"""Plan and collect event-aligned GPM IMERG Final V07 granules.

The script deliberately separates source acquisition from feature extraction.
It creates deduplicated causal half-hourly IMERG tasks for every timestamp in
the multi-source collection plan, records the direct GES DISC and OPeNDAP URLs, and
only downloads data when locally configured Earthdata credentials are present.

The direct files are global HDF5 granules. The source windows contain the 48
complete intervals required for every causal 24-hour accumulation. A follow-on
extractor can spatially subset them around the matching IBTrACS position and
write the feature contract recorded in the generated manifest. This keeps the collector dependency-light:
it uses only Python's standard library and does not pretend to derive rainfall
statistics without an HDF5 reader.

NASA references used for the URL shape and authentication flow:
* GES DISC collection: GPM_3IMERGHH.07 (IMERG Final half-hourly V07)
* CMR collection concept ID: C2723754847-GES_DISC
* Earthdata Login's documented urllib + CookieJar authentication pattern
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import netrc
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPCookieProcessor,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "imerg"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "imerg_manifest.json"

# Current official identifiers for GPM IMERG Final V07, verified against the
# NASA CMR virtual directory and its GES DISC/OPeNDAP links on 2026-09-26.
DATASET = "GPM_3IMERGHH.07"
COLLECTION_CONCEPT_ID = "C2723754847-GES_DISC"
VERSION = "V07B"
GES_DISC_DATA_ROOT = "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3"
OPENDAP_COLLECTION_ROOT = "https://opendap.earthdata.nasa.gov/collections"
EARTHDATA_AUTH_HOST = "urs.earthdata.nasa.gov"

# The feature file is intentionally not created by this collector.  It is the
# explicit hand-off contract for a later HDF5-aware extraction step.
FEATURE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "imerg_features.csv"
FEATURE_COLUMNS = [
    "storm_id",
    "timestamp_utc",
    "imerg_granule_start_utc",
    "imerg_center_precipitation_cal_mm_hr",
    "imerg_window_mean_precipitation_cal_mm_hr",
    "imerg_window_p95_precipitation_cal_mm_hr",
    "imerg_window_max_precipitation_cal_mm_hr",
    "imerg_window_mean_random_error_mm_hr",
    "imerg_window_mean_quality_index",
    "imerg_accumulation_6h_mm",
    "imerg_accumulation_24h_mm",
    "imerg_missing",
]

USER_AGENT = "CycloneSentinel-IMERGCollector/1.0 (+https://github.com/)"
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class EarthdataCredentials:
    """Credentials discovered from a local, user-managed source."""

    username: str
    password: str
    source: str


def resolve_path(value: Path, *, default: Path) -> Path:
    """Resolve a command-line path relative to the repository when needed."""
    path = value if value != default else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Return a stable project-relative path whenever the path is in the repo."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Return a SHA-256 digest without loading a potentially large file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(text: object, *, context: str) -> datetime:
    """Parse an ISO-8601 timestamp and require an explicit UTC offset."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{context} must be a non-empty ISO-8601 timestamp.")
    try:
        value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{context} is not a valid ISO-8601 timestamp: {text!r}") from error
    if value.tzinfo is None:
        raise ValueError(f"{context} must include UTC offset or Z: {text!r}")
    return value.astimezone(UTC)


def utc_text(value: datetime) -> str:
    """Format an aware UTC time in the plan's canonical representation."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def imerg_granule_start(timestamp: datetime) -> datetime:
    """Return the latest *completed* half-hourly interval's start time.

    Feature values must be knowable at the prediction issue time.  We therefore
    use the interval ending at the most recent half-hour boundary, rather than
    the interval that contains the timestamp (which would leak future rainfall).
    """
    normalized = timestamp.astimezone(UTC).replace(second=0, microsecond=0)
    completed_end = normalized - timedelta(minutes=normalized.minute % 30)
    return completed_end - timedelta(minutes=30)


def imerg_filename(granule_start: datetime) -> str:
    """Create the NASA V07B half-hourly IMERG filename for a granule start."""
    start = granule_start.astimezone(UTC)
    if start.minute not in {0, 30} or start.second or start.microsecond:
        raise ValueError("IMERG granule starts must be on a UTC half-hour boundary.")
    end = start + timedelta(minutes=30) - timedelta(seconds=1)
    minute_index = start.hour * 60 + start.minute
    return (
        "3B-HHR.MS.MRG.3IMERG."
        f"{start:%Y%m%d}-S{start:%H%M%S}-E{end:%H%M%S}."
        f"{minute_index:04d}.{VERSION}.HDF5"
    )


def build_urls(granule_start: datetime, filename: str) -> tuple[str, str]:
    """Return official direct-download and OPeNDAP URLs for one known granule.

    The direct URL is the actual HDF5 download target.  The OPeNDAP URL is the
    CMR-linked DMR HTML endpoint for the same granule and is retained for a
    later spatial-subsetting implementation; it is not treated as a
    byte-for-byte HDF5 fallback.
    """
    start = granule_start.astimezone(UTC)
    day_of_year = start.timetuple().tm_yday
    direct_url = (
        f"{GES_DISC_DATA_ROOT}/{DATASET}/{start.year}/{day_of_year:03d}/{filename}"
    )
    granule_identifier = quote(f"{DATASET}:{filename}", safe="")
    opendap_url = (
        f"{OPENDAP_COLLECTION_ROOT}/{COLLECTION_CONCEPT_ID}/granules/"
        f"{granule_identifier}.dmr.html"
    )
    return direct_url, opendap_url


def task_id(storm_id: str, granule_start: datetime, filename: str) -> str:
    """Create a compact, reproducible task identifier without secrets."""
    key = f"{DATASET}|{storm_id}|{utc_text(granule_start)}|{filename}".encode("utf-8")
    return f"imerg-{hashlib.sha256(key).hexdigest()[:16]}"


def read_plan(path: Path) -> dict[str, Any]:
    """Load and validate the limited portion of the multi-source plan we use."""
    if not path.is_file():
        raise FileNotFoundError(f"Collection plan is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Collection plan is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Collection plan root must be a JSON object.")
    storms = payload.get("storms")
    if not isinstance(storms, list) or not storms:
        raise ValueError("Collection plan must contain a non-empty 'storms' list.")
    return payload


def selected_storms(plan: dict[str, Any], storm_id: str | None) -> list[dict[str, Any]]:
    """Validate and filter plan storms while keeping their stable plan order."""
    selected: list[dict[str, Any]] = []
    seen_storm_ids: set[str] = set()
    for index, storm in enumerate(plan["storms"]):
        if not isinstance(storm, dict):
            raise ValueError(f"storms[{index}] must be a JSON object.")
        current_id = storm.get("storm_id")
        if not isinstance(current_id, str) or not current_id.strip():
            raise ValueError(f"storms[{index}].storm_id must be a non-empty string.")
        if current_id in seen_storm_ids:
            raise ValueError(f"Collection plan has duplicate storm_id: {current_id}")
        seen_storm_ids.add(current_id)
        time_steps = storm.get("time_steps_utc")
        if not isinstance(time_steps, list) or not time_steps:
            raise ValueError(f"Storm {current_id} must have a non-empty time_steps_utc list.")
        if storm_id is None or current_id == storm_id:
            selected.append(storm)
    if not selected:
        raise ValueError(f"No matching storm in collection plan: {storm_id}")
    return selected


def build_tasks(storms: Iterable[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """Build raw tasks for causal instantaneous, 6h, and 24h rainfall features.

    The 6h and 24h features require 12 and 48 complete half-hourly intervals,
    respectively.  We deduplicate the union across a storm's issue times so the
    downloaded raw collection is compact but still sufficient to calculate the
    stated feature contract without filling gaps.
    """
    tasks: list[dict[str, Any]] = []
    for storm in storms:
        storm_id = str(storm["storm_id"])
        parsed_steps: list[tuple[datetime, str]] = []
        seen_timestamps: set[datetime] = set()
        for step_index, raw_timestamp in enumerate(storm["time_steps_utc"]):
            timestamp = parse_timestamp(
                raw_timestamp,
                context=f"Storm {storm_id} time_steps_utc[{step_index}]",
            )
            if timestamp in seen_timestamps:
                raise ValueError(f"Storm {storm_id} has a duplicate timestamp: {utc_text(timestamp)}")
            seen_timestamps.add(timestamp)
            parsed_steps.append((timestamp, utc_text(timestamp)))

        required_granules: dict[datetime, set[str]] = {}
        for timestamp, timestamp_text in sorted(parsed_steps, key=lambda item: item[0]):
            granule_start = imerg_granule_start(timestamp)
            # `granule_start` is the latest completed interval.  Forty-eight
            # intervals (including it) cover the causal 24 hours ending at the
            # issue time; this also subsumes the 6-hour window.
            for offset in range(48):
                candidate = granule_start - timedelta(minutes=30 * offset)
                required_granules.setdefault(candidate, set()).add(timestamp_text)

        for granule_start, issue_times in sorted(required_granules.items()):
            filename = imerg_filename(granule_start)
            direct_url, opendap_url = build_urls(granule_start, filename)
            day_of_year = granule_start.timetuple().tm_yday
            target = output_dir / storm_id / f"{granule_start.year}" / f"{day_of_year:03d}" / filename
            tasks.append({
                "task_id": task_id(storm_id, granule_start, filename),
                "storm_id": storm_id,
                "granule_start_utc": utc_text(granule_start),
                "contributes_to_issue_times_utc": sorted(issue_times),
                "imerg_granule_start_utc": utc_text(granule_start),
                "temporal_alignment": "completed half-hour granule ending at or before each issue time",
                "dataset": DATASET,
                "collection_concept_id": COLLECTION_CONCEPT_ID,
                "filename": filename,
                "download_url": direct_url,
                "opendap_url": opendap_url,
                "target": project_relative(target),
                "status": "planned",
            })
    return tasks


def local_netrc_candidates() -> list[Path]:
    """Return conventional local netrc locations, without creating any files."""
    candidates: list[Path] = []
    configured = os.environ.get("NETRC")
    if configured:
        candidates.append(Path(configured).expanduser())
    home = Path.home()
    candidates.extend([home / ".netrc", home / "_netrc"])
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_credentials() -> EarthdataCredentials | None:
    """Discover Earthdata credentials without prompting or serializing secrets.

    A complete ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD`` pair wins over
    netrc.  Otherwise, the standard ``.netrc`` / Windows ``_netrc`` entry for
    ``urs.earthdata.nasa.gov`` is used.  The username, password, and exact
    credential-file path are never printed or written to the manifest.
    """
    env_username = os.environ.get("EARTHDATA_USERNAME", "").strip()
    env_password = os.environ.get("EARTHDATA_PASSWORD", "")
    if env_username and env_password:
        return EarthdataCredentials(env_username, env_password, "environment")

    for candidate in local_netrc_candidates():
        if not candidate.is_file():
            continue
        try:
            parsed = netrc.netrc(str(candidate))
            authenticators = parsed.authenticators(EARTHDATA_AUTH_HOST)
        except (OSError, netrc.NetrcParseError):
            # A malformed unrelated netrc must not cause a network request. A
            # later valid candidate or a configured environment pair can still
            # be used on a subsequent run.
            continue
        if authenticators is None:
            continue
        login, account, password = authenticators
        username = (login or account or "").strip()
        if username and password:
            return EarthdataCredentials(username, password, "netrc")
    return None


def build_authenticated_opener(credentials: EarthdataCredentials):
    """Build NASA's documented urllib authentication + in-memory-cookie flow."""
    password_manager = HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(
        None,
        f"https://{EARTHDATA_AUTH_HOST}",
        credentials.username,
        credentials.password,
    )
    # Some installations challenge on the data host before redirecting to URS;
    # supplying this realm lets urllib complete either documented flow.
    password_manager.add_password(
        None,
        "https://data.gesdisc.earthdata.nasa.gov",
        credentials.username,
        credentials.password,
    )
    opener = build_opener(
        HTTPBasicAuthHandler(password_manager),
        HTTPCookieProcessor(CookieJar()),
    )
    opener.addheaders = [("User-Agent", USER_AGENT)]
    return opener


def absolute_target(task: dict[str, Any]) -> Path:
    """Convert a serialized target value back to the local target path."""
    target = Path(str(task["target"]))
    return target if target.is_absolute() else PROJECT_ROOT / target


def download_task(opener: Any, task: dict[str, Any], *, timeout_seconds: float, overwrite: bool) -> dict[str, Any]:
    """Download one file atomically and return non-secret provenance metadata."""
    target = absolute_target(task)
    if target.exists() and not overwrite:
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"Existing target is not a usable file: {target}")
        return {
            "status": "already_present",
            "bytes": target.stat().st_size,
        }

    temporary: Path | None = None
    digest = hashlib.sha256()
    bytes_written = 0
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_text = tempfile.mkstemp(
            prefix=f"{target.name}.part-",
            dir=target.parent,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_text)
        request = Request(str(task["download_url"]), headers={"User-Agent": USER_AGENT})
        with opener.open(request, timeout=timeout_seconds) as response:
            content_type = response.headers.get_content_type().lower()
            # A successful HTML response is typically an authorization/login
            # page; never persist it as if it were an HDF5 scientific product.
            if content_type in {"text/html", "application/xhtml+xml"}:
                raise RuntimeError(
                    "Received an HTML response instead of an IMERG HDF5 file; "
                    "verify Earthdata authorization for GES DISC."
                )
            with temporary.open("wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)
        if bytes_written <= 0:
            raise RuntimeError("Downloaded zero bytes; target was not committed.")
        # Default runs never replace a valid existing artifact.  --overwrite is
        # an explicit user decision and enables the atomic replacement below.
        if target.exists() and not overwrite:
            raise RuntimeError(f"Target appeared during download; refusing to replace it: {target}")
        os.replace(temporary, target)
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError(str(error)) from error
    return {
        "status": "downloaded",
        "bytes": bytes_written,
        "sha256": digest.hexdigest(),
        "downloaded_at_utc": utc_text(datetime.now(UTC)),
    }


def feature_contract() -> dict[str, Any]:
    """Describe the exact, non-fabricated feature hand-off for model assembly."""
    return {
        "path": project_relative(FEATURE_OUTPUT_PATH),
        "key_columns": ["storm_id", "timestamp_utc"],
        "row_grain": "exactly one row for every requested storm_id and plan timestamp_utc",
        "columns": FEATURE_COLUMNS,
        "source_fields": {
            "precipitation": "Grid/precipitationCal (mm hr-1)",
            "random_error": "Grid/randomError (mm hr-1)",
            "quality_index": "Grid/precipitationQualityIndex (unitless)",
        },
        "spatial_window": "2.5 degrees in latitude and longitude around the matched IBTrACS centre",
        "temporal_policy": {
            "instantaneous": (
                "Use the completed half-hour interval ending at or before the issue time; "
                "do not relabel the join key."
            ),
            "accumulations": (
                "Sum complete half-hourly precipitationCal intervals ending at the "
                "matched timestamp (rate * 0.5 hours). Do not fill missing intervals."
            ),
        },
        "missingness": (
            "Set imerg_missing to 1 and leave every imerg_* value blank when the "
            "matching raw granule cannot be downloaded or extracted. Never encode "
            "missing precipitation as 0. For partially unavailable 6h/24h windows, "
            "leave the affected accumulation value blank and retain imerg_missing=0 "
            "only when the instantaneous granule was successfully extracted."
        ),
        "collector_scope": (
            "This script acquires raw HDF5 granules and writes this contract; an "
            "HDF5-aware extractor is responsible for producing the CSV."
        ),
    }


def task_summary(tasks: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count task states in a deterministic key order for manifest consumers."""
    counts: dict[str, int] = {}
    for task in tasks:
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def build_manifest(
    *,
    plan_path: Path,
    plan: dict[str, Any],
    selected: list[dict[str, Any]],
    output_dir: Path,
    tasks: list[dict[str, Any]],
    run_status: str,
    credential_source: str | None = None,
) -> dict[str, Any]:
    """Create serializable provenance for both dry runs and real collection."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": utc_text(datetime.now(UTC)),
        "status": run_status,
        "source": {
            "provider": "NASA GES DISC",
            "dataset": DATASET,
            "product": "GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 degree V07",
            "collection_concept_id": COLLECTION_CONCEPT_ID,
            "direct_download_root": GES_DISC_DATA_ROOT,
            "opendap_collection_root": OPENDAP_COLLECTION_ROOT,
        },
        "plan": {
            "path": project_relative(plan_path),
            "sha256": sha256_file(plan_path),
            "schema_version": plan.get("schema_version"),
            "storm_ids": [str(storm["storm_id"]) for storm in selected],
        },
        "raw_output_dir": project_relative(output_dir),
        "feature_contract": feature_contract(),
        "tasks": tasks,
        "summary": {
            "storms": len(selected),
            "tasks": len(tasks),
            "task_statuses": task_summary(tasks),
        },
    }
    if credential_source is not None:
        # This signals that the run used a local configuration without exposing
        # its username, password, or location.
        manifest["authentication"] = {"credential_source": credential_source}
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a readable manifest atomically so an interruption cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_text = tempfile.mkstemp(
        prefix=f"{path.name}.part-",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary = Path(temporary_text)
    try:
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def mark_pending_credentials(tasks: Iterable[dict[str, Any]]) -> None:
    """Mark planned work as deliberately not attempted, rather than failed."""
    for task in tasks:
        task["status"] = "pending_local_earthdata_credentials"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan/download event-aligned NASA GPM IMERG Final V07 HDF5 granules."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Multi-source plan JSON.")
    parser.add_argument("--storm-id", help="Collect one plan storm ID; omit to include every plan storm.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Raw IMERG target root.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output provenance manifest JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic tasks/manifest without network access.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Per-file HTTPS timeout for authenticated downloads (default: 120).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing raw targets after a successful re-download.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be a positive finite number.")

    plan_path = resolve_path(args.plan, default=DEFAULT_PLAN)
    output_dir = resolve_path(args.output_dir, default=DEFAULT_OUTPUT_DIR)
    manifest_path = resolve_path(args.manifest, default=DEFAULT_MANIFEST)
    plan = read_plan(plan_path)
    selected = selected_storms(plan, args.storm_id)
    tasks = build_tasks(selected, output_dir)

    if args.dry_run:
        manifest = build_manifest(
            plan_path=plan_path,
            plan=plan,
            selected=selected,
            output_dir=output_dir,
            tasks=tasks,
            run_status="planned_dry_run",
        )
        write_manifest(manifest_path, manifest)
        print(
            f"Dry run: planned {len(tasks):,} IMERG Final V07 task(s) for "
            f"{len(selected)} storm(s)."
        )
        print(f"Manifest: {project_relative(manifest_path)}")
        return

    credentials = resolve_credentials()
    if credentials is None:
        mark_pending_credentials(tasks)
        manifest = build_manifest(
            plan_path=plan_path,
            plan=plan,
            selected=selected,
            output_dir=output_dir,
            tasks=tasks,
            run_status="blocked_local_earthdata_credentials_missing",
        )
        write_manifest(manifest_path, manifest)
        print("No local Earthdata credentials were found; no network request was made.", file=sys.stderr)
        print(
            "Configure EARTHDATA_USERNAME/EARTHDATA_PASSWORD or a local netrc entry for "
            "urs.earthdata.nasa.gov, then rerun.",
            file=sys.stderr,
        )
        print(f"Manifest: {project_relative(manifest_path)}", file=sys.stderr)
        return

    manifest = build_manifest(
        plan_path=plan_path,
        plan=plan,
        selected=selected,
        output_dir=output_dir,
        tasks=tasks,
        run_status="collecting",
        credential_source=credentials.source,
    )
    write_manifest(manifest_path, manifest)
    opener = build_authenticated_opener(credentials)
    failures = 0
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task['storm_id']} {task['granule_start_utc']}…")
        try:
            task.update(
                download_task(
                    opener,
                    task,
                    timeout_seconds=args.timeout_seconds,
                    overwrite=args.overwrite,
                )
            )
        except RuntimeError as error:
            failures += 1
            task["status"] = "failed"
            task["error"] = str(error)
            print(f"  Failed: {error}", file=sys.stderr)
        # Preserve useful provenance even if the process is stopped part-way
        # through a large selected cohort.
        manifest["generated_at_utc"] = utc_text(datetime.now(UTC))
        manifest["summary"]["task_statuses"] = task_summary(tasks)
        write_manifest(manifest_path, manifest)

    manifest["generated_at_utc"] = utc_text(datetime.now(UTC))
    manifest["status"] = "complete" if failures == 0 else "completed_with_failures"
    manifest["summary"]["task_statuses"] = task_summary(tasks)
    write_manifest(manifest_path, manifest)
    print(f"Manifest: {project_relative(manifest_path)}")
    if failures:
        raise SystemExit(f"IMERG collection completed with {failures} failed task(s).")
    print(f"Complete: {len(tasks):,} IMERG task(s) are available under {project_relative(output_dir)}")


if __name__ == "__main__":
    main()
