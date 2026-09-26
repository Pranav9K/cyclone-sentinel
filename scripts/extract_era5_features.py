"""Extract event-aligned ERA5 surface features for the cyclone cohort.

This is intentionally a separate, offline stage after ``collect_era5.py``:
it never contacts CDS and never needs a CDS credential.  The default path is
therefore safe to run before data have been downloaded.  In that state it
writes the stable feature-table header and a manifest explaining that no ERA5
records are available yet.

The preferred input is the GRIB output produced by ``collect_era5.py``.  GRIB
and NetCDF support are optional because decoding them requires ``xarray`` and
``cfgrib``.  A dependency-free CSV input contract is also supported, which is
useful for a CDS CSV export and for validation in constrained environments.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"
DEFAULT_TRACKS = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "era5_features.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "era5_features_manifest.json"

OUTPUT_COLUMNS = [
    "storm_id",
    "timestamp_utc",
    "era5_u10_mps",
    "era5_v10_mps",
    "era5_wind10_mps",
    "era5_mslp_pa",
    "era5_sst_k",
    "era5_tcw_kg_m2",
    "era5_dewpoint_k",
    "era5_missing",
]

# The extractor accepts short names written by cfgrib as well as the verbose
# names that occur in CDS CSV exports.  The values must remain in the canonical
# ERA5 SI units documented in docs/ERA5_FEATURE_EXTRACTION.md.
SOURCE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp_utc": ("timestamp_utc", "valid_time", "time", "datetime", "date_time"),
    "latitude": ("latitude", "lat"),
    "longitude": ("longitude", "lon", "lng"),
    "u10": (
        "u10",
        "u10_mps",
        "10m_u_component_of_wind",
        "10m_u_component_of_wind_m_s",
    ),
    "v10": (
        "v10",
        "v10_mps",
        "10m_v_component_of_wind",
        "10m_v_component_of_wind_m_s",
    ),
    "mslp": ("msl", "mslp", "mslp_pa", "mean_sea_level_pressure"),
    "sst": ("sst", "sst_k", "sea_surface_temperature"),
    "tcw": ("tcwv", "tcw", "tcw_kg_m2", "total_column_water_vapour"),
    "dewpoint": (
        "d2m",
        "dewpoint",
        "dewpoint_k",
        "2m_dewpoint_temperature",
    ),
    "storm_id": ("storm_id", "sid", "storm"),
}

REQUIRED_SOURCE_CONCEPTS = (
    "timestamp_utc",
    "latitude",
    "longitude",
    "u10",
    "v10",
    "mslp",
    "sst",
    "tcw",
    "dewpoint",
)
SUPPORTED_SUFFIXES = {".csv", ".grib", ".grb", ".grib2", ".nc", ".netcdf"}


class InputValidationError(ValueError):
    """Raised when a required local input cannot define the event cohort."""


@dataclass(frozen=True, order=True)
class EventKey:
    """An expected ERA5 feature location at one cyclone issue time."""

    storm_id: str
    timestamp: datetime
    latitude: float
    longitude: float

    @property
    def key(self) -> tuple[str, str]:
        return self.storm_id, format_timestamp(self.timestamp)


@dataclass(frozen=True)
class Candidate:
    """The closest valid ERA5 grid point observed for an event key."""

    distance_km: float
    source_file: str
    source_row: int | None
    u10: float
    v10: float
    mslp: float
    sst: float
    tcw: float
    dewpoint: float


@dataclass
class FileResult:
    """Compact provenance and validation result for one raw source file."""

    path: str
    kind: str
    bytes: int
    status: str
    rows_seen: int = 0
    rows_valid: int = 0
    rows_considered: int = 0
    error: str | None = None
    missing_columns: list[str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "bytes": self.bytes,
            "status": self.status,
            "rows_seen": self.rows_seen,
            "rows_valid": self.rows_valid,
            "rows_considered": self.rows_considered,
            "error": self.error,
            "missing_columns": self.missing_columns or [],
        }


def resolve_path(value: Path | str) -> Path:
    """Resolve a CLI path relative to the project without changing absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_relative(path: Path) -> str:
    """Use a portable relative path in manifests when the file is in this repo."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Hash a modest metadata input in chunks for reproducible provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_column_name(value: str) -> str:
    """Normalize source headings such as ``10m U-component of wind``."""
    return "".join(character.lower() for character in value.strip() if character.isalnum())


def find_column_map(fieldnames: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Resolve a raw-table header to canonical source concepts."""
    normalized = {normalize_column_name(name): name for name in fieldnames if name}
    resolved: dict[str, str] = {}
    for concept, aliases in SOURCE_COLUMN_ALIASES.items():
        for alias in aliases:
            source_name = normalized.get(normalize_column_name(alias))
            if source_name is not None:
                resolved[concept] = source_name
                break
    missing = [concept for concept in REQUIRED_SOURCE_CONCEPTS if concept not in resolved]
    return resolved, missing


def parse_timestamp(value: object, *, label: str) -> datetime:
    """Parse an ISO-like timestamp and normalize it to UTC.

    A timezone-less ERA5 CSV time is interpreted as UTC, which is the CDS
    convention.  The action is recorded in the input contract rather than
    guessing a local timezone.
    """
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is blank")
    candidate = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        parsed = None
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"{label} is not an ISO UTC timestamp: {text!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    """Format all output timestamps as a stable UTC key."""
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_finite_float(value: object, *, label: str) -> float:
    """Parse a finite numeric value or report a useful source error."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is blank")
    try:
        number = float(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {text!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def normalize_longitude(longitude: float) -> float:
    """Normalize a valid source longitude to [-180, 180)."""
    normalized = (longitude + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == 0.0 else normalized


def validate_feature_values(values: dict[str, float]) -> str | None:
    """Return a validation reason for non-physical canonical ERA5 values."""
    bounds = {
        "u10": (-150.0, 150.0),
        "v10": (-150.0, 150.0),
        "mslp": (80_000.0, 110_000.0),
        "sst": (250.0, 330.0),
        "tcw": (0.0, 100.0),
        "dewpoint": (180.0, 330.0),
    }
    for name, (lower, upper) in bounds.items():
        value = values[name]
        if not lower <= value <= upper:
            return f"{name}_out_of_range"
    return None


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    """Calculate great-circle separation while handling a longitude seam."""
    earth_radius_km = 6371.0088
    latitude_a_rad = math.radians(latitude_a)
    latitude_b_rad = math.radians(latitude_b)
    delta_latitude = latitude_b_rad - latitude_a_rad
    delta_longitude = math.radians(normalize_longitude(longitude_b - longitude_a))
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_a_rad) * math.cos(latitude_b_rad) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(min(1.0, math.sqrt(haversine)))


def load_plan(path: Path) -> dict[str, set[datetime]]:
    """Read the expected storm/time keys from the event-aligned collection plan."""
    if not path.exists():
        raise InputValidationError(f"Collection plan is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise InputValidationError(f"Collection plan is not valid JSON: {path}") from error
    storms = payload.get("storms")
    if not isinstance(storms, list) or not storms:
        raise InputValidationError("Collection plan has no non-empty 'storms' list")

    expected: dict[str, set[datetime]] = {}
    for index, storm in enumerate(storms, start=1):
        if not isinstance(storm, dict):
            raise InputValidationError(f"Collection plan storm entry {index} is not an object")
        storm_id = str(storm.get("storm_id", "")).strip()
        times = storm.get("time_steps_utc")
        if not storm_id or not isinstance(times, list) or not times:
            raise InputValidationError(
                f"Collection plan storm entry {index} needs a storm_id and non-empty time_steps_utc"
            )
        if storm_id in expected:
            raise InputValidationError(f"Collection plan duplicates storm_id {storm_id!r}")
        parsed_times: set[datetime] = set()
        for value in times:
            try:
                parsed_times.add(parse_timestamp(value, label=f"plan {storm_id} time_steps_utc"))
            except ValueError as error:
                raise InputValidationError(str(error)) from error
        if len(parsed_times) != len(times):
            raise InputValidationError(f"Collection plan duplicates time steps for {storm_id}")
        expected[storm_id] = parsed_times
    return expected


def load_events(plan: dict[str, set[datetime]], tracks_path: Path) -> tuple[list[EventKey], list[str]]:
    """Join planned timestamps to their recorded IBTrACS cyclone centre."""
    if not tracks_path.exists():
        raise InputValidationError(f"Normalized track table is missing: {tracks_path}")
    with tracks_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"storm_id", "timestamp_utc", "latitude", "longitude"}
        missing = sorted(required - fields)
        if missing:
            raise InputValidationError(
                f"Track table is missing required column(s): {', '.join(missing)}"
            )

        positions: dict[tuple[str, datetime], tuple[float, float]] = {}
        duplicate_positions: set[tuple[str, datetime]] = set()
        selected_ids = set(plan)
        for row_number, row in enumerate(reader, start=2):
            storm_id = (row.get("storm_id") or "").strip()
            if storm_id not in selected_ids:
                continue
            try:
                timestamp = parse_timestamp(row.get("timestamp_utc", ""), label=f"tracks:{row_number} timestamp_utc")
                latitude = parse_finite_float(row.get("latitude", ""), label=f"tracks:{row_number} latitude")
                longitude = parse_finite_float(row.get("longitude", ""), label=f"tracks:{row_number} longitude")
            except ValueError as error:
                raise InputValidationError(str(error)) from error
            if not -90.0 <= latitude <= 90.0:
                raise InputValidationError(f"tracks:{row_number} latitude is outside [-90, 90]")
            if not -360.0 <= longitude <= 360.0:
                raise InputValidationError(f"tracks:{row_number} longitude is outside [-360, 360]")
            key = (storm_id, timestamp)
            value = (latitude, normalize_longitude(longitude))
            existing = positions.get(key)
            if existing is not None and existing != value:
                duplicate_positions.add(key)
            positions[key] = value

    events: list[EventKey] = []
    missing_events: list[str] = []
    for storm_id, timestamps in plan.items():
        for timestamp in timestamps:
            key = (storm_id, timestamp)
            if key in duplicate_positions:
                raise InputValidationError(
                    f"Track table has conflicting positions for {storm_id} at {format_timestamp(timestamp)}"
                )
            position = positions.get(key)
            if position is None:
                missing_events.append(f"{storm_id}@{format_timestamp(timestamp)}")
                continue
            events.append(EventKey(storm_id, timestamp, *position))
    events.sort(key=lambda item: (item.timestamp, item.storm_id))
    return events, missing_events


def discover_raw_files(raw_dir: Path) -> list[Path]:
    """Find supported downloaded/extracted ERA5 files, deterministically."""
    if not raw_dir.exists():
        return []
    if not raw_dir.is_dir():
        raise InputValidationError(f"--raw-dir is not a directory: {raw_dir}")
    return sorted(
        (path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda path: str(path).lower(),
    )


def closest_event_times(
    timestamp: datetime,
    target_times: list[datetime],
    time_tolerance: timedelta,
) -> list[datetime]:
    """Return planned times close enough to a raw source time, in UTC."""
    if not target_times:
        return []
    index = bisect.bisect_left(target_times, timestamp)
    candidates: list[datetime] = []
    for candidate_index in (index - 1, index):
        if 0 <= candidate_index < len(target_times):
            candidate = target_times[candidate_index]
            if abs(candidate - timestamp) <= time_tolerance:
                candidates.append(candidate)
    return candidates


def parse_source_row(
    row: dict[str, object],
    columns: dict[str, str],
    *,
    source_label: str,
) -> tuple[datetime, float, float, str | None, dict[str, float]]:
    """Convert one raw row into canonical ERA5 values with strict units."""
    timestamp = parse_timestamp(row.get(columns["timestamp_utc"], ""), label=f"{source_label} timestamp")
    latitude = parse_finite_float(row.get(columns["latitude"], ""), label=f"{source_label} latitude")
    longitude = parse_finite_float(row.get(columns["longitude"], ""), label=f"{source_label} longitude")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"{source_label} latitude is outside [-90, 90]")
    if not -360.0 <= longitude <= 360.0:
        raise ValueError(f"{source_label} longitude is outside [-360, 360]")

    values = {
        name: parse_finite_float(row.get(columns[name], ""), label=f"{source_label} {name}")
        for name in ("u10", "v10", "mslp", "sst", "tcw", "dewpoint")
    }
    invalid_reason = validate_feature_values(values)
    if invalid_reason:
        raise ValueError(f"{source_label} {invalid_reason}")
    storm_column = columns.get("storm_id")
    raw_storm_id = str(row.get(storm_column, "")).strip() if storm_column else None
    return timestamp, latitude, normalize_longitude(longitude), raw_storm_id or None, values


def source_error_bucket(error: ValueError) -> str:
    """Map potentially unique bad cell values to bounded manifest counters."""
    message = str(error).lower()
    for concept in ("timestamp", "latitude", "longitude", "u10", "v10", "mslp", "sst", "tcw", "dewpoint"):
        if concept in message:
            return concept
    if "not numeric" in message or "not finite" in message or "blank" in message:
        return "numeric_value"
    return "other"


def accept_candidate(
    *,
    timestamp: datetime,
    latitude: float,
    longitude: float,
    raw_storm_id: str | None,
    values: dict[str, float],
    source_file: str,
    source_row: int | None,
    events_by_time: dict[datetime, list[EventKey]],
    target_times: list[datetime],
    time_tolerance: timedelta,
    max_grid_distance_km: float,
    candidates: dict[EventKey, Candidate],
    counts: Counter[str],
) -> int:
    """Keep a source record only if it is closest to a planned cyclone centre."""
    relevant_times = closest_event_times(timestamp, target_times, time_tolerance)
    if not relevant_times:
        counts["raw_rows_outside_event_times"] += 1
        return 0
    considered = 0
    for event_time in relevant_times:
        for event in events_by_time[event_time]:
            if raw_storm_id is not None and raw_storm_id != event.storm_id:
                counts["raw_rows_storm_id_not_in_scope"] += 1
                continue
            distance = haversine_km(event.latitude, event.longitude, latitude, longitude)
            if distance > max_grid_distance_km:
                counts["raw_grid_points_too_far"] += 1
                continue
            considered += 1
            candidate = Candidate(
                distance_km=distance,
                source_file=source_file,
                source_row=source_row,
                u10=values["u10"],
                v10=values["v10"],
                mslp=values["mslp"],
                sst=values["sst"],
                tcw=values["tcw"],
                dewpoint=values["dewpoint"],
            )
            existing = candidates.get(event)
            if existing is None or candidate.distance_km < existing.distance_km:
                candidates[event] = candidate
                counts["candidate_selected"] += 1
    return considered


def inspect_csv_header(path: Path) -> tuple[dict[str, str], list[str]]:
    """Validate a dependency-free raw CSV header without reading all grid rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}, list(REQUIRED_SOURCE_CONCEPTS)
        return find_column_map(reader.fieldnames)


def process_csv_file(
    path: Path,
    result: FileResult,
    *,
    events_by_time: dict[datetime, list[EventKey]],
    target_times: list[datetime],
    time_tolerance: timedelta,
    max_grid_distance_km: float,
    candidates: dict[EventKey, Candidate],
    counts: Counter[str],
) -> None:
    """Stream a flat ERA5 grid CSV and retain only nearest event rows."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                result.status = "invalid_header"
                result.missing_columns = list(REQUIRED_SOURCE_CONCEPTS)
                return
            columns, missing = find_column_map(reader.fieldnames)
            if missing:
                result.status = "invalid_header"
                result.missing_columns = missing
                return
            result.status = "read"
            for row_number, row in enumerate(reader, start=2):
                result.rows_seen += 1
                counts["raw_rows_seen"] += 1
                try:
                    timestamp, latitude, longitude, raw_storm_id, values = parse_source_row(
                        row, columns, source_label=f"{project_relative(path)}:{row_number}"
                    )
                except ValueError as error:
                    counts[f"invalid_source_{source_error_bucket(error)}"] += 1
                    continue
                result.rows_valid += 1
                counts["raw_rows_valid"] += 1
                result.rows_considered += accept_candidate(
                    timestamp=timestamp,
                    latitude=latitude,
                    longitude=longitude,
                    raw_storm_id=raw_storm_id,
                    values=values,
                    source_file=project_relative(path),
                    source_row=row_number,
                    events_by_time=events_by_time,
                    target_times=target_times,
                    time_tolerance=time_tolerance,
                    max_grid_distance_km=max_grid_distance_km,
                    candidates=candidates,
                    counts=counts,
                )
    except (OSError, UnicodeError, csv.Error) as error:
        result.status = "unreadable"
        result.error = str(error)


def xarray_availability(kind: str) -> tuple[bool, str | None]:
    """Check optional file-decoder availability without importing it by default."""
    try:
        import xarray  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False, "Install optional decoder dependencies: pip install xarray"
    if kind == "grib":
        try:
            import cfgrib  # noqa: F401  # type: ignore[import-not-found]
        except ImportError:
            return False, "Install optional GRIB decoder dependencies: pip install xarray cfgrib"
    return True, None


def process_xarray_file(
    path: Path,
    result: FileResult,
    *,
    kind: str,
    events_by_time: dict[datetime, list[EventKey]],
    target_times: list[datetime],
    time_tolerance: timedelta,
    max_grid_distance_km: float,
    candidates: dict[EventKey, Candidate],
    counts: Counter[str],
) -> None:
    """Optionally flatten a CDS GRIB/NetCDF grid through xarray then reuse validation."""
    available, reason = xarray_availability(kind)
    if not available:
        result.status = "decoder_unavailable"
        result.error = reason
        return
    try:
        import xarray as xr  # type: ignore[import-not-found]

        open_kwargs: dict[str, Any] = {}
        if kind == "grib":
            # Avoid persisting cfgrib index sidecars within data/raw.
            open_kwargs = {"engine": "cfgrib", "backend_kwargs": {"indexpath": ""}}
        dataset = xr.open_dataset(path, **open_kwargs)
        try:
            dataframe = dataset.to_dataframe().reset_index()
        finally:
            dataset.close()
    except Exception as error:  # Optional decoders have package-specific errors.
        result.status = "unreadable"
        result.error = f"{type(error).__name__}: {error}"
        return

    columns, missing = find_column_map(str(column) for column in dataframe.columns)
    if missing:
        result.status = "invalid_header"
        result.missing_columns = missing
        return
    result.status = "read"
    for row_number, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        result.rows_seen += 1
        counts["raw_rows_seen"] += 1
        try:
            timestamp, latitude, longitude, raw_storm_id, values = parse_source_row(
                row, columns, source_label=f"{project_relative(path)}:{row_number}"
            )
        except ValueError as error:
            counts[f"invalid_source_{source_error_bucket(error)}"] += 1
            continue
        result.rows_valid += 1
        counts["raw_rows_valid"] += 1
        result.rows_considered += accept_candidate(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            raw_storm_id=raw_storm_id,
            values=values,
            source_file=project_relative(path),
            source_row=row_number,
            events_by_time=events_by_time,
            target_times=target_times,
            time_tolerance=time_tolerance,
            max_grid_distance_km=max_grid_distance_km,
            candidates=candidates,
            counts=counts,
        )


def source_kind(path: Path) -> str:
    """Map a discovered suffix to the appropriate parser category."""
    if path.suffix.lower() == ".csv":
        return "csv"
    if path.suffix.lower() in {".grib", ".grb", ".grib2"}:
        return "grib"
    return "netcdf"


def clean_number(value: float) -> str:
    """Write compact, locale-independent decimal values to CSV."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_features(path: Path, candidates: dict[EventKey, Candidate]) -> int:
    """Write only validated, resolved source features in a stable key order.

    An absent event key is deliberately not represented by a blank row.  This
    preserves a clear left-join contract for the downstream merger: a missing
    key means the source layer was unavailable, while every emitted row has
    ``era5_missing=0`` and usable physical values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for event, candidate in sorted(candidates.items(), key=lambda item: (item[0].timestamp, item[0].storm_id)):
            writer.writerow({
                "storm_id": event.storm_id,
                "timestamp_utc": format_timestamp(event.timestamp),
                "era5_u10_mps": clean_number(candidate.u10),
                "era5_v10_mps": clean_number(candidate.v10),
                "era5_wind10_mps": clean_number(math.hypot(candidate.u10, candidate.v10)),
                "era5_mslp_pa": clean_number(candidate.mslp),
                "era5_sst_k": clean_number(candidate.sst),
                "era5_tcw_kg_m2": clean_number(candidate.tcw),
                "era5_dewpoint_k": clean_number(candidate.dewpoint),
                "era5_missing": "0",
            })
    return len(candidates)


def build_contract() -> dict[str, object]:
    """Expose the stable source/output contracts in the manifest and dry run."""
    return {
        "output_csv_columns": OUTPUT_COLUMNS,
        "output_units": {
            "era5_u10_mps": "m s-1",
            "era5_v10_mps": "m s-1",
            "era5_wind10_mps": "m s-1; sqrt(u10^2 + v10^2)",
            "era5_mslp_pa": "Pa",
            "era5_sst_k": "K",
            "era5_tcw_kg_m2": "kg m-2",
            "era5_dewpoint_k": "K",
            "era5_missing": "0 for every emitted row",
        },
        "raw_csv_required_concepts": list(REQUIRED_SOURCE_CONCEPTS),
        "raw_csv_aliases": {name: list(aliases) for name, aliases in SOURCE_COLUMN_ALIASES.items()},
        "selection": "nearest grid point to the IBTrACS centre at the exact planned UTC time",
        "missing_data_behavior": (
            "Only fully valid, spatially matched source rows are emitted. Missing, invalid, or unmatched "
            "event keys are omitted; the downstream left join must treat an absent key as unavailable. "
            "If no raw ERA5 source is readable, the CSV contains only its header."
        ),
    }


def create_manifest(
    *,
    status: str,
    plan_path: Path,
    tracks_path: Path,
    raw_dir: Path,
    output_path: Path,
    events: list[EventKey],
    missing_track_events: list[str],
    raw_files: list[Path],
    file_results: list[FileResult],
    candidates: dict[EventKey, Candidate],
    counts: Counter[str],
    time_tolerance: timedelta,
    max_grid_distance_km: float,
    dry_run: bool,
) -> dict[str, object]:
    """Create compact provenance, quality, and missing-data metadata."""
    per_storm = Counter(event.storm_id for event in events)
    resolved_per_storm = Counter(event.storm_id for event in candidates)
    unresolved = len(events) - len(candidates)
    selected_distances = [candidate.distance_km for candidate in candidates.values()]
    return {
        "schema_version": 1,
        "status": status,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "source": "Copernicus Climate Data Store ERA5 hourly single levels",
        "contract": build_contract(),
        "inputs": {
            "collection_plan": {
                "path": project_relative(plan_path),
                "sha256": sha256_file(plan_path),
            },
            "tracks": {
                "path": project_relative(tracks_path),
                "sha256": sha256_file(tracks_path),
            },
            "raw_era5_directory": project_relative(raw_dir),
            "raw_files_discovered": len(raw_files),
            "raw_files": [result.as_dict() for result in file_results],
        },
        "alignment": {
            "event_keys_expected": len(events),
            "event_keys_by_storm": dict(sorted(per_storm.items())),
            "event_keys_without_track_position": len(missing_track_events),
            "missing_track_position_examples": missing_track_events[:10],
            "time_tolerance_minutes": int(time_tolerance.total_seconds() // 60),
            "max_grid_distance_km": max_grid_distance_km,
        },
        "output": {
            "path": project_relative(output_path),
            "records_written": len(candidates),
            "unresolved_event_keys": unresolved,
            "resolved_by_storm": dict(sorted(resolved_per_storm.items())),
            "selected_grid_distance_km": {
                "minimum": round(min(selected_distances), 4) if selected_distances else None,
                "maximum": round(max(selected_distances), 4) if selected_distances else None,
                "mean": round(sum(selected_distances) / len(selected_distances), 4) if selected_distances else None,
            },
        },
        "quality_counts": dict(sorted(counts.items())),
        "next_action": (
            "Run scripts/collect_era5.py after configuring CDS, then run this extractor again."
            if not raw_files
            else "Inspect raw file validation above, install optional decoders for GRIB/NetCDF if needed, then rerun."
        ),
    }


def print_dry_run(
    *,
    plan_path: Path,
    tracks_path: Path,
    raw_dir: Path,
    events: list[EventKey],
    missing_track_events: list[str],
    raw_files: list[Path],
    file_results: list[FileResult],
    time_tolerance: timedelta,
    max_grid_distance_km: float,
) -> None:
    """Emit a credential-free source-readiness report without writing artifacts."""
    payload = {
        "dry_run": True,
        "plan": project_relative(plan_path),
        "tracks": project_relative(tracks_path),
        "raw_era5_directory": project_relative(raw_dir),
        "event_keys_expected": len(events),
        "event_keys_without_track_position": len(missing_track_events),
        "raw_files_discovered": len(raw_files),
        "raw_file_readiness": [result.as_dict() for result in file_results],
        "time_tolerance_minutes": int(time_tolerance.total_seconds() // 60),
        "max_grid_distance_km": max_grid_distance_km,
        "contract": build_contract(),
    }
    print(json.dumps(payload, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract validated, cyclone-centred ERA5 features from downloaded local data."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN, help="Event-aligned collection plan JSON.")
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS, help="Normalized IBTrACS track CSV.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Downloaded ERA5 source directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output event feature CSV.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output provenance manifest JSON.")
    parser.add_argument(
        "--max-grid-distance-km",
        type=float,
        default=100.0,
        help="Reject a nearest ERA5 point farther from the recorded centre (default: 100).",
    )
    parser.add_argument(
        "--time-tolerance-minutes",
        type=int,
        default=0,
        help="Maximum difference from the planned UTC time; defaults to exact matching.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local plan/tracks and inspect source readiness without writing artifacts or contacting CDS.",
    )
    args = parser.parse_args()

    if args.max_grid_distance_km <= 0:
        parser.error("--max-grid-distance-km must be positive")
    if args.time_tolerance_minutes < 0:
        parser.error("--time-tolerance-minutes cannot be negative")

    plan_path = resolve_path(args.plan)
    tracks_path = resolve_path(args.tracks)
    raw_dir = resolve_path(args.raw_dir)
    output_path = resolve_path(args.output)
    manifest_path = resolve_path(args.manifest)
    time_tolerance = timedelta(minutes=args.time_tolerance_minutes)

    try:
        plan = load_plan(plan_path)
        events, missing_track_events = load_events(plan, tracks_path)
        if missing_track_events:
            raise InputValidationError(
                "The plan has timestamps absent from normalized tracks; examples: "
                + ", ".join(missing_track_events[:3])
            )
        if not events:
            raise InputValidationError("No event keys could be built from the plan and tracks")
        raw_files = discover_raw_files(raw_dir)
    except InputValidationError as error:
        parser.error(str(error))

    file_results: list[FileResult] = []
    for raw_file in raw_files:
        kind = source_kind(raw_file)
        result = FileResult(
            path=project_relative(raw_file),
            kind=kind,
            bytes=raw_file.stat().st_size,
            status="discovered",
        )
        if kind == "csv":
            try:
                _, missing = inspect_csv_header(raw_file)
                if missing:
                    result.status = "invalid_header"
                    result.missing_columns = missing
                else:
                    result.status = "ready"
            except (OSError, UnicodeError, csv.Error) as error:
                result.status = "unreadable"
                result.error = str(error)
        else:
            available, reason = xarray_availability(kind)
            result.status = "ready" if available else "decoder_unavailable"
            result.error = reason
        file_results.append(result)

    if args.dry_run:
        print_dry_run(
            plan_path=plan_path,
            tracks_path=tracks_path,
            raw_dir=raw_dir,
            events=events,
            missing_track_events=missing_track_events,
            raw_files=raw_files,
            file_results=file_results,
            time_tolerance=time_tolerance,
            max_grid_distance_km=args.max_grid_distance_km,
        )
        return 0

    events_by_time: dict[datetime, list[EventKey]] = defaultdict(list)
    for event in events:
        events_by_time[event.timestamp].append(event)
    target_times = sorted(events_by_time)
    candidates: dict[EventKey, Candidate] = {}
    counts: Counter[str] = Counter()

    for raw_file, result in zip(raw_files, file_results, strict=True):
        if result.status != "ready":
            continue
        kind = source_kind(raw_file)
        if kind == "csv":
            process_csv_file(
                raw_file,
                result,
                events_by_time=events_by_time,
                target_times=target_times,
                time_tolerance=time_tolerance,
                max_grid_distance_km=args.max_grid_distance_km,
                candidates=candidates,
                counts=counts,
            )
        else:
            process_xarray_file(
                raw_file,
                result,
                kind=kind,
                events_by_time=events_by_time,
                target_times=target_times,
                time_tolerance=time_tolerance,
                max_grid_distance_km=args.max_grid_distance_km,
                candidates=candidates,
                counts=counts,
            )

    records_written = write_features(output_path, candidates)
    readable_files = sum(1 for result in file_results if result.status == "read")
    if not raw_files:
        status = "waiting_for_raw_era5"
    elif records_written == len(events):
        status = "ready"
    elif records_written:
        status = "partial"
    elif readable_files:
        status = "waiting_for_event_aligned_era5"
    else:
        status = "waiting_for_valid_raw_era5"

    manifest = create_manifest(
        status=status,
        plan_path=plan_path,
        tracks_path=tracks_path,
        raw_dir=raw_dir,
        output_path=output_path,
        events=events,
        missing_track_events=missing_track_events,
        raw_files=raw_files,
        file_results=file_results,
        candidates=candidates,
        counts=counts,
        time_tolerance=time_tolerance,
        max_grid_distance_km=args.max_grid_distance_km,
        dry_run=False,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"ERA5 extraction {status}: wrote {records_written:,}/{len(events):,} resolved event feature row(s)."
    )
    print(f"Feature CSV: {project_relative(output_path)}")
    print(f"Manifest: {project_relative(manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
