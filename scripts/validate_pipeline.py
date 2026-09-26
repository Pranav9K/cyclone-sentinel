"""Validate the generated Cyclone Sentinel data and baseline artifacts.

The first project stages intentionally use plain CSV and JSON files.  This
validator checks that their schemas, row counts, cross-file references, and
core forecasting invariants still agree before a later collection or training
step consumes them.  It uses only the Python standard library.

Run from any directory with:

    python scripts/validate_pipeline.py

Use ``--require-complete-multisource`` only after ERA5 and IMERG features have
been collected; the initial placeholder multi-source table is valid with zero
coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRACK_COLUMNS = {
    "storm_id",
    "season",
    "name",
    "timestamp_utc",
    "latitude",
    "longitude",
    "wind_knots",
    "pressure_hpa",
    "status",
    "source",
}
TRAINING_COLUMNS = {
    "sample_id",
    "storm_id",
    "input_timestamp_utc",
    "season",
    "input_year",
    "input_month",
    "input_day_of_year",
    "input_hour_utc",
    "latitude",
    "longitude",
    "wind_knots",
    "pressure_hpa",
    "status",
    "previous_6h_available",
    "previous_6h_delta_latitude",
    "previous_6h_delta_longitude",
    "previous_6h_wind_change_knots",
    "target_timestamp_utc",
    "target_interval_hours",
    "target_latitude",
    "target_longitude",
    "target_delta_latitude_24h",
    "target_delta_longitude_24h",
    "target_wind_knots",
    "target_wind_change_knots_24h",
    "split",
}
VALID_SPLITS = {"train", "validation", "test"}
MAX_REPORTED_ERRORS = 40


@dataclass(frozen=True)
class TrackObservation:
    """Fields needed to cross-check downstream artifacts."""

    latitude: float
    longitude: float
    wind_knots: float | None
    pressure_hpa: float | None


@dataclass
class TracksInfo:
    path: Path
    digest: str
    records: int
    by_storm_time: dict[tuple[str, datetime], TrackObservation]
    storm_ids: set[str]


@dataclass
class TrainingInfo:
    path: Path
    records: int
    sample_ids: set[str]
    sample_storms: dict[str, str]


class ValidationResult:
    """Collect actionable failures while allowing independent checks to run."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self._suppressed_errors = 0
        self.summary: dict[str, Any] = {}

    def error(self, message: str) -> None:
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(message)
        else:
            self._suppressed_errors += 1

    def require(self, condition: bool, message: str) -> bool:
        if not condition:
            self.error(message)
            return False
        return True

    @property
    def ok(self) -> bool:
        return not self.errors and self._suppressed_errors == 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str, result: ValidationResult) -> dict[str, Any] | None:
    if not path.is_file():
        result.error(f"{label}: missing file at {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        result.error(f"{label}: cannot parse JSON at {path}: {error}")
        return None
    if not isinstance(payload, dict):
        result.error(f"{label}: top-level JSON value must be an object")
        return None
    return payload


def as_mapping(value: Any, label: str, result: ValidationResult) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        result.error(f"{label}: expected an object")
        return None
    return value


def as_list(value: Any, label: str, result: ValidationResult) -> list[Any] | None:
    if not isinstance(value, list):
        result.error(f"{label}: expected a list")
        return None
    return value


def finite_number(value: Any, label: str, result: ValidationResult) -> float | None:
    if isinstance(value, bool):
        result.error(f"{label}: expected a finite number, got boolean")
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        result.error(f"{label}: expected a finite number, got {value!r}")
        return None
    if not math.isfinite(number):
        result.error(f"{label}: value must be finite, got {value!r}")
        return None
    return number


def integer(value: Any, label: str, result: ValidationResult) -> int | None:
    if isinstance(value, bool):
        result.error(f"{label}: expected an integer, got boolean")
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        result.error(f"{label}: expected an integer, got {value!r}")
        return None
    if str(parsed) != str(value).strip() and not (isinstance(value, int) and not isinstance(value, bool)):
        result.error(f"{label}: expected an integer, got {value!r}")
        return None
    return parsed


def parse_time(value: Any, label: str, result: ValidationResult) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        result.error(f"{label}: missing timestamp")
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        result.error(f"{label}: invalid ISO timestamp {value!r}")
        return None
    if parsed.tzinfo is None:
        # The normalized IBTrACS CSV deliberately stores UTC without an offset.
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def bool_text(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def resolve_manifest_path(project_root: Path, value: Any, label: str, result: ValidationResult) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        result.error(f"{label}: missing path")
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def same_path(left: Path, right: Path) -> bool:
    """Compare expected artifact paths without requiring either to exist."""
    return left.resolve(strict=False) == right.resolve(strict=False)


def csv_reader(path: Path, label: str, result: ValidationResult) -> tuple[csv.DictReader, Any] | None:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        result.error(f"{label}: cannot read {path}: {error}")
        return None
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        handle.close()
        result.error(f"{label}: CSV has no header row")
        return None
    return reader, handle


def validate_tracks(
    processed_dir: Path,
    result: ValidationResult,
) -> TracksInfo | None:
    """Validate the normalized IBTrACS manifest and its compact track CSV."""
    label = "IBTrACS"
    manifest_path = processed_dir / "ibtracs_ni_manifest.json"
    tracks_path = processed_dir / "ibtracs_ni_tracks.csv"
    manifest = load_json(manifest_path, f"{label} manifest", result)
    if manifest is None:
        return None

    result.require(manifest.get("status") == "ready", f"{label} manifest: status must be 'ready'")
    start_year = integer(manifest.get("start_year"), f"{label} manifest.start_year", result)
    record_count = integer(manifest.get("records"), f"{label} manifest.records", result)
    result.require(isinstance(manifest.get("source"), str) and bool(manifest["source"].strip()), f"{label} manifest.source: missing source description")
    result.require(isinstance(manifest.get("source_url"), str) and manifest["source_url"].startswith(("http://", "https://")), f"{label} manifest.source_url: expected an HTTP(S) URL")
    processed_reference = resolve_manifest_path(PROJECT_ROOT, manifest.get("processed_path"), f"{label} manifest.processed_path", result)
    if processed_reference is not None:
        result.require(same_path(processed_reference, tracks_path), f"{label} manifest.processed_path points to {processed_reference}, expected {tracks_path}")
    if start_year is not None:
        result.require(1800 <= start_year <= 2200, f"{label} manifest.start_year must be between 1800 and 2200")
    if record_count is not None:
        result.require(record_count >= 0, f"{label} manifest.records must not be negative")

    if not tracks_path.is_file():
        result.error(f"{label} tracks: missing file at {tracks_path}; run scripts/collect_ibtracs.py")
        return None
    reader_and_handle = csv_reader(tracks_path, f"{label} tracks", result)
    if reader_and_handle is None:
        return None
    reader, handle = reader_and_handle
    try:
        header = set(reader.fieldnames or [])
        missing = sorted(TRACK_COLUMNS - header)
        if missing:
            result.error(f"{label} tracks: missing required column(s): {', '.join(missing)}")
            return None

        by_storm_time: dict[tuple[str, datetime], TrackObservation] = {}
        storm_ids: set[str] = set()
        rows = 0
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            context = f"{label} tracks:{row_number}"
            storm_id = (row.get("storm_id") or "").strip()
            if not storm_id:
                result.error(f"{context}: storm_id is blank")
                continue
            season = integer(row.get("season"), f"{context}.season", result)
            timestamp = parse_time(row.get("timestamp_utc"), f"{context}.timestamp_utc", result)
            latitude = finite_number(row.get("latitude"), f"{context}.latitude", result)
            longitude = finite_number(row.get("longitude"), f"{context}.longitude", result)
            if season is not None and not 1800 <= season <= 2200:
                result.error(f"{context}.season must be between 1800 and 2200")
            if start_year is not None and season is not None and season < start_year:
                result.error(f"{context}.season is before manifest start_year {start_year}")
            if latitude is not None and not -90 <= latitude <= 90:
                result.error(f"{context}.latitude must be between -90 and 90")
            if longitude is not None and not -360 <= longitude <= 360:
                result.error(f"{context}.longitude must be between -360 and 360")
            wind_text = (row.get("wind_knots") or "").strip()
            wind = finite_number(wind_text, f"{context}.wind_knots", result) if wind_text else None
            if wind is not None and not 0 <= wind <= 250:
                result.error(f"{context}.wind_knots must be between 0 and 250")
            pressure_text = (row.get("pressure_hpa") or "").strip()
            pressure = finite_number(pressure_text, f"{context}.pressure_hpa", result) if pressure_text else None
            if pressure is not None and not 800 <= pressure <= 1100:
                result.error(f"{context}.pressure_hpa must be between 800 and 1100")
            if not (row.get("status") or "").strip():
                result.error(f"{context}.status is blank")
            if not (row.get("source") or "").strip():
                result.error(f"{context}.source is blank")
            if timestamp is None or latitude is None or longitude is None:
                continue
            key = (storm_id, timestamp)
            if key in by_storm_time:
                result.error(f"{context}: duplicate storm/timestamp key {storm_id} {timestamp.isoformat()}")
                continue
            by_storm_time[key] = TrackObservation(latitude, longitude, wind, pressure)
            storm_ids.add(storm_id)
    finally:
        handle.close()

    if record_count is not None and rows != record_count:
        result.error(f"{label} tracks: found {rows:,} rows but manifest.records says {record_count:,}; rerun scripts/collect_ibtracs.py")
    if rows == 0:
        result.error(f"{label} tracks: CSV has no observations")
        return None
    result.summary["tracks"] = {"records": rows, "storms": len(storm_ids)}
    return TracksInfo(
        path=tracks_path,
        digest=sha256_file(tracks_path),
        records=rows,
        by_storm_time=by_storm_time,
        storm_ids=storm_ids,
    )


def validate_training(
    processed_dir: Path,
    tracks: TracksInfo | None,
    result: ValidationResult,
) -> TrainingInfo | None:
    """Validate supervised samples and their reproducibility manifest."""
    label = "Training samples"
    manifest_path = processed_dir / "training_samples_manifest.json"
    samples_path = processed_dir / "training_samples.csv"
    manifest = load_json(manifest_path, f"{label} manifest", result)
    if manifest is None:
        return None
    dataset = as_mapping(manifest.get("dataset"), f"{label} manifest.dataset", result)
    source = as_mapping(manifest.get("source"), f"{label} manifest.source", result)
    configuration = as_mapping(manifest.get("configuration"), f"{label} manifest.configuration", result)
    quality = as_mapping(manifest.get("quality"), f"{label} manifest.quality", result)
    if manifest.get("status") != "ready":
        result.error(f"{label} manifest: status must be 'ready'")
    if manifest.get("schema_version") != 1:
        result.error(f"{label} manifest.schema_version must be 1")

    horizon: int | None = None
    allowed_statuses: set[str] = set()
    if configuration is not None:
        horizon = integer(configuration.get("exact_horizon_hours"), f"{label} manifest.configuration.exact_horizon_hours", result)
        if horizon is not None and horizon <= 0:
            result.error(f"{label} manifest.configuration.exact_horizon_hours must be positive")
        statuses = as_list(configuration.get("allowed_statuses"), f"{label} manifest.configuration.allowed_statuses", result)
        if statuses is not None:
            allowed_statuses = {item.strip() for item in statuses if isinstance(item, str) and item.strip()}
            if not allowed_statuses:
                result.error(f"{label} manifest.configuration.allowed_statuses must contain at least one status")
        fractions = []
        for name in ("train_fraction", "validation_fraction", "test_fraction"):
            value = finite_number(configuration.get(name), f"{label} manifest.configuration.{name}", result)
            if value is not None:
                fractions.append(value)
                if not 0 <= value <= 1:
                    result.error(f"{label} manifest.configuration.{name} must be between 0 and 1")
        if len(fractions) == 3 and not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            result.error(f"{label} manifest configuration split fractions sum to {sum(fractions)}, expected 1")

    if source is not None and tracks is not None:
        source_path = resolve_manifest_path(PROJECT_ROOT, source.get("path"), f"{label} manifest.source.path", result)
        if source_path is not None:
            result.require(same_path(source_path, tracks.path), f"{label} manifest.source.path points to {source_path}, expected {tracks.path}")
        if source.get("sha256") != tracks.digest:
            result.error(f"{label} manifest.source.sha256 does not match ibtracs_ni_tracks.csv; rerun scripts/prepare_training_data.py")
        rows_read = integer(source.get("rows_read"), f"{label} manifest.source.rows_read", result)
        if rows_read is not None and rows_read != tracks.records:
            result.error(f"{label} manifest.source.rows_read is {rows_read:,}, but IBTrACS has {tracks.records:,} rows")

    dataset_records: int | None = None
    expected_columns: list[str] | None = None
    split_summary: dict[str, Any] | None = None
    if dataset is not None:
        dataset_reference = resolve_manifest_path(PROJECT_ROOT, dataset.get("path"), f"{label} manifest.dataset.path", result)
        if dataset_reference is not None:
            result.require(same_path(dataset_reference, samples_path), f"{label} manifest.dataset.path points to {dataset_reference}, expected {samples_path}")
        dataset_records = integer(dataset.get("records"), f"{label} manifest.dataset.records", result)
        columns = as_list(dataset.get("columns"), f"{label} manifest.dataset.columns", result)
        if columns is not None and all(isinstance(column, str) for column in columns):
            expected_columns = [str(column) for column in columns]
            missing = sorted(TRAINING_COLUMNS - set(expected_columns))
            if missing:
                result.error(f"{label} manifest.dataset.columns is missing required column(s): {', '.join(missing)}")
        elif columns is not None:
            result.error(f"{label} manifest.dataset.columns must contain only strings")
        split_summary = as_mapping(dataset.get("split_summary"), f"{label} manifest.dataset.split_summary", result)

    if not samples_path.is_file():
        result.error(f"{label}: missing file at {samples_path}; run scripts/prepare_training_data.py")
        return None
    reader_and_handle = csv_reader(samples_path, label, result)
    if reader_and_handle is None:
        return None
    reader, handle = reader_and_handle
    sample_ids: set[str] = set()
    sample_storms: dict[str, str] = {}
    split_rows: Counter[str] = Counter()
    split_storms: dict[str, set[str]] = defaultdict(set)
    split_times: dict[str, list[datetime]] = defaultdict(list)
    rows = 0
    try:
        header_list = reader.fieldnames or []
        header = set(header_list)
        missing = sorted(TRAINING_COLUMNS - header)
        if missing:
            result.error(f"{label}: CSV is missing required column(s): {', '.join(missing)}")
            return None
        if expected_columns is not None and header_list != expected_columns:
            result.error(f"{label}: CSV header differs from manifest.dataset.columns; rerun scripts/prepare_training_data.py")

        for row_number, row in enumerate(reader, start=2):
            rows += 1
            context = f"{label}:{row_number}"
            sample_id = (row.get("sample_id") or "").strip()
            storm_id = (row.get("storm_id") or "").strip()
            if not sample_id:
                result.error(f"{context}.sample_id is blank")
            elif sample_id in sample_ids:
                result.error(f"{context}: duplicate sample_id {sample_id}")
            else:
                sample_ids.add(sample_id)
                sample_storms[sample_id] = storm_id
            if not storm_id:
                result.error(f"{context}.storm_id is blank")
            elif tracks is not None and storm_id not in tracks.storm_ids:
                result.error(f"{context}.storm_id {storm_id!r} is not present in IBTrACS tracks")

            input_time = parse_time(row.get("input_timestamp_utc"), f"{context}.input_timestamp_utc", result)
            target_time = parse_time(row.get("target_timestamp_utc"), f"{context}.target_timestamp_utc", result)
            target_interval = integer(row.get("target_interval_hours"), f"{context}.target_interval_hours", result)
            if horizon is not None and target_interval is not None and target_interval != horizon:
                result.error(f"{context}.target_interval_hours is {target_interval}, expected manifest horizon {horizon}")
            if horizon is not None and input_time is not None and target_time is not None:
                actual_hours = (target_time - input_time).total_seconds() / 3600
                if not math.isclose(actual_hours, horizon, abs_tol=1e-9):
                    result.error(f"{context}: target timestamp is {actual_hours:g}h after input, expected {horizon}h")

            latitude = finite_number(row.get("latitude"), f"{context}.latitude", result)
            longitude = finite_number(row.get("longitude"), f"{context}.longitude", result)
            target_latitude = finite_number(row.get("target_latitude"), f"{context}.target_latitude", result)
            target_longitude = finite_number(row.get("target_longitude"), f"{context}.target_longitude", result)
            wind = finite_number(row.get("wind_knots"), f"{context}.wind_knots", result)
            target_wind = finite_number(row.get("target_wind_knots"), f"{context}.target_wind_knots", result)
            # Pressure is a useful feature but deliberately optional in the
            # source contract; the preparation stage retains those samples.
            pressure_text = (row.get("pressure_hpa") or "").strip()
            pressure = finite_number(pressure_text, f"{context}.pressure_hpa", result) if pressure_text else None
            for name, value, low, high in (
                ("latitude", latitude, -90, 90),
                ("target_latitude", target_latitude, -90, 90),
                ("longitude", longitude, -180, 180),
                ("target_longitude", target_longitude, -180, 180),
                ("wind_knots", wind, 0, 250),
                ("target_wind_knots", target_wind, 0, 250),
                ("pressure_hpa", pressure, 800, 1100),
            ):
                if value is not None and not low <= value <= high:
                    result.error(f"{context}.{name} must be between {low} and {high}")
            for name in (
                "target_delta_latitude_24h",
                "target_delta_longitude_24h",
                "target_wind_change_knots_24h",
                "previous_6h_delta_latitude",
                "previous_6h_delta_longitude",
                "previous_6h_wind_change_knots",
            ):
                text = (row.get(name) or "").strip()
                if text:
                    finite_number(text, f"{context}.{name}", result)
            available = integer(row.get("previous_6h_available"), f"{context}.previous_6h_available", result)
            if available is not None and available not in {0, 1}:
                result.error(f"{context}.previous_6h_available must be 0 or 1")
            status = (row.get("status") or "").strip()
            if allowed_statuses and status not in allowed_statuses:
                result.error(f"{context}.status {status!r} is not in the manifest allowed_statuses")
            split = (row.get("split") or "").strip()
            if split not in VALID_SPLITS:
                result.error(f"{context}.split must be one of {', '.join(sorted(VALID_SPLITS))}")
            else:
                split_rows[split] += 1
                if storm_id:
                    split_storms[split].add(storm_id)
                if input_time is not None:
                    split_times[split].append(input_time)

            if tracks is not None and storm_id and input_time is not None:
                observation = tracks.by_storm_time.get((storm_id, input_time))
                if observation is None:
                    result.error(f"{context}: input storm/timestamp is absent from IBTrACS tracks")
                elif latitude is not None and not math.isclose(latitude, observation.latitude, abs_tol=1e-6):
                    result.error(f"{context}.latitude does not match its IBTrACS input observation")
            if tracks is not None and storm_id and target_time is not None:
                target_observation = tracks.by_storm_time.get((storm_id, target_time))
                if target_observation is None:
                    result.error(f"{context}: target storm/timestamp is absent from IBTrACS tracks")
                elif target_latitude is not None and not math.isclose(target_latitude, target_observation.latitude, abs_tol=1e-6):
                    result.error(f"{context}.target_latitude does not match its IBTrACS target observation")
    finally:
        handle.close()

    if rows == 0:
        result.error(f"{label}: CSV has no supervised samples")
        return None
    if dataset_records is not None and rows != dataset_records:
        result.error(f"{label}: found {rows:,} rows but manifest.dataset.records says {dataset_records:,}")
    if quality is not None:
        sample_counts = as_mapping(quality.get("sample_counts"), f"{label} manifest.quality.sample_counts", result)
        if sample_counts is not None:
            emitted = integer(sample_counts.get("emitted_samples"), f"{label} manifest.quality.sample_counts.emitted_samples", result)
            if emitted is not None and emitted != rows:
                result.error(f"{label}: manifest quality emitted_samples is {emitted:,}, but CSV has {rows:,} rows")
    storm_split_memberships: dict[str, set[str]] = defaultdict(set)
    for split, storm_set in split_storms.items():
        for storm_id in storm_set:
            storm_split_memberships[storm_id].add(split)
    leaked_storms = sorted(storm for storm, memberships in storm_split_memberships.items() if len(memberships) > 1)
    if leaked_storms:
        preview = ", ".join(leaked_storms[:5])
        suffix = "" if len(leaked_storms) <= 5 else f" (+{len(leaked_storms) - 5} more)"
        result.error(f"{label}: storm-disjoint split violated by {preview}{suffix}")
    if split_summary is not None:
        for split in VALID_SPLITS:
            summary = as_mapping(split_summary.get(split), f"{label} manifest.dataset.split_summary.{split}", result)
            if summary is None:
                continue
            expected_samples = integer(summary.get("samples"), f"{label} manifest.dataset.split_summary.{split}.samples", result)
            expected_storms = integer(summary.get("storms"), f"{label} manifest.dataset.split_summary.{split}.storms", result)
            if expected_samples is not None and expected_samples != split_rows[split]:
                result.error(f"{label}: {split} sample count is {split_rows[split]:,}, manifest says {expected_samples:,}")
            if expected_storms is not None and expected_storms != len(split_storms[split]):
                result.error(f"{label}: {split} storm count is {len(split_storms[split]):,}, manifest says {expected_storms:,}")
            if split_times[split]:
                first = min(split_times[split]).isoformat().replace("+00:00", "Z")
                last = max(split_times[split]).isoformat().replace("+00:00", "Z")
                if summary.get("first_input_timestamp_utc") != first:
                    result.error(f"{label}: {split} first input timestamp differs from manifest")
                if summary.get("last_input_timestamp_utc") != last:
                    result.error(f"{label}: {split} last input timestamp differs from manifest")
    result.summary["training"] = {
        "records": rows,
        "splits": {split: split_rows[split] for split in ("train", "validation", "test")},
    }
    return TrainingInfo(path=samples_path, records=rows, sample_ids=sample_ids, sample_storms=sample_storms)


def validate_regression_model(model: Any, label: str, result: ValidationResult) -> None:
    """Validate a serialized standardized ridge-regression component."""
    payload = as_mapping(model, label, result)
    if payload is None:
        return
    if payload.get("type") != "ridge_linear_regression":
        result.error(f"{label}.type must be 'ridge_linear_regression'")
    alpha = finite_number(payload.get("alpha"), f"{label}.alpha", result)
    if alpha is not None and alpha < 0:
        result.error(f"{label}.alpha must not be negative")
    names = as_list(payload.get("feature_names"), f"{label}.feature_names", result)
    means = as_list(payload.get("feature_means"), f"{label}.feature_means", result)
    scales = as_list(payload.get("feature_scales"), f"{label}.feature_scales", result)
    coefficients = as_list(payload.get("coefficients_standardized"), f"{label}.coefficients_standardized", result)
    if names is None or means is None or scales is None or coefficients is None:
        return
    if not names or not all(isinstance(name, str) and name for name in names):
        result.error(f"{label}.feature_names must be a non-empty string list")
    width = len(names)
    if len(means) != width or len(scales) != width or len(coefficients) != width + 1:
        result.error(f"{label}: expected {width} means/scales and {width + 1} coefficients")
    for index, value in enumerate(means):
        finite_number(value, f"{label}.feature_means[{index}]", result)
    for index, value in enumerate(scales):
        scale = finite_number(value, f"{label}.feature_scales[{index}]", result)
        if scale is not None and scale <= 0:
            result.error(f"{label}.feature_scales[{index}] must be positive")
    for index, value in enumerate(coefficients):
        finite_number(value, f"{label}.coefficients_standardized[{index}]", result)


def validate_baseline(
    processed_dir: Path,
    tracks: TracksInfo | None,
    training_horizon: int | None,
    result: ValidationResult,
) -> None:
    """Validate metrics and serialized baseline model against their contract."""
    metrics = load_json(processed_dir / "baseline_metrics.json", "Baseline metrics", result)
    model = load_json(processed_dir / "baseline_model.json", "Baseline model", result)
    if metrics is None or model is None:
        return
    horizon_metrics = integer(metrics.get("forecast_horizon_hours"), "Baseline metrics.forecast_horizon_hours", result)
    horizon_model = integer(model.get("forecast_horizon_hours"), "Baseline model.forecast_horizon_hours", result)
    if horizon_metrics is not None and horizon_model is not None and horizon_metrics != horizon_model:
        result.error("Baseline metrics/model forecast horizons differ")
    if training_horizon is not None and horizon_metrics is not None and training_horizon != horizon_metrics:
        result.error(f"Baseline horizon is {horizon_metrics}h but training-sample horizon is {training_horizon}h")
    for payload, label in ((metrics, "Baseline metrics"), (model, "Baseline model")):
        if not isinstance(payload.get("model_note"), str) or not payload["model_note"].strip():
            result.error(f"{label}.model_note is missing")
        if tracks is not None:
            input_path = resolve_manifest_path(PROJECT_ROOT, payload.get("input_path"), f"{label}.input_path", result)
            if input_path is not None:
                result.require(same_path(input_path, tracks.path), f"{label}.input_path points to {input_path}, expected {tracks.path}")
    if metrics.get("split") != model.get("split"):
        result.error("Baseline metrics/model split metadata differs")
    split = as_mapping(metrics.get("split"), "Baseline metrics.split", result)
    if split is not None:
        train_samples = integer(split.get("train_samples"), "Baseline metrics.split.train_samples", result)
        test_samples = integer(split.get("test_samples"), "Baseline metrics.split.test_samples", result)
        if train_samples is not None and train_samples <= 0:
            result.error("Baseline metrics.split.train_samples must be positive")
        if test_samples is not None and test_samples <= 0:
            result.error("Baseline metrics.split.test_samples must be positive")
        parse_time(split.get("cutoff_utc"), "Baseline metrics.split.cutoff_utc", result)
    test_metrics = as_mapping(metrics.get("test_metrics"), "Baseline metrics.test_metrics", result)
    track_mae: float | None = None
    wind_mae: float | None = None
    if test_metrics is not None:
        for task, fields in {
            "track": ("endpoint_mae_km", "endpoint_median_km", "endpoint_p90_km", "latitude_mae_degrees", "longitude_delta_mae_degrees", "motion_persistence_endpoint_mae_km", "samples"),
            "intensity": ("persistence_wind_mae_knots", "wind_bias_knots", "wind_mae_knots", "wind_rmse_knots", "samples"),
        }.items():
            values = as_mapping(test_metrics.get(task), f"Baseline metrics.test_metrics.{task}", result)
            if values is None:
                continue
            for field in fields:
                number = finite_number(values.get(field), f"Baseline metrics.test_metrics.{task}.{field}", result)
                if number is not None and field != "wind_bias_knots" and number < 0:
                    result.error(f"Baseline metrics.test_metrics.{task}.{field} must not be negative")
            if task == "track":
                track_mae = finite_number(values.get("endpoint_mae_km"), "Baseline metrics.test_metrics.track.endpoint_mae_km", result)
            else:
                wind_mae = finite_number(values.get("wind_mae_knots"), "Baseline metrics.test_metrics.intensity.wind_mae_knots", result)
    model_track = as_mapping(model.get("track"), "Baseline model.track", result)
    if model_track is not None:
        validate_regression_model(model_track.get("latitude_delta_degrees"), "Baseline model.track.latitude_delta_degrees", result)
        validate_regression_model(model_track.get("longitude_delta_degrees"), "Baseline model.track.longitude_delta_degrees", result)
    model_intensity = as_mapping(model.get("intensity"), "Baseline model.intensity", result)
    if model_intensity is not None:
        validate_regression_model(model_intensity.get("wind_delta_knots"), "Baseline model.intensity.wind_delta_knots", result)
    result.summary["baseline"] = {"horizon": horizon_metrics, "track_mae": track_mae, "wind_mae": wind_mae}


def validate_collection_plan(
    processed_dir: Path,
    tracks: TracksInfo | None,
    result: ValidationResult,
) -> set[str]:
    """Check every planned storm/time step against the normalized track data."""
    label = "Multi-source collection plan"
    payload = load_json(processed_dir / "multisource_collection_plan.json", label, result)
    if payload is None:
        return set()
    if payload.get("schema_version") != 1:
        result.error(f"{label}.schema_version must be 1")
    buffer = finite_number(payload.get("buffer_degrees"), f"{label}.buffer_degrees", result)
    if buffer is not None and buffer <= 0:
        result.error(f"{label}.buffer_degrees must be positive")
    storm_entries = as_list(payload.get("storms"), f"{label}.storms", result)
    if not storm_entries:
        result.error(f"{label}.storms must contain at least one entry")
        return set()
    storm_ids: set[str] = set()
    total_observations = 0
    for index, entry in enumerate(storm_entries):
        context = f"{label}.storms[{index}]"
        storm = as_mapping(entry, context, result)
        if storm is None:
            continue
        storm_id = storm.get("storm_id")
        if not isinstance(storm_id, str) or not storm_id.strip():
            result.error(f"{context}.storm_id is missing")
            continue
        if storm_id in storm_ids:
            result.error(f"{context}.storm_id {storm_id!r} is duplicated")
            continue
        storm_ids.add(storm_id)
        if tracks is not None and storm_id not in tracks.storm_ids:
            result.error(f"{context}.storm_id {storm_id!r} is absent from IBTrACS tracks")
        observations = integer(storm.get("observations"), f"{context}.observations", result)
        time_steps = as_list(storm.get("time_steps_utc"), f"{context}.time_steps_utc", result)
        area = as_list(storm.get("era5_area_nwse"), f"{context}.era5_area_nwse", result)
        start = parse_time(storm.get("start_utc"), f"{context}.start_utc", result)
        end = parse_time(storm.get("end_utc"), f"{context}.end_utc", result)
        if start is not None and end is not None and start > end:
            result.error(f"{context}: start_utc is after end_utc")
        parsed_steps: list[datetime] = []
        if time_steps is not None:
            for time_index, value in enumerate(time_steps):
                parsed = parse_time(value, f"{context}.time_steps_utc[{time_index}]", result)
                if parsed is not None:
                    parsed_steps.append(parsed)
            if parsed_steps != sorted(parsed_steps) or len(set(parsed_steps)) != len(parsed_steps):
                result.error(f"{context}.time_steps_utc must be strictly increasing without duplicates")
        if observations is not None and observations != len(parsed_steps):
            result.error(f"{context}.observations is {observations}, but time_steps_utc has {len(parsed_steps)} entries")
        if parsed_steps:
            if start is not None and start != parsed_steps[0]:
                result.error(f"{context}.start_utc does not equal the first planned time step")
            if end is not None and end != parsed_steps[-1]:
                result.error(f"{context}.end_utc does not equal the final planned time step")
        if area is None or len(area) != 4:
            result.error(f"{context}.era5_area_nwse must contain [north, west, south, east]")
        else:
            north, west, south, east = [finite_number(value, f"{context}.era5_area_nwse[{position}]", result) for position, value in enumerate(area)]
            if None not in (north, west, south, east):
                assert north is not None and west is not None and south is not None and east is not None
                if not -90 <= south <= north <= 90:
                    result.error(f"{context}.era5_area_nwse has invalid north/south bounds")
                if not -180 <= west <= east <= 180:
                    result.error(f"{context}.era5_area_nwse has invalid west/east bounds")
        if tracks is not None:
            planned_observations: list[TrackObservation] = []
            for step in parsed_steps:
                observation = tracks.by_storm_time.get((storm_id, step))
                if observation is None:
                    result.error(f"{context}: planned time {step.isoformat()} is absent from IBTrACS tracks")
                else:
                    planned_observations.append(observation)
            if area is not None and len(area) == 4 and planned_observations:
                numbers = [finite_number(value, f"{context}.era5_area_nwse[{position}]", result) for position, value in enumerate(area)]
                if all(value is not None for value in numbers):
                    north, west, south, east = numbers  # type: ignore[misc]
                    for observation in planned_observations:
                        if not (south <= observation.latitude <= north and west <= observation.longitude <= east):
                            result.error(f"{context}.era5_area_nwse does not enclose every planned IBTrACS point")
                            break
        total_observations += len(parsed_steps)
    summary = as_mapping(payload.get("summary"), f"{label}.summary", result)
    if summary is not None:
        summary_storms = integer(summary.get("storms"), f"{label}.summary.storms", result)
        summary_observations = integer(summary.get("observations"), f"{label}.summary.observations", result)
        if summary_storms is not None and summary_storms != len(storm_ids):
            result.error(f"{label}.summary.storms is {summary_storms}, expected {len(storm_ids)}")
        if summary_observations is not None and summary_observations != total_observations:
            result.error(f"{label}.summary.observations is {summary_observations}, expected {total_observations}")
    result.summary["plan"] = {"storms": len(storm_ids), "observations": total_observations}
    return storm_ids


def validate_multisource(
    processed_dir: Path,
    plan_storms: set[str],
    training: TrainingInfo | None,
    result: ValidationResult,
    require_complete: bool,
) -> None:
    """Validate the joined multi-source CSV and its availability accounting."""
    label = "Multi-source training"
    manifest_path = processed_dir / "multisource_training_manifest.json"
    dataset_path = processed_dir / "multisource_training.csv"
    manifest = load_json(manifest_path, f"{label} manifest", result)
    if manifest is None:
        return
    if manifest.get("schema_version") != 1:
        result.error(f"{label} manifest.schema_version must be 1")
    expected_cohort = integer(manifest.get("cohort_storms"), f"{label} manifest.cohort_storms", result)
    if expected_cohort is not None and expected_cohort != len(plan_storms):
        result.error(f"{label} manifest.cohort_storms is {expected_cohort}, but collection plan has {len(plan_storms)} storms")
    output_reference = resolve_manifest_path(PROJECT_ROOT, manifest.get("output"), f"{label} manifest.output", result)
    if output_reference is not None:
        result.require(same_path(output_reference, dataset_path), f"{label} manifest.output points to {output_reference}, expected {dataset_path}")
    expected_columns_list = as_list(manifest.get("columns"), f"{label} manifest.columns", result)
    expected_columns = [value for value in expected_columns_list if isinstance(value, str)] if expected_columns_list is not None else None
    if expected_columns_list is not None and expected_columns is not None and len(expected_columns) != len(expected_columns_list):
        result.error(f"{label} manifest.columns must contain only strings")
    counts_manifest = as_mapping(manifest.get("counts"), f"{label} manifest.counts", result)
    if not dataset_path.is_file():
        result.error(f"{label}: missing file at {dataset_path}; run scripts/build_multisource_dataset.py")
        return
    reader_and_handle = csv_reader(dataset_path, label, result)
    if reader_and_handle is None:
        return
    reader, handle = reader_and_handle
    counts: Counter[str] = Counter()
    observed_storms: set[str] = set()
    try:
        header = reader.fieldnames or []
        required = {"sample_id", "storm_id", "input_timestamp_utc", "era5_available", "imerg_available", "multisource_complete"}
        missing = sorted(required - set(header))
        if missing:
            result.error(f"{label}: CSV is missing required column(s): {', '.join(missing)}")
            return
        if expected_columns is not None and header != expected_columns:
            result.error(f"{label}: CSV header differs from manifest.columns; rerun scripts/build_multisource_dataset.py")
        for row_number, row in enumerate(reader, start=2):
            context = f"{label}:{row_number}"
            counts["rows"] += 1
            sample_id = (row.get("sample_id") or "").strip()
            storm_id = (row.get("storm_id") or "").strip()
            if not sample_id:
                result.error(f"{context}.sample_id is blank")
            elif training is not None:
                source_storm = training.sample_storms.get(sample_id)
                if source_storm is None:
                    result.error(f"{context}.sample_id {sample_id!r} is not present in training_samples.csv")
                elif source_storm != storm_id:
                    result.error(f"{context}.storm_id does not match the source training sample")
            if not storm_id:
                result.error(f"{context}.storm_id is blank")
            elif plan_storms and storm_id not in plan_storms:
                result.error(f"{context}.storm_id {storm_id!r} is not in the collection plan cohort")
            else:
                observed_storms.add(storm_id)
            parse_time(row.get("input_timestamp_utc"), f"{context}.input_timestamp_utc", result)
            era5 = bool_text(row.get("era5_available"))
            imerg = bool_text(row.get("imerg_available"))
            complete = bool_text(row.get("multisource_complete"))
            if era5 is None:
                result.error(f"{context}.era5_available must be 'true' or 'false'")
            if imerg is None:
                result.error(f"{context}.imerg_available must be 'true' or 'false'")
            if complete is None:
                result.error(f"{context}.multisource_complete must be 'true' or 'false'")
            if era5 is not None:
                counts["era5_available"] += int(era5)
            if imerg is not None:
                counts["imerg_available"] += int(imerg)
            if complete is not None:
                counts["complete"] += int(complete)
            if None not in (era5, imerg, complete) and complete != (era5 and imerg):
                result.error(f"{context}.multisource_complete must equal era5_available AND imerg_available")
    finally:
        handle.close()
    if counts["rows"] == 0:
        result.error(f"{label}: CSV has no cohort samples")
    if counts_manifest is not None:
        for key in ("rows", "era5_available", "imerg_available", "complete"):
            expected = integer(counts_manifest.get(key), f"{label} manifest.counts.{key}", result)
            if expected is not None and expected != counts[key]:
                result.error(f"{label}: {key} count is {counts[key]:,}, manifest says {expected:,}")
    if require_complete and counts["complete"] != counts["rows"]:
        result.error(
            f"{label}: complete multi-source coverage is {counts['complete']:,}/{counts['rows']:,}; "
            "collect ERA5 and IMERG features before using --require-complete-multisource"
        )
    result.summary["multisource"] = {
        "rows": counts["rows"],
        "era5": counts["era5_available"],
        "imerg": counts["imerg_available"],
        "complete": counts["complete"],
        "storms": len(observed_storms),
    }


def training_horizon_from_manifest(processed_dir: Path) -> int | None:
    """Read only the horizon for the baseline cross-check without duplicating errors."""
    try:
        payload = json.loads((processed_dir / "training_samples_manifest.json").read_text(encoding="utf-8"))
        value = payload["configuration"]["exact_horizon_hours"]
        return int(value) if int(value) > 0 else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def print_result(result: ValidationResult) -> None:
    if not result.ok:
        total = len(result.errors) + result._suppressed_errors
        print(f"FAIL pipeline validation: {total} issue(s)")
        for message in result.errors:
            print(f"- {message}")
        if result._suppressed_errors:
            print(f"- {result._suppressed_errors} additional issue(s) suppressed; fix the errors above and rerun.")
        return
    tracks = result.summary.get("tracks", {})
    training = result.summary.get("training", {})
    baseline = result.summary.get("baseline", {})
    plan = result.summary.get("plan", {})
    multisource = result.summary.get("multisource", {})
    splits = training.get("splits", {})
    print("PASS pipeline validation")
    print(f"- IBTrACS: {tracks.get('records', 0):,} observations across {tracks.get('storms', 0):,} storms")
    print(
        "- Training: "
        f"{training.get('records', 0):,} samples "
        f"(train {splits.get('train', 0):,}; validation {splits.get('validation', 0):,}; test {splits.get('test', 0):,})"
    )
    track_mae = baseline.get("track_mae")
    wind_mae = baseline.get("wind_mae")
    if isinstance(track_mae, (int, float)) and isinstance(wind_mae, (int, float)):
        print(f"- Baseline: {baseline.get('horizon')}h; track MAE {track_mae:.3f} km; wind MAE {wind_mae:.3f} kt")
    else:
        print(f"- Baseline: {baseline.get('horizon')}h contract validated")
    print(f"- Collection plan: {plan.get('storms', 0):,} storms / {plan.get('observations', 0):,} observations")
    print(
        "- Multi-source: "
        f"{multisource.get('rows', 0):,} rows; ERA5 {multisource.get('era5', 0):,}; "
        f"IMERG {multisource.get('imerg', 0):,}; complete {multisource.get('complete', 0):,}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated IBTrACS, baseline, and multi-source pipeline artifacts.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing generated JSON and CSV artifacts (default: data/processed).",
    )
    parser.add_argument(
        "--require-complete-multisource",
        action="store_true",
        help="Fail unless every multi-source row has both ERA5 and IMERG availability.",
    )
    args = parser.parse_args()
    processed_dir = args.processed_dir if args.processed_dir.is_absolute() else PROJECT_ROOT / args.processed_dir
    result = ValidationResult()
    tracks = validate_tracks(processed_dir, result)
    training = validate_training(processed_dir, tracks, result)
    validate_baseline(processed_dir, tracks, training_horizon_from_manifest(processed_dir), result)
    plan_storms = validate_collection_plan(processed_dir, tracks, result)
    validate_multisource(processed_dir, plan_storms, training, result, args.require_complete_multisource)
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
