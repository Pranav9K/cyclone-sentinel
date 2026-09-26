"""Prepare leakage-aware supervised samples from normalized IBTrACS tracks.

The collector deliberately retains all source observations.  This script is the
next, separate stage: it validates observations, keeps tropical/disturbance
records suitable for the initial North Indian Ocean model, and creates samples
where the label is exactly ``horizon_hours`` after the input observation.

It intentionally uses only the Python standard library so that the first ML
baseline stays reproducible on a fresh machine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_samples.csv"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "training_samples_manifest.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "training_samples_report.md"
FORECAST_HORIZON_HOURS = 24

REQUIRED_COLUMNS = {
    "storm_id",
    "season",
    "timestamp_utc",
    "latitude",
    "longitude",
    "wind_knots",
    "pressure_hpa",
    "status",
}
DEFAULT_ALLOWED_STATUSES = ("TS", "DS")
DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
)

FEATURE_COLUMNS = [
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
]
TARGET_COLUMNS = [
    "target_delta_latitude_24h",
    "target_delta_longitude_24h",
    "target_wind_knots",
    "target_wind_change_knots_24h",
]
OUTPUT_COLUMNS = [
    "sample_id",
    "storm_id",
    "input_timestamp_utc",
    *FEATURE_COLUMNS,
    "target_timestamp_utc",
    "target_interval_hours",
    "target_latitude",
    "target_longitude",
    *TARGET_COLUMNS,
    "split",
]


@dataclass(frozen=True)
class Observation:
    """One validated source observation after core track checks."""

    row_number: int
    storm_id: str
    season: int
    timestamp: datetime
    latitude: float
    longitude: float
    wind_knots: float | None
    pressure_hpa: float | None
    status: str

    @property
    def completeness_score(self) -> int:
        """Prefer a duplicate row with more usable measurements."""
        return int(self.wind_knots is not None) + int(self.pressure_hpa is not None)


def resolve_path(value: str | None, default: Path) -> Path:
    """Resolve CLI paths from project root, keeping absolute paths untouched."""
    if value is None:
        return default
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def project_relative(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    """Hash a source file in chunks for a reproducibility manifest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def text_value(value: object) -> str:
    """Coerce a potentially absent CSV field into safe stripped text."""
    return value.strip() if isinstance(value, str) else ""


def parse_timestamp(value: object) -> datetime | None:
    """Parse expected IBTrACS timestamps as timezone-aware UTC values."""
    text = text_value(value)
    for format_string in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, format_string).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def parse_finite_number(value: object) -> float | None:
    """Return a finite float, or None for blanks/non-numeric values."""
    text = text_value(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_longitude(longitude: float) -> float:
    """Normalize a valid longitude to [-180, 180) for stable displacement."""
    normalized = (longitude + 180.0) % 360.0 - 180.0
    # Avoid a negative zero in the CSV, which complicates simple comparisons.
    return 0.0 if normalized == 0 else normalized


def shortest_longitude_delta(start_longitude: float, end_longitude: float) -> float:
    """Calculate the signed shortest longitudinal movement in degrees."""
    return normalize_longitude(end_longitude - start_longitude)


def clean_number(value: float | None) -> str:
    """Write concise, locale-independent numeric values to CSV."""
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def parse_statuses(raw_value: str) -> tuple[str, ...]:
    """Normalize a comma-separated status allow-list."""
    statuses = tuple(sorted({item.strip().upper() for item in raw_value.split(",") if item.strip()}))
    if not statuses:
        raise ValueError("At least one allowed status is required.")
    return statuses


def validate_fractions(train_fraction: float, validation_fraction: float) -> None:
    """Ensure chronological split fractions leave a non-negative test share."""
    if not 0 < train_fraction < 1:
        raise ValueError("--train-fraction must be strictly between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 (inclusive) and 1 (exclusive).")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a positive test fraction.")


def validate_row(
    row: dict[str, str],
    row_number: int,
    allowed_statuses: set[str],
    validation_counts: Counter[str],
) -> Observation | None:
    """Validate source fields needed for track labels and return an observation.

    Wind and pressure are intentionally optional at this stage.  They are
    counted here, then a sample is only emitted when both input and target wind
    values are available for the requested next-wind label.
    """
    storm_id = text_value(row["storm_id"])
    if not storm_id:
        validation_counts["missing_storm_id"] += 1
        return None

    timestamp = parse_timestamp(row["timestamp_utc"])
    if timestamp is None:
        validation_counts["invalid_timestamp"] += 1
        return None

    try:
        season = int(text_value(row["season"]))
    except ValueError:
        validation_counts["invalid_season"] += 1
        return None
    if not 1800 <= season <= 2200:
        validation_counts["season_out_of_range"] += 1
        return None
    if season != timestamp.year:
        validation_counts["season_timestamp_mismatch"] += 1

    status = text_value(row["status"]).upper()
    if status not in allowed_statuses:
        validation_counts["status_not_allowed"] += 1
        return None

    latitude = parse_finite_number(row["latitude"])
    if latitude is None or not -90.0 <= latitude <= 90.0:
        validation_counts["invalid_latitude"] += 1
        return None

    raw_longitude = parse_finite_number(row["longitude"])
    if raw_longitude is None or not -360.0 <= raw_longitude <= 360.0:
        validation_counts["invalid_longitude"] += 1
        return None

    raw_wind = text_value(row["wind_knots"])
    wind_knots = parse_finite_number(raw_wind)
    if raw_wind and (wind_knots is None or not 0.0 <= wind_knots <= 250.0):
        validation_counts["invalid_wind_measurement"] += 1
        wind_knots = None
    elif wind_knots is None:
        validation_counts["missing_wind_measurement"] += 1

    raw_pressure = text_value(row["pressure_hpa"])
    pressure_hpa = parse_finite_number(raw_pressure)
    if raw_pressure and (pressure_hpa is None or not 800.0 <= pressure_hpa <= 1100.0):
        validation_counts["invalid_pressure_measurement"] += 1
        pressure_hpa = None
    elif pressure_hpa is None:
        validation_counts["missing_pressure_measurement"] += 1

    validation_counts["valid_core_observations"] += 1
    return Observation(
        row_number=row_number,
        storm_id=storm_id,
        season=season,
        timestamp=timestamp,
        latitude=latitude,
        longitude=normalize_longitude(raw_longitude),
        wind_knots=wind_knots,
        pressure_hpa=pressure_hpa,
        status=status,
    )


def load_observations(
    input_path: Path, allowed_statuses: set[str]
) -> tuple[dict[str, list[Observation]], Counter[str], Counter[str]]:
    """Load source records, validate them, and deterministically deduplicate times."""
    validation_counts: Counter[str] = Counter()
    best_by_storm_time: dict[tuple[str, datetime], Observation] = {}

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - source_columns)
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Input is missing required column(s): {missing}")

        for row_number, row in enumerate(reader, start=2):
            validation_counts["rows_read"] += 1
            observation = validate_row(row, row_number, allowed_statuses, validation_counts)
            if observation is None:
                continue
            key = (observation.storm_id, observation.timestamp)
            existing = best_by_storm_time.get(key)
            if existing is None:
                best_by_storm_time[key] = observation
                continue

            validation_counts["duplicate_storm_timestamp"] += 1
            if observation.completeness_score > existing.completeness_score:
                best_by_storm_time[key] = observation
                validation_counts["duplicate_replaced_with_more_complete_row"] += 1

    observations_by_storm: dict[str, list[Observation]] = defaultdict(list)
    for observation in best_by_storm_time.values():
        observations_by_storm[observation.storm_id].append(observation)
    for observations in observations_by_storm.values():
        observations.sort(key=lambda observation: observation.timestamp)

    validation_counts["unique_valid_observations"] = len(best_by_storm_time)
    validation_counts["storms_with_valid_observations"] = len(observations_by_storm)
    status_counts = Counter(
        observation.status
        for observations in observations_by_storm.values()
        for observation in observations
    )
    return dict(observations_by_storm), validation_counts, status_counts


def optional_previous_motion(
    observation: Observation,
    observations_by_time: dict[datetime, Observation],
) -> tuple[int, float | None, float | None, float | None]:
    """Derive non-required six-hour history features without interpolation."""
    previous = observations_by_time.get(observation.timestamp - timedelta(hours=6))
    if previous is None:
        return 0, None, None, None

    previous_wind_change = None
    if previous.wind_knots is not None and observation.wind_knots is not None:
        previous_wind_change = observation.wind_knots - previous.wind_knots
    return (
        1,
        observation.latitude - previous.latitude,
        shortest_longitude_delta(previous.longitude, observation.longitude),
        previous_wind_change,
    )


def build_samples(
    observations_by_storm: dict[str, list[Observation]], horizon_hours: int
) -> tuple[list[dict[str, str]], Counter[str]]:
    """Build samples with exact-horizon labels, preventing temporal leakage."""
    sample_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    storms_with_samples: set[str] = set()
    horizon = timedelta(hours=horizon_hours)

    for storm_id, observations in observations_by_storm.items():
        observations_by_time = {observation.timestamp: observation for observation in observations}
        for observation in observations:
            sample_counts["candidate_input_observations"] += 1
            target = observations_by_time.get(observation.timestamp + horizon)
            if target is None:
                sample_counts["excluded_no_exact_horizon_target"] += 1
                continue
            if observation.wind_knots is None:
                sample_counts["excluded_missing_input_wind"] += 1
                continue
            if target.wind_knots is None:
                sample_counts["excluded_missing_target_wind"] += 1
                continue

            previous_available, previous_lat_delta, previous_lon_delta, previous_wind_change = optional_previous_motion(
                observation, observations_by_time
            )
            target_interval_hours = (target.timestamp - observation.timestamp).total_seconds() / 3600
            sample = {
                "sample_id": f"{storm_id}_{observation.timestamp.strftime('%Y%m%dT%H%M%SZ')}",
                "storm_id": storm_id,
                "input_timestamp_utc": observation.timestamp.isoformat().replace("+00:00", "Z"),
                "season": str(observation.season),
                "input_year": str(observation.timestamp.year),
                "input_month": str(observation.timestamp.month),
                "input_day_of_year": str(observation.timestamp.timetuple().tm_yday),
                "input_hour_utc": str(observation.timestamp.hour),
                "latitude": clean_number(observation.latitude),
                "longitude": clean_number(observation.longitude),
                "wind_knots": clean_number(observation.wind_knots),
                "pressure_hpa": clean_number(observation.pressure_hpa),
                "status": observation.status,
                "previous_6h_available": str(previous_available),
                "previous_6h_delta_latitude": clean_number(previous_lat_delta),
                "previous_6h_delta_longitude": clean_number(previous_lon_delta),
                "previous_6h_wind_change_knots": clean_number(previous_wind_change),
                "target_timestamp_utc": target.timestamp.isoformat().replace("+00:00", "Z"),
                "target_interval_hours": clean_number(target_interval_hours),
                "target_latitude": clean_number(target.latitude),
                "target_longitude": clean_number(target.longitude),
                "target_delta_latitude_24h": clean_number(target.latitude - observation.latitude),
                "target_delta_longitude_24h": clean_number(
                    shortest_longitude_delta(observation.longitude, target.longitude)
                ),
                "target_wind_knots": clean_number(target.wind_knots),
                "target_wind_change_knots_24h": clean_number(target.wind_knots - observation.wind_knots),
                "split": "",
            }
            samples.append(sample)
            storms_with_samples.add(storm_id)
            sample_counts["emitted_samples"] += 1

    sample_counts["storms_with_supervised_samples"] = len(storms_with_samples)
    sample_counts["storms_without_supervised_samples"] = len(observations_by_storm) - len(storms_with_samples)

    def sample_sort_key(sample: dict[str, str]) -> tuple[str, str]:
        return sample["input_timestamp_utc"], sample["storm_id"]

    samples.sort(key=sample_sort_key)
    return samples, sample_counts


def chronological_storm_split(
    samples: list[dict[str, str]], train_fraction: float, validation_fraction: float
) -> dict[str, dict[str, object]]:
    """Assign whole storms to chronological train/validation/test partitions.

    Every sample from a storm receives one split, avoiding an easy but invalid
    evaluation where the model sees the same cyclone in both train and test.
    Storms are ordered by their first usable input timestamp.
    """
    earliest_by_storm: dict[str, str] = {}
    for sample in samples:
        earliest_by_storm.setdefault(sample["storm_id"], sample["input_timestamp_utc"])
    storm_ids = sorted(earliest_by_storm, key=lambda storm_id: (earliest_by_storm[storm_id], storm_id))
    storm_count = len(storm_ids)
    if not storm_count:
        return {}

    train_end = int(math.floor(storm_count * train_fraction))
    validation_end = train_end + int(math.floor(storm_count * validation_fraction))
    # A tiny dataset remains usable.  The collected source is much larger, but
    # this makes the script safe for a small fixture in CI or a new region.
    if storm_count >= 3:
        train_end = max(1, min(train_end, storm_count - 2))
        validation_end = max(train_end + 1, min(validation_end, storm_count - 1))
    elif storm_count == 2:
        train_end, validation_end = 1, 1
    else:
        train_end, validation_end = 1, 1

    split_by_storm: dict[str, str] = {}
    for index, storm_id in enumerate(storm_ids):
        if index < train_end:
            split_by_storm[storm_id] = "train"
        elif index < validation_end:
            split_by_storm[storm_id] = "validation"
        else:
            split_by_storm[storm_id] = "test"

    sample_counts = Counter()
    storms_by_split: dict[str, list[str]] = defaultdict(list)
    timestamps_by_split: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        split = split_by_storm[sample["storm_id"]]
        sample["split"] = split
        sample_counts[split] += 1
        timestamps_by_split[split].append(sample["input_timestamp_utc"])
    for storm_id, split in split_by_storm.items():
        storms_by_split[split].append(storm_id)

    summary: dict[str, dict[str, object]] = {}
    for split_name in ("train", "validation", "test"):
        split_timestamps = timestamps_by_split[split_name]
        summary[split_name] = {
            "storms": len(storms_by_split[split_name]),
            "samples": sample_counts[split_name],
            "first_input_timestamp_utc": min(split_timestamps) if split_timestamps else None,
            "last_input_timestamp_utc": max(split_timestamps) if split_timestamps else None,
        }
    return summary


def write_csv(samples: Iterable[dict[str, str]], output_path: Path) -> int:
    """Write the fixed supervised-data schema in chronological order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)
            count += 1
    return count


def write_report(
    report_path: Path,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    horizon_hours: int,
    allowed_statuses: tuple[str, ...],
    validation_counts: Counter[str],
    sample_counts: Counter[str],
    split_summary: dict[str, dict[str, object]],
) -> None:
    """Write a readable quality report beside the generated dataset."""
    total_rows = validation_counts["rows_read"]
    valid_rows = validation_counts["unique_valid_observations"]
    emitted = sample_counts["emitted_samples"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IBTrACS supervised training-data report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Result",
        "",
        f"- Input: `{project_relative(input_path)}`",
        f"- Output: `{project_relative(output_path)}`",
        f"- Manifest: `{project_relative(manifest_path)}`",
        f"- Exact forecast horizon: {horizon_hours} hours",
        f"- Allowed source statuses: {', '.join(allowed_statuses)}",
        f"- Source rows read: {total_rows:,}",
        f"- Unique valid observations: {valid_rows:,}",
        f"- Supervised samples emitted: {emitted:,}",
        "",
        "## Validation and exclusions",
        "",
        "| Check | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(validation_counts):
        if key in {"rows_read", "valid_core_observations", "unique_valid_observations", "storms_with_valid_observations"}:
            continue
        lines.append(f"| {key.replace('_', ' ')} | {validation_counts[key]:,} |")
    for key in sorted(sample_counts):
        lines.append(f"| {key.replace('_', ' ')} | {sample_counts[key]:,} |")

    lines.extend([
        "",
        "## Chronological, storm-disjoint split",
        "",
        "Each storm is assigned once, ordered by its first usable timestamp. This prevents the same cyclone appearing in both training and evaluation data.",
        "",
        "| Split | Storms | Samples | First input | Last input |",
        "| --- | ---: | ---: | --- | --- |",
    ])
    for split_name in ("train", "validation", "test"):
        summary = split_summary.get(split_name, {})
        lines.append(
            "| {split} | {storms:,} | {samples:,} | {first} | {last} |".format(
                split=split_name,
                storms=int(summary.get("storms", 0)),
                samples=int(summary.get("samples", 0)),
                first=summary.get("first_input_timestamp_utc") or "—",
                last=summary.get("last_input_timestamp_utc") or "—",
            )
        )

    lines.extend([
        "",
        "## Model columns",
        "",
        f"- Features: `{', '.join(FEATURE_COLUMNS)}`",
        f"- Targets: `{', '.join(TARGET_COLUMNS)}`",
        "- Identifiers and timestamps (`sample_id`, `storm_id`, `input_timestamp_utc`, `target_timestamp_utc`) are traceability fields, not model features.",
        "- `pressure_hpa` and six-hour history fields may be blank; retain an imputation strategy within each training split only.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create chronological 24-hour track-displacement and next-wind training samples."
    )
    parser.add_argument("--input", help="Input CSV, relative to project root by default.")
    parser.add_argument("--output", help="Output training CSV, relative to project root by default.")
    parser.add_argument("--manifest", help="Output JSON manifest, relative to project root by default.")
    parser.add_argument("--report", help="Output Markdown report, relative to project root by default.")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=FORECAST_HORIZON_HOURS,
        help="Exact forecast horizon used for labels (default: 24).",
    )
    parser.add_argument(
        "--allowed-statuses",
        default=",".join(DEFAULT_ALLOWED_STATUSES),
        help="Comma-separated IBTrACS NATURE values to retain (default: TS,DS).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Fraction of storms assigned to chronological training data (default: 0.70).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Fraction of storms assigned to chronological validation data (default: 0.15).",
    )
    args = parser.parse_args()

    if args.horizon_hours != FORECAST_HORIZON_HOURS:
        parser.error(
            "This builder intentionally writes the fixed 24-hour schema; "
            "--horizon-hours must be 24."
        )
    try:
        allowed_statuses = parse_statuses(args.allowed_statuses)
        validate_fractions(args.train_fraction, args.validation_fraction)
    except ValueError as error:
        parser.error(str(error))

    input_path = resolve_path(args.input, DEFAULT_INPUT_PATH)
    output_path = resolve_path(args.output, DEFAULT_OUTPUT_PATH)
    manifest_path = resolve_path(args.manifest, DEFAULT_MANIFEST_PATH)
    report_path = resolve_path(args.report, DEFAULT_REPORT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input data is missing: {input_path}")

    observations_by_storm, validation_counts, status_counts = load_observations(
        input_path, set(allowed_statuses)
    )
    samples, sample_counts = build_samples(observations_by_storm, args.horizon_hours)
    if not samples:
        raise RuntimeError(
            "No supervised samples were created. Check --allowed-statuses, horizon, and source coverage."
        )
    split_summary = chronological_storm_split(samples, args.train_fraction, args.validation_fraction)
    written_samples = write_csv(samples, output_path)

    manifest = {
        "schema_version": 1,
        "status": "ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": project_relative(input_path),
            "sha256": sha256_file(input_path),
            "rows_read": validation_counts["rows_read"],
        },
        "configuration": {
            "exact_horizon_hours": args.horizon_hours,
            "allowed_statuses": list(allowed_statuses),
            "split_strategy": "storm-disjoint chronological split by first usable input timestamp",
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": 1 - args.train_fraction - args.validation_fraction,
        },
        "quality": {
            "validation_counts": dict(sorted(validation_counts.items())),
            "retained_status_counts": dict(sorted(status_counts.items())),
            "sample_counts": dict(sorted(sample_counts.items())),
        },
        "dataset": {
            "path": project_relative(output_path),
            "records": written_samples,
            "columns": OUTPUT_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "split_summary": split_summary,
            "first_input_timestamp_utc": samples[0]["input_timestamp_utc"],
            "last_input_timestamp_utc": samples[-1]["input_timestamp_utc"],
        },
        "report_path": project_relative(report_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_report(
        report_path,
        input_path,
        output_path,
        manifest_path,
        args.horizon_hours,
        allowed_statuses,
        validation_counts,
        sample_counts,
        split_summary,
    )

    print(
        "Prepared "
        f"{written_samples:,} supervised samples from "
        f"{validation_counts['unique_valid_observations']:,} valid observations."
    )
    print(f"Training CSV: {project_relative(output_path)}")
    print(f"Manifest: {project_relative(manifest_path)}")
    print(f"Report: {project_relative(report_path)}")


if __name__ == "__main__":
    main()
