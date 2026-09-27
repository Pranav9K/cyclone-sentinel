"""Extract causal, event-aligned rainfall features from local GPM IMERG HDF5.

This stage never downloads data and never reads Earthdata credentials. It turns
the raw HDF5 files planned by ``collect_imerg.py`` into one CSV row per planned
storm issue time. If source files are absent, it writes explicit missing rows
instead of silently substituting zero rainfall.

HDF5 decoding is optional: install ``h5py`` and ``numpy`` only after raw IMERG
files have been downloaded. A no-data run uses the standard library alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"
DEFAULT_TRACKS = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "imerg"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "imerg_features.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "imerg_features_manifest.json"

WINDOW_DEGREES = 2.5
OUTPUT_COLUMNS = [
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


@dataclass(frozen=True)
class Event:
    storm_id: str
    timestamp: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True)
class GranuleStats:
    center_precipitation: float
    window_mean_precipitation: float
    window_p95_precipitation: float
    window_max_precipitation: float
    window_mean_random_error: float
    window_mean_quality_index: float


def parse_timestamp(value: object, label: str, *, assume_utc_if_naive: bool = False) -> datetime:
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} is not an ISO timestamp: {text!r}") from error
    if parsed.tzinfo is None:
        if not assume_utc_if_naive:
            raise ValueError(f"{label} must include a UTC offset: {text!r}")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def timestamp_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def completed_granule_start(issue_time: datetime) -> datetime:
    """Return the start of the last half-hour interval fully known at issue time."""
    minute = issue_time.astimezone(UTC).replace(second=0, microsecond=0)
    boundary = minute - timedelta(minutes=minute.minute % 30)
    return boundary - timedelta(minutes=30)


def imerg_filename(granule_start: datetime) -> str:
    end = granule_start + timedelta(minutes=30) - timedelta(seconds=1)
    minute_index = granule_start.hour * 60 + granule_start.minute
    return (
        "3B-HHR.MS.MRG.3IMERG."
        f"{granule_start:%Y%m%d}-S{granule_start:%H%M%S}-E{end:%H%M%S}."
        f"{minute_index:04d}.V07B.HDF5"
    )


def resolve(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def read_plan(path: Path) -> list[tuple[str, datetime]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read collection plan {path}: {error}") from error
    storms = payload.get("storms")
    if not isinstance(storms, list) or not storms:
        raise ValueError("Collection plan has no storms.")
    events: list[tuple[str, datetime]] = []
    seen: set[tuple[str, datetime]] = set()
    for index, storm in enumerate(storms, start=1):
        if not isinstance(storm, dict):
            raise ValueError(f"Plan storm {index} is not an object.")
        storm_id = str(storm.get("storm_id", "")).strip()
        steps = storm.get("time_steps_utc")
        if not storm_id or not isinstance(steps, list) or not steps:
            raise ValueError(f"Plan storm {index} needs storm_id and time_steps_utc.")
        for step in steps:
            issue_time = parse_timestamp(step, f"Plan {storm_id} time step")
            key = (storm_id, issue_time)
            if key in seen:
                raise ValueError(f"Plan duplicates {storm_id} at {timestamp_text(issue_time)}")
            seen.add(key)
            events.append(key)
    return sorted(events, key=lambda item: (item[0], item[1]))


def load_events(plan_events: list[tuple[str, datetime]], tracks_path: Path) -> list[Event]:
    if not tracks_path.is_file():
        raise ValueError(f"Missing normalized IBTrACS tracks: {tracks_path}")
    expected = set(plan_events)
    positions: dict[tuple[str, datetime], tuple[float, float]] = {}
    with tracks_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"storm_id", "timestamp_utc", "latitude", "longitude"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("IBTrACS tracks are missing storm_id, timestamp_utc, latitude, or longitude.")
        for row_number, row in enumerate(reader, start=2):
            storm_id = (row.get("storm_id") or "").strip()
            try:
                issue_time = parse_timestamp(
                    row.get("timestamp_utc", ""),
                    f"Tracks row {row_number}",
                    assume_utc_if_naive=True,
                )
            except ValueError:
                continue
            key = (storm_id, issue_time)
            if key not in expected:
                continue
            try:
                latitude = float(row.get("latitude", ""))
                longitude = float(row.get("longitude", ""))
            except ValueError as error:
                raise ValueError(f"Tracks row {row_number} has invalid coordinates.") from error
            if not math.isfinite(latitude) or not math.isfinite(longitude):
                raise ValueError(f"Tracks row {row_number} has non-finite coordinates.")
            if key in positions:
                raise ValueError(f"Tracks duplicate {storm_id} at {timestamp_text(issue_time)}")
            positions[key] = (latitude, longitude)
    missing = [f"{storm}@{timestamp_text(time)}" for storm, time in plan_events if (storm, time) not in positions]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"IBTrACS is missing {len(missing)} planned event position(s): {preview}")
    return [Event(storm, time, *positions[(storm, time)]) for storm, time in plan_events]


def discover_raw_files(raw_dir: Path) -> dict[str, Path]:
    if not raw_dir.exists():
        return {}
    if not raw_dir.is_dir():
        raise ValueError(f"--raw-dir is not a directory: {raw_dir}")
    files: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*.HDF5"), key=lambda item: str(item).lower()):
        existing = files.get(path.name)
        if existing is not None and existing != path:
            raise ValueError(f"Duplicate IMERG filename in raw directory: {path.name}")
        files[path.name] = path
    return files


def load_hdf_dependencies() -> tuple[Any, Any]:
    try:
        import h5py  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Raw IMERG files are present, but HDF5 support is missing. Install optional dependencies: "
            "pip install h5py numpy"
        ) from error
    return h5py, np


def hdf_dataset(handle: Any, candidates: tuple[str, ...], label: str) -> Any:
    for name in candidates:
        if name in handle:
            return handle[name]
    raise ValueError(f"IMERG HDF5 is missing {label}; checked {', '.join(candidates)}")


def normalize_values(values: Any, dataset: Any, np: Any) -> Any:
    output = np.asarray(values, dtype=float)
    for key in ("_FillValue", "missing_value"):
        if key in dataset.attrs:
            fill = float(dataset.attrs[key])
            output[np.isclose(output, fill, equal_nan=False)] = np.nan
    output[~np.isfinite(output)] = np.nan
    return output


def window_indices(values: Any, centre: float, half_width: float, *, longitude: bool, np: Any) -> Any:
    if longitude:
        delta = np.abs((values - centre + 180.0) % 360.0 - 180.0)
    else:
        delta = np.abs(values - centre)
    indices = np.flatnonzero(delta <= half_width + 1e-9)
    if indices.size == 0:
        raise ValueError("No grid cells fall inside the requested IMERG spatial window.")
    return indices


def slice_grid(
    dataset: Any,
    latitudes: Any,
    longitudes: Any,
    lat_indices: Any,
    lon_indices: Any,
    centre_latitude: float,
    centre_longitude: float,
    np: Any,
) -> tuple[Any, Any]:
    """Read only the local grid window and its centre, supporting both axis orders."""
    if dataset.ndim != 2:
        raise ValueError(f"IMERG grid {dataset.name} must be two-dimensional, got {dataset.ndim} dimensions.")
    if not (np.all(np.diff(lat_indices) == 1) and np.all(np.diff(lon_indices) == 1)):
        raise ValueError("IMERG spatial window crosses a grid seam; split-window extraction is not implemented.")
    lat_start, lat_stop = int(lat_indices[0]), int(lat_indices[-1]) + 1
    lon_start, lon_stop = int(lon_indices[0]), int(lon_indices[-1]) + 1
    lat_centre = int(np.argmin(np.abs(latitudes - centre_latitude)))
    lon_centre = int(np.argmin(np.abs((longitudes - centre_longitude + 180) % 360 - 180)))
    if dataset.shape == (len(latitudes), len(longitudes)):
        return dataset[lat_start:lat_stop, lon_start:lon_stop], dataset[lat_centre, lon_centre]
    if dataset.shape == (len(longitudes), len(latitudes)):
        return dataset[lon_start:lon_stop, lat_start:lat_stop], dataset[lon_centre, lat_centre]
    raise ValueError(
        f"IMERG grid {dataset.name} shape {dataset.shape} does not match latitude/longitude sizes "
        f"({len(latitudes)}, {len(longitudes)})."
    )


def feature_stats(path: Path, latitude: float, longitude: float, h5py: Any, np: Any) -> GranuleStats:
    with h5py.File(path, "r") as handle:
        latitudes = normalize_values(hdf_dataset(handle, ("Grid/lat", "lat", "latitude"), "latitude" )[:], hdf_dataset(handle, ("Grid/lat", "lat", "latitude"), "latitude"), np)
        longitudes = normalize_values(hdf_dataset(handle, ("Grid/lon", "lon", "longitude"), "longitude")[:], hdf_dataset(handle, ("Grid/lon", "lon", "longitude"), "longitude"), np)
        if latitudes.ndim != 1 or longitudes.ndim != 1:
            raise ValueError("IMERG latitude and longitude arrays must be one-dimensional.")
        lat_indices = window_indices(latitudes, latitude, WINDOW_DEGREES, longitude=False, np=np)
        lon_indices = window_indices(longitudes, longitude, WINDOW_DEGREES, longitude=True, np=np)
        precipitation = hdf_dataset(handle, ("Grid/precipitationCal", "precipitationCal"), "precipitationCal")
        random_error = hdf_dataset(handle, ("Grid/randomError", "randomError"), "randomError")
        quality = hdf_dataset(handle, ("Grid/precipitationQualityIndex", "precipitationQualityIndex"), "precipitationQualityIndex")
        grid_args = (latitudes, longitudes, lat_indices, lon_indices, latitude, longitude, np)
        precip_window, precip_centre = slice_grid(precipitation, *grid_args)
        error_window, _ = slice_grid(random_error, *grid_args)
        quality_window, _ = slice_grid(quality, *grid_args)
        precip_window = normalize_values(precip_window, precipitation, np)
        error_window = normalize_values(error_window, random_error, np)
        quality_window = normalize_values(quality_window, quality, np)
        precip_centre = float(normalize_values([precip_centre], precipitation, np)[0])
        if not math.isfinite(precip_centre):
            raise ValueError("IMERG centre precipitation is missing.")
        if not np.isfinite(precip_window).any() or not np.isfinite(error_window).any() or not np.isfinite(quality_window).any():
            raise ValueError("IMERG source fields have no valid values in the requested spatial window.")
        return GranuleStats(
            center_precipitation=precip_centre,
            window_mean_precipitation=float(np.nanmean(precip_window)),
            window_p95_precipitation=float(np.nanpercentile(precip_window, 95)),
            window_max_precipitation=float(np.nanmax(precip_window)),
            window_mean_random_error=float(np.nanmean(error_window)),
            window_mean_quality_index=float(np.nanmean(quality_window)),
        )


def empty_row(event: Event) -> dict[str, str]:
    return {
        "storm_id": event.storm_id,
        "timestamp_utc": timestamp_text(event.timestamp),
        "imerg_granule_start_utc": "",
        "imerg_center_precipitation_cal_mm_hr": "",
        "imerg_window_mean_precipitation_cal_mm_hr": "",
        "imerg_window_p95_precipitation_cal_mm_hr": "",
        "imerg_window_max_precipitation_cal_mm_hr": "",
        "imerg_window_mean_random_error_mm_hr": "",
        "imerg_window_mean_quality_index": "",
        "imerg_accumulation_6h_mm": "",
        "imerg_accumulation_24h_mm": "",
        "imerg_missing": "1",
    }


def number(value: float) -> str:
    return format(value, ".6g")


def build_row(event: Event, raw_files: dict[str, Path], h5py: Any, np: Any, cache: dict[tuple[Path, float, float], GranuleStats | None], errors: list[str]) -> tuple[dict[str, str], bool, bool]:
    endpoint = completed_granule_start(event.timestamp)
    starts = [endpoint - timedelta(minutes=30 * offset) for offset in range(48)]
    statistics: list[GranuleStats | None] = []
    for start in starts:
        path = raw_files.get(imerg_filename(start))
        if path is None:
            statistics.append(None)
            continue
        key = (path, round(event.latitude, 5), round(event.longitude, 5))
        if key not in cache:
            try:
                cache[key] = feature_stats(path, event.latitude, event.longitude, h5py, np)
            except (OSError, ValueError) as error:
                cache[key] = None
                if len(errors) < 25:
                    errors.append(f"{path.name}: {error}")
        statistics.append(cache[key])
    current = statistics[0]
    if current is None:
        return empty_row(event), False, False
    row = {
        "storm_id": event.storm_id,
        "timestamp_utc": timestamp_text(event.timestamp),
        "imerg_granule_start_utc": timestamp_text(endpoint),
        "imerg_center_precipitation_cal_mm_hr": number(current.center_precipitation),
        "imerg_window_mean_precipitation_cal_mm_hr": number(current.window_mean_precipitation),
        "imerg_window_p95_precipitation_cal_mm_hr": number(current.window_p95_precipitation),
        "imerg_window_max_precipitation_cal_mm_hr": number(current.window_max_precipitation),
        "imerg_window_mean_random_error_mm_hr": number(current.window_mean_random_error),
        "imerg_window_mean_quality_index": number(current.window_mean_quality_index),
        "imerg_accumulation_6h_mm": "",
        "imerg_accumulation_24h_mm": "",
        "imerg_missing": "0",
    }
    for label, window_size, field in (("6h", 12, "imerg_accumulation_6h_mm"), ("24h", 48, "imerg_accumulation_24h_mm")):
        window = statistics[:window_size]
        if all(item is not None for item in window):
            accumulation = sum(item.window_mean_precipitation * 0.5 for item in window if item is not None)
            row[field] = number(accumulation)
    return row, True, all(item is not None for item in statistics)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract causal event-aligned IMERG rainfall features from local HDF5 files.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Inspect local readiness without writing feature artifacts.")
    args = parser.parse_args()

    plan_path = resolve(args.plan)
    tracks_path = resolve(args.tracks)
    raw_dir = resolve(args.raw_dir)
    output_path = resolve(args.output)
    manifest_path = resolve(args.manifest)
    events = load_events(read_plan(plan_path), tracks_path)
    raw_files = discover_raw_files(raw_dir)
    expected_granules = {
        imerg_filename(completed_granule_start(event.timestamp) - timedelta(minutes=30 * offset))
        for event in events
        for offset in range(48)
    }
    if args.dry_run:
        print(json.dumps({
            "event_keys_expected": len(events),
            "raw_files_discovered": len(raw_files),
            "required_unique_granules": len(expected_granules),
            "required_granules_present": sum(name in raw_files for name in expected_granules),
            "feature_output": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }, indent=2))
        return

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    cache: dict[tuple[Path, float, float], GranuleStats | None] = {}
    endpoint_available = 0
    complete_24h = 0
    if raw_files:
        h5py, np = load_hdf_dependencies()
        for event in events:
            row, endpoint_ok, full_window = build_row(event, raw_files, h5py, np, cache, errors)
            rows.append(row)
            endpoint_available += int(endpoint_ok)
            complete_24h += int(full_window)
    else:
        rows = [empty_row(event) for event in events]

    write_csv(output_path, rows)
    if endpoint_available == 0:
        status = "waiting_for_raw_imerg"
    elif complete_24h == len(events):
        status = "ready"
    else:
        status = "partial"
    manifest = {
        "schema_version": 1,
        "status": status,
        "source": "NASA GES DISC GPM IMERG Final V07 half-hourly HDF5",
        "output": {"path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "records_written": len(rows)},
        "alignment": {"event_keys_expected": len(events), "window_degrees": WINDOW_DEGREES, "causal_interval_policy": "completed half-hour intervals ending at or before issue time"},
        "coverage": {"raw_files_discovered": len(raw_files), "endpoint_features_available": endpoint_available, "complete_24h_accumulations": complete_24h},
        "errors": errors,
        "next_action": "Collect missing IMERG HDF5 files and rerun extraction." if status != "ready" else "Join imerg_features.csv with the multi-source cohort.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"IMERG extraction {status}: wrote {len(rows):,} feature row(s); endpoints {endpoint_available:,}/{len(events):,}; complete 24h windows {complete_24h:,}/{len(events):,}.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
