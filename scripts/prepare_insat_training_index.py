"""Pair catalogued INSAT files with historical IBTrACS labels for ML research.

The input catalog is produced only from real local files by
``index_insat_imagery.py``.  This stage matches each parsed capture time to the
closest IBTrACS observation within a conservative tolerance and writes a
supervised-index CSV. It does not decode pixels, infer a cloud pattern, or
invent a storm label when no trustworthy time match exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "processed" / "insat_image_catalog.json"
DEFAULT_TRACKS = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "insat_training_index.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "insat_training_index_manifest.json"
OUTPUT_COLUMNS = [
    "image_path", "image_sha256", "capture_time_utc", "format", "bytes",
    "storm_id", "storm_name", "season", "track_timestamp_utc", "time_offset_minutes",
    "latitude", "longitude", "wind_knots", "pressure_hpa", "intensity_label",
]


@dataclass(frozen=True)
class TrackObservation:
    storm_id: str
    name: str
    season: str
    timestamp: datetime
    latitude: str
    longitude: str
    wind_knots: str
    pressure_hpa: str


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def intensity_label(wind_knots: str) -> str:
    try:
        wind = float(wind_knots)
    except ValueError:
        return "Intensity unavailable"
    if wind >= 120: return "Super Cyclonic Storm"
    if wind >= 90: return "Extremely Severe Cyclonic Storm"
    if wind >= 64: return "Very Severe Cyclonic Storm"
    if wind >= 48: return "Severe Cyclonic Storm"
    if wind >= 34: return "Cyclonic Storm"
    if wind >= 28: return "Deep Depression"
    if wind >= 17: return "Depression"
    return "Low Pressure Area"


def load_tracks(path: Path) -> list[TrackObservation]:
    if not path.exists():
        raise FileNotFoundError(f"IBTrACS track table not found: {path}")
    records: list[TrackObservation] = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                records.append(TrackObservation(
                    storm_id=row["storm_id"], name=row.get("name", "UNNAMED"), season=row["season"],
                    timestamp=parse_time(row["timestamp_utc"]), latitude=row["latitude"], longitude=row["longitude"],
                    wind_knots=row.get("wind_knots", ""), pressure_hpa=row.get("pressure_hpa", ""),
                ))
            except (KeyError, ValueError):
                continue
    return records


def nearest_observation(capture_time: datetime, tracks: list[TrackObservation], tolerance_minutes: int) -> tuple[TrackObservation | None, float | None]:
    """Return the closest best-track record only when its time is in tolerance."""
    if not tracks:
        return None, None
    closest = min(tracks, key=lambda item: abs((item.timestamp - capture_time).total_seconds()))
    difference = abs((closest.timestamp - capture_time).total_seconds()) / 60
    return (closest, difference) if difference <= tolerance_minutes else (None, difference)


def load_catalog(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"INSAT catalog is not valid JSON: {path}") from error
    images = payload.get("images", [])
    return [item for item in images if isinstance(item, dict)] if isinstance(images, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a timestamp-matched INSAT/IBTrACS supervised index.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tolerance-minutes", type=int, default=90)
    args = parser.parse_args()
    if args.tolerance_minutes < 0:
        parser.error("--tolerance-minutes must be non-negative.")

    catalog = args.catalog if args.catalog.is_absolute() else PROJECT_ROOT / args.catalog
    tracks_path = args.tracks if args.tracks.is_absolute() else PROJECT_ROOT / args.tracks
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    images, tracks = load_catalog(catalog), load_tracks(tracks_path)
    records: list[dict[str, str | int | float]] = []
    counters = {"catalog_images": len(images), "without_capture_time": 0, "outside_tolerance": 0, "paired": 0}
    for image in images:
        capture = image.get("capture_time_utc")
        if not isinstance(capture, str):
            counters["without_capture_time"] += 1
            continue
        try:
            capture_time = parse_time(capture)
        except ValueError:
            counters["without_capture_time"] += 1
            continue
        track, offset = nearest_observation(capture_time, tracks, args.tolerance_minutes)
        if track is None:
            counters["outside_tolerance"] += 1
            continue
        counters["paired"] += 1
        records.append({
            "image_path": str(image.get("path", "")), "image_sha256": str(image.get("sha256", "")),
            "capture_time_utc": utc_text(capture_time), "format": str(image.get("format", "")), "bytes": int(image.get("bytes", 0)),
            "storm_id": track.storm_id, "storm_name": track.name, "season": track.season,
            "track_timestamp_utc": utc_text(track.timestamp), "time_offset_minutes": round(offset or 0, 1),
            "latitude": track.latitude, "longitude": track.longitude, "wind_knots": track.wind_knots,
            "pressure_hpa": track.pressure_hpa, "intensity_label": intensity_label(track.wind_knots),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": 1,
        "status": "ready_for_image_feature_extraction" if records else "waiting_for_time_matched_images",
        "generated_at_utc": utc_text(datetime.now(UTC)),
        "catalog": str(catalog), "tracks": str(tracks_path), "output": str(output),
        "tolerance_minutes": args.tolerance_minutes, "counts": counters,
        "limitations": [
            "Labels come from the nearest historical IBTrACS timestamp, not from the satellite image itself.",
            "This index does not prove that the image footprint contains the matched cyclone centre.",
            "Do not use it as a real-time or operational satellite classification product.",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"INSAT supervised index: {counters['paired']} paired / {counters['catalog_images']} catalog images")


if __name__ == "__main__":
    main()
