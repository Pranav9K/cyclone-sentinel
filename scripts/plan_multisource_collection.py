"""Build a checked collection plan for the first multi-source storm cohort.

The plan is intentionally separate from downloading: it guarantees every
external request is traceable to a storm, time window, and geographic envelope
before credentials or large downloads are involved.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKS = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_COHORT = PROJECT_ROOT / "config" / "mvp_storms.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the event-aligned ERA5/IMERG collection plan.")
    parser.add_argument("--tracks", type=Path, default=DEFAULT_TRACKS)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buffer-degrees", type=float, default=5.0)
    args = parser.parse_args()
    if args.buffer_degrees <= 0:
        raise ValueError("--buffer-degrees must be positive")

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    wanted = set(cohort["storm_ids"])
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.tracks.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row.get("storm_id") in wanted:
                grouped[row["storm_id"]].append(row)

    missing = sorted(wanted - set(grouped))
    if missing:
        raise ValueError(f"Cohort IDs absent from tracks: {', '.join(missing)}")

    storms: list[dict[str, object]] = []
    total_observations = 0
    for storm_id in cohort["storm_ids"]:
        rows = sorted(grouped[storm_id], key=lambda item: item["timestamp_utc"])
        times = [parse_time(row["timestamp_utc"]) for row in rows]
        lats = [float(row["latitude"]) for row in rows]
        lons = [float(row["longitude"]) for row in rows]
        total_observations += len(rows)
        storms.append({
            "storm_id": storm_id,
            "name": rows[0]["name"],
            "season": int(rows[0]["season"]),
            "observations": len(rows),
            "start_utc": times[0].isoformat().replace("+00:00", "Z"),
            "end_utc": times[-1].isoformat().replace("+00:00", "Z"),
            "era5_area_nwse": [
                round(min(90.0, max(lats) + args.buffer_degrees), 3),
                round(max(-180.0, min(lons) - args.buffer_degrees), 3),
                round(max(-90.0, min(lats) - args.buffer_degrees), 3),
                round(min(180.0, max(lons) + args.buffer_degrees), 3),
            ],
            "time_steps_utc": [value.isoformat().replace("+00:00", "Z") for value in times],
        })

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "track_source": "NOAA IBTrACS v04r01 North Indian Ocean normalized tracks",
        "buffer_degrees": args.buffer_degrees,
        "storms": storms,
        "summary": {"storms": len(storms), "observations": total_observations},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Planned {len(storms)} storms and {total_observations} aligned observations: {args.output}")


if __name__ == "__main__":
    main()
