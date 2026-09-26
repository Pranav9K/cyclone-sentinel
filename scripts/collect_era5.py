"""Collect compact, event-aligned ERA5 single-level data from CDS.

Run with --dry-run first. Real requests require a free CDS account, acceptance
of the ERA5 dataset terms, and a configured CDS API key. The collector requests
one storm/month envelope at a time instead of a wasteful all-basin archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "era5_manifest.json"
DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "total_column_water_vapour",
    "2m_dewpoint_temperature",
]


def request_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_requests(storm: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Group observed timestamps by calendar month for compact CDS requests."""
    grouped: dict[tuple[int, int], list[datetime]] = defaultdict(list)
    for text in storm["time_steps_utc"]:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        grouped[(value.year, value.month)].append(value)

    requests: list[tuple[str, dict[str, Any]]] = []
    for (year, month), times in sorted(grouped.items()):
        request = {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": sorted({f"{value.day:02d}" for value in times}),
            "time": sorted({f"{value.hour:02d}:00" for value in times}),
            "area": storm["era5_area_nwse"],
            "data_format": "grib",
            "download_format": "unarchived",
        }
        requests.append((f"{year}{month:02d}", request))
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Download event-aligned ERA5 data through the CDS API.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--storm-id", help="Collect one selected storm ID; omit to collect every plan entry.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true", help="Print request plan without contacting CDS.")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    storms = [item for item in plan["storms"] if not args.storm_id or item["storm_id"] == args.storm_id]
    if not storms:
        raise ValueError("No matching storm in collection plan.")

    planned: list[dict[str, Any]] = []
    for storm in storms:
        for month_key, request in build_requests(storm):
            target = args.output_dir / storm["storm_id"] / f"{month_key}.grib"
            planned.append({
                "storm_id": storm["storm_id"],
                "month": month_key,
                "request": request,
                "target": str(target.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "request_hash": request_hash(request),
            })

    if args.dry_run:
        print(json.dumps({"dataset": DATASET, "requests": planned}, indent=2))
        print(f"Dry run: {len(planned)} CDS request(s); no data downloaded.")
        return

    try:
        import cdsapi  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("Install the optional CDS client: pip install cdsapi, then configure your CDS API key.") from error

    client = cdsapi.Client()
    completed: list[dict[str, Any]] = []
    for item in planned:
        target = PROJECT_ROOT / item["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {item['storm_id']} {item['month']}…")
            client.retrieve(DATASET, item["request"], str(target))
        completed.append({**item, "downloaded_at_utc": datetime.now(UTC).isoformat(), "bytes": target.stat().st_size})

    manifest = {
        "schema_version": 1,
        "source": "Copernicus Climate Data Store ERA5 hourly single levels",
        "dataset": DATASET,
        "variables": VARIABLES,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "requests": completed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Complete: {len(completed)} ERA5 file(s) recorded in {args.manifest}")


if __name__ == "__main__":
    main()
