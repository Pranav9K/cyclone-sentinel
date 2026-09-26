"""Download and normalize official NOAA IBTrACS North Indian Ocean tracks.

This collector is deliberately dependency-free so the first project dataset can
be reproduced with the standard Python library. It writes raw source data and
a compact CSV suitable for feature engineering.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r01/access/csv/ibtracs.NI.list.v04r01.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ibtracs" / "ibtracs.NI.list.v04r01.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_manifest.json"


def first_present(row: dict[str, str], *fields: str) -> str:
    """Use the first populated agency measurement, retaining the source field."""
    return next((row.get(field, "").strip() for field in fields if row.get(field, "").strip()), "")


def download() -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    request = Request(SOURCE_URL, headers={"User-Agent": "Cyclone-Sentinel/0.1 research"})
    with urlopen(request, timeout=90) as response, RAW_PATH.open("wb") as target:
        shutil.copyfileobj(response, target)


def normalize(start_year: int) -> int:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["storm_id", "season", "name", "timestamp_utc", "latitude", "longitude", "wind_knots", "pressure_hpa", "status", "source"]
    count = 0
    with RAW_PATH.open("r", encoding="utf-8-sig", newline="") as source, PROCESSED_PATH.open("w", encoding="utf-8", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            try:
                season = int(row.get("SEASON", ""))
            except ValueError:  # IBTrACS metadata/unit row
                continue
            if season < start_year or not row.get("ISO_TIME", "").strip():
                continue
            wind = first_present(row, "WMO_WIND", "USA_WIND", "BOM_WIND", "TOKYO_WIND")
            pressure = first_present(row, "WMO_PRES", "USA_PRES", "BOM_PRES", "TOKYO_PRES")
            writer.writerow({
                "storm_id": row.get("SID", "").strip(),
                "season": season,
                "name": row.get("NAME", "").strip() or "UNNAMED",
                "timestamp_utc": row.get("ISO_TIME", "").strip(),
                "latitude": row.get("LAT", "").strip(),
                "longitude": row.get("LON", "").strip(),
                "wind_knots": wind,
                "pressure_hpa": pressure,
                "status": row.get("NATURE", "").strip(),
                "source": "NOAA IBTrACS v04r01",
            })
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect NOAA IBTrACS North Indian Ocean tracks.")
    parser.add_argument("--start-year", type=int, default=2000, help="First season to retain (default: 2000).")
    parser.add_argument("--skip-download", action="store_true", help="Normalize an already downloaded raw file.")
    args = parser.parse_args()
    if not args.skip_download:
        print("Downloading NOAA IBTrACS North Indian Ocean archive…")
        download()
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw file is missing: {RAW_PATH}")
    records = normalize(args.start_year)
    manifest = {
        "status": "ready",
        "source": "NOAA IBTrACS v04r01 — North Indian Ocean subset",
        "source_url": SOURCE_URL,
        "license_note": "Use the NOAA citation and technical documentation when publishing results.",
        "start_year": args.start_year,
        "records": records,
        "raw_path": str(RAW_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "processed_path": str(PROCESSED_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "collected_at_utc": datetime.now(UTC).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Ready: {records:,} time-stamped observations written to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
