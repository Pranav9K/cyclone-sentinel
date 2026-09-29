"""Report local multi-source collection readiness without reading secrets.

The report checks only whether a credential location or expected environment
variables exist. It never reads, prints, or writes credential values.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROCESSED_DIR / "collection_readiness.json"


def dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def credential_presence(home: Path, environment: Mapping[str, str]) -> dict[str, bool]:
    """Return booleans only; callers never receive a secret or its path."""
    cds_config = Path(environment.get("CDSAPI_RC", home / ".cdsapirc"))
    return {
        "cds_configuration_present": cds_config.is_file(),
        "earthdata_environment_present": bool(environment.get("EARTHDATA_USERNAME") and environment.get("EARTHDATA_PASSWORD")),
        "earthdata_netrc_present": (home / ".netrc").is_file(),
    }


def count_files(directory: Path, suffixes: set[str]) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def count_available_csv_rows(path: Path, missing_column: str) -> int:
    """Count measured feature rows, excluding deliberate missing placeholders."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for row in csv.DictReader(handle) if row.get(missing_column) != "1")
    except OSError:
        return 0


def source_state(has_raw_data: bool, prerequisites_ready: bool) -> str:
    if has_raw_data:
        return "collection_started"
    return "ready_to_collect" if prerequisites_ready else "setup_required"


def build_report(home: Path | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    home = home or Path.home()
    environment = environment or os.environ
    credentials = credential_presence(home, environment)
    dependencies = {
        "cdsapi": dependency_available("cdsapi"),
        "numpy": dependency_available("numpy"),
        "h5py": dependency_available("h5py"),
    }
    era5_files = count_files(RAW_DIR / "era5", {".grib", ".grb", ".nc", ".csv"})
    imerg_files = count_files(RAW_DIR / "imerg", {".h5", ".hdf5"})
    insat_files = count_files(RAW_DIR / "insat", {".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    era5_ready = credentials["cds_configuration_present"] and dependencies["cdsapi"]
    imerg_ready = (credentials["earthdata_environment_present"] or credentials["earthdata_netrc_present"]) and dependencies["numpy"] and dependencies["h5py"]
    sources = [
        {
            "id": "era5", "name": "ERA5 atmospheric context", "status": source_state(era5_files > 0, era5_ready),
            "raw_files": era5_files, "feature_rows": count_available_csv_rows(PROCESSED_DIR / "era5_features.csv", "era5_missing"),
            "next_action": "Run collect_era5.py for a selected storm." if era5_ready else "Install cdsapi and configure a local CDS API credential file.",
        },
        {
            "id": "imerg", "name": "GPM IMERG Final rainfall", "status": source_state(imerg_files > 0, imerg_ready),
            "raw_files": imerg_files, "feature_rows": count_available_csv_rows(PROCESSED_DIR / "imerg_features.csv", "imerg_missing"),
            "next_action": "Run collect_imerg.py, then extract_imerg_features.py." if imerg_ready else "Configure Earthdata access and install numpy plus h5py.",
        },
        {
            "id": "insat", "name": "MOSDAC INSAT imagery", "status": "collection_started" if insat_files else "registration_required",
            "raw_files": insat_files, "feature_rows": 0,
            "next_action": "Index locally authorised imagery." if insat_files else "Register for authorised MOSDAC imagery access, then place files under data/raw/insat/.",
        },
    ]
    next_source = next((source for source in sources if source["status"] in {"setup_required", "registration_required"}), None)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "credentials": credentials,
        "dependencies": dependencies,
        "sources": sources,
        "next_action": next_source["next_action"] if next_source else "Run the matching feature extractors, rebuild the joined dataset, then validate coverage.",
        "safety_note": "This report contains setup booleans and file counts only; it never reads or exposes credentials.",
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local ERA5, IMERG, and INSAT collection readiness without exposing credentials.")
    parser.add_argument("--write", action="store_true", help="Write the safe report for the dashboard under data/processed/.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Report destination used with --write.")
    args = parser.parse_args()
    report = build_report()
    print(json.dumps(report, indent=2))
    if args.write:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        write_report(output, report)
        print(f"Readiness report written to {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
