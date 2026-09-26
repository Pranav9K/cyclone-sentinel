"""Join track labels with event-aligned ERA5 and IMERG feature tables.

The script deliberately keeps missing-source rows instead of fabricating data
or silently excluding hard examples.  A model training run can later choose a
complete-case cohort, an imputation strategy, or a modality-specific model.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES = PROJECT_ROOT / "data" / "processed" / "training_samples.csv"
DEFAULT_PLAN = PROJECT_ROOT / "data" / "processed" / "multisource_collection_plan.json"
DEFAULT_ERA5 = PROJECT_ROOT / "data" / "processed" / "era5_features.csv"
DEFAULT_IMERG = PROJECT_ROOT / "data" / "processed" / "imerg_features.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "multisource_training.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "multisource_training_manifest.json"

SOURCE_PREFIXES = {"era5": "era5_", "imerg": "imerg_"}


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def read_feature_table(path: Path, source_name: str) -> tuple[dict[tuple[str, str], dict[str, str]], list[str]]:
    """Read one optional source table and enforce its time-keyed contract."""
    if not path.exists():
        return {}, []
    prefix = SOURCE_PREFIXES[source_name]
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        required = {"storm_id", "timestamp_utc"}
        missing = required - set(fields)
        if missing:
            raise ValueError(f"{path.name} is missing key column(s): {', '.join(sorted(missing))}")
        feature_fields = [field for field in fields if field.startswith(prefix)]
        records: dict[tuple[str, str], dict[str, str]] = {}
        for row_number, row in enumerate(reader, start=2):
            storm_id = row["storm_id"].strip()
            timestamp = row["timestamp_utc"].strip()
            try:
                parse_time(timestamp)
            except ValueError as error:
                raise ValueError(f"{path.name}:{row_number} has invalid timestamp_utc: {timestamp!r}") from error
            key = (storm_id, timestamp)
            if key in records:
                raise ValueError(f"{path.name}:{row_number} duplicates source key {storm_id} {timestamp}")
            records[key] = {field: row.get(field, "").strip() for field in feature_fields}
    return records, feature_fields


def read_cohort(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    storm_ids = {entry["storm_id"] for entry in payload.get("storms", [])}
    if not storm_ids:
        raise ValueError(f"No storm entries were found in {path}")
    return storm_ids


def empty_values(fields: Iterable[str]) -> dict[str, str]:
    return {field: "" for field in fields}


def source_available(record: dict[str, str] | None, source_name: str) -> bool:
    """A source row explicitly marked missing must not count as coverage."""
    if record is None:
        return False
    missing_value = record.get(f"{source_name}_missing", "").strip().lower()
    return missing_value not in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the event-aligned multi-source training table.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--era5", type=Path, default=DEFAULT_ERA5)
    parser.add_argument("--imerg", type=Path, default=DEFAULT_IMERG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-complete", action="store_true", help="Fail unless every selected sample has both source layers.")
    args = parser.parse_args()

    cohort_ids = read_cohort(args.plan)
    era5_records, era5_fields = read_feature_table(args.era5, "era5")
    imerg_records, imerg_fields = read_feature_table(args.imerg, "imerg")

    with args.samples.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        sample_fields = reader.fieldnames or []
        required = {"sample_id", "storm_id", "input_timestamp_utc"}
        missing = required - set(sample_fields)
        if missing:
            raise ValueError(f"Training sample table is missing: {', '.join(sorted(missing))}")
        selected = [row for row in reader if row["storm_id"] in cohort_ids]

    if not selected:
        raise ValueError("The selected cohort has no supervised samples. Run prepare_training_data.py first.")

    output_fields = [
        *sample_fields,
        *era5_fields,
        *imerg_fields,
        "era5_available",
        "imerg_available",
        "multisource_complete",
    ]
    counts: Counter[str] = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output_fields)
        writer.writeheader()
        for row in selected:
            key = (row["storm_id"], row["input_timestamp_utc"])
            era5 = era5_records.get(key)
            imerg = imerg_records.get(key)
            era5_is_available = source_available(era5, "era5")
            imerg_is_available = source_available(imerg, "imerg")
            complete = era5_is_available and imerg_is_available
            if args.require_complete and not complete:
                raise ValueError(f"Missing source features for {key[0]} at {key[1]}")
            writer.writerow({
                **row,
                **(era5 if era5 is not None else empty_values(era5_fields)),
                **(imerg if imerg is not None else empty_values(imerg_fields)),
                "era5_available": str(era5_is_available).lower(),
                "imerg_available": str(imerg_is_available).lower(),
                "multisource_complete": str(complete).lower(),
            })
            counts["rows"] += 1
            counts["era5_available"] += int(era5_is_available)
            counts["imerg_available"] += int(imerg_is_available)
            counts["complete"] += int(complete)

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cohort_storms": len(cohort_ids),
        "input_samples": str(args.samples.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_plan": str(args.plan.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "era5_feature_source": str(args.era5.relative_to(PROJECT_ROOT)).replace("\\", "/") if args.era5.exists() else None,
        "imerg_feature_source": str(args.imerg.relative_to(PROJECT_ROOT)).replace("\\", "/") if args.imerg.exists() else None,
        "output": str(args.output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "columns": output_fields,
        "counts": dict(counts),
        "missing_data_policy": "Rows are retained; availability flags describe each source layer.",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Built {counts['rows']} cohort samples "
        f"(ERA5: {counts['era5_available']}, IMERG: {counts['imerg_available']}, complete: {counts['complete']})."
    )


if __name__ == "__main__":
    main()
