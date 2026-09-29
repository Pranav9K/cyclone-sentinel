"""Train a strict complete-case multi-source research benchmark.

This stage deliberately refuses to run until exact, non-missing ERA5 and
IMERG rows exist.  It never converts missing observations into zero or an
imputed value, so its reported metrics cannot be mistaken for a result from
uncollected source layers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from train_ibtracs_baseline import RidgeRegressor, haversine_km, normalize_longitude


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "multisource_training.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "latitude", "longitude", "wind_knots", "previous_6h_available",
    "previous_6h_delta_latitude", "previous_6h_delta_longitude", "previous_6h_wind_change_knots",
    "input_day_of_year", "input_hour_utc",
    "era5_u10_mps", "era5_v10_mps", "era5_wind10_mps", "era5_mslp_pa", "era5_sst_k",
    "era5_tcw_kg_m2", "era5_dewpoint_k",
    "imerg_center_precipitation_cal_mm_hr", "imerg_window_mean_precipitation_cal_mm_hr",
    "imerg_window_p95_precipitation_cal_mm_hr", "imerg_window_max_precipitation_cal_mm_hr",
    "imerg_window_mean_random_error_mm_hr", "imerg_window_mean_quality_index",
    "imerg_accumulation_6h_mm", "imerg_accumulation_24h_mm",
]
TARGET_COLUMNS = ["target_delta_latitude_24h", "target_delta_longitude_24h", "target_wind_change_knots_24h"]


def finite_float(row: dict[str, str], column: str) -> float:
    value = row.get(column, "").strip()
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{column} is blank or non-numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{column} is not finite")
    return parsed


def is_complete(row: dict[str, str]) -> bool:
    return row.get("multisource_complete", "").strip().lower() == "true"


def load_complete_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Multi-source training table not found: {path}")
    rows: list[dict[str, Any]] = []
    counts = {"total": 0, "incomplete": 0, "invalid_complete": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"split", "storm_id", "input_timestamp_utc", "target_latitude", "target_longitude", *FEATURE_COLUMNS, *TARGET_COLUMNS, "multisource_complete"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Multi-source training table is missing columns: {', '.join(missing)}")
        for source_row in reader:
            counts["total"] += 1
            if not is_complete(source_row):
                counts["incomplete"] += 1
                continue
            try:
                row = {
                    "storm_id": source_row["storm_id"],
                    "issue_time": source_row["input_timestamp_utc"],
                    "split": source_row["split"],
                    "features": [finite_float(source_row, column) for column in FEATURE_COLUMNS],
                    "target_latitude": finite_float(source_row, "target_latitude"),
                    "target_longitude": finite_float(source_row, "target_longitude"),
                    "targets": [finite_float(source_row, column) for column in TARGET_COLUMNS],
                }
            except ValueError:
                counts["invalid_complete"] += 1
                continue
            rows.append(row)
    return rows, counts


def metric_values(rows: list[dict[str, Any]], latitude_model: RidgeRegressor, longitude_model: RidgeRegressor, wind_model: RidgeRegressor) -> dict[str, float | int]:
    endpoint_errors: list[float] = []
    wind_errors: list[float] = []
    for row in rows:
        latitude = row["features"][0] + latitude_model.predict(row["features"])
        longitude = normalize_longitude(row["features"][1] + longitude_model.predict(row["features"]))
        endpoint_errors.append(haversine_km(latitude, longitude, row["target_latitude"], row["target_longitude"]))
        target_wind = row["features"][2] + row["targets"][2]
        predicted_wind = max(0.0, row["features"][2] + wind_model.predict(row["features"]))
        wind_errors.append(abs(predicted_wind - target_wind))
    return {
        "samples": len(rows),
        "endpoint_mae_km": round(fmean(endpoint_errors), 3),
        "wind_mae_knots": round(fmean(wind_errors), 3),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a complete-case ERA5 + IMERG ridge research benchmark.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--minimum-complete-rows", type=int, default=30)
    args = parser.parse_args()
    if args.ridge_alpha < 0 or args.minimum_complete_rows < 10:
        parser.error("--ridge-alpha must be non-negative and --minimum-complete-rows must be at least 10.")

    rows, counts = load_complete_rows(args.input)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(rows) < args.minimum_complete_rows or len(train_rows) < 10 or len(test_rows) < 5:
        raise RuntimeError(
            "Not enough complete ERA5 + IMERG examples to train. "
            f"Found {len(rows)} complete rows (train {len(train_rows)}, test {len(test_rows)}); "
            "collect and extract both source layers, then rebuild the multi-source dataset."
        )

    features = [row["features"] for row in train_rows]
    latitude_model = RidgeRegressor(args.ridge_alpha).fit(features, [row["targets"][0] for row in train_rows])
    longitude_model = RidgeRegressor(args.ridge_alpha).fit(features, [row["targets"][1] for row in train_rows])
    wind_model = RidgeRegressor(args.ridge_alpha).fit(features, [row["targets"][2] for row in train_rows])
    generated_at = datetime.now(UTC).isoformat()
    metadata = {
        "generated_at_utc": generated_at,
        "input_path": str(args.input.resolve()),
        "feature_columns": FEATURE_COLUMNS,
        "selection_policy": "Complete-case only: era5_available AND imerg_available; no missing values are imputed.",
        "model_note": "Research benchmark only; not an operational forecast or safety product.",
        "coverage": {**counts, "complete_used": len(rows), "train": len(train_rows), "test": len(test_rows)},
    }
    write_json(args.output_dir / "multisource_baseline_model.json", {
        **metadata,
        "models": {
            "latitude_delta_degrees": latitude_model.as_dict(FEATURE_COLUMNS),
            "longitude_delta_degrees": longitude_model.as_dict(FEATURE_COLUMNS),
            "wind_delta_knots": wind_model.as_dict(FEATURE_COLUMNS),
        },
    })
    metrics = metric_values(test_rows, latitude_model, longitude_model, wind_model)
    write_json(args.output_dir / "multisource_baseline_metrics.json", {**metadata, "test_metrics": metrics})
    print("Multi-source ridge research benchmark complete")
    print(f"Complete rows used: {len(rows):,} (train {len(train_rows):,}; test {len(test_rows):,})")
    print(f"Held-out endpoint MAE: {metrics['endpoint_mae_km']} km; wind MAE: {metrics['wind_mae_knots']} kt")


if __name__ == "__main__":
    main()
