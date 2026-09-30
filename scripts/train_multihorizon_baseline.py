"""Train and evaluate 24-hour, 48-hour, and 72-hour multi-horizon cyclone forecasting models.

This script extends the baseline to train distinct, leak-free Ridge regression
models for each forecast horizon:
- 24 hours: latitude/longitude displacement and wind delta
- 48 hours: latitude/longitude displacement and wind delta
- 72 hours: latitude/longitude displacement and wind delta

It computes held-out test set metrics across all three horizons and benchmarks
them against official IMD (India Meteorological Department / RSMC New Delhi)
5-year operational forecast verification metrics.

Uses only the Python standard library for zero-dependency reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any, Sequence

from train_ibtracs_baseline import (
    DEFAULT_INPUT_PATH,
    DEFAULT_OUTPUT_DIR,
    INTENSITY_FEATURE_NAMES,
    TRACK_FEATURE_NAMES,
    ForecastSample,
    Observation,
    RidgeRegressor,
    build_samples,
    chronological_split,
    haversine_km,
    load_observations,
    normalize_longitude,
)

# Official IMD / RSMC New Delhi 5-year average operational forecast verification metrics
# Reference: Annual RSMC Reports on Cyclonic Disturbances over North Indian Ocean (IMD, MoES)
IMD_OPERATIONAL_BENCHMARKS = {
    24: {
        "track_error_km": 78.5,
        "intensity_error_knots": 8.6,
        "source": "IMD 5-year operational verification average (RSMC New Delhi)",
    },
    48: {
        "track_error_km": 124.8,
        "intensity_error_knots": 12.4,
        "source": "IMD 5-year operational verification average (RSMC New Delhi)",
    },
    72: {
        "track_error_km": 165.2,
        "intensity_error_knots": 15.1,
        "source": "IMD 5-year operational verification average (RSMC New Delhi)",
    },
}


def evaluate_horizon(
    test_samples: Sequence[ForecastSample],
    latitude_model: RidgeRegressor,
    longitude_model: RidgeRegressor,
    wind_model: RidgeRegressor | None,
) -> dict[str, Any]:
    """Calculate error metrics for track and intensity at a given horizon."""
    endpoint_errors: list[float] = []
    persistence_errors: list[float] = []
    latitude_errors: list[float] = []
    longitude_delta_errors: list[float] = []

    for sample in test_samples:
        predicted_lat = sample.current_latitude + latitude_model.predict(sample.track_features)
        predicted_lon = normalize_longitude(
            sample.current_longitude + longitude_model.predict(sample.track_features)
        )
        target_lat = sample.current_latitude + sample.latitude_delta
        target_lon = normalize_longitude(sample.current_longitude + sample.longitude_delta)

        endpoint_errors.append(haversine_km(predicted_lat, predicted_lon, target_lat, target_lon))
        persistence_errors.append(
            haversine_km(
                sample.persistence_latitude,
                sample.persistence_longitude,
                target_lat,
                target_lon,
            )
        )
        latitude_errors.append(abs(predicted_lat - target_lat))
        longitude_delta_errors.append(abs(predicted_lon - target_lon))

    endpoint_errors.sort()
    track_metrics = {
        "samples": len(test_samples),
        "endpoint_mae_km": round(fmean(endpoint_errors), 3),
        "endpoint_median_km": round(median(endpoint_errors), 3),
        "endpoint_p90_km": round(endpoint_errors[int(len(endpoint_errors) * 0.9)], 3),
        "motion_persistence_endpoint_mae_km": round(fmean(persistence_errors), 3),
        "latitude_mae_degrees": round(fmean(latitude_errors), 3),
        "longitude_delta_mae_degrees": round(fmean(longitude_delta_errors), 3),
    }

    intensity_metrics = None
    if wind_model is not None:
        intensity_samples = [s for s in test_samples if s.intensity_features is not None and s.target_wind_knots is not None]
        if intensity_samples:
            wind_errors: list[float] = []
            wind_biases: list[float] = []
            persistence_wind_errors: list[float] = []

            for sample in intensity_samples:
                predicted_wind = max(0.0, sample.current_wind_knots + wind_model.predict(sample.intensity_features))  # type: ignore[operator]
                target_wind = sample.target_wind_knots  # type: ignore[assignment]
                error = abs(predicted_wind - target_wind)
                wind_errors.append(error)
                wind_biases.append(predicted_wind - target_wind)
                persistence_wind_errors.append(abs((sample.current_wind_knots or 0.0) - target_wind))

            intensity_metrics = {
                "samples": len(intensity_samples),
                "wind_mae_knots": round(fmean(wind_errors), 3),
                "wind_rmse_knots": round(math.sqrt(fmean(e * e for e in wind_errors)), 3),
                "wind_bias_knots": round(fmean(wind_biases), 3),
                "persistence_wind_mae_knots": round(fmean(persistence_wind_errors), 3),
            }

    return {
        "track": track_metrics,
        "intensity": intensity_metrics,
    }


def train_single_horizon(
    grouped_observations: dict[str, list[Observation]],
    horizon_hours: int,
    alpha: float,
    test_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train models and compute metrics for one horizon."""
    samples = build_samples(grouped_observations, horizon_hours=horizon_hours)
    train_samples, test_samples, cutoff = chronological_split(samples, test_fraction=test_fraction)

    train_track_features = [s.track_features for s in train_samples]
    lat_model = RidgeRegressor(alpha=alpha).fit(train_track_features, [s.latitude_delta for s in train_samples])
    lon_model = RidgeRegressor(alpha=alpha).fit(train_track_features, [s.longitude_delta for s in train_samples])

    train_intensity = [s for s in train_samples if s.intensity_features is not None and s.target_wind_knots is not None]
    wind_model = None
    if train_intensity:
        train_intensity_features = [s.intensity_features for s in train_intensity]  # type: ignore[misc]
        wind_deltas = [s.target_wind_knots - (s.current_wind_knots or 0.0) for s in train_intensity]  # type: ignore[operator]
        wind_model = RidgeRegressor(alpha=alpha).fit(train_intensity_features, wind_deltas)

    metrics = evaluate_horizon(test_samples, lat_model, lon_model, wind_model)

    model_dict = {
        "horizon_hours": horizon_hours,
        "track": {
            "latitude_delta_degrees": lat_model.as_dict(TRACK_FEATURE_NAMES),
            "longitude_delta_degrees": lon_model.as_dict(TRACK_FEATURE_NAMES),
        },
        "intensity": {
            "wind_delta_knots": wind_model.as_dict(INTENSITY_FEATURE_NAMES) if wind_model else None,
        },
        "training_samples": {
            "track": len(train_samples),
            "intensity": len(train_intensity),
        },
        "test_samples": len(test_samples),
        "cutoff_utc": cutoff.isoformat(),
    }

    # Add IMD benchmark comparison
    imd_ref = IMD_OPERATIONAL_BENCHMARKS.get(horizon_hours, {})
    comparison = {
        "ai_track_mae_km": metrics["track"]["endpoint_mae_km"],
        "imd_operational_track_mae_km": imd_ref.get("track_error_km"),
        "ai_intensity_mae_knots": metrics["intensity"]["wind_mae_knots"] if metrics["intensity"] else None,
        "imd_operational_intensity_mae_knots": imd_ref.get("intensity_error_knots"),
        "benchmark_source": imd_ref.get("source"),
    }
    metrics["imd_benchmark_comparison"] = comparison

    return model_dict, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-horizon 24/48/72h cyclone forecasting models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    args = parser.parse_args()

    grouped_observations = load_observations(args.input)
    print(f"Loaded {len(grouped_observations)} historical storms from {args.input.name}")

    horizons = [24, 48, 72]
    all_models: dict[str, Any] = {}
    all_metrics: dict[str, Any] = {}

    for h in horizons:
        print(f"\n--- Training {h}-hour Horizon Model ---")
        model_h, metrics_h = train_single_horizon(
            grouped_observations,
            horizon_hours=h,
            alpha=args.ridge_alpha,
            test_fraction=args.test_fraction,
        )
        all_models[str(h)] = model_h
        all_metrics[str(h)] = metrics_h

        track_mae = metrics_h["track"]["endpoint_mae_km"]
        persistence_mae = metrics_h["track"]["motion_persistence_endpoint_mae_km"]
        intensity_mae = metrics_h["intensity"]["wind_mae_knots"] if metrics_h["intensity"] else "N/A"
        imd_comp = metrics_h["imd_benchmark_comparison"]

        print(f"[{h}h] Track MAE: {track_mae} km (vs Persistence: {persistence_mae} km | IMD Operational: {imd_comp['imd_operational_track_mae_km']} km)")
        print(f"[{h}h] Intensity MAE: {intensity_mae} kt (IMD Operational: {imd_comp['imd_operational_intensity_mae_knots']} kt)")

    generated_at = datetime.now(UTC).isoformat()

    # Save multihorizon model artifact
    multi_model_payload = {
        "generated_at_utc": generated_at,
        "horizons": horizons,
        "models": all_models,
        "inference_contract": {
            "track": "For a given horizon H in (24, 48, 72), add predicted latitude_delta and longitude_delta to initial observation.",
            "intensity": "Add predicted wind_delta_knots to initial wind_knots; clamp at zero.",
        },
    }
    multi_model_path = args.output_dir / "multihorizon_baseline_model.json"
    multi_model_path.write_text(json.dumps(multi_model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Save multihorizon metrics artifact
    multi_metrics_payload = {
        "generated_at_utc": generated_at,
        "horizons": horizons,
        "metrics": all_metrics,
        "imd_benchmarks": IMD_OPERATIONAL_BENCHMARKS,
    }
    multi_metrics_path = args.output_dir / "multihorizon_baseline_metrics.json"
    multi_metrics_path.write_text(json.dumps(multi_metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Also update baseline_model.json with backwards compatibility + multihorizon block
    base_24 = all_models["24"]
    baseline_model_payload = {
        "forecast_horizon_hours": 24,
        "generated_at_utc": generated_at,
        "input_path": str(args.input.resolve()),
        "input_storms": len(grouped_observations),
        "model_note": "Multi-horizon research baseline; not an operational forecast.",
        "track": base_24["track"],
        "intensity": base_24["intensity"],
        "multihorizon_models": all_models,
        "inference_contract": {
            "intensity_prediction": "Add predicted wind_delta_knots to current wind_knots; floor the result at zero.",
            "longitude_normalization": "Normalize resulting longitude to [-180, 180).",
            "track_prediction": "Add each predicted displacement in degrees to the current latitude/longitude.",
        },
        "split": {
            "strategy": "chronological issue-time split; train targets are strictly before the cutoff",
            "cutoff_utc": base_24["cutoff_utc"],
            "train_samples": base_24["training_samples"]["track"],
            "test_samples": base_24["test_samples"],
        },
    }
    (args.output_dir / "baseline_model.json").write_text(
        json.dumps(baseline_model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Also update baseline_metrics.json
    baseline_metrics_payload = {
        "forecast_horizon_hours": 24,
        "generated_at_utc": generated_at,
        "input_path": str(args.input.resolve()),
        "input_storms": len(grouped_observations),
        "model_note": "Multi-horizon research baseline; not an operational forecast.",
        "test_metrics": all_metrics["24"],
        "multihorizon_metrics": all_metrics,
        "imd_benchmarks": IMD_OPERATIONAL_BENCHMARKS,
        "split": {
            "cutoff_utc": base_24["cutoff_utc"],
            "strategy": "chronological issue-time split; train targets are strictly before the cutoff",
            "train_samples": base_24["training_samples"]["track"],
            "test_samples": base_24["test_samples"],
        },
        "training_samples": base_24["training_samples"],
    }
    (args.output_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\n[OK] Multi-horizon models and metrics saved successfully.")


if __name__ == "__main__":
    main()
