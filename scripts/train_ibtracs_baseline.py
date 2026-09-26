"""Train and evaluate a dependency-free 24-hour IBTrACS forecasting baseline.

The project starts with NOAA IBTrACS best-track observations.  This script
turns each storm's 3-hourly positions into forecasting samples, then trains
small ridge-regression models for:

* the 24-hour latitude/longitude displacement; and
* the 24-hour maximum-wind change.

It intentionally uses only the Python standard library so a teammate can run
the first reproducible benchmark without a GPU or a separate ML environment.
The model is a benchmark, not an operational forecast and must not be used for
public safety decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ibtracs_ni_tracks.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
EARTH_RADIUS_KM = 6_371.0088


@dataclass(frozen=True)
class Observation:
    """One normalized IBTrACS observation."""

    storm_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    wind_knots: float | None


@dataclass(frozen=True)
class ForecastSample:
    """Features and targets available at a storm observation time."""

    issue_time: datetime
    target_time: datetime
    storm_id: str
    track_features: tuple[float, ...]
    intensity_features: tuple[float, ...] | None
    latitude_delta: float
    longitude_delta: float
    target_wind_knots: float | None
    current_latitude: float
    current_longitude: float
    current_wind_knots: float | None
    persistence_latitude: float
    persistence_longitude: float


TRACK_FEATURE_NAMES = [
    "latitude_deg",
    "longitude_deg",
    "latitude_motion_last_6h_deg",
    "longitude_motion_last_6h_deg",
    "storm_age_days",
    "season_sin",
    "season_cos",
]
INTENSITY_FEATURE_NAMES = [
    "wind_knots",
    "wind_change_last_6h_knots",
    "wind_change_available",
    "latitude_deg",
    "longitude_deg",
    "storm_age_days",
    "season_sin",
    "season_cos",
]


class RidgeRegressor:
    """A tiny, standardized ridge regressor implemented for reproducibility.

    The model stores standardized-feature coefficients.  Keeping this class in
    the script avoids adding heavyweight dependencies before the project has a
    stable training-data contract.
    """

    def __init__(self, alpha: float) -> None:
        if alpha < 0:
            raise ValueError("ridge alpha must be non-negative")
        self.alpha = alpha
        self.feature_means: list[float] = []
        self.feature_scales: list[float] = []
        self.coefficients: list[float] = []  # Intercept followed by feature weights.

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "RidgeRegressor":
        if not rows or len(rows) != len(targets):
            raise ValueError("features and targets must be non-empty and equal length")
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("feature rows must have a consistent non-zero width")

        self.feature_means = [fmean(row[column] for row in rows) for column in range(width)]
        self.feature_scales = []
        for column, mean_value in enumerate(self.feature_means):
            variance = fmean((row[column] - mean_value) ** 2 for row in rows)
            # Constant columns should remain usable rather than divide by zero.
            self.feature_scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

        # Normal equations have a tiny dimensionality here (at most 9 x 9),
        # while sample counts are in the thousands.
        dimensions = width + 1
        gram = [[0.0 for _ in range(dimensions)] for _ in range(dimensions)]
        right_hand = [0.0 for _ in range(dimensions)]
        for row, target in zip(rows, targets):
            design = [1.0] + [
                (value - mean_value) / scale
                for value, mean_value, scale in zip(row, self.feature_means, self.feature_scales)
            ]
            for left in range(dimensions):
                right_hand[left] += design[left] * target
                for right in range(left, dimensions):
                    gram[left][right] += design[left] * design[right]

        for left in range(dimensions):
            for right in range(left):
                gram[left][right] = gram[right][left]
            if left != 0:  # Do not penalize the intercept.
                gram[left][left] += self.alpha

        self.coefficients = solve_linear_system(gram, right_hand)
        return self

    def predict(self, row: Sequence[float]) -> float:
        if not self.coefficients or len(row) != len(self.feature_means):
            raise ValueError("model is not fitted or feature width does not match")
        prediction = self.coefficients[0]
        for value, mean_value, scale, coefficient in zip(
            row,
            self.feature_means,
            self.feature_scales,
            self.coefficients[1:],
        ):
            prediction += ((value - mean_value) / scale) * coefficient
        return prediction

    def as_dict(self, feature_names: Sequence[str]) -> dict[str, object]:
        return {
            "type": "ridge_linear_regression",
            "alpha": self.alpha,
            "feature_names": list(feature_names),
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
            "coefficients_standardized": self.coefficients,
        }


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve Ax=b with partial-pivot Gaussian elimination."""

    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for pivot_column in range(size):
        pivot_row = max(range(pivot_column, size), key=lambda row: abs(augmented[row][pivot_column]))
        if abs(augmented[pivot_row][pivot_column]) < 1e-12:
            raise ValueError("singular ridge-regression design matrix")
        augmented[pivot_column], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_column]
        pivot = augmented[pivot_column][pivot_column]
        for column in range(pivot_column, size + 1):
            augmented[pivot_column][column] /= pivot
        for row in range(size):
            if row == pivot_column:
                continue
            factor = augmented[row][pivot_column]
            if factor == 0:
                continue
            for column in range(pivot_column, size + 1):
                augmented[row][column] -= factor * augmented[pivot_column][column]
    return [augmented[row][size] for row in range(size)]


def parse_optional_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def load_observations(input_path: Path) -> dict[str, list[Observation]]:
    """Load observations grouped and ordered by storm."""

    grouped: dict[str, list[Observation]] = {}
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                observation = Observation(
                    storm_id=row["storm_id"],
                    timestamp=datetime.fromisoformat(row["timestamp_utc"]).replace(tzinfo=UTC),
                    latitude=float(row["latitude"]),
                    longitude=float(row["longitude"]),
                    wind_knots=parse_optional_float(row.get("wind_knots")),
                )
            except (KeyError, TypeError, ValueError):
                # The normalized collector is responsible for schema quality;
                # skipping a malformed row makes the training run resilient.
                continue
            grouped.setdefault(observation.storm_id, []).append(observation)
    for observations in grouped.values():
        observations.sort(key=lambda item: item.timestamp)
    return grouped


def season_features(timestamp: datetime) -> tuple[float, float]:
    day_of_year = timestamp.timetuple().tm_yday
    angle = 2 * math.pi * day_of_year / 365.25
    return math.sin(angle), math.cos(angle)


def normalized_longitude_delta(start: float, end: float) -> float:
    """Return the shortest signed longitude displacement in degrees."""

    return (end - start + 180.0) % 360.0 - 180.0


def normalize_longitude(longitude: float) -> float:
    return (longitude + 180.0) % 360.0 - 180.0


def build_samples(
    grouped_observations: dict[str, list[Observation]],
    horizon_hours: int,
    motion_window_hours: int = 6,
) -> list[ForecastSample]:
    """Create exact-horizon samples without interpolating best-track data."""

    if horizon_hours <= 0 or motion_window_hours <= 0:
        raise ValueError("forecast and motion horizons must be positive")
    horizon = timedelta(hours=horizon_hours)
    motion_window = timedelta(hours=motion_window_hours)
    samples: list[ForecastSample] = []

    for storm_id, observations in grouped_observations.items():
        if len(observations) < 3:
            continue
        # Find exact timestamps instead of assuming every historical storm uses
        # the same cadence. This remains valid when a future source has gaps.
        by_time = {observation.timestamp: observation for observation in observations}
        first_time = observations[0].timestamp
        for current in observations:
            previous = by_time.get(current.timestamp - motion_window)
            target = by_time.get(current.timestamp + horizon)
            if previous is None or target is None:
                continue

            latitude_motion = current.latitude - previous.latitude
            longitude_motion = normalized_longitude_delta(previous.longitude, current.longitude)
            age_days = (current.timestamp - first_time).total_seconds() / 86_400
            season_sin, season_cos = season_features(current.timestamp)
            track_features = (
                current.latitude,
                current.longitude,
                latitude_motion,
                longitude_motion,
                age_days,
                season_sin,
                season_cos,
            )

            intensity_features: tuple[float, ...] | None = None
            if current.wind_knots is not None:
                if previous.wind_knots is None:
                    wind_motion, wind_motion_available = 0.0, 0.0
                else:
                    wind_motion = current.wind_knots - previous.wind_knots
                    wind_motion_available = 1.0
                intensity_features = (
                    current.wind_knots,
                    wind_motion,
                    wind_motion_available,
                    current.latitude,
                    current.longitude,
                    age_days,
                    season_sin,
                    season_cos,
                )

            scale = horizon_hours / motion_window_hours
            samples.append(
                ForecastSample(
                    issue_time=current.timestamp,
                    target_time=target.timestamp,
                    storm_id=storm_id,
                    track_features=track_features,
                    intensity_features=intensity_features,
                    latitude_delta=target.latitude - current.latitude,
                    longitude_delta=normalized_longitude_delta(current.longitude, target.longitude),
                    target_wind_knots=target.wind_knots,
                    current_latitude=current.latitude,
                    current_longitude=current.longitude,
                    current_wind_knots=current.wind_knots,
                    persistence_latitude=current.latitude + latitude_motion * scale,
                    persistence_longitude=normalize_longitude(current.longitude + longitude_motion * scale),
                )
            )
    return sorted(samples, key=lambda item: (item.issue_time, item.storm_id))


def chronological_split(
    samples: Sequence[ForecastSample], test_fraction: float
) -> tuple[list[ForecastSample], list[ForecastSample], datetime]:
    """Split by issue time and prevent target observations leaking over cutoff."""

    if not 0 < test_fraction < 0.5:
        raise ValueError("test fraction must be greater than 0 and less than 0.5")
    if len(samples) < 10:
        raise ValueError("not enough samples to split")
    cutoff_index = int(len(samples) * (1 - test_fraction))
    cutoff = samples[cutoff_index].issue_time
    train = [sample for sample in samples if sample.target_time < cutoff]
    test = [sample for sample in samples if sample.issue_time >= cutoff]
    if not train or not test:
        raise ValueError("chronological split produced an empty train or test set")
    return train, test, cutoff


def haversine_km(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    latitude_a_rad, latitude_b_rad = math.radians(latitude_a), math.radians(latitude_b)
    delta_latitude = latitude_b_rad - latitude_a_rad
    delta_longitude = math.radians(normalized_longitude_delta(longitude_a, longitude_b))
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(latitude_a_rad) * math.cos(latitude_b_rad) * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(haversine)))


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    values = [abs(left - right) for left, right in zip(actual, predicted)]
    return fmean(values)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    values = [(left - right) ** 2 for left, right in zip(actual, predicted)]
    return math.sqrt(fmean(values))


def rounded_metrics(values: dict[str, float | int]) -> dict[str, float | int]:
    return {key: round(value, 3) if isinstance(value, float) else value for key, value in values.items()}


def evaluate_track(
    samples: Sequence[ForecastSample], latitude_model: RidgeRegressor, longitude_model: RidgeRegressor
) -> dict[str, float | int]:
    endpoint_errors: list[float] = []
    persistence_errors: list[float] = []
    actual_latitudes: list[float] = []
    predicted_latitudes: list[float] = []
    actual_longitude_deltas: list[float] = []
    predicted_longitude_deltas: list[float] = []
    for sample in samples:
        predicted_latitude = sample.current_latitude + latitude_model.predict(sample.track_features)
        predicted_longitude_delta = longitude_model.predict(sample.track_features)
        predicted_longitude = normalize_longitude(sample.current_longitude + predicted_longitude_delta)
        actual_latitude = sample.current_latitude + sample.latitude_delta
        actual_longitude = normalize_longitude(sample.current_longitude + sample.longitude_delta)
        endpoint_errors.append(
            haversine_km(predicted_latitude, predicted_longitude, actual_latitude, actual_longitude)
        )
        persistence_errors.append(
            haversine_km(
                sample.persistence_latitude,
                sample.persistence_longitude,
                actual_latitude,
                actual_longitude,
            )
        )
        actual_latitudes.append(actual_latitude)
        predicted_latitudes.append(predicted_latitude)
        actual_longitude_deltas.append(sample.longitude_delta)
        predicted_longitude_deltas.append(predicted_longitude_delta)
    sorted_errors = sorted(endpoint_errors)
    percentile_90 = sorted_errors[math.ceil(len(sorted_errors) * 0.9) - 1]
    return rounded_metrics({
        "samples": len(samples),
        "endpoint_mae_km": fmean(endpoint_errors),
        "endpoint_median_km": median(endpoint_errors),
        "endpoint_p90_km": percentile_90,
        "motion_persistence_endpoint_mae_km": fmean(persistence_errors),
        "latitude_mae_degrees": mae(actual_latitudes, predicted_latitudes),
        "longitude_delta_mae_degrees": mae(actual_longitude_deltas, predicted_longitude_deltas),
    })


def evaluate_intensity(samples: Sequence[ForecastSample], model: RidgeRegressor) -> dict[str, float | int]:
    eligible = [
        sample
        for sample in samples
        if sample.intensity_features is not None
        and sample.target_wind_knots is not None
        and sample.current_wind_knots is not None
    ]
    if not eligible:
        raise ValueError("no test samples have current and target wind observations")
    actual = [sample.target_wind_knots for sample in eligible]
    predicted = [
        max(0.0, sample.current_wind_knots + model.predict(sample.intensity_features))
        for sample in eligible
    ]
    persistence = [sample.current_wind_knots for sample in eligible]
    bias = fmean(prediction - target for target, prediction in zip(actual, predicted))
    return rounded_metrics({
        "samples": len(eligible),
        "wind_mae_knots": mae(actual, predicted),
        "wind_rmse_knots": rmse(actual, predicted),
        "wind_bias_knots": bias,
        "persistence_wind_mae_knots": mae(actual, persistence),
    })


def build_prediction_records(
    samples: Sequence[ForecastSample], latitude_model: RidgeRegressor, longitude_model: RidgeRegressor, intensity_model: RidgeRegressor
) -> list[dict[str, object]]:
    """Create API-friendly held-out forecast records for replay/inspection."""

    records: list[dict[str, object]] = []
    for sample in samples:
        predicted_latitude = sample.current_latitude + latitude_model.predict(sample.track_features)
        predicted_longitude = normalize_longitude(
            sample.current_longitude + longitude_model.predict(sample.track_features)
        )
        actual_latitude = sample.current_latitude + sample.latitude_delta
        actual_longitude = normalize_longitude(sample.current_longitude + sample.longitude_delta)
        predicted_wind: float | None = None
        if sample.intensity_features is not None and sample.current_wind_knots is not None:
            predicted_wind = max(0.0, sample.current_wind_knots + intensity_model.predict(sample.intensity_features))
        record: dict[str, object] = {
            "storm_id": sample.storm_id,
            "issue_time_utc": sample.issue_time.isoformat(),
            "target_time_utc": sample.target_time.isoformat(),
            "predicted": {
                "latitude": round(predicted_latitude, 4),
                "longitude": round(predicted_longitude, 4),
                "wind_knots": round(predicted_wind, 2) if predicted_wind is not None else None,
            },
            "actual": {
                "latitude": round(actual_latitude, 4),
                "longitude": round(actual_longitude, 4),
                "wind_knots": round(sample.target_wind_knots, 2) if sample.target_wind_knots is not None else None,
            },
            "endpoint_error_km": round(
                haversine_km(predicted_latitude, predicted_longitude, actual_latitude, actual_longitude), 3
            ),
        }
        records.append(record)
    return records


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a chronological 24-hour IBTrACS ridge baseline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Normalized IBTrACS CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for ignored model/result JSON artifacts.")
    parser.add_argument("--horizon-hours", type=int, default=24, help="Forecast horizon in hours (default: 24).")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="Latest chronological share held out for evaluation.")
    parser.add_argument("--ridge-alpha", type=float, default=10.0, help="L2 penalty on standardized coefficients.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")
    observations = load_observations(args.input)
    samples = build_samples(observations, args.horizon_hours)
    train_samples, test_samples, cutoff = chronological_split(samples, args.test_fraction)

    track_train_features = [sample.track_features for sample in train_samples]
    latitude_targets = [sample.latitude_delta for sample in train_samples]
    longitude_targets = [sample.longitude_delta for sample in train_samples]
    latitude_model = RidgeRegressor(args.ridge_alpha).fit(track_train_features, latitude_targets)
    longitude_model = RidgeRegressor(args.ridge_alpha).fit(track_train_features, longitude_targets)

    intensity_train_samples = [
        sample
        for sample in train_samples
        if sample.intensity_features is not None
        and sample.target_wind_knots is not None
        and sample.current_wind_knots is not None
    ]
    intensity_model = RidgeRegressor(args.ridge_alpha).fit(
        [sample.intensity_features for sample in intensity_train_samples if sample.intensity_features is not None],
        [
            sample.target_wind_knots - sample.current_wind_knots
            for sample in intensity_train_samples
            if sample.target_wind_knots is not None and sample.current_wind_knots is not None
        ],
    )

    track_metrics = evaluate_track(test_samples, latitude_model, longitude_model)
    intensity_metrics = evaluate_intensity(test_samples, intensity_model)
    split_payload = {
        "strategy": "chronological issue-time split; train targets are strictly before the cutoff",
        "cutoff_utc": cutoff.isoformat(),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "dropped_boundary_samples": len(samples) - len(train_samples) - len(test_samples),
    }
    metadata = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_path": str(args.input.resolve()),
        "input_storms": len(observations),
        "forecast_horizon_hours": args.horizon_hours,
        "model_note": "Research baseline only; not an operational or safety-critical forecast.",
        "split": split_payload,
    }
    model_payload: dict[str, object] = {
        **metadata,
        "inference_contract": {
            "track_prediction": "Add each predicted displacement in degrees to the current latitude/longitude.",
            "intensity_prediction": "Add predicted wind_delta_knots to current wind_knots; floor the result at zero.",
            "longitude_normalization": "Normalize resulting longitude to [-180, 180).",
        },
        "track": {
            "latitude_delta_degrees": latitude_model.as_dict(TRACK_FEATURE_NAMES),
            "longitude_delta_degrees": longitude_model.as_dict(TRACK_FEATURE_NAMES),
        },
        "intensity": {
            "wind_delta_knots": intensity_model.as_dict(INTENSITY_FEATURE_NAMES),
        },
    }
    metrics_payload: dict[str, object] = {
        **metadata,
        "training_samples": {
            "track": len(train_samples),
            "intensity": len(intensity_train_samples),
        },
        "test_metrics": {
            "track": track_metrics,
            "intensity": intensity_metrics,
        },
    }
    predictions_payload: dict[str, object] = {
        **metadata,
        "records_note": "Chronological held-out forecasts. Each record predicts exactly forecast_horizon_hours after issue_time_utc.",
        "records": build_prediction_records(test_samples, latitude_model, longitude_model, intensity_model),
    }
    # Stable generic names make the artifacts easy for the FastAPI layer to
    # consume, while their contents carry the exact data/model metadata.
    model_path = args.output_dir / "baseline_model.json"
    metrics_path = args.output_dir / "baseline_metrics.json"
    predictions_path = args.output_dir / "baseline_predictions.json"
    write_json(model_path, model_payload)
    write_json(metrics_path, metrics_payload)
    write_json(predictions_path, predictions_payload)

    print("24-hour IBTrACS ridge baseline complete")
    print(f"Input: {args.input}")
    print(f"Chronological cutoff: {cutoff.isoformat()}")
    print(f"Track samples - train: {len(train_samples):,}; test: {len(test_samples):,}")
    print(
        "Track endpoint MAE: "
        f"{track_metrics['endpoint_mae_km']} km "
        f"(motion persistence: {track_metrics['motion_persistence_endpoint_mae_km']} km)"
    )
    print(
        "Intensity MAE: "
        f"{intensity_metrics['wind_mae_knots']} kt "
        f"(persistence: {intensity_metrics['persistence_wind_mae_knots']} kt; "
        f"test samples: {intensity_metrics['samples']})"
    )
    print(f"Model artifact: {model_path}")
    print(f"Metrics artifact: {metrics_path}")
    print(f"Held-out prediction artifact: {predictions_path}")


if __name__ == "__main__":
    main()
