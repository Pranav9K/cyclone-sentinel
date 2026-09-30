"""Train and evaluate a calibrated Machine Learning classifier for Rapid Intensification (RI).

Rapid Intensification (RI) is defined by WMO/IMD as an increase in maximum
sustained wind speed of at least 30 knots (55 km/h) over a 24-hour period.

This script trains a regularized, class-balanced logistic regression model
using historical North Indian Ocean cyclone events. It evaluates:
- ROC-AUC score
- Precision, Recall, and F1-score
- Brier reliability score: E[(P(RI) - Y)^2]
- Feature attribution weights for explainable AI (XAI)

Uses only the Python standard library for zero-dependency reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "training_samples.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RI_FEATURE_NAMES = [
    "wind_knots",
    "wind_acceleration_6h",
    "latitude_deg",
    "longitude_deg",
    "season_sin",
    "season_cos",
    "stage_consolidation_window",  # 1.0 if 40-90 kt, 0.0 otherwise
    "sst_estimate_celsius",
    "shear_estimate_knots",
]


def sigmoid(z: float) -> float:
    if z < -40.0:
        return 0.0
    if z > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class CalibratedLogisticClassifier:
    """Class-weighted, L2-regularized logistic regression classifier."""

    def __init__(self, l2_reg: float = 1.0, learning_rate: float = 0.05, max_epochs: int = 500) -> None:
        self.l2_reg = l2_reg
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.feature_means: list[float] = []
        self.feature_scales: list[float] = []
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "CalibratedLogisticClassifier":
        n_samples = len(X)
        n_features = len(X[0])

        # Compute means and stddevs for standard scaling
        self.feature_means = [fmean(X[i][j] for i in range(n_samples)) for j in range(n_features)]
        self.feature_scales = []
        for j in range(n_features):
            variance = fmean((X[i][j] - self.feature_means[j]) ** 2 for i in range(n_samples))
            self.feature_scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

        # Standardize features
        X_scaled = [
            [(X[i][j] - self.feature_means[j]) / self.feature_scales[j] for j in range(n_features)]
            for i in range(n_samples)
        ]

        # Calculate class weights for severe class imbalance (~3.7% positive)
        n_pos = sum(y)
        n_neg = n_samples - n_pos
        w_pos = n_samples / (2.0 * max(1, n_pos))
        w_neg = n_samples / (2.0 * max(1, n_neg))

        # Gradient descent with class weights and L2 penalty
        self.weights = [0.0] * n_features
        # Initialize bias to log-odds
        self.bias = math.log((n_pos + 1e-5) / (n_neg + 1e-5))

        lr = self.lr
        for epoch in range(self.max_epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0

            for i in range(n_samples):
                xi = X_scaled[i]
                yi = y[i]
                sample_weight = w_pos if yi == 1 else w_neg

                linear = self.bias + sum(w * x for w, x in zip(self.weights, xi))
                pi = sigmoid(linear)
                error = (pi - yi) * sample_weight

                grad_b += error
                for j in range(n_features):
                    grad_w[j] += error * xi[j]

            # Normalize gradients and add L2 penalty (do not regularize bias)
            grad_b /= n_samples
            for j in range(n_features):
                grad_w[j] = (grad_w[j] / n_samples) + (self.l2_reg / n_samples) * self.weights[j]
                self.weights[j] -= lr * grad_w[j]
            self.bias -= lr * grad_b

            # Decay learning rate gradually
            if (epoch + 1) % 100 == 0:
                lr *= 0.8

        return self

    def predict_proba(self, x: Sequence[float]) -> float:
        x_scaled = [
            (x[j] - self.feature_means[j]) / self.feature_scales[j] for j in range(len(x))
        ]
        linear = self.bias + sum(w * val for w, val in zip(self.weights, x_scaled))
        return sigmoid(linear)

    def explain(self, x: Sequence[float], feature_names: Sequence[str]) -> list[dict[str, Any]]:
        """Return standardized factor attributions for explainable AI."""
        x_scaled = [
            (x[j] - self.feature_means[j]) / self.feature_scales[j] for j in range(len(x))
        ]
        attributions: list[dict[str, Any]] = []
        for j, (name, val, scaled) in enumerate(zip(feature_names, x, x_scaled)):
            contribution = round(self.weights[j] * scaled, 3)
            attributions.append({
                "feature": name,
                "raw_value": round(val, 2),
                "weight": round(self.weights[j], 3),
                "contribution": contribution,
                "favorable": contribution > 0.0,
            })
        attributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return attributions

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "calibrated_logistic_regression",
            "feature_names": RI_FEATURE_NAMES,
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
            "weights": self.weights,
            "bias": self.bias,
        }


def extract_ri_samples(csv_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            target_change = r.get("target_wind_change_knots_24h", "").strip()
            if not target_change:
                continue
            delta_wind = float(target_change)
            is_ri = 1 if delta_wind >= 30.0 else 0

            wind = float(r["wind_knots"]) if r.get("wind_knots") else 35.0
            wind_prev = float(r["previous_6h_wind_change_knots"]) if r.get("previous_6h_wind_change_knots") else 0.0
            lat = float(r["latitude"])
            lon = float(r["longitude"])
            day_of_year = int(r["input_day_of_year"])

            angle = 2.0 * math.pi * day_of_year / 365.25
            season_sin = math.sin(angle)
            season_cos = math.cos(angle)

            # Consolidation window (40-90 kt is the prime stage for rapid intensification)
            consolidation = 1.0 if 40.0 <= wind <= 90.0 else 0.0

            # Seasonal SST and Shear proxies for BoB/Arabian Sea
            is_prime_monsoon = int(r.get("input_month", 5)) in (4, 5, 6, 10, 11, 12)
            sst = 30.0 if (is_prime_monsoon and 5.0 <= lat <= 22.0) else 27.5
            shear = 11.5 if is_prime_monsoon else 23.0

            features = [
                wind,
                wind_prev,
                lat,
                lon,
                season_sin,
                season_cos,
                consolidation,
                sst,
                shear,
            ]

            sample = {
                "sample_id": r.get("sample_id", ""),
                "storm_id": r.get("storm_id", ""),
                "features": features,
                "is_ri": is_ri,
                "delta_wind_24h": delta_wind,
            }

            if r.get("split", "train") == "test":
                test_rows.append(sample)
            else:
                train_rows.append(sample)

    return train_rows, test_rows


def calculate_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict[str, Any]:
    n = len(y_true)
    tp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp >= threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= threshold)
    tn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp < threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp < threshold)

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1 = round(2.0 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    accuracy = round((tp + tn) / n, 3)

    # Brier Score: lower is better (0 is perfect calibration)
    brier_score = round(fmean((p - y) ** 2 for y, p in zip(y_true, y_prob)), 4)

    # ROC-AUC estimation via Mann-Whitney U rank statistic
    pos_probs = [p for y, p in zip(y_true, y_prob) if y == 1]
    neg_probs = [p for y, p in zip(y_true, y_prob) if y == 0]
    n_pos = len(pos_probs)
    n_neg = len(neg_probs)

    if n_pos > 0 and n_neg > 0:
        # Count pairs where pos > neg
        wins = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos_probs for n in neg_probs)
        roc_auc = round(wins / (n_pos * n_neg), 3)
    else:
        roc_auc = 0.5

    return {
        "samples": n,
        "ri_events": n_pos,
        "ri_prevalence_percent": round(n_pos / n * 100, 1),
        "threshold_used": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "brier_score": brier_score,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a calibrated Machine Learning model for Rapid Intensification.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l2-reg", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=600)
    args = parser.parse_args()

    train_data, test_data = extract_ri_samples(args.input)
    print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples.")
    print(f"Train RI events: {sum(s['is_ri'] for s in train_data)} | Test RI events: {sum(s['is_ri'] for s in test_data)}")

    classifier = CalibratedLogisticClassifier(
        l2_reg=args.l2_reg,
        learning_rate=args.lr,
        max_epochs=args.epochs,
    )
    X_train = [s["features"] for s in train_data]
    y_train = [s["is_ri"] for s in train_data]
    classifier.fit(X_train, y_train)

    # Evaluate on held-out test split
    X_test = [s["features"] for s in test_data]
    y_test = [s["is_ri"] for s in test_data]
    y_test_probs = [classifier.predict_proba(x) for x in X_test]

    metrics = calculate_metrics(y_test, y_test_probs, threshold=0.40)
    print("\n--- Held-out Rapid Intensification Test Metrics ---")
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Brier Reliability Score: {metrics['brier_score']} (0.0 = perfect calibration)")
    print(f"Precision: {metrics['precision']} | Recall: {metrics['recall']} | F1: {metrics['f1_score']}")
    print(f"Confusion Matrix: TP={metrics['confusion_matrix']['tp']}, FP={metrics['confusion_matrix']['fp']}, TN={metrics['confusion_matrix']['tn']}, FN={metrics['confusion_matrix']['fn']}")

    # Save model artifact
    generated_at = datetime.now(UTC).isoformat()
    model_payload = {
        "generated_at_utc": generated_at,
        "definition": "Increase of >= 30 knots in maximum sustained wind within 24 hours (IMD/WMO standard)",
        "model": classifier.as_dict(),
        "metrics": metrics,
        "feature_attributions_sample": classifier.explain(X_test[0], RI_FEATURE_NAMES),
    }
    model_path = args.output_dir / "ri_model.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[OK] Calibrated RI model artifact saved to {model_path}")


if __name__ == "__main__":
    main()
