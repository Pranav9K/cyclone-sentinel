"""Train and evaluate a multi-class Machine Learning model for Tropical Cyclone Pattern Classification.

The five canonical cloud pattern classes defined by Dvorak (WMO / IMD) are:
1. Eye Pattern
2. Central Dense Overcast (CDO)
3. Curved Band Pattern
4. Embedded Center Pattern
5. Shear Pattern

This script trains a regularized Softmax (multinomial logistic) classification model
that predicts pattern probabilities, dominant pattern type, and model confidence
from structural morphometrics (convective symmetry, rainband wrapping, thermal contrast,
pressure deficit, and vertical wind shear).

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

PATTERN_CLASSES = [
    "Eye Pattern",
    "Central Dense Overcast (CDO)",
    "Curved Band Pattern",
    "Embedded Center Pattern",
    "Shear Pattern",
]

PATTERN_FEATURE_NAMES = [
    "wind_knots",
    "pressure_deficit_hpa",
    "convective_symmetry_pct",
    "log_spiral_wrap_turns",
    "cloud_shield_diameter_km",
    "eyewall_thermal_contrast_c",
    "vertical_wind_shear_kt",
]


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_vals = [math.exp(val - max_logit) for val in logits]
    total = sum(exp_vals)
    return [val / total for val in exp_vals]


class MulticlassSoftmaxClassifier:
    """Multinomial Softmax Classifier trained with gradient descent and L2 regularization."""

    def __init__(self, num_classes: int, num_features: int, l2_reg: float = 0.5, lr: float = 0.08, epochs: int = 500) -> None:
        self.K = num_classes
        self.D = num_features
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs

        self.feature_means: list[float] = [0.0] * num_features
        self.feature_scales: list[float] = [1.0] * num_features
        # Weights matrix of shape (K, D)
        self.weights: list[list[float]] = [[0.0] * num_features for _ in range(num_classes)]
        self.biases: list[float] = [0.0] * num_classes

    def fit(self, X: list[list[float]], y: list[int]) -> "MulticlassSoftmaxClassifier":
        n_samples = len(X)

        # Standardize features
        self.feature_means = [fmean(X[i][j] for i in range(n_samples)) for j in range(self.D)]
        self.feature_scales = []
        for j in range(self.D):
            variance = fmean((X[i][j] - self.feature_means[j]) ** 2 for i in range(n_samples))
            self.feature_scales.append(math.sqrt(variance) if variance > 1e-12 else 1.0)

        X_scaled = [
            [(X[i][j] - self.feature_means[j]) / self.feature_scales[j] for j in range(self.D)]
            for i in range(n_samples)
        ]

        lr = self.lr
        for epoch in range(self.epochs):
            grad_w = [[0.0] * self.D for _ in range(self.K)]
            grad_b = [0.0] * self.K

            for i in range(n_samples):
                xi = X_scaled[i]
                yi = y[i]

                logits = [self.biases[k] + sum(self.weights[k][j] * xi[j] for j in range(self.D)) for k in range(self.K)]
                probs = softmax(logits)

                for k in range(self.K):
                    indicator = 1.0 if yi == k else 0.0
                    error = probs[k] - indicator
                    grad_b[k] += error
                    for j in range(self.D):
                        grad_w[k][j] += error * xi[j]

            # Update weights with L2 regularization
            for k in range(self.K):
                self.biases[k] -= lr * (grad_b[k] / n_samples)
                for j in range(self.D):
                    grad = (grad_w[k][j] / n_samples) + (self.l2_reg / n_samples) * self.weights[k][j]
                    self.weights[k][j] -= lr * grad

            if (epoch + 1) % 100 == 0:
                lr *= 0.85

        return self

    def predict_proba(self, x: Sequence[float]) -> list[float]:
        x_scaled = [(x[j] - self.feature_means[j]) / self.feature_scales[j] for j in range(self.D)]
        logits = [self.biases[k] + sum(self.weights[k][j] * x_scaled[j] for j in range(self.D)) for k in range(self.K)]
        return softmax(logits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "multiclass_softmax_classifier",
            "classes": PATTERN_CLASSES,
            "feature_names": PATTERN_FEATURE_NAMES,
            "feature_means": self.feature_means,
            "feature_scales": self.feature_scales,
            "weights": self.weights,
            "biases": self.biases,
        }


def derive_pattern_ground_truth(wind: float, shear: float) -> tuple[int, list[float]]:
    """Derive ground truth label and synthetic structural features from meteorological physical principles."""
    # Mishra-Gupta NIO central pressure deficit
    delta_p = 0.0076 * (max(wind, 10.0) ** 1.89)

    # Dvorak T-number approximation
    t_num = min(8.0, max(1.0, 1.0 + (wind - 15.0) / 18.0))

    if wind >= 75.0 and shear < 20.0:
        pattern_idx = 0  # Eye Pattern
        symmetry = min(98.0, 75.0 + t_num * 3.0)
        spiral_turns = min(1.5, 0.9 + t_num * 0.08)
        cloud_diameter = 400.0 + t_num * 40.0
        thermal_contrast = 20.0 + (t_num - 4.5) * 5.0
    elif wind >= 55.0:
        pattern_idx = 1  # Central Dense Overcast (CDO)
        symmetry = min(85.0, 60.0 + t_num * 3.0)
        spiral_turns = min(1.2, 0.7 + t_num * 0.07)
        cloud_diameter = 350.0 + t_num * 35.0
        thermal_contrast = 10.0 + (t_num - 3.5) * 3.0
    elif shear >= 18.0 and wind < 50.0:
        pattern_idx = 4  # Shear Pattern
        symmetry = max(35.0, 50.0 - (shear - 18.0) * 1.5)
        spiral_turns = 0.35
        cloud_diameter = 220.0 + t_num * 25.0
        thermal_contrast = 4.0
    elif wind >= 42.0:
        pattern_idx = 3  # Embedded Center Pattern
        symmetry = 65.0
        spiral_turns = 0.75
        cloud_diameter = 300.0 + t_num * 30.0
        thermal_contrast = 8.0
    else:
        pattern_idx = 2  # Curved Band Pattern
        symmetry = 58.0
        spiral_turns = min(1.0, 0.45 + t_num * 0.1)
        cloud_diameter = 260.0 + t_num * 28.0
        thermal_contrast = 6.0

    features = [
        wind,
        round(delta_p, 1),
        round(symmetry, 1),
        round(spiral_turns, 2),
        round(cloud_diameter, 1),
        round(thermal_contrast, 1),
        round(shear, 1),
    ]
    return pattern_idx, features


def load_pattern_dataset(csv_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples: list[dict[str, Any]] = []
    test_samples: list[dict[str, Any]] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for r in reader:
            wind = float(r["wind_knots"]) if r.get("wind_knots") else 35.0
            month = int(r.get("input_month", 5))
            shear = 11.5 if month in (4, 5, 6, 10, 11, 12) else 22.5

            label_idx, features = derive_pattern_ground_truth(wind, shear)
            item = {
                "storm_id": r.get("storm_id", ""),
                "features": features,
                "label": label_idx,
                "pattern_name": PATTERN_CLASSES[label_idx],
            }
            if r.get("split", "train") == "test":
                test_samples.append(item)
            else:
                train_samples.append(item)

    return train_samples, test_samples


def evaluate_classifier(classifier: MulticlassSoftmaxClassifier, test_samples: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(test_samples)
    correct = 0
    class_totals = [0] * len(PATTERN_CLASSES)
    class_correct = [0] * len(PATTERN_CLASSES)
    brier_scores: list[float] = []

    for item in test_samples:
        probs = classifier.predict_proba(item["features"])
        pred = max(range(len(PATTERN_CLASSES)), key=lambda k: probs[k])
        true_k = item["label"]

        class_totals[true_k] += 1
        if pred == true_k:
            correct += 1
            class_correct[true_k] += 1

        brier = sum((probs[k] - (1.0 if k == true_k else 0.0)) ** 2 for k in range(len(PATTERN_CLASSES)))
        brier_scores.append(brier)

    accuracy = round(correct / n, 4)
    per_class_recall = {
        PATTERN_CLASSES[k]: round(class_correct[k] / class_totals[k], 3) if class_totals[k] > 0 else 0.0
        for k in range(len(PATTERN_CLASSES))
    }
    macro_recall = round(fmean(per_class_recall.values()), 3)

    return {
        "samples": n,
        "overall_accuracy": accuracy,
        "macro_recall": macro_recall,
        "brier_multiclass_score": round(fmean(brier_scores), 4),
        "per_class_recall": per_class_recall,
        "class_support": {PATTERN_CLASSES[k]: class_totals[k] for k in range(len(PATTERN_CLASSES))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-class tropical cyclone pattern classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--l2-reg", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train_data, test_data = load_pattern_dataset(args.input)
    print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples.")

    classifier = MulticlassSoftmaxClassifier(
        num_classes=len(PATTERN_CLASSES),
        num_features=len(PATTERN_FEATURE_NAMES),
        l2_reg=args.l2_reg,
        lr=args.lr,
        epochs=args.epochs,
    )
    X_train = [s["features"] for s in train_data]
    y_train = [s["label"] for s in train_data]
    classifier.fit(X_train, y_train)

    metrics = evaluate_classifier(classifier, test_data)
    print("\n--- Pattern Classification Test Metrics ---")
    print(f"Overall Accuracy: {metrics['overall_accuracy'] * 100:.1f}%")
    print(f"Macro-Recall: {metrics['macro_recall'] * 100:.1f}%")
    print(f"Brier Multiclass Score: {metrics['brier_multiclass_score']}")
    print(f"Per-Class Recall: {metrics['per_class_recall']}")

    generated_at = datetime.now(UTC).isoformat()
    model_payload = {
        "generated_at_utc": generated_at,
        "pattern_classes": PATTERN_CLASSES,
        "feature_names": PATTERN_FEATURE_NAMES,
        "model": classifier.as_dict(),
        "metrics": metrics,
    }

    model_path = args.output_dir / "pattern_model.json"
    model_path.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[OK] Pattern classification model saved to {model_path}")


if __name__ == "__main__":
    main()
