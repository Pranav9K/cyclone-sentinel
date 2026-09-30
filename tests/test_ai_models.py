"""Unit and integration tests for multi-horizon models, pattern classifier, and calibrated RI classifier."""

from __future__ import annotations

import unittest

from app.dvorak_engine import classify_pattern
from app.historical_replay import prediction_payload, replay
from app.main import model_benchmarks, model_pattern_info, model_ri_info
from app.ri_engine import evaluate_rapid_intensification


class AIModelsIntegrationTests(unittest.TestCase):
    def test_multihorizon_forecast_and_benchmarks(self) -> None:
        res = replay("current")
        self.assertIn("forecast_track", res)
        forecast = res["forecast_track"]
        self.assertEqual(len(forecast), 3)

        horizons = [pt["hours"] for pt in forecast]
        self.assertEqual(horizons, [24, 48, 72])

        # Verify radii are monotonically increasing with horizon uncertainty
        radii = [pt["radius_km"] for pt in forecast]
        self.assertGreater(radii[1], radii[0])
        self.assertGreater(radii[2], radii[1])

        # Check IMD benchmark comparison in model_metrics
        metrics = res["model_metrics"]
        self.assertIn("multihorizon", metrics)
        self.assertIn("imd_benchmarks", metrics)
        self.assertIn("24", metrics["imd_benchmarks"])

    def test_calibrated_ri_classifier(self) -> None:
        result = evaluate_rapid_intensification(
            current_wind_knots=75.0,
            prior_wind_knots=65.0,
            latitude=14.5,
            longitude=85.0,
            season_month=5,
            sea_surface_temperature_c=30.2,
        )
        self.assertGreaterEqual(result.ri_score, 0.0)
        self.assertLessEqual(result.ri_score, 1.0)
        self.assertIsNotNone(result.roc_auc)
        self.assertGreaterEqual(result.roc_auc, 0.70)
        self.assertIsNotNone(result.brier_score)
        self.assertIn("SCREENING", result.status_label)
        self.assertTrue(len(result.favorable_factors) > 0)

    def test_pattern_classification_5_classes(self) -> None:
        # High-intensity cyclone with low shear should yield Eye Pattern
        result_eye = classify_pattern(
            wind_knots=105.0,
            pressure_hpa=940.0,
            latitude=15.0,
            longitude=86.0,
            vertical_wind_shear_knots=10.0,
        )
        self.assertEqual(result_eye.pattern_type, "Eye Pattern")
        self.assertGreater(result_eye.confidence_percent, 50.0)
        self.assertIn("Eye Pattern", result_eye.pattern_probabilities)
        self.assertEqual(len(result_eye.pattern_probabilities), 5)
        self.assertAlmostEqual(sum(result_eye.pattern_probabilities.values()), 1.0, places=2)

        # Low-intensity system with high shear should yield Shear Pattern
        result_shear = classify_pattern(
            wind_knots=30.0,
            pressure_hpa=1002.0,
            latitude=12.0,
            longitude=88.0,
            vertical_wind_shear_knots=25.0,
        )
        self.assertEqual(result_shear.pattern_type, "Shear Pattern")

    def test_prediction_payload_has_pattern_and_benchmarks(self) -> None:
        pred = prediction_payload("current")
        self.assertIn("pattern_classification", pred)
        self.assertIn("predicted_pattern", pred["pattern_classification"])
        self.assertIn("probabilities", pred["pattern_classification"])
        self.assertIn("benchmarks_vs_imd", pred)

    def test_model_api_functions(self) -> None:
        data_bench = model_benchmarks()
        self.assertIn("benchmarks", data_bench)
        self.assertIn("imd_reference", data_bench)

        data_pat = model_pattern_info()
        self.assertIn("pattern_classes", data_pat)
        self.assertEqual(len(data_pat["pattern_classes"]), 5)

        data_ri = model_ri_info()
        self.assertIn("model", data_ri)
        self.assertIn("metrics", data_ri)
        self.assertGreater(data_ri["metrics"]["roc_auc"], 0.70)


if __name__ == "__main__":
    unittest.main()
