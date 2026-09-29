"""Contract tests for transparent, non-operational replay outputs."""

from __future__ import annotations

from datetime import UTC, datetime
import unittest
from unittest.mock import patch

from app.bulletin_generator import generate_research_brief
from app.dvorak_engine import classify_pattern
from app.landfall_impact_engine import evaluate_landfall_and_impact
from app.ri_engine import evaluate_rapid_intensification
from app.historical_replay import _source_features


class ResearchScreenContractTests(unittest.TestCase):
    def test_dvorak_result_identifies_best_track_proxy(self) -> None:
        result = classify_pattern(65.0, 980.0, 15.0, 85.0)
        self.assertEqual(result.assessment_mode, "best_track_intensity_proxy")
        self.assertTrue(any("satellite" in item.lower() for item in result.limitations))

    def test_ri_is_a_bounded_screening_score_not_probability(self) -> None:
        result = evaluate_rapid_intensification(60.0, 54.0, 14.0, 82.0, season_month=5)
        self.assertGreaterEqual(result.ri_score, 0.0)
        self.assertLessEqual(result.ri_score, 1.0)
        self.assertIn("SCREENING", result.status_label)
        self.assertIn("seasonal_proxies", result.input_mode)

    def test_era5_sst_is_used_only_when_an_extracted_row_is_available(self) -> None:
        result = evaluate_rapid_intensification(
            60.0, 54.0, 14.0, 82.0, season_month=5, sea_surface_temperature_c=29.4
        )
        self.assertIn("era5_sst", result.input_mode)

    def test_extracted_source_values_are_keyed_to_the_exact_replay_time(self) -> None:
        timestamp = datetime(2025, 5, 1, tzinfo=UTC)
        key = ("test-storm", "2025-05-01T00:00:00Z")
        with (
            patch("app.historical_replay.load_era5_features", return_value={key: {"era5_missing": "0", "era5_sst_k": "301.15"}}),
            patch("app.historical_replay.load_imerg_features", return_value={key: {"imerg_missing": "0", "imerg_center_precipitation_cal_mm_hr": "7.25"}}),
        ):
            features = _source_features("test-storm", timestamp)
        self.assertEqual(features["era5_sst_celsius"], 28.0)
        self.assertEqual(features["imerg_center_precipitation_mm_hr"], 7.25)
        self.assertTrue(features["era5_available"])
        self.assertTrue(features["imerg_available"])

    def test_landfall_output_is_static_proximity_screening(self) -> None:
        forecast = [{"hours": 24, "lat": 20.4, "lon": 86.3, "wind_kmph": 110}]
        result, districts = evaluate_landfall_and_impact(
            forecast, 18.0, 86.0, 60.0, datetime(2025, 5, 1, tzinfo=UTC)
        )
        self.assertEqual(result.assessment_mode, "static_district_proximity_screening")
        self.assertTrue(any("not a landfall detector" in item.lower() for item in result.limitations))
        self.assertTrue(all(item["alert_tier"].startswith("RESEARCH SCREEN") for item in districts))

    def test_brief_cannot_be_mistaken_for_an_agency_advisory(self) -> None:
        text = generate_research_brief({
            "name": "Cyclone Test", "status": "Cyclonic Storm", "last_updated": "2025-05-01T00:00:00Z",
            "current": {"lat": 14.0, "lon": 82.0, "wind_kmph": 90, "pressure_hpa": 985},
            "dvorak": {"t_number": 3.5, "ci_number": 3.5, "pattern_type": "Proxy"},
            "rapid_intensification": {"ri_score": 0.6, "status_label": "ELEVATED RI SCREENING"},
        })
        self.assertIn("HISTORICAL RESEARCH REPLAY BRIEF", text)
        self.assertIn("NOT AN IMD/RSMC ADVISORY", text)
        self.assertIn("NOT a probability", text)


if __name__ == "__main__":
    unittest.main()
