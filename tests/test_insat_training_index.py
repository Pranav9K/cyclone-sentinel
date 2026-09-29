"""Tests for conservative timestamp matching of real INSAT files to labels."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from index_insat_imagery import capture_time_from_name
from prepare_insat_training_index import TrackObservation, intensity_label, nearest_observation


class InsatTrainingIndexTests(unittest.TestCase):
    def test_mosdac_filename_timestamp_is_parsed_without_guessing(self) -> None:
        self.assertEqual(capture_time_from_name("3DIMG_23AUG2022_0800_L2P_VSW.gif"), "2022-08-23T08:00:00Z")
        self.assertIsNone(capture_time_from_name("unlabelled-image.png"))

    def test_only_a_time_match_inside_tolerance_is_paired(self) -> None:
        timestamp = datetime(2023, 5, 15, 12, tzinfo=UTC)
        observation = TrackObservation("storm", "TEST", "2023", timestamp, "14", "85", "50", "980")
        paired, offset = nearest_observation(timestamp + timedelta(minutes=60), [observation], 90)
        self.assertEqual(paired, observation)
        self.assertEqual(offset, 60)
        unpaired, offset = nearest_observation(timestamp + timedelta(minutes=91), [observation], 90)
        self.assertIsNone(unpaired)
        self.assertEqual(offset, 91)

    def test_target_intensity_label_comes_from_best_track_wind(self) -> None:
        self.assertEqual(intensity_label("64"), "Very Severe Cyclonic Storm")
        self.assertEqual(intensity_label(""), "Intensity unavailable")


if __name__ == "__main__":
    unittest.main()
