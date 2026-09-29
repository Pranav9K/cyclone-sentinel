"""Regression checks for every listed historical cyclone replay."""

from __future__ import annotations

import unittest

from app.dvorak_engine import wind_to_t_number
from app.historical_replay import list_replays, replay


class ReplayCatalogTests(unittest.TestCase):
    def test_duplicate_25_knot_threshold_is_safe(self) -> None:
        self.assertEqual(wind_to_t_number(25.0), 1.5)
        self.assertGreaterEqual(wind_to_t_number(20.0), 1.0)

    def test_every_listed_storm_builds_a_complete_replay_payload(self) -> None:
        for item in list_replays():
            with self.subTest(storm_id=item["id"]):
                payload = replay(str(item["id"]))
                self.assertEqual(payload["id"], item["id"])
                self.assertTrue(payload["observed_track"])
                self.assertIn("dvorak", payload)


if __name__ == "__main__":
    unittest.main()
