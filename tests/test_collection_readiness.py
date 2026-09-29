"""Tests for safe, credential-value-free collection readiness reporting."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_collection_readiness import count_available_csv_rows, credential_presence, source_state


class CollectionReadinessTests(unittest.TestCase):
    def test_credential_presence_reports_booleans_without_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            (home / ".cdsapirc").write_text("url: hidden\nkey: hidden\n", encoding="utf-8")
            presence = credential_presence(home, {"EARTHDATA_USERNAME": "user", "EARTHDATA_PASSWORD": "secret"})
        self.assertEqual(presence, {
            "cds_configuration_present": True,
            "earthdata_environment_present": True,
            "earthdata_netrc_present": False,
        })

    def test_source_state_requires_prerequisites_before_collection(self) -> None:
        self.assertEqual(source_state(False, False), "setup_required")
        self.assertEqual(source_state(False, True), "ready_to_collect")
        self.assertEqual(source_state(True, False), "collection_started")

    def test_missing_placeholder_rows_do_not_count_as_feature_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.csv"
            path.write_text("imerg_missing,value\n1,\n0,4.2\n", encoding="utf-8")
            self.assertEqual(count_available_csv_rows(path, "imerg_missing"), 1)


if __name__ == "__main__":
    unittest.main()
