"""Keep the final acceptance manifest executable in regression tests."""

import unittest

from scripts.run_acceptance import run_acceptance


class FinalAcceptanceTests(unittest.TestCase):
    def test_all_manifest_cases_pass(self) -> None:
        results = run_acceptance()
        self.assertGreaterEqual(len(results), 10)
        failures = [item for item in results if not item["passed"]]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
