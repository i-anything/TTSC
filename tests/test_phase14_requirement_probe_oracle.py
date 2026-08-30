from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from scripts import verify_phase14_requirement_probe_oracle as oracle


class Phase14RequirementProbeOracleTest(unittest.TestCase):
    def test_fixed_oracle_prints_only_count_and_frozen_digest(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = oracle.main()

        self.assertIsNone(result)
        self.assertGreaterEqual(oracle.RANDOM_CASES, 30_000)
        self.assertEqual(output.getvalue().count("\n"), 1)
        self.assertEqual(
            json.loads(output.getvalue()),
            {
                "cases": oracle.RANDOM_CASES,
                "digest": oracle.EXPECTED_SHA256,
            },
        )
        self.assertEqual(
            output.getvalue(),
            json.dumps(
                {
                    "cases": oracle.RANDOM_CASES,
                    "digest": oracle.EXPECTED_SHA256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )


if __name__ == "__main__":
    unittest.main()
