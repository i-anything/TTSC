from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_policy_ablations import _write_json_atomic, run_policies


class PolicyAblationTest(unittest.TestCase):
    def test_policy_grid_rejects_empty_duplicate_and_unknown_names_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_policies("unused", "unused", [])
        with self.assertRaisesRegex(ValueError, "unique"):
            run_policies("unused", "unused", ["phase1", "phase1"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            run_policies("unused", "unused", ["not-a-policy"])

    def test_atomic_writer_replaces_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            _write_json_atomic(output, {"version": 1})
            _write_json_atomic(output, {"version": 2, "items": ["a", "b"]})

            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"version": 2, "items": ["a", "b"]},
            )
            self.assertEqual(list(output.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
