from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_orchestration_ablations import (
    SOURCE_PATHS,
    _freeze,
    _lookup_accounting_exact,
    _validate_output,
    _validate_publication_privacy,
)


class OrchestrationAblationGuardTests(unittest.TestCase):
    def test_freeze_is_order_independent_for_mapping_keys(self) -> None:
        first = {"b": [2, {"x": 3}], "a": 1}
        second = {"a": 1, "b": [2, {"x": 3}]}

        self.assertEqual(_freeze(first), _freeze(second))

    def test_lookup_accounting_requires_a_complete_partition(self) -> None:
        valid = {
            "lookups": 10,
            "hits": 4,
            "cold_misses": 2,
            "dependency_misses": 2,
            "backend_invalidations": 1,
            "fault_invalidations": 1,
        }
        self.assertTrue(_lookup_accounting_exact(valid))
        invalid = {**valid, "dependency_misses": 1}
        self.assertFalse(_lookup_accounting_exact(invalid))

    def test_publication_privacy_rejects_raw_fields_and_product_ids(self) -> None:
        _validate_publication_privacy({"safe": {"aggregate": 3}})
        for payload in (
            {"sessions": []},
            {"nested": {"user_message": "private"}},
            {"value": "B012345678"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    _validate_publication_privacy(payload)

    def test_output_guard_rejects_inputs_sources_and_publication_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        catalog = repository_root / "data" / "catalog.jsonl"
        dataset = repository_root / "data" / "public_set.jsonl"
        rejected = (
            catalog,
            dataset,
            repository_root / SOURCE_PATHS[0],
            repository_root / "docs" / "phase7_results.json",
            repository_root / "benchmarks" / "phase7.json",
        )
        for output in rejected:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    _validate_output(output, catalog, dataset)

    def test_output_guard_accepts_a_separate_temporary_result(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            _validate_output(
                Path(directory) / "phase7-result.json",
                repository_root / "data" / "catalog.jsonl",
                repository_root / "data" / "public_set.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
