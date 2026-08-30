from __future__ import annotations

import json
import unittest

from scripts.build_semantic_rescue_activation_suite import (
    _jsonl_bytes,
    select_unique_target,
    validate_activation_suite,
)


def _row(sample_id: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "scenario_type": "buying",
        "user_profile": {},
        "ground_truth": {"parent_asin": target},
        "intent_card": {
            "target_category": "Shoes",
            "hard_constraints": ["waterproof"],
            "soft_preferences": [],
        },
        "behavior": {"scenario_type": "buying"},
    }


class SemanticRescueActivationSuiteTests(unittest.TestCase):
    def test_target_selection_is_deterministic_and_excludes_used_targets(self) -> None:
        keyword = {
            "support_ids": ("PUBLIC", "USED", "A", "B", "A"),
            "public_targets": frozenset({"PUBLIC"}),
            "selected_targets": frozenset({"USED"}),
            "category": "Shoes",
            "constraint": "waterproof",
        }

        first = select_unique_target(**keyword)
        second = select_unique_target(**keyword)

        self.assertEqual(first, second)
        self.assertIn(first, {"A", "B"})

    def test_suite_validation_rejects_public_or_duplicate_targets(self) -> None:
        valid = [_row("activation_a", "TARGET_A")]
        validate_activation_suite(valid, public_targets=frozenset())

        with self.assertRaisesRegex(ValueError, "not target-disjoint"):
            validate_activation_suite(
                valid,
                public_targets=frozenset({"TARGET_A"}),
            )
        with self.assertRaisesRegex(ValueError, "not target-disjoint"):
            validate_activation_suite(
                [
                    _row("activation_a", "TARGET_A"),
                    _row("activation_b", "TARGET_A"),
                ],
                public_targets=frozenset(),
            )

    def test_jsonl_serialization_is_canonical_and_newline_terminated(self) -> None:
        payload = _jsonl_bytes([_row("activation_a", "TARGET_A")])

        self.assertTrue(payload.endswith(b"\n"))
        decoded = json.loads(payload.decode("utf-8"))
        self.assertEqual(decoded["sample_id"], "activation_a")


if __name__ == "__main__":
    unittest.main()
