from __future__ import annotations

import unittest

from conversational_search.field_semantic import (
    FieldSemanticAssessment,
    FieldSemanticStatus,
    rank_field_semantic,
)


def _assessment(
    parent_asin: str,
    *,
    exclusion: float | None = None,
    minimum: float | None = None,
    mean: float | None = None,
    category: float | None = None,
) -> FieldSemanticAssessment:
    return FieldSemanticAssessment(
        parent_asin,
        exclusion,
        minimum,
        mean,
        category,
    )


class FieldSemanticRankingTest(unittest.TestCase):
    def test_exclusion_then_weakest_requirement_define_order(self) -> None:
        result = rank_field_semantic(
            ("BASE", "EXCLUDED", "BALANCED"),
            (
                _assessment(
                    "BASE",
                    exclusion=0.10,
                    minimum=0.40,
                    mean=0.80,
                    category=0.90,
                ),
                _assessment(
                    "EXCLUDED",
                    exclusion=0.80,
                    minimum=0.90,
                    mean=0.90,
                    category=0.90,
                ),
                _assessment(
                    "BALANCED",
                    exclusion=0.10,
                    minimum=0.70,
                    mean=0.75,
                    category=0.80,
                ),
            ),
        )

        self.assertIs(result.status, FieldSemanticStatus.REORDERED)
        self.assertEqual(result.ranked_ids, ("BALANCED", "BASE", "EXCLUDED"))

    def test_no_signal_preserves_input_order(self) -> None:
        result = rank_field_semantic(
            ("ONE", "TWO"),
            (_assessment("ONE"), _assessment("TWO")),
        )

        self.assertIs(result.status, FieldSemanticStatus.NO_SIGNAL)
        self.assertEqual(result.ranked_ids, ("ONE", "TWO"))

    def test_partial_candidate_coverage_and_misalignment_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate-complete"):
            rank_field_semantic(
                ("ONE", "TWO"),
                (
                    _assessment("ONE", minimum=0.5, mean=0.5),
                    _assessment("TWO"),
                ),
            )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            rank_field_semantic(
                ("ONE", "TWO"),
                (_assessment("TWO"), _assessment("ONE")),
            )


if __name__ == "__main__":
    unittest.main()
