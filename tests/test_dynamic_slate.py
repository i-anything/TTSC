from __future__ import annotations

import unittest

from conversational_search.dynamic_slate import plan_answer_conditioned_slate
from conversational_search.exact_evidence import CandidateBelief


class DynamicSlatePlannerTests(unittest.TestCase):
    def test_defers_candidates_that_the_answer_will_rank_more_strongly(self) -> None:
        ranked = ("A", "B", "C", "D", "E")
        beliefs = tuple(
            CandidateBelief(parent_asin, weight)
            for parent_asin, weight in zip(
                ranked,
                (0.44, 0.22, 0.15, 0.11, 0.08),
            )
        )

        plan = plan_answer_conditioned_slate(
            ranked,
            beliefs,
            (("A", None), ("B", 1), ("C", None), ("D", 2), ("E", None)),
            current_turn=3,
            width=3,
            top_k=3,
        )

        self.assertTrue(plan.changed)
        self.assertEqual(plan.selected_ids, ("A", "C", "E"))
        self.assertGreater(plan.planned_value, plan.baseline_value)

    def test_keeps_exact_prefix_when_no_candidate_is_recoverable(self) -> None:
        ranked = ("A", "B", "C", "D")
        beliefs = tuple(
            CandidateBelief(parent_asin, weight)
            for parent_asin, weight in zip(ranked, (0.48, 0.24, 0.16, 0.12))
        )

        plan = plan_answer_conditioned_slate(
            ranked,
            beliefs,
            tuple((parent_asin, None) for parent_asin in ranked),
            current_turn=2,
            width=2,
            top_k=2,
        )

        self.assertFalse(plan.changed)
        self.assertEqual(plan.selected_ids, ("A", "B"))
        self.assertEqual(plan.planned_value, plan.baseline_value)

    def test_requires_a_complete_branch_rank_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "cover every candidate"):
            plan_answer_conditioned_slate(
                ("A", "B"),
                (CandidateBelief("A", 0.6), CandidateBelief("B", 0.4)),
                (("A", 1),),
                current_turn=1,
                width=1,
                top_k=2,
            )


if __name__ == "__main__":
    unittest.main()
