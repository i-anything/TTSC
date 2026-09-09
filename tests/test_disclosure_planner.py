from __future__ import annotations

import unittest

from conversational_search.disclosure_planner import (
    plan_disclosure_question,
    plan_ranked_enumeration_width,
)
from conversational_search.protocol import (
    DisclosureCard, ObservedProtocolEvent, ProductProtocolEvidence, ProtocolEventKind,
)
from conversational_search.protocol_index import resolve_protocol_transcript


def resolution(cards, disclosed=False):
    return resolve_protocol_transcript(
        tuple(
            ProductProtocolEvidence(f"P{i}", "Shoes", card)
            for i, card in enumerate(cards)
        ),
        (ObservedProtocolEvent(
            1,
            ProtocolEventKind.INITIAL_EXPLICIT if disclosed else ProtocolEventKind.INITIAL_BROWSING,
            values=("cotton",) if disclosed else (),
        ),),
        observed_turn_count=1,
    )


class DisclosurePlannerTests(unittest.TestCase):
    def test_unique_feature_beats_two_shared_initial_disclosures(self):
        support = resolution([
            DisclosureCard(f"shoe {i}", ("cotton", "color: black"), (f"special property {i}",))
            for i in range(8)
        ])
        self.assertEqual(plan_disclosure_question(
            support.candidate_ids, support, current_turn=1, top_k=10,
        ), "feature")

    def test_initial_explicit_near_tie_prefers_broader_wildcard(self):
        support = resolution([
            DisclosureCard("one", ("cotton", "color: blue"), ("machine wash", "Imported")),
            DisclosureCard("two", ("cotton", "color: blue"), ("rubber sole", "special a")),
            DisclosureCard("three", ("cotton", "color: blue"), ("rubber sole", "lightweight")),
            DisclosureCard("four", ("cotton", "color: red"), ("special a", "special b")),
        ], disclosed=True)

        self.assertEqual(
            plan_disclosure_question(
                support.candidate_ids,
                support,
                current_turn=1,
                top_k=10,
            ),
            "feature",
        )
        self.assertEqual(
            plan_disclosure_question(
                support.candidate_ids,
                support,
                current_turn=1,
                top_k=10,
                prefer_wildcard_near_tie=True,
            ),
            "other",
        )

    def test_wildcard_preference_flag_requires_a_boolean(self):
        support = resolution([DisclosureCard("one", ("cotton",), ("warm",))])
        with self.assertRaises(TypeError):
            plan_disclosure_question(
                support.candidate_ids,
                support,
                current_turn=1,
                top_k=10,
                prefer_wildcard_near_tie=1,
            )

    def test_full_support_is_retained_outside_search_prefix(self):
        support = resolution([
            DisclosureCard(f"shoe {i}", ("cotton", "color: black"), (f"special property {i}",))
            for i in range(8)
        ])
        self.assertEqual(plan_disclosure_question(
            support.candidate_ids[:2], support, current_turn=1, top_k=10,
        ), "feature")

    def test_identical_serialized_replies_are_not_separate_outcomes(self):
        values = [
            ("black", ("alpha", "beta; gamma")),
            ("blue", ("alpha", "beta; gamma")),
            ("blue", ("alpha; beta", "gamma")),
            ("black", ("alpha", "beta; gamma")),
            ("black", ("alpha; beta", "delta")),
            ("black", ("alpha; beta", "gamma")),
            ("black", ("alpha; beta", "gamma")),
        ]
        support = resolution([
            DisclosureCard(f"shoe {i}", ("cotton", f"color: {color}"), soft)
            for i, (color, soft) in enumerate(values)
        ])
        # Splitting by value tuples incorrectly prefers feature; the shopper
        # emits the same "alpha; beta; gamma" string for both representations.
        self.assertEqual(plan_disclosure_question(
            support.candidate_ids, support, current_turn=1, top_k=10,
        ), "other")

    def test_rank_prior_preserves_rank_one_when_more_turns_are_available(self):
        self.assertEqual(plan_ranked_enumeration_width(10, current_turn=2, top_k=10), 1)

    def test_final_turn_uses_all_available_recommendation_slots(self):
        for support, top_k in ((3, 10), (1000, 10), (8, 2)):
            self.assertEqual(
                plan_ranked_enumeration_width(support, current_turn=10, top_k=top_k),
                min(support, top_k),
            )

    def test_width_planner_rejects_invalid_contract_values(self):
        for support, turn, top_k in ((0, 1, 10), (True, 1, 10), (5, 11, 10), (5, 1, 0)):
            with self.assertRaises(ValueError):
                plan_ranked_enumeration_width(support, current_turn=turn, top_k=top_k)

    def test_replanning_cannot_sacrifice_reachable_products_for_rank(self):
        for top_k in (1, 2, 3, 10):
            for start_turn in (1, 5, 9):
                capacity = top_k * (11 - start_turn)
                for count in (1, capacity // 2 + 1, capacity, capacity + 10):
                    remaining = count
                    shown = 0
                    for turn in range(start_turn, 11):
                        if not remaining:
                            break
                        width = plan_ranked_enumeration_width(
                            remaining, current_turn=turn, top_k=top_k,
                        )
                        remaining -= width
                        shown += width
                    self.assertEqual(shown, min(count, capacity))

    def test_no_new_question_at_deadline(self):
        support = resolution([
            DisclosureCard("one", ("cotton",), ("insulated",)),
            DisclosureCard("two", ("cotton",), ("lightweight",)),
        ])
        self.assertEqual(plan_disclosure_question(
            support.candidate_ids, support, current_turn=10, top_k=10,
        ), "other")


if __name__ == "__main__":
    unittest.main()
