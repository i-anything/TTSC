from __future__ import annotations

import unittest

from conversational_search.intent import (
    IntentState,
    Requirement,
    RequirementImportance,
)
from conversational_search.slates import (
    REPEAT_TOP_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    SlateState,
    ranking_signature,
    select_slate,
)
from conversational_search.strategy import RouteWeights


class SlateSelectionTest(unittest.TestCase):
    def test_short_pool_stays_unique_and_zero_limit_does_not_mutate_state(self) -> None:
        signature = ("rank",)
        state = SlateState()

        first = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            state,
            signature,
            ("A", "B"),
            10,
        )
        zero = select_slate(
            STAGNATION_AWARE_SLATE_POLICY,
            first.state,
            signature,
            ("A", "B"),
            0,
        )

        self.assertEqual(first.selected_ids, ("A", "B"))
        self.assertEqual(zero.selected_ids, ())
        self.assertEqual(zero.state, first.state)

    def test_ranked_pool_rejects_invalid_or_duplicate_ids(self) -> None:
        for ranked_ids in (("A", "A"), (" A",), ("",)):
            with self.subTest(ranked_ids=ranked_ids):
                with self.assertRaises(ValueError):
                    select_slate(
                        STAGNATION_AWARE_SLATE_POLICY,
                        SlateState(),
                        ("rank",),
                        ranked_ids,
                        10,
                    )

    def test_repeat_policy_is_stateless(self) -> None:
        state = SlateState(signature=("old",), shown_ids=("A",))

        selection = select_slate(
            REPEAT_TOP_SLATE_POLICY,
            state,
            ("new",),
            ("B", "C"),
            1,
        )

        self.assertEqual(selection.selected_ids, ("B",))
        self.assertEqual(selection.state, state)


class RankingSignatureTest(unittest.TestCase):
    def test_dialogue_only_fields_do_not_change_the_signature(self) -> None:
        requirement = Requirement("leather", "answer", 2, "material")
        first = IntentState(
            category="Shoes",
            requirements=(requirement,),
            last_turn=2,
        )
        second = IntentState(
            category="Shoes",
            requirements=(requirement,),
            no_preference=frozenset({"color"}),
            asked_attributes=("feature", "material"),
            last_asked_attribute="material",
            last_turn=7,
        )
        arguments = (
            "Category: Shoes\nAttributes: Material: leather",
            "Shoes leather",
            RouteWeights(bm25=0.5, dense=0.5),
            "stage_a",
            ("A", "B"),
            10,
        )

        self.assertEqual(
            ranking_signature(first, *arguments),
            ranking_signature(second, *arguments),
        )

    def test_intent_version_changes_the_signature(self) -> None:
        arguments = (
            "Category: Shoes",
            "Shoes",
            RouteWeights(bm25=0.4, dense=0.6),
            "stage_a",
            ("A",),
            10,
        )

        self.assertNotEqual(
            ranking_signature(IntentState(), *arguments),
            ranking_signature(IntentState(intent_version=1), *arguments),
        )

    def test_requirement_strength_changes_the_signature(self) -> None:
        arguments = (
            "Category: Shoes\nSearch Clues: flexible",
            "Shoes flexible",
            RouteWeights(bm25=0.5, dense=0.5),
            "stage_a",
            ("A",),
            10,
        )
        hard = IntentState(
            category="Shoes",
            requirements=(
                Requirement("flexible", "answer", 2, "feature", "hard"),
            ),
        )
        soft = IntentState(
            category="Shoes",
            requirements=(
                Requirement("flexible", "answer", 2, "feature", "soft"),
            ),
        )

        self.assertNotEqual(
            ranking_signature(hard, *arguments),
            ranking_signature(soft, *arguments),
        )

    def test_requirement_importance_changes_the_signature(self) -> None:
        arguments = (
            "Category: Shoes\nAttributes: Feature: flexible",
            "Shoes flexible",
            RouteWeights(bm25=0.5, dense=0.5),
            "importance-aware-satisfaction-lexicographic-v1",
            ("A",),
            10,
        )
        should = IntentState(
            category="Shoes",
            requirements=(
                Requirement(
                    "flexible",
                    "answer",
                    2,
                    "feature",
                    importance=RequirementImportance.SHOULD,
                ),
            ),
        )
        prefer = IntentState(
            category="Shoes",
            requirements=(
                Requirement(
                    "flexible",
                    "answer",
                    2,
                    "feature",
                    importance=RequirementImportance.PREFER,
                ),
            ),
        )

        self.assertNotEqual(
            ranking_signature(should, *arguments),
            ranking_signature(prefer, *arguments),
        )

    def test_ranked_pool_order_and_membership_change_the_signature(self) -> None:
        common = (
            IntentState(category="Shoes"),
            "Category: Shoes",
            "Shoes",
            RouteWeights(bm25=0.4, dense=0.6),
            "stage_a",
        )
        original = ranking_signature(*common, ("A", "B"), 10)

        self.assertNotEqual(original, ranking_signature(*common, ("B", "A"), 10))
        self.assertNotEqual(original, ranking_signature(*common, ("A", "C"), 10))

    def test_result_count_changes_the_signature(self) -> None:
        common = (
            IntentState(category="Shoes"),
            "Category: Shoes",
            "Shoes",
            RouteWeights(bm25=0.4, dense=0.6),
            "stage_a",
            ("A", "B"),
        )

        self.assertNotEqual(
            ranking_signature(*common, 1),
            ranking_signature(*common, 2),
        )


if __name__ == "__main__":
    unittest.main()
