from __future__ import annotations

import unittest

from conversational_search.intent import (
    IntentState,
    Requirement,
    RequirementImportance,
    requirement_semantic_payload,
)
from conversational_search.profiles import ProductTheme, ProfilePrior
from conversational_search.protocol import DisclosureCard, ProductProtocolEvidence
from conversational_search.ranking import CandidateDocument
from conversational_search.requirement_satisfaction import (
    ImportanceAwareStatus,
    RequirementSatisfaction,
    rank_importance_aware_satisfaction,
)


def _evidence(
    parent_asin: str,
    *,
    hard: tuple[str, ...] = (),
    soft: tuple[str, ...] = (),
    price: str | None = None,
    category: str = "Shoes",
    text: str = "",
) -> tuple[ProductProtocolEvidence, CandidateDocument]:
    card = DisclosureCard(parent_asin, hard, soft)
    protocol = ProductProtocolEvidence(
        parent_asin,
        category,
        card,
        text=" ".join((parent_asin, *hard, *soft)),
        price=price,
    )
    document = CandidateDocument(
        parent_asin,
        text or "\n".join((f"Title: {parent_asin}", *hard, *soft)),
    )
    return protocol, document


def _rank(
    state: IntentState,
    rows: tuple[tuple[ProductProtocolEvidence, CandidateDocument], ...],
    *,
    base: tuple[str, ...] | None = None,
    bm25: tuple[str, ...] | None = None,
    profile: ProfilePrior = ProfilePrior(),
):
    ids = tuple(row[0].parent_asin for row in rows)
    return rank_importance_aware_satisfaction(
        ids if base is None else base,
        ids if bm25 is None else bm25,
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        state,
        profile_prior=profile,
    )


class RequirementImportanceTests(unittest.TestCase):
    def test_default_importance_is_independent_of_strength(self) -> None:
        examples = (
            ("initial_explicit", RequirementImportance.MUST),
            ("initial_tentative", RequirementImportance.PREFER),
            ("answer", RequirementImportance.SHOULD),
            ("override", RequirementImportance.MUST),
            ("free_text", RequirementImportance.PREFER),
        )
        for source, expected in examples:
            with self.subTest(source=source):
                requirement = Requirement("waterproof", source, 1, "feature")
                self.assertIs(requirement.importance, expected)

    def test_explicit_cues_and_maximum_budget_raise_importance(self) -> None:
        self.assertIs(
            Requirement("must be waterproof", "free_text", 1).importance,
            RequirementImportance.MUST,
        )
        self.assertIs(
            Requirement("strongly prefer blue", "free_text", 1).importance,
            RequirementImportance.SHOULD,
        )
        self.assertIs(
            Requirement("under $120", "answer", 2, "budget").importance,
            RequirementImportance.MUST,
        )
        forced = Requirement(
            "under $120",
            "answer",
            2,
            "budget",
            importance=RequirementImportance.PREFER,
        )
        self.assertIs(forced.importance, RequirementImportance.MUST)

    def test_cues_are_anchored_and_semantic_payload_is_preserved(self) -> None:
        examples = (
            ("must be waterproof", RequirementImportance.MUST, "waterproof"),
            (
                "waterproof is important",
                RequirementImportance.SHOULD,
                "waterproof",
            ),
            (
                "blue would be nice",
                RequirementImportance.PREFER,
                "blue",
            ),
            ("ideally lightweight", RequirementImportance.PREFER, "lightweight"),
        )
        for value, importance, payload in examples:
            with self.subTest(value=value):
                requirement = Requirement(value, "free_text", 1)
                self.assertIs(requirement.importance, importance)
                self.assertEqual(requirement_semantic_payload(value), payload)

    def test_negated_cues_do_not_escalate_importance(self) -> None:
        for value in (
            "waterproof is not required",
            "no need for leather",
            "blue is not important",
            "not a must-have: insulated",
        ):
            with self.subTest(value=value):
                self.assertIs(
                    Requirement(value, "free_text", 1).importance,
                    RequirementImportance.PREFER,
                )
                self.assertEqual(requirement_semantic_payload(value), value)


class ImportanceAwareSatisfactionTests(unittest.TestCase):
    def test_must_budget_and_feature_dominate_preferred_color(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_explicit", 1, "feature"),
                Requirement("under $120", "answer", 2, "budget"),
                Requirement("blue", "initial_tentative", 1, "color"),
            ),
        )
        candidate_a = _evidence(
            "A",
            hard=("waterproof", "hiking"),
            soft=("color: brown",),
            price="110",
            text="Title: hiking shoe\nFeatures: waterproof\nDetails: color brown",
        )
        candidate_b = _evidence(
            "B",
            hard=("water-resistant", "hiking"),
            soft=("color: blue",),
            price="140",
            text="Title: hiking shoe\nFeatures: water-resistant\nDetails: color blue",
        )

        result = _rank(state, (candidate_b, candidate_a))

        self.assertEqual(result.ranked_ids[:2], ("A", "B"))
        self.assertEqual(result.trace.must_violation_candidate_count, 1)
        self.assertIn("A", result.fully_satisfied_best_ids)

    def test_many_preferences_cannot_compensate_for_weaker_must(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_explicit", 1, "feature"),
                Requirement("blue", "initial_tentative", 1, "color"),
                Requirement("lightweight", "initial_tentative", 1, "feature"),
                Requirement("cushioned", "initial_tentative", 1, "feature"),
            ),
        )
        must_only = _evidence(
            "MUST",
            hard=("waterproof",),
            text="Title: waterproof hiking shoe",
        )
        preferences = _evidence(
            "PREFS",
            hard=("water-resistant",),
            soft=("color: blue", "lightweight cushioned"),
            text=(
                "Title: water-resistant shoe\nFeatures: lightweight cushioned\n"
                "Details: color blue"
            ),
        )

        result = _rank(state, (preferences, must_only))

        self.assertEqual(result.ranked_ids[0], "MUST")

    def test_should_requirement_dominates_many_prefer_matches(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("insulated", "answer", 1, "feature"),
                Requirement("blue", "initial_tentative", 1, "color"),
                Requirement("lightweight", "initial_tentative", 1, "feature"),
                Requirement("cushioned", "initial_tentative", 1, "feature"),
            ),
        )
        should = _evidence("SHOULD", hard=("insulated",))
        preferences = _evidence(
            "PREFERENCES",
            soft=("blue", "lightweight cushioned"),
        )

        self.assertEqual(
            _rank(state, (preferences, should)).ranked_ids[0],
            "SHOULD",
        )

    def test_same_level_violation_cannot_be_averaged_away(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "answer", 1, "feature"),
                Requirement("insulated", "answer", 1, "feature"),
            ),
        )
        full_and_violated = _evidence(
            "MIXED",
            text="Features: waterproof but not insulated",
        )
        partial_and_unknown = _evidence(
            "SAFE",
            text="Features: water-resistant upper",
        )

        result = _rank(state, (full_and_violated, partial_and_unknown))

        self.assertEqual(result.ranked_ids[0], "SAFE")

    def test_unknown_must_is_retained_for_every_candidate(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("quantum weave", "initial_explicit", 1, "feature"),
            ),
        )
        rows = (_evidence("A", hard=("mesh",)), _evidence("B", hard=("leather",)))

        result = _rank(state, rows, bm25=("B", "A"))

        requirement_index = next(
            index
            for index, requirement in enumerate(result.requirements)
            if requirement.normalized_value == "quantum weave"
        )
        self.assertTrue(
            all(
                assessment.satisfactions[requirement_index]
                is RequirementSatisfaction.UNKNOWN
                for assessment in result.assessments
            )
        )
        self.assertEqual(result.trace.must_unknown_candidate_count, 2)
        self.assertEqual(result.trace.all_must_full_candidate_count, 0)
        self.assertEqual(result.ranked_ids[0], "B")

    def test_typed_maximum_budget_distinguishes_full_unknown_and_violated(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(Requirement("under $120", "answer", 2, "budget"),),
        )
        rows = (
            _evidence("OVER", price="140"),
            _evidence("UNKNOWN", price=None),
            _evidence("EQUAL", price="120"),
            _evidence("UNDER", price="119.99"),
        )

        result = _rank(state, rows)

        self.assertEqual(result.ranked_ids, ("UNDER", "UNKNOWN", "OVER", "EQUAL"))
        budget_index = next(
            index
            for index, requirement in enumerate(result.requirements)
            if requirement.attribute == "budget"
        )
        status_by_id = {
            assessment.parent_asin: assessment.satisfactions[budget_index]
            for assessment in result.assessments
        }
        self.assertIs(status_by_id["UNDER"], RequirementSatisfaction.FULL)
        self.assertIs(status_by_id["UNKNOWN"], RequirementSatisfaction.UNKNOWN)
        self.assertIs(status_by_id["EQUAL"], RequirementSatisfaction.VIOLATED)
        self.assertIs(status_by_id["OVER"], RequirementSatisfaction.VIOLATED)

    def test_inclusive_budget_accepts_equal_price(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(Requirement("at most $120", "answer", 2, "budget"),),
        )
        rows = (_evidence("OVER", price="121"), _evidence("EQUAL", price="120"))
        self.assertEqual(_rank(state, rows).ranked_ids[0], "EQUAL")

    def test_budget_operators_ranges_and_minimums_remain_typed(self) -> None:
        cases = (
            ("<= $120", "120", "121"),
            ("less than $120", "119.99", "120"),
            ("at least $100", "100", "99.99"),
            ("over $100", "100.01", "100"),
            ("between $100 and $120", "110", "121"),
            ("from $100 to $120", "120", "99"),
        )
        for requirement, accepted, rejected in cases:
            with self.subTest(requirement=requirement):
                state = IntentState(
                    category="Shoes",
                    requirements=(
                        Requirement(requirement, "answer", 1, "budget"),
                    ),
                )
                result = _rank(
                    state,
                    (
                        _evidence("REJECTED", price=rejected),
                        _evidence("ACCEPTED", price=accepted),
                    ),
                )
                self.assertEqual(result.ranked_ids[0], "ACCEPTED")

    def test_water_resistant_is_partial_for_waterproof(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_explicit", 1, "feature"),
            ),
        )
        rows = (
            _evidence(
                "PARTIAL",
                hard=("water-resistant",),
                text="Features: water-resistant upper",
            ),
            _evidence("UNKNOWN", hard=("mesh",), text="Features: mesh upper"),
        )

        result = _rank(state, rows)
        requirement_index = next(
            index
            for index, requirement in enumerate(result.requirements)
            if requirement.normalized_value == "waterproof"
        )
        status_by_id = {
            assessment.parent_asin: assessment.satisfactions[requirement_index]
            for assessment in result.assessments
        }
        self.assertIs(status_by_id["PARTIAL"], RequirementSatisfaction.PARTIAL)
        self.assertIs(status_by_id["UNKNOWN"], RequirementSatisfaction.UNKNOWN)
        self.assertEqual(result.ranked_ids[0], "PARTIAL")

    def test_protection_partial_relation_is_not_water_specific(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("fireproof", "answer", 1, "feature"),
            ),
        )
        rows = (
            _evidence("UNKNOWN", text="Features: mesh"),
            _evidence("PARTIAL", text="Features: fire-resistant shell"),
        )
        self.assertEqual(_rank(state, rows).ranked_ids[0], "PARTIAL")

    def test_negated_candidate_phrase_is_a_violation_not_a_full_match(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_explicit", 1, "feature"),
            ),
        )
        result = _rank(
            state,
            (
                _evidence("NEGATED", text="Features: not waterproof"),
                _evidence("UNKNOWN", text="Features: mesh"),
            ),
        )
        requirement_index = next(
            index
            for index, requirement in enumerate(result.requirements)
            if requirement.normalized_value == "waterproof"
        )
        status_by_id = {
            item.parent_asin: item.satisfactions[requirement_index]
            for item in result.assessments
        }
        self.assertIs(
            status_by_id["NEGATED"],
            RequirementSatisfaction.VIOLATED,
        )
        self.assertEqual(result.ranked_ids[0], "UNKNOWN")

    def test_negated_excluded_phrase_is_positive_safety_evidence(self) -> None:
        state = IntentState(category="Shoes", excluded=("leather",))
        result = _rank(
            state,
            (
                _evidence("LEATHER", text="Details: leather upper"),
                _evidence("SAFE", text="Details: no leather materials"),
            ),
        )
        self.assertEqual(result.ranked_ids[0], "SAFE")

    def test_explicit_exclusion_dominates_positive_preferences(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("blue", "initial_tentative", 1, "color"),
                Requirement("lightweight", "initial_tentative", 1, "feature"),
            ),
            excluded=("leather",),
        )
        violating = _evidence(
            "LEATHER",
            hard=("leather", "color: blue"),
            soft=("lightweight",),
            text="Details: leather color blue lightweight",
        )
        safe = _evidence("MESH", hard=("mesh",), text="Details: mesh upper")

        result = _rank(state, (violating, safe))

        self.assertEqual(result.ranked_ids[0], "MESH")
        self.assertEqual(result.trace.must_violation_candidate_count, 1)

    def test_duplicate_requirement_keeps_highest_importance_once(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_tentative", 1, "feature"),
                Requirement("waterproof", "initial_explicit", 2, "feature"),
            ),
        )
        result = _rank(state, (_evidence("A", hard=("waterproof",)),))
        matching = tuple(
            requirement
            for requirement in result.requirements
            if requirement.normalized_value == "waterproof"
        )
        self.assertEqual(len(matching), 1)
        self.assertIs(matching[0].importance, RequirementImportance.MUST)

    def test_explicit_importance_is_not_reinferred_by_the_reranker(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement(
                    "must be waterproof",
                    "initial_explicit",
                    1,
                    "feature",
                    importance=RequirementImportance.PREFER,
                ),
            ),
        )
        result = _rank(state, (_evidence("A", hard=("waterproof",)),))
        requirement = next(
            item
            for item in result.requirements
            if item.normalized_value == "waterproof"
        )
        self.assertIs(requirement.importance, RequirementImportance.PREFER)

    def test_profile_prior_is_a_prefer_tier(self) -> None:
        state = IntentState(category="Shoes")
        rows = (
            _evidence("PLAIN", text="Title: plain shoe"),
            _evidence("COMFORT", text="Title: cushioned comfortable shoe"),
        )

        result = _rank(
            state,
            rows,
            profile=ProfilePrior(ProductTheme.COMFORT),
        )

        self.assertEqual(result.ranked_ids[0], "COMFORT")
        self.assertEqual(result.trace.profile_preference_count, 1)

    def test_bm25_breaks_an_exact_satisfaction_tie(self) -> None:
        state = IntentState(category="Shoes")
        rows = (_evidence("A"), _evidence("B"))
        result = _rank(state, rows, bm25=("B", "A"))
        self.assertEqual(result.ranked_ids, ("B", "A"))

    def test_no_requirements_preserves_base_order(self) -> None:
        state = IntentState()
        rows = (_evidence("A"), _evidence("B"))
        result = _rank(state, rows, bm25=("B", "A"))
        self.assertIs(result.status, ImportanceAwareStatus.NO_REQUIREMENTS)
        self.assertEqual(result.ranked_ids, ("A", "B"))

    def test_misaligned_protocol_evidence_is_rejected(self) -> None:
        state = IntentState(category="Shoes")
        a = _evidence("A")
        b = _evidence("B")
        with self.assertRaisesRegex(ValueError, "positionally misaligned"):
            rank_importance_aware_satisfaction(
                ("A", "B"),
                ("A", "B"),
                (b[0], a[0]),
                (a[1], b[1]),
                state,
            )

    def test_over_bound_requirement_fails_closed_instead_of_truncating(self) -> None:
        state = IntentState(
            requirements=(
                Requirement("x" * 1_025, "free_text", 1, "feature"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            _rank(state, (_evidence("A"),))

    def test_ranking_is_deterministic_and_a_complete_permutation(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(Requirement("waterproof", "answer", 1, "feature"),),
        )
        rows = (
            _evidence("A", hard=("mesh",)),
            _evidence("B", hard=("water-resistant",)),
            _evidence("C", hard=("waterproof",)),
        )

        first = _rank(state, rows)
        second = _rank(state, rows)

        self.assertEqual(first, second)
        self.assertEqual(set(first.ranked_ids), {"A", "B", "C"})
        self.assertEqual(len(first.ranked_ids), 3)


if __name__ == "__main__":
    unittest.main()
