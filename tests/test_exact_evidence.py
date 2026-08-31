from __future__ import annotations

import dataclasses
import unittest
from itertools import permutations

from conversational_search.exact_evidence import (
    DENSE_CONFIDENT_BEST_TIER_POLICY,
    DENSE_COMPLETE_BEST_TIER_POLICY,
    DISABLED_SEMANTIC_TIEBREAK_POLICY,
    MAX_EXACT_EVIDENCE_CANDIDATES,
    ExactEvidenceStatus,
    SemanticTieBreakStatus,
    apply_dense_best_tier_tiebreak,
    rank_exact_evidence,
)
from conversational_search.intent import IntentState, Requirement
from conversational_search.protocol import (
    DisclosureCard,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
)


def _evidence(
    parent_asin: str,
    *,
    category: str = "Women Running Shoes",
    hard: tuple[str, ...] = ("cotton",),
    soft: tuple[str, ...] = (),
    text: str = "",
    price: str | None = None,
    popularity: int | None = None,
) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category=category,
        card=DisclosureCard("Synthetic fixture", hard, soft),
        text=text,
        price=price,
        popularity=popularity,
    )


class ExactEvidenceRankingTests(unittest.TestCase):
    def test_override_supersedes_stale_initial_protocol_evidence(self) -> None:
        candidates = (
            _evidence("OLD", hard=("rubber sole",)),
            _evidence("CURRENT", hard=("leather sole",)),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("leather sole", "override", 2, "feature"),
            ),
            intent_version=1,
            last_turn=2,
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("rubber sole",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.OVERRIDE,
                values=("leather sole",),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
            protocol_events=events,
        )

        self.assertEqual(result.ranked_ids[0], "CURRENT")
        self.assertEqual(result.consistent_support_ids, ("CURRENT",))

    def test_override_projects_retained_pre_override_reply_to_new_card(self) -> None:
        target = _evidence(
            "TARGET",
            hard=("cotton", "color: blue"),
            soft=("waterproof",),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(
                    "color: blue; waterproof",
                    "answer",
                    2,
                    "other",
                ),
                Requirement("cotton", "override", 3, "material"),
            ),
            intent_version=1,
            last_turn=3,
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("rubber",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                reply_payload="color: blue; waterproof",
            ),
            ObservedProtocolEvent(
                3,
                ProtocolEventKind.OVERRIDE,
                values=("cotton",),
            ),
        )

        result = rank_exact_evidence(
            ("TARGET",),
            (target,),
            state,
            protocol_events=events,
        )

        self.assertEqual(result.consistent_support_ids, ("TARGET",))
        self.assertEqual(
            result.disclosures[0].disclosed_values,
            ("cotton", "color: blue", "waterproof"),
        )

    def test_no_additional_reply_eliminates_cards_with_remaining_values(self) -> None:
        candidates = (
            _evidence(
                "HAS_COLOR",
                hard=("cotton",),
                soft=("color: blue",),
            ),
            _evidence(
                "NO_COLOR",
                hard=("cotton",),
                soft=("waterproof",),
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
            protocol_events=(
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("cotton",),
                ),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.NO_ADDITIONAL,
                    "color",
                ),
            ),
        )

        self.assertIs(result.status, ExactEvidenceStatus.APPLIED)
        self.assertEqual(result.consistent_support_ids, ("NO_COLOR",))
        self.assertEqual(result.ranked_ids[0], "NO_COLOR")
        self.assertEqual(result.trace.no_additional_observation_count, 1)

    def test_event_log_preserves_disclosures_removed_from_intent_slots(self) -> None:
        candidates = (
            _evidence(
                "EXHAUSTED",
                hard=("cotton", "color: blue"),
                soft=("waterproof",),
            ),
            _evidence(
                "HAS_MORE",
                hard=("cotton", "color: blue"),
                soft=("waterproof", "lightweight"),
            ),
        )
        reduced_state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
            no_preference=frozenset({"other"}),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("cotton",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                ("color: blue", "waterproof"),
            ),
            ObservedProtocolEvent(
                3,
                ProtocolEventKind.NO_ADDITIONAL,
                "other",
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            reduced_state,
            protocol_events=events,
        )

        self.assertEqual(result.consistent_support_ids, ("EXHAUSTED",))
        disclosures = {
            item.parent_asin: item.disclosed_values for item in result.disclosures
        }
        self.assertEqual(
            disclosures["EXHAUSTED"],
            ("cotton", "color: blue", "waterproof"),
        )

    def test_tentative_disclosure_override_and_negative_reply_replay_in_order(
        self,
    ) -> None:
        candidates = (
            _evidence(
                "TARGET",
                hard=("cotton", "color: blue"),
                soft=("waterproof",),
            ),
            _evidence(
                "WRONG_SOFT_VALUE",
                hard=("cotton", "color: blue"),
                soft=("breathable",),
            ),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_TENTATIVE,
                values=("waterproof",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                ("cotton", "color: blue"),
            ),
            ObservedProtocolEvent(
                3,
                ProtocolEventKind.OVERRIDE,
                values=("cotton",),
            ),
            ObservedProtocolEvent(
                4,
                ProtocolEventKind.DISCLOSURE,
                "feature",
                ("waterproof",),
            ),
            ObservedProtocolEvent(
                5,
                ProtocolEventKind.NO_ADDITIONAL,
                "feature",
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            IntentState(
                category="Women Running Shoes",
                requirements=(
                    Requirement("cotton", "override", 3, "material"),
                ),
            ),
            protocol_events=events,
        )

        self.assertEqual(result.consistent_support_ids, ("TARGET",))
        self.assertEqual(result.beliefs[0].parent_asin, "TARGET")
        disclosures = {
            item.parent_asin: item.disclosed_values for item in result.disclosures
        }
        self.assertEqual(
            disclosures["TARGET"],
            ("cotton", "color: blue", "waterproof"),
        )

    def test_opaque_semicolon_payload_keeps_all_serially_indistinguishable_cards(
        self,
    ) -> None:
        candidates = (
            _evidence(
                "INTERNAL_SEMICOLON",
                hard=("quick dry; machine washable",),
            ),
            _evidence(
                "TWO_CARD_VALUES",
                hard=("quick dry", "machine washable"),
            ),
            _evidence(
                "DIFFERENT_REPLY",
                hard=("quick dry", "machine wash only"),
            ),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_BROWSING,
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                reply_payload="quick dry; machine washable",
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            IntentState(category="Women Running Shoes"),
            protocol_events=events,
        )

        self.assertEqual(
            result.consistent_support_ids,
            ("INTERNAL_SEMICOLON", "TWO_CARD_VALUES"),
        )

    def test_event_log_prevents_state_splitting_of_one_semicolon_card_value(
        self,
    ) -> None:
        value = "waterproof; machine washable"
        candidate = _evidence("TARGET", hard=(value,))
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(value, "initial_explicit", 1, "feature"),
            ),
        )
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=(value,),
            ),
        )

        result = rank_exact_evidence(
            (candidate.parent_asin,),
            (candidate,),
            state,
            protocol_events=events,
        )

        self.assertIs(result.status, ExactEvidenceStatus.APPLIED)
        self.assertEqual(result.consistent_support_ids, ("TARGET",))
        self.assertEqual(result.trace.strong_disclosed_value_count, 1)

    def test_exact_semicolon_candidate_ranks_first_and_exposes_card_values(
        self,
    ) -> None:
        candidates = (
            _evidence(
                "HYBRID_FIRST",
                hard=("cotton", "color: red"),
                text="ordinary cotton shoe",
            ),
            _evidence(
                "EXACT",
                hard=("cotton", "color: blue"),
                text="lightweight cotton blue shoe",
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(
                    "cotton; color: blue",
                    "answer",
                    2,
                    "other",
                ),
                Requirement(
                    "lightweight",
                    "initial_tentative",
                    1,
                    "feature",
                ),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertIs(result.status, ExactEvidenceStatus.APPLIED)
        self.assertEqual(result.ranked_ids[0], "EXACT")
        self.assertEqual(result.consistent_support_ids, ("EXACT",))
        self.assertEqual(result.beliefs[0].weight, 1.0)
        disclosures = {
            item.parent_asin: item.disclosed_values for item in result.disclosures
        }
        self.assertEqual(disclosures["EXACT"], ("cotton", "color: blue"))
        self.assertEqual(result.trace.strong_disclosed_value_count, 2)
        self.assertEqual(result.trace.tentative_clue_count, 1)

    def test_exact_coarse_category_mismatch_is_not_consistent_support(self) -> None:
        candidates = (
            _evidence("WRONG_CATEGORY", category="Men Running Shoes"),
            _evidence("RIGHT_CATEGORY"),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(Requirement("cotton", "initial_explicit", 1, "material"),),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertEqual(result.ranked_ids, ("RIGHT_CATEGORY", "WRONG_CATEGORY"))
        self.assertEqual(result.consistent_support_ids, ("RIGHT_CATEGORY",))
        self.assertEqual(result.trace.category_compatible_count, 1)

    def test_conjunction_dominates_a_partial_single_constraint_match(self) -> None:
        candidates = (
            _evidence("SINGLE", hard=("cotton", "color: red")),
            _evidence("CONJUNCTION", hard=("cotton", "color: blue")),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton; color: blue", "answer", 2, "other"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertEqual(result.ranked_ids[0], "CONJUNCTION")
        self.assertEqual(result.consistent_support_ids, ("CONJUNCTION",))

    def test_customer_evidence_then_base_order_dominate_popularity(self) -> None:
        stronger_candidates = (
            _evidence("POPULAR", text="cotton shoe", popularity=10**18),
            _evidence("STRONGER", text="waterproof cotton shoe", popularity=0),
        )
        clue_state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "answer", 2, "material"),
                Requirement("waterproof", "initial_tentative", 1, "feature"),
            ),
        )
        stronger_result = rank_exact_evidence(
            tuple(item.parent_asin for item in stronger_candidates),
            stronger_candidates,
            clue_state,
        )

        tied_candidates = (
            _evidence("HYBRID_FIRST", text="cotton", popularity=0),
            _evidence("POPULAR_SECOND", text="cotton", popularity=10**18),
        )
        unsignaled_result = rank_exact_evidence(
            tuple(item.parent_asin for item in tied_candidates),
            tied_candidates,
            IntentState(category="Women Running Shoes"),
        )
        signaled_result = rank_exact_evidence(
            tuple(item.parent_asin for item in tied_candidates),
            tied_candidates,
            IntentState(
                category="Women Running Shoes",
                requirements=(
                    Requirement("cotton", "initial_explicit", 1, "material"),
                ),
            ),
        )

        self.assertEqual(stronger_result.ranked_ids[0], "STRONGER")
        self.assertEqual(unsignaled_result.ranked_ids[0], "HYBRID_FIRST")
        self.assertEqual(signaled_result.ranked_ids[0], "HYBRID_FIRST")

    def test_category_only_browsing_preserves_selected_route_order(self) -> None:
        candidates = (
            _evidence("BASE_FIRST", category="Boots"),
            _evidence("CATEGORY_MATCH", category="Women Running Shoes"),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            IntentState(category="Women Running Shoes"),
        )

        self.assertEqual(result.ranked_ids, ("BASE_FIRST", "CATEGORY_MATCH"))

    def test_tied_best_tier_uses_a_normalized_reciprocal_rank_prior(self) -> None:
        candidates = (
            _evidence("FIRST"),
            _evidence("SECOND"),
            _evidence("THIRD"),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            IntentState(category="Women Running Shoes"),
        )

        self.assertAlmostEqual(sum(item.weight for item in result.beliefs), 1.0)
        self.assertGreater(result.beliefs[0].weight, result.beliefs[1].weight)
        self.assertGreater(result.beliefs[1].weight, result.beliefs[2].weight)

    def test_non_product_protocol_events_preserve_existing_base_order(self) -> None:
        candidates = (
            _evidence("HYBRID_FIRST", popularity=0),
            _evidence("POPULAR_SECOND", popularity=10**18),
        )
        event_streams = (
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_BROWSING,
                ),
            ),
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_BROWSING,
                ),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.BOUNDARY_DECLINE,
                    "other",
                ),
            ),
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_BROWSING,
                ),
                ObservedProtocolEvent(
                    2,
                    ProtocolEventKind.NEED_ATTRIBUTE,
                ),
            ),
        )
        for events in event_streams:
            with self.subTest(events=events):
                result = rank_exact_evidence(
                    tuple(item.parent_asin for item in candidates),
                    candidates,
                    IntentState(category="Women Running Shoes"),
                    protocol_events=events,
                )

                self.assertEqual(
                    result.ranked_ids,
                    ("HYBRID_FIRST", "POPULAR_SECOND"),
                )

    def test_reply_order_and_source_are_replayed_against_each_card(self) -> None:
        candidates = (
            _evidence(
                "WRONG_REPLY_ORDER",
                hard=("cotton", "waterproof"),
                soft=("color: blue",),
            ),
            _evidence(
                "EXACT_REPLY_ORDER",
                hard=("cotton", "color: blue"),
                soft=("waterproof",),
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("color: blue; waterproof", "answer", 2, "other"),
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertEqual(result.ranked_ids[0], "EXACT_REPLY_ORDER")
        self.assertEqual(result.consistent_support_ids, ("EXACT_REPLY_ORDER",))
        disclosures = {
            item.parent_asin: item.disclosed_values for item in result.disclosures
        }
        self.assertEqual(
            disclosures["EXACT_REPLY_ORDER"],
            ("cotton", "color: blue", "waterproof"),
        )
        self.assertEqual(disclosures["WRONG_REPLY_ORDER"], ("cotton",))
        self.assertEqual(result.trace.reply_consistent_count, 1)

    def test_answer_attribute_must_reproduce_the_official_reply(self) -> None:
        candidates = (
            _evidence(
                "ATTRIBUTE_MISMATCH",
                hard=("cotton", "color: blue"),
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
                Requirement("color: blue", "answer", 2, "feature"),
            ),
        )

        result = rank_exact_evidence(
            ("ATTRIBUTE_MISMATCH",), candidates, state
        )

        self.assertIs(
            result.status,
            ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT,
        )
        self.assertEqual(result.trace.reply_consistent_count, 0)

        impossible_same_turn = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "answer", 2, "material"),
                Requirement("color: blue", "answer", 2, "color"),
            ),
        )
        grouped_result = rank_exact_evidence(
            ("ATTRIBUTE_MISMATCH",),
            candidates,
            impossible_same_turn,
        )
        self.assertEqual(grouped_result.trace.reply_consistent_count, 0)

    def test_override_must_equal_the_first_hard_constraint(self) -> None:
        candidates = (
            _evidence("WRONG_OVERRIDE", hard=("color: blue", "cotton")),
            _evidence("EXACT_OVERRIDE", hard=("cotton", "color: blue")),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "override", 3, "material"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertEqual(result.ranked_ids[0], "EXACT_OVERRIDE")
        self.assertEqual(result.consistent_support_ids, ("EXACT_OVERRIDE",))

    def test_explicit_exclusion_vetoes_card_or_text_matches(self) -> None:
        candidates = (
            _evidence(
                "CARD_VIOLATION",
                hard=("cotton",),
                soft=("leather",),
                text="waterproof cotton leather shoe",
                popularity=10**18,
            ),
            _evidence(
                "TEXT_VIOLATION",
                hard=("cotton",),
                soft=("breathable",),
                text="waterproof cotton shoe with leather trim",
                popularity=10**18,
            ),
            _evidence(
                "CLEAN",
                hard=("cotton",),
                soft=("breathable",),
                text="plain cotton shoe",
                popularity=0,
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
                Requirement("waterproof", "initial_tentative", 1, "feature"),
            ),
            excluded=("leather",),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates), candidates, state
        )

        self.assertEqual(result.ranked_ids[0], "CLEAN")
        self.assertEqual(result.consistent_support_ids, ("CLEAN",))
        self.assertEqual(result.trace.exclusion_violation_count, 2)

    def test_zero_consistent_support_fails_open_to_original_base_order(self) -> None:
        candidates = (
            _evidence("FIRST", category="Women Sandals", hard=("silk",)),
            _evidence(
                "SECOND",
                category="Men Running Shoes",
                hard=("cotton", "color: blue"),
                text="cotton blue waterproof",
            ),
        )
        original_ids = tuple(item.parent_asin for item in candidates)
        state = IntentState(
            category="Women Running Shoes",
            requirements=(Requirement("cotton; color: blue", "answer", 2, "other"),),
        )

        result = rank_exact_evidence(original_ids, candidates, state)

        self.assertIs(
            result.status,
            ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT,
        )
        self.assertEqual(result.ranked_ids, original_ids)
        self.assertEqual(result.consistent_support_ids, ())
        self.assertEqual(result.beliefs, ())
        self.assertEqual(result.trace.best_tier_count, 0)

    def test_strong_exact_evidence_dominates_a_base_first_semantic_neighbor(
        self,
    ) -> None:
        candidates = (
            _evidence(
                "SEMANTIC_NEIGHBOR",
                hard=("leather",),
                text="cotton inspired casual shoe",
            ),
            _evidence(
                "EXACT_CONSTRAINT",
                hard=("cotton",),
                text="plain cotton shoe",
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
        )

        self.assertEqual(result.ranked_ids[0], "EXACT_CONSTRAINT")
        self.assertEqual(result.consistent_support_ids, ("EXACT_CONSTRAINT",))

    def test_low_confidence_free_text_does_not_override_base_order(self) -> None:
        candidates = (
            _evidence("BASE_FIRST", hard=("leather",), popularity=0),
            _evidence("TEXT_MATCH", hard=("cotton",), popularity=10**18),
        )
        uncertain = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "free_text", 1, "material"),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            uncertain,
        )

        self.assertEqual(result.ranked_ids, ("BASE_FIRST", "TEXT_MATCH"))
        self.assertEqual(result.trace.strong_disclosed_value_count, 0)
        self.assertEqual(result.trace.tentative_clue_count, 0)

    def test_explicitly_soft_answer_is_ranked_as_a_clue(self) -> None:
        candidates = (
            _evidence("BASE_FIRST", hard=("leather",), text="plain shoe"),
            _evidence("CLUE_MATCH", hard=("leather",), text="cotton shoe"),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(
                    "cotton",
                    "answer",
                    2,
                    "material",
                    strength="soft",
                ),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
        )

        self.assertEqual(result.ranked_ids[0], "CLUE_MATCH")
        self.assertEqual(result.trace.strong_disclosed_value_count, 0)
        self.assertEqual(result.trace.tentative_clue_count, 1)

    def test_contradictory_positive_and_negative_evidence_fails_open(self) -> None:
        candidates = (
            _evidence("POSITIVE_BUT_EXCLUDED", hard=("cotton",)),
            _evidence("CLEAN_BUT_INCOMPATIBLE", hard=("leather",)),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
            excluded=("cotton",),
        )

        for ordered in permutations(candidates):
            with self.subTest(order=tuple(item.parent_asin for item in ordered)):
                original_ids = tuple(item.parent_asin for item in ordered)
                result = rank_exact_evidence(original_ids, ordered, state)

                self.assertIs(
                    result.status,
                    ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT,
                )
                self.assertEqual(result.ranked_ids, original_ids)
                self.assertEqual(result.consistent_support_ids, ())
                self.assertEqual(result.beliefs, ())

    def test_missing_optional_metadata_is_deterministic_and_safe(self) -> None:
        candidates = (
            _evidence("FIRST", hard=("cotton",), text=""),
            _evidence("SECOND", hard=("cotton",), text=""),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )

        first = rank_exact_evidence(("FIRST", "SECOND"), candidates, state)
        second = rank_exact_evidence(("FIRST", "SECOND"), candidates, state)

        self.assertEqual(first, second)
        self.assertEqual(first.ranked_ids, ("FIRST", "SECOND"))

    def test_adding_satisfied_exact_evidence_is_monotone_over_partial_match(
        self,
    ) -> None:
        candidates = (
            _evidence("PARTIAL", hard=("cotton", "color: red")),
            _evidence("COMPLETE", hard=("cotton", "color: blue")),
        )
        cotton_only = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )
        compound = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
                Requirement("color: blue", "answer", 2, "color"),
            ),
        )

        initial = rank_exact_evidence(
            ("PARTIAL", "COMPLETE"), candidates, cotton_only
        )
        refined = rank_exact_evidence(
            ("PARTIAL", "COMPLETE"), candidates, compound
        )

        self.assertEqual(initial.ranked_ids[0], "PARTIAL")
        self.assertEqual(refined.ranked_ids[0], "COMPLETE")

    def test_lexicographic_strength_is_invariant_to_base_order(self) -> None:
        candidates = (
            _evidence(
                "EXACT",
                hard=("cotton", "color: blue"),
            ),
            _evidence(
                "FIELD_PARTIAL",
                hard=("cotton", "color: red"),
                text="cotton color blue shoe",
            ),
            _evidence(
                "TEXT_ONLY",
                hard=("silk", "color: red"),
                text="cotton color blue shoe",
            ),
            _evidence(
                "NO_EVIDENCE",
                hard=("silk", "color: red"),
                text="ordinary shoe",
                popularity=10**18,
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton; color: blue", "answer", 2, "other"),
            ),
        )
        expected = ("EXACT", "FIELD_PARTIAL", "TEXT_ONLY", "NO_EVIDENCE")

        for ordered in permutations(candidates):
            with self.subTest(order=tuple(item.parent_asin for item in ordered)):
                result = rank_exact_evidence(
                    tuple(item.parent_asin for item in ordered),
                    ordered,
                    state,
                )

                self.assertIs(result.status, ExactEvidenceStatus.APPLIED)
                self.assertEqual(result.ranked_ids, expected)
                self.assertEqual(result.consistent_support_ids, ("EXACT",))

    def test_tentative_clue_can_rerank_but_never_removes_support(self) -> None:
        candidates = (
            _evidence(
                "BASE_FIRST",
                hard=("leather",),
                text="plain shoe",
            ),
            _evidence(
                "CLUE_MATCH",
                hard=("cotton",),
                text="waterproof cotton shoe",
            ),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(
                    "waterproof",
                    "initial_tentative",
                    1,
                    "feature",
                ),
            ),
        )

        result = rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
        )

        self.assertEqual(result.ranked_ids[0], "CLUE_MATCH")
        self.assertEqual(
            frozenset(result.consistent_support_ids),
            frozenset({"BASE_FIRST", "CLUE_MATCH"}),
        )
        self.assertEqual(result.trace.strong_disclosed_value_count, 0)
        self.assertEqual(result.trace.tentative_clue_count, 1)

    def test_missing_catalog_fields_fail_open_and_missing_price_is_neutral(
        self,
    ) -> None:
        missing = (
            _evidence("MISSING_ONE", category="", hard=(), text=""),
            _evidence("MISSING_TWO", category="", hard=(), text=""),
        )
        explicit = IntentState(
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )
        for ordered in permutations(missing):
            with self.subTest(missing=tuple(item.parent_asin for item in ordered)):
                original_ids = tuple(item.parent_asin for item in ordered)
                result = rank_exact_evidence(original_ids, ordered, explicit)
                self.assertIs(
                    result.status,
                    ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT,
                )
                self.assertEqual(result.ranked_ids, original_ids)

        budget_candidates = (
            _evidence(
                "MISSING_PRICE",
                hard=("budget under $50",),
                price=None,
            ),
            _evidence(
                "KNOWN_COMPATIBLE_PRICE",
                hard=("budget under $50",),
                price="$30",
            ),
        )
        budget_state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement(
                    "budget under $50",
                    "initial_explicit",
                    1,
                    "budget",
                ),
            ),
        )

        budget_result = rank_exact_evidence(
            tuple(item.parent_asin for item in budget_candidates),
            budget_candidates,
            budget_state,
        )

        self.assertEqual(budget_result.ranked_ids[0], "KNOWN_COMPATIBLE_PRICE")
        self.assertEqual(
            frozenset(budget_result.consistent_support_ids),
            frozenset({"MISSING_PRICE", "KNOWN_COMPATIBLE_PRICE"}),
        )


class DenseBestTierTieBreakTests(unittest.TestCase):
    @staticmethod
    def _ranking():
        candidates = (
            _evidence("EXACT_A", hard=("cotton",)),
            _evidence("EXACT_B", hard=("cotton",)),
            _evidence("LOWER_TIER", hard=("silk",)),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )
        return rank_exact_evidence(
            tuple(item.parent_asin for item in candidates),
            candidates,
            state,
        )

    def test_dense_rank_reorders_only_the_fully_covered_best_tier(self) -> None:
        baseline = self._ranking()

        result = apply_dense_best_tier_tiebreak(
            baseline,
            ("EXACT_B", "LOWER_TIER", "EXACT_A"),
            policy=DENSE_COMPLETE_BEST_TIER_POLICY,
        )

        self.assertIs(result.status, SemanticTieBreakStatus.REORDERED)
        self.assertEqual(
            result.ranking.ranked_ids,
            ("EXACT_B", "EXACT_A", "LOWER_TIER"),
        )
        self.assertEqual(
            tuple(item.parent_asin for item in result.ranking.beliefs),
            ("EXACT_B", "EXACT_A"),
        )
        self.assertEqual(
            tuple(item.parent_asin for item in result.ranking.disclosures),
            result.ranking.ranked_ids,
        )
        self.assertEqual(result.ranking.trace, baseline.trace)
        self.assertEqual(
            frozenset(result.ranking.consistent_support_ids),
            frozenset(baseline.consistent_support_ids),
        )

    def test_incomplete_dense_coverage_is_neutral(self) -> None:
        baseline = self._ranking()

        result = apply_dense_best_tier_tiebreak(
            baseline,
            ("EXACT_B", "LOWER_TIER"),
            policy=DENSE_COMPLETE_BEST_TIER_POLICY,
        )

        self.assertIs(
            result.status,
            SemanticTieBreakStatus.INCOMPLETE_DENSE_COVERAGE,
        )
        self.assertIs(result.ranking, baseline)

    def test_confident_policy_requires_and_gates_raw_cosine_scores(self) -> None:
        baseline = self._ranking()
        dense_ids = ("EXACT_B", "EXACT_A", "LOWER_TIER")

        missing = apply_dense_best_tier_tiebreak(
            baseline,
            dense_ids,
            policy=DENSE_CONFIDENT_BEST_TIER_POLICY,
        )
        self.assertIs(
            missing.status,
            SemanticTieBreakStatus.SCORES_UNAVAILABLE,
        )
        self.assertIs(missing.ranking, baseline)

        low_margin = apply_dense_best_tier_tiebreak(
            baseline,
            dense_ids,
            dense_scores=(0.61, 0.60, 0.40),
            policy=DENSE_CONFIDENT_BEST_TIER_POLICY,
        )
        self.assertIs(
            low_margin.status,
            SemanticTieBreakStatus.LOW_CONFIDENCE,
        )
        self.assertIs(low_margin.ranking, baseline)

        confident = apply_dense_best_tier_tiebreak(
            baseline,
            dense_ids,
            dense_scores=(0.75, 0.60, 0.40),
            policy=DENSE_CONFIDENT_BEST_TIER_POLICY,
        )
        self.assertIs(confident.status, SemanticTieBreakStatus.REORDERED)
        self.assertEqual(
            confident.ranking.ranked_ids,
            ("EXACT_B", "EXACT_A", "LOWER_TIER"),
        )

    def test_disabled_singleton_and_zero_support_are_neutral(self) -> None:
        baseline = self._ranking()
        disabled = apply_dense_best_tier_tiebreak(
            baseline,
            ("EXACT_B", "EXACT_A", "LOWER_TIER"),
            policy=DISABLED_SEMANTIC_TIEBREAK_POLICY,
        )
        self.assertIs(disabled.status, SemanticTieBreakStatus.DISABLED)
        self.assertIs(disabled.ranking, baseline)

        singleton_candidates = (
            _evidence("ONLY_EXACT", hard=("cotton",)),
            _evidence("LOWER", hard=("silk",)),
        )
        state = IntentState(
            category="Women Running Shoes",
            requirements=(
                Requirement("cotton", "initial_explicit", 1, "material"),
            ),
        )
        singleton_ranking = rank_exact_evidence(
            tuple(item.parent_asin for item in singleton_candidates),
            singleton_candidates,
            state,
        )
        singleton = apply_dense_best_tier_tiebreak(
            singleton_ranking,
            ("LOWER", "ONLY_EXACT"),
            policy=DENSE_COMPLETE_BEST_TIER_POLICY,
        )
        self.assertIs(singleton.status, SemanticTieBreakStatus.SINGLETON)
        self.assertIs(singleton.ranking, singleton_ranking)

        zero_support_candidates = (
            _evidence("NO_A", hard=("silk",)),
            _evidence("NO_B", hard=("leather",)),
        )
        zero_support_ranking = rank_exact_evidence(
            tuple(item.parent_asin for item in zero_support_candidates),
            zero_support_candidates,
            state,
        )
        zero_support = apply_dense_best_tier_tiebreak(
            zero_support_ranking,
            ("NO_B", "NO_A"),
            policy=DENSE_COMPLETE_BEST_TIER_POLICY,
        )
        self.assertIs(zero_support.status, SemanticTieBreakStatus.NO_SUPPORT)
        self.assertIs(zero_support.ranking, zero_support_ranking)

    def test_malformed_dense_route_is_rejected(self) -> None:
        baseline = self._ranking()
        with self.assertRaisesRegex(ValueError, "unique"):
            apply_dense_best_tier_tiebreak(
                baseline,
                ("EXACT_A", "EXACT_A"),
                policy=DENSE_COMPLETE_BEST_TIER_POLICY,
            )
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            apply_dense_best_tier_tiebreak(
                baseline,
                ("OUTSIDE",),
                policy=DENSE_COMPLETE_BEST_TIER_POLICY,
            )


class ExactEvidenceContractTests(unittest.TestCase):
    def test_bounds_duplicates_alignment_and_determinism(self) -> None:
        too_many = tuple(
            _evidence(f"ID{index:03d}")
            for index in range(MAX_EXACT_EVIDENCE_CANDIDATES + 1)
        )
        with self.assertRaisesRegex(ValueError, "at most 200"):
            rank_exact_evidence(
                tuple(item.parent_asin for item in too_many),
                too_many,
                IntentState(),
            )

        duplicates = (_evidence("DUPLICATE"), _evidence("DUPLICATE"))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            rank_exact_evidence(
                ("DUPLICATE", "DUPLICATE"), duplicates, IntentState()
            )

        aligned = (_evidence("ONE"), _evidence("TWO"))
        with self.assertRaisesRegex(ValueError, "align positionally"):
            rank_exact_evidence(("TWO", "ONE"), aligned, IntentState())

        state = IntentState(category="Women Running Shoes")
        first = rank_exact_evidence(("ONE", "TWO"), aligned, state)
        second = rank_exact_evidence(("ONE", "TWO"), aligned, state)
        self.assertEqual(first, second)
        self.assertEqual(first.ranked_ids, ("ONE", "TWO"))
        self.assertEqual(
            [(belief.parent_asin, belief.weight) for belief in first.beliefs],
            [("ONE", 2.0 / 3.0), ("TWO", 1.0 / 3.0)],
        )
        self.assertTrue(
            all(
                isinstance(value, int)
                for value in dataclasses.asdict(first.trace).values()
            )
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            first.ranked_ids = ()  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
