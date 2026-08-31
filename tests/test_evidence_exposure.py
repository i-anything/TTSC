from __future__ import annotations

import unittest

from conversational_search.exact_evidence import rank_exact_evidence
from conversational_search.exposure import (
    plan_evidence_gated_exposure,
    plan_protocol_enumeration_width,
    plan_protocol_reply_tree_width,
)
from conversational_search.exposure_policy import EvidenceExposureStatus
from conversational_search.intent import IntentState, Requirement
from conversational_search.protocol import (
    DisclosureCard,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
)
from conversational_search.protocol_index import resolve_protocol_transcript


def _evidence(
    parent_asin: str,
    color: str,
) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin=parent_asin,
        coarse_category="Shoes",
        card=DisclosureCard(
            f"{parent_asin} shoe",
            ("waterproof",),
            (f"color: {color}",),
        ),
        text=f"waterproof {color} shoe",
    )


def _state(
    *,
    source: str = "initial_explicit",
    excluded: tuple[str, ...] = (),
) -> IntentState:
    return IntentState(
        category="Shoes",
        requirements=(
            Requirement("waterproof", source, 1, "feature"),
        ),
        excluded=excluded,
        last_turn=1,
    )


class EvidenceExposureTests(unittest.TestCase):
    def _plan(
        self,
        colors: tuple[str, ...],
        *,
        state: IntentState | None = None,
        current_turn: int = 1,
        top_k: int = 10,
        retrieval_fault: bool = False,
        require_initial_explicit_buying: bool = False,
        question_prefix_limit: int = 0,
        initial_ambiguous_prefix_limit: int = 0,
    ):
        evidence = tuple(
            _evidence(f"P{index}", color)
            for index, color in enumerate(colors)
        )
        active_state = state or _state()
        exact = rank_exact_evidence(
            tuple(item.parent_asin for item in evidence),
            evidence,
            active_state,
        )
        return plan_evidence_gated_exposure(
            active_state,
            exact,
            evidence,
            current_turn=current_turn,
            requested_top_k=top_k,
            retrieval_fault_or_fallback=retrieval_fault,
            require_initial_explicit_buying=require_initial_explicit_buying,
            question_prefix_limit=question_prefix_limit,
            initial_ambiguous_prefix_limit=initial_ambiguous_prefix_limit,
        )

    def test_initial_category_only_state_exposes_rank_one_and_asks_other(self) -> None:
        state = IntentState(category="Shoes", last_turn=1)

        decision = self._plan(
            ("blue", "red", "green", "black"),
            state=state,
            initial_ambiguous_prefix_limit=1,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW,
        )
        self.assertEqual(decision.presentation_ids, ("P0",))
        self.assertEqual((decision.width, decision.question), (1, "other"))
        self.assertEqual(decision.plausible_count, 4)

    def test_ambiguous_preview_requires_turn_one(self) -> None:
        later = self._plan(
            ("blue", "red", "green", "black"),
            state=IntentState(category="Shoes", last_turn=2),
            current_turn=2,
            top_k=3,
            initial_ambiguous_prefix_limit=1,
        )
        shared_reply = self._plan(
            ("blue", "blue", "blue", "blue"),
            state=IntentState(category="Shoes", last_turn=1),
            top_k=3,
            initial_ambiguous_prefix_limit=1,
        )
        retrieval_fault = self._plan(
            ("blue", "red", "green", "black"),
            state=IntentState(category="Shoes", last_turn=1),
            top_k=3,
            retrieval_fault=True,
            initial_ambiguous_prefix_limit=1,
        )

        self.assertIs(later.status, EvidenceExposureStatus.UNSAFE_STATE)
        self.assertIs(
            shared_reply.status,
            EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW,
        )
        self.assertEqual(later.width, 3)
        self.assertIsNone(later.question)
        self.assertEqual(shared_reply.presentation_ids, ("P0",))
        self.assertEqual((shared_reply.width, shared_reply.question), (1, "other"))
        self.assertIs(
            retrieval_fault.status,
            EvidenceExposureStatus.RETRIEVAL_FAIL_OPEN,
        )
        self.assertEqual(retrieval_fault.width, 3)
        self.assertIsNone(retrieval_fault.question)

    def test_one_through_three_plausible_candidates_fit_the_exposed_prefix(self) -> None:
        for count in (1, 2, 3):
            with self.subTest(count=count):
                decision = self._plan(tuple(f"color{index}" for index in range(count)))

                self.assertIs(
                    decision.status,
                    EvidenceExposureStatus.TOP3_CONFIDENT,
                )
                self.assertEqual(decision.width, count)
                self.assertEqual(len(decision.presentation_ids), count)
                self.assertIsNone(decision.question)

    def test_more_than_three_plausible_candidates_withhold_for_best_question(self) -> None:
        decision = self._plan(("blue", "red", "green", "black"))

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.QUESTION_WITHHELD,
        )
        self.assertEqual(decision.width, 0)
        self.assertEqual(decision.question, "color")
        self.assertEqual(len(decision.presentation_ids), 4)

    def test_prefix_too_small_uses_the_same_gate_even_when_tier_has_three(self) -> None:
        decision = self._plan(("blue", "red", "green"), top_k=2)

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.QUESTION_WITHHELD,
        )
        self.assertEqual((decision.width, decision.question), (0, "color"))

    def test_question_prefix_exposes_only_the_literal_top_three(self) -> None:
        decision = self._plan(
            ("blue", "red", "green", "black"),
            top_k=10,
            require_initial_explicit_buying=True,
            question_prefix_limit=3,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.QUESTION_WITH_PREFIX,
        )
        self.assertEqual(decision.presentation_ids, ("P0", "P1", "P2"))
        self.assertEqual((decision.width, decision.question), (3, "color"))
        self.assertEqual(decision.plausible_count, 4)

    def test_question_prefix_respects_a_smaller_api_width(self) -> None:
        decision = self._plan(
            ("blue", "red", "green"),
            top_k=2,
            require_initial_explicit_buying=True,
            question_prefix_limit=3,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.QUESTION_WITH_PREFIX,
        )
        self.assertEqual(decision.presentation_ids, ("P0", "P1"))
        self.assertEqual((decision.width, decision.question), (2, "color"))

    def test_no_informative_question_returns_the_full_available_slate(self) -> None:
        decision = self._plan(("blue", "blue", "blue", "blue"), top_k=3)

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.NO_INFORMATIVE_QUESTION,
        )
        self.assertEqual(decision.width, 3)
        self.assertEqual(len(decision.presentation_ids), 4)
        self.assertIsNone(decision.question)

    def test_final_turn_returns_full_slate_without_a_question(self) -> None:
        state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "initial_explicit", 10, "feature"),
            ),
            last_turn=10,
        )
        decision = self._plan(
            ("blue", "red", "green", "black"),
            state=state,
            current_turn=10,
            top_k=3,
        )

        self.assertIs(decision.status, EvidenceExposureStatus.FINAL_TURN)
        self.assertEqual(decision.width, 3)
        self.assertIsNone(decision.question)

    def test_fault_override_and_contradiction_fail_open(self) -> None:
        fault = self._plan(
            ("blue", "red", "green", "black"),
            top_k=3,
            retrieval_fault=True,
        )
        override = self._plan(
            ("blue", "red", "green", "black"),
            state=_state(source="override"),
            top_k=3,
        )
        contradiction = self._plan(
            ("blue", "red", "green", "black"),
            state=_state(excluded=("waterproof",)),
            top_k=3,
        )

        self.assertIs(fault.status, EvidenceExposureStatus.RETRIEVAL_FAIL_OPEN)
        self.assertIs(override.status, EvidenceExposureStatus.UNSAFE_STATE)
        self.assertIs(contradiction.status, EvidenceExposureStatus.UNSAFE_STATE)
        for decision in (fault, override, contradiction):
            self.assertEqual(decision.width, 3)
            self.assertIsNone(decision.question)

    def test_buying_only_gate_rejects_answer_only_browsing_state(self) -> None:
        browsing_state = IntentState(
            category="Shoes",
            requirements=(
                Requirement("waterproof", "answer", 2, "feature"),
            ),
            asked_attributes=("feature",),
            last_asked_attribute="feature",
            last_turn=2,
        )

        decision = self._plan(
            ("blue", "red", "green", "black"),
            state=browsing_state,
            current_turn=2,
            top_k=3,
            require_initial_explicit_buying=True,
        )

        self.assertIs(decision.status, EvidenceExposureStatus.UNSAFE_STATE)
        self.assertEqual(decision.width, 3)
        self.assertIsNone(decision.question)

    def test_buying_only_gate_accepts_stable_initial_explicit_state(self) -> None:
        decision = self._plan(
            ("blue", "red", "green", "black"),
            top_k=3,
            require_initial_explicit_buying=True,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.QUESTION_WITHHELD,
        )
        self.assertEqual((decision.width, decision.question), (0, "color"))

    def test_protocol_posterior_exposes_singleton_at_rank_one(self) -> None:
        evidence = (_evidence("P0", "blue"),)
        state = _state()
        exact = rank_exact_evidence(("P0",), evidence, state)
        resolution = resolve_protocol_transcript(
            evidence,
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("waterproof",),
                ),
            ),
            observed_turn_count=1,
        )

        decision = plan_evidence_gated_exposure(
            state,
            exact,
            evidence,
            current_turn=1,
            requested_top_k=10,
            protocol_resolution=resolution,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.POSTERIOR_SINGLETON,
        )
        self.assertEqual(decision.presentation_ids, ("P0",))
        self.assertEqual((decision.width, decision.question), (1, None))

    def test_protocol_posterior_repeats_other_with_a_rank_one_probe(self) -> None:
        evidence = tuple(_evidence(f"P{index}", color) for index, color in enumerate(("blue", "red", "green")))
        state = _state()
        ids = tuple(item.parent_asin for item in evidence)
        exact = rank_exact_evidence(ids, evidence, state)
        resolution = resolve_protocol_transcript(
            evidence,
            (
                ObservedProtocolEvent(
                    1,
                    ProtocolEventKind.INITIAL_EXPLICIT,
                    values=("waterproof",),
                ),
            ),
            observed_turn_count=1,
        )

        decision = plan_evidence_gated_exposure(
            state,
            exact,
            evidence,
            current_turn=1,
            requested_top_k=10,
            retrieval_fault_or_fallback=True,
            protocol_resolution=resolution,
        )

        self.assertIs(decision.status, EvidenceExposureStatus.POSTERIOR_PROBE)
        self.assertEqual(decision.presentation_ids, ("P0",))
        self.assertEqual((decision.width, decision.question), (1, "other"))
        self.assertEqual(decision.plausible_count, 3)

    def test_exhausted_protocol_posterior_uses_a_batch_and_final_turn_never_asks(
        self,
    ) -> None:
        evidence = tuple(_evidence(f"P{index}", "blue") for index in range(3))
        ids = tuple(item.parent_asin for item in evidence)
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("waterproof",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                reply_payload="color: blue",
            ),
        )
        resolution = resolve_protocol_transcript(
            evidence,
            events,
            observed_turn_count=2,
        )
        state = IntentState(category="Shoes", last_turn=2)
        exact = rank_exact_evidence(ids, evidence, state, protocol_events=events)

        decision = plan_evidence_gated_exposure(
            state,
            exact,
            evidence,
            current_turn=2,
            requested_top_k=2,
            protocol_resolution=resolution,
        )

        self.assertIs(decision.status, EvidenceExposureStatus.POSTERIOR_BATCH)
        self.assertEqual(decision.width, 2)
        self.assertIsNone(decision.question)

    def test_metric_aware_enumeration_uses_future_refutation_value(self) -> None:
        evidence = tuple(_evidence(f"P{index}", "blue") for index in range(3))
        ids = tuple(item.parent_asin for item in evidence)
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("waterproof",),
            ),
            ObservedProtocolEvent(
                2,
                ProtocolEventKind.DISCLOSURE,
                "other",
                reply_payload="color: blue",
            ),
        )
        resolution = resolve_protocol_transcript(
            evidence,
            events,
            observed_turn_count=2,
        )
        state = IntentState(category="Shoes", last_turn=2)
        exact = rank_exact_evidence(ids, evidence, state, protocol_events=events)

        decision = plan_evidence_gated_exposure(
            state,
            exact,
            evidence,
            current_turn=2,
            requested_top_k=2,
            protocol_resolution=resolution,
            metric_aware_protocol_enumeration=True,
        )

        self.assertIs(
            decision.status,
            EvidenceExposureStatus.POSTERIOR_ENUMERATION,
        )
        self.assertEqual(decision.width, 1)
        self.assertIsNone(decision.question)

    def test_metric_aware_width_is_derived_from_the_official_metric(self) -> None:
        self.assertEqual(
            plan_protocol_enumeration_width(3, current_turn=2, top_k=10),
            1,
        )
        self.assertEqual(
            plan_protocol_enumeration_width(10, current_turn=2, top_k=10),
            2,
        )
        self.assertEqual(
            plan_protocol_enumeration_width(3, current_turn=10, top_k=10),
            3,
        )

    def test_reply_tree_enumerates_only_when_the_next_reply_is_shared(self) -> None:
        card = DisclosureCard("same shoe", ("waterproof",), ("color: blue",))
        evidence = tuple(
            ProductProtocolEvidence(
                parent_asin=f"P{index}",
                coarse_category="Shoes",
                card=card,
                text="waterproof blue shoe",
            )
            for index in range(10)
        )
        ids = tuple(item.parent_asin for item in evidence)
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("waterproof",),
            ),
        )
        resolution = resolve_protocol_transcript(
            evidence,
            events,
            observed_turn_count=1,
        )

        self.assertEqual(
            plan_protocol_reply_tree_width(
                ids,
                resolution,
                current_turn=1,
                top_k=10,
            ),
            2,
        )

    def test_reply_tree_falls_back_when_ranked_pool_is_incomplete(self) -> None:
        evidence = tuple(_evidence(f"P{index}", "blue") for index in range(3))
        events = (
            ObservedProtocolEvent(
                1,
                ProtocolEventKind.INITIAL_EXPLICIT,
                values=("waterproof",),
            ),
        )
        resolution = resolve_protocol_transcript(
            evidence,
            events,
            observed_turn_count=1,
        )

        self.assertEqual(
            plan_protocol_reply_tree_width(
                ("P0", "P1"),
                resolution,
                current_turn=1,
                top_k=10,
            ),
            1,
        )

if __name__ == "__main__":
    unittest.main()
