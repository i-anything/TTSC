from __future__ import annotations

import unittest

from conversational_search.exact_evidence import rank_exact_evidence
from conversational_search.exposure import plan_evidence_gated_exposure
from conversational_search.exposure_policy import EvidenceExposureStatus
from conversational_search.intent import IntentState, Requirement
from conversational_search.protocol import DisclosureCard, ProductProtocolEvidence


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
        )

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


if __name__ == "__main__":
    unittest.main()
