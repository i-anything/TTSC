from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from typing import cast

from conversational_search.intent import IntentState, Requirement, RequirementSource
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    EQUAL_RRF_POLICY,
    FusionPolicy,
    RouteWeights,
    intent_completeness,
)


def _state(*sources: RequirementSource) -> IntentState:
    return IntentState(
        requirements=tuple(
            Requirement(value=f"requirement-{index}", source=source, turn=index)
            for index, source in enumerate(sources, start=1)
        )
    )


class IntentCompletenessTests(unittest.TestCase):
    def test_empty_intent_has_zero_completeness(self) -> None:
        self.assertEqual(intent_completeness(IntentState()), 0.0)

    def test_strong_provenance_contributes_one_point(self) -> None:
        for source in ("initial_explicit", "answer", "override"):
            with self.subTest(source=source):
                self.assertAlmostEqual(intent_completeness(_state(source)), 1.0 / 3.0)

    def test_weak_provenance_contributes_half_a_point(self) -> None:
        for source in ("initial_tentative", "free_text"):
            with self.subTest(source=source):
                self.assertAlmostEqual(intent_completeness(_state(source)), 1.0 / 6.0)

    def test_mixed_evidence_is_additive_and_input_is_unchanged(self) -> None:
        state = _state("answer", "free_text")
        original_requirements = state.requirements

        self.assertEqual(intent_completeness(state), 0.5)
        self.assertIs(state.requirements, original_requirements)

    def test_completeness_clamps_at_one(self) -> None:
        complete = _state("answer", "answer", "answer")
        overcomplete = _state("answer", "answer", "answer", "answer")

        self.assertEqual(intent_completeness(complete), 1.0)
        self.assertEqual(intent_completeness(overcomplete), 1.0)

    def test_unknown_provenance_is_rejected(self) -> None:
        unknown = cast(RequirementSource, "unknown")
        with self.assertRaisesRegex(ValueError, "unsupported requirement source"):
            intent_completeness(_state(unknown))


class FusionPolicyTests(unittest.TestCase):
    def test_equal_policy_is_exactly_balanced(self) -> None:
        self.assertEqual(
            EQUAL_RRF_POLICY.choose(IntentState()),
            RouteWeights(bm25=0.5, dense=0.5),
        )
        self.assertEqual(
            EQUAL_RRF_POLICY.choose(_state("answer")),
            RouteWeights(bm25=0.5, dense=0.5),
        )

    def test_adaptive_policy_matches_predeclared_endpoints_and_midpoint(self) -> None:
        cases = (
            (IntentState(), RouteWeights(bm25=0.4, dense=0.6)),
            (
                _state("answer", "initial_tentative"),
                RouteWeights(bm25=0.5, dense=0.5),
            ),
            (
                _state("answer", "override", "initial_explicit"),
                RouteWeights(bm25=0.6, dense=0.4),
            ),
        )
        for state, expected in cases:
            with self.subTest(completeness=intent_completeness(state)):
                actual = COMPLETENESS_ADAPTIVE_RRF_POLICY.choose(state)
                self.assertAlmostEqual(actual.bm25, expected.bm25)
                self.assertAlmostEqual(actual.dense, expected.dense)

    def test_adaptive_policy_is_finite_bounded_normalized_and_monotonic(self) -> None:
        states = (
            IntentState(),
            _state("free_text"),
            _state("answer"),
            _state("answer", "free_text"),
            _state("answer", "override"),
            _state("answer", "override", "initial_explicit"),
            _state("answer", "override", "initial_explicit", "answer"),
        )
        completeness = [intent_completeness(state) for state in states]
        weights = [COMPLETENESS_ADAPTIVE_RRF_POLICY.choose(state) for state in states]

        self.assertTrue(all(math.isfinite(value) for value in completeness))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in completeness))
        self.assertEqual(completeness, sorted(completeness))
        self.assertTrue(
            all(
                math.isfinite(weight.bm25) and math.isfinite(weight.dense)
                for weight in weights
            )
        )
        self.assertTrue(
            all(
                0.0 <= weight.bm25 <= 1.0
                and 0.0 <= weight.dense <= 1.0
                and math.isclose(
                    weight.bm25 + weight.dense,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for weight in weights
            )
        )
        self.assertEqual([weight.bm25 for weight in weights], sorted(weight.bm25 for weight in weights))
        self.assertEqual(
            [weight.dense for weight in weights],
            sorted((weight.dense for weight in weights), reverse=True),
        )

    def test_weight_value_is_immutable(self) -> None:
        weights = RouteWeights(bm25=0.5, dense=0.5)
        with self.assertRaises(FrozenInstanceError):
            weights.bm25 = 0.6  # type: ignore[misc]

    def test_policy_is_an_immutable_enum(self) -> None:
        self.assertIsInstance(EQUAL_RRF_POLICY, FusionPolicy)
        self.assertIsInstance(COMPLETENESS_ADAPTIVE_RRF_POLICY, FusionPolicy)
        with self.assertRaises(AttributeError):
            EQUAL_RRF_POLICY.value = "changed"  # type: ignore[misc]

    def test_weight_value_rejects_nonfinite_unbounded_or_unnormalized_values(self) -> None:
        invalid = (
            (math.nan, math.nan),
            (math.inf, -math.inf),
            (-0.1, 1.1),
            (0.0, 1.0),
            (0.4, 0.5),
        )
        for bm25, dense in invalid:
            with self.subTest(bm25=bm25, dense=dense):
                with self.assertRaises(ValueError):
                    RouteWeights(bm25=bm25, dense=dense)


if __name__ == "__main__":
    unittest.main()
