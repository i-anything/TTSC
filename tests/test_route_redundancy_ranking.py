from __future__ import annotations

import math
import random
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from conversational_search.intent import IntentState, Requirement
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    NEUTRAL_PROFILE_PRIOR,
    ProductTheme,
    ProfilePrior,
)
from conversational_search.ranking import (
    ROUTE_REDUNDANCY_CORRECTED_RANKING_POLICY,
    CandidateDocument,
    ProfileResidualStatus,
    RankingPolicy,
    RouteRedundancyRankingResult,
    RouteRedundancyStatus,
    _redundancy_corrected_route_scores,
    rerank_stage_a_with_profile,
    rerank_stage_a_with_profile_and_route_redundancy,
    route_redundancy_coefficient,
)
from conversational_search.strategy import RouteWeights


_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)


def _documents(identifiers: tuple[str, ...]) -> tuple[CandidateDocument, ...]:
    return tuple(
        CandidateDocument(parent_asin, f"plain synthetic item {index}")
        for index, parent_asin in enumerate(identifiers)
    )


def _rankings(
    bm25: tuple[str, ...],
    dense: tuple[str, ...],
    fused: tuple[str, ...],
    *,
    state: IntentState = IntentState(),
    weights: RouteWeights = _WEIGHTS,
    profile_prior: ProfilePrior = NEUTRAL_PROFILE_PRIOR,
    profile_policy=DISABLED_PROFILE_POLICY,
):
    documents = _documents(fused)
    baseline = rerank_stage_a_with_profile(
        state,
        documents,
        bm25_ids=bm25,
        dense_ids=dense,
        fused_ids=fused,
        route_weights=weights,
        profile_prior=profile_prior,
        profile_policy=profile_policy,
    )
    candidate = rerank_stage_a_with_profile_and_route_redundancy(
        state,
        documents,
        bm25_ids=bm25,
        dense_ids=dense,
        fused_ids=fused,
        route_weights=weights,
        profile_prior=profile_prior,
        profile_policy=profile_policy,
    )
    return baseline, candidate


class RouteRedundancyEquationTests(unittest.TestCase):
    def test_policy_result_and_status_contracts_are_immutable(self) -> None:
        self.assertIs(
            ROUTE_REDUNDANCY_CORRECTED_RANKING_POLICY,
            RankingPolicy.ROUTE_REDUNDANCY_CORRECTED,
        )
        _baseline, result = _rankings(("A",), ("A",), ("A",))
        self.assertIsInstance(result, RouteRedundancyRankingResult)
        self.assertIs(result.status, RouteRedundancyStatus.IDENTICAL)
        with self.assertRaises(FrozenInstanceError):
            result.status = RouteRedundancyStatus.APPLIED  # type: ignore[misc]

    def test_coefficient_is_bounded_and_matches_overlap_coefficient(self) -> None:
        cases = (
            ((), (), 0.0),
            (("A",), (), 0.0),
            (("A", "B"), ("C", "D"), 0.0),
            (("A", "B"), ("A", "C"), 0.5),
            (("A", "B"), ("B", "A", "C"), 1.0),
        )
        for bm25, dense, expected in cases:
            with self.subTest(bm25=bm25, dense=dense):
                actual = route_redundancy_coefficient(bm25, dense)
                self.assertEqual(actual, expected)
                self.assertTrue(math.isfinite(actual))
                self.assertTrue(0.0 <= actual <= 1.0)

        with self.assertRaisesRegex(ValueError, "duplicate"):
            route_redundancy_coefficient(("A", "A"), ("A",))
        with self.assertRaisesRegex(ValueError, "at most 100"):
            route_redundancy_coefficient(
                tuple(f"B{index:03d}" for index in range(101)),
                (),
            )

    def test_frozen_equation_matches_direct_reference_and_is_symmetric(self) -> None:
        bm25 = ("A", "B", "C", "D")
        dense = ("C", "E", "A", "F")
        fused = ("A", "C", "B", "E", "D", "F")
        weights = RouteWeights(bm25=0.4, dense=0.6)

        actual = _redundancy_corrected_route_scores(
            bm25,
            dense,
            fused,
            weights,
        )
        coefficient = 2 / 4
        bm25_ranks = {value: rank for rank, value in enumerate(bm25, 1)}
        dense_ranks = {value: rank for rank, value in enumerate(dense, 1)}
        expected: dict[str, float] = {}
        for value in fused:
            x_value = (
                weights.bm25 * 61 / (60 + bm25_ranks[value])
                if value in bm25_ranks
                else 0.0
            )
            y_value = (
                weights.dense * 61 / (60 + dense_ranks[value])
                if value in dense_ranks
                else 0.0
            )
            expected[value] = (
                x_value
                + y_value
                - coefficient * min(x_value, y_value)
            )

        self.assertEqual(actual, expected)
        self.assertTrue(all(0.0 < value <= 1.0 for value in actual.values()))
        swapped = _redundancy_corrected_route_scores(
            dense,
            bm25,
            fused,
            RouteWeights(bm25=0.6, dense=0.4),
        )
        self.assertEqual(actual, swapped)

    def test_fixed_route_evidence_is_monotonic_and_subadditive(self) -> None:
        bm25 = ("A", "B", "C")
        dense = ("C", "B", "D")
        fused = ("A", "B", "C", "D")
        scores = _redundancy_corrected_route_scores(
            bm25,
            dense,
            fused,
            _WEIGHTS,
        )
        coefficient = route_redundancy_coefficient(bm25, dense)
        self.assertEqual(coefficient, 2 / 3)
        for value in fused:
            bm25_rank = bm25.index(value) + 1 if value in bm25 else None
            dense_rank = dense.index(value) + 1 if value in dense else None
            x_value = 0.5 * 61 / (60 + bm25_rank) if bm25_rank else 0.0
            y_value = 0.5 * 61 / (60 + dense_rank) if dense_rank else 0.0
            self.assertGreaterEqual(scores[value], max(x_value, y_value))
            self.assertLessEqual(scores[value], x_value + y_value)
        self.assertGreater(scores["C"], scores["B"])

    def test_exact_baseline_cells_return_the_exact_phase9_result(self) -> None:
        cases = (
            ((), (), (), RouteRedundancyStatus.EMPTY),
            (("A", "B"), (), ("A", "B"), RouteRedundancyStatus.SINGLE_ROUTE),
            ((), ("A", "B"), ("A", "B"), RouteRedundancyStatus.SINGLE_ROUTE),
            (
                ("A", "B"),
                ("C", "D"),
                ("A", "C", "B", "D"),
                RouteRedundancyStatus.DISJOINT,
            ),
            (
                ("A", "B", "C"),
                ("A", "B", "C"),
                ("A", "B", "C"),
                RouteRedundancyStatus.IDENTICAL,
            ),
        )
        for bm25, dense, fused, expected_status in cases:
            with self.subTest(status=expected_status.value):
                baseline, candidate = _rankings(bm25, dense, fused)
                self.assertEqual(candidate.ranking, baseline.ranking)
                self.assertIs(candidate.status, expected_status)
                self.assertIs(
                    candidate.profile_status,
                    baseline.status,
                )

    def test_partial_overlap_changes_only_order_not_membership_or_stage_a_shape(self) -> None:
        bm25 = ("A", "B", "C", "D")
        dense = ("D", "C", "E", "F")
        fused = ("A", "D", "B", "C", "E", "F")
        state = IntentState(
            category="synthetic footwear",
            requirements=(
                Requirement("plain", "answer", 2, "feature"),
            ),
        )
        baseline, candidate = _rankings(
            bm25,
            dense,
            fused,
            state=state,
            weights=RouteWeights(bm25=0.6, dense=0.4),
        )

        self.assertIs(candidate.status, RouteRedundancyStatus.APPLIED)
        self.assertEqual(set(candidate.ranking.ranked_ids), set(fused))
        self.assertEqual(len(candidate.ranking.ranked_ids), len(fused))
        self.assertEqual(candidate.ranking.trace.input_ids, fused)
        self.assertEqual(
            candidate.ranking.trace.beta,
            baseline.ranking.trace.beta,
        )
        self.assertEqual(
            candidate.ranking.trace.observable_clause_count,
            baseline.ranking.trace.observable_clause_count,
        )

    def test_profile_composition_preserves_exact_profile_contract(self) -> None:
        bm25 = ("A", "B", "C")
        dense = ("C", "B", "D")
        fused = ("A", "B", "C", "D")
        documents = (
            CandidateDocument("A", "ordinary shoe"),
            CandidateDocument("B", "comfortable cushioned shoe"),
            CandidateDocument("C", "ordinary boot"),
            CandidateDocument("D", "padded comfort sandal"),
        )
        result = rerank_stage_a_with_profile_and_route_redundancy(
            IntentState(),
            documents,
            bm25_ids=bm25,
            dense_ids=dense,
            fused_ids=fused,
            route_weights=_WEIGHTS,
            profile_prior=ProfilePrior(ProductTheme.COMFORT),
            profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        )

        self.assertIs(result.status, RouteRedundancyStatus.APPLIED)
        self.assertIs(result.profile_status, ProfileResidualStatus.APPLIED)
        self.assertEqual(result.requested_theme_count, 1)
        self.assertEqual(result.represented_theme_count, 1)
        self.assertEqual(set(result.ranking.ranked_ids), set(fused))

    def test_candidate_only_fault_fails_closed_to_exact_phase9(self) -> None:
        bm25 = ("A", "B", "C")
        dense = ("C", "B", "D")
        fused = ("A", "B", "C", "D")
        baseline = rerank_stage_a_with_profile(
            IntentState(),
            _documents(fused),
            bm25_ids=bm25,
            dense_ids=dense,
            fused_ids=fused,
            route_weights=_WEIGHTS,
            profile_prior=NEUTRAL_PROFILE_PRIOR,
            profile_policy=DISABLED_PROFILE_POLICY,
        )
        with patch(
            "conversational_search.ranking."
            "_redundancy_corrected_route_scores",
            side_effect=RuntimeError("synthetic candidate fault"),
        ):
            result = rerank_stage_a_with_profile_and_route_redundancy(
                IntentState(),
                _documents(fused),
                bm25_ids=bm25,
                dense_ids=dense,
                fused_ids=fused,
                route_weights=_WEIGHTS,
                profile_prior=NEUTRAL_PROFILE_PRIOR,
                profile_policy=DISABLED_PROFILE_POLICY,
            )

        self.assertIs(result.status, RouteRedundancyStatus.SCORING_FALLBACK)
        self.assertEqual(result.ranking, baseline.ranking)

    def test_random_valid_scores_are_bounded_symmetric_and_deterministic(self) -> None:
        rng = random.Random(0x1200)
        weights = (
            RouteWeights(bm25=0.4, dense=0.6),
            RouteWeights(bm25=0.5, dense=0.5),
            RouteWeights(bm25=0.6, dense=0.4),
        )
        for case_index in range(2_000):
            size = 100 if case_index == 0 else rng.randrange(1, 16)
            identifiers = tuple(f"P{index:03d}" for index in range(size))
            memberships = [rng.randrange(1, 4) for _ in identifiers]
            bm25 = [
                value
                for value, membership in zip(identifiers, memberships)
                if membership & 1
            ]
            dense = [
                value
                for value, membership in zip(identifiers, memberships)
                if membership & 2
            ]
            rng.shuffle(bm25)
            rng.shuffle(dense)
            fused = list(identifiers)
            rng.shuffle(fused)
            route_weights = rng.choice(weights)
            actual = _redundancy_corrected_route_scores(
                tuple(bm25),
                tuple(dense),
                tuple(fused),
                route_weights,
            )
            replay = _redundancy_corrected_route_scores(
                tuple(bm25),
                tuple(dense),
                tuple(fused),
                route_weights,
            )
            swapped = _redundancy_corrected_route_scores(
                tuple(dense),
                tuple(bm25),
                tuple(fused),
                RouteWeights(
                    bm25=route_weights.dense,
                    dense=route_weights.bm25,
                ),
            )
            self.assertEqual(actual, replay)
            self.assertEqual(actual, swapped)
            self.assertEqual(set(actual), set(identifiers))
            self.assertTrue(
                all(math.isfinite(value) and 0.0 < value <= 1.0 for value in actual.values())
            )


if __name__ == "__main__":
    unittest.main()
