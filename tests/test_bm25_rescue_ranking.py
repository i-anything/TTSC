from __future__ import annotations

import random
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from conversational_search.intent import IntentState, Requirement
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    ProductTheme,
    ProfilePrior,
)
from conversational_search.ranking import (
    COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
    Bm25RescueRankingResult,
    Bm25RescueStatus,
    CandidateDocument,
    ProfileResidualStatus,
    RankingPolicy,
    RankingResult,
    RankingTrace,
    _Bm25RescueComputation,
    _StageAComputation,
    _apply_bm25_rescue,
    rerank_stage_a_with_profile,
    rerank_stage_a_with_profile_and_bm25_rescue,
)
from conversational_search.strategy import RouteWeights


_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)
_COMFORT = ProfilePrior(ProductTheme.COMFORT)


def _computation(
    bm25_order: tuple[int, ...],
    dense_order: tuple[int, ...],
    satisfaction: tuple[float, ...],
    completeness: float,
    route_weights: RouteWeights = _WEIGHTS,
) -> _Bm25RescueComputation:
    identifiers = tuple(f"P{index:03d}" for index in range(len(satisfaction)))
    if set(bm25_order) | set(dense_order) != set(range(len(identifiers))):
        raise ValueError("synthetic routes must cover every identifier")
    bm25_ids = tuple(identifiers[index] for index in bm25_order)
    dense_ids = tuple(identifiers[index] for index in dense_order)
    bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, 1)}
    dense_ranks = {index: rank for rank, index in enumerate(dense_order, 1)}
    raw_rrf = tuple(
        (
            route_weights.bm25 / (60.0 + bm25_ranks[index])
            if index in bm25_ranks
            else 0.0
        )
        + (
            route_weights.dense / (60.0 + dense_ranks[index])
            if index in dense_ranks
            else 0.0
        )
        for index in range(len(identifiers))
    )
    maximum_rrf = max(raw_rrf, default=1.0)
    rrf = tuple(score / maximum_rrf for score in raw_rrf)
    beta = 0.20 + 0.25 * completeness
    base_scores = tuple(
        (1.0 - beta) * retrieval + beta * clause
        for retrieval, clause in zip(rrf, satisfaction)
    )
    ranked_ids = tuple(
        identifiers[index]
        for index in sorted(
            range(len(identifiers)),
            key=lambda index: (-base_scores[index], index),
        )
    )
    trace = RankingTrace(
        input_ids=identifiers,
        output_ids=ranked_ids,
        beta=beta,
        observable_clause_count=0,
    )
    phase9 = _StageAComputation(
        RankingResult(ranked_ids, trace),
        base_scores,
        tuple(("plain",) for _ in identifiers),
    )
    return _Bm25RescueComputation(
        phase9=phase9,
        completeness=completeness,
        bm25_ids=bm25_ids,
        dense_ids=dense_ids,
        route_weights=route_weights,
    )


def _reference(
    computation: _Bm25RescueComputation,
) -> tuple[tuple[str, ...], tuple[float, ...], Bm25RescueStatus]:
    phase9 = computation.phase9
    base = phase9.ranking
    if computation.completeness == 0.0:
        return (
            base.ranked_ids,
            phase9.base_scores,
            Bm25RescueStatus.ZERO_COMPLETENESS,
        )
    if not computation.bm25_ids:
        return base.ranked_ids, phase9.base_scores, Bm25RescueStatus.EMPTY_BM25
    bm25_ranks = {
        parent_asin: rank
        for rank, parent_asin in enumerate(computation.bm25_ids, 1)
    }
    dense_ranks = {
        parent_asin: rank
        for rank, parent_asin in enumerate(computation.dense_ids, 1)
    }
    candidate_ids = base.trace.input_ids
    raw_rrf = tuple(
        (
            computation.route_weights.bm25 / (60.0 + bm25_ranks[parent_asin])
            if parent_asin in bm25_ranks
            else 0.0
        )
        + (
            computation.route_weights.dense / (60.0 + dense_ranks[parent_asin])
            if parent_asin in dense_ranks
            else 0.0
        )
        for parent_asin in candidate_ids
    )
    maximum_rrf = max(raw_rrf)
    normalized_rrf = tuple(score / maximum_rrf for score in raw_rrf)
    normalized_bm25 = tuple(
        61.0 / (60.0 + bm25_ranks[parent_asin])
        if parent_asin in bm25_ranks
        else 0.0
        for parent_asin in candidate_ids
    )
    uplift = tuple(
        computation.completeness * max(0.0, lexical - hybrid)
        for hybrid, lexical in zip(
            normalized_rrf,
            normalized_bm25,
        )
    )
    if not any(value > 0.0 for value in uplift):
        return (
            base.ranked_ids,
            phase9.base_scores,
            Bm25RescueStatus.NO_POSITIVE_UPLIFT,
        )
    if all(value == uplift[0] for value in uplift[1:]):
        return (
            base.ranked_ids,
            phase9.base_scores,
            Bm25RescueStatus.CONSTANT_UPLIFT,
        )
    scores = tuple(
        score + (1.0 - base.trace.beta) * delta
        for score, delta in zip(phase9.base_scores, uplift)
    )
    positions = {value: index for index, value in enumerate(base.ranked_ids)}
    ranked = tuple(
        value
        for value, _score in sorted(
            zip(base.trace.input_ids, scores),
            key=lambda item: (-item[1], positions[item[0]]),
        )
    )
    if ranked == base.ranked_ids:
        return (
            base.ranked_ids,
            phase9.base_scores,
            Bm25RescueStatus.UNCHANGED_ORDER,
        )
    return ranked, scores, Bm25RescueStatus.REORDERED


class Bm25RescueEquationTests(unittest.TestCase):
    def test_policy_and_result_contract_are_immutable_and_label_free(self) -> None:
        self.assertIs(
            COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            RankingPolicy.COMPLETENESS_BM25_RESCUE,
        )
        result = Bm25RescueRankingResult(
            ranking=RankingResult(
                ("A",),
                RankingTrace(("A",), ("A",), 0.2, 0),
            ),
            status=Bm25RescueStatus.ZERO_COMPLETENESS,
            profile_status=ProfileResidualStatus.NEUTRAL,
            requested_theme_count=0,
            represented_theme_count=0,
        )
        with self.assertRaises(FrozenInstanceError):
            result.status = Bm25RescueStatus.REORDERED  # type: ignore[misc]
        for forbidden in ("target", "ground_truth", "query", "document"):
            self.assertNotIn(forbidden, repr(result).casefold())
        rescue_fields = {field.name for field in fields(_Bm25RescueComputation)}
        self.assertNotIn("normalized_rrf_scores", rescue_fields)
        self.assertNotIn("normalized_bm25_scores", rescue_fields)

    def test_in_range_but_impossible_phase9_scores_fail_closed(self) -> None:
        computation = _computation(
            (0, 1),
            (1, 0),
            (0.0, 0.0),
            1.0,
        )
        impossible = replace(
            computation,
            phase9=replace(computation.phase9, base_scores=(0.1, 0.0)),
        )

        with self.assertRaises(ValueError):
            _apply_bm25_rescue(impossible)

    def test_boundary_outcomes_return_the_exact_input_computation(self) -> None:
        cases = (
            (
                _computation((0, 1), (0, 1), (0.0, 0.0), 0.0),
                Bm25RescueStatus.ZERO_COMPLETENESS,
            ),
            (
                _computation((), (0, 1), (0.0, 0.0), 1.0),
                Bm25RescueStatus.EMPTY_BM25,
            ),
            (
                _computation((0, 1), (1, 0), (0.0, 0.0), 1.0),
                Bm25RescueStatus.NO_POSITIVE_UPLIFT,
            ),
            (
                _computation((0, 1), (), (0.0, 0.0), 1.0),
                Bm25RescueStatus.UNCHANGED_ORDER,
            ),
        )
        for computation, expected_status in cases:
            with self.subTest(status=expected_status.value):
                candidate, status = _apply_bm25_rescue(computation)
                self.assertIs(candidate, computation.phase9)
                self.assertEqual(status, expected_status)

    def test_rescue_equation_reorders_and_preserves_phase9_ties(self) -> None:
        computation = _computation(
            (0, 1, 2),
            (0, 2, 1),
            (0.0, 0.0, 0.0),
            1.0,
            RouteWeights(bm25=0.4, dense=0.6),
        )

        candidate, status = _apply_bm25_rescue(computation)
        expected_ids, expected_scores, expected_status = _reference(computation)

        self.assertEqual(status, expected_status)
        self.assertEqual(status, Bm25RescueStatus.REORDERED)
        self.assertEqual(candidate.ranking.ranked_ids, expected_ids)
        self.assertEqual(candidate.base_scores, expected_scores)
        self.assertEqual(
            candidate.ranking.trace.beta,
            computation.phase9.ranking.trace.beta,
        )

    def test_exact_rescued_score_tie_keeps_phase9_order(self) -> None:
        computation = _computation(
            (1,),
            (0,),
            (0.0, 0.0),
            1.0,
            RouteWeights(bm25=0.4, dense=0.6),
        )
        beta = computation.phase9.ranking.trace.beta
        first_final = (1.0 - beta) * 1.0
        second_final = (
            (1.0 - beta) * (2.0 / 3.0)
            + (1.0 - beta) * (1.0 - 2.0 / 3.0)
        )

        candidate, status = _apply_bm25_rescue(computation)

        self.assertEqual(first_final, second_final)
        self.assertEqual(status, Bm25RescueStatus.UNCHANGED_ORDER)
        self.assertIs(candidate, computation.phase9)
        self.assertEqual(candidate.ranking.ranked_ids, ("P000", "P001"))

    def test_zero_weak_and_strong_satisfaction_cells_match_reference(self) -> None:
        for satisfaction in (0.0, 0.5, 1.0):
            with self.subTest(satisfaction=satisfaction):
                computation = _computation(
                    (0, 1, 2),
                    (0, 2, 1),
                    (satisfaction,) * 3,
                    1.0,
                    RouteWeights(bm25=0.4, dense=0.6),
                )
                candidate, status = _apply_bm25_rescue(computation)
                expected_ids, expected_scores, expected_status = _reference(
                    computation
                )

                self.assertEqual(status, expected_status)
                self.assertEqual(candidate.ranking.ranked_ids, expected_ids)
                self.assertEqual(candidate.base_scores, expected_scores)

    def test_bm25_rank_boundaries_one_two_sixty_one_hundred_and_missing(self) -> None:
        size = 101
        computation = _computation(
            tuple(range(100)),
            tuple(range(100, -1, -1)),
            (0.0,) * size,
            1.0,
            RouteWeights(bm25=0.4, dense=0.6),
        )
        candidate, status = _apply_bm25_rescue(computation)
        expected_ids, expected_scores, expected_status = _reference(computation)

        self.assertEqual(status, expected_status)
        self.assertEqual(candidate.ranking.ranked_ids, expected_ids)
        self.assertEqual(candidate.base_scores, expected_scores)
        bm25_ranks = (1, 2, 60, 100)
        self.assertEqual(
            tuple(61.0 / (60.0 + rank) for rank in bm25_ranks),
            (1.0, 61.0 / 62.0, 61.0 / 120.0, 61.0 / 160.0),
        )
        self.assertNotIn("P100", computation.bm25_ids)

    def test_positive_constant_uplift_is_unreachable_for_valid_routes(self) -> None:
        from itertools import permutations

        for size in range(1, 5):
            identifiers = tuple(range(size))
            for bm25_order in permutations(identifiers):
                for dense_order in permutations(identifiers):
                    computation = _computation(
                        bm25_order,
                        dense_order,
                        (0.0,) * size,
                        1.0,
                    )
                    _candidate, status = _apply_bm25_rescue(computation)
                    self.assertIsNot(status, Bm25RescueStatus.CONSTANT_UPLIFT)

    def test_completeness_levels_scale_the_frozen_formula(self) -> None:
        last_rescue_delta: float | None = None
        for completeness in (
            0.0,
            1.0 / 6.0,
            1.0 / 3.0,
            0.5,
            2.0 / 3.0,
            1.0,
        ):
            computation = _computation(
                (0, 1, 2),
                (0, 2, 1),
                (0.0, 0.0, 0.0),
                completeness,
                RouteWeights(bm25=0.4, dense=0.6),
            )
            candidate, _status = _apply_bm25_rescue(computation)
            reference_ranked, reference_scores, _reference_status = _reference(
                computation
            )
            self.assertEqual(candidate.ranking.ranked_ids, reference_ranked)
            self.assertEqual(candidate.base_scores, reference_scores)
            rescue_delta = (
                reference_scores[1] - computation.phase9.base_scores[1]
            )
            if last_rescue_delta is not None:
                self.assertGreaterEqual(rescue_delta, last_rescue_delta)
            last_rescue_delta = rescue_delta

    def test_ten_thousand_valid_and_two_thousand_malformed_streams(self) -> None:
        rng = random.Random(0xB025)
        valid: list[_Bm25RescueComputation] = []
        completeness_values = (0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0)
        for case_index in range(10_000):
            size = 100 if case_index % 997 == 0 else rng.randrange(0, 13)
            memberships = [rng.randrange(1, 4) for _ in range(size)]
            bm25_route = [
                index
                for index, membership in enumerate(memberships)
                if membership & 1
            ]
            dense_route = [
                index
                for index, membership in enumerate(memberships)
                if membership & 2
            ]
            rng.shuffle(bm25_route)
            rng.shuffle(dense_route)
            bm25_weight = rng.choice((0.4, 0.5, 0.6, 0.7))
            dense_weight = 1.0 - bm25_weight
            satisfaction = tuple(rng.random() for _ in range(size))
            computation = _computation(
                tuple(bm25_route),
                tuple(dense_route),
                satisfaction,
                rng.choice(completeness_values),
                RouteWeights(bm25=bm25_weight, dense=dense_weight),
            )
            valid.append(computation)
            expected_ids, expected_scores, expected_status = _reference(computation)
            candidate, status = _apply_bm25_rescue(computation)
            self.assertEqual(status, expected_status)
            self.assertEqual(candidate.ranking.ranked_ids, expected_ids)
            self.assertEqual(candidate.base_scores, expected_scores)

        for index in range(2_000):
            source_index = index
            source = valid[source_index]
            while (
                source.completeness == 0.0
                or len(source.bm25_ids) < 2
                or len(source.phase9.ranking.trace.input_ids) < 2
            ):
                source_index = (source_index + 1) % len(valid)
                source = valid[source_index]
            mutation = index % 10
            if mutation == 0:
                malformed = replace(source, completeness=float("nan"))
            elif mutation == 1:
                malformed = replace(source, completeness=1.01)
            elif mutation == 2:
                malformed_phase9 = replace(
                    source.phase9,
                    base_scores=source.phase9.base_scores[:-1],
                )
                malformed = replace(source, phase9=malformed_phase9)
            elif mutation == 3:
                route = source.bm25_ids or source.dense_ids
                malformed = replace(source, bm25_ids=route + (route[0],))
            elif mutation == 4:
                omitted = source.bm25_ids[-1]
                malformed = replace(
                    source,
                    bm25_ids=source.bm25_ids[:-1],
                    dense_ids=tuple(
                        parent_asin
                        for parent_asin in source.dense_ids
                        if parent_asin != omitted
                    ),
                )
            elif mutation == 5:
                malformed_phase9 = replace(
                    source.phase9,
                    tokenized_documents=source.phase9.tokenized_documents[:-1],
                )
                malformed = replace(source, phase9=malformed_phase9)
            elif mutation == 6:
                malformed_trace = replace(
                    source.phase9.ranking.trace,
                    beta=0.19,
                )
                malformed_ranking = replace(
                    source.phase9.ranking,
                    trace=malformed_trace,
                )
                malformed = replace(
                    source,
                    phase9=replace(source.phase9, ranking=malformed_ranking),
                )
            elif mutation == 7:
                malformed_trace = replace(
                    source.phase9.ranking.trace,
                    output_ids=(),
                )
                malformed_ranking = replace(
                    source.phase9.ranking,
                    trace=malformed_trace,
                )
                malformed = replace(
                    source,
                    phase9=replace(source.phase9, ranking=malformed_ranking),
                )
            elif mutation == 8:
                inconsistent_scores = (0.0,) + source.phase9.base_scores[1:]
                malformed = replace(
                    source,
                    phase9=replace(
                        source.phase9,
                        base_scores=inconsistent_scores,
                    ),
                )
            else:
                reversed_ids = tuple(
                    reversed(source.phase9.ranking.ranked_ids)
                )
                if len(reversed_ids) < 2:
                    source = _computation((0, 1), (0, 1), (0.0, 1.0), 1.0)
                    reversed_ids = tuple(
                        reversed(source.phase9.ranking.ranked_ids)
                    )
                malformed_ranking = replace(
                    source.phase9.ranking,
                    ranked_ids=reversed_ids,
                    trace=replace(
                        source.phase9.ranking.trace,
                        output_ids=reversed_ids,
                    ),
                )
                malformed = replace(
                    source,
                    phase9=replace(source.phase9, ranking=malformed_ranking),
                )
            with self.assertRaises(
                (TypeError, ValueError),
                msg=f"malformed mutation {mutation} at case {index}",
            ):
                _apply_bm25_rescue(malformed)


class Bm25RescueCompositionTests(unittest.TestCase):
    def test_zero_completeness_is_exact_phase9_including_profile_residual(self) -> None:
        state = IntentState(category="office chair")
        documents = (
            CandidateDocument("BASE", "plain everyday item"),
            CandidateDocument("PROFILE", "ergonomic cushioned item"),
        )
        identifiers = tuple(document.parent_asin for document in documents)
        arguments = {
            "bm25_ids": identifiers,
            "dense_ids": identifiers,
            "fused_ids": identifiers,
            "route_weights": _WEIGHTS,
            "profile_prior": _COMFORT,
            "profile_policy": BOUNDED_RESIDUAL_PROFILE_POLICY,
        }
        phase9 = rerank_stage_a_with_profile(state, documents, **arguments)

        phase10 = rerank_stage_a_with_profile_and_bm25_rescue(
            state,
            documents,
            **arguments,
        )

        self.assertEqual(phase10.ranking, phase9.ranking)
        self.assertEqual(phase10.status, Bm25RescueStatus.ZERO_COMPLETENESS)
        self.assertEqual(phase10.profile_status, phase9.status)
        self.assertEqual(phase10.requested_theme_count, phase9.requested_theme_count)
        self.assertEqual(
            phase10.represented_theme_count,
            phase9.represented_theme_count,
        )

    def test_one_stage_a_computation_and_rescue_precedes_profile(self) -> None:
        state = IntentState(
            requirements=tuple(
                Requirement(f"feature{index}", "answer", index + 1, "feature")
                for index in range(3)
            )
        )
        computation = _computation(
            (0, 1, 2),
            (0, 2, 1),
            (0.0, 0.0, 0.0),
            1.0,
            RouteWeights(bm25=0.4, dense=0.6),
        )
        documents = tuple(
            CandidateDocument(parent_asin, "plain")
            for parent_asin in computation.phase9.ranking.trace.input_ids
        )
        with patch(
            "conversational_search.ranking._compute_stage_a",
            return_value=computation.phase9,
        ) as compute:
            result = rerank_stage_a_with_profile_and_bm25_rescue(
                state,
                documents,
                bm25_ids=computation.bm25_ids,
                dense_ids=computation.dense_ids,
                fused_ids=computation.phase9.ranking.trace.input_ids,
                route_weights=computation.route_weights,
                profile_prior=_COMFORT,
                profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            )

        compute.assert_called_once()
        self.assertEqual(result.status, Bm25RescueStatus.REORDERED)
        self.assertEqual(
            result.profile_status,
            ProfileResidualStatus.ACTIVE_REQUIREMENTS,
        )
        self.assertEqual(result.ranking.ranked_ids, ("P000", "P001", "P002"))

    def test_rescue_fault_returns_exact_phase9_with_noncacheable_status(self) -> None:
        state = IntentState(category="office chair")
        documents = (
            CandidateDocument("BASE", "plain everyday item"),
            CandidateDocument("PROFILE", "ergonomic cushioned item"),
        )
        identifiers = tuple(document.parent_asin for document in documents)
        arguments = {
            "bm25_ids": identifiers,
            "dense_ids": identifiers,
            "fused_ids": identifiers,
            "route_weights": _WEIGHTS,
            "profile_prior": _COMFORT,
            "profile_policy": BOUNDED_RESIDUAL_PROFILE_POLICY,
        }
        phase9 = rerank_stage_a_with_profile(state, documents, **arguments)
        with patch(
            "conversational_search.ranking._apply_bm25_rescue",
            side_effect=RuntimeError("synthetic rescue fault"),
        ):
            phase10 = rerank_stage_a_with_profile_and_bm25_rescue(
                state,
                documents,
                **arguments,
            )

        self.assertEqual(phase10.ranking, phase9.ranking)
        self.assertEqual(phase10.profile_status, phase9.status)
        self.assertEqual(phase10.status, Bm25RescueStatus.SCORING_FALLBACK)

    def test_profile_fault_discards_a_successful_rescue(self) -> None:
        state = IntentState(
            requirements=tuple(
                Requirement(f"feature{index}", "answer", index + 1, "feature")
                for index in range(3)
            )
        )
        computation = _computation(
            (0, 1, 2),
            (0, 2, 1),
            (0.0, 0.0, 0.0),
            1.0,
            RouteWeights(bm25=0.4, dense=0.6),
        )
        with patch(
            "conversational_search.ranking._compute_stage_a",
            return_value=computation.phase9,
        ):
            result = rerank_stage_a_with_profile_and_bm25_rescue(
                state,
                (),
                bm25_ids=computation.bm25_ids,
                dense_ids=computation.dense_ids,
                fused_ids=computation.phase9.ranking.trace.input_ids,
                route_weights=computation.route_weights,
                profile_prior=_COMFORT,
                profile_policy=object(),  # type: ignore[arg-type]
            )

        self.assertEqual(result.status, Bm25RescueStatus.SCORING_FALLBACK)
        self.assertEqual(result.profile_status, ProfileResidualStatus.SCORING_FALLBACK)
        self.assertEqual(result.ranking, computation.phase9.ranking)


if __name__ == "__main__":
    unittest.main()
