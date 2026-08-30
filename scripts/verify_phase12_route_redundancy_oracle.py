from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections.abc import Sequence

from conversational_search.intent import IntentState
from conversational_search.profiles import (
    DISABLED_PROFILE_POLICY,
    NEUTRAL_PROFILE_PRIOR,
)
from conversational_search.ranking import (
    CandidateDocument,
    RouteRedundancyStatus,
    _redundancy_corrected_route_scores,
    rerank_stage_a_with_profile,
    rerank_stage_a_with_profile_and_route_redundancy,
    route_redundancy_coefficient,
)
from conversational_search.strategy import RouteWeights


SEED = 120260830
RANDOM_CASES = 30_000
EXPECTED_SHA256 = "5fa70591a4761a1e4bde33ad246c6288938284e1ed1cc7137805a03eefadf1c8"
UNIVERSE = ("P000", "P001", "P002", "P003")
WEIGHTS = (
    RouteWeights(bm25=0.4, dense=0.6),
    RouteWeights(bm25=0.5, dense=0.5),
    RouteWeights(bm25=0.6, dense=0.4),
)
EXACT_STATUSES = frozenset(
    {
        RouteRedundancyStatus.EMPTY,
        RouteRedundancyStatus.SINGLE_ROUTE,
        RouteRedundancyStatus.DISJOINT,
        RouteRedundancyStatus.IDENTICAL,
    }
)


def _ordered_routes(universe: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        route
        for length in range(len(universe) + 1)
        for route in itertools.permutations(universe, length)
    )


def _additive_fused(
    bm25: Sequence[str],
    dense: Sequence[str],
    weights: RouteWeights,
) -> tuple[str, ...]:
    union = tuple(dict.fromkeys((*bm25, *dense)))
    positions = {value: index for index, value in enumerate(union)}
    bm25_ranks = {value: rank for rank, value in enumerate(bm25, 1)}
    dense_ranks = {value: rank for rank, value in enumerate(dense, 1)}
    scores = {
        value: (
            weights.bm25 / (60 + bm25_ranks[value])
            if value in bm25_ranks
            else 0.0
        )
        + (
            weights.dense / (60 + dense_ranks[value])
            if value in dense_ranks
            else 0.0
        )
        for value in union
    }
    return tuple(
        sorted(union, key=lambda value: (-scores[value], positions[value]))
    )


def _assert_equations(
    bm25: tuple[str, ...],
    dense: tuple[str, ...],
    fused: tuple[str, ...],
    weights: RouteWeights,
) -> tuple[RouteRedundancyStatus, tuple[str, ...], tuple[str, ...], str]:
    coefficient = route_redundancy_coefficient(bm25, dense)
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise AssertionError("coefficient escaped [0, 1]")
    scores = _redundancy_corrected_route_scores(
        bm25,
        dense,
        fused,
        weights,
    )
    replay = _redundancy_corrected_route_scores(
        bm25,
        dense,
        fused,
        weights,
    )
    swapped = _redundancy_corrected_route_scores(
        dense,
        bm25,
        fused,
        RouteWeights(bm25=weights.dense, dense=weights.bm25),
    )
    if scores != replay or scores != swapped:
        raise AssertionError("candidate route scores are not exact and symmetric")
    if set(scores) != set(fused):
        raise AssertionError("candidate score domain drifted from fused IDs")

    bm25_ranks = {value: rank for rank, value in enumerate(bm25, 1)}
    dense_ranks = {value: rank for rank, value in enumerate(dense, 1)}
    for value, score in scores.items():
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
        expected = x_value + y_value - coefficient * min(x_value, y_value)
        if score != expected:
            raise AssertionError("candidate equation drifted from frozen reference")
        if not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise AssertionError("candidate score escaped (0, 1]")
        lower = max(x_value, y_value)
        upper = x_value + y_value
        lower_holds = score >= lower or math.isclose(
            score,
            lower,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        upper_holds = score <= upper or math.isclose(
            score,
            upper,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        if not lower_holds or not upper_holds:
            raise AssertionError("candidate score violated submodular bounds")

    documents = tuple(CandidateDocument(value, "") for value in fused)
    arguments = {
        "bm25_ids": bm25,
        "dense_ids": dense,
        "fused_ids": fused,
        "route_weights": weights,
        "profile_prior": NEUTRAL_PROFILE_PRIOR,
        "profile_policy": DISABLED_PROFILE_POLICY,
    }
    baseline = rerank_stage_a_with_profile(
        IntentState(),
        documents,
        **arguments,
    )
    candidate = rerank_stage_a_with_profile_and_route_redundancy(
        IntentState(),
        documents,
        **arguments,
    )
    candidate_replay = rerank_stage_a_with_profile_and_route_redundancy(
        IntentState(),
        documents,
        **arguments,
    )
    if candidate != candidate_replay:
        raise AssertionError("candidate ranking replay drifted")
    if set(candidate.ranking.ranked_ids) != set(fused):
        raise AssertionError("candidate ranking is not a complete permutation")
    if candidate.status in EXACT_STATUSES and candidate.ranking != baseline.ranking:
        raise AssertionError("exact candidate cell drifted from Phase 9")
    if candidate.status is RouteRedundancyStatus.SCORING_FALLBACK:
        raise AssertionError("valid oracle input reached scoring fallback")

    score_digest = hashlib.sha256()
    for value in fused:
        score_digest.update(value.encode("ascii"))
        score_digest.update(b"\0")
        score_digest.update(scores[value].hex().encode("ascii"))
        score_digest.update(b"\0")
    return (
        candidate.status,
        baseline.ranking.ranked_ids,
        candidate.ranking.ranked_ids,
        score_digest.hexdigest(),
    )


def verify() -> dict[str, object]:
    digest = hashlib.sha256()
    case_count = 0
    routes = _ordered_routes(UNIVERSE)

    def consume(
        bm25: tuple[str, ...],
        dense: tuple[str, ...],
        weights: RouteWeights,
    ) -> None:
        nonlocal case_count
        fused = _additive_fused(bm25, dense, weights)
        status, baseline, candidate, score_digest = _assert_equations(
            bm25,
            dense,
            fused,
            weights,
        )
        payload = {
            "case": case_count,
            "bm25": bm25,
            "dense": dense,
            "fused": fused,
            "weights": (weights.bm25.hex(), weights.dense.hex()),
            "status": status.value,
            "baseline": baseline,
            "candidate": candidate,
            "scores": score_digest,
        }
        digest.update(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
        digest.update(b"\n")
        case_count += 1

    for weights in WEIGHTS:
        for bm25 in routes:
            for dense in routes:
                consume(bm25, dense, weights)

    exhaustive_cases = case_count
    rng = random.Random(SEED)
    for random_index in range(RANDOM_CASES):
        size = 100 if random_index == 0 else rng.randrange(0, 21)
        identifiers = tuple(f"R{index:03d}" for index in range(size))
        memberships = [rng.randrange(1, 4) for _ in identifiers]
        bm25_values = [
            value
            for value, membership in zip(identifiers, memberships)
            if membership & 1
        ]
        dense_values = [
            value
            for value, membership in zip(identifiers, memberships)
            if membership & 2
        ]
        rng.shuffle(bm25_values)
        rng.shuffle(dense_values)
        consume(
            tuple(bm25_values),
            tuple(dense_values),
            rng.choice(WEIGHTS),
        )

    expected_exhaustive = len(WEIGHTS) * len(routes) * len(routes)
    if exhaustive_cases != expected_exhaustive or case_count != (
        expected_exhaustive + RANDOM_CASES
    ):
        raise AssertionError("oracle case accounting drifted")
    observed_digest = digest.hexdigest()
    if observed_digest != EXPECTED_SHA256:
        raise AssertionError("Phase 12 oracle digest drifted")
    return {
        "status": "ok",
        "seed": SEED,
        "exhaustive_cases": exhaustive_cases,
        "random_cases": RANDOM_CASES,
        "cases": case_count,
        "digest": observed_digest,
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
