from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.ranking import (
    ROUTE_REDUNDANCY_CORRECTED_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
)
from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights
from tests.test_service import CacheableRecordingRetriever


_A = "B000000001"
_B = "B000000002"
_FUSED = (_A, _B)
_BM25 = (_A,)
_DENSE = (_B, _A)
_BROWSING = "I'm looking for Shoes, but I'm still exploring."
_NO_FEATURE = "I don't have a preference for feature; please use your judgment."
_HEALTH_KEYS = {
    "policy",
    "attempts",
    "empty_exact_baseline",
    "single_route_exact_baseline",
    "disjoint_exact_baseline",
    "identical_order_exact_baseline",
    "correction_applied",
    "validation_or_scoring_fallbacks",
}


class PartialOverlapRetriever(CacheableRecordingRetriever):
    def __init__(self) -> None:
        super().__init__(
            _FUSED,
            fused_ids=_FUSED,
            documents={_A: "ordinary item", _B: "plain object"},
        )

    def search_with_trace(
        self,
        dense_query: str,
        lexical_query: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> RetrievalResult:
        result = super().search_with_trace(
            dense_query,
            lexical_query,
            top_k,
            route_weights=route_weights,
        )
        if not isinstance(result, RetrievalResult):
            raise TypeError("test retriever must return RetrievalResult")
        trace = result.trace
        return RetrievalResult(
            recommendations=_FUSED,
            trace=RetrievalTrace(
                bm25_ids=_BM25,
                dense_ids=_DENSE,
                fused_ids=_FUSED,
                bm25_status=(
                    trace.bm25_status
                    if self.bm25_status_override is None
                    else self.bm25_status_override
                ),
                dense_status=(
                    trace.dense_status
                    if self.dense_status_override is None
                    else self.dense_status_override
                ),
                used_fallback=(
                    trace.used_fallback
                    if self.used_fallback_override is None
                    else self.used_fallback_override
                ),
            ),
        )


def _agent(
    retriever: object,
    *,
    candidate: bool,
) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        "unused.jsonl",
        retriever=retriever,
        ranking_policy=(
            ROUTE_REDUNDANCY_CORRECTED_RANKING_POLICY
            if candidate
            else STAGE_A_RANKING_POLICY
        ),
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )


class ServiceRouteRedundancyTests(unittest.TestCase):
    def test_candidate_reorders_partial_overlap_and_exact_reuse_adds_no_work(self) -> None:
        candidate_retriever = PartialOverlapRetriever()
        baseline_retriever = PartialOverlapRetriever()
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})

        candidate_first = candidate.respond("candidate", _BROWSING, 1, 2)
        baseline_first = baseline.respond("baseline", _BROWSING, 1, 2)
        candidate_second = candidate.respond("candidate", _NO_FEATURE, 2, 2)

        self.assertEqual(
            candidate_first["recommendations"],
            [{"parent_asin": _B}, {"parent_asin": _A}],
        )
        self.assertEqual(
            baseline_first["recommendations"],
            [{"parent_asin": _A}, {"parent_asin": _B}],
        )
        self.assertEqual(
            candidate_second["recommendations"],
            candidate_first["recommendations"],
        )
        self.assertEqual(len(candidate_retriever.calls), 1)
        self.assertEqual(len(candidate_retriever.document_calls), 1)
        self.assertEqual(candidate.orchestration_health["hits"], 1)
        self.assertEqual(candidate.ranking_health["attempts"], 1)
        self.assertEqual(candidate.route_redundancy_health["attempts"], 1)
        self.assertEqual(
            candidate.route_redundancy_health["correction_applied"],
            1,
        )

    def test_identical_routes_are_exact_phase9_and_cache_normally(self) -> None:
        candidate_retriever = CacheableRecordingRetriever(
            (_A, _B),
            fused_ids=(_A, _B),
        )
        baseline_retriever = CacheableRecordingRetriever(
            (_A, _B),
            fused_ids=(_A, _B),
        )
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})

        candidate_first = candidate.respond("candidate", _BROWSING, 1, 2)
        baseline_first = baseline.respond("baseline", _BROWSING, 1, 2)
        candidate.respond("candidate", _NO_FEATURE, 2, 2)
        baseline.respond("baseline", _NO_FEATURE, 2, 2)

        self.assertEqual(candidate_first, baseline_first)
        self.assertEqual(candidate_retriever.calls, baseline_retriever.calls)
        self.assertEqual(
            candidate_retriever.document_calls,
            baseline_retriever.document_calls,
        )
        self.assertEqual(candidate.orchestration_health["hits"], 1)
        self.assertEqual(baseline.orchestration_health["hits"], 1)
        self.assertEqual(
            candidate.route_redundancy_health[
                "identical_order_exact_baseline"
            ],
            1,
        )

    def test_route_fault_returns_exact_phase9_and_is_not_cached(self) -> None:
        candidate_retriever = PartialOverlapRetriever()
        baseline_retriever = PartialOverlapRetriever()
        candidate_retriever.bm25_status_override = "error"
        baseline_retriever.bm25_status_override = "error"
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})

        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_route_redundancy",
            side_effect=AssertionError("faulted route entered candidate"),
        ) as candidate_ranker:
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING,
                1,
                2,
            )
        baseline_response = baseline.respond("baseline", _BROWSING, 1, 2)

        candidate_ranker.assert_not_called()
        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(
            candidate.route_redundancy_health[
                "validation_or_scoring_fallbacks"
            ],
            1,
        )

    def test_unexpected_candidate_fault_returns_exact_phase9_and_never_caches(self) -> None:
        candidate_retriever = PartialOverlapRetriever()
        baseline = _agent(PartialOverlapRetriever(), candidate=False)
        candidate = _agent(candidate_retriever, candidate=True)
        baseline.reset("baseline", {})
        candidate.reset("candidate", {})
        baseline_response = baseline.respond("baseline", _BROWSING, 1, 2)

        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_route_redundancy",
            side_effect=RuntimeError("synthetic candidate fault"),
        ):
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING,
                1,
                2,
            )

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(
            candidate.route_redundancy_health[
                "validation_or_scoring_fallbacks"
            ],
            1,
        )
        candidate.respond("candidate", _NO_FEATURE, 2, 2)
        self.assertEqual(len(candidate_retriever.calls), 2)

    def test_health_is_fixed_cardinality_partition_without_sensitive_data(self) -> None:
        agent = _agent(PartialOverlapRetriever(), candidate=True)
        agent.reset("private-sensitive-session", {})
        agent.respond("private-sensitive-session", _BROWSING, 1, 2)

        health = agent.route_redundancy_health
        self.assertEqual(set(health), _HEALTH_KEYS)
        self.assertEqual(
            health["policy"],
            "phase12-route-redundancy-corrected-stage-a-v1",
        )
        self.assertEqual(
            sum(
                value
                for key, value in health.items()
                if key not in {"policy", "attempts"}
            ),
            health["attempts"],
        )
        self.assertNotIn("private-sensitive-session", repr(health))
        self.assertNotIn("Shoes", repr(health))

    def test_reset_and_interleaved_sessions_do_not_share_candidate_state(self) -> None:
        retriever = PartialOverlapRetriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("one", {})
        agent.reset("two", {})

        one = agent.respond("one", _BROWSING, 1, 2)
        two = agent.respond("two", _BROWSING, 1, 2)
        agent.respond("one", _NO_FEATURE, 2, 2)
        agent.reset("one", {})
        one_after_reset = agent.respond("one", _BROWSING, 1, 2)

        self.assertEqual(one, two)
        self.assertEqual(one, one_after_reset)
        self.assertEqual(len(retriever.calls), 3)
        self.assertEqual(agent.orchestration_health["hits"], 1)
        self.assertEqual(agent.route_redundancy_health["attempts"], 3)


if __name__ == "__main__":
    unittest.main()
