from __future__ import annotations

import unittest
from dataclasses import fields
from unittest import mock

from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.ranking import (
    COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    Bm25RescueRankingResult,
    Bm25RescueStatus,
    ProfileResidualStatus,
    RankingResult,
    RankingTrace,
)
from conversational_search.service import ConversationalSearchAgent
from tests.test_service import CacheableRecordingRetriever


_IDS = ("B000000001", "B000000002")
_BROWSING_MESSAGE = "I'm looking for Shoes, but I'm still exploring."
_BUYING_MESSAGE = "I'm looking for Shoes. A key requirement is: leather."
_NO_FEATURE_PREFERENCE = (
    "I don't have a preference for feature; please use your judgment."
)
_PROFILE = {"preference_tags": ["comfort"]}
_DOCUMENTS = {
    _IDS[0]: "ordinary shoe",
    _IDS[1]: "comfortable cushioned shoe",
}
_RESCUE_HEALTH_KEYS = {
    "policy",
    "attempts",
    "zero_completeness_neutral",
    "bm25_unavailable_or_empty_neutral",
    "no_positive_uplift_neutral",
    "constant_uplift_neutral",
    "unchanged_order_neutral",
    "successful_reorders",
    "validation_or_scoring_fallbacks",
}


def _agent(
    retriever: object,
    *,
    ranking_policy: object = COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        "unused.jsonl",
        retriever=retriever,
        ranking_policy=ranking_policy,  # type: ignore[arg-type]
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )


def _retriever() -> CacheableRecordingRetriever:
    return CacheableRecordingRetriever(
        _IDS,
        fused_ids=_IDS,
        documents=_DOCUMENTS,
    )


def _cache_snapshot(agent: ConversationalSearchAgent) -> tuple[object, ...]:
    entries = agent._orchestrator._entries  # type: ignore[attr-defined]
    return tuple(
        (
            key,
            entry.dependency_digest,
            entry.ranked_ids,
            tuple(field.name for field in fields(entry)),
        )
        for key, entry in entries.items()
    )


class ServiceBm25RescueTests(unittest.TestCase):
    def test_zero_completeness_is_exact_phase9_and_both_policies_reuse(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever)
        baseline = _agent(
            baseline_retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        candidate.reset("session", _PROFILE)
        baseline.reset("session", _PROFILE)

        candidate_first = candidate.respond("session", _BROWSING_MESSAGE, 1, 2)
        baseline_first = baseline.respond("session", _BROWSING_MESSAGE, 1, 2)
        candidate.respond("session", _NO_FEATURE_PREFERENCE, 2, 2)
        baseline.respond("session", _NO_FEATURE_PREFERENCE, 2, 2)

        self.assertEqual(candidate_first, baseline_first)
        self.assertEqual(candidate.session_state("session"), baseline.session_state("session"))
        self.assertEqual(candidate_retriever.calls, baseline_retriever.calls)
        self.assertEqual(candidate_retriever.document_calls, baseline_retriever.document_calls)
        self.assertEqual(len(candidate_retriever.calls), 1)
        self.assertEqual(candidate.orchestration_health["hits"], 1)
        self.assertEqual(baseline.orchestration_health["hits"], 1)
        self.assertEqual(candidate.rescue_health["attempts"], 1)
        self.assertEqual(candidate.rescue_health["zero_completeness_neutral"], 1)
        self.assertEqual(baseline.rescue_health["attempts"], 0)

    def test_unusable_bm25_route_bypasses_rescue_and_returns_exact_phase9(self) -> None:
        candidate_retriever = _retriever()
        candidate_retriever.bm25_status_override = "error"
        candidate_retriever.used_fallback_override = False
        baseline_retriever = _retriever()
        baseline_retriever.bm25_status_override = "error"
        baseline_retriever.used_fallback_override = False
        candidate = _agent(candidate_retriever)
        baseline = _agent(
            baseline_retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        candidate.reset("candidate", _PROFILE)
        baseline.reset("baseline", _PROFILE)

        baseline_response = baseline.respond("baseline", _BROWSING_MESSAGE, 1, 2)
        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_bm25_rescue",
            side_effect=AssertionError("unusable BM25 route entered rescue"),
        ) as rescue:
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        rescue.assert_not_called()
        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.rescue_health["attempts"], 1)
        self.assertEqual(
            candidate.rescue_health["bm25_unavailable_or_empty_neutral"],
            1,
        )
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(candidate.ranking_health["successes"], 1)

    def test_unexpected_composite_fault_recomputes_phase9_and_never_caches(self) -> None:
        candidate_retriever = _retriever()
        baseline = _agent(_retriever(), ranking_policy=STAGE_A_RANKING_POLICY)
        candidate = _agent(candidate_retriever)
        baseline.reset("baseline", _PROFILE)
        candidate.reset("candidate", _PROFILE)
        baseline_response = baseline.respond("baseline", _BROWSING_MESSAGE, 1, 2)

        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_bm25_rescue",
            side_effect=RuntimeError("synthetic composite fault"),
        ):
            candidate_response = candidate.respond(
                "candidate",
                _BROWSING_MESSAGE,
                1,
                2,
            )

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(candidate.rescue_health["attempts"], 1)
        self.assertEqual(
            candidate.rescue_health["validation_or_scoring_fallbacks"],
            1,
        )
        self.assertEqual(
            candidate.profile_health["successful_residual_applications"],
            1,
        )

        candidate.respond("candidate", _NO_FEATURE_PREFERENCE, 2, 2)

        self.assertEqual(len(candidate_retriever.calls), 2)
        self.assertEqual(candidate.orchestration_health["hits"], 0)
        self.assertEqual(candidate.orchestration_health["stores"], 1)
        self.assertEqual(candidate.rescue_health["attempts"], 2)
        self.assertEqual(candidate.rescue_health["zero_completeness_neutral"], 1)

    def test_malformed_composite_results_fail_closed_and_never_cache(self) -> None:
        expected_beta = 0.20 + 0.25 / 3.0
        malformed_rankings = {
            "unknown_id": RankingResult(
                ranked_ids=("UNKNOWN", _IDS[0]),
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=("UNKNOWN", _IDS[0]),
                    beta=expected_beta,
                    observable_clause_count=1,
                ),
            ),
            "missing_id": RankingResult(
                ranked_ids=(_IDS[0],),
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=(_IDS[0],),
                    beta=expected_beta,
                    observable_clause_count=1,
                ),
            ),
            "duplicate_id": RankingResult(
                ranked_ids=(_IDS[0], _IDS[0]),
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=(_IDS[0], _IDS[0]),
                    beta=expected_beta,
                    observable_clause_count=1,
                ),
            ),
            "trace_mismatch": RankingResult(
                ranked_ids=_IDS,
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=tuple(reversed(_IDS)),
                    beta=expected_beta,
                    observable_clause_count=1,
                ),
            ),
            "wrong_beta": RankingResult(
                ranked_ids=_IDS,
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=_IDS,
                    beta=0.20,
                    observable_clause_count=1,
                ),
            ),
            "unbounded_clause_count": RankingResult(
                ranked_ids=_IDS,
                trace=RankingTrace(
                    input_ids=_IDS,
                    output_ids=_IDS,
                    beta=expected_beta,
                    observable_clause_count=33,
                ),
            ),
        }

        for label, malformed_ranking in malformed_rankings.items():
            with self.subTest(label=label):
                baseline = _agent(
                    _retriever(),
                    ranking_policy=STAGE_A_RANKING_POLICY,
                )
                candidate = _agent(_retriever())
                baseline.reset("baseline", _PROFILE)
                candidate.reset("candidate", _PROFILE)
                baseline_response = baseline.respond(
                    "baseline",
                    _BUYING_MESSAGE,
                    1,
                    2,
                )
                malformed = Bm25RescueRankingResult(
                    ranking=malformed_ranking,
                    status=Bm25RescueStatus.REORDERED,
                    profile_status=ProfileResidualStatus.ACTIVE_REQUIREMENTS,
                    requested_theme_count=1,
                    represented_theme_count=0,
                )

                with mock.patch(
                    "conversational_search.service."
                    "rerank_stage_a_with_profile_and_bm25_rescue",
                    return_value=malformed,
                ):
                    candidate_response = candidate.respond(
                        "candidate",
                        _BUYING_MESSAGE,
                        1,
                        2,
                    )

                self.assertEqual(candidate_response, baseline_response)
                self.assertEqual(candidate.orchestration_health["stores"], 0)
                self.assertEqual(
                    candidate.rescue_health[
                        "validation_or_scoring_fallbacks"
                    ],
                    1,
                )

    def test_successful_reorder_is_returned_and_health_is_fixed_partition(self) -> None:
        agent = _agent(_retriever())
        agent.reset("private-session", _PROFILE)
        ranking = RankingResult(
            ranked_ids=(_IDS[1], _IDS[0]),
            trace=RankingTrace(
                input_ids=_IDS,
                output_ids=(_IDS[1], _IDS[0]),
                beta=0.20 + 0.25 / 3.0,
                observable_clause_count=1,
            ),
        )
        result = Bm25RescueRankingResult(
            ranking=ranking,
            status=Bm25RescueStatus.REORDERED,
            profile_status=ProfileResidualStatus.ACTIVE_REQUIREMENTS,
            requested_theme_count=1,
            represented_theme_count=0,
        )

        with mock.patch(
            "conversational_search.service."
            "rerank_stage_a_with_profile_and_bm25_rescue",
            return_value=result,
        ):
            response = agent.respond("private-session", _BUYING_MESSAGE, 1, 2)

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": _IDS[1]}, {"parent_asin": _IDS[0]}],
        )
        health = agent.rescue_health
        self.assertEqual(set(health), _RESCUE_HEALTH_KEYS)
        self.assertEqual(
            health["policy"],
            "phase10-completeness-gated-bm25-rescue-v1",
        )
        self.assertEqual(health["attempts"], 1)
        self.assertEqual(health["successful_reorders"], 1)
        self.assertEqual(
            sum(
                value
                for key, value in health.items()
                if key not in {"policy", "attempts"}
            ),
            health["attempts"],
        )
        self.assertTrue(
            all(
                type(value) is int and value >= 0
                for key, value in health.items()
                if key != "policy"
            )
        )
        self.assertNotIn("private-session", repr(health))
        self.assertNotIn("leather", repr(health))

    def test_independent_cache_snapshots_are_exact_and_contain_no_rescue_state(self) -> None:
        first = _agent(_retriever())
        second = _agent(_retriever())
        baseline = _agent(_retriever(), ranking_policy=STAGE_A_RANKING_POLICY)
        for agent in (first, second, baseline):
            agent.reset("same-session", _PROFILE)
            agent.respond("same-session", _BROWSING_MESSAGE, 1, 2)

        first_snapshot = _cache_snapshot(first)
        second_snapshot = _cache_snapshot(second)
        baseline_snapshot = _cache_snapshot(baseline)

        self.assertEqual(first_snapshot, second_snapshot)
        self.assertEqual(len(first_snapshot), 1)
        self.assertEqual(
            first_snapshot[0][3],
            ("dependency_digest", "backend_snapshot_token", "ranked_ids"),
        )
        self.assertNotEqual(first_snapshot[0][1], baseline_snapshot[0][1])
        self.assertNotIn("score", repr(first_snapshot).casefold())
        self.assertNotIn("uplift", repr(first_snapshot).casefold())


if __name__ == "__main__":
    unittest.main()
