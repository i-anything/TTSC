from __future__ import annotations

import unittest

from conversational_search.orchestration import (
    EXACT_RANKING_CACHE_CAPABILITY,
    BackendSnapshotToken,
)
from conversational_search.ranking import (
    FUSED_ONLY_RANKING_POLICY,
    CandidateDocument,
)
from conversational_search.retrieval import (
    CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    REQUIREMENT_PROBE_CAPABILITY,
    RequirementProbeRetrievalResult,
    RequirementProbeTrace,
    RequirementProbePolicy,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights


class ProbeRetriever:
    def __init__(self, *, malformed_candidate: bool = False) -> None:
        self.ranking_cache_capability = EXACT_RANKING_CACHE_CAPABILITY
        self.requirement_probe_capability = REQUIREMENT_PROBE_CAPABILITY
        self.snapshot_token = BackendSnapshotToken()
        self.malformed_candidate = malformed_candidate
        self.calls: list[dict[str, object]] = []
        self.document_calls: list[tuple[str, ...]] = []

    def search_with_trace(
        self,
        dense_query: str,
        lexical_query: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
        **kwargs: object,
    ) -> RetrievalResult:
        self.calls.append(
            {
                "dense_query": dense_query,
                "lexical_query": lexical_query,
                "top_k": top_k,
                "route_weights": route_weights,
                **kwargs,
            }
        )
        if "requirement_probe_policy" not in kwargs:
            return RetrievalResult(
                recommendations=("B000000001",),
                trace=RetrievalTrace(
                    bm25_ids=("B000000001",),
                    dense_ids=(),
                    fused_ids=("B000000001",),
                    bm25_status="ok",
                    dense_status="empty",
                    used_fallback=False,
                ),
            )
        bm25_ids = (
            ("B000000001",)
            if self.malformed_candidate
            else ("B000000001", "B000000002")
        )
        fused_ids = bm25_ids
        return RequirementProbeRetrievalResult(
            recommendations=fused_ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=bm25_ids,
                dense_ids=(),
                fused_ids=fused_ids,
                bm25_status="ok",
                dense_status="empty",
                used_fallback=False,
            ),
            probe_trace=RequirementProbeTrace(
                base_bm25_ids=("B000000001",),
                supplemental_ids=("B000000002",),
                status="ok",
                query_count=1,
            ),
        )

    def candidate_documents(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[CandidateDocument, ...]:
        self.document_calls.append(parent_asins)
        texts = {
            "B000000001": "Title: basic shoes",
            "B000000002": "Title: waterproof shoes",
        }
        return tuple(
            CandidateDocument(parent_asin, texts[parent_asin])
            for parent_asin in parent_asins
        )


class ServiceRequirementProbeTest(unittest.TestCase):
    @staticmethod
    def _message() -> str:
        return "I'm looking for Shoes. A key requirement is: waterproof"

    def test_enabled_policy_passes_bounded_candidates_and_records_aggregates(self) -> None:
        retriever = ProbeRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond("session", self._message(), 1, 2)

        self.assertEqual(len(retriever.calls), 1)
        self.assertIs(
            retriever.calls[0]["requirement_probe_policy"],
            CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        )
        self.assertEqual(
            retriever.calls[0]["requirement_probe_candidates"],
            ("waterproof",),
        )
        self.assertEqual(
            {item["parent_asin"] for item in response["recommendations"]},
            {"B000000001", "B000000002"},
        )
        self.assertEqual(
            agent.requirement_probe_health,
            {
                "policy": "catalog_idf_requirement_probes",
                "attempts": 1,
                "disabled": 0,
                "no_eligible": 0,
                "capacity": 0,
                "successful_supplements": 1,
                "empty_routes": 0,
                "no_additions": 0,
                "unavailable": 0,
                "errors": 0,
                "selected_probe_queries": 1,
                "supplemental_ids": 1,
                "validation_or_execution_fallbacks": 0,
            },
        )
        self.assertEqual(agent.orchestration_health["stores"], 1)

    def test_malformed_capability_result_fails_closed_without_a_second_search(self) -> None:
        retriever = ProbeRetriever(malformed_candidate=True)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond("session", self._message(), 1, 2)

        self.assertEqual(len(retriever.calls), 1)
        self.assertIn("requirement_probe_policy", retriever.calls[0])
        self.assertEqual(response["recommendations"], [])
        self.assertEqual(agent.requirement_probe_health["errors"], 1)
        self.assertEqual(
            agent.requirement_probe_health[
                "validation_or_execution_fallbacks"
            ],
            1,
        )
        self.assertEqual(agent.orchestration_health["stores"], 0)

    def test_exact_ranking_reuse_performs_no_second_probe_or_route_call(self) -> None:
        retriever = ProbeRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        )
        agent.reset("session", {})
        first = agent.respond("session", self._message(), 1, 2)
        asked = first["ask_attribute"]
        self.assertIsInstance(asked, str)

        agent.respond(
            "session",
            f"I don't have an additional preference for {asked}.",
            2,
            2,
        )

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(agent.requirement_probe_health["attempts"], 1)
        self.assertEqual(agent.orchestration_health["reuses"], 1)

    def test_policy_is_rejected_for_ranking_paths_with_phase10_assumptions(self) -> None:
        with self.assertRaises(ValueError):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=ProbeRetriever(),
                ranking_policy=FUSED_ONLY_RANKING_POLICY,
                requirement_probe_policy=(
                    CATALOG_IDF_REQUIREMENT_PROBE_POLICY
                ),
            )

    def test_policy_argument_requires_the_closed_enum(self) -> None:
        with self.assertRaises(TypeError):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=ProbeRetriever(),
                requirement_probe_policy="enabled",  # type: ignore[arg-type]
            )
        self.assertEqual(
            tuple(policy.value for policy in RequirementProbePolicy),
            ("disabled", "catalog_idf_requirement_probes"),
        )


if __name__ == "__main__":
    unittest.main()
