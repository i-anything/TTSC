from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.decision_policy import (
    PROTOCOL_UTILITY_DECISION_POLICY,
)
from conversational_search.intent import LOSSLESS_MULTI_SLOT_INTENT_POLICY
from conversational_search.orchestration import (
    BackendSnapshotToken,
    EXACT_RANKING_CACHE_CAPABILITY,
)
from conversational_search.protocol import DisclosureCard, ProductProtocolEvidence
from conversational_search.ranking import (
    IMPORTANCE_AWARE_SATISFACTION_RANKING_POLICY,
    CandidateDocument,
)
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import REPEAT_TOP_SLATE_POLICY
from conversational_search.strategy import RouteWeights


class _SatisfactionRetriever:
    def __init__(
        self,
        evidence: tuple[ProductProtocolEvidence, ...],
        *,
        capable: bool = True,
        evidence_error: bool = False,
    ) -> None:
        self._evidence = {item.parent_asin: item for item in evidence}
        self._ids = tuple(self._evidence)
        self._capable = capable
        self._evidence_error = evidence_error
        self._snapshot = BackendSnapshotToken()
        self.search_calls = 0
        self.evidence_calls = 0

    @property
    def ranking_cache_capability(self) -> object:
        return EXACT_RANKING_CACHE_CAPABILITY

    @property
    def snapshot_token(self) -> BackendSnapshotToken:
        return self._snapshot

    @property
    def protocol_evidence_capability(self) -> object | None:
        return PROTOCOL_EVIDENCE_CAPABILITY if self._capable else None

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
        **kwargs: object,
    ) -> RetrievalResult:
        self.search_calls += 1
        return RetrievalResult(
            self._ids[:top_k],
            RetrievalTrace(
                bm25_ids=self._ids,
                dense_ids=self._ids,
                fused_ids=self._ids,
                bm25_status="ok",
                dense_status="ok",
                used_fallback=False,
            ),
        )

    def candidate_documents(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[CandidateDocument, ...]:
        return tuple(
            CandidateDocument(parent_asin, "Title: generic hiking shoe")
            for parent_asin in parent_asins
        )

    def candidate_protocol_evidence(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[ProductProtocolEvidence, ...]:
        self.evidence_calls += 1
        if self._evidence_error:
            raise RuntimeError("synthetic evidence failure")
        return tuple(self._evidence[parent_asin] for parent_asin in parent_asins)


def _item(
    parent_asin: str,
    *,
    hard: tuple[str, ...],
    soft: tuple[str, ...],
    price: str | None,
) -> ProductProtocolEvidence:
    return ProductProtocolEvidence(
        parent_asin,
        "Hiking Shoes",
        DisclosureCard("Hiking Shoes", hard, soft),
        text=" ".join(("Hiking Shoes", *hard, *soft)),
        price=price,
    )


def _response_ids(response: dict) -> tuple[str, ...]:
    return tuple(item["parent_asin"] for item in response["recommendations"])


class ServiceRequirementSatisfactionTests(unittest.TestCase):
    def _agent(
        self,
        retriever: _SatisfactionRetriever,
    ) -> ConversationalSearchAgent:
        return ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=IMPORTANCE_AWARE_SATISFACTION_RANKING_POLICY,
            intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
            slate_policy=REPEAT_TOP_SLATE_POLICY,
        )

    @staticmethod
    def _evidence() -> tuple[ProductProtocolEvidence, ...]:
        return (
            _item(
                "BLUE_OVER_BUDGET",
                hard=("water-resistant",),
                soft=("blue",),
                price="$140",
            ),
            _item(
                "BROWN_VALID",
                hard=("waterproof",),
                soft=("brown",),
                price="$110",
            ),
        )

    def test_policy_reorders_by_musts_before_preferred_color(self) -> None:
        retriever = _SatisfactionRetriever(self._evidence())
        agent = self._agent(retriever)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Hiking Shoes. A key requirement is: "
            "waterproof; under $120; blue would be nice.",
            1,
            10,
        )

        self.assertEqual(
            _response_ids(response),
            ("BROWN_VALID", "BLUE_OVER_BUDGET"),
        )
        state = agent.session_state("session")
        self.assertEqual(
            tuple(item.importance.value for item in state.requirements),
            ("must", "must", "prefer"),
        )
        health = agent.importance_satisfaction_health
        self.assertEqual(health["attempts"], 1)
        self.assertEqual(health["applied_reordered"], 1)
        self.assertEqual(health["validation_errors"], 0)
        self.assertEqual(health["budget_requirements"], 1)

    def test_exact_cached_order_reuses_without_reloading_evidence(self) -> None:
        retriever = _SatisfactionRetriever(self._evidence())
        agent = self._agent(retriever)
        agent.reset("session", {})
        message = (
            "I'm looking for Hiking Shoes. A key requirement is: "
            "waterproof; under $120; blue would be nice."
        )

        first = agent.respond("session", message, 1, 2)
        second = agent.respond(
            "session",
            "Those options are not quite right yet. "
            "Ask me about one specific attribute.",
            2,
            2,
        )

        self.assertEqual(_response_ids(first), _response_ids(second))
        self.assertEqual(retriever.search_calls, 1)
        self.assertEqual(retriever.evidence_calls, 1)
        self.assertEqual(agent.orchestration_health["retrievals_avoided"], 1)

    def test_missing_capability_fails_open_to_complete_stage_a_order(self) -> None:
        retriever = _SatisfactionRetriever(self._evidence(), capable=False)
        agent = self._agent(retriever)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Hiking Shoes. A key requirement is: waterproof.",
            1,
            10,
        )

        self.assertEqual(_response_ids(response), retriever._ids)
        self.assertEqual(
            agent.importance_satisfaction_health["capability_unavailable"],
            1,
        )
        self.assertEqual(agent.orchestration_health["stores"], 0)

    def test_malformed_result_fails_open_and_is_not_cached(self) -> None:
        retriever = _SatisfactionRetriever(self._evidence())
        agent = self._agent(retriever)
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.requirement_satisfaction."
            "rank_importance_aware_satisfaction",
            return_value=object(),
        ):
            response = agent.respond(
                "session",
                "I'm looking for Hiking Shoes. A key requirement is: waterproof.",
                1,
                10,
            )

        self.assertEqual(_response_ids(response), retriever._ids)
        self.assertEqual(
            agent.importance_satisfaction_health["validation_errors"],
            1,
        )
        self.assertEqual(agent.orchestration_health["stores"], 0)

    def test_policy_is_rejected_when_combined_with_decision_ablation(self) -> None:
        with self.assertRaisesRegex(ValueError, "isolated reranker ablation"):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=_SatisfactionRetriever(self._evidence()),
                ranking_policy=(
                    IMPORTANCE_AWARE_SATISFACTION_RANKING_POLICY
                ),
                decision_policy=PROTOCOL_UTILITY_DECISION_POLICY,
            )


if __name__ == "__main__":
    unittest.main()
