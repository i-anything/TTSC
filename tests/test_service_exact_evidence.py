from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.decision_policy import PROTOCOL_UTILITY_DECISION_POLICY
from conversational_search.protocol import (
    DisclosureCard,
    ProductProtocolEvidence,
)
from conversational_search.orchestration import (
    BackendSnapshotToken,
    EXACT_RANKING_CACHE_CAPABILITY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    CandidateDocument,
)
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights


class _ExactEvidenceRetriever:
    def __init__(
        self,
        cards: dict[str, tuple[str, ...]],
        *,
        capable: bool = True,
        evidence_error: bool = False,
    ) -> None:
        self._ids = tuple(cards)
        self._cards = cards
        self._capable = capable
        self._evidence_error = evidence_error
        self._snapshot_token = BackendSnapshotToken()
        self.search_calls = 0
        self.evidence_calls = 0
        self.dense_requests: list[bool] = []

    @property
    def ranking_cache_capability(self) -> object:
        return EXACT_RANKING_CACHE_CAPABILITY

    @property
    def snapshot_token(self) -> BackendSnapshotToken:
        return self._snapshot_token

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
        use_dense = bool(kwargs.get("use_dense", True))
        self.dense_requests.append(use_dense)
        return RetrievalResult(
            recommendations=self._ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=self._ids,
                dense_ids=self._ids if use_dense else (),
                fused_ids=self._ids,
                bm25_status="ok",
                dense_status="ok" if use_dense else "skipped",
                used_fallback=False,
            ),
        )

    def candidate_documents(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[CandidateDocument, ...]:
        return tuple(
            CandidateDocument(parent_asin, "cotton shoe")
            for parent_asin in parent_asins
        )

    def candidate_protocol_evidence(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[ProductProtocolEvidence, ...]:
        self.evidence_calls += 1
        if self._evidence_error:
            raise RuntimeError("protocol evidence unavailable")
        if not self._capable:
            return ()
        return tuple(
            ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    f"{parent_asin} fixture",
                    self._cards[parent_asin],
                    (),
                ),
                text="cotton shoe",
            )
            for parent_asin in parent_asins
        )


def _response_ids(response: dict) -> tuple[str, ...]:
    return tuple(item["parent_asin"] for item in response["recommendations"])


class ServiceExactEvidenceTests(unittest.TestCase):
    def test_exact_policy_reorders_after_stage_a_without_changing_question(self) -> None:
        cards = {
            "BASE_FIRST": ("leather",),
            "EXACT_SECOND": ("cotton",),
        }
        protected = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=_ExactEvidenceRetriever(cards),
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        candidate_retriever = _ExactEvidenceRetriever(cards)
        candidate = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=candidate_retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        for agent, session_id in (
            (protected, "protected"),
            (candidate, "candidate"),
        ):
            agent.reset(session_id, {})
        message = "I'm looking for Shoes. A key requirement is: cotton."

        protected_response = protected.respond("protected", message, 1, 10)
        candidate_response = candidate.respond("candidate", message, 1, 10)

        self.assertEqual(_response_ids(protected_response)[0], "BASE_FIRST")
        self.assertEqual(_response_ids(candidate_response)[0], "EXACT_SECOND")
        self.assertEqual(
            protected_response["ask_attribute"],
            candidate_response["ask_attribute"],
        )
        self.assertEqual(len(protected_response["recommendations"]), 2)
        self.assertEqual(len(candidate_response["recommendations"]), 2)
        self.assertEqual(
            frozenset(_response_ids(candidate_response)),
            frozenset(cards),
        )
        self.assertEqual(candidate_retriever.dense_requests, [True])
        self.assertEqual(candidate.exact_evidence_health["applied"], 1)
        self.assertEqual(candidate.exact_evidence_health["validation_errors"], 0)

    def test_zero_support_fails_open_to_stage_a_order(self) -> None:
        retriever = _ExactEvidenceRetriever(
            {
                "BASE_FIRST": ("leather",),
                "BASE_SECOND": ("silk",),
            }
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertEqual(
            _response_ids(response),
            ("BASE_FIRST", "BASE_SECOND"),
        )
        self.assertEqual(
            agent.exact_evidence_health["zero_support_fail_open"],
            1,
        )

    def test_missing_capability_fails_open_without_response_fault(self) -> None:
        cards = {"BASE_FIRST": ("leather",), "EXACT_SECOND": ("cotton",)}
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=_ExactEvidenceRetriever(cards, capable=False),
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertEqual(_response_ids(response)[0], "BASE_FIRST")
        self.assertEqual(
            agent.exact_evidence_health["capability_unavailable"],
            1,
        )

    def test_evidence_backend_error_fails_open_without_response_fault(self) -> None:
        cards = {"BASE_FIRST": ("leather",), "EXACT_SECOND": ("cotton",)}
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=_ExactEvidenceRetriever(cards, evidence_error=True),
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertEqual(_response_ids(response), tuple(cards))
        self.assertEqual(agent.exact_evidence_health["evidence_errors"], 1)

    def test_malformed_exact_result_fails_open_to_stage_a_order(self) -> None:
        cards = {"BASE_FIRST": ("leather",), "EXACT_SECOND": ("cotton",)}
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=_ExactEvidenceRetriever(cards),
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.exact_evidence.rank_exact_evidence",
            return_value=object(),
        ):
            response = agent.respond(
                "session",
                "I'm looking for Shoes. A key requirement is: cotton.",
                1,
                10,
            )

        self.assertEqual(_response_ids(response), tuple(cards))
        self.assertEqual(agent.exact_evidence_health["validation_errors"], 1)

    def test_reuse_reads_the_cached_exact_order_without_reloading_evidence(
        self,
    ) -> None:
        cards = {
            "BASE_FIRST": ("leather",),
            "EXACT_SECOND": ("cotton",),
        }
        retriever = _ExactEvidenceRetriever(cards)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            2,
        )
        agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            2,
        )

        self.assertEqual(retriever.search_calls, 1)
        self.assertEqual(retriever.evidence_calls, 1)
        self.assertEqual(agent.exact_evidence_health["attempts"], 1)
        self.assertEqual(agent.exact_evidence_health["applied_reordered"], 1)
        self.assertEqual(agent.orchestration_health["retrievals_avoided"], 1)

    def test_default_ranking_policy_keeps_exact_layer_inactive(self) -> None:
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=_ExactEvidenceRetriever({"ONLY": ("cotton",)}),
        )
        self.assertEqual(agent.ranking_health["policy"], "stage_a")
        self.assertEqual(agent.exact_evidence_health["attempts"], 0)

    def test_invalid_ranking_policy_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=_ExactEvidenceRetriever({"ONLY": ("cotton",)}),
                ranking_policy="enabled",  # type: ignore[arg-type]
            )

    def test_static_exact_and_protocol_utility_cannot_double_apply(self) -> None:
        with self.assertRaisesRegex(ValueError, "unified planner integration"):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=_ExactEvidenceRetriever({"ONLY": ("cotton",)}),
                ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
                decision_policy=PROTOCOL_UTILITY_DECISION_POLICY,
            )


if __name__ == "__main__":
    unittest.main()
