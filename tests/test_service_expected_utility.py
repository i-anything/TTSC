from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.decision_policy import (
    EXPECTED_UTILITY_DECISION_POLICY,
)
from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    BackendSnapshotToken,
    EXACT_RANKING_CACHE_CAPABILITY,
)
from conversational_search.protocol import (
    DisclosureCard,
    ProductProtocolEvidence,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    CandidateDocument,
)
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
)
from conversational_search.strategy import RouteWeights


class _ExpectedRetriever:
    def __init__(
        self,
        cards: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
        *,
        evidence_error: bool = False,
    ) -> None:
        self._cards = cards
        self._ids = tuple(cards)
        self._snapshot = BackendSnapshotToken()
        self._evidence_error = evidence_error
        self.search_calls = 0
        self.evidence_calls = 0
        self.dense_requests: list[bool] = []

    @property
    def ranking_cache_capability(self) -> object:
        return EXACT_RANKING_CACHE_CAPABILITY

    @property
    def snapshot_token(self) -> BackendSnapshotToken:
        return self._snapshot

    @property
    def protocol_evidence_capability(self) -> object:
        return PROTOCOL_EVIDENCE_CAPABILITY

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
            raise RuntimeError("evidence unavailable")
        return tuple(
            ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    f"{parent_asin} shoe",
                    self._cards[parent_asin][0],
                    self._cards[parent_asin][1],
                ),
                text=" ".join(
                    (
                        "cotton shoe",
                        *self._cards[parent_asin][0],
                        *self._cards[parent_asin][1],
                    )
                ),
            )
            for parent_asin in parent_asins
        )


def _agent(retriever: _ExpectedRetriever) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        "unused.jsonl",
        retriever=retriever,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        decision_policy=EXPECTED_UTILITY_DECISION_POLICY,
    )


def _ids(response: dict) -> tuple[str, ...]:
    return tuple(item["parent_asin"] for item in response["recommendations"])


class ServiceExpectedUtilityTests(unittest.TestCase):
    def test_constructor_requires_the_coherent_phase4_stack(self) -> None:
        retriever = _ExpectedRetriever({"A": (("cotton",), ())})
        with self.assertRaisesRegex(ValueError, "lexicographic"):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=retriever,
                slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
                decision_policy=EXPECTED_UTILITY_DECISION_POLICY,
            )
        with self.assertRaisesRegex(ValueError, "intent-epoch"):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=retriever,
                ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
                slate_policy=STAGNATION_AWARE_SLATE_POLICY,
                decision_policy=EXPECTED_UTILITY_DECISION_POLICY,
            )
        with self.assertRaisesRegex(ValueError, "orchestration"):
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=retriever,
                ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
                slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
                decision_policy=EXPECTED_UTILITY_DECISION_POLICY,
                orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
            )

    def test_search_uses_one_evidence_pass_and_the_full_hybrid_route(self) -> None:
        retriever = _ExpectedRetriever(
            {
                "A": (("cotton",), ("color: blue",)),
                "B": (("cotton",), ("color: red",)),
            }
        )
        agent = _agent(retriever)
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            2,
        )

        self.assertIs(agent.decision_policy, EXPECTED_UTILITY_DECISION_POLICY)
        self.assertEqual(retriever.search_calls, 1)
        self.assertEqual(retriever.evidence_calls, 1)
        self.assertEqual(retriever.dense_requests, [True])
        self.assertEqual(agent.exact_evidence_health["attempts"], 1)
        self.assertEqual(agent.protocol_decision_health["applied"], 1)
        self.assertGreaterEqual(len(response["recommendations"]), 1)
        trace = agent.last_action_trace("session")
        assert trace is not None
        self.assertEqual(trace["world_mode"], "exact")
        self.assertEqual(trace["dense_policy"], "dense-always-v1")
        self.assertIn("expected_utility", trace)

    def test_reuse_skips_search_but_reranks_exact_evidence_once(self) -> None:
        retriever = _ExpectedRetriever(
            {
                "A": (("cotton",), ()),
                "B": (("cotton",), ()),
            }
        )
        agent = _agent(retriever)
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            2,
        )
        second = agent.respond(
            "session",
            "Those options are not quite right yet. "
            "Ask me about one specific attribute.",
            2,
            2,
        )

        self.assertEqual(retriever.search_calls, 1)
        self.assertEqual(retriever.evidence_calls, 2)
        self.assertEqual(agent.exact_evidence_health["attempts"], 2)
        self.assertEqual(agent.orchestration_health["retrievals_avoided"], 1)
        self.assertNotEqual(_ids(first), _ids(second))
        self.assertEqual(_ids(second)[0], "B")

    def test_tentative_override_interval_forces_zero_width_without_slate_mutation(
        self,
    ) -> None:
        retriever = _ExpectedRetriever(
            {
                "A": (("cotton",), ("color: blue",)),
                "B": (("cotton",), ("color: blue",)),
            }
        )
        agent = _agent(retriever)
        agent.reset("session", {})
        before = agent.slate_state("session")

        response = agent.respond(
            "session",
            "I'm looking for Shoes. color: blue",
            1,
            2,
        )

        self.assertEqual(response["recommendations"], [])
        self.assertEqual(agent.slate_state("session"), before)
        trace = agent.last_action_trace("session")
        assert trace is not None
        self.assertEqual(trace["presented_width"], 0)

    def test_planner_fault_fails_open_and_retains_the_valid_observation(self) -> None:
        retriever = _ExpectedRetriever(
            {
                "A": (("cotton",), ()),
                "B": (("cotton",), ()),
            }
        )
        agent = _agent(retriever)
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.decision.plan_expected_utility_decision",
            side_effect=RuntimeError("planner fault"),
        ):
            response = agent.respond(
                "session",
                "I'm looking for Shoes. A key requirement is: cotton.",
                1,
                2,
            )

        self.assertEqual(len(response["recommendations"]), 2)
        self.assertEqual(response["ask_attribute"], "feature")
        self.assertEqual(len(agent._protocol_events["session"]), 1)
        self.assertEqual(agent.protocol_decision_health["applied"], 0)

    def test_evidence_fault_and_free_form_both_return_protected_full_width(self) -> None:
        evidence_fault = _agent(
            _ExpectedRetriever(
                {"A": (("cotton",), ()), "B": (("cotton",), ())},
                evidence_error=True,
            )
        )
        evidence_fault.reset("evidence", {})
        evidence_response = evidence_fault.respond(
            "evidence",
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            2,
        )

        free_form = _agent(
            _ExpectedRetriever(
                {"A": (("cotton",), ()), "B": (("cotton",), ())}
            )
        )
        free_form.reset("free", {})
        free_response = free_form.respond(
            "free",
            "Please find comfortable cotton shoes for commuting",
            1,
            2,
        )

        self.assertEqual(len(evidence_response["recommendations"]), 2)
        self.assertEqual(evidence_response["ask_attribute"], "feature")
        self.assertEqual(len(free_response["recommendations"]), 2)
        self.assertEqual(free_response["ask_attribute"], "feature")
        trace = free_form.last_action_trace("free")
        assert trace is not None
        self.assertEqual(trace["protocol_mode"], "free_form_fail_open")


if __name__ == "__main__":
    unittest.main()
