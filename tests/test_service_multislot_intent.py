from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.intent import (
    LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from starter.agent import DEFAULT_CATALOG_PATH, Agent
from tests.test_service import CacheableRecordingRetriever


_IDS = ("B000000001", "B000000002", "B000000003")
_DOCUMENTS = {
    _IDS[0]: "red leather walking shoe",
    _IDS[1]: "blue wool boot",
    _IDS[2]: "waterproof synthetic shoe",
}


def _retriever() -> CacheableRecordingRetriever:
    return CacheableRecordingRetriever(
        _IDS,
        fused_ids=_IDS,
        documents=_DOCUMENTS,
    )


def _agent(
    retriever: CacheableRecordingRetriever,
    *,
    candidate: bool,
) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        "unused.jsonl",
        retriever=retriever,
        intent_policy=(
            LOSSLESS_MULTI_SLOT_INTENT_POLICY
            if candidate
            else ROBUST_INTENT_POLICY
        ),
    )


class MultiSlotServiceIntegrationTest(unittest.TestCase):
    def test_single_slot_candidate_path_is_exact_phase9(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})
        message = "I'm looking for Shoes. A key requirement is: leather."

        candidate_response = candidate.respond("candidate", message, 1, 3)
        baseline_response = baseline.respond("baseline", message, 1, 3)

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(
            candidate.session_state("candidate"),
            baseline.session_state("baseline"),
        )
        self.assertEqual(candidate_retriever.calls, baseline_retriever.calls)
        self.assertEqual(
            candidate_retriever.document_calls,
            baseline_retriever.document_calls,
        )
        self.assertEqual(candidate.orchestration_health, baseline.orchestration_health)

    def test_multi_slot_state_drives_typed_queries_without_extra_calls(self) -> None:
        retriever = _retriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("session", {})

        response = agent.respond("session", "I want red and leather", 1, 3)

        state = agent.session_state("session")
        self.assertEqual(
            [(item.value, item.attribute) for item in state.requirements],
            [("red", "color"), ("leather", "material")],
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(retriever.document_calls), 1)
        dense_query, lexical_query, _ = retriever.calls[0]
        self.assertIn("Material: leather", dense_query)
        self.assertIn("Color: red", dense_query)
        self.assertIn("red", lexical_query)
        self.assertIn("leather", lexical_query)
        self.assertEqual(len(response["recommendations"]), 3)
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})

    def test_exact_no_change_reuses_and_new_multi_slot_evidence_searches(self) -> None:
        retriever = _retriever()
        agent = _agent(retriever, candidate=True)
        agent.reset("session", {})

        agent.respond("session", "I want red and leather", 1, 3)
        agent.respond(
            "session",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            2,
            3,
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(agent.orchestration_health["hits"], 1)

        agent.respond("session", "now blue and wool", 3, 3)
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(agent.orchestration_health["dependency_misses"], 1)

    def test_parser_exception_returns_phase9_and_does_not_commit_cache(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})
        message = "I want red and leather"

        baseline_response = baseline.respond("baseline", message, 1, 3)
        with mock.patch(
            "conversational_search.intent._apply_candidate_atoms",
            side_effect=RuntimeError("synthetic parser fault"),
        ):
            candidate_response = candidate.respond("candidate", message, 1, 3)

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(
            candidate.session_state("candidate"),
            baseline.session_state("baseline"),
        )
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(baseline.orchestration_health["stores"], 1)

    def test_bound_fallback_is_exact_phase9_and_does_not_commit_cache(self) -> None:
        candidate_retriever = _retriever()
        baseline_retriever = _retriever()
        candidate = _agent(candidate_retriever, candidate=True)
        baseline = _agent(baseline_retriever, candidate=False)
        candidate.reset("candidate", {})
        baseline.reset("baseline", {})
        message = "x" * 2049

        candidate_response = candidate.respond("candidate", message, 1, 3)
        baseline_response = baseline.respond("baseline", message, 1, 3)

        self.assertEqual(candidate_response, baseline_response)
        self.assertEqual(
            candidate.session_state("candidate"),
            baseline.session_state("baseline"),
        )
        self.assertEqual(candidate.orchestration_health["stores"], 0)
        self.assertEqual(baseline.orchestration_health["stores"], 1)

    def test_reset_and_interleaved_sessions_remain_isolated(self) -> None:
        agent = _agent(_retriever(), candidate=True)
        agent.reset("one", {})
        agent.reset("two", {})

        agent.respond("one", "red and leather", 1, 3)
        agent.respond("two", "blue and wool", 1, 3)

        self.assertEqual(
            [item.value for item in agent.session_state("one").requirements],
            ["red", "leather"],
        )
        self.assertEqual(
            [item.value for item in agent.session_state("two").requirements],
            ["blue", "wool"],
        )
        agent.reset("one", {})
        self.assertEqual(agent.session_state("one").requirements, ())
        self.assertEqual(
            [item.value for item in agent.session_state("two").requirements],
            ["blue", "wool"],
        )

    def test_starter_keeps_phase11_parser_rejected_after_phase13_promotion(self) -> None:
        with mock.patch.object(
            ConversationalSearchAgent,
            "__init__",
            return_value=None,
        ) as initialize:
            Agent()
        initialize.assert_called_once_with(
            DEFAULT_CATALOG_PATH,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )

        with self.assertRaises(TypeError):
            Agent(intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
