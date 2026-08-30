from __future__ import annotations

import unittest
from unittest import mock

from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
)
from tests.test_service import CacheableRecordingRetriever, _product_ids


def _ids(response: dict) -> tuple[str, ...]:
    return tuple(
        item["parent_asin"] for item in response["recommendations"]
    )


class IntentEpochSlateServiceTests(unittest.TestCase):
    def _agent(
        self,
        *,
        policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    ) -> tuple[ConversationalSearchAgent, CacheableRecordingRetriever]:
        parent_asins = _product_ids(12)
        retriever = CacheableRecordingRetriever(
            results=parent_asins,
            fused_ids=parent_asins,
        )
        return (
            ConversationalSearchAgent(
                "unused.jsonl",
                retriever=retriever,
                slate_policy=policy,
            ),
            retriever,
        )

    def test_same_epoch_refinement_avoids_previously_shown_products(self) -> None:
        candidate, candidate_retriever = self._agent()
        baseline, baseline_retriever = self._agent(
            policy=STAGNATION_AWARE_SLATE_POLICY
        )
        for name, agent in (("candidate", candidate), ("baseline", baseline)):
            agent.reset(name, {})

        first_candidate = candidate.respond(
            "candidate",
            "I'm looking for shoes, but I'm still exploring.",
            1,
            3,
        )
        first_baseline = baseline.respond(
            "baseline",
            "I'm looking for shoes, but I'm still exploring.",
            1,
            3,
        )
        self.assertEqual(_ids(first_candidate), _ids(first_baseline))
        self.assertEqual(_ids(first_candidate), _product_ids(3))

        refined_candidate = candidate.respond(
            "candidate",
            "For that, what matters is: cotton.",
            2,
            3,
        )
        refined_baseline = baseline.respond(
            "baseline",
            "For that, what matters is: cotton.",
            2,
            3,
        )
        self.assertEqual(_ids(refined_candidate), _product_ids(6)[3:])
        self.assertEqual(_ids(refined_baseline), _product_ids(3))
        self.assertEqual(len(candidate_retriever.calls), 2)
        self.assertEqual(len(baseline_retriever.calls), 2)
        health = candidate.intent_epoch_slate_health
        self.assertEqual(health["attempts"], 2)
        self.assertEqual(health["first_slate_exact_baseline"], 1)
        self.assertEqual(health["same_epoch_history_carried"], 1)
        self.assertEqual(health["eligible_prior_shown_total"], 3)
        self.assertEqual(health["validation_fallbacks"], 0)

    def test_explicit_override_epoch_resets_history(self) -> None:
        agent, _retriever = self._agent()
        agent.reset("override", {})
        first = agent.respond(
            "override",
            "I'm looking for shoes. A key requirement is: leather.",
            1,
            3,
        )
        replaced = agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            2,
            3,
        )
        self.assertEqual(_ids(first), _product_ids(3))
        self.assertEqual(_ids(replaced), _product_ids(3))
        health = agent.intent_epoch_slate_health
        self.assertEqual(health["first_slate_exact_baseline"], 1)
        self.assertEqual(health["changed_epoch_exact_baseline"], 1)
        self.assertEqual(health["same_epoch_history_carried"], 0)

    def test_unchanged_ranking_reuses_cache_and_existing_novelty(self) -> None:
        agent, retriever = self._agent()
        agent.reset("reuse", {})
        first = agent.respond(
            "reuse",
            "I'm looking for shoes, but I'm still exploring.",
            1,
            3,
        )
        second = agent.respond(
            "reuse",
            "I don't have an additional preference for material.",
            2,
            3,
        )
        self.assertEqual(_ids(first), _product_ids(3))
        self.assertEqual(_ids(second), _product_ids(6)[3:])
        self.assertEqual(len(retriever.calls), 1)
        health = agent.intent_epoch_slate_health
        self.assertEqual(health["unchanged_signature_exact_baseline"], 1)
        self.assertEqual(health["same_epoch_history_carried"], 0)

    def test_sessions_and_resets_do_not_share_shown_history(self) -> None:
        agent, _retriever = self._agent()
        for session_id in ("one", "two"):
            agent.reset(session_id, {})
            response = agent.respond(
                session_id,
                "I'm looking for shoes, but I'm still exploring.",
                1,
                3,
            )
            self.assertEqual(_ids(response), _product_ids(3))
        agent.reset("one", {})
        reset_response = agent.respond(
            "one",
            "I'm looking for shoes, but I'm still exploring.",
            1,
            3,
        )
        self.assertEqual(_ids(reset_response), _product_ids(3))

    def test_candidate_fault_fails_closed_without_slate_failure(self) -> None:
        agent, _retriever = self._agent()
        agent.reset("fault", {})
        with mock.patch(
            "conversational_search.service."
            "select_slate_with_intent_epoch_novelty",
            return_value=object(),
        ):
            response = agent.respond(
                "fault",
                "I'm looking for shoes, but I'm still exploring.",
                1,
                3,
            )
        self.assertEqual(_ids(response), _product_ids(3))
        self.assertEqual(agent.slate_health["failures"], 0)
        self.assertEqual(
            agent.intent_epoch_slate_health["validation_fallbacks"],
            1,
        )

    def test_candidate_telemetry_is_lazy_and_baseline_is_zero(self) -> None:
        candidate, _candidate_retriever = self._agent()
        baseline, _baseline_retriever = self._agent(
            policy=STAGNATION_AWARE_SLATE_POLICY
        )
        self.assertNotIn("_intent_epoch_slate_counts", vars(candidate))
        self.assertNotIn("_intent_epoch_slate_counts", vars(baseline))
        self.assertEqual(candidate.intent_epoch_slate_health["attempts"], 0)
        self.assertEqual(baseline.intent_epoch_slate_health["attempts"], 0)
        self.assertEqual(
            baseline.intent_epoch_slate_health["policy"],
            STAGNATION_AWARE_SLATE_POLICY.value,
        )


if __name__ == "__main__":
    unittest.main()
