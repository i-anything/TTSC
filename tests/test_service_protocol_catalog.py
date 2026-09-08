from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conversational_search.exposure_policy import (
    PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
    PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
)
from conversational_search.protocol_index import (
    ELIGIBLE_CONTINUATION_REFUTATION_POLICY,
    FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import HybridRetriever
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY


class ServiceProtocolCatalogTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog_path = Path(directory.name) / "catalog.jsonl"
        products = (
            {
                "parent_asin": "A",
                "title": "Popular shoe",
                "categories": ["Shoes"],
                "features": ["waterproof", "wide", "warm"],
                "details": {},
                "description": [],
                "store": "One",
                "rating_number": 100,
            },
            {
                "parent_asin": "B",
                "title": "Less popular shoe",
                "categories": ["Shoes"],
                "features": ["waterproof", "wide", "warm"],
                "details": {},
                "description": [],
                "store": "Two",
                "rating_number": 10,
            },
            {
                "parent_asin": "C",
                "title": "Different shoe",
                "categories": ["Shoes"],
                "features": ["water resistant", "narrow", "cool"],
                "details": {},
                "description": [],
                "store": "Three",
                "rating_number": 1,
            },
        )
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def _agent(
        self, *, normalize_language: bool = False, metric_aware: bool = False,
    ) -> ConversationalSearchAgent:
        retriever = HybridRetriever(
            self.catalog_path,
            None,
            None,
            protocol_evidence=True,
        )
        self.addCleanup(retriever._connection.close)
        return ConversationalSearchAgent(
            self.catalog_path,
            retriever=retriever,
            normalize_language=normalize_language,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=(
                PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY
                if metric_aware else PROTOCOL_POSTERIOR_EXPOSURE_POLICY
            ),
            protocol_catalog_policy=FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
            protocol_refutation_policy=(
                ELIGIBLE_CONTINUATION_REFUTATION_POLICY
            ),
            slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )

    def test_rollout_question_reaches_service_and_preserves_override_guard(self) -> None:
        products = [
            {
                "parent_asin": f"P{i}", "title": f"Shoe {i}",
                "categories": ["Shoes"],
                "features": ["cotton", "color: black", f"special property {i}", "warm"],
                "rating_number": 100 - i,
            }
            for i in range(8)
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        agent = self._agent(metric_aware=True)
        agent.reset("browse", {})
        first = agent.respond("browse", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(len(first["recommendations"]), 1)
        second = agent.respond("browse", "For that, what matters is: special property 7; warm.", 2, 10)
        self.assertEqual(second["recommendations"], [{"parent_asin": "P7"}])
        self.assertIsNone(second["ask_attribute"])

        agent.reset("override", {})
        pending = agent.respond("override", "I'm looking for Shoes. warm", 1, 10)
        self.assertEqual(pending["ask_attribute"], "other")
        self.assertFalse(agent._protocol_pending_refutable["override"])

    def test_continuation_refutes_only_the_prior_score_eligible_product(self) -> None:
        agent = self._agent()
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: waterproof.",
            1,
            10,
        )
        second = agent.respond(
            "session",
            "For that, what matters is: wide; warm.",
            2,
            10,
        )

        self.assertEqual(first["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(first["ask_attribute"], "other")
        self.assertEqual(second["recommendations"], [{"parent_asin": "B"}])
        self.assertIsNone(second["ask_attribute"])
        self.assertEqual(agent._protocol_refuted_ids["session"], ("A",))

    def test_paraphrased_transcript_preserves_actions_state_and_refutation(self) -> None:
        agent = self._agent(normalize_language=True)
        canonical = (
            "I'm looking for Shoes. warm",
            "For that, what matters is: waterproof; wide.",
            "Actually, ignore my earlier preference. What I need is: waterproof.",
            "For that, what matters is: warm.",
        )
        paraphrased = (
            "Please help me find Shoes. One tentative preference is: warm.",
            "My priorities are: waterproof; wide.",
            "I changed my mind. Please prioritize: waterproof.",
            "For that attribute, I prefer warm.",
        )
        agent.reset("canonical", {})
        agent.reset("paraphrased", {})
        for turn, (original, variant) in enumerate(zip(canonical, paraphrased), 1):
            with self.subTest(turn=turn):
                expected = agent.respond("canonical", original, turn, 10)
                actual = agent.respond("paraphrased", variant, turn, 10)
                self.assertEqual(actual, expected)
                self.assertEqual(agent.session_state("canonical"), agent.session_state("paraphrased"))
                self.assertEqual(agent._protocol_events["canonical"], agent._protocol_events["paraphrased"])
                self.assertEqual(agent._protocol_refuted_ids["canonical"], agent._protocol_refuted_ids["paraphrased"])
                trace = agent.last_action_trace("paraphrased")
                self.assertTrue(trace["language_normalized"])
                self.assertEqual(trace["protocol_mode"], "applied")
                self.assertNotEqual(trace["planner_outcome"], "candidate_or_evidence_error")
                self.assertEqual(trace["presented_width"], len(actual["recommendations"]))
                self.assertEqual(trace["question"], actual["ask_attribute"])

    def test_recognized_envelope_does_not_establish_catalog_support(self) -> None:
        agent = self._agent(normalize_language=True)
        agent.reset("unsupported", {})
        agent.respond("unsupported", "I need Shoes. It must be invisible.", 1, 10)
        self.assertFalse(agent._protocol_pending_refutable["unsupported"])
        agent.respond("unsupported", "My priorities are: wide; warm.", 2, 10)
        self.assertEqual(agent._protocol_refuted_ids["unsupported"], ())

    def test_unsupported_turn_cannot_rejoin_a_partial_protocol_transcript(self) -> None:
        agent = self._agent(normalize_language=True)
        agent.reset("free", {})
        agent.respond("free", "I need Shoes. It must be waterproof.", 1, 10)
        agent.respond("free", "Can these shoes be repaired locally?", 2, 10)
        agent.respond("free", "My priorities are: wide; warm.", 3, 10)
        self.assertFalse(agent._protocol_consistency["free"])
        self.assertFalse(agent._protocol_pending_refutable["free"])
        self.assertEqual(agent._protocol_refuted_ids["free"], ())

    def test_pre_override_products_are_not_refuted(self) -> None:
        agent = self._agent()
        agent.reset("override", {})

        first = agent.respond(
            "override",
            "I'm looking for Shoes. warm",
            1,
            10,
        )
        second = agent.respond(
            "override",
            "For that, what matters is: waterproof; wide.",
            2,
            10,
        )
        third = agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: waterproof.",
            3,
            10,
        )
        refuted_after_override = agent._protocol_refuted_ids["override"]
        fourth = agent.respond(
            "override",
            "For that, what matters is: warm.",
            4,
            10,
        )

        self.assertEqual(first["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(second["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(refuted_after_override, ())
        self.assertEqual(agent._protocol_refuted_ids["override"], ("A",))
        self.assertEqual(third["recommendations"], [{"parent_asin": "A"}])
        self.assertEqual(fourth["recommendations"], [{"parent_asin": "B"}])

    def test_metric_aware_policy_requires_refutation(self) -> None:
        retriever = HybridRetriever(
            self.catalog_path,
            None,
            None,
            protocol_evidence=True,
        )
        self.addCleanup(retriever._connection.close)

        with self.assertRaisesRegex(ValueError, "requires continuation refutation"):
            ConversationalSearchAgent(
                self.catalog_path,
                retriever=retriever,
                evidence_exposure_policy=(
                    PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY
                ),
                protocol_catalog_policy=FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
            )


if __name__ == "__main__":
    unittest.main()
