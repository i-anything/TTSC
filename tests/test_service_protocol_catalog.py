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

    def _agent(self) -> ConversationalSearchAgent:
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
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
            protocol_catalog_policy=FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
            protocol_refutation_policy=(
                ELIGIBLE_CONTINUATION_REFUTATION_POLICY
            ),
            slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )

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
