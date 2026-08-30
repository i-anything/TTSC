from __future__ import annotations

import unittest

from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    TOP3_STRUCTURAL_EXPOSURE_POLICY,
)
from conversational_search.orchestration import (
    BackendSnapshotToken,
    EXACT_RANKING_CACHE_CAPABILITY,
)
from conversational_search.protocol import DisclosureCard, ProductProtocolEvidence
from conversational_search.ranking import (
    CandidateDocument,
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    PROTOCOL_EVIDENCE_CAPABILITY,
    SHARED_DENSE_TERMS_RESCUE_POLICY,
    RetrievalResult,
    RetrievalTrace,
    SemanticLexicalRescueStatus,
    SemanticLexicalRescueTrace,
    SemanticLexicalRetrievalResult,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights


PLAUSIBLE_IDS = ("BLUE", "RED", "GREEN", "BLACK")
BASE_IDS = ("DISTRACTOR", *PLAUSIBLE_IDS)


class _FactorialRetriever:
    def __init__(self) -> None:
        self._snapshot_token = BackendSnapshotToken()
        self.search_kwargs: list[dict[str, object]] = []

    @property
    def ranking_cache_capability(self) -> object:
        return EXACT_RANKING_CACHE_CAPABILITY

    @property
    def snapshot_token(self) -> BackendSnapshotToken:
        return self._snapshot_token

    @property
    def protocol_evidence_capability(self) -> object:
        return PROTOCOL_EVIDENCE_CAPABILITY

    def protocol_category_exists(self, category: str) -> bool:
        return " ".join(category.split()).casefold() == "shoes"

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        return PLAUSIBLE_IDS if constraints == ("cotton",) else ()

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
        **kwargs: object,
    ) -> RetrievalResult:
        self.search_kwargs.append(dict(kwargs))
        semantic_policy = kwargs.get("semantic_lexical_rescue_policy")
        if semantic_policy is SHARED_DENSE_TERMS_RESCUE_POLICY:
            return SemanticLexicalRetrievalResult(
                recommendations=PLAUSIBLE_IDS[:top_k],
                trace=RetrievalTrace(
                    bm25_ids=PLAUSIBLE_IDS,
                    dense_ids=(),
                    fused_ids=PLAUSIBLE_IDS,
                    bm25_status="ok",
                    dense_status="skipped",
                    used_fallback=False,
                ),
                semantic_trace=SemanticLexicalRescueTrace(
                    status=SemanticLexicalRescueStatus.APPLIED,
                    base_bm25_ids=("DISTRACTOR",),
                    retry_bm25_ids=PLAUSIBLE_IDS,
                    base_bm25_status="ok",
                    retry_bm25_status="ok",
                    private_dense_status="ok",
                    private_dense_candidate_count=4,
                    compatible_dense_candidate_count=4,
                    expansion_term_count=1,
                    retry_count=1,
                ),
            )
        return RetrievalResult(
            recommendations=BASE_IDS[:top_k],
            trace=RetrievalTrace(
                bm25_ids=BASE_IDS,
                dense_ids=BASE_IDS,
                fused_ids=BASE_IDS,
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
            CandidateDocument(
                parent_asin,
                "leather shoe" if parent_asin == "DISTRACTOR" else "cotton shoe",
            )
            for parent_asin in parent_asins
        )

    def candidate_protocol_evidence(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[ProductProtocolEvidence, ...]:
        colors = {
            "BLUE": "blue",
            "RED": "red",
            "GREEN": "green",
            "BLACK": "black",
        }
        return tuple(
            ProductProtocolEvidence(
                parent_asin=parent_asin,
                coarse_category="Shoes",
                card=DisclosureCard(
                    f"{parent_asin} shoe",
                    (
                        "leather"
                        if parent_asin == "DISTRACTOR"
                        else "cotton",
                    ),
                    (
                        ()
                        if parent_asin == "DISTRACTOR"
                        else (f"color: {colors[parent_asin]}",)
                    ),
                ),
                text=(
                    "leather shoe"
                    if parent_asin == "DISTRACTOR"
                    else f"cotton {colors[parent_asin]} shoe"
                ),
            )
            for parent_asin in parent_asins
        )


def _ids(response: dict) -> tuple[str, ...]:
    return tuple(item["parent_asin"] for item in response["recommendations"])


class SemanticExposureFactorialTests(unittest.TestCase):
    def _agent(
        self,
        *,
        rescue: bool,
        gate: bool,
    ) -> tuple[ConversationalSearchAgent, _FactorialRetriever]:
        retriever = _FactorialRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            semantic_lexical_rescue_policy=(
                SHARED_DENSE_TERMS_RESCUE_POLICY
                if rescue
                else DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
            ),
            evidence_exposure_policy=(
                TOP3_STRUCTURAL_EXPOSURE_POLICY
                if gate
                else DISABLED_EVIDENCE_EXPOSURE_POLICY
            ),
        )
        return agent, retriever

    def test_baseline_rescue_gate_and_combined_are_independent_arms(self) -> None:
        outcomes: dict[str, tuple[dict, ConversationalSearchAgent, _FactorialRetriever]] = {}
        for name, rescue, gate in (
            ("baseline", False, False),
            ("rescue_only", True, False),
            ("gate_only", False, True),
            ("combined", True, True),
        ):
            agent, retriever = self._agent(rescue=rescue, gate=gate)
            agent.reset(name, {})
            response = agent.respond(
                name,
                "I'm looking for Shoes. A key requirement is: cotton.",
                1,
                10,
            )
            outcomes[name] = (response, agent, retriever)

        self.assertEqual(_ids(outcomes["baseline"][0]), PLAUSIBLE_IDS + ("DISTRACTOR",))
        self.assertEqual(_ids(outcomes["rescue_only"][0]), PLAUSIBLE_IDS)
        self.assertEqual(_ids(outcomes["gate_only"][0]), ())
        self.assertEqual(_ids(outcomes["combined"][0]), ())
        self.assertEqual(outcomes["gate_only"][0]["ask_attribute"], "color")
        self.assertEqual(outcomes["combined"][0]["ask_attribute"], "color")

        for name in ("baseline", "gate_only"):
            kwargs = outcomes[name][2].search_kwargs[0]
            self.assertNotIn("semantic_lexical_rescue_policy", kwargs)
            self.assertNotEqual(kwargs.get("use_dense"), False)
        for name in ("rescue_only", "combined"):
            kwargs = outcomes[name][2].search_kwargs[0]
            self.assertIs(kwargs.get("use_dense"), False)
            self.assertIs(
                kwargs.get("semantic_lexical_rescue_policy"),
                SHARED_DENSE_TERMS_RESCUE_POLICY,
            )
            self.assertEqual(
                outcomes[name][1].semantic_lexical_rescue_health["applied"],
                1,
            )
        for name in ("gate_only", "combined"):
            self.assertEqual(
                outcomes[name][1].evidence_exposure_health["question_withheld"],
                1,
            )

    def test_buying_only_gate_never_withholds_an_answer_only_browsing_session(self) -> None:
        retriever = _FactorialRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=(
                BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY
            ),
        )
        session_id = "answer-only-browsing"
        agent.reset(session_id, {})
        first = agent.respond(
            session_id,
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )
        self.assertIsNotNone(first["ask_attribute"])

        second = agent.respond(
            session_id,
            "For that, what matters is: cotton.",
            2,
            10,
        )

        self.assertGreater(len(second["recommendations"]), 0)
        self.assertEqual(agent.evidence_exposure_health["question_withheld"], 0)
        self.assertEqual(agent.evidence_exposure_health["unsafe_state"], 2)

    def test_buying_only_gate_still_withholds_for_explicit_buying(self) -> None:
        retriever = _FactorialRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=(
                BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY
            ),
        )
        session_id = "explicit-buying"
        agent.reset(session_id, {})

        response = agent.respond(
            session_id,
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertEqual(response["recommendations"], [])
        self.assertEqual(response["ask_attribute"], "color")
        self.assertEqual(agent.evidence_exposure_health["question_withheld"], 1)

    def test_buying_prefix_returns_top_three_and_still_asks(self) -> None:
        retriever = _FactorialRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
        )
        session_id = "explicit-buying-prefix"
        agent.reset(session_id, {})

        response = agent.respond(
            session_id,
            "I'm looking for Shoes. A key requirement is: cotton.",
            1,
            10,
        )

        self.assertEqual(_ids(response), PLAUSIBLE_IDS[:3])
        self.assertEqual(response["ask_attribute"], "color")
        self.assertEqual(
            agent.evidence_exposure_health["question_with_prefix"],
            1,
        )
        self.assertEqual(agent.evidence_exposure_health["withheld_turns"], 0)

    def test_buying_prefix_preserves_browsing_fail_open(self) -> None:
        retriever = _FactorialRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
            evidence_exposure_policy=BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
        )
        session_id = "answer-only-browsing-prefix"
        agent.reset(session_id, {})
        first = agent.respond(
            session_id,
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )
        second = agent.respond(
            session_id,
            "For that, what matters is: cotton.",
            2,
            10,
        )

        self.assertGreater(len(first["recommendations"]), 0)
        self.assertGreater(len(second["recommendations"]), 0)
        self.assertEqual(
            agent.evidence_exposure_health["question_with_prefix"],
            0,
        )
        self.assertEqual(agent.evidence_exposure_health["unsafe_state"], 2)


if __name__ == "__main__":
    unittest.main()
