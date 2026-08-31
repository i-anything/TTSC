from __future__ import annotations

import unittest

from conversational_search.intent import IntentState, Requirement
from conversational_search.ranking import CandidateDocument
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.retrieval_routing import (
    ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY,
    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
    RetrievalRouteMode,
    RetrievalRouteReason,
    plan_retrieval_route,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights


class _EvidenceRetriever:
    protocol_evidence_capability = PROTOCOL_EVIDENCE_CAPABILITY

    def __init__(
        self,
        *,
        category_exists: bool = True,
        exact_count: int = 1,
        support_ids: tuple[str, ...] = ("B000000001",),
        fail: bool = False,
    ) -> None:
        self.category_exists = category_exists
        self.exact_count = exact_count
        self.support_ids = support_ids
        self.fail = fail

    def protocol_category_exists(self, category: str) -> bool:
        if self.fail:
            raise RuntimeError("evidence unavailable")
        return self.category_exists

    def protocol_exact_constraint_count(
        self,
        category: str,
        constraints: tuple[str, ...],
    ) -> int:
        if self.fail:
            raise RuntimeError("evidence unavailable")
        return self.exact_count

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        if self.fail:
            raise RuntimeError("evidence unavailable")
        return self.support_ids


class _RoutingServiceRetriever(_EvidenceRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.search_options: list[dict[str, object]] = []

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
        **options: object,
    ) -> RetrievalResult:
        self.search_options.append(dict(options))
        use_dense = bool(options.get("use_dense", True))
        dense_ids = ("B000000002",) if use_dense else ()
        fused_ids = ("B000000001", *dense_ids)
        return RetrievalResult(
            recommendations=fused_ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=("B000000001",),
                dense_ids=dense_ids,
                fused_ids=fused_ids,
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
            CandidateDocument(parent_asin, "rubber sole shoe")
            for parent_asin in parent_asins
        )


def _state(
    *requirements: Requirement,
    category: str | None = "Shoes",
    excluded: tuple[str, ...] = (),
) -> IntentState:
    return IntentState(
        category=category,
        requirements=tuple(requirements),
        excluded=excluded,
        last_turn=1,
    )


def _requirement(
    value: str = "rubber sole",
    *,
    source: str = "initial_explicit",
    attribute: str | None = "feature",
) -> Requirement:
    return Requirement(
        value,
        source,  # type: ignore[arg-type]
        1,
        attribute,
    )


class RetrievalRoutingTests(unittest.TestCase):
    def test_always_hybrid_policy_never_reads_catalog_evidence(self) -> None:
        plan = plan_retrieval_route(
            ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY,
            _state(_requirement()),
            _EvidenceRetriever(fail=True),
            intent_cacheable=True,
        )

        self.assertIs(plan.mode, RetrievalRouteMode.HYBRID)
        self.assertIs(
            plan.reason,
            RetrievalRouteReason.POLICY_REQUIRES_HYBRID,
        )
        self.assertTrue(plan.use_dense)
        self.assertEqual(plan.structural_support_ids, ())

    def test_exact_structured_intent_uses_bm25_first_with_catalog_support(
        self,
    ) -> None:
        plan = plan_retrieval_route(
            SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
            _state(_requirement()),
            _EvidenceRetriever(),
            intent_cacheable=True,
        )

        self.assertIs(plan.mode, RetrievalRouteMode.BM25_FIRST)
        self.assertIs(
            plan.reason,
            RetrievalRouteReason.EXACT_STRUCTURAL_SUPPORT,
        )
        self.assertFalse(plan.use_dense)
        self.assertEqual(plan.structural_support_ids, ("B000000001",))

    def test_semantic_uncertainty_always_fails_open_to_hybrid(self) -> None:
        cases = (
            (
                _state(_requirement()),
                False,
                RetrievalRouteReason.INTENT_FALLBACK,
            ),
            (
                _state(_requirement(), category=None),
                True,
                RetrievalRouteReason.MISSING_CATEGORY,
            ),
            (
                _state(),
                True,
                RetrievalRouteReason.NO_EXACT_REQUIREMENTS,
            ),
            (
                _state(_requirement(source="initial_tentative")),
                True,
                RetrievalRouteReason.SEMANTIC_OR_SOFT_REQUIREMENT,
            ),
            (
                _state(_requirement(source="override")),
                True,
                RetrievalRouteReason.OVERRIDE_OR_EXCLUSION,
            ),
            (
                _state(_requirement(), excluded=("leather",)),
                True,
                RetrievalRouteReason.OVERRIDE_OR_EXCLUSION,
            ),
            (
                _state(_requirement(attribute="other")),
                True,
                RetrievalRouteReason.UNTYPED_REQUIREMENT,
            ),
        )
        for state, cacheable, reason in cases:
            with self.subTest(reason=reason.value):
                plan = plan_retrieval_route(
                    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
                    state,
                    _EvidenceRetriever(),
                    intent_cacheable=cacheable,
                )
                self.assertIs(plan.mode, RetrievalRouteMode.HYBRID)
                self.assertIs(plan.reason, reason)

    def test_catalog_uncertainty_always_fails_open_to_hybrid(self) -> None:
        state = _state(_requirement())
        cases = (
            (
                object(),
                RetrievalRouteReason.EVIDENCE_CAPABILITY_UNAVAILABLE,
            ),
            (
                _EvidenceRetriever(category_exists=False),
                RetrievalRouteReason.CATEGORY_UNRECOGNIZED,
            ),
            (
                _EvidenceRetriever(exact_count=0),
                RetrievalRouteReason.PARTIAL_EXACT_SUPPORT,
            ),
            (
                _EvidenceRetriever(support_ids=()),
                RetrievalRouteReason.NO_JOINT_STRUCTURAL_SUPPORT,
            ),
            (
                _EvidenceRetriever(
                    support_ids=(
                        "B000000001",
                        "B000000002",
                        "B000000003",
                        "B000000004",
                    )
                ),
                RetrievalRouteReason.AMBIGUOUS_STRUCTURAL_SUPPORT,
            ),
            (
                _EvidenceRetriever(fail=True),
                RetrievalRouteReason.EVIDENCE_ERROR,
            ),
        )
        for retriever, reason in cases:
            with self.subTest(reason=reason.value):
                plan = plan_retrieval_route(
                    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
                    state,
                    retriever,
                    intent_cacheable=True,
                )
                self.assertIs(plan.mode, RetrievalRouteMode.HYBRID)
                self.assertIs(plan.reason, reason)

    def test_every_constraint_must_have_exact_and_joint_support(self) -> None:
        state = _state(
            _requirement("rubber sole"),
            _requirement("color: blue", source="answer", attribute="color"),
        )

        partial = plan_retrieval_route(
            SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
            state,
            _EvidenceRetriever(exact_count=1),
            intent_cacheable=True,
        )
        supported = plan_retrieval_route(
            SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
            state,
            _EvidenceRetriever(exact_count=2),
            intent_cacheable=True,
        )

        self.assertIs(
            partial.reason,
            RetrievalRouteReason.PARTIAL_EXACT_SUPPORT,
        )
        self.assertIs(supported.mode, RetrievalRouteMode.BM25_FIRST)

    def test_service_passes_structural_gate_and_reports_route_health(self) -> None:
        retriever = _RoutingServiceRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            retrieval_routing_policy=(
                SMART_HYBRID_RETRIEVAL_ROUTING_POLICY
            ),
        )
        agent.reset("buying", {})
        agent.reset("browsing", {})

        agent.respond(
            "buying",
            "I'm looking for Shoes. A key requirement is: rubber sole.",
            1,
            10,
        )
        agent.respond(
            "browsing",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )

        self.assertEqual(
            retriever.search_options[0],
            {
                "use_dense": False,
                "bm25_only_support_ids": ("B000000001",),
                "bm25_only_requires_all_support": True,
            },
        )
        self.assertEqual(retriever.search_options[1], {})
        health = agent.retrieval_routing_health
        self.assertEqual(health["planned_bm25_first"], 1)
        self.assertEqual(health["planned_hybrid"], 1)
        self.assertEqual(health["executed_bm25_only"], 1)
        self.assertEqual(health["executed_hybrid"], 1)


if __name__ == "__main__":
    unittest.main()
