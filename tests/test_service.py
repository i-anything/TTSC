from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from conversational_search.intent import (
    CANONICAL_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
)
from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    EXACT_RANKING_CACHE_CAPABILITY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    BackendSnapshotToken,
)
from conversational_search.ranking import (
    FUSED_ONLY_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    CandidateDocument,
)
from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.service import (
    ConversationalSearchAgent,
    _validate_catalog_pair,
)
from conversational_search.slates import (
    REPEAT_TOP_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
)
from conversational_search.strategy import RouteWeights


def _product_ids(count: int) -> tuple[str, ...]:
    return tuple(f"B{index:09d}" for index in range(1, count + 1))


class RecordingRetriever:
    def __init__(
        self,
        results: object = None,
        *,
        fail: bool = False,
        document_fail: bool = False,
        documents: dict[str, str] | None = None,
        fused_ids: tuple[str, ...] | None = None,
        documents_unavailable: bool = False,
    ) -> None:
        self.results = ["B000000001", "B000000002"] if results is None else results
        self.fail = fail
        self.document_fail = document_fail
        self.documents = documents or {}
        self.fused_ids = fused_ids
        self.documents_unavailable = documents_unavailable
        self.calls: list[tuple[str, str, int]] = []
        self.weight_calls: list[RouteWeights] = []
        self.document_calls: list[tuple[str, ...]] = []

    def search_with_trace(
        self,
        dense_query: str,
        lexical_query: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> object:
        self.calls.append((dense_query, lexical_query, top_k))
        self.weight_calls.append(route_weights)
        if self.fail:
            raise RuntimeError("retrieval failed")
        recommendations = (
            tuple(self.results)
            if isinstance(self.results, (list, tuple))
            else self.results
        )
        fused_ids = self.fused_ids or tuple(
            item for item in recommendations if isinstance(item, str) and item
        )
        return RetrievalResult(
            recommendations=recommendations,
            trace=RetrievalTrace(
                bm25_ids=fused_ids,
                dense_ids=fused_ids,
                fused_ids=fused_ids,
                bm25_status="ok" if fused_ids else "empty",
                dense_status="ok" if fused_ids else "empty",
                used_fallback=not fused_ids,
            ),
        )

    def candidate_documents(
        self,
        parent_asins: tuple[str, ...],
    ) -> tuple[CandidateDocument, ...]:
        self.document_calls.append(parent_asins)
        if self.document_fail:
            raise RuntimeError("candidate documents unavailable")
        if self.documents_unavailable:
            return ()
        return tuple(
            CandidateDocument(parent_asin, self.documents.get(parent_asin, ""))
            for parent_asin in parent_asins
        )


class CacheableRecordingRetriever(RecordingRetriever):
    """Controllable immutable-backend double for exact-reuse integration tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.ranking_cache_capability = EXACT_RANKING_CACHE_CAPABILITY
        self.snapshot_token = BackendSnapshotToken()
        self.bm25_status_override: str | None = None
        self.dense_status_override: str | None = None
        self.used_fallback_override: bool | None = None

    def search_with_trace(
        self,
        dense_query: str,
        lexical_query: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> object:
        result = super().search_with_trace(
            dense_query,
            lexical_query,
            top_k,
            route_weights=route_weights,
        )
        if not isinstance(result, RetrievalResult):
            return result
        trace = result.trace
        return RetrievalResult(
            recommendations=result.recommendations,
            trace=RetrievalTrace(
                bm25_ids=trace.bm25_ids,
                dense_ids=trace.dense_ids,
                fused_ids=trace.fused_ids,
                bm25_status=(
                    trace.bm25_status
                    if self.bm25_status_override is None
                    else self.bm25_status_override
                ),
                dense_status=(
                    trace.dense_status
                    if self.dense_status_override is None
                    else self.dense_status_override
                ),
                used_fallback=(
                    trace.used_fallback
                    if self.used_fallback_override is None
                    else self.used_fallback_override
                ),
            ),
        )


class ConversationalSearchAgentTest(unittest.TestCase):
    def test_intent_policy_is_wired_into_service_state_and_queries(self) -> None:
        message = "Show me some Shoes; I'm open to options."
        canonical_retriever = RecordingRetriever()
        robust_retriever = RecordingRetriever()
        canonical = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=canonical_retriever,
            intent_policy=CANONICAL_INTENT_POLICY,
        )
        robust = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=robust_retriever,
            intent_policy=ROBUST_INTENT_POLICY,
        )
        canonical.reset("canonical", {})
        robust.reset("robust", {})

        canonical.respond("canonical", message, 1, 10)
        robust.respond("robust", message, 1, 10)

        self.assertIsNone(canonical.session_state("canonical").category)
        self.assertEqual(robust.session_state("robust").category, "Shoes")
        self.assertNotEqual(
            canonical_retriever.calls[-1][:2],
            robust_retriever.calls[-1][:2],
        )

    def test_reset_is_required_and_replaces_prior_session_state(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)

        with self.assertRaises(RuntimeError):
            agent.respond("session", "hello", 1, 10)

        agent.reset("session", {})
        agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
            10,
        )
        self.assertIn("leather", retriever.calls[-1][0])

        agent.reset("session", {})
        agent.respond(
            "session",
            "I'm looking for Bags, but I'm still exploring.",
            1,
            10,
        )
        self.assertEqual(retriever.calls[-1][0], "Category: Bags")
        self.assertNotIn("leather", retriever.calls[-1][1])

    def test_interleaved_sessions_do_not_leak_intent(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)
        agent.reset("shoes", {})
        agent.reset("bags", {})

        agent.respond(
            "shoes",
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
            10,
        )
        agent.respond(
            "bags",
            "I'm looking for Bags, but I'm still exploring.",
            1,
            10,
        )
        agent.respond(
            "shoes",
            "For that, what matters is: waterproof.",
            2,
            10,
        )

        self.assertEqual(retriever.calls[1][0], "Category: Bags")
        self.assertIn("leather", retriever.calls[2][0])
        self.assertIn("waterproof", retriever.calls[2][0])
        self.assertNotIn("Bags", retriever.calls[2][0])

    def test_every_turn_retrieves_from_complete_updated_state(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Accessories Belts. Buckle closure",
            1,
            10,
        )
        self.assertEqual(first["ask_attribute"], "material")
        agent.respond(
            "session",
            "For that, what matters is: canvas.",
            2,
            10,
        )
        third = agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
            10,
        )

        override_dense, override_lexical, _ = retriever.calls[-1]
        self.assertNotIn("Buckle closure", override_dense)
        self.assertNotIn("Buckle closure", override_lexical)
        self.assertIn("canvas", override_dense)
        self.assertIn("leather", override_dense)
        # The color question from turn 2 was interrupted by the override, so
        # the active Phase 2 policy makes it eligible again.
        self.assertEqual(third["ask_attribute"], "color")

    def test_boundary_answer_is_not_added_to_query_and_question_advances(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)
        agent.reset("session", {})
        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            10,
        )

        self.assertEqual(first["ask_attribute"], "feature")
        self.assertEqual(second["ask_attribute"], "material")
        dense, lexical, _ = retriever.calls[-1]
        self.assertEqual(dense, "Category: Shoes")
        self.assertEqual(lexical, "Shoes")

    def test_unchanged_evidence_advances_recommendation_window(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(
            first["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": value} for value in product_ids[3:6]],
        )
        self.assertEqual(agent.slate_health["stagnant_turns"], 1)
        self.assertEqual(agent.slate_health["repeat_backfills"], 0)

    def test_new_requirement_resets_recommendation_window(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        third = agent.respond(
            "session",
            "For that, what matters is: leather.",
            3,
            3,
        )

        self.assertEqual(third["recommendations"], first["recommendations"])
        self.assertEqual(agent.slate_health["initializations"], 1)
        self.assertEqual(agent.slate_health["ranking_resets"], 1)

    def test_override_resets_even_when_rendered_query_is_equal(self) -> None:
        product_ids = _product_ids(8)
        retriever = RecordingRetriever(product_ids[:3], fused_ids=product_ids)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes. leather.",
            1,
            3,
        )
        agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        before_override_query = retriever.calls[-1][:2]
        third = agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            3,
            3,
        )

        self.assertEqual(retriever.calls[-1][:2], before_override_query)
        self.assertEqual(third["recommendations"], first["recommendations"])

    def test_exhausted_tail_backfills_deterministically(self) -> None:
        product_ids = _product_ids(5)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        expected = (product_ids[3], product_ids[4], product_ids[0])
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": value} for value in expected],
        )
        self.assertEqual(len({item["parent_asin"] for item in second["recommendations"]}), 3)
        self.assertEqual(agent.slate_health["repeat_backfills"], 1)

    def test_paging_replay_is_deterministic_and_reset_clears_cursor(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        messages = (
            "I'm looking for Shoes, but I'm still exploring.",
            "I don't have a preference for feature; please use your judgment.",
            "I don't have an additional preference for material.",
        )

        sequences: list[list[list[dict[str, str]]]] = []
        for _ in range(2):
            agent.reset("session", {})
            sequences.append(
                [
                    agent.respond("session", message, turn, 3)["recommendations"]
                    for turn, message in enumerate(messages, start=1)
                ]
            )

        self.assertEqual(sequences[0], sequences[1])

    def test_paging_cursor_is_session_local(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("first", {})
        agent.reset("second", {})

        first_page = agent.respond(
            "first",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        agent.respond(
            "first",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        other_session = agent.respond(
            "second",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )

        self.assertEqual(other_session["recommendations"], first_page["recommendations"])

    def test_reranking_failure_does_not_advance_or_reset_cursor(self) -> None:
        product_ids = _product_ids(8)
        retriever = RecordingRetriever(product_ids[:3], fused_ids=product_ids)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        retriever.document_fail = True
        failed = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        retriever.document_fail = False
        recovered = agent.respond(
            "session",
            "I don't have an additional preference for material.",
            3,
            3,
        )

        self.assertEqual(
            failed["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(
            recovered["recommendations"],
            [{"parent_asin": value} for value in product_ids[3:6]],
        )
        self.assertEqual(agent.slate_health["initializations"], 1)
        self.assertEqual(agent.slate_health["ranking_resets"], 0)
        self.assertEqual(agent.slate_health["stagnant_turns"], 1)

    def test_zero_top_k_does_not_consume_the_first_window(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        empty = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            0,
        )
        first_window = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(empty["recommendations"], [])
        self.assertEqual(
            first_window["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(agent.slate_health["initializations"], 1)
        self.assertEqual(agent.slate_health["ranking_resets"], 0)
        self.assertEqual(agent.slate_health["stagnant_turns"], 0)

    def test_repeat_top_policy_preserves_phase4_behavior(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=REPEAT_TOP_SLATE_POLICY,
        )
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.service.ranking_signature",
            side_effect=AssertionError("repeat-top must use the Phase 4 fast path"),
        ):
            first = agent.respond(
                "session",
                "I'm looking for Shoes, but I'm still exploring.",
                1,
                3,
            )
            second = agent.respond(
                "session",
                "I don't have a preference for feature; please use your judgment.",
                2,
                3,
            )

        self.assertEqual(second["recommendations"], first["recommendations"])
        self.assertEqual(agent.slate_health["attempts"], 0)

    def test_slate_failure_returns_phase4_order_without_mutating_cursor(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.service.ranking_signature",
            side_effect=RuntimeError("slate failed"),
        ):
            failed = agent.respond(
                "session",
                "I'm looking for Shoes, but I'm still exploring.",
                1,
                3,
            )
        recovered = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        expected = [{"parent_asin": value} for value in product_ids[:3]]
        self.assertEqual(failed["recommendations"], expected)
        self.assertEqual(recovered["recommendations"], expected)
        self.assertEqual(agent.slate_health["attempts"], 2)
        self.assertEqual(agent.slate_health["successes"], 1)
        self.assertEqual(agent.slate_health["failures"], 1)

    def test_top_k_change_resets_the_recommendation_window(self) -> None:
        product_ids = _product_ids(8)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(product_ids[:3], fused_ids=product_ids),
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            2,
        )
        resized = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(
            resized["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(agent.slate_health["ranking_resets"], 1)

    def test_exact_reuse_avoids_retrieval_and_reranking_but_advances_slate(
        self,
    ) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(
            first["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(
            second["recommendations"],
            [{"parent_asin": value} for value in product_ids[3:6]],
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.document_calls, [product_ids])
        health = agent.orchestration_health
        self.assertEqual(health["hits"], 1)
        self.assertEqual(health["retrievals_avoided"], 1)
        self.assertEqual(health["reranks_avoided"], 1)
        self.assertEqual(agent.ranking_health["attempts"], 1)
        self.assertEqual(agent.slate_health["stagnant_turns"], 1)

    def test_changed_requirement_forces_exact_reuse_miss(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: leather.",
            1,
            3,
        )
        agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: canvas.",
            2,
            3,
        )

        self.assertEqual(len(retriever.calls), 2)
        self.assertNotEqual(retriever.calls[0][:2], retriever.calls[1][:2])
        self.assertEqual(retriever.weight_calls[0], retriever.weight_calls[1])
        self.assertEqual(agent.orchestration_health["dependency_misses"], 1)

    def test_provenance_and_route_weight_change_force_miss_for_equal_query(
        self,
    ) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
        agent.reset("session", {})

        agent.respond("session", "I'm looking for Shoes. leather.", 1, 3)
        agent.respond(
            "session",
            "Actually, ignore my earlier preference. What I need is: leather.",
            2,
            3,
        )

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[0][:2], retriever.calls[1][:2])
        self.assertNotEqual(retriever.weight_calls[0], retriever.weight_calls[1])
        self.assertEqual(agent.orchestration_health["dependency_misses"], 1)

    def test_top_k_change_reuses_full_ranking_and_resets_slate(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:2],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            2,
        )
        resized = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(
            resized["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(retriever.document_calls), 1)
        self.assertEqual(agent.orchestration_health["hits"], 1)
        self.assertEqual(agent.slate_health["ranking_resets"], 1)

    def test_zero_top_k_skips_query_without_consuming_cache_or_slate(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        agent.reset("session", {})

        empty = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            0,
        )
        first_window = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(empty["recommendations"], [])
        self.assertEqual(
            first_window["recommendations"],
            [{"parent_asin": value} for value in product_ids[:3]],
        )
        self.assertEqual(len(retriever.calls), 1)
        health = agent.orchestration_health
        self.assertEqual(health["skips"], 1)
        self.assertEqual(health["lookups"], 1)
        self.assertEqual(agent.slate_health["attempts"], 1)
        self.assertEqual(agent.slate_health["initializations"], 1)

    def test_reset_invalidates_exact_reuse_entry(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
        agent.reset("session", {})
        initial = "I'm looking for Shoes, but I'm still exploring."

        agent.respond("session", initial, 1, 3)
        agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        agent.reset("session", {})
        agent.respond("session", initial, 1, 3)

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(agent.orchestration_health["hits"], 1)
        self.assertEqual(agent.orchestration_health["reset_invalidations"], 1)

    def test_backend_snapshot_change_invalidates_then_recaches(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        retriever.snapshot_token = BackendSnapshotToken()
        agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )
        agent.respond(
            "session",
            "I don't have an additional preference for material.",
            3,
            3,
        )

        self.assertEqual(len(retriever.calls), 2)
        health = agent.orchestration_health
        self.assertEqual(health["backend_invalidations"], 1)
        self.assertEqual(health["hits"], 1)
        self.assertEqual(health["stores"], 2)

    def test_failed_or_unsafe_rankings_never_cache_and_recover(self) -> None:
        product_ids = _product_ids(8)
        cases = (
            "retrieval_failure",
            "document_failure",
            "fallback",
            "route_error",
        )
        for case in cases:
            with self.subTest(case=case):
                retriever = CacheableRecordingRetriever(
                    product_ids[:3],
                    fused_ids=product_ids,
                )
                if case == "retrieval_failure":
                    retriever.fail = True
                elif case == "document_failure":
                    retriever.document_fail = True
                elif case == "fallback":
                    retriever.used_fallback_override = True
                else:
                    retriever.bm25_status_override = "error"
                agent = ConversationalSearchAgent(
                    "unused.jsonl",
                    retriever=retriever,
                    orchestration_policy=(
                        EXACT_RANKING_REUSE_ORCHESTRATION_POLICY
                    ),
                )
                agent.reset("session", {})

                agent.respond(
                    "session",
                    "I'm looking for Shoes, but I'm still exploring.",
                    1,
                    3,
                )
                self.assertEqual(agent.orchestration_health["stores"], 0)
                self.assertEqual(agent.orchestration_health["entries"], 0)

                retriever.fail = False
                retriever.document_fail = False
                retriever.used_fallback_override = None
                retriever.bm25_status_override = None
                recovered = agent.respond(
                    "session",
                    "I don't have a preference for feature; please use your judgment.",
                    2,
                    3,
                )
                reused = agent.respond(
                    "session",
                    "I don't have an additional preference for material.",
                    3,
                    3,
                )

                self.assertTrue(recovered["recommendations"])
                self.assertTrue(reused["recommendations"])
                self.assertEqual(len(retriever.calls), 2)
                health = agent.orchestration_health
                self.assertEqual(health["stores"], 1)
                self.assertEqual(health["hits"], 1)
                self.assertEqual(health["entries"], 1)

    def test_untrusted_stateful_backend_cannot_opt_into_reuse_by_token_alone(
        self,
    ) -> None:
        class UntrustedRetriever(RecordingRetriever):
            ranking_cache_capability = object()

            def __init__(self, *args: object, **kwargs: object) -> None:
                super().__init__(*args, **kwargs)
                self.snapshot_reads = 0

            @property
            def snapshot_token(self) -> BackendSnapshotToken:
                self.snapshot_reads += 1
                return BackendSnapshotToken()

        product_ids = _product_ids(8)
        retriever = UntrustedRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        # A property with the right-looking name and an opaque token is not the
        # explicit immutable/full-pool capability required by production.
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            1,
        )
        retriever.fail = True
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertTrue(first["recommendations"])
        self.assertEqual(second["recommendations"], [])
        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(agent.orchestration_health["lookups"], 0)
        self.assertEqual(agent.orchestration_health["stores"], 0)
        self.assertEqual(agent.orchestration_health["hits"], 0)
        self.assertEqual(retriever.snapshot_reads, 0)

    def test_repeat_top_policy_reuses_exact_ranked_pool(self) -> None:
        product_ids = _product_ids(8)
        retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=REPEAT_TOP_SLATE_POLICY,
        )
        agent.reset("session", {})

        first = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            3,
        )
        second = agent.respond(
            "session",
            "I don't have a preference for feature; please use your judgment.",
            2,
            3,
        )

        self.assertEqual(second["recommendations"], first["recommendations"])
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(len(retriever.document_calls), 1)
        self.assertEqual(agent.orchestration_health["hits"], 1)
        self.assertEqual(agent.slate_health["attempts"], 0)

    def test_exact_reuse_matches_always_search_over_synthetic_trajectory(
        self,
    ) -> None:
        product_ids = _product_ids(12)
        baseline_retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        candidate_retriever = CacheableRecordingRetriever(
            product_ids[:3],
            fused_ids=product_ids,
        )
        baseline = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=baseline_retriever,
            orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        candidate = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=candidate_retriever,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
            slate_policy=STAGNATION_AWARE_SLATE_POLICY,
        )
        baseline.reset("session", {})
        candidate.reset("session", {})
        trajectory = (
            ("I'm looking for Shoes. leather.", 3),
            (
                "Actually, ignore my earlier preference. "
                "What I need is: leather.",
                3,
            ),
            (
                "I don't have a preference for feature; "
                "please use your judgment.",
                3,
            ),
            ("For that, what matters is: black.", 3),
            ("I don't have an additional preference for size.", 2),
            (
                "Those options are not quite right yet. "
                "Ask me about one specific attribute.",
                2,
            ),
        )

        for turn, (message, top_k) in enumerate(trajectory, start=1):
            baseline_response = baseline.respond(
                "session", message, turn, top_k
            )
            candidate_response = candidate.respond(
                "session", message, turn, top_k
            )
            self.assertEqual(candidate_response, baseline_response)
            self.assertEqual(
                candidate.session_state("session"),
                baseline.session_state("session"),
            )
            self.assertEqual(
                candidate.slate_state("session"),
                baseline.slate_state("session"),
            )

        self.assertEqual(len(baseline_retriever.calls), len(trajectory))
        self.assertLess(
            len(candidate_retriever.calls),
            len(baseline_retriever.calls),
        )
        self.assertGreater(candidate.orchestration_health["hits"], 0)

    def test_response_is_sanitized_and_turn_ten_never_asks(self) -> None:
        retriever = RecordingRetriever(
            [
                " B000000001 ",
                "B000000001",
                {"parent_asin": "B000000002"},
                None,
                {"wrong": "B000000003"},
            ]
        )
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)
        agent.reset("session", {})
        response = agent.respond("session", "plain useful text", 10, 100)

        self.assertEqual(response["ask_attribute"], None)
        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": "B000000001"},
                {"parent_asin": "B000000002"},
            ],
        )
        self.assertEqual(response["usage"], {"prompt_tokens": 0, "completion_tokens": 0})
        self.assertEqual(retriever.calls[0][2], 10)

    def test_retrieval_failure_still_returns_a_schema_safe_clarification(self) -> None:
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(fail=True),
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )

        self.assertIsInstance(response["message"], str)
        self.assertEqual(response["ask_attribute"], "feature")
        self.assertEqual(response["recommendations"], [])

    def test_default_fusion_policy_receives_updated_intent_each_turn(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
        )
        agent.reset("session", {})

        agent.respond(
            "session",
            "I'm looking for Shoes, but I'm still exploring.",
            1,
            10,
        )
        agent.respond(
            "session",
            "For that, what matters is: waterproof.",
            2,
            10,
        )

        self.assertEqual(retriever.weight_calls[0], RouteWeights(bm25=0.4, dense=0.6))
        self.assertAlmostEqual(retriever.weight_calls[1].bm25, 0.4 + 0.2 / 3.0)
        self.assertAlmostEqual(retriever.weight_calls[1].dense, 1.0 - (0.4 + 0.2 / 3.0))

    def test_stage_a_reranks_the_union_and_reports_label_free_health(self) -> None:
        retriever = RecordingRetriever(
            ["B000000001"],
            documents={
                "B000000001": "plain shoe",
                "B000000002": "waterproof trail shoe",
            },
            fused_ids=("B000000001", "B000000002"),
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond(
            "session",
            "I'm looking for Shoes. A key requirement is: waterproof.",
            1,
            1,
        )

        self.assertEqual(
            response["recommendations"],
            [{"parent_asin": "B000000002"}],
        )
        self.assertEqual(
            retriever.document_calls,
            [("B000000001", "B000000002")],
        )
        self.assertEqual(
            agent.ranking_health,
            {
                "policy": "stage_a",
                "attempts": 1,
                "successes": 1,
                "failures": 0,
                "unavailable_skips": 0,
            },
        )

    def test_stage_a_failure_returns_the_exact_incumbent_order(self) -> None:
        retriever = RecordingRetriever(document_fail=True)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": "B000000001"},
                {"parent_asin": "B000000002"},
            ],
        )
        self.assertEqual(agent.ranking_health["failures"], 1)

    def test_stage_a_scoring_failure_returns_the_exact_incumbent_order(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        agent.reset("session", {})

        with mock.patch(
            "conversational_search.service.rerank_stage_a",
            side_effect=RuntimeError("scorer failed"),
        ):
            response = agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": "B000000001"},
                {"parent_asin": "B000000002"},
            ],
        )
        health = agent.ranking_health
        self.assertEqual(health["failures"], 1)
        self.assertEqual(
            health["attempts"],
            health["successes"] + health["failures"] + health["unavailable_skips"],
        )

    def test_stage_a_unavailable_documents_are_an_expected_skip(self) -> None:
        retriever = RecordingRetriever(documents_unavailable=True)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=STAGE_A_RANKING_POLICY,
        )
        agent.reset("session", {})

        response = agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(
            response["recommendations"],
            [
                {"parent_asin": "B000000001"},
                {"parent_asin": "B000000002"},
            ],
        )
        self.assertEqual(agent.ranking_health["failures"], 0)
        self.assertEqual(agent.ranking_health["unavailable_skips"], 1)

    def test_fused_only_policy_does_not_read_candidate_documents(self) -> None:
        retriever = RecordingRetriever(document_fail=True)
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            ranking_policy=FUSED_ONLY_RANKING_POLICY,
        )
        agent.reset("session", {})

        agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(retriever.document_calls, [])
        self.assertEqual(agent.ranking_health["attempts"], 0)

    def test_default_ranking_policy_is_promoted_stage_a(self) -> None:
        retriever = RecordingRetriever()
        agent = ConversationalSearchAgent("unused.jsonl", retriever=retriever)
        agent.reset("session", {})

        agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(agent.ranking_health["policy"], "stage_a")
        self.assertEqual(agent.ranking_health["successes"], 1)

    def test_default_slate_policy_is_promoted_stagnation_aware(self) -> None:
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=RecordingRetriever(),
        )
        agent.reset("session", {})

        agent.respond("session", "waterproof shoes", 1, 10)

        self.assertEqual(agent.slate_health["policy"], "stagnation_aware")
        self.assertEqual(agent.slate_health["successes"], 1)

    def test_dense_catalog_identity_rejects_stale_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            payload = b'{"parent_asin":"B000000001"}\n'
            catalog_path.write_bytes(payload)
            dense_index = SimpleNamespace(
                row_count=1,
                manifest={
                    "catalog": {
                        "rows": 1,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            )

            _validate_catalog_pair(catalog_path, dense_index)
            catalog_path.write_bytes(payload + b"stale")
            with self.assertRaisesRegex(ValueError, "catalog checksum"):
                _validate_catalog_pair(catalog_path, dense_index)

    def test_default_agent_catalog_path_is_working_directory_independent(self) -> None:
        from starter.agent import Agent, DEFAULT_CATALOG_PATH

        self.assertTrue(DEFAULT_CATALOG_PATH.is_absolute())
        with mock.patch.object(
            ConversationalSearchAgent,
            "__init__",
            return_value=None,
        ) as initialize:
            Agent()
        initialize.assert_called_once_with(
            DEFAULT_CATALOG_PATH,
            orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        )


if __name__ == "__main__":
    unittest.main()
