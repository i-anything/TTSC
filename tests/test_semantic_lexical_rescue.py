from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from conversational_search.retrieval import (
    SHARED_DENSE_TERMS_RESCUE_POLICY,
    HybridRetriever,
    SemanticLexicalRescueStatus,
    SemanticLexicalRetrievalResult,
)


@dataclass(frozen=True)
class _Hit:
    parent_asin: str


class _Encoder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode_queries(self, texts: list[str], batch_size: int = 1) -> list[list[float]]:
        self.calls.extend(texts)
        return [[0.25, 0.75]]


class _DenseIndex:
    def __init__(self, ids: tuple[str, ...]) -> None:
        self.ids = ids
        self.calls = 0

    def search(self, query_vector: object, top_k: int) -> list[_Hit]:
        self.calls += 1
        return [_Hit(parent_asin) for parent_asin in self.ids]


class SemanticLexicalRescueTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.catalog_path = Path(temporary_directory.name) / "catalog.jsonl"
        products: list[dict[str, object]] = []
        for index in range(60):
            parent_asin = f"P{index:03d}"
            is_rescue_hit = index < 3
            products.append(
                {
                    "parent_asin": parent_asin,
                    "title": f"Product {index}",
                    "categories": ["Shoes" if index < 50 else "Boots"],
                    "features": [
                        "generic",
                        *(
                            [f"cushioned unique{index} 2025"]
                            if is_rescue_hit
                            else []
                        ),
                    ],
                    "details": {},
                    "store": "Brandword" if is_rescue_hit else f"Store {index}",
                    "description": ["generic"],
                }
            )
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def _retriever(
        self,
        dense_ids: tuple[str, ...] = ("P000", "P001", "P002", "P055"),
    ) -> tuple[HybridRetriever, _Encoder, _DenseIndex]:
        encoder = _Encoder()
        dense = _DenseIndex(dense_ids)
        return HybridRetriever(self.catalog_path, encoder, dense), encoder, dense

    def test_credible_bm25_support_skips_private_dense_and_retry(self) -> None:
        retriever, encoder, dense = self._retriever()

        result = retriever.search_with_trace(
            "semantic cotton shoe",
            "Shoes cushioned",
            top_k=3,
            use_dense=False,
            bm25_only_support_ids=("P000", "P001", "P002"),
            semantic_lexical_rescue_policy=SHARED_DENSE_TERMS_RESCUE_POLICY,
            semantic_rescue_category="Shoes",
        )

        self.assertIsInstance(result, SemanticLexicalRetrievalResult)
        self.assertIs(
            result.semantic_trace.status,
            SemanticLexicalRescueStatus.NOT_NEEDED,
        )
        self.assertEqual(encoder.calls, [])
        self.assertEqual(dense.calls, 0)
        self.assertEqual(result.semantic_trace.retry_count, 0)
        self.assertEqual(result.trace.dense_ids, ())

    def test_broad_category_only_support_uses_safe_terms_and_retries_once(self) -> None:
        retriever, encoder, dense = self._retriever()

        with mock.patch.object(retriever, "_bm25", wraps=retriever._bm25) as bm25:
            result = retriever.search_with_trace(
                "semantically soft running shoe",
                "Shoes opaquephrase",
                top_k=3,
                use_dense=False,
                bm25_only_support_ids=("P000", "P001", "P002"),
                semantic_lexical_rescue_policy=SHARED_DENSE_TERMS_RESCUE_POLICY,
                semantic_rescue_category="Shoes",
            )

        self.assertIs(
            result.semantic_trace.status,
            SemanticLexicalRescueStatus.APPLIED,
        )
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(dense.calls, 1)
        self.assertEqual(bm25.call_count, 2)
        retry_query = bm25.call_args_list[1].args[0]
        self.assertIn("cushioned", retry_query)
        self.assertNotIn("brandword", retry_query.casefold())
        self.assertNotIn("2025", retry_query)
        self.assertNotIn("generic", retry_query)
        self.assertEqual(result.semantic_trace.expansion_term_count, 1)
        self.assertEqual(result.semantic_trace.retry_count, 1)
        self.assertEqual(result.trace.dense_ids, ())
        self.assertNotIn("P055", result.trace.fused_ids)
        self.assertTrue(
            set(result.recommendations).intersection({"P000", "P001", "P002"})
        )

    def test_no_shared_safe_terms_fails_open_without_retry(self) -> None:
        retriever, encoder, dense = self._retriever(
            ("P003", "P004", "P005")
        )

        with mock.patch.object(retriever, "_bm25", wraps=retriever._bm25) as bm25:
            result = retriever.search_with_trace(
                "semantic query",
                "Shoes opaquephrase",
                top_k=2,
                use_dense=False,
                bm25_only_support_ids=("P003", "P004", "P005"),
                semantic_lexical_rescue_policy=SHARED_DENSE_TERMS_RESCUE_POLICY,
                semantic_rescue_category="Shoes",
            )

        self.assertIs(
            result.semantic_trace.status,
            SemanticLexicalRescueStatus.NO_SAFE_TERMS,
        )
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(dense.calls, 1)
        self.assertEqual(bm25.call_count, 1)
        self.assertEqual(result.semantic_trace.retry_count, 0)
        self.assertEqual(result.trace.bm25_ids, result.semantic_trace.base_bm25_ids)

    def test_retry_without_structural_support_keeps_original_bm25_route(self) -> None:
        retriever, _, _ = self._retriever()
        with (
            mock.patch.object(
                retriever,
                "_bm25",
                side_effect=[["P010"], ["P011"]],
            ) as bm25,
            mock.patch.object(
                retriever,
                "_safe_semantic_expansion_terms",
                return_value=("cushioned",),
            ),
        ):
            result = retriever.search_with_trace(
                "semantic query",
                "Shoes opaquephrase",
                top_k=2,
                use_dense=False,
                bm25_only_support_ids=("P000", "P001", "P002"),
                semantic_lexical_rescue_policy=SHARED_DENSE_TERMS_RESCUE_POLICY,
                semantic_rescue_category="Shoes",
            )

        self.assertEqual(bm25.call_count, 2)
        self.assertIs(
            result.semantic_trace.status,
            SemanticLexicalRescueStatus.RETRY_NO_STRUCTURAL_SUPPORT,
        )
        self.assertEqual(result.trace.bm25_ids, ("P010",))
        self.assertEqual(result.recommendations, ("P010",))
        self.assertEqual(result.trace.dense_ids, ())

    def test_enabled_policy_rejects_an_exposed_dense_route(self) -> None:
        retriever, _, _ = self._retriever()
        with self.assertRaisesRegex(ValueError, "exposed dense route"):
            retriever.search_with_trace(
                "semantic query",
                "Shoes",
                top_k=2,
                use_dense=True,
                bm25_only_support_ids=("P000",),
                semantic_lexical_rescue_policy=SHARED_DENSE_TERMS_RESCUE_POLICY,
                semantic_rescue_category="Shoes",
            )


if __name__ == "__main__":
    unittest.main()
