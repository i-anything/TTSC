from __future__ import annotations

import json
import tempfile
import unittest
import sqlite3
from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
from unittest import mock

from conversational_search.retrieval import (
    HybridRetriever,
    MAX_CANDIDATE_DOCUMENTS,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.ranking import CandidateDocument
from conversational_search.strategy import RouteWeights


@dataclass(frozen=True)
class FakeHit:
    parent_asin: str


class FakeEncoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[list[str], int]] = []

    def encode_queries(self, texts: list[str], batch_size: int = 1) -> list[list[float]]:
        self.calls.append((list(texts), batch_size))
        if self.fail:
            raise RuntimeError("encoder unavailable")
        return [[0.25, 0.75]]


class FakeDenseIndex:
    def __init__(self, hits: list[object], *, fail: bool = False) -> None:
        self.hits = hits
        self.fail = fail
        self.calls: list[tuple[object, int]] = []

    def search(self, query_vector: object, top_k: int) -> list[object]:
        self.calls.append((query_vector, top_k))
        if self.fail:
            raise RuntimeError("dense index unavailable")
        return self.hits


class HybridRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "B000000001",
                "title": "Alpha trail shoe",
                "categories": ["Shoes"],
                "features": [],
                "details": {},
                "store": "First",
                "description": [],
            },
            {
                "parent_asin": "B000000002",
                "title": "Second product",
                "categories": ["Shoes"],
                "features": [],
                "details": {},
                "store": "Second",
                "description": ["Alpha"],
            },
            {
                "parent_asin": "B000000003",
                "title": "Third product",
                "categories": ["Boots"],
                "features": ["Waterproof"],
                "details": {},
                "store": "Third",
                "description": [],
            },
            {
                "parent_asin": "B000000004",
                "title": "Fourth product",
                "categories": ["Accessories"],
                "features": [],
                "details": {},
                "store": "Fourth",
                "description": [],
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def test_encodes_once_requests_top_100_and_fuses_equal_weight_rrf(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex(
            [
                FakeHit("B000000002"),
                FakeHit("B000000003"),
                FakeHit("B000000001"),
                FakeHit("B000000002"),
                FakeHit("NOT_IN_CATALOG"),
            ]
        )
        retriever = HybridRetriever(self.catalog_path, encoder, dense)

        with mock.patch.object(retriever, "_bm25", wraps=retriever._bm25) as bm25:
            result = retriever.search("dense alpha query", "alpha", top_k=3)

        self.assertEqual(result, ["B000000002", "B000000001", "B000000003"])
        bm25.assert_called_once_with("alpha")
        self.assertEqual(encoder.calls, [(["dense alpha query"], 1)])
        self.assertEqual(dense.calls, [([0.25, 0.75], 100)])
        self.assertEqual(len(result), len(set(result)))
        self.assertFalse(hasattr(retriever, "last_trace"))

    def test_detailed_result_is_immutable_and_keeps_the_full_fused_union(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex(
            [
                FakeHit("B000000002"),
                FakeHit("B000000004"),
                FakeHit("B000000001"),
            ]
        )
        retriever = HybridRetriever(self.catalog_path, encoder, dense)
        with mock.patch.object(
            retriever,
            "_bm25",
            return_value=["B000000001", "B000000002", "B000000003"],
        ):
            result = retriever.search_with_trace("dense query", "lexical query", top_k=2)

        self.assertIsInstance(result, RetrievalResult)
        self.assertIsInstance(result.trace, RetrievalTrace)
        self.assertEqual(result.recommendations, ("B000000002", "B000000001"))
        self.assertEqual(
            result.trace.bm25_ids,
            ("B000000001", "B000000002", "B000000003"),
        )
        self.assertEqual(
            result.trace.dense_ids,
            ("B000000002", "B000000004", "B000000001"),
        )
        self.assertEqual(
            result.trace.fused_ids,
            ("B000000002", "B000000001", "B000000004", "B000000003"),
        )
        self.assertEqual(result.trace.bm25_status, "ok")
        self.assertEqual(result.trace.dense_status, "ok")
        self.assertFalse(result.trace.used_fallback)
        self.assertEqual(
            {field.name for field in fields(RetrievalTrace)},
            {
                "bm25_ids",
                "dense_ids",
                "fused_ids",
                "bm25_status",
                "dense_status",
                "used_fallback",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            result.trace.used_fallback = True
        with self.assertRaises(FrozenInstanceError):
            result.recommendations = ()

    def test_trace_distinguishes_unavailable_empty_and_error_routes(self) -> None:
        no_dense = HybridRetriever(self.catalog_path, None, None)
        unavailable = no_dense.search_with_trace("ignored", "alpha", top_k=1)
        self.assertEqual(unavailable.trace.bm25_status, "ok")
        self.assertEqual(unavailable.trace.dense_status, "unavailable")
        self.assertFalse(unavailable.trace.used_fallback)

        empty_dense = FakeDenseIndex([])
        both_empty = HybridRetriever(self.catalog_path, FakeEncoder(), empty_dense)
        empty = both_empty.search_with_trace("nothing", "the and please", top_k=2)
        self.assertEqual(empty.trace.bm25_status, "empty")
        self.assertEqual(empty.trace.dense_status, "empty")
        self.assertEqual(empty.trace.fused_ids, ())
        self.assertTrue(empty.trace.used_fallback)
        self.assertEqual(
            empty.recommendations,
            ("B000000001", "B000000002"),
        )

        failed = HybridRetriever(
            self.catalog_path,
            FakeEncoder(fail=True),
            FakeDenseIndex([]),
        )
        failed._connection.close()
        error = failed.search_with_trace("query", "alpha", top_k=2)
        self.assertEqual(error.trace.bm25_status, "error")
        self.assertEqual(error.trace.dense_status, "error")
        self.assertTrue(error.trace.used_fallback)
        self.assertEqual(
            error.recommendations,
            ("B000000001", "B000000002"),
        )

    def test_dense_failure_keeps_weighted_bm25_route(self) -> None:
        encoder = FakeEncoder(fail=True)
        dense = FakeDenseIndex([])
        retriever = HybridRetriever(self.catalog_path, encoder, dense)

        result = retriever.search("query", "alpha", top_k=2)

        self.assertEqual(result, ["B000000001", "B000000002"])
        self.assertEqual(len(encoder.calls), 1)
        self.assertEqual(dense.calls, [])

    def test_explicit_route_weights_change_cross_route_order(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex([FakeHit("B000000003")])
        retriever = HybridRetriever(self.catalog_path, encoder, dense)
        with mock.patch.object(
            retriever,
            "_bm25",
            return_value=["B000000001"],
        ):
            bm25_first = retriever.search(
                "dense query",
                "lexical query",
                top_k=2,
                route_weights=RouteWeights(bm25=0.6, dense=0.4),
            )
            dense_first = retriever.search(
                "dense query",
                "lexical query",
                top_k=2,
                route_weights=RouteWeights(bm25=0.4, dense=0.6),
            )

        self.assertEqual(bm25_first, ["B000000001", "B000000003"])
        self.assertEqual(dense_first, ["B000000003", "B000000001"])

    def test_route_weights_are_type_checked_before_search(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex([FakeHit("B000000003")])
        retriever = HybridRetriever(self.catalog_path, encoder, dense)

        with self.assertRaisesRegex(TypeError, "RouteWeights"):
            retriever.search(
                "dense query",
                "lexical query",
                route_weights=(0.5, 0.5),  # type: ignore[arg-type]
            )

        self.assertEqual(encoder.calls, [])
        self.assertEqual(dense.calls, [])

    def test_bm25_failure_keeps_dense_route(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex(
            [FakeHit("B000000003"), FakeHit("B000000002"), FakeHit("UNKNOWN")]
        )
        retriever = HybridRetriever(self.catalog_path, encoder, dense)
        retriever._connection.close()

        result = retriever.search("waterproof boots", "alpha", top_k=3)

        self.assertEqual(result, ["B000000003", "B000000002"])
        self.assertEqual(len(encoder.calls), 1)

    def test_missing_dense_dependencies_use_bm25_without_encoding(self) -> None:
        retriever = HybridRetriever(self.catalog_path, encoder=None, dense_index=None)

        result = retriever.search("ignored", "alpha", top_k=2)

        self.assertFalse(retriever.dense_available)
        self.assertEqual(result, ["B000000001", "B000000002"])

    def test_both_empty_or_failed_routes_use_catalog_order_fallback(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex([])
        retriever = HybridRetriever(self.catalog_path, encoder, dense)

        first = retriever.search("nothing", "the and please", top_k=3)
        retriever._connection.close()
        dense.fail = True
        second = retriever.search("nothing", "alpha", top_k=2)

        self.assertEqual(first, ["B000000001", "B000000002", "B000000003"])
        self.assertEqual(second, ["B000000001", "B000000002"])
        self.assertEqual(len(encoder.calls), 2)

    def test_catalog_is_loaded_only_during_initialization(self) -> None:
        retriever = HybridRetriever(self.catalog_path, None, None)
        self.catalog_path.write_text("not valid json\n", encoding="utf-8")

        result = retriever.search("ignored", "waterproof", top_k=1)

        self.assertEqual(result, ["B000000003"])

    def test_candidate_documents_are_transient_ordered_and_deduplicated(self) -> None:
        retriever = HybridRetriever(self.catalog_path, None, None)
        self.catalog_path.write_text("not valid json\n", encoding="utf-8")

        documents = retriever.candidate_documents(
            ("B000000003", "B000000001", "B000000003")
        )

        self.assertEqual(
            documents,
            (
                CandidateDocument(
                    "B000000003",
                    "Title: Third product\n"
                    "Categories: Boots\n"
                    "Features: Waterproof\n"
                    "Store: Third",
                ),
                CandidateDocument(
                    "B000000001",
                    "Title: Alpha trail shoe\nCategories: Shoes\nStore: First",
                ),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            documents[0].text = "changed"  # type: ignore[misc]

    def test_candidate_document_field_labels_and_order_are_exact(self) -> None:
        complete_catalog = Path(self.temporary_directory.name) / "complete.jsonl"
        complete_catalog.write_text(
            json.dumps(
                {
                    "parent_asin": "COMPLETE",
                    "title": "Complete product",
                    "categories": ["Shoes", "Trail"],
                    "features": ["Waterproof", "Lightweight"],
                    "details": {"Material": "Leather", "Color": "Brown"},
                    "store": "Complete Store",
                    "description": ["First sentence", "Second sentence"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        retriever = HybridRetriever(complete_catalog, None, None)

        document = retriever.candidate_documents(("COMPLETE",))[0]

        self.assertEqual(
            document.text,
            "Title: Complete product\n"
            "Categories: Shoes Trail\n"
            "Features: Waterproof Lightweight\n"
            "Details: Material Leather Color Brown\n"
            "Store: Complete Store\n"
            "Description: First sentence Second sentence",
        )

    def test_candidate_document_lookup_is_bounded_and_validated(self) -> None:
        retriever = HybridRetriever(self.catalog_path, None, None)
        with self.assertRaisesRegex(TypeError, "sequence"):
            retriever.candidate_documents("B000000001")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "sequence"):
            retriever.candidate_documents(iter(("B000000001",)))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            retriever.candidate_documents((1,))  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unknown"):
            retriever.candidate_documents(("UNKNOWN",))

        bounded_catalog = Path(self.temporary_directory.name) / "bounded.jsonl"
        identifiers = tuple(
            f"PRODUCT{index:03d}" for index in range(MAX_CANDIDATE_DOCUMENTS)
        )
        bounded_catalog.write_text(
            "".join(
                json.dumps({"parent_asin": parent_asin, "title": parent_asin})
                + "\n"
                for parent_asin in identifiers
            ),
            encoding="utf-8",
        )
        bounded = HybridRetriever(bounded_catalog, None, None)
        requested = tuple(reversed(identifiers))

        exact_limit = bounded.candidate_documents(requested)

        self.assertEqual(
            tuple(document.parent_asin for document in exact_limit),
            requested,
        )
        with self.assertRaisesRegex(ValueError, "at most"):
            bounded.candidate_documents((*requested, identifiers[0]))

    def test_candidate_documents_detect_missing_or_misaligned_fts_rows(self) -> None:
        missing = HybridRetriever(self.catalog_path, None, None)
        missing._connection.execute("DELETE FROM products WHERE rowid = 1")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            missing.candidate_documents(("B000000001",))

        misaligned = HybridRetriever(self.catalog_path, None, None)
        misaligned._connection.execute(
            "UPDATE products SET parent_asin = ? WHERE rowid = 1",
            ("B000000002",),
        )
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            misaligned.candidate_documents(("B000000001",))

    def test_candidate_document_lookup_uses_one_rowid_query(self) -> None:
        retriever = HybridRetriever(self.catalog_path, None, None)
        statements: list[str] = []
        retriever._connection.set_trace_callback(statements.append)

        retriever.candidate_documents(("B000000004", "B000000002"))

        selects = [statement for statement in statements if statement.startswith("SELECT")]
        self.assertEqual(len(selects), 1)
        self.assertIn("WHERE rowid IN", selects[0])
        self.assertNotIn("WHERE parent_asin", selects[0])

    def test_explicit_fts_rowids_ignore_blank_catalog_lines(self) -> None:
        payload = self.catalog_path.read_text(encoding="utf-8").splitlines()
        self.catalog_path.write_text(
            "\n" + payload[0] + "\n\n" + payload[1] + "\n",
            encoding="utf-8",
        )
        retriever = HybridRetriever(self.catalog_path, None, None)

        rows = retriever._connection.execute(
            "SELECT rowid, parent_asin FROM products ORDER BY rowid"
        ).fetchall()
        self.assertEqual(rows, [(1, "B000000001"), (2, "B000000002")])

    def test_non_positive_top_k_is_empty_without_encoding(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex([])
        retriever = HybridRetriever(self.catalog_path, encoder, dense)

        self.assertEqual(retriever.search("query", "alpha", top_k=0), [])
        detailed = retriever.search_with_trace("query", "alpha", top_k=0)
        self.assertEqual(detailed.recommendations, ())
        self.assertEqual(detailed.trace.bm25_status, "skipped")
        self.assertEqual(detailed.trace.dense_status, "skipped")
        self.assertFalse(detailed.trace.used_fallback)
        self.assertEqual(encoder.calls, [])
        self.assertEqual(dense.calls, [])

    def test_missing_fts5_keeps_dense_and_catalog_fallback_available(self) -> None:
        encoder = FakeEncoder()
        dense = FakeDenseIndex([FakeHit("B000000003")])
        with mock.patch.object(
            HybridRetriever,
            "_create_bm25_table",
            side_effect=sqlite3.OperationalError("no such module: fts5"),
        ):
            retriever = HybridRetriever(self.catalog_path, encoder, dense)

        self.assertFalse(retriever.bm25_available)
        self.assertIn("fts5", retriever.bm25_initialization_error or "")
        self.assertEqual(retriever.candidate_documents(("B000000003",)), ())
        result = retriever.search_with_trace("waterproof", "waterproof", top_k=1)
        self.assertEqual(
            result.recommendations,
            ("B000000003",),
        )
        self.assertEqual(result.trace.bm25_status, "unavailable")
        self.assertEqual(result.trace.dense_status, "ok")


if __name__ == "__main__":
    unittest.main()
