from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conversational_search.intent import (
    IntentState,
    Requirement,
    render_requirement_probe_candidates,
)
from conversational_search.retrieval import (
    CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    DISABLED_REQUIREMENT_PROBE_POLICY,
    HybridRetriever,
    RequirementProbeRetrievalResult,
)


class RequirementProbeIntentTest(unittest.TestCase):
    def test_renderer_keeps_only_deduplicated_strong_nonbudget_clauses(self) -> None:
        state = IntentState(
            requirements=(
                Requirement("Material: Silk", "initial_explicit", 1, "material"),
                Requirement("silk", "answer", 2, "material"),
                Requirement("waterproof", "override", 3, "feature"),
                Requirement("under $50", "answer", 4, "budget"),
                Requirement("casual", "free_text", 5, "style"),
                Requirement("blue", "initial_tentative", 1, "color"),
            )
        )

        self.assertEqual(
            render_requirement_probe_candidates(state),
            ("Silk", "waterproof"),
        )

    def test_renderer_is_bounded_and_does_not_retain_invalid_state(self) -> None:
        state = IntentState(
            requirements=tuple(
                Requirement(f"feature {index}", "answer", 1, "feature")
                for index in range(25)
            )
        )
        self.assertEqual(render_requirement_probe_candidates(state), ())


class RequirementProbeRetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = (
            ("B000000001", "shoe common cotton"),
            ("B000000002", "shoe common waterproof"),
            ("B000000003", "silk scarf"),
            ("B000000004", "waterproof jacket"),
            ("B000000005", "titanium bracelet"),
        )
        self.catalog_path.write_text(
            "".join(
                json.dumps(
                    {
                        "parent_asin": parent_asin,
                        "title": title,
                        "categories": ["Products"],
                        "features": [],
                        "details": {},
                        "store": "Store",
                        "description": [],
                    }
                )
                + "\n"
                for parent_asin, title in products
            ),
            encoding="utf-8",
        )

    def test_disabled_policy_is_exactly_one_protected_bm25_call(self) -> None:
        retriever = HybridRetriever(self.catalog_path)
        with mock.patch.object(retriever, "_bm25", wraps=retriever._bm25) as bm25:
            result = retriever.search_with_trace(
                "shoe",
                "shoe",
                top_k=5,
                requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
                requirement_probe_candidates=("silk", "waterproof"),
            )

        bm25.assert_called_once_with("shoe")
        self.assertNotIsInstance(result, RequirementProbeRetrievalResult)

    def test_rarest_known_clauses_add_only_unseen_products_after_base_bm25(self) -> None:
        retriever = HybridRetriever(self.catalog_path)
        with mock.patch.object(retriever, "_bm25", wraps=retriever._bm25) as bm25:
            result = retriever.search_with_trace(
                "shoe",
                "shoe",
                top_k=5,
                requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
                requirement_probe_candidates=("waterproof", "silk", "unknownword"),
            )

        self.assertEqual(
            [call.args[0] for call in bm25.call_args_list],
            ["shoe", "silk", "waterproof"],
        )
        self.assertEqual(
            result.probe_trace.base_bm25_ids,
            ("B000000001", "B000000002"),
        )
        self.assertEqual(
            result.probe_trace.supplemental_ids,
            ("B000000003", "B000000004"),
        )
        self.assertEqual(
            result.trace.bm25_ids,
            (
                "B000000001",
                "B000000002",
                "B000000003",
                "B000000004",
            ),
        )
        self.assertEqual(result.probe_trace.status, "ok")
        self.assertEqual(result.probe_trace.query_count, 2)
        self.assertEqual(
            set(result.trace.fused_ids),
            set(result.trace.bm25_ids) | set(result.trace.dense_ids),
        )

    def test_oov_suffixes_cannot_consume_two_slots_for_one_known_signature(self) -> None:
        retriever = HybridRetriever(self.catalog_path)
        self.assertTrue(retriever._ensure_bm25_vocabulary())

        selected = retriever._select_requirement_probes(
            (
                "silk missingalpha",
                "silk missingbeta",
                "waterproof",
            ),
            "shoe",
        )

        self.assertEqual(selected, ("silk", "waterproof"))

    def test_unexpected_vocabulary_fault_returns_the_exact_main_route(self) -> None:
        retriever = HybridRetriever(self.catalog_path)
        with (
            mock.patch.object(
                retriever,
                "_ensure_bm25_vocabulary",
                side_effect=RuntimeError("vocabulary fault"),
            ),
            mock.patch.object(
                retriever,
                "_bm25",
                wraps=retriever._bm25,
            ) as bm25,
        ):
            result = retriever.search_with_trace(
                "shoe",
                "shoe",
                top_k=5,
                requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
                requirement_probe_candidates=("silk",),
            )

        bm25.assert_called_once_with("shoe")
        self.assertEqual(result.trace.bm25_ids, result.probe_trace.base_bm25_ids)
        self.assertEqual(result.probe_trace.supplemental_ids, ())
        self.assertEqual(result.probe_trace.status, "error")
        self.assertEqual(result.probe_trace.query_count, 0)

    def test_any_probe_route_error_discards_all_supplements(self) -> None:
        retriever = HybridRetriever(self.catalog_path)
        protected = tuple(retriever._bm25("shoe"))

        def fail_probe(query: str) -> list[str]:
            if query == "shoe":
                return list(protected)
            raise RuntimeError("probe failed")

        with mock.patch.object(retriever, "_bm25", side_effect=fail_probe):
            result = retriever.search_with_trace(
                "shoe",
                "shoe",
                top_k=5,
                requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
                requirement_probe_candidates=("silk", "waterproof"),
            )

        self.assertEqual(result.probe_trace.base_bm25_ids, protected)
        self.assertEqual(result.trace.bm25_ids, protected)
        self.assertEqual(result.probe_trace.supplemental_ids, ())
        self.assertEqual(result.probe_trace.status, "error")
        self.assertEqual(result.probe_trace.query_count, 1)


if __name__ == "__main__":
    unittest.main()
