from __future__ import annotations

import unittest

from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.strategy import RouteWeights
from scripts.run_fusion_ablations import (
    RouteHealthRetriever,
    _policy_contract,
    run_fusion_policies,
)
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    EQUAL_RRF_POLICY,
)


class _Backend:
    def __init__(self, trace: RetrievalTrace) -> None:
        self.trace = trace
        self.calls: list[tuple[str, str, int, RouteWeights]] = []

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int,
        *,
        route_weights: RouteWeights,
    ) -> RetrievalResult:
        self.calls.append((dense_query_text, lexical_text, top_k, route_weights))
        return RetrievalResult(recommendations=("A",), trace=self.trace)


def _trace(
    bm25_status: str = "ok",
    dense_status: str = "ok",
    *,
    fallback: bool = False,
) -> RetrievalTrace:
    return RetrievalTrace(
        bm25_ids=("A",) if bm25_status == "ok" else (),
        dense_ids=("B",) if dense_status == "ok" else (),
        fused_ids=("A", "B") if not fallback else (),
        bm25_status=bm25_status,  # type: ignore[arg-type]
        dense_status=dense_status,  # type: ignore[arg-type]
        used_fallback=fallback,
    )


class FusionAblationTest(unittest.TestCase):
    def test_policy_grid_rejects_invalid_names_before_loading_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_fusion_policies("unused", "unused", [])
        with self.assertRaisesRegex(ValueError, "unique"):
            run_fusion_policies("unused", "unused", ["equal", "equal"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            run_fusion_policies("unused", "unused", ["unknown"])

    def test_health_adapter_delegates_once_and_preserves_weights(self) -> None:
        backend = _Backend(_trace())
        adapter = RouteHealthRetriever(backend)
        weights = RouteWeights(bm25=0.6, dense=0.4)

        result = adapter.search(
            "dense",
            "lexical",
            10,
            route_weights=weights,
        )

        self.assertEqual(result, ["A"])
        self.assertEqual(backend.calls, [("dense", "lexical", 10, weights)])
        self.assertEqual(
            adapter.summary(),
            {
                "bm25": {"ok": 1},
                "dense": {"ok": 1},
                "fallback_turns": 0,
            },
        )
        adapter.validate(expected_turns=1)
        with self.assertRaisesRegex(RuntimeError, "evaluator turns"):
            adapter.validate(expected_turns=2)

    def test_health_adapter_allows_empty_but_rejects_operational_faults(self) -> None:
        empty = RouteHealthRetriever(_Backend(_trace("empty", "empty", fallback=True)))
        empty.search(
            "dense",
            "lexical",
            10,
            route_weights=RouteWeights(bm25=0.5, dense=0.5),
        )
        empty.validate()
        self.assertEqual(empty.summary()["fallback_turns"], 1)

        failed = RouteHealthRetriever(_Backend(_trace("error", "unavailable")))
        failed.search(
            "dense",
            "lexical",
            10,
            route_weights=RouteWeights(bm25=0.5, dense=0.5),
        )
        with self.assertRaisesRegex(RuntimeError, "route faults"):
            failed.validate()

    def test_policy_contracts_are_explicit_and_label_free(self) -> None:
        equal = _policy_contract(EQUAL_RRF_POLICY)
        adaptive = _policy_contract(COMPLETENESS_ADAPTIVE_RRF_POLICY)

        self.assertEqual(equal["bm25"], "0.5")
        self.assertEqual(adaptive["bm25"], "0.4 + 0.2 * C_t")
        serialized = repr({"equal": equal, "adaptive": adaptive})
        for forbidden in ("target", "ground_truth", "sample_id", "profile"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
