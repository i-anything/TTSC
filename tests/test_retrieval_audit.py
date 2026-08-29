from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from conversational_search.strategy import RouteWeights
from scripts.run_retrieval_audit import (
    OFFICIAL_METRIC_KEYS,
    UnlabeledTurnTrace,
    TraceCaptureRetriever,
    _LabeledTurn,
    _aggregate,
    _classify_waterfall,
    _join_labels,
    _rank,
    _validate_trace_coverage,
    _write_json_exclusive,
)


def _trace(
    *,
    bm25: tuple[str, ...] = (),
    dense: tuple[str, ...] = (),
    fused: tuple[str, ...] = (),
    fallback: bool = False,
) -> RetrievalTrace:
    return RetrievalTrace(
        bm25_ids=bm25,
        dense_ids=dense,
        fused_ids=fused,
        bm25_status="ok" if bm25 else "empty",
        dense_status="ok" if dense else "empty",
        used_fallback=fallback,
    )


class _FakeBackend:
    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
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
        return self.result


def _row(
    *,
    bm25_rank: int | None = None,
    dense_rank: int | None = None,
    fused_rank: int | None = None,
    output_rank: int | None = None,
    waterfall: str = "not_retrieved",
) -> _LabeledTurn:
    return _LabeledTurn(
        session_ordinal=0,
        scenario_type="buying",
        turn=1,
        eligible=True,
        intent_completeness_proxy=0.0,
        bm25_status="ok",
        dense_status="ok",
        bm25_rank=bm25_rank,
        dense_rank=dense_rank,
        fused_rank=fused_rank,
        output_rank=output_rank,
        bm25_count=100,
        dense_count=100,
        fused_count=180,
        route_jaccard=20 / 180,
        used_fallback=False,
        waterfall=waterfall,
    )


class RetrievalAuditTest(unittest.TestCase):
    def test_capture_executes_one_search_and_requires_one_pop(self) -> None:
        result = RetrievalResult(
            recommendations=("A", "B"),
            trace=_trace(bm25=("A",), dense=("B",), fused=("A", "B")),
        )
        backend = _FakeBackend(result)
        capture = TraceCaptureRetriever(backend)
        weights = RouteWeights(bm25=0.5, dense=0.5)

        self.assertEqual(
            capture.search(
                "dense",
                "lexical",
                2,
                route_weights=weights,
            ),
            ["A", "B"],
        )
        self.assertEqual(backend.calls, [("dense", "lexical", 2, weights)])
        self.assertEqual(capture.pop(), result)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            capture.pop()

        capture.search("dense", "lexical", 2, route_weights=weights)
        capture.search("dense", "lexical", 2, route_weights=weights)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            capture.pop()

    def test_unlabeled_trace_has_no_label_or_query_fields(self) -> None:
        names = {field.name for field in fields(UnlabeledTurnTrace)}
        self.assertEqual(
            names,
            {
                "session_ordinal",
                "turn",
                "output_ids",
                "trace",
                "intent_completeness_proxy",
            },
        )
        forbidden = {"target", "sample_id", "scenario_type", "query", "profile"}
        self.assertTrue(names.isdisjoint(forbidden))

    def test_rank_is_one_based_and_absence_is_null(self) -> None:
        self.assertEqual(_rank(("A", "B", "C"), "A"), 1)
        self.assertEqual(_rank(("A", "B", "C"), "C"), 3)
        self.assertIsNone(_rank(("A", "B", "C"), "D"))

    def test_waterfall_categories_are_mutually_exclusive(self) -> None:
        cases = [
            ((1, 2, 1, 1, False), "agreement_kept"),
            ((1, None, 1, 1, False), "bm25_rescue"),
            ((None, 1, 1, 1, False), "dense_rescue"),
            ((11, 12, 1, 1, False), "deep_rrf_promotion"),
            ((1, 2, 11, None, False), "agreement_lost"),
            ((1, None, 11, None, False), "bm25_lost"),
            ((None, 1, 11, None, False), "dense_lost"),
            ((11, None, 11, None, False), "union_not_promoted"),
            ((None, None, None, None, False), "not_retrieved"),
            ((None, None, None, 1, True), "fallback_hit"),
            ((None, None, None, None, True), "fallback_miss"),
        ]
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                actual = _classify_waterfall(
                    bm25_rank=arguments[0],
                    dense_rank=arguments[1],
                    fused_rank=arguments[2],
                    output_rank=arguments[3],
                    used_fallback=arguments[4],
                )
                self.assertEqual(actual, expected)

    def test_aggregate_reports_route_recall_and_partitions_rows(self) -> None:
        rows = [
            _row(
                bm25_rank=1,
                dense_rank=None,
                fused_rank=1,
                output_rank=1,
                waterfall="bm25_rescue",
            ),
            _row(
                bm25_rank=None,
                dense_rank=20,
                fused_rank=30,
                waterfall="union_not_promoted",
            ),
        ]

        result = _aggregate(rows)

        self.assertEqual(result["eligible_observations"], 2)
        self.assertEqual(result["bm25"]["recall_at_10"], 0.5)
        self.assertEqual(result["dense"]["recall_at_10"], 0.0)
        self.assertEqual(result["dense"]["recall_at_50"], 0.5)
        self.assertEqual(result["fused"]["union_recall"], 1.0)
        self.assertEqual(result["output"]["recall_at_10"], 0.5)
        self.assertEqual(result["waterfall_counts"]["bm25_rescue"], 1)
        self.assertEqual(result["waterfall_counts"]["union_not_promoted"], 1)
        self.assertEqual(sum(result["waterfall_counts"].values()), 2)

    def test_coverage_validation_detects_missing_or_noncontiguous_traces(self) -> None:
        trace = _trace(bm25=("A",), fused=("A",))
        sessions = [
            {"first_hit_turn": 2},
            {"first_hit_turn": None},
        ]
        records = [
            UnlabeledTurnTrace(0, turn, ("A",), trace, 0.0)
            for turn in (1, 2)
        ] + [
            UnlabeledTurnTrace(1, turn, (), trace, 0.0)
            for turn in range(1, 11)
        ]

        _validate_trace_coverage(records, sessions)
        with self.assertRaisesRegex(RuntimeError, "captured"):
            _validate_trace_coverage(records[:-1], sessions)
        noncontiguous = list(records)
        noncontiguous[1] = UnlabeledTurnTrace(0, 3, ("A",), trace, 0.0)
        with self.assertRaisesRegex(RuntimeError, "non-contiguous"):
            _validate_trace_coverage(noncontiguous, sessions)

    def test_official_metric_projection_is_an_explicit_privacy_allowlist(self) -> None:
        self.assertEqual(
            set(OFFICIAL_METRIC_KEYS),
            {
                "sample_count",
                "hit_rate_at_10",
                "mrr",
                "mttc",
                "efficiency",
                "recommended_technical_score",
                "reported_token_usage",
                "scenario_metrics",
            },
        )
        self.assertTrue(
            set(OFFICIAL_METRIC_KEYS).isdisjoint(
                {"sessions", "sample_id", "user_profile", "ground_truth"}
            )
        )

    def test_label_join_excludes_pre_override_turns_from_waterfall(self) -> None:
        sample = {
            "sample_id": "public_0001",
            "scenario_type": "intent_override",
            "user_profile": {},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "target_category": "shoe",
                "hard_constraints": ["leather"],
                "soft_preferences": ["black"],
            },
            "behavior": {
                "scenario_type": "intent_override",
                "override": {
                    "turn": 3,
                    "old_value": "black",
                    "new_value": "leather",
                    "message": "Actually, ignore my earlier preference. What I need is: leather.",
                },
            },
        }
        trace = _trace(
            bm25=("TARGET",),
            dense=("OTHER",),
            fused=("TARGET", "OTHER"),
        )
        records = [
            UnlabeledTurnTrace(0, 2, ("TARGET",), trace, 0.5),
            UnlabeledTurnTrace(0, 3, ("TARGET",), trace, 0.5),
        ]

        rows = _join_labels(records, [sample], {})

        self.assertFalse(rows[0].eligible)
        self.assertIsNone(rows[0].waterfall)
        self.assertTrue(rows[1].eligible)
        self.assertEqual(rows[1].waterfall, "bm25_rescue")

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "audit.json"
            _write_json_exclusive(destination, {"value": 1})
            self.assertEqual(destination.read_text(encoding="utf-8"), '{\n  "value": 1\n}\n')
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(destination, {"value": 2})


if __name__ == "__main__":
    unittest.main()
