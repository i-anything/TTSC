from __future__ import annotations

import ast
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from conversational_search.intent import IntentState
from conversational_search.orchestration import QueryAction
from conversational_search.retrieval import RetrievalResult, RetrievalTrace
from scripts.run_phase1_failure_diagnostics import (
    FAILURE_CATEGORIES,
    LabelFreeTurnTrace,
    ScoreLoss,
    _TurnCapture,
    _choose_primary,
    _diagnose_session,
    _rank_histogram,
    _score_loss,
    _validate_public_report,
    _write_json_exclusive,
)


def _record(
    turn: int,
    *,
    pool: tuple[str, ...] = (),
    output: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
    version: int = 0,
    action: str = "reuse",
) -> LabelFreeTurnTrace:
    return LabelFreeTurnTrace(
        dialogue_ordinal=0,
        turn=turn,
        output_ids=output,
        ask_attribute=None,
        response_error=False,
        latency_ms=1.0,
        decision_action=action,
        decision_reason="test",
        retrieval_executed=False,
        bm25_ids=(),
        dense_ids=(),
        fused_ids=(),
        bm25_status="not_executed",
        dense_status="not_executed",
        retrieval_fallback=False,
        rerank_executed=False,
        reranked_ids=(),
        pre_slate_executed=bool(pool),
        pre_slate_ids=pool,
        selected_ids=output,
        slate_status="test",
        intent_version=version,
        requirement_sources=sources,
        asked_attributes=(),
        no_preference=(),
        retrieval_dependency_digest=f"digest-{turn}",
    )


def _product(parent_asin: str = "P") -> dict:
    return {
        "parent_asin": parent_asin,
        "title": "Test shoe",
        "categories": ["Shoes"],
        "features": ["durable"],
        "details": {},
        "description": [],
    }


def _sample(scenario: str = "buying") -> dict:
    behavior = {"scenario_type": scenario}
    if scenario == "intent_override":
        behavior["override"] = {
            "turn": 3,
            "old_value": "durable",
            "new_value": "durable",
            "message": (
                "Actually, ignore my earlier preference. "
                "What I need is: durable."
            ),
        }
    return {
        "ground_truth": {"parent_asin": "P"},
        "scenario_type": scenario,
        "user_profile": {},
        "intent_card": {
            "target_category": "Test shoe",
            "hard_constraints": ["durable"],
            "soft_preferences": ["durable"],
        },
        "behavior": behavior,
    }


class Phase1FailureDiagnosticTests(unittest.TestCase):
    def test_label_free_trace_has_no_label_or_text_fields(self) -> None:
        names = {field.name for field in fields(LabelFreeTurnTrace)}
        forbidden = {
            "target",
            "sample_id",
            "session_id",
            "user_message",
            "dense_query",
            "lexical_query",
            "user_profile",
        }
        self.assertTrue(forbidden.isdisjoint(names))

    def test_score_loss_exact_cases(self) -> None:
        rank_two_turn_three = _score_loss(3, 2)
        self.assertAlmostEqual(rank_two_turn_three.total, 0.19)
        self.assertEqual(_score_loss(None, None), ScoreLoss(0.5, 0.3, 0.2))
        self.assertAlmostEqual(_score_loss(1, 1).total, 0.0)

    def test_primary_precedence_and_no_loss(self) -> None:
        flags = frozenset(
            {
                "weak_question_selected",
                "target_absent_from_retrieved_candidates",
            }
        )
        self.assertEqual(
            _choose_primary(flags, ScoreLoss(0.5, 0.3, 0.2)),
            "target_absent_from_retrieved_candidates",
        )
        self.assertEqual(
            _choose_primary(flags, ScoreLoss(0.0, 0.0, 0.0)),
            "no_loss",
        )

    def test_rank_histogram_distinguishes_absent_and_not_executed(self) -> None:
        histogram = _rank_histogram(
            ((False, None), (True, None), (True, 1), (True, 15), (True, 75))
        )
        self.assertEqual(histogram["not_executed"], 1)
        self.assertEqual(histogram["absent"], 1)
        self.assertEqual(histogram["1"], 1)
        self.assertEqual(histogram["11-20"], 1)
        self.assertEqual(histogram["51-100"], 1)

    def test_turn_capture_accepts_search_and_reuse(self) -> None:
        capture = _TurnCapture()
        capture.begin(0, 1)
        capture.record_decision(QueryAction.SEARCH, "cold")
        capture.record_retrieval(
            RetrievalResult(
                recommendations=("P",),
                trace=RetrievalTrace(
                    bm25_ids=("P",),
                    dense_ids=(),
                    fused_ids=("P",),
                    bm25_status="ok",
                    dense_status="unavailable",
                    used_fallback=False,
                ),
            )
        )
        search = capture.finish(
            {"message": "ok", "ask_attribute": None, "recommendations": []},
            IntentState(),
            response_error=False,
            latency_ms=1.0,
        )
        self.assertTrue(search.retrieval_executed)

        capture.begin(0, 2)
        capture.record_decision(QueryAction.REUSE, "hit")
        reuse = capture.finish(
            {"message": "ok", "ask_attribute": None, "recommendations": []},
            IntentState(),
            response_error=False,
            latency_ms=1.0,
        )
        self.assertFalse(reuse.retrieval_executed)
        self.assertEqual(reuse.bm25_status, "not_executed")

    def test_turn_capture_rejects_more_than_one_search(self) -> None:
        capture = _TurnCapture()
        capture.begin(0, 1)
        capture.record_decision(QueryAction.SEARCH, "cold")
        result = RetrievalResult(
            recommendations=(),
            trace=RetrievalTrace(
                bm25_ids=(),
                dense_ids=(),
                fused_ids=(),
                bm25_status="empty",
                dense_status="empty",
                used_fallback=True,
            ),
        )
        capture.record_retrieval(result)
        capture.record_retrieval(result)
        with self.assertRaises(RuntimeError):
            capture.finish(
                {"message": "ok", "ask_attribute": None, "recommendations": []},
                IntentState(),
                response_error=False,
                latency_ms=1.0,
            )

    def test_absent_failure_is_classified_without_persisting_identifier(self) -> None:
        records = tuple(_record(turn) for turn in range(1, 11))
        product = _product()
        diagnostic = _diagnose_session(
            0,
            _sample(),
            product,
            ["Shoes"],
            records,
            {"first_hit_turn": None, "best_rank": None},
            {"P": product},
            {},
        )
        self.assertIn(
            "target_absent_from_retrieved_candidates",
            diagnostic.flags,
        )
        self.assertEqual(
            diagnostic.primary,
            "target_absent_from_retrieved_candidates",
        )

    def test_poor_and_premature_exposure_are_distinct_flags(self) -> None:
        records = [
            _record(1, pool=("P",), output=("X", "P")),
            _record(2, pool=("P",), output=("P",)),
        ]
        records.extend(_record(turn) for turn in range(3, 11))
        product = _product()
        diagnostic = _diagnose_session(
            0,
            _sample(),
            product,
            ["Shoes"],
            tuple(records),
            {"first_hit_turn": 1, "best_rank": 2},
            {"P": product},
            {},
        )
        self.assertIn("target_in_top_10_but_ordered_poorly", diagnostic.flags)
        self.assertIn(
            "target_exposed_prematurely_at_low_reciprocal_rank",
            diagnostic.flags,
        )

    def test_bad_override_state_is_detected(self) -> None:
        records = tuple(
            _record(
                turn,
                sources=("initial_tentative",),
                version=0,
            )
            for turn in range(1, 11)
        )
        product = _product()
        diagnostic = _diagnose_session(
            0,
            _sample("intent_override"),
            product,
            ["Shoes"],
            records,
            {"first_hit_turn": None, "best_rank": None},
            {"P": product},
            {},
        )
        self.assertIn("intent_override_handled_incorrectly", diagnostic.flags)
        self.assertEqual(
            diagnostic.primary,
            "intent_override_handled_incorrectly",
        )

    def test_all_ten_requested_categories_are_declared(self) -> None:
        self.assertEqual(len(FAILURE_CATEGORIES), 10)
        self.assertEqual(len(set(FAILURE_CATEGORIES)), 10)

    def test_public_report_rejects_raw_identifiers(self) -> None:
        with self.assertRaises(ValueError):
            _validate_public_report(
                {"parent_asin": "P", "invariants": {"ok": True}}
            )

    def test_exclusive_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _write_json_exclusive(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                _write_json_exclusive(path, {"ok": True})

    def test_runtime_packages_do_not_import_diagnostic_module(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = "scripts.run_phase1_failure_diagnostics"
        for directory in (root / "conversational_search", root / "starter"):
            for path in directory.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertNotIn(forbidden, {alias.name for alias in node.names})
                    elif isinstance(node, ast.ImportFrom):
                        self.assertNotEqual(node.module, forbidden)


if __name__ == "__main__":
    unittest.main()
