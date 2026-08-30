from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from conversational_search.intent import IntentState, Requirement
from conversational_search.orchestration import ranking_dependency_digest
from conversational_search.ranking import STAGE_A_RANKING_POLICY
from conversational_search.retrieval import (
    CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    DISABLED_REQUIREMENT_PROBE_POLICY,
    REQUIREMENT_PROBE_CAPABILITY,
    RequirementProbeRetrievalResult,
    RequirementProbeTrace,
    RetrievalTrace,
)
from conversational_search.strategy import RouteWeights
from scripts import run_requirement_probe_ablations as ablations


def _summary(*, mrr: float, mttc: float, score: float) -> dict:
    return {
        "sample_count": 4,
        "hit_rate_at_10": 0.75,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": round((11.0 - mttc) / 10.0, 6),
        "recommended_technical_score": score,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _run(*, mrr: float, mttc: float, score: float) -> ablations.VariantRun:
    return ablations.VariantRun(
        summary=_summary(mrr=mrr, mttc=mttc, score=score),
        sessions=[],
        diagnostics={},
        evaluator_digest="digest",
        private_digest="private",
    )


def _paired(lower: float) -> dict:
    return {
        "transitions": {
            "both_hit": 3,
            "candidate_only_hit": 0,
            "baseline_only_hit": 0,
            "both_miss": 1,
        },
        "bootstrap": {
            "seed": ablations.BOOTSTRAP_SEED,
            "replicates": 10_000,
            "strata": 1,
            "lower_95": lower,
            "upper_95": 0.1,
        },
    }


def _privacy_payload() -> dict:
    return {
        "schema_version": ablations.SCHEMA_VERSION,
        "experiment_id": ablations.EXPERIMENT_ID,
        "suite": "fresh",
        "dataset": {},
        "run_configuration": {},
        "metrics": {},
        "paired_quality": {},
        "health": {},
        "call_accounting": {},
        "performance": {},
        "startup": {},
        "exactness": {},
        "privacy": {
            "aggregate_metrics_and_fixed_counters_only": True,
            "row_scenario_message_profile_target_product_and_trace_data_absent": True,
            "per_case_fingerprints_absent": True,
            "manual_failure_inspection_performed": False,
        },
        "reproducibility": {},
        "decision_gate": {},
    }


def _verification() -> dict:
    return {
        "focused_suite_command": ablations.FOCUSED_SUITE_COMMAND,
        "focused_tests_passed": 24,
        "complete_suite_command": ablations.FULL_SUITE_COMMAND,
        "complete_unit_tests_passed": 459,
        "phase7_oracle_command": ablations.PHASE7_ORACLE_COMMAND,
        "phase7_oracle_cases": ablations.PHASE7_ORACLE_CASES,
        "phase7_oracle_sha256": ablations.PHASE7_ORACLE_SHA256,
        "phase9_oracle_command": ablations.PHASE9_ORACLE_COMMAND,
        "phase9_oracle_cases": ablations.PHASE9_ORACLE_CASES,
        "phase9_oracle_sha256": ablations.PHASE9_ORACLE_SHA256,
        "phase13_oracle_command": ablations.PHASE13_ORACLE_COMMAND,
        "phase13_oracle_cases": 45_600,
        "phase13_random_oracle_cases": ablations.PHASE13_RANDOM_ORACLE_CASES,
        "phase13_oracle_sha256": ablations.PHASE13_ORACLE_SHA256,
        "phase14_oracle_command": ablations.PHASE14_ORACLE_COMMAND,
        "phase14_oracle_cases": ablations.PHASE14_RANDOM_ORACLE_CASES,
        "phase14_random_oracle_cases": ablations.PHASE14_RANDOM_ORACLE_CASES,
        "phase14_oracle_sha256": ablations.PHASE14_ORACLE_SHA256,
        "completed_before_lock": True,
    }


class _TraceBackend:
    ranking_cache_capability = object()
    snapshot_token = object()
    requirement_probe_capability = REQUIREMENT_PROBE_CAPABILITY

    def __init__(self, result: RequirementProbeRetrievalResult) -> None:
        self.result = result
        self.document_calls = 0

    def search_with_trace(self, *args: object, **kwargs: object) -> object:
        return self.result

    def candidate_documents(self, parent_asins: object) -> tuple:
        self.document_calls += 1
        return ()


class _ManualAgent:
    def __init__(self, target: str) -> None:
        self.target = target
        self.reset_calls: list[tuple[str, dict]] = []
        self.messages: list[tuple[str, str, int, int]] = []

    def reset(self, session_id: str, profile: dict) -> None:
        self.reset_calls.append((session_id, profile))

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        self.messages.append((session_id, user_message, turn, top_k))
        return {
            "message": "ok",
            "ask_attribute": "other",
            "recommendations": [] if turn == 1 else [self.target],
        }


class _LazyVocabularyBackend:
    requirement_probe_capability = REQUIREMENT_PROBE_CAPABILITY
    requirement_probe_vocabulary_available = False

    def __init__(self) -> None:
        self.initializations = 0

    def _ensure_bm25_vocabulary(self) -> bool:
        self.initializations += 1
        self.requirement_probe_vocabulary_available = True
        return True


class RequirementProbeAblationHarnessTests(unittest.TestCase):
    def test_frozen_suite_order_and_source_scope(self) -> None:
        self.assertEqual(tuple(ablations.SUITES), (
            "fresh",
            "development",
            "validation",
            "public",
        ))
        self.assertEqual(ablations.SUITES["fresh"].prerequisites, ())
        self.assertEqual(
            ablations.SUITES["public"].prerequisites,
            ("fresh", "development", "validation"),
        )
        self.assertNotIn("starter/agent.py", ablations.ALLOWED_RUNTIME_CHANGES)
        self.assertIn(
            "scripts/run_requirement_probe_ablations.py",
            ablations.SOURCE_PATHS,
        )
        self.assertIn(
            "tests/test_requirement_probe_ablations.py",
            ablations.SOURCE_PATHS,
        )

    def test_disabled_ranking_digest_is_exact_old_path_and_candidate_differs(self) -> None:
        state = IntentState(
            category="shoes",
            requirements=(Requirement("wide toe box", "answer", 2, "size"),),
        )
        weights = RouteWeights(bm25=0.6, dense=0.4)
        arguments = (
            state,
            "Category: shoes\nAttributes: Size: wide toe box",
            "shoes wide toe box",
            weights,
            STAGE_A_RANKING_POLICY,
        )

        omitted = ranking_dependency_digest(*arguments)
        explicit_disabled = ranking_dependency_digest(
            *arguments,
            retrieval_policy=None,
        )
        candidate = ranking_dependency_digest(
            *arguments,
            retrieval_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY.value,
        )

        self.assertEqual(omitted, explicit_disabled)
        self.assertNotEqual(candidate, omitted)

    def test_call_audit_counts_probe_work_without_retaining_ids(self) -> None:
        result = RequirementProbeRetrievalResult(
            recommendations=("A", "B", "C"),
            trace=RetrievalTrace(
                bm25_ids=("A", "C"),
                dense_ids=("B",),
                fused_ids=("A", "B", "C"),
                bm25_status="ok",
                dense_status="ok",
                used_fallback=False,
            ),
            probe_trace=RequirementProbeTrace(
                base_bm25_ids=("A",),
                supplemental_ids=("C",),
                status="ok",
                query_count=1,
            ),
        )
        backend = _TraceBackend(result)
        audit = ablations.ProbeCallAuditRetriever(backend, {"A", "B", "C"})

        observed = audit.search_with_trace(
            "dense",
            "lexical",
            route_weights=RouteWeights(bm25=0.5, dense=0.5),
            requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
            requirement_probe_candidates=("wide toe box",),
        )
        audit.candidate_documents(("A", "B", "C"))
        audit.validate(1)
        summary = audit.summary()

        self.assertIs(observed, result)
        self.assertIs(audit.requirement_probe_capability, REQUIREMENT_PROBE_CAPABILITY)
        self.assertEqual(summary["probe_bm25_queries"], 1)
        self.assertEqual(summary["max_probe_queries_per_search"], 1)
        self.assertEqual(summary["candidate_document_calls"], 1)
        self.assertEqual(summary["incumbent_loss_violations"], 0)
        self.assertEqual(summary["probe_overlap_violations"], 0)
        serialized = json.dumps(summary, sort_keys=True)
        self.assertNotIn('"A"', serialized)
        self.assertNotIn('"B"', serialized)
        self.assertNotIn('"C"', serialized)

    def test_probe_health_is_partitioned_and_bounded(self) -> None:
        candidate = {
            key: (
                CATALOG_IDF_REQUIREMENT_PROBE_POLICY.value
                if key == "policy"
                else 0
            )
            for key in ablations.PROBE_HEALTH_KEYS
        }
        candidate.update(
            {
                "attempts": 3,
                "no_eligible": 1,
                "successful_supplements": 2,
                "selected_probe_queries": 4,
                "supplemental_ids": 7,
            }
        )
        projected = ablations._project_probe_health(
            candidate,
            policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        )
        self.assertEqual(projected, candidate)

        invalid = dict(candidate)
        invalid["selected_probe_queries"] = 7
        with self.assertRaises(RuntimeError):
            ablations._project_probe_health(
                invalid,
                policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
            )

        disabled = {
            key: DISABLED_REQUIREMENT_PROBE_POLICY.value if key == "policy" else 0
            for key in ablations.PROBE_HEALTH_KEYS
        }
        self.assertEqual(
            ablations._project_probe_health(
                disabled,
                policy=DISABLED_REQUIREMENT_PROBE_POLICY,
            ),
            disabled,
        )

    def test_lazy_probe_vocabulary_is_initialized_without_query_work(self) -> None:
        backend = _LazyVocabularyBackend()
        ablations._ensure_probe_vocabulary(backend)
        self.assertEqual(backend.initializations, 1)
        self.assertTrue(backend.requirement_probe_vocabulary_available)

        backend.requirement_probe_capability = object()
        with self.assertRaises(RuntimeError):
            ablations._ensure_probe_vocabulary(backend)

    def test_fresh_loader_uses_whole_rows_and_requires_two_manual_messages(self) -> None:
        first = {
            "sample_id": "first",
            "scenario_type": "buying",
            "user_profile": {},
            "ground_truth": {"parent_asin": "TARGET"},
            "phase14_messages": ["first message", "second message"],
        }
        second = {**first, "sample_id": "second"}
        fingerprints = {
            ablations._fresh_row_fingerprint(first),
            ablations._fresh_row_fingerprint(second),
        }
        config = ablations.SuiteConfig(
            "synthetic",
            Path("unused.jsonl"),
            "a" * 64,
            2,
            2,
            ablations._fingerprint_set_digest(fingerprints),
            False,
            True,
        )

        with mock.patch.object(ablations, "load_jsonl", return_value=[first, second]):
            selected, evidence = ablations._load_suite_samples(config)

        self.assertEqual(selected, [first, second])
        self.assertEqual(evidence["manual_message_cases"], 2)
        self.assertEqual(evidence["manual_messages_per_case"], 2)

        malformed = {**first, "phase14_messages": ["only one"]}
        with mock.patch.object(ablations, "load_jsonl", return_value=[malformed]):
            bad_config = ablations.SuiteConfig(
                "bad",
                Path("unused.jsonl"),
                "a" * 64,
                1,
                1,
                "b" * 64,
                False,
                True,
            )
            with self.assertRaises(RuntimeError):
                ablations._load_suite_samples(bad_config)

    def test_manual_evaluator_uses_locked_first_two_messages_and_stable_id(self) -> None:
        sample = {
            "sample_id": "opaque",
            "scenario_type": "buying",
            "user_profile": {"preference_tags": []},
            "ground_truth": {"parent_asin": "TARGET"},
            "intent_card": {
                "hard_constraints": ["first constraint", "second constraint"],
                "soft_preferences": [],
            },
            "behavior": {},
            "phase14_messages": ["locked first", "locked second"],
        }
        agent = _ManualAgent("TARGET")

        result = ablations._evaluate_manual_messages(
            agent,  # type: ignore[arg-type]
            [sample],
            {"TARGET"},
            {},
            {},
        )

        self.assertEqual([call[1] for call in agent.messages], [
            "locked first",
            "locked second",
        ])
        self.assertEqual(
            agent.reset_calls[0][0],
            "public_00000000000000000000000000000001",
        )
        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(result["mttc"], 2.0)
        self.assertEqual(result["mrr"], 1.0)

    def test_suite_specific_bootstrap_threshold_is_frozen(self) -> None:
        baseline = _run(mrr=0.5, mttc=3.0, score=0.75)
        candidate = _run(mrr=0.51, mttc=3.0, score=0.76)

        fresh = ablations._quality_gates(
            ablations.SUITES["fresh"],
            baseline,
            candidate,
            _paired(0.0),
        )
        development = ablations._quality_gates(
            ablations.SUITES["development"],
            baseline,
            candidate,
            _paired(0.0),
        )

        self.assertTrue(all(fresh.values()))
        self.assertFalse(
            development["paired_bootstrap_lower_95_passes_suite_threshold"]
        )
        self.assertEqual(sum(not value for value in development.values()), 1)

    def test_publication_validator_rejects_private_rows_messages_and_ids(self) -> None:
        payload = _privacy_payload()
        self.assertTrue(ablations.publication_privacy_is_valid(payload))

        for mutation in (
            {"sessions": []},
            {"phase14_messages": ["secret"]},
            {"product": "B000000001"},
            {"small_cell": "buying"},
        ):
            with self.subTest(mutation=mutation):
                unsafe = copy.deepcopy(payload)
                unsafe["health"].update(mutation)
                self.assertFalse(ablations.publication_privacy_is_valid(unsafe))

    def test_attempt_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            ablations._claim_attempt(path, "fresh")
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ablations._claim_attempt(path, "fresh")
            self.assertEqual(path.read_bytes(), first)

    def test_prelock_verification_is_exact_and_rejects_oracle_drift(self) -> None:
        verification = _verification()
        self.assertEqual(
            ablations._validate_prelock_verification(verification),
            verification,
        )

        drifted = dict(verification)
        drifted["phase14_oracle_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            ablations._validate_prelock_verification(drifted)

    def test_startup_probe_keeps_disabled_path_unmodified_and_candidate_bounded(self) -> None:
        probe = ablations._startup_probe(object(), iterations=100)
        self.assertEqual(probe["iterations"], 100)
        self.assertGreaterEqual(
            probe["candidate_empty_retained_bytes"],
            probe["baseline_empty_retained_bytes"],
        )
        self.assertGreater(float(probe["candidate_startup_time_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
