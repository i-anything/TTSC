from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.ranking import (
    COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
)
from conversational_search.service import ConversationalSearchAgent
from scripts.run_bm25_rescue_ablations import (
    BASELINE_ID,
    CALL_ACCOUNTING_KEYS,
    CANDIDATE_ID,
    CONTRACT_RELATIVE,
    DECISION_GATE_KEYS,
    EXACTNESS_KEYS,
    EXPERIMENT_ID,
    FOCUSED_SUITE_COMMAND,
    FROZEN_INPUT_SHA256,
    FULL_SUITE_COMMAND,
    IMPLEMENTATION_LOCK_ID,
    IMPLEMENTATION_LOCK_RELATIVE,
    PHASE10_FORMULA_ORACLE_CASES,
    PHASE10_PHASE9_ORACLE_CASES,
    PHASE10_PHASE9_ORACLE_COMMAND,
    PHASE10_PHASE9_ORACLE_SHA256,
    PHASE7_ORACLE_CASES,
    PHASE7_ORACLE_COMMAND,
    PHASE7_ORACLE_SHA256,
    PHASE9_ORACLE_CASES,
    PHASE9_ORACLE_COMMAND,
    PHASE9_ORACLE_SHA256,
    RAW_RESULT_RELATIVE,
    SOURCE_PATHS,
    _build_decision_gates,
    _call_accounting,
    _canonical_private_cache_shape,
    _canonical_private_cache_snapshot,
    _claim_run_output,
    _evaluate_with_deterministic_session_ids,
    _inspect_retained_rescue_state,
    _overall_official_summary,
    _project_rescue_health,
    _publication_privacy_is_valid,
    _validate_implementation_lock,
    _validate_output,
    _validate_prelock_verification,
    _validate_publication_privacy,
    _validate_variant_accounting,
    run_bm25_rescue_ablations,
)
from scripts.run_fusion_ablations import _sha256
from tests.test_service import CacheableRecordingRetriever


def _verification(*, focused: int = 20, complete: int = 40) -> dict:
    return {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "focused_tests_passed": focused,
        "complete_suite_command": FULL_SUITE_COMMAND,
        "complete_unit_tests_passed": complete,
        "phase10_formula_oracle_cases": PHASE10_FORMULA_ORACLE_CASES,
        "phase7_exact_oracle_command": PHASE7_ORACLE_COMMAND,
        "phase7_exact_oracle_cases": PHASE7_ORACLE_CASES,
        "phase7_exact_oracle_sha256": PHASE7_ORACLE_SHA256,
        "phase9_exact_oracle_command": PHASE9_ORACLE_COMMAND,
        "phase9_exact_oracle_cases": PHASE9_ORACLE_CASES,
        "phase9_exact_oracle_sha256": PHASE9_ORACLE_SHA256,
        "phase10_phase9_exact_oracle_command": PHASE10_PHASE9_ORACLE_COMMAND,
        "phase10_phase9_exact_oracle_cases": PHASE10_PHASE9_ORACLE_CASES,
        "phase10_phase9_exact_oracle_sha256": PHASE10_PHASE9_ORACLE_SHA256,
        "completed_before_lock": True,
    }


def _profile_health(*, sessions: int = 200) -> dict:
    return {
        "policy": BOUNDED_RESIDUAL_PROFILE_POLICY.value,
        "session_entries": sessions,
        "logical_profile_bytes": sessions * 2,
        "profiles_reset": sessions,
        "zero_mask_profiles": sessions,
        "nonzero_mask_profiles": 0,
        "recognized_theme_count": 0,
        "turns_disabled_by_active_requirements": 0,
        "eligible_stage_a_attempts": 0,
        "empty_represented_theme_fallbacks": 0,
        "constant_score_neutral_fallbacks": 0,
        "successful_residual_applications": 0,
        "parsing_or_scoring_fallbacks": 0,
    }


def _rescue_health(*, candidate: bool, attempts: int) -> dict:
    return {
        "policy": (
            COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value
            if candidate
            else STAGE_A_RANKING_POLICY.value
        ),
        "attempts": attempts,
        "zero_completeness_neutral": 0,
        "bm25_unavailable_or_empty_neutral": 0,
        "no_positive_uplift_neutral": 0,
        "constant_uplift_neutral": 0,
        "unchanged_order_neutral": 0,
        "successful_reorders": attempts,
        "validation_or_scoring_fallbacks": 0,
    }


def _diagnostics(
    *,
    candidate: bool,
    searches: int,
    turns: int,
    wall: float,
    p95: float,
) -> dict:
    policy = (
        COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value
        if candidate
        else STAGE_A_RANKING_POLICY.value
    )
    reuses = turns - searches
    return {
        "expected_turns": turns,
        "route_health": {
            "bm25": {"ok": searches},
            "dense": {"ok": searches},
            "fallback_turns": 0,
            "candidate_document_calls": searches,
        },
        "ranking_health": {
            "policy": policy,
            "attempts": searches,
            "successes": searches,
            "failures": 0,
            "unavailable_skips": 0,
        },
        "rescue_health": _rescue_health(
            candidate=candidate,
            attempts=searches if candidate else 0,
        ),
        "profile_health": _profile_health(),
        "slate_health": {
            "policy": "stagnation_aware",
            "attempts": turns,
            "successes": turns,
            "failures": 0,
            "initializations": 200,
            "ranking_resets": 0,
            "stagnant_turns": 0,
            "unseen_selected_on_stagnant": 0,
            "repeat_backfills": 0,
        },
        "orchestration_health": {
            "policy": "exact_ranking_reuse",
            "capacity": 256,
            "maximum_ids_per_entry": 200,
            "maximum_id_characters": 64,
            "decisions": turns,
            "searches": searches,
            "reuses": reuses,
            "skips": 0,
            "reasons": {"cold_cache": searches, "exact_dependency_hit": reuses},
            "lookups": turns,
            "hits": reuses,
            "cold_misses": searches,
            "dependency_misses": 0,
            "backend_invalidations": 0,
            "fault_invalidations": 0,
            "stores": searches,
            "store_rejections": 0,
            "reset_invalidations": 0,
            "capacity_evictions": 0,
            "retrievals_avoided": reuses,
            "reranks_avoided": reuses,
            "entries": 200,
            "cached_id_references": 400,
            "cached_id_utf8_bytes": 4000,
            "retained_cache_bytes": 8000,
        },
        "retained_profile_state_valid": True,
        "retained_rescue_state_valid": True,
        "evaluation_wall_seconds": wall,
        "respond_latency_ms": {
            "count": turns,
            "warm_count": turns - 1,
            "p50": p95,
            "p90": p95,
            "p95": p95,
            "p99": p95,
            "warm_p95": p95,
            "max": p95,
            "total": p95 * turns,
        },
    }


def _result(*, mrr: float, mttc: float, score: float, efficiency: float) -> dict:
    return {
        "sample_count": 200,
        "hit_rate_at_10": 0.99,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": efficiency,
        "recommended_technical_score": score,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "scenario_metrics": {},
        "sessions": [],
    }


def _publication() -> dict:
    baseline_result = _result(
        mrr=0.529558,
        mttc=3.065,
        score=0.812567,
        efficiency=0.7935,
    )
    candidate_result = _result(
        mrr=0.54,
        mttc=3.0,
        score=0.82,
        efficiency=0.8,
    )
    baseline = _diagnostics(
        candidate=False,
        searches=10,
        turns=12,
        wall=10.0,
        p95=20.0,
    )
    candidate = _diagnostics(
        candidate=True,
        searches=9,
        turns=12,
        wall=10.4,
        p95=20.8,
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "candidate": CANDIDATE_ID,
        "baseline": BASELINE_ID,
        "run_configuration": {
            "execution": "strictly_sequential",
            "onnx_threads": 1,
            "shared_immutable_backend_for_ablation_variants": True,
            "fresh_agent_state_per_variant": True,
            "run_order": [
                CANDIDATE_ID,
                BASELINE_ID,
                CANDIDATE_ID,
                "independent_starter_agent_default",
            ],
            "backend_warmup": "one_fixed_unlabeled_request",
            "external_api_calls": 0,
            "post_run_tuning_or_second_candidate_run_allowed": False,
        },
        "official_metrics": {
            "baseline": _overall_official_summary(baseline_result),
            "candidate": _overall_official_summary(candidate_result),
            "delta": {
                "hit_rate_at_10": 0.0,
                "mrr": 0.010442,
                "mttc": -0.065,
                "efficiency": 0.0065,
                "recommended_technical_score": 0.007433,
            },
        },
        "paired_quality": {"phase9_hit_to_phase10_miss_count": 0},
        "health": {
            "phase9_baseline": baseline,
            "phase10_candidate": candidate,
            "phase10_replay": candidate,
            "independent_phase10": candidate,
        },
        "call_accounting": {
            "baseline": _call_accounting(baseline),
            "candidate": _call_accounting(candidate),
        },
        "latency": {
            "baseline_wall_seconds": 10.0,
            "candidate_wall_seconds": 10.4,
            "candidate_wall_time_ratio": 1.04,
            "baseline_warm_p95_ms": 20.0,
            "candidate_warm_p95_ms": 20.8,
            "candidate_warm_p95_ratio": 1.04,
        },
        "exactness": {key: True for key in EXACTNESS_KEYS},
        "privacy": {
            "queries_messages_profiles_and_tags_absent": True,
            "product_sample_session_and_turn_rows_absent": True,
            "targets_scenarios_ranks_scores_and_action_traces_absent": True,
            "private_cache_snapshots_absent": True,
            "rescue_telemetry_is_fixed_global_counters_only": True,
        },
        "reproducibility": {
            "platform": "synthetic-platform",
            "python": "3.synthetic",
            "frozen_input_sha256": {
                key: "a" * 64 for key in FROZEN_INPUT_SHA256
            },
            "implementation_lock_id": IMPLEMENTATION_LOCK_ID,
            "contract_sha256": "b" * 64,
            "source_sha256": {key: "c" * 64 for key in SOURCE_PATHS},
            "pre_lock_verification": _verification(),
            "lock_revalidated_after_independent": True,
        },
        "decision_gate": {key: True for key in DECISION_GATE_KEYS},
    }


class RescueHealthTests(unittest.TestCase):
    def test_candidate_outcomes_exactly_partition_attempts(self) -> None:
        health = _rescue_health(candidate=True, attempts=7)
        self.assertEqual(
            _project_rescue_health(
                health,
                expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            ),
            health,
        )

        invalid = {**health, "successful_reorders": 6}
        with self.assertRaisesRegex(RuntimeError, "partition"):
            _project_rescue_health(
                invalid,
                expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            )

    def test_phase9_comparator_must_have_zero_rescue_activity(self) -> None:
        health = _rescue_health(candidate=False, attempts=0)
        self.assertEqual(
            _project_rescue_health(
                health,
                expected_policy=STAGE_A_RANKING_POLICY,
            ),
            health,
        )
        invalid = {**health, "attempts": 1, "successful_reorders": 1}
        with self.assertRaisesRegex(RuntimeError, "comparator"):
            _project_rescue_health(
                invalid,
                expected_policy=STAGE_A_RANKING_POLICY,
            )

    def test_variant_accounting_is_exact_for_candidate_and_comparator(self) -> None:
        candidate = _diagnostics(
            candidate=True,
            searches=9,
            turns=12,
            wall=1.0,
            p95=1.0,
        )
        baseline = _diagnostics(
            candidate=False,
            searches=10,
            turns=12,
            wall=1.0,
            p95=1.0,
        )
        _validate_variant_accounting(
            candidate,
            expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
        )
        _validate_variant_accounting(
            baseline,
            expected_policy=STAGE_A_RANKING_POLICY,
        )
        self.assertEqual(
            set(_call_accounting(candidate)),
            set(CALL_ACCOUNTING_KEYS),
        )


class RescueCacheAndStateTests(unittest.TestCase):
    def test_cache_snapshot_is_complete_canonical_and_private(self) -> None:
        identifiers = ("SYNTHETIC-ONE", "SYNTHETIC-TWO")
        retriever = CacheableRecordingRetriever(
            identifiers,
            fused_ids=identifiers,
            documents={
                identifiers[0]: "ordinary synthetic item",
                identifiers[1]: "durable synthetic item",
            },
        )
        agent = ConversationalSearchAgent(
            "unused-synthetic.jsonl",
            retriever=retriever,
            ranking_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
        )
        agent.reset("private-synthetic-session", {})
        agent.respond(
            "private-synthetic-session",
            "I'm looking for synthetic shoes, but I'm still exploring.",
            1,
            2,
        )

        first = _canonical_private_cache_snapshot(agent)
        second = _canonical_private_cache_snapshot(agent)
        self.assertEqual(first, second)
        self.assertEqual(len(first[2]), 1)
        self.assertEqual(
            _canonical_private_cache_shape(first),
            ("dependency_digest", "backend_snapshot_token", "ranked_ids"),
        )

    def test_rescue_state_inspector_accepts_counters_and_rejects_scores(self) -> None:
        agent = ConversationalSearchAgent(
            "unused-synthetic.jsonl",
            retriever=object(),
            ranking_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
        )
        self.assertTrue(
            _inspect_retained_rescue_state(
                agent,
                expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            )
        )
        agent._candidate_scores = (0.5,)  # type: ignore[attr-defined]
        self.assertFalse(
            _inspect_retained_rescue_state(
                agent,
                expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            )
        )
        unexpected_state = {
            "_candidate_ids": ("PRIVATEPRODUCT",),
            "_product_documents": {"PRIVATE": "text"},
            "_memo": ("PRIVATEPRODUCT",),
        }
        for name, value in unexpected_state.items():
            with self.subTest(name=name):
                extra_agent = ConversationalSearchAgent(
                    "unused-synthetic.jsonl",
                    retriever=object(),
                    ranking_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
                )
                setattr(extra_agent, name, value)
                self.assertFalse(
                    _inspect_retained_rescue_state(
                        extra_agent,
                        expected_policy=(
                            COMPLETENESS_BM25_RESCUE_RANKING_POLICY
                        ),
                    )
                )


class RescueDecisionGateTests(unittest.TestCase):
    def test_every_frozen_gate_can_pass(self) -> None:
        baseline = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        candidate = _result(mrr=0.54, mttc=3.0, score=0.82, efficiency=0.8)
        baseline_diagnostics = _diagnostics(
            candidate=False,
            searches=10,
            turns=12,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            candidate=True,
            searches=9,
            turns=12,
            wall=10.4,
            p95=20.8,
        )
        trace = [("private-exact-trace",)]
        cache = ("exact_ranking_reuse", 256, ())
        gates, latency = _build_decision_gates(
            baseline_run=(baseline, baseline_diagnostics, trace, cache),
            candidate_run=(candidate, candidate_diagnostics, trace, cache),
            replay_run=(candidate, candidate_diagnostics, trace, cache),
            independent_run=(candidate, candidate_diagnostics, trace, cache),
            hit_to_miss_count=0,
            implementation_lock={"verification": _verification()},
            publication_privacy_valid=True,
            implementation_lock_revalidated=True,
        )

        self.assertTrue(gates["adopt"])
        self.assertEqual(latency["candidate_wall_time_ratio"], 1.04)
        self.assertEqual(latency["candidate_warm_p95_ratio"], 1.04)

    def test_quality_regression_cache_drift_or_extra_calls_rejects(self) -> None:
        baseline = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        candidate = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        baseline_diagnostics = _diagnostics(
            candidate=False,
            searches=9,
            turns=12,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            candidate=True,
            searches=10,
            turns=12,
            wall=10.6,
            p95=21.2,
        )
        trace = [("private-exact-trace",)]
        cache = ("exact_ranking_reuse", 256, ())
        drifted_cache = ("exact_ranking_reuse", 256, (("drift",),))
        gates, _ = _build_decision_gates(
            baseline_run=(baseline, baseline_diagnostics, trace, cache),
            candidate_run=(candidate, candidate_diagnostics, trace, cache),
            replay_run=(candidate, candidate_diagnostics, trace, drifted_cache),
            independent_run=(candidate, candidate_diagnostics, trace, cache),
            hit_to_miss_count=1,
            implementation_lock={"verification": _verification()},
            publication_privacy_valid=False,
            implementation_lock_revalidated=False,
        )

        self.assertFalse(gates["candidate_mrr_strictly_above_0_529558"])
        self.assertFalse(gates["phase9_hit_to_phase10_miss_count_is_zero"])
        self.assertFalse(
            gates["candidate_total_counted_calls_not_above_comparator"]
        )
        self.assertFalse(
            gates["candidate_replay_payload_intent_slate_cache_and_health_exact"]
        )
        self.assertFalse(gates["adopt"])

    def test_nonzero_reported_model_or_api_tokens_rejects(self) -> None:
        baseline = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        candidate = _result(mrr=0.54, mttc=3.0, score=0.82, efficiency=0.8)
        candidate["reported_token_usage"] = {
            "prompt_tokens": 1,
            "completion_tokens": 0,
            "total_tokens": 1,
        }
        baseline_diagnostics = _diagnostics(
            candidate=False,
            searches=10,
            turns=12,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            candidate=True,
            searches=9,
            turns=12,
            wall=10.4,
            p95=20.8,
        )
        trace = [("private-exact-trace",)]
        cache = ("exact_ranking_reuse", 256, ())

        gates, _ = _build_decision_gates(
            baseline_run=(baseline, baseline_diagnostics, trace, cache),
            candidate_run=(candidate, candidate_diagnostics, trace, cache),
            replay_run=(candidate, candidate_diagnostics, trace, cache),
            independent_run=(candidate, candidate_diagnostics, trace, cache),
            hit_to_miss_count=0,
            implementation_lock={"verification": _verification()},
            publication_privacy_valid=True,
            implementation_lock_revalidated=True,
        )

        self.assertFalse(
            gates["candidate_and_comparator_report_zero_model_api_tokens"]
        )
        self.assertFalse(gates["adopt"])

    def test_backend_snapshot_invalidation_rejects(self) -> None:
        baseline = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        candidate = _result(mrr=0.54, mttc=3.0, score=0.82, efficiency=0.8)
        baseline_diagnostics = _diagnostics(
            candidate=False, searches=10, turns=12, wall=10.0, p95=20.0
        )
        candidate_diagnostics = _diagnostics(
            candidate=True, searches=9, turns=12, wall=10.4, p95=20.8
        )
        candidate_diagnostics["orchestration_health"]["backend_invalidations"] = 1
        candidate_diagnostics["orchestration_health"]["reasons"] = {
            **candidate_diagnostics["orchestration_health"]["reasons"],
            "backend_snapshot_changed": 1,
        }
        trace = [("private-exact-trace",)]
        cache = ("exact_ranking_reuse", 256, ())

        gates, _ = _build_decision_gates(
            baseline_run=(baseline, baseline_diagnostics, trace, cache),
            candidate_run=(candidate, candidate_diagnostics, trace, cache),
            replay_run=(candidate, candidate_diagnostics, trace, cache),
            independent_run=(candidate, candidate_diagnostics, trace, cache),
            hit_to_miss_count=0,
            implementation_lock={"verification": _verification()},
            publication_privacy_valid=True,
            implementation_lock_revalidated=True,
        )

        self.assertFalse(gates["candidate_and_comparator_faults_are_zero"])
        self.assertFalse(gates["adopt"])


class RescuePrivacyTests(unittest.TestCase):
    def test_publication_accepts_only_complete_aggregate_payload(self) -> None:
        payload = _publication()
        self.assertTrue(_publication_privacy_is_valid(payload))
        _validate_publication_privacy(payload)
        self.assertNotIn("sessions", json.dumps(payload, sort_keys=True))

    def test_publication_rejects_rows_scenarios_ids_traces_and_cache(self) -> None:
        mutations = (
            lambda payload: payload.update({"sessions": []}),
            lambda payload: payload["health"].update({"sample_id": "private"}),
            lambda payload: payload["health"].update({"scenario_metrics": {}}),
            lambda payload: payload["health"].update({"parent_asin": "synthetic"}),
            lambda payload: payload["health"].update({"value": "B012345678"}),
            lambda payload: payload["exactness"].update({"action_trace": []}),
            lambda payload: payload["exactness"].update({"cache_snapshot": []}),
        )
        for mutate in mutations:
            payload = _publication()
            mutate(payload)
            with self.subTest(payload=payload):
                self.assertFalse(_publication_privacy_is_valid(payload))
                with self.assertRaises(RuntimeError):
                    _validate_publication_privacy(payload)

    def test_publication_rejects_identity_metric_and_policy_drift(self) -> None:
        mutations = (
            lambda payload: payload.update({"schema_version": 999}),
            lambda payload: payload.update({"experiment_id": "wrong"}),
            lambda payload: payload.update({"candidate": "wrong"}),
            lambda payload: payload.update({"baseline": "wrong"}),
            lambda payload: payload["official_metrics"]["candidate"].update(
                {"mrr": float("nan")}
            ),
            lambda payload: payload["official_metrics"]["candidate"].update(
                {"mrr": "private user message"}
            ),
            lambda payload: payload["official_metrics"]["baseline"].update(
                {"sample_count": True}
            ),
            lambda payload: payload["official_metrics"]["delta"].update(
                {"mrr": float("inf")}
            ),
            lambda payload: payload["health"]["phase10_candidate"][
                "slate_health"
            ].update({"policy": "wrong"}),
            lambda payload: payload["health"]["phase10_candidate"][
                "orchestration_health"
            ].update({"policy": "wrong"}),
        )
        for mutate in mutations:
            payload = _publication()
            mutate(payload)
            with self.subTest(mutation=mutate):
                self.assertFalse(_publication_privacy_is_valid(payload))
                with self.assertRaises(RuntimeError):
                    _validate_publication_privacy(payload)


class RescueLockAndOutputTests(unittest.TestCase):
    def _manual_lock_repository(
        self,
        directory: str,
    ) -> tuple[Path, tuple[str, ...], dict]:
        root = Path(directory)
        contract = root / CONTRACT_RELATIVE
        contract.parent.mkdir(parents=True)
        contract.write_text('{"frozen":true}\n', encoding="utf-8")
        source = root / "candidate.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        sources = ("candidate.py",)
        lock = {
            "schema_version": 1,
            "lock_id": IMPLEMENTATION_LOCK_ID,
            "status": "locked_before_public_confirmation",
            "contract_sha256": _sha256(contract),
            "source_sha256": {"candidate.py": _sha256(source)},
            "verification": _verification(),
        }
        lock_path = root / IMPLEMENTATION_LOCK_RELATIVE
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        return root, sources, lock

    def test_manually_patched_lock_validates_and_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, sources, lock = self._manual_lock_repository(directory)
            self.assertEqual(
                _validate_implementation_lock(root, source_paths=sources),
                lock,
            )
            (root / sources[0]).write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "drifted"):
                _validate_implementation_lock(root, source_paths=sources)

    def test_prelock_verification_rejects_counts_or_oracle_drift(self) -> None:
        self.assertEqual(_validate_prelock_verification(_verification()), _verification())
        for invalid in (
            {**_verification(), "focused_tests_passed": 0},
            {**_verification(), "phase9_exact_oracle_sha256": "0" * 64},
            {**_verification(), "phase10_phase9_exact_oracle_sha256": "f" * 64},
            {**_verification(), "phase10_formula_oracle_cases": 9_999},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    _validate_prelock_verification(invalid)

    def test_exclusive_run_claim_survives_and_refuses_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            _claim_run_output(output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "experiment_id": EXPERIMENT_ID,
                    "status": "run_started_no_retry",
                },
            )
            with self.assertRaises(FileExistsError):
                _claim_run_output(output)

    def test_evaluator_session_ids_are_unique_and_exactly_replayable(self) -> None:
        from evaluator import local_evaluator

        samples = [{}, {}, {}]

        def fake_evaluate(*_args: object) -> dict:
            return {
                "session_ids": [
                    local_evaluator.uuid.uuid4().hex for _sample in samples
                ]
            }

        with patch(
            "scripts.run_bm25_rescue_ablations.evaluate",
            side_effect=fake_evaluate,
        ):
            first = _evaluate_with_deterministic_session_ids(
                Mock(), samples, set(), {}, {}
            )
            second = _evaluate_with_deterministic_session_ids(
                Mock(), samples, set(), {}, {}
            )

        self.assertEqual(first, second)
        self.assertEqual(len(set(first["session_ids"])), len(samples))

    def test_output_guard_accepts_only_the_frozen_root_path(self) -> None:
        root = Path(__file__).resolve().parents[1]
        catalog = root / "data" / "catalog.jsonl"
        dataset = root / "data" / "public_set.jsonl"
        with patch.object(Path, "exists", return_value=False):
            _validate_output(root / RAW_RESULT_RELATIVE, catalog, dataset)
            with self.assertRaises(ValueError):
                _validate_output(root / "different.json", catalog, dataset)

    def test_lock_failure_precedes_claim_and_dataset_load(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with (
            patch(
                "scripts.run_bm25_rescue_ablations._validate_frozen_inputs",
                return_value={},
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._validate_implementation_lock",
                side_effect=RuntimeError("synthetic lock failure"),
            ),
            patch("scripts.run_bm25_rescue_ablations._claim_run_output") as claim,
            patch("scripts.run_bm25_rescue_ablations.load_jsonl") as load,
        ):
            with self.assertRaisesRegex(RuntimeError, "lock failure"):
                run_bm25_rescue_ablations(
                    "synthetic-catalog.jsonl",
                    "synthetic-dataset.jsonl",
                    root / RAW_RESULT_RELATIVE,
                )
        claim.assert_not_called()
        load.assert_not_called()

    def test_sealed_order_and_second_lock_validation_use_only_mocks(self) -> None:
        baseline = _result(
            mrr=0.529558,
            mttc=3.065,
            score=0.812567,
            efficiency=0.7935,
        )
        candidate = _result(mrr=0.54, mttc=3.0, score=0.82, efficiency=0.8)
        baseline_diagnostics = _diagnostics(
            candidate=False,
            searches=10,
            turns=12,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            candidate=True,
            searches=9,
            turns=12,
            wall=10.4,
            p95=20.8,
        )
        trace: list[tuple[object, object, object]] = []
        cache = ("exact_ranking_reuse", 256, ())
        lock = {
            "lock_id": IMPLEMENTATION_LOCK_ID,
            "contract_sha256": "a" * 64,
            "source_sha256": {},
            "verification": _verification(),
        }
        runtime = Mock()
        runtime.retrieval_backend = Mock(dense_available=True, bm25_available=True)
        root = Path(__file__).resolve().parents[1]
        variants = (
            (candidate, candidate_diagnostics, trace, cache),
            (baseline, baseline_diagnostics, trace, cache),
            (candidate, candidate_diagnostics, trace, cache),
        )
        variant_results = iter(variants)
        events: list[str] = []

        def validate_lock_event(*_args: object, **_kwargs: object) -> dict:
            events.append("lock")
            return lock

        def claim_event(*_args: object, **_kwargs: object) -> None:
            events.append("claim")

        def load_event(*_args: object, **_kwargs: object) -> list[dict]:
            events.append("load")
            return []

        def warm_event(_catalog: object, backend: object) -> None:
            self.assertIs(backend, runtime.retrieval_backend)
            events.append("warm")

        def variant_event(*args: object, **_kwargs: object) -> tuple:
            backend = args[-2]
            policy = args[-1]
            self.assertIs(backend, runtime.retrieval_backend)
            events.append(
                "candidate"
                if policy is COMPLETENESS_BM25_RESCUE_RANKING_POLICY
                else "baseline"
            )
            return next(variant_results)

        def independent_event(*_args: object, **_kwargs: object) -> tuple:
            events.append("independent")
            return (candidate, candidate_diagnostics, trace, cache)

        with (
            patch(
                "scripts.run_bm25_rescue_ablations._validate_frozen_inputs",
                return_value={},
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._validate_implementation_lock",
                side_effect=validate_lock_event,
            ) as validate_lock,
            patch("scripts.run_bm25_rescue_ablations._validate_output"),
            patch(
                "scripts.run_bm25_rescue_ablations._claim_run_output",
                side_effect=claim_event,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations.load_jsonl",
                side_effect=load_event,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations.catalog_index",
                return_value=(set(), {}, {}),
            ),
            patch(
                "scripts.run_bm25_rescue_ablations.ConversationalSearchAgent",
                return_value=runtime,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._warm_backend",
                side_effect=warm_event,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._run_variant",
                side_effect=variant_event,
            ) as run_variant,
            patch(
                "scripts.run_bm25_rescue_ablations._run_independent",
                side_effect=independent_event,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._paired_hit_to_miss_count",
                return_value=0,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._publication_privacy_is_valid",
                return_value=True,
            ),
            patch(
                "scripts.run_bm25_rescue_ablations._build_decision_gates",
                return_value=(
                    {"adopt": True},
                    {
                        "baseline_wall_seconds": 10.0,
                        "candidate_wall_seconds": 10.4,
                        "candidate_wall_time_ratio": 1.04,
                        "baseline_warm_p95_ms": 20.0,
                        "candidate_warm_p95_ms": 20.8,
                        "candidate_warm_p95_ratio": 1.04,
                    },
                ),
            ),
            patch("scripts.run_bm25_rescue_ablations._validate_publication_privacy"),
            patch("scripts.run_bm25_rescue_ablations._write_json_atomic"),
        ):
            run_bm25_rescue_ablations(
                "synthetic-catalog.jsonl",
                "synthetic-dataset.jsonl",
                root / RAW_RESULT_RELATIVE,
            )

        self.assertEqual(validate_lock.call_count, 2)
        self.assertEqual(
            events,
            [
                "lock",
                "claim",
                "load",
                "warm",
                "candidate",
                "baseline",
                "candidate",
                "independent",
                "lock",
            ],
        )
        self.assertEqual(
            [call.args[-1] for call in run_variant.call_args_list],
            [
                COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
                STAGE_A_RANKING_POLICY,
                COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            ],
        )


if __name__ == "__main__":
    unittest.main()
