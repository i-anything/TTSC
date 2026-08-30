from __future__ import annotations

import copy
import importlib
import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from conversational_search.decision_policy import (
    PROTECTED_DECISION_POLICY,
    PROTOCOL_UTILITY_DECISION_POLICY,
)
import evaluator.local_evaluator as evaluator_module
from scripts import run_protocol_utility_ablations as ablations


def _summary(*, candidate: bool = False) -> dict:
    return {
        "sample_count": 4,
        "hit_rate_at_10": 0.75,
        "mrr": 0.51 if candidate else 0.50,
        "mttc": 2.9 if candidate else 3.0,
        "efficiency": 0.81 if candidate else 0.80,
        "recommended_technical_score": 0.69 if candidate else 0.685,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _protocol_health(*, candidate: bool, turns: int = 4) -> dict:
    outcomes = {key: 0 for key in ablations.PROTOCOL_OUTCOMES}
    questions = {key: 0 for key in ablations.PROTOCOL_QUESTION_ACTIONS}
    widths = {str(width): 0 for width in range(11)}
    if candidate:
        outcomes["applied"] = turns
        questions["other"] = turns
        widths["1"] = turns
    return {
        "policy": (
            PROTOCOL_UTILITY_DECISION_POLICY.value
            if candidate
            else PROTECTED_DECISION_POLICY.value
        ),
        "turns": turns if candidate else 0,
        **outcomes,
        "question_action_counts": questions,
        "width_action_counts": widths,
        "requested_total": turns * 10 if candidate else 0,
        "presented_total": turns if candidate else 0,
    }


def _calibration(turns: int) -> dict:
    value = ablations._empty_calibration_summary()
    value.update(
        {
            "observations": turns,
            "target_in_support": turns,
            "mean_multiclass_brier": 0.25,
            "ece_10": 0.1,
            "bin_counts": [turns, *([0] * 9)],
            "bin_mean_confidence": [0.5, *([0.0] * 9)],
            "bin_accuracy": [0.6, *([0.0] * 9)],
        }
    )
    return value


def _diagnostics(*, candidate: bool, turns: int = 4) -> dict:
    protocol = ablations._project_protocol_health(
        _protocol_health(candidate=candidate, turns=turns),
        expected_policy=(
            PROTOCOL_UTILITY_DECISION_POLICY
            if candidate
            else PROTECTED_DECISION_POLICY
        ),
        expected_turns=turns if candidate else 0,
    )
    orchestration = {
        key: 0 for key in ablations.ORCHESTRATION_HEALTH_KEYS
        if key not in {"policy", "reasons"}
    }
    orchestration.update(
        {
            "policy": ablations.EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value,
            "decisions": turns,
            "searches": turns,
            "reasons": {"search": turns},
            "lookups": turns,
            "cold_misses": turns,
        }
    )
    profile = {
        key: 0
        for key in ablations.PROFILE_HEALTH_KEYS
        if key != "policy"
    }
    profile.update(
        {
            "policy": ablations.BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            "session_entries": 4,
        }
    )
    latency = {
        "count": turns,
        "warm_count": max(0, turns - 1),
        "p50": 1.0,
        "p90": 1.0,
        "p95": 1.0,
        "p99": 1.0,
        "warm_p95": 1.0,
        "max": 1.0,
        "total": float(turns),
    }
    return {
        "expected_turns": turns,
        "route_health": {
            "bm25": {"ok": turns},
            "dense": {"skipped": turns} if candidate else {"ok": turns},
            "fallback_turns": 0,
            "candidate_document_calls": turns,
            "protocol_exact_candidate_calls": turns if candidate else 0,
            "protocol_candidate_evidence_calls": turns if candidate else 0,
        },
        "ranking_health": {
            "policy": "test-ranking-policy",
            "attempts": turns,
            "successes": turns,
            "failures": 0,
            "unavailable_skips": 0,
        },
        "rescue_health": {
            key: (
                "test-ranking-policy"
                if key == "policy"
                else 0
            )
            for key in ablations.RESCUE_HEALTH_KEYS
        },
        "route_redundancy_health": {
            "policy": "test-ranking-policy",
            "attempts": 0,
            "empty_exact_baseline": 0,
            "single_route_exact_baseline": 0,
            "disjoint_exact_baseline": 0,
            "identical_order_exact_baseline": 0,
            "correction_applied": 0,
            "validation_or_scoring_fallbacks": 0,
        },
        "intent_epoch_slate_health": {
            "policy": ablations.INTENT_EPOCH_NOVELTY_SLATE_POLICY.value,
            "attempts": 0,
            "empty_exact_baseline": 0,
            "first_slate_exact_baseline": 0,
            "unchanged_signature_exact_baseline": 0,
            "changed_epoch_exact_baseline": 0,
            "same_epoch_history_carried": 0,
            "validation_fallbacks": 0,
            "eligible_prior_shown_total": 0,
        },
        "profile_health": profile,
        "slate_health": {
            "policy": ablations.INTENT_EPOCH_NOVELTY_SLATE_POLICY.value,
            "attempts": turns,
            "successes": turns,
            "failures": 0,
            "initializations": 0,
            "ranking_resets": 0,
            "stagnant_turns": 0,
            "unseen_selected_on_stagnant": 0,
            "repeat_backfills": 0,
        },
        "orchestration_health": orchestration,
        "protocol_decision_health": protocol,
        "response_audit": {
            "response_exceptions": 0,
            "invalid_api_responses": 0,
        },
        "runtime_network_attempts": 0,
        "calibration": _calibration(turns) if candidate else (
            ablations._empty_calibration_summary()
        ),
        "retained_agent_bytes": 2_000 if candidate else 1_000,
        "protocol_retained_bytes": 1_000 if candidate else 0,
        "evaluation_wall_seconds": 1.0,
        "respond_latency_ms": latency,
    }


def _run(
    *,
    candidate: bool,
    digest: str,
    behavior: str | None = None,
) -> ablations.VariantRun:
    return ablations.VariantRun(
        summary=_summary(candidate=candidate),
        sessions=[],
        diagnostics=_diagnostics(candidate=candidate),
        evaluator_digest=digest,
        behavior_digest=behavior or digest,
        private_digest=digest,
    )


def _paired(lower: float = 0.001) -> dict:
    return {
        "transitions": {
            "both_hit": 3,
            "candidate_only_hit": 0,
            "baseline_only_hit": 0,
            "both_miss": 1,
        },
        "mean_utility_delta": 0.01,
        "bootstrap": {
            "seed": ablations.BOOTSTRAP_SEED,
            "replicates": ablations.BOOTSTRAP_REPLICATES,
            "strata": 1,
            "lower_95": lower,
            "upper_95": 0.02,
        },
        "mcnemar_exact_two_sided_p": 1.0,
    }


def _suite_lock_payload() -> dict:
    source_names = sorted(
        {key for config in ablations.SUITES.values() for key in config.source_keys}
    )
    disjoint_source_names = set(source_names) | set(
        ablations.PRIOR_SOURCE_KEYS
    )
    pairs = {
        pair: 0
        for pair in ablations._required_target_disjoint_pairs(
            disjoint_source_names
        )
    }
    source = {
        "path": "unused.jsonl",
        "sha256": "a" * 64,
        "rows": 4,
        "case_fingerprint_set_sha256": "b" * 64,
        "target_fingerprint_set_sha256": "c" * 64,
    }
    return {
        "schema_version": ablations.SUITE_LOCK_SCHEMA_VERSION,
        "lock_id": ablations.SUITE_LOCK_ID,
        "experiment_id": ablations.EXPERIMENT_ID,
        "status": "locked_before_phase15_candidate_execution",
        "catalog_sha256": "d" * 64,
        "ordered_gates": list(ablations.SUITES),
        "public_confirmation_is_last": True,
        "public_metrics_are_comparison_only": True,
        "robustness_manifest": {"path": "manifest.json", "sha256": "e" * 64},
        "generator_source_sha256": {
            ablations.ROBUSTNESS_GENERATOR_RELATIVE: "f" * 64,
            ablations.ROBUSTNESS_REFERENCE_RELATIVES["evaluator"]: "1" * 64,
            ablations.ROBUSTNESS_REFERENCE_RELATIVES["phase14_builder"]: (
                "2" * 64
            ),
        },
        "gate_sources": {
            name: list(config.source_keys)
            for name, config in ablations.SUITES.items()
        },
        "sources": {name: dict(source) for name in source_names},
        "prior_sources": {
            name: dict(source) for name in ablations.PRIOR_SOURCE_KEYS
        },
        "target_disjointness": {
            "generated_sources_are_mutually_and_forbidden_disjoint": True,
            "pairwise_overlap_counts": pairs,
        },
    }


def _privacy_payload(suite: str) -> dict:
    return {
        "schema_version": ablations.SCHEMA_VERSION,
        "experiment_id": ablations.EXPERIMENT_ID,
        "suite": suite,
        "dataset": {},
        "run_configuration": {},
        "metrics": {},
        "paired_quality": {},
        "health": {},
        "call_accounting": {},
        "performance": {},
        "startup": {},
        "exactness": {},
        "calibration": ablations._empty_calibration_summary(),
        "privacy": {
            "aggregate_metrics_and_fixed_counters_only": True,
            "row_scenario_message_profile_target_product_and_trace_data_absent": True,
            "per_case_belief_and_fingerprint_values_absent": True,
            "manual_failure_or_small_cell_inspection_performed": False,
            "public_metrics_used_for_fitting": False,
        },
        "reproducibility": {},
        "decision_gate": {},
    }


def _limits() -> dict:
    return {
        "candidate_warm_p95_ratio_at_most": 1.10,
        "candidate_wall_time_ratio_at_most": 1.10,
        "candidate_startup_ratio_at_most": 1.10,
        "candidate_additional_startup_rss_bytes_at_most": 64 * 1024 * 1024,
        "candidate_additional_post_warm_peak_rss_bytes_at_most": (
            64 * 1024 * 1024
        ),
        "candidate_additional_retained_session_bytes_at_most": 65_536,
        "no_duplicate_catalog_copy": True,
        "candidate_metadata_is_compact_card_only": True,
        "candidate_metadata_fetches_full_fts_documents": False,
        "additional_external_model_or_api_calls_per_turn": 0,
        "model_and_api_tokens": 0,
        "response_exceptions": 0,
        "invalid_api_responses": 0,
        "runtime_network_attempts": 0,
        "faults": 0,
        "replay_exact": True,
        "independent_construction_exact": True,
    }


def _exact_paired() -> dict:
    value = _paired(lower=0.0)
    value["mean_utility_delta"] = 0.0
    value["bootstrap"]["upper_95"] = 0.0
    return value


def _valid_startup() -> dict:
    return {
        "accounting": "max_candidate_vs_min_protected_across_both_orders",
        "baseline_probe_count": 2,
        "candidate_probe_count": 2,
        "baseline_startup_seconds": 1.0,
        "candidate_startup_seconds": 1.0,
        "candidate_startup_time_ratio": 1.0,
        "baseline_startup_rss_bytes": 1_000,
        "candidate_startup_rss_bytes": 2_000,
        "candidate_additional_startup_rss_bytes": 1_000,
        "baseline_post_warm_peak_rss_bytes": 1_500,
        "candidate_post_warm_peak_rss_bytes": 2_500,
        "candidate_additional_post_warm_peak_rss_bytes": 1_000,
        "baseline_empty_retained_bytes": 100,
        "candidate_empty_retained_bytes": 200,
    }


def _valid_result_payload(
    suite: str,
    suite_lock: dict,
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
) -> dict:
    config = ablations.SUITES[suite]
    baseline = _run(candidate=False, digest="protected")
    candidate = _run(candidate=True, digest="candidate")
    paired = _paired()
    if config.gate_mode == "exact_fail_open":
        candidate.summary = copy.deepcopy(baseline.summary)
        raw = _protocol_health(candidate=True)
        raw["applied"] = 0
        raw["unsupported_or_disabled"] = 4
        candidate.diagnostics["protocol_decision_health"] = (
            ablations._project_protocol_health(
                raw,
                expected_policy=PROTOCOL_UTILITY_DECISION_POLICY,
                expected_turns=4,
            )
        )
        candidate.diagnostics["route_health"]["dense"] = {"ok": 4}
        paired = _exact_paired()
    baseline_health = ablations._aggregate_health(baseline.diagnostics)
    candidate_health = ablations._aggregate_health(candidate.diagnostics)
    metrics = {
        "baseline": baseline.summary,
        "candidate": candidate.summary,
        "delta": ablations._metric_deltas(
            baseline.summary,
            candidate.summary,
        ),
    }
    payload = _privacy_payload(suite)
    payload.update(
        {
            "dataset": ablations._expected_dataset_evidence(
                config,
                suite_lock,
            ),
            "run_configuration": {
                "execution": "strictly_sequential_cpu",
                "threads": 1,
                "processes_during_evaluation": 1,
                "variant_policies": [
                    ablations.BASELINE_ID,
                    ablations.CANDIDATE_ID,
                ],
                "verification_runs_are_not_ablation_arms": True,
                "fresh_agent_state_per_variant": True,
                "startup_probe_policy_order": [
                    policy.value
                    for pair in ablations.STARTUP_PROBE_ORDERS
                    for policy in pair
                ],
                "startup_probe_conservative_accounting": (
                    "max_candidate_vs_min_protected_across_both_orders"
                ),
                "conditional_dense_routing_enabled": True,
                "additional_external_model_or_api_calls": 0,
                "gpu_or_mps": False,
                "thermal_safe_acknowledged": True,
                "public_metrics_can_tune_policy": False,
            },
            "metrics": metrics,
            "paired_quality": paired,
            "health": {
                "baseline": baseline_health,
                "protected_reference": copy.deepcopy(baseline_health),
                "candidate": candidate_health,
                "candidate_replay": copy.deepcopy(candidate_health),
                "independent_candidate": copy.deepcopy(candidate_health),
            },
            "call_accounting": {
                "baseline": ablations._call_accounting(baseline_health),
                "candidate": ablations._call_accounting(candidate_health),
            },
            "performance": ablations._performance_summary(
                baseline,
                candidate,
            ),
            "startup": _valid_startup(),
            "exactness": {
                "protected_reference": {
                    key: True
                    for key in ablations._run_exactness(
                        baseline,
                        baseline,
                    )
                },
                "candidate_replay": {
                    key: True
                    for key in ablations._run_exactness(
                        candidate,
                        candidate,
                    )
                },
                "independent_candidate": {
                    key: True
                    for key in ablations._run_exactness(
                        candidate,
                        candidate,
                    )
                },
                "candidate_vs_baseline_fail_open": {
                    "evaluator_payload_equal": (
                        config.gate_mode == "exact_fail_open"
                    ),
                    "response_state_slate_cache_equal": (
                        config.gate_mode == "exact_fail_open"
                    ),
                },
            },
            "calibration": copy.deepcopy(candidate_health["calibration"]),
            "reproducibility": {
                "platform": "test-platform",
                "python": "3.test",
                "environment": dict(ablations.REQUIRED_ENVIRONMENT),
                "implementation_lock_id": ablations.IMPLEMENTATION_LOCK_ID,
                "implementation_lock_sha256": implementation_lock_sha256,
                "suite_lock_id": ablations.SUITE_LOCK_ID,
                "suite_lock_sha256": suite_lock_sha256,
                "locks_revalidated_after_all_variants": True,
            },
        }
    )
    payload["decision_gate"] = ablations._recompute_published_decision_gates(
        config,
        payload,
        _limits(),
        suite_evidence_valid=True,
    )
    return payload


class ProtocolUtilityAblationHarnessTests(unittest.TestCase):
    def test_runtime_network_audit_denies_and_counts_connections(self) -> None:
        audit = ablations.RuntimeNetworkAudit()

        with audit.deny():
            with self.assertRaisesRegex(RuntimeError, "network access"):
                socket.create_connection(("example.invalid", 443))

        self.assertEqual(audit.attempts, 1)

    def test_suite_order_has_public_last_and_only_two_policy_arms(self) -> None:
        self.assertEqual(
            tuple(ablations.SUITES),
            (
                "fresh_exact",
                "paraphrase_fail_open",
                "card_perturbed",
                "scenario_balanced",
                "target_disjoint_development",
                "target_disjoint_validation",
                "public_confirmation",
            ),
        )
        self.assertEqual(
            ablations.SUITES["card_perturbed"].source_keys,
            ("card_perturbed",),
        )
        self.assertEqual(
            ablations.SUITES["scenario_balanced"].source_keys,
            ("scenario_balanced",),
        )
        self.assertEqual(
            ablations.SUITES["target_disjoint_development"].source_keys,
            ("target_disjoint_development",),
        )
        self.assertEqual(
            ablations.SUITES["target_disjoint_validation"].source_keys,
            ("target_disjoint_validation",),
        )
        self.assertNotIn(
            "legacy_development",
            {
                source
                for config in ablations.SUITES.values()
                for source in config.source_keys
            },
        )
        self.assertEqual(
            {ablations.BASELINE_ID, ablations.CANDIDATE_ID},
            {
                PROTECTED_DECISION_POLICY.value,
                PROTOCOL_UTILITY_DECISION_POLICY.value,
            },
        )
        self.assertEqual(
            ablations.CANDIDATE_BACKEND_RUN_ORDER,
            ("candidate", "protected_reference", "candidate_replay"),
        )
        self.assertEqual(
            ablations.STARTUP_PROBE_ORDERS,
            (
                (
                    PROTECTED_DECISION_POLICY,
                    PROTOCOL_UTILITY_DECISION_POLICY,
                ),
                (
                    PROTOCOL_UTILITY_DECISION_POLICY,
                    PROTECTED_DECISION_POLICY,
                ),
            ),
        )
        self.assertIn(
            "scripts/run_protocol_utility_ablations.py",
            ablations.SOURCE_PATHS,
        )
        self.assertIn(
            "tests/test_protocol_utility_ablations.py",
            ablations.SOURCE_PATHS,
        )

    def test_suite_lock_validation_is_exact_and_target_disjoint(self) -> None:
        payload = _suite_lock_payload()
        self.assertIs(ablations._validate_suite_lock_payload(payload), payload)

        reordered = copy.deepcopy(payload)
        reordered["ordered_gates"] = list(reversed(reordered["ordered_gates"]))
        with self.assertRaises(RuntimeError):
            ablations._validate_suite_lock_payload(reordered)

        overlapping = copy.deepcopy(payload)
        key = next(iter(overlapping["target_disjointness"]["pairwise_overlap_counts"]))
        overlapping["target_disjointness"]["pairwise_overlap_counts"][key] = 1
        with self.assertRaises(RuntimeError):
            ablations._validate_suite_lock_payload(overlapping)

        missing_prior = copy.deepcopy(payload)
        missing_prior["prior_sources"] = {}
        with self.assertRaises(RuntimeError):
            ablations._validate_suite_lock_payload(missing_prior)

        forbidden_pairs = {
            "legacy_development|legacy_validation",
            "legacy_development|public_confirmation",
            "legacy_validation|public_confirmation",
            "legacy_development|phase14_fresh",
            "legacy_validation|phase14_fresh",
            "phase14_fresh|public_confirmation",
        }
        self.assertTrue(
            forbidden_pairs.isdisjoint(
                payload["target_disjointness"]["pairwise_overlap_counts"]
            )
        )

        arbitrary_generator = copy.deepcopy(payload)
        arbitrary_generator["generator_source_sha256"] = {
            "generator.py": "f" * 64
        }
        with self.assertRaises(RuntimeError):
            ablations._validate_suite_lock_payload(arbitrary_generator)

    def test_legacy_exclusion_source_may_repeat_targets(self) -> None:
        rows = [
            {
                "sample_id": "legacy-a",
                "ground_truth": {"parent_asin": "B000000001"},
            },
            {
                "sample_id": "legacy-b",
                "ground_truth": {"parent_asin": "B000000001"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            locked = {
                "path": str(path),
                "sha256": ablations._sha256(path),
                "rows": len(rows),
                "case_fingerprint_set_sha256": (
                    ablations._case_fingerprint_set_digest(rows)
                ),
                "target_fingerprint_set_sha256": (
                    ablations._target_set_digest({"B000000001"})
                ),
            }

            loaded_rows, targets = ablations._locked_rows_and_targets(
                locked,
                Path(directory),
                require_unique_targets=False,
            )
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(targets, {"B000000001"})
            with self.assertRaisesRegex(RuntimeError, "must be unique"):
                ablations._locked_rows_and_targets(
                    locked,
                    Path(directory),
                    require_unique_targets=True,
                )

    def test_protocol_telemetry_partitions_every_turn_and_width(self) -> None:
        projected = ablations._project_protocol_health(
            _protocol_health(candidate=True),
            expected_policy=PROTOCOL_UTILITY_DECISION_POLICY,
            expected_turns=4,
        )
        self.assertEqual(projected["planner_decisions"], 4)
        self.assertEqual(sum(projected["question_action_counts"].values()), 4)
        self.assertEqual(sum(projected["width_action_counts"].values()), 4)
        self.assertEqual(projected["presented_total"], 4)

        invalid = _protocol_health(candidate=True)
        invalid["width_action_counts"]["1"] = 3
        with self.assertRaises(RuntimeError):
            ablations._project_protocol_health(
                invalid,
                expected_policy=PROTOCOL_UTILITY_DECISION_POLICY,
                expected_turns=4,
            )

        with self.assertRaises(RuntimeError):
            ablations._project_protocol_health(
                _protocol_health(candidate=True),
                expected_policy=PROTECTED_DECISION_POLICY,
                expected_turns=4,
            )

    def test_brier_and_ece_are_aggregate_fixed_and_unfitted(self) -> None:
        audit = ablations.BeliefCalibrationAudit(("T", "Y"))
        audit.bind_session("one")
        with audit.activate("one"):
            audit.observe(
                SimpleNamespace(
                    beliefs=(
                        SimpleNamespace(parent_asin="T", weight=0.75),
                        SimpleNamespace(parent_asin="X", weight=0.25),
                    )
                )
            )
        audit.bind_session("two")
        with audit.activate("two"):
            audit.observe(
                SimpleNamespace(
                    beliefs=(
                        SimpleNamespace(parent_asin="T", weight=0.75),
                        SimpleNamespace(parent_asin="Y", weight=0.25),
                    )
                )
            )
        summary = audit.summary()

        self.assertEqual(summary["observations"], 2)
        self.assertEqual(summary["target_in_support"], 2)
        self.assertEqual(summary["mean_multiclass_brier"], 0.625)
        self.assertEqual(summary["ece_10"], 0.25)
        self.assertFalse(summary["learned_or_fitted_calibration"])
        self.assertFalse(summary["used_as_promotion_threshold"])
        self.assertTrue(
            ablations._calibration_is_valid(summary, expected_turns=2)
        )

    def test_calibration_failure_is_deferred_without_changing_ranking_payload(self) -> None:
        audit = ablations.BeliefCalibrationAudit(("T",))
        audit.bind_session("session")
        ranking_payload = SimpleNamespace(
            marker="unchanged",
            beliefs=(SimpleNamespace(parent_asin="T", weight=float("nan")),),
        )
        exact_evidence_module = importlib.import_module(
            "conversational_search.exact_evidence"
        )
        with mock.patch.object(
            exact_evidence_module,
            "rank_exact_evidence",
            return_value=ranking_payload,
        ):
            with audit.activate("session"), ablations._capture_belief_calibration(
                audit
            ):
                observed = exact_evidence_module.rank_exact_evidence()

        self.assertIs(observed, ranking_payload)
        self.assertEqual(observed.marker, "unchanged")
        with self.assertRaisesRegex(RuntimeError, "outside evaluator"):
            audit.raise_deferred_failure()

    def test_startup_probe_is_order_balanced_and_conservatively_accounted(self) -> None:
        observations = iter(
            (
                {
                    "policy": PROTECTED_DECISION_POLICY.value,
                    "elapsed_seconds": 4.0,
                    "max_rss_bytes": 100,
                    "post_warm_max_rss_bytes": 110,
                    "empty_retained_bytes": 10,
                },
                {
                    "policy": PROTOCOL_UTILITY_DECISION_POLICY.value,
                    "elapsed_seconds": 2.0,
                    "max_rss_bytes": 130,
                    "post_warm_max_rss_bytes": 170,
                    "empty_retained_bytes": 15,
                },
                {
                    "policy": PROTOCOL_UTILITY_DECISION_POLICY.value,
                    "elapsed_seconds": 5.0,
                    "max_rss_bytes": 150,
                    "post_warm_max_rss_bytes": 160,
                    "empty_retained_bytes": 17,
                },
                {
                    "policy": PROTECTED_DECISION_POLICY.value,
                    "elapsed_seconds": 3.0,
                    "max_rss_bytes": 90,
                    "post_warm_max_rss_bytes": 95,
                    "empty_retained_bytes": 9,
                },
            )
        )

        def completed(*args: object, **kwargs: object) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(stdout=json.dumps(next(observations)))

        with mock.patch.object(
            ablations.subprocess,
            "run",
            side_effect=completed,
        ) as run:
            summary = ablations._startup_probe(Path("unused.jsonl"))

        policies = [
            call.args[0][4]
            for call in run.call_args_list
        ]
        self.assertEqual(
            policies,
            [
                PROTECTED_DECISION_POLICY.value,
                PROTOCOL_UTILITY_DECISION_POLICY.value,
                PROTOCOL_UTILITY_DECISION_POLICY.value,
                PROTECTED_DECISION_POLICY.value,
            ],
        )
        self.assertEqual(summary["baseline_startup_seconds"], 3.0)
        self.assertEqual(summary["candidate_startup_seconds"], 5.0)
        self.assertEqual(summary["candidate_startup_time_ratio"], 1.666667)
        self.assertEqual(summary["baseline_startup_rss_bytes"], 90)
        self.assertEqual(summary["candidate_startup_rss_bytes"], 150)
        self.assertEqual(summary["candidate_additional_startup_rss_bytes"], 60)
        self.assertEqual(summary["baseline_post_warm_peak_rss_bytes"], 95)
        self.assertEqual(summary["candidate_post_warm_peak_rss_bytes"], 170)
        self.assertEqual(
            summary["candidate_additional_post_warm_peak_rss_bytes"],
            75,
        )
        self.assertEqual(summary["baseline_empty_retained_bytes"], 9)
        self.assertEqual(summary["candidate_empty_retained_bytes"], 17)

    def test_locked_paraphrase_dialog_wraps_only_message_surfaces(self) -> None:
        sample = {
            "scenario_type": "buying",
            "intent_card": {
                "hard_constraints": ["cotton", "color: blue"],
                "soft_preferences": [],
            },
            "phase15_dialog": {
                "mode": "paraphrase_fail_open_v1",
                "initial_message": "Locked initial surface.",
                "reply_shapes": {
                    "disclosure": "Details for {attribute}: {values}",
                    "boundary_decline": "No choice for {attribute}",
                    "no_additional": "Nothing else for {attribute}",
                    "need_attribute": "Ask one attribute",
                    "override": "Use {value} instead",
                },
            },
        }
        disclosed: set[str] = set()
        with ablations._phase15_dialog_surface():
            initial = evaluator_module.initial_message(
                sample,
                "shirts",
                disclosed,
            )
            reply, boundary = evaluator_module.customer_reply(
                sample,
                "color",
                disclosed,
                False,
            )

        self.assertEqual(initial, "Locked initial surface.")
        self.assertEqual(disclosed, {"cotton", "color: blue"})
        self.assertEqual(reply, "Details for color: color: blue")
        self.assertFalse(boundary)

    def test_gate_builder_enforces_exactness_calls_and_contract_limits(self) -> None:
        baseline = _run(candidate=False, digest="protected")
        protected_reference = _run(candidate=False, digest="protected")
        candidate = _run(candidate=True, digest="candidate")
        replay = _run(candidate=True, digest="candidate")
        independent = _run(candidate=True, digest="candidate")
        performance = {
            "candidate_warm_p95_ratio": 1.0,
            "candidate_wall_time_ratio": 1.0,
            "candidate_additional_retained_session_bytes": 250,
        }
        startup = {
            "candidate_startup_time_ratio": 1.0,
            "candidate_additional_startup_rss_bytes": 1_024,
            "candidate_additional_post_warm_peak_rss_bytes": 2_048,
        }
        limits = _limits()

        gates = ablations._build_gates(
            ablations.SUITES["fresh_exact"],
            baseline,
            protected_reference,
            candidate,
            replay,
            independent,
            _paired(),
            performance,
            startup,
            limits,
            locks_revalidated=True,
            suite_evidence_valid=True,
            privacy_valid=True,
        )
        self.assertTrue(gates["advance"])
        self.assertTrue(all(gates.values()))
        self.assertEqual(
            set(gates),
            ablations._expected_decision_gate_keys(
                ablations.SUITES["fresh_exact"]
            ),
        )

        candidate.diagnostics["route_health"]["dense"] = {"ok": 5}
        failed = ablations._build_gates(
            ablations.SUITES["fresh_exact"],
            baseline,
            protected_reference,
            candidate,
            replay,
            independent,
            _paired(),
            performance,
            startup,
            limits,
            locks_revalidated=True,
            suite_evidence_valid=True,
            privacy_valid=True,
        )
        self.assertFalse(
            failed[
                "candidate_dense_executions_and_planned_skips_partition_searches"
            ]
        )
        self.assertFalse(failed["advance"])

    def test_publication_rejects_private_rows_messages_ids_and_beliefs(self) -> None:
        payload = _privacy_payload("fresh_exact")
        self.assertTrue(ablations.publication_privacy_is_valid(payload))
        for mutation in (
            {"sessions": []},
            {"phase15_dialog": {}},
            {"parent_asin": "B000000001"},
            {"beliefs": [0.5, 0.5]},
            {"small_cell": "intent_override"},
        ):
            with self.subTest(mutation=mutation):
                unsafe = copy.deepcopy(payload)
                unsafe["health"].update(mutation)
                self.assertFalse(ablations.publication_privacy_is_valid(unsafe))

    def test_prerequisite_payload_binds_suite_identity_and_current_locks(self) -> None:
        suite_lock = _suite_lock_payload()
        payload = _valid_result_payload(
            "fresh_exact",
            suite_lock,
            implementation_lock_sha256="a" * 64,
            suite_lock_sha256="b" * 64,
        )
        ablations._validate_prerequisite_payload(
            "fresh_exact",
            payload,
            implementation_lock_sha256="a" * 64,
            suite_lock_sha256="b" * 64,
            suite_lock=suite_lock,
            limits=_limits(),
        )

        with self.assertRaisesRegex(RuntimeError, "wrong suite"):
            ablations._validate_prerequisite_payload(
                "paraphrase_fail_open",
                payload,
                implementation_lock_sha256="a" * 64,
                suite_lock_sha256="b" * 64,
                suite_lock=suite_lock,
                limits=_limits(),
            )
        with self.assertRaisesRegex(RuntimeError, "incomplete, stale"):
            ablations._validate_prerequisite_payload(
                "fresh_exact",
                payload,
                implementation_lock_sha256="c" * 64,
                suite_lock_sha256="b" * 64,
                suite_lock=suite_lock,
                limits=_limits(),
            )

        minimal = _privacy_payload("fresh_exact")
        minimal["decision_gate"] = {"advance": True}
        with self.assertRaisesRegex(RuntimeError, "incomplete, stale"):
            ablations._validate_prerequisite_payload(
                "fresh_exact",
                minimal,
                implementation_lock_sha256="a" * 64,
                suite_lock_sha256="b" * 64,
                suite_lock=suite_lock,
                limits=_limits(),
            )

    def test_promotion_requires_nonpublic_gates_but_never_public_metrics(self) -> None:
        suite_lock = _suite_lock_payload()
        implementation_lock_sha256 = "a" * 64
        suite_lock_sha256 = "b" * 64
        payloads = [
            _valid_result_payload(
                name,
                suite_lock,
                implementation_lock_sha256=implementation_lock_sha256,
                suite_lock_sha256=suite_lock_sha256,
            )
            for name in ablations.SUITES
        ]
        allowed = {
            "implementation_lock_sha256": implementation_lock_sha256,
            "suite_lock_sha256": suite_lock_sha256,
            "suite_lock": suite_lock,
            "limits": _limits(),
            "public_baseline_metrics": payloads[-1]["metrics"]["baseline"],
        }
        self.assertTrue(
            ablations._promotion_payloads_are_allowed(payloads, **allowed)
        )

        better_public = copy.deepcopy(payloads)
        candidate = better_public[-1]["metrics"]["candidate"]
        candidate["mrr"] = 0.99
        candidate["recommended_technical_score"] = round(
            0.50 * candidate["hit_rate_at_10"]
            + 0.30 * candidate["mrr"]
            + 0.20 * candidate["efficiency"],
            6,
        )
        better_public[-1]["metrics"]["delta"] = ablations._metric_deltas(
            better_public[-1]["metrics"]["baseline"],
            candidate,
        )
        self.assertTrue(
            ablations._promotion_payloads_are_allowed(
                better_public,
                **allowed,
            )
        )

        fabricated = copy.deepcopy(payloads)
        fabricated[0]["health"] = {}
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(fabricated, **allowed)
        )

        inconsistent_metrics = copy.deepcopy(payloads)
        inconsistent_metrics[0]["metrics"]["candidate"]["mrr"] = 0.52
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(
                inconsistent_metrics,
                **allowed,
            )
        )
        failed_nonpublic = copy.deepcopy(payloads)
        failed_nonpublic[3]["decision_gate"]["advance"] = False
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(
                failed_nonpublic,
                **allowed,
            )
        )

        stale_dataset = copy.deepcopy(payloads)
        stale_dataset[4]["dataset"]["evaluated_cases"] -= 1
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(
                stale_dataset,
                **allowed,
            )
        )

        stale_lock = copy.deepcopy(payloads)
        stale_lock[0]["reproducibility"]["suite_lock_sha256"] = "c" * 64
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(
                stale_lock,
                **allowed,
            )
        )

        incomplete_gate = copy.deepcopy(payloads)
        incomplete_gate[0]["decision_gate"].pop(
            "candidate_mrr_not_below_baseline"
        )
        self.assertFalse(
            ablations._promotion_payloads_are_allowed(
                incomplete_gate,
                **allowed,
            )
        )

    def test_attempt_is_exclusive_and_no_run_happens_without_thermal_ack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            ablations._claim_attempt(path, "fresh_exact")
            first = path.read_bytes()
            with self.assertRaises(FileExistsError):
                ablations._claim_attempt(path, "fresh_exact")
            self.assertEqual(path.read_bytes(), first)

        with mock.patch.object(ablations, "_validate_execution_environment") as validate:
            with self.assertRaisesRegex(RuntimeError, "thermal safety"):
                ablations.run_protocol_utility_ablation(
                    "fresh_exact",
                    "unused.json",
                    thermal_safe_ack=False,
                )
            validate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
