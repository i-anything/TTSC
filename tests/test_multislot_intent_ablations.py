from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import conversational_search.service as service_module
from conversational_search.intent import LOSSLESS_MULTI_SLOT_INTENT_POLICY
from conversational_search.service import ConversationalSearchAgent
from scripts import run_multislot_intent_ablations as ablations
from tests.test_service import CacheableRecordingRetriever


def _row(index: int) -> dict:
    return {
        "sample_id": f"synthetic-{index}",
        "scenario_type": "synthetic-stratum",
        "user_profile": {"synthetic": index},
        "ground_truth": {"parent_asin": f"SYNTHETIC-{index}"},
    }


def _session(
    index: int,
    *,
    hit: bool,
    rank: int | None,
    turn: int | None,
    stratum: str,
) -> dict:
    return {
        "sample_id": f"opaque-{index}",
        "scenario_type": stratum,
        "hit": hit,
        "best_rank": rank,
        "first_hit_turn": turn,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
    }


def _summary(
    *,
    hit_rate: float,
    mrr: float,
    mttc: float,
    score: float,
) -> dict:
    return {
        "sample_count": 4,
        "hit_rate_at_10": hit_rate,
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


def _privacy_payload() -> dict:
    baseline = _summary(hit_rate=0.5, mrr=0.3, mttc=6.0, score=0.44)
    candidate = _summary(hit_rate=0.75, mrr=0.4, mttc=5.0, score=0.615)
    route = {
        key: {} if key in {"bm25", "dense"} else 0
        for key in ablations.ROUTE_HEALTH_KEYS
    }

    def counters(keys: tuple[str, ...]) -> dict:
        return {
            key: "policy" if key == "policy" else {} if key == "reasons" else 0
            for key in keys
        }

    diagnostic = {
        "expected_turns": 0,
        "route_health": route,
        "ranking_health": counters(ablations.RANKING_HEALTH_KEYS),
        "rescue_health": counters(ablations.RESCUE_HEALTH_KEYS),
        "profile_health": counters(ablations.PROFILE_HEALTH_KEYS),
        "slate_health": counters(ablations.SLATE_HEALTH_KEYS),
        "orchestration_health": counters(ablations.ORCHESTRATION_HEALTH_KEYS),
        "parser_health": counters(ablations.PARSER_HEALTH_KEYS),
        "retained_profile_state_valid": True,
        "retained_agent_bytes": 0,
        "evaluation_wall_seconds": 0.0,
        "respond_latency_ms": {key: 0 for key in ablations.LATENCY_KEYS},
    }
    diagnostic["ranking_health"]["policy"] = (
        ablations.STAGE_A_RANKING_POLICY.value
    )
    diagnostic["rescue_health"]["policy"] = (
        ablations.STAGE_A_RANKING_POLICY.value
    )
    diagnostic["profile_health"]["policy"] = (
        ablations.BOUNDED_RESIDUAL_PROFILE_POLICY.value
    )
    diagnostic["slate_health"]["policy"] = (
        ablations.STAGNATION_AWARE_SLATE_POLICY.value
    )
    diagnostic["orchestration_health"]["policy"] = (
        ablations.EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value
    )
    calls = {
        "searches": 0,
        "bm25_route_calls": 0,
        "dense_route_calls": 0,
        "candidate_document_calls": 0,
        "stage_a_attempts": 0,
    }
    performance = {
        "baseline_wall_seconds": 1.0,
        "candidate_wall_seconds": 1.0,
        "candidate_wall_time_ratio": 1.0,
        "baseline_warm_p95_ms": 1.0,
        "candidate_warm_p95_ms": 1.0,
        "candidate_warm_p95_ratio": 1.0,
        "baseline_retained_agent_bytes": 0,
        "candidate_retained_agent_bytes": 0,
        "candidate_additional_retained_agent_bytes": 0,
    }
    startup = {
        "iterations": 400,
        "baseline_total_ms": 1.0,
        "candidate_total_ms": 1.0,
        "candidate_startup_time_ratio": 1.0,
        "candidate_additional_startup_rss_bytes": 0,
        "baseline_empty_retained_bytes": 0,
        "candidate_empty_retained_bytes": 0,
    }
    exactness = {
        "candidate_replay_evaluator_payload_equal": True,
        "candidate_replay_action_state_slate_cache_equal": True,
        "candidate_replay_aggregate_health_equal": True,
        "independent_evaluator_payload_equal": True,
        "independent_action_state_slate_cache_equal": True,
        "independent_aggregate_health_equal": True,
    }
    return {
        "schema_version": 1,
        "experiment_id": ablations.EXPERIMENT_ID,
        "suite": "development",
        "dataset": {
            "source_sha256": "a" * 64,
            "source_rows": 4,
            "evaluated_cases": 4,
            "public_rows_excluded": 0,
            "duplicate_rows_excluded": 0,
            "fingerprint_set_sha256": "b" * 64,
        },
        "run_configuration": {
            "execution": "strictly_sequential_cpu",
            "threads": 1,
            "processes": 1,
            "shared_immutable_backend": True,
            "fresh_agent_state_per_variant": True,
            "variant_order": [
                ablations.CANDIDATE_ID,
                ablations.BASELINE_ID,
                ablations.CANDIDATE_ID,
                "independent_explicit_policy_starter",
            ],
            "backend_warmup": "one_fixed_label_free_request",
            "thermal_safe_acknowledged": True,
            "external_api_calls": 0,
            "gpu_or_mps": False,
        },
        "metrics": {
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                key: 0.0
                for key in ablations.OVERALL_METRIC_KEYS
                if key != "sample_count"
            },
        },
        "paired_quality": {
            "transitions": {
                "both_hit": 2,
                "candidate_only_hit": 1,
                "baseline_only_hit": 0,
                "both_miss": 1,
            },
            "mean_utility_delta": 0.1,
            "bootstrap": {
                "seed": 20260830,
                "replicates": 10_000,
                "strata": 2,
                "lower_95": 0.0,
                "upper_95": 0.2,
            },
            "mcnemar_exact_two_sided_p": 1.0,
        },
        "health": {
            "baseline": diagnostic,
            "candidate": diagnostic,
            "candidate_replay": diagnostic,
            "independent_candidate": diagnostic,
        },
        "call_accounting": {
            "baseline": calls,
            "candidate": calls,
        },
        "performance": performance,
        "startup": startup,
        "exactness": exactness,
        "privacy": {
            "aggregate_metrics_and_fixed_counters_only": True,
            "row_scenario_message_profile_target_product_and_trace_data_absent": True,
            "per_case_fingerprints_absent": True,
            "manual_failure_inspection_performed": False,
        },
        "reproducibility": {
            "platform": "synthetic-platform",
            "python": "3.12",
            "environment": dict(ablations.REQUIRED_ENVIRONMENT),
            "implementation_lock_id": ablations.IMPLEMENTATION_LOCK_ID,
            "contract_sha256": "c" * 64,
            "source_sha256": {
                path: "d" * 64 for path in ablations.SOURCE_PATHS
            },
            "phase11_oracle_sha256": ablations.PHASE11_ORACLE_SHA256,
            "implementation_lock_revalidated_after_independent": True,
        },
        "decision_gate": {
            key: False for key in ablations._COMMON_GATE_KEYS
        },
    }


class Phase11DatasetContractTest(unittest.TestCase):
    def test_content_fingerprint_ignores_identifier_and_metadata(self) -> None:
        left = _row(1)
        right = {
            **left,
            "sample_id": "different",
            "scenario_type": "different",
            "extra": "ignored",
        }

        self.assertEqual(
            ablations._content_fingerprint(left),
            ablations._content_fingerprint(right),
        )
        changed = {**right, "user_profile": {"synthetic": 2}}
        self.assertNotEqual(
            ablations._content_fingerprint(left),
            ablations._content_fingerprint(changed),
        )

    def test_suite_loader_excludes_public_before_stable_deduplication(self) -> None:
        public = [_row(0), _row(1)]
        first = _row(2)
        duplicate = {**first, "sample_id": "duplicate-id"}
        second = _row(3)
        source = [public[0], first, duplicate, second]
        selected_set = {
            ablations._content_fingerprint(first),
            ablations._content_fingerprint(second),
        }
        config = ablations.SuiteConfig(
            "synthetic",
            Path("unused.jsonl"),
            "a" * 64,
            4,
            2,
            ablations._fingerprint_set_digest(selected_set),
            True,
        )
        public_set_digest = ablations._fingerprint_set_digest(
            {ablations._content_fingerprint(row) for row in public}
        )

        with (
            mock.patch.object(
                ablations,
                "load_jsonl",
                side_effect=[public, source],
            ),
            mock.patch.object(
                ablations,
                "PUBLIC_SET_SHA256",
                public_set_digest,
            ),
            mock.patch.object(ablations, "PUBLIC_CASES", 2),
        ):
            selected, evidence = ablations._load_suite_samples(config)

        self.assertEqual(selected, [first, second])
        self.assertEqual(
            evidence,
            {
                "source_rows": 4,
                "evaluated_cases": 2,
                "public_rows_excluded": 1,
                "duplicate_rows_excluded": 1,
                "fingerprint_set_sha256": config.fingerprint_set_sha256,
            },
        )

    def test_locked_development_split_matches_aggregate_audit(self) -> None:
        selected, evidence = ablations._load_suite_samples(
            ablations.SUITES["development"]
        )

        self.assertEqual(len(selected), 996)
        self.assertEqual(evidence["public_rows_excluded"], 202)
        self.assertEqual(evidence["duplicate_rows_excluded"], 2)
        self.assertEqual(
            evidence["fingerprint_set_sha256"],
            ablations.DEVELOPMENT_SET_SHA256,
        )


class Phase11PairedStatisticsTest(unittest.TestCase):
    def test_statistics_are_fixed_seed_stratified_and_aggregate(self) -> None:
        baseline = [
            _session(0, hit=True, rank=2, turn=3, stratum="secret-alpha-label"),
            _session(1, hit=False, rank=None, turn=None, stratum="secret-alpha-label"),
            _session(2, hit=True, rank=4, turn=5, stratum="secret-beta-label"),
            _session(3, hit=False, rank=None, turn=None, stratum="secret-beta-label"),
        ]
        candidate = [
            _session(0, hit=True, rank=1, turn=2, stratum="secret-alpha-label"),
            _session(1, hit=True, rank=5, turn=6, stratum="secret-alpha-label"),
            _session(2, hit=True, rank=3, turn=4, stratum="secret-beta-label"),
            _session(3, hit=False, rank=None, turn=None, stratum="secret-beta-label"),
        ]

        first = ablations._paired_statistics(baseline, candidate)
        second = ablations._paired_statistics(baseline, candidate)

        self.assertEqual(first, second)
        self.assertEqual(
            first["transitions"],
            {
                "both_hit": 2,
                "candidate_only_hit": 1,
                "baseline_only_hit": 0,
                "both_miss": 1,
            },
        )
        self.assertEqual(first["bootstrap"]["seed"], 20260830)
        self.assertEqual(first["bootstrap"]["replicates"], 10_000)
        self.assertEqual(first["bootstrap"]["strata"], 2)
        self.assertGreaterEqual(first["bootstrap"]["lower_95"], 0.0)
        serialized = json.dumps(first)
        self.assertNotIn("secret-alpha-label", serialized)
        self.assertNotIn("secret-beta-label", serialized)

    def test_quality_gate_is_conjunctive_and_rejects_one_regression(self) -> None:
        baseline = ablations.VariantRun(
            _summary(hit_rate=0.5, mrr=0.3, mttc=6.0, score=0.44),
            [],
            {},
            "a" * 64,
            "b" * 64,
        )
        candidate = ablations.VariantRun(
            _summary(hit_rate=0.75, mrr=0.4, mttc=5.0, score=0.615),
            [],
            {},
            "c" * 64,
            "d" * 64,
        )
        paired = {
            "transitions": {"baseline_only_hit": 0},
            "bootstrap": {"lower_95": 0.0},
        }

        gates = ablations._quality_gates(baseline, candidate, paired)
        self.assertTrue(all(gates.values()))
        paired["transitions"]["baseline_only_hit"] = 1
        gates = ablations._quality_gates(baseline, candidate, paired)
        self.assertFalse(gates["baseline_hit_to_candidate_miss_is_zero"])

    def test_exact_mcnemar_is_bounded_and_symmetric(self) -> None:
        self.assertEqual(ablations._exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(
            ablations._exact_mcnemar_p(4, 1),
            ablations._exact_mcnemar_p(1, 4),
        )
        self.assertLessEqual(ablations._exact_mcnemar_p(10, 0), 1.0)


class Phase11HarnessSafetyTest(unittest.TestCase):
    def test_candidate_parser_audit_partitions_and_validates_live_state(self) -> None:
        retriever = CacheableRecordingRetriever(
            ("B000000001", "B000000002"),
            fused_ids=("B000000001", "B000000002"),
            documents={
                "B000000001": "red leather synthetic item",
                "B000000002": "blue wool synthetic item",
            },
        )
        agent = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=retriever,
            intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
        )
        audit = ablations.ParserAudit()
        wrapped = ablations.Phase11AuditAgent(agent, audit)
        wrapped.reset("session", {})

        original = service_module.apply_user_message_with_trace
        with ablations._capture_candidate_reductions(audit):
            response = wrapped.respond("session", "red and leather", 1, 2)

        self.assertIs(service_module.apply_user_message_with_trace, original)
        self.assertEqual(len(response["recommendations"]), 2)
        health = audit.summary()
        self.assertEqual(health["attempts"], 1)
        self.assertEqual(health["applied"], 1)
        self.assertEqual(health["positive_atoms"], 2)
        self.assertEqual(sum(health[key] for key in ablations._PARSER_OUTCOME_KEYS), 1)

    def test_candidate_and_baseline_variant_use_exact_call_accounting(self) -> None:
        identifier = "B000000001"
        samples = [
            {
                "sample_id": "opaque",
                "scenario_type": "buying",
                "user_profile": {},
                "ground_truth": {"parent_asin": identifier},
                "intent_card": {
                    "target_category": "synthetic item",
                    "hard_constraints": ["red"],
                    "soft_preferences": ["leather"],
                },
                "behavior": {"scenario_type": "buying"},
            }
        ]
        products = {identifier: {"parent_asin": identifier, "title": "synthetic item"}}
        backend = CacheableRecordingRetriever(
            (identifier,),
            fused_ids=(identifier,),
            documents={identifier: "red leather synthetic item"},
        )

        candidate = ablations._run_variant(
            Path("unused.jsonl"),
            samples,
            {identifier},
            {identifier: ["Synthetic"]},
            products,
            backend,
            candidate=True,
        )
        baseline = ablations._run_variant(
            Path("unused.jsonl"),
            samples,
            {identifier},
            {identifier: ["Synthetic"]},
            products,
            backend,
            candidate=False,
        )

        self.assertEqual(candidate.summary, baseline.summary)
        self.assertEqual(candidate.diagnostics["parser_health"]["attempts"], 1)
        self.assertEqual(baseline.diagnostics["parser_health"]["attempts"], 0)
        for run in (candidate, baseline):
            calls = ablations._call_accounting(run.diagnostics)
            self.assertEqual(
                len(set(calls.values())),
                1,
            )

    def test_attempt_claim_is_exclusive_and_aggregate_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.json"
            ablations._claim_attempt(path, "synthetic")
            with self.assertRaises(FileExistsError):
                ablations._claim_attempt(path, "synthetic")
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            payload,
            {
                "schema_version": 1,
                "experiment_id": ablations.EXPERIMENT_ID,
                "suite": "synthetic",
                "status": "claimed",
            },
        )

    def test_execution_environment_requires_every_single_thread_pin(self) -> None:
        pinned = {
            **ablations.REQUIRED_ENVIRONMENT,
            "TOKENIZERS_PARALLELISM": "false",
        }
        with mock.patch.dict(os.environ, pinned, clear=True):
            ablations._validate_execution_environment()
        pinned["OMP_NUM_THREADS"] = "2"
        with mock.patch.dict(os.environ, pinned, clear=True):
            with self.assertRaisesRegex(RuntimeError, "single-thread"):
                ablations._validate_execution_environment()

    def test_phase11_lock_preserves_its_historical_phase9_rollback(self) -> None:
        lock = json.loads(
            (
                ablations.REPOSITORY_ROOT
                / ablations.BASELINE_LOCK_RELATIVE
            ).read_text(encoding="utf-8")
        )
        expected = lock["source_sha256"]["starter/agent.py"]
        results = json.loads(
            (
                ablations.REPOSITORY_ROOT / "docs/phase11_results.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            expected,
            results["rollback"][
                "starter_restored_to_protected_phase9_sha256"
            ],
        )


class Phase11PublicationPrivacyTest(unittest.TestCase):
    def test_aggregate_payload_passes_and_sensitive_shapes_fail(self) -> None:
        payload = _privacy_payload()
        self.assertTrue(ablations.publication_privacy_is_valid(payload))
        before_gate = json.loads(json.dumps(payload))
        before_gate.pop("decision_gate")
        self.assertTrue(
            ablations.publication_privacy_is_valid(
                before_gate,
                allow_missing_decision_gate=True,
            )
        )
        self.assertFalse(ablations.publication_privacy_is_valid(before_gate))

        mutations = (
            lambda value: value["health"].update({"sessions": []}),
            lambda value: value["health"].update({"scenario_type": "x"}),
            lambda value: value["health"].update({"message": "secret"}),
            lambda value: value["health"].update({"opaque": "B000000001"}),
            lambda value: value["health"].update({"opaque": "buying"}),
            lambda value: value["dataset"].update({"fingerprints": []}),
            lambda value: value["dataset"].update({"opaque": "free form"}),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = json.loads(json.dumps(payload))
                mutate(changed)
                self.assertFalse(ablations.publication_privacy_is_valid(changed))

    def test_private_canonicalization_is_deterministic_without_object_addresses(self) -> None:
        value = {
            "set": frozenset({"two", "one"}),
            "policy": LOSSLESS_MULTI_SLOT_INTENT_POLICY,
            "bytes": b"opaque",
        }
        first = ablations._private_digest(value)
        second = ablations._private_digest(value)

        self.assertEqual(first, second)
        self.assertRegex(first, r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
