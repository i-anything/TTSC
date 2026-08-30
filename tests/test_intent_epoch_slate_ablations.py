from __future__ import annotations

import copy
import unittest

from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
)
from scripts.run_intent_epoch_slate_ablations import (
    ALLOWED_CANDIDATE_SOURCE_CHANGES,
    CANDIDATE_ID,
    EXPERIMENT_ID,
    PHASE9_DEVELOPMENT_METRICS,
    SOURCE_PATHS,
    SUITES,
    VariantRun,
    _build_gates,
    _faults_are_zero,
    _quality_gates,
    _startup_probe,
    _validate_phase13_accounting,
    publication_privacy_is_valid,
)


def _summary(*, candidate: bool) -> dict:
    baseline = copy.deepcopy(PHASE9_DEVELOPMENT_METRICS)
    if not candidate:
        return baseline
    return {
        **baseline,
        "mrr": 0.53588,
        "mttc": 3.048233,
        "efficiency": 0.795177,
        "recommended_technical_score": 0.815783,
    }


def _novelty_health(*, candidate: bool, attempts: int) -> dict:
    return {
        "policy": (
            INTENT_EPOCH_NOVELTY_SLATE_POLICY.value
            if candidate
            else STAGNATION_AWARE_SLATE_POLICY.value
        ),
        "attempts": attempts if candidate else 0,
        "empty_exact_baseline": 0,
        "first_slate_exact_baseline": 1 if candidate and attempts else 0,
        "unchanged_signature_exact_baseline": 0,
        "changed_epoch_exact_baseline": 0,
        "same_epoch_history_carried": (
            attempts - 1 if candidate and attempts else 0
        ),
        "validation_fallbacks": 0,
        "eligible_prior_shown_total": attempts * 3 if candidate else 0,
    }


def _diagnostics(*, candidate: bool, searches: int) -> dict:
    slate_policy = (
        INTENT_EPOCH_NOVELTY_SLATE_POLICY.value
        if candidate
        else STAGNATION_AWARE_SLATE_POLICY.value
    )
    return {
        "expected_turns": searches,
        "route_health": {
            "bm25": {"ok": searches},
            "dense": {"ok": searches},
            "fallback_turns": 0,
            "candidate_document_calls": searches,
        },
        "ranking_health": {
            "policy": "stage_a",
            "attempts": searches,
            "successes": searches,
            "failures": 0,
            "unavailable_skips": 0,
        },
        "rescue_health": {"policy": "stage_a", "attempts": 0},
        "route_redundancy_health": {
            "policy": "stage_a",
            "attempts": 0,
            "validation_or_scoring_fallbacks": 0,
        },
        "intent_epoch_slate_health": _novelty_health(
            candidate=candidate,
            attempts=searches,
        ),
        "profile_health": {"parsing_or_scoring_fallbacks": 0},
        "slate_health": {
            "policy": slate_policy,
            "attempts": searches,
            "successes": searches,
            "failures": 0,
        },
        "orchestration_health": {
            "searches": searches,
            "fault_invalidations": 0,
            "store_rejections": 0,
        },
        "retained_profile_state_valid": True,
        "retained_agent_bytes": 1_024,
        "evaluation_wall_seconds": 1.0,
        "respond_latency_ms": {"warm_p95": 1.0},
    }


def _run(*, candidate: bool, searches: int, digest: str) -> VariantRun:
    return VariantRun(
        summary=_summary(candidate=candidate),
        sessions=[],
        diagnostics=_diagnostics(candidate=candidate, searches=searches),
        evaluator_digest=digest,
        private_digest=digest,
    )


def _paired(lower: float = 0.0) -> dict:
    return {
        "transitions": {
            "both_hit": 990,
            "candidate_only_hit": 1,
            "baseline_only_hit": 0,
            "both_miss": 5,
        },
        "mean_utility_delta": 0.0005,
        "bootstrap": {
            "seed": 130260830,
            "replicates": 10_000,
            "strata": 4,
            "lower_95": lower,
            "upper_95": 0.001,
        },
        "mcnemar_exact_two_sided_p": 1.0,
    }


class IntentEpochSlateHarnessTests(unittest.TestCase):
    def test_source_scope_matches_frozen_candidate_boundary(self) -> None:
        self.assertNotIn("starter/agent.py", ALLOWED_CANDIDATE_SOURCE_CHANGES)
        self.assertIn(
            "conversational_search/slates.py",
            ALLOWED_CANDIDATE_SOURCE_CHANGES,
        )
        self.assertIn(
            "conversational_search/service.py",
            ALLOWED_CANDIDATE_SOURCE_CHANGES,
        )
        self.assertIn(
            "scripts/run_intent_epoch_slate_ablations.py",
            SOURCE_PATHS,
        )
        self.assertIn("docs/phase13_experiment_contract.json", SOURCE_PATHS)

    def test_candidate_and_baseline_accounting_are_disjoint_and_complete(self) -> None:
        candidate = _diagnostics(candidate=True, searches=7)
        baseline = _diagnostics(candidate=False, searches=8)

        _validate_phase13_accounting(
            candidate,
            INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )
        _validate_phase13_accounting(
            baseline,
            STAGNATION_AWARE_SLATE_POLICY,
        )
        self.assertTrue(_faults_are_zero(candidate))
        self.assertTrue(_faults_are_zero(baseline))

        candidate["intent_epoch_slate_health"]["validation_fallbacks"] = 1
        candidate["intent_epoch_slate_health"][
            "same_epoch_history_carried"
        ] -= 1
        _validate_phase13_accounting(
            candidate,
            INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        )
        self.assertFalse(_faults_are_zero(candidate))

    def test_quality_gate_is_conjunctive_including_confidence_bound(self) -> None:
        baseline = _run(candidate=False, searches=11, digest="baseline")
        candidate = _run(candidate=True, searches=10, digest="candidate")

        passing = _quality_gates(baseline, candidate, _paired())
        failing = _quality_gates(baseline, candidate, _paired(-0.000000001))

        self.assertTrue(all(passing.values()))
        self.assertFalse(failing["paired_bootstrap_lower_95_not_below_zero"])
        self.assertEqual(sum(not value for value in failing.values()), 1)

    def test_complete_gate_builder_requires_exact_safe_candidate(self) -> None:
        baseline = _run(candidate=False, searches=11, digest="baseline")
        candidate = _run(candidate=True, searches=10, digest="candidate")
        replay = _run(candidate=True, searches=10, digest="candidate")
        independent = _run(candidate=True, searches=10, digest="candidate")
        performance = {
            "candidate_warm_p95_ratio": 1.0,
            "candidate_wall_time_ratio": 1.0,
            "candidate_additional_retained_agent_bytes": 256,
        }
        startup = {
            "candidate_startup_time_ratio": 1.0,
            "candidate_additional_startup_rss_bytes": 0,
        }

        gates = _build_gates(
            SUITES["development"],
            baseline,
            candidate,
            replay,
            independent,
            _paired(),
            performance,
            startup,
            lock_revalidated=True,
            privacy_valid=True,
        )

        self.assertTrue(gates["advance"])
        self.assertTrue(all(gates.values()))

    def test_publication_validator_rejects_rows_ids_and_scenario_cells(self) -> None:
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "suite": "development",
            "dataset": {},
            "run_configuration": {"variant": CANDIDATE_ID},
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
        self.assertTrue(publication_privacy_is_valid(payload))

        for mutation in (
            {"sessions": []},
            {"parent_asin": "B000000001"},
            {"small_cell": "intent_override"},
        ):
            with self.subTest(mutation=mutation):
                unsafe = copy.deepcopy(payload)
                unsafe["health"].update(mutation)
                self.assertFalse(publication_privacy_is_valid(unsafe))

    def test_startup_probe_keeps_candidate_constructor_empty_and_bounded(self) -> None:
        probe = _startup_probe(object(), iterations=100)
        self.assertEqual(probe["iterations"], 100)
        self.assertEqual(
            probe["candidate_empty_retained_bytes"],
            probe["baseline_empty_retained_bytes"],
        )
        self.assertEqual(probe["candidate_additional_startup_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
