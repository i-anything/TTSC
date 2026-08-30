"""Sequential, aggregate-only Phase 4 expected-utility A/B evaluation.

The comparator is the accepted Phase 2 lexicographic-evidence agent.  The
candidate changes only the decision policy: it builds the Phase 3 protocol
world, simulates bounded reply/rerank continuations, and selects a question and
legal recommendation width by the official evaluator utility.  Evaluation
labels are joined only after each label-free replay and are never serialized.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from conversational_search.decision_policy import (
    EXPECTED_UTILITY_DECISION_POLICY,
    PROTECTED_DECISION_POLICY,
    DecisionPolicy,
)
from conversational_search.intent import ROBUST_INTENT_POLICY
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    PROTOCOL_EVIDENCE_CAPABILITY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.run_fusion_ablations import _sha256
from scripts.run_multislot_intent_ablations import (
    _canonical_json,
    _current_max_rss_bytes,
    _deep_size,
    _evaluate_with_deterministic_session_ids,
    _metric_deltas,
    _overall_summary,
    _paired_statistics,
    _safe_ratio,
)
from scripts.run_orchestration_ablations import _lookup_accounting_exact
from scripts.run_phase2_exact_evidence_ablations import (
    AggregateAuditAgent,
    ExactEvidenceCallAuditRetriever,
    RuntimeNetworkAudit,
    VariantRun,
    _scenario_report,
    _write_json_exclusive,
    build_fold_report,
    select_smoke_indices,
)
from scripts.run_profile_ablations import (
    TOKEN_USAGE_KEYS,
    _project_profile_health,
)
from scripts.run_reranking_ablations import _expected_turns


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORT_ID = "phase4-protocol-expected-utility-bounded-world-ab-20260830"
EXPERIMENT_ID = "phase4-protocol-expected-utility-v4-bounded-world"
PHASE2_LOCK_RELATIVE = (
    "benchmarks/diagnostics/"
    "phase2-lexicographic-exact-evidence-replayplan-ab-20260830.json"
)
BASELINE_ID = PROTECTED_DECISION_POLICY.value
CANDIDATE_ID = EXPECTED_UTILITY_DECISION_POLICY.value
MAX_ADDITIONAL_RSS_BYTES = 64 * 1024 * 1024
MAX_ADDITIONAL_AGENT_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_RATIO = 1.10
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
_FORBIDDEN_PUBLICATION_KEYS = (
    '"sessions"',
    '"sample_id"',
    '"ground_truth"',
    '"parent_asin"',
    '"user_message"',
    '"user_profile"',
    '"recommendations"',
    '"per_session"',
    '"rows"',
)

REQUIRED_ENVIRONMENT = {
    "TTSC_ONNX_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_ENABLE_MPS_FALLBACK": "0",
}

SOURCE_PATHS = (
    "conversational_search/decision.py",
    "conversational_search/decision_policy.py",
    "conversational_search/exact_evidence.py",
    "conversational_search/protocol.py",
    "conversational_search/service.py",
    "conversational_search/utility_planner.py",
    "evaluator/local_evaluator.py",
    "scripts/run_phase4_expected_utility_ablation.py",
    "starter/agent.py",
)


class TraceAggregateAuditAgent(AggregateAuditAgent):
    """Aggregate only fixed-vocabulary planner trace fields after each turn."""

    def __init__(self, delegate: ConversationalSearchAgent, catalog_ids: set[str]) -> None:
        super().__init__(delegate, catalog_ids)
        self._trace_counts: dict[str, Counter[str]] = {
            "planner_outcome": Counter(),
            "world_mode": Counter(),
            "fallback_reason": Counter(),
            "simulated_reply_partitions": Counter(),
            "pruned_questions": Counter(),
        }

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        trace = self._search_delegate.last_action_trace(session_id)
        if isinstance(trace, dict):
            for output_key, trace_key in (
                ("planner_outcome", "planner_outcome"),
                ("world_mode", "world_mode"),
                ("fallback_reason", "planner_fallback_reason"),
                ("simulated_reply_partitions", "simulated_reply_partitions"),
                ("pruned_questions", "pruned_questions"),
            ):
                value = trace.get(trace_key)
                normalized = "none" if value is None else str(value)
                self._trace_counts[output_key][normalized] += 1
        return response

    def trace_summary(self) -> dict[str, dict[str, int]]:
        return {
            key: dict(sorted(counter.items()))
            for key, counter in self._trace_counts.items()
        }


def _validate_execution_environment() -> None:
    mismatches = {
        key: (expected, os.environ.get(key))
        for key, expected in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"single-thread execution environment is not pinned: {mismatches}"
        )


def _new_agent(
    catalog: Path,
    backend: object,
    decision_policy: DecisionPolicy,
) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        catalog,
        retriever=backend,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        intent_policy=ROBUST_INTENT_POLICY,
        decision_policy=decision_policy,
        requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    return _deep_size(
        (
            getattr(agent, "_sessions", None),
            getattr(agent, "_slates", None),
            getattr(agent, "_profile_priors", None),
            getattr(planner, "_entries", None),
            getattr(agent, "_exact_evidence_counts", None),
            getattr(agent, "_protocol_consistency", None),
            getattr(agent, "_protocol_events", None),
            getattr(agent, "_protocol_override_pending", None),
            getattr(agent, "_protocol_shown_ids", None),
            getattr(agent, "_protocol_action_traces", None),
            getattr(agent, "_expected_exact_replays", None),
        )
    )


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    decision_policy: DecisionPolicy,
) -> VariantRun:
    guarded = ExactEvidenceCallAuditRetriever(backend)
    agent = _new_agent(catalog, guarded, decision_policy)
    audited = TraceAggregateAuditAgent(agent, catalog_ids)
    network = RuntimeNetworkAudit()
    started = time.perf_counter()
    with network.deny():
        result = _evaluate_with_deterministic_session_ids(
            audited,
            samples,
            catalog_ids,
            categories,
            products,
        )
    wall_seconds = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    searches = int(agent.orchestration_health["searches"])
    guarded.validate(searches)
    latency = audited.latency_summary()
    if int(latency["count"]) != expected_turns:
        raise RuntimeError("response latency coverage is incomplete")
    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": guarded.summary(),
        "ranking_health": agent.ranking_health,
        "exact_evidence_health": agent.exact_evidence_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "slate_health": agent.slate_health,
        "intent_epoch_slate_health": agent.intent_epoch_slate_health,
        "orchestration_health": agent.orchestration_health,
        "protocol_decision_health": agent.protocol_decision_health,
        "decision_trace_health": audited.trace_summary(),
        "response_audit": {
            "response_exceptions": audited.response_exceptions,
            "invalid_api_responses": audited.invalid_api_responses,
        },
        "runtime_network_attempts": network.attempts,
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
        "peak_rss_bytes": _current_max_rss_bytes(),
    }
    _validate_variant_accounting(diagnostics, decision_policy)
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("evaluator sessions are unavailable")
    return VariantRun(
        summary=_overall_summary(result),
        sessions=sessions,
        diagnostics=diagnostics,
        evaluator_digest=hashlib.sha256(_canonical_json(result)).hexdigest(),
        behavior_digest=audited.behavior_digest,
        actions=audited.actions,
    )


def _validate_variant_accounting(
    diagnostics: Mapping[str, object],
    decision_policy: DecisionPolicy,
) -> None:
    turns = int(diagnostics["expected_turns"])
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    exact = diagnostics["exact_evidence_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
    protocol = diagnostics["protocol_decision_health"]
    if not all(
        isinstance(value, dict)
        for value in (route, ranking, exact, slate, orchestration, protocol)
    ):
        raise RuntimeError("variant diagnostics have an invalid schema")

    searches = int(orchestration["searches"])
    reuses = int(orchestration["reuses"])
    if int(orchestration["decisions"]) != turns or searches + reuses != turns:
        raise RuntimeError("orchestration accounting is incomplete")
    if int(orchestration["skips"]) or not _lookup_accounting_exact(orchestration):
        raise RuntimeError("orchestration cache accounting is inconsistent")
    route_calls = sum(int(value) for value in (route.get("bm25") or {}).values())
    if route_calls != searches:
        raise RuntimeError("retrieval-route accounting is incomplete")
    if (
        int(ranking["attempts"]) != searches
        or int(ranking["successes"]) != searches
        or int(route["candidate_document_calls"]) != searches
    ):
        raise RuntimeError("Stage-A accounting is incomplete")
    if int(ranking["failures"]) or int(ranking["unavailable_skips"]):
        raise RuntimeError("Stage-A faults invalidate the experiment")

    exact_attempts = int(exact["attempts"])
    exact_outcomes = sum(
        int(exact[key])
        for key in (
            "applied",
            "zero_support_fail_open",
            "capability_unavailable",
            "evidence_errors",
            "validation_errors",
        )
    )
    if exact_attempts != exact_outcomes:
        raise RuntimeError("exact-evidence outcomes do not partition attempts")
    expected_exact_attempts = (
        turns
        if decision_policy is EXPECTED_UTILITY_DECISION_POLICY
        else searches
    )
    if (
        exact_attempts != expected_exact_attempts
        or int(route["candidate_protocol_evidence_calls"]) != exact_attempts
    ):
        raise RuntimeError("exact-evidence call accounting drifted")

    protocol_outcomes = sum(
        int(protocol[key])
        for key in (
            "applied",
            "unsupported_or_disabled",
            "capability_unavailable",
            "candidate_or_evidence_error",
            "fail_open_evidence",
            "fail_open_no_candidates",
            "fail_open_no_support",
            "fail_open_validation",
        )
    )
    question_total = sum(
        int(value) for value in protocol["question_action_counts"].values()
    )
    width_total = sum(
        int(value) for value in protocol["width_action_counts"].values()
    )
    if decision_policy is EXPECTED_UTILITY_DECISION_POLICY:
        if (
            protocol["policy"] != CANDIDATE_ID
            or int(protocol["turns"]) != turns
            or protocol_outcomes != turns
            or question_total != turns
            or width_total != turns
        ):
            raise RuntimeError("expected-utility accounting is incomplete")
        zero_width = int(protocol["width_action_counts"]["0"])
        expected_slate_attempts = turns - zero_width
    else:
        if (
            protocol["policy"] != BASELINE_ID
            or int(protocol["turns"])
            or protocol_outcomes
            or question_total
            or width_total
        ):
            raise RuntimeError("protected comparator performed protocol planning")
        expected_slate_attempts = turns
    if (
        int(slate["attempts"]) != expected_slate_attempts
        or int(slate["successes"]) != expected_slate_attempts
        or int(slate["failures"])
    ):
        raise RuntimeError("slate accounting is incomplete")


def _warm_backend(catalog: Path, backend: object) -> None:
    agent = _new_agent(catalog, backend, PROTECTED_DECISION_POLICY)
    session_id = "phase4-label-free-backend-warmup"
    agent.reset(session_id, {})
    agent.respond(
        session_id,
        "I'm exploring a generic clothing item and have no firm preference yet.",
        1,
        10,
    )
    if int(agent.ranking_health["successes"]) != 1:
        raise RuntimeError("label-free backend warm-up failed")


def _faults_are_zero(run: VariantRun) -> bool:
    diagnostics = run.diagnostics
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    exact = diagnostics["exact_evidence_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
    protocol = diagnostics["protocol_decision_health"]
    response = diagnostics["response_audit"]
    return (
        int(route["fallback_turns"]) == 0
        and int(ranking["failures"]) == 0
        and int(ranking["unavailable_skips"]) == 0
        and int(exact["capability_unavailable"]) == 0
        and int(exact["evidence_errors"]) == 0
        and int(exact["validation_errors"]) == 0
        and int(slate["failures"]) == 0
        and int(orchestration["fault_invalidations"]) == 0
        and int(orchestration["store_rejections"]) == 0
        and int(protocol["capability_unavailable"]) == 0
        and int(protocol["candidate_or_evidence_error"]) == 0
        and int(protocol["fail_open_validation"]) == 0
        and int(response["response_exceptions"]) == 0
        and int(response["invalid_api_responses"]) == 0
        and int(diagnostics["runtime_network_attempts"]) == 0
    )


def _tokens_are_zero(summary: Mapping[str, object]) -> bool:
    usage = summary.get("reported_token_usage")
    return isinstance(usage, dict) and all(
        type(usage.get(key)) is int and usage[key] == 0
        for key in TOKEN_USAGE_KEYS
    )


def _deterministic_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        key: diagnostics[key]
        for key in (
            "expected_turns",
            "route_health",
            "ranking_health",
            "exact_evidence_health",
            "profile_health",
            "slate_health",
            "intent_epoch_slate_health",
            "orchestration_health",
            "protocol_decision_health",
            "decision_trace_health",
            "response_audit",
            "runtime_network_attempts",
            "retained_agent_bytes",
        )
    }


def _action_change_summary(baseline: VariantRun, candidate: VariantRun) -> dict:
    baseline_actions = {
        (ordinal, turn): (question, width)
        for ordinal, turn, question, width in baseline.actions
    }
    candidate_actions = {
        (ordinal, turn): (question, width)
        for ordinal, turn, question, width in candidate.actions
    }
    common = sorted(set(baseline_actions).intersection(candidate_actions))
    return {
        "common_turns": len(common),
        "question_changes": sum(
            baseline_actions[key][0] != candidate_actions[key][0] for key in common
        ),
        "width_changes": sum(
            baseline_actions[key][1] != candidate_actions[key][1] for key in common
        ),
        "narrower_candidate_slates": sum(
            candidate_actions[key][1] < baseline_actions[key][1] for key in common
        ),
        "zero_width_candidate_slates": sum(
            candidate_actions[key][1] == 0 for key in common
        ),
    }


def _variant_public(run: VariantRun) -> dict:
    diagnostics = run.diagnostics
    return {
        "official_metrics": run.summary,
        "route_health": diagnostics["route_health"],
        "ranking_health": diagnostics["ranking_health"],
        "exact_evidence_health": diagnostics["exact_evidence_health"],
        "orchestration_health": diagnostics["orchestration_health"],
        "protocol_decision_health": diagnostics["protocol_decision_health"],
        "decision_trace_health": diagnostics["decision_trace_health"],
        "response_audit": diagnostics["response_audit"],
        "runtime_network_attempts": diagnostics["runtime_network_attempts"],
        "retained_agent_bytes": diagnostics["retained_agent_bytes"],
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
        "peak_rss_bytes": diagnostics["peak_rss_bytes"],
    }


def _performance(baseline: VariantRun, candidate: VariantRun) -> dict:
    baseline_wall = float(baseline.diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate.diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(baseline.diagnostics["respond_latency_ms"]["warm_p95"])
    candidate_p95 = float(candidate.diagnostics["respond_latency_ms"]["warm_p95"])
    baseline_bytes = int(baseline.diagnostics["retained_agent_bytes"])
    candidate_bytes = int(candidate.diagnostics["retained_agent_bytes"])
    return {
        "baseline_wall_seconds": baseline_wall,
        "candidate_wall_seconds": candidate_wall,
        "candidate_wall_time_ratio": round(
            _safe_ratio(candidate_wall, baseline_wall), 6
        ),
        "baseline_warm_p95_ms": baseline_p95,
        "candidate_warm_p95_ms": candidate_p95,
        "candidate_warm_p95_ratio": round(
            _safe_ratio(candidate_p95, baseline_p95), 6
        ),
        "baseline_retained_agent_bytes": baseline_bytes,
        "candidate_retained_agent_bytes": candidate_bytes,
        "candidate_additional_retained_agent_bytes": max(
            0, candidate_bytes - baseline_bytes
        ),
        "candidate_peak_rss_bytes": int(candidate.diagnostics["peak_rss_bytes"]),
    }


def _locked_phase2_metrics() -> tuple[dict, int]:
    payload = json.loads(
        (REPOSITORY_ROOT / PHASE2_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    full = payload.get("full")
    if not isinstance(full, dict) or full.get("status") != "completed":
        raise RuntimeError("accepted Phase 2 benchmark lock is incomplete")
    candidate = full.get("candidate")
    if not isinstance(candidate, dict):
        raise RuntimeError("accepted Phase 2 candidate lock is missing")
    metrics = candidate.get("official_metrics")
    peak = candidate.get("peak_rss_bytes")
    if not isinstance(metrics, dict) or isinstance(peak, bool) or not isinstance(peak, int):
        raise RuntimeError("accepted Phase 2 resource lock is invalid")
    return dict(metrics), peak


def _smoke_gates(
    baseline: VariantRun,
    candidate: VariantRun,
    replay: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    transitions = paired["transitions"]
    gates = {
        "candidate_has_no_baseline_hit_loss": (
            int(transitions["baseline_only_hit"]) == 0
        ),
        "candidate_hit_rate_not_below_baseline": (
            float(candidate.summary["hit_rate_at_10"])
            >= float(baseline.summary["hit_rate_at_10"])
        ),
        "candidate_exercises_expected_utility": (
            int(candidate.diagnostics["protocol_decision_health"]["applied"]) > 0
        ),
        "candidate_replay_evaluator_exact": (
            candidate.evaluator_digest == replay.evaluator_digest
        ),
        "candidate_replay_behavior_exact": (
            candidate.behavior_digest == replay.behavior_digest
            and candidate.actions == replay.actions
        ),
        "candidate_replay_deterministic_health_exact": (
            _deterministic_health(candidate.diagnostics)
            == _deterministic_health(replay.diagnostics)
        ),
        "all_smoke_variants_fault_free": all(
            _faults_are_zero(run) for run in (baseline, candidate, replay)
        ),
        "all_smoke_variants_zero_token": all(
            _tokens_are_zero(run.summary) for run in (baseline, candidate, replay)
        ),
    }
    gates["authorize_full"] = all(gates.values())
    return gates


def _full_gates(
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
    folds: Mapping[str, object],
    scenarios: Mapping[str, object],
    performance: Mapping[str, object],
    *,
    phase2_metrics: Mapping[str, object],
    phase2_peak_rss: int,
) -> dict[str, bool]:
    transitions = paired["transitions"]
    fold_summary = folds["summary"]
    scenario_nonnegative = sum(
        float(value["delta"]["recommended_technical_score"]) >= 0
        for value in scenarios.values()
    )
    metric_keys = (
        "sample_count",
        "hit_rate_at_10",
        "mrr",
        "mttc",
        "efficiency",
        "recommended_technical_score",
        "reported_token_usage",
    )
    baseline_projection = {key: baseline.summary[key] for key in metric_keys}
    phase2_projection = {key: phase2_metrics[key] for key in metric_keys}
    gates = {
        "shared_backend_baseline_matches_accepted_phase2": (
            baseline_projection == phase2_projection
        ),
        "candidate_has_no_baseline_hit_loss": (
            int(transitions["baseline_only_hit"]) == 0
        ),
        "candidate_hit_rate_not_below_baseline": (
            float(candidate.summary["hit_rate_at_10"])
            >= float(baseline.summary["hit_rate_at_10"])
        ),
        "candidate_mrr_strictly_improves": (
            float(candidate.summary["mrr"]) > float(baseline.summary["mrr"])
        ),
        "candidate_mttc_not_above_baseline": (
            float(candidate.summary["mttc"]) <= float(baseline.summary["mttc"])
        ),
        "candidate_technical_score_strictly_improves": (
            float(candidate.summary["recommended_technical_score"])
            > float(baseline.summary["recommended_technical_score"])
        ),
        "mrr_nonnegative_in_at_least_four_folds": (
            int(fold_summary["mrr_nonnegative_fold_count"]) >= 4
        ),
        "mrr_positive_in_at_least_three_folds": (
            int(fold_summary["mrr_positive_fold_count"]) >= 3
        ),
        "median_fold_mrr_delta_positive": (
            float(fold_summary["median_mrr_delta"]) > 0
        ),
        "worst_fold_technical_score_delta_at_least_minus_0_005": (
            float(fold_summary["worst_technical_score_delta"]) >= -0.005
        ),
        "technical_score_nonnegative_in_at_least_three_scenarios": (
            scenario_nonnegative >= 3
        ),
        "both_full_variants_fault_free": (
            _faults_are_zero(baseline) and _faults_are_zero(candidate)
        ),
        "both_full_variants_zero_token": (
            _tokens_are_zero(baseline.summary)
            and _tokens_are_zero(candidate.summary)
        ),
        "candidate_wall_time_ratio_at_most_1_10": (
            float(performance["candidate_wall_time_ratio"]) <= MAX_RUNTIME_RATIO
        ),
        "candidate_warm_p95_ratio_at_most_1_10": (
            float(performance["candidate_warm_p95_ratio"]) <= MAX_RUNTIME_RATIO
        ),
        "candidate_additional_retained_agent_bytes_at_most_2mib": (
            int(performance["candidate_additional_retained_agent_bytes"])
            <= MAX_ADDITIONAL_AGENT_BYTES
        ),
        "candidate_peak_rss_within_phase2_plus_64mib": (
            int(performance["candidate_peak_rss_bytes"])
            <= phase2_peak_rss + MAX_ADDITIONAL_RSS_BYTES
        ),
    }
    gates["adopt"] = all(gates.values())
    return gates


def run_phase4_expected_utility_ablation(
    catalog_path: str | Path,
    dataset_path: str | Path,
) -> dict:
    _validate_execution_environment()
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    smoke_indices = select_smoke_indices(samples)
    smoke_samples = [samples[index] for index in smoke_indices]
    catalog_ids, categories, products = catalog_index(catalog)
    phase2_metrics, phase2_peak_rss = _locked_phase2_metrics()

    rss_before_backend = _current_max_rss_bytes()
    backend_started = time.perf_counter()
    bootstrap = ConversationalSearchAgent(
        catalog,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    backend = bootstrap.retrieval_backend
    backend_wall_seconds = time.perf_counter() - backend_started
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable")
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable")
    if (
        getattr(backend, "protocol_evidence_capability", None)
        is not PROTOCOL_EVIDENCE_CAPABILITY
    ):
        raise RuntimeError("protocol evidence is unavailable")
    rss_after_backend = _current_max_rss_bytes()
    del bootstrap
    _warm_backend(catalog, backend)

    smoke_candidate = _run_variant(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
        EXPECTED_UTILITY_DECISION_POLICY,
    )
    smoke_baseline = _run_variant(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
        PROTECTED_DECISION_POLICY,
    )
    smoke_replay = _run_variant(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
        EXPECTED_UTILITY_DECISION_POLICY,
    )
    smoke_paired = _paired_statistics(
        smoke_baseline.sessions,
        smoke_candidate.sessions,
    )
    smoke_gates = _smoke_gates(
        smoke_baseline,
        smoke_candidate,
        smoke_replay,
        smoke_paired,
    )
    smoke_payload = {
        "sample_count": len(smoke_samples),
        "baseline": _variant_public(smoke_baseline),
        "candidate": _variant_public(smoke_candidate),
        "metric_delta": _metric_deltas(
            smoke_baseline.summary,
            smoke_candidate.summary,
        ),
        "paired_quality": smoke_paired,
        "action_changes": _action_change_summary(
            smoke_baseline,
            smoke_candidate,
        ),
        "candidate_replay_exactness": {
            "evaluator_payload_equal": (
                smoke_candidate.evaluator_digest == smoke_replay.evaluator_digest
            ),
            "response_behavior_equal": (
                smoke_candidate.behavior_digest == smoke_replay.behavior_digest
            ),
            "actions_equal": smoke_candidate.actions == smoke_replay.actions,
            "deterministic_health_equal": (
                _deterministic_health(smoke_candidate.diagnostics)
                == _deterministic_health(smoke_replay.diagnostics)
            ),
        },
        "decision_gate": smoke_gates,
    }

    full_payload: dict[str, object]
    decision: dict[str, object]
    if not smoke_gates["authorize_full"]:
        full_payload = {"status": "not_run", "reason": "smoke_gate_rejected"}
        decision = {
            "status": "rejected_at_smoke",
            "adopt": False,
            "reason": "The candidate failed at least one frozen smoke safety gate.",
        }
    else:
        del smoke_replay
        gc.collect()
        full_candidate = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            EXPECTED_UTILITY_DECISION_POLICY,
        )
        gc.collect()
        full_baseline = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            PROTECTED_DECISION_POLICY,
        )
        full_paired = _paired_statistics(
            full_baseline.sessions,
            full_candidate.sessions,
        )
        folds = build_fold_report(
            samples,
            full_baseline.sessions,
            full_candidate.sessions,
        )
        scenarios = _scenario_report(
            full_baseline.sessions,
            full_candidate.sessions,
        )
        performance = _performance(full_baseline, full_candidate)
        full_gates = _full_gates(
            full_baseline,
            full_candidate,
            full_paired,
            folds,
            scenarios,
            performance,
            phase2_metrics=phase2_metrics,
            phase2_peak_rss=phase2_peak_rss,
        )
        full_payload = {
            "status": "completed",
            "baseline": _variant_public(full_baseline),
            "candidate": _variant_public(full_candidate),
            "metric_delta": _metric_deltas(
                full_baseline.summary,
                full_candidate.summary,
            ),
            "paired_quality": full_paired,
            "scenario_subsets": scenarios,
            "disjoint_folds": folds,
            "action_changes": _action_change_summary(
                full_baseline,
                full_candidate,
            ),
            "performance": performance,
            "decision_gate": full_gates,
        }
        decision = {
            "status": "accepted" if full_gates["adopt"] else "rejected",
            "adopt": full_gates["adopt"],
            "reason": (
                "Every frozen quality, fold, fault, token, latency, and memory "
                "gate passed."
                if full_gates["adopt"]
                else "At least one frozen full-evaluation promotion gate failed."
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "A protocol-aware, reply-simulating expected-utility planner can "
            "improve reciprocal rank and conversion decisions over the accepted "
            "Phase 2 exact-evidence agent without reducing recall."
        ),
        "isolated_change": {
            "baseline_decision_policy": BASELINE_ID,
            "candidate_decision_policy": CANDIDATE_ID,
            "shared_ranking_policy": (
                LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY.value
            ),
            "shared_slate_policy": INTENT_EPOCH_NOVELTY_SLATE_POLICY.value,
            "candidate_set_additions_or_filters": 0,
            "external_model_or_api_added": False,
        },
        "dataset": {
            "catalog_sha256": _sha256(catalog),
            "evaluation_sha256": _sha256(dataset),
            "full_sample_count": len(samples),
        },
        "run_configuration": {
            "execution": "strictly_sequential_cpu",
            "threads": 1,
            "processes": 1,
            "shared_protocol_capable_backend": True,
            "fresh_agent_state_per_variant": True,
            "run_order": [
                "smoke_candidate",
                "smoke_baseline",
                "smoke_candidate_replay",
                "full_candidate_if_authorized",
                "full_baseline_if_authorized",
            ],
            "external_api_calls": 0,
            "gpu_or_mps": False,
            "environment": dict(REQUIRED_ENVIRONMENT),
        },
        "backend_startup": {
            "wall_seconds": round(backend_wall_seconds, 6),
            "peak_rss_before_bytes": rss_before_backend,
            "peak_rss_after_bytes": rss_after_backend,
            "phase2_peak_rss_reference_bytes": phase2_peak_rss,
        },
        "smoke": smoke_payload,
        "full": full_payload,
        "decision": decision,
        "privacy": {
            "aggregate_only": True,
            "labels_used_for_deterministic_subset_selection": True,
            "labels_joined_to_agent_outputs_only_after_replay": True,
            "runtime_received_labels_directly": False,
            "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "phase2_baseline_sha256": _sha256(
                REPOSITORY_ROOT / PHASE2_LOCK_RELATIVE
            ),
            "source_sha256": {
                relative: _sha256(REPOSITORY_ROOT / relative)
                for relative in SOURCE_PATHS
            },
        },
    }
    validate_phase4_publication(payload)
    return payload


def validate_phase4_publication(payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    if any(key in serialized for key in _FORBIDDEN_PUBLICATION_KEYS):
        raise ValueError("aggregate report contains a forbidden raw-data key")
    if _ASIN_RE.search(serialized):
        raise ValueError("aggregate report contains a product identifier")
    expected_privacy = {
        "aggregate_only": True,
        "labels_used_for_deterministic_subset_selection": True,
        "labels_joined_to_agent_outputs_only_after_replay": True,
        "runtime_received_labels_directly": False,
        "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
    }
    if payload.get("privacy") != expected_privacy:
        raise ValueError("aggregate report privacy assertions are incomplete")


def _validate_output_path(output: Path, catalog: Path, dataset: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        *(REPOSITORY_ROOT / relative for relative in SOURCE_PATHS),
        REPOSITORY_ROOT / PHASE2_LOCK_RELATIVE,
    }
    if output.resolve() in protected:
        raise ValueError("output must not overwrite an input or source file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sequential aggregate-only Phase 4 expected-utility A/B"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    catalog = Path(arguments.catalog).resolve()
    dataset = Path(arguments.dataset).resolve()
    output = Path(arguments.output).resolve()
    _validate_output_path(output, catalog, dataset)
    payload = run_phase4_expected_utility_ablation(catalog, dataset)
    _write_json_exclusive(output, payload)


if __name__ == "__main__":
    main()
