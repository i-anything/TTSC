"""Sealed aggregate-only Phase 13 generator-separated evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    SlatePolicy,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_bm25_rescue_ablations import (
    _canonical_private_cache_snapshot,
)
from scripts.run_fusion_ablations import _sha256
from scripts.run_multislot_intent_ablations import (
    _canonical_json,
    _content_fingerprint,
    _current_max_rss_bytes,
    _deep_size,
    _evaluate_with_deterministic_session_ids,
    _fingerprint_set_digest,
    _metric_deltas,
    _overall_summary,
    _paired_statistics,
    _private_digest,
    _safe_ratio,
)
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_profile_ablations import (
    TOKEN_USAGE_KEYS,
    _AuditAgent,
    _CallAuditRetriever,
    _inspect_retained_profile_state,
    _project_profile_health,
    _route_call_count,
    _validate_variant_accounting,
)
from scripts.run_reranking_ablations import _expected_turns
from scripts.verify_phase13_slate_oracle import (
    EXPECTED_SHA256 as PHASE13_ORACLE_SHA256,
    RANDOM_CASES as PHASE13_RANDOM_ORACLE_CASES,
    verify as verify_phase13_oracle,
)
from scripts.verify_phase7_stage_a_oracle import (
    EXPECTED_SHA256 as PHASE7_ORACLE_SHA256,
    ORACLE_CASES as PHASE7_ORACLE_CASES,
)
from scripts.verify_phase9_ranking_oracle import (
    EXPECTED_SHA256 as PHASE9_ORACLE_SHA256,
    ORACLE_CASES as PHASE9_ORACLE_CASES,
)


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase13-intent-epoch-continuation-novelty-v1"
CANDIDATE_ID = INTENT_EPOCH_NOVELTY_SLATE_POLICY.value
BASELINE_ID = "phase9-bounded-profile-residual-v1"
IMPLEMENTATION_LOCK_ID = "phase13-intent-epoch-continuation-novelty-implementation-v1"
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase13_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase13_experiment_contract.json"
BASELINE_LOCK_RELATIVE = "docs/phase13_baseline_lock.json"
DATASET_AUDIT_RELATIVE = "docs/phase11_dataset_audit.json"
RESEARCH_PLAN_RELATIVE = "docs/phase13_research_plan.md"

FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest "
    "tests.test_intent_epoch_slates "
    "tests.test_service_intent_epoch_slates "
    "tests.test_intent_epoch_slate_ablations tests.test_slates "
    "tests.test_service tests.test_orchestration -q"
)
PHASE7_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase7_stage_a_oracle"
PHASE9_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase9_ranking_oracle"
PHASE13_ORACLE_COMMAND = (
    ".venv/bin/python -m scripts.verify_phase13_slate_oracle"
)

_LOCKED_PYTHON_DIRECTORIES = (
    "conversational_search",
    "evaluator",
    "preprocessing",
    "scripts",
    "starter",
    "tests",
)
SOURCE_PATHS = tuple(
    sorted(
        {
            ".gitignore",
            CONTRACT_RELATIVE,
            BASELINE_LOCK_RELATIVE,
            DATASET_AUDIT_RELATIVE,
            RESEARCH_PLAN_RELATIVE,
            *(
                path.relative_to(REPOSITORY_ROOT).as_posix()
                for path in REPOSITORY_ROOT.glob("requirements*.txt")
            ),
            *(
                path.relative_to(REPOSITORY_ROOT).as_posix()
                for directory in _LOCKED_PYTHON_DIRECTORIES
                for path in (REPOSITORY_ROOT / directory).glob("*.py")
            ),
        }
    )
)
ALLOWED_CANDIDATE_SOURCE_CHANGES = frozenset(
    {
        "conversational_search/slates.py",
        "conversational_search/service.py",
        "scripts/run_intent_epoch_slate_ablations.py",
        "scripts/verify_phase13_slate_oracle.py",
        "tests/test_intent_epoch_slate_ablations.py",
        "tests/test_intent_epoch_slates.py",
        "tests/test_service_intent_epoch_slates.py",
    }
)

CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
PUBLIC_SHA256 = "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579"
PLAIN_SHA256 = "f2cdf94b8dbdf22373f42dd661f22372e92715d7bcbc924590db26cf824894db"
SCENARIO_AWARE_SHA256 = "78da5e8402bd7d6c7d9eee86de24eec9a13d3e433ed6f3cfe720bf85cb3319c9"
DEVELOPMENT_SET_SHA256 = "37bb8265543198c33305d40c6facf26f76e9109ac6f68afb529ef6a53b19eabd"
VALIDATION_SET_SHA256 = "a2677eb857c8df9ed963818c7c854c2d3ec936b7d597ed9810248fbc467f8ad1"
PUBLIC_SET_SHA256 = "a10adf5ef5e6424c749b0d97e2c7186da62ec76572de4bfbd2a28e8d42b3cb34"
PUBLIC_CASES = 200
BOOTSTRAP_SEED = 130260830

REQUIRED_ENVIRONMENT = {
    "TTSC_ONNX_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PHASE9_DEVELOPMENT_METRICS = {
    "sample_count": 996,
    "hit_rate_at_10": 0.991968,
    "mrr": 0.53488,
    "mttc": 3.058233,
    "efficiency": 0.794177,
    "recommended_technical_score": 0.815283,
    "reported_token_usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    },
}
PHASE9_PUBLIC_METRICS = {
    "sample_count": 200,
    "hit_rate_at_10": 0.99,
    "mrr": 0.529558,
    "mttc": 3.065,
    "efficiency": 0.7935,
    "recommended_technical_score": 0.812567,
    "reported_token_usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    },
}

_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
_FORBIDDEN_PUBLICATION_KEYS = frozenset(
    {
        "sessions",
        "session_id",
        "sample_id",
        "scenario_type",
        "scenario_metrics",
        "ground_truth",
        "target",
        "user_message",
        "user_profile",
        "preference_tags",
        "recommendations",
        "parent_asin",
        "product_id",
        "query",
        "message",
        "rows",
        "traces",
        "fingerprints",
        "per_session",
    }
)


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    name: str
    source: Path
    source_sha256: str
    source_rows: int
    evaluated_cases: int
    fingerprint_set_sha256: str
    exclude_public: bool
    prerequisites: tuple[str, ...] = ()

    @property
    def output(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase13-{self.name}.json"

    @property
    def attempt(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase13-{self.name}-attempt.json"


SUITES = {
    "development": SuiteConfig(
        "development",
        Path("/Users/limzichao/Downloads/public_plus_synthetic_1200.jsonl"),
        PLAIN_SHA256,
        1_200,
        996,
        DEVELOPMENT_SET_SHA256,
        True,
    ),
    "validation": SuiteConfig(
        "validation",
        Path(
            "/Users/limzichao/Downloads/"
            "public_plus_synthetic_scenario_aware_1200.jsonl"
        ),
        SCENARIO_AWARE_SHA256,
        1_200,
        1_000,
        VALIDATION_SET_SHA256,
        True,
        ("development",),
    ),
    "public": SuiteConfig(
        "public",
        REPOSITORY_ROOT / "data/public_set.jsonl",
        PUBLIC_SHA256,
        200,
        200,
        PUBLIC_SET_SHA256,
        False,
        ("development", "validation"),
    ),
}


@dataclass(slots=True)
class VariantRun:
    summary: dict
    sessions: list[dict]
    diagnostics: dict
    evaluator_digest: str
    private_digest: str


def _load_suite_samples(
    config: SuiteConfig,
) -> tuple[list[dict], dict[str, int | str]]:
    public_rows = load_jsonl(REPOSITORY_ROOT / "data/public_set.jsonl")
    public_fingerprints = {_content_fingerprint(row) for row in public_rows}
    if len(public_rows) != PUBLIC_CASES or len(public_fingerprints) != PUBLIC_CASES:
        raise RuntimeError("released public fingerprint contract drifted")
    if _fingerprint_set_digest(public_fingerprints) != PUBLIC_SET_SHA256:
        raise RuntimeError("released public fingerprint digest drifted")

    source_rows = load_jsonl(config.source)
    if len(source_rows) != config.source_rows:
        raise RuntimeError("suite source row count drifted")
    selected: list[dict] = []
    selected_fingerprints: set[bytes] = set()
    public_excluded = 0
    duplicate_excluded = 0
    for row in source_rows:
        fingerprint = _content_fingerprint(row)
        if config.exclude_public and fingerprint in public_fingerprints:
            public_excluded += 1
            continue
        if fingerprint in selected_fingerprints:
            duplicate_excluded += 1
            continue
        selected_fingerprints.add(fingerprint)
        selected.append(row)
    observed_set = _fingerprint_set_digest(selected_fingerprints)
    if len(selected) != config.evaluated_cases:
        raise RuntimeError("deduplicated suite case count drifted")
    if observed_set != config.fingerprint_set_sha256:
        raise RuntimeError("deduplicated suite fingerprint set drifted")
    return selected, {
        "source_rows": len(source_rows),
        "evaluated_cases": len(selected),
        "public_rows_excluded": public_excluded,
        "duplicate_rows_excluded": duplicate_excluded,
        "fingerprint_set_sha256": observed_set,
    }


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    candidate_counters = tuple(
        value
        for name, value in vars(agent).items()
        if name.startswith("_intent_epoch_slate_")
    )
    retained = (
        getattr(agent, "_sessions", None),
        getattr(agent, "_slates", None),
        getattr(agent, "_profile_priors", None),
        getattr(planner, "_entries", None),
        candidate_counters,
    )
    return _deep_size(retained)


def _startup_probe(backend: object, *, iterations: int = 10_000) -> dict:
    if iterations < 100:
        raise ValueError("startup probe requires at least 100 iterations")
    baseline_total = 0
    candidate_total = 0
    baseline: ConversationalSearchAgent | None = None
    candidate: ConversationalSearchAgent | None = None
    for index in range(iterations):
        policies = (
            (
                STAGNATION_AWARE_SLATE_POLICY,
                INTENT_EPOCH_NOVELTY_SLATE_POLICY,
            )
            if index % 2 == 0
            else (
                INTENT_EPOCH_NOVELTY_SLATE_POLICY,
                STAGNATION_AWARE_SLATE_POLICY,
            )
        )
        for policy in policies:
            started = time.perf_counter_ns()
            agent = ConversationalSearchAgent(
                "unused.jsonl",
                retriever=backend,
                slate_policy=policy,
            )
            elapsed = time.perf_counter_ns() - started
            if policy is STAGNATION_AWARE_SLATE_POLICY:
                baseline = agent
                baseline_total += elapsed
            else:
                candidate = agent
                candidate_total += elapsed
    if baseline is None or candidate is None:
        raise RuntimeError("startup probe did not construct agents")
    baseline_bytes = _retained_agent_bytes(baseline)
    rss_before = _current_max_rss_bytes()
    rss_candidate = ConversationalSearchAgent(
        "unused.jsonl",
        retriever=backend,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    candidate_rss_delta = max(0, _current_max_rss_bytes() - rss_before)
    candidate_bytes = _retained_agent_bytes(rss_candidate)
    return {
        "iterations": iterations,
        "baseline_total_ms": round(baseline_total / 1_000_000.0, 6),
        "candidate_total_ms": round(candidate_total / 1_000_000.0, 6),
        "candidate_startup_time_ratio": round(
            _safe_ratio(candidate_total, baseline_total),
            6,
        ),
        "candidate_additional_startup_rss_bytes": candidate_rss_delta,
        "baseline_empty_retained_bytes": baseline_bytes,
        "candidate_empty_retained_bytes": candidate_bytes,
    }


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    slate_policy: SlatePolicy,
) -> VariantRun:
    guarded = _CallAuditRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        slate_policy=slate_policy,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    audited = _AuditAgent(agent)
    started = time.perf_counter()
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
        "rescue_health": agent.rescue_health,
        "route_redundancy_health": agent.route_redundancy_health,
        "intent_epoch_slate_health": agent.intent_epoch_slate_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "slate_health": agent.slate_health,
        "orchestration_health": agent.orchestration_health,
        "retained_profile_state_valid": _inspect_retained_profile_state(
            agent,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }
    _validate_variant_accounting(
        diagnostics,
        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    _validate_phase13_accounting(diagnostics, slate_policy)
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("evaluator sessions are unavailable for paired checks")
    return VariantRun(
        summary=_overall_summary(result),
        sessions=sessions,
        diagnostics=diagnostics,
        evaluator_digest=hashlib.sha256(_canonical_json(result)).hexdigest(),
        private_digest=_private_digest(
            audited.action_trace,
            _canonical_private_cache_snapshot(agent),
        ),
    )


def _run_independent(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> VariantRun:
    runtime = ConversationalSearchAgent(
        catalog,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable independently")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable independently")
    return _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )


def _warm_backend(catalog: Path, backend: object) -> None:
    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    session_id = "phase13-label-free-runtime-warmup"
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    if int(warmup.ranking_health["successes"]) != 1:
        raise RuntimeError("label-free backend warm-up did not complete")


def _validate_phase13_accounting(
    diagnostics: Mapping[str, object],
    slate_policy: SlatePolicy,
) -> None:
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    rescue = diagnostics["rescue_health"]  # type: ignore[assignment]
    route_redundancy = diagnostics["route_redundancy_health"]  # type: ignore[assignment]
    slate = diagnostics["slate_health"]  # type: ignore[assignment]
    novelty = diagnostics["intent_epoch_slate_health"]  # type: ignore[assignment]
    attempts = int(slate["attempts"])  # type: ignore[index]
    if ranking["policy"] != "stage_a":  # type: ignore[index]
        raise RuntimeError("protected Stage-A policy drifted")
    if slate["policy"] != slate_policy.value:  # type: ignore[index]
        raise RuntimeError("slate policy telemetry drifted")
    if int(rescue["attempts"]):  # type: ignore[index]
        raise RuntimeError("Phase 13 unexpectedly enabled Phase 10 rescue")
    if int(route_redundancy["attempts"]):  # type: ignore[index]
        raise RuntimeError("Phase 13 unexpectedly enabled Phase 12 ranking")
    outcomes = (
        "empty_exact_baseline",
        "first_slate_exact_baseline",
        "unchanged_signature_exact_baseline",
        "changed_epoch_exact_baseline",
        "same_epoch_history_carried",
        "validation_fallbacks",
    )
    novelty_attempts = int(novelty["attempts"])  # type: ignore[index]
    if sum(int(novelty[key]) for key in outcomes) != novelty_attempts:  # type: ignore[index]
        raise RuntimeError("intent-epoch slate outcomes do not partition attempts")
    eligible_total = int(novelty["eligible_prior_shown_total"])  # type: ignore[index]
    if not 0 <= eligible_total <= novelty_attempts * 200:
        raise RuntimeError("eligible prior shown telemetry is invalid")
    if slate_policy is INTENT_EPOCH_NOVELTY_SLATE_POLICY:
        if novelty_attempts != attempts:
            raise RuntimeError("candidate intent-epoch slate coverage is incomplete")
    elif any(
        int(novelty[key])
        for key in ("attempts", *outcomes, "eligible_prior_shown_total")
    ):  # type: ignore[index]
        raise RuntimeError("protected baseline performed candidate work")


def _deterministic_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        key: diagnostics[key]
        for key in (
            "expected_turns",
            "route_health",
            "ranking_health",
            "rescue_health",
            "route_redundancy_health",
            "intent_epoch_slate_health",
            "profile_health",
            "slate_health",
            "orchestration_health",
            "retained_profile_state_valid",
            "retained_agent_bytes",
        )
    }


def _aggregate_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        **_deterministic_health(diagnostics),
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
    }


def _performance_summary(baseline: VariantRun, candidate: VariantRun) -> dict:
    baseline_wall = float(baseline.diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate.diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(
        baseline.diagnostics["respond_latency_ms"]["warm_p95"]  # type: ignore[index]
    )
    candidate_p95 = float(
        candidate.diagnostics["respond_latency_ms"]["warm_p95"]  # type: ignore[index]
    )
    baseline_bytes = int(baseline.diagnostics["retained_agent_bytes"])
    candidate_bytes = int(candidate.diagnostics["retained_agent_bytes"])
    return {
        "baseline_wall_seconds": round(baseline_wall, 6),
        "candidate_wall_seconds": round(candidate_wall, 6),
        "candidate_wall_time_ratio": round(
            _safe_ratio(candidate_wall, baseline_wall),
            6,
        ),
        "baseline_warm_p95_ms": round(baseline_p95, 6),
        "candidate_warm_p95_ms": round(candidate_p95, 6),
        "candidate_warm_p95_ratio": round(
            _safe_ratio(candidate_p95, baseline_p95),
            6,
        ),
        "baseline_retained_agent_bytes": baseline_bytes,
        "candidate_retained_agent_bytes": candidate_bytes,
        "candidate_additional_retained_agent_bytes": max(
            0,
            candidate_bytes - baseline_bytes,
        ),
    }


def _call_accounting(diagnostics: Mapping[str, object]) -> dict[str, int]:
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    return {
        "searches": int(orchestration["searches"]),  # type: ignore[index]
        "bm25_route_calls": _route_call_count(route),  # type: ignore[arg-type]
        "dense_route_calls": sum(
            int(value) for value in route["dense"].values()  # type: ignore[index,union-attr]
        ),
        "candidate_document_calls": int(route["candidate_document_calls"]),  # type: ignore[index]
        "stage_a_attempts": int(ranking["attempts"]),  # type: ignore[index]
    }


def _faults_are_zero(diagnostics: Mapping[str, object]) -> bool:
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    rescue = diagnostics["rescue_health"]  # type: ignore[assignment]
    redundancy = diagnostics["route_redundancy_health"]  # type: ignore[assignment]
    novelty = diagnostics["intent_epoch_slate_health"]  # type: ignore[assignment]
    profile = diagnostics["profile_health"]  # type: ignore[assignment]
    slate = diagnostics["slate_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    return (
        int(route["fallback_turns"]) == 0  # type: ignore[index]
        and int(ranking["failures"]) == 0  # type: ignore[index]
        and int(ranking["unavailable_skips"]) == 0  # type: ignore[index]
        and int(rescue["attempts"]) == 0  # type: ignore[index]
        and int(redundancy["validation_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(novelty["validation_fallbacks"]) == 0  # type: ignore[index]
        and int(profile["parsing_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(slate["failures"]) == 0  # type: ignore[index]
        and int(orchestration["fault_invalidations"]) == 0  # type: ignore[index]
        and int(orchestration["store_rejections"]) == 0  # type: ignore[index]
    )


def _tokens_are_zero(summary: Mapping[str, object]) -> bool:
    usage = summary.get("reported_token_usage")
    return isinstance(usage, dict) and all(
        type(usage.get(key)) is int and usage[key] == 0
        for key in TOKEN_USAGE_KEYS
    )


def _quality_gates(
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    baseline_metrics = baseline.summary
    candidate_metrics = candidate.summary
    transitions = paired["transitions"]  # type: ignore[assignment]
    bootstrap = paired["bootstrap"]  # type: ignore[assignment]
    return {
        "candidate_hit_rate_not_below_baseline": (
            float(candidate_metrics["hit_rate_at_10"])
            >= float(baseline_metrics["hit_rate_at_10"])
        ),
        "baseline_hit_to_candidate_miss_is_zero": (
            int(transitions["baseline_only_hit"]) == 0  # type: ignore[index]
        ),
        "candidate_mrr_not_below_baseline": (
            float(candidate_metrics["mrr"]) >= float(baseline_metrics["mrr"])
        ),
        "candidate_mttc_not_above_baseline": (
            float(candidate_metrics["mttc"]) <= float(baseline_metrics["mttc"])
        ),
        "candidate_technical_score_strictly_improves": (
            float(candidate_metrics["recommended_technical_score"])
            > float(baseline_metrics["recommended_technical_score"])
        ),
        "candidate_mrr_or_mttc_strictly_improves": (
            float(candidate_metrics["mrr"]) > float(baseline_metrics["mrr"])
            or float(candidate_metrics["mttc"])
            < float(baseline_metrics["mttc"])
        ),
        "paired_bootstrap_lower_95_not_below_zero": (
            float(bootstrap["lower_95"]) >= 0.0  # type: ignore[index]
        ),
    }


def _build_gates(
    config: SuiteConfig,
    baseline: VariantRun,
    candidate: VariantRun,
    replay: VariantRun,
    independent: VariantRun,
    paired: Mapping[str, object],
    performance: Mapping[str, object],
    startup: Mapping[str, object],
    *,
    lock_revalidated: bool,
    privacy_valid: bool,
) -> dict[str, bool]:
    baseline_calls = _call_accounting(baseline.diagnostics)
    candidate_calls = _call_accounting(candidate.diagnostics)
    candidate_total = sum(
        candidate_calls[key]
        for key in (
            "bm25_route_calls",
            "dense_route_calls",
            "candidate_document_calls",
            "stage_a_attempts",
        )
    )
    baseline_total = sum(
        baseline_calls[key]
        for key in (
            "bm25_route_calls",
            "dense_route_calls",
            "candidate_document_calls",
            "stage_a_attempts",
        )
    )
    one_call_per_search = all(
        calls["searches"]
        == calls["bm25_route_calls"]
        == calls["dense_route_calls"]
        == calls["candidate_document_calls"]
        == calls["stage_a_attempts"]
        for calls in (baseline_calls, candidate_calls)
    )
    candidate_replay_exact = (
        replay.evaluator_digest == candidate.evaluator_digest
        and replay.private_digest == candidate.private_digest
        and _deterministic_health(replay.diagnostics)
        == _deterministic_health(candidate.diagnostics)
    )
    independent_exact = (
        independent.evaluator_digest == candidate.evaluator_digest
        and independent.private_digest == candidate.private_digest
        and _deterministic_health(independent.diagnostics)
        == _deterministic_health(candidate.diagnostics)
    )
    gates = {
        **_quality_gates(baseline, candidate, paired),
        "candidate_replay_is_exact": candidate_replay_exact,
        "independent_explicit_policy_is_exact": independent_exact,
        "baseline_and_candidate_faults_are_zero": all(
            _faults_are_zero(run.diagnostics)
            for run in (baseline, candidate, replay, independent)
        ),
        "candidate_intent_epoch_slate_outcomes_partition_attempts": (
            int(candidate.diagnostics["intent_epoch_slate_health"]["attempts"])  # type: ignore[index]
            == int(candidate.diagnostics["slate_health"]["attempts"])  # type: ignore[index]
        ),
        "one_route_document_and_stage_a_call_per_search": one_call_per_search,
        "candidate_total_counted_calls_at_most_baseline": (
            candidate_total <= baseline_total
        ),
        "all_variants_report_zero_model_and_api_tokens": all(
            _tokens_are_zero(run.summary)
            for run in (baseline, candidate, replay, independent)
        ),
        "candidate_warm_p95_ratio_at_most_1_05": (
            float(performance["candidate_warm_p95_ratio"]) <= 1.05
        ),
        "candidate_wall_time_ratio_at_most_1_05": (
            float(performance["candidate_wall_time_ratio"]) <= 1.05
        ),
        "candidate_startup_time_ratio_at_most_1_05": (
            float(startup["candidate_startup_time_ratio"]) <= 1.05
        ),
        "candidate_additional_startup_rss_at_most_1mib": (
            int(startup["candidate_additional_startup_rss_bytes"]) <= 1_048_576
        ),
        "candidate_additional_retained_agent_bytes_at_most_2mib": (
            int(performance["candidate_additional_retained_agent_bytes"])
            <= 2_097_152
        ),
        "implementation_lock_revalidated_after_all_variants": lock_revalidated,
        "aggregate_publication_privacy_valid": privacy_valid,
    }
    if config.name == "development":
        gates["development_baseline_matches_protected_phase9"] = (
            baseline.summary == PHASE9_DEVELOPMENT_METRICS
        )
    elif config.name == "public":
        gates["public_baseline_matches_protected_phase9"] = (
            baseline.summary == PHASE9_PUBLIC_METRICS
        )
    gates["advance"] = all(gates.values())
    return gates


def _validate_execution_environment() -> None:
    if any(
        os.environ.get(key) != value
        for key, value in REQUIRED_ENVIRONMENT.items()
    ):
        raise RuntimeError("single-thread execution environment is not pinned")


def _validate_baseline_scope(repository_root: Path) -> dict:
    lock = json.loads(
        (repository_root / BASELINE_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    baseline_hashes = lock.get("source_sha256")
    if not isinstance(baseline_hashes, dict):
        raise RuntimeError("protected baseline source lock is malformed")
    for relative, expected in baseline_hashes.items():
        if relative in ALLOWED_CANDIDATE_SOURCE_CHANGES:
            continue
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("source outside the frozen candidate scope drifted")
    baseline_paths = set(baseline_hashes)
    new_python_paths = {
        relative
        for relative in SOURCE_PATHS
        if relative.endswith(".py") and relative not in baseline_paths
    }
    if not new_python_paths.issubset(ALLOWED_CANDIDATE_SOURCE_CHANGES):
        raise RuntimeError("candidate introduced Python outside frozen scope")
    if _sha256(repository_root / "starter/agent.py") != (
        lock["active_agent"]["starter_sha256"]
    ):
        raise RuntimeError("protected starter changed before promotion")
    return lock


def _validate_prelock_verification(value: object) -> dict:
    expected_keys = {
        "focused_suite_command",
        "focused_tests_passed",
        "complete_suite_command",
        "complete_unit_tests_passed",
        "phase7_oracle_command",
        "phase7_oracle_cases",
        "phase7_oracle_sha256",
        "phase9_oracle_command",
        "phase9_oracle_cases",
        "phase9_oracle_sha256",
        "phase13_oracle_command",
        "phase13_oracle_cases",
        "phase13_random_oracle_cases",
        "phase13_oracle_sha256",
        "completed_before_lock",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("Phase 13 pre-lock verification schema drifted")
    expected = {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "complete_suite_command": FULL_SUITE_COMMAND,
        "phase7_oracle_command": PHASE7_ORACLE_COMMAND,
        "phase7_oracle_cases": PHASE7_ORACLE_CASES,
        "phase7_oracle_sha256": PHASE7_ORACLE_SHA256,
        "phase9_oracle_command": PHASE9_ORACLE_COMMAND,
        "phase9_oracle_cases": PHASE9_ORACLE_CASES,
        "phase9_oracle_sha256": PHASE9_ORACLE_SHA256,
        "phase13_oracle_command": PHASE13_ORACLE_COMMAND,
        "phase13_oracle_cases": 45_600,
        "phase13_random_oracle_cases": PHASE13_RANDOM_ORACLE_CASES,
        "phase13_oracle_sha256": PHASE13_ORACLE_SHA256,
        "completed_before_lock": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RuntimeError("Phase 13 pre-lock verification evidence drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise RuntimeError("Phase 13 pre-lock test count is invalid")
    return value


def _validate_implementation_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    lock = json.loads(
        (repository_root / IMPLEMENTATION_LOCK_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    expected_keys = {
        "schema_version",
        "lock_id",
        "status",
        "contract_sha256",
        "baseline_lock_sha256",
        "dataset_audit_sha256",
        "research_plan_sha256",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 13 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 13 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 13 implementation lock identity")
    if lock.get("status") != "locked_before_generator_evaluation":
        raise RuntimeError("Phase 13 candidate is not frozen")
    documents = {
        "contract_sha256": CONTRACT_RELATIVE,
        "baseline_lock_sha256": BASELINE_LOCK_RELATIVE,
        "dataset_audit_sha256": DATASET_AUDIT_RELATIVE,
        "research_plan_sha256": RESEARCH_PLAN_RELATIVE,
    }
    for key, relative in documents.items():
        value = lock.get(key)
        if (
            not isinstance(value, str)
            or _HEX_SHA256_RE.fullmatch(value) is None
            or value != _sha256(repository_root / relative)
        ):
            raise RuntimeError("Phase 13 planning artifact drifted after lock")
    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 13 source lock is incomplete")
    observed = {
        relative: _sha256(repository_root / relative)
        for relative in SOURCE_PATHS
    }
    if observed != source_hashes:
        raise RuntimeError("Phase 13 implementation drifted after lock")
    _validate_prelock_verification(lock.get("verification"))
    _validate_baseline_scope(repository_root)
    return lock


def _claim_attempt(path: Path, suite: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(
            descriptor,
            _canonical_json(
                {
                    "schema_version": 1,
                    "experiment_id": EXPERIMENT_ID,
                    "suite": suite,
                    "status": "claimed",
                }
            )
            + b"\n",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_prerequisites(config: SuiteConfig) -> None:
    for suite in config.prerequisites:
        path = SUITES[suite].output
        if not path.is_file():
            raise RuntimeError("prior aggregate suite result is unavailable")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not publication_privacy_is_valid(payload):
            raise RuntimeError("prior suite result violates privacy")
        gates = payload.get("decision_gate")
        if not isinstance(gates, dict) or gates.get("advance") is not True:
            raise RuntimeError("prior suite did not pass every gate")


def _validate_run_paths(config: SuiteConfig, output: Path) -> None:
    if output.resolve() != config.output.resolve():
        raise ValueError("suite has one frozen aggregate output path")
    if output.exists():
        raise FileExistsError("suite output already exists")
    if config.attempt.exists():
        raise FileExistsError("suite attempt was already consumed")
    if not config.source.is_file():
        raise FileNotFoundError("suite source is unavailable")


def publication_privacy_is_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    required_top = {
        "schema_version",
        "experiment_id",
        "suite",
        "dataset",
        "run_configuration",
        "metrics",
        "paired_quality",
        "health",
        "call_accounting",
        "performance",
        "startup",
        "exactness",
        "privacy",
        "reproducibility",
        "decision_gate",
    }
    if set(payload) != required_top:
        return False
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("experiment_id") != EXPERIMENT_ID:
        return False
    if payload.get("suite") not in SUITES:
        return False

    keys: set[str] = set()
    strings: set[str] = set()

    def visit(value: object) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    return False
                keys.add(key)
                if not visit(item):
                    return False
            return True
        if isinstance(value, (list, tuple)):
            return all(visit(item) for item in value)
        if isinstance(value, str):
            strings.add(value)
            return True
        if value is None or type(value) in {bool, int}:
            return True
        return isinstance(value, float) and math.isfinite(value)

    if not visit(payload):
        return False
    if not keys.isdisjoint(_FORBIDDEN_PUBLICATION_KEYS):
        return False
    if not strings.isdisjoint(
        {"boundary", "browsing", "buying", "intent_override"}
    ):
        return False
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
    if _ASIN_RE.search(serialized):
        return False
    privacy = payload.get("privacy")
    return privacy == {
        "aggregate_metrics_and_fixed_counters_only": True,
        "row_scenario_message_profile_target_product_and_trace_data_absent": True,
        "per_case_fingerprints_absent": True,
        "manual_failure_inspection_performed": False,
    }


def run_intent_epoch_slate_ablation(
    suite: str,
    output_path: str | Path,
    *,
    thermal_safe_ack: bool,
) -> dict:
    if suite not in SUITES:
        raise ValueError("unsupported Phase 13 suite")
    if thermal_safe_ack is not True:
        raise RuntimeError("thermal safety must be checked before claiming a suite")
    config = SUITES[suite]
    output = Path(output_path).resolve()
    _validate_execution_environment()
    implementation_lock = _validate_implementation_lock()
    _validate_prerequisites(config)
    _validate_run_paths(config, output)

    _claim_attempt(config.attempt, config.name)
    if _sha256(REPOSITORY_ROOT / "data/catalog.jsonl") != CATALOG_SHA256:
        raise RuntimeError("catalog drifted after suite claim")
    if _sha256(config.source) != config.source_sha256:
        raise RuntimeError("suite source drifted after suite claim")
    samples, dataset_evidence = _load_suite_samples(config)
    catalog = REPOSITORY_ROOT / "data/catalog.jsonl"
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; suite consumed")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; suite consumed")
    _warm_backend(catalog, backend)
    startup = _startup_probe(backend)

    candidate = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    baseline = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        STAGNATION_AWARE_SLATE_POLICY,
    )
    replay = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    independent = _run_independent(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
    )
    lock_revalidated = _validate_implementation_lock() == implementation_lock

    paired = _paired_statistics(
        baseline.sessions,
        candidate.sessions,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    performance = _performance_summary(baseline, candidate)
    exactness = {
        "candidate_replay_evaluator_payload_equal": (
            replay.evaluator_digest == candidate.evaluator_digest
        ),
        "candidate_replay_action_state_slate_cache_equal": (
            replay.private_digest == candidate.private_digest
        ),
        "candidate_replay_aggregate_health_equal": (
            _deterministic_health(replay.diagnostics)
            == _deterministic_health(candidate.diagnostics)
        ),
        "independent_evaluator_payload_equal": (
            independent.evaluator_digest == candidate.evaluator_digest
        ),
        "independent_action_state_slate_cache_equal": (
            independent.private_digest == candidate.private_digest
        ),
        "independent_aggregate_health_equal": (
            _deterministic_health(independent.diagnostics)
            == _deterministic_health(candidate.diagnostics)
        ),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "suite": config.name,
        "dataset": {
            "source_sha256": config.source_sha256,
            **dataset_evidence,
        },
        "run_configuration": {
            "execution": "strictly_sequential_cpu",
            "threads": 1,
            "processes": 1,
            "shared_immutable_backend": True,
            "fresh_agent_state_per_variant": True,
            "variant_order": [
                CANDIDATE_ID,
                BASELINE_ID,
                CANDIDATE_ID,
                "independent_explicit_policy_candidate",
            ],
            "backend_warmup": "one_fixed_label_free_request",
            "thermal_safe_acknowledged": True,
            "external_api_calls": 0,
            "gpu_or_mps": False,
        },
        "metrics": {
            "baseline": baseline.summary,
            "candidate": candidate.summary,
            "delta": _metric_deltas(baseline.summary, candidate.summary),
        },
        "paired_quality": paired,
        "health": {
            "baseline": _aggregate_health(baseline.diagnostics),
            "candidate": _aggregate_health(candidate.diagnostics),
            "candidate_replay": _aggregate_health(replay.diagnostics),
            "independent_candidate": _aggregate_health(independent.diagnostics),
        },
        "call_accounting": {
            "baseline": _call_accounting(baseline.diagnostics),
            "candidate": _call_accounting(candidate.diagnostics),
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
            "platform": platform.platform(),
            "python": platform.python_version(),
            "environment": dict(REQUIRED_ENVIRONMENT),
            "implementation_lock_id": implementation_lock["lock_id"],
            "contract_sha256": implementation_lock["contract_sha256"],
            "source_sha256": implementation_lock["source_sha256"],
            "phase13_oracle_sha256": PHASE13_ORACLE_SHA256,
            "implementation_lock_revalidated_after_independent": (
                lock_revalidated
            ),
        },
        "decision_gate": {},
    }
    privacy_before_gates = publication_privacy_is_valid(payload)
    gates = _build_gates(
        config,
        baseline,
        candidate,
        replay,
        independent,
        paired,
        performance,
        startup,
        lock_revalidated=lock_revalidated,
        privacy_valid=privacy_before_gates,
    )
    payload["decision_gate"] = gates
    if not publication_privacy_is_valid(payload):
        raise RuntimeError("Phase 13 aggregate publication violates privacy")
    _write_json_atomic(output, payload)

    for run in (baseline, candidate, replay, independent):
        run.sessions.clear()
    samples.clear()
    products.clear()
    categories.clear()
    catalog_ids.clear()
    gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen Phase 13 generator-separated suite"
    )
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thermal-safe-ack", action="store_true")
    args = parser.parse_args()
    run_intent_epoch_slate_ablation(
        args.suite,
        args.output,
        thermal_safe_ack=args.thermal_safe_ack,
    )


if __name__ == "__main__":
    main()
