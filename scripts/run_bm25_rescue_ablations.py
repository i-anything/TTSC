"""Sealed aggregate-only Phase 10 BM25-rescue confirmation.

Evaluator rows and exact action/cache traces exist only in memory for paired
checks. The persisted result contains fixed-cardinality aggregate evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import fields
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    BackendSnapshotToken,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    RankingCacheEntry,
)
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    PROFILE_THEME_MASK_BYTES,
)
from conversational_search.ranking import (
    COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import STAGNATION_AWARE_SLATE_POLICY
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import _sha256
from scripts.run_orchestration_ablations import _lookup_accounting_exact
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_profile_ablations import (
    FROZEN_INPUT_SHA256 as PHASE9_FROZEN_INPUT_SHA256,
    OVERALL_METRIC_KEYS,
    PROFILE_HEALTH_KEYS,
    TOKEN_USAGE_KEYS,
    _CallAuditRetriever,
    _AuditAgent,
    _inspect_retained_profile_state,
    _latency_comparison,
    _overall_official_summary,
    _paired_hit_to_miss_count,
    _project_profile_health,
    _publication_privacy_is_valid as _phase9_publication_privacy_is_valid,
    _route_call_count,
    _safe_ratio,
    _validate_variant_accounting as _validate_phase9_variant_accounting,
)
from scripts.run_reranking_ablations import _expected_turns, _metric_deltas
from scripts.verify_phase7_stage_a_oracle import (
    EXPECTED_SHA256 as PHASE7_ORACLE_SHA256,
    ORACLE_CASES as PHASE7_ORACLE_CASES,
)
from scripts.verify_phase9_ranking_oracle import (
    EXPECTED_SHA256 as PHASE9_ORACLE_SHA256,
    ORACLE_CASES as PHASE9_ORACLE_CASES,
)
from scripts.verify_phase10_phase9_exact_oracle import (
    EXPECTED_SHA256 as PHASE10_PHASE9_ORACLE_SHA256,
    ORACLE_CASES as PHASE10_PHASE9_ORACLE_CASES,
)
from starter.agent import Agent


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase10-completeness-gated-bm25-rescue-v1"
CANDIDATE_ID = COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value
BASELINE_ID = "phase9-bounded-profile-residual-v1"
IMPLEMENTATION_LOCK_ID = (
    "phase10-completeness-gated-bm25-rescue-implementation-v1"
)
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase10_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase10_experiment_contract.json"
RAW_RESULT_RELATIVE = "results-phase10-bm25-rescue.json"
FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest tests.test_bm25_rescue_ranking "
    "tests.test_service_bm25_rescue tests.test_bm25_rescue_ablations "
    "tests.test_phase10_phase9_exact_oracle "
    "tests.test_phase9_ranking_oracle tests.test_phase7_stage_a_oracle "
    "tests.test_ranking tests.test_profile_ranking tests.test_service "
    "tests.test_service_profiles tests.test_orchestration -q"
)
PHASE7_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase7_stage_a_oracle"
PHASE9_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase9_ranking_oracle"
PHASE10_FORMULA_ORACLE_CASES = 10_000
PHASE10_PHASE9_ORACLE_COMMAND = (
    ".venv/bin/python -m scripts.verify_phase10_phase9_exact_oracle"
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

FROZEN_INPUT_SHA256 = {
    **PHASE9_FROZEN_INPUT_SHA256,
    "benchmarks/phase9.json": (
        "a9a0ff54b89b19736c7ae2125539cbdb1ec5e9a7cd3735e72aa54e656c245fce"
    ),
    "docs/phase9_implementation_lock.json": (
        "bdbe9df7efbbe3cca85b5c414bc5a14f4b8c3f31ae866ae2bb90b0aff3507338"
    ),
    "docs/phase9_results.json": (
        "54aee431b5c70cb0a2808a05cd63e47ea60e5e6beae7b42bc3d58e079a64e55c"
    ),
}

PHASE9_OFFICIAL = {
    "sample_count": 200,
    "hit_rate_at_10": 0.99,
    "mrr": 0.529558,
    "mttc": 3.065,
    "efficiency": 0.7935,
    "recommended_technical_score": 0.812567,
}

RANKING_HEALTH_KEYS = tuple(
    "policy attempts successes failures unavailable_skips".split()
)
RESCUE_HEALTH_KEYS = tuple(
    (
        "policy attempts zero_completeness_neutral "
        "bm25_unavailable_or_empty_neutral no_positive_uplift_neutral "
        "constant_uplift_neutral unchanged_order_neutral successful_reorders "
        "validation_or_scoring_fallbacks"
    ).split()
)
_RESCUE_OUTCOME_KEYS = RESCUE_HEALTH_KEYS[2:]
SLATE_HEALTH_KEYS = tuple(
    ("policy attempts successes failures initializations ranking_resets "
     "stagnant_turns unseen_selected_on_stagnant repeat_backfills").split()
)
ORCHESTRATION_HEALTH_KEYS = tuple(
    (
        "policy capacity maximum_ids_per_entry maximum_id_characters decisions "
        "searches reuses skips reasons lookups hits cold_misses dependency_misses "
        "backend_invalidations fault_invalidations stores store_rejections "
        "reset_invalidations capacity_evictions retrievals_avoided reranks_avoided "
        "entries cached_id_references cached_id_utf8_bytes retained_cache_bytes"
    ).split()
)
ROUTE_HEALTH_KEYS = tuple(
    "bm25 dense fallback_turns candidate_document_calls".split()
)
LATENCY_KEYS = tuple(
    "count warm_count p50 p90 p95 p99 warm_p95 max total".split()
)
DIAGNOSTIC_KEYS = tuple(
    (
        "expected_turns route_health ranking_health rescue_health profile_health "
        "slate_health orchestration_health retained_profile_state_valid "
        "retained_rescue_state_valid evaluation_wall_seconds respond_latency_ms"
    ).split()
)
CALL_ACCOUNTING_KEYS = tuple(
    ("bm25_route_calls dense_route_calls candidate_document_calls "
     "stage_a_attempts total_counted_calls").split()
)
RUN_CONFIGURATION_KEYS = tuple(
    (
        "execution onnx_threads shared_immutable_backend_for_ablation_variants "
        "fresh_agent_state_per_variant run_order backend_warmup external_api_calls "
        "post_run_tuning_or_second_candidate_run_allowed"
    ).split()
)
LATENCY_COMPARISON_KEYS = tuple(
    ("baseline_wall_seconds candidate_wall_seconds candidate_wall_time_ratio "
     "baseline_warm_p95_ms candidate_warm_p95_ms candidate_warm_p95_ratio").split()
)
EXACTNESS_KEYS = tuple(
    (
        "candidate_replay_evaluator_payload_equal "
        "candidate_replay_action_intent_slate_trace_equal "
        "candidate_replay_private_cache_snapshot_equal "
        "candidate_replay_aggregate_health_equal independent_evaluator_payload_equal "
        "independent_action_intent_slate_trace_equal "
        "independent_private_cache_snapshot_equal independent_aggregate_health_equal"
    ).split()
)
PRIVACY_KEYS = tuple(
    (
        "queries_messages_profiles_and_tags_absent "
        "product_sample_session_and_turn_rows_absent "
        "targets_scenarios_ranks_scores_and_action_traces_absent "
        "private_cache_snapshots_absent rescue_telemetry_is_fixed_global_counters_only"
    ).split()
)
REPRODUCIBILITY_KEYS = tuple(
    ("platform python frozen_input_sha256 implementation_lock_id contract_sha256 "
     "source_sha256 pre_lock_verification lock_revalidated_after_independent").split()
)
DECISION_GATE_KEYS = tuple(
    (
        "phase9_comparator_metrics_exact candidate_sample_count_is_200 "
        "candidate_hit_rate_at_10_at_least_0_99 candidate_mrr_strictly_above_0_529558 "
        "candidate_mttc_at_most_3_065 candidate_technical_score_strictly_above_0_812567 "
        "phase9_hit_to_phase10_miss_count_is_zero "
        "candidate_replay_payload_intent_slate_cache_and_health_exact "
        "independent_payload_intent_slate_cache_and_health_exact "
        "candidate_rescue_outcomes_partition_attempts phase9_comparator_rescue_attempts_are_zero "
        "candidate_and_comparator_have_one_call_per_search "
        "candidate_total_counted_calls_not_above_comparator "
        "candidate_and_comparator_report_zero_model_api_tokens "
        "candidate_and_comparator_faults_are_zero "
        "candidate_warm_p95_ratio_at_most_1_05 candidate_wall_time_ratio_at_most_1_05 "
        "synthetic_formula_phase7_phase9_and_complete_suites_passed "
        "bounded_profile_and_rescue_retained_state_valid aggregate_publication_privacy_valid "
        "source_contract_and_lock_validated_before_and_after_run "
        "additional_persistent_session_candidate_or_product_state_is_zero "
        "post_run_tuning_repair_or_second_candidate_run_absent adopt"
    ).split()
)
_RESCUE_STATE_ATTRIBUTES = frozenset(
    {
        "_bm25_rescue_attempts",
        "_bm25_rescue_zero_completeness",
        "_bm25_rescue_unavailable_or_empty",
        "_bm25_rescue_no_positive_uplift",
        "_bm25_rescue_constant_uplift",
        "_bm25_rescue_unchanged_order",
        "_bm25_rescue_successful_reorders",
        "_bm25_rescue_fallbacks",
    }
)
_FROZEN_PROFILE_SCORE_COUNTERS = frozenset(
    {"_constant_score_neutral_fallbacks"}
)
_FROZEN_AGENT_ATTRIBUTES = frozenset(
    {
        "dense_initialization_error",
        "_retriever",
        "_question_policy",
        "_fusion_policy",
        "_ranking_policy",
        "_profile_policy",
        "_slate_policy",
        "_intent_policy",
        "_orchestrator",
        "_reranking_attempts",
        "_reranking_successes",
        "_reranking_failures",
        "_reranking_unavailable_skips",
        "_bm25_rescue_attempts",
        "_bm25_rescue_zero_completeness",
        "_bm25_rescue_unavailable_or_empty",
        "_bm25_rescue_no_positive_uplift",
        "_bm25_rescue_constant_uplift",
        "_bm25_rescue_unchanged_order",
        "_bm25_rescue_successful_reorders",
        "_bm25_rescue_fallbacks",
        "_sessions",
        "_profile_priors",
        "_profiles_reset",
        "_zero_mask_profiles",
        "_nonzero_mask_profiles",
        "_recognized_theme_count",
        "_turns_disabled_by_active_requirements",
        "_eligible_stage_a_attempts",
        "_empty_represented_theme_fallbacks",
        "_constant_score_neutral_fallbacks",
        "_successful_residual_applications",
        "_parsing_or_scoring_fallbacks",
        "_slates",
        "_slate_attempts",
        "_slate_successes",
        "_slate_failures",
        "_slate_initializations",
        "_slate_ranking_resets",
        "_slate_stagnant_turns",
        "_slate_unseen_selected_on_stagnant",
        "_slate_repeat_backfills",
    }
)
_ROUTE_STATUSES = frozenset({"ok", "empty"})
_ORCHESTRATION_REASONS = frozenset(
    {
        "cold_cache",
        "exact_dependency_hit",
        "ranking_dependencies_changed",
        "backend_snapshot_changed",
        "cache_fault",
        "ranking_not_cache_eligible",
        "backend_snapshot_unavailable",
        "policy_requires_search",
        "empty_result_request",
    }
)
_SENSITIVE_STATE_TOKENS = frozenset(
    {"bm25", "rescue", "uplift", "uplifts", "score", "scores", "rank", "ranks"}
)
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _project_health(
    health: object,
    keys: Sequence[str],
    name: str,
    *,
    policy: RankingPolicy | None = None,
) -> dict:
    if not isinstance(health, dict) or set(health) != set(keys):
        raise RuntimeError(f"{name} schema drifted")
    projected = dict(health)
    if policy is not None and projected.get("policy") != policy.value:
        raise RuntimeError(f"{name} policy drifted")
    for key, value in projected.items():
        if key == "policy":
            if type(value) is not str or not value:
                raise RuntimeError(f"invalid {name} policy")
        elif key in {"bm25", "dense", "reasons"}:
            if not isinstance(value, dict) or any(
                type(label) is not str or not label or type(count) is not int or count < 0
                for label, count in value.items()
            ):
                raise RuntimeError(f"invalid {name} mapping: {key}")
            allowed = _ORCHESTRATION_REASONS if key == "reasons" else _ROUTE_STATUSES
            if not set(value).issubset(allowed):
                raise RuntimeError(f"unexpected {name} label: {key}")
            projected[key] = dict(sorted(value.items()))
        elif type(value) is not int or value < 0:
            raise RuntimeError(f"invalid {name} counter: {key}")
    return projected


def _project_rescue_health(
    health: object,
    *,
    expected_policy: RankingPolicy,
) -> dict[str, int | str]:
    projected = _project_health(
        health, RESCUE_HEALTH_KEYS, "rescue health", policy=expected_policy
    )
    attempts = int(projected["attempts"])
    if sum(int(projected[key]) for key in _RESCUE_OUTCOME_KEYS) != attempts:
        raise RuntimeError("rescue outcomes do not partition attempts")
    if expected_policy is STAGE_A_RANKING_POLICY and attempts:
        raise RuntimeError("Phase 9 comparator performed rescue work")
    return projected


def _canonical_private_cache_snapshot(
    agent: ConversationalSearchAgent,
) -> tuple[object, ...]:
    """Copy the complete bounded cache canonically for in-memory equality."""

    planner = getattr(agent, "_orchestrator", None)
    entries = getattr(planner, "_entries", None)
    capacity = getattr(planner, "_capacity", None)
    policy = getattr(planner, "policy", None)
    if type(entries) is not OrderedDict or type(capacity) is not int:
        raise RuntimeError("orchestration cache representation drifted")
    canonical_entries: list[tuple[object, ...]] = []
    expected_fields = ("dependency_digest", "backend_snapshot_token", "ranked_ids")
    for key, entry in entries.items():
        if type(key) is not bytes or len(key) != 32:
            raise RuntimeError("cache session key is not a SHA-256 digest")
        if type(entry) is not RankingCacheEntry:
            raise RuntimeError("cache entry type drifted")
        if tuple(field.name for field in fields(entry)) != expected_fields:
            raise RuntimeError("cache entry retained unexpected state")
        if type(entry.backend_snapshot_token) is not BackendSnapshotToken:
            raise RuntimeError("cache backend token type drifted")
        canonical_entries.append(
            (
                key.hex(),
                entry.dependency_digest.hex(),
                tuple(entry.ranked_ids),
                expected_fields,
            )
        )
    return (
        getattr(policy, "value", None),
        capacity,
        tuple(canonical_entries),
    )


def _inspect_retained_rescue_state(
    agent: ConversationalSearchAgent,
    *,
    expected_policy: RankingPolicy,
) -> bool:
    """Confirm that rescue persistence is counters and policy only."""

    try:
        attributes = vars(agent)
        if attributes.get("_ranking_policy") is not expected_policy:
            return False
        if set(attributes) != _FROZEN_AGENT_ATTRIBUTES:
            return False
        suspicious = set()
        for name in attributes:
            if name == "_retriever":
                continue
            tokens = set(name.strip("_").casefold().split("_"))
            if tokens & _SENSITIVE_STATE_TOKENS:
                suspicious.add(name)
        if suspicious != _RESCUE_STATE_ATTRIBUTES | _FROZEN_PROFILE_SCORE_COUNTERS:
            return False
        for name in _RESCUE_STATE_ATTRIBUTES:
            value = attributes.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return False
        _canonical_private_cache_snapshot(agent)
    except Exception:
        return False
    return True


def _validate_variant_accounting(
    diagnostics: dict,
    *,
    expected_policy: RankingPolicy,
) -> None:
    _validate_phase9_variant_accounting(
        diagnostics,
        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    ranking = diagnostics["ranking_health"]
    rescue = diagnostics["rescue_health"]
    searches = int(diagnostics["orchestration_health"]["searches"])
    if ranking["policy"] != expected_policy.value:
        raise RuntimeError("ranking policy accounting drifted")
    if rescue["policy"] != expected_policy.value:
        raise RuntimeError("rescue policy accounting drifted")
    if expected_policy is COMPLETENESS_BM25_RESCUE_RANKING_POLICY:
        if int(rescue["attempts"]) != int(ranking["attempts"]):
            raise RuntimeError("candidate rescue coverage is incomplete")
    elif any(int(rescue[key]) for key in RESCUE_HEALTH_KEYS[1:]):
        raise RuntimeError("Phase 9 comparator contains rescue activity")
    if int(ranking["attempts"]) != searches:
        raise RuntimeError("Stage-A attempts do not cover every search")


def _evaluate_with_deterministic_session_ids(
    audited: _AuditAgent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    """Run the unchanged evaluator with replayable opaque session identities."""

    identifiers = (uuid.UUID(int=index + 1) for index in range(len(samples)))
    with patch(
        "evaluator.local_evaluator.uuid.uuid4",
        side_effect=identifiers,
    ) as uuid4:
        result = evaluate(audited, samples, catalog_ids, categories, products)
    if uuid4.call_count != len(samples):
        raise RuntimeError("evaluator session isolation count drifted")
    return result


def _evaluate_agent(
    agent: ConversationalSearchAgent,
    guarded_backend: _CallAuditRetriever,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    *,
    expected_policy: RankingPolicy,
) -> tuple[
    dict,
    dict,
    list[tuple[object, object, object]],
    tuple[object, ...],
]:
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
    guarded_backend.validate(searches)
    latency = audited.latency_summary()
    if int(latency["count"]) != expected_turns:
        raise RuntimeError("response timing coverage is incomplete")

    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": _project_health(
            guarded_backend.summary(), ROUTE_HEALTH_KEYS, "route health"
        ),
        "ranking_health": _project_health(
            agent.ranking_health,
            RANKING_HEALTH_KEYS,
            "ranking health",
            policy=expected_policy,
        ),
        "rescue_health": _project_rescue_health(
            agent.rescue_health,
            expected_policy=expected_policy,
        ),
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "slate_health": _project_health(
            agent.slate_health, SLATE_HEALTH_KEYS, "slate health"
        ),
        "orchestration_health": _project_health(
            agent.orchestration_health,
            ORCHESTRATION_HEALTH_KEYS,
            "orchestration health",
        ),
        "retained_profile_state_valid": _inspect_retained_profile_state(
            agent,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_rescue_state_valid": _inspect_retained_rescue_state(
            agent,
            expected_policy=expected_policy,
        ),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }
    _validate_variant_accounting(
        diagnostics,
        expected_policy=expected_policy,
    )
    return (
        result,
        diagnostics,
        audited.action_trace,
        _canonical_private_cache_snapshot(agent),
    )


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    ranking_policy: RankingPolicy,
) -> tuple[dict, dict, list[tuple[object, object, object]], tuple[object, ...]]:
    guarded_backend = _CallAuditRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded_backend,
        ranking_policy=ranking_policy,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    return _evaluate_agent(
        agent,
        guarded_backend,
        samples,
        catalog_ids,
        categories,
        products,
        expected_policy=ranking_policy,
    )


def _run_independent(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[dict, dict, list[tuple[object, object, object]], tuple[object, ...]]:
    agent = Agent(catalog)
    backend = agent.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable for independent verification")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable for independent verification")
    guarded_backend = _CallAuditRetriever(backend)
    agent._retriever = guarded_backend  # type: ignore[attr-defined]
    return _evaluate_agent(
        agent,
        guarded_backend,
        samples,
        catalog_ids,
        categories,
        products,
        expected_policy=COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
    )


def _warm_backend(catalog: Path, backend: object) -> None:
    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        ranking_policy=STAGE_A_RANKING_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    session_id = "phase10-label-free-runtime-warmup"
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    health = warmup.ranking_health
    if int(health["attempts"]) != 1 or int(health["successes"]) != 1:
        raise RuntimeError("label-free runtime warm-up did not complete safely")
    if int(warmup.rescue_health["attempts"]) != 0:
        raise RuntimeError("label-free warm-up unexpectedly entered rescue")


def _validate_frozen_inputs(
    repository_root: Path,
    catalog: Path,
    dataset: Path,
) -> dict[str, str]:
    observed = {
        "catalog": _sha256(catalog),
        "dataset": _sha256(dataset),
        **{
            relative: _sha256(repository_root / relative)
            for relative in FROZEN_INPUT_SHA256
            if relative not in {"catalog", "dataset"}
        },
    }
    if observed != FROZEN_INPUT_SHA256:
        raise RuntimeError("frozen Phase 10 inputs drifted; refusing sealed run")
    return observed


def _validate_prelock_verification(verification: object) -> dict:
    expected_keys = {
        "focused_suite_command",
        "focused_tests_passed",
        "complete_suite_command",
        "complete_unit_tests_passed",
        "phase10_formula_oracle_cases",
        "phase7_exact_oracle_command",
        "phase7_exact_oracle_cases",
        "phase7_exact_oracle_sha256",
        "phase9_exact_oracle_command",
        "phase9_exact_oracle_cases",
        "phase9_exact_oracle_sha256",
        "phase10_phase9_exact_oracle_command",
        "phase10_phase9_exact_oracle_cases",
        "phase10_phase9_exact_oracle_sha256",
        "completed_before_lock",
    }
    if not isinstance(verification, dict) or set(verification) != expected_keys:
        raise ValueError("Phase 10 pre-lock verification schema is incomplete")
    expected_values = {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "complete_suite_command": FULL_SUITE_COMMAND,
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
    if any(verification.get(key) != value for key, value in expected_values.items()):
        raise ValueError("Phase 10 pre-lock verification evidence drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        value = verification.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("pre-lock test counts must be positive integers")
    return {key: verification[key] for key in expected_keys}


def _validate_implementation_lock(
    repository_root: Path,
    *,
    source_paths: Sequence[str] = SOURCE_PATHS,
    lock_relative: str = IMPLEMENTATION_LOCK_RELATIVE,
) -> dict:
    lock_path = repository_root / lock_relative
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "lock_id",
        "status",
        "contract_sha256",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 10 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 10 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 10 implementation lock identity")
    if lock.get("status") != "locked_before_public_confirmation":
        raise RuntimeError("Phase 10 implementation is not frozen")
    contract_sha256 = lock.get("contract_sha256")
    if (
        not isinstance(contract_sha256, str)
        or not _HEX_SHA256_RE.fullmatch(contract_sha256)
        or contract_sha256 != _sha256(repository_root / CONTRACT_RELATIVE)
    ):
        raise RuntimeError("Phase 10 experiment contract drifted after lock")

    paths = tuple(source_paths)
    if len(paths) != len(set(paths)):
        raise RuntimeError("Phase 10 source lock contains duplicate paths")
    expected_source = lock.get("source_sha256")
    if not isinstance(expected_source, dict) or set(expected_source) != set(paths):
        raise RuntimeError("Phase 10 implementation source lock is incomplete")
    if any(
        not isinstance(value, str) or not _HEX_SHA256_RE.fullmatch(value)
        for value in expected_source.values()
    ):
        raise RuntimeError("Phase 10 implementation source hash is malformed")
    observed_source = {
        relative: _sha256(repository_root / relative)
        for relative in sorted(paths)
    }
    if observed_source != expected_source:
        raise RuntimeError("Phase 10 implementation drifted after lock")
    try:
        _validate_prelock_verification(lock.get("verification"))
    except ValueError as error:
        raise RuntimeError("Phase 10 pre-lock verification drifted") from error
    return lock


def _claim_run_output(output: Path) -> None:
    sentinel = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "run_started_no_retry",
    }
    serialized = (json.dumps(sentinel, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(
            "Phase 10 run was already claimed; refusing a second run"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _phase9_summary_matches(result: dict) -> bool:
    summary = _overall_official_summary(result)
    return {key: summary.get(key) for key in PHASE9_OFFICIAL} == PHASE9_OFFICIAL


def _reported_token_usage_is_zero(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    usage = result.get("reported_token_usage")
    return (
        isinstance(usage, dict)
        and set(usage) == set(TOKEN_USAGE_KEYS)
        and all(type(usage[key]) is int and usage[key] == 0 for key in TOKEN_USAGE_KEYS)
    )


def _faults_are_zero(diagnostics: dict) -> bool:
    reasons = diagnostics["orchestration_health"]["reasons"]
    return (
        int(diagnostics["route_health"]["fallback_turns"]) == 0
        and int(diagnostics["ranking_health"]["failures"]) == 0
        and int(diagnostics["ranking_health"]["unavailable_skips"]) == 0
        and int(diagnostics["rescue_health"]["validation_or_scoring_fallbacks"])
        == 0
        and int(diagnostics["profile_health"]["parsing_or_scoring_fallbacks"])
        == 0
        and int(diagnostics["slate_health"]["failures"]) == 0
        and int(diagnostics["orchestration_health"]["fault_invalidations"]) == 0
        and int(diagnostics["orchestration_health"]["backend_invalidations"]) == 0
        and int(reasons.get("backend_snapshot_changed", 0)) == 0
        and int(reasons.get("backend_snapshot_unavailable", 0)) == 0
        and int(reasons.get("cache_fault", 0)) == 0
        and int(diagnostics["orchestration_health"]["store_rejections"]) == 0
    )


def _call_accounting(diagnostics: dict) -> dict[str, int]:
    route = diagnostics["route_health"]
    bm25 = _route_call_count(route)
    dense = sum(int(value) for value in route["dense"].values())
    documents = int(route["candidate_document_calls"])
    stage_a = int(diagnostics["ranking_health"]["attempts"])
    return {
        "bm25_route_calls": bm25,
        "dense_route_calls": dense,
        "candidate_document_calls": documents,
        "stage_a_attempts": stage_a,
        "total_counted_calls": bm25 + dense + documents + stage_a,
    }


def _no_extra_per_search_calls(diagnostics: dict) -> bool:
    searches = int(diagnostics["orchestration_health"]["searches"])
    calls = _call_accounting(diagnostics)
    return all(
        calls[key] == searches
        for key in CALL_ACCOUNTING_KEYS
        if key != "total_counted_calls"
    )


def _deterministic_health(diagnostics: dict) -> dict:
    return {
        key: diagnostics[key]
        for key in DIAGNOSTIC_KEYS
        if key not in {"evaluation_wall_seconds", "respond_latency_ms"}
    }


def _build_decision_gates(
    *,
    baseline_run: tuple,
    candidate_run: tuple,
    replay_run: tuple,
    independent_run: tuple,
    hit_to_miss_count: int,
    implementation_lock: dict,
    publication_privacy_valid: bool,
    implementation_lock_revalidated: bool,
) -> tuple[dict[str, bool], dict[str, float]]:
    baseline, baseline_diagnostics, _baseline_trace, _baseline_cache = baseline_run
    candidate, candidate_diagnostics, candidate_trace, candidate_cache = (
        candidate_run
    )
    replay, replay_diagnostics, replay_trace, replay_cache = replay_run
    independent, independent_diagnostics, independent_trace, independent_cache = (
        independent_run
    )
    latency = _latency_comparison(baseline_diagnostics, candidate_diagnostics)
    baseline_metrics = _overall_official_summary(baseline)
    candidate_metrics = _overall_official_summary(candidate)
    candidate_calls = _call_accounting(candidate_diagnostics)
    baseline_calls = _call_accounting(baseline_diagnostics)
    verification = implementation_lock["verification"]
    replay_exact = (
        replay == candidate
        and replay_trace == candidate_trace
        and replay_cache == candidate_cache
        and _deterministic_health(replay_diagnostics)
        == _deterministic_health(candidate_diagnostics)
    )
    independent_exact = (
        independent == candidate
        and independent_trace == candidate_trace
        and independent_cache == candidate_cache
        and _deterministic_health(independent_diagnostics)
        == _deterministic_health(candidate_diagnostics)
    )
    state_valid = all(
        diagnostics["retained_profile_state_valid"] is True
        and diagnostics["retained_rescue_state_valid"] is True
        for diagnostics in (
            baseline_diagnostics,
            candidate_diagnostics,
            replay_diagnostics,
            independent_diagnostics,
        )
    )
    prelock_valid = (
        verification["completed_before_lock"] is True
        and int(verification["focused_tests_passed"]) > 0
        and int(verification["complete_unit_tests_passed"]) > 0
        and verification["phase10_formula_oracle_cases"]
        == PHASE10_FORMULA_ORACLE_CASES
        and verification["phase7_exact_oracle_cases"] == PHASE7_ORACLE_CASES
        and verification["phase7_exact_oracle_sha256"] == PHASE7_ORACLE_SHA256
        and verification["phase9_exact_oracle_cases"] == PHASE9_ORACLE_CASES
        and verification["phase9_exact_oracle_sha256"] == PHASE9_ORACLE_SHA256
        and verification["phase10_phase9_exact_oracle_cases"]
        == PHASE10_PHASE9_ORACLE_CASES
        and verification["phase10_phase9_exact_oracle_sha256"]
        == PHASE10_PHASE9_ORACLE_SHA256
    )
    wall_ratio = _safe_ratio(
        float(candidate_diagnostics["evaluation_wall_seconds"]),
        float(baseline_diagnostics["evaluation_wall_seconds"]),
    )
    p95_ratio = _safe_ratio(
        float(candidate_diagnostics["respond_latency_ms"]["warm_p95"]),
        float(baseline_diagnostics["respond_latency_ms"]["warm_p95"]),
    )

    gates = {
        "phase9_comparator_metrics_exact": _phase9_summary_matches(baseline),
        "candidate_sample_count_is_200": (
            int(candidate_metrics["sample_count"]) == 200
        ),
        "candidate_hit_rate_at_10_at_least_0_99": (
            float(candidate_metrics["hit_rate_at_10"]) >= 0.99
        ),
        "candidate_mrr_strictly_above_0_529558": (
            float(candidate_metrics["mrr"]) > 0.529558
        ),
        "candidate_mttc_at_most_3_065": (
            float(candidate_metrics["mttc"]) <= 3.065
        ),
        "candidate_technical_score_strictly_above_0_812567": (
            float(candidate_metrics["recommended_technical_score"]) > 0.812567
        ),
        "phase9_hit_to_phase10_miss_count_is_zero": hit_to_miss_count == 0,
        "candidate_replay_payload_intent_slate_cache_and_health_exact": replay_exact,
        "independent_payload_intent_slate_cache_and_health_exact": independent_exact,
        "candidate_rescue_outcomes_partition_attempts": (
            int(candidate_diagnostics["rescue_health"]["attempts"])
            == sum(
                int(candidate_diagnostics["rescue_health"][key])
                for key in _RESCUE_OUTCOME_KEYS
            )
        ),
        "phase9_comparator_rescue_attempts_are_zero": (
            int(baseline_diagnostics["rescue_health"]["attempts"]) == 0
        ),
        "candidate_and_comparator_have_one_call_per_search": (
            _no_extra_per_search_calls(candidate_diagnostics)
            and _no_extra_per_search_calls(baseline_diagnostics)
        ),
        "candidate_total_counted_calls_not_above_comparator": (
            candidate_calls["total_counted_calls"]
            <= baseline_calls["total_counted_calls"]
        ),
        "candidate_and_comparator_report_zero_model_api_tokens": (
            _reported_token_usage_is_zero(candidate)
            and _reported_token_usage_is_zero(baseline)
        ),
        "candidate_and_comparator_faults_are_zero": (
            _faults_are_zero(candidate_diagnostics)
            and _faults_are_zero(baseline_diagnostics)
        ),
        "candidate_warm_p95_ratio_at_most_1_05": p95_ratio <= 1.05,
        "candidate_wall_time_ratio_at_most_1_05": wall_ratio <= 1.05,
        "synthetic_formula_phase7_phase9_and_complete_suites_passed": prelock_valid,
        "bounded_profile_and_rescue_retained_state_valid": state_valid,
        "aggregate_publication_privacy_valid": publication_privacy_valid,
        "source_contract_and_lock_validated_before_and_after_run": (
            implementation_lock_revalidated
        ),
        "additional_persistent_session_candidate_or_product_state_is_zero": (
            state_valid
            and PROFILE_THEME_MASK_BYTES == 2
            and all(
                set(_canonical_private_cache_shape(cache))
                == {"dependency_digest", "backend_snapshot_token", "ranked_ids"}
                for cache in (candidate_cache, replay_cache, independent_cache)
            )
        ),
        "post_run_tuning_repair_or_second_candidate_run_absent": True,
    }
    gates["adopt"] = all(gates.values())
    if set(gates) != set(DECISION_GATE_KEYS):
        raise RuntimeError("Phase 10 decision-gate schema drifted")
    return gates, latency


def _canonical_private_cache_shape(snapshot: tuple[object, ...]) -> tuple[str, ...]:
    if len(snapshot) != 3 or not isinstance(snapshot[2], tuple):
        return ()
    entries = snapshot[2]
    if not entries:
        return ("dependency_digest", "backend_snapshot_token", "ranked_ids")
    first = entries[0]
    if not isinstance(first, tuple) or len(first) != 4:
        return ()
    fields_value = first[3]
    return tuple(fields_value) if isinstance(fields_value, tuple) else ()


def _publication_privacy_is_valid(
    payload: object,
    *,
    allow_missing_decision_gate: bool = False,
) -> bool:
    if not isinstance(payload, dict):
        return False
    top = {
        "schema_version", "experiment_id", "candidate", "baseline",
        "run_configuration", "official_metrics", "paired_quality", "health",
        "call_accounting", "latency", "exactness", "privacy",
        "reproducibility", "decision_gate",
    }
    if allow_missing_decision_gate:
        top.remove("decision_gate")
    if set(payload) != top:
        return False
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("candidate") != CANDIDATE_ID
        or payload.get("baseline") != BASELINE_ID
    ):
        return False

    # Reuse the Phase 9 recursive sensitive-data and aggregate-metric guard.
    inherited = {key: value for key, value in payload.items() if key != "call_accounting"}
    if not _phase9_publication_privacy_is_valid(
        inherited,
        allow_missing_decision_gate=allow_missing_decision_gate,
    ):
        return False

    def exact(value: object, keys: Sequence[str] | set[str]) -> bool:
        return isinstance(value, dict) and set(value) == set(keys)

    official = payload["official_metrics"]
    if not exact(official, {"baseline", "candidate", "delta"}):
        return False
    summary_keys = {*OVERALL_METRIC_KEYS, "reported_token_usage"}
    for variant in ("baseline", "candidate"):
        summary = official.get(variant)
        if not exact(summary, summary_keys):
            return False
        if type(summary["sample_count"]) is not int or summary["sample_count"] < 0:
            return False
        if any(
            type(summary[key]) not in {int, float}
            or not math.isfinite(float(summary[key]))
            or summary[key] < 0
            for key in OVERALL_METRIC_KEYS[1:]
        ):
            return False
        usage = summary["reported_token_usage"]
        if not exact(usage, TOKEN_USAGE_KEYS) or any(
            type(usage[key]) is not int or usage[key] < 0
            for key in TOKEN_USAGE_KEYS
        ):
            return False
    delta = official.get("delta")
    if not exact(delta, OVERALL_METRIC_KEYS[1:]) or any(
        type(delta[key]) not in {int, float}
        or not math.isfinite(float(delta[key]))
        for key in OVERALL_METRIC_KEYS[1:]
    ):
        return False

    run = payload["run_configuration"]
    if not exact(run, RUN_CONFIGURATION_KEYS) or run.get("run_order") != [
        CANDIDATE_ID, BASELINE_ID, CANDIDATE_ID, "independent_starter_agent_default"
    ]:
        return False
    if (
        run.get("execution") != "strictly_sequential"
        or run.get("onnx_threads") != 1
        or run.get("shared_immutable_backend_for_ablation_variants") is not True
        or run.get("fresh_agent_state_per_variant") is not True
        or run.get("backend_warmup") != "one_fixed_unlabeled_request"
        or run.get("external_api_calls") != 0
        or run.get("post_run_tuning_or_second_candidate_run_allowed") is not False
    ):
        return False

    paired = payload["paired_quality"]
    if not exact(paired, {"phase9_hit_to_phase10_miss_count"}):
        return False
    paired_count = paired["phase9_hit_to_phase10_miss_count"]
    if type(paired_count) is not int or paired_count < 0:
        return False

    health = payload["health"]
    variants = {
        "phase9_baseline", "phase10_candidate", "phase10_replay",
        "independent_phase10",
    }
    nested = (
        ("route_health", ROUTE_HEALTH_KEYS),
        ("ranking_health", RANKING_HEALTH_KEYS),
        ("rescue_health", RESCUE_HEALTH_KEYS),
        ("profile_health", PROFILE_HEALTH_KEYS),
        ("slate_health", SLATE_HEALTH_KEYS),
        ("orchestration_health", ORCHESTRATION_HEALTH_KEYS),
        ("respond_latency_ms", LATENCY_KEYS),
    )
    if not exact(health, variants):
        return False
    if any(
        not exact(diagnostics, DIAGNOSTIC_KEYS)
        or any(not exact(diagnostics.get(key), keys) for key, keys in nested)
        for diagnostics in health.values()
    ):
        return False
    expected_ranking_policies = {
        "phase9_baseline": STAGE_A_RANKING_POLICY.value,
        "phase10_candidate": COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value,
        "phase10_replay": COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value,
        "independent_phase10": COMPLETENESS_BM25_RESCUE_RANKING_POLICY.value,
    }
    for variant, diagnostics in health.items():
        if (
            diagnostics["ranking_health"].get("policy")
            != expected_ranking_policies[variant]
            or diagnostics["rescue_health"].get("policy")
            != expected_ranking_policies[variant]
            or diagnostics["profile_health"].get("policy")
            != BOUNDED_RESIDUAL_PROFILE_POLICY.value
            or diagnostics["slate_health"].get("policy")
            != STAGNATION_AWARE_SLATE_POLICY.value
            or diagnostics["orchestration_health"].get("policy")
            != EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value
        ):
            return False

    calls = payload["call_accounting"]
    if not exact(calls, {"baseline", "candidate"}) or any(
        not exact(values, CALL_ACCOUNTING_KEYS)
        or any(type(count) is not int or count < 0 for count in values.values())
        for values in calls.values()
    ):
        return False
    latency = payload["latency"]
    if not exact(latency, LATENCY_COMPARISON_KEYS) or any(
        type(value) not in {int, float}
        or not math.isfinite(float(value))
        or value < 0
        for value in latency.values()
    ):
        return False
    for key, keys in (("exactness", EXACTNESS_KEYS), ("privacy", PRIVACY_KEYS)):
        values = payload[key]
        if not exact(values, keys) or any(type(value) is not bool for value in values.values()):
            return False
    if not all(payload["privacy"].values()):
        return False

    reproducibility = payload["reproducibility"]
    if not exact(reproducibility, REPRODUCIBILITY_KEYS):
        return False
    if (
        type(reproducibility.get("platform")) is not str
        or type(reproducibility.get("python")) is not str
        or reproducibility.get("implementation_lock_id") != IMPLEMENTATION_LOCK_ID
        or reproducibility.get("lock_revalidated_after_independent") is not True
    ):
        return False
    for key, paths in (
        ("frozen_input_sha256", FROZEN_INPUT_SHA256),
        ("source_sha256", SOURCE_PATHS),
    ):
        hashes = reproducibility.get(key)
        if not exact(hashes, set(paths)) or any(
            type(value) is not str or not _HEX_SHA256_RE.fullmatch(value)
            for value in hashes.values()
        ):
            return False
    if not _HEX_SHA256_RE.fullmatch(str(reproducibility.get("contract_sha256"))):
        return False
    try:
        _validate_prelock_verification(reproducibility.get("pre_lock_verification"))
    except ValueError:
        return False
    if not allow_missing_decision_gate:
        gates = payload["decision_gate"]
        if not exact(gates, DECISION_GATE_KEYS) or any(
            type(value) is not bool for value in gates.values()
        ):
            return False
    return True


def _validate_publication_privacy(payload: dict) -> None:
    if not _publication_privacy_is_valid(payload):
        raise RuntimeError("Phase 10 publication violates the aggregate-only allowlist")


def _validate_output(output: Path, catalog: Path, dataset: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    resolved = output.resolve()
    if resolved != (repository_root / RAW_RESULT_RELATIVE).resolve():
        raise ValueError("Phase 10 has one frozen raw-result output path")
    if resolved.exists():
        raise FileExistsError("Phase 10 raw result already exists; refusing a second run")
    if any(
        resolved.is_relative_to(repository_root / directory)
        for directory in ("docs", "benchmarks")
    ):
        raise ValueError("the runner cannot write append-only publication paths")
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        *(repository_root / relative for relative in SOURCE_PATHS),
        repository_root / IMPLEMENTATION_LOCK_RELATIVE,
    }
    if resolved in protected:
        raise ValueError("output must not overwrite an input, source, or lock file")


def run_bm25_rescue_ablations(
    catalog_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    output = Path(output_path).resolve()
    repository_root = Path(__file__).resolve().parents[1]

    # Lock/input/output checks all precede the exclusive claim and dataset load.
    implementation_lock = _validate_implementation_lock(repository_root)
    frozen_inputs = _validate_frozen_inputs(repository_root, catalog, dataset)
    _validate_output(output, catalog, dataset)
    _claim_run_output(output)
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        ranking_policy=STAGE_A_RANKING_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing Phase 10 run")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing Phase 10 run")
    _warm_backend(catalog, backend)

    candidate_run, baseline_run, replay_run = (
        _run_variant(
            catalog, samples, catalog_ids, categories, products, backend, policy
        )
        for policy in (
            COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
            STAGE_A_RANKING_POLICY,
            COMPLETENESS_BM25_RESCUE_RANKING_POLICY,
        )
    )
    candidate, candidate_diagnostics, candidate_trace, candidate_cache = candidate_run
    baseline, baseline_diagnostics, _baseline_trace, _baseline_cache = baseline_run
    if not _phase9_summary_matches(baseline):
        raise RuntimeError("Phase 9 comparator drifted; refusing Phase 10 comparison")
    replay, replay_diagnostics, replay_trace, replay_cache = replay_run
    independent_run = _run_independent(
        catalog, samples, catalog_ids, categories, products
    )
    independent, independent_diagnostics, independent_trace, independent_cache = (
        independent_run
    )

    revalidated_lock = _validate_implementation_lock(repository_root)
    if revalidated_lock != implementation_lock:
        raise RuntimeError("Phase 10 implementation lock changed during the sealed run")
    hit_to_miss_count = _paired_hit_to_miss_count(baseline, candidate)
    latency = _latency_comparison(baseline_diagnostics, candidate_diagnostics)
    payload = {
        "schema_version": SCHEMA_VERSION,
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
            "baseline": _overall_official_summary(baseline),
            "candidate": _overall_official_summary(candidate),
            "delta": _metric_deltas(baseline, candidate),
        },
        "paired_quality": {
            "phase9_hit_to_phase10_miss_count": hit_to_miss_count,
        },
        "health": {
            "phase9_baseline": baseline_diagnostics,
            "phase10_candidate": candidate_diagnostics,
            "phase10_replay": replay_diagnostics,
            "independent_phase10": independent_diagnostics,
        },
        "call_accounting": {
            "baseline": _call_accounting(baseline_diagnostics),
            "candidate": _call_accounting(candidate_diagnostics),
        },
        "latency": latency,
        "exactness": {
            "candidate_replay_evaluator_payload_equal": replay == candidate,
            "candidate_replay_action_intent_slate_trace_equal": (
                replay_trace == candidate_trace
            ),
            "candidate_replay_private_cache_snapshot_equal": (
                replay_cache == candidate_cache
            ),
            "candidate_replay_aggregate_health_equal": (
                _deterministic_health(replay_diagnostics)
                == _deterministic_health(candidate_diagnostics)
            ),
            "independent_evaluator_payload_equal": independent == candidate,
            "independent_action_intent_slate_trace_equal": (
                independent_trace == candidate_trace
            ),
            "independent_private_cache_snapshot_equal": (
                independent_cache == candidate_cache
            ),
            "independent_aggregate_health_equal": (
                _deterministic_health(independent_diagnostics)
                == _deterministic_health(candidate_diagnostics)
            ),
        },
        "privacy": {
            "queries_messages_profiles_and_tags_absent": True,
            "product_sample_session_and_turn_rows_absent": True,
            "targets_scenarios_ranks_scores_and_action_traces_absent": True,
            "private_cache_snapshots_absent": True,
            "rescue_telemetry_is_fixed_global_counters_only": True,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "frozen_input_sha256": frozen_inputs,
            "implementation_lock_id": implementation_lock["lock_id"],
            "contract_sha256": implementation_lock["contract_sha256"],
            "source_sha256": implementation_lock["source_sha256"],
            "pre_lock_verification": implementation_lock["verification"],
            "lock_revalidated_after_independent": True,
        },
    }
    publication_privacy_valid = _publication_privacy_is_valid(
        payload,
        allow_missing_decision_gate=True,
    )
    gates, validated_latency = _build_decision_gates(
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        replay_run=replay_run,
        independent_run=independent_run,
        hit_to_miss_count=hit_to_miss_count,
        implementation_lock=implementation_lock,
        publication_privacy_valid=publication_privacy_valid,
        implementation_lock_revalidated=True,
    )
    if validated_latency != latency:
        raise RuntimeError("latency projection drifted while building gates")
    payload["decision_gate"] = gates
    _validate_publication_privacy(payload)
    _write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the one sealed Phase 10 candidate versus Phase 9 confirmation"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output(output, catalog, dataset)
    run_bm25_rescue_ablations(catalog, dataset, output)


if __name__ == "__main__":
    main()
