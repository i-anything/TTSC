"""Versioned buying-only exposure and semantic-rescue diagnostics.

This runner publishes aggregate evidence only.  The target-disjoint suite is
activation-only, while the reused public suite is explicitly exploratory and
cannot promote or mutate the active starter.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    EvidenceExposurePolicy,
    EvidenceExposureStatus,
)
from conversational_search.intent import ROBUST_INTENT_POLICY
from conversational_search.orchestration import (
    EXACT_RANKING_CACHE_CAPABILITY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    PROTOCOL_EVIDENCE_CAPABILITY,
    SHARED_DENSE_TERMS_RESCUE_POLICY,
    SemanticLexicalRescuePolicy,
    SemanticLexicalRescueStatus,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.build_semantic_rescue_activation_suite import (
    validate_activation_suite,
)
from scripts.run_fusion_ablations import _sha256
from scripts.run_multislot_intent_ablations import (
    _canonical_json,
    _current_max_rss_bytes,
    _evaluate_with_deterministic_session_ids,
    _metric_deltas,
    _overall_summary,
    _paired_statistics,
)
from scripts.run_phase2_exact_evidence_ablations import (
    AggregateAuditAgent,
    RuntimeNetworkAudit,
    VariantRun,
    _common_action_equivalence,
    _performance,
    _scenario_report,
    select_smoke_indices,
)
from scripts.run_profile_ablations import (
    TOKEN_USAGE_KEYS,
    _project_profile_health,
)
from scripts.run_reranking_ablations import _expected_turns
from scripts.run_semantic_exposure_ablations import (
    FactorialAuditRetriever,
    _action_summary,
    _claim_output,
    _replace_claim,
    _retained_agent_bytes,
    _status_sum,
)


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase16b-buying-gated-semantic-exposure-factorial-v1"
REPORT_ID = "phase16b-buying-gated-semantic-exposure-public-20260830"
CONTRACT_RELATIVE = "docs/phase16b_semantic_exposure_contract.json"
IMPLEMENTATION_LOCK_RELATIVE = (
    "docs/phase16b_semantic_exposure_implementation_lock.json"
)
IMPLEMENTATION_LOCK_ID = "phase16b-semantic-exposure-implementation-v1"
ACTIVATION_LOCK_RELATIVE = (
    "docs/phase16b_semantic_rescue_activation_lock.json"
)
CATALOG_RELATIVE = "data/catalog.jsonl"
PUBLIC_RELATIVE = "data/public_set.jsonl"
ACTIVATION_RELATIVE = "benchmarks/phase16b_semantic_rescue_activation.jsonl"
STARTER_RELATIVE = "starter/agent.py"
EXPECTED_PUBLIC_CASES = 200
EXPECTED_ACTIVATION_CASES = 16

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
    "conversational_search/exact_evidence.py",
    "conversational_search/exposure.py",
    "conversational_search/exposure_policy.py",
    "conversational_search/intent.py",
    "conversational_search/orchestration.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/slates.py",
    "conversational_search/strategy.py",
    "evaluator/local_evaluator.py",
    "scripts/build_semantic_rescue_activation_suite.py",
    "scripts/run_semantic_exposure_ablations.py",
    "scripts/run_semantic_exposure_v2_ablations.py",
    "starter/agent.py",
    "tests/test_evidence_exposure.py",
    "tests/test_semantic_exposure_factorial.py",
    "tests/test_semantic_exposure_v2_ablations.py",
    "tests/test_semantic_lexical_rescue.py",
    "tests/test_semantic_rescue_activation_suite.py",
)

_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
_FORBIDDEN_RESULT_KEYS = (
    '"sessions"',
    '"sample_id"',
    '"ground_truth"',
    '"parent_asin"',
    '"user_message"',
    '"user_profile"',
    '"recommendations"',
    '"per_session"',
    '"rows"',
    '"actions"',
)
_SEMANTIC_TECHNICAL_FAULTS = (
    SemanticLexicalRescueStatus.BM25_UNAVAILABLE,
    SemanticLexicalRescueStatus.DENSE_UNAVAILABLE,
    SemanticLexicalRescueStatus.DENSE_ERROR,
    SemanticLexicalRescueStatus.TERM_EXTRACTION_ERROR,
    SemanticLexicalRescueStatus.RETRY_ERROR,
)


@dataclass(frozen=True, slots=True)
class ArmConfig:
    arm_id: str
    semantic_policy: SemanticLexicalRescuePolicy
    exposure_policy: EvidenceExposurePolicy

    @property
    def rescue_enabled(self) -> bool:
        return self.semantic_policy is SHARED_DENSE_TERMS_RESCUE_POLICY

    @property
    def gate_enabled(self) -> bool:
        return self.exposure_policy is not DISABLED_EVIDENCE_EXPOSURE_POLICY

    def public_contract(self) -> dict[str, str]:
        return {
            "id": self.arm_id,
            "semantic_lexical_rescue": self.semantic_policy.value,
            "evidence_exposure": self.exposure_policy.value,
        }


ARM_CONFIGS = (
    ArmConfig(
        "baseline",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        DISABLED_EVIDENCE_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "rescue_only",
        SHARED_DENSE_TERMS_RESCUE_POLICY,
        DISABLED_EVIDENCE_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "buying_gate_only",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "combined_v2",
        SHARED_DENSE_TERMS_RESCUE_POLICY,
        BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    ),
)
ARM_ORDER = tuple(config.arm_id for config in ARM_CONFIGS)


@dataclass(slots=True)
class ActivationRun:
    diagnostics: dict[str, object]
    behavior_digest: str
    actions: tuple[tuple[int, int, str | None, int], ...]
    reported_token_usage: dict[str, int]


def _fixed_architecture_contract() -> dict[str, str]:
    return {
        "question_policy": CONSERVATIVE_EARLY_OTHER_POLICY.name,
        "fusion_policy": COMPLETENESS_ADAPTIVE_RRF_POLICY.value,
        "ranking_policy": LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY.value,
        "profile_policy": BOUNDED_RESIDUAL_PROFILE_POLICY.value,
        "slate_policy": INTENT_EPOCH_NOVELTY_SLATE_POLICY.value,
        "intent_policy": ROBUST_INTENT_POLICY.value,
        "decision_policy": PROTECTED_DECISION_POLICY.value,
        "requirement_probe_policy": DISABLED_REQUIREMENT_PROBE_POLICY.value,
        "orchestration_policy": EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value,
    }


def _load_contract(repository_root: Path = REPOSITORY_ROOT) -> dict:
    contract = json.loads(
        (repository_root / CONTRACT_RELATIVE).read_text(encoding="utf-8")
    )
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("experiment_id") != EXPERIMENT_ID
        or contract.get("status") != "frozen_exploratory_diagnostic_only"
    ):
        raise RuntimeError("Phase 16b contract identity or status drifted")
    if contract.get("arms") != [
        config.public_contract() for config in ARM_CONFIGS
    ]:
        raise RuntimeError("Phase 16b arm definitions drifted")
    if contract.get("fixed_shared_architecture") != _fixed_architecture_contract():
        raise RuntimeError("Phase 16b shared architecture drifted")
    execution = contract.get("execution")
    if not isinstance(execution, dict) or execution.get(
        "public_arm_order"
    ) != list(ARM_ORDER):
        raise RuntimeError("Phase 16b execution order drifted")
    authority = contract.get("promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise RuntimeError("Phase 16b acquired promotion authority")
    return contract


def _validate_hashes(
    repository_root: Path,
    hashes: Mapping[str, object],
) -> None:
    if not hashes:
        raise RuntimeError("Phase 16b hash lock is empty")
    for relative, expected in hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _HEX_SHA256_RE.fullmatch(expected) is None
        ):
            raise RuntimeError("Phase 16b hash lock is malformed")
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Phase 16b locked path drifted: {relative}")


def _validate_activation_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    lock = json.loads(
        (repository_root / ACTIVATION_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    suite = lock.get("suite") if isinstance(lock, dict) else None
    inputs = lock.get("inputs") if isinstance(lock, dict) else None
    selection = lock.get("selection") if isinstance(lock, dict) else None
    if (
        lock.get("schema_version") != 1
        or lock.get("suite_id") != "phase16b-semantic-rescue-activation-v1"
        or lock.get("status") != "locked_before_evaluation"
        or not isinstance(suite, dict)
        or not isinstance(inputs, dict)
        or not isinstance(selection, dict)
        or suite.get("path") != ACTIVATION_RELATIVE
        or suite.get("case_count") != EXPECTED_ACTIVATION_CASES
        or suite.get("unique_target_count") != EXPECTED_ACTIVATION_CASES
        or suite.get("public_target_overlap") != 0
        or selection.get("candidate_conditioned") is not True
        or selection.get("quality_or_promotion_claim_allowed") is not False
    ):
        raise RuntimeError("Phase 16b activation lock drifted")
    _validate_hashes(
        repository_root,
        {
            str(suite["path"]): suite["sha256"],
            str(inputs["catalog_path"]): inputs["catalog_sha256"],
            str(inputs["public_path"]): inputs["public_sha256"],
        },
    )
    return lock


def _validate_implementation_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    contract = _load_contract(repository_root)
    _validate_activation_lock(repository_root)
    lock = json.loads(
        (repository_root / IMPLEMENTATION_LOCK_RELATIVE).read_text(
            encoding="utf-8"
        )
    )
    expected_keys = {
        "schema_version",
        "lock_id",
        "experiment_id",
        "status",
        "contract_sha256",
        "activation_lock_sha256",
        "input_sha256",
        "arm_order",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 16b implementation lock schema drifted")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("lock_id") != IMPLEMENTATION_LOCK_ID
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("status") != "locked_before_diagnostic_execution"
        or lock.get("arm_order") != list(ARM_ORDER)
        or lock.get("contract_sha256")
        != _sha256(repository_root / CONTRACT_RELATIVE)
        or lock.get("activation_lock_sha256")
        != _sha256(repository_root / ACTIVATION_LOCK_RELATIVE)
    ):
        raise RuntimeError("Phase 16b implementation lock identity drifted")

    frozen = contract.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise RuntimeError("Phase 16b frozen inputs are unavailable")
    expected_inputs: dict[str, str] = {}
    for name in ("catalog", "public_diagnostic", "activation_suite", "starter"):
        item = frozen.get(name)
        if not isinstance(item, dict):
            raise RuntimeError("Phase 16b frozen input entry is invalid")
        expected_inputs[str(item["path"])] = str(item["sha256"])
    if lock.get("input_sha256") != expected_inputs:
        raise RuntimeError("Phase 16b input lock disagrees with contract")
    _validate_hashes(repository_root, expected_inputs)

    sources = lock.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 16b source lock is incomplete")
    _validate_hashes(repository_root, sources)
    verification = lock.get("verification")
    if not isinstance(verification, dict) or set(verification) != {
        "focused_test_command",
        "focused_tests_passed",
        "complete_test_command",
        "complete_tests_passed",
        "phase13_oracle_command",
        "phase13_oracle_cases",
        "phase13_oracle_sha256",
        "completed_before_lock",
    }:
        raise RuntimeError("Phase 16b verification schema drifted")
    if verification.get("completed_before_lock") is not True:
        raise RuntimeError("Phase 16b verification was not completed before lock")
    for key in (
        "focused_tests_passed",
        "complete_tests_passed",
        "phase13_oracle_cases",
    ):
        if type(verification.get(key)) is not int or int(verification[key]) <= 0:
            raise RuntimeError("Phase 16b verification count is invalid")
    return lock


def _validate_environment() -> None:
    mismatches = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Phase 16b environment is not pinned: {mismatches}")


def _new_agent(
    catalog: Path,
    backend: object,
    config: ArmConfig,
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
        decision_policy=PROTECTED_DECISION_POLICY,
        requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
        semantic_lexical_rescue_policy=config.semantic_policy,
        evidence_exposure_policy=config.exposure_policy,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )


def _validate_accounting(
    diagnostics: Mapping[str, object],
    config: ArmConfig,
) -> None:
    turns = int(diagnostics["expected_turns"])
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    exact = diagnostics["exact_evidence_health"]
    semantic = diagnostics["semantic_lexical_rescue_health"]
    exposure = diagnostics["evidence_exposure_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
    protocol = diagnostics["protocol_decision_health"]
    if not all(
        isinstance(value, dict)
        for value in (
            route,
            ranking,
            exact,
            semantic,
            exposure,
            slate,
            orchestration,
            protocol,
        )
    ):
        raise RuntimeError("Phase 16b diagnostic schema is invalid")
    searches = int(orchestration["searches"])
    if (
        int(orchestration["decisions"]) != turns
        or searches + int(orchestration["reuses"]) != turns
        or int(orchestration["skips"])
        or sum(int(value) for value in route["bm25"].values()) != searches
        or int(route["candidate_document_calls"]) != searches
        or int(ranking["attempts"]) != searches
        or int(ranking["successes"]) != searches
        or int(ranking["failures"])
        or int(ranking["unavailable_skips"])
    ):
        raise RuntimeError("Phase 16b route/ranking accounting drifted")

    exact_attempts = int(exact["attempts"])
    expected_exact = turns if config.gate_enabled else searches
    if (
        exact_attempts != expected_exact
        or int(route["candidate_protocol_evidence_calls"]) != exact_attempts
        or sum(
            int(exact[key])
            for key in (
                "applied",
                "zero_support_fail_open",
                "capability_unavailable",
                "evidence_errors",
                "validation_errors",
            )
        )
        != exact_attempts
    ):
        raise RuntimeError("Phase 16b exact-evidence accounting drifted")

    semantic_attempts = int(semantic["attempts"])
    semantic_outcomes = _status_sum(semantic, tuple(SemanticLexicalRescueStatus))
    if config.rescue_enabled:
        if (
            semantic["policy"] != config.semantic_policy.value
            or semantic_attempts != searches
            or semantic_outcomes != semantic_attempts
        ):
            raise RuntimeError("Phase 16b semantic accounting drifted")
    elif semantic_attempts or semantic_outcomes:
        raise RuntimeError("disabled Phase 16b semantic policy performed work")

    exposure_attempts = int(exposure["attempts"])
    exposure_outcomes = _status_sum(exposure, tuple(EvidenceExposureStatus))
    if config.gate_enabled:
        if (
            exposure["policy"] != config.exposure_policy.value
            or exposure_attempts != turns
            or exposure_outcomes != exposure_attempts
            or int(exposure["withheld_turns"])
            != int(exposure[EvidenceExposureStatus.QUESTION_WITHHELD.value])
        ):
            raise RuntimeError("Phase 16b exposure accounting drifted")
    elif exposure_attempts or exposure_outcomes or int(exposure["withheld_turns"]):
        raise RuntimeError("disabled Phase 16b exposure policy performed work")

    expected_slates = turns - int(exposure["withheld_turns"])
    if (
        int(slate["attempts"]) != expected_slates
        or int(slate["successes"]) != expected_slates
        or int(slate["failures"])
        or any(
            int(protocol[key])
            for key in (
                "turns",
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
    ):
        raise RuntimeError("Phase 16b slate/protocol accounting drifted")


def _diagnostics(
    agent: ConversationalSearchAgent,
    guarded: FactorialAuditRetriever,
    audited: AggregateAuditAgent,
    network: RuntimeNetworkAudit,
    *,
    expected_turns: int,
    sample_count: int,
    wall_seconds: float,
) -> dict[str, object]:
    return {
        "expected_turns": expected_turns,
        "route_health": guarded.summary(),
        "ranking_health": agent.ranking_health,
        "exact_evidence_health": agent.exact_evidence_health,
        "semantic_lexical_rescue_health": agent.semantic_lexical_rescue_health,
        "evidence_exposure_health": agent.evidence_exposure_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=sample_count,
        ),
        "slate_health": agent.slate_health,
        "intent_epoch_slate_health": agent.intent_epoch_slate_health,
        "orchestration_health": agent.orchestration_health,
        "protocol_decision_health": agent.protocol_decision_health,
        "response_audit": {
            "response_exceptions": audited.response_exceptions,
            "invalid_api_responses": audited.invalid_api_responses,
        },
        "runtime_network_attempts": network.attempts,
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": audited.latency_summary(),
        "peak_rss_bytes": _current_max_rss_bytes(),
    }


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    config: ArmConfig,
) -> VariantRun:
    guarded = FactorialAuditRetriever(backend)
    agent = _new_agent(catalog, guarded, config)
    audited = AggregateAuditAgent(agent, catalog_ids)
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
    guarded.validate(
        expected_searches=searches,
        rescue_enabled=config.rescue_enabled,
    )
    diagnostics = _diagnostics(
        agent,
        guarded,
        audited,
        network,
        expected_turns=expected_turns,
        sample_count=int(result["sample_count"]),
        wall_seconds=wall_seconds,
    )
    if int(diagnostics["respond_latency_ms"]["count"]) != expected_turns:
        raise RuntimeError("Phase 16b response timing coverage is incomplete")
    _validate_accounting(diagnostics, config)
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("Phase 16b evaluator sessions are unavailable")
    return VariantRun(
        summary=_overall_summary(result),
        sessions=sessions,
        diagnostics=diagnostics,
        evaluator_digest=hashlib.sha256(_canonical_json(result)).hexdigest(),
        behavior_digest=audited.behavior_digest,
        actions=audited.actions,
    )


def _run_public_pass(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
) -> dict[str, VariantRun]:
    runs: dict[str, VariantRun] = {}
    for config in ARM_CONFIGS:
        runs[config.arm_id] = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            config,
        )
        gc.collect()
    return runs


def _technical_faults_are_zero(run: VariantRun | ActivationRun) -> bool:
    diagnostics = run.diagnostics
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    exact = diagnostics["exact_evidence_health"]
    semantic = diagnostics["semantic_lexical_rescue_health"]
    exposure = diagnostics["evidence_exposure_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
    response = diagnostics["response_audit"]
    return bool(
        not int(route["fallback_turns"])
        and not int(ranking["failures"])
        and not int(ranking["unavailable_skips"])
        and not int(exact["capability_unavailable"])
        and not int(exact["evidence_errors"])
        and not int(exact["validation_errors"])
        and not _status_sum(semantic, _SEMANTIC_TECHNICAL_FAULTS)
        and not int(semantic["validation_or_execution_fallbacks"])
        and not int(exposure["validation_fallbacks"])
        and not int(slate["failures"])
        and not int(orchestration["fault_invalidations"])
        and not int(orchestration["store_rejections"])
        and not int(response["response_exceptions"])
        and not int(response["invalid_api_responses"])
        and not int(diagnostics["runtime_network_attempts"])
    )


def _tokens_are_zero(summary: Mapping[str, object]) -> bool:
    usage = summary.get("reported_token_usage")
    return isinstance(usage, dict) and all(
        type(usage.get(key)) is int and int(usage[key]) == 0
        for key in TOKEN_USAGE_KEYS
    )


def _variant_public(run: VariantRun) -> dict[str, object]:
    diagnostics = run.diagnostics
    exposure = diagnostics["evidence_exposure_health"]
    return {
        "official_metrics": run.summary,
        "route_health": diagnostics["route_health"],
        "ranking_health": diagnostics["ranking_health"],
        "exact_evidence_health": diagnostics["exact_evidence_health"],
        "semantic_lexical_rescue_health": diagnostics[
            "semantic_lexical_rescue_health"
        ],
        "evidence_exposure_health": exposure,
        "safe_exposure_fail_open_total": (
            int(exposure[EvidenceExposureStatus.RETRIEVAL_FAIL_OPEN.value])
            + int(exposure[EvidenceExposureStatus.EVIDENCE_FAIL_OPEN.value])
        ),
        "orchestration_health": diagnostics["orchestration_health"],
        "slate_health": diagnostics["slate_health"],
        "response_audit": diagnostics["response_audit"],
        "runtime_network_attempts": diagnostics["runtime_network_attempts"],
        "retained_agent_bytes": diagnostics["retained_agent_bytes"],
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
        "peak_rss_bytes": diagnostics["peak_rss_bytes"],
        "action_summary": _action_summary(run),
        "technical_fault_free": _technical_faults_are_zero(run),
    }


def _diagnostic_observations(
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    transitions = paired["transitions"]
    observations = {
        "no_baseline_only_hits": int(transitions["baseline_only_hit"]) == 0,
        "hit_rate_not_below_baseline": (
            float(candidate.summary["hit_rate_at_10"])
            >= float(baseline.summary["hit_rate_at_10"])
        ),
        "mrr_not_below_baseline": (
            float(candidate.summary["mrr"])
            >= float(baseline.summary["mrr"])
        ),
        "mttc_not_above_baseline": (
            float(candidate.summary["mttc"])
            <= float(baseline.summary["mttc"])
        ),
        "technical_score_not_below_baseline": (
            float(candidate.summary["recommended_technical_score"])
            >= float(baseline.summary["recommended_technical_score"])
        ),
        "technical_fault_free": _technical_faults_are_zero(candidate),
    }
    observations["all_descriptive_nonregression_observations_hold"] = all(
        observations.values()
    )
    return observations


def _pass_report(runs: Mapping[str, VariantRun]) -> dict[str, object]:
    if tuple(runs) != ARM_ORDER:
        raise RuntimeError("Phase 16b public pass order drifted")
    baseline = runs["baseline"]
    comparisons: dict[str, object] = {}
    for arm_id in ARM_ORDER[1:]:
        candidate = runs[arm_id]
        paired = _paired_statistics(baseline.sessions, candidate.sessions)
        comparisons[arm_id] = {
            "metric_delta_vs_baseline": _metric_deltas(
                baseline.summary,
                candidate.summary,
            ),
            "paired_quality": paired,
            "scenario_subsets": _scenario_report(
                baseline.sessions,
                candidate.sessions,
            ),
            "question_and_width_comparison": _common_action_equivalence(
                baseline,
                candidate,
            ),
            "performance": _performance(baseline, candidate),
            "diagnostic_observations": _diagnostic_observations(
                baseline,
                candidate,
                paired,
            ),
        }
    return {
        "arms": {
            arm_id: _variant_public(runs[arm_id]) for arm_id in ARM_ORDER
        },
        "comparisons": comparisons,
    }


def _deterministic_health(
    run: VariantRun | ActivationRun,
) -> dict[str, object]:
    return {
        key: run.diagnostics[key]
        for key in (
            "expected_turns",
            "route_health",
            "ranking_health",
            "exact_evidence_health",
            "semantic_lexical_rescue_health",
            "evidence_exposure_health",
            "profile_health",
            "slate_health",
            "intent_epoch_slate_health",
            "orchestration_health",
            "protocol_decision_health",
            "response_audit",
            "runtime_network_attempts",
            "retained_agent_bytes",
        )
    }


def _public_smoke_gate(
    primary: Mapping[str, VariantRun],
    replay: Mapping[str, VariantRun],
) -> tuple[dict[str, object], bool]:
    exactness: dict[str, dict[str, bool]] = {}
    for arm_id in ARM_ORDER:
        first = primary[arm_id]
        second = replay[arm_id]
        exactness[arm_id] = {
            "evaluator_payload_equal": (
                first.evaluator_digest == second.evaluator_digest
            ),
            "response_behavior_equal": (
                first.behavior_digest == second.behavior_digest
            ),
            "question_and_width_trace_equal": first.actions == second.actions,
            "deterministic_health_equal": (
                _deterministic_health(first) == _deterministic_health(second)
            ),
        }
    gate = {
        "all_primary_arms_technically_safe": all(
            _technical_faults_are_zero(primary[arm_id]) for arm_id in ARM_ORDER
        ),
        "all_replay_arms_technically_safe": all(
            _technical_faults_are_zero(replay[arm_id]) for arm_id in ARM_ORDER
        ),
        "all_arms_zero_token": all(
            _tokens_are_zero(run.summary)
            for collection in (primary, replay)
            for run in collection.values()
        ),
        "all_arms_exactly_replay": all(
            all(values.values()) for values in exactness.values()
        ),
        "metric_direction_used_as_gate": False,
    }
    gate["authorize_full_public_diagnostic"] = all(
        value for key, value in gate.items() if key != "metric_direction_used_as_gate"
    )
    return {"arm_exactness": exactness, "technical_safety_gate": gate}, bool(
        gate["authorize_full_public_diagnostic"]
    )


def _activation_message(row: Mapping[str, object]) -> str:
    card = row.get("intent_card")
    if not isinstance(card, dict):
        raise RuntimeError("activation card is unavailable")
    category = card.get("target_category")
    constraints = card.get("hard_constraints")
    if (
        not isinstance(category, str)
        or not isinstance(constraints, list)
        or len(constraints) != 1
        or not isinstance(constraints[0], str)
    ):
        raise RuntimeError("activation card schema drifted")
    return (
        f"I'm looking for {category}. "
        f"A key requirement is: {constraints[0]}."
    )


def _run_activation(
    catalog: Path,
    activation_rows: list[dict],
    catalog_ids: set[str],
    backend: object,
) -> ActivationRun:
    config = ARM_CONFIGS[1]
    guarded = FactorialAuditRetriever(backend)
    agent = _new_agent(catalog, guarded, config)
    audited = AggregateAuditAgent(agent, catalog_ids)
    network = RuntimeNetworkAudit()
    usage = {key: 0 for key in TOKEN_USAGE_KEYS}
    started = time.perf_counter()
    with network.deny():
        for index, row in enumerate(activation_rows):
            session_id = f"phase16b_activation_{index + 1:03d}"
            audited.reset(session_id, {})
            response = audited.respond(
                session_id,
                _activation_message(row),
                1,
                10,
            )
            response_usage = response.get("usage")
            if not isinstance(response_usage, dict):
                raise RuntimeError("activation response usage is unavailable")
            for key in ("prompt_tokens", "completion_tokens"):
                value = response_usage.get(key)
                if type(value) is not int or value < 0:
                    raise RuntimeError("activation token usage is invalid")
                usage[key] += value
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    wall_seconds = time.perf_counter() - started
    turns = len(activation_rows)
    searches = int(agent.orchestration_health["searches"])
    guarded.validate(expected_searches=searches, rescue_enabled=True)
    diagnostics = _diagnostics(
        agent,
        guarded,
        audited,
        network,
        expected_turns=turns,
        sample_count=turns,
        wall_seconds=wall_seconds,
    )
    _validate_accounting(diagnostics, config)
    semantic = diagnostics["semantic_lexical_rescue_health"]
    route = diagnostics["route_health"]
    if (
        searches != turns
        or int(semantic[SemanticLexicalRescueStatus.APPLIED.value]) != turns
        or int(route["bm25_retry_total"]) != turns
        or not turns
        <= int(route["safe_expansion_term_total"])
        <= 3 * turns
    ):
        raise RuntimeError("activation suite did not exercise every rescue")
    return ActivationRun(
        diagnostics=diagnostics,
        behavior_digest=audited.behavior_digest,
        actions=audited.actions,
        reported_token_usage=usage,
    )


def _activation_report(
    primary: ActivationRun,
    replay: ActivationRun,
) -> tuple[dict[str, object], bool]:
    exactness = {
        "response_behavior_equal": (
            primary.behavior_digest == replay.behavior_digest
        ),
        "question_and_width_trace_equal": primary.actions == replay.actions,
        "deterministic_health_equal": (
            _deterministic_health(primary) == _deterministic_health(replay)
        ),
    }
    gate = {
        "all_cases_applied_in_primary": (
            int(
                primary.diagnostics["semantic_lexical_rescue_health"][
                    SemanticLexicalRescueStatus.APPLIED.value
                ]
            )
            == EXPECTED_ACTIVATION_CASES
        ),
        "all_cases_applied_in_replay": (
            int(
                replay.diagnostics["semantic_lexical_rescue_health"][
                    SemanticLexicalRescueStatus.APPLIED.value
                ]
            )
            == EXPECTED_ACTIVATION_CASES
        ),
        "primary_technically_safe": _technical_faults_are_zero(primary),
        "replay_technically_safe": _technical_faults_are_zero(replay),
        "zero_token": all(
            value == 0
            for run in (primary, replay)
            for value in run.reported_token_usage.values()
        ),
        "exact_replay": all(exactness.values()),
    }
    gate["authorize_public_smoke"] = all(gate.values())
    report = {
        "classification": "candidate_conditioned_activation_only",
        "quality_or_promotion_claim_allowed": False,
        "case_count": EXPECTED_ACTIVATION_CASES,
        "primary": {
            "route_health": primary.diagnostics["route_health"],
            "semantic_lexical_rescue_health": primary.diagnostics[
                "semantic_lexical_rescue_health"
            ],
            "response_audit": primary.diagnostics["response_audit"],
            "runtime_network_attempts": primary.diagnostics[
                "runtime_network_attempts"
            ],
            "reported_token_usage": primary.reported_token_usage,
        },
        "exactness": exactness,
        "gate": gate,
    }
    return report, bool(gate["authorize_public_smoke"])


def _warm_backend(catalog: Path, backend: object) -> dict[str, object]:
    started = time.perf_counter()
    warmup = _new_agent(catalog, backend, ARM_CONFIGS[0])
    session_id = "phase16b-label-free-warmup"
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    vocabulary = getattr(backend, "_ensure_bm25_vocabulary", None)
    if (
        int(warmup.ranking_health["successes"]) != 1
        or not callable(vocabulary)
        or vocabulary() is not True
    ):
        raise RuntimeError("Phase 16b label-free backend warm-up failed")
    return {
        "label_free": True,
        "dense_bm25_and_catalog_frequency_cache_warmed": True,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def _decision(full: Mapping[str, object]) -> dict[str, object]:
    comparisons = full.get("comparisons")
    if not isinstance(comparisons, dict):
        raise RuntimeError("Phase 16b full comparisons are unavailable")
    signals: dict[str, object] = {}
    for component, arm_id in {
        "semantic_to_lexical_rescue": "rescue_only",
        "buying_only_evidence_gate": "buying_gate_only",
        "combined_v2": "combined_v2",
    }.items():
        observations = comparisons[arm_id]["diagnostic_observations"]
        signals[component] = {
            "arm": arm_id,
            "descriptive_nonregression": observations[
                "all_descriptive_nonregression_observations_hold"
            ],
            "may_change_starter": False,
        }
    return {
        "status": "exploratory_diagnostic_not_promotable",
        "automatic_adoption": False,
        "reason": (
            "The only quality suite is previously used public data, and the "
            "fresh suite was deliberately selected on rescue activation."
        ),
        "components": signals,
    }


def run_phase16b(
    catalog_path: str | Path,
    public_path: str | Path,
    activation_path: str | Path,
    *,
    smoke_only: bool = False,
) -> dict[str, object]:
    _validate_environment()
    contract = _load_contract()
    lock = _validate_implementation_lock()
    catalog = Path(catalog_path).resolve()
    public = Path(public_path).resolve()
    activation = Path(activation_path).resolve()
    expected_paths = {
        catalog: (REPOSITORY_ROOT / CATALOG_RELATIVE).resolve(),
        public: (REPOSITORY_ROOT / PUBLIC_RELATIVE).resolve(),
        activation: (REPOSITORY_ROOT / ACTIVATION_RELATIVE).resolve(),
    }
    if any(observed != expected for observed, expected in expected_paths.items()):
        raise RuntimeError("Phase 16b requires every frozen input path")

    public_rows = load_jsonl(public)
    activation_rows = load_jsonl(activation)
    if len(public_rows) != EXPECTED_PUBLIC_CASES:
        raise RuntimeError("Phase 16b public case count drifted")
    if len(activation_rows) != EXPECTED_ACTIVATION_CASES:
        raise RuntimeError("Phase 16b activation case count drifted")
    public_targets = frozenset(
        str(row["ground_truth"]["parent_asin"]) for row in public_rows
    )
    validate_activation_suite(
        activation_rows,
        public_targets=public_targets,
    )
    catalog_ids, categories, products = catalog_index(catalog)

    rss_before = _current_max_rss_bytes()
    backend_started = time.perf_counter()
    bootstrap = ConversationalSearchAgent(
        catalog,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    backend = bootstrap.retrieval_backend
    backend_seconds = time.perf_counter() - backend_started
    if (
        not getattr(backend, "bm25_available", False)
        or not getattr(backend, "dense_available", False)
        or getattr(backend, "ranking_cache_capability", None)
        is not EXACT_RANKING_CACHE_CAPABILITY
        or getattr(backend, "protocol_evidence_capability", None)
        is not PROTOCOL_EVIDENCE_CAPABILITY
    ):
        raise RuntimeError("Phase 16b required backend capabilities are unavailable")
    rss_after = _current_max_rss_bytes()
    del bootstrap
    warmup = _warm_backend(catalog, backend)

    activation_primary = _run_activation(
        catalog,
        activation_rows,
        catalog_ids,
        backend,
    )
    activation_replay = _run_activation(
        catalog,
        activation_rows,
        catalog_ids,
        backend,
    )
    activation_report, activation_safe = _activation_report(
        activation_primary,
        activation_replay,
    )

    smoke_indices = select_smoke_indices(public_rows)
    smoke_rows = [public_rows[index] for index in smoke_indices]
    if activation_safe:
        smoke_primary = _run_public_pass(
            catalog,
            smoke_rows,
            catalog_ids,
            categories,
            products,
            backend,
        )
        smoke_replay = _run_public_pass(
            catalog,
            smoke_rows,
            catalog_ids,
            categories,
            products,
            backend,
        )
        smoke_gate, smoke_safe = _public_smoke_gate(
            smoke_primary,
            smoke_replay,
        )
        smoke_report: dict[str, object] = {
            "status": "completed",
            "sample_count": len(smoke_rows),
            "cases_per_scenario": 10,
            "primary": _pass_report(smoke_primary),
            "replay": smoke_gate,
        }
    else:
        smoke_safe = False
        smoke_report = {
            "status": "not_run",
            "reason": "activation_gate_rejected",
        }

    if smoke_only:
        full_report: dict[str, object] = {
            "status": "not_run",
            "reason": "smoke_only_requested",
        }
        decision = {
            "status": "smoke_only",
            "automatic_adoption": False,
        }
    elif not activation_safe or not smoke_safe:
        full_report = {
            "status": "not_run",
            "reason": "frozen_technical_gate_rejected",
        }
        decision = {
            "status": "rejected_before_full",
            "automatic_adoption": False,
        }
    else:
        del activation_replay, smoke_replay
        gc.collect()
        full_runs = _run_public_pass(
            catalog,
            public_rows,
            catalog_ids,
            categories,
            products,
            backend,
        )
        full_report = {
            "status": "completed",
            "sample_count": len(public_rows),
            **_pass_report(full_runs),
        }
        decision = _decision(full_report)

    if _validate_implementation_lock() != lock:
        raise RuntimeError("Phase 16b lock changed during execution")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "exploratory_diagnostic_only",
        "factorial": {
            "arm_order": list(ARM_ORDER),
            "arms": [config.public_contract() for config in ARM_CONFIGS],
            "shared_architecture": _fixed_architecture_contract(),
            "active_starter_changed": False,
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog),
            "public_sha256": _sha256(public),
            "public_case_count": len(public_rows),
            "activation_sha256": _sha256(activation),
            "activation_case_count": len(activation_rows),
            "activation_public_target_overlap": 0,
        },
        "execution": {
            "mode": "strictly_sequential_single_process_cpu",
            "threads": 1,
            "processes": 1,
            "fresh_agent_state_per_arm": True,
            "shared_backend": True,
            "network_access": False,
            "external_model_calls": 0,
            "environment": dict(REQUIRED_ENVIRONMENT),
            "warmup": warmup,
            "backend_startup_wall_seconds": round(backend_seconds, 6),
            "peak_rss_before_backend_bytes": rss_before,
            "peak_rss_after_backend_bytes": rss_after,
        },
        "activation": activation_report,
        "public_smoke": smoke_report,
        "public_full": full_report,
        "decision": decision,
        "fault_taxonomy": contract["fault_taxonomy"],
        "privacy": {
            "aggregate_only": True,
            "labels_used_only_after_agent_replay": True,
            "runtime_received_evaluation_labels": False,
            "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
        },
        "promotion_authority": contract["promotion_authority"],
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "contract_sha256": _sha256(REPOSITORY_ROOT / CONTRACT_RELATIVE),
            "activation_lock_sha256": _sha256(
                REPOSITORY_ROOT / ACTIVATION_LOCK_RELATIVE
            ),
            "implementation_lock_sha256": _sha256(
                REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE
            ),
            "starter_sha256": _sha256(REPOSITORY_ROOT / STARTER_RELATIVE),
            "source_sha256": dict(lock["source_sha256"]),
        },
    }
    validate_publication(payload)
    return payload


def validate_publication(payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    if any(key in serialized for key in _FORBIDDEN_RESULT_KEYS):
        raise ValueError("Phase 16b result contains a forbidden raw-data key")
    if _ASIN_RE.search(serialized):
        raise ValueError("Phase 16b result contains a product identifier")
    if payload.get("privacy") != {
        "aggregate_only": True,
        "labels_used_only_after_agent_replay": True,
        "runtime_received_evaluation_labels": False,
        "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
    }:
        raise ValueError("Phase 16b privacy assertions are incomplete")
    authority = payload.get("promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise ValueError("Phase 16b result claims promotion authority")


def _validate_output_path(
    output: Path,
    catalog: Path,
    public: Path,
    activation: Path,
) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    protected = {
        catalog.resolve(),
        public.resolve(),
        activation.resolve(),
        REPOSITORY_ROOT / CONTRACT_RELATIVE,
        REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE,
        REPOSITORY_ROOT / ACTIVATION_LOCK_RELATIVE,
        *(REPOSITORY_ROOT / relative for relative in SOURCE_PATHS),
    }
    if output.resolve() in protected:
        raise ValueError("Phase 16b output must not overwrite an input or source")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 16b activation and public diagnostics"
    )
    parser.add_argument("--catalog", default=CATALOG_RELATIVE)
    parser.add_argument("--public", default=PUBLIC_RELATIVE)
    parser.add_argument("--activation", default=ACTIVATION_RELATIVE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    catalog = Path(arguments.catalog).resolve()
    public = Path(arguments.public).resolve()
    activation = Path(arguments.activation).resolve()
    output = Path(arguments.output).resolve()
    _validate_output_path(output, catalog, public, activation)
    _claim_output(output, smoke_only=arguments.smoke_only)
    payload = run_phase16b(
        catalog,
        public,
        activation,
        smoke_only=arguments.smoke_only,
    )
    _replace_claim(output, payload)


if __name__ == "__main__":
    main()
