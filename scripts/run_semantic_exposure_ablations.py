"""Frozen four-arm diagnostic for semantic rescue and evidence exposure.

The public evaluator is reused only as a diagnostic.  Product IDs, messages,
profiles, queries, candidate pools, and per-session outcomes remain private and
are reduced to aggregate evidence before publication.  No arm mutates the
active starter configuration.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.exposure_policy import (
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    TOP3_STRUCTURAL_EXPOSURE_POLICY,
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
    RetrievalResult,
    SemanticLexicalRescuePolicy,
    SemanticLexicalRescueStatus,
    SemanticLexicalRetrievalResult,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    RouteWeights,
)
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


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase16-semantic-lexical-exposure-factorial-v1"
REPORT_ID = "phase16-semantic-exposure-factorial-public-20260830"
CONTRACT_RELATIVE = "docs/phase16_semantic_exposure_contract.json"
IMPLEMENTATION_LOCK_RELATIVE = (
    "docs/phase16_semantic_exposure_implementation_lock.json"
)
IMPLEMENTATION_LOCK_ID = "phase16-semantic-exposure-implementation-v1"
EXPECTED_CASES = 200

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
    "scripts/run_semantic_exposure_ablations.py",
    "starter/agent.py",
    "tests/test_evidence_exposure.py",
    "tests/test_semantic_exposure_ablations.py",
    "tests/test_semantic_exposure_factorial.py",
    "tests/test_semantic_lexical_rescue.py",
)

_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")
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
    '"actions"',
)
_SAFE_ROUTE_STATUSES = frozenset({"ok", "empty", "skipped"})
_SEMANTIC_FAULT_STATUSES = (
    SemanticLexicalRescueStatus.BM25_UNAVAILABLE,
    SemanticLexicalRescueStatus.DENSE_UNAVAILABLE,
    SemanticLexicalRescueStatus.DENSE_ERROR,
    SemanticLexicalRescueStatus.TERM_EXTRACTION_ERROR,
    SemanticLexicalRescueStatus.RETRY_ERROR,
)
_EXPOSURE_FAULT_STATUSES = (
    EvidenceExposureStatus.RETRIEVAL_FAIL_OPEN,
    EvidenceExposureStatus.EVIDENCE_FAIL_OPEN,
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
        return self.exposure_policy is TOP3_STRUCTURAL_EXPOSURE_POLICY

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
        "gate_only",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        TOP3_STRUCTURAL_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "combined",
        SHARED_DENSE_TERMS_RESCUE_POLICY,
        TOP3_STRUCTURAL_EXPOSURE_POLICY,
    ),
)
ARM_ORDER = tuple(config.arm_id for config in ARM_CONFIGS)


class FactorialAuditRetriever:
    """Forward the full opt-in protocol while retaining aggregate counters only."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._bm25_statuses: Counter[str] = Counter()
        self._dense_statuses: Counter[str] = Counter()
        self._semantic_statuses: Counter[str] = Counter()
        self._private_dense_statuses: Counter[str] = Counter()
        self._fallbacks = 0
        self._candidate_document_calls = 0
        self._protocol_evidence_calls = 0
        self._category_support_calls = 0
        self._constraint_support_calls = 0
        self._exact_constraint_count_calls = 0
        self._private_dense_candidates = 0
        self._compatible_dense_candidates = 0
        self._expansion_terms = 0
        self._bm25_retries = 0

    @property
    def ranking_cache_capability(self) -> object:
        return self._backend.ranking_cache_capability

    @property
    def snapshot_token(self) -> object:
        return self._backend.snapshot_token

    @property
    def protocol_evidence_capability(self) -> object:
        return self._backend.protocol_evidence_capability

    def protocol_category_exists(self, category: str) -> bool:
        self._category_support_calls += 1
        return bool(self._backend.protocol_category_exists(category))

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: Sequence[str],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        self._constraint_support_calls += 1
        return tuple(
            self._backend.protocol_exact_candidates(
                category,
                constraints,
                limit=limit,
            )
        )

    def protocol_exact_constraint_count(
        self,
        category: str,
        constraints: Sequence[str],
    ) -> int:
        self._exact_constraint_count_calls += 1
        return int(
            self._backend.protocol_exact_constraint_count(category, constraints)
        )

    def candidate_documents(self, parent_asins: Sequence[str]) -> tuple:
        self._candidate_document_calls += 1
        return self._backend.candidate_documents(parent_asins)

    def candidate_protocol_evidence(self, parent_asins: Sequence[str]) -> tuple:
        self._protocol_evidence_calls += 1
        return self._backend.candidate_protocol_evidence(parent_asins)

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        route_weights: RouteWeights,
        **kwargs: object,
    ) -> RetrievalResult:
        result = self._backend.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            route_weights=route_weights,
            **kwargs,
        )
        if not isinstance(result, RetrievalResult):
            raise TypeError("search_with_trace must return RetrievalResult")
        self._bm25_statuses[result.trace.bm25_status] += 1
        self._dense_statuses[result.trace.dense_status] += 1
        self._fallbacks += int(result.trace.used_fallback)
        if isinstance(result, SemanticLexicalRetrievalResult):
            trace = result.semantic_trace
            self._semantic_statuses[trace.status.value] += 1
            self._private_dense_statuses[trace.private_dense_status] += 1
            self._private_dense_candidates += trace.private_dense_candidate_count
            self._compatible_dense_candidates += (
                trace.compatible_dense_candidate_count
            )
            self._expansion_terms += trace.expansion_term_count
            self._bm25_retries += trace.retry_count
        return result

    def validate(self, *, expected_searches: int, rescue_enabled: bool) -> None:
        observed = sum(self._bm25_statuses.values())
        if observed != expected_searches:
            raise RuntimeError("retrieval call accounting drifted")
        route_faults = {
            "bm25": sorted(set(self._bm25_statuses) - _SAFE_ROUTE_STATUSES),
            "dense": sorted(set(self._dense_statuses) - _SAFE_ROUTE_STATUSES),
        }
        if route_faults["bm25"] or route_faults["dense"]:
            raise RuntimeError(f"retrieval route fault: {route_faults}")
        semantic_traces = sum(self._semantic_statuses.values())
        expected_semantic = expected_searches if rescue_enabled else 0
        if semantic_traces != expected_semantic:
            raise RuntimeError("semantic trace coverage drifted")

    def summary(self) -> dict[str, object]:
        return {
            "bm25": dict(sorted(self._bm25_statuses.items())),
            "dense": dict(sorted(self._dense_statuses.items())),
            "fallback_turns": self._fallbacks,
            "candidate_document_calls": self._candidate_document_calls,
            "candidate_protocol_evidence_calls": self._protocol_evidence_calls,
            "category_support_calls": self._category_support_calls,
            "constraint_support_calls": self._constraint_support_calls,
            "exact_constraint_count_calls": self._exact_constraint_count_calls,
            "semantic_trace_status": dict(sorted(self._semantic_statuses.items())),
            "private_dense_status": dict(
                sorted(self._private_dense_statuses.items())
            ),
            "private_dense_candidate_total": self._private_dense_candidates,
            "compatible_dense_candidate_total": (
                self._compatible_dense_candidates
            ),
            "safe_expansion_term_total": self._expansion_terms,
            "bm25_retry_total": self._bm25_retries,
        }


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
    payload = json.loads(
        (repository_root / CONTRACT_RELATIVE).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError("Phase 16 contract must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 16 contract")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("unexpected Phase 16 experiment identity")
    if payload.get("status") != "frozen_public_diagnostic_only":
        raise RuntimeError("Phase 16 contract is not frozen")
    if payload.get("arms") != [config.public_contract() for config in ARM_CONFIGS]:
        raise RuntimeError("Phase 16 arm definitions drifted")
    if payload.get("fixed_shared_architecture") != _fixed_architecture_contract():
        raise RuntimeError("Phase 16 shared architecture drifted")
    execution = payload.get("execution")
    if not isinstance(execution, dict) or execution.get("arm_order") != list(
        ARM_ORDER
    ):
        raise RuntimeError("Phase 16 arm order drifted")
    authority = payload.get("promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise RuntimeError("Phase 16 diagnostic acquired promotion authority")
    return payload


def _validate_locked_paths(
    repository_root: Path,
    expected_hashes: Mapping[str, object],
) -> None:
    if not expected_hashes:
        raise RuntimeError("implementation source lock is empty")
    for relative, expected in expected_hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _HEX_SHA256_RE.fullmatch(expected) is None
        ):
            raise RuntimeError("implementation source lock is malformed")
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"locked path drifted: {relative}")


def _validate_implementation_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    contract = _load_contract(repository_root)
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
        "input_sha256",
        "arm_order",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 16 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 16 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 16 implementation lock identity")
    if lock.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Phase 16 lock experiment identity drifted")
    if lock.get("status") != "locked_before_public_diagnostic":
        raise RuntimeError("Phase 16 implementation is not frozen")
    if lock.get("contract_sha256") != _sha256(
        repository_root / CONTRACT_RELATIVE
    ):
        raise RuntimeError("Phase 16 contract drifted after lock")
    if lock.get("arm_order") != list(ARM_ORDER):
        raise RuntimeError("Phase 16 locked arm order drifted")

    frozen_inputs = contract.get("frozen_inputs")
    if not isinstance(frozen_inputs, dict):
        raise RuntimeError("Phase 16 frozen inputs are unavailable")
    expected_inputs: dict[str, str] = {}
    for name in ("catalog", "evaluation", "starter"):
        item = frozen_inputs.get(name)
        if not isinstance(item, dict):
            raise RuntimeError("Phase 16 frozen input entry is invalid")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RuntimeError("Phase 16 frozen input hash is invalid")
        expected_inputs[relative] = digest
    if lock.get("input_sha256") != expected_inputs:
        raise RuntimeError("Phase 16 input lock disagrees with the contract")
    _validate_locked_paths(repository_root, expected_inputs)

    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(
        SOURCE_PATHS
    ):
        raise RuntimeError("Phase 16 implementation source lock is incomplete")
    _validate_locked_paths(repository_root, source_hashes)

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
        raise RuntimeError("Phase 16 pre-lock verification schema drifted")
    if verification.get("completed_before_lock") is not True:
        raise RuntimeError("Phase 16 verification was not completed before lock")
    for key in (
        "focused_tests_passed",
        "complete_tests_passed",
        "phase13_oracle_cases",
    ):
        if type(verification.get(key)) is not int or int(verification[key]) <= 0:
            raise RuntimeError("Phase 16 verification count is invalid")
    oracle_digest = verification.get("phase13_oracle_sha256")
    if (
        not isinstance(oracle_digest, str)
        or _HEX_SHA256_RE.fullmatch(oracle_digest) is None
    ):
        raise RuntimeError("Phase 16 oracle digest is invalid")
    return lock


def _validate_execution_environment() -> None:
    mismatches = {
        key: {"expected": expected, "observed": os.environ.get(key)}
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


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    return _deep_size(
        (
            getattr(agent, "_sessions", None),
            getattr(agent, "_slates", None),
            getattr(agent, "_profile_priors", None),
            getattr(planner, "_entries", None),
            getattr(agent, "_exact_evidence_counts", None),
            getattr(agent, "_semantic_rescue_counts", None),
            getattr(agent, "_evidence_exposure_counts", None),
        )
    )


def _status_sum(health: Mapping[str, object], statuses: Sequence[object]) -> int:
    return sum(int(health[getattr(status, "value")]) for status in statuses)


def _validate_variant_accounting(
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
    response = diagnostics["response_audit"]
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
            response,
        )
    ):
        raise RuntimeError("Phase 16 diagnostic schema is invalid")

    searches = int(orchestration["searches"])
    reuses = int(orchestration["reuses"])
    if int(orchestration["decisions"]) != turns or searches + reuses != turns:
        raise RuntimeError("Phase 16 orchestration coverage is incomplete")
    if int(orchestration["skips"]):
        raise RuntimeError("Phase 16 official workload unexpectedly skipped")
    if sum(int(value) for value in route["bm25"].values()) != searches:
        raise RuntimeError("Phase 16 retrieval coverage is incomplete")
    if (
        int(route["candidate_document_calls"]) != searches
        or int(ranking["attempts"]) != searches
        or int(ranking["successes"]) != searches
        or int(ranking["failures"])
        or int(ranking["unavailable_skips"])
    ):
        raise RuntimeError("Phase 16 ranking accounting drifted")

    exact_attempts = int(exact["attempts"])
    expected_exact_attempts = turns if config.gate_enabled else searches
    if (
        exact_attempts != expected_exact_attempts
        or int(route["candidate_protocol_evidence_calls"]) != exact_attempts
    ):
        raise RuntimeError("Phase 16 exact-evidence accounting drifted")
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
    if exact_outcomes != exact_attempts:
        raise RuntimeError("Phase 16 exact-evidence outcomes do not partition")

    semantic_attempts = int(semantic["attempts"])
    semantic_outcomes = _status_sum(semantic, tuple(SemanticLexicalRescueStatus))
    if config.rescue_enabled:
        if (
            semantic["policy"] != SHARED_DENSE_TERMS_RESCUE_POLICY.value
            or semantic_attempts != searches
            or semantic_outcomes != semantic_attempts
        ):
            raise RuntimeError("Phase 16 semantic rescue accounting drifted")
    elif (
        semantic["policy"] != DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY.value
        or semantic_attempts
        or semantic_outcomes
    ):
        raise RuntimeError("disabled semantic rescue performed work")

    exposure_attempts = int(exposure["attempts"])
    exposure_outcomes = _status_sum(exposure, tuple(EvidenceExposureStatus))
    if config.gate_enabled:
        if (
            exposure["policy"] != TOP3_STRUCTURAL_EXPOSURE_POLICY.value
            or exposure_attempts != turns
            or exposure_outcomes != exposure_attempts
            or int(exposure["withheld_turns"])
            != int(exposure[EvidenceExposureStatus.QUESTION_WITHHELD.value])
        ):
            raise RuntimeError("Phase 16 exposure accounting drifted")
    elif (
        exposure["policy"] != DISABLED_EVIDENCE_EXPOSURE_POLICY.value
        or exposure_attempts
        or exposure_outcomes
        or int(exposure["withheld_turns"])
    ):
        raise RuntimeError("disabled evidence exposure performed work")

    expected_slates = turns - int(exposure["withheld_turns"])
    if (
        int(slate["attempts"]) != expected_slates
        or int(slate["successes"]) != expected_slates
        or int(slate["failures"])
    ):
        raise RuntimeError("Phase 16 slate accounting drifted")
    if any(
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
    ):
        raise RuntimeError("Phase 16 unexpectedly enabled protocol planning")


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
    latency = audited.latency_summary()
    if int(latency["count"]) != expected_turns:
        raise RuntimeError("Phase 16 response timing coverage is incomplete")
    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": guarded.summary(),
        "ranking_health": agent.ranking_health,
        "exact_evidence_health": agent.exact_evidence_health,
        "semantic_lexical_rescue_health": agent.semantic_lexical_rescue_health,
        "evidence_exposure_health": agent.evidence_exposure_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
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
        "respond_latency_ms": latency,
        "peak_rss_bytes": _current_max_rss_bytes(),
    }
    _validate_variant_accounting(diagnostics, config)
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("Phase 16 evaluator sessions are unavailable")
    return VariantRun(
        summary=_overall_summary(result),
        sessions=sessions,
        diagnostics=diagnostics,
        evaluator_digest=hashlib.sha256(_canonical_json(result)).hexdigest(),
        behavior_digest=audited.behavior_digest,
        actions=audited.actions,
    )


def _run_pass(
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


def _tokens_are_zero(summary: Mapping[str, object]) -> bool:
    usage = summary.get("reported_token_usage")
    return isinstance(usage, dict) and all(
        type(usage.get(key)) is int and int(usage[key]) == 0
        for key in TOKEN_USAGE_KEYS
    )


def _faults_are_zero(run: VariantRun) -> bool:
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
        and not _status_sum(semantic, _SEMANTIC_FAULT_STATUSES)
        and not int(semantic["validation_or_execution_fallbacks"])
        and not _status_sum(exposure, _EXPOSURE_FAULT_STATUSES)
        and not int(exposure["validation_fallbacks"])
        and not int(slate["failures"])
        and not int(orchestration["fault_invalidations"])
        and not int(orchestration["store_rejections"])
        and not int(response["response_exceptions"])
        and not int(response["invalid_api_responses"])
        and not int(diagnostics["runtime_network_attempts"])
    )


def _action_summary(run: VariantRun) -> dict[str, object]:
    questions = Counter(
        ask if ask is not None else "none"
        for _, _, ask, _ in run.actions
    )
    widths = Counter(width for _, _, _, width in run.actions)
    total = len(run.actions)
    return {
        "turns": total,
        "question_turns": total - int(questions["none"]),
        "question_attribute_counts": dict(sorted(questions.items())),
        "width_counts": {
            str(width): count for width, count in sorted(widths.items())
        },
        "mean_exposed_width": round(
            sum(width * count for width, count in widths.items()) / total,
            6,
        )
        if total
        else 0.0,
    }


def _variant_public(run: VariantRun) -> dict[str, object]:
    diagnostics = run.diagnostics
    return {
        "official_metrics": run.summary,
        "route_health": diagnostics["route_health"],
        "ranking_health": diagnostics["ranking_health"],
        "exact_evidence_health": diagnostics["exact_evidence_health"],
        "semantic_lexical_rescue_health": diagnostics[
            "semantic_lexical_rescue_health"
        ],
        "evidence_exposure_health": diagnostics["evidence_exposure_health"],
        "orchestration_health": diagnostics["orchestration_health"],
        "slate_health": diagnostics["slate_health"],
        "response_audit": diagnostics["response_audit"],
        "runtime_network_attempts": diagnostics["runtime_network_attempts"],
        "retained_agent_bytes": diagnostics["retained_agent_bytes"],
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
        "peak_rss_bytes": diagnostics["peak_rss_bytes"],
        "action_summary": _action_summary(run),
        "fault_free": _faults_are_zero(run),
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
        "fault_free": _faults_are_zero(candidate),
    }
    observations["all_nonregression_observations_hold"] = all(
        observations.values()
    )
    return observations


def _pass_report(runs: Mapping[str, VariantRun]) -> dict[str, object]:
    if tuple(runs) != ARM_ORDER:
        raise RuntimeError("Phase 16 pass order drifted")
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


def _deterministic_health(run: VariantRun) -> dict[str, object]:
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


def _smoke_replay_gate(
    primary: Mapping[str, VariantRun],
    replay: Mapping[str, VariantRun],
) -> tuple[dict[str, object], bool]:
    exactness: dict[str, object] = {}
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
    gates = {
        "all_primary_arms_fault_free": all(
            _faults_are_zero(primary[arm_id]) for arm_id in ARM_ORDER
        ),
        "all_replay_arms_fault_free": all(
            _faults_are_zero(replay[arm_id]) for arm_id in ARM_ORDER
        ),
        "all_arms_zero_token": all(
            _tokens_are_zero(run.summary)
            for collection in (primary, replay)
            for run in collection.values()
        ),
        "all_arms_exactly_replay": all(
            all(values.values()) for values in exactness.values()
        ),
    }
    gates["authorize_full_public_diagnostic"] = all(gates.values())
    return {"arm_exactness": exactness, "safety_gate": gates}, bool(
        gates["authorize_full_public_diagnostic"]
    )


def _component_decision(full_report: Mapping[str, object]) -> dict[str, object]:
    comparisons = full_report.get("comparisons")
    if not isinstance(comparisons, dict):
        raise RuntimeError("Phase 16 full comparisons are unavailable")
    recommendations: dict[str, object] = {}
    labels = {
        "semantic_to_lexical_rescue": "rescue_only",
        "evidence_gated_exposure": "gate_only",
        "combined": "combined",
    }
    for component, arm_id in labels.items():
        comparison = comparisons[arm_id]
        observations = comparison["diagnostic_observations"]
        delta = comparison["metric_delta_vs_baseline"]
        improved = bool(
            float(delta["mrr"]) > 0
            or float(delta["recommended_technical_score"]) > 0
            or float(delta["mttc"]) < 0
        )
        promising = bool(
            observations["all_nonregression_observations_hold"] and improved
        )
        recommendations[component] = {
            "arm": arm_id,
            "public_diagnostic_signal": (
                "promising_for_fresh_validation"
                if promising
                else "not_supported_for_adoption"
            ),
            "may_change_starter": False,
        }
    return {
        "status": "diagnostic_not_promotable",
        "automatic_adoption": False,
        "reason": (
            "The public suite was previously used. Any promising component "
            "must pass a fresh target-disjoint confirmation before adoption."
        ),
        "components": recommendations,
    }


def _warm_backend(catalog: Path, backend: object) -> dict[str, object]:
    baseline = ARM_CONFIGS[0]
    warmup = _new_agent(catalog, backend, baseline)
    session_id = "phase16-label-free-backend-warmup"
    started = time.perf_counter()
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    if int(warmup.ranking_health["successes"]) != 1:
        raise RuntimeError("Phase 16 label-free backend warm-up failed")
    vocabulary_builder = getattr(backend, "_ensure_bm25_vocabulary", None)
    if not callable(vocabulary_builder) or vocabulary_builder() is not True:
        raise RuntimeError("Phase 16 catalog-frequency cache warm-up failed")
    return {
        "label_free": True,
        "dense_and_bm25_warmed": True,
        "catalog_frequency_cache_warmed": True,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def run_semantic_exposure_factorial(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    smoke_only: bool = False,
) -> dict[str, object]:
    _validate_execution_environment()
    contract = _load_contract()
    lock = _validate_implementation_lock()
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    if catalog != (REPOSITORY_ROOT / "data/catalog.jsonl").resolve():
        raise RuntimeError("Phase 16 requires the frozen catalog path")
    if dataset != (REPOSITORY_ROOT / "data/public_set.jsonl").resolve():
        raise RuntimeError("Phase 16 requires the frozen public diagnostic path")

    samples = load_jsonl(dataset)
    if len(samples) != EXPECTED_CASES:
        raise RuntimeError("Phase 16 public diagnostic case count drifted")
    smoke_indices = select_smoke_indices(samples)
    smoke_samples = [samples[index] for index in smoke_indices]
    catalog_ids, categories, products = catalog_index(catalog)

    rss_before_backend = _current_max_rss_bytes()
    backend_started = time.perf_counter()
    bootstrap = ConversationalSearchAgent(
        catalog,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    backend = bootstrap.retrieval_backend
    backend_seconds = time.perf_counter() - backend_started
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("Phase 16 BM25 retrieval is unavailable")
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("Phase 16 dense retrieval is unavailable")
    if (
        getattr(backend, "ranking_cache_capability", None)
        is not EXACT_RANKING_CACHE_CAPABILITY
    ):
        raise RuntimeError("Phase 16 exact ranking cache is unavailable")
    if (
        getattr(backend, "protocol_evidence_capability", None)
        is not PROTOCOL_EVIDENCE_CAPABILITY
    ):
        raise RuntimeError("Phase 16 protocol evidence is unavailable")
    rss_after_backend = _current_max_rss_bytes()
    del bootstrap
    warmup = _warm_backend(catalog, backend)

    smoke_primary = _run_pass(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
    )
    smoke_replay = _run_pass(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
    )
    replay_report, authorize_full = _smoke_replay_gate(
        smoke_primary,
        smoke_replay,
    )
    smoke_report = {
        "sample_count": len(smoke_samples),
        "cases_per_scenario": 10,
        "primary": _pass_report(smoke_primary),
        "replay": replay_report,
    }

    if smoke_only:
        full_report: dict[str, object] = {
            "status": "not_run",
            "reason": "smoke_only_requested",
        }
        decision = {
            "status": "smoke_only",
            "automatic_adoption": False,
            "reason": "A smoke run cannot support adoption.",
        }
    elif not authorize_full:
        full_report = {
            "status": "not_run",
            "reason": "frozen_smoke_safety_gate_rejected",
        }
        decision = {
            "status": "rejected_at_smoke",
            "automatic_adoption": False,
            "reason": "At least one frozen safety or replay requirement failed.",
        }
    else:
        del smoke_replay
        gc.collect()
        full_runs = _run_pass(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
        )
        full_report = {
            "status": "completed",
            "sample_count": len(samples),
            **_pass_report(full_runs),
        }
        decision = _component_decision(full_report)

    final_lock = _validate_implementation_lock()
    if final_lock != lock:
        raise RuntimeError("Phase 16 lock changed during execution")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "public_diagnostic_only",
        "isolated_factorial": {
            "arm_order": list(ARM_ORDER),
            "arms": [config.public_contract() for config in ARM_CONFIGS],
            "shared_architecture": _fixed_architecture_contract(),
            "active_starter_changed": False,
        },
        "dataset": {
            "catalog_sha256": _sha256(catalog),
            "evaluation_sha256": _sha256(dataset),
            "sample_count": len(samples),
            "classification": "previously_used_public_diagnostic",
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
            "peak_rss_before_backend_bytes": rss_before_backend,
            "peak_rss_after_backend_bytes": rss_after_backend,
        },
        "smoke": smoke_report,
        "full": full_report,
        "decision": decision,
        "privacy": {
            "aggregate_only": True,
            "labels_used_only_after_agent_replay": True,
            "runtime_received_evaluation_labels": False,
            "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "contract_sha256": _sha256(REPOSITORY_ROOT / CONTRACT_RELATIVE),
            "implementation_lock_sha256": _sha256(
                REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE
            ),
            "starter_sha256": _sha256(REPOSITORY_ROOT / "starter/agent.py"),
            "source_sha256": dict(final_lock["source_sha256"]),
        },
        "contract_promotion_authority": contract["promotion_authority"],
    }
    validate_publication(payload)
    return payload


def validate_publication(payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    if any(key in serialized for key in _FORBIDDEN_PUBLICATION_KEYS):
        raise ValueError("aggregate report contains a forbidden raw-data key")
    if _ASIN_RE.search(serialized):
        raise ValueError("aggregate report contains a product identifier")
    if payload.get("privacy") != {
        "aggregate_only": True,
        "labels_used_only_after_agent_replay": True,
        "runtime_received_evaluation_labels": False,
        "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
    }:
        raise ValueError("aggregate report privacy assertions are incomplete")
    authority = payload.get("contract_promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise ValueError("aggregate report claims unauthorized promotion")


def _validate_output_path(output: Path, catalog: Path, dataset: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        REPOSITORY_ROOT / CONTRACT_RELATIVE,
        REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE,
        *(REPOSITORY_ROOT / relative for relative in SOURCE_PATHS),
    }
    if output.resolve() in protected:
        raise ValueError("output must not overwrite an input or source file")


def _claim_output(path: Path, *, smoke_only: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "status": "run_claimed_no_retry",
                "scope": "smoke" if smoke_only else "full_public_diagnostic",
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_claim(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen aggregate-only semantic rescue/exposure factorial"
        )
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    catalog = Path(arguments.catalog).resolve()
    dataset = Path(arguments.dataset).resolve()
    output = Path(arguments.output).resolve()
    _validate_output_path(output, catalog, dataset)
    _claim_output(output, smoke_only=arguments.smoke_only)
    payload = run_semantic_exposure_factorial(
        catalog,
        dataset,
        smoke_only=arguments.smoke_only,
    )
    _replace_claim(output, payload)


if __name__ == "__main__":
    main()
