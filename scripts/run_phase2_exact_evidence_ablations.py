"""Sequential, aggregate-only Phase 2 exact-evidence A/B evaluation.

The runner changes one runtime axis only: protected Phase 13 Stage-A ranking
versus the lexicographic exact-evidence ranking policy.  Both arms share one
protocol-capable retrieval backend, use fresh agent state, and run in one
process.  Evaluation labels are used only after each label-free replay to form
aggregate metrics and deterministic target-group folds; they are never passed
to the agent or serialized.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import re
import socket
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.intent import ROBUST_INTENT_POLICY
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.questions import (
    CONSERVATIVE_EARLY_OTHER_POLICY,
    QUESTION_TEXT,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
)
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    PROTOCOL_EVIDENCE_CAPABILITY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import catalog_index, load_jsonl, metric_summary
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
from scripts.run_profile_ablations import (
    TOKEN_USAGE_KEYS,
    _CallAuditRetriever,
    _project_profile_health,
    _validate_variant_accounting,
)
from scripts.run_reranking_ablations import RespondLatencyAgent, _expected_turns


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORT_ID = "phase2-lexicographic-exact-evidence-ab-20260830"
EXPERIMENT_ID = "phase2-lexicographic-exact-evidence-v1"
PHASE0_BASELINE_RELATIVE = (
    "benchmarks/diagnostics/phase0-active-phase13-baseline-20260830.json"
)
BASELINE_ID = STAGE_A_RANKING_POLICY.value
CANDIDATE_ID = LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY.value
SMOKE_PER_SCENARIO = 10
FOLD_COUNT = 5
SMOKE_SELECTION_SALT = "phase2-exact-evidence-smoke-v1"
FOLD_SELECTION_SALT = "phase2-exact-evidence-fold-v1"
MAX_ADDITIONAL_RSS_BYTES = 64 * 1024 * 1024
MAX_ADDITIONAL_AGENT_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_RATIO = 1.10

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
    "conversational_search/orchestration.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "evaluator/local_evaluator.py",
    "scripts/run_phase2_exact_evidence_ablations.py",
    "starter/agent.py",
)

_OFFICIAL_SCENARIOS = ("boundary", "browsing", "buying", "intent_override")
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


@dataclass(slots=True)
class VariantRun:
    summary: dict
    sessions: list[dict]
    diagnostics: dict
    evaluator_digest: str
    behavior_digest: str
    actions: tuple[tuple[int, int, str | None, int], ...]


class ExactEvidenceCallAuditRetriever(_CallAuditRetriever):
    """Count metadata lookups while retaining no candidate IDs or text."""

    def __init__(self, backend: object) -> None:
        super().__init__(backend)
        self._protocol_evidence_calls = 0

    @property
    def protocol_evidence_capability(self) -> object:
        return self._backend.protocol_evidence_capability

    def candidate_protocol_evidence(self, parent_asins: Sequence[str]) -> tuple:
        self._protocol_evidence_calls += 1
        return self._backend.candidate_protocol_evidence(parent_asins)

    def summary(self) -> dict:
        return {
            **super().summary(),
            "candidate_protocol_evidence_calls": self._protocol_evidence_calls,
        }


class AggregateAuditAgent(RespondLatencyAgent):
    """Time responses and retain only private, ordinal action signatures."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        catalog_ids: set[str],
    ) -> None:
        super().__init__(delegate)
        self._search_delegate = delegate
        self._catalog_ids = catalog_ids
        self._ordinals: dict[str, int] = {}
        self._actions: list[tuple[int, int, str | None, int]] = []
        self._response_digests: list[bytes] = []
        self.response_exceptions = 0
        self.invalid_api_responses = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        if session_id in self._ordinals:
            raise RuntimeError("diagnostic session ID was reused")
        self._ordinals[session_id] = len(self._ordinals)
        super().reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            response = super().respond(session_id, user_message, turn, top_k)
        except Exception:
            self.response_exceptions += 1
            raise
        if not self._valid_response(response, top_k):
            self.invalid_api_responses += 1
        ask = response.get("ask_attribute")
        recommendations = response.get("recommendations")
        width = len(recommendations) if isinstance(recommendations, list) else -1
        self._actions.append((self._ordinals[session_id], turn, ask, width))
        self._response_digests.append(hashlib.sha256(_canonical_json(response)).digest())
        return response

    def _valid_response(self, response: object, top_k: int) -> bool:
        if not isinstance(response, dict) or not isinstance(response.get("message"), str):
            return False
        ask = response.get("ask_attribute")
        if ask is not None and ask not in QUESTION_TEXT:
            return False
        recommendations = response.get("recommendations")
        if not isinstance(recommendations, list) or len(recommendations) > top_k:
            return False
        identifiers: list[str] = []
        for item in recommendations:
            value = item.get("parent_asin") if isinstance(item, dict) else item
            if not isinstance(value, str) or value not in self._catalog_ids:
                return False
            identifiers.append(value)
        if len(identifiers) != len(set(identifiers)):
            return False
        usage = response.get("usage")
        return isinstance(usage, dict) and all(
            type(usage.get(key)) is int and int(usage[key]) >= 0
            for key in ("prompt_tokens", "completion_tokens")
        )

    @property
    def actions(self) -> tuple[tuple[int, int, str | None, int], ...]:
        return tuple(self._actions)

    @property
    def behavior_digest(self) -> str:
        return hashlib.sha256(b"".join(self._response_digests)).hexdigest()


class RuntimeNetworkAudit:
    """Deny and count evaluator-visible socket connections."""

    def __init__(self) -> None:
        self.attempts = 0

    @contextlib.contextmanager
    def deny(self) -> Iterator[None]:
        audit = self
        original_socket = socket.socket

        class DeniedSocket(original_socket):
            def connect(self, address: object) -> None:
                del address
                audit.attempts += 1
                raise RuntimeError("runtime network access is disabled")

            def connect_ex(self, address: object) -> int:
                del address
                audit.attempts += 1
                return 1

        def denied_create_connection(*args: object, **kwargs: object) -> object:
            del args, kwargs
            audit.attempts += 1
            raise RuntimeError("runtime network access is disabled")

        with patch.object(socket, "socket", DeniedSocket), patch.object(
            socket,
            "create_connection",
            denied_create_connection,
        ):
            yield


def _selection_digest(salt: str, value: str) -> bytes:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).digest()


def _sample_target(sample: Mapping[str, object]) -> str:
    ground_truth = sample.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("evaluation sample has no ground-truth object")
    value = ground_truth.get("parent_asin")
    if not isinstance(value, str) or not value:
        raise ValueError("evaluation sample has no target product")
    return value


def select_smoke_indices(
    samples: Sequence[Mapping[str, object]],
    *,
    per_scenario: int = SMOKE_PER_SCENARIO,
) -> tuple[int, ...]:
    """Choose one deterministic, scenario-balanced safety slice."""

    if isinstance(per_scenario, bool) or not isinstance(per_scenario, int):
        raise TypeError("per_scenario must be an integer")
    if per_scenario <= 0:
        raise ValueError("per_scenario must be positive")
    groups: dict[str, list[tuple[bytes, int]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        scenario = sample.get("scenario_type")
        if not isinstance(scenario, str) or not scenario:
            raise ValueError("evaluation sample has no scenario")
        target = _sample_target(sample)
        key = _selection_digest(
            SMOKE_SELECTION_SALT,
            f"{scenario}\0{target}\0{index}",
        )
        groups[scenario].append((key, index))
    if set(groups) != set(_OFFICIAL_SCENARIOS):
        raise ValueError("smoke selection requires every official scenario")
    selected: list[int] = []
    for scenario in _OFFICIAL_SCENARIOS:
        values = sorted(groups[scenario])
        if len(values) < per_scenario:
            raise ValueError(f"scenario {scenario} has too few smoke samples")
        selected.extend(index for _, index in values[:per_scenario])
    return tuple(sorted(selected))


def assign_disjoint_folds(
    samples: Sequence[Mapping[str, object]],
    *,
    fold_count: int = FOLD_COUNT,
) -> tuple[tuple[int, ...], ...]:
    """Assign every repeated product group to one balanced deterministic fold."""

    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise TypeError("fold_count must be an integer")
    if fold_count < 2:
        raise ValueError("fold_count must be at least two")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[_sample_target(sample)].append(index)
    if len(grouped) < fold_count:
        raise ValueError("too few disjoint product groups for the requested folds")

    folds: list[list[int]] = [[] for _ in range(fold_count)]
    loads = [0] * fold_count
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: _selection_digest(FOLD_SELECTION_SALT, item[0]),
    )
    for _, indices in ordered_groups:
        fold = min(range(fold_count), key=lambda value: (loads[value], value))
        folds[fold].extend(indices)
        loads[fold] += len(indices)
    return tuple(tuple(sorted(indices)) for indices in folds)


def _summary_from_sessions(sessions: Sequence[Mapping[str, object]]) -> dict:
    projected = [dict(item) for item in sessions]
    summary = metric_summary(projected)
    if not projected or summary["mttc"] is None:
        raise ValueError("aggregate subset must not be empty")
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    score = (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )
    return {
        **summary,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(score, 6),
    }


def _subset_report(
    baseline_sessions: Sequence[Mapping[str, object]],
    candidate_sessions: Sequence[Mapping[str, object]],
    indices: Sequence[int],
) -> dict:
    baseline = _summary_from_sessions([baseline_sessions[index] for index in indices])
    candidate = _summary_from_sessions([candidate_sessions[index] for index in indices])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": _metric_deltas(baseline, candidate),
    }


def build_fold_report(
    samples: Sequence[Mapping[str, object]],
    baseline_sessions: Sequence[Mapping[str, object]],
    candidate_sessions: Sequence[Mapping[str, object]],
) -> dict:
    if not (
        len(samples) == len(baseline_sessions) == len(candidate_sessions)
        and samples
    ):
        raise ValueError("fold inputs must have equal nonzero lengths")
    for sample, baseline, candidate in zip(
        samples,
        baseline_sessions,
        candidate_sessions,
    ):
        if (
            sample.get("sample_id") != baseline.get("sample_id")
            or baseline.get("sample_id") != candidate.get("sample_id")
        ):
            raise RuntimeError("fold outcome order drifted")
    folds = assign_disjoint_folds(samples)
    reports = {
        f"fold_{index}": _subset_report(
            baseline_sessions,
            candidate_sessions,
            indices,
        )
        for index, indices in enumerate(folds)
    }
    mrr_deltas = [float(value["delta"]["mrr"]) for value in reports.values()]
    score_deltas = [
        float(value["delta"]["recommended_technical_score"])
        for value in reports.values()
    ]
    return {
        "assignment": {
            "fold_count": len(folds),
            "sample_counts": [len(indices) for indices in folds],
            "repeated_products_confined_to_one_fold": True,
            "selection_salt_sha256": hashlib.sha256(
                FOLD_SELECTION_SALT.encode("utf-8")
            ).hexdigest(),
        },
        "folds": reports,
        "summary": {
            "mrr_nonnegative_fold_count": sum(value >= 0 for value in mrr_deltas),
            "mrr_positive_fold_count": sum(value > 0 for value in mrr_deltas),
            "median_mrr_delta": round(sorted(mrr_deltas)[len(mrr_deltas) // 2], 6),
            "worst_mrr_delta": round(min(mrr_deltas), 6),
            "worst_technical_score_delta": round(min(score_deltas), 6),
        },
    }


def _scenario_report(
    baseline_sessions: Sequence[Mapping[str, object]],
    candidate_sessions: Sequence[Mapping[str, object]],
) -> dict:
    result: dict[str, dict] = {}
    for scenario in _OFFICIAL_SCENARIOS:
        indices = [
            index
            for index, session in enumerate(baseline_sessions)
            if session.get("scenario_type") == scenario
        ]
        if not indices:
            raise RuntimeError(f"scenario {scenario} is absent")
        if any(
            candidate_sessions[index].get("scenario_type") != scenario
            for index in indices
        ):
            raise RuntimeError("scenario outcome order drifted")
        result[scenario] = _subset_report(
            baseline_sessions,
            candidate_sessions,
            indices,
        )
    return result


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    return _deep_size(
        (
            getattr(agent, "_sessions", None),
            getattr(agent, "_slates", None),
            getattr(agent, "_profile_priors", None),
            getattr(planner, "_entries", None),
            getattr(agent, "_exact_evidence_counts", None),
        )
    )


def _new_agent(
    catalog: Path,
    backend: object,
    ranking_policy: RankingPolicy,
) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        catalog,
        retriever=backend,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=ranking_policy,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        intent_policy=ROBUST_INTENT_POLICY,
        decision_policy=PROTECTED_DECISION_POLICY,
        requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    ranking_policy: RankingPolicy,
) -> VariantRun:
    guarded = ExactEvidenceCallAuditRetriever(backend)
    agent = _new_agent(catalog, guarded, ranking_policy)
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
    _validate_variant_accounting(
        diagnostics,
        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    _validate_exact_accounting(diagnostics, ranking_policy)
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


def _validate_exact_accounting(
    diagnostics: Mapping[str, object],
    ranking_policy: RankingPolicy,
) -> None:
    exact = diagnostics["exact_evidence_health"]
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    protocol = diagnostics["protocol_decision_health"]
    if not isinstance(exact, dict) or not isinstance(route, dict):
        raise RuntimeError("exact-evidence diagnostics are invalid")
    if not isinstance(ranking, dict) or not isinstance(protocol, dict):
        raise RuntimeError("policy diagnostics are invalid")
    attempts = int(exact["attempts"])
    outcomes = sum(
        int(exact[key])
        for key in (
            "applied",
            "zero_support_fail_open",
            "capability_unavailable",
            "evidence_errors",
            "validation_errors",
        )
    )
    if attempts != outcomes:
        raise RuntimeError("exact-evidence outcomes do not partition attempts")
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
        raise RuntimeError("Phase 2 unexpectedly enabled protocol planning")
    evidence_calls = int(route["candidate_protocol_evidence_calls"])
    if ranking_policy is LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY:
        if exact["policy"] != CANDIDATE_ID:
            raise RuntimeError("candidate exact-evidence policy drifted")
        if attempts != int(ranking["attempts"]) or evidence_calls != attempts:
            raise RuntimeError("candidate exact-evidence call accounting drifted")
    else:
        if exact["policy"] != BASELINE_ID or attempts or evidence_calls:
            raise RuntimeError("baseline performed exact-evidence work")


def _warm_backend(catalog: Path, backend: object) -> None:
    agent = _new_agent(catalog, backend, STAGE_A_RANKING_POLICY)
    session_id = "phase2-label-free-backend-warmup"
    agent.reset(session_id, {})
    agent.respond(
        session_id,
        "I'm exploring a generic clothing item and have no firm preference yet.",
        1,
        10,
    )
    if int(agent.ranking_health["successes"]) != 1:
        raise RuntimeError("label-free backend warm-up failed")


def _common_action_equivalence(
    baseline: VariantRun,
    candidate: VariantRun,
) -> dict[str, int | bool]:
    baseline_actions = {
        (ordinal, turn): (ask, width)
        for ordinal, turn, ask, width in baseline.actions
    }
    candidate_actions = {
        (ordinal, turn): (ask, width)
        for ordinal, turn, ask, width in candidate.actions
    }
    common = sorted(set(baseline_actions).intersection(candidate_actions))
    mismatches = sum(
        baseline_actions[key] != candidate_actions[key] for key in common
    )
    return {
        "common_turns": len(common),
        "question_or_width_mismatches": mismatches,
        "questions_and_widths_equal_on_common_turns": mismatches == 0,
    }


def _faults_are_zero(run: VariantRun) -> bool:
    diagnostics = run.diagnostics
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    exact = diagnostics["exact_evidence_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
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


def _smoke_gates(
    baseline: VariantRun,
    candidate: VariantRun,
    replay: VariantRun,
    paired: Mapping[str, object],
    common_actions: Mapping[str, object],
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
        "candidate_exercises_exact_evidence": (
            int(candidate.diagnostics["exact_evidence_health"]["applied"]) > 0
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
        "questions_and_widths_unchanged_on_common_turns": bool(
            common_actions["questions_and_widths_equal_on_common_turns"]
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
            "response_audit",
            "runtime_network_attempts",
            "retained_agent_bytes",
        )
    }


def _performance(baseline: VariantRun, candidate: VariantRun) -> dict:
    baseline_wall = float(baseline.diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate.diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(
        baseline.diagnostics["respond_latency_ms"]["warm_p95"]
    )
    candidate_p95 = float(
        candidate.diagnostics["respond_latency_ms"]["warm_p95"]
    )
    return {
        "baseline_wall_seconds": baseline_wall,
        "candidate_wall_seconds": candidate_wall,
        "candidate_wall_time_ratio": round(
            _safe_ratio(candidate_wall, baseline_wall),
            6,
        ),
        "baseline_warm_p95_ms": baseline_p95,
        "candidate_warm_p95_ms": candidate_p95,
        "candidate_warm_p95_ratio": round(
            _safe_ratio(candidate_p95, baseline_p95),
            6,
        ),
        "baseline_retained_agent_bytes": int(
            baseline.diagnostics["retained_agent_bytes"]
        ),
        "candidate_retained_agent_bytes": int(
            candidate.diagnostics["retained_agent_bytes"]
        ),
        "candidate_additional_retained_agent_bytes": max(
            0,
            int(candidate.diagnostics["retained_agent_bytes"])
            - int(baseline.diagnostics["retained_agent_bytes"]),
        ),
        "candidate_peak_rss_bytes": int(candidate.diagnostics["peak_rss_bytes"]),
    }


def _locked_phase0_metrics() -> tuple[dict, int]:
    payload = json.loads(
        (REPOSITORY_ROOT / PHASE0_BASELINE_RELATIVE).read_text(encoding="utf-8")
    )
    metrics = payload.get("metrics")
    resources = payload.get("process_resources")
    if not isinstance(metrics, dict) or not isinstance(resources, dict):
        raise RuntimeError("Phase 0 baseline lock is incomplete")
    peak = resources.get("absolute_peak_rss_bytes")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise RuntimeError("Phase 0 peak RSS lock is invalid")
    return dict(metrics), peak


def _full_gates(
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
    folds: Mapping[str, object],
    scenarios: Mapping[str, object],
    performance: Mapping[str, object],
    common_actions: Mapping[str, object],
    *,
    phase0_metrics: Mapping[str, object],
    phase0_peak_rss: int,
) -> dict[str, bool]:
    transitions = paired["transitions"]
    fold_summary = folds["summary"]
    scenario_nonnegative = sum(
        float(value["delta"]["mrr"]) >= 0 for value in scenarios.values()
    )
    baseline_projection = {
        key: baseline.summary[key]
        for key in phase0_metrics
    }
    gates = {
        "shared_backend_baseline_matches_phase0": baseline_projection == phase0_metrics,
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
        "mrr_nonnegative_in_at_least_three_scenarios": scenario_nonnegative >= 3,
        "questions_and_widths_unchanged_on_common_turns": bool(
            common_actions["questions_and_widths_equal_on_common_turns"]
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
        "candidate_peak_rss_within_phase0_plus_64mib": (
            int(performance["candidate_peak_rss_bytes"])
            <= phase0_peak_rss + MAX_ADDITIONAL_RSS_BYTES
        ),
    }
    gates["adopt"] = all(gates.values())
    return gates


def _variant_public(run: VariantRun) -> dict:
    diagnostics = run.diagnostics
    return {
        "official_metrics": run.summary,
        "route_health": diagnostics["route_health"],
        "ranking_health": diagnostics["ranking_health"],
        "exact_evidence_health": diagnostics["exact_evidence_health"],
        "orchestration_health": diagnostics["orchestration_health"],
        "response_audit": diagnostics["response_audit"],
        "runtime_network_attempts": diagnostics["runtime_network_attempts"],
        "retained_agent_bytes": diagnostics["retained_agent_bytes"],
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
        "peak_rss_bytes": diagnostics["peak_rss_bytes"],
    }


def run_phase2_exact_evidence_ablation(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    smoke_only: bool = False,
) -> dict:
    _validate_execution_environment()
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    smoke_indices = select_smoke_indices(samples)
    smoke_samples = [samples[index] for index in smoke_indices]
    catalog_ids, categories, products = catalog_index(catalog)
    phase0_metrics, phase0_peak_rss = _locked_phase0_metrics()

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
    if getattr(backend, "protocol_evidence_capability", None) is not PROTOCOL_EVIDENCE_CAPABILITY:
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
        LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    smoke_baseline = _run_variant(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
        STAGE_A_RANKING_POLICY,
    )
    smoke_replay = _run_variant(
        catalog,
        smoke_samples,
        catalog_ids,
        categories,
        products,
        backend,
        LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    smoke_paired = _paired_statistics(
        smoke_baseline.sessions,
        smoke_candidate.sessions,
    )
    smoke_actions = _common_action_equivalence(smoke_baseline, smoke_candidate)
    smoke_gates = _smoke_gates(
        smoke_baseline,
        smoke_candidate,
        smoke_replay,
        smoke_paired,
        smoke_actions,
    )
    smoke_payload = {
        "sample_count": len(smoke_samples),
        "selection": {
            "cases_per_scenario": SMOKE_PER_SCENARIO,
            "selection_salt_sha256": hashlib.sha256(
                SMOKE_SELECTION_SALT.encode("utf-8")
            ).hexdigest(),
        },
        "baseline": _variant_public(smoke_baseline),
        "candidate": _variant_public(smoke_candidate),
        "metric_delta": _metric_deltas(
            smoke_baseline.summary,
            smoke_candidate.summary,
        ),
        "paired_quality": smoke_paired,
        "common_action_equivalence": smoke_actions,
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
    if smoke_only:
        full_payload = {"status": "not_run", "reason": "smoke_only_requested"}
        decision = {
            "status": "smoke_only",
            "adopt": False,
            "reason": "The smoke run is diagnostic and cannot promote a policy.",
        }
    elif not smoke_gates["authorize_full"]:
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
            LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        )
        gc.collect()
        full_baseline = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            STAGE_A_RANKING_POLICY,
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
        common_actions = _common_action_equivalence(full_baseline, full_candidate)
        performance = _performance(full_baseline, full_candidate)
        full_gates = _full_gates(
            full_baseline,
            full_candidate,
            full_paired,
            folds,
            scenarios,
            performance,
            common_actions,
            phase0_metrics=phase0_metrics,
            phase0_peak_rss=phase0_peak_rss,
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
            "common_action_equivalence": common_actions,
            "performance": performance,
            "decision_gate": full_gates,
        }
        decision = {
            "status": "accepted" if full_gates["adopt"] else "rejected",
            "adopt": full_gates["adopt"],
            "reason": (
                "Every frozen quality, fold, isolation, fault, token, latency, "
                "and memory gate passed."
                if full_gates["adopt"]
                else "At least one frozen full-evaluation promotion gate failed."
            ),
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": (
            "A lexicographic exact-evidence layer can improve reciprocal rank "
            "without changing Phase 13 recall, questions, retrieval, or slate width."
        ),
        "isolated_change": {
            "baseline_ranking": BASELINE_ID,
            "candidate_ranking": CANDIDATE_ID,
            "candidate_set_additions_or_filters": 0,
            "question_policy_changed": False,
            "recommendation_width_changed": False,
            "decision_policy": PROTECTED_DECISION_POLICY.value,
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
            "phase0_peak_rss_reference_bytes": phase0_peak_rss,
        },
        "smoke": smoke_payload,
        "full": full_payload,
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
            "phase0_baseline_sha256": _sha256(
                REPOSITORY_ROOT / PHASE0_BASELINE_RELATIVE
            ),
            "source_sha256": {
                relative: _sha256(REPOSITORY_ROOT / relative)
                for relative in SOURCE_PATHS
            },
        },
    }
    validate_publication(payload)
    return payload


def _validate_execution_environment() -> None:
    mismatches = {
        key: (expected, os.environ.get(key))
        for key, expected in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"single-thread execution environment is not pinned: {mismatches}")


def validate_publication(payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True)
    if any(key in serialized for key in _FORBIDDEN_PUBLICATION_KEYS):
        raise ValueError("aggregate report contains a forbidden raw-data key")
    if _ASIN_RE.search(serialized):
        raise ValueError("aggregate report contains a product identifier")
    privacy = payload.get("privacy")
    if privacy != {
        "aggregate_only": True,
        "labels_used_only_after_agent_replay": True,
        "runtime_received_evaluation_labels": False,
        "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
    }:
        raise ValueError("aggregate report privacy assertions are incomplete")


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite diagnostic: {path}")
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
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_output_path(output: Path, catalog: Path, dataset: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        *(REPOSITORY_ROOT / relative for relative in SOURCE_PATHS),
        REPOSITORY_ROOT / PHASE0_BASELINE_RELATIVE,
    }
    if output.resolve() in protected:
        raise ValueError("output must not overwrite an input or source file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the sequential aggregate-only Phase 2 exact-evidence A/B"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run the frozen smoke/replay gate without the full public A/B",
    )
    arguments = parser.parse_args()
    catalog = Path(arguments.catalog).resolve()
    dataset = Path(arguments.dataset).resolve()
    output = Path(arguments.output).resolve()
    _validate_output_path(output, catalog, dataset)
    payload = run_phase2_exact_evidence_ablation(
        catalog,
        dataset,
        smoke_only=arguments.smoke_only,
    )
    _write_json_exclusive(output, payload)


if __name__ == "__main__":
    main()
