"""One-shot, aggregate-only Phase 15 protocol-utility evaluation.

The harness compares exactly two policies: the protected Phase 13 policy and
the opt-in protocol-utility policy.  Candidate replay, independent candidate
construction, and an explicit protected run on the candidate-capable backend
are exactness checks, not additional ablation arms.

No row, message, target, recommendation, per-session outcome, or belief is
written.  Private paired rows and action traces live only long enough to form
aggregate metrics, confidence intervals, and equality digests.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import evaluator.local_evaluator as evaluator_module
from conversational_search.decision_policy import (
    PROTECTED_DECISION_POLICY,
    PROTOCOL_UTILITY_DECISION_POLICY,
    DecisionPolicy,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.questions import QUESTION_TEXT
from conversational_search.retrieval import (
    PROTOCOL_EVIDENCE_CAPABILITY,
    RetrievalResult,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.run_bm25_rescue_ablations import (
    LATENCY_KEYS,
    ORCHESTRATION_HEALTH_KEYS,
    RANKING_HEALTH_KEYS,
    RESCUE_HEALTH_KEYS,
    SLATE_HEALTH_KEYS,
    _canonical_private_cache_snapshot,
)
from scripts.build_phase15_protocol_robustness_suites import (
    FAMILY_ORDER as ROBUSTNESS_FAMILY_ORDER,
    PARAPHRASE_ORDER as ROBUSTNESS_PARAPHRASE_ORDER,
    PERTURBATION_ORDER as ROBUSTNESS_PERTURBATION_ORDER,
    POPULARITY_ORDER as ROBUSTNESS_POPULARITY_ORDER,
    SCENARIO_ORDER as ROBUSTNESS_SCENARIO_ORDER,
    SELECTION_SALT as ROBUSTNESS_SELECTION_SALT,
    SUITE_CASES_PER_CELL as ROBUSTNESS_SUITE_CASES_PER_CELL,
    SUITE_ORDER as ROBUSTNESS_SUITE_ORDER,
    _aggregate_counts as _robustness_aggregate_counts,
    _case_fingerprint_set_digest,
    _target_set_digest as _robustness_target_set_digest,
    _validate_manifest_privacy as _validate_robustness_manifest_privacy,
    _variant_counts as _robustness_variant_counts,
)
from scripts.run_fusion_ablations import _sha256
from scripts.run_intent_epoch_slate_ablations import (
    _validate_phase13_accounting,
)
from scripts.run_multislot_intent_ablations import (
    _canonical_json,
    _current_max_rss_bytes,
    _deep_size,
    _evaluate_with_deterministic_session_ids,
    _exact_mcnemar_p,
    _fingerprint_set_digest,
    _metric_deltas,
    _overall_summary,
    _paired_statistics,
    _private_digest,
    _safe_ratio,
)
from scripts.run_orchestration_ablations import _lookup_accounting_exact
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_profile_ablations import (
    OVERALL_METRIC_KEYS,
    PROFILE_HEALTH_KEYS,
    TOKEN_USAGE_KEYS,
    _AuditAgent,
    _project_profile_health,
    _route_call_count,
    _validate_variant_accounting,
)
from scripts.run_reranking_ablations import _expected_turns


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 2
SUITE_LOCK_SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase15-decision-aware-evidence-acquisition-v2"
BASELINE_ID = PROTECTED_DECISION_POLICY.value
CANDIDATE_ID = PROTOCOL_UTILITY_DECISION_POLICY.value
IMPLEMENTATION_LOCK_ID = "phase15-protocol-utility-v2-implementation-v1"
SUITE_LOCK_ID = "phase15-protocol-utility-suite-execution-v2"

CONTRACT_RELATIVE = "docs/phase15_experiment_contract.json"
BASELINE_LOCK_RELATIVE = "docs/phase15_baseline_lock.json"
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase15_implementation_lock.json"
RESEARCH_PLAN_RELATIVE = "docs/phase15_research_plan.md"
SUITE_LOCK_RELATIVE = "docs/phase15_protocol_utility_suite_lock.json"
ROBUSTNESS_GENERATOR_RELATIVE = (
    "scripts/build_phase15_protocol_robustness_suites.py"
)
ROBUSTNESS_REFERENCE_RELATIVES = {
    "evaluator": "evaluator/local_evaluator.py",
    "phase14_builder": "scripts/build_phase14_explicit_card_suite.py",
}
ROBUSTNESS_INPUT_TO_LOCKED_SOURCE = {
    "public": "public_confirmation",
    "development": "legacy_development",
    "validation": "legacy_validation",
    "phase14_fresh": "phase14_fresh",
}
PRIOR_SOURCE_KEYS = frozenset(
    {"phase14_fresh", "legacy_development", "legacy_validation"}
)
GENERATED_SOURCE_KEYS = frozenset(ROBUSTNESS_SUITE_ORDER)

BOOTSTRAP_SEED = 150_260_830
BOOTSTRAP_REPLICATES = 10_000
CALIBRATION_BINS = 10
MAX_TOP_K = 10
CANDIDATE_BACKEND_RUN_ORDER = (
    "candidate",
    "protected_reference",
    "candidate_replay",
)
STARTUP_PROBE_ORDERS = (
    (PROTECTED_DECISION_POLICY, PROTOCOL_UTILITY_DECISION_POLICY),
    (PROTOCOL_UTILITY_DECISION_POLICY, PROTECTED_DECISION_POLICY),
)
STARTUP_CANDIDATE_MODULES = (
    "conversational_search.protocol",
    "conversational_search.exact_evidence",
    "conversational_search.utility_planner",
    "conversational_search.decision",
)

FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest "
    "tests.test_protocol tests.test_exact_evidence "
    "tests.test_utility_planner tests.test_decision "
    "tests.test_hybrid_retrieval tests.test_service_protocol_decision "
    "tests.test_protocol_utility_ablations tests.test_service -q"
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
            RESEARCH_PLAN_RELATIVE,
            SUITE_LOCK_RELATIVE,
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

PROTOCOL_OUTCOMES = (
    "applied",
    "unsupported_or_disabled",
    "capability_unavailable",
    "candidate_or_evidence_error",
    "fail_open_evidence",
    "fail_open_no_candidates",
    "fail_open_no_support",
    "fail_open_validation",
)
PROTOCOL_QUESTION_ACTIONS = (*QUESTION_TEXT, "none")
_SAFE_ROUTE_STATUSES = frozenset({"ok", "empty", "skipped"})
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
_ROUTE_HEALTH_KEYS = frozenset(
    {
        "bm25",
        "dense",
        "fallback_turns",
        "candidate_document_calls",
        "protocol_exact_candidate_calls",
        "protocol_candidate_evidence_calls",
    }
)
_ROUTE_REDUNDANCY_HEALTH_KEYS = frozenset(
    {
        "policy",
        "attempts",
        "empty_exact_baseline",
        "single_route_exact_baseline",
        "disjoint_exact_baseline",
        "identical_order_exact_baseline",
        "correction_applied",
        "validation_or_scoring_fallbacks",
    }
)
_INTENT_EPOCH_SLATE_HEALTH_KEYS = frozenset(
    {
        "policy",
        "attempts",
        "empty_exact_baseline",
        "first_slate_exact_baseline",
        "unchanged_signature_exact_baseline",
        "changed_epoch_exact_baseline",
        "same_epoch_history_carried",
        "validation_fallbacks",
        "eligible_prior_shown_total",
    }
)
_RESPONSE_AUDIT_KEYS = frozenset(
    {"response_exceptions", "invalid_api_responses"}
)
_AGGREGATE_HEALTH_KEYS = frozenset(
    {
        "expected_turns",
        "route_health",
        "ranking_health",
        "rescue_health",
        "route_redundancy_health",
        "intent_epoch_slate_health",
        "profile_health",
        "slate_health",
        "orchestration_health",
        "protocol_decision_health",
        "response_audit",
        "runtime_network_attempts",
        "calibration",
        "retained_agent_bytes",
        "protocol_retained_bytes",
        "evaluation_wall_seconds",
        "respond_latency_ms",
    }
)
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
        "phase15_dialog",
        "phase15_card_perturbation",
        "intent_card",
        "behavior",
        "rows",
        "traces",
        "fingerprints",
        "per_session",
        "beliefs",
    }
)


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    """Frozen execution order and gate semantics; data lives in the lock."""

    name: str
    source_keys: tuple[str, ...]
    gate_mode: str
    prerequisites: tuple[str, ...] = ()

    @property
    def output(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase15-{self.name}.json"

    @property
    def attempt(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase15-{self.name}-attempt.json"


SUITES = {
    "fresh_exact": SuiteConfig(
        "fresh_exact",
        ("fresh_exact",),
        "strict_improvement",
    ),
    "paraphrase_fail_open": SuiteConfig(
        "paraphrase_fail_open",
        ("paraphrase_fail_open",),
        "exact_fail_open",
        ("fresh_exact",),
    ),
    "card_perturbed": SuiteConfig(
        "card_perturbed",
        ("card_perturbed",),
        "non_regression",
        ("fresh_exact", "paraphrase_fail_open"),
    ),
    "scenario_balanced": SuiteConfig(
        "scenario_balanced",
        ("scenario_balanced",),
        "non_regression",
        ("fresh_exact", "paraphrase_fail_open", "card_perturbed"),
    ),
    "target_disjoint_development": SuiteConfig(
        "target_disjoint_development",
        ("target_disjoint_development",),
        "strict_confidence",
        (
            "fresh_exact",
            "paraphrase_fail_open",
            "card_perturbed",
            "scenario_balanced",
        ),
    ),
    "target_disjoint_validation": SuiteConfig(
        "target_disjoint_validation",
        ("target_disjoint_validation",),
        "strict_confidence",
        (
            "fresh_exact",
            "paraphrase_fail_open",
            "card_perturbed",
            "scenario_balanced",
            "target_disjoint_development",
        ),
    ),
    "public_confirmation": SuiteConfig(
        "public_confirmation",
        ("public_confirmation",),
        "comparison_only",
        (
            "fresh_exact",
            "paraphrase_fail_open",
            "card_perturbed",
            "scenario_balanced",
            "target_disjoint_development",
            "target_disjoint_validation",
        ),
    ),
}


@dataclass(slots=True)
class VariantRun:
    """Private rows are deliberately excluded from every serialized projection."""

    summary: dict
    sessions: list[dict]
    diagnostics: dict
    evaluator_digest: str
    behavior_digest: str
    private_digest: str


class ProtocolCallAuditRetriever:
    """Count fixed route/protocol operations without retaining candidate IDs."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._bm25: Counter[str] = Counter()
        self._dense: Counter[str] = Counter()
        self._fallbacks = 0
        self._candidate_document_calls = 0
        self._protocol_exact_calls = 0
        self._protocol_evidence_calls = 0

    @property
    def ranking_cache_capability(self) -> object:
        return self._backend.ranking_cache_capability

    @property
    def snapshot_token(self) -> object:
        return self._backend.snapshot_token

    @property
    def requirement_probe_capability(self) -> object:
        return self._backend.requirement_probe_capability

    @property
    def protocol_evidence_capability(self) -> object:
        return self._backend.protocol_evidence_capability

    def search_with_trace(self, *args: object, **kwargs: object) -> RetrievalResult:
        result = self._backend.search_with_trace(*args, **kwargs)
        if not isinstance(result, RetrievalResult):
            raise TypeError("search_with_trace must return RetrievalResult")
        self._bm25[result.trace.bm25_status] += 1
        self._dense[result.trace.dense_status] += 1
        self._fallbacks += int(result.trace.used_fallback)
        return result

    def candidate_documents(self, parent_asins: Sequence[str]) -> tuple:
        self._candidate_document_calls += 1
        return self._backend.candidate_documents(parent_asins)

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: Sequence[str],
        *,
        limit: int,
    ) -> tuple[str, ...]:
        self._protocol_exact_calls += 1
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
        return int(
            self._backend.protocol_exact_constraint_count(category, constraints)
        )

    def protocol_category_exists(self, category: str) -> bool:
        return bool(self._backend.protocol_category_exists(category))

    def candidate_protocol_evidence(
        self,
        parent_asins: Sequence[str],
    ) -> tuple:
        self._protocol_evidence_calls += 1
        return tuple(self._backend.candidate_protocol_evidence(parent_asins))

    def validate(self, expected_searches: int) -> None:
        if sum(self._bm25.values()) != expected_searches:
            raise RuntimeError("BM25 route accounting is incomplete")
        if sum(self._dense.values()) != expected_searches:
            raise RuntimeError("dense route accounting is incomplete")

    def summary(self) -> dict[str, object]:
        return {
            "bm25": dict(sorted(self._bm25.items())),
            "dense": dict(sorted(self._dense.items())),
            "fallback_turns": self._fallbacks,
            "candidate_document_calls": self._candidate_document_calls,
            "protocol_exact_candidate_calls": self._protocol_exact_calls,
            "protocol_candidate_evidence_calls": self._protocol_evidence_calls,
        }


class BeliefCalibrationAudit:
    """Aggregate diagnostics for the fixed, label-free reciprocal-rank belief."""

    def __init__(self, targets: Sequence[str]) -> None:
        if any(not isinstance(target, str) or not target for target in targets):
            raise ValueError("calibration targets must be non-empty strings")
        self._pending = iter(tuple(targets))
        self._target_by_session: dict[str, str] = {}
        self._active_target: str | None = None
        self._observations = 0
        self._target_in_support = 0
        self._brier_total = 0.0
        self._bin_counts = [0] * CALIBRATION_BINS
        self._bin_confidence = [0.0] * CALIBRATION_BINS
        self._bin_correct = [0] * CALIBRATION_BINS
        self._deferred_failure: str | None = None

    def bind_session(self, session_id: str) -> None:
        if session_id in self._target_by_session:
            raise RuntimeError("calibration session was bound twice")
        try:
            self._target_by_session[session_id] = next(self._pending)
        except StopIteration as error:
            raise RuntimeError("calibration target order was exhausted") from error

    @contextlib.contextmanager
    def activate(self, session_id: str) -> Iterator[None]:
        target = self._target_by_session.get(session_id)
        if target is None or self._active_target is not None:
            raise RuntimeError("calibration session context is invalid")
        self._active_target = target
        try:
            yield
        finally:
            self._active_target = None

    def observe(self, result: object) -> None:
        target = self._active_target
        beliefs = getattr(result, "beliefs", None)
        if target is None or not isinstance(beliefs, tuple) or not beliefs:
            return
        pairs: list[tuple[str, float]] = []
        for belief in beliefs:
            parent_asin = getattr(belief, "parent_asin", None)
            weight = getattr(belief, "weight", None)
            if (
                not isinstance(parent_asin, str)
                or isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(float(weight))
                or float(weight) < 0.0
            ):
                raise RuntimeError("belief calibration input is invalid")
            pairs.append((parent_asin, float(weight)))
        total = sum(weight for _, weight in pairs)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("belief calibration mass is not normalized")
        target_probability = sum(
            weight for parent_asin, weight in pairs if parent_asin == target
        )
        self._target_in_support += int(target_probability > 0.0)
        self._brier_total += (
            1.0
            - 2.0 * target_probability
            + sum(weight * weight for _, weight in pairs)
        )
        predicted_id, confidence = max(pairs, key=lambda item: item[1])
        bin_index = min(
            CALIBRATION_BINS - 1,
            int(confidence * CALIBRATION_BINS),
        )
        self._bin_counts[bin_index] += 1
        self._bin_confidence[bin_index] += confidence
        self._bin_correct[bin_index] += int(predicted_id == target)
        self._observations += 1

    def summary(self) -> dict[str, object]:
        observations = self._observations
        mean_confidence: list[float] = []
        accuracy: list[float] = []
        ece = 0.0
        for count, confidence_sum, correct in zip(
            self._bin_counts,
            self._bin_confidence,
            self._bin_correct,
        ):
            average_confidence = confidence_sum / count if count else 0.0
            average_accuracy = correct / count if count else 0.0
            mean_confidence.append(round(average_confidence, 9))
            accuracy.append(round(average_accuracy, 9))
            if observations:
                ece += (
                    count
                    / observations
                    * abs(average_accuracy - average_confidence)
                )
        return {
            "method": "fixed_reciprocal_rank_uncalibrated_diagnostic_only",
            "observations": observations,
            "target_in_support": self._target_in_support,
            "mean_multiclass_brier": round(
                self._brier_total / observations if observations else 0.0,
                9,
            ),
            "ece_10": round(ece, 9),
            "bin_counts": list(self._bin_counts),
            "bin_mean_confidence": mean_confidence,
            "bin_accuracy": accuracy,
            "learned_or_fitted_calibration": False,
            "used_as_promotion_threshold": False,
        }

    def defer_failure(self, error: BaseException) -> None:
        """Remember only a bounded error class; never retain labels or rows."""

        if self._deferred_failure is None:
            self._deferred_failure = type(error).__name__[:80]

    def raise_deferred_failure(self) -> None:
        if self._deferred_failure is not None:
            raise RuntimeError(
                "belief calibration audit failed outside evaluator: "
                + self._deferred_failure
            )

    def clear(self) -> None:
        self._pending = iter(())
        self._target_by_session.clear()
        self._active_target = None
        self._deferred_failure = None


class ProtocolAuditAgent(_AuditAgent):
    """Bind evaluator order to private calibration without label leakage."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        calibration: BeliefCalibrationAudit | None,
        catalog_ids: set[str],
    ) -> None:
        super().__init__(delegate)
        self._calibration = calibration
        self._catalog_ids = catalog_ids
        self._response_exceptions = 0
        self._invalid_responses = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self._calibration is not None:
            self._calibration.bind_session(session_id)
        super().reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            if self._calibration is None:
                response = super().respond(
                    session_id,
                    user_message,
                    turn,
                    top_k,
                )
            else:
                with self._calibration.activate(session_id):
                    response = super().respond(
                        session_id,
                        user_message,
                        turn,
                        top_k,
                    )
        except Exception:
            self._response_exceptions += 1
            raise
        if not self._response_is_valid(response, top_k):
            self._invalid_responses += 1
        return response

    def _response_is_valid(self, response: object, top_k: int) -> bool:
        if not isinstance(response, dict) or not isinstance(
            response.get("message"), str
        ):
            return False
        ask_attribute = response.get("ask_attribute")
        if ask_attribute is not None and ask_attribute not in QUESTION_TEXT:
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
        if not isinstance(usage, dict):
            return False
        return all(
            type(usage.get(key)) is int and int(usage[key]) >= 0
            for key in ("prompt_tokens", "completion_tokens")
        )

    @property
    def response_audit(self) -> dict[str, int]:
        return {
            "response_exceptions": self._response_exceptions,
            "invalid_api_responses": self._invalid_responses,
        }


class RuntimeNetworkAudit:
    """Count and deny socket connections during evaluator-visible turns."""

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


@contextlib.contextmanager
def _capture_belief_calibration(
    audit: BeliefCalibrationAudit | None,
) -> Iterator[None]:
    if audit is None:
        yield
        return
    exact_evidence_module = importlib.import_module(
        "conversational_search.exact_evidence"
    )
    original = exact_evidence_module.rank_exact_evidence

    def wrapped(*args: object, **kwargs: object) -> object:
        result = original(*args, **kwargs)
        try:
            audit.observe(result)
        except Exception as error:
            audit.defer_failure(error)
        return result

    exact_evidence_module.rank_exact_evidence = wrapped  # type: ignore[assignment]
    try:
        yield
    finally:
        exact_evidence_module.rank_exact_evidence = original


def _dialog_spec(sample: Mapping[str, object]) -> dict[str, object] | None:
    raw = sample.get("phase15_dialog")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuntimeError("phase15_dialog must be an object")
    mode = raw.get("mode")
    initial = raw.get("initial_message")
    shapes = raw.get("reply_shapes")
    expected_shapes = {
        "disclosure",
        "boundary_decline",
        "no_additional",
        "need_attribute",
        "override",
    }
    if (
        mode not in {"canonical_v1", "paraphrase_fail_open_v1"}
        or not isinstance(initial, str)
        or not initial
        or not isinstance(shapes, dict)
        or set(shapes) != expected_shapes
        or any(not isinstance(value, str) or not value for value in shapes.values())
    ):
        raise RuntimeError("phase15_dialog schema drifted")
    return raw


@contextlib.contextmanager
def _phase15_dialog_surface() -> Iterator[None]:
    """Replay locked initial/reply surfaces while retaining evaluator semantics."""

    original_initial = evaluator_module.initial_message
    original_reply = evaluator_module.customer_reply

    def initial(sample: dict, category: str, disclosed: set[str]) -> str:
        spec = _dialog_spec(sample)
        if spec is None:
            return original_initial(sample, category, disclosed)
        hard = sample["intent_card"].get("hard_constraints") or []
        if sample.get("scenario_type") == "buying" and hard:
            disclosed.add(str(hard[0]))
        return str(spec["initial_message"])

    def reply(
        sample: dict,
        ask_attribute: object,
        disclosed: set[str],
        boundary_used: bool,
    ) -> tuple[str, bool]:
        spec = _dialog_spec(sample)
        if spec is None or spec.get("mode") == "canonical_v1":
            return original_reply(sample, ask_attribute, disclosed, boundary_used)
        before = set(disclosed)
        canonical, next_boundary = original_reply(
            sample,
            ask_attribute,
            disclosed,
            boundary_used,
        )
        attribute = ask_attribute if isinstance(ask_attribute, str) else "other"
        constraints = [
            *[
                str(value)
                for value in sample["intent_card"].get("hard_constraints", [])
            ],
            *[
                str(value)
                for value in sample["intent_card"].get("soft_preferences", [])
            ],
        ]
        new_values = [value for value in constraints if value not in before and value in disclosed]
        if next_boundary and not boundary_used:
            shape = "boundary_decline"
        elif not isinstance(ask_attribute, str):
            shape = "need_attribute"
        elif new_values:
            shape = "disclosure"
        else:
            shape = "no_additional"
        template = spec["reply_shapes"][shape]  # type: ignore[index]
        try:
            transformed = str(template).format(
                attribute=attribute,
                values="; ".join(new_values),
                value=new_values[0] if new_values else "",
            )
        except (IndexError, KeyError, ValueError) as error:
            raise RuntimeError("phase15 reply shape is invalid") from error
        if not transformed:
            raise RuntimeError("phase15 reply shape rendered empty")
        del canonical
        return transformed, next_boundary

    with patch.object(evaluator_module, "initial_message", initial), patch.object(
        evaluator_module,
        "customer_reply",
        reply,
    ):
        yield


def _prepare_dialog_samples(samples: Sequence[dict]) -> list[dict]:
    """Copy only the override envelope needed for a locked paraphrase."""

    prepared: list[dict] = []
    for sample in samples:
        spec = _dialog_spec(sample)
        if spec is None or spec.get("mode") == "canonical_v1":
            prepared.append(sample)
            continue
        behavior = sample.get("behavior")
        override = behavior.get("override") if isinstance(behavior, dict) else None
        if not isinstance(override, dict):
            prepared.append(sample)
            continue
        new_value = str(override.get("new_value", ""))
        template = spec["reply_shapes"]["override"]  # type: ignore[index]
        try:
            message = str(template).format(
                attribute="other",
                values=new_value,
                value=new_value,
            )
        except (IndexError, KeyError, ValueError) as error:
            raise RuntimeError("phase15 override shape is invalid") from error
        copied_override = {**override, "message": message}
        prepared.append(
            {
                **sample,
                "behavior": {**behavior, "override": copied_override},
            }
        )
    return prepared


def _project_protocol_health(
    health: object,
    *,
    expected_policy: DecisionPolicy,
    expected_turns: int,
) -> dict[str, object]:
    """Validate fixed-cardinality telemetry and return its safe projection."""

    expected_keys = {
        "policy",
        "turns",
        *PROTOCOL_OUTCOMES,
        "question_action_counts",
        "width_action_counts",
        "requested_total",
        "presented_total",
    }
    if not isinstance(health, dict) or set(health) != expected_keys:
        raise RuntimeError("protocol decision health schema drifted")
    if health.get("policy") != expected_policy.value:
        raise RuntimeError("protocol decision policy telemetry drifted")
    scalar_keys = (
        "turns",
        *PROTOCOL_OUTCOMES,
        "requested_total",
        "presented_total",
    )
    if any(
        type(health.get(key)) is not int or int(health[key]) < 0
        for key in scalar_keys
    ):
        raise RuntimeError("protocol decision counter is invalid")
    turns = int(health["turns"])
    if turns != expected_turns:
        raise RuntimeError("protocol decision turn coverage is incomplete")
    if sum(int(health[key]) for key in PROTOCOL_OUTCOMES) != turns:
        raise RuntimeError("protocol outcomes do not partition turns")

    questions = health.get("question_action_counts")
    if not isinstance(questions, dict) or set(questions) != set(
        PROTOCOL_QUESTION_ACTIONS
    ):
        raise RuntimeError("protocol question action schema drifted")
    if any(type(value) is not int or value < 0 for value in questions.values()):
        raise RuntimeError("protocol question action counter is invalid")
    if sum(questions.values()) != turns:
        raise RuntimeError("protocol question actions do not partition turns")

    widths = health.get("width_action_counts")
    expected_widths = {str(width) for width in range(MAX_TOP_K + 1)}
    if not isinstance(widths, dict) or set(widths) != expected_widths:
        raise RuntimeError("protocol width action schema drifted")
    if any(type(value) is not int or value < 0 for value in widths.values()):
        raise RuntimeError("protocol width action counter is invalid")
    if sum(widths.values()) != turns:
        raise RuntimeError("protocol width actions do not partition turns")
    presented_total = int(health["presented_total"])
    if sum(int(width) * int(count) for width, count in widths.items()) != (
        presented_total
    ):
        raise RuntimeError("protocol presented-width accounting drifted")
    requested_total = int(health["requested_total"])
    if not 0 <= presented_total <= requested_total <= turns * MAX_TOP_K:
        raise RuntimeError("protocol requested/presented totals are invalid")

    if expected_policy is PROTECTED_DECISION_POLICY:
        if any(
            int(health[key])
            for key in (
                "turns",
                *PROTOCOL_OUTCOMES,
                "requested_total",
                "presented_total",
            )
        ) or any(questions.values()) or any(widths.values()):
            raise RuntimeError("protected policy performed protocol work")

    planner_decisions = sum(
        int(health[key])
        for key in (
            "applied",
            "fail_open_evidence",
            "fail_open_no_candidates",
            "fail_open_no_support",
            "fail_open_validation",
        )
    )
    return {
        **{key: health[key] for key in ("policy", "turns", *PROTOCOL_OUTCOMES)},
        "planner_decisions": planner_decisions,
        "protocol_mode_turns": int(health["applied"]),
        "fail_open_turns_by_reason_code": {
            key: int(health[key])
            for key in PROTOCOL_OUTCOMES
            if key != "applied"
        },
        "question_action_counts": {
            key: int(questions[key]) for key in PROTOCOL_QUESTION_ACTIONS
        },
        "width_action_counts": {
            str(width): int(widths[str(width)])
            for width in range(MAX_TOP_K + 1)
        },
        "requested_total": requested_total,
        "presented_total": presented_total,
    }


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    candidate_state = tuple(
        value
        for name, value in vars(agent).items()
        if name.startswith("_protocol_")
    )
    retained = (
        getattr(agent, "_sessions", None),
        getattr(agent, "_slates", None),
        getattr(agent, "_profile_priors", None),
        getattr(planner, "_entries", None),
        candidate_state,
    )
    return _deep_size(retained)


def _candidate_private_state(agent: ConversationalSearchAgent) -> tuple:
    return tuple(
        (name, value)
        for name, value in sorted(vars(agent).items())
        if name.startswith("_protocol_")
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
    guarded = ProtocolCallAuditRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        decision_policy=decision_policy,
    )
    evaluation_samples = _prepare_dialog_samples(samples)
    targets = (
        [
            str(sample["ground_truth"]["parent_asin"])
            for sample in evaluation_samples
        ]
        if decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
        else []
    )
    calibration = BeliefCalibrationAudit(targets) if targets else None
    audited = ProtocolAuditAgent(agent, calibration, catalog_ids)
    network_audit = RuntimeNetworkAudit()
    started = time.perf_counter()
    with (
        _capture_belief_calibration(calibration),
        _phase15_dialog_surface(),
        network_audit.deny(),
    ):
        result = _evaluate_with_deterministic_session_ids(
            audited,
            evaluation_samples,
            catalog_ids,
            categories,
            products,
        )
    if calibration is not None:
        calibration.raise_deferred_failure()
    wall_seconds = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    searches = int(agent.orchestration_health["searches"])
    guarded.validate(searches)
    latency = audited.latency_summary()
    if int(latency["count"]) != expected_turns:
        raise RuntimeError("response latency coverage is incomplete")
    protocol_health = _project_protocol_health(
        agent.protocol_decision_health,
        expected_policy=decision_policy,
        expected_turns=(
            expected_turns
            if decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
            else 0
        ),
    )
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
        "protocol_decision_health": protocol_health,
        "response_audit": audited.response_audit,
        "runtime_network_attempts": network_audit.attempts,
        "calibration": (
            calibration.summary()
            if calibration is not None
            else _empty_calibration_summary()
        ),
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "protocol_retained_bytes": (
            _deep_size(_candidate_private_state(agent))
            if decision_policy is PROTOCOL_UTILITY_DECISION_POLICY
            else 0
        ),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }
    if decision_policy is PROTECTED_DECISION_POLICY:
        _validate_variant_accounting(
            diagnostics,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        )
    else:
        _validate_protocol_variant_accounting(diagnostics)
    _validate_phase13_accounting(
        diagnostics,
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("evaluator sessions are unavailable for paired checks")
    behavior = _private_digest(
        audited.action_trace,
        _canonical_private_cache_snapshot(agent),
    )
    private = _private_digest(
        behavior,
        _candidate_private_state(agent),
    )
    if calibration is not None:
        calibration.clear()
    return VariantRun(
        summary=_overall_summary(result),
        sessions=sessions,
        diagnostics=diagnostics,
        evaluator_digest=hashlib.sha256(_canonical_json(result)).hexdigest(),
        behavior_digest=behavior,
        private_digest=private,
    )


def _validate_protocol_variant_accounting(
    diagnostics: Mapping[str, object],
) -> None:
    turns = int(diagnostics["expected_turns"])
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    slate = diagnostics["slate_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    profile = diagnostics["profile_health"]  # type: ignore[assignment]
    protocol = diagnostics["protocol_decision_health"]  # type: ignore[assignment]
    searches = int(orchestration["searches"])  # type: ignore[index]
    if int(orchestration["decisions"]) != turns:  # type: ignore[index]
        raise RuntimeError("orchestration decision coverage is incomplete")
    if searches + int(orchestration["reuses"]) != turns:  # type: ignore[index]
        raise RuntimeError("candidate turns must search or exactly reuse")
    if int(orchestration["skips"]) != 0:  # type: ignore[index]
        raise RuntimeError("the official top-k workload must not skip")
    if not _lookup_accounting_exact(orchestration):  # type: ignore[arg-type]
        raise RuntimeError("cache lookup accounting is inconsistent")
    if _route_call_count(route) != searches:  # type: ignore[arg-type]
        raise RuntimeError("route-call accounting is incomplete")
    dense_statuses = route["dense"]  # type: ignore[index,assignment]
    dense_coverage = sum(
        int(value)
        for value in dense_statuses.values()  # type: ignore[union-attr]
    )
    dense_skips = int(dense_statuses.get("skipped", 0))  # type: ignore[union-attr]
    dense_executions = sum(
        int(dense_statuses.get(status, 0))  # type: ignore[union-attr]
        for status in ("ok", "empty", "error")
    )
    if dense_coverage != searches:
        raise RuntimeError("dense route accounting is incomplete")
    if dense_executions + dense_skips != searches:
        raise RuntimeError("dense executions and planned skips must partition searches")
    if int(dense_statuses.get("error", 0)):  # type: ignore[union-attr]
        raise RuntimeError("dense route errors invalidate Phase 15")
    if int(ranking["attempts"]) != searches:  # type: ignore[index]
        raise RuntimeError("reranker-call accounting is incomplete")
    if int(route["candidate_document_calls"]) != searches:  # type: ignore[index]
        raise RuntimeError("candidate-document accounting is incomplete")
    if int(ranking["successes"]) != searches:  # type: ignore[index]
        raise RuntimeError("every Stage-A call must succeed")
    if int(ranking["failures"]) or int(ranking["unavailable_skips"]):  # type: ignore[index]
        raise RuntimeError("reranker faults invalidate Phase 15")
    zero_width_turns = int(protocol["width_action_counts"]["0"])  # type: ignore[index]
    if int(slate["attempts"]) + zero_width_turns != turns:  # type: ignore[index]
        raise RuntimeError("candidate slate/withholding coverage is incomplete")
    if int(slate["successes"]) != int(slate["attempts"]):  # type: ignore[index]
        raise RuntimeError("candidate slate success coverage is incomplete")
    if int(slate["failures"]):  # type: ignore[index]
        raise RuntimeError("slate faults invalidate Phase 15")
    if int(route["fallback_turns"]):  # type: ignore[index]
        raise RuntimeError("retrieval fallbacks invalidate Phase 15")
    if int(profile["eligible_stage_a_attempts"]) > searches:  # type: ignore[index]
        raise RuntimeError("profile attempts exceed Stage-A attempts")


def _empty_calibration_summary() -> dict[str, object]:
    return {
        "method": "fixed_reciprocal_rank_uncalibrated_diagnostic_only",
        "observations": 0,
        "target_in_support": 0,
        "mean_multiclass_brier": 0.0,
        "ece_10": 0.0,
        "bin_counts": [0] * CALIBRATION_BINS,
        "bin_mean_confidence": [0.0] * CALIBRATION_BINS,
        "bin_accuracy": [0.0] * CALIBRATION_BINS,
        "learned_or_fitted_calibration": False,
        "used_as_promotion_threshold": False,
    }


def _calibration_is_valid(value: object, *, expected_turns: int) -> bool:
    if not isinstance(value, dict) or set(value) != set(
        _empty_calibration_summary()
    ):
        return False
    try:
        observations = int(value["observations"])
        target_in_support = int(value["target_in_support"])
        brier = float(value["mean_multiclass_brier"])
        ece = float(value["ece_10"])
    except (TypeError, ValueError):
        return False
    vectors = (
        value["bin_counts"],
        value["bin_mean_confidence"],
        value["bin_accuracy"],
    )
    if any(not isinstance(vector, list) or len(vector) != CALIBRATION_BINS for vector in vectors):
        return False
    counts = vectors[0]
    if any(type(count) is not int or count < 0 for count in counts):
        return False
    if sum(counts) != observations:
        return False
    if not 0 <= target_in_support <= observations <= expected_turns:
        return False
    if not (math.isfinite(brier) and 0.0 <= brier <= 2.0):
        return False
    if not (math.isfinite(ece) and 0.0 <= ece <= 1.0):
        return False
    for vector in vectors[1:]:
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in vector
        ):
            return False
    return (
        value.get("learned_or_fitted_calibration") is False
        and value.get("used_as_promotion_threshold") is False
    )


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
            "protocol_decision_health",
            "response_audit",
            "runtime_network_attempts",
            "calibration",
            "retained_agent_bytes",
            "protocol_retained_bytes",
        )
    }


def _aggregate_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        **_deterministic_health(diagnostics),
        "evaluation_wall_seconds": diagnostics["evaluation_wall_seconds"],
        "respond_latency_ms": diagnostics["respond_latency_ms"],
    }


def _call_accounting(diagnostics: Mapping[str, object]) -> dict[str, int]:
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    return {
        "searches": int(orchestration["searches"]),  # type: ignore[index]
        "bm25_route_calls": _route_call_count(route),  # type: ignore[arg-type]
        "dense_route_statuses": sum(
            int(value) for value in route["dense"].values()  # type: ignore[index,union-attr]
        ),
        "dense_route_executions": sum(
            int(route["dense"].get(status, 0))  # type: ignore[index,union-attr]
            for status in ("ok", "empty", "error")
        ),
        "dense_route_skips": int(  # type: ignore[index,union-attr]
            route["dense"].get("skipped", 0)
        ),
        "candidate_document_calls": int(route["candidate_document_calls"]),  # type: ignore[index]
        "stage_a_attempts": int(ranking["attempts"]),  # type: ignore[index]
        "protocol_exact_candidate_calls": int(
            route["protocol_exact_candidate_calls"]  # type: ignore[index]
        ),
        "protocol_candidate_evidence_calls": int(
            route["protocol_candidate_evidence_calls"]  # type: ignore[index]
        ),
    }


def _tokens_are_zero(summary: Mapping[str, object]) -> bool:
    usage = summary.get("reported_token_usage")
    return isinstance(usage, dict) and all(
        type(usage.get(key)) is int and usage[key] == 0
        for key in TOKEN_USAGE_KEYS
    )


def _faults_are_zero(diagnostics: Mapping[str, object]) -> bool:
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    rescue = diagnostics["rescue_health"]  # type: ignore[assignment]
    redundancy = diagnostics["route_redundancy_health"]  # type: ignore[assignment]
    novelty = diagnostics["intent_epoch_slate_health"]  # type: ignore[assignment]
    profile = diagnostics["profile_health"]  # type: ignore[assignment]
    slate = diagnostics["slate_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    protocol = diagnostics["protocol_decision_health"]  # type: ignore[assignment]
    response_audit = diagnostics["response_audit"]  # type: ignore[assignment]
    return (
        int(route["fallback_turns"]) == 0  # type: ignore[index]
        and set(route["bm25"]).issubset(_SAFE_ROUTE_STATUSES)  # type: ignore[arg-type,index]
        and set(route["dense"]).issubset(_SAFE_ROUTE_STATUSES)  # type: ignore[arg-type,index]
        and int(ranking["failures"]) == 0  # type: ignore[index]
        and int(ranking["unavailable_skips"]) == 0  # type: ignore[index]
        and int(rescue["attempts"]) == 0  # type: ignore[index]
        and int(redundancy["validation_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(novelty["validation_fallbacks"]) == 0  # type: ignore[index]
        and int(profile["parsing_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(slate["failures"]) == 0  # type: ignore[index]
        and int(orchestration["fault_invalidations"]) == 0  # type: ignore[index]
        and int(orchestration["store_rejections"]) == 0  # type: ignore[index]
        and int(protocol["capability_unavailable"]) == 0  # type: ignore[index]
        and int(protocol["candidate_or_evidence_error"]) == 0  # type: ignore[index]
        and int(protocol["fail_open_validation"]) == 0  # type: ignore[index]
        and int(response_audit["response_exceptions"]) == 0  # type: ignore[index]
        and int(response_audit["invalid_api_responses"]) == 0  # type: ignore[index]
        and int(diagnostics["runtime_network_attempts"]) == 0
    )


def _performance_summary(
    baseline: VariantRun,
    candidate: VariantRun,
) -> dict[str, float | int]:
    baseline_wall = float(baseline.diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate.diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(
        baseline.diagnostics["respond_latency_ms"]["warm_p95"]
    )
    candidate_p95 = float(
        candidate.diagnostics["respond_latency_ms"]["warm_p95"]
    )
    baseline_bytes = int(baseline.diagnostics["retained_agent_bytes"])
    candidate_bytes = int(candidate.diagnostics["retained_agent_bytes"])
    sessions = max(1, int(candidate.summary["sample_count"]))
    additional_bytes = max(0, candidate_bytes - baseline_bytes)
    protocol_bytes = int(candidate.diagnostics["protocol_retained_bytes"])
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
        "candidate_additional_retained_agent_bytes": additional_bytes,
        "candidate_protocol_retained_bytes": protocol_bytes,
        "candidate_additional_retained_session_bytes": math.ceil(
            protocol_bytes / sessions
        ),
    }


def _startup_summary(
    baseline_observations: Sequence[Mapping[str, object]],
    candidate_observations: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(baseline_observations) != 2 or len(candidate_observations) != 2:
        raise RuntimeError("startup probe must measure both policies twice")

    baseline_seconds = min(
        float(observation["elapsed_seconds"])
        for observation in baseline_observations
    )
    candidate_seconds = max(
        float(observation["elapsed_seconds"])
        for observation in candidate_observations
    )
    baseline_rss = min(
        int(observation["max_rss_bytes"])
        for observation in baseline_observations
    )
    candidate_rss = max(
        int(observation["max_rss_bytes"])
        for observation in candidate_observations
    )
    baseline_post_warm_rss = min(
        int(observation["post_warm_max_rss_bytes"])
        for observation in baseline_observations
    )
    candidate_post_warm_rss = max(
        int(observation["post_warm_max_rss_bytes"])
        for observation in candidate_observations
    )
    baseline_retained = min(
        int(observation["empty_retained_bytes"])
        for observation in baseline_observations
    )
    candidate_retained = max(
        int(observation["empty_retained_bytes"])
        for observation in candidate_observations
    )
    if (
        not math.isfinite(baseline_seconds)
        or not math.isfinite(candidate_seconds)
        or baseline_seconds < 0.0
        or candidate_seconds < 0.0
        or min(
            baseline_rss,
            candidate_rss,
            baseline_post_warm_rss,
            candidate_post_warm_rss,
            baseline_retained,
            candidate_retained,
        )
        < 0
    ):
        raise RuntimeError("startup probe observation is invalid")
    return {
        "accounting": "max_candidate_vs_min_protected_across_both_orders",
        "baseline_probe_count": len(baseline_observations),
        "candidate_probe_count": len(candidate_observations),
        "baseline_startup_seconds": round(baseline_seconds, 6),
        "candidate_startup_seconds": round(candidate_seconds, 6),
        "candidate_startup_time_ratio": round(
            _safe_ratio(candidate_seconds, baseline_seconds),
            6,
        ),
        "baseline_startup_rss_bytes": baseline_rss,
        "candidate_startup_rss_bytes": candidate_rss,
        "candidate_additional_startup_rss_bytes": max(
            0,
            candidate_rss - baseline_rss,
        ),
        "baseline_post_warm_peak_rss_bytes": baseline_post_warm_rss,
        "candidate_post_warm_peak_rss_bytes": candidate_post_warm_rss,
        "candidate_additional_post_warm_peak_rss_bytes": max(
            0,
            candidate_post_warm_rss - baseline_post_warm_rss,
        ),
        "baseline_empty_retained_bytes": baseline_retained,
        "candidate_empty_retained_bytes": candidate_retained,
    }


def _quality_gates(
    config: SuiteConfig,
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    if config.gate_mode == "comparison_only":
        return {"public_metrics_are_comparison_only": True}
    baseline_metrics = baseline.summary
    candidate_metrics = candidate.summary
    transitions = paired["transitions"]  # type: ignore[assignment]
    bootstrap = paired["bootstrap"]  # type: ignore[assignment]
    common = {
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
            float(candidate_metrics["mttc"])
            <= float(baseline_metrics["mttc"])
        ),
        "candidate_technical_score_not_below_baseline": (
            float(candidate_metrics["recommended_technical_score"])
            >= float(baseline_metrics["recommended_technical_score"])
        ),
    }
    lower = float(bootstrap["lower_95"])  # type: ignore[index]
    if config.gate_mode == "exact_fail_open":
        return {
            **common,
            "candidate_metrics_exactly_equal_protected": (
                candidate_metrics == baseline_metrics
            ),
            "paired_mean_delta_is_exactly_zero": (
                float(paired["mean_utility_delta"]) == 0.0
            ),
            "paired_bootstrap_interval_is_exactly_zero": (
                lower == 0.0
                and float(bootstrap["upper_95"]) == 0.0  # type: ignore[index]
            ),
        }
    gates = {
        **common,
        "paired_bootstrap_lower_95_not_below_zero": lower >= 0.0,
    }
    if config.gate_mode in {"strict_improvement", "strict_confidence"}:
        gates.update(
            {
                "candidate_technical_score_strictly_improves": (
                    float(candidate_metrics["recommended_technical_score"])
                    > float(baseline_metrics["recommended_technical_score"])
                ),
                "candidate_mrr_or_mttc_strictly_improves": (
                    float(candidate_metrics["mrr"])
                    > float(baseline_metrics["mrr"])
                    or float(candidate_metrics["mttc"])
                    < float(baseline_metrics["mttc"])
                ),
            }
        )
    if config.gate_mode == "strict_confidence":
        gates["paired_bootstrap_lower_95_strictly_above_zero"] = lower > 0.0
    return gates


def _operational_limits(contract: Mapping[str, object]) -> dict[str, object]:
    raw = contract.get("operational_gates")
    numeric_keys = {
        "candidate_warm_p95_ratio_at_most",
        "candidate_wall_time_ratio_at_most",
        "candidate_startup_ratio_at_most",
        "candidate_additional_startup_rss_bytes_at_most",
        "candidate_additional_post_warm_peak_rss_bytes_at_most",
        "candidate_additional_retained_session_bytes_at_most",
    }
    architecture = {
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
    if not isinstance(raw, dict) or not numeric_keys.issubset(raw):
        raise RuntimeError("Phase 15 operational contract is incomplete")
    limits: dict[str, object] = {}
    for key in numeric_keys:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("Phase 15 operational limit is invalid")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise RuntimeError("Phase 15 operational limit is invalid")
        limits[key] = value
    if any(raw.get(key) != expected for key, expected in architecture.items()):
        raise RuntimeError("Phase 15 operational architecture contract drifted")
    limits.update(architecture)
    return limits


def _run_exactness(
    first: VariantRun,
    second: VariantRun,
) -> dict[str, bool]:
    return {
        "evaluator_payload_equal": first.evaluator_digest == second.evaluator_digest,
        "response_state_slate_cache_equal": (
            first.behavior_digest == second.behavior_digest
        ),
        "complete_private_state_equal": first.private_digest == second.private_digest,
        "aggregate_health_equal": (
            _deterministic_health(first.diagnostics)
            == _deterministic_health(second.diagnostics)
        ),
    }


def _build_gates(
    config: SuiteConfig,
    baseline: VariantRun,
    protected_reference: VariantRun,
    candidate: VariantRun,
    replay: VariantRun,
    independent: VariantRun,
    paired: Mapping[str, object],
    performance: Mapping[str, object],
    startup: Mapping[str, object],
    limits: Mapping[str, object],
    *,
    locks_revalidated: bool,
    suite_evidence_valid: bool,
    privacy_valid: bool,
) -> dict[str, bool]:
    runs = (baseline, protected_reference, candidate, replay, independent)
    calls = [_call_accounting(run.diagnostics) for run in runs]
    protected_phase13_search = all(
        item["searches"]
        == item["bm25_route_calls"]
        == item["dense_route_executions"]
        == item["dense_route_statuses"]
        == item["candidate_document_calls"]
        == item["stage_a_attempts"]
        and item["dense_route_skips"] == 0
        for item in calls[:2]
    )
    candidate_route_partition = all(
        item["searches"]
        == item["bm25_route_calls"]
        == item["dense_route_statuses"]
        == item["candidate_document_calls"]
        == item["stage_a_attempts"]
        and item["dense_route_executions"] + item["dense_route_skips"]
        == item["searches"]
        and item["dense_route_executions"] <= item["searches"]
        for item in calls[2:]
    )
    conditional_dense_behavior = True
    if config.name == "fresh_exact":
        conditional_dense_behavior = all(
            item["dense_route_skips"] > 0 for item in calls[2:]
        )
    elif config.gate_mode == "exact_fail_open":
        conditional_dense_behavior = all(
            item["dense_route_skips"] == 0 for item in calls[2:]
        )
    protected_protocol_calls_are_zero = all(
        item["protocol_exact_candidate_calls"] == 0
        and item["protocol_candidate_evidence_calls"] == 0
        for item in calls[:2]
    )
    candidate_protocol = candidate.diagnostics["protocol_decision_health"]
    protocol_behavior_gate = True
    if config.gate_mode == "strict_improvement":
        protocol_behavior_gate = (
            int(candidate_protocol["applied"]) > 0
            and int(candidate_protocol["planner_decisions"]) > 0
        )
    elif config.gate_mode == "exact_fail_open":
        protocol_behavior_gate = (
            int(candidate_protocol["applied"]) == 0
            and int(candidate_protocol["unsupported_or_disabled"])
            == int(candidate_protocol["turns"])
            and candidate.evaluator_digest == baseline.evaluator_digest
            and candidate.behavior_digest == baseline.behavior_digest
        )

    protected_exactness = _run_exactness(baseline, protected_reference)
    replay_exactness = _run_exactness(candidate, replay)
    independent_exactness = _run_exactness(candidate, independent)
    gates = {
        **_quality_gates(config, baseline, candidate, paired),
        "protected_explicit_policy_is_exact_phase13": all(
            protected_exactness.values()
        ),
        "candidate_replay_is_exact": all(replay_exactness.values()),
        "independent_candidate_construction_is_exact": all(
            independent_exactness.values()
        ),
        "baseline_and_candidate_faults_are_zero": all(
            _faults_are_zero(run.diagnostics) for run in runs
        ),
        "protected_phase13_executes_one_bm25_dense_document_and_stage_a_call_per_search": (
            protected_phase13_search
        ),
        "candidate_dense_executions_and_planned_skips_partition_searches": (
            candidate_route_partition
        ),
        "candidate_dense_execution_is_bounded_to_one_per_search": all(
            item["dense_route_executions"] <= item["searches"]
            for item in calls[2:]
        ),
        "conditional_dense_behavior_matches_suite_world": (
            conditional_dense_behavior
        ),
        "protected_policy_performs_zero_protocol_route_or_evidence_calls": (
            protected_protocol_calls_are_zero
        ),
        "protocol_action_telemetry_is_fixed_and_complete": (
            int(candidate_protocol["turns"])
            == int(candidate.diagnostics["expected_turns"])
        ),
        "suite_specific_protocol_behavior_passes": protocol_behavior_gate,
        "belief_brier_and_ece_are_aggregate_unfitted_diagnostics": (
            _calibration_is_valid(
                candidate.diagnostics["calibration"],
                expected_turns=int(candidate.diagnostics["expected_turns"]),
            )
        ),
        "all_variants_report_zero_model_and_api_tokens": all(
            _tokens_are_zero(run.summary) for run in runs
        ),
        "zero_copy_compact_card_only_operational_contract_is_locked": (
            limits["no_duplicate_catalog_copy"] is True
            and limits["candidate_metadata_is_compact_card_only"] is True
            and limits["candidate_metadata_fetches_full_fts_documents"] is False
            and limits["additional_external_model_or_api_calls_per_turn"] == 0
            and limits["model_and_api_tokens"] == 0
            and limits["response_exceptions"] == 0
            and limits["invalid_api_responses"] == 0
            and limits["runtime_network_attempts"] == 0
            and limits["faults"] == 0
            and limits["replay_exact"] is True
            and limits["independent_construction_exact"] is True
        ),
        "candidate_warm_p95_ratio_within_contract": (
            float(performance["candidate_warm_p95_ratio"])
            <= float(limits["candidate_warm_p95_ratio_at_most"])
        ),
        "candidate_wall_time_ratio_within_contract": (
            float(performance["candidate_wall_time_ratio"])
            <= float(limits["candidate_wall_time_ratio_at_most"])
        ),
        "candidate_startup_ratio_within_contract": (
            float(startup["candidate_startup_time_ratio"])
            <= float(limits["candidate_startup_ratio_at_most"])
        ),
        "candidate_additional_startup_rss_within_contract": (
            int(startup["candidate_additional_startup_rss_bytes"])
            <= int(limits["candidate_additional_startup_rss_bytes_at_most"])
        ),
        "candidate_additional_post_warm_peak_rss_within_contract": (
            int(startup["candidate_additional_post_warm_peak_rss_bytes"])
            <= int(
                limits[
                    "candidate_additional_post_warm_peak_rss_bytes_at_most"
                ]
            )
        ),
        "candidate_additional_retained_session_state_within_contract": (
            int(performance["candidate_additional_retained_session_bytes"])
            <= int(
                limits[
                    "candidate_additional_retained_session_bytes_at_most"
                ]
            )
        ),
        "implementation_and_suite_locks_revalidated_after_all_variants": (
            locks_revalidated
        ),
        "suite_hash_target_disjointness_and_order_evidence_valid": (
            suite_evidence_valid
        ),
        "aggregate_publication_privacy_valid": privacy_valid,
    }
    if config.gate_mode == "comparison_only":
        gates["public_confirmation_complete"] = all(
            value
            for key, value in gates.items()
            if key != "public_metrics_are_comparison_only"
        )
        gates["advance"] = False
    else:
        gates["advance"] = all(gates.values())
    return gates


def _target_set_digest(values: set[str]) -> str:
    fingerprints = {
        hashlib.sha256(value.encode("utf-8")).digest() for value in values
    }
    return _fingerprint_set_digest(fingerprints)


def _validate_locked_source_entry(value: object) -> dict:
    keys = {
        "path",
        "sha256",
        "rows",
        "case_fingerprint_set_sha256",
        "target_fingerprint_set_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("Phase 15 source lock entry drifted")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise RuntimeError("Phase 15 source path is invalid")
    if type(value.get("rows")) is not int or value["rows"] <= 0:
        raise RuntimeError("Phase 15 source row count is invalid")
    for key in (
        "sha256",
        "case_fingerprint_set_sha256",
        "target_fingerprint_set_sha256",
    ):
        if (
            not isinstance(value.get(key), str)
            or _HEX_SHA256_RE.fullmatch(value[key]) is None
        ):
            raise RuntimeError("Phase 15 source digest is invalid")
    return value


def _required_target_disjoint_pairs(
    source_names: Iterable[str],
) -> set[str]:
    """Require disjointness only where a newly generated suite participates.

    The legacy development, validation, Phase 14, and public sources are
    exclusion inputs.  They may overlap one another; the new Phase 15 suites
    must be mutually disjoint and disjoint from their entire union.
    """

    ordered = sorted(source_names)
    return {
        f"{left}|{right}"
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
        if left in GENERATED_SOURCE_KEYS or right in GENERATED_SOURCE_KEYS
    }


def _validate_suite_lock_payload(payload: object) -> dict:
    expected_top = {
        "schema_version",
        "lock_id",
        "experiment_id",
        "status",
        "catalog_sha256",
        "ordered_gates",
        "public_confirmation_is_last",
        "public_metrics_are_comparison_only",
        "robustness_manifest",
        "generator_source_sha256",
        "gate_sources",
        "sources",
        "prior_sources",
        "target_disjointness",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top:
        raise RuntimeError("Phase 15 suite lock schema drifted")
    if payload.get("schema_version") != SUITE_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 15 suite lock schema")
    if payload.get("lock_id") != SUITE_LOCK_ID:
        raise RuntimeError("unexpected Phase 15 suite lock identity")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("Phase 15 suite experiment identity drifted")
    if payload.get("status") != "locked_before_phase15_candidate_execution":
        raise RuntimeError("Phase 15 suites are not frozen")
    catalog_sha256 = payload.get("catalog_sha256")
    if (
        not isinstance(catalog_sha256, str)
        or _HEX_SHA256_RE.fullmatch(catalog_sha256) is None
    ):
        raise RuntimeError("Phase 15 catalog lock is invalid")
    if payload.get("ordered_gates") != list(SUITES):
        raise RuntimeError("Phase 15 suite execution order drifted")
    if payload.get("public_confirmation_is_last") is not True:
        raise RuntimeError("public confirmation must remain last")
    if payload.get("public_metrics_are_comparison_only") is not True:
        raise RuntimeError("public metrics must remain comparison-only")

    gate_sources = payload.get("gate_sources")
    if not isinstance(gate_sources, dict) or set(gate_sources) != set(SUITES):
        raise RuntimeError("Phase 15 gate-source mapping is incomplete")
    for name, config in SUITES.items():
        if gate_sources.get(name) != list(config.source_keys):
            raise RuntimeError("Phase 15 gate-source mapping drifted")
    expected_sources = {
        key for config in SUITES.values() for key in config.source_keys
    }
    sources = payload.get("sources")
    if not isinstance(sources, dict) or set(sources) != expected_sources:
        raise RuntimeError("Phase 15 suite source lock is incomplete")
    for value in sources.values():
        _validate_locked_source_entry(value)
    prior_sources = payload.get("prior_sources")
    if not isinstance(prior_sources, dict) or set(prior_sources) != set(
        PRIOR_SOURCE_KEYS
    ):
        raise RuntimeError("Phase 15 prior-source lock is incomplete")
    for value in prior_sources.values():
        _validate_locked_source_entry(value)

    robustness = payload.get("robustness_manifest")
    if (
        not isinstance(robustness, dict)
        or set(robustness) != {"path", "sha256"}
        or not isinstance(robustness.get("path"), str)
        or not robustness["path"]
        or not isinstance(robustness.get("sha256"), str)
        or _HEX_SHA256_RE.fullmatch(robustness["sha256"]) is None
    ):
        raise RuntimeError("Phase 15 robustness-manifest lock is invalid")
    generator_hashes = payload.get("generator_source_sha256")
    expected_generator_paths = {
        ROBUSTNESS_GENERATOR_RELATIVE,
        *ROBUSTNESS_REFERENCE_RELATIVES.values(),
    }
    if (
        not isinstance(generator_hashes, dict)
        or set(generator_hashes) != expected_generator_paths
    ):
        raise RuntimeError("Phase 15 generator source lock is incomplete")
    for relative, digest in generator_hashes.items():
        if (
            not isinstance(relative, str)
            or not relative
            or not isinstance(digest, str)
            or _HEX_SHA256_RE.fullmatch(digest) is None
        ):
            raise RuntimeError("Phase 15 generator source lock is invalid")

    disjoint = payload.get("target_disjointness")
    disjoint_source_names = expected_sources | set(PRIOR_SOURCE_KEYS)
    pairs = _required_target_disjoint_pairs(disjoint_source_names)
    if (
        not isinstance(disjoint, dict)
        or set(disjoint) != {
            "generated_sources_are_mutually_and_forbidden_disjoint",
            "pairwise_overlap_counts",
        }
        or disjoint.get(
            "generated_sources_are_mutually_and_forbidden_disjoint"
        )
        is not True
        or not isinstance(disjoint.get("pairwise_overlap_counts"), dict)
        or set(disjoint["pairwise_overlap_counts"]) != pairs
        or any(
            type(value) is not int or value != 0
            for value in disjoint["pairwise_overlap_counts"].values()
        )
    ):
        raise RuntimeError("Phase 15 target-disjointness proof drifted")
    return payload


def _resolve_locked_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _locked_rows_and_targets(
    locked: Mapping[str, object],
    repository_root: Path,
    *,
    require_unique_targets: bool,
) -> tuple[list[dict], set[str]]:
    path = _resolve_locked_path(str(locked["path"]), repository_root)
    if not path.is_file() or _sha256(path) != locked["sha256"]:
        raise RuntimeError("Phase 15 suite source drifted")
    rows = load_jsonl(path)
    if len(rows) != int(locked["rows"]):
        raise RuntimeError("Phase 15 suite source row count drifted")
    if _case_fingerprint_set_digest(rows) != locked[
        "case_fingerprint_set_sha256"
    ]:
        raise RuntimeError("Phase 15 case fingerprint set drifted")
    targets: set[str] = set()
    for row in rows:
        ground_truth = row.get("ground_truth") if isinstance(row, dict) else None
        target = (
            ground_truth.get("parent_asin")
            if isinstance(ground_truth, dict)
            else None
        )
        if not isinstance(target, str) or not target:
            raise RuntimeError("Phase 15 suite target is invalid")
        targets.add(target)
    if require_unique_targets and len(targets) != len(rows):
        raise RuntimeError("Phase 15 suite targets must be unique")
    if _target_set_digest(targets) != locked[
        "target_fingerprint_set_sha256"
    ]:
        raise RuntimeError("Phase 15 target fingerprint set drifted")
    return rows, targets


def _builder_target_set_digest(domain: str, values: set[str]) -> str:
    builder = importlib.import_module(
        "scripts.build_phase15_protocol_robustness_suites"
    )
    fingerprints = sorted(
        hashlib.sha256(
            f"{builder.SELECTION_SALT}\0{domain}\0{value}".encode("utf-8")
        ).digest()
        for value in values
    )
    return hashlib.sha256(b"".join(fingerprints)).hexdigest()


def _validate_robustness_distribution(
    generated_rows: Mapping[str, Sequence[Mapping[str, object]]],
    manifest: Mapping[str, object],
    expected_counts: Mapping[str, int],
) -> None:
    expected_family_counts = _robustness_aggregate_counts(
        generated_rows,
        "phase15_family",
        ROBUSTNESS_FAMILY_ORDER,
    )
    expected_popularity_counts = _robustness_aggregate_counts(
        generated_rows,
        "phase15_popularity_stratum",
        ROBUSTNESS_POPULARITY_ORDER,
    )
    expected_scenario_counts = _robustness_aggregate_counts(
        generated_rows,
        "scenario_type",
        ROBUSTNESS_SCENARIO_ORDER,
    )
    expected_variant_counts = _robustness_variant_counts(generated_rows)
    if (
        manifest.get("family_counts") != expected_family_counts
        or manifest.get("popularity_counts") != expected_popularity_counts
        or manifest.get("scenario_counts") != expected_scenario_counts
        or manifest.get("variant_counts") != expected_variant_counts
    ):
        raise RuntimeError("Phase 15 robustness balance proof drifted")

    for name in ROBUSTNESS_SUITE_ORDER:
        cases_per_cell = ROBUSTNESS_SUITE_CASES_PER_CELL[name]
        joint = Counter(
            (
                str(row["phase15_family"]),
                str(row["phase15_popularity_stratum"]),
            )
            for row in generated_rows[name]
        )
        expected_joint = {
            (family, popularity): cases_per_cell
            for family in ROBUSTNESS_FAMILY_ORDER
            for popularity in ROBUSTNESS_POPULARITY_ORDER
        }
        if dict(joint) != expected_joint:
            raise RuntimeError("Phase 15 robustness joint strata are unbalanced")
        if name in {
            "scenario_balanced",
            "target_disjoint_development",
            "target_disjoint_validation",
        } and set(expected_scenario_counts[name].values()) != {
            expected_counts[name] // len(ROBUSTNESS_SCENARIO_ORDER)
        }:
            raise RuntimeError("Phase 15 robustness scenarios are unbalanced")

        if name == "paraphrase_fail_open":
            expected_variants = {
                variant: expected_counts[name]
                // len(ROBUSTNESS_PARAPHRASE_ORDER)
                for variant in ROBUSTNESS_PARAPHRASE_ORDER
            }
        elif name == "card_perturbed":
            expected_variants = {
                variant: expected_counts[name]
                // len(ROBUSTNESS_PERTURBATION_ORDER)
                for variant in ROBUSTNESS_PERTURBATION_ORDER
            }
        else:
            expected_variants = {"canonical": expected_counts[name]}
        if expected_variant_counts[name] != expected_variants:
            raise RuntimeError("Phase 15 robustness variant coverage drifted")


def _validate_robustness_manifest(
    manifest: object,
    suite_lock: Mapping[str, object],
    rows_by_source: Mapping[str, list[dict]],
    targets_by_source: Mapping[str, set[str]],
    repository_root: Path,
) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("Phase 15 robustness manifest is invalid")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("lock_id")
        != "phase15-protocol-robustness-target-disjoint-v2"
        or manifest.get("status")
        != "generated_before_phase15_candidate_data_execution"
        or manifest.get("privacy")
        != {
            "aggregate_only": True,
            "target_ids_published": False,
            "messages_published": False,
            "cards_published": False,
            "case_records_published": False,
        }
        or manifest.get("overlap_proof")
        != {
            "forbidden_overlap": 0,
            "inter_suite_overlap": 0,
            "all_selected_targets_unique": True,
        }
        or manifest.get("runtime_agent_or_retriever_used") is not False
    ):
        raise RuntimeError("Phase 15 robustness manifest guarantees drifted")

    generator_hashes = suite_lock["generator_source_sha256"]
    if manifest.get("generator_sha256") != generator_hashes[
        ROBUSTNESS_GENERATOR_RELATIVE
    ]:
        raise RuntimeError("Phase 15 robustness generator hash drifted")
    expected_references = {
        name: generator_hashes[relative]
        for name, relative in ROBUSTNESS_REFERENCE_RELATIVES.items()
    }
    if manifest.get("reference_source_hashes") != expected_references:
        raise RuntimeError("Phase 15 robustness reference hashes drifted")
    expected_inputs = {"catalog": suite_lock["catalog_sha256"]}
    for input_name, source_name in ROBUSTNESS_INPUT_TO_LOCKED_SOURCE.items():
        source_group = (
            suite_lock["prior_sources"]
            if source_name in PRIOR_SOURCE_KEYS
            else suite_lock["sources"]
        )
        expected_inputs[input_name] = source_group[source_name]["sha256"]
    if manifest.get("input_source_hashes") != expected_inputs:
        raise RuntimeError("Phase 15 robustness input hashes drifted")

    builder = importlib.import_module(
        "scripts.build_phase15_protocol_robustness_suites"
    )
    if manifest.get("selection_salt_sha256") != hashlib.sha256(
        ROBUSTNESS_SELECTION_SALT.encode("utf-8")
    ).hexdigest():
        raise RuntimeError("Phase 15 robustness selection salt drifted")
    expected_selection_policy = {
        "cases_per_family_popularity_cell": dict(
            ROBUSTNESS_SUITE_CASES_PER_CELL
        ),
        "family_order": list(ROBUSTNESS_FAMILY_ORDER),
        "popularity_order": list(ROBUSTNESS_POPULARITY_ORDER),
        "popularity_quantiles": (
            "catalog-only stable rank thirds; target exclusions applied later"
        ),
        "suite_order": list(ROBUSTNESS_SUITE_ORDER),
        "without_replacement_across_suites": True,
    }
    if manifest.get("selection_policy") != expected_selection_policy:
        raise RuntimeError("Phase 15 robustness selection policy drifted")
    outputs = manifest.get("outputs")
    counts = manifest.get("case_counts")
    if (
        not isinstance(outputs, dict)
        or set(outputs) != set(builder.SUITE_ORDER)
        or not isinstance(counts, dict)
        or set(counts) != set(builder.SUITE_ORDER)
    ):
        raise RuntimeError("Phase 15 robustness outputs are incomplete")
    expected_counts = {
        name: ROBUSTNESS_SUITE_CASES_PER_CELL[name]
        * len(ROBUSTNESS_FAMILY_ORDER)
        * len(ROBUSTNESS_POPULARITY_ORDER)
        for name in ROBUSTNESS_SUITE_ORDER
    }
    if counts != expected_counts:
        raise RuntimeError("Phase 15 robustness suite cardinality drifted")
    generated_rows = {
        name: rows_by_source[name] for name in ROBUSTNESS_SUITE_ORDER
    }
    _validate_robustness_distribution(
        generated_rows,
        manifest,
        expected_counts,
    )
    selected_targets: set[str] = set()
    for name in builder.SUITE_ORDER:
        locked = suite_lock["sources"][name]
        rows = rows_by_source[name]
        targets = targets_by_source[name]
        metadata = outputs[name]
        path = _resolve_locked_path(str(locked["path"]), repository_root)
        if (
            not isinstance(metadata, dict)
            or metadata.get("filename") != path.name
            or metadata.get("bytes") != path.stat().st_size
            or metadata.get("sha256") != locked["sha256"]
            or metadata.get("case_fingerprint_set_sha256")
            != locked["case_fingerprint_set_sha256"]
            or metadata.get("target_set_sha256")
            != _builder_target_set_digest(f"selected:{name}", targets)
            or counts[name] != len(rows)
        ):
            raise RuntimeError("Phase 15 robustness output provenance drifted")
        selected_targets.update(targets)
    forbidden_targets = set().union(
        *(targets_by_source[name] for name in ROBUSTNESS_INPUT_TO_LOCKED_SOURCE.values())
    )
    if selected_targets & forbidden_targets:
        raise RuntimeError("Phase 15 robustness target exclusion drifted")
    if (
        manifest.get("selected_target_count") != len(selected_targets)
        or manifest.get("selected_target_set_sha256")
        != _builder_target_set_digest("selected:all", selected_targets)
        or manifest.get("forbidden_target_count") != len(forbidden_targets)
        or manifest.get("forbidden_target_set_sha256")
        != _builder_target_set_digest("forbidden", forbidden_targets)
    ):
        raise RuntimeError("Phase 15 robustness target proof drifted")
    _validate_robustness_manifest_privacy(
        manifest,
        selected_targets,
        generated_rows,
    )


def _validate_suite_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    path = repository_root / SUITE_LOCK_RELATIVE
    payload = _validate_suite_lock_payload(
        json.loads(path.read_text(encoding="utf-8"))
    )
    robustness = payload["robustness_manifest"]
    robustness_path = _resolve_locked_path(robustness["path"], repository_root)
    if (
        not robustness_path.is_file()
        or _sha256(robustness_path) != robustness["sha256"]
    ):
        raise RuntimeError("Phase 15 robustness manifest drifted")
    for relative, expected in payload["generator_source_sha256"].items():
        source = _resolve_locked_path(relative, repository_root)
        if not source.is_file() or _sha256(source) != expected:
            raise RuntimeError("Phase 15 suite generator drifted")
    all_locked_sources = {
        **payload["sources"],
        **payload["prior_sources"],
    }
    rows_by_source: dict[str, list[dict]] = {}
    targets_by_source: dict[str, set[str]] = {}
    for name, locked in all_locked_sources.items():
        rows, targets = _locked_rows_and_targets(
            locked,
            repository_root,
            require_unique_targets=name in GENERATED_SOURCE_KEYS,
        )
        rows_by_source[name] = rows
        targets_by_source[name] = targets
    names = sorted(all_locked_sources)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if (
                left in GENERATED_SOURCE_KEYS
                or right in GENERATED_SOURCE_KEYS
            ) and targets_by_source[left] & targets_by_source[right]:
                raise RuntimeError("Phase 15 locked targets overlap")
    manifest = json.loads(robustness_path.read_text(encoding="utf-8"))
    _validate_robustness_manifest(
        manifest,
        payload,
        rows_by_source,
        targets_by_source,
        repository_root,
    )
    return payload


def _load_suite_samples(
    config: SuiteConfig,
    suite_lock: Mapping[str, object],
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[list[dict], dict[str, object]]:
    sources = suite_lock["sources"]  # type: ignore[assignment]
    selected: list[dict] = []
    all_cases: set[bytes] = set()
    all_targets: set[str] = set()
    source_hashes: list[bytes] = []
    source_case_counts: list[int] = []
    duplicate_targets = 0
    for key in config.source_keys:
        locked = sources[key]  # type: ignore[index]
        path = _resolve_locked_path(locked["path"], repository_root)
        if not path.is_file() or _sha256(path) != locked["sha256"]:
            raise RuntimeError("Phase 15 suite source drifted")
        rows = load_jsonl(path)
        if len(rows) != int(locked["rows"]):
            raise RuntimeError("Phase 15 suite source row count drifted")
        if _case_fingerprint_set_digest(rows) != locked[
            "case_fingerprint_set_sha256"
        ]:
            raise RuntimeError("Phase 15 case fingerprint set drifted")
        targets: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("Phase 15 suite row must be an object")
            ground_truth = row.get("ground_truth")
            parent_asin = (
                ground_truth.get("parent_asin")
                if isinstance(ground_truth, dict)
                else None
            )
            if not isinstance(parent_asin, str) or not parent_asin:
                raise RuntimeError("Phase 15 suite target is invalid")
            targets.add(parent_asin)
            if key in ROBUSTNESS_SUITE_ORDER:
                _dialog_spec(row)
        source_duplicate_targets = len(rows) - len(targets)
        if key in GENERATED_SOURCE_KEYS and source_duplicate_targets:
            raise RuntimeError("Phase 15 suite targets must be unique")
        duplicate_targets += source_duplicate_targets
        if _target_set_digest(targets) != locked[
            "target_fingerprint_set_sha256"
        ]:
            raise RuntimeError("Phase 15 target fingerprint set drifted")
        if targets & all_targets:
            raise RuntimeError("Phase 15 gate sources overlap by target")
        case_fingerprints = {
            hashlib.sha256(_canonical_json(row)).digest() for row in rows
        }
        if len(case_fingerprints) != len(rows) or case_fingerprints & all_cases:
            raise RuntimeError("Phase 15 gate sources overlap by case")
        all_cases.update(case_fingerprints)
        all_targets.update(targets)
        source_hashes.append(bytes.fromhex(locked["sha256"]))
        source_case_counts.append(len(rows))
        selected.extend(rows)
    return selected, {
        "source_count": len(config.source_keys),
        "source_rows": sum(source_case_counts),
        "evaluated_cases": len(selected),
        "source_sha256_set_digest": hashlib.sha256(
            b"".join(sorted(source_hashes))
        ).hexdigest(),
        "case_fingerprint_set_sha256": _fingerprint_set_digest(all_cases),
        "target_fingerprint_set_sha256": _target_set_digest(all_targets),
        "duplicate_cases": 0,
        "duplicate_targets": duplicate_targets,
    }


def _validate_execution_environment() -> None:
    if any(
        os.environ.get(key) != value
        for key, value in REQUIRED_ENVIRONMENT.items()
    ):
        raise RuntimeError("single-thread CPU execution environment is not pinned")


def _validate_prelock_verification(value: object) -> dict:
    expected = {
        "focused_suite_command",
        "focused_tests_passed",
        "complete_suite_command",
        "complete_unit_tests_passed",
        "completed_before_lock",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("Phase 15 pre-lock verification schema drifted")
    if (
        value.get("focused_suite_command") != FOCUSED_SUITE_COMMAND
        or value.get("complete_suite_command") != FULL_SUITE_COMMAND
        or value.get("completed_before_lock") is not True
    ):
        raise RuntimeError("Phase 15 pre-lock verification drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise RuntimeError("Phase 15 pre-lock test count is invalid")
    return value


def _validate_implementation_lock(
    repository_root: Path = REPOSITORY_ROOT,
) -> dict:
    path = repository_root / IMPLEMENTATION_LOCK_RELATIVE
    lock = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "lock_id",
        "status",
        "contract_sha256",
        "baseline_lock_sha256",
        "suite_lock_sha256",
        "research_plan_sha256",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 15 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 15 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 15 implementation lock identity")
    if lock.get("status") != "locked_before_phase15_data_suite_evaluation":
        raise RuntimeError("Phase 15 implementation is not frozen")
    documents = {
        "contract_sha256": CONTRACT_RELATIVE,
        "baseline_lock_sha256": BASELINE_LOCK_RELATIVE,
        "suite_lock_sha256": SUITE_LOCK_RELATIVE,
        "research_plan_sha256": RESEARCH_PLAN_RELATIVE,
    }
    for key, relative in documents.items():
        expected = lock.get(key)
        if (
            not isinstance(expected, str)
            or _HEX_SHA256_RE.fullmatch(expected) is None
            or expected != _sha256(repository_root / relative)
        ):
            raise RuntimeError("Phase 15 planning artifact drifted after lock")
    hashes = lock.get("source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 15 source lock is incomplete")
    observed = {
        relative: _sha256(repository_root / relative) for relative in SOURCE_PATHS
    }
    if observed != hashes:
        raise RuntimeError("Phase 15 implementation drifted after lock")
    baseline = json.loads(
        (repository_root / BASELINE_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    protected_files = baseline.get("protected_files")
    starter_hash = (
        protected_files.get("starter/agent.py")
        if isinstance(protected_files, dict)
        else None
    )
    if (
        not isinstance(starter_hash, str)
        or _sha256(repository_root / "starter/agent.py") != starter_hash
    ):
        raise RuntimeError("protected Phase 13 starter drifted")
    _validate_prelock_verification(lock.get("verification"))
    _validate_suite_lock(repository_root)
    return lock


def _claim_attempt(
    path: Path,
    suite: str,
    *,
    implementation_lock_sha256: str = "",
    suite_lock_sha256: str = "",
) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "suite": suite,
            "status": "claimed_before_rows_loaded",
            "implementation_lock_sha256": implementation_lock_sha256,
            "suite_lock_sha256": suite_lock_sha256,
        }
        os.write(descriptor, _canonical_json(payload) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_prerequisite_payload(
    expected_suite: str,
    payload: object,
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
    suite_lock: Mapping[str, object],
    limits: Mapping[str, object],
) -> None:
    if not isinstance(payload, dict) or payload.get("suite") != expected_suite:
        raise RuntimeError("prior suite result has the wrong suite identity")
    if not _published_result_is_valid(
        SUITES[expected_suite],
        payload,
        implementation_lock_sha256=implementation_lock_sha256,
        suite_lock_sha256=suite_lock_sha256,
        suite_lock=suite_lock,
        limits=limits,
    ):
        raise RuntimeError(
            "prior suite result is incomplete, stale, or failed recomputed gates"
        )


def _validate_prerequisites(
    config: SuiteConfig,
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
    suite_lock: Mapping[str, object],
    limits: Mapping[str, object],
) -> None:
    for name in config.prerequisites:
        path = SUITES[name].output
        if not path.is_file():
            raise RuntimeError("prior aggregate suite result is unavailable")
        _validate_prerequisite_payload(
            name,
            json.loads(path.read_text(encoding="utf-8")),
            implementation_lock_sha256=implementation_lock_sha256,
            suite_lock_sha256=suite_lock_sha256,
            suite_lock=suite_lock,
            limits=limits,
        )


def _validate_run_paths(config: SuiteConfig, output: Path) -> None:
    if output.resolve() != config.output.resolve():
        raise ValueError("suite has one frozen aggregate output path")
    if output.exists():
        raise FileExistsError("suite output already exists")
    if config.attempt.exists():
        raise FileExistsError("suite attempt was already consumed")


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
        "calibration",
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
    return payload.get("privacy") == {
        "aggregate_metrics_and_fixed_counters_only": True,
        "row_scenario_message_profile_target_product_and_trace_data_absent": True,
        "per_case_belief_and_fingerprint_values_absent": True,
        "manual_failure_or_small_cell_inspection_performed": False,
        "public_metrics_used_for_fitting": False,
    }


_COMMON_QUALITY_GATE_KEYS = frozenset(
    {
        "candidate_hit_rate_not_below_baseline",
        "baseline_hit_to_candidate_miss_is_zero",
        "candidate_mrr_not_below_baseline",
        "candidate_mttc_not_above_baseline",
        "candidate_technical_score_not_below_baseline",
    }
)
_COMMON_DECISION_GATE_KEYS = frozenset(
    {
        "protected_explicit_policy_is_exact_phase13",
        "candidate_replay_is_exact",
        "independent_candidate_construction_is_exact",
        "baseline_and_candidate_faults_are_zero",
        "protected_phase13_executes_one_bm25_dense_document_and_stage_a_call_per_search",
        "candidate_dense_executions_and_planned_skips_partition_searches",
        "candidate_dense_execution_is_bounded_to_one_per_search",
        "conditional_dense_behavior_matches_suite_world",
        "protected_policy_performs_zero_protocol_route_or_evidence_calls",
        "protocol_action_telemetry_is_fixed_and_complete",
        "suite_specific_protocol_behavior_passes",
        "belief_brier_and_ece_are_aggregate_unfitted_diagnostics",
        "all_variants_report_zero_model_and_api_tokens",
        "zero_copy_compact_card_only_operational_contract_is_locked",
        "candidate_warm_p95_ratio_within_contract",
        "candidate_wall_time_ratio_within_contract",
        "candidate_startup_ratio_within_contract",
        "candidate_additional_startup_rss_within_contract",
        "candidate_additional_post_warm_peak_rss_within_contract",
        "candidate_additional_retained_session_state_within_contract",
        "implementation_and_suite_locks_revalidated_after_all_variants",
        "suite_hash_target_disjointness_and_order_evidence_valid",
        "aggregate_publication_privacy_valid",
    }
)


def _expected_decision_gate_keys(config: SuiteConfig) -> set[str]:
    keys = set(_COMMON_DECISION_GATE_KEYS)
    if config.gate_mode == "comparison_only":
        keys.update(
            {
                "public_metrics_are_comparison_only",
                "public_confirmation_complete",
            }
        )
    else:
        keys.update(_COMMON_QUALITY_GATE_KEYS)
        if config.gate_mode == "exact_fail_open":
            keys.update(
                {
                    "candidate_metrics_exactly_equal_protected",
                    "paired_mean_delta_is_exactly_zero",
                    "paired_bootstrap_interval_is_exactly_zero",
                }
            )
        else:
            keys.add("paired_bootstrap_lower_95_not_below_zero")
        if config.gate_mode in {"strict_improvement", "strict_confidence"}:
            keys.update(
                {
                    "candidate_technical_score_strictly_improves",
                    "candidate_mrr_or_mttc_strictly_improves",
                }
            )
        if config.gate_mode == "strict_confidence":
            keys.add("paired_bootstrap_lower_95_strictly_above_zero")
    keys.add("advance")
    return keys


def _decision_gate_is_valid(config: SuiteConfig, value: object) -> bool:
    if (
        not isinstance(value, dict)
        or set(value) != _expected_decision_gate_keys(config)
        or any(type(item) is not bool for item in value.values())
    ):
        return False
    if config.gate_mode == "comparison_only":
        return (
            value["advance"] is False
            and value["public_metrics_are_comparison_only"] is True
            and value["public_confirmation_complete"] is True
            and all(
                item
                for key, item in value.items()
                if key
                not in {
                    "advance",
                    "public_metrics_are_comparison_only",
                    "public_confirmation_complete",
                }
            )
        )
    return value["advance"] is True and all(
        item for key, item in value.items() if key != "advance"
    )


def _expected_dataset_evidence(
    config: SuiteConfig,
    suite_lock: Mapping[str, object],
) -> dict[str, object]:
    if len(config.source_keys) != 1:
        raise RuntimeError("Phase 15 promotion requires one frozen source per gate")
    sources = suite_lock.get("sources")
    if not isinstance(sources, dict):
        raise RuntimeError("Phase 15 promotion suite lock is incomplete")
    locked = sources.get(config.source_keys[0])
    if not isinstance(locked, dict):
        raise RuntimeError("Phase 15 promotion source is missing")
    _validate_locked_source_entry(locked)
    rows = int(locked["rows"])
    return {
        "source_count": 1,
        "source_rows": rows,
        "evaluated_cases": rows,
        "source_sha256_set_digest": hashlib.sha256(
            bytes.fromhex(str(locked["sha256"]))
        ).hexdigest(),
        "case_fingerprint_set_sha256": locked[
            "case_fingerprint_set_sha256"
        ],
        "target_fingerprint_set_sha256": locked[
            "target_fingerprint_set_sha256"
        ],
        "duplicate_cases": 0,
        "duplicate_targets": 0,
    }


def _exact_keys(value: object, expected: Iterable[str]) -> bool:
    return isinstance(value, dict) and set(value) == set(expected)


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: object, *, minimum: float | None = None) -> bool:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return False
    return minimum is None or float(value) >= minimum


def _counter_health_is_valid(
    value: object,
    expected_keys: Iterable[str],
    *,
    mapping_keys: Iterable[str] = (),
) -> bool:
    if not _exact_keys(value, expected_keys):
        return False
    mappings = set(mapping_keys)
    for key, item in value.items():  # type: ignore[union-attr]
        if key == "policy":
            if not isinstance(item, str) or not item:
                return False
        elif key in mappings:
            if not isinstance(item, dict) or any(
                not isinstance(label, str)
                or not label
                or not _nonnegative_integer(count)
                for label, count in item.items()
            ):
                return False
        elif not _nonnegative_integer(item):
            return False
    return True


def _published_summary_is_valid(value: object, *, sample_count: int) -> bool:
    expected = {*OVERALL_METRIC_KEYS, "reported_token_usage"}
    if not _exact_keys(value, expected):
        return False
    summary = value  # type: ignore[assignment]
    usage = summary["reported_token_usage"]
    if (
        summary["sample_count"] != sample_count
        or not _exact_keys(usage, TOKEN_USAGE_KEYS)
        or any(not _nonnegative_integer(usage[key]) for key in TOKEN_USAGE_KEYS)
    ):
        return False
    bounded = ("hit_rate_at_10", "mrr", "efficiency", "recommended_technical_score")
    if any(
        not _finite_number(summary[key])
        or not 0.0 <= float(summary[key]) <= 1.0
        for key in bounded
    ):
        return False
    if (
        not _finite_number(summary["mttc"], minimum=0.0)
        or float(summary["mttc"]) > 10.0
        or int(usage["total_tokens"])
        != int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
    ):
        return False
    efficiency = round(
        max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0)),
        6,
    )
    technical_score = round(
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency,
        6,
    )
    return (
        float(summary["efficiency"]) == efficiency
        and float(summary["recommended_technical_score"]) == technical_score
    )


def _published_metrics_are_valid(
    value: object,
    *,
    sample_count: int,
) -> bool:
    if not _exact_keys(value, {"baseline", "candidate", "delta"}):
        return False
    metrics = value  # type: ignore[assignment]
    baseline = metrics["baseline"]
    candidate = metrics["candidate"]
    if not all(
        _published_summary_is_valid(summary, sample_count=sample_count)
        for summary in (baseline, candidate)
    ):
        return False
    delta = metrics["delta"]
    expected_delta = _metric_deltas(baseline, candidate)
    return (
        _exact_keys(delta, expected_delta)
        and delta == expected_delta
    )


def _published_paired_quality_is_valid(
    value: object,
    *,
    sample_count: int,
    baseline_summary: Mapping[str, object],
    candidate_summary: Mapping[str, object],
) -> bool:
    if not _exact_keys(
        value,
        {
            "transitions",
            "mean_utility_delta",
            "bootstrap",
            "mcnemar_exact_two_sided_p",
        },
    ):
        return False
    paired = value  # type: ignore[assignment]
    transition_keys = {
        "both_hit",
        "candidate_only_hit",
        "baseline_only_hit",
        "both_miss",
    }
    transitions = paired["transitions"]
    if (
        not _exact_keys(transitions, transition_keys)
        or any(
            not _nonnegative_integer(transitions[key])
            for key in transition_keys
        )
        or sum(int(transitions[key]) for key in transition_keys) != sample_count
    ):
        return False
    baseline_hits = int(transitions["both_hit"]) + int(
        transitions["baseline_only_hit"]
    )
    candidate_hits = int(transitions["both_hit"]) + int(
        transitions["candidate_only_hit"]
    )
    if (
        round(baseline_hits / sample_count, 6)
        != float(baseline_summary["hit_rate_at_10"])
        or round(candidate_hits / sample_count, 6)
        != float(candidate_summary["hit_rate_at_10"])
    ):
        return False
    bootstrap = paired["bootstrap"]
    if not _exact_keys(
        bootstrap,
        {"seed", "replicates", "strata", "lower_95", "upper_95"},
    ):
        return False
    lower = bootstrap["lower_95"]
    upper = bootstrap["upper_95"]
    probability = paired["mcnemar_exact_two_sided_p"]
    if (
        bootstrap["seed"] != BOOTSTRAP_SEED
        or bootstrap["replicates"] != BOOTSTRAP_REPLICATES
        or not _nonnegative_integer(bootstrap["strata"])
        or not 1 <= int(bootstrap["strata"]) <= sample_count
        or not _finite_number(paired["mean_utility_delta"])
        or not _finite_number(lower)
        or not _finite_number(upper)
        or not -1.0 <= float(lower) <= float(upper) <= 1.0
        or not _finite_number(probability)
        or not 0.0 <= float(probability) <= 1.0
    ):
        return False
    return float(probability) == round(
        _exact_mcnemar_p(
            int(transitions["candidate_only_hit"]),
            int(transitions["baseline_only_hit"]),
        ),
        9,
    )


def _published_protocol_health_is_valid(
    value: object,
    *,
    candidate: bool,
    expected_turns: int,
) -> bool:
    expected_keys = {
        "policy",
        "turns",
        *PROTOCOL_OUTCOMES,
        "planner_decisions",
        "protocol_mode_turns",
        "fail_open_turns_by_reason_code",
        "question_action_counts",
        "width_action_counts",
        "requested_total",
        "presented_total",
    }
    if not _exact_keys(value, expected_keys):
        return False
    health = value  # type: ignore[assignment]
    expected_policy = (
        PROTOCOL_UTILITY_DECISION_POLICY.value
        if candidate
        else PROTECTED_DECISION_POLICY.value
    )
    expected_protocol_turns = expected_turns if candidate else 0
    scalar_keys = {
        "turns",
        *PROTOCOL_OUTCOMES,
        "planner_decisions",
        "protocol_mode_turns",
        "requested_total",
        "presented_total",
    }
    if (
        health["policy"] != expected_policy
        or any(not _nonnegative_integer(health[key]) for key in scalar_keys)
        or int(health["turns"]) != expected_protocol_turns
        or sum(int(health[key]) for key in PROTOCOL_OUTCOMES)
        != expected_protocol_turns
    ):
        return False
    reasons = health["fail_open_turns_by_reason_code"]
    expected_reasons = {
        key: int(health[key]) for key in PROTOCOL_OUTCOMES if key != "applied"
    }
    questions = health["question_action_counts"]
    widths = health["width_action_counts"]
    if (
        reasons != expected_reasons
        or not _exact_keys(questions, PROTOCOL_QUESTION_ACTIONS)
        or any(not _nonnegative_integer(item) for item in questions.values())
        or sum(int(item) for item in questions.values())
        != expected_protocol_turns
        or not _exact_keys(widths, {str(width) for width in range(MAX_TOP_K + 1)})
        or any(not _nonnegative_integer(item) for item in widths.values())
        or sum(int(item) for item in widths.values()) != expected_protocol_turns
    ):
        return False
    planner = sum(
        int(health[key])
        for key in (
            "applied",
            "fail_open_evidence",
            "fail_open_no_candidates",
            "fail_open_no_support",
            "fail_open_validation",
        )
    )
    presented = sum(
        width * int(widths[str(width)]) for width in range(MAX_TOP_K + 1)
    )
    return (
        int(health["planner_decisions"]) == planner
        and int(health["protocol_mode_turns"]) == int(health["applied"])
        and int(health["presented_total"]) == presented
        and 0
        <= presented
        <= int(health["requested_total"])
        <= expected_protocol_turns * MAX_TOP_K
    )


def _published_health_is_valid(
    value: object,
    *,
    candidate: bool,
    sample_count: int,
) -> bool:
    if not _exact_keys(value, _AGGREGATE_HEALTH_KEYS):
        return False
    health = value  # type: ignore[assignment]
    if (
        not _nonnegative_integer(health["expected_turns"])
        or not sample_count <= int(health["expected_turns"]) <= sample_count * 10
        or not _nonnegative_integer(health["runtime_network_attempts"])
        or not _nonnegative_integer(health["retained_agent_bytes"])
        or not _nonnegative_integer(health["protocol_retained_bytes"])
        or int(health["protocol_retained_bytes"])
        > int(health["retained_agent_bytes"])
        or not _finite_number(health["evaluation_wall_seconds"], minimum=0.0)
    ):
        return False
    route = health["route_health"]
    if not _counter_health_is_valid(
        route,
        _ROUTE_HEALTH_KEYS,
        mapping_keys={"bm25", "dense"},
    ):
        return False
    if any(
        not set(route[key]).issubset({"ok", "empty", "skipped", "unavailable", "error"})
        for key in ("bm25", "dense")
    ):
        return False
    counter_sections = (
        ("ranking_health", RANKING_HEALTH_KEYS, ()),
        ("rescue_health", RESCUE_HEALTH_KEYS, ()),
        ("route_redundancy_health", _ROUTE_REDUNDANCY_HEALTH_KEYS, ()),
        ("intent_epoch_slate_health", _INTENT_EPOCH_SLATE_HEALTH_KEYS, ()),
        ("profile_health", PROFILE_HEALTH_KEYS, ()),
        ("slate_health", SLATE_HEALTH_KEYS, ()),
        ("orchestration_health", ORCHESTRATION_HEALTH_KEYS, ("reasons",)),
        ("response_audit", _RESPONSE_AUDIT_KEYS, ()),
    )
    if any(
        not _counter_health_is_valid(
            health[key],
            expected,
            mapping_keys=mappings,
        )
        for key, expected, mappings in counter_sections
    ):
        return False
    if (
        health["profile_health"]["policy"]
        != BOUNDED_RESIDUAL_PROFILE_POLICY.value
        or health["slate_health"]["policy"]
        != INTENT_EPOCH_NOVELTY_SLATE_POLICY.value
        or health["intent_epoch_slate_health"]["policy"]
        != INTENT_EPOCH_NOVELTY_SLATE_POLICY.value
        or health["orchestration_health"]["policy"]
        != EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value
    ):
        return False
    turns = int(health["expected_turns"])
    orchestration = health["orchestration_health"]
    searches = int(orchestration["searches"])
    if (
        int(orchestration["decisions"]) != turns
        or searches + int(orchestration["reuses"]) + int(orchestration["skips"])
        != turns
        or not _lookup_accounting_exact(orchestration)
        or sum(int(item) for item in route["bm25"].values()) != searches
        or sum(int(item) for item in route["dense"].values()) != searches
        or int(health["ranking_health"]["attempts"]) != searches
        or int(route["candidate_document_calls"]) != searches
        or int(health["profile_health"]["session_entries"]) != sample_count
    ):
        return False
    if not _published_protocol_health_is_valid(
        health["protocol_decision_health"],
        candidate=candidate,
        expected_turns=turns,
    ):
        return False
    calibration = health["calibration"]
    if not _calibration_is_valid(calibration, expected_turns=turns):
        return False
    latency = health["respond_latency_ms"]
    if (
        not _exact_keys(latency, LATENCY_KEYS)
        or not _nonnegative_integer(latency["count"])
        or not _nonnegative_integer(latency["warm_count"])
        or any(not _finite_number(latency[key], minimum=0.0) for key in LATENCY_KEYS)
        or int(latency["count"]) != turns
        or int(latency["warm_count"]) > turns
        or not float(latency["p50"])
        <= float(latency["p90"])
        <= float(latency["p95"])
        <= float(latency["p99"])
        <= float(latency["max"])
        or float(latency["warm_p95"]) > float(latency["max"])
    ):
        return False
    if candidate:
        zero_widths = int(
            health["protocol_decision_health"]["width_action_counts"]["0"]
        )
        if int(health["slate_health"]["attempts"]) + zero_widths != turns:
            return False
    elif (
        int(health["protocol_retained_bytes"]) != 0
        or int(health["slate_health"]["attempts"]) != turns
    ):
        return False
    return True


def _published_exactness_is_valid(value: object) -> bool:
    groups = {
        "protected_reference",
        "candidate_replay",
        "independent_candidate",
        "candidate_vs_baseline_fail_open",
    }
    if not _exact_keys(value, groups):
        return False
    exact_keys = {
        "evaluator_payload_equal",
        "response_state_slate_cache_equal",
        "complete_private_state_equal",
        "aggregate_health_equal",
    }
    fail_open_keys = {
        "evaluator_payload_equal",
        "response_state_slate_cache_equal",
    }
    return all(
        _exact_keys(value[name], exact_keys)
        and all(type(item) is bool for item in value[name].values())
        for name in groups - {"candidate_vs_baseline_fail_open"}
    ) and (
        _exact_keys(value["candidate_vs_baseline_fail_open"], fail_open_keys)
        and all(
            type(item) is bool
            for item in value["candidate_vs_baseline_fail_open"].values()
        )
    )


def _published_performance_is_valid(
    value: object,
    *,
    baseline_health: Mapping[str, object],
    candidate_health: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    candidate_summary: Mapping[str, object],
) -> bool:
    expected_keys = {
        "baseline_wall_seconds",
        "candidate_wall_seconds",
        "candidate_wall_time_ratio",
        "baseline_warm_p95_ms",
        "candidate_warm_p95_ms",
        "candidate_warm_p95_ratio",
        "baseline_retained_agent_bytes",
        "candidate_retained_agent_bytes",
        "candidate_additional_retained_agent_bytes",
        "candidate_protocol_retained_bytes",
        "candidate_additional_retained_session_bytes",
    }
    if not _exact_keys(value, expected_keys):
        return False
    expected = _performance_summary(
        VariantRun(dict(baseline_summary), [], dict(baseline_health), "", "", ""),
        VariantRun(dict(candidate_summary), [], dict(candidate_health), "", "", ""),
    )
    return value == expected


def _published_startup_is_valid(value: object) -> bool:
    expected_keys = {
        "accounting",
        "baseline_probe_count",
        "candidate_probe_count",
        "baseline_startup_seconds",
        "candidate_startup_seconds",
        "candidate_startup_time_ratio",
        "baseline_startup_rss_bytes",
        "candidate_startup_rss_bytes",
        "candidate_additional_startup_rss_bytes",
        "baseline_post_warm_peak_rss_bytes",
        "candidate_post_warm_peak_rss_bytes",
        "candidate_additional_post_warm_peak_rss_bytes",
        "baseline_empty_retained_bytes",
        "candidate_empty_retained_bytes",
    }
    if not _exact_keys(value, expected_keys):
        return False
    startup = value  # type: ignore[assignment]
    integer_keys = expected_keys - {
        "accounting",
        "baseline_startup_seconds",
        "candidate_startup_seconds",
        "candidate_startup_time_ratio",
    }
    if (
        startup["accounting"]
        != "max_candidate_vs_min_protected_across_both_orders"
        or startup["baseline_probe_count"] != 2
        or startup["candidate_probe_count"] != 2
        or any(not _nonnegative_integer(startup[key]) for key in integer_keys)
        or any(
            not _finite_number(startup[key], minimum=0.0)
            for key in (
                "baseline_startup_seconds",
                "candidate_startup_seconds",
                "candidate_startup_time_ratio",
            )
        )
        or float(startup["baseline_startup_seconds"]) <= 0.0
    ):
        return False
    return (
        float(startup["candidate_startup_time_ratio"])
        == round(
            _safe_ratio(
                float(startup["candidate_startup_seconds"]),
                float(startup["baseline_startup_seconds"]),
            ),
            6,
        )
        and int(startup["candidate_additional_startup_rss_bytes"])
        == max(
            0,
            int(startup["candidate_startup_rss_bytes"])
            - int(startup["baseline_startup_rss_bytes"]),
        )
        and int(startup["candidate_additional_post_warm_peak_rss_bytes"])
        == max(
            0,
            int(startup["candidate_post_warm_peak_rss_bytes"])
            - int(startup["baseline_post_warm_peak_rss_bytes"]),
        )
    )


def _published_run_configuration_is_valid(value: object) -> bool:
    expected = {
        "execution": "strictly_sequential_cpu",
        "threads": 1,
        "processes_during_evaluation": 1,
        "variant_policies": [BASELINE_ID, CANDIDATE_ID],
        "verification_runs_are_not_ablation_arms": True,
        "fresh_agent_state_per_variant": True,
        "startup_probe_policy_order": [
            policy.value for pair in STARTUP_PROBE_ORDERS for policy in pair
        ],
        "startup_probe_conservative_accounting": (
            "max_candidate_vs_min_protected_across_both_orders"
        ),
        "conditional_dense_routing_enabled": True,
        "additional_external_model_or_api_calls": 0,
        "gpu_or_mps": False,
        "thermal_safe_acknowledged": True,
        "public_metrics_can_tune_policy": False,
    }
    return value == expected


def _published_reproducibility_is_valid(
    value: object,
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
) -> bool:
    expected_keys = {
        "platform",
        "python",
        "environment",
        "implementation_lock_id",
        "implementation_lock_sha256",
        "suite_lock_id",
        "suite_lock_sha256",
        "locks_revalidated_after_all_variants",
    }
    return (
        _exact_keys(value, expected_keys)
        and value["implementation_lock_id"] == IMPLEMENTATION_LOCK_ID  # type: ignore[index]
        and value["implementation_lock_sha256"]  # type: ignore[index]
        == implementation_lock_sha256
        and value["suite_lock_id"] == SUITE_LOCK_ID  # type: ignore[index]
        and value["suite_lock_sha256"] == suite_lock_sha256  # type: ignore[index]
        and value["locks_revalidated_after_all_variants"] is True  # type: ignore[index]
        and value["environment"] == REQUIRED_ENVIRONMENT  # type: ignore[index]
        and isinstance(value["platform"], str)  # type: ignore[index]
        and bool(value["platform"])  # type: ignore[index]
        and isinstance(value["python"], str)  # type: ignore[index]
        and bool(value["python"])  # type: ignore[index]
    )


def _recompute_published_decision_gates(
    config: SuiteConfig,
    payload: Mapping[str, object],
    limits: Mapping[str, object],
    *,
    suite_evidence_valid: bool,
) -> dict[str, bool]:
    metrics = payload["metrics"]  # type: ignore[assignment]
    paired = payload["paired_quality"]  # type: ignore[assignment]
    health = payload["health"]  # type: ignore[assignment]
    performance = payload["performance"]  # type: ignore[assignment]
    startup = payload["startup"]  # type: ignore[assignment]
    exactness = payload["exactness"]  # type: ignore[assignment]
    baseline_summary = metrics["baseline"]
    candidate_summary = metrics["candidate"]
    runs = tuple(
        health[name]
        for name in (
            "baseline",
            "protected_reference",
            "candidate",
            "candidate_replay",
            "independent_candidate",
        )
    )
    calls = [_call_accounting(run) for run in runs]
    protected_search = all(
        item["searches"]
        == item["bm25_route_calls"]
        == item["dense_route_executions"]
        == item["dense_route_statuses"]
        == item["candidate_document_calls"]
        == item["stage_a_attempts"]
        and item["dense_route_skips"] == 0
        for item in calls[:2]
    )
    candidate_partition = all(
        item["searches"]
        == item["bm25_route_calls"]
        == item["dense_route_statuses"]
        == item["candidate_document_calls"]
        == item["stage_a_attempts"]
        and item["dense_route_executions"] + item["dense_route_skips"]
        == item["searches"]
        for item in calls[2:]
    )
    conditional_dense = True
    if config.name == "fresh_exact":
        conditional_dense = all(item["dense_route_skips"] > 0 for item in calls[2:])
    elif config.gate_mode == "exact_fail_open":
        conditional_dense = all(item["dense_route_skips"] == 0 for item in calls[2:])
    candidate_protocol = health["candidate"]["protocol_decision_health"]
    protocol_behavior = True
    if config.gate_mode == "strict_improvement":
        protocol_behavior = (
            int(candidate_protocol["applied"]) > 0
            and int(candidate_protocol["planner_decisions"]) > 0
        )
    elif config.gate_mode == "exact_fail_open":
        fail_open = exactness["candidate_vs_baseline_fail_open"]
        protocol_behavior = (
            int(candidate_protocol["applied"]) == 0
            and int(candidate_protocol["unsupported_or_disabled"])
            == int(candidate_protocol["turns"])
            and fail_open["evaluator_payload_equal"] is True
            and fail_open["response_state_slate_cache_equal"] is True
        )
    quality = _quality_gates(
        config,
        VariantRun(dict(baseline_summary), [], {}, "", "", ""),
        VariantRun(dict(candidate_summary), [], {}, "", "", ""),
        paired,
    )
    gates = {
        **quality,
        "protected_explicit_policy_is_exact_phase13": all(
            exactness["protected_reference"].values()
        ),
        "candidate_replay_is_exact": all(
            exactness["candidate_replay"].values()
        ),
        "independent_candidate_construction_is_exact": all(
            exactness["independent_candidate"].values()
        ),
        "baseline_and_candidate_faults_are_zero": all(
            _faults_are_zero(run) for run in runs
        ),
        "protected_phase13_executes_one_bm25_dense_document_and_stage_a_call_per_search": protected_search,
        "candidate_dense_executions_and_planned_skips_partition_searches": candidate_partition,
        "candidate_dense_execution_is_bounded_to_one_per_search": all(
            item["dense_route_executions"] <= item["searches"]
            for item in calls[2:]
        ),
        "conditional_dense_behavior_matches_suite_world": conditional_dense,
        "protected_policy_performs_zero_protocol_route_or_evidence_calls": all(
            item["protocol_exact_candidate_calls"] == 0
            and item["protocol_candidate_evidence_calls"] == 0
            for item in calls[:2]
        ),
        "protocol_action_telemetry_is_fixed_and_complete": (
            int(candidate_protocol["turns"])
            == int(health["candidate"]["expected_turns"])
        ),
        "suite_specific_protocol_behavior_passes": protocol_behavior,
        "belief_brier_and_ece_are_aggregate_unfitted_diagnostics": (
            _calibration_is_valid(
                payload["calibration"],
                expected_turns=int(health["candidate"]["expected_turns"]),
            )
        ),
        "all_variants_report_zero_model_and_api_tokens": (
            _tokens_are_zero(baseline_summary)
            and _tokens_are_zero(candidate_summary)
        ),
        "zero_copy_compact_card_only_operational_contract_is_locked": (
            limits["no_duplicate_catalog_copy"] is True
            and limits["candidate_metadata_is_compact_card_only"] is True
            and limits["candidate_metadata_fetches_full_fts_documents"] is False
            and limits["additional_external_model_or_api_calls_per_turn"] == 0
            and limits["model_and_api_tokens"] == 0
            and limits["response_exceptions"] == 0
            and limits["invalid_api_responses"] == 0
            and limits["runtime_network_attempts"] == 0
            and limits["faults"] == 0
            and limits["replay_exact"] is True
            and limits["independent_construction_exact"] is True
        ),
        "candidate_warm_p95_ratio_within_contract": (
            float(performance["candidate_warm_p95_ratio"])
            <= float(limits["candidate_warm_p95_ratio_at_most"])
        ),
        "candidate_wall_time_ratio_within_contract": (
            float(performance["candidate_wall_time_ratio"])
            <= float(limits["candidate_wall_time_ratio_at_most"])
        ),
        "candidate_startup_ratio_within_contract": (
            float(startup["candidate_startup_time_ratio"])
            <= float(limits["candidate_startup_ratio_at_most"])
        ),
        "candidate_additional_startup_rss_within_contract": (
            int(startup["candidate_additional_startup_rss_bytes"])
            <= int(limits["candidate_additional_startup_rss_bytes_at_most"])
        ),
        "candidate_additional_post_warm_peak_rss_within_contract": (
            int(startup["candidate_additional_post_warm_peak_rss_bytes"])
            <= int(limits["candidate_additional_post_warm_peak_rss_bytes_at_most"])
        ),
        "candidate_additional_retained_session_state_within_contract": (
            int(performance["candidate_additional_retained_session_bytes"])
            <= int(limits["candidate_additional_retained_session_bytes_at_most"])
        ),
        "implementation_and_suite_locks_revalidated_after_all_variants": (
            payload["reproducibility"]["locks_revalidated_after_all_variants"]
            is True
        ),
        "suite_hash_target_disjointness_and_order_evidence_valid": suite_evidence_valid,
        "aggregate_publication_privacy_valid": True,
    }
    if config.gate_mode == "comparison_only":
        gates["public_confirmation_complete"] = all(
            item
            for key, item in gates.items()
            if key != "public_metrics_are_comparison_only"
        )
        gates["advance"] = False
    else:
        gates["advance"] = all(gates.values())
    return gates


def _published_result_is_valid(
    config: SuiteConfig,
    payload: object,
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
    suite_lock: Mapping[str, object],
    limits: Mapping[str, object],
    public_baseline_metrics: Mapping[str, object] | None = None,
) -> bool:
    try:
        if (
            not publication_privacy_is_valid(payload)
            or not isinstance(payload, dict)
            or payload.get("suite") != config.name
            or payload.get("dataset")
            != _expected_dataset_evidence(config, suite_lock)
            or not _published_run_configuration_is_valid(
                payload.get("run_configuration")
            )
            or not _published_reproducibility_is_valid(
                payload.get("reproducibility"),
                implementation_lock_sha256=implementation_lock_sha256,
                suite_lock_sha256=suite_lock_sha256,
            )
        ):
            return False
        sample_count = int(payload["dataset"]["evaluated_cases"])
        if not _published_metrics_are_valid(
            payload.get("metrics"),
            sample_count=sample_count,
        ):
            return False
        metrics = payload["metrics"]
        if not _published_paired_quality_is_valid(
            payload.get("paired_quality"),
            sample_count=sample_count,
            baseline_summary=metrics["baseline"],
            candidate_summary=metrics["candidate"],
        ):
            return False
        health = payload.get("health")
        health_names = {
            "baseline": False,
            "protected_reference": False,
            "candidate": True,
            "candidate_replay": True,
            "independent_candidate": True,
        }
        if not _exact_keys(health, health_names) or any(
            not _published_health_is_valid(
                health[name],
                candidate=is_candidate,
                sample_count=sample_count,
            )
            for name, is_candidate in health_names.items()
        ):
            return False
        call_accounting = payload.get("call_accounting")
        if (
            not _exact_keys(call_accounting, {"baseline", "candidate"})
            or call_accounting["baseline"]
            != _call_accounting(health["baseline"])
            or call_accounting["candidate"]
            != _call_accounting(health["candidate"])
        ):
            return False
        if (
            not _published_performance_is_valid(
                payload.get("performance"),
                baseline_health=health["baseline"],
                candidate_health=health["candidate"],
                baseline_summary=metrics["baseline"],
                candidate_summary=metrics["candidate"],
            )
            or not _published_startup_is_valid(payload.get("startup"))
            or not _published_exactness_is_valid(payload.get("exactness"))
            or payload.get("calibration") != health["candidate"]["calibration"]
            or not _calibration_is_valid(
                payload.get("calibration"),
                expected_turns=int(health["candidate"]["expected_turns"]),
            )
        ):
            return False
        exactness = payload["exactness"]
        aggregate_pairs = (
            ("protected_reference", "baseline", "protected_reference"),
            ("candidate_replay", "candidate", "candidate_replay"),
            (
                "independent_candidate",
                "candidate",
                "independent_candidate",
            ),
        )
        if any(
            bool(exactness[group]["aggregate_health_equal"])
            != (
                _deterministic_health(health[first])
                == _deterministic_health(health[second])
            )
            for group, first, second in aggregate_pairs
        ):
            return False
        fail_open_exactness = exactness["candidate_vs_baseline_fail_open"]
        if (
            fail_open_exactness["evaluator_payload_equal"] is True
            and metrics["candidate"] != metrics["baseline"]
        ):
            return False
        suite_evidence_valid = True
        if config.name == "public_confirmation":
            if public_baseline_metrics is None or any(
                metrics["baseline"].get(key) != expected
                for key, expected in public_baseline_metrics.items()
            ):
                suite_evidence_valid = False
        recomputed = _recompute_published_decision_gates(
            config,
            payload,
            limits,
            suite_evidence_valid=suite_evidence_valid,
        )
        return (
            payload.get("decision_gate") == recomputed
            and _decision_gate_is_valid(config, recomputed)
        )
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError):
        return False


def _promotion_payloads_are_allowed(
    payloads: Sequence[object],
    *,
    implementation_lock_sha256: str,
    suite_lock_sha256: str,
    suite_lock: Mapping[str, object],
    limits: Mapping[str, object],
    public_baseline_metrics: Mapping[str, object],
) -> bool:
    """Recompute every promotion gate from locked aggregate evidence."""

    if (
        _HEX_SHA256_RE.fullmatch(implementation_lock_sha256) is None
        or _HEX_SHA256_RE.fullmatch(suite_lock_sha256) is None
        or len(payloads) != len(SUITES)
    ):
        return False
    for expected_name, payload in zip(SUITES, payloads):
        config = SUITES[expected_name]
        if not _published_result_is_valid(
            config,
            payload,
            implementation_lock_sha256=implementation_lock_sha256,
            suite_lock_sha256=suite_lock_sha256,
            suite_lock=suite_lock,
            limits=limits,
            public_baseline_metrics=public_baseline_metrics,
        ):
            return False
    return True


def promotion_is_allowed(
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> bool:
    """Revalidate locks and fixed result files before authorizing promotion."""

    try:
        implementation_lock = _validate_implementation_lock(repository_root)
        suite_lock = _validate_suite_lock(repository_root)
        contract = json.loads(
            (repository_root / CONTRACT_RELATIVE).read_text(encoding="utf-8")
        )
        limits = _operational_limits(contract)
        baseline_lock = json.loads(
            (repository_root / BASELINE_LOCK_RELATIVE).read_text(
                encoding="utf-8"
            )
        )
        protected_metrics = baseline_lock.get("protected_metrics")
        public_baseline_metrics = (
            protected_metrics.get("public")
            if isinstance(protected_metrics, dict)
            else None
        )
        if not isinstance(public_baseline_metrics, dict):
            return False
        implementation_lock_sha256 = _sha256(
            repository_root / IMPLEMENTATION_LOCK_RELATIVE
        )
        suite_lock_sha256 = _sha256(repository_root / SUITE_LOCK_RELATIVE)
        payloads = [
            json.loads(
                (repository_root / SUITES[name].output.name).read_text(
                    encoding="utf-8"
                )
            )
            for name in SUITES
        ]
        if implementation_lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
            return False
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return _promotion_payloads_are_allowed(
        payloads,
        implementation_lock_sha256=implementation_lock_sha256,
        suite_lock_sha256=suite_lock_sha256,
        suite_lock=suite_lock,
        limits=limits,
        public_baseline_metrics=public_baseline_metrics,
    )


def _new_runtime(catalog: Path, policy: DecisionPolicy) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        catalog,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        decision_policy=policy,
    )


def _validate_runtime_backend(backend: object, policy: DecisionPolicy) -> None:
    if getattr(backend, "dense_available", False) is not True:
        raise RuntimeError("dense retrieval is unavailable")
    if getattr(backend, "bm25_available", False) is not True:
        raise RuntimeError("BM25 retrieval is unavailable")
    if policy is PROTOCOL_UTILITY_DECISION_POLICY and (
        getattr(backend, "protocol_evidence_capability", None)
        is not PROTOCOL_EVIDENCE_CAPABILITY
    ):
        raise RuntimeError("protocol evidence is unavailable")


def _startup_worker(catalog: Path, policy: DecisionPolicy) -> dict[str, object]:
    started = time.perf_counter()
    agent = _new_runtime(catalog, policy)
    elapsed = time.perf_counter() - started
    _validate_runtime_backend(agent.retrieval_backend, policy)
    module_presence = {
        name: name in sys.modules for name in STARTUP_CANDIDATE_MODULES
    }
    if policy is PROTECTED_DECISION_POLICY and any(module_presence.values()):
        raise RuntimeError("protected startup imported a candidate-only module")
    if (
        policy is PROTOCOL_UTILITY_DECISION_POLICY
        and not module_presence["conversational_search.protocol"]
    ):
        raise RuntimeError("candidate startup did not build protocol evidence")
    empty_retained_bytes = _retained_agent_bytes(agent)
    startup_rss_bytes = _current_max_rss_bytes()
    agent.reset("phase15_resource_probe", {})
    response = agent.respond(
        "phase15_resource_probe",
        "Recommend versatile everyday footwear suitable for long walks.",
        1,
        MAX_TOP_K,
    )
    if not isinstance(response, dict) or not isinstance(
        response.get("message"), str
    ):
        raise RuntimeError("post-warm resource probe returned an invalid response")
    return {
        "policy": policy.value,
        "elapsed_seconds": round(elapsed, 9),
        "max_rss_bytes": startup_rss_bytes,
        "post_warm_max_rss_bytes": _current_max_rss_bytes(),
        "empty_retained_bytes": empty_retained_bytes,
        "candidate_module_presence": module_presence,
    }


def _startup_probe(catalog: Path) -> dict[str, object]:
    observations: dict[DecisionPolicy, list[dict[str, object]]] = {
        PROTECTED_DECISION_POLICY: [],
        PROTOCOL_UTILITY_DECISION_POLICY: [],
    }
    environment = {**os.environ, **REQUIRED_ENVIRONMENT}
    policy_order = tuple(
        policy for pair in STARTUP_PROBE_ORDERS for policy in pair
    )
    for policy in policy_order:
        command = (
            sys.executable,
            "-m",
            "scripts.run_protocol_utility_ablations",
            "--startup-worker",
            policy.value,
            "--catalog",
            str(catalog),
        )
        completed = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=900,
        )
        try:
            observation = json.loads(completed.stdout.strip())
        except json.JSONDecodeError as error:
            raise RuntimeError("startup worker output is invalid") from error
        if observation.get("policy") != policy.value:
            raise RuntimeError("startup worker policy drifted")
        observations[policy].append(observation)
    return _startup_summary(
        observations[PROTECTED_DECISION_POLICY],
        observations[PROTOCOL_UTILITY_DECISION_POLICY],
    )


def _baseline_matches_protected_lock(
    config: SuiteConfig,
    baseline: VariantRun,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> bool:
    # Target-disjoint Phase 15 development/validation suites intentionally do
    # not reuse Phase 13 metric expectations.  Runtime exactness is checked by
    # the explicit protected-reference run; only the unchanged public source
    # has a locked aggregate metric expectation.
    if config.name != "public_confirmation":
        return True
    key = "public"
    lock = json.loads(
        (repository_root / BASELINE_LOCK_RELATIVE).read_text(encoding="utf-8")
    )
    expected = lock.get("protected_metrics", {}).get(key)
    if not isinstance(expected, dict):
        return False
    return all(baseline.summary.get(metric) == value for metric, value in expected.items())


def run_protocol_utility_ablation(
    suite: str,
    output_path: str | Path,
    *,
    thermal_safe_ack: bool,
) -> dict:
    """Consume one sealed suite attempt and emit aggregate evidence only."""

    if suite not in SUITES:
        raise ValueError("unsupported Phase 15 suite")
    if thermal_safe_ack is not True:
        raise RuntimeError("thermal safety must be acknowledged before a suite")
    config = SUITES[suite]
    output = Path(output_path).resolve()
    _validate_execution_environment()
    implementation_lock = _validate_implementation_lock()
    suite_lock = _validate_suite_lock()
    implementation_lock_sha256 = _sha256(
        REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE
    )
    suite_lock_sha256 = _sha256(REPOSITORY_ROOT / SUITE_LOCK_RELATIVE)
    contract = json.loads(
        (REPOSITORY_ROOT / CONTRACT_RELATIVE).read_text(encoding="utf-8")
    )
    limits = _operational_limits(contract)
    _validate_prerequisites(
        config,
        implementation_lock_sha256=implementation_lock_sha256,
        suite_lock_sha256=suite_lock_sha256,
        suite_lock=suite_lock,
        limits=limits,
    )
    _validate_run_paths(config, output)
    catalog = REPOSITORY_ROOT / "data/catalog.jsonl"
    if _sha256(catalog) != suite_lock["catalog_sha256"]:
        raise RuntimeError("catalog drifted before Phase 15 suite claim")

    _claim_attempt(
        config.attempt,
        config.name,
        implementation_lock_sha256=implementation_lock_sha256,
        suite_lock_sha256=suite_lock_sha256,
    )
    samples, dataset_evidence = _load_suite_samples(config, suite_lock)
    catalog_ids, categories, products = catalog_index(catalog)
    startup = _startup_probe(catalog)

    baseline_runtime = _new_runtime(catalog, PROTECTED_DECISION_POLICY)
    baseline_backend = baseline_runtime.retrieval_backend
    _validate_runtime_backend(baseline_backend, PROTECTED_DECISION_POLICY)
    baseline = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        baseline_backend,
        PROTECTED_DECISION_POLICY,
    )
    del baseline_runtime, baseline_backend
    gc.collect()

    candidate_runtime = _new_runtime(catalog, PROTOCOL_UTILITY_DECISION_POLICY)
    candidate_backend = candidate_runtime.retrieval_backend
    _validate_runtime_backend(candidate_backend, PROTOCOL_UTILITY_DECISION_POLICY)
    candidate_backend_runs: dict[str, VariantRun] = {}
    for name in CANDIDATE_BACKEND_RUN_ORDER:
        policy = (
            PROTECTED_DECISION_POLICY
            if name == "protected_reference"
            else PROTOCOL_UTILITY_DECISION_POLICY
        )
        candidate_backend_runs[name] = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            candidate_backend,
            policy,
        )
    candidate = candidate_backend_runs["candidate"]
    protected_reference = candidate_backend_runs["protected_reference"]
    replay = candidate_backend_runs["candidate_replay"]
    del candidate_runtime, candidate_backend
    gc.collect()

    independent_runtime = _new_runtime(
        catalog,
        PROTOCOL_UTILITY_DECISION_POLICY,
    )
    independent_backend = independent_runtime.retrieval_backend
    _validate_runtime_backend(
        independent_backend,
        PROTOCOL_UTILITY_DECISION_POLICY,
    )
    independent = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        independent_backend,
        PROTOCOL_UTILITY_DECISION_POLICY,
    )
    del independent_runtime, independent_backend
    gc.collect()

    locks_revalidated = (
        _validate_implementation_lock() == implementation_lock
        and _validate_suite_lock() == suite_lock
        and _sha256(REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE)
        == implementation_lock_sha256
        and _sha256(REPOSITORY_ROOT / SUITE_LOCK_RELATIVE)
        == suite_lock_sha256
    )
    paired = _paired_statistics(
        baseline.sessions,
        candidate.sessions,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    performance = _performance_summary(baseline, candidate)
    exactness = {
        "protected_reference": _run_exactness(baseline, protected_reference),
        "candidate_replay": _run_exactness(candidate, replay),
        "independent_candidate": _run_exactness(candidate, independent),
        "candidate_vs_baseline_fail_open": {
            "evaluator_payload_equal": (
                candidate.evaluator_digest == baseline.evaluator_digest
            ),
            "response_state_slate_cache_equal": (
                candidate.behavior_digest == baseline.behavior_digest
            ),
        },
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "suite": config.name,
        "dataset": dataset_evidence,
        "run_configuration": {
            "execution": "strictly_sequential_cpu",
            "threads": 1,
            "processes_during_evaluation": 1,
            "variant_policies": [BASELINE_ID, CANDIDATE_ID],
            "verification_runs_are_not_ablation_arms": True,
            "fresh_agent_state_per_variant": True,
            "startup_probe_policy_order": [
                policy.value
                for pair in STARTUP_PROBE_ORDERS
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
        "metrics": {
            "baseline": baseline.summary,
            "candidate": candidate.summary,
            "delta": _metric_deltas(baseline.summary, candidate.summary),
        },
        "paired_quality": paired,
        "health": {
            "baseline": _aggregate_health(baseline.diagnostics),
            "protected_reference": _aggregate_health(
                protected_reference.diagnostics
            ),
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
        "calibration": candidate.diagnostics["calibration"],
        "privacy": {
            "aggregate_metrics_and_fixed_counters_only": True,
            "row_scenario_message_profile_target_product_and_trace_data_absent": True,
            "per_case_belief_and_fingerprint_values_absent": True,
            "manual_failure_or_small_cell_inspection_performed": False,
            "public_metrics_used_for_fitting": False,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "environment": dict(REQUIRED_ENVIRONMENT),
            "implementation_lock_id": implementation_lock["lock_id"],
            "implementation_lock_sha256": implementation_lock_sha256,
            "suite_lock_id": suite_lock["lock_id"],
            "suite_lock_sha256": suite_lock_sha256,
            "locks_revalidated_after_all_variants": locks_revalidated,
        },
        "decision_gate": {},
    }
    privacy_before_gates = publication_privacy_is_valid(payload)
    gates = _build_gates(
        config,
        baseline,
        protected_reference,
        candidate,
        replay,
        independent,
        paired,
        performance,
        startup,
        limits,
        locks_revalidated=locks_revalidated,
        suite_evidence_valid=(
            dataset_evidence["evaluated_cases"] == len(samples)
            and dataset_evidence["duplicate_cases"] == 0
            and dataset_evidence["duplicate_targets"] == 0
            and _baseline_matches_protected_lock(config, baseline)
        ),
        privacy_valid=privacy_before_gates,
    )
    payload["decision_gate"] = gates
    if not publication_privacy_is_valid(payload):
        raise RuntimeError("Phase 15 aggregate publication violates privacy")
    _write_json_atomic(output, payload)

    for run in (baseline, protected_reference, candidate, replay, independent):
        run.sessions.clear()
    samples.clear()
    products.clear()
    categories.clear()
    catalog_ids.clear()
    gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one sealed Phase 15 aggregate-only suite"
    )
    parser.add_argument("--suite", choices=tuple(SUITES))
    parser.add_argument("--output")
    parser.add_argument("--thermal-safe-ack", action="store_true")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument(
        "--startup-worker",
        choices=(BASELINE_ID, CANDIDATE_ID),
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments.startup_worker:
        policy = DecisionPolicy(arguments.startup_worker)
        result = _startup_worker(Path(arguments.catalog).resolve(), policy)
        print(json.dumps(result, sort_keys=True))
        return
    if not arguments.suite or not arguments.output:
        parser.error("--suite and --output are required")
    run_protocol_utility_ablation(
        arguments.suite,
        arguments.output,
        thermal_safe_ack=arguments.thermal_safe_ack,
    )


if __name__ == "__main__":
    main()
