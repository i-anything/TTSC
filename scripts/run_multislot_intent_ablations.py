"""One-shot, aggregate-only Phase 11 multi-slot intent evaluation.

Every suite is claimed before its rows are loaded.  Candidate, protected
baseline, exact candidate replay, and an independent starter adapter run
strictly sequentially.  Row-level evaluator data and traces exist only in
memory and are reduced to fixed aggregate evidence before publication.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import sys
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Mapping, Sequence
from unittest.mock import patch

import conversational_search.service as service_module
from conversational_search.intent import (
    LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentReduction,
    IntentReductionStatus,
    apply_user_message_with_trace,
    record_question,
)
from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.ranking import STAGE_A_RANKING_POLICY
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import STAGNATION_AWARE_SLATE_POLICY
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_bm25_rescue_ablations import (
    LATENCY_KEYS,
    ORCHESTRATION_HEALTH_KEYS,
    RANKING_HEALTH_KEYS,
    RESCUE_HEALTH_KEYS,
    ROUTE_HEALTH_KEYS,
    SLATE_HEALTH_KEYS,
    _canonical_private_cache_snapshot,
)
from scripts.run_fusion_ablations import _sha256
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_profile_ablations import (
    OVERALL_METRIC_KEYS,
    PROFILE_HEALTH_KEYS,
    TOKEN_USAGE_KEYS,
    _AuditAgent,
    _CallAuditRetriever,
    _inspect_retained_profile_state,
    _project_profile_health,
    _route_call_count,
    _validate_variant_accounting,
)
from scripts.run_reranking_ablations import _expected_turns
from scripts.verify_phase11_intent_oracle import (
    BASELINE_EQUIVALENCE_CASES as PHASE11_BASELINE_EQUIVALENCE_CASES,
    EXPECTED_SHA256 as PHASE11_ORACLE_SHA256,
    ORACLE_CASES as PHASE11_ORACLE_CASES,
    VALID_ORACLE_CASES as PHASE11_VALID_ORACLE_CASES,
)
from scripts.verify_phase7_stage_a_oracle import (
    EXPECTED_SHA256 as PHASE7_ORACLE_SHA256,
    ORACLE_CASES as PHASE7_ORACLE_CASES,
)
from scripts.verify_phase9_ranking_oracle import (
    EXPECTED_SHA256 as PHASE9_ORACLE_SHA256,
    ORACLE_CASES as PHASE9_ORACLE_CASES,
)
from starter.agent import Agent


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase11-lossless-conservative-multislot-intent-v1"
CANDIDATE_ID = LOSSLESS_MULTI_SLOT_INTENT_POLICY.value
BASELINE_ID = "phase9-bounded-profile-residual-v1"
IMPLEMENTATION_LOCK_ID = "phase11-lossless-multislot-implementation-v1"
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase11_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase11_experiment_contract.json"
BASELINE_LOCK_RELATIVE = "docs/phase11_baseline_lock.json"
DATASET_AUDIT_RELATIVE = "docs/phase11_dataset_audit.json"
RESEARCH_PLAN_RELATIVE = "docs/phase11_research_plan.md"

FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest tests.test_multislot_intent "
    "tests.test_service_multislot_intent "
    "tests.test_multislot_intent_ablations tests.test_intent "
    "tests.test_service tests.test_orchestration tests.test_profile_ranking -q"
)
PHASE7_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase7_stage_a_oracle"
PHASE9_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase9_ranking_oracle"
PHASE11_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase11_intent_oracle"

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
        "conversational_search/intent.py",
        "conversational_search/service.py",
        "starter/agent.py",
        "scripts/run_multislot_intent_ablations.py",
        "scripts/verify_phase11_intent_oracle.py",
        "tests/test_multislot_intent.py",
        "tests/test_service_multislot_intent.py",
        "tests/test_multislot_intent_ablations.py",
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

PARSER_HEALTH_KEYS = (
    "attempts",
    "applied",
    "single_slot",
    "ambiguous",
    "bounds",
    "validation_fallbacks",
    "positive_atoms",
    "exclusion_atoms",
    "clear_atoms",
    "replacement_atoms",
    "residual_atoms",
)
_PARSER_OUTCOME_KEYS = (
    "applied",
    "single_slot",
    "ambiguous",
    "bounds",
    "validation_fallbacks",
)
_STATUS_HEALTH_KEY = {
    IntentReductionStatus.APPLIED: "applied",
    IntentReductionStatus.SINGLE_SLOT: "single_slot",
    IntentReductionStatus.AMBIGUOUS: "ambiguous",
    IntentReductionStatus.BOUNDS: "bounds",
    IntentReductionStatus.VALIDATION_FALLBACK: "validation_fallbacks",
}
_HEX_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
_SAFE_ROUTE_STATUSES = frozenset({"ok", "empty"})
_SAFE_ORCHESTRATION_REASONS = frozenset(
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

_QUALITY_GATE_KEYS = frozenset(
    {
        "candidate_hit_rate_not_below_baseline",
        "baseline_hit_to_candidate_miss_is_zero",
        "candidate_mrr_not_below_baseline",
        "candidate_mttc_not_above_baseline",
        "candidate_technical_score_strictly_improves",
        "candidate_mrr_or_mttc_strictly_improves",
        "paired_bootstrap_lower_95_not_below_zero",
    }
)
_COMMON_GATE_KEYS = _QUALITY_GATE_KEYS | {
    "candidate_replay_is_exact",
    "independent_starter_is_exact",
    "baseline_and_candidate_faults_are_zero",
    "candidate_parser_outcomes_partition_attempts",
    "candidate_parser_validation_fallbacks_are_zero",
    "one_route_document_and_stage_a_call_per_search",
    "all_variants_report_zero_model_and_api_tokens",
    "candidate_warm_p95_ratio_at_most_1_05",
    "candidate_wall_time_ratio_at_most_1_05",
    "candidate_startup_time_ratio_at_most_1_05",
    "candidate_additional_startup_rss_at_most_1mib",
    "candidate_additional_retained_agent_bytes_at_most_2mib",
    "dataset_fingerprint_public_exclusion_and_generator_separation_valid",
    "prelock_tests_oracles_and_source_scope_valid",
    "implementation_lock_revalidated_after_all_variants",
    "aggregate_publication_privacy_valid",
    "advance",
}


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
        return REPOSITORY_ROOT / f"results-phase11-{self.name}.json"

    @property
    def attempt(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase11-{self.name}-attempt.json"


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


class ParserAudit:
    """Fixed-cardinality parser counters plus one transient state prediction."""

    __slots__ = ("_counts", "_pending")

    def __init__(self) -> None:
        self._counts = {key: 0 for key in PARSER_HEALTH_KEYS}
        self._pending: IntentReduction | None = None

    def reduce(self, *args: object, **kwargs: object) -> IntentReduction:
        if self._pending is not None:
            raise RuntimeError("parser audit observed overlapping reductions")
        reduction = apply_user_message_with_trace(*args, **kwargs)  # type: ignore[arg-type]
        key = _STATUS_HEALTH_KEY.get(reduction.status)
        if key is None:
            raise RuntimeError("candidate reducer returned an invalid aggregate status")
        self._counts["attempts"] += 1
        self._counts[key] += 1
        self._counts["positive_atoms"] += reduction.positive_atoms
        self._counts["exclusion_atoms"] += reduction.exclusion_atoms
        self._counts["clear_atoms"] += reduction.clear_atoms
        self._counts["replacement_atoms"] += reduction.replacement_atoms
        self._counts["residual_atoms"] += reduction.residual_atoms
        self._pending = reduction
        return reduction

    def validate_and_clear(
        self,
        agent: ConversationalSearchAgent,
        session_id: str,
        response: object,
    ) -> None:
        reduction = self._pending
        self._pending = None
        if reduction is None:
            raise RuntimeError("candidate response bypassed parser audit")
        expected = reduction.state
        if isinstance(response, dict) and isinstance(
            response.get("ask_attribute"), str
        ):
            expected = record_question(expected, response["ask_attribute"])
        if agent.session_state(session_id) != expected:
            raise RuntimeError("candidate parser state validation failed")

    def summary(self) -> dict[str, int]:
        if self._pending is not None:
            raise RuntimeError("parser audit retained an incomplete reduction")
        if sum(self._counts[key] for key in _PARSER_OUTCOME_KEYS) != self._counts[
            "attempts"
        ]:
            raise RuntimeError("candidate parser outcomes do not partition attempts")
        return dict(self._counts)


class Phase11AuditAgent(_AuditAgent):
    """Existing exact action audit with parser state validation."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        parser_audit: ParserAudit | None,
    ) -> None:
        super().__init__(delegate)
        self._phase11_delegate = delegate
        self._parser_audit = parser_audit

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        if self._parser_audit is not None:
            self._parser_audit.validate_and_clear(
                self._phase11_delegate,
                session_id,
                response,
            )
        return response


@contextmanager
def _capture_candidate_reductions(audit: ParserAudit | None) -> Iterator[None]:
    if audit is None:
        yield
        return
    original = service_module.apply_user_message_with_trace
    service_module.apply_user_message_with_trace = audit.reduce
    try:
        yield
    finally:
        service_module.apply_user_message_with_trace = original


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _content_fingerprint(row: Mapping[str, object]) -> bytes:
    if "user_profile" not in row or "ground_truth" not in row:
        raise RuntimeError("evaluation row schema is incomplete")
    return hashlib.sha256(
        _canonical_json(
            {
                "user_profile": row["user_profile"],
                "ground_truth": row["ground_truth"],
            }
        )
    ).digest()


def _fingerprint_set_digest(fingerprints: set[bytes]) -> str:
    return hashlib.sha256(b"".join(sorted(fingerprints))).hexdigest()


def _load_suite_samples(config: SuiteConfig) -> tuple[list[dict], dict[str, int | str]]:
    public_rows = load_jsonl(REPOSITORY_ROOT / "data/public_set.jsonl")
    public_fingerprints = {_content_fingerprint(row) for row in public_rows}
    if len(public_rows) != PUBLIC_CASES or len(public_fingerprints) != PUBLIC_CASES:
        raise RuntimeError("released public-set fingerprint contract drifted")
    if _fingerprint_set_digest(public_fingerprints) != PUBLIC_SET_SHA256:
        raise RuntimeError("released public-set fingerprint digest drifted")

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
    if len(selected) != config.evaluated_cases:
        raise RuntimeError("deduplicated suite case count drifted")
    observed_set = _fingerprint_set_digest(selected_fingerprints)
    if observed_set != config.fingerprint_set_sha256:
        raise RuntimeError("deduplicated suite fingerprint set drifted")
    return selected, {
        "source_rows": len(source_rows),
        "evaluated_cases": len(selected),
        "public_rows_excluded": public_excluded,
        "duplicate_rows_excluded": duplicate_excluded,
        "fingerprint_set_sha256": observed_set,
    }


def _overall_summary(result: Mapping[str, object]) -> dict:
    usage = result.get("reported_token_usage")
    if not isinstance(usage, dict):
        raise RuntimeError("evaluator token summary is unavailable")
    return {
        **{key: result[key] for key in OVERALL_METRIC_KEYS},
        "reported_token_usage": {
            key: usage[key]
            for key in TOKEN_USAGE_KEYS
        },
    }


def _metric_deltas(baseline: Mapping[str, object], candidate: Mapping[str, object]) -> dict:
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 6)
        for key in OVERALL_METRIC_KEYS
        if key != "sample_count"
    }


def _session_utility(session: Mapping[str, object]) -> float:
    hit = 1.0 if session.get("hit") is True else 0.0
    reciprocal_rank = float(session.get("reciprocal_rank") or 0.0)
    raw_turn = session.get("first_hit_turn")
    first_hit_turn = float(raw_turn) if isinstance(raw_turn, int) else 11.0
    efficiency = max(0.0, min(1.0, (11.0 - first_hit_turn) / 10.0))
    return 0.50 * hit + 0.30 * reciprocal_rank + 0.20 * efficiency


def _exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(candidate_only, baseline_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_statistics(
    baseline_sessions: Sequence[Mapping[str, object]],
    candidate_sessions: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260830,
) -> dict:
    if len(baseline_sessions) != len(candidate_sessions) or not baseline_sessions:
        raise RuntimeError("paired evaluator session counts are invalid")
    strata: dict[str, list[float]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    total_delta = 0.0
    for baseline, candidate in zip(baseline_sessions, candidate_sessions):
        if (
            baseline.get("sample_id") != candidate.get("sample_id")
            or baseline.get("scenario_type") != candidate.get("scenario_type")
        ):
            raise RuntimeError("paired evaluator session order drifted")
        label = baseline.get("scenario_type")
        if not isinstance(label, str) or not label:
            raise RuntimeError("paired evaluator stratum is invalid")
        baseline_hit = baseline.get("hit") is True
        candidate_hit = candidate.get("hit") is True
        transition = (
            "both_hit"
            if baseline_hit and candidate_hit
            else "baseline_only_hit"
            if baseline_hit
            else "candidate_only_hit"
            if candidate_hit
            else "both_miss"
        )
        transitions[transition] += 1
        delta = _session_utility(candidate) - _session_utility(baseline)
        strata[label].append(delta)
        total_delta += delta

    if bootstrap_replicates != 10_000:
        raise ValueError("the frozen bootstrap requires exactly 10,000 replicates")
    rng = random.Random(bootstrap_seed)
    total = len(baseline_sessions)
    replicates: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled_sum = 0.0
        for deltas in strata.values():
            sampled_sum += sum(
                deltas[rng.randrange(len(deltas))]
                for _ in range(len(deltas))
            )
        replicates.append(sampled_sum / total)
    replicates.sort()
    lower_index = math.floor(0.025 * bootstrap_replicates)
    upper_index = math.floor(0.975 * bootstrap_replicates)
    candidate_only = transitions["candidate_only_hit"]
    baseline_only = transitions["baseline_only_hit"]
    return {
        "transitions": {
            key: transitions[key]
            for key in (
                "both_hit",
                "candidate_only_hit",
                "baseline_only_hit",
                "both_miss",
            )
        },
        "mean_utility_delta": round(total_delta / total, 9),
        "bootstrap": {
            "seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "strata": len(strata),
            "lower_95": round(replicates[lower_index], 9),
            "upper_95": round(replicates[upper_index], 9),
        },
        "mcnemar_exact_two_sided_p": round(
            _exact_mcnemar_p(candidate_only, baseline_only),
            9,
        ),
    }


def _canonicalize_private(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__type__": type(value).__qualname__,
            **{
                field.name: _canonicalize_private(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): _canonicalize_private(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_private(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize_private(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("private exactness trace contains an unsupported type")


def _private_digest(*values: object) -> str:
    return hashlib.sha256(
        _canonical_json([_canonicalize_private(value) for value in values])
    ).hexdigest()


def _deep_size(value: object, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    identifier = id(value)
    if identifier in seen:
        return 0
    seen.add(identifier)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        return size + sum(
            _deep_size(key, seen) + _deep_size(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            _deep_size(getattr(value, field.name), seen)
            for field in fields(value)
        )
    return size


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    retained = (
        getattr(agent, "_sessions", None),
        getattr(agent, "_slates", None),
        getattr(agent, "_profile_priors", None),
        getattr(planner, "_entries", None),
    )
    return _deep_size(retained)


def _current_max_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _startup_probe(backend: object, *, iterations: int = 400) -> dict:
    if iterations < 100:
        raise ValueError("startup probe requires at least 100 paired constructions")
    baseline_total = 0
    candidate_total = 0
    for _ in range(iterations):
        started = time.perf_counter_ns()
        baseline = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=backend,
            intent_policy=ROBUST_INTENT_POLICY,
        )
        baseline_total += time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        candidate = ConversationalSearchAgent(
            "unused.jsonl",
            retriever=backend,
            intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
        )
        candidate_total += time.perf_counter_ns() - started
    baseline_bytes = _retained_agent_bytes(baseline)
    rss_before_candidate = _current_max_rss_bytes()
    rss_candidate = ConversationalSearchAgent(
        "unused.jsonl",
        retriever=backend,
        intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    )
    candidate_rss_delta = max(0, _current_max_rss_bytes() - rss_before_candidate)
    candidate_bytes = _retained_agent_bytes(rss_candidate)
    return {
        "iterations": iterations,
        "baseline_total_ms": round(baseline_total / 1_000_000.0, 6),
        "candidate_total_ms": round(candidate_total / 1_000_000.0, 6),
        "candidate_startup_time_ratio": round(
            candidate_total / baseline_total if baseline_total else math.inf,
            6,
        ),
        "candidate_additional_startup_rss_bytes": candidate_rss_delta,
        "baseline_empty_retained_bytes": baseline_bytes,
        "candidate_empty_retained_bytes": candidate_bytes,
    }


def _evaluate_with_deterministic_session_ids(
    audited: Phase11AuditAgent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    identifiers = (uuid.UUID(int=index + 1) for index in range(len(samples)))
    with patch("evaluator.local_evaluator.uuid.uuid4", side_effect=identifiers) as uuid4:
        result = evaluate(audited, samples, catalog_ids, categories, products)
    if uuid4.call_count != len(samples):
        raise RuntimeError("evaluator session isolation count drifted")
    return result


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    *,
    candidate: bool,
) -> VariantRun:
    guarded = _CallAuditRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        intent_policy=(
            LOSSLESS_MULTI_SLOT_INTENT_POLICY
            if candidate
            else ROBUST_INTENT_POLICY
        ),
    )
    parser_audit = ParserAudit() if candidate else None
    audited = Phase11AuditAgent(agent, parser_audit)
    started = time.perf_counter()
    with _capture_candidate_reductions(parser_audit):
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
        raise RuntimeError("response timing coverage is incomplete")
    parser_health = (
        parser_audit.summary()
        if parser_audit is not None
        else {key: 0 for key in PARSER_HEALTH_KEYS}
    )
    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": guarded.summary(),
        "ranking_health": agent.ranking_health,
        "rescue_health": agent.rescue_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "slate_health": agent.slate_health,
        "orchestration_health": agent.orchestration_health,
        "parser_health": parser_health,
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
    if int(diagnostics["rescue_health"]["attempts"]) != 0:
        raise RuntimeError("Phase 11 unexpectedly enabled Phase 10 rescue")
    private = _private_digest(
        audited.action_trace,
        _canonical_private_cache_snapshot(agent),
    )
    evaluator_digest = hashlib.sha256(_canonical_json(result)).hexdigest()
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("evaluator sessions are unavailable for paired checks")
    return VariantRun(
        _overall_summary(result),
        sessions,
        diagnostics,
        evaluator_digest,
        private,
    )


def _run_independent(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> VariantRun:
    agent = Agent(catalog, intent_policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY)
    backend = agent.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable for independent verification")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable for independent verification")
    guarded = _CallAuditRetriever(backend)
    agent._retriever = guarded  # type: ignore[attr-defined]
    parser_audit = ParserAudit()
    audited = Phase11AuditAgent(agent, parser_audit)
    started = time.perf_counter()
    with _capture_candidate_reductions(parser_audit):
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
    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": guarded.summary(),
        "ranking_health": agent.ranking_health,
        "rescue_health": agent.rescue_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "slate_health": agent.slate_health,
        "orchestration_health": agent.orchestration_health,
        "parser_health": parser_audit.summary(),
        "retained_profile_state_valid": _inspect_retained_profile_state(
            agent,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": audited.latency_summary(),
    }
    _validate_variant_accounting(
        diagnostics,
        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    private = _private_digest(
        audited.action_trace,
        _canonical_private_cache_snapshot(agent),
    )
    evaluator_digest = hashlib.sha256(_canonical_json(result)).hexdigest()
    sessions = result.get("sessions")
    if not isinstance(sessions, list):
        raise RuntimeError("independent evaluator sessions are unavailable")
    return VariantRun(
        _overall_summary(result),
        sessions,
        diagnostics,
        evaluator_digest,
        private,
    )


def _warm_backend(catalog: Path, backend: object) -> None:
    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    session_id = "phase11-label-free-runtime-warmup"
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    if int(warmup.ranking_health["successes"]) != 1:
        raise RuntimeError("label-free backend warm-up did not complete safely")


def _deterministic_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        key: diagnostics[key]
        for key in (
            "expected_turns",
            "route_health",
            "ranking_health",
            "rescue_health",
            "profile_health",
            "slate_health",
            "orchestration_health",
            "parser_health",
            "retained_profile_state_valid",
            "retained_agent_bytes",
        )
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.inf


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
        "dense_route_calls": sum(int(value) for value in route["dense"].values()),  # type: ignore[index,union-attr]
        "candidate_document_calls": int(route["candidate_document_calls"]),  # type: ignore[index]
        "stage_a_attempts": int(ranking["attempts"]),  # type: ignore[index]
    }


def _faults_are_zero(diagnostics: Mapping[str, object]) -> bool:
    route = diagnostics["route_health"]  # type: ignore[assignment]
    ranking = diagnostics["ranking_health"]  # type: ignore[assignment]
    rescue = diagnostics["rescue_health"]  # type: ignore[assignment]
    profile = diagnostics["profile_health"]  # type: ignore[assignment]
    slate = diagnostics["slate_health"]  # type: ignore[assignment]
    orchestration = diagnostics["orchestration_health"]  # type: ignore[assignment]
    parser = diagnostics["parser_health"]  # type: ignore[assignment]
    return (
        int(route["fallback_turns"]) == 0  # type: ignore[index]
        and int(ranking["failures"]) == 0  # type: ignore[index]
        and int(ranking["unavailable_skips"]) == 0  # type: ignore[index]
        and int(rescue["attempts"]) == 0  # type: ignore[index]
        and int(profile["parsing_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(slate["failures"]) == 0  # type: ignore[index]
        and int(orchestration["fault_invalidations"]) == 0  # type: ignore[index]
        and int(orchestration["store_rejections"]) == 0  # type: ignore[index]
        and int(parser["validation_fallbacks"]) == 0  # type: ignore[index]
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
            or float(candidate_metrics["mttc"]) < float(baseline_metrics["mttc"])
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
    candidate_calls = _call_accounting(candidate.diagnostics)
    baseline_calls = _call_accounting(baseline.diagnostics)
    quality = _quality_gates(baseline, candidate, paired)
    exact_candidate_replay = (
        replay.evaluator_digest == candidate.evaluator_digest
        and replay.private_digest == candidate.private_digest
        and _deterministic_health(replay.diagnostics)
        == _deterministic_health(candidate.diagnostics)
    )
    exact_independent = (
        independent.evaluator_digest == candidate.evaluator_digest
        and independent.private_digest == candidate.private_digest
        and _deterministic_health(independent.diagnostics)
        == _deterministic_health(candidate.diagnostics)
    )
    candidate_parser = candidate.diagnostics["parser_health"]
    parser_partition = (
        sum(int(candidate_parser[key]) for key in _PARSER_OUTCOME_KEYS)
        == int(candidate_parser["attempts"])
        == int(candidate.diagnostics["expected_turns"])
    )
    call_contract = all(
        calls["searches"]
        == calls["bm25_route_calls"]
        == calls["dense_route_calls"]
        == calls["candidate_document_calls"]
        == calls["stage_a_attempts"]
        for calls in (baseline_calls, candidate_calls)
    )
    gates = {
        **quality,
        "candidate_replay_is_exact": exact_candidate_replay,
        "independent_starter_is_exact": exact_independent,
        "baseline_and_candidate_faults_are_zero": (
            _faults_are_zero(baseline.diagnostics)
            and _faults_are_zero(candidate.diagnostics)
            and _faults_are_zero(replay.diagnostics)
            and _faults_are_zero(independent.diagnostics)
        ),
        "candidate_parser_outcomes_partition_attempts": parser_partition,
        "candidate_parser_validation_fallbacks_are_zero": (
            int(candidate_parser["validation_fallbacks"]) == 0
        ),
        "one_route_document_and_stage_a_call_per_search": call_contract,
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
        "dataset_fingerprint_public_exclusion_and_generator_separation_valid": True,
        "prelock_tests_oracles_and_source_scope_valid": True,
        "implementation_lock_revalidated_after_all_variants": lock_revalidated,
        "aggregate_publication_privacy_valid": privacy_valid,
    }
    if config.name == "public":
        gates["public_baseline_matches_protected_phase9"] = (
            baseline.summary == PHASE9_PUBLIC_METRICS
        )
    gates["advance"] = all(gates.values())
    return gates


def _validate_execution_environment() -> None:
    if any(os.environ.get(key) != value for key, value in REQUIRED_ENVIRONMENT.items()):
        raise RuntimeError("single-thread execution environment is not fully pinned")


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
        raise RuntimeError("candidate introduced Python outside the frozen source scope")
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
        "phase11_oracle_command",
        "phase11_oracle_cases",
        "phase11_valid_oracle_cases",
        "phase11_baseline_equivalence_cases",
        "phase11_oracle_sha256",
        "completed_before_lock",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("Phase 11 pre-lock verification schema drifted")
    expected = {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "complete_suite_command": FULL_SUITE_COMMAND,
        "phase7_oracle_command": PHASE7_ORACLE_COMMAND,
        "phase7_oracle_cases": PHASE7_ORACLE_CASES,
        "phase7_oracle_sha256": PHASE7_ORACLE_SHA256,
        "phase9_oracle_command": PHASE9_ORACLE_COMMAND,
        "phase9_oracle_cases": PHASE9_ORACLE_CASES,
        "phase9_oracle_sha256": PHASE9_ORACLE_SHA256,
        "phase11_oracle_command": PHASE11_ORACLE_COMMAND,
        "phase11_oracle_cases": PHASE11_ORACLE_CASES,
        "phase11_valid_oracle_cases": PHASE11_VALID_ORACLE_CASES,
        "phase11_baseline_equivalence_cases": (
            PHASE11_BASELINE_EQUIVALENCE_CASES
        ),
        "phase11_oracle_sha256": PHASE11_ORACLE_SHA256,
        "completed_before_lock": True,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("Phase 11 pre-lock verification evidence drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] <= 0:
            raise RuntimeError("Phase 11 pre-lock test counts are invalid")
    return value


def _validate_implementation_lock(repository_root: Path = REPOSITORY_ROOT) -> dict:
    lock_path = repository_root / IMPLEMENTATION_LOCK_RELATIVE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
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
        raise RuntimeError("Phase 11 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 11 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 11 implementation lock identity")
    if lock.get("status") != "locked_before_generator_evaluation":
        raise RuntimeError("Phase 11 candidate is not frozen")
    expected_documents = {
        "contract_sha256": CONTRACT_RELATIVE,
        "baseline_lock_sha256": BASELINE_LOCK_RELATIVE,
        "dataset_audit_sha256": DATASET_AUDIT_RELATIVE,
        "research_plan_sha256": RESEARCH_PLAN_RELATIVE,
    }
    for key, relative in expected_documents.items():
        value = lock.get(key)
        if (
            not isinstance(value, str)
            or not _HEX_SHA256_RE.fullmatch(value)
            or value != _sha256(repository_root / relative)
        ):
            raise RuntimeError("Phase 11 planning artifact drifted after lock")
    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 11 implementation source lock is incomplete")
    observed = {
        relative: _sha256(repository_root / relative)
        for relative in SOURCE_PATHS
    }
    if observed != source_hashes:
        raise RuntimeError("Phase 11 implementation drifted after lock")
    _validate_prelock_verification(lock.get("verification"))
    _validate_baseline_scope(repository_root)
    return lock


def _claim_attempt(path: Path, suite: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = _canonical_json(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "suite": suite,
                "status": "claimed",
            }
        ) + b"\n"
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_prerequisites(config: SuiteConfig) -> None:
    for suite in config.prerequisites:
        path = SUITES[suite].output
        if not path.is_file():
            raise RuntimeError("a prior aggregate suite result is unavailable")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not publication_privacy_is_valid(payload):
            raise RuntimeError("a prior suite result violates the privacy contract")
        gates = payload.get("decision_gate")
        if not isinstance(gates, dict) or gates.get("advance") is not True:
            raise RuntimeError("a prior suite did not pass every promotion gate")


def _validate_run_paths(config: SuiteConfig, output: Path) -> None:
    if output.resolve() != config.output.resolve():
        raise ValueError("each Phase 11 suite has one frozen aggregate output path")
    if output.exists():
        raise FileExistsError("Phase 11 suite output already exists")
    if config.attempt.exists():
        raise FileExistsError("Phase 11 suite attempt was already consumed")
    if not config.source.is_file():
        raise FileNotFoundError("Phase 11 suite source is unavailable")


def publication_privacy_is_valid(
    payload: object,
    *,
    allow_missing_decision_gate: bool = False,
) -> bool:
    if not isinstance(payload, dict):
        return False
    expected_top = {
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
    if allow_missing_decision_gate:
        expected_top.remove("decision_gate")
    if set(payload) != expected_top:
        return False
    if payload.get("schema_version") != SCHEMA_VERSION:
        return False
    if payload.get("experiment_id") != EXPERIMENT_ID:
        return False
    suite = payload.get("suite")
    if suite not in SUITES:
        return False

    def exact_keys(value: object, expected: Sequence[str] | set[str]) -> bool:
        return isinstance(value, dict) and set(value) == set(expected)

    def nonnegative_integer(value: object) -> bool:
        return type(value) is int and value >= 0

    def finite_number(value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )

    def hash_value(value: object) -> bool:
        return isinstance(value, str) and _HEX_SHA256_RE.fullmatch(value) is not None

    dataset = payload.get("dataset")
    if not exact_keys(
        dataset,
        {
            "source_sha256",
            "source_rows",
            "evaluated_cases",
            "public_rows_excluded",
            "duplicate_rows_excluded",
            "fingerprint_set_sha256",
        },
    ):
        return False
    if not hash_value(dataset["source_sha256"]) or not hash_value(  # type: ignore[index]
        dataset["fingerprint_set_sha256"]  # type: ignore[index]
    ):
        return False
    if any(
        not nonnegative_integer(dataset[key])  # type: ignore[index]
        for key in (
            "source_rows",
            "evaluated_cases",
            "public_rows_excluded",
            "duplicate_rows_excluded",
        )
    ):
        return False

    run_configuration = payload.get("run_configuration")
    if not exact_keys(
        run_configuration,
        {
            "execution",
            "threads",
            "processes",
            "shared_immutable_backend",
            "fresh_agent_state_per_variant",
            "variant_order",
            "backend_warmup",
            "thermal_safe_acknowledged",
            "external_api_calls",
            "gpu_or_mps",
        },
    ):
        return False
    expected_run_configuration = {
        "execution": "strictly_sequential_cpu",
        "threads": 1,
        "processes": 1,
        "shared_immutable_backend": True,
        "fresh_agent_state_per_variant": True,
        "variant_order": [
            CANDIDATE_ID,
            BASELINE_ID,
            CANDIDATE_ID,
            "independent_explicit_policy_starter",
        ],
        "backend_warmup": "one_fixed_label_free_request",
        "thermal_safe_acknowledged": True,
        "external_api_calls": 0,
        "gpu_or_mps": False,
    }
    if run_configuration != expected_run_configuration:
        return False

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"baseline", "candidate", "delta"}:
        return False
    summary_keys = {*OVERALL_METRIC_KEYS, "reported_token_usage"}
    for name in ("baseline", "candidate"):
        summary = metrics.get(name)
        if not isinstance(summary, dict) or set(summary) != summary_keys:
            return False
        usage = summary.get("reported_token_usage")
        if not isinstance(usage, dict) or set(usage) != set(TOKEN_USAGE_KEYS):
            return False
        if not nonnegative_integer(summary["sample_count"]):
            return False
        if any(
            not finite_number(summary[key])
            for key in OVERALL_METRIC_KEYS
            if key != "sample_count"
        ):
            return False
        if any(not nonnegative_integer(usage[key]) for key in TOKEN_USAGE_KEYS):
            return False
    delta_keys = set(OVERALL_METRIC_KEYS) - {"sample_count"}
    delta = metrics.get("delta")
    if not exact_keys(delta, delta_keys) or any(
        not finite_number(delta[key]) for key in delta_keys  # type: ignore[index]
    ):
        return False

    paired = payload.get("paired_quality")
    if not exact_keys(
        paired,
        {
            "transitions",
            "mean_utility_delta",
            "bootstrap",
            "mcnemar_exact_two_sided_p",
        },
    ):
        return False
    transitions = paired["transitions"]  # type: ignore[index]
    transition_keys = {
        "both_hit",
        "candidate_only_hit",
        "baseline_only_hit",
        "both_miss",
    }
    if not exact_keys(transitions, transition_keys) or any(
        not nonnegative_integer(transitions[key])  # type: ignore[index]
        for key in transition_keys
    ):
        return False
    bootstrap = paired["bootstrap"]  # type: ignore[index]
    if not exact_keys(
        bootstrap,
        {"seed", "replicates", "strata", "lower_95", "upper_95"},
    ):
        return False
    if (
        bootstrap["seed"] != 20260830  # type: ignore[index]
        or bootstrap["replicates"] != 10_000  # type: ignore[index]
        or not nonnegative_integer(bootstrap["strata"])  # type: ignore[index]
        or not finite_number(bootstrap["lower_95"])  # type: ignore[index]
        or not finite_number(bootstrap["upper_95"])  # type: ignore[index]
        or not finite_number(paired["mean_utility_delta"])  # type: ignore[index]
        or not finite_number(paired["mcnemar_exact_two_sided_p"])  # type: ignore[index]
    ):
        return False

    def counter_health(
        value: object,
        expected_keys: Sequence[str],
        *,
        mapping_keys: set[str] | frozenset[str] = frozenset(),
    ) -> bool:
        if not exact_keys(value, expected_keys):
            return False
        for key, item in value.items():  # type: ignore[union-attr]
            if key == "policy":
                if not isinstance(item, str) or not item:
                    return False
            elif key in mapping_keys:
                if not isinstance(item, dict) or any(
                    not isinstance(label, str)
                    or not label
                    or not nonnegative_integer(count)
                    for label, count in item.items()
                ):
                    return False
            elif not nonnegative_integer(item):
                return False
        return True

    diagnostic_keys = {
        "expected_turns",
        "route_health",
        "ranking_health",
        "rescue_health",
        "profile_health",
        "slate_health",
        "orchestration_health",
        "parser_health",
        "retained_profile_state_valid",
        "retained_agent_bytes",
        "evaluation_wall_seconds",
        "respond_latency_ms",
    }

    def diagnostic_is_valid(value: object) -> bool:
        if not exact_keys(value, diagnostic_keys):
            return False
        if not nonnegative_integer(value["expected_turns"]):  # type: ignore[index]
            return False
        if not counter_health(
            value["route_health"],  # type: ignore[index]
            ROUTE_HEALTH_KEYS,
            mapping_keys={"bm25", "dense"},
        ):
            return False
        route = value["route_health"]  # type: ignore[index]
        if any(
            not set(route[key]).issubset(_SAFE_ROUTE_STATUSES)  # type: ignore[index]
            for key in ("bm25", "dense")
        ):
            return False
        for key, expected in (
            ("ranking_health", RANKING_HEALTH_KEYS),
            ("rescue_health", RESCUE_HEALTH_KEYS),
            ("profile_health", PROFILE_HEALTH_KEYS),
            ("slate_health", SLATE_HEALTH_KEYS),
            ("parser_health", PARSER_HEALTH_KEYS),
        ):
            if not counter_health(value[key], expected):  # type: ignore[index]
                return False
        if not counter_health(
            value["orchestration_health"],  # type: ignore[index]
            ORCHESTRATION_HEALTH_KEYS,
            mapping_keys={"reasons"},
        ):
            return False
        expected_policies = {
            "ranking_health": STAGE_A_RANKING_POLICY.value,
            "rescue_health": STAGE_A_RANKING_POLICY.value,
            "profile_health": BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            "slate_health": STAGNATION_AWARE_SLATE_POLICY.value,
            "orchestration_health": (
                EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value
            ),
        }
        if any(
            value[key]["policy"] != policy  # type: ignore[index]
            for key, policy in expected_policies.items()
        ):
            return False
        reasons = value["orchestration_health"]["reasons"]  # type: ignore[index]
        if not set(reasons).issubset(_SAFE_ORCHESTRATION_REASONS):
            return False
        latency = value["respond_latency_ms"]  # type: ignore[index]
        if not exact_keys(latency, LATENCY_KEYS) or any(
            not finite_number(latency[key]) for key in LATENCY_KEYS  # type: ignore[index]
        ):
            return False
        return (
            type(value["retained_profile_state_valid"]) is bool  # type: ignore[index]
            and nonnegative_integer(value["retained_agent_bytes"])  # type: ignore[index]
            and finite_number(value["evaluation_wall_seconds"])  # type: ignore[index]
        )

    health = payload.get("health")
    health_keys = {
        "baseline",
        "candidate",
        "candidate_replay",
        "independent_candidate",
    }
    if not exact_keys(health, health_keys) or any(
        not diagnostic_is_valid(health[key]) for key in health_keys  # type: ignore[index]
    ):
        return False

    call_accounting = payload.get("call_accounting")
    call_keys = {
        "searches",
        "bm25_route_calls",
        "dense_route_calls",
        "candidate_document_calls",
        "stage_a_attempts",
    }
    if not exact_keys(call_accounting, {"baseline", "candidate"}):
        return False
    for name in ("baseline", "candidate"):
        calls = call_accounting[name]  # type: ignore[index]
        if not exact_keys(calls, call_keys) or any(
            not nonnegative_integer(calls[key]) for key in call_keys  # type: ignore[index]
        ):
            return False

    performance_keys = {
        "baseline_wall_seconds",
        "candidate_wall_seconds",
        "candidate_wall_time_ratio",
        "baseline_warm_p95_ms",
        "candidate_warm_p95_ms",
        "candidate_warm_p95_ratio",
        "baseline_retained_agent_bytes",
        "candidate_retained_agent_bytes",
        "candidate_additional_retained_agent_bytes",
    }
    performance = payload.get("performance")
    if not exact_keys(performance, performance_keys) or any(
        not finite_number(performance[key]) for key in performance_keys  # type: ignore[index]
    ):
        return False
    startup_keys = {
        "iterations",
        "baseline_total_ms",
        "candidate_total_ms",
        "candidate_startup_time_ratio",
        "candidate_additional_startup_rss_bytes",
        "baseline_empty_retained_bytes",
        "candidate_empty_retained_bytes",
    }
    startup = payload.get("startup")
    if not exact_keys(startup, startup_keys) or any(
        not finite_number(startup[key]) for key in startup_keys  # type: ignore[index]
    ):
        return False

    exactness_keys = {
        "candidate_replay_evaluator_payload_equal",
        "candidate_replay_action_state_slate_cache_equal",
        "candidate_replay_aggregate_health_equal",
        "independent_evaluator_payload_equal",
        "independent_action_state_slate_cache_equal",
        "independent_aggregate_health_equal",
    }
    exactness = payload.get("exactness")
    if not exact_keys(exactness, exactness_keys) or any(
        type(exactness[key]) is not bool for key in exactness_keys  # type: ignore[index]
    ):
        return False
    privacy_keys = {
        "aggregate_metrics_and_fixed_counters_only",
        "row_scenario_message_profile_target_product_and_trace_data_absent",
        "per_case_fingerprints_absent",
        "manual_failure_inspection_performed",
    }
    privacy = payload.get("privacy")
    if not exact_keys(privacy, privacy_keys) or any(
        type(privacy[key]) is not bool for key in privacy_keys  # type: ignore[index]
    ):
        return False

    reproducibility_keys = {
        "platform",
        "python",
        "environment",
        "implementation_lock_id",
        "contract_sha256",
        "source_sha256",
        "phase11_oracle_sha256",
        "implementation_lock_revalidated_after_independent",
    }
    reproducibility = payload.get("reproducibility")
    if not exact_keys(reproducibility, reproducibility_keys):
        return False
    if (
        not isinstance(reproducibility["platform"], str)  # type: ignore[index]
        or not isinstance(reproducibility["python"], str)  # type: ignore[index]
        or reproducibility["environment"] != REQUIRED_ENVIRONMENT  # type: ignore[index]
        or reproducibility["implementation_lock_id"] != IMPLEMENTATION_LOCK_ID  # type: ignore[index]
        or not hash_value(reproducibility["contract_sha256"])  # type: ignore[index]
        or reproducibility["phase11_oracle_sha256"] != PHASE11_ORACLE_SHA256  # type: ignore[index]
        or type(reproducibility["implementation_lock_revalidated_after_independent"])  # type: ignore[index]
        is not bool
    ):
        return False
    source_hashes = reproducibility["source_sha256"]  # type: ignore[index]
    if not exact_keys(source_hashes, set(SOURCE_PATHS)) or any(
        not hash_value(source_hashes[key]) for key in SOURCE_PATHS  # type: ignore[index]
    ):
        return False

    if not allow_missing_decision_gate:
        expected_gate_keys = set(_COMMON_GATE_KEYS)
        if suite == "public":
            expected_gate_keys.add("public_baseline_matches_protected_phase9")
        decision_gate = payload.get("decision_gate")
        if not exact_keys(decision_gate, expected_gate_keys) or any(
            type(decision_gate[key]) is not bool for key in expected_gate_keys  # type: ignore[index]
        ):
            return False

    forbidden_keys = {
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
        "summary",
        "rating_style",
        "average_prior_rating",
        "purchase_frequency",
        "action_trace",
        "slate_state",
        "intent_state",
        "profile_mask",
        "theme_mask",
        "category",
        "categories",
        "product_id",
        "product_ids",
        "parent_asin",
        "recommendations",
        "query",
        "dense_query",
        "lexical_query",
        "message",
        "rows",
        "per_session",
        "fingerprints",
    }

    def visit(value: object) -> tuple[set[str], set[str]]:
        if isinstance(value, dict):
            keys = {str(key) for key in value}
            strings: set[str] = set()
            for item in value.values():
                child_keys, child_strings = visit(item)
                keys.update(child_keys)
                strings.update(child_strings)
            return keys, strings
        if isinstance(value, (list, tuple)):
            keys: set[str] = set()
            strings: set[str] = set()
            for item in value:
                child_keys, child_strings = visit(item)
                keys.update(child_keys)
                strings.update(child_strings)
            return keys, strings
        return set(), ({value} if isinstance(value, str) else set())

    keys, strings = visit(payload)
    if not forbidden_keys.isdisjoint(keys):
        return False
    if not {"boundary", "browsing", "buying", "intent_override"}.isdisjoint(
        keys | strings
    ):
        return False
    if _ASIN_RE.search(json.dumps(payload, sort_keys=True)):
        return False
    return True


def _aggregate_health(diagnostics: Mapping[str, object]) -> dict:
    return {
        key: diagnostics[key]
        for key in (
            "expected_turns",
            "route_health",
            "ranking_health",
            "rescue_health",
            "profile_health",
            "slate_health",
            "orchestration_health",
            "parser_health",
            "retained_profile_state_valid",
            "retained_agent_bytes",
            "evaluation_wall_seconds",
            "respond_latency_ms",
        )
    }


def run_multislot_intent_ablation(
    suite: str,
    output_path: str | Path,
    *,
    thermal_safe_ack: bool,
) -> dict:
    if suite not in SUITES:
        raise ValueError("unsupported Phase 11 suite")
    if thermal_safe_ack is not True:
        raise RuntimeError("thermal safety must be checked before claiming a suite")
    config = SUITES[suite]
    output = Path(output_path).resolve()
    _validate_execution_environment()
    implementation_lock = _validate_implementation_lock()
    _validate_prerequisites(config)
    _validate_run_paths(config, output)
    if _sha256(REPOSITORY_ROOT / "data/catalog.jsonl") != CATALOG_SHA256:
        raise RuntimeError("catalog drifted before the sealed suite")

    # The exclusive attempt claim deliberately precedes source hashing or loading.
    _claim_attempt(config.attempt, config.name)
    if _sha256(config.source) != config.source_sha256:
        raise RuntimeError("suite source hash drifted after its attempt was claimed")
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
        raise RuntimeError("dense retrieval is unavailable; sealed suite consumed")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; sealed suite consumed")
    _warm_backend(catalog, backend)
    startup = _startup_probe(backend)

    candidate = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        candidate=True,
    )
    baseline = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        candidate=False,
    )
    replay = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        candidate=True,
    )
    independent = _run_independent(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
    )
    revalidated_lock = _validate_implementation_lock()
    lock_revalidated = revalidated_lock == implementation_lock

    paired = _paired_statistics(baseline.sessions, candidate.sessions)
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
                "independent_explicit_policy_starter",
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
            "phase11_oracle_sha256": PHASE11_ORACLE_SHA256,
            "implementation_lock_revalidated_after_independent": lock_revalidated,
        },
    }
    privacy_valid = publication_privacy_is_valid(
        payload,
        allow_missing_decision_gate=True,
    )
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
        privacy_valid=privacy_valid,
    )
    payload["decision_gate"] = gates
    if not publication_privacy_is_valid(payload):
        raise RuntimeError("Phase 11 aggregate publication violates its privacy contract")
    _write_json_atomic(output, payload)

    # Drop row-level objects explicitly after all in-memory paired checks.
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
        description="Run one frozen Phase 11 generator-separated suite"
    )
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thermal-safe-ack", action="store_true")
    args = parser.parse_args()
    run_multislot_intent_ablation(
        args.suite,
        args.output,
        thermal_safe_ack=args.thermal_safe_ack,
    )


if __name__ == "__main__":
    main()
