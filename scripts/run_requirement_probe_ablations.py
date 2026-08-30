"""Sealed aggregate-only Phase 14 requirement-probe evaluation.

The harness never publishes evaluator rows, target IDs, messages, profiles,
queries, route lists, or per-case outcomes.  Private rows exist only in memory
for paired statistics and deterministic equality digests and are cleared before
the aggregate result is returned.
"""

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
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.retrieval import (
    CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    DISABLED_REQUIREMENT_PROBE_POLICY,
    MAX_CANDIDATE_DOCUMENTS,
    MAX_REQUIREMENT_PROBES,
    REQUIREMENT_PROBE_CAPABILITY,
    RequirementProbePolicy,
    RequirementProbeRetrievalResult,
    RetrievalResult,
)
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    customer_reply,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from scripts.run_bm25_rescue_ablations import (
    _canonical_private_cache_snapshot,
)
from scripts.run_fusion_ablations import _sha256
from scripts.run_intent_epoch_slate_ablations import (
    _validate_phase13_accounting,
)
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
    _inspect_retained_profile_state,
    _project_profile_health,
    _validate_variant_accounting,
)
from scripts.run_reranking_ablations import _expected_turns
from scripts.verify_phase13_slate_oracle import (
    EXPECTED_SHA256 as PHASE13_ORACLE_SHA256,
    RANDOM_CASES as PHASE13_RANDOM_ORACLE_CASES,
)
from scripts.verify_phase14_requirement_probe_oracle import (
    EXPECTED_SHA256 as PHASE14_ORACLE_SHA256,
    RANDOM_CASES as PHASE14_RANDOM_ORACLE_CASES,
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
EXPERIMENT_ID = "phase14-catalog-idf-requirement-probes-v1"
CANDIDATE_ID = CATALOG_IDF_REQUIREMENT_PROBE_POLICY.value
BASELINE_ID = "phase13-intent-epoch-continuation-novelty-v1"
IMPLEMENTATION_LOCK_ID = "phase14-catalog-idf-requirement-probes-implementation-v1"
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase14_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase14_experiment_contract.json"
BASELINE_LOCK_RELATIVE = "docs/phase14_baseline_lock.json"
FRESH_LOCK_RELATIVE = "docs/phase14_fresh_suite_lock.json"
RESEARCH_PLAN_RELATIVE = "docs/phase14_research_plan.md"

FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest "
    "tests.test_requirement_probes "
    "tests.test_service_requirement_probes "
    "tests.test_phase14_requirement_probe_oracle "
    "tests.test_requirement_probe_ablations -q"
)
PHASE7_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase7_stage_a_oracle"
PHASE9_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase9_ranking_oracle"
PHASE13_ORACLE_COMMAND = ".venv/bin/python -m scripts.verify_phase13_slate_oracle"
PHASE14_ORACLE_COMMAND = (
    ".venv/bin/python -m scripts.verify_phase14_requirement_probe_oracle"
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
            FRESH_LOCK_RELATIVE,
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
ALLOWED_RUNTIME_CHANGES = frozenset(
    {
        "conversational_search/intent.py",
        "conversational_search/orchestration.py",
        "conversational_search/retrieval.py",
        "conversational_search/service.py",
    }
)

CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
PUBLIC_SHA256 = "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579"
PLAIN_SHA256 = "f2cdf94b8dbdf22373f42dd661f22372e92715d7bcbc924590db26cf824894db"
SCENARIO_AWARE_SHA256 = "78da5e8402bd7d6c7d9eee86de24eec9a13d3e433ed6f3cfe720bf85cb3319c9"
FRESH_SHA256 = "d3fd6c75ad36ff5057e56388d0e2e26abc8359452dd76b4a135479defac257a1"
DEVELOPMENT_SET_SHA256 = "37bb8265543198c33305d40c6facf26f76e9109ac6f68afb529ef6a53b19eabd"
VALIDATION_SET_SHA256 = "a2677eb857c8df9ed963818c7c854c2d3ec936b7d597ed9810248fbc467f8ad1"
PUBLIC_SET_SHA256 = "a10adf5ef5e6424c749b0d97e2c7186da62ec76572de4bfbd2a28e8d42b3cb34"
FRESH_SET_SHA256 = "452a25afef920b7c1e4aff38e5501b243f46f9dd7cb7f25d6eba7e5c236fe008"
PUBLIC_CASES = 200
BOOTSTRAP_SEED = 140260830

REQUIRED_ENVIRONMENT = {
    "TTSC_ONNX_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PHASE13_PROTECTED_METRICS = {
    "development": {
        "sample_count": 996,
        "hit_rate_at_10": 0.991968,
        "mrr": 0.566126,
        "mttc": 2.90261,
        "efficiency": 0.809739,
        "recommended_technical_score": 0.82777,
    },
    "validation": {
        "sample_count": 1000,
        "hit_rate_at_10": 0.975,
        "mrr": 0.639632,
        "mttc": 2.831,
        "efficiency": 0.8169,
        "recommended_technical_score": 0.84277,
    },
    "public": {
        "sample_count": 200,
        "hit_rate_at_10": 0.99,
        "mrr": 0.556748,
        "mttc": 2.91,
        "efficiency": 0.809,
        "recommended_technical_score": 0.823824,
    },
}

PROBE_HEALTH_KEYS = (
    "policy",
    "attempts",
    "disabled",
    "no_eligible",
    "capacity",
    "successful_supplements",
    "empty_routes",
    "no_additions",
    "unavailable",
    "errors",
    "selected_probe_queries",
    "supplemental_ids",
    "validation_or_execution_fallbacks",
)
_PROBE_OUTCOME_KEYS = PROBE_HEALTH_KEYS[2:9]
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
        "phase14_messages",
        "phase14_family",
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
    manual_messages: bool = False
    prerequisites: tuple[str, ...] = ()

    @property
    def output(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase14-{self.name}.json"

    @property
    def attempt(self) -> Path:
        return REPOSITORY_ROOT / f"results-phase14-{self.name}-attempt.json"


SUITES = {
    "fresh": SuiteConfig(
        "fresh",
        Path("/private/tmp/ttsc-phase14-explicit-card-v1.jsonl"),
        FRESH_SHA256,
        384,
        384,
        FRESH_SET_SHA256,
        False,
        True,
    ),
    "development": SuiteConfig(
        "development",
        Path("/Users/limzichao/Downloads/public_plus_synthetic_1200.jsonl"),
        PLAIN_SHA256,
        1_200,
        996,
        DEVELOPMENT_SET_SHA256,
        True,
        False,
        ("fresh",),
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
        False,
        ("fresh", "development"),
    ),
    "public": SuiteConfig(
        "public",
        REPOSITORY_ROOT / "data/public_set.jsonl",
        PUBLIC_SHA256,
        200,
        200,
        PUBLIC_SET_SHA256,
        False,
        False,
        ("fresh", "development", "validation"),
    ),
}


@dataclass(slots=True)
class VariantRun:
    summary: dict
    sessions: list[dict]
    diagnostics: dict
    evaluator_digest: str
    private_digest: str


class ProbeCallAuditRetriever:
    """Audit aggregate route/probe invariants without retaining route IDs."""

    def __init__(self, backend: object, catalog_ids: set[str]) -> None:
        self._backend = backend
        self._catalog_ids = catalog_ids
        self._bm25: Counter[str] = Counter()
        self._dense: Counter[str] = Counter()
        self._fallbacks = 0
        self._candidate_document_calls = 0
        self._enabled_searches = 0
        self._vocabulary_lookup_upper_bound = 0
        self._probe_bm25_queries = 0
        self._max_probe_queries = 0
        self._max_fused_union = 0
        self._incumbent_loss_violations = 0
        self._candidate_bound_violations = 0
        self._invalid_or_duplicate_id_violations = 0
        self._probe_overlap_violations = 0

    @property
    def ranking_cache_capability(self) -> object:
        return self._backend.ranking_cache_capability

    @property
    def snapshot_token(self) -> object:
        return self._backend.snapshot_token

    @property
    def requirement_probe_capability(self) -> object:
        return self._backend.requirement_probe_capability

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        route_weights: RouteWeights,
        requirement_probe_policy: RequirementProbePolicy = (
            DISABLED_REQUIREMENT_PROBE_POLICY
        ),
        requirement_probe_candidates: Sequence[str] = (),
    ) -> RetrievalResult:
        result = self._backend.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            route_weights=route_weights,
            requirement_probe_policy=requirement_probe_policy,
            requirement_probe_candidates=requirement_probe_candidates,
        )
        if not isinstance(result, RetrievalResult):
            raise TypeError("search_with_trace must return RetrievalResult")
        trace = result.trace
        self._bm25[trace.bm25_status] += 1
        self._dense[trace.dense_status] += 1
        self._fallbacks += int(trace.used_fallback)
        self._max_fused_union = max(self._max_fused_union, len(trace.fused_ids))
        base_bm25_ids = trace.bm25_ids
        supplemental_ids: tuple[str, ...] = ()
        if requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY:
            if not isinstance(result, RequirementProbeRetrievalResult):
                raise TypeError(
                    "enabled search must return RequirementProbeRetrievalResult"
                )
            self._enabled_searches += 1
            probe_trace = result.probe_trace
            base_bm25_ids = probe_trace.base_bm25_ids
            supplemental_ids = probe_trace.supplemental_ids
            queries = probe_trace.query_count
            self._probe_bm25_queries += queries
            self._max_probe_queries = max(self._max_probe_queries, queries)
            if probe_trace.status not in {"capacity", "unavailable"}:
                self._vocabulary_lookup_upper_bound += 1

        incumbent = set((*base_bm25_ids, *trace.dense_ids))
        effective = set(trace.fused_ids)
        if not incumbent.issubset(effective):
            self._incumbent_loss_violations += 1
        if len(trace.fused_ids) > MAX_CANDIDATE_DOCUMENTS:
            self._candidate_bound_violations += 1
        routes = (
            base_bm25_ids,
            supplemental_ids,
            trace.bm25_ids,
            trace.dense_ids,
            trace.fused_ids,
        )
        if any(
            len(route) != len(set(route))
            or any(parent_asin not in self._catalog_ids for parent_asin in route)
            for route in routes
        ):
            self._invalid_or_duplicate_id_violations += 1
        if set(supplemental_ids) & incumbent:
            self._probe_overlap_violations += 1
        return result

    def candidate_documents(self, parent_asins: Sequence[str]) -> tuple:
        self._candidate_document_calls += 1
        return self._backend.candidate_documents(parent_asins)

    def validate(self, expected_searches: int) -> None:
        if sum(self._bm25.values()) != expected_searches:
            raise RuntimeError("route trace coverage is incomplete")
        if set(self._bm25) - {"ok", "empty"} or set(self._dense) - {"ok", "empty"}:
            raise RuntimeError("route faults invalidate Phase 14")

    def summary(self) -> dict[str, object]:
        return {
            "bm25": dict(sorted(self._bm25.items())),
            "dense": dict(sorted(self._dense.items())),
            "fallback_turns": self._fallbacks,
            "candidate_document_calls": self._candidate_document_calls,
            "probe_enabled_searches": self._enabled_searches,
            "vocabulary_lookup_upper_bound": self._vocabulary_lookup_upper_bound,
            "probe_bm25_queries": self._probe_bm25_queries,
            "max_probe_queries_per_search": self._max_probe_queries,
            "max_fused_union": self._max_fused_union,
            "incumbent_loss_violations": self._incumbent_loss_violations,
            "candidate_bound_violations": self._candidate_bound_violations,
            "invalid_or_duplicate_id_violations": (
                self._invalid_or_duplicate_id_violations
            ),
            "probe_overlap_violations": self._probe_overlap_violations,
        }


def _fresh_row_fingerprint(row: Mapping[str, object]) -> bytes:
    return hashlib.sha256(_canonical_json(row)).digest()


def _load_suite_samples(
    config: SuiteConfig,
) -> tuple[list[dict], dict[str, int | str]]:
    if not config.manual_messages:
        public_rows = load_jsonl(REPOSITORY_ROOT / "data/public_set.jsonl")
        public_fingerprints = {_content_fingerprint(row) for row in public_rows}
        if len(public_rows) != PUBLIC_CASES or len(public_fingerprints) != PUBLIC_CASES:
            raise RuntimeError("released public fingerprint contract drifted")
        if _fingerprint_set_digest(public_fingerprints) != PUBLIC_SET_SHA256:
            raise RuntimeError("released public fingerprint digest drifted")
    else:
        public_fingerprints = set()

    source_rows = load_jsonl(config.source)
    if len(source_rows) != config.source_rows:
        raise RuntimeError("suite source row count drifted")
    selected: list[dict] = []
    selected_fingerprints: set[bytes] = set()
    public_excluded = 0
    duplicate_excluded = 0
    for row in source_rows:
        fingerprint = (
            _fresh_row_fingerprint(row)
            if config.manual_messages
            else _content_fingerprint(row)
        )
        if config.exclude_public and fingerprint in public_fingerprints:
            public_excluded += 1
            continue
        if fingerprint in selected_fingerprints:
            duplicate_excluded += 1
            continue
        if config.manual_messages:
            messages = row.get("phase14_messages")
            if (
                row.get("scenario_type") != "buying"
                or not isinstance(messages, list)
                or len(messages) != 2
                or any(not isinstance(message, str) or not message for message in messages)
            ):
                raise RuntimeError("fresh manual-message schema drifted")
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
        "manual_message_cases": len(selected) if config.manual_messages else 0,
        "manual_messages_per_case": 2 if config.manual_messages else 0,
    }


def _evaluate_manual_messages(
    agent: _AuditAgent,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> dict:
    """Evaluate the locked first-two-turn messages, then resume the simulator."""

    del categories
    sessions: list[dict] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for index, sample in enumerate(samples):
        session_id = f"public_{uuid.UUID(int=index + 1).hex}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        messages = sample.get("phase14_messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise RuntimeError("fresh suite messages are unavailable")
        hard = [str(value) for value in card.get("hard_constraints") or []]
        disclosed: set[str] = set()
        if hard:
            disclosed.add(hard[0])
        boundary_used = False
        user_message = str(messages[0])
        hit_turn: int | None = None
        best_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            if not isinstance(response, dict) or not isinstance(response.get("message"), str):
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if type(prompt) is int and prompt >= 0:
                    total_prompt_tokens += prompt
                if type(completion) is int and completion >= 0:
                    total_completion_tokens += completion
            ranked = normalize_recommendations(
                response.get("recommendations"),
                catalog_ids,
            )
            if target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break
            if turn == 1:
                user_message = str(messages[1])
                if len(hard) > 1:
                    disclosed.add(hard[1])
            else:
                user_message, boundary_used = customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        sessions.append(
            {
                "sample_id": sample["sample_id"],
                "scenario_type": sample["scenario_type"],
                "hit": hit_turn is not None,
                "first_hit_turn": hit_turn,
                "best_rank": best_rank,
                "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
            }
        )

    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical, 6),
        "reported_token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "sessions": sessions,
    }


def _retained_agent_bytes(agent: ConversationalSearchAgent) -> int:
    planner = getattr(agent, "_orchestrator", None)
    retained = (
        getattr(agent, "_sessions", None),
        getattr(agent, "_slates", None),
        getattr(agent, "_profile_priors", None),
        getattr(planner, "_entries", None),
        getattr(agent, "_requirement_probe_counts", None),
    )
    return _deep_size(retained)


def _inspect_retained_probe_state(
    agent: ConversationalSearchAgent,
    policy: RequirementProbePolicy,
) -> bool:
    try:
        attributes = vars(agent)
        suspicious = {
            name
            for name in attributes
            if "probe" in name.casefold() and name != "_retriever"
        }
        expected = (
            set()
            if policy is DISABLED_REQUIREMENT_PROBE_POLICY
            else {"_requirement_probe_policy", "_requirement_probe_counts"}
        )
        if suspicious != expected:
            return False
        if policy is DISABLED_REQUIREMENT_PROBE_POLICY:
            _canonical_private_cache_snapshot(agent)
            return True
        counts = attributes.get("_requirement_probe_counts")
        if (
            type(counts) is not list
            or len(counts) != 12
            or any(type(value) is not int or value < 0 for value in counts)
        ):
            return False
        if not isinstance(
            attributes.get("_requirement_probe_policy"),
            RequirementProbePolicy,
        ):
            return False
        _canonical_private_cache_snapshot(agent)
    except Exception:
        return False
    return True


def _startup_probe(backend: object, *, iterations: int = 2_000) -> dict:
    if iterations < 100:
        raise ValueError("startup probe requires at least 100 paired constructions")
    baseline_total = 0
    candidate_total = 0
    baseline: ConversationalSearchAgent | None = None
    candidate: ConversationalSearchAgent | None = None
    for index in range(iterations):
        policies = (
            (
                DISABLED_REQUIREMENT_PROBE_POLICY,
                CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
            )
            if index % 2 == 0
            else (
                CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
                DISABLED_REQUIREMENT_PROBE_POLICY,
            )
        )
        for policy in policies:
            started = time.perf_counter_ns()
            agent = ConversationalSearchAgent(
                "unused.jsonl",
                retriever=backend,
                requirement_probe_policy=policy,
                slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
            )
            elapsed = time.perf_counter_ns() - started
            if policy is DISABLED_REQUIREMENT_PROBE_POLICY:
                baseline = agent
                baseline_total += elapsed
            else:
                candidate = agent
                candidate_total += elapsed
    if baseline is None or candidate is None:
        raise RuntimeError("startup probe did not construct both variants")
    baseline_bytes = _retained_agent_bytes(baseline)
    rss_before = _current_max_rss_bytes()
    rss_candidate = ConversationalSearchAgent(
        "unused.jsonl",
        retriever=backend,
        requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    rss_delta = max(0, _current_max_rss_bytes() - rss_before)
    candidate_bytes = _retained_agent_bytes(rss_candidate)
    return {
        "iterations": iterations,
        "baseline_total_ms": round(baseline_total / 1_000_000.0, 6),
        "candidate_total_ms": round(candidate_total / 1_000_000.0, 6),
        "candidate_startup_time_ratio": round(
            _safe_ratio(candidate_total, baseline_total),
            6,
        ),
        "candidate_additional_startup_rss_bytes": rss_delta,
        "baseline_empty_retained_bytes": baseline_bytes,
        "candidate_empty_retained_bytes": candidate_bytes,
    }


def _project_probe_health(
    health: object,
    *,
    policy: RequirementProbePolicy,
) -> dict[str, int | str]:
    if not isinstance(health, dict) or set(health) != set(PROBE_HEALTH_KEYS):
        raise RuntimeError("requirement-probe health schema drifted")
    if health.get("policy") != policy.value:
        raise RuntimeError("requirement-probe policy telemetry drifted")
    for key in PROBE_HEALTH_KEYS[1:]:
        value = health.get(key)
        if type(value) is not int or value < 0:
            raise RuntimeError("requirement-probe counter is invalid")
    attempts = int(health["attempts"])
    if sum(int(health[key]) for key in _PROBE_OUTCOME_KEYS) != attempts:
        raise RuntimeError("requirement-probe outcomes do not partition attempts")
    if int(health["selected_probe_queries"]) > attempts * MAX_REQUIREMENT_PROBES:
        raise RuntimeError("requirement-probe call bound was exceeded")
    if policy is DISABLED_REQUIREMENT_PROBE_POLICY and any(
        int(health[key]) for key in PROBE_HEALTH_KEYS[1:]
    ):
        raise RuntimeError("protected baseline performed requirement-probe work")
    return {key: health[key] for key in PROBE_HEALTH_KEYS}


def _run_variant(
    config: SuiteConfig,
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    policy: RequirementProbePolicy,
) -> VariantRun:
    guarded = ProbeCallAuditRetriever(backend, catalog_ids)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        requirement_probe_policy=policy,
    )
    audited = _AuditAgent(agent)
    started = time.perf_counter()
    if config.manual_messages:
        result = _evaluate_manual_messages(
            audited,
            samples,
            catalog_ids,
            categories,
            products,
        )
    else:
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
        "requirement_probe_health": _project_probe_health(
            agent.requirement_probe_health,
            policy=policy,
        ),
        "retained_profile_state_valid": _inspect_retained_profile_state(
            agent,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_probe_state_valid": _inspect_retained_probe_state(
            agent,
            policy,
        ),
        "retained_agent_bytes": _retained_agent_bytes(agent),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }
    _validate_variant_accounting(
        diagnostics,
        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    _validate_phase13_accounting(
        diagnostics,
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    probe = diagnostics["requirement_probe_health"]
    if policy is CATALOG_IDF_REQUIREMENT_PROBE_POLICY:
        if int(probe["attempts"]) != searches:
            raise RuntimeError("candidate probe coverage does not match searches")
    elif int(probe["attempts"]):
        raise RuntimeError("baseline performed candidate retrieval work")
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
    config: SuiteConfig,
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> VariantRun:
    runtime = ConversationalSearchAgent(
        catalog,
        requirement_probe_policy=CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable independently")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable independently")
    _ensure_probe_vocabulary(backend)
    return _run_variant(
        config,
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    )


def _warm_backend(catalog: Path, backend: object) -> None:
    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
    )
    session_id = "phase14-label-free-runtime-warmup"
    warmup.reset(session_id, {})
    warmup.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    if int(warmup.ranking_health["successes"]) != 1:
        raise RuntimeError("label-free backend warm-up did not complete")
    if int(warmup.requirement_probe_health["attempts"]):
        raise RuntimeError("label-free warm-up performed candidate work")


def _ensure_probe_vocabulary(backend: object) -> None:
    """Materialize the lazy catalog-only vocabulary before any suite attempt."""

    try:
        capable = (
            getattr(backend, "requirement_probe_capability")
            is REQUIREMENT_PROBE_CAPABILITY
        )
        initialize = getattr(backend, "_ensure_bm25_vocabulary")
    except Exception as error:
        raise RuntimeError("requirement-probe capability is unavailable") from error
    if not capable or not callable(initialize):
        raise RuntimeError("requirement-probe capability is unavailable")
    try:
        initialized = initialize()
    except Exception as error:
        raise RuntimeError("requirement-probe vocabulary initialization failed") from error
    if initialized is not True or not getattr(
        backend,
        "requirement_probe_vocabulary_available",
        False,
    ):
        raise RuntimeError("requirement-probe vocabulary is unavailable")


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
            "requirement_probe_health",
            "retained_profile_state_valid",
            "retained_probe_state_valid",
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
        "reuses": int(orchestration["reuses"]),  # type: ignore[index]
        "main_bm25_route_calls": sum(  # type: ignore[index,union-attr]
            int(value) for value in route["bm25"].values()
        ),
        "dense_route_calls": sum(  # type: ignore[index,union-attr]
            int(value) for value in route["dense"].values()
        ),
        "vocabulary_lookup_upper_bound": int(  # type: ignore[index]
            route["vocabulary_lookup_upper_bound"]
        ),
        "probe_bm25_queries": int(route["probe_bm25_queries"]),  # type: ignore[index]
        "candidate_document_calls": int(route["candidate_document_calls"]),  # type: ignore[index]
        "stage_a_attempts": int(ranking["attempts"]),  # type: ignore[index]
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
    probe = diagnostics["requirement_probe_health"]  # type: ignore[assignment]
    return (
        int(route["fallback_turns"]) == 0  # type: ignore[index]
        and all(
            int(route[key]) == 0  # type: ignore[index]
            for key in (
                "incumbent_loss_violations",
                "candidate_bound_violations",
                "invalid_or_duplicate_id_violations",
                "probe_overlap_violations",
            )
        )
        and int(ranking["failures"]) == 0  # type: ignore[index]
        and int(ranking["unavailable_skips"]) == 0  # type: ignore[index]
        and int(rescue["attempts"]) == 0  # type: ignore[index]
        and int(redundancy["validation_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(novelty["validation_fallbacks"]) == 0  # type: ignore[index]
        and int(profile["parsing_or_scoring_fallbacks"]) == 0  # type: ignore[index]
        and int(slate["failures"]) == 0  # type: ignore[index]
        and int(orchestration["fault_invalidations"]) == 0  # type: ignore[index]
        and int(orchestration["store_rejections"]) == 0  # type: ignore[index]
        and int(probe["unavailable"]) == 0  # type: ignore[index]
        and int(probe["errors"]) == 0  # type: ignore[index]
        and int(probe["validation_or_execution_fallbacks"]) == 0  # type: ignore[index]
        and diagnostics["retained_profile_state_valid"] is True
        and diagnostics["retained_probe_state_valid"] is True
    )


def _quality_gates(
    config: SuiteConfig,
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    baseline_metrics = baseline.summary
    candidate_metrics = candidate.summary
    transitions = paired["transitions"]  # type: ignore[assignment]
    bootstrap = paired["bootstrap"]  # type: ignore[assignment]
    lower = float(bootstrap["lower_95"])  # type: ignore[index]
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
        "paired_bootstrap_lower_95_passes_suite_threshold": (
            lower > 0.0 if config.name in {"development", "validation"} else lower >= 0.0
        ),
    }


def _baseline_matches_protected(config: SuiteConfig, baseline: VariantRun) -> bool:
    protected = PHASE13_PROTECTED_METRICS.get(config.name)
    if protected is None:
        return True
    observed = {
        key: baseline.summary[key]
        for key in protected
    }
    return observed == protected


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
    candidate_probe = candidate.diagnostics["requirement_probe_health"]
    replay_exact = (
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
    one_ordinary_call = all(
        calls["searches"]
        == calls["main_bm25_route_calls"]
        == calls["dense_route_calls"]
        == calls["candidate_document_calls"]
        == calls["stage_a_attempts"]
        for calls in (baseline_calls, candidate_calls)
    )
    candidate_route = candidate.diagnostics["route_health"]
    probe_partition = (
        sum(int(candidate_probe[key]) for key in _PROBE_OUTCOME_KEYS)
        == int(candidate_probe["attempts"])
        == candidate_calls["searches"]
    )
    gates = {
        **_quality_gates(config, baseline, candidate, paired),
        "candidate_replay_is_exact": replay_exact,
        "independent_construction_is_exact": independent_exact,
        "all_variants_are_fault_free": all(
            _faults_are_zero(run.diagnostics)
            for run in (baseline, candidate, replay, independent)
        ),
        "candidate_probe_outcomes_partition_searches": probe_partition,
        "at_most_two_probe_bm25_queries_per_search": (
            int(candidate_route["max_probe_queries_per_search"])
            <= MAX_REQUIREMENT_PROBES
            and candidate_calls["probe_bm25_queries"]
            <= MAX_REQUIREMENT_PROBES * candidate_calls["searches"]
        ),
        "at_most_one_vocabulary_lookup_per_search": (
            candidate_calls["vocabulary_lookup_upper_bound"]
            <= candidate_calls["searches"]
        ),
        "complete_union_bound_and_incumbent_preservation_hold": all(
            int(candidate_route[key]) == 0
            for key in (
                "incumbent_loss_violations",
                "candidate_bound_violations",
                "invalid_or_duplicate_id_violations",
                "probe_overlap_violations",
            )
        )
        and int(candidate_route["max_fused_union"]) <= MAX_CANDIDATE_DOCUMENTS,
        "exact_reuse_performs_no_route_or_probe_attempt": (
            int(candidate_probe["attempts"]) == candidate_calls["searches"]
            and candidate_calls["searches"] + candidate_calls["reuses"]
            == int(candidate.diagnostics["expected_turns"])
        ),
        "one_main_route_dense_document_and_stage_a_call_per_search": (
            one_ordinary_call
        ),
        "baseline_performed_zero_probe_work": (
            all(
                int(baseline.diagnostics["requirement_probe_health"][key]) == 0
                for key in PROBE_HEALTH_KEYS[1:]
            )
            and baseline_calls["probe_bm25_queries"] == 0
            and baseline_calls["vocabulary_lookup_upper_bound"] == 0
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
        "candidate_additional_retained_session_state_at_most_64kib": (
            int(performance["candidate_additional_retained_agent_bytes"]) <= 65_536
        ),
        "baseline_matches_protected_phase13_metrics": (
            _baseline_matches_protected(config, baseline)
        ),
        "fresh_suite_lock_and_manual_protocol_valid": True,
        "implementation_lock_revalidated_after_all_variants": lock_revalidated,
        "aggregate_publication_privacy_valid": privacy_valid,
    }
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
    runtime = lock.get("runtime_source_sha256")
    if not isinstance(runtime, dict):
        raise RuntimeError("protected Phase 13 runtime lock is malformed")
    for relative, expected in runtime.items():
        if relative in ALLOWED_RUNTIME_CHANGES:
            continue
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("runtime outside the Phase 14 scope drifted")
    starter = repository_root / "starter/agent.py"
    if _sha256(starter) != lock["active_agent"]["starter_sha256"]:
        raise RuntimeError("protected starter changed before promotion")
    return lock


def _validate_fresh_lock(repository_root: Path) -> dict:
    path = repository_root / FRESH_LOCK_RELATIVE
    lock = json.loads(path.read_text(encoding="utf-8"))
    if (
        lock.get("schema_version") != 1
        or lock.get("lock_id")
        != "phase14-explicit-card-target-disjoint-suite-v1"
        or lock.get("status")
        != "generated_before_phase14_candidate_implementation"
        or lock.get("selected_target_count") != 384
        or lock.get("prior_target_overlap") != 0
        or lock.get("output_sha256") != FRESH_SHA256
        or lock.get("row_fingerprint_set_sha256") != FRESH_SET_SHA256
    ):
        raise RuntimeError("fresh Phase 14 suite lock drifted")
    output = Path(str(lock.get("output_path", "")))
    if not output.is_file() or _sha256(output) != FRESH_SHA256:
        raise RuntimeError("fresh Phase 14 suite artifact drifted")
    if lock.get("contract_sha256") != _sha256(repository_root / CONTRACT_RELATIVE):
        raise RuntimeError("fresh suite contract hash drifted")
    if lock.get("baseline_lock_sha256") != _sha256(
        repository_root / BASELINE_LOCK_RELATIVE
    ):
        raise RuntimeError("fresh suite baseline hash drifted")
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
        "phase14_oracle_command",
        "phase14_oracle_cases",
        "phase14_random_oracle_cases",
        "phase14_oracle_sha256",
        "completed_before_lock",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("Phase 14 pre-lock verification schema drifted")
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
        "phase14_oracle_command": PHASE14_ORACLE_COMMAND,
        "phase14_oracle_cases": PHASE14_RANDOM_ORACLE_CASES,
        "phase14_random_oracle_cases": PHASE14_RANDOM_ORACLE_CASES,
        "phase14_oracle_sha256": PHASE14_ORACLE_SHA256,
        "completed_before_lock": True,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise RuntimeError("Phase 14 pre-lock verification evidence drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise RuntimeError("Phase 14 pre-lock test count is invalid")
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
        "fresh_suite_lock_sha256",
        "research_plan_sha256",
        "source_sha256",
        "verification",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 14 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 14 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 14 implementation lock identity")
    if lock.get("status") != "locked_before_phase14_data_suite_evaluation":
        raise RuntimeError("Phase 14 implementation is not frozen")
    documents = {
        "contract_sha256": CONTRACT_RELATIVE,
        "baseline_lock_sha256": BASELINE_LOCK_RELATIVE,
        "fresh_suite_lock_sha256": FRESH_LOCK_RELATIVE,
        "research_plan_sha256": RESEARCH_PLAN_RELATIVE,
    }
    for key, relative in documents.items():
        value = lock.get(key)
        if (
            not isinstance(value, str)
            or _HEX_SHA256_RE.fullmatch(value) is None
            or value != _sha256(repository_root / relative)
        ):
            raise RuntimeError("Phase 14 planning artifact drifted after lock")
    source_hashes = lock.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 14 source lock is incomplete")
    observed = {
        relative: _sha256(repository_root / relative)
        for relative in SOURCE_PATHS
    }
    if observed != source_hashes:
        raise RuntimeError("Phase 14 implementation drifted after lock")
    _validate_prelock_verification(lock.get("verification"))
    _validate_baseline_scope(repository_root)
    _validate_fresh_lock(repository_root)
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
            raise RuntimeError("prior suite result violates aggregate privacy")
        gates = payload.get("decision_gate")
        if not isinstance(gates, dict) or gates.get("advance") is not True:
            raise RuntimeError("prior suite did not pass every frozen gate")


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
    if _ASIN_RE.search(json.dumps(payload, sort_keys=True, allow_nan=False)):
        return False
    return payload.get("privacy") == {
        "aggregate_metrics_and_fixed_counters_only": True,
        "row_scenario_message_profile_target_product_and_trace_data_absent": True,
        "per_case_fingerprints_absent": True,
        "manual_failure_inspection_performed": False,
    }


def run_requirement_probe_ablation(
    suite: str,
    output_path: str | Path,
    *,
    thermal_safe_ack: bool,
) -> dict:
    if suite not in SUITES:
        raise ValueError("unsupported Phase 14 suite")
    if thermal_safe_ack is not True:
        raise RuntimeError("thermal safety must be checked before claiming a suite")
    config = SUITES[suite]
    output = Path(output_path).resolve()
    _validate_execution_environment()
    implementation_lock = _validate_implementation_lock()
    _validate_prerequisites(config)
    _validate_run_paths(config, output)

    if _sha256(REPOSITORY_ROOT / "data/catalog.jsonl") != CATALOG_SHA256:
        raise RuntimeError("catalog drifted before suite claim")
    if _sha256(config.source) != config.source_sha256:
        raise RuntimeError("suite source drifted before suite claim")
    catalog = REPOSITORY_ROOT / "data/catalog.jsonl"

    runtime = ConversationalSearchAgent(
        catalog,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable before suite claim")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable before suite claim")
    _ensure_probe_vocabulary(backend)

    _claim_attempt(config.attempt, config.name)
    samples, dataset_evidence = _load_suite_samples(config)
    catalog_ids, categories, products = catalog_index(catalog)
    _warm_backend(catalog, backend)
    startup = _startup_probe(backend)

    candidate = _run_variant(
        config,
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    )
    baseline = _run_variant(
        config,
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        DISABLED_REQUIREMENT_PROBE_POLICY,
    )
    replay = _run_variant(
        config,
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        CATALOG_IDF_REQUIREMENT_PROBE_POLICY,
    )
    independent = _run_independent(
        config,
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
            "manual_first_two_messages": config.manual_messages,
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
            "phase14_oracle_sha256": PHASE14_ORACLE_SHA256,
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
        raise RuntimeError("Phase 14 aggregate publication violates privacy")
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
        description="Run one frozen Phase 14 aggregate-only suite"
    )
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thermal-safe-ack", action="store_true")
    arguments = parser.parse_args()
    run_requirement_probe_ablation(
        arguments.suite,
        arguments.output,
        thermal_safe_ack=arguments.thermal_safe_ack,
    )


if __name__ == "__main__":
    main()
