"""Sealed Phase 9 bounded-profile candidate versus Phase 7 comparator.

The evaluator payload and response/state traces are held only in memory for
paired exactness checks.  The returned publication is aggregate-only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Sequence

from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    PROFILE_THEME_MASK_BYTES,
    ProductTheme,
    ProfilePolicy,
    ProfilePrior,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever, _sha256
from scripts.run_orchestration_ablations import _lookup_accounting_exact
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_reranking_ablations import (
    RespondLatencyAgent,
    _expected_turns,
    _metric_deltas,
)
from scripts.verify_phase7_stage_a_oracle import (
    EXPECTED_SHA256 as PHASE7_ORACLE_SHA256,
    ORACLE_CASES as PHASE7_ORACLE_CASES,
)
from starter.agent import Agent


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_SCHEMA_VERSION = 1
EXPERIMENT_ID = "phase9-bounded-profile-aware-stage-a-v1"
CANDIDATE_ID = "phase9-bounded-profile-residual-v1"
BASELINE_ID = "phase7-exact-ranking-reuse-v1"
IMPLEMENTATION_LOCK_ID = (
    "phase9-bounded-profile-aware-stage-a-implementation-v1"
)
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase9_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase9_experiment_contract.json"
RAW_RESULT_RELATIVE = "results-phase9-profile-aware-ranking.json"
FULL_SUITE_COMMAND = ".venv/bin/python -m unittest discover -s tests -q"
FOCUSED_SUITE_COMMAND = (
    ".venv/bin/python -m unittest tests.test_profiles "
    "tests.test_profile_ranking tests.test_ranking tests.test_orchestration "
    "tests.test_service tests.test_service_profiles "
    "tests.test_profile_ablations tests.test_phase7_stage_a_oracle -q"
)
PHASE7_DIFFERENTIAL_COMMAND = (
    ".venv/bin/python -m scripts.verify_phase7_stage_a_oracle"
)
_UNITTEST_COUNT_RE = re.compile(r"\bRan\s+(\d+)\s+tests?\b")

# The lock deliberately covers the complete production path, its experiment
# harness, and all profile/cache integration tests.  It excludes itself.
SOURCE_PATHS = (
    ".gitignore",
    "conversational_search/intent.py",
    "conversational_search/orchestration.py",
    "conversational_search/profiles.py",
    "conversational_search/questions.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/slates.py",
    "conversational_search/strategy.py",
    CONTRACT_RELATIVE,
    "evaluator/local_evaluator.py",
    "preprocessing/encoder.py",
    "requirements-runtime.txt",
    "scripts/run_fusion_ablations.py",
    "scripts/run_orchestration_ablations.py",
    "scripts/run_policy_ablations.py",
    "scripts/run_profile_ablations.py",
    "scripts/run_reranking_ablations.py",
    "scripts/verify_phase7_stage_a_oracle.py",
    "starter/agent.py",
    "starter/dense.py",
    "tests/test_orchestration.py",
    "tests/test_phase7_stage_a_oracle.py",
    "tests/test_profile_ablations.py",
    "tests/test_profile_ranking.py",
    "tests/test_profiles.py",
    "tests/test_ranking.py",
    "tests/test_service.py",
    "tests/test_service_profiles.py",
)

FROZEN_INPUT_SHA256 = {
    "catalog": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    "dataset": "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    "assets/bge-small-en-v1.5-int8/model_manifest.json": (
        "f1130079f60555f7e35dc84344a33cd8e9afdcb4743c42afc94fb42b3991fd76"
    ),
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json": (
        "c9b7291004d6ef78473b24886899ea51f427fc2e179c8216c8e8b65f6cf929b2"
    ),
    "benchmarks/phase7.json": (
        "bc82249a24eff47720ba2d18785f40e370602f908e258bbde3a722e2dc814080"
    ),
    "docs/phase7_implementation_lock.json": (
        "bde98911a18932331c6e0702ef53a43b9575f56ece3dd8e0953ef8e3ae7444d6"
    ),
    "docs/phase7_results.json": (
        "598bbf53c26a77c4b8020c08c834e43c142725337e6a07443551c2b6fa3c1c09"
    ),
    "evaluator/local_evaluator.py": (
        "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"
    ),
    "requirements-runtime.txt": (
        "db8bcb738aa8a27746b78473e7ba0806b9ddb03011917a82285ee7ca0e9523b5"
    ),
}

PHASE7_OFFICIAL = {
    "sample_count": 200,
    "hit_rate_at_10": 0.99,
    "mrr": 0.52223,
    "mttc": 3.07,
    "recommended_technical_score": 0.810269,
}
OVERALL_METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
TOKEN_USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)

PROFILE_HEALTH_KEYS = (
    "policy",
    "session_entries",
    "logical_profile_bytes",
    "profiles_reset",
    "zero_mask_profiles",
    "nonzero_mask_profiles",
    "recognized_theme_count",
    "turns_disabled_by_active_requirements",
    "eligible_stage_a_attempts",
    "empty_represented_theme_fallbacks",
    "constant_score_neutral_fallbacks",
    "successful_residual_applications",
    "parsing_or_scoring_fallbacks",
)
_PROFILE_COUNTER_KEYS = PROFILE_HEALTH_KEYS[1:]
_PROFILE_STATE_ATTRIBUTES = frozenset(
    {
        "_profile_policy",
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
    }
)
_PROFILE_STATE_NAME_TOKENS = frozenset(
    {
        "profile",
        "profiles",
        "theme",
        "themes",
        "preference",
        "preferences",
        "rating",
        "ratings",
        "purchase",
        "purchases",
        "tag",
        "tags",
        "represented",
        "residual",
    }
)
_RAW_PROFILE_FIELD_KEYS = frozenset(
    {
        "user_profile",
        "preference_tags",
        "summary",
        "rating_style",
        "average_prior_rating",
        "purchase_frequency",
    }
)
_ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")


class _CallAuditRetriever(RouteHealthRetriever):
    """Audit existing route and candidate-document calls without their data."""

    def __init__(self, backend: object) -> None:
        super().__init__(backend)
        self._candidate_document_calls = 0

    def candidate_documents(self, parent_asins: Sequence[str]) -> tuple:
        self._candidate_document_calls += 1
        return super().candidate_documents(parent_asins)

    def summary(self) -> dict:
        return {
            **super().summary(),
            "candidate_document_calls": self._candidate_document_calls,
        }


class _AuditAgent(RespondLatencyAgent):
    """Keep private exactness traces in memory; never return or serialize them."""

    def __init__(self, delegate: ConversationalSearchAgent) -> None:
        super().__init__(delegate)
        self._search_delegate = delegate
        self.action_trace: list[tuple[object, object, object]] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = super().respond(session_id, user_message, turn, top_k)
        self.action_trace.append(
            (
                _freeze(response),
                self._search_delegate.session_state(session_id),
                self._search_delegate.slate_state(session_id),
            )
        )
        return response


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _route_call_count(summary: dict) -> int:
    return sum(int(value) for value in (summary.get("bm25") or {}).values())


def _project_profile_health(
    health: object,
    *,
    expected_policy: ProfilePolicy,
    expected_sessions: int,
) -> dict[str, int | str]:
    """Validate and project the frozen aggregate-only profile telemetry."""

    if not isinstance(health, dict) or set(health) != set(PROFILE_HEALTH_KEYS):
        raise RuntimeError("profile health schema drifted")
    if health.get("policy") != expected_policy.value:
        raise RuntimeError("profile policy telemetry does not match the variant")
    for key in _PROFILE_COUNTER_KEYS:
        value = health.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid aggregate profile counter: {key}")

    entries = int(health["session_entries"])
    resets = int(health["profiles_reset"])
    zero = int(health["zero_mask_profiles"])
    nonzero = int(health["nonzero_mask_profiles"])
    recognized = int(health["recognized_theme_count"])
    eligible = int(health["eligible_stage_a_attempts"])
    nonfault_outcomes = sum(
        int(health[key])
        for key in (
            "empty_represented_theme_fallbacks",
            "constant_score_neutral_fallbacks",
            "successful_residual_applications",
        )
    )
    fallbacks = int(health["parsing_or_scoring_fallbacks"])
    if entries != expected_sessions or resets != expected_sessions:
        raise RuntimeError("profile session/reset accounting is incomplete")
    if zero + nonzero != resets:
        raise RuntimeError("profile reset classes do not partition resets")
    if not nonzero <= recognized <= 10 * nonzero:
        raise RuntimeError("recognized theme accounting exceeds the ten-bit mask")
    if int(health["logical_profile_bytes"]) != entries * PROFILE_THEME_MASK_BYTES:
        raise RuntimeError("logical retained profile memory is not two bytes per session")
    if not nonfault_outcomes <= eligible <= nonfault_outcomes + fallbacks:
        raise RuntimeError("profile outcomes are inconsistent with eligible attempts")
    if expected_policy is DISABLED_PROFILE_POLICY and any(
        int(health[key])
        for key in (
            "turns_disabled_by_active_requirements",
            "eligible_stage_a_attempts",
            "empty_represented_theme_fallbacks",
            "constant_score_neutral_fallbacks",
            "successful_residual_applications",
        )
    ):
        raise RuntimeError("disabled profile comparator performed profile work")
    return {key: health[key] for key in PROFILE_HEALTH_KEYS}


def _contains_raw_profile_mapping(value: object, *, depth: int = 0) -> bool:
    if depth > 3:
        return False
    if isinstance(value, dict):
        if any(str(key) in _RAW_PROFILE_FIELD_KEYS for key in value):
            return True
        return any(
            _contains_raw_profile_mapping(item, depth=depth + 1)
            for item in value.values()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(
            _contains_raw_profile_mapping(item, depth=depth + 1)
            for item in value
        )
    return False


def _inspect_retained_profile_state(
    agent: ConversationalSearchAgent,
    *,
    expected_sessions: int,
) -> bool:
    """Inspect private state once, returning only one aggregate-safe boolean."""

    try:
        attributes = vars(agent)
        if not isinstance(attributes, dict):
            return False
        suspicious_names = {
            name
            for name in attributes
            if set(name.strip("_").casefold().split("_"))
            & _PROFILE_STATE_NAME_TOKENS
        }
        expected_suspicious_names = {
            name
            for name in _PROFILE_STATE_ATTRIBUTES
            if set(name.strip("_").casefold().split("_"))
            & _PROFILE_STATE_NAME_TOKENS
        }
        if (
            not _PROFILE_STATE_ATTRIBUTES.issubset(attributes)
            or suspicious_names != expected_suspicious_names
        ):
            return False
        if attributes.get("_profile_policy") not in {
            BOUNDED_RESIDUAL_PROFILE_POLICY,
            DISABLED_PROFILE_POLICY,
        }:
            return False
        store = attributes.get("_profile_priors")
        if type(store) is not dict or len(store) != expected_sessions:
            return False
        for key, prior in store.items():
            if type(key) is not bytes or len(key) != 32:
                return False
            if type(prior) is not ProfilePrior or hasattr(prior, "__dict__"):
                return False
            if tuple(getattr(type(prior), "__slots__", ())) != ("theme_mask",):
                return False
            mask = prior.theme_mask
            if type(mask) is not ProductTheme or int(mask) & ~((1 << 10) - 1):
                return False
        for name, value in attributes.items():
            if name == "_retriever":
                continue
            if _contains_raw_profile_mapping(value):
                return False
        for name in _PROFILE_STATE_ATTRIBUTES - {"_profile_policy", "_profile_priors"}:
            value = attributes.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return False
    except Exception:
        return False
    return True


def _validate_variant_accounting(
    diagnostics: dict,
    *,
    expected_policy: ProfilePolicy,
) -> None:
    turns = int(diagnostics["expected_turns"])
    route = diagnostics["route_health"]
    ranking = diagnostics["ranking_health"]
    slate = diagnostics["slate_health"]
    orchestration = diagnostics["orchestration_health"]
    profile = diagnostics["profile_health"]
    searches = int(orchestration["searches"])

    if int(orchestration["decisions"]) != turns:
        raise RuntimeError("orchestration decision coverage is incomplete")
    if searches + int(orchestration["reuses"]) != turns:
        raise RuntimeError("ordinary evaluator turns must search or reuse")
    if int(orchestration["skips"]) != 0:
        raise RuntimeError("the official top-k workload must not skip")
    if not _lookup_accounting_exact(orchestration):
        raise RuntimeError("cache lookup accounting is inconsistent")
    if _route_call_count(route) != searches:
        raise RuntimeError("route-call accounting is incomplete")
    if int(ranking["attempts"]) != searches:
        raise RuntimeError("reranker-call accounting is incomplete")
    if int(route["candidate_document_calls"]) != int(ranking["attempts"]):
        raise RuntimeError("candidate-document call accounting is incomplete")
    if int(ranking["successes"]) != int(ranking["attempts"]):
        raise RuntimeError("every Stage-A call must succeed")
    if int(ranking["failures"]) or int(ranking["unavailable_skips"]):
        raise RuntimeError("reranker faults invalidate Phase 9 confirmation")
    if int(slate["attempts"]) != turns or int(slate["successes"]) != turns:
        raise RuntimeError("slate coverage is incomplete")
    if int(slate["failures"]):
        raise RuntimeError("slate faults invalidate Phase 9 confirmation")
    if int(route["fallback_turns"]):
        raise RuntimeError("fallback turns invalidate Phase 9 confirmation")
    if int(profile["eligible_stage_a_attempts"]) > int(ranking["attempts"]):
        raise RuntimeError("profile attempts exceed ordinary Stage-A attempts")
    if profile["policy"] != expected_policy.value:
        raise RuntimeError("profile policy accounting drifted")


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    profile_policy: ProfilePolicy,
) -> tuple[dict, dict, list[tuple[object, object, object]]]:
    guarded_backend = _CallAuditRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded_backend,
        profile_policy=profile_policy,
    )
    audited = _AuditAgent(agent)
    started = time.perf_counter()
    result = evaluate(audited, samples, catalog_ids, categories, products)
    wall_seconds = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    searches = int(agent.orchestration_health["searches"])
    guarded_backend.validate(searches)
    latency = audited.latency_summary()
    if int(latency["count"]) != expected_turns:
        raise RuntimeError("response timing coverage is incomplete")
    retained_profile_state_valid = _inspect_retained_profile_state(
        agent,
        expected_sessions=int(result["sample_count"]),
    )

    diagnostics = {
        "expected_turns": expected_turns,
        "route_health": guarded_backend.summary(),
        "ranking_health": agent.ranking_health,
        "slate_health": agent.slate_health,
        "orchestration_health": agent.orchestration_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=profile_policy,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_profile_state_valid": retained_profile_state_valid,
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }
    _validate_variant_accounting(
        diagnostics,
        expected_policy=profile_policy,
    )
    return result, diagnostics, audited.action_trace


def _core_health(diagnostics: dict) -> dict:
    return {
        key: diagnostics[key]
        for key in (
            "expected_turns",
            "ranking_health",
            "slate_health",
            "orchestration_health",
            "profile_health",
            "retained_profile_state_valid",
        )
    }


def _deterministic_variant_health(diagnostics: dict) -> dict:
    return {
        **_core_health(diagnostics),
        "route_health": diagnostics["route_health"],
    }


def _run_independent(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[dict, dict, list[tuple[object, object, object]]]:
    agent = Agent(catalog)
    if not getattr(agent.retrieval_backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable for independent verification")
    if not getattr(agent.retrieval_backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable for independent verification")
    audited = _AuditAgent(agent)
    result = evaluate(audited, samples, catalog_ids, categories, products)
    expected_turns = _expected_turns(result)
    if int(audited.latency_summary()["count"]) != expected_turns:
        raise RuntimeError("independent response timing coverage is incomplete")
    retained_profile_state_valid = _inspect_retained_profile_state(
        agent,
        expected_sessions=int(result["sample_count"]),
    )
    diagnostics = {
        "expected_turns": expected_turns,
        "ranking_health": agent.ranking_health,
        "slate_health": agent.slate_health,
        "orchestration_health": agent.orchestration_health,
        "profile_health": _project_profile_health(
            agent.profile_health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=int(result["sample_count"]),
        ),
        "retained_profile_state_valid": retained_profile_state_valid,
    }
    return result, diagnostics, audited.action_trace


def _warm_backend(catalog: Path, backend: object) -> None:
    """Initialize runtime kernels using one fixed, unlabeled synthetic turn."""

    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=DISABLED_PROFILE_POLICY,
    )
    session_id = "phase9-label-free-runtime-warmup"
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
        raise RuntimeError("frozen Phase 9 inputs drifted; refusing public run")
    return observed


def _run_prelock_command(
    repository_root: Path,
    command: str,
) -> subprocess.CompletedProcess[str]:
    parts = command.split()
    if not parts or parts[0] != ".venv/bin/python":
        raise ValueError("pre-lock command must use the frozen repository Python")
    completed = subprocess.run(
        [str(repository_root / parts[0]), *parts[1:]],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"pre-lock verification command failed with exit {completed.returncode}: "
            f"{command}"
        )
    return completed


def _run_prelock_unittest_suite(repository_root: Path, command: str) -> int:
    completed = _run_prelock_command(repository_root, command)
    matches = _UNITTEST_COUNT_RE.findall(
        f"{completed.stdout}\n{completed.stderr}"
    )
    if len(matches) != 1:
        raise RuntimeError("could not parse one exact unittest count")
    count = int(matches[0])
    if count <= 0:
        raise RuntimeError("pre-lock unittest suite executed zero tests")
    return count


def _run_prelock_differential(repository_root: Path) -> tuple[int, str]:
    completed = _run_prelock_command(
        repository_root,
        PHASE7_DIFFERENTIAL_COMMAND,
    )
    try:
        evidence = json.loads(completed.stdout.strip())
    except (json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("Phase 7 differential evidence is not valid JSON") from error
    if not isinstance(evidence, dict) or set(evidence) != {
        "cases",
        "digest",
        "status",
    }:
        raise RuntimeError("Phase 7 differential evidence schema drifted")
    if (
        evidence.get("status") != "ok"
        or evidence.get("cases") != PHASE7_ORACLE_CASES
        or evidence.get("digest") != PHASE7_ORACLE_SHA256
    ):
        raise RuntimeError("Phase 7 differential oracle did not verify")
    return PHASE7_ORACLE_CASES, PHASE7_ORACLE_SHA256


def _collect_prelock_verification(repository_root: Path) -> dict:
    """Execute every frozen pre-lock check and retain aggregate evidence only."""

    differential_cases, differential_sha256 = _run_prelock_differential(
        repository_root
    )
    focused_tests = _run_prelock_unittest_suite(
        repository_root,
        FOCUSED_SUITE_COMMAND,
    )
    complete_tests = _run_prelock_unittest_suite(
        repository_root,
        FULL_SUITE_COMMAND,
    )
    return {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "focused_tests_passed": focused_tests,
        "complete_suite_command": FULL_SUITE_COMMAND,
        "complete_unit_tests_passed": complete_tests,
        "phase7_exact_differential_command": PHASE7_DIFFERENTIAL_COMMAND,
        "phase7_exact_differential_cases": differential_cases,
        "phase7_exact_oracle_sha256": differential_sha256,
        "completed_before_lock": True,
    }


def _validate_prelock_verification(verification: object) -> dict:
    expected_keys = {
        "focused_suite_command",
        "focused_tests_passed",
        "complete_suite_command",
        "complete_unit_tests_passed",
        "phase7_exact_differential_command",
        "phase7_exact_differential_cases",
        "phase7_exact_oracle_sha256",
        "completed_before_lock",
    }
    if not isinstance(verification, dict) or set(verification) != expected_keys:
        raise ValueError("pre-lock verification schema is incomplete")
    if (
        verification.get("focused_suite_command") != FOCUSED_SUITE_COMMAND
        or verification.get("complete_suite_command") != FULL_SUITE_COMMAND
        or verification.get("phase7_exact_differential_command")
        != PHASE7_DIFFERENTIAL_COMMAND
        or verification.get("phase7_exact_differential_cases")
        != PHASE7_ORACLE_CASES
        or verification.get("phase7_exact_oracle_sha256")
        != PHASE7_ORACLE_SHA256
        or verification.get("completed_before_lock") is not True
    ):
        raise ValueError("pre-lock verification evidence drifted")
    for key in ("focused_tests_passed", "complete_unit_tests_passed"):
        value = verification.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("pre-lock test counts must be positive integers")
    return {key: verification[key] for key in expected_keys}


def _build_implementation_lock(
    repository_root: Path,
    *,
    verification: dict,
    source_paths: Sequence[str] = SOURCE_PATHS,
) -> dict:
    """Build deterministic pre-public lock content; perform no evaluation."""

    validated_verification = _validate_prelock_verification(verification)
    paths = tuple(source_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("implementation lock source paths must be unique")
    source_sha256 = {
        relative: _sha256(repository_root / relative)
        for relative in sorted(paths)
    }
    return {
        "schema_version": IMPLEMENTATION_LOCK_SCHEMA_VERSION,
        "lock_id": IMPLEMENTATION_LOCK_ID,
        "status": "locked_before_public_confirmation",
        "contract_sha256": _sha256(repository_root / CONTRACT_RELATIVE),
        "source_sha256": source_sha256,
        "verification": validated_verification,
    }


def _write_implementation_lock(lock_path: Path, lock: dict) -> None:
    """Atomically create a lock and refuse to replace an existing lock."""

    lock_path = lock_path.resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(lock, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=lock_path.parent,
            prefix=f".{lock_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, lock_path)
        except FileExistsError as error:
            raise FileExistsError(
                "Phase 9 implementation lock already exists; refusing overwrite"
            ) from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _claim_run_output(output: Path) -> None:
    """Exclusively record that the sole public run started; never self-remove."""

    sentinel = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "run_started_no_retry",
    }
    serialized = (json.dumps(sentinel, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "Phase 9 run was already claimed; refusing a second run"
        ) from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


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
        raise RuntimeError("Phase 9 implementation lock schema drifted")
    if lock.get("schema_version") != IMPLEMENTATION_LOCK_SCHEMA_VERSION:
        raise RuntimeError("unsupported Phase 9 implementation lock")
    if lock.get("lock_id") != IMPLEMENTATION_LOCK_ID:
        raise RuntimeError("unexpected Phase 9 implementation lock identity")
    if lock.get("status") != "locked_before_public_confirmation":
        raise RuntimeError("Phase 9 implementation is not frozen")
    if lock.get("contract_sha256") != _sha256(repository_root / CONTRACT_RELATIVE):
        raise RuntimeError("Phase 9 experiment contract drifted after lock")

    paths = tuple(source_paths)
    expected_source = lock.get("source_sha256")
    if not isinstance(expected_source, dict) or set(expected_source) != set(paths):
        raise RuntimeError("Phase 9 implementation source lock is incomplete")
    observed_source = {
        relative: _sha256(repository_root / relative)
        for relative in sorted(paths)
    }
    if observed_source != expected_source:
        raise RuntimeError("Phase 9 implementation drifted after lock")

    try:
        _validate_prelock_verification(lock.get("verification"))
    except ValueError as error:
        raise RuntimeError("Phase 9 pre-lock verification drifted") from error
    return lock


def _phase7_summary_matches(result: dict) -> bool:
    summary = _overall_official_summary(result)
    return {
        key: summary.get(key)
        for key in PHASE7_OFFICIAL
    } == PHASE7_OFFICIAL


def _overall_official_summary(result: dict) -> dict:
    """Project only overall evaluator metrics; exclude every scenario cell."""

    usage = result.get("reported_token_usage") or {}
    return {
        **{key: result[key] for key in OVERALL_METRIC_KEYS},
        "reported_token_usage": {
            key: usage[key]
            for key in TOKEN_USAGE_KEYS
            if key in usage
        },
    }


def _paired_hit_to_miss_count(baseline: dict, candidate: dict) -> int:
    """Compute one aggregate count and retain no paired row projection."""

    baseline_sessions = baseline.get("sessions")
    candidate_sessions = candidate.get("sessions")
    if not isinstance(baseline_sessions, list) or not isinstance(
        candidate_sessions, list
    ):
        raise RuntimeError("paired evaluator sessions are unavailable")
    if len(baseline_sessions) != len(candidate_sessions):
        raise RuntimeError("paired evaluator session counts differ")
    hit_to_miss = 0
    for baseline_session, candidate_session in zip(
        baseline_sessions,
        candidate_sessions,
    ):
        if not isinstance(baseline_session, dict) or not isinstance(
            candidate_session, dict
        ):
            raise RuntimeError("paired evaluator session is malformed")
        if (
            baseline_session.get("sample_id")
            != candidate_session.get("sample_id")
            or baseline_session.get("scenario_type")
            != candidate_session.get("scenario_type")
        ):
            raise RuntimeError("paired evaluator session order drifted")
        hit_to_miss += int(
            baseline_session.get("hit") is True
            and candidate_session.get("hit") is not True
        )
    return hit_to_miss


def _faults_are_zero(diagnostics: dict) -> bool:
    return (
        int(diagnostics["route_health"]["fallback_turns"]) == 0
        and int(diagnostics["ranking_health"]["failures"]) == 0
        and int(diagnostics["ranking_health"]["unavailable_skips"]) == 0
        and int(diagnostics["slate_health"]["failures"]) == 0
        and int(diagnostics["orchestration_health"]["fault_invalidations"]) == 0
        and int(diagnostics["orchestration_health"]["store_rejections"]) == 0
        and int(diagnostics["profile_health"]["parsing_or_scoring_fallbacks"])
        == 0
    )


def _ten_bit_profile_contract_holds() -> bool:
    supported = sum(int(theme) for theme in ProductTheme)
    return supported == (1 << 10) - 1 and PROFILE_THEME_MASK_BYTES == 2


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.inf


def _latency_comparison(
    baseline_diagnostics: dict,
    candidate_diagnostics: dict,
) -> dict[str, float]:
    baseline_wall = float(baseline_diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate_diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(
        baseline_diagnostics["respond_latency_ms"]["warm_p95"]
    )
    candidate_p95 = float(
        candidate_diagnostics["respond_latency_ms"]["warm_p95"]
    )
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
    }


def _build_decision_gates(
    *,
    baseline: dict,
    candidate: dict,
    replay: dict,
    independent: dict,
    baseline_diagnostics: dict,
    candidate_diagnostics: dict,
    replay_diagnostics: dict,
    independent_diagnostics: dict,
    candidate_trace: list[tuple[object, object, object]],
    replay_trace: list[tuple[object, object, object]],
    independent_trace: list[tuple[object, object, object]],
    hit_to_miss_count: int,
    implementation_lock: dict,
    publication_privacy_valid: bool,
) -> tuple[dict[str, bool], dict[str, float]]:
    latency = _latency_comparison(baseline_diagnostics, candidate_diagnostics)
    wall_ratio = _safe_ratio(
        float(candidate_diagnostics["evaluation_wall_seconds"]),
        float(baseline_diagnostics["evaluation_wall_seconds"]),
    )
    p95_ratio = _safe_ratio(
        float(candidate_diagnostics["respond_latency_ms"]["warm_p95"]),
        float(baseline_diagnostics["respond_latency_ms"]["warm_p95"]),
    )
    candidate_profile = candidate_diagnostics["profile_health"]
    baseline_calls = baseline_diagnostics["route_health"]
    candidate_calls = candidate_diagnostics["route_health"]
    candidate_metrics = _overall_official_summary(candidate)
    baseline_metrics = _overall_official_summary(baseline)
    verification = implementation_lock["verification"]

    replay_exact = (
        replay == candidate
        and replay_trace == candidate_trace
        and _deterministic_variant_health(replay_diagnostics)
        == _deterministic_variant_health(candidate_diagnostics)
    )
    independent_exact = (
        independent == candidate
        and independent_trace == candidate_trace
        and independent_diagnostics == _core_health(candidate_diagnostics)
    )
    stage_call_accounting = (
        int(candidate_calls["candidate_document_calls"])
        == int(candidate_diagnostics["ranking_health"]["attempts"])
        == int(candidate_diagnostics["orchestration_health"]["searches"])
        and int(baseline_calls["candidate_document_calls"])
        == int(baseline_diagnostics["ranking_health"]["attempts"])
        == int(baseline_diagnostics["orchestration_health"]["searches"])
    )
    candidate_has_no_additional_calls = (
        stage_call_accounting
        and _route_call_count(candidate_calls)
        <= _route_call_count(baseline_calls)
        and int(candidate_calls["candidate_document_calls"])
        <= int(baseline_calls["candidate_document_calls"])
    )
    profile_memory_holds = (
        int(candidate_profile["logical_profile_bytes"])
        == int(candidate_profile["session_entries"])
        * PROFILE_THEME_MASK_BYTES
        and PROFILE_THEME_MASK_BYTES <= 2
    )
    retained_profile_state_valid = all(
        diagnostics.get("retained_profile_state_valid") is True
        for diagnostics in (
            baseline_diagnostics,
            candidate_diagnostics,
            replay_diagnostics,
            independent_diagnostics,
        )
    )

    gates = {
        "phase7_comparator_metrics_exact": _phase7_summary_matches(baseline),
        "candidate_hit_rate_at_10_at_least_0_99": (
            float(candidate_metrics["hit_rate_at_10"]) >= 0.99
        ),
        "candidate_mrr_at_least_0_52223": (
            float(candidate_metrics["mrr"]) >= 0.52223
        ),
        "candidate_mttc_at_most_3_07": (
            float(candidate_metrics["mttc"]) <= 3.07
        ),
        "candidate_technical_score_strictly_above_0_810269": (
            float(candidate_metrics["recommended_technical_score"]) > 0.810269
        ),
        "candidate_mrr_or_mttc_strictly_improves": (
            float(candidate_metrics["mrr"]) > float(baseline_metrics["mrr"])
            or float(candidate_metrics["mttc"])
            < float(baseline_metrics["mttc"])
        ),
        "phase7_hit_to_phase9_miss_count_is_zero": hit_to_miss_count == 0,
        "candidate_replay_payload_trace_and_telemetry_exact": replay_exact,
        "independent_starter_payload_trace_and_telemetry_exact": independent_exact,
        "synthetic_and_complete_unit_suites_passed_before_lock": (
            verification["completed_before_lock"] is True
            and int(verification["focused_tests_passed"]) > 0
            and int(verification["complete_unit_tests_passed"]) > 0
            and verification["phase7_exact_differential_cases"]
            == PHASE7_ORACLE_CASES
            and verification["phase7_exact_oracle_sha256"]
            == PHASE7_ORACLE_SHA256
        ),
        "baseline_and_candidate_faults_are_zero": (
            _faults_are_zero(baseline_diagnostics)
            and _faults_are_zero(candidate_diagnostics)
        ),
        "no_additional_model_api_embedding_route_rerank_or_document_calls": (
            candidate_has_no_additional_calls
        ),
        "candidate_warm_p95_ratio_at_most_1_05": p95_ratio <= 1.05,
        "candidate_wall_time_ratio_at_most_1_05": wall_ratio <= 1.05,
        "logical_profile_payload_at_most_two_bytes_per_session": (
            profile_memory_holds
        ),
        "retained_profile_state_is_only_valid_ten_bit_mask": (
            _ten_bit_profile_contract_holds() and retained_profile_state_valid
        ),
        "raw_profile_and_per_session_telemetry_absent": (
            retained_profile_state_valid
            and set(candidate_profile) == set(PROFILE_HEALTH_KEYS)
            and int(candidate_profile["logical_profile_bytes"])
            == int(candidate_profile["session_entries"])
            * PROFILE_THEME_MASK_BYTES
        ),
        "aggregate_publication_privacy_valid": publication_privacy_valid,
        "source_and_contract_lock_validated_before_public_run": True,
        "post_run_tuning_or_second_candidate_run_absent": True,
    }
    gates["adopt"] = all(gates.values())
    return gates, latency


def run_profile_ablations(
    catalog_path: str | Path,
    dataset_path: str | Path,
) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    repository_root = Path(__file__).resolve().parents[1]

    # These checks intentionally precede dataset loading and runtime creation.
    frozen_inputs = _validate_frozen_inputs(repository_root, catalog, dataset)
    implementation_lock = _validate_implementation_lock(repository_root)
    output = (repository_root / RAW_RESULT_RELATIVE).resolve()
    _validate_output(output, catalog, dataset)
    _claim_run_output(output)
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
        profile_policy=DISABLED_PROFILE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing Phase 9 run")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing Phase 9 run")
    _warm_backend(catalog, backend)

    candidate, candidate_diagnostics, candidate_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    baseline, baseline_diagnostics, _baseline_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        DISABLED_PROFILE_POLICY,
    )
    if not _phase7_summary_matches(baseline):
        raise RuntimeError("Phase 7 comparator drifted; refusing Phase 9 comparison")
    replay, replay_diagnostics, replay_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        BOUNDED_RESIDUAL_PROFILE_POLICY,
    )
    independent, independent_diagnostics, independent_trace = _run_independent(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
    )

    hit_to_miss_count = _paired_hit_to_miss_count(baseline, candidate)
    latency = _latency_comparison(
        baseline_diagnostics,
        candidate_diagnostics,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "candidate": CANDIDATE_ID,
        "baseline": BASELINE_ID,
        "run_configuration": {
            "execution": "strictly_sequential",
            "onnx_threads": 1,
            "shared_immutable_backend": True,
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
            "phase7_hit_to_phase9_miss_count": hit_to_miss_count,
        },
        "health": {
            "baseline": baseline_diagnostics,
            "candidate": candidate_diagnostics,
            "candidate_replay": replay_diagnostics,
            "independent_core": independent_diagnostics,
        },
        "latency": latency,
        "exactness": {
            "candidate_replay_evaluator_payload_equal": replay == candidate,
            "candidate_replay_action_intent_slate_trace_equal": (
                replay_trace == candidate_trace
            ),
            "candidate_replay_aggregate_telemetry_equal": (
                _deterministic_variant_health(replay_diagnostics)
                == _deterministic_variant_health(candidate_diagnostics)
            ),
            "independent_evaluator_payload_equal": independent == candidate,
            "independent_action_intent_slate_trace_equal": (
                independent_trace == candidate_trace
            ),
            "independent_aggregate_telemetry_equal": (
                independent_diagnostics == _core_health(candidate_diagnostics)
            ),
        },
        "privacy": {
            "contains_queries_or_messages": False,
            "contains_profiles_or_tags": False,
            "contains_product_or_sample_ids": False,
            "contains_session_or_turn_rows": False,
            "contains_raw_masks_candidate_scores_or_action_traces": False,
            "profile_telemetry_is_fixed_global_counters_only": True,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "frozen_input_sha256": frozen_inputs,
            "implementation_lock_id": implementation_lock["lock_id"],
            "contract_sha256": implementation_lock["contract_sha256"],
            "source_sha256": implementation_lock["source_sha256"],
            "pre_lock_verification": implementation_lock["verification"],
        },
    }
    publication_privacy_valid = _publication_privacy_is_valid(
        payload,
        allow_missing_decision_gate=True,
    )
    gates, _validated_latency = _build_decision_gates(
        baseline=baseline,
        candidate=candidate,
        replay=replay,
        independent=independent,
        baseline_diagnostics=baseline_diagnostics,
        candidate_diagnostics=candidate_diagnostics,
        replay_diagnostics=replay_diagnostics,
        independent_diagnostics=independent_diagnostics,
        candidate_trace=candidate_trace,
        replay_trace=replay_trace,
        independent_trace=independent_trace,
        hit_to_miss_count=hit_to_miss_count,
        implementation_lock=implementation_lock,
        publication_privacy_valid=publication_privacy_valid,
    )
    if _validated_latency != latency:
        raise RuntimeError("latency projection drifted while building gates")
    payload["decision_gate"] = gates
    _validate_publication_privacy(payload)
    _write_json_atomic(output, payload)
    return payload


def _publication_privacy_is_valid(
    payload: object,
    *,
    allow_missing_decision_gate: bool = False,
) -> bool:
    """Validate the exact aggregate publication allowlist without raising."""

    if not isinstance(payload, dict):
        return False
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "candidate",
        "baseline",
        "run_configuration",
        "official_metrics",
        "paired_quality",
        "health",
        "latency",
        "exactness",
        "privacy",
        "reproducibility",
        "decision_gate",
    }
    if allow_missing_decision_gate:
        expected_top_level.remove("decision_gate")
    if set(payload) != expected_top_level:
        return False

    official = payload.get("official_metrics")
    if not isinstance(official, dict) or set(official) != {
        "baseline",
        "candidate",
        "delta",
    }:
        return False
    expected_summary_keys = {*OVERALL_METRIC_KEYS, "reported_token_usage"}
    for name in ("baseline", "candidate"):
        summary = official.get(name)
        if not isinstance(summary, dict) or set(summary) != expected_summary_keys:
            return False
        usage = summary.get("reported_token_usage")
        if not isinstance(usage, dict) or set(usage) != set(TOKEN_USAGE_KEYS):
            return False
    delta = official.get("delta")
    if not isinstance(delta, dict) or set(delta) != set(OVERALL_METRIC_KEYS[1:]):
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
        "theme",
        "themes",
        "theme_counts",
        "theme_frequencies",
        "category",
        "categories",
        "category_metrics",
        "tag",
        "tags",
        "tag_counts",
        "tag_frequencies",
        "candidate_documents",
        "candidate_scores",
        "candidate_ids",
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
    scenario_labels = {"boundary", "browsing", "buying", "intent_override"}
    if not scenario_labels.isdisjoint(keys | strings):
        return False
    serialized = json.dumps(payload, sort_keys=True)
    if _ASIN_RE.search(serialized):
        return False
    return True


def _validate_publication_privacy(payload: dict) -> None:
    """Reject any publication outside the aggregate-only allowlist."""

    if not _publication_privacy_is_valid(payload):
        raise RuntimeError("Phase 9 publication violates the aggregate-only allowlist")


def _validate_output(output: Path, catalog: Path, dataset: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    resolved = output.resolve()
    if resolved != (repository_root / RAW_RESULT_RELATIVE).resolve():
        raise ValueError("Phase 9 has one frozen raw-result output path")
    if resolved.exists():
        raise FileExistsError("Phase 9 raw result already exists; refusing a second run")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the Phase 9 source lock or run its sealed A/B confirmation"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--create-implementation-lock", action="store_true")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]

    if args.create_implementation_lock:
        if args.output is not None:
            parser.error("--output cannot be used while creating the implementation lock")
        if (repository_root / RAW_RESULT_RELATIVE).exists():
            parser.error("cannot create the implementation lock after a Phase 9 run")
        if (repository_root / IMPLEMENTATION_LOCK_RELATIVE).exists():
            parser.error("Phase 9 implementation lock already exists")
        verification = _collect_prelock_verification(repository_root)
        lock = _build_implementation_lock(
            repository_root,
            verification=verification,
        )
        _write_implementation_lock(
            repository_root / IMPLEMENTATION_LOCK_RELATIVE,
            lock,
        )
        return

    if args.output is None:
        parser.error("--output is required for the sealed A/B confirmation")
    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output(output, catalog, dataset)
    run_profile_ablations(catalog, dataset)


if __name__ == "__main__":
    main()
