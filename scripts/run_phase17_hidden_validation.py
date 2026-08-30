"""Run the one-shot aggregate-only Phase 17 hidden-generalization validation."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import hmac
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

import evaluator.local_evaluator as evaluator_module
from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    EvidenceExposurePolicy,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.build_phase17_clean_room_suite import (
    CASE_COUNT,
    SCENARIO_COUNTS,
    SURFACE_SCENARIO_COUNTS,
    _forbidden_targets,
    _key,
    _set_commitment,
    _sha256,
    _target,
)
from scripts.run_multislot_intent_ablations import (
    _evaluate_with_deterministic_session_ids,
)
from scripts.run_phase2_exact_evidence_ablations import (
    AggregateAuditAgent,
    RuntimeNetworkAudit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_RELATIVE = "docs/phase17_hidden_generalization_contract.json"
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase17_implementation_lock.json"
TEMPLATE_RELATIVE = "docs/phase17_language_templates.json"
SUITE_LOCK_RELATIVE = "docs/phase17_suite_lock.json"
EXPERIMENT_ID = "phase17-clean-room-hidden-generalization-v1"
LOCK_ID = "phase17-clean-room-implementation-v1"
SUITE_LOCK_ID = "phase17-clean-room-hidden-generalization-suite-v1"
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 170_260_831
SCENARIO_ORDER = ("buying", "browsing", "intent_override", "boundary")
OVERALL_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
SOURCE_PATHS = (
    "starter/agent.py",
    "starter/dense.py",
    "conversational_search/__init__.py",
    "conversational_search/decision.py",
    "conversational_search/decision_policy.py",
    "conversational_search/exact_evidence.py",
    "conversational_search/exposure.py",
    "conversational_search/exposure_policy.py",
    "conversational_search/intent.py",
    "conversational_search/orchestration.py",
    "conversational_search/profiles.py",
    "conversational_search/protocol.py",
    "conversational_search/questions.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/slates.py",
    "conversational_search/strategy.py",
    "conversational_search/utility_planner.py",
    "preprocessing/__init__.py",
    "preprocessing/catalog.py",
    "preprocessing/embeddings.py",
    "preprocessing/encoder.py",
    "evaluator/local_evaluator.py",
    "docs/agent_api_contract.json",
    CONTRACT_RELATIVE,
    TEMPLATE_RELATIVE,
    "scripts/build_phase17_clean_room_suite.py",
    "scripts/run_phase17_hidden_validation.py",
    "tests/test_phase17_hidden_validation.py",
    "requirements-runtime.txt",
    "assets/bge-small-en-v1.5-int8/model_manifest.json",
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json",
)
ASIN_RE = re.compile(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])")
FORBIDDEN_REPORT_KEYS = (
    '"sessions"',
    '"sample_id"',
    '"ground_truth"',
    '"parent_asin"',
    '"user_message"',
    '"user_profile"',
    '"recommendations"',
    '"actions"',
    '"rows"',
)


@dataclass(frozen=True, slots=True)
class Arm:
    arm_id: str
    ranking_policy: RankingPolicy
    exposure_policy: EvidenceExposurePolicy


@dataclass(slots=True)
class Run:
    summary: dict[str, object]
    scenario_metrics: dict[str, object]
    surface_metrics: dict[str, object]
    private_outcomes: list[dict]
    action_signature: tuple[tuple[int, int, str | None, int], ...]
    behavior_digest: str
    health: dict[str, object]


ARMS = (
    Arm(
        "active-phase16c-buying-prefix-v3",
        LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    ),
    Arm(
        "exact-evidence-without-exposure",
        LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        DISABLED_EVIDENCE_EXPOSURE_POLICY,
    ),
    Arm(
        "protected-phase13",
        STAGE_A_RANKING_POLICY,
        DISABLED_EVIDENCE_EXPOSURE_POLICY,
    ),
    Arm(
        "active-phase16c-buying-prefix-v3-replay",
        LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    ),
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} is not a JSON object")
    return value


def _validate_implementation_lock() -> dict[str, object]:
    contract_path = REPOSITORY_ROOT / CONTRACT_RELATIVE
    contract = _load_json(contract_path)
    if (
        contract.get("experiment_id") != EXPERIMENT_ID
        or contract.get("status")
        != "frozen_after_pre_execution_activation_coverage_review_before_generator_lock_or_evaluation"
    ):
        raise RuntimeError("Phase 17 contract identity or status drifted")
    lock = _load_json(REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE)
    if (
        lock.get("schema_version") != 1
        or lock.get("lock_id") != LOCK_ID
        or lock.get("status") != "locked_before_suite_generation_or_evaluation"
        or lock.get("contract_sha256") != _sha256(contract_path)
        or lock.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES
        or lock.get("bootstrap_seed") != BOOTSTRAP_SEED
    ):
        raise RuntimeError("Phase 17 implementation lock identity drifted")
    hashes = lock.get("source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 17 source lock is incomplete")
    for relative in SOURCE_PATHS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file() or _sha256(path) != hashes[relative]:
            raise RuntimeError(f"Phase 17 locked source drifted: {relative}")
    candidate = contract.get("candidate")
    if (
        not isinstance(candidate, dict)
        or candidate.get("starter_sha256") != hashes["starter/agent.py"]
    ):
        raise RuntimeError("Phase 17 candidate hash disagrees with its contract")
    return lock


def _suite_dialog(sample: Mapping[str, object]) -> dict[str, object]:
    value = sample.get("phase17_dialog")
    if not isinstance(value, dict):
        raise RuntimeError("Phase 17 row has no dialog object")
    mode = value.get("mode")
    if mode == "official_exact" and set(value) == {"mode"}:
        return value
    initial = value.get("initial_message")
    replies = value.get("reply_templates")
    expected = {
        "need_attribute",
        "disclosure",
        "no_additional",
        "boundary_indifference",
    }
    if (
        mode != "clean_room_language_shift"
        or not isinstance(initial, str)
        or not initial
        or not isinstance(replies, dict)
        or set(replies) != expected
        or any(not isinstance(item, str) or not item for item in replies.values())
    ):
        raise RuntimeError("Phase 17 dialog schema drifted")
    return value


@contextlib.contextmanager
def _language_surface() -> Iterator[None]:
    original_initial = evaluator_module.initial_message
    original_reply = evaluator_module.customer_reply

    def initial(sample: dict, category: str, disclosed: set[str]) -> str:
        dialog = _suite_dialog(sample)
        if dialog["mode"] == "official_exact":
            return original_initial(sample, category, disclosed)
        hard = sample["intent_card"].get("hard_constraints") or []
        if sample.get("scenario_type") == "buying" and hard:
            disclosed.add(str(hard[0]))
        return str(dialog["initial_message"])

    def reply(
        sample: dict,
        ask_attribute: object,
        disclosed: set[str],
        boundary_used: bool,
    ) -> tuple[str, bool]:
        dialog = _suite_dialog(sample)
        if dialog["mode"] == "official_exact":
            return original_reply(sample, ask_attribute, disclosed, boundary_used)
        before = set(disclosed)
        _, next_boundary = original_reply(
            sample,
            ask_attribute,
            disclosed,
            boundary_used,
        )
        constraints = [
            *[str(value) for value in sample["intent_card"].get("hard_constraints", [])],
            *[str(value) for value in sample["intent_card"].get("soft_preferences", [])],
        ]
        new_values = [
            value for value in constraints if value not in before and value in disclosed
        ]
        if next_boundary and not boundary_used:
            event = "boundary_indifference"
        elif not isinstance(ask_attribute, str):
            event = "need_attribute"
        elif new_values:
            event = "disclosure"
        else:
            event = "no_additional"
        attribute = ask_attribute if isinstance(ask_attribute, str) else "other"
        template = dialog["reply_templates"][event]  # type: ignore[index]
        try:
            rendered = str(template).format(
                attribute=attribute,
                values="; ".join(new_values),
            )
        except (IndexError, KeyError, ValueError) as error:
            raise RuntimeError("Phase 17 reply template failed") from error
        if not rendered:
            raise RuntimeError("Phase 17 reply rendered empty")
        return rendered, next_boundary

    with patch.object(evaluator_module, "initial_message", initial), patch.object(
        evaluator_module,
        "customer_reply",
        reply,
    ):
        yield


def _validate_suite(
    suite_path: Path,
    manifest_path: Path,
    key_path: Path,
    exclusion_paths: Sequence[Path],
) -> tuple[list[dict], dict[str, object]]:
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("lock_id") != SUITE_LOCK_ID
        or manifest.get("status") != "generated_before_candidate_evaluation"
        or manifest.get("case_count") != CASE_COUNT
        or manifest.get("unique_target_count") != CASE_COUNT
    ):
        raise RuntimeError("Phase 17 suite lock identity drifted")
    suite = manifest.get("suite")
    if (
        not isinstance(suite, dict)
        or suite.get("sha256") != _sha256(suite_path)
        or suite.get("path") != str(suite_path)
    ):
        raise RuntimeError("Phase 17 suite payload drifted")
    key = _key(key_path)
    if manifest.get("key_commitment_sha256") != hashlib.sha256(key).hexdigest():
        raise RuntimeError("Phase 17 selection key drifted")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("Phase 17 suite input lock is absent")
    if inputs.get("catalog_sha256") != _sha256(REPOSITORY_ROOT / "data/catalog.jsonl"):
        raise RuntimeError("Phase 17 catalog input drifted")
    source_hashes = inputs.get("exclusion_source_sha256")
    expected_source_hashes = {path.name: _sha256(path) for path in exclusion_paths}
    if source_hashes != expected_source_hashes:
        raise RuntimeError("Phase 17 exclusion sources drifted")
    rows = load_jsonl(suite_path)
    if len(rows) != CASE_COUNT:
        raise RuntimeError("Phase 17 suite cardinality drifted")
    targets = [_target(row) for row in rows]
    if any(value is None for value in targets):
        raise RuntimeError("Phase 17 suite target is missing")
    selected = {str(value) for value in targets}
    forbidden, _ = _forbidden_targets(exclusion_paths)
    scenarios = Counter(str(row.get("scenario_type")) for row in rows)
    surfaces = Counter(str(row.get("phase17_surface")) for row in rows)
    expected_surfaces = {
        surface: sum(counts.values())
        for surface, counts in SURFACE_SCENARIO_COUNTS.items()
    }
    surface_scenarios = Counter(
        (str(row.get("phase17_surface")), str(row.get("scenario_type")))
        for row in rows
    )
    if (
        len(selected) != CASE_COUNT
        or selected & forbidden
        or scenarios != Counter(SCENARIO_COUNTS)
        or surfaces != Counter(expected_surfaces)
        or any(
            surface_scenarios[(surface, scenario)] != count
            for surface, counts in SURFACE_SCENARIO_COUNTS.items()
            for scenario, count in counts.items()
        )
        or manifest.get("selected_target_set_hmac_sha256")
        != _set_commitment(key, "selected", selected)
        or manifest.get("forbidden_target_set_hmac_sha256")
        != _set_commitment(key, "forbidden", forbidden)
    ):
        raise RuntimeError("Phase 17 target or scenario integrity failed")
    for row in rows:
        _suite_dialog(row)
    integrity = {
        "suite_sha256_matches": True,
        "case_count": len(rows),
        "unique_target_count": len(selected),
        "historical_target_overlap": 0,
        "scenario_counts": dict(sorted(scenarios.items())),
        "surface_counts": dict(sorted(surfaces.items())),
        "surface_scenario_counts": {
            surface: {
                scenario: surface_scenarios[(surface, scenario)]
                for scenario in SCENARIO_ORDER
            }
            for surface in SURFACE_SCENARIO_COUNTS
        },
        "keyed_commitments_match": True,
        "dialog_schema_valid": True,
        "all_checks_pass": True,
    }
    return rows, integrity


def _new_agent(catalog: Path, backend: object, arm: Arm) -> ConversationalSearchAgent:
    return ConversationalSearchAgent(
        catalog,
        retriever=backend,
        ranking_policy=arm.ranking_policy,
        evidence_exposure_policy=arm.exposure_policy,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )


def _summary(result: Mapping[str, object]) -> dict[str, object]:
    usage = result.get("reported_token_usage")
    if not isinstance(usage, dict):
        raise RuntimeError("evaluator token usage is unavailable")
    return {
        **{key: result[key] for key in OVERALL_KEYS},
        "reported_token_usage": {
            key: usage[key]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }


def _summary_from_outcomes(outcomes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    projected = [dict(value) for value in outcomes]
    summary = evaluator_module.metric_summary(projected)
    if summary["mttc"] is None:
        raise RuntimeError("subset metric is unavailable")
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


def _surface_metrics(
    rows: Sequence[Mapping[str, object]],
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(rows) != len(outcomes):
        raise RuntimeError("surface metric pairing drifted")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row, outcome in zip(rows, outcomes):
        if row.get("sample_id") != outcome.get("sample_id"):
            raise RuntimeError("surface metric order drifted")
        surface = str(row.get("phase17_surface"))
        grouped[surface].append(outcome)
    if set(grouped) != set(SURFACE_SCENARIO_COUNTS):
        raise RuntimeError("surface metric groups drifted")
    return {
        surface: _summary_from_outcomes(grouped[surface])
        for surface in SURFACE_SCENARIO_COUNTS
    }


def _action_summary(
    actions: Sequence[tuple[int, int, str | None, int]],
) -> dict[str, object]:
    return {
        "turn_count": len(actions),
        "question_counts": dict(
            sorted(Counter(question or "none" for _, _, question, _ in actions).items())
        ),
        "width_counts": {
            str(width): count
            for width, count in sorted(Counter(width for _, _, _, width in actions).items())
        },
    }


def _technical_fault_free(agent: ConversationalSearchAgent, audited: AggregateAuditAgent, network: RuntimeNetworkAudit) -> bool:
    ranking = agent.ranking_health
    exact = agent.exact_evidence_health
    exposure = agent.evidence_exposure_health
    slate = agent.slate_health
    orchestration = agent.orchestration_health
    return bool(
        agent.dense_initialization_error is None
        and int(ranking["failures"]) == 0
        and int(ranking["unavailable_skips"]) == 0
        and int(exact["capability_unavailable"]) == 0
        and int(exact["evidence_errors"]) == 0
        and int(exact["validation_errors"]) == 0
        and int(exposure["validation_fallbacks"]) == 0
        and int(slate["failures"]) == 0
        and int(orchestration["fault_invalidations"]) == 0
        and int(orchestration["store_rejections"]) == 0
        and audited.response_exceptions == 0
        and audited.invalid_api_responses == 0
        and network.attempts == 0
    )


def _run_arm(
    catalog: Path,
    rows: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    arm: Arm,
) -> Run:
    agent = _new_agent(catalog, backend, arm)
    audited = AggregateAuditAgent(agent, catalog_ids)
    network = RuntimeNetworkAudit()
    started = time.perf_counter()
    with _language_surface(), network.deny():
        result = _evaluate_with_deterministic_session_ids(
            audited,
            rows,
            catalog_ids,
            categories,
            products,
        )
    wall_seconds = time.perf_counter() - started
    private_outcomes = result.get("sessions")
    scenario_metrics = result.get("scenario_metrics")
    if not isinstance(private_outcomes, list) or not isinstance(scenario_metrics, dict):
        raise RuntimeError("evaluator aggregate or private paired outcomes are unavailable")
    summary = _summary(result)
    tokens = summary["reported_token_usage"]
    if not isinstance(tokens, dict):
        raise RuntimeError("token summary is malformed")
    fault_free = _technical_fault_free(agent, audited, network)
    health = {
        "technical_fault_free": fault_free,
        "ranking": agent.ranking_health,
        "exact_evidence": agent.exact_evidence_health,
        "evidence_exposure": agent.evidence_exposure_health,
        "slate": agent.slate_health,
        "orchestration": agent.orchestration_health,
        "response_exceptions": audited.response_exceptions,
        "invalid_api_responses": audited.invalid_api_responses,
        "runtime_network_attempts": network.attempts,
        "reported_tokens_zero": all(int(value) == 0 for value in tokens.values()),
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "action_summary": _action_summary(audited.actions),
    }
    return Run(
        summary=summary,
        scenario_metrics=scenario_metrics,
        surface_metrics=_surface_metrics(rows, private_outcomes),
        private_outcomes=private_outcomes,
        action_signature=audited.actions,
        behavior_digest=audited.behavior_digest,
        health=health,
    )


def _turn(session: Mapping[str, object]) -> float:
    value = session.get("first_hit_turn")
    return float(value) if isinstance(value, int) else 11.0


def _utility(session: Mapping[str, object]) -> float:
    hit = 1.0 if session.get("hit") is True else 0.0
    reciprocal = float(session.get("reciprocal_rank") or 0.0)
    efficiency = max(0.0, min(1.0, (11.0 - _turn(session)) / 10.0))
    return 0.50 * hit + 0.30 * reciprocal + 0.20 * efficiency


def _paired_deltas(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
) -> tuple[dict[str, np.ndarray], Counter[str]]:
    if len(reference) != len(candidate) or len(reference) != CASE_COUNT:
        raise RuntimeError("paired outcome counts drifted")
    by_scenario: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    for left, right in zip(reference, candidate):
        if (
            left.get("sample_id") != right.get("sample_id")
            or left.get("scenario_type") != right.get("scenario_type")
        ):
            raise RuntimeError("paired outcome order drifted")
        left_hit = left.get("hit") is True
        right_hit = right.get("hit") is True
        transitions[
            "both_hit"
            if left_hit and right_hit
            else "candidate_only_hit"
            if right_hit
            else "reference_only_hit"
            if left_hit
            else "both_miss"
        ] += 1
        scenario = str(left["scenario_type"])
        by_scenario[scenario].append(
            (
                _utility(right) - _utility(left),
                float(right_hit) - float(left_hit),
                float(right.get("reciprocal_rank") or 0.0)
                - float(left.get("reciprocal_rank") or 0.0),
                _turn(right) - _turn(left),
            )
        )
    if set(by_scenario) != set(SCENARIO_ORDER):
        raise RuntimeError("paired scenario strata drifted")
    return {
        scenario: np.asarray(by_scenario[scenario], dtype=np.float64)
        for scenario in SCENARIO_ORDER
    }, transitions


def _bootstrap(
    strata: Mapping[str, np.ndarray],
) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.empty((BOOTSTRAP_REPLICATES, 4), dtype=np.float64)
    batch_size = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(BOOTSTRAP_REPLICATES, start + batch_size)
        aggregate = np.zeros((stop - start, 4), dtype=np.float64)
        for values in strata.values():
            indices = rng.integers(
                0,
                len(values),
                size=(stop - start, len(values)),
            )
            aggregate += values[indices].sum(axis=1)
        replicates[start:stop] = aggregate / CASE_COUNT
    replicates.sort(axis=0)
    lower = replicates[math.floor(0.05 * BOOTSTRAP_REPLICATES)]
    upper = replicates[math.floor(0.95 * BOOTSTRAP_REPLICATES)]
    means = sum(values.sum(axis=0) for values in strata.values()) / CASE_COUNT
    names = ("technical_score", "hit_rate_at_10", "mrr", "mttc")
    return {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "strata": len(strata),
        "mean_delta": {name: round(float(means[index]), 9) for index, name in enumerate(names)},
        "lower_95_one_sided": {name: round(float(lower[index]), 9) for index, name in enumerate(names)},
        "upper_95_one_sided": {name: round(float(upper[index]), 9) for index, name in enumerate(names)},
    }


def _scenario_hr_deltas(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for scenario in SCENARIO_ORDER:
        left = reference.get(scenario)
        right = candidate.get(scenario)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError("scenario metric is unavailable")
        result[scenario] = round(
            float(right["hit_rate_at_10"]) - float(left["hit_rate_at_10"]),
            6,
        )
    return result


def _surface_hr_deltas(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for surface in SURFACE_SCENARIO_COUNTS:
        left = reference.get(surface)
        right = candidate.get(surface)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise RuntimeError("surface metric is unavailable")
        result[surface] = round(
            float(right["hit_rate_at_10"]) - float(left["hit_rate_at_10"]),
            6,
        )
    return result


def _comparison(reference: Run, candidate: Run, reference_id: str) -> dict[str, object]:
    strata, transitions = _paired_deltas(reference.private_outcomes, candidate.private_outcomes)
    bootstrap = _bootstrap(strata)
    scenario_deltas = _scenario_hr_deltas(reference.scenario_metrics, candidate.scenario_metrics)
    surface_deltas = _surface_hr_deltas(reference.surface_metrics, candidate.surface_metrics)
    lower = bootstrap["lower_95_one_sided"]
    upper = bootstrap["upper_95_one_sided"]
    if not isinstance(lower, dict) or not isinstance(upper, dict):
        raise RuntimeError("bootstrap output is malformed")
    scenario_limits = {
        "buying": -0.05,
        "browsing": -0.05,
        "intent_override": -0.05,
        "boundary": -0.10,
    }
    gates = {
        "technical_score_lower_95_at_least_zero": float(lower["technical_score"]) >= 0.0,
        "hit_rate_lower_95_at_least_minus_0_02": float(lower["hit_rate_at_10"]) >= -0.02,
        "mrr_lower_95_at_least_minus_0_01": float(lower["mrr"]) >= -0.01,
        "mttc_upper_95_at_most_plus_0_25": float(upper["mttc"]) <= 0.25,
        "no_scenario_hit_rate_collapse": all(
            scenario_deltas[name] >= limit for name, limit in scenario_limits.items()
        ),
        "no_surface_hit_rate_collapse": all(
            value >= -0.05 for value in surface_deltas.values()
        ),
    }
    gates["all_quality_gates_pass"] = all(gates.values())
    return {
        "reference": reference_id,
        "candidate": ARMS[0].arm_id,
        "reference_metrics": reference.summary,
        "candidate_metrics": candidate.summary,
        "scenario_hit_rate_point_deltas": scenario_deltas,
        "surface_hit_rate_point_deltas": surface_deltas,
        "paired_transitions": {
            key: transitions[key]
            for key in (
                "both_hit",
                "candidate_only_hit",
                "reference_only_hit",
                "both_miss",
            )
        },
        "paired_bootstrap": bootstrap,
        "quality_gates": gates,
    }


def _replay_exact(first: Run, second: Run) -> dict[str, bool]:
    return {
        "metrics_equal": first.summary == second.summary,
        "scenario_metrics_equal": first.scenario_metrics == second.scenario_metrics,
        "surface_metrics_equal": first.surface_metrics == second.surface_metrics,
        "actions_equal": first.action_signature == second.action_signature,
        "responses_equal": first.behavior_digest == second.behavior_digest,
        "health_equal_except_timing": {
            key: value for key, value in first.health.items() if key != "evaluation_wall_seconds"
        }
        == {
            key: value for key, value in second.health.items() if key != "evaluation_wall_seconds"
        },
    }


def _public_arm(run: Run) -> dict[str, object]:
    return {
        "metrics": run.summary,
        "scenario_metrics": run.scenario_metrics,
        "surface_metrics": run.surface_metrics,
        "health": run.health,
    }


def _claim(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "status": "claimed",
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")


def _publish(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    compact = json.dumps(payload, sort_keys=True, allow_nan=False)
    if ASIN_RE.search(compact) or any(key in compact for key in FORBIDDEN_REPORT_KEYS):
        raise RuntimeError("Phase 17 report contains private row-level data")
    temporary = path.with_name(f".{path.name}.complete")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def run(
    *,
    catalog: Path,
    suite_path: Path,
    suite_lock_path: Path,
    key_path: Path,
    exclusion_paths: Sequence[Path],
) -> dict[str, object]:
    implementation_lock = _validate_implementation_lock()
    rows, integrity = _validate_suite(
        suite_path,
        suite_lock_path,
        key_path,
        exclusion_paths,
    )
    catalog_ids, categories, products = catalog_index(catalog)
    bootstrap_agent = ConversationalSearchAgent(
        catalog,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    backend = bootstrap_agent.retrieval_backend
    if not getattr(backend, "bm25_available", False) or not getattr(
        backend,
        "dense_available",
        False,
    ):
        raise RuntimeError("Phase 17 requires healthy BM25 and dense routes")
    del bootstrap_agent
    runs: dict[str, Run] = {}
    for arm in ARMS:
        runs[arm.arm_id] = _run_arm(
            catalog,
            rows,
            catalog_ids,
            categories,
            products,
            backend,
            arm,
        )
        gc.collect()
    candidate = runs[ARMS[0].arm_id]
    replay = runs[ARMS[3].arm_id]
    comparisons = {
        arm.arm_id: _comparison(runs[arm.arm_id], candidate, arm.arm_id)
        for arm in ARMS[1:3]
    }
    replay_checks = _replay_exact(candidate, replay)
    execution_gates = {
        "integrity": integrity["all_checks_pass"] is True,
        "all_arms_technical_fault_free": all(
            run.health["technical_fault_free"] is True for run in runs.values()
        ),
        "all_arms_zero_token": all(
            run.health["reported_tokens_zero"] is True for run in runs.values()
        ),
        "candidate_replay_exact": all(replay_checks.values()),
        "both_reference_comparisons_pass": all(
            comparison["quality_gates"]["all_quality_gates_pass"] is True
            for comparison in comparisons.values()
        ),
    }
    execution_gates["all_gates_pass"] = all(execution_gates.values())
    return {
        "schema_version": 1,
        "report_id": "phase17-clean-room-hidden-generalization-20260831",
        "experiment_id": EXPERIMENT_ID,
        "status": "passed" if execution_gates["all_gates_pass"] else "failed",
        "interpretation": (
            "fresh target- and language-shift validation passed; this is strong finite-sample evidence, not a mathematical guarantee"
            if execution_gates["all_gates_pass"]
            else "the frozen active candidate did not satisfy every predeclared fresh-suite gate; no tuning or rerun is authorized"
        ),
        "integrity": integrity,
        "arms": {
            arm.arm_id: _public_arm(runs[arm.arm_id])
            for arm in ARMS[:3]
        },
        "comparisons": comparisons,
        "replay": replay_checks,
        "decision_gate": execution_gates,
        "execution": {
            "arm_order": [arm.arm_id for arm in ARMS],
            "single_process_single_thread_cpu": True,
            "shared_immutable_backend": True,
            "fresh_agent_state_per_arm": True,
            "network_denied": True,
            "external_model_or_api_calls": 0,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "public_200_scored": False,
            "shadow_source_scored": False,
        },
        "privacy": {
            "aggregate_only": True,
            "contains_rows_messages_profiles_targets_products_queries_ranks_or_action_traces": False,
            "failed_case_inspection": False,
        },
        "reproducibility": {
            "contract_sha256": _sha256(REPOSITORY_ROOT / CONTRACT_RELATIVE),
            "implementation_lock_sha256": _sha256(
                REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE
            ),
            "suite_lock_sha256": _sha256(suite_lock_path),
            "suite_sha256": _sha256(suite_path),
            "source_sha256": implementation_lock["source_sha256"],
        },
        "next_action": (
            "retain the frozen active candidate without further public tuning"
            if execution_gates["all_gates_pass"]
            else "do not inspect failures or tune on this suite; assess any replacement only on a newly committed disjoint suite"
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=REPOSITORY_ROOT / "data/catalog.jsonl")
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--suite-lock", type=Path, default=REPOSITORY_ROOT / SUITE_LOCK_RELATIVE)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=REPOSITORY_ROOT / "data/public_set.jsonl")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--phase14", type=Path, required=True)
    parser.add_argument("--phase16-activation", type=Path, default=REPOSITORY_ROOT / "benchmarks/phase16b_semantic_rescue_activation.jsonl")
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    _claim(output)
    payload = run(
        catalog=args.catalog.resolve(),
        suite_path=args.suite.resolve(),
        suite_lock_path=args.suite_lock.resolve(),
        key_path=args.key_file.resolve(),
        exclusion_paths=(
            args.public.resolve(),
            args.development.resolve(),
            args.validation.resolve(),
            args.phase14.resolve(),
            args.phase16_activation.resolve(),
            args.shadow.resolve(),
        ),
    )
    _publish(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
