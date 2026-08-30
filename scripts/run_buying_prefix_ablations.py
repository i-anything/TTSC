"""Frozen exploratory comparison of withholding versus a ranked top-three prefix.

The public suite informed this follow-up, so the runner may diagnose behavior
but cannot promote a policy or mutate the active starter. Published output is
aggregate-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import time
from collections.abc import Mapping
from pathlib import Path

from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
)
from conversational_search.orchestration import (
    EXACT_RANKING_CACHE_CAPABILITY,
)
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    PROTOCOL_EVIDENCE_CAPABILITY,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import catalog_index, load_jsonl
from scripts.run_fusion_ablations import _sha256
from scripts.run_multislot_intent_ablations import (
    _metric_deltas,
    _paired_statistics,
)
from scripts.run_phase2_exact_evidence_ablations import (
    VariantRun,
    _common_action_equivalence,
    _performance,
    _scenario_report,
    select_smoke_indices,
)
from scripts.run_semantic_exposure_ablations import (
    _claim_output,
    _replace_claim,
)
from scripts.run_semantic_exposure_v2_ablations import (
    REQUIRED_ENVIRONMENT,
    SOURCE_PATHS as PHASE16B_SOURCE_PATHS,
    ArmConfig,
    _deterministic_health,
    _fixed_architecture_contract as _phase16b_architecture_contract,
    _run_variant,
    _technical_faults_are_zero,
    _tokens_are_zero,
    _validate_environment,
    _variant_public,
)


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "phase16c-buying-prefix-exposure-ablation-v1"
REPORT_ID = "phase16c-buying-prefix-public-20260830"
CONTRACT_RELATIVE = "docs/phase16c_buying_prefix_contract.json"
IMPLEMENTATION_LOCK_RELATIVE = (
    "docs/phase16c_buying_prefix_implementation_lock.json"
)
IMPLEMENTATION_LOCK_ID = "phase16c-buying-prefix-implementation-v1"
CATALOG_RELATIVE = "data/catalog.jsonl"
PUBLIC_RELATIVE = "data/public_set.jsonl"
STARTER_RELATIVE = "starter/agent.py"
EXPECTED_PUBLIC_CASES = 200

SOURCE_PATHS = tuple(
    dict.fromkeys(
        (
            *PHASE16B_SOURCE_PATHS,
            "scripts/run_buying_prefix_ablations.py",
            "tests/test_buying_prefix_ablations.py",
        )
    )
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


ARM_CONFIGS = (
    ArmConfig(
        "baseline",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        DISABLED_EVIDENCE_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "buying_withhold_v2",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    ),
    ArmConfig(
        "buying_prefix_v3",
        DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
        BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    ),
)
ARM_ORDER = tuple(config.arm_id for config in ARM_CONFIGS)


def _arm_contract(config: ArmConfig) -> dict[str, str]:
    return {
        "id": config.arm_id,
        "evidence_exposure": config.exposure_policy.value,
    }


def _fixed_architecture_contract() -> dict[str, str]:
    return {
        **_phase16b_architecture_contract(),
        "semantic_lexical_rescue_policy": (
            DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY.value
        ),
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
        or contract.get("single_varying_axis")
        != "evidence_exposure_policy"
    ):
        raise RuntimeError("Phase 16c contract identity or status drifted")
    if contract.get("arms") != [_arm_contract(config) for config in ARM_CONFIGS]:
        raise RuntimeError("Phase 16c arm definitions drifted")
    if contract.get("fixed_shared_architecture") != _fixed_architecture_contract():
        raise RuntimeError("Phase 16c shared architecture drifted")
    execution = contract.get("execution")
    if not isinstance(execution, dict) or execution.get(
        "public_arm_order"
    ) != list(ARM_ORDER):
        raise RuntimeError("Phase 16c execution order drifted")
    authority = contract.get("promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise RuntimeError("Phase 16c acquired promotion authority")
    return contract


def _validate_hashes(
    repository_root: Path,
    hashes: Mapping[str, object],
) -> None:
    if not hashes:
        raise RuntimeError("Phase 16c hash lock is empty")
    for relative, expected in hashes.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _HEX_SHA256_RE.fullmatch(expected) is None
        ):
            raise RuntimeError("Phase 16c hash lock is malformed")
        path = repository_root / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"Phase 16c locked path drifted: {relative}")


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
        raise RuntimeError("Phase 16c implementation lock schema drifted")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("lock_id") != IMPLEMENTATION_LOCK_ID
        or lock.get("experiment_id") != EXPERIMENT_ID
        or lock.get("status") != "locked_before_diagnostic_execution"
        or lock.get("arm_order") != list(ARM_ORDER)
        or lock.get("contract_sha256")
        != _sha256(repository_root / CONTRACT_RELATIVE)
    ):
        raise RuntimeError("Phase 16c implementation lock identity drifted")

    frozen = contract.get("frozen_inputs")
    if not isinstance(frozen, dict):
        raise RuntimeError("Phase 16c frozen inputs are unavailable")
    expected_inputs: dict[str, str] = {}
    for name in ("catalog", "public_diagnostic", "starter"):
        item = frozen.get(name)
        if not isinstance(item, dict):
            raise RuntimeError("Phase 16c frozen input entry is invalid")
        expected_inputs[str(item["path"])] = str(item["sha256"])
    if lock.get("input_sha256") != expected_inputs:
        raise RuntimeError("Phase 16c input lock disagrees with contract")
    _validate_hashes(repository_root, expected_inputs)

    sources = lock.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 16c source lock is incomplete")
    _validate_hashes(repository_root, sources)

    verification = lock.get("verification")
    expected_verification = {
        "focused_test_command",
        "focused_tests_passed",
        "complete_test_command",
        "complete_tests_passed",
        "phase13_oracle_command",
        "phase13_oracle_cases",
        "phase13_oracle_sha256",
        "completed_before_lock",
    }
    if (
        not isinstance(verification, dict)
        or set(verification) != expected_verification
        or verification.get("completed_before_lock") is not True
    ):
        raise RuntimeError("Phase 16c verification lock drifted")
    for key in (
        "focused_tests_passed",
        "complete_tests_passed",
        "phase13_oracle_cases",
    ):
        if type(verification.get(key)) is not int or int(verification[key]) <= 0:
            raise RuntimeError("Phase 16c verification count is invalid")
    return lock


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


def _diagnostic_observations(
    baseline: VariantRun,
    candidate: VariantRun,
    paired: Mapping[str, object],
) -> dict[str, bool]:
    transitions = paired["transitions"]
    observations = {
        "no_reference_only_hits": int(transitions["baseline_only_hit"]) == 0,
        "hit_rate_not_below_reference": (
            float(candidate.summary["hit_rate_at_10"])
            >= float(baseline.summary["hit_rate_at_10"])
        ),
        "mrr_not_below_reference": (
            float(candidate.summary["mrr"])
            >= float(baseline.summary["mrr"])
        ),
        "mttc_not_above_reference": (
            float(candidate.summary["mttc"])
            <= float(baseline.summary["mttc"])
        ),
        "technical_score_not_below_reference": (
            float(candidate.summary["recommended_technical_score"])
            >= float(baseline.summary["recommended_technical_score"])
        ),
        "technical_fault_free": _technical_faults_are_zero(candidate),
    }
    observations["all_descriptive_nonregression_observations_hold"] = all(
        observations.values()
    )
    return observations


def _comparison(
    reference: VariantRun,
    candidate: VariantRun,
) -> dict[str, object]:
    paired = _paired_statistics(reference.sessions, candidate.sessions)
    return {
        "metric_delta_vs_reference": _metric_deltas(
            reference.summary,
            candidate.summary,
        ),
        "paired_quality": paired,
        "scenario_subsets": _scenario_report(
            reference.sessions,
            candidate.sessions,
        ),
        "question_and_width_comparison": _common_action_equivalence(
            reference,
            candidate,
        ),
        "performance": _performance(reference, candidate),
        "diagnostic_observations": _diagnostic_observations(
            reference,
            candidate,
            paired,
        ),
    }


def _pass_report(runs: Mapping[str, VariantRun]) -> dict[str, object]:
    if tuple(runs) != ARM_ORDER:
        raise RuntimeError("Phase 16c public pass order drifted")
    baseline = runs["baseline"]
    withhold = runs["buying_withhold_v2"]
    prefix = runs["buying_prefix_v3"]
    return {
        "arms": {
            arm_id: _variant_public(runs[arm_id]) for arm_id in ARM_ORDER
        },
        "comparisons": {
            "buying_withhold_v2_vs_baseline": _comparison(
                baseline,
                withhold,
            ),
            "buying_prefix_v3_vs_baseline": _comparison(
                baseline,
                prefix,
            ),
            "buying_prefix_v3_vs_buying_withhold_v2": _comparison(
                withhold,
                prefix,
            ),
        },
    }


def _smoke_gate(
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
        value
        for key, value in gate.items()
        if key != "metric_direction_used_as_gate"
    )
    return {
        "arm_exactness": exactness,
        "technical_safety_gate": gate,
    }, bool(gate["authorize_full_public_diagnostic"])


def _warm_backend(catalog: Path, backend: object) -> dict[str, object]:
    started = time.perf_counter()
    agent = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
    )
    session_id = "phase16c-label-free-warmup"
    agent.reset(session_id, {})
    agent.respond(
        session_id,
        "I'm looking for a generic clothing item, but I'm still exploring.",
        1,
        10,
    )
    vocabulary = getattr(backend, "_ensure_bm25_vocabulary", None)
    if (
        int(agent.ranking_health["successes"]) != 1
        or not callable(vocabulary)
        or vocabulary() is not True
    ):
        raise RuntimeError("Phase 16c label-free backend warm-up failed")
    return {
        "label_free": True,
        "dense_bm25_and_catalog_frequency_cache_warmed": True,
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def _decision(full: Mapping[str, object]) -> dict[str, object]:
    comparisons = full.get("comparisons")
    if not isinstance(comparisons, dict):
        raise RuntimeError("Phase 16c full comparisons are unavailable")
    prefix_vs_withhold = comparisons[
        "buying_prefix_v3_vs_buying_withhold_v2"
    ]
    paired = prefix_vs_withhold["paired_quality"]
    transitions = paired["transitions"]
    return {
        "status": "exploratory_diagnostic_not_promotable",
        "automatic_adoption": False,
        "reason": (
            "The same public outcomes informed the prefix design, so this run "
            "can select the better exploratory direction but cannot validate "
            "generalization or change the starter."
        ),
        "preferred_exploratory_direction": "buying_prefix_v3",
        "prefix_vs_withhold": {
            "recovered_hits": int(transitions["candidate_only_hit"]),
            "lost_hits": int(transitions["baseline_only_hit"]),
            "metric_delta": prefix_vs_withhold["metric_delta_vs_reference"],
            "paired_bootstrap": paired["bootstrap"],
        },
        "next_evidence_required": "fresh_target_disjoint_quality_holdout",
        "may_change_starter": False,
    }


def run_phase16c(
    catalog_path: str | Path,
    public_path: str | Path,
    *,
    smoke_only: bool = False,
) -> dict[str, object]:
    _validate_environment()
    contract = _load_contract()
    lock = _validate_implementation_lock()
    catalog = Path(catalog_path).resolve()
    public = Path(public_path).resolve()
    if catalog != (REPOSITORY_ROOT / CATALOG_RELATIVE).resolve() or public != (
        REPOSITORY_ROOT / PUBLIC_RELATIVE
    ).resolve():
        raise RuntimeError("Phase 16c requires both frozen input paths")

    public_rows = load_jsonl(public)
    if len(public_rows) != EXPECTED_PUBLIC_CASES:
        raise RuntimeError("Phase 16c public case count drifted")
    catalog_ids, categories, products = catalog_index(catalog)

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
        raise RuntimeError("Phase 16c required backend capabilities are unavailable")
    del bootstrap
    warmup = _warm_backend(catalog, backend)

    smoke_indices = select_smoke_indices(public_rows)
    smoke_rows = [public_rows[index] for index in smoke_indices]
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
    smoke_gate, smoke_safe = _smoke_gate(smoke_primary, smoke_replay)
    smoke_report: dict[str, object] = {
        "status": "completed",
        "sample_count": len(smoke_rows),
        "cases_per_scenario": 10,
        "primary": _pass_report(smoke_primary),
        "replay": smoke_gate,
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
    elif not smoke_safe:
        full_report = {
            "status": "not_run",
            "reason": "frozen_technical_gate_rejected",
        }
        decision = {
            "status": "rejected_before_full",
            "automatic_adoption": False,
        }
    else:
        del smoke_replay
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
        raise RuntimeError("Phase 16c lock changed during execution")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "report_id": REPORT_ID,
        "experiment_id": EXPERIMENT_ID,
        "status": "exploratory_diagnostic_only",
        "ablation": {
            "single_varying_axis": "evidence_exposure_policy",
            "arm_order": list(ARM_ORDER),
            "arms": [_arm_contract(config) for config in ARM_CONFIGS],
            "shared_architecture": _fixed_architecture_contract(),
            "active_starter_changed": False,
        },
        "inputs": {
            "catalog_sha256": _sha256(catalog),
            "public_sha256": _sha256(public),
            "public_case_count": len(public_rows),
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
        },
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
        raise ValueError("Phase 16c result contains a forbidden raw-data key")
    if _ASIN_RE.search(serialized):
        raise ValueError("Phase 16c result contains a product identifier")
    if payload.get("privacy") != {
        "aggregate_only": True,
        "labels_used_only_after_agent_replay": True,
        "runtime_received_evaluation_labels": False,
        "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
    }:
        raise ValueError("Phase 16c privacy assertions are incomplete")
    authority = payload.get("promotion_authority")
    if not isinstance(authority, dict) or any(
        authority.get(key) is not False
        for key in (
            "automatic_promotion_allowed",
            "starter_may_be_changed_by_this_run",
        )
    ):
        raise ValueError("Phase 16c result claims promotion authority")


def _validate_output_path(output: Path, catalog: Path, public: Path) -> None:
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite diagnostic: {output}")
    protected = {
        catalog.resolve(),
        public.resolve(),
        REPOSITORY_ROOT / CONTRACT_RELATIVE,
        REPOSITORY_ROOT / IMPLEMENTATION_LOCK_RELATIVE,
        *(REPOSITORY_ROOT / relative for relative in SOURCE_PATHS),
    }
    if output.resolve() in protected:
        raise ValueError("Phase 16c output must not overwrite an input or source")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 16c buying-prefix diagnostic"
    )
    parser.add_argument("--catalog", default=CATALOG_RELATIVE)
    parser.add_argument("--public", default=PUBLIC_RELATIVE)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args()
    catalog = Path(arguments.catalog).resolve()
    public = Path(arguments.public).resolve()
    output = Path(arguments.output).resolve()
    _validate_output_path(output, catalog, public)
    _claim_output(output, smoke_only=arguments.smoke_only)
    payload = run_phase16c(
        catalog,
        public,
        smoke_only=arguments.smoke_only,
    )
    _replace_claim(output, payload)


if __name__ == "__main__":
    main()
