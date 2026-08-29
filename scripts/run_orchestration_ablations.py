"""Strict sequential Phase 6 versus Phase 7 orchestration confirmation."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

from conversational_search.orchestration import (
    ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    OrchestrationPolicy,
)
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever, _sha256
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_reranking_ablations import (
    RespondLatencyAgent,
    _expected_turns,
    _metric_deltas,
    _official_summary,
)
from starter.agent import Agent


SCHEMA_VERSION = 1
IMPLEMENTATION_LOCK_RELATIVE = "docs/phase7_implementation_lock.json"
CONTRACT_RELATIVE = "docs/phase7_experiment_contract.json"
SOURCE_PATHS = (
    "conversational_search/intent.py",
    "conversational_search/orchestration.py",
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
    "scripts/run_reranking_ablations.py",
    "starter/agent.py",
    "starter/dense.py",
    "tests/test_orchestration.py",
    "tests/test_orchestration_ablations.py",
    "tests/test_service.py",
)
PHASE6_OFFICIAL = {
    "sample_count": 200,
    "hit_rate_at_10": 0.99,
    "mrr": 0.52223,
    "mttc": 3.07,
    "efficiency": 0.793,
    "recommended_technical_score": 0.810269,
    "scenario_metrics": {
        "boundary": {
            "sample_count": 10,
            "hit_rate_at_10": 0.9,
            "mrr": 0.385952,
            "mttc": 4.3,
        },
        "browsing": {
            "sample_count": 80,
            "hit_rate_at_10": 1.0,
            "mrr": 0.513219,
            "mttc": 2.5625,
        },
        "buying": {
            "sample_count": 80,
            "hit_rate_at_10": 0.9875,
            "mrr": 0.46,
            "mttc": 2.9,
        },
        "intent_override": {
            "sample_count": 30,
            "hit_rate_at_10": 1.0,
            "mrr": 0.757632,
            "mttc": 4.466667,
        },
    },
}
FROZEN_INPUT_SHA256 = {
    "catalog": "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    "dataset": "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    "evaluator/local_evaluator.py": "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564",
    "docs/phase6_results.json": "7c9048ac2fd87600e6f83f9b5e5b61b5ea79c40f917130802cee0001c5e1482d",
    "assets/bge-small-en-v1.5-int8/model_manifest.json": "f1130079f60555f7e35dc84344a33cd8e9afdcb4743c42afc94fb42b3991fd76",
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json": "c9b7291004d6ef78473b24886899ea51f427fc2e179c8216c8e8b65f6cf929b2",
}


class _AuditAgent(RespondLatencyAgent):
    """Retain private response/state traces only long enough to compare runs."""

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


def _lookup_accounting_exact(health: dict) -> bool:
    misses = sum(
        int(health.get(key, 0))
        for key in (
            "cold_misses",
            "dependency_misses",
            "backend_invalidations",
            "fault_invalidations",
        )
    )
    return int(health.get("lookups", -1)) == int(health.get("hits", -2)) + misses


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    policy: OrchestrationPolicy,
) -> tuple[dict, dict, list[tuple[object, object, object]]]:
    guarded_backend = RouteHealthRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded_backend,
        orchestration_policy=policy,
    )
    audited = _AuditAgent(agent)
    started = time.perf_counter()
    result = evaluate(audited, samples, catalog_ids, categories, products)
    wall_seconds = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    orchestration = agent.orchestration_health
    expected_route_calls = int(orchestration["searches"])
    guarded_backend.validate(expected_route_calls)
    route = guarded_backend.summary()
    ranking = agent.ranking_health
    slate = agent.slate_health
    latency = audited.latency_summary()

    if latency["count"] != expected_turns:
        raise RuntimeError("response timing coverage is incomplete")
    if int(orchestration["decisions"]) != expected_turns:
        raise RuntimeError("orchestration decision coverage is incomplete")
    if int(orchestration["searches"]) + int(orchestration["reuses"]) != expected_turns:
        raise RuntimeError("ordinary evaluator turns must search or reuse")
    if int(orchestration["skips"]) != 0:
        raise RuntimeError("the official top-k workload must not skip")
    if _route_call_count(route) != expected_route_calls:
        raise RuntimeError("route-call accounting is incomplete")
    if int(ranking["attempts"]) != expected_route_calls:
        raise RuntimeError("reranker-call accounting is incomplete")
    if int(ranking["successes"]) != expected_route_calls:
        raise RuntimeError("every eligible reranker call must succeed")
    if int(ranking["failures"]) or int(ranking["unavailable_skips"]):
        raise RuntimeError("reranker faults invalidate orchestration confirmation")
    if int(slate["attempts"]) != expected_turns or int(slate["successes"]) != expected_turns:
        raise RuntimeError("slate coverage is incomplete")
    if int(slate["failures"]):
        raise RuntimeError("slate faults invalidate orchestration confirmation")
    if int(route["fallback_turns"]):
        raise RuntimeError("fallback turns invalidate orchestration confirmation")
    if not _lookup_accounting_exact(orchestration):
        raise RuntimeError("cache lookup accounting is inconsistent")

    return result, {
        "expected_turns": expected_turns,
        "route_calls": expected_route_calls,
        "route_health": route,
        "ranking_health": ranking,
        "slate_health": slate,
        "orchestration_health": orchestration,
        "evaluation_wall_seconds": round(wall_seconds, 6),
        "respond_latency_ms": latency,
    }, audited.action_trace


def _run_independent(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[dict, list[tuple[object, object, object]]]:
    agent = Agent(catalog)
    if not getattr(agent.retrieval_backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable for independent verification")
    audited = _AuditAgent(agent)
    return (
        evaluate(audited, samples, catalog_ids, categories, products),
        audited.action_trace,
    )


def _warm_backend(catalog: Path, backend: object) -> None:
    """Initialize runtime kernels with one fixed, unlabeled synthetic request."""

    warmup = ConversationalSearchAgent(
        catalog,
        retriever=backend,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    )
    warmup.reset("phase7-label-free-runtime-warmup", {})
    warmup.respond(
        "phase7-label-free-runtime-warmup",
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
        raise RuntimeError("frozen Phase 7 inputs drifted; refusing public run")
    return observed


def _validate_implementation_lock(repository_root: Path) -> dict:
    lock_path = repository_root / IMPLEMENTATION_LOCK_RELATIVE
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "lock_id",
        "status",
        "contract_sha256",
        "source_sha256",
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("Phase 7 implementation lock schema drifted")
    if lock.get("schema_version") != 1:
        raise RuntimeError("unsupported Phase 7 implementation lock")
    if lock.get("status") != "locked_before_public_confirmation":
        raise RuntimeError("Phase 7 implementation is not frozen")
    if lock.get("contract_sha256") != _sha256(repository_root / CONTRACT_RELATIVE):
        raise RuntimeError("Phase 7 experiment contract drifted after lock")
    expected_source = lock.get("source_sha256")
    if not isinstance(expected_source, dict) or set(expected_source) != set(SOURCE_PATHS):
        raise RuntimeError("Phase 7 implementation source lock is incomplete")
    observed_source = {
        relative: _sha256(repository_root / relative)
        for relative in SOURCE_PATHS
    }
    if observed_source != expected_source:
        raise RuntimeError("Phase 7 implementation drifted after lock")
    return lock


def _phase6_summary_matches(result: dict) -> bool:
    summary = _official_summary(result)
    observed = {
        key: summary[key]
        for key in (
            "sample_count",
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "efficiency",
            "recommended_technical_score",
            "scenario_metrics",
        )
    }
    return observed == PHASE6_OFFICIAL


def run_orchestration_ablations(
    catalog_path: str | Path,
    dataset_path: str | Path,
) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    repository_root = Path(__file__).resolve().parents[1]
    frozen_inputs = _validate_frozen_inputs(repository_root, catalog, dataset)
    implementation_lock = _validate_implementation_lock(repository_root)
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        orchestration_policy=ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing Phase 7 run")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing Phase 7 run")
    _warm_backend(catalog, backend)

    candidate, candidate_diagnostics, candidate_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )
    baseline, baseline_diagnostics, baseline_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        ALWAYS_SEARCH_ORCHESTRATION_POLICY,
    )
    if not _phase6_summary_matches(baseline):
        raise RuntimeError("Phase 6 baseline drifted; refusing Phase 7 comparison")
    replay, replay_diagnostics, replay_trace = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )
    independent, independent_trace = _run_independent(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
    )

    baseline_wall = float(baseline_diagnostics["evaluation_wall_seconds"])
    candidate_wall = float(candidate_diagnostics["evaluation_wall_seconds"])
    baseline_p95 = float(baseline_diagnostics["respond_latency_ms"]["warm_p95"])
    candidate_p95 = float(candidate_diagnostics["respond_latency_ms"]["warm_p95"])
    wall_ratio = candidate_wall / baseline_wall if baseline_wall > 0 else math.inf
    p95_ratio = candidate_p95 / baseline_p95 if baseline_p95 > 0 else math.inf
    candidate_orchestration = candidate_diagnostics["orchestration_health"]
    baseline_orchestration = baseline_diagnostics["orchestration_health"]
    candidate_turns = int(candidate_diagnostics["expected_turns"])
    baseline_turns = int(baseline_diagnostics["expected_turns"])
    hits = int(candidate_orchestration["hits"])

    gates = {
        "phase6_baseline_metrics_exact": _phase6_summary_matches(baseline),
        "candidate_full_evaluator_payload_exact": candidate == baseline,
        "candidate_full_action_trace_exact": candidate_trace == baseline_trace,
        "candidate_replay_payload_and_trace_identical": (
            replay == candidate and replay_trace == candidate_trace
        ),
        "independent_starter_agent_matches_candidate": (
            independent == candidate and independent_trace == candidate_trace
        ),
        "cache_hits_and_avoided_work_are_positive": (
            hits > 0
            and int(candidate_orchestration["retrievals_avoided"]) > 0
            and int(candidate_orchestration["reranks_avoided"]) > 0
        ),
        "lookup_accounting_exact": _lookup_accounting_exact(candidate_orchestration),
        "hits_equal_avoided_retrievals_and_reranks": (
            hits
            == int(candidate_orchestration["retrievals_avoided"])
            == int(candidate_orchestration["reranks_avoided"])
        ),
        "candidate_route_and_rerank_accounting_exact": (
            int(candidate_diagnostics["route_calls"]) + hits == candidate_turns
            and int(candidate_diagnostics["ranking_health"]["attempts"]) + hits
            == candidate_turns
        ),
        "baseline_route_and_rerank_accounting_exact": (
            int(baseline_diagnostics["route_calls"]) == baseline_turns
            and int(baseline_diagnostics["ranking_health"]["attempts"])
            == baseline_turns
            and int(baseline_orchestration["hits"]) == 0
        ),
        "cache_memory_bounds_hold": (
            int(candidate_orchestration["entries"])
            <= int(candidate_orchestration["capacity"])
            and int(candidate_orchestration["cached_id_references"])
            <= int(candidate_orchestration["entries"])
            * int(candidate_orchestration["maximum_ids_per_entry"])
            and int(candidate_orchestration["cached_id_utf8_bytes"])
            <= int(candidate_orchestration["cached_id_references"])
            * int(candidate_orchestration["maximum_id_characters"])
            and int(candidate_orchestration["retained_cache_bytes"])
            <= 8 * 1024 * 1024
        ),
        "route_reranker_slate_cache_faults_are_zero": (
            int(candidate_diagnostics["route_health"]["fallback_turns"]) == 0
            and int(candidate_diagnostics["ranking_health"]["failures"]) == 0
            and int(candidate_diagnostics["slate_health"]["failures"]) == 0
            and int(candidate_orchestration["fault_invalidations"]) == 0
            and int(candidate_orchestration["store_rejections"]) == 0
        ),
        "warm_p95_ratio_at_most_1_05": p95_ratio <= 1.05,
        "wall_time_ratio_at_most_0_9": wall_ratio <= 0.9,
        "aggregate_privacy_projection_valid": True,
    }
    gates["adopt"] = all(gates.values())

    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": "phase7-exact-value-orchestration-v1",
        "candidate": "phase7-exact-ranking-reuse-v1",
        "baseline": "phase6-robust-intent-reducer-v1",
        "run_configuration": {
            "execution": "strictly_sequential",
            "onnx_threads": 1,
            "shared_immutable_backend": True,
            "cache_start": "cold",
            "run_order": [
                EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value,
                ALWAYS_SEARCH_ORCHESTRATION_POLICY.value,
                EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value,
                "independent_starter_agent",
            ],
            "backend_warmup": "one_fixed_unlabeled_request",
            "external_api_calls": 0,
        },
        "official_metrics": {
            "baseline": _official_summary(baseline),
            "candidate": _official_summary(candidate),
            "delta": _metric_deltas(baseline, candidate),
        },
        "health": {
            "baseline": baseline_diagnostics,
            "candidate": candidate_diagnostics,
            "candidate_replay": replay_diagnostics,
        },
        "latency": {
            "baseline_wall_seconds": round(baseline_wall, 6),
            "candidate_wall_seconds": round(candidate_wall, 6),
            "wall_time_ratio": round(wall_ratio, 6),
            "baseline_warm_p95_ms": round(baseline_p95, 6),
            "candidate_warm_p95_ms": round(candidate_p95, 6),
            "warm_p95_ratio": round(p95_ratio, 6),
        },
        "equivalence": {
            "candidate_evaluator_payload_equal": candidate == baseline,
            "candidate_action_state_slate_trace_equal": candidate_trace == baseline_trace,
            "candidate_replay_equal": replay == candidate and replay_trace == candidate_trace,
            "independent_entry_point_equal": independent == candidate,
            "independent_action_state_slate_trace_equal": independent_trace == candidate_trace,
        },
        "decision_gate": gates,
        "privacy": {
            "contains_queries_or_messages": False,
            "contains_profiles": False,
            "contains_product_or_sample_ids": False,
            "contains_session_or_turn_rows": False,
            "contains_raw_action_or_state_traces": False,
        },
        "reproducibility": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "frozen_input_sha256": frozen_inputs,
            "implementation_lock_id": implementation_lock["lock_id"],
            "contract_sha256": implementation_lock["contract_sha256"],
            "source_sha256": implementation_lock["source_sha256"],
        },
    }
    _validate_publication_privacy(payload)
    return payload


def _validate_publication_privacy(payload: dict) -> None:
    serialized = json.dumps(payload, sort_keys=True)
    forbidden_keys = {
        "sessions",
        "sample_id",
        "scenario_type",
        "ground_truth",
        "target",
        "user_message",
        "user_profile",
        "action_trace",
        "slate_state",
        "intent_state",
    }

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    if not forbidden_keys.isdisjoint(keys(payload)):
        raise RuntimeError("Phase 7 publication contains a forbidden field")
    if "B0" in serialized:
        raise RuntimeError("Phase 7 publication appears to contain product IDs")


def _validate_output(output: Path, catalog: Path, dataset: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    resolved = output.resolve()
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
        raise ValueError("output must not overwrite an input or source file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen sequential Phase 7 orchestration ablation"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output(output, catalog, dataset)
    _write_json_atomic(output, run_orchestration_ablations(catalog, dataset))


if __name__ == "__main__":
    main()
