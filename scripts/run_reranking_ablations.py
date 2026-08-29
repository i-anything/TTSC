from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.ranking import (
    FUSED_ONLY_RANKING_POLICY,
    STAGE_A_RANKING_POLICY,
    RankingPolicy,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever, _sha256
from scripts.run_policy_ablations import _write_json_atomic


SCHEMA_VERSION = 1
METRIC_KEYS = (
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
SCENARIO_METRIC_KEYS = ("sample_count", "hit_rate_at_10", "mrr", "mttc")
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
SOURCE_PATHS = (
    "conversational_search/intent.py",
    "conversational_search/questions.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/strategy.py",
    "evaluator/local_evaluator.py",
    "scripts/run_reranking_ablations.py",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency_summary(durations_ns: list[int]) -> dict[str, int | float]:
    milliseconds = [duration / 1_000_000.0 for duration in durations_ns]
    warm = milliseconds[1:] if len(milliseconds) > 1 else milliseconds
    return {
        "count": len(milliseconds),
        "warm_count": len(warm),
        "p50": round(_percentile(milliseconds, 50), 6),
        "p90": round(_percentile(milliseconds, 90), 6),
        "p95": round(_percentile(milliseconds, 95), 6),
        "p99": round(_percentile(milliseconds, 99), 6),
        "warm_p95": round(_percentile(warm, 95), 6),
        "max": round(max(milliseconds, default=0.0), 6),
        "total": round(sum(milliseconds), 6),
    }


class RespondLatencyAgent:
    """Time delegate responses while retaining no request or label data."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        *,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._delegate = delegate
        self._clock_ns = time.perf_counter_ns if clock_ns is None else clock_ns
        self._durations_ns: list[int] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        started = self._clock_ns()
        try:
            return self._delegate.respond(session_id, user_message, turn, top_k)
        finally:
            self._durations_ns.append(max(0, self._clock_ns() - started))

    def latency_summary(self) -> dict[str, int | float]:
        return _latency_summary(self._durations_ns)


def _official_summary(result: dict) -> dict:
    """Project the evaluator result onto an aggregate-only allowlist."""

    usage = result.get("reported_token_usage") or {}
    scenarios = result.get("scenario_metrics") or {}
    return {
        "sample_count": result["sample_count"],
        **{key: result[key] for key in METRIC_KEYS},
        "reported_token_usage": {
            key: usage[key] for key in TOKEN_KEYS if key in usage
        },
        "scenario_metrics": {
            str(name): {
                key: metrics[key]
                for key in SCENARIO_METRIC_KEYS
                if key in metrics
            }
            for name, metrics in sorted(scenarios.items())
            if isinstance(metrics, dict)
        },
    }


def _scenario_hit_counts(result: dict) -> dict[str, int]:
    scenarios = set((result.get("scenario_metrics") or {}).keys())
    counts = {str(scenario): 0 for scenario in scenarios}
    for session in result.get("sessions") or ():
        if not isinstance(session, dict):
            continue
        scenario = session.get("scenario_type")
        if isinstance(scenario, str):
            counts.setdefault(scenario, 0)
            counts[scenario] += int(session.get("hit") is True)
    return dict(sorted(counts.items()))


def _metric_deltas(baseline: dict, candidate: dict) -> dict[str, float]:
    return {
        key: round(float(candidate[key]) - float(baseline[key]), 6)
        for key in METRIC_KEYS
    }


def _scenario_hit_count_deltas(
    baseline: dict,
    candidate: dict,
) -> dict[str, int]:
    baseline_counts = _scenario_hit_counts(baseline)
    candidate_counts = _scenario_hit_counts(candidate)
    return {
        scenario: candidate_counts.get(scenario, 0)
        - baseline_counts.get(scenario, 0)
        for scenario in sorted(set(baseline_counts) | set(candidate_counts))
    }


def _expected_turns(result: dict) -> int:
    return sum(
        int(session["first_hit_turn"])
        if session.get("first_hit_turn") is not None
        else MAX_TURNS
        for session in result.get("sessions") or ()
    )


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    ranking_policy: RankingPolicy,
) -> tuple[dict, dict]:
    guarded_backend = RouteHealthRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded_backend,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=ranking_policy,
    )
    timed_agent = RespondLatencyAgent(agent)
    started = time.perf_counter()
    result = evaluate(timed_agent, samples, catalog_ids, categories, products)
    evaluation_wall_seconds = time.perf_counter() - started
    expected_turns = _expected_turns(result)
    guarded_backend.validate(expected_turns)
    latency = timed_agent.latency_summary()
    if latency["count"] != expected_turns:
        raise RuntimeError(
            f"captured {latency['count']} response timings for "
            f"{expected_turns} evaluator turns"
        )
    return result, {
        "route_health": guarded_backend.summary(),
        "ranking_health": agent.ranking_health,
        "evaluation_wall_seconds": round(evaluation_wall_seconds, 6),
        "respond_latency_ms": latency,
    }


def run_reranking_ablations(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    verify_determinism: bool = False,
) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(
        catalog,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=FUSED_ONLY_RANKING_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing reranking ablation")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing reranking ablation")

    baseline_result, baseline_diagnostics = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        FUSED_ONLY_RANKING_POLICY,
    )
    candidate_result, candidate_diagnostics = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        STAGE_A_RANKING_POLICY,
    )

    repeat_diagnostics: dict | None = None
    payloads_equal: bool | None = None
    if verify_determinism:
        repeated_result, repeat_diagnostics = _run_variant(
            catalog,
            samples,
            catalog_ids,
            categories,
            products,
            backend,
            STAGE_A_RANKING_POLICY,
        )
        payloads_equal = repeated_result == candidate_result

    repository_root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_sha256": _sha256(catalog),
        "dataset_sha256": _sha256(dataset),
        "source_sha256": {
            relative: _sha256(repository_root / relative)
            for relative in SOURCE_PATHS
        },
        "run_order": [
            FUSED_ONLY_RANKING_POLICY.value,
            STAGE_A_RANKING_POLICY.value,
            *(
                [STAGE_A_RANKING_POLICY.value]
                if verify_determinism
                else []
            ),
        ],
        "question_policy": CONSERVATIVE_EARLY_OTHER_POLICY.name,
        "fusion_policy": COMPLETENESS_ADAPTIVE_RRF_POLICY.value,
        "variants": {
            FUSED_ONLY_RANKING_POLICY.value: {
                "official_metrics": _official_summary(baseline_result),
                **baseline_diagnostics,
            },
            STAGE_A_RANKING_POLICY.value: {
                "official_metrics": _official_summary(candidate_result),
                **candidate_diagnostics,
            },
        },
        "comparison": {
            "metric_delta": _metric_deltas(
                baseline_result,
                candidate_result,
            ),
            "scenario_hit_count_delta": _scenario_hit_count_deltas(
                baseline_result,
                candidate_result,
            ),
        },
        "determinism": {
            "requested": verify_determinism,
            "stage_a_evaluator_payload_equal": payloads_equal,
            "repeat_diagnostics": repeat_diagnostics,
        },
    }


def _validate_output_path(output: Path, catalog: Path, dataset: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    protected = {
        catalog.resolve(),
        dataset.resolve(),
        *(repository_root / relative for relative in SOURCE_PATHS),
    }
    if output.resolve() in protected:
        raise ValueError("output must not overwrite an input or source file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential Phase 3 versus Stage-A reranking ablations"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="run Stage-A a second time and compare evaluator payloads",
    )
    args = parser.parse_args()

    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output_path(output, catalog, dataset)
    payload = run_reranking_ablations(
        catalog,
        dataset,
        verify_determinism=args.verify_determinism,
    )
    _write_json_atomic(output, payload)


if __name__ == "__main__":
    main()
