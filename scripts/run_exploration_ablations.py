from __future__ import annotations

import argparse
import time
from pathlib import Path

from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.ranking import STAGE_A_RANKING_POLICY
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import (
    REPEAT_TOP_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    SlatePolicy,
)
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever, _sha256
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_reranking_ablations import (
    RespondLatencyAgent,
    _expected_turns,
    _metric_deltas,
    _official_summary,
    _scenario_hit_count_deltas,
)


SCHEMA_VERSION = 1
SOURCE_PATHS = (
    "assets/bge-small-en-v1.5-int8/model_manifest.json",
    "assets/search-index-bge-small-en-v1.5-v2/manifest.json",
    "conversational_search/intent.py",
    "conversational_search/questions.py",
    "conversational_search/ranking.py",
    "conversational_search/retrieval.py",
    "conversational_search/service.py",
    "conversational_search/slates.py",
    "conversational_search/strategy.py",
    "docs/phase5_experiment_contract.json",
    "evaluator/local_evaluator.py",
    "preprocessing/encoder.py",
    "scripts/run_exploration_ablations.py",
    "scripts/run_fusion_ablations.py",
    "scripts/run_policy_ablations.py",
    "scripts/run_reranking_ablations.py",
    "starter/dense.py",
)

PHASE4_BASELINE = {
    "sample_count": 200,
    "hit_rate_at_10": 0.885,
    "mrr": 0.514109,
    "mttc": 3.695,
    "efficiency": 0.7305,
    "recommended_technical_score": 0.742833,
    "scenario_metrics": {
        "boundary": {
            "sample_count": 10,
            "hit_rate_at_10": 0.7,
            "mrr": 0.349286,
            "mttc": 5.5,
        },
        "browsing": {
            "sample_count": 80,
            "hit_rate_at_10": 0.9,
            "mrr": 0.497426,
            "mttc": 3.2375,
        },
        "buying": {
            "sample_count": 80,
            "hit_rate_at_10": 0.8875,
            "mrr": 0.468824,
            "mttc": 3.5,
        },
        "intent_override": {
            "sample_count": 30,
            "hit_rate_at_10": 0.9,
            "mrr": 0.734299,
            "mttc": 4.833333,
        },
    },
}


def _validate_phase4_baseline(result: dict) -> None:
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
    if observed != PHASE4_BASELINE:
        raise RuntimeError(
            "Phase 4 baseline metrics drifted; refusing Phase 5 comparison"
        )


def _validate_variant_health(
    expected_turns: int,
    slate_policy: SlatePolicy,
    route_health: dict,
    ranking_health: dict,
    slate_health: dict,
) -> None:
    if route_health.get("fallback_turns") != 0:
        raise RuntimeError("fallback turns invalidate exploration ablation")
    expected_ranking = {
        "attempts": expected_turns,
        "successes": expected_turns,
        "failures": 0,
        "unavailable_skips": 0,
    }
    if any(
        ranking_health.get(key) != value
        for key, value in expected_ranking.items()
    ):
        raise RuntimeError("reranker health invalidates exploration ablation")

    if slate_policy is REPEAT_TOP_SLATE_POLICY:
        expected_slate = {"attempts": 0, "successes": 0, "failures": 0}
    else:
        expected_slate = {
            "attempts": expected_turns,
            "successes": expected_turns,
            "failures": 0,
        }
    if any(
        slate_health.get(key) != value for key, value in expected_slate.items()
    ):
        raise RuntimeError("slate health invalidates exploration ablation")
    if slate_policy is STAGNATION_AWARE_SLATE_POLICY:
        classified_turns = sum(
            int(slate_health.get(key, -expected_turns - 1))
            for key in ("initializations", "ranking_resets", "stagnant_turns")
        )
        if classified_turns != expected_turns:
            raise RuntimeError("slate trace coverage is incomplete")


def _run_variant(
    catalog: Path,
    samples: list[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
    backend: object,
    slate_policy: SlatePolicy,
) -> tuple[dict, dict]:
    guarded_backend = RouteHealthRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded_backend,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=STAGE_A_RANKING_POLICY,
        slate_policy=slate_policy,
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
    route_health = guarded_backend.summary()
    ranking_health = agent.ranking_health
    slate_health = agent.slate_health
    _validate_variant_health(
        expected_turns,
        slate_policy,
        route_health,
        ranking_health,
        slate_health,
    )
    return result, {
        "route_health": route_health,
        "ranking_health": ranking_health,
        "slate_health": slate_health,
        "evaluation_wall_seconds": round(evaluation_wall_seconds, 6),
        "respond_latency_ms": latency,
    }


def run_exploration_ablations(
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
        ranking_policy=STAGE_A_RANKING_POLICY,
        slate_policy=REPEAT_TOP_SLATE_POLICY,
    )
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing exploration ablation")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing exploration ablation")

    baseline_result, baseline_diagnostics = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        REPEAT_TOP_SLATE_POLICY,
    )
    _validate_phase4_baseline(baseline_result)
    candidate_result, candidate_diagnostics = _run_variant(
        catalog,
        samples,
        catalog_ids,
        categories,
        products,
        backend,
        STAGNATION_AWARE_SLATE_POLICY,
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
            STAGNATION_AWARE_SLATE_POLICY,
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
            REPEAT_TOP_SLATE_POLICY.value,
            STAGNATION_AWARE_SLATE_POLICY.value,
            *(
                [STAGNATION_AWARE_SLATE_POLICY.value]
                if verify_determinism
                else []
            ),
        ],
        "question_policy": CONSERVATIVE_EARLY_OTHER_POLICY.name,
        "fusion_policy": COMPLETENESS_ADAPTIVE_RRF_POLICY.value,
        "ranking_policy": STAGE_A_RANKING_POLICY.value,
        "variants": {
            REPEAT_TOP_SLATE_POLICY.value: {
                "official_metrics": _official_summary(baseline_result),
                **baseline_diagnostics,
            },
            STAGNATION_AWARE_SLATE_POLICY.value: {
                "official_metrics": _official_summary(candidate_result),
                **candidate_diagnostics,
            },
        },
        "comparison": {
            "metric_delta": _metric_deltas(baseline_result, candidate_result),
            "scenario_hit_count_delta": _scenario_hit_count_deltas(
                baseline_result,
                candidate_result,
            ),
        },
        "determinism": {
            "requested": verify_determinism,
            "candidate_evaluator_payload_equal": payloads_equal,
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
        description="Run sequential Phase 4 versus stagnation-aware slate ablations"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="run the stagnation-aware candidate twice and compare evaluator payloads",
    )
    args = parser.parse_args()

    catalog = Path(args.catalog).resolve()
    dataset = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    _validate_output_path(output, catalog, dataset)
    payload = run_exploration_ablations(
        catalog,
        dataset,
        verify_determinism=args.verify_determinism,
    )
    _write_json_atomic(output, payload)


if __name__ == "__main__":
    main()
