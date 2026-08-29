from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.retrieval import RetrievalResult
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import (
    COMPLETENESS_ADAPTIVE_RRF_POLICY,
    EQUAL_RRF_POLICY,
    FusionPolicy,
    RouteWeights,
)
from evaluator.local_evaluator import MAX_TURNS, catalog_index, evaluate, load_jsonl
from scripts.run_policy_ablations import _write_json_atomic


SCHEMA_VERSION = 1
FUSION_POLICIES = {
    EQUAL_RRF_POLICY.value: EQUAL_RRF_POLICY,
    COMPLETENESS_ADAPTIVE_RRF_POLICY.value: COMPLETENESS_ADAPTIVE_RRF_POLICY,
}
SAFE_ROUTE_STATUSES = frozenset({"ok", "empty"})


class RouteHealthRetriever:
    """A/B-only adapter that audits route health without seeing labels."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._bm25_statuses: Counter[str] = Counter()
        self._dense_statuses: Counter[str] = Counter()
        self._fallbacks = 0

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        route_weights: RouteWeights,
    ) -> RetrievalResult:
        result = self._backend.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            route_weights=route_weights,
        )
        if not isinstance(result, RetrievalResult):
            raise TypeError("search_with_trace must return RetrievalResult")
        self._bm25_statuses[result.trace.bm25_status] += 1
        self._dense_statuses[result.trace.dense_status] += 1
        self._fallbacks += int(result.trace.used_fallback)
        return result

    def search(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        route_weights: RouteWeights,
    ) -> list[str]:
        result = self.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            route_weights=route_weights,
        )
        return list(result.recommendations)

    def candidate_documents(self, parent_asins: Sequence[str]) -> tuple:
        return self._backend.candidate_documents(parent_asins)

    @property
    def ranking_cache_capability(self) -> object:
        """Forward only the backend's explicit exact-ranking capability."""

        return self._backend.ranking_cache_capability

    @property
    def snapshot_token(self) -> object:
        """Forward the immutable backend identity for exact-reuse experiments."""

        return self._backend.snapshot_token

    def validate(self, expected_turns: int | None = None) -> None:
        observed_turns = sum(self._bm25_statuses.values())
        if expected_turns is not None and observed_turns != expected_turns:
            raise RuntimeError(
                f"captured {observed_turns} route traces for {expected_turns} evaluator turns"
            )
        faults = {
            "bm25": sorted(set(self._bm25_statuses) - SAFE_ROUTE_STATUSES),
            "dense": sorted(set(self._dense_statuses) - SAFE_ROUTE_STATUSES),
        }
        if faults["bm25"] or faults["dense"]:
            raise RuntimeError(f"route faults invalidate fusion ablation: {faults}")

    def summary(self) -> dict:
        return {
            "bm25": dict(sorted(self._bm25_statuses.items())),
            "dense": dict(sorted(self._dense_statuses.items())),
            "fallback_turns": self._fallbacks,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _policy_contract(policy: FusionPolicy) -> dict:
    if policy is EQUAL_RRF_POLICY:
        return {
            "bm25": "0.5",
            "dense": "0.5",
            "completeness": "unused",
        }
    return {
        "bm25": "0.4 + 0.2 * C_t",
        "dense": "1 - bm25",
        "completeness": (
            "clip((strong_requirements + 0.5 * weak_requirements) / 3, 0, 1)"
        ),
    }


def run_fusion_policies(
    catalog_path: str | Path,
    dataset_path: str | Path,
    policy_names: Sequence[str],
) -> dict:
    if not policy_names:
        raise ValueError("at least one fusion policy is required")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("fusion policies must be unique")
    unknown = [name for name in policy_names if name not in FUSION_POLICIES]
    if unknown:
        raise ValueError(f"unknown fusion policies: {unknown}")

    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    runtime = ConversationalSearchAgent(catalog)
    backend = runtime.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing fusion ablation")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing fusion ablation")

    results: dict[str, dict] = {}
    route_health: dict[str, dict] = {}
    for name in policy_names:
        policy = FUSION_POLICIES[name]
        guarded_backend = RouteHealthRetriever(backend)
        agent = ConversationalSearchAgent(
            catalog,
            retriever=guarded_backend,
            question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
            fusion_policy=policy,
        )
        started = time.perf_counter()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        elapsed = time.perf_counter() - started
        expected_turns = sum(
            int(session["first_hit_turn"])
            if session["first_hit_turn"] is not None
            else MAX_TURNS
            for session in result["sessions"]
        )
        guarded_backend.validate(expected_turns)
        results[name] = result
        route_health[name] = guarded_backend.summary()
        print(
            f"{name}: {elapsed:.3f}s, "
            f"score={result['recommended_technical_score']:.6f}"
        )

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = (
        "conversational_search/intent.py",
        "conversational_search/questions.py",
        "conversational_search/retrieval.py",
        "conversational_search/service.py",
        "conversational_search/strategy.py",
        "evaluator/local_evaluator.py",
        "scripts/run_fusion_ablations.py",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_sha256": _sha256(catalog),
        "dataset_sha256": _sha256(dataset),
        "source_sha256": {
            relative: _sha256(repository_root / relative)
            for relative in source_paths
        },
        "question_policy": CONSERVATIVE_EARLY_OTHER_POLICY.name,
        "policies": {
            name: {
                "contract": _policy_contract(FUSION_POLICIES[name]),
                "route_health": route_health[name],
                "result": results[name],
            }
            for name in policy_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential fusion-policy ablations on one shared backend"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(FUSION_POLICIES),
        default=tuple(FUSION_POLICIES),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    protected = {Path(args.catalog).resolve(), Path(args.dataset).resolve()}
    if output in protected:
        raise ValueError("output must not overwrite the catalog or dataset")
    payload = run_fusion_policies(args.catalog, args.dataset, args.policies)
    _write_json_atomic(output, payload)


if __name__ == "__main__":
    main()
