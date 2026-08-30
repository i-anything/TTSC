from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from pathlib import Path
from unittest import mock

from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.intent import ROBUST_INTENT_POLICY
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.ranking import STAGE_A_RANKING_POLICY
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    HybridRetriever,
)
from conversational_search.service import ConversationalSearchAgent, _load_dense_runtime
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from scripts.run_fusion_ablations import RouteHealthRetriever
from scripts.run_policy_ablations import _write_json_atomic
from scripts.run_reranking_ablations import RespondLatencyAgent


SCHEMA_VERSION = 1
OFFICIAL_METRICS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
)
SOURCE_DIRECTORIES = (
    "conversational_search",
    "evaluator",
    "preprocessing",
    "starter",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _tree_identity(repository_root: Path) -> dict[str, object]:
    paths = [Path(__file__).resolve()]
    for directory in SOURCE_DIRECTORIES:
        paths.extend((repository_root / directory).rglob("*.py"))
    unique = sorted({path.resolve() for path in paths})
    digest = hashlib.sha256()
    for path in unique:
        relative = path.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return {"file_count": len(unique), "sha256": digest.hexdigest()}


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_variant(
    *,
    catalog_path: str | Path,
    dataset_path: str | Path,
    model_assets: str | Path,
    dense_index_path: str | Path,
    variant: str,
) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[1]
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    model_root = Path(model_assets).resolve()
    index_root = Path(dense_index_path).resolve()
    model_manifest = model_root / "model_manifest.json"
    index_manifest = index_root / "manifest.json"
    ready = index_root / "READY"
    for required in (catalog, dataset, model_manifest, index_manifest, ready):
        if not required.is_file():
            raise FileNotFoundError(required)

    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    encoder, dense_index = _load_dense_runtime(catalog, model_root, index_root)
    backend = HybridRetriever(
        catalog,
        encoder=encoder,
        dense_index=dense_index,
        protocol_evidence=False,
    )
    if not backend.bm25_available or not backend.dense_available:
        raise RuntimeError("both BM25 and dense routes must be available")
    guarded = RouteHealthRetriever(backend)
    agent = ConversationalSearchAgent(
        catalog,
        retriever=guarded,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
        fusion_policy=COMPLETENESS_ADAPTIVE_RRF_POLICY,
        ranking_policy=STAGE_A_RANKING_POLICY,
        profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        intent_policy=ROBUST_INTENT_POLICY,
        decision_policy=PROTECTED_DECISION_POLICY,
        requirement_probe_policy=DISABLED_REQUIREMENT_PROBE_POLICY,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    )
    timed = RespondLatencyAgent(agent)
    identifiers = [uuid.UUID(int=index + 1) for index in range(len(samples))]
    started = time.perf_counter()
    with mock.patch(
        "evaluator.local_evaluator.uuid.uuid4",
        side_effect=identifiers,
    ) as identifier_factory:
        raw = evaluate(timed, samples, catalog_ids, categories, products)
    elapsed = time.perf_counter() - started
    if identifier_factory.call_count != len(samples):
        raise RuntimeError("evaluator session identity count drifted")

    orchestration = dict(agent.orchestration_health)
    guarded.validate(int(orchestration["searches"]))
    route_health = guarded.summary()
    if route_health["fallback_turns"]:
        raise RuntimeError("retrieval fallback invalidates encoder comparison")
    ranking_health = dict(agent.ranking_health)
    if ranking_health.get("failures") or ranking_health.get("unavailable_skips"):
        raise RuntimeError("ranking failure invalidates encoder comparison")

    metrics = {key: raw[key] for key in OFFICIAL_METRICS}
    token_usage = dict(raw.get("reported_token_usage") or {})
    del raw
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "embedding-package-ablation-v1",
        "variant": variant,
        "policy_freeze": {
            "question": CONSERVATIVE_EARLY_OTHER_POLICY.name,
            "fusion": COMPLETENESS_ADAPTIVE_RRF_POLICY.value,
            "ranking": STAGE_A_RANKING_POLICY.value,
            "profile": BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            "slate": INTENT_EPOCH_NOVELTY_SLATE_POLICY.value,
            "intent": ROBUST_INTENT_POLICY.value,
            "decision": PROTECTED_DECISION_POLICY.value,
            "requirement_probe": DISABLED_REQUIREMENT_PROBE_POLICY.value,
            "orchestration": EXACT_RANKING_REUSE_ORCHESTRATION_POLICY.value,
        },
        "metrics": metrics,
        "reported_token_usage": token_usage,
        "runtime": {
            "cpu_only": True,
            "onnx_threads": 1,
            "evaluation_wall_seconds": round(elapsed, 6),
            "response_latency_ms": timed.latency_summary(),
            "route_health": route_health,
            "orchestration_health": orchestration,
            "ranking_health": ranking_health,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "artifacts": {
            "catalog_sha256": _sha256(catalog),
            "dataset_sha256": _sha256(dataset),
            "model_manifest_sha256": _sha256(model_manifest),
            "index_manifest_sha256": _sha256(index_manifest),
            "index_ready_sha256": ready.read_text(encoding="ascii").strip(),
            "model_asset_identity_sha256": encoder.metadata.asset_identity_sha256,
            "model_bytes": _directory_bytes(model_root),
            "index_bytes": _directory_bytes(index_root),
            "dimension": encoder.metadata.dimension,
            "source_tree": _tree_identity(repository_root),
        },
        "environment": {
            "external_api_calls": 0,
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "privacy": {
            "individual_sessions_persisted": False,
            "scenario_metrics_persisted": False,
            "individual_outcomes_inspected": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one aggregate-only frozen embedding-package variant"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model-assets", required=True)
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_variant(
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        model_assets=args.model_assets,
        dense_index_path=args.dense_index,
        variant=args.variant,
    )
    _write_json_atomic(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
