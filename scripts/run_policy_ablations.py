from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

from conversational_search.questions import QUESTION_POLICIES
from conversational_search.service import ConversationalSearchAgent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run_policies(
    catalog_path: str | Path,
    dataset_path: str | Path,
    policy_names: Sequence[str],
) -> dict:
    if not policy_names:
        raise ValueError("at least one question policy is required")
    if len(set(policy_names)) != len(policy_names):
        raise ValueError("question policies must be unique")
    unknown = [name for name in policy_names if name not in QUESTION_POLICIES]
    if unknown:
        raise ValueError(f"unknown question policies: {unknown}")

    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    first_policy = QUESTION_POLICIES[policy_names[0]]
    first_agent = ConversationalSearchAgent(catalog, question_policy=first_policy)
    backend = first_agent.retrieval_backend
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing hybrid ablation")
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 retrieval is unavailable; refusing hybrid ablation")

    results: dict[str, dict] = {}
    for index, name in enumerate(policy_names):
        policy = QUESTION_POLICIES[name]
        agent = first_agent if index == 0 else ConversationalSearchAgent(
            catalog,
            retriever=backend,
            question_policy=policy,
        )
        started = time.perf_counter()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        elapsed = time.perf_counter() - started
        results[name] = result
        print(f"{name}: {elapsed:.3f}s, score={result['recommended_technical_score']:.6f}")

    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_sha256": _sha256(catalog),
        "dataset_sha256": _sha256(dataset),
        "policies": {
            name: {
                "priority": list(QUESTION_POLICIES[name].priority),
                "requeue_interrupted": QUESTION_POLICIES[name].requeue_interrupted,
                "result": results[name],
            }
            for name in policy_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential question-policy ablations on one shared backend"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(QUESTION_POLICIES),
        default=tuple(QUESTION_POLICIES),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    protected = {Path(args.catalog).resolve(), Path(args.dataset).resolve()}
    if output in protected:
        raise ValueError("output must not overwrite the catalog or dataset")
    result = run_policies(args.catalog, args.dataset, args.policies)
    _write_json_atomic(output, result)


if __name__ == "__main__":
    main()
