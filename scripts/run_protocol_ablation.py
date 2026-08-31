#!/usr/bin/env python3
"""Run aggregate-only catalog/protocol ablations with one shared retriever."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from conversational_search.decision_policy import (
    EXPECTED_UTILITY_DECISION_POLICY,
)
from conversational_search.exposure_policy import (
    BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY,
    PROTOCOL_POSTERIOR_EXPOSURE_POLICY,
    PROTOCOL_REPLY_TREE_EXPOSURE_POLICY,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
)
from conversational_search.protocol_index import (
    DISABLED_PROTOCOL_REFUTATION_POLICY,
    ELIGIBLE_CONTINUATION_REFUTATION_POLICY,
    FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
)
from conversational_search.questions import WILDCARD_OTHER_POLICY
from conversational_search.ranking import (
    LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
)
from conversational_search.retrieval_routing import (
    SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


ARMS = (
    "baseline",
    "catalog_current",
    "catalog_refute_current",
    "catalog_posterior",
    "catalog_refute_posterior",
    "catalog_metric_aware",
    "catalog_reply_tree",
    "catalog_expected",
    "catalog_refute_expected",
)


def build_agent(
    arm: str,
    catalog_path: Path,
    retrieval_backend: object,
) -> ConversationalSearchAgent:
    if arm not in ARMS or arm == "baseline":
        raise ValueError("arm must name a non-baseline protocol ablation")
    refutation = (
        ELIGIBLE_CONTINUATION_REFUTATION_POLICY
        if "refute" in arm
        or arm in {"catalog_metric_aware", "catalog_reply_tree"}
        else DISABLED_PROTOCOL_REFUTATION_POLICY
    )
    if arm.endswith("expected"):
        exposure = DISABLED_EVIDENCE_EXPOSURE_POLICY
        decision_options = {"decision_policy": EXPECTED_UTILITY_DECISION_POLICY}
    else:
        exposure = (
            PROTOCOL_REPLY_TREE_EXPOSURE_POLICY
            if arm == "catalog_reply_tree"
            else PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY
            if arm == "catalog_metric_aware"
            else
            PROTOCOL_POSTERIOR_EXPOSURE_POLICY
            if arm.endswith("posterior")
            else BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY
        )
        decision_options = {}
    return ConversationalSearchAgent(
        catalog_path,
        retriever=retrieval_backend,
        evidence_exposure_policy=exposure,
        orchestration_policy=EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
        protocol_catalog_policy=FULL_TRANSCRIPT_PROTOCOL_CATALOG_POLICY,
        protocol_refutation_policy=refutation,
        question_policy=WILDCARD_OTHER_POLICY,
        ranking_policy=LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY,
        retrieval_routing_policy=SMART_HYBRID_RETRIEVAL_ROUTING_POLICY,
        slate_policy=INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        **decision_options,
    )


def aggregate(result: dict, elapsed_seconds: float) -> dict:
    return {
        key: result[key]
        for key in (
            "sample_count",
            "hit_rate_at_10",
            "mrr",
            "mttc",
            "efficiency",
            "recommended_technical_score",
            "reported_token_usage",
            "scenario_metrics",
        )
    } | {"elapsed_seconds": round(elapsed_seconds, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="comma-separated frozen arm names",
    )
    args = parser.parse_args()
    requested = tuple(value.strip() for value in args.arms.split(",") if value.strip())
    if not requested or any(value not in ARMS for value in requested):
        raise SystemExit(f"--arms must use: {', '.join(ARMS)}")

    catalog_path = Path(args.catalog)
    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(catalog_path)
    baseline = Agent(catalog_path)
    output: dict[str, object] = {
        "schema": "protocol-ablation-aggregate-v1",
        "dataset": str(Path(args.dataset)),
        "arms": {},
    }
    for arm in requested:
        agent = (
            baseline
            if arm == "baseline"
            else build_agent(arm, catalog_path, baseline.retrieval_backend)
        )
        started = time.perf_counter()
        result = evaluate(agent, samples, catalog_ids, categories, products)
        summary = aggregate(result, time.perf_counter() - started)
        output["arms"][arm] = summary
        print(json.dumps({"arm": arm, **summary}, sort_keys=True), flush=True)
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
