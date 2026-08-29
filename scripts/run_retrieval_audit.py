from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from conversational_search.questions import CONSERVATIVE_EARLY_OTHER_POLICY
from conversational_search.retrieval import (
    ROUTE_LIMIT,
    RRF_K,
    RetrievalResult,
    RetrievalTrace,
)
from conversational_search.service import ConversationalSearchAgent
from conversational_search.strategy import RouteWeights, intent_completeness
from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    evaluate,
    load_jsonl,
    materialize_hidden_fields,
)


SCHEMA_VERSION = 1
RANK_CUTOFFS = (10, 50, 100)
WATERFALL_CATEGORIES = (
    "agreement_kept",
    "bm25_rescue",
    "dense_rescue",
    "deep_rrf_promotion",
    "agreement_lost",
    "bm25_lost",
    "dense_lost",
    "union_not_promoted",
    "not_retrieved",
    "fallback_hit",
    "fallback_miss",
)
OFFICIAL_METRIC_KEYS = (
    "sample_count",
    "hit_rate_at_10",
    "mrr",
    "mttc",
    "efficiency",
    "recommended_technical_score",
    "reported_token_usage",
    "scenario_metrics",
)


@dataclass(frozen=True, slots=True)
class UnlabeledTurnTrace:
    """A production turn trace captured before any target label is joined."""

    session_ordinal: int
    turn: int
    output_ids: tuple[str, ...]
    trace: RetrievalTrace
    intent_completeness_proxy: float


@dataclass(frozen=True, slots=True)
class _LabeledTurn:
    session_ordinal: int
    scenario_type: str
    turn: int
    eligible: bool
    intent_completeness_proxy: float
    bm25_status: str
    dense_status: str
    bm25_rank: int | None
    dense_rank: int | None
    fused_rank: int | None
    output_rank: int | None
    bm25_count: int
    dense_count: int
    fused_count: int
    route_jaccard: float
    used_fallback: bool
    waterfall: str | None


class TraceCaptureRetriever:
    """Experiment-only wrapper; it never receives targets, scenarios, or profiles."""

    def __init__(self, backend: object) -> None:
        self._backend = backend
        self._pending: list[RetrievalResult] = []

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
        self._pending.append(result)
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

    def pop(self) -> RetrievalResult:
        if len(self._pending) != 1:
            raise RuntimeError(
                "each agent response must execute exactly one traced retrieval"
            )
        return self._pending.pop()

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class AuditAgent:
    """Delegates production behavior and captures only label-free turn outputs."""

    def __init__(
        self,
        delegate: ConversationalSearchAgent,
        capture: TraceCaptureRetriever,
    ) -> None:
        self._delegate = delegate
        self._capture = capture
        self._session_ordinals: dict[str, int] = {}
        self.records: list[UnlabeledTurnTrace] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        if self._capture.pending_count:
            raise RuntimeError("unconsumed retrieval trace before reset")
        if session_id in self._session_ordinals:
            raise RuntimeError("audit evaluator reused a session identifier")
        self._session_ordinals[session_id] = len(self._session_ordinals)
        self._delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        response = self._delegate.respond(session_id, user_message, turn, top_k)
        retrieval = self._capture.pop()
        output_ids = tuple(
            item["parent_asin"]
            for item in response.get("recommendations", [])
            if isinstance(item, dict) and isinstance(item.get("parent_asin"), str)
        )
        state = self._delegate.session_state(session_id)
        self.records.append(
            UnlabeledTurnTrace(
                session_ordinal=self._session_ordinals[session_id],
                turn=turn,
                output_ids=output_ids,
                trace=retrieval.trace,
                intent_completeness_proxy=intent_completeness(state),
            )
        )
        return response

    @property
    def session_count(self) -> int:
        return len(self._session_ordinals)


def _rank(items: Sequence[str], target: str) -> int | None:
    try:
        return items.index(target) + 1
    except ValueError:
        return None


def _at(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def _validate_trace_coverage(
    records: Sequence[UnlabeledTurnTrace],
    sessions: Sequence[dict],
) -> None:
    """Reject evaluator runs where a swallowed audit failure changed the replay."""

    expected_total = sum(
        int(session["first_hit_turn"])
        if session.get("first_hit_turn") is not None
        else MAX_TURNS
        for session in sessions
    )
    if len(records) != expected_total:
        raise RuntimeError(
            f"captured {len(records)} traces but evaluator reached {expected_total} turns"
        )
    grouped: dict[int, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.session_ordinal].append(record.turn)
    if set(grouped) != set(range(len(sessions))):
        raise RuntimeError("trace session ordinals do not match evaluator sessions")
    for ordinal, session in enumerate(sessions):
        terminal = (
            int(session["first_hit_turn"])
            if session.get("first_hit_turn") is not None
            else MAX_TURNS
        )
        expected_turns = list(range(1, terminal + 1))
        if grouped[ordinal] != expected_turns:
            raise RuntimeError(
                f"session ordinal {ordinal} has non-contiguous audit turns"
            )


def _classify_waterfall(
    *,
    bm25_rank: int | None,
    dense_rank: int | None,
    fused_rank: int | None,
    output_rank: int | None,
    used_fallback: bool,
) -> str:
    if used_fallback:
        return "fallback_hit" if _at(output_rank, TOP_K) else "fallback_miss"

    bm25_top = _at(bm25_rank, TOP_K)
    dense_top = _at(dense_rank, TOP_K)
    fused_top = _at(fused_rank, TOP_K)
    if fused_top:
        if bm25_top and dense_top:
            return "agreement_kept"
        if bm25_top:
            return "bm25_rescue"
        if dense_top:
            return "dense_rescue"
        return "deep_rrf_promotion"
    if bm25_top and dense_top:
        return "agreement_lost"
    if bm25_top:
        return "bm25_lost"
    if dense_top:
        return "dense_lost"
    if bm25_rank is not None or dense_rank is not None:
        return "union_not_promoted"
    return "not_retrieved"


def _join_labels(
    records: Sequence[UnlabeledTurnTrace],
    samples: Sequence[dict],
    products: dict[str, dict],
) -> list[_LabeledTurn]:
    rows: list[_LabeledTurn] = []
    for record in records:
        if not 0 <= record.session_ordinal < len(samples):
            raise ValueError("trace session ordinal is outside the sample sequence")
        sample = samples[record.session_ordinal]
        target = str(sample["ground_truth"]["parent_asin"])
        _, behavior = materialize_hidden_fields(sample, products)
        scenario = str(sample["scenario_type"])
        if scenario == "intent_override":
            override = behavior.get("override") or {}
            eligible = record.turn >= int(override.get("turn", 3))
        else:
            eligible = True

        trace = record.trace
        bm25_rank = _rank(trace.bm25_ids, target)
        dense_rank = _rank(trace.dense_ids, target)
        fused_rank = _rank(trace.fused_ids, target)
        output_rank = _rank(record.output_ids, target)
        route_union = set(trace.bm25_ids) | set(trace.dense_ids)
        intersection = set(trace.bm25_ids) & set(trace.dense_ids)
        route_jaccard = (
            len(intersection) / len(route_union) if route_union else 0.0
        )
        rows.append(
            _LabeledTurn(
                session_ordinal=record.session_ordinal,
                scenario_type=scenario,
                turn=record.turn,
                eligible=eligible,
                intent_completeness_proxy=record.intent_completeness_proxy,
                bm25_status=trace.bm25_status,
                dense_status=trace.dense_status,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                fused_rank=fused_rank,
                output_rank=output_rank,
                bm25_count=len(trace.bm25_ids),
                dense_count=len(trace.dense_ids),
                fused_count=len(trace.fused_ids),
                route_jaccard=route_jaccard,
                used_fallback=trace.used_fallback,
                waterfall=(
                    _classify_waterfall(
                        bm25_rank=bm25_rank,
                        dense_rank=dense_rank,
                        fused_rank=fused_rank,
                        output_rank=output_rank,
                        used_fallback=trace.used_fallback,
                    )
                    if eligible
                    else None
                ),
            )
        )
    return rows


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _rank_histogram(ranks: Iterable[int | None]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rank in ranks:
        if rank is None:
            bucket = "absent"
        elif rank <= 10:
            bucket = str(rank)
        elif rank <= 20:
            bucket = "11-20"
        elif rank <= 50:
            bucket = "21-50"
        elif rank <= 100:
            bucket = "51-100"
        elif rank <= 200:
            bucket = "101-200"
        else:
            bucket = ">200"
        counts[bucket] += 1
    order = (
        *(str(rank) for rank in range(1, 11)),
        "11-20",
        "21-50",
        "51-100",
        "101-200",
        ">200",
        "absent",
    )
    return {bucket: counts.get(bucket, 0) for bucket in order}


def _session_metrics(rows: Sequence[_LabeledTurn]) -> dict:
    grouped: dict[int, list[_LabeledTurn]] = defaultdict(list)
    for row in rows:
        if row.eligible:
            grouped[row.session_ordinal].append(row)

    def summarize(route: str, cutoff: int) -> dict:
        observed_hits = 0
        right_censored = 0
        first_turns: list[float] = []
        for session_rows in grouped.values():
            ranks = [getattr(row, f"{route}_rank") for row in session_rows]
            hit_turns = [
                row.turn
                for row, rank in zip(session_rows, ranks)
                if _at(rank, cutoff)
            ]
            if hit_turns:
                observed_hits += 1
                first_turns.append(float(hit_turns[0]))
            elif session_rows[-1].turn < MAX_TURNS:
                right_censored += 1
        denominator = len(grouped)
        return {
            "sessions": denominator,
            "observed_hit_lower_bound": (
                round(observed_hits / denominator, 6) if denominator else 0.0
            ),
            "first_observed_hit_turn_mean": (
                _mean(first_turns) if first_turns else None
            ),
            "right_censored_without_observed_hit": right_censored,
        }

    return {
        "observation_unit": "session_with_at_least_one_eligible_reached_turn",
        "censoring": (
            "Sessions stop on production output Top-10 hits; route values are "
            "observed lower bounds, not counterfactual route metrics."
        ),
        "session_count": len(grouped),
        "bm25_at_10": summarize("bm25", 10),
        "bm25_at_100": summarize("bm25", 100),
        "dense_at_10": summarize("dense", 10),
        "dense_at_100": summarize("dense", 100),
        "fused_at_10": summarize("fused", 10),
        "output_at_10": summarize("output", 10),
    }


def _route_metrics(
    rows: Sequence[_LabeledTurn],
    route: str,
) -> dict:
    ranks = [getattr(row, f"{route}_rank") for row in rows]
    statuses = [getattr(row, f"{route}_status") for row in rows]
    counts = [getattr(row, f"{route}_count") for row in rows]
    metrics = {
        f"recall_at_{cutoff}": _mean(
            float(_at(rank, cutoff)) for rank in ranks
        )
        for cutoff in RANK_CUTOFFS
    }
    metrics["mrr_at_10"] = _mean(
        1.0 / rank if _at(rank, TOP_K) else 0.0 for rank in ranks
    )
    metrics["rank_histogram"] = _rank_histogram(ranks)
    metrics["mean_candidate_count"] = _mean(float(count) for count in counts)
    ok_ranks = [rank for rank, status in zip(ranks, statuses) if status == "ok"]
    metrics["recall_at_10_given_ok"] = _mean(
        float(_at(rank, TOP_K)) for rank in ok_ranks
    )
    return metrics


def _aggregate(rows: Sequence[_LabeledTurn]) -> dict:
    eligible = [row for row in rows if row.eligible]
    bm25_status = Counter(row.bm25_status for row in eligible)
    dense_status = Counter(row.dense_status for row in eligible)
    waterfall = Counter(
        row.waterfall for row in eligible if row.waterfall is not None
    )
    fused_ranks = [row.fused_rank for row in eligible]
    output_ranks = [row.output_rank for row in eligible]
    result = {
        "observation_unit": "eligible_reached_turn",
        "eligible_observations": len(eligible),
        "route_status_counts": {
            "bm25": dict(sorted(bm25_status.items())),
            "dense": dict(sorted(dense_status.items())),
        },
        "bm25": _route_metrics(eligible, "bm25"),
        "dense": _route_metrics(eligible, "dense"),
        "fused": {
            **{
                f"recall_at_{cutoff}": _mean(
                    float(_at(rank, cutoff)) for rank in fused_ranks
                )
                for cutoff in (*RANK_CUTOFFS, 200)
            },
            "union_recall": _mean(
                float(rank is not None) for rank in fused_ranks
            ),
            "mrr_at_10": _mean(
                1.0 / rank if _at(rank, TOP_K) else 0.0
                for rank in fused_ranks
            ),
            "rank_histogram": _rank_histogram(fused_ranks),
            "mean_candidate_count": _mean(
                float(row.fused_count) for row in eligible
            ),
        },
        "output": {
            "recall_at_10": _mean(
                float(_at(rank, TOP_K)) for rank in output_ranks
            ),
            "mrr_at_10": _mean(
                1.0 / rank if _at(rank, TOP_K) else 0.0
                for rank in output_ranks
            ),
            "rank_histogram": _rank_histogram(output_ranks),
        },
        "oracle": {
            f"either_route_at_{cutoff}": _mean(
                float(_at(row.bm25_rank, cutoff) or _at(row.dense_rank, cutoff))
                for row in eligible
            )
            for cutoff in RANK_CUTOFFS
        },
        "mean_route_jaccard": _mean(row.route_jaccard for row in eligible),
        "fallback_rate": _mean(float(row.used_fallback) for row in eligible),
        "waterfall_counts": {
            category: waterfall.get(category, 0)
            for category in WATERFALL_CATEGORIES
        },
    }
    if not math.isclose(
        sum(result["waterfall_counts"].values()),
        len(eligible),
    ):
        raise AssertionError("waterfall categories must partition eligible turns")
    return result


def _grouped(rows: Sequence[_LabeledTurn], key) -> dict[str, dict]:
    groups: dict[str, list[_LabeledTurn]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    return {name: _aggregate(groups[name]) for name in sorted(groups)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_audit(catalog_path: str | Path, dataset_path: str | Path) -> dict:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)

    production_agent = ConversationalSearchAgent(catalog)
    backend = production_agent.retrieval_backend
    if not getattr(backend, "bm25_available", False):
        raise RuntimeError("BM25 is unavailable; refusing a misleading route audit")
    if not getattr(backend, "dense_available", False):
        raise RuntimeError("dense retrieval is unavailable; refusing route audit")
    capture = TraceCaptureRetriever(backend)
    delegate = ConversationalSearchAgent(
        catalog,
        retriever=capture,
        question_policy=CONSERVATIVE_EARLY_OTHER_POLICY,
    )
    audit_agent = AuditAgent(delegate, capture)
    evaluator_result = evaluate(
        audit_agent,
        samples,
        catalog_ids,
        categories,
        products,
    )
    if audit_agent.session_count != len(samples):
        raise AssertionError("audit reset order no longer matches evaluator samples")
    if capture.pending_count:
        raise AssertionError("audit finished with an unconsumed retrieval trace")
    evaluator_sessions = evaluator_result.get("sessions")
    if not isinstance(evaluator_sessions, list):
        raise RuntimeError("evaluator did not return its session summaries")
    _validate_trace_coverage(audit_agent.records, evaluator_sessions)

    rows = _join_labels(audit_agent.records, samples, products)
    by_scenario_turn: dict[str, dict[str, dict]] = {}
    for scenario in sorted({row.scenario_type for row in rows}):
        scenario_rows = [row for row in rows if row.scenario_type == scenario]
        by_scenario_turn[scenario] = _grouped(scenario_rows, lambda row: row.turn)

    repository_root = Path(__file__).resolve().parents[1]
    source_paths = {
        "evaluator/local_evaluator.py": repository_root / "evaluator/local_evaluator.py",
        "conversational_search/intent.py": repository_root / "conversational_search/intent.py",
        "conversational_search/questions.py": repository_root / "conversational_search/questions.py",
        "conversational_search/retrieval.py": repository_root / "conversational_search/retrieval.py",
        "conversational_search/service.py": repository_root / "conversational_search/service.py",
        "preprocessing/encoder.py": repository_root / "preprocessing/encoder.py",
        "scripts/run_retrieval_audit.py": Path(__file__).resolve(),
        "starter/dense.py": repository_root / "starter/dense.py",
    }
    model_manifest = repository_root / "assets/bge-small-en-v1.5-int8/model_manifest.json"
    index_manifest = repository_root / "assets/search-index-bge-small-en-v1.5-v2/manifest.json"
    official = {key: evaluator_result[key] for key in OFFICIAL_METRIC_KEYS}
    ignored_pre_override = sum(not row.eligible for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "audit": "retrieval-loss-waterfall-v1",
        "trajectory": "production-output-top10-stop",
        "censoring": (
            "The official evaluator stops after a sanitized output Top-10 hit; route metrics "
            "describe only observed on-policy turns."
        ),
        "configuration": {
            "question_policy": CONSERVATIVE_EARLY_OTHER_POLICY.name,
            "route_limit": ROUTE_LIMIT,
            "rrf_k": RRF_K,
            "top_k": TOP_K,
            "onnx_threads": 1,
            "external_api_calls": 0,
        },
        "run": {
            "sample_count": len(samples),
            "captured_turns": len(rows),
            "eligible_turns": len(rows) - ignored_pre_override,
            "ignored_pre_override_turns": ignored_pre_override,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "runtime_versions": {
                "numpy": _version("numpy"),
                "onnxruntime": _version("onnxruntime"),
                "sqlite": sqlite3.sqlite_version,
                "tokenizers": _version("tokenizers"),
            },
        },
        "reproducibility": {
            "catalog_sha256": _sha256(catalog),
            "dataset_sha256": _sha256(dataset),
            "model_manifest_sha256": _sha256(model_manifest),
            "index_manifest_sha256": _sha256(index_manifest),
            "source_sha256": {
                name: _sha256(path) for name, path in sorted(source_paths.items())
            },
        },
        "official_metrics": official,
        "overall": _aggregate(rows),
        "session_level": {
            "overall": _session_metrics(rows),
            "by_scenario": {
                scenario: _session_metrics(
                    [row for row in rows if row.scenario_type == scenario]
                )
                for scenario in sorted({row.scenario_type for row in rows})
            },
        },
        "by_scenario": _grouped(rows, lambda row: row.scenario_type),
        "by_turn": _grouped(rows, lambda row: row.turn),
        "intent_completeness_proxy": {
            "status": "predeclared_uncalibrated_proxy",
            "formula": (
                "clip((strong_requirements + 0.5*weak_requirements)/3, 0, 1)"
            ),
            "groups": _grouped(
                rows,
                lambda row: f"{row.intent_completeness_proxy:.6f}",
            ),
        },
        "by_scenario_turn": by_scenario_turn,
    }


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit BM25, dense, and fused retrieval without label leakage"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    protected = {Path(args.catalog).resolve(), Path(args.dataset).resolve()}
    if output in protected:
        raise ValueError("output must not overwrite the catalog or dataset")
    report = run_audit(args.catalog, args.dataset)
    _write_json_exclusive(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "official_metrics": report["official_metrics"],
                "overall": report["overall"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
