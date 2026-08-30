"""Evaluator-only failure attribution for the active conversational agent.

The runtime-facing capture layer never receives a target identifier, scenario
label, sample row, or evaluator outcome.  Targets are joined by ordinal only
after the complete label-free replay, and only aggregate diagnostics are
serialized.  Nothing in ``conversational_search`` or ``starter`` imports this
module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import conversational_search.service as service_module
from conversational_search.decision_policy import PROTECTED_DECISION_POLICY
from conversational_search.intent import (
    ROBUST_INTENT_POLICY,
    IntentState,
    render_dense_query,
    render_lexical_query,
)
from conversational_search.orchestration import (
    EXACT_RANKING_REUSE_ORCHESTRATION_POLICY,
    QueryAction,
)
from conversational_search.profiles import BOUNDED_RESIDUAL_PROFILE_POLICY
from conversational_search.protocol import (
    CandidateReplyStatus,
    ProductProtocolEvidence,
    build_product_protocol_evidence,
    remaining_reply,
)
from conversational_search.questions import (
    CONSERVATIVE_EARLY_OTHER_POLICY,
    QUESTION_TEXT,
)
from conversational_search.ranking import STAGE_A_RANKING_POLICY
from conversational_search.retrieval import (
    DISABLED_REQUIREMENT_PROBE_POLICY,
    HybridRetriever,
    RetrievalResult,
)
from conversational_search.service import (
    ConversationalSearchAgent,
    _load_dense_runtime,
)
from conversational_search.slates import INTENT_EPOCH_NOVELTY_SLATE_POLICY
from conversational_search.strategy import COMPLETENESS_ADAPTIVE_RRF_POLICY
from evaluator.local_evaluator import (
    ALLOWED_ATTRIBUTES,
    MAX_TURNS,
    TOP_K,
    catalog_index,
    classify_constraint,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)


SCHEMA_VERSION = 1
FAILURE_CATEGORIES = (
    "intent_override_handled_incorrectly",
    "protocol_prediction_mismatch",
    "candidate_pool_became_too_narrow",
    "target_absent_from_retrieved_candidates",
    "target_retrieved_but_ranked_below_top_10",
    "target_exposed_prematurely_at_low_reciprocal_rank",
    "target_in_top_10_but_ordered_poorly",
    "repeated_or_uninformative_question",
    "weak_question_selected",
    "unnecessary_retrieval_or_dense_model_call",
)
PRIMARY_PRECEDENCE = (
    "intent_override_handled_incorrectly",
    "candidate_pool_became_too_narrow",
    "target_absent_from_retrieved_candidates",
    "target_retrieved_but_ranked_below_top_10",
    "target_exposed_prematurely_at_low_reciprocal_rank",
    "target_in_top_10_but_ordered_poorly",
    "repeated_or_uninformative_question",
    "weak_question_selected",
    "unnecessary_retrieval_or_dense_model_call",
    "protocol_prediction_mismatch",
)
PRIMARY_CATEGORIES = (*FAILURE_CATEGORIES, "no_loss", "unclassified")
QUESTION_ACTIONS = tuple(QUESTION_TEXT)
_STRONG_OVERRIDE_SOURCE = "override"
_STALE_OVERRIDE_SOURCE = "initial_tentative"


@dataclass(frozen=True, slots=True)
class LabelFreeTurnTrace:
    """One turn captured without evaluator labels, messages, or query text."""

    dialogue_ordinal: int
    turn: int
    output_ids: tuple[str, ...]
    ask_attribute: str | None
    response_error: bool
    latency_ms: float
    decision_action: str
    decision_reason: str
    retrieval_executed: bool
    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    bm25_status: str
    dense_status: str
    retrieval_fallback: bool
    rerank_executed: bool
    reranked_ids: tuple[str, ...]
    pre_slate_executed: bool
    pre_slate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    slate_status: str
    intent_version: int
    requirement_sources: tuple[str, ...]
    asked_attributes: tuple[str, ...]
    no_preference: tuple[str, ...]
    retrieval_dependency_digest: str


@dataclass(frozen=True, slots=True)
class ScoreLoss:
    coverage: float
    ranking: float
    efficiency: float

    @property
    def total(self) -> float:
        return self.coverage + self.ranking + self.efficiency


@dataclass(frozen=True, slots=True)
class SessionDiagnostic:
    """Target-aware private row.  It is aggregated and then discarded."""

    scenario: str
    hit: bool
    first_hit_turn: int | None
    first_hit_rank: int | None
    first_any_exposure_turn: int | None
    first_any_exposure_rank: int | None
    flags: frozenset[str]
    primary: str
    loss: ScoreLoss
    reached_checkpoint_rows: tuple[dict[str, object], ...]
    weak_questions: int
    repeated_or_uninformative_questions: int
    protocol_mismatches: int
    search_calls: int
    reuse_calls: int
    dense_calls: int
    unnecessary_search_calls: int
    unnecessary_dense_calls: int
    pre_override_exposures: int


class _TurnCapture:
    """Single-threaded label-free event collector used by transparent wrappers."""

    def __init__(self) -> None:
        self._active: tuple[int, int] | None = None
        self._retrievals: list[RetrievalResult] = []
        self._rerankings: list[tuple[str, ...]] = []
        self._pre_slates: list[tuple[str, ...]] = []
        self._selected: list[tuple[str, ...]] = []
        self._slate_statuses: list[str] = []
        self._decisions: list[tuple[str, str]] = []

    def begin(self, dialogue_ordinal: int, turn: int) -> None:
        if self._active is not None:
            raise RuntimeError("a diagnostic turn is already active")
        self._active = (dialogue_ordinal, turn)
        self._retrievals.clear()
        self._rerankings.clear()
        self._pre_slates.clear()
        self._selected.clear()
        self._slate_statuses.clear()
        self._decisions.clear()

    def _require_active(self) -> None:
        if self._active is None:
            raise RuntimeError("diagnostic event occurred outside an active turn")

    def record_retrieval(self, result: RetrievalResult) -> None:
        self._require_active()
        if not isinstance(result, RetrievalResult):
            raise TypeError("captured retrieval must be RetrievalResult")
        self._retrievals.append(result)

    def record_reranking(self, ranked_ids: Sequence[str]) -> None:
        self._require_active()
        self._rerankings.append(tuple(ranked_ids))

    def record_pre_slate(self, ranked_ids: Sequence[str]) -> None:
        self._require_active()
        self._pre_slates.append(tuple(ranked_ids))

    def record_selected(
        self,
        selected_ids: Sequence[str],
        status: object = "unknown",
    ) -> None:
        self._require_active()
        self._selected.append(tuple(selected_ids))
        self._slate_statuses.append(
            str(getattr(status, "value", status))
        )

    def record_decision(self, action: object, reason: object) -> None:
        self._require_active()
        self._decisions.append(
            (str(getattr(action, "value", action)), str(reason))
        )

    def finish(
        self,
        response: dict[str, object],
        state: IntentState,
        *,
        response_error: bool,
        latency_ms: float,
    ) -> LabelFreeTurnTrace:
        if self._active is None:
            raise RuntimeError("cannot finish an inactive diagnostic turn")
        dialogue_ordinal, turn = self._active
        try:
            if len(self._retrievals) > 1:
                raise RuntimeError("a turn executed more than one retrieval")
            if len(self._decisions) != 1:
                raise RuntimeError("a turn must execute exactly one orchestration decision")
            action, reason = self._decisions[0]
            retrieval_executed = bool(self._retrievals)
            if (action == QueryAction.SEARCH.value) != retrieval_executed:
                raise RuntimeError("retrieval execution disagrees with orchestration")
            retrieval = self._retrievals[0] if self._retrievals else None
            trace = retrieval.trace if retrieval is not None else None
            output_ids = tuple(
                str(item.get("parent_asin"))
                for item in response.get("recommendations", [])
                if isinstance(item, dict)
                and isinstance(item.get("parent_asin"), str)
            )
            matching_selected = tuple(
                (index, selected)
                for index, selected in enumerate(self._selected)
                if selected == output_ids
            )
            if self._selected and not matching_selected:
                raise RuntimeError("captured slate does not match response output")
            selected_index = matching_selected[0][0] if matching_selected else None
            selected = output_ids if matching_selected else ()
            digest = hashlib.sha256(
                (
                    render_dense_query(state)
                    + "\0"
                    + render_lexical_query(state)
                ).encode("utf-8")
            ).hexdigest()
            return LabelFreeTurnTrace(
                dialogue_ordinal=dialogue_ordinal,
                turn=turn,
                output_ids=output_ids,
                ask_attribute=(
                    response.get("ask_attribute")
                    if isinstance(response.get("ask_attribute"), str)
                    else None
                ),
                response_error=response_error,
                latency_ms=latency_ms,
                decision_action=action,
                decision_reason=reason,
                retrieval_executed=retrieval_executed,
                bm25_ids=() if trace is None else trace.bm25_ids,
                dense_ids=() if trace is None else trace.dense_ids,
                fused_ids=() if trace is None else trace.fused_ids,
                bm25_status="not_executed" if trace is None else trace.bm25_status,
                dense_status="not_executed" if trace is None else trace.dense_status,
                retrieval_fallback=False if trace is None else trace.used_fallback,
                rerank_executed=bool(self._rerankings),
                reranked_ids=(self._rerankings[-1] if self._rerankings else ()),
                pre_slate_executed=bool(self._pre_slates),
                pre_slate_ids=(self._pre_slates[-1] if self._pre_slates else ()),
                selected_ids=selected,
                slate_status=(
                    self._slate_statuses[selected_index]
                    if selected_index is not None
                    else "not_executed"
                ),
                intent_version=state.intent_version,
                requirement_sources=tuple(
                    requirement.source for requirement in state.requirements
                ),
                asked_attributes=state.asked_attributes,
                no_preference=tuple(sorted(state.no_preference)),
                retrieval_dependency_digest=digest,
            )
        finally:
            self._active = None


class _CaptureRetriever:
    """Forward every capability while observing only completed retrievals."""

    def __init__(self, backend: object, capture: _TurnCapture) -> None:
        self._backend = backend
        self._capture = capture

    def __getattr__(self, name: str) -> object:
        return getattr(self._backend, name)

    def search_with_trace(self, *args: object, **kwargs: object) -> RetrievalResult:
        result = self._backend.search_with_trace(*args, **kwargs)
        if not isinstance(result, RetrievalResult):
            raise TypeError("search_with_trace must return RetrievalResult")
        self._capture.record_retrieval(result)
        return result

    def search(self, *args: object, **kwargs: object) -> list[str]:
        result = self.search_with_trace(*args, **kwargs)
        return list(result.recommendations)


class _CaptureOrchestrator:
    def __init__(self, delegate: object, capture: _TurnCapture) -> None:
        self._delegate = delegate
        self._capture = capture

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    def decide(self, *args: object, **kwargs: object) -> object:
        decision = self._delegate.decide(*args, **kwargs)
        self._capture.record_decision(decision.action, decision.reason)
        return decision


class _AuditAgent:
    """API wrapper that emits label-free turn traces in evaluator order."""

    def __init__(self, delegate: ConversationalSearchAgent, capture: _TurnCapture) -> None:
        self._delegate = delegate
        self._capture = capture
        self._ordinals: dict[str, int] = {}
        self.records: list[LabelFreeTurnTrace] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        if session_id in self._ordinals:
            raise RuntimeError("diagnostic replay reused a dialogue identifier")
        self._ordinals[session_id] = len(self._ordinals)
        self._delegate.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        ordinal = self._ordinals[session_id]
        self._capture.begin(ordinal, turn)
        started = time.perf_counter()
        response_error = False
        try:
            response = self._delegate.respond(session_id, user_message, turn, top_k)
            if not isinstance(response, dict) or not isinstance(
                response.get("message"), str
            ):
                raise TypeError("agent response is invalid")
        except Exception:
            response_error = True
            response = {
                "message": "",
                "ask_attribute": None,
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        latency_ms = (time.perf_counter() - started) * 1000.0
        state = self._delegate.session_state(session_id)
        record = self._capture.finish(
            response,
            state,
            response_error=response_error,
            latency_ms=latency_ms,
        )
        self.records.append(record)
        return response


@contextmanager
def _capture_runtime(capture: _TurnCapture) -> Iterable[None]:
    """Temporarily wrap service-local ranking/slate symbols and restore them."""

    original_stage_a = service_module.rerank_stage_a
    original_profile = service_module.rerank_stage_a_with_profile
    original_rescue = service_module.rerank_stage_a_with_profile_and_bm25_rescue
    original_redundancy = (
        service_module.rerank_stage_a_with_profile_and_route_redundancy
    )
    original_epoch = service_module.select_slate_with_intent_epoch_novelty
    original_slate = service_module.select_slate

    def stage_a(*args: object, **kwargs: object) -> object:
        result = original_stage_a(*args, **kwargs)
        capture.record_reranking(result.ranked_ids)
        return result

    def profile(*args: object, **kwargs: object) -> object:
        result = original_profile(*args, **kwargs)
        capture.record_reranking(result.ranking.ranked_ids)
        return result

    def rescue(*args: object, **kwargs: object) -> object:
        result = original_rescue(*args, **kwargs)
        capture.record_reranking(result.ranking.ranked_ids)
        return result

    def redundancy(*args: object, **kwargs: object) -> object:
        result = original_redundancy(*args, **kwargs)
        capture.record_reranking(result.ranking.ranked_ids)
        return result

    def epoch(
        prior_state: object,
        signature: tuple[object, ...],
        ranked_ids: Sequence[str],
        limit: int,
    ) -> object:
        capture.record_pre_slate(ranked_ids)
        result = original_epoch(prior_state, signature, ranked_ids, limit)
        capture.record_selected(result.selection.selected_ids, result.status)
        return result

    def slate(
        policy: object,
        prior_state: object,
        signature: tuple[object, ...],
        ranked_ids: Sequence[str],
        limit: int,
    ) -> object:
        capture.record_pre_slate(ranked_ids)
        result = original_slate(policy, prior_state, signature, ranked_ids, limit)
        capture.record_selected(result.selected_ids, "fallback")
        return result

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(service_module, "rerank_stage_a", stage_a))
        stack.enter_context(
            mock.patch.object(service_module, "rerank_stage_a_with_profile", profile)
        )
        stack.enter_context(
            mock.patch.object(
                service_module,
                "rerank_stage_a_with_profile_and_bm25_rescue",
                rescue,
            )
        )
        stack.enter_context(
            mock.patch.object(
                service_module,
                "rerank_stage_a_with_profile_and_route_redundancy",
                redundancy,
            )
        )
        stack.enter_context(
            mock.patch.object(
                service_module,
                "select_slate_with_intent_epoch_novelty",
                epoch,
            )
        )
        stack.enter_context(mock.patch.object(service_module, "select_slate", slate))
        yield


def _rank(items: Sequence[str], target: str) -> int | None:
    try:
        return items.index(target) + 1
    except ValueError:
        return None


def _score_loss(hit_turn: int | None, hit_rank: int | None) -> ScoreLoss:
    if hit_turn is None or hit_rank is None:
        return ScoreLoss(0.5, 0.3, 0.2)
    return ScoreLoss(
        0.0,
        0.3 * (1.0 - 1.0 / hit_rank),
        0.02 * (hit_turn - 1),
    )


def _hit_utility(turn: int, rank: int) -> float:
    return 0.5 + 0.3 / rank + 0.02 * (11 - turn)


def _choose_primary(flags: frozenset[str], loss: ScoreLoss) -> str:
    if math.isclose(loss.total, 0.0, abs_tol=1e-15):
        return "no_loss"
    for category in PRIMARY_PRECEDENCE:
        if category in flags:
            return category
    return "unclassified"


def _reply_partition_entropy(
    ids: Sequence[str],
    evidence_cache: dict[str, ProductProtocolEvidence],
    products: dict[str, dict],
    attribute: str,
    disclosed: Iterable[str],
    *,
    boundary_pending: bool,
    limit: int = 50,
) -> float:
    pool = tuple(ids[:limit])
    if not pool:
        return 0.0
    raw_weights = tuple(1.0 / rank for rank in range(1, len(pool) + 1))
    denominator = sum(raw_weights)
    partitions: defaultdict[tuple[object, ...], float] = defaultdict(float)
    for parent_asin, raw_weight in zip(pool, raw_weights):
        evidence = evidence_cache.get(parent_asin)
        if evidence is None:
            evidence = build_product_protocol_evidence(
                products[parent_asin],
                include_text=False,
            )
            evidence_cache[parent_asin] = evidence
        reply = remaining_reply(
            evidence.card,
            attribute,
            disclosed,
            boundary_pending=boundary_pending,
        )
        key = (reply.status.value, reply.attribute, *reply.values)
        partitions[key] += raw_weight / denominator
    return -sum(probability * math.log(probability) for probability in partitions.values())


def _validate_trace_coverage(
    records: Sequence[LabelFreeTurnTrace],
    dialogue_count: int,
) -> None:
    if len(records) != dialogue_count * MAX_TURNS:
        raise RuntimeError("shadow replay did not capture ten turns per dialogue")
    grouped: defaultdict[int, list[int]] = defaultdict(list)
    for record in records:
        grouped[record.dialogue_ordinal].append(record.turn)
    if set(grouped) != set(range(dialogue_count)):
        raise RuntimeError("trace ordinals do not cover every dialogue")
    for ordinal in range(dialogue_count):
        if grouped[ordinal] != list(range(1, MAX_TURNS + 1)):
            raise RuntimeError("trace turns are not contiguous")


def _shadow_replay(
    agent: _AuditAgent,
    samples: Sequence[dict],
    catalog_ids: set[str],
    categories: dict[str, list[str]],
    products: dict[str, dict],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Mirror the official simulator but continue after a hit for diagnostics."""

    outcomes: list[dict[str, object]] = []
    prompt_tokens = 0
    completion_tokens = 0
    for ordinal, sample in enumerate(samples):
        session_id = f"phase1_shadow_{ordinal:04d}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        first_hit_turn: int | None = None
        first_hit_rank: int | None = None
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            usage = response.get("usage")
            if isinstance(usage, dict):
                prompt = usage.get("prompt_tokens")
                completion = usage.get("completion_tokens")
                if isinstance(prompt, int) and prompt >= 0:
                    prompt_tokens += prompt
                if isinstance(completion, int) and completion >= 0:
                    completion_tokens += completion
            ranked = normalize_recommendations(
                response.get("recommendations"),
                catalog_ids,
            )
            if (
                first_hit_turn is None
                and override_applied
                and target in ranked
            ):
                first_hit_turn = turn
                first_hit_rank = ranked.index(target) + 1
            if turn == MAX_TURNS:
                continue
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get(
                        "message",
                        "Actually, please ignore my earlier preference.",
                    )
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )
        outcomes.append(
            {
                "scenario_type": str(sample["scenario_type"]),
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": first_hit_rank,
                "reciprocal_rank": (
                    0.0 if first_hit_rank is None else 1.0 / first_hit_rank
                ),
            }
        )
    metrics = metric_summary(outcomes)
    efficiency = max(
        0.0,
        min(1.0, (11.0 - float(metrics["mttc"])) / 10.0),
    )
    score = (
        0.50 * float(metrics["hit_rate_at_10"])
        + 0.30 * float(metrics["mrr"])
        + 0.20 * efficiency
    )
    return (
        {
            **metrics,
            "efficiency": round(efficiency, 6),
            "recommended_technical_score": round(score, 6),
            "reported_token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        outcomes,
    )


def _diagnose_session(
    ordinal: int,
    sample: dict,
    product: dict,
    categories: list[str],
    records: Sequence[LabelFreeTurnTrace],
    outcome: dict[str, object],
    products: dict[str, dict],
    evidence_cache: dict[str, ProductProtocolEvidence],
) -> SessionDiagnostic:
    target = str(sample["ground_truth"]["parent_asin"])
    scenario = str(sample["scenario_type"])
    card, behavior = materialize_hidden_fields(sample, products)
    override = behavior.get("override") or {}
    override_turn = int(override.get("turn", 3)) if scenario == "intent_override" else 1
    hit_turn = outcome.get("first_hit_turn")
    hit_rank = outcome.get("best_rank")
    if hit_turn is not None:
        hit_turn = int(hit_turn)
    if hit_rank is not None:
        hit_rank = int(hit_rank)
    loss = _score_loss(hit_turn, hit_rank)

    rows: list[dict[str, object]] = []
    first_any_turn: int | None = None
    first_any_rank: int | None = None
    pre_override_exposures = 0
    for record in records:
        eligible = record.turn >= override_turn
        exposure_rank = _rank(record.output_ids, target)
        if exposure_rank is not None and first_any_turn is None:
            first_any_turn = record.turn
            first_any_rank = exposure_rank
        if exposure_rank is not None and not eligible:
            pre_override_exposures += 1
        rows.append(
            {
                "turn": record.turn,
                "eligible": eligible,
                "retrieval_executed": record.retrieval_executed,
                "retrieval_rank": _rank(record.fused_ids, target),
                "bm25_rank": _rank(record.bm25_ids, target),
                "dense_rank": _rank(record.dense_ids, target),
                "rerank_executed": record.rerank_executed,
                "rerank_rank": _rank(record.reranked_ids, target),
                "pre_slate_executed": record.pre_slate_executed,
                "pre_slate_rank": _rank(record.pre_slate_ids, target),
                "exposure_rank": exposure_rank,
            }
        )

    reached = [
        row
        for row in rows
        if bool(row["eligible"])
        and (hit_turn is None or int(row["turn"]) <= hit_turn)
    ]
    present_in_pool = any(row["pre_slate_rank"] is not None for row in reached)
    ever_top10 = any(
        isinstance(row["pre_slate_rank"], int)
        and int(row["pre_slate_rank"]) <= TOP_K
        for row in reached
    )
    flags: set[str] = set()
    if not present_in_pool:
        flags.add("target_absent_from_retrieved_candidates")
    elif not ever_top10:
        flags.add("target_retrieved_but_ranked_below_top_10")
    if hit_rank is not None and hit_rank > 1:
        flags.add("target_in_top_10_but_ordered_poorly")

    seen_in_pool = False
    for row in reached:
        if row["pre_slate_rank"] is not None:
            seen_in_pool = True
        elif seen_in_pool:
            flags.add("candidate_pool_became_too_narrow")
        if row["retrieval_rank"] is not None and row["pre_slate_rank"] is None:
            flags.add("candidate_pool_became_too_narrow")

    if scenario == "intent_override":
        override_record = records[override_turn - 1]
        prior_record = records[override_turn - 2] if override_turn > 1 else None
        override_ok = (
            _STRONG_OVERRIDE_SOURCE in override_record.requirement_sources
            and _STALE_OVERRIDE_SOURCE not in override_record.requirement_sources
            and (
                prior_record is None
                or override_record.intent_version > prior_record.intent_version
            )
        )
        if not override_ok:
            flags.add("intent_override_handled_incorrectly")

    disclosed: set[str] = set()
    effective = {**sample, "intent_card": card, "behavior": behavior}
    initial_message(effective, coarse_category(categories), disclosed)
    boundary_used = False
    asked: set[str] = set()
    weak_questions = 0
    repeated_or_uninformative = 0
    protocol_mismatches = 0
    target_protocol = build_product_protocol_evidence(product, include_text=False)
    for record in records:
        if hit_turn is not None and record.turn >= hit_turn:
            break
        if record.turn == MAX_TURNS:
            break
        scheduled_override = (
            scenario == "intent_override"
            and record.turn + 1 == override_turn
        )
        boundary_pending = scenario == "boundary" and not boundary_used
        question = record.ask_attribute
        unresolved = tuple(
            attribute
            for attribute in QUESTION_ACTIONS
            if attribute not in asked
            and attribute not in record.no_preference
        )
        pool = record.pre_slate_ids
        entropies = {
            attribute: _reply_partition_entropy(
                pool,
                evidence_cache,
                products,
                attribute,
                disclosed,
                boundary_pending=boundary_pending,
            )
            for attribute in unresolved
        }
        best_entropy = max(entropies.values(), default=0.0)
        chosen_entropy = entropies.get(question, 0.0)
        repeated = question is not None and question in asked
        if repeated:
            repeated_or_uninformative += 1
        if (
            not scheduled_override
            and not boundary_pending
            and question is not None
            and best_entropy > chosen_entropy + 1e-12
        ):
            weak_questions += 1
        if question is not None:
            asked.add(question)

        if scheduled_override:
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            continue

        predicted = remaining_reply(
            target_protocol.card,
            question,
            disclosed,
            boundary_pending=boundary_pending,
        )
        actual_text, next_boundary_used = customer_reply(
            effective,
            question,
            disclosed,
            boundary_used,
        )
        if predicted.reply_text != actual_text:
            protocol_mismatches += 1
        if (
            predicted.status
            in {CandidateReplyStatus.NO_ADDITIONAL, CandidateReplyStatus.NEED_ATTRIBUTE}
            and best_entropy > 1e-12
            and not boundary_pending
        ):
            repeated_or_uninformative += 1
        boundary_used = next_boundary_used

    if weak_questions:
        flags.add("weak_question_selected")
    if repeated_or_uninformative:
        flags.add("repeated_or_uninformative_question")
    if protocol_mismatches:
        flags.add("protocol_prediction_mismatch")

    if hit_turn is not None and hit_rank is not None and hit_rank > 1:
        immediate = _hit_utility(hit_turn, hit_rank)
        later_better = any(
            bool(row["eligible"])
            and int(row["turn"]) > hit_turn
            and isinstance(row["exposure_rank"], int)
            and _hit_utility(int(row["turn"]), int(row["exposure_rank"]))
            > immediate + 1e-12
            for row in rows
        )
        if later_better:
            flags.add("target_exposed_prematurely_at_low_reciprocal_rank")

    search_calls = sum(record.decision_action == QueryAction.SEARCH.value for record in records)
    reuse_calls = sum(record.decision_action == QueryAction.REUSE.value for record in records)
    dense_calls = sum(
        record.retrieval_executed and record.dense_status in {"ok", "empty", "error"}
        for record in records
    )
    unnecessary_search = 0
    unnecessary_dense = 0
    prior: LabelFreeTurnTrace | None = None
    for record, row in zip(records, rows):
        if hit_turn is not None and record.turn > hit_turn:
            break
        if (
            prior is not None
            and record.decision_action == QueryAction.SEARCH.value
            and record.retrieval_dependency_digest
            == prior.retrieval_dependency_digest
            and record.pre_slate_ids == prior.pre_slate_ids
        ):
            unnecessary_search += 1
        bm25_rank = row["bm25_rank"]
        pre_slate_rank = row["pre_slate_rank"]
        if (
            record.retrieval_executed
            and record.dense_status == "ok"
            and isinstance(bm25_rank, int)
            and bm25_rank <= TOP_K
            and (
                pre_slate_rank is None
                or int(pre_slate_rank) >= bm25_rank
            )
        ):
            unnecessary_dense += 1
        prior = record
    if unnecessary_search or unnecessary_dense:
        flags.add("unnecessary_retrieval_or_dense_model_call")

    frozen_flags = frozenset(flags)
    return SessionDiagnostic(
        scenario=scenario,
        hit=hit_turn is not None,
        first_hit_turn=hit_turn,
        first_hit_rank=hit_rank,
        first_any_exposure_turn=first_any_turn,
        first_any_exposure_rank=first_any_rank,
        flags=frozen_flags,
        primary=_choose_primary(frozen_flags, loss),
        loss=loss,
        reached_checkpoint_rows=tuple(reached),
        weak_questions=weak_questions,
        repeated_or_uninformative_questions=repeated_or_uninformative,
        protocol_mismatches=protocol_mismatches,
        search_calls=search_calls,
        reuse_calls=reuse_calls,
        dense_calls=dense_calls,
        unnecessary_search_calls=unnecessary_search,
        unnecessary_dense_calls=unnecessary_dense,
        pre_override_exposures=pre_override_exposures,
    )


def _rank_histogram(
    observations: Iterable[tuple[bool, int | None]],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for executed, rank in observations:
        if not executed:
            bucket = "not_executed"
        elif rank is None:
            bucket = "absent"
        elif rank <= 10:
            bucket = str(rank)
        elif rank <= 20:
            bucket = "11-20"
        elif rank <= 50:
            bucket = "21-50"
        elif rank <= 100:
            bucket = "51-100"
        else:
            bucket = "101+"
        counts[bucket] += 1
    order = (
        *(str(rank) for rank in range(1, 11)),
        "11-20",
        "21-50",
        "51-100",
        "101+",
        "absent",
        "not_executed",
    )
    return {key: counts.get(key, 0) for key in order}


def _loss_payload(rows: Sequence[SessionDiagnostic]) -> dict[str, float]:
    return {
        "coverage": round(sum(row.loss.coverage for row in rows), 9),
        "ranking": round(sum(row.loss.ranking for row in rows), 9),
        "efficiency": round(sum(row.loss.efficiency for row in rows), 9),
        "total": round(sum(row.loss.total for row in rows), 9),
    }


def _aggregate_report(
    diagnostics: Sequence[SessionDiagnostic],
    official_metrics: dict[str, object],
    records: Sequence[LabelFreeTurnTrace],
    *,
    wall_seconds: float,
) -> dict[str, object]:
    primary_groups = {
        category: [row for row in diagnostics if row.primary == category]
        for category in PRIMARY_CATEGORIES
    }
    flag_groups = {
        category: [row for row in diagnostics if category in row.flags]
        for category in FAILURE_CATEGORIES
    }
    checkpoint_rows = [
        checkpoint
        for row in diagnostics
        for checkpoint in row.reached_checkpoint_rows
    ]
    total_loss = sum(row.loss.total for row in diagnostics)
    reconstructed_score = 1.0 - total_loss / len(diagnostics)
    primary_total = sum(
        sum(row.loss.total for row in group)
        for group in primary_groups.values()
    )
    latency = [record.latency_ms for record in records]
    warm = latency[1:]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "phase1-active-agent-failure-attribution-v1",
        "privacy": {
            "runtime_received_evaluation_labels": False,
            "labels_joined_after_label_free_replay": True,
            "individual_rows_persisted": False,
            "identifiers_messages_queries_profiles_persisted": False,
            "failure_rows_aggregated_before_serialization": True,
        },
        "capture_contract": {
            "shadow_turns_per_dialogue": MAX_TURNS,
            "official_stop_metric_reconstructed_from_first_eligible_exposure": True,
            "retrieval_rank_is_null_on_cache_reuse": True,
            "rank_checkpoints": (
                "fresh_fused_retrieval",
                "active_reranker_output",
                "final_order_before_slate",
                "exposed_output",
            ),
            "issue_flags_are_nonexclusive": True,
            "associated_flag_loss_is_nonadditive": True,
            "primary_failure_is_exclusive": True,
            "primary_precedence": PRIMARY_PRECEDENCE,
        },
        "official_metrics": official_metrics,
        "trace_coverage": {
            "evaluated_dialogues": len(diagnostics),
            "captured_shadow_turns": len(records),
            "official_reached_eligible_turns": len(checkpoint_rows),
            "response_errors": sum(record.response_error for record in records),
        },
        "score_gap": {
            "component_loss_sums": _loss_payload(diagnostics),
            "mean_total_loss": round(total_loss / len(diagnostics), 9),
            "reconstructed_technical_score": round(reconstructed_score, 9),
            "reported_technical_score": official_metrics[
                "recommended_technical_score"
            ],
        },
        "primary_failure": {
            category: {
                "count": len(group),
                "attributed_score_loss": _loss_payload(group),
            }
            for category, group in primary_groups.items()
        },
        "issue_flags": {
            category: {
                "count": len(group),
                "associated_score_loss": _loss_payload(group),
            }
            for category, group in flag_groups.items()
        },
        "checkpoint_rank_histograms": {
            "immediately_after_retrieval": _rank_histogram(
                (
                    bool(row["retrieval_executed"]),
                    row["retrieval_rank"] if isinstance(row["retrieval_rank"], int) else None,
                )
                for row in checkpoint_rows
            ),
            "after_reranking": _rank_histogram(
                (
                    bool(row["rerank_executed"]),
                    row["rerank_rank"] if isinstance(row["rerank_rank"], int) else None,
                )
                for row in checkpoint_rows
            ),
            "before_slate_selection": _rank_histogram(
                (
                    bool(row["pre_slate_executed"]),
                    row["pre_slate_rank"] if isinstance(row["pre_slate_rank"], int) else None,
                )
                for row in checkpoint_rows
            ),
            "exposed_output": _rank_histogram(
                (
                    True,
                    row["exposure_rank"] if isinstance(row["exposure_rank"], int) else None,
                )
                for row in checkpoint_rows
            ),
        },
        "question_diagnostics": {
            "weak_selected": sum(row.weak_questions for row in diagnostics),
            "repeated_or_uninformative": sum(
                row.repeated_or_uninformative_questions for row in diagnostics
            ),
            "pre_override_target_exposures": sum(
                row.pre_override_exposures for row in diagnostics
            ),
        },
        "protocol_diagnostics": {
            "predicted_reply_mismatches": sum(
                row.protocol_mismatches for row in diagnostics
            ),
        },
        "route_diagnostics": {
            "search_calls": sum(row.search_calls for row in diagnostics),
            "reuse_calls": sum(row.reuse_calls for row in diagnostics),
            "dense_calls": sum(row.dense_calls for row in diagnostics),
            "unnecessary_search_calls": sum(
                row.unnecessary_search_calls for row in diagnostics
            ),
            "dense_calls_unnecessary_for_observed_outcome": sum(
                row.unnecessary_dense_calls for row in diagnostics
            ),
        },
        "runtime": {
            "single_process": True,
            "single_numerical_thread": True,
            "wall_seconds": round(wall_seconds, 6),
            "respond_latency_ms": {
                "count": len(latency),
                "p50": round(statistics.median(latency), 6),
                "warm_p95": round(
                    sorted(warm)[max(0, math.ceil(0.95 * len(warm)) - 1)],
                    6,
                ) if warm else 0.0,
                "max": round(max(latency, default=0.0), 6),
            },
        },
        "by_scenario": {},
        "invariants": {
            "primary_counts_equal_dialogue_count": sum(
                len(group) for group in primary_groups.values()
            ) == len(diagnostics),
            "primary_loss_reconstructs_total_loss": math.isclose(
                primary_total,
                total_loss,
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
            "score_reconstructs_reported_metric": math.isclose(
                round(reconstructed_score, 6),
                float(official_metrics["recommended_technical_score"]),
                rel_tol=0.0,
                abs_tol=2e-6,
            ),
            "all_failure_categories_reported": set(flag_groups)
            == set(FAILURE_CATEGORIES),
            "zero_response_errors": not any(
                record.response_error for record in records
            ),
        },
    }
    scenarios = sorted({row.scenario for row in diagnostics})
    result["by_scenario"] = {
        scenario: {
            "count": len(group),
            "hits": sum(row.hit for row in group),
            "score_loss": _loss_payload(group),
            "primary_counts": {
                category: sum(row.primary == category for row in group)
                for category in PRIMARY_CATEGORIES
            },
        }
        for scenario in scenarios
        for group in ([row for row in diagnostics if row.scenario == scenario],)
    }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_public_report(report: dict[str, object]) -> None:
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    forbidden_keys = (
        '"sessions"',
        '"sample_id"',
        '"target_id"',
        '"parent_asin"',
        '"user_message"',
        '"dense_query"',
        '"lexical_query"',
        '"user_profile"',
        '"recommendations"',
    )
    if any(key in encoded for key in forbidden_keys):
        raise ValueError("diagnostic report contains a forbidden raw-data field")
    invariants = report.get("invariants")
    if not isinstance(invariants, dict) or not all(invariants.values()):
        raise ValueError("diagnostic invariants did not all pass")


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite diagnostic: {path}")
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


def run_diagnostics(
    catalog_path: str | Path,
    dataset_path: str | Path,
    *,
    model_assets: str | Path,
    dense_index_path: str | Path,
    limit: int | None = None,
) -> dict[str, object]:
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    samples = load_jsonl(dataset)
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer or None")
        samples = samples[:limit]
    if not samples:
        raise ValueError("diagnostic dataset must not be empty")
    catalog_ids, categories, products = catalog_index(catalog)
    encoder, dense_index = _load_dense_runtime(
        catalog,
        model_assets,
        dense_index_path,
    )
    backend = HybridRetriever(
        catalog,
        encoder=encoder,
        dense_index=dense_index,
        protocol_evidence=False,
    )
    capture = _TurnCapture()
    retriever = _CaptureRetriever(backend, capture)
    delegate = ConversationalSearchAgent(
        catalog,
        retriever=retriever,
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
    delegate._orchestrator = _CaptureOrchestrator(  # type: ignore[attr-defined]
        delegate._orchestrator,  # type: ignore[attr-defined]
        capture,
    )
    audit = _AuditAgent(delegate, capture)
    started = time.perf_counter()
    with _capture_runtime(capture):
        official_metrics, outcomes = _shadow_replay(
            audit,
            samples,
            catalog_ids,
            categories,
            products,
        )
    wall_seconds = time.perf_counter() - started
    _validate_trace_coverage(audit.records, len(samples))
    by_ordinal: defaultdict[int, list[LabelFreeTurnTrace]] = defaultdict(list)
    for record in audit.records:
        by_ordinal[record.dialogue_ordinal].append(record)
    evidence_cache: dict[str, ProductProtocolEvidence] = {}
    diagnostics = tuple(
        _diagnose_session(
            ordinal,
            sample,
            products[str(sample["ground_truth"]["parent_asin"])],
            categories.get(str(sample["ground_truth"]["parent_asin"]), []),
            by_ordinal[ordinal],
            outcomes[ordinal],
            products,
            evidence_cache,
        )
        for ordinal, sample in enumerate(samples)
    )
    report = _aggregate_report(
        diagnostics,
        official_metrics,
        audit.records,
        wall_seconds=wall_seconds,
    )
    report["reproducibility"] = {
        "catalog_sha256": _sha256(catalog),
        "dataset_sha256": _sha256(dataset),
        "source_sha256": _sha256(Path(__file__).resolve()),
        "active_policy": {
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
    }
    _validate_public_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run aggregate-only Phase 1 active-agent failure diagnostics"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--model-assets",
        default="assets/bge-small-en-v1.5-int8",
    )
    parser.add_argument(
        "--dense-index",
        default="assets/search-index-bge-small-en-v1.5-v2",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        help="evaluate only the first N dialogues for a smoke run",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    protected = {Path(args.catalog).resolve(), Path(args.dataset).resolve()}
    if output in protected:
        raise ValueError("output must not overwrite input data")
    report = run_diagnostics(
        args.catalog,
        args.dataset,
        model_assets=args.model_assets,
        dense_index_path=args.dense_index,
        limit=args.limit,
    )
    _write_json_exclusive(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "official_metrics": report["official_metrics"],
                "score_gap": report["score_gap"],
                "primary_failure": report["primary_failure"],
                "invariants": report["invariants"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
