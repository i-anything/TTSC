"""Deterministic post-ranking exploration for unchanged conversational intent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from conversational_search.intent import IntentState
from conversational_search.strategy import RouteWeights


MAX_SLATE_CANDIDATES = 200


class SlatePolicy(Enum):
    """Supported immutable policies for selecting from one ranked candidate pool."""

    REPEAT_TOP = "repeat_top"
    STAGNATION_AWARE = "stagnation_aware"


REPEAT_TOP_SLATE_POLICY = SlatePolicy.REPEAT_TOP
STAGNATION_AWARE_SLATE_POLICY = SlatePolicy.STAGNATION_AWARE


@dataclass(frozen=True, slots=True)
class SlateState:
    """Bounded session-local memory for the current ranking signature."""

    signature: tuple[object, ...] | None = None
    shown_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SlateTrace:
    """Aggregate-safe selection facts containing no query or product IDs."""

    signature_changed: bool
    stagnant_turn: bool
    unseen_selected: int
    repeat_backfills: int


@dataclass(frozen=True, slots=True)
class SlateSelection:
    selected_ids: tuple[str, ...]
    state: SlateState
    trace: SlateTrace


def ranking_signature(
    state: IntentState,
    dense_query: str,
    lexical_query: str,
    route_weights: RouteWeights,
    ranking_policy: str,
    ranked_ids: Sequence[str],
    result_count: int,
) -> tuple[object, ...]:
    """Describe every label-free input that can change the ranked slate."""

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if not isinstance(dense_query, str) or not isinstance(lexical_query, str):
        raise TypeError("rendered queries must be strings")
    if not isinstance(route_weights, RouteWeights):
        raise TypeError("route_weights must be RouteWeights")
    if not isinstance(ranking_policy, str) or not ranking_policy:
        raise ValueError("ranking_policy must be a non-empty string")
    if (
        isinstance(result_count, bool)
        or not isinstance(result_count, int)
        or not 1 <= result_count <= 10
    ):
        raise ValueError("result_count must be an integer from 1 through 10")
    pool = _ranked_pool(ranked_ids)
    return (
        state.intent_version,
        state.category,
        dense_query,
        lexical_query,
        route_weights.bm25,
        route_weights.dense,
        ranking_policy,
        tuple(
            (requirement.value, requirement.source, requirement.attribute)
            for requirement in state.requirements
        ),
        state.excluded,
        pool,
        result_count,
    )


def select_slate(
    policy: SlatePolicy,
    state: SlateState,
    signature: tuple[object, ...],
    ranked_ids: Sequence[str],
    limit: int,
) -> SlateSelection:
    """Select unseen candidates first, then deterministically backfill on exhaustion."""

    if not isinstance(policy, SlatePolicy):
        raise TypeError("policy must be SlatePolicy")
    if not isinstance(state, SlateState):
        raise TypeError("state must be SlateState")
    if not isinstance(signature, tuple):
        raise TypeError("signature must be a tuple")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer")
    pool = _ranked_pool(ranked_ids)
    if limit <= 0 or not pool:
        return SlateSelection(
            selected_ids=(),
            state=state,
            trace=SlateTrace(False, False, 0, 0),
        )

    if policy is SlatePolicy.REPEAT_TOP:
        return SlateSelection(
            selected_ids=pool[:limit],
            state=state,
            trace=SlateTrace(False, False, 0, 0),
        )

    signature_changed = state.signature != signature
    pool_set = frozenset(pool)
    prior_shown = (
        ()
        if signature_changed
        else tuple(parent_asin for parent_asin in state.shown_ids if parent_asin in pool_set)
    )
    shown_set = frozenset(prior_shown)
    unseen = tuple(parent_asin for parent_asin in pool if parent_asin not in shown_set)
    selected = list(unseen[:limit])
    unseen_selected = len(selected)
    if len(selected) < limit:
        selected_set = frozenset(selected)
        selected.extend(
            parent_asin
            for parent_asin in pool
            if parent_asin not in selected_set
        )
        del selected[limit:]

    repeat_backfills = len(selected) - unseen_selected
    next_shown = tuple(dict.fromkeys((*prior_shown, *selected)))
    return SlateSelection(
        selected_ids=tuple(selected),
        state=SlateState(signature=signature, shown_ids=next_shown),
        trace=SlateTrace(
            signature_changed=signature_changed,
            stagnant_turn=not signature_changed,
            unseen_selected=unseen_selected,
            repeat_backfills=repeat_backfills,
        ),
    )


def _ranked_pool(ranked_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ranked_ids, (str, bytes)) or not isinstance(ranked_ids, Sequence):
        raise TypeError("ranked_ids must be a sequence of product IDs")
    pool = tuple(ranked_ids)
    if len(pool) > MAX_SLATE_CANDIDATES:
        raise ValueError(f"at most {MAX_SLATE_CANDIDATES} ranked IDs are supported")
    if any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        for parent_asin in pool
    ):
        raise ValueError("ranked_ids must contain normalized non-empty strings")
    if len(pool) != len(set(pool)):
        raise ValueError("ranked_ids must not contain duplicates")
    return pool
