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
    INTENT_EPOCH_NOVELTY = "phase13-intent-epoch-continuation-novelty-v1"


REPEAT_TOP_SLATE_POLICY = SlatePolicy.REPEAT_TOP
STAGNATION_AWARE_SLATE_POLICY = SlatePolicy.STAGNATION_AWARE
INTENT_EPOCH_NOVELTY_SLATE_POLICY = SlatePolicy.INTENT_EPOCH_NOVELTY


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


class IntentEpochSlateStatus(str, Enum):
    """Mutually exclusive aggregate outcomes for the Phase 13 policy."""

    EMPTY = "empty_exact_baseline"
    FIRST = "first_slate_exact_baseline"
    UNCHANGED = "unchanged_signature_exact_baseline"
    EPOCH_RESET = "changed_epoch_exact_baseline"
    CARRIED = "same_epoch_history_carried"
    VALIDATION_FALLBACK = "validation_fallback"


@dataclass(frozen=True, slots=True)
class IntentEpochSlateSelection:
    """One transient Phase 13 decision around an ordinary slate selection."""

    selection: SlateSelection
    status: IntentEpochSlateStatus
    eligible_prior_shown: int


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
    if policy is SlatePolicy.INTENT_EPOCH_NOVELTY:
        return select_slate_with_intent_epoch_novelty(
            state,
            signature,
            pool,
            limit,
        ).selection

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


def select_slate_with_intent_epoch_novelty(
    state: SlateState,
    signature: tuple[object, ...],
    ranked_ids: Sequence[str],
    limit: int,
) -> IntentEpochSlateSelection:
    """Carry shown IDs across ranking changes inside one explicit intent epoch.

    The protected stagnation-aware result is computed first and is returned
    exactly for every neutral or candidate-validation case. Only a changed
    signature with equal valid intent epochs takes the candidate path.
    """

    baseline = select_slate(
        STAGNATION_AWARE_SLATE_POLICY,
        state,
        signature,
        ranked_ids,
        limit,
    )
    pool = _ranked_pool(ranked_ids)
    if limit <= 0 or not pool:
        return IntentEpochSlateSelection(
            baseline,
            IntentEpochSlateStatus.EMPTY,
            0,
        )
    if state.signature is None:
        return IntentEpochSlateSelection(
            baseline,
            IntentEpochSlateStatus.FIRST,
            0,
        )
    if state.signature == signature:
        pool_set = frozenset(pool)
        eligible_prior = tuple(
            dict.fromkeys(
                parent_asin
                for parent_asin in state.shown_ids
                if parent_asin in pool_set
            )
        )
        return IntentEpochSlateSelection(
            baseline,
            IntentEpochSlateStatus.UNCHANGED,
            len(eligible_prior),
        )
    try:
        prior_epoch = _intent_epoch(state.signature)
        current_epoch = _intent_epoch(signature)
    except (TypeError, ValueError):
        return IntentEpochSlateSelection(
            baseline,
            IntentEpochSlateStatus.VALIDATION_FALLBACK,
            0,
        )
    if prior_epoch != current_epoch:
        return IntentEpochSlateSelection(
            baseline,
            IntentEpochSlateStatus.EPOCH_RESET,
            0,
        )

    pool_set = frozenset(pool)
    prior_shown = tuple(
        dict.fromkeys(
            parent_asin
            for parent_asin in state.shown_ids
            if parent_asin in pool_set
        )
    )
    shown_set = frozenset(prior_shown)
    unseen = tuple(
        parent_asin for parent_asin in pool if parent_asin not in shown_set
    )
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
    candidate = SlateSelection(
        selected_ids=tuple(selected),
        state=SlateState(signature=signature, shown_ids=next_shown),
        trace=SlateTrace(
            signature_changed=True,
            stagnant_turn=False,
            unseen_selected=unseen_selected,
            repeat_backfills=repeat_backfills,
        ),
    )
    return IntentEpochSlateSelection(
        candidate,
        IntentEpochSlateStatus.CARRIED,
        len(prior_shown),
    )


def _intent_epoch(signature: tuple[object, ...]) -> int:
    if not isinstance(signature, tuple) or not signature:
        raise TypeError("ranking signature must contain an intent epoch")
    epoch = signature[0]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("intent epoch must be a non-negative integer")
    return epoch


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
