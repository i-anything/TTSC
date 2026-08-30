"""Deterministic label-free oracle for Phase 13 intent-epoch slates."""

from __future__ import annotations

import hashlib
import itertools
import json
import random

from conversational_search.slates import (
    INTENT_EPOCH_NOVELTY_SLATE_POLICY,
    STAGNATION_AWARE_SLATE_POLICY,
    IntentEpochSlateStatus,
    SlateState,
    select_slate,
    select_slate_with_intent_epoch_novelty,
)


ORACLE_SEED = 130260830
RANDOM_CASES = 30_000
EXPECTED_SHA256 = "b0b83da7d111ff66c3f3452e64337143ed376a9b9926779cd37259dc9ea114be"


def _ordered_pools(values: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        pool
        for length in range(len(values) + 1)
        for pool in itertools.permutations(values, length)
    )


def _histories(values: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        history
        for length in range(len(values) + 1)
        for history in itertools.combinations(values, length)
    )


def _transition(
    kind: str,
    shown: tuple[str, ...],
) -> tuple[SlateState, tuple[object, ...]]:
    if kind == "first":
        return SlateState(shown_ids=shown), (0, "new")
    if kind == "unchanged":
        signature = (0, "same")
        return SlateState(signature=signature, shown_ids=shown), signature
    if kind == "carried":
        return SlateState(signature=(0, "old"), shown_ids=shown), (0, "new")
    if kind == "epoch_reset":
        return SlateState(signature=(0, "old"), shown_ids=shown), (1, "new")
    if kind == "malformed":
        return (
            SlateState(signature=("invalid", "old"), shown_ids=shown),
            (0, "new"),
        )
    raise ValueError("unsupported oracle transition")


def _visible_rank(values: tuple[str, ...], target: str) -> int:
    return values.index(target) + 1 if target in values else 1_000_000


def _record_case(
    digest: hashlib._Hash,
    *,
    state: SlateState,
    signature: tuple[object, ...],
    pool: tuple[str, ...],
    limit: int,
) -> None:
    baseline = select_slate(
        STAGNATION_AWARE_SLATE_POLICY,
        state,
        signature,
        pool,
        limit,
    )
    result = select_slate_with_intent_epoch_novelty(
        state,
        signature,
        pool,
        limit,
    )
    direct = select_slate(
        INTENT_EPOCH_NOVELTY_SLATE_POLICY,
        state,
        signature,
        pool,
        limit,
    )
    if direct != result.selection:
        raise AssertionError("public policy and traced selector diverged")
    selected = result.selection.selected_ids
    if len(selected) != len(set(selected)) or not set(selected).issubset(pool):
        raise AssertionError("candidate slate is not a unique pool subset")
    if len(selected) > min(max(limit, 0), len(pool)):
        raise AssertionError("candidate slate exceeded its output bound")
    shown = result.selection.state.shown_ids
    if len(shown) != len(set(shown)) or len(shown) > 200:
        raise AssertionError("candidate retained state is duplicate or unbounded")
    if pool and limit > 0 and not set(shown).issubset(pool):
        raise AssertionError("active candidate state escaped the current pool")
    if result.status is not IntentEpochSlateStatus.CARRIED:
        if result.selection != baseline:
            raise AssertionError("neutral candidate case is not exact baseline")
    else:
        prior_pool_history = frozenset(state.shown_ids).intersection(pool)
        for target in pool:
            if target in prior_pool_history:
                continue
            if _visible_rank(selected, target) > _visible_rank(
                baseline.selected_ids,
                target,
            ):
                raise AssertionError("continuation target rank regressed")
    if not 0 <= result.eligible_prior_shown <= 200:
        raise AssertionError("eligible history count is out of bounds")

    payload = {
        "prior_signature": state.signature,
        "prior_shown": state.shown_ids,
        "signature": signature,
        "pool": pool,
        "limit": limit,
        "selected": selected,
        "next_shown": shown,
        "status": result.status.value,
        "eligible_prior_shown": result.eligible_prior_shown,
        "trace": {
            "signature_changed": result.selection.trace.signature_changed,
            "stagnant_turn": result.selection.trace.stagnant_turn,
            "unseen_selected": result.selection.trace.unseen_selected,
            "repeat_backfills": result.selection.trace.repeat_backfills,
        },
    }
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def oracle_digest() -> tuple[int, int, str]:
    digest = hashlib.sha256()
    values = ("A", "B", "C", "D")
    exhaustive_cases = 0
    for pool in _ordered_pools(values):
        for shown in _histories(values):
            for kind in (
                "first",
                "unchanged",
                "carried",
                "epoch_reset",
                "malformed",
            ):
                state, signature = _transition(kind, shown)
                for limit in (0, 1, 3):
                    _record_case(
                        digest,
                        state=state,
                        signature=signature,
                        pool=pool,
                        limit=limit,
                    )
                    exhaustive_cases += 1

    rng = random.Random(ORACLE_SEED)
    universe = tuple(f"P{index:02d}" for index in range(24))
    kinds = ("first", "unchanged", "carried", "epoch_reset", "malformed")
    for _ in range(RANDOM_CASES):
        pool = tuple(rng.sample(universe, rng.randrange(0, 21)))
        shown = tuple(rng.sample(universe, rng.randrange(0, 21)))
        state, signature = _transition(rng.choice(kinds), shown)
        _record_case(
            digest,
            state=state,
            signature=signature,
            pool=pool,
            limit=rng.randrange(0, 11),
        )
    return exhaustive_cases, exhaustive_cases + RANDOM_CASES, digest.hexdigest()


def verify() -> dict[str, int | str]:
    exhaustive_cases, cases, observed = oracle_digest()
    if observed != EXPECTED_SHA256:
        raise AssertionError(
            "Phase 13 slate oracle drifted: "
            f"expected {EXPECTED_SHA256}, observed {observed}"
        )
    return {
        "status": "ok",
        "seed": ORACLE_SEED,
        "cases": cases,
        "exhaustive_cases": exhaustive_cases,
        "random_cases": RANDOM_CASES,
        "digest": observed,
    }


def main() -> None:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
