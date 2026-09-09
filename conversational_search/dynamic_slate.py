"""Answer-conditioned scheduling for a bounded recommendation slate.

The planner receives label-free candidate beliefs and the rank each candidate
would obtain after its own observable answer branch.  It then assigns the
current slate to candidates that would otherwise be difficult to recover,
while deferring candidates that the next turn can rank more strongly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite

from conversational_search.exact_evidence import CandidateBelief
from conversational_search.utility_planner import MAX_TURN, hit_utility


class DynamicSlatePolicy(str, Enum):
    """Supported policies for answer-conditioned slate scheduling."""

    DISABLED = "disabled"
    NEXT_TURN_RECOVERABILITY = "next-turn-recoverability-v1"


DISABLED_DYNAMIC_SLATE_POLICY = DynamicSlatePolicy.DISABLED
NEXT_TURN_RECOVERABILITY_SLATE_POLICY = (
    DynamicSlatePolicy.NEXT_TURN_RECOVERABILITY
)


@dataclass(frozen=True, slots=True)
class DynamicSlatePlan:
    """Selected current IDs and their two-turn expected utility."""

    selected_ids: tuple[str, ...]
    baseline_value: float
    planned_value: float

    @property
    def changed(self) -> bool:
        return self.planned_value > self.baseline_value + 1e-12


def plan_answer_conditioned_slate(
    ranked_ids: Sequence[str],
    beliefs: Sequence[CandidateBelief],
    post_reply_ranks: Sequence[tuple[str, int | None]],
    *,
    current_turn: int,
    width: int,
    top_k: int,
) -> DynamicSlatePlan:
    """Schedule one slate against exact per-branch next-turn ranks.

    A candidate can earn the official hit utility either now, at its assigned
    display rank, or on the next turn at the predicted rank for the answer its
    product card would emit.  Dynamic programming finds the exact best ordered
    subset while preserving the current evidence order among selected items.
    """

    if isinstance(ranked_ids, (str, bytes)) or not isinstance(
        ranked_ids, Sequence
    ):
        raise TypeError("ranked_ids must be a sequence")
    ranked = tuple(ranked_ids)
    if (
        not ranked
        or len(ranked) != len(set(ranked))
        or any(not isinstance(value, str) or not value for value in ranked)
    ):
        raise ValueError("ranked_ids must contain unique non-empty strings")
    if isinstance(beliefs, (str, bytes)) or not isinstance(beliefs, Sequence):
        raise TypeError("beliefs must be a sequence")
    candidate_beliefs = tuple(beliefs)
    if not candidate_beliefs or any(
        not isinstance(belief, CandidateBelief) for belief in candidate_beliefs
    ):
        raise ValueError("beliefs must contain CandidateBelief values")
    if isinstance(post_reply_ranks, (str, bytes)) or not isinstance(
        post_reply_ranks, Sequence
    ):
        raise TypeError("post_reply_ranks must be a sequence")
    if (
        isinstance(current_turn, bool)
        or not isinstance(current_turn, int)
        or not 1 <= current_turn < MAX_TURN
    ):
        raise ValueError("current_turn must be from one through nine")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= 10
    ):
        raise ValueError("top_k must be from one through ten")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= top_k
    ):
        raise ValueError("width must be from one through top_k")

    belief_by_id: dict[str, float] = {}
    for belief in candidate_beliefs:
        if (
            belief.parent_asin in belief_by_id
            or belief.parent_asin not in ranked
            or isinstance(belief.weight, bool)
            or not isinstance(belief.weight, (int, float))
            or not isfinite(float(belief.weight))
            or float(belief.weight) <= 0.0
        ):
            raise ValueError("candidate beliefs are malformed")
        belief_by_id[belief.parent_asin] = float(belief.weight)
    candidates = tuple(value for value in ranked if value in belief_by_id)
    if len(candidates) < width or ranked[:width] != candidates[:width]:
        raise ValueError("the exposed prefix must belong to the belief tier")

    rank_by_id: dict[str, int | None] = {}
    for value in post_reply_ranks:
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("post-reply ranks must contain two-tuples")
        parent_asin, rank = value
        if parent_asin in rank_by_id or parent_asin not in belief_by_id:
            raise ValueError("post-reply candidate IDs are malformed")
        if rank is not None and (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 1 <= rank <= top_k
        ):
            raise ValueError("post-reply ranks must be within top_k or None")
        rank_by_id[parent_asin] = rank
    if set(rank_by_id) != set(belief_by_id):
        raise ValueError("post-reply ranks must cover every candidate belief")

    future_utility = {
        parent_asin: (
            0.0
            if rank_by_id[parent_asin] is None
            else hit_utility(current_turn + 1, rank_by_id[parent_asin])
        )
        for parent_asin in candidates
    }

    def value_for(selected: tuple[str, ...]) -> float:
        current_rank = {
            parent_asin: rank
            for rank, parent_asin in enumerate(selected, start=1)
        }
        return sum(
            belief_by_id[parent_asin]
            * (
                hit_utility(current_turn, current_rank[parent_asin])
                if parent_asin in current_rank
                else future_utility[parent_asin]
            )
            for parent_asin in candidates
        )

    # Each cell stores the best utility for the processed prefix and the IDs
    # selected from it. Skipped candidates receive their predicted next-turn
    # utility; the j-th selected candidate receives current display rank j.
    table: list[list[tuple[float, tuple[str, ...]] | None]] = [
        [None] * (width + 1) for _ in range(len(candidates) + 1)
    ]
    table[0][0] = (0.0, ())
    for index, parent_asin in enumerate(candidates, start=1):
        probability = belief_by_id[parent_asin]
        deferred = probability * future_utility[parent_asin]
        for selected_count in range(width + 1):
            prior = table[index - 1][selected_count]
            if prior is not None:
                table[index][selected_count] = (
                    prior[0] + deferred,
                    prior[1],
                )
            if selected_count == 0:
                continue
            prior = table[index - 1][selected_count - 1]
            if prior is None:
                continue
            selected_value = prior[0] + probability * hit_utility(
                current_turn,
                selected_count,
            )
            incumbent = table[index][selected_count]
            if (
                incumbent is None
                or selected_value > incumbent[0] + 1e-12
            ):
                table[index][selected_count] = (
                    selected_value,
                    (*prior[1], parent_asin),
                )

    planned = table[-1][width]
    if planned is None:
        raise RuntimeError("dynamic slate plan is incomplete")
    baseline_ids = ranked[:width]
    baseline_value = value_for(baseline_ids)
    if planned[0] <= baseline_value + 1e-12:
        return DynamicSlatePlan(
            baseline_ids,
            baseline_value,
            baseline_value,
        )
    return DynamicSlatePlan(planned[1], baseline_value, planned[0])
