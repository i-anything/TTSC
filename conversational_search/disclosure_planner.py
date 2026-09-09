"""Catalog-based clarification and ranked recommendation planning.

The reciprocal-rank prior is a heuristic, not a learned target probability.
Rollouts retain current candidate order; actual answers trigger live retrieval
and transcript resolution again. No evaluation labels enter this module.
"""

from __future__ import annotations

from functools import lru_cache

from conversational_search.protocol import CandidateReplyStatus, remaining_reply
from conversational_search.protocol_index import ProtocolResolution
from conversational_search.questions import QUESTION_TEXT
from conversational_search.utility_planner import MAX_TURN, hit_utility


@lru_cache(maxsize=512)
def _ranked_enumeration_plan(
    weights: tuple[float, ...], current_turn: int, top_k: int,
) -> tuple[float, int]:
    """Compare showing a prefix now with showing its survivors on later turns.

    Weights remain fixed throughout the rollout. The reachable prefix must be
    covered by the deadline, so a heuristic prior cannot discard its tail to
    improve rank. Only generic rank weights enter the bounded cache.
    """

    @lru_cache(maxsize=None)
    def value(offset: int, turn: int) -> tuple[float, int]:
        if offset >= len(weights) or turn > MAX_TURN:
            return 0.0, 0
        best_value, best_width = -1.0, 1
        immediate = 0.0
        minimum_width = max(1, len(weights) - offset - top_k * (MAX_TURN - turn))
        for width in range(1, min(top_k, len(weights) - offset) + 1):
            immediate += weights[offset + width - 1] * hit_utility(turn, width)
            if width < minimum_width:
                continue
            reward = immediate + value(offset + width, turn + 1)[0]
            if reward > best_value + 1e-12 or (
                abs(reward - best_value) <= 1e-12 and width > best_width
            ):
                best_value, best_width = reward, width
        return best_value, best_width

    return value(0, current_turn)


def plan_ranked_enumeration_width(
    support_count: int, *, current_turn: int, top_k: int,
) -> int:
    """Use the same reciprocal-rank prior as exact-evidence candidate beliefs."""

    if type(support_count) is not int or support_count <= 0:
        raise ValueError("support_count must be a positive integer")
    if type(current_turn) is not int or not 1 <= current_turn <= MAX_TURN:
        raise ValueError("current_turn must be from one through ten")
    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be from one through ten")
    # A finite conversation cannot expose products beyond this prefix.
    count = min(support_count, top_k * (MAX_TURN - current_turn + 1))
    weights = tuple(1.0 / rank for rank in range(1, count + 1))
    return _ranked_enumeration_plan(weights, current_turn, top_k)[1]


def plan_disclosure_question(
    ranked_ids: tuple[str, ...],
    resolution: ProtocolResolution,
    *,
    current_turn: int,
    top_k: int,
    prefer_wildcard_near_tie: bool = False,
) -> str:
    """Compare a question's replies through the remaining conversation.

    The current rank-one preview is held fixed. Each possible answer is
    followed by wildcard disclosures and ranked enumeration until success or
    the turn limit. Complete catalog support remains in the model, including
    products outside the bounded retrieval prefix. Exact ties retain ``other``.
    """

    if type(prefer_wildcard_near_tie) is not bool:
        raise TypeError("prefer_wildcard_near_tie must be a boolean")
    if not resolution.exact or not ranked_ids or current_turn >= MAX_TURN:
        return "other"
    groups = resolution.groups
    by_id = {
        parent_asin: index
        for index, group in enumerate(groups)
        for parent_asin in group.parent_asins
    }
    if not set(ranked_ids).issubset(by_id):
        return "other"
    ordered = tuple(dict.fromkeys((*ranked_ids, *resolution.candidate_ids)))
    # Each product gets its own mass, even when multiple products share a card.
    hypotheses = tuple(
        (by_id[parent_asin], groups[by_id[parent_asin]].disclosed_values, 1.0 / rank)
        for rank, parent_asin in enumerate(ordered, start=1)
    )

    @lru_cache(maxsize=None)
    def reply(index: int, disclosed: tuple[str, ...], question: str):
        signature = remaining_reply(groups[index].card, question, disclosed)
        updated = disclosed
        if signature.status is CandidateReplyStatus.DISCLOSURE:
            updated = tuple(sorted(set(disclosed).union(signature.values)))
        return signature.reply_text, updated

    def branches(remaining, question):
        partitions = {}
        for index, disclosed, weight in remaining:
            text, updated = reply(index, disclosed, question)
            # Tuple equality would incorrectly separate indistinguishable
            # answers when a catalog value itself contains a semicolon.
            partitions.setdefault(text, []).append((index, updated, weight))
        return tuple(tuple(partition) for partition in partitions.values())

    @lru_cache(maxsize=None)
    def continuation(remaining, turn):
        if not remaining or turn > MAX_TURN:
            return 0.0
        if len(remaining) == 1:
            return remaining[0][2] * hit_utility(turn, 1)
        if turn == MAX_TURN:
            return sum(
                item[2] * hit_utility(turn, rank)
                for rank, item in enumerate(remaining[:top_k], start=1)
            )
        has_disclosure = any(
            reply(index, disclosed, "other")[1] != disclosed
            for index, disclosed, _ in remaining
        )
        if not has_disclosure:
            capacity = top_k * (MAX_TURN - turn + 1)
            weights = tuple(item[2] for item in remaining[:capacity])
            return _ranked_enumeration_plan(weights, turn, top_k)[0]
        return remaining[0][2] * hit_utility(turn, 1) + sum(
            continuation(branch, turn + 1)
            for branch in branches(remaining[1:], "other")
        )

    best_question, best_value = "other", -1.0
    other_value = -1.0
    for question in ("other", *(q for q in QUESTION_TEXT if q != "other")):
        value = sum(
            continuation(branch, current_turn + 1)
            for branch in branches(hypotheses[1:], question)
        )
        if question == "other":
            other_value = value
        if value > best_value + 1e-12:
            best_question, best_value = question, value
    # Rollouts retain today's order while the live system re-ranks after the
    # answer.  When a specific question's modeled edge is smaller than the
    # value of moving the highest unshown hypothesis by one turn, prefer the
    # wildcard: it reveals up to two values and is less exposed to that
    # approximation.  This bound comes directly from the metric and the
    # reciprocal-rank prior (0.02 turn value times rank-two weight 1/2).
    ordering_uncertainty = 0.01
    if (
        prefer_wildcard_near_tie
        and best_question != "other"
        and best_value - other_value <= ordering_uncertainty
    ):
        return "other"
    return best_question
