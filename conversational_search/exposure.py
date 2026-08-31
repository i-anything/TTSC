"""Conservative, evidence-gated recommendation exposure.

This module is deliberately retrieval-agnostic.  It can narrow or withhold an
already ranked lexical candidate pool, but it cannot introduce or reorder IDs.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Sequence

from conversational_search.exposure_policy import (
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    TOP3_STRUCTURAL_EXPOSURE_POLICY,
    EvidenceExposureDecision,
    EvidenceExposurePolicy,
    EvidenceExposureStatus,
)
from conversational_search.exact_evidence import (
    ExactEvidenceResult,
    ExactEvidenceStatus,
)
from conversational_search.intent import IntentState, active_attributes
from conversational_search.protocol import (
    CandidateReplySignature,
    CandidateReplyStatus,
    DisclosureCard,
    ProductProtocolEvidence,
    remaining_reply,
)
from conversational_search.protocol_index import (
    ProtocolResolution,
    protocol_probe_question,
)
from conversational_search.questions import QUESTION_TEXT
from conversational_search.utility_planner import MAX_TURN, hit_utility


TOP3_EXPOSURE_LIMIT = 3
_TYPED_HARD_ATTRIBUTES = frozenset(
    {
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
    }
)


def plan_evidence_gated_exposure(
    state: IntentState,
    exact_result: ExactEvidenceResult,
    evidence: Sequence[ProductProtocolEvidence],
    *,
    current_turn: int,
    requested_top_k: int,
    retrieval_fault_or_fallback: bool = False,
    require_initial_explicit_buying: bool = False,
    question_prefix_limit: int = 0,
    initial_ambiguous_prefix_limit: int = 0,
    protocol_resolution: ProtocolResolution | None = None,
    metric_aware_protocol_enumeration: bool = False,
    reply_tree_protocol_planning: bool = False,
) -> EvidenceExposureDecision:
    """Expose only when the best structural tier fits inside the API prefix.

    Every non-confident branch fails open to the existing ranked pool, except
    when a genuinely discriminating question is available before the final
    turn.  The default branch withholds recommendations for exactly one turn;
    an explicitly bounded experiment may instead expose the literal ranked
    prefix while asking the same question.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if not isinstance(exact_result, ExactEvidenceResult):
        raise TypeError("exact_result must be ExactEvidenceResult")
    if isinstance(evidence, (str, bytes)) or not isinstance(evidence, Sequence):
        raise TypeError("evidence must be a sequence")
    candidates = tuple(evidence)
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    if isinstance(current_turn, bool) or not isinstance(current_turn, int):
        raise TypeError("current_turn must be an integer")
    if isinstance(requested_top_k, bool) or not isinstance(requested_top_k, int):
        raise TypeError("requested_top_k must be an integer")
    if type(retrieval_fault_or_fallback) is not bool:
        raise TypeError("retrieval_fault_or_fallback must be a boolean")
    if type(require_initial_explicit_buying) is not bool:
        raise TypeError("require_initial_explicit_buying must be a boolean")
    if isinstance(question_prefix_limit, bool) or not isinstance(
        question_prefix_limit,
        int,
    ):
        raise TypeError("question_prefix_limit must be an integer")
    if not 0 <= question_prefix_limit <= TOP3_EXPOSURE_LIMIT:
        raise ValueError("question_prefix_limit must be between zero and three")
    if isinstance(initial_ambiguous_prefix_limit, bool) or not isinstance(
        initial_ambiguous_prefix_limit,
        int,
    ):
        raise TypeError("initial_ambiguous_prefix_limit must be an integer")
    if initial_ambiguous_prefix_limit not in {0, 1}:
        raise ValueError("initial_ambiguous_prefix_limit must be zero or one")
    if protocol_resolution is not None and not isinstance(
        protocol_resolution,
        ProtocolResolution,
    ):
        raise TypeError("protocol_resolution must be a ProtocolResolution or None")
    if type(metric_aware_protocol_enumeration) is not bool:
        raise TypeError("metric_aware_protocol_enumeration must be a boolean")
    if type(reply_tree_protocol_planning) is not bool:
        raise TypeError("reply_tree_protocol_planning must be a boolean")
    if reply_tree_protocol_planning and not metric_aware_protocol_enumeration:
        raise ValueError("reply-tree planning requires metric-aware enumeration")
    ranked_ids = exact_result.ranked_ids
    full_width = min(max(requested_top_k, 0), len(ranked_ids))
    if full_width == 0:
        return EvidenceExposureDecision(
            EvidenceExposureStatus.EMPTY_REQUEST,
            ranked_ids,
            0,
            None,
            0,
        )
    if protocol_resolution is not None and protocol_resolution.exact:
        return _plan_protocol_posterior_exposure(
            ranked_ids,
            protocol_resolution,
            current_turn=current_turn,
            requested_top_k=requested_top_k,
            metric_aware_enumeration=metric_aware_protocol_enumeration,
            reply_tree_planning=reply_tree_protocol_planning,
        )
    if current_turn >= 10:
        return EvidenceExposureDecision(
            EvidenceExposureStatus.FINAL_TURN,
            ranked_ids,
            full_width,
            None,
            exact_result.trace.best_tier_count,
        )
    if retrieval_fault_or_fallback:
        return EvidenceExposureDecision(
            EvidenceExposureStatus.RETRIEVAL_FAIL_OPEN,
            ranked_ids,
            full_width,
            None,
            exact_result.trace.best_tier_count,
        )
    if initial_ambiguous_prefix_limit and _state_is_initial_ambiguous(
        state,
        current_turn,
    ):
        return EvidenceExposureDecision(
            EvidenceExposureStatus.AMBIGUOUS_TOP1_PREVIEW,
            ranked_ids[:1],
            1,
            "other",
            len(ranked_ids),
        )
    if not _state_is_safe_for_exposure(
        state,
        current_turn,
        require_initial_explicit_buying=require_initial_explicit_buying,
    ):
        return EvidenceExposureDecision(
            EvidenceExposureStatus.UNSAFE_STATE,
            ranked_ids,
            full_width,
            None,
            exact_result.trace.best_tier_count,
        )

    plausible_ids = tuple(item.parent_asin for item in exact_result.beliefs)
    evidence_by_id = {item.parent_asin: item for item in candidates}
    disclosure_by_id = {
        item.parent_asin: item.disclosed_values
        for item in exact_result.disclosures
    }
    if not _evidence_is_safe(
        state,
        exact_result,
        candidates,
        plausible_ids,
        evidence_by_id,
        disclosure_by_id,
    ):
        return EvidenceExposureDecision(
            EvidenceExposureStatus.EVIDENCE_FAIL_OPEN,
            ranked_ids,
            full_width,
            None,
            len(plausible_ids),
        )

    plausible_count = len(plausible_ids)
    if (
        plausible_count <= TOP3_EXPOSURE_LIMIT
        and plausible_count <= requested_top_k
    ):
        return EvidenceExposureDecision(
            EvidenceExposureStatus.TOP3_CONFIDENT,
            plausible_ids,
            plausible_count,
            None,
            plausible_count,
        )

    question = _most_discriminating_question(
        state,
        plausible_ids,
        evidence_by_id,
        disclosure_by_id,
    )
    if question is None:
        return EvidenceExposureDecision(
            EvidenceExposureStatus.NO_INFORMATIVE_QUESTION,
            ranked_ids,
            full_width,
            None,
            plausible_count,
        )
    if question_prefix_limit:
        prefix_width = min(
            question_prefix_limit,
            full_width,
            plausible_count,
        )
        return EvidenceExposureDecision(
            EvidenceExposureStatus.QUESTION_WITH_PREFIX,
            plausible_ids[:prefix_width],
            prefix_width,
            question,
            plausible_count,
        )
    return EvidenceExposureDecision(
        EvidenceExposureStatus.QUESTION_WITHHELD,
        ranked_ids,
        0,
        question,
        plausible_count,
    )


def _plan_protocol_posterior_exposure(
    ranked_ids: tuple[str, ...],
    resolution: ProtocolResolution,
    *,
    current_turn: int,
    requested_top_k: int,
    metric_aware_enumeration: bool,
    reply_tree_planning: bool,
) -> EvidenceExposureDecision:
    """Expose a rank-one probe until the complete posterior is exhausted."""

    support_count = resolution.support_count
    if (
        not ranked_ids
        or support_count <= 0
        or not set(ranked_ids).issubset(resolution.candidate_ids)
    ):
        return EvidenceExposureDecision(
            EvidenceExposureStatus.EVIDENCE_FAIL_OPEN,
            ranked_ids,
            min(requested_top_k, len(ranked_ids)),
            None,
            support_count,
        )
    if support_count == 1:
        return EvidenceExposureDecision(
            EvidenceExposureStatus.POSTERIOR_SINGLETON,
            ranked_ids[:1],
            1,
            None,
            1,
        )
    if current_turn < 10:
        question = protocol_probe_question(resolution)
        if question is not None:
            if reply_tree_planning:
                width = plan_protocol_reply_tree_width(
                    ranked_ids,
                    resolution,
                    current_turn=current_turn,
                    top_k=min(requested_top_k, len(ranked_ids)),
                )
                return EvidenceExposureDecision(
                    EvidenceExposureStatus.POSTERIOR_REPLY_TREE,
                    ranked_ids,
                    width,
                    question,
                    support_count,
                )
            return EvidenceExposureDecision(
                EvidenceExposureStatus.POSTERIOR_PROBE,
                ranked_ids[:1],
                1,
                question,
                support_count,
            )
        if metric_aware_enumeration:
            top_k = min(requested_top_k, len(ranked_ids))
            width = plan_protocol_enumeration_width(
                support_count,
                current_turn=current_turn,
                top_k=top_k,
            )
            return EvidenceExposureDecision(
                EvidenceExposureStatus.POSTERIOR_ENUMERATION,
                ranked_ids,
                width,
                None,
                support_count,
            )
    width = min(requested_top_k, len(ranked_ids), support_count)
    return EvidenceExposureDecision(
        EvidenceExposureStatus.POSTERIOR_BATCH,
        ranked_ids,
        width,
        None,
        support_count,
    )


def plan_protocol_enumeration_width(
    support_count: int,
    *,
    current_turn: int,
    top_k: int,
) -> int:
    """Choose a slate width for protocol-indistinguishable survivors.

    The plan assumes a uniform posterior only after the complete published
    card has been exhausted.  A continued score-eligible session refutes the
    displayed prefix, so dynamic programming can compare a hit now against a
    rank-one opportunity on a later turn.  The only reward is the official
    per-session metric; no evaluator labels or fitted constants are used.
    """

    if (
        isinstance(support_count, bool)
        or not isinstance(support_count, int)
        or support_count <= 0
    ):
        raise ValueError("support_count must be a positive integer")
    if (
        isinstance(current_turn, bool)
        or not isinstance(current_turn, int)
        or not 1 <= current_turn <= MAX_TURN
    ):
        raise ValueError("current_turn must be from one through ten")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    return _protocol_enumeration_plan(
        support_count,
        current_turn=current_turn,
        top_k=top_k,
    )[1]


def plan_protocol_reply_tree_width(
    ranked_ids: Sequence[str],
    resolution: ProtocolResolution,
    *,
    current_turn: int,
    top_k: int,
) -> int:
    """Plan exposure against exact deterministic future ``other`` replies.

    The posterior is uniform because no evaluator labels or fitted priors are
    available. Planning is enabled only when the ranked pool contains the
    complete protocol support; incomplete bounded pools conservatively retain
    the proven rank-one probe. Candidate order is otherwise immutable, and a
    continued session refutes precisely the exposed prefix.
    """

    ids = tuple(ranked_ids)
    if not isinstance(resolution, ProtocolResolution) or not resolution.exact:
        raise ValueError("resolution must be exact")
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(not isinstance(value, str) or not value for value in ids)
    ):
        raise ValueError("ranked_ids must contain unique non-empty strings")
    if (
        isinstance(current_turn, bool)
        or not isinstance(current_turn, int)
        or not 1 <= current_turn <= MAX_TURN
    ):
        raise ValueError("current_turn must be from one through ten")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if set(ids) != set(resolution.candidate_ids):
        return 1

    group_by_id = {
        parent_asin: (group.card, group.disclosed_values)
        for group in resolution.groups
        for parent_asin in group.parent_asins
    }
    if set(group_by_id) != set(ids):
        return 1
    hypotheses = tuple(
        (parent_asin, *group_by_id[parent_asin])
        for parent_asin in ids
    )

    @lru_cache(maxsize=None)
    def best_value(
        remaining: tuple[tuple[str, DisclosureCard, tuple[str, ...]], ...],
        turn: int,
    ) -> tuple[float, int]:
        count = len(remaining)
        if count <= 0 or turn > MAX_TURN:
            return 0.0, 0
        signatures = tuple(
            remaining_reply(card, "other", disclosed)
            for _, card, disclosed in remaining
        )
        if not any(
            signature.status is CandidateReplyStatus.DISCLOSURE
            for signature in signatures
        ):
            return _protocol_enumeration_plan(
                count,
                current_turn=turn,
                top_k=min(top_k, count),
            )

        best_reward = -1.0
        best_width = 1
        for width in range(1, min(top_k, count) + 1):
            reward = sum(
                hit_utility(turn, rank)
                for rank in range(1, width + 1)
            ) / count
            if turn < MAX_TURN and count > width:
                branches: dict[
                    CandidateReplySignature,
                    list[tuple[str, DisclosureCard, tuple[str, ...]]],
                ] = {}
                for hypothesis, signature in zip(
                    remaining[width:],
                    signatures[width:],
                ):
                    parent_asin, card, disclosed = hypothesis
                    if signature.status is CandidateReplyStatus.DISCLOSURE:
                        disclosed = tuple(
                            sorted(set(disclosed).union(signature.values))
                        )
                    branches.setdefault(signature, []).append(
                        (parent_asin, card, disclosed)
                    )
                for branch in branches.values():
                    continuation, _ = best_value(tuple(branch), turn + 1)
                    reward += (len(branch) / count) * continuation
            if reward > best_reward + 1e-12 or (
                abs(reward - best_reward) <= 1e-12
                and width > best_width
            ):
                best_reward = reward
                best_width = width
        return best_reward, best_width

    return best_value(hypotheses, current_turn)[1]


def _protocol_enumeration_plan(
    support_count: int,
    *,
    current_turn: int,
    top_k: int,
) -> tuple[float, int]:
    """Return exact value and width for an indistinguishable posterior."""

    @lru_cache(maxsize=None)
    def best_value(remaining: int, turn: int) -> tuple[float, int]:
        if remaining <= 0 or turn > MAX_TURN:
            return 0.0, 0
        best_reward = -1.0
        best_width = 1
        for width in range(1, min(top_k, remaining) + 1):
            reward = sum(
                hit_utility(turn, rank)
                for rank in range(1, width + 1)
            ) / remaining
            if turn < MAX_TURN and remaining > width:
                continuation, _ = best_value(remaining - width, turn + 1)
                reward += ((remaining - width) / remaining) * continuation
            if reward > best_reward + 1e-12 or (
                abs(reward - best_reward) <= 1e-12
                and width > best_width
            ):
                best_reward = reward
                best_width = width
        return best_reward, best_width

    return best_value(support_count, current_turn)


def _state_is_initial_ambiguous(
    state: IntentState,
    current_turn: int,
) -> bool:
    return bool(
        current_turn == 1
        and state.last_turn == 1
        and state.category
        and not state.requirements
        and not state.excluded
        and not state.no_preference
        and not state.asked_attributes
        and state.last_asked_attribute is None
    )


def _evidence_is_safe(
    state: IntentState,
    exact_result: ExactEvidenceResult,
    candidates: Sequence[ProductProtocolEvidence],
    plausible_ids: tuple[str, ...],
    evidence_by_id: dict[str, ProductProtocolEvidence],
    disclosure_by_id: dict[str, tuple[str, ...]],
) -> bool:
    ranked_ids = exact_result.ranked_ids
    return bool(
        exact_result.status is ExactEvidenceStatus.APPLIED
        and plausible_ids
        and len(plausible_ids) == exact_result.trace.best_tier_count
        and plausible_ids == ranked_ids[: len(plausible_ids)]
        and set(plausible_ids).issubset(exact_result.consistent_support_ids)
        and set(ranked_ids) == set(evidence_by_id)
        and len(ranked_ids) == len(evidence_by_id) == len(candidates)
        and set(plausible_ids).issubset(disclosure_by_id)
        and _plausible_categories_match(
            state.category or "",
            plausible_ids,
            evidence_by_id,
        )
    )


def _state_is_safe_for_exposure(
    state: IntentState,
    current_turn: int,
    *,
    require_initial_explicit_buying: bool,
) -> bool:
    hard = tuple(
        requirement
        for requirement in state.requirements
        if requirement.strength == "hard"
    )
    if (
        not state.category
        or state.last_turn != current_turn
        or not hard
        or (
            require_initial_explicit_buying
            and not any(
                requirement.source == "initial_explicit"
                for requirement in hard
            )
        )
        or any(
            requirement.source in {"initial_tentative", "override", "free_text"}
            for requirement in state.requirements
        )
        or any(
            requirement.attribute not in _TYPED_HARD_ATTRIBUTES
            for requirement in hard
        )
    ):
        return False
    hard_attributes = tuple(requirement.attribute for requirement in hard)
    if len(hard_attributes) != len(set(hard_attributes)):
        return False
    if any(
        requirement.attribute in state.no_preference
        for requirement in state.requirements
        if requirement.attribute is not None
    ):
        return False
    positives = {
        " ".join(requirement.value.split()).casefold()
        for requirement in state.requirements
    }
    negatives = {
        " ".join(value.split()).casefold()
        for value in state.excluded
        if isinstance(value, str)
    }
    return not positives.intersection(negatives)


def _plausible_categories_match(
    category: str,
    plausible_ids: Sequence[str],
    evidence_by_id: dict[str, ProductProtocolEvidence],
) -> bool:
    normalized = " ".join(category.split()).casefold()
    return bool(normalized) and all(
        " ".join(evidence_by_id[parent_asin].coarse_category.split()).casefold()
        == normalized
        for parent_asin in plausible_ids
    )


def _most_discriminating_question(
    state: IntentState,
    plausible_ids: Sequence[str],
    evidence_by_id: dict[str, ProductProtocolEvidence],
    disclosure_by_id: dict[str, tuple[str, ...]],
) -> str | None:
    resolved = active_attributes(state) | state.no_preference
    ranked_actions: list[tuple[int, int, int, str]] = []
    for action_index, action in enumerate(QUESTION_TEXT):
        if action in resolved or action in state.asked_attributes:
            continue
        partitions = Counter(
            remaining_reply(
                evidence_by_id[parent_asin].card,
                action,
                disclosure_by_id[parent_asin],
            )
            for parent_asin in plausible_ids
        )
        if len(partitions) < 2:
            continue
        ranked_actions.append(
            (
                max(partitions.values()),
                -len(partitions),
                action_index,
                action,
            )
        )
    return min(ranked_actions)[3] if ranked_actions else None
