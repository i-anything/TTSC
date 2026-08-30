"""Conservative, evidence-gated recommendation exposure.

This module is deliberately retrieval-agnostic.  It can narrow or withhold an
already ranked lexical candidate pool, but it cannot introduce or reorder IDs.
"""

from __future__ import annotations

from collections import Counter
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
    ProductProtocolEvidence,
    remaining_reply,
)
from conversational_search.questions import QUESTION_TEXT


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
    evidence_is_safe = (
        exact_result.status is ExactEvidenceStatus.APPLIED
        and bool(plausible_ids)
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
    if not evidence_is_safe:
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
