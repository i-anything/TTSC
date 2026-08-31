"""Bounded field-level semantic evidence for validated local intent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


MAX_FIELD_SEMANTIC_CANDIDATES = 20
MAX_FIELD_SEMANTIC_REQUIREMENTS = 8


class FieldSemanticPolicy(str, Enum):
    """Reversible policies for semantic matching against catalog card atoms."""

    DISABLED = "disabled"
    LOCAL_INTENT_CARD_ATOMS = "local-intent-card-atoms-v1"


DISABLED_FIELD_SEMANTIC_POLICY = FieldSemanticPolicy.DISABLED
LOCAL_INTENT_CARD_ATOMS_POLICY = FieldSemanticPolicy.LOCAL_INTENT_CARD_ATOMS


class FieldSemanticStatus(str, Enum):
    """Mutually exclusive outcomes of a validated semantic ranking pass."""

    NO_SIGNAL = "no_signal"
    UNCHANGED = "unchanged"
    REORDERED = "reordered"


@dataclass(frozen=True, slots=True)
class FieldSemanticAssessment:
    """Candidate affinities derived only from frozen catalog card fields."""

    parent_asin: str
    exclusion_affinity: float | None
    minimum_requirement_affinity: float | None
    mean_requirement_affinity: float | None
    category_affinity: float | None


@dataclass(frozen=True, slots=True)
class FieldSemanticResult:
    ranked_ids: tuple[str, ...]
    status: FieldSemanticStatus


def rank_field_semantic(
    candidate_ids: Sequence[str],
    assessments: Sequence[FieldSemanticAssessment],
) -> FieldSemanticResult:
    """Rank aligned candidates with a weight-free semantic lexicographic key.

    Exclusion avoidance is considered first, followed by weakest-requirement
    coverage, average requirement coverage, and category affinity. Missing
    signals are permitted only when they are missing for every candidate.
    """

    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence")
    if isinstance(assessments, (str, bytes)):
        raise TypeError("assessments must be a sequence")
    ids = tuple(candidate_ids)
    values = tuple(assessments)
    if not ids or len(ids) > MAX_FIELD_SEMANTIC_CANDIDATES:
        raise ValueError("field-semantic candidates are empty or out of bounds")
    if len(ids) != len(values) or len(set(ids)) != len(ids):
        raise ValueError("field-semantic candidates must be unique and aligned")
    if any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        for parent_asin in ids
    ):
        raise ValueError("candidate IDs must be normalized strings")
    if any(not isinstance(item, FieldSemanticAssessment) for item in values):
        raise TypeError("assessments must contain FieldSemanticAssessment values")
    if tuple(item.parent_asin for item in values) != ids:
        raise ValueError("field-semantic assessments are positionally misaligned")

    fields = (
        "exclusion_affinity",
        "minimum_requirement_affinity",
        "mean_requirement_affinity",
        "category_affinity",
    )
    for field in fields:
        field_values = tuple(getattr(item, field) for item in values)
        present = tuple(value is not None for value in field_values)
        if any(present) and not all(present):
            raise ValueError("semantic evidence coverage must be candidate-complete")
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not -1.0 <= float(value) <= 1.0
            )
            for value in field_values
        ):
            raise ValueError("semantic affinities must be finite cosine values")

    if all(
        getattr(values[0], field) is None
        for field in fields
    ):
        return FieldSemanticResult(ids, FieldSemanticStatus.NO_SIGNAL)

    def key(index: int) -> tuple[float, float, float, float, int]:
        item = values[index]
        return (
            float(item.exclusion_affinity)
            if item.exclusion_affinity is not None
            else -1.0,
            -float(item.minimum_requirement_affinity)
            if item.minimum_requirement_affinity is not None
            else 1.0,
            -float(item.mean_requirement_affinity)
            if item.mean_requirement_affinity is not None
            else 1.0,
            -float(item.category_affinity)
            if item.category_affinity is not None
            else 1.0,
            index,
        )

    ranked_ids = tuple(ids[index] for index in sorted(range(len(ids)), key=key))
    return FieldSemanticResult(
        ranked_ids,
        FieldSemanticStatus.UNCHANGED
        if ranked_ids == ids
        else FieldSemanticStatus.REORDERED,
    )
