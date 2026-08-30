"""Lightweight policy and result types for evidence-gated exposure."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EvidenceExposurePolicy(str, Enum):
    DISABLED = "disabled"
    TOP3_STRUCTURAL = "top3-structural-evidence-gate-v1"
    BUYING_ONLY_TOP3_STRUCTURAL = (
        "buying-only-top3-structural-evidence-gate-v2"
    )
    BUYING_ONLY_TOP3_PREFIX = "buying-only-top3-prefix-question-v3"


DISABLED_EVIDENCE_EXPOSURE_POLICY = EvidenceExposurePolicy.DISABLED
TOP3_STRUCTURAL_EXPOSURE_POLICY = EvidenceExposurePolicy.TOP3_STRUCTURAL
BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.BUYING_ONLY_TOP3_STRUCTURAL
)
BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.BUYING_ONLY_TOP3_PREFIX
)


class EvidenceExposureStatus(str, Enum):
    """Why the gate narrowed, withheld, or failed open."""

    TOP3_CONFIDENT = "top3_confident"
    QUESTION_WITHHELD = "question_withheld"
    QUESTION_WITH_PREFIX = "question_with_prefix"
    NO_INFORMATIVE_QUESTION = "no_informative_question"
    FINAL_TURN = "final_turn"
    UNSAFE_STATE = "unsafe_state"
    RETRIEVAL_FAIL_OPEN = "retrieval_fail_open"
    EVIDENCE_FAIL_OPEN = "evidence_fail_open"
    EMPTY_REQUEST = "empty_request"


@dataclass(frozen=True, slots=True)
class EvidenceExposureDecision:
    """A width/question decision over an immutable pre-existing ranking."""

    status: EvidenceExposureStatus
    presentation_ids: tuple[str, ...]
    width: int
    question: str | None
    plausible_count: int
