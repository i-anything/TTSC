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
    BUYING_TOP3_AMBIGUOUS_TOP1 = (
        "buying-top3-ambiguous-top1-preview-v4"
    )
    PROTOCOL_POSTERIOR = "full-protocol-posterior-probe-v5"
    PROTOCOL_METRIC_AWARE = "full-protocol-metric-aware-enumeration-v6"
    PROTOCOL_REPLY_TREE = "full-protocol-reply-tree-planning-v7"


DISABLED_EVIDENCE_EXPOSURE_POLICY = EvidenceExposurePolicy.DISABLED
TOP3_STRUCTURAL_EXPOSURE_POLICY = EvidenceExposurePolicy.TOP3_STRUCTURAL
BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.BUYING_ONLY_TOP3_STRUCTURAL
)
BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.BUYING_ONLY_TOP3_PREFIX
)
BUYING_TOP3_AMBIGUOUS_TOP1_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.BUYING_TOP3_AMBIGUOUS_TOP1
)
PROTOCOL_POSTERIOR_EXPOSURE_POLICY = EvidenceExposurePolicy.PROTOCOL_POSTERIOR
PROTOCOL_METRIC_AWARE_EXPOSURE_POLICY = (
    EvidenceExposurePolicy.PROTOCOL_METRIC_AWARE
)
PROTOCOL_REPLY_TREE_EXPOSURE_POLICY = EvidenceExposurePolicy.PROTOCOL_REPLY_TREE


class EvidenceExposureStatus(str, Enum):
    """Why the gate narrowed, withheld, or failed open."""

    TOP3_CONFIDENT = "top3_confident"
    QUESTION_WITHHELD = "question_withheld"
    QUESTION_WITH_PREFIX = "question_with_prefix"
    AMBIGUOUS_TOP1_PREVIEW = "ambiguous_top1_preview"
    POSTERIOR_SINGLETON = "posterior_singleton"
    POSTERIOR_PROBE = "posterior_probe"
    POSTERIOR_BATCH = "posterior_batch"
    POSTERIOR_ENUMERATION = "posterior_enumeration"
    POSTERIOR_REPLY_TREE = "posterior_reply_tree"
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
