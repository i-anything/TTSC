"""Pure fusion-weight policies derived from conversational intent."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .intent import IntentState


_STRONG_SOURCES = frozenset({"initial_explicit", "answer", "override"})
_WEAK_SOURCES = frozenset({"initial_tentative", "free_text"})


@dataclass(frozen=True, slots=True)
class RouteWeights:
    """Normalized weights for lexical and dense reciprocal-rank fusion."""

    bm25: float
    dense: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.bm25) or not math.isfinite(self.dense):
            raise ValueError("fusion weights must be finite")
        if not 0.0 < self.bm25 <= 1.0 or not 0.0 < self.dense <= 1.0:
            raise ValueError("fusion weights must be greater than zero and at most one")
        if not math.isclose(
            self.bm25 + self.dense,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("fusion weights must sum to one")


def intent_completeness(state: IntentState) -> float:
    """Return bounded intent evidence from active-requirement provenance.

    Strong requirements contribute one point and weak requirements contribute
    half a point. Three points constitute complete intent for this policy.
    """

    evidence = 0.0
    for requirement in state.requirements:
        if requirement.source in _STRONG_SOURCES:
            evidence += 1.0
        elif requirement.source in _WEAK_SOURCES:
            evidence += 0.5
        else:
            raise ValueError(f"unsupported requirement source: {requirement.source!r}")
    return min(1.0, max(0.0, evidence / 3.0))


class FusionPolicy(Enum):
    """Supported immutable policies for selecting reciprocal-rank weights."""

    EQUAL = "equal"
    COMPLETENESS_ADAPTIVE = "completeness_adaptive"

    def choose(self, state: IntentState) -> RouteWeights:
        if self is FusionPolicy.EQUAL:
            return RouteWeights(bm25=0.5, dense=0.5)

        alpha_bm25 = 0.4 + 0.2 * intent_completeness(state)
        return RouteWeights(bm25=alpha_bm25, dense=1.0 - alpha_bm25)


EQUAL_RRF_POLICY = FusionPolicy.EQUAL
COMPLETENESS_ADAPTIVE_RRF_POLICY = FusionPolicy.COMPLETENESS_ADAPTIVE
