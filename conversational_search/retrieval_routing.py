"""Target-independent routing between lexical and hybrid retrieval.

The smart policy is deliberately conservative: hybrid retrieval remains the
default, and BM25 may stand alone only when frozen catalog evidence proves that
the parsed intent is fully structured and BM25 can retrieve at least one item
that jointly satisfies every exact constraint.  No evaluator labels, scenario
IDs, rank thresholds, or product-specific rules participate in the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from conversational_search.decision import exact_query_constraints
from conversational_search.intent import IntentState
from conversational_search.retrieval import (
    MAX_CANDIDATE_DOCUMENTS,
    PROTOCOL_EVIDENCE_CAPABILITY,
)


class RetrievalRoutingPolicy(str, Enum):
    """Reversible policies for deciding whether dense retrieval is needed."""

    ALWAYS_HYBRID = "always-hybrid-v1"
    SMART_HYBRID = "smart-bm25-first-complete-support-v2"


ALWAYS_HYBRID_RETRIEVAL_ROUTING_POLICY = (
    RetrievalRoutingPolicy.ALWAYS_HYBRID
)
SMART_HYBRID_RETRIEVAL_ROUTING_POLICY = RetrievalRoutingPolicy.SMART_HYBRID
MAX_BM25_ONLY_STRUCTURAL_SUPPORT = 3


class RetrievalRouteMode(str, Enum):
    """Pre-retrieval execution mode selected from observable evidence."""

    HYBRID = "hybrid"
    BM25_FIRST = "bm25_first"


class RetrievalRouteReason(str, Enum):
    """Fixed-cardinality, aggregate-safe explanations for routing decisions."""

    POLICY_REQUIRES_HYBRID = "policy_requires_hybrid"
    INTENT_FALLBACK = "intent_fallback"
    MISSING_CATEGORY = "missing_category"
    NO_EXACT_REQUIREMENTS = "no_exact_requirements"
    SEMANTIC_OR_SOFT_REQUIREMENT = "semantic_or_soft_requirement"
    OVERRIDE_OR_EXCLUSION = "override_or_exclusion"
    UNTYPED_REQUIREMENT = "untyped_requirement"
    EVIDENCE_CAPABILITY_UNAVAILABLE = "evidence_capability_unavailable"
    CATEGORY_UNRECOGNIZED = "category_unrecognized"
    PARTIAL_EXACT_SUPPORT = "partial_exact_support"
    NO_JOINT_STRUCTURAL_SUPPORT = "no_joint_structural_support"
    AMBIGUOUS_STRUCTURAL_SUPPORT = "ambiguous_structural_support"
    EVIDENCE_ERROR = "evidence_error"
    EXACT_STRUCTURAL_SUPPORT = "exact_structural_support"


_ROUTABLE_ATTRIBUTES = frozenset(
    {
        "material",
        "color",
        "size",
        "feature",
        "use_case",
        "brand",
        "style",
        "budget",
    }
)
_STABLE_EXACT_SOURCES = frozenset({"initial_explicit", "answer"})


@dataclass(frozen=True, slots=True)
class RetrievalRoutePlan:
    """One fail-open routing decision and its bounded structural support set."""

    policy: RetrievalRoutingPolicy
    mode: RetrievalRouteMode
    reason: RetrievalRouteReason
    structural_support_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RetrievalRoutingPolicy):
            raise TypeError("policy must be RetrievalRoutingPolicy")
        if not isinstance(self.mode, RetrievalRouteMode):
            raise TypeError("mode must be RetrievalRouteMode")
        if not isinstance(self.reason, RetrievalRouteReason):
            raise TypeError("reason must be RetrievalRouteReason")
        if not isinstance(self.structural_support_ids, tuple):
            raise TypeError("structural_support_ids must be a tuple")
        if (
            len(self.structural_support_ids) > MAX_CANDIDATE_DOCUMENTS
            or len(self.structural_support_ids)
            != len(set(self.structural_support_ids))
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in self.structural_support_ids
            )
        ):
            raise ValueError("structural support IDs are malformed")
        if self.mode is RetrievalRouteMode.BM25_FIRST:
            if (
                self.reason is not RetrievalRouteReason.EXACT_STRUCTURAL_SUPPORT
                or not self.structural_support_ids
            ):
                raise ValueError(
                    "BM25-first requires non-empty exact structural support"
                )
        elif self.structural_support_ids:
            raise ValueError("hybrid plans must not retain structural support IDs")

    @property
    def use_dense(self) -> bool:
        return self.mode is RetrievalRouteMode.HYBRID

    @property
    def dependency_key(self) -> str:
        """Return a bounded cache dependency without retaining query data."""

        return f"{self.policy.value}:{self.mode.value}:{self.reason.value}"


def _hybrid_plan(
    policy: RetrievalRoutingPolicy,
    reason: RetrievalRouteReason,
) -> RetrievalRoutePlan:
    return RetrievalRoutePlan(policy, RetrievalRouteMode.HYBRID, reason)


def plan_retrieval_route(
    policy: RetrievalRoutingPolicy,
    state: IntentState,
    retriever: object,
    *,
    intent_cacheable: bool,
) -> RetrievalRoutePlan:
    """Choose a conservative route using only parsed intent and catalog facts.

    Dense retrieval is skipped only after all semantic uncertainty gates pass
    and the immutable catalog index returns at least one joint exact match.
    The retriever performs the final post-BM25 support check; if BM25 misses the
    returned support set, dense retrieval runs as a rescue in the same search.
    """

    if not isinstance(policy, RetrievalRoutingPolicy):
        raise TypeError("policy must be RetrievalRoutingPolicy")
    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if type(intent_cacheable) is not bool:
        raise TypeError("intent_cacheable must be a boolean")
    if policy is RetrievalRoutingPolicy.ALWAYS_HYBRID:
        return _hybrid_plan(
            policy,
            RetrievalRouteReason.POLICY_REQUIRES_HYBRID,
        )
    if not intent_cacheable:
        return _hybrid_plan(policy, RetrievalRouteReason.INTENT_FALLBACK)
    if not state.category:
        return _hybrid_plan(policy, RetrievalRouteReason.MISSING_CATEGORY)
    if state.excluded or any(
        requirement.source == "override" for requirement in state.requirements
    ):
        return _hybrid_plan(policy, RetrievalRouteReason.OVERRIDE_OR_EXCLUSION)
    if not state.requirements:
        return _hybrid_plan(policy, RetrievalRouteReason.NO_EXACT_REQUIREMENTS)
    if any(
        requirement.strength != "hard"
        or requirement.source not in _STABLE_EXACT_SOURCES
        for requirement in state.requirements
    ):
        return _hybrid_plan(
            policy,
            RetrievalRouteReason.SEMANTIC_OR_SOFT_REQUIREMENT,
        )
    if any(
        requirement.attribute not in _ROUTABLE_ATTRIBUTES
        for requirement in state.requirements
    ):
        return _hybrid_plan(policy, RetrievalRouteReason.UNTYPED_REQUIREMENT)

    constraints = exact_query_constraints(state)
    if not constraints:
        return _hybrid_plan(policy, RetrievalRouteReason.NO_EXACT_REQUIREMENTS)
    try:
        capable = (
            getattr(retriever, "protocol_evidence_capability")
            is PROTOCOL_EVIDENCE_CAPABILITY
        )
    except Exception:
        capable = False
    if not capable:
        return _hybrid_plan(
            policy,
            RetrievalRouteReason.EVIDENCE_CAPABILITY_UNAVAILABLE,
        )

    try:
        if not retriever.protocol_category_exists(state.category):
            return _hybrid_plan(
                policy,
                RetrievalRouteReason.CATEGORY_UNRECOGNIZED,
            )
        exact_count = retriever.protocol_exact_constraint_count(
            state.category,
            constraints,
        )
        if type(exact_count) is not int or not 0 <= exact_count <= len(constraints):
            raise ValueError("exact constraint count is malformed")
        if exact_count != len(constraints):
            return _hybrid_plan(
                policy,
                RetrievalRouteReason.PARTIAL_EXACT_SUPPORT,
            )
        raw_support_ids = retriever.protocol_exact_candidates(
            state.category,
            constraints,
            limit=MAX_CANDIDATE_DOCUMENTS,
        )
        if not isinstance(raw_support_ids, tuple):
            raise TypeError("exact candidates must be a tuple")
        support_ids = tuple(dict.fromkeys(raw_support_ids))
        if (
            support_ids != raw_support_ids
            or len(support_ids) > MAX_CANDIDATE_DOCUMENTS
            or any(
                not isinstance(parent_asin, str) or not parent_asin
                for parent_asin in support_ids
            )
        ):
            raise ValueError("exact candidates are malformed")
    except Exception:
        return _hybrid_plan(policy, RetrievalRouteReason.EVIDENCE_ERROR)
    if not support_ids:
        return _hybrid_plan(
            policy,
            RetrievalRouteReason.NO_JOINT_STRUCTURAL_SUPPORT,
        )
    if len(support_ids) > MAX_BM25_ONLY_STRUCTURAL_SUPPORT:
        return _hybrid_plan(
            policy,
            RetrievalRouteReason.AMBIGUOUS_STRUCTURAL_SUPPORT,
        )
    return RetrievalRoutePlan(
        policy,
        RetrievalRouteMode.BM25_FIRST,
        RetrievalRouteReason.EXACT_STRUCTURAL_SUPPORT,
        support_ids,
    )
