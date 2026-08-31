"""Deterministic protocol evidence ranking and label-free candidate beliefs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import lru_cache
from typing import Callable, Sequence

from conversational_search.intent import IntentState, Requirement
from conversational_search.protocol import (
    CandidateReplyStatus,
    ObservedProtocolEvent,
    ProductProtocolEvidence,
    ProtocolEventKind,
    classify_constraint,
    remaining_reply,
)


MAX_EXACT_EVIDENCE_CANDIDATES = 200
MIN_DENSE_TIEBREAK_MARGIN = 0.02

_STRONG_SOURCES = frozenset({"initial_explicit", "answer", "override"})
_CLUE_SOURCES = frozenset({"initial_tentative"})
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_LEADING_PROTOCOL_LABEL_RE = re.compile(
    r"^\s*(?:category|material|color|size|style|brand|budget|price|"
    r"feature|features|use[_ ]case|other)\s*:\s*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


class ExactEvidenceStatus(str, Enum):
    """Mutually exclusive outcomes of one bounded evidence pass."""

    APPLIED = "applied"
    FAIL_OPEN_ZERO_SUPPORT = "fail_open_zero_support"


class SemanticTieBreakPolicy(str, Enum):
    """Optional dense ordering applied only inside the exact best tier."""

    DISABLED = "disabled"
    DENSE_COMPLETE_BEST_TIER = "dense-complete-best-tier-v1"
    DENSE_CONFIDENT_BEST_TIER = "dense-confident-best-tier-v2"


DISABLED_SEMANTIC_TIEBREAK_POLICY = SemanticTieBreakPolicy.DISABLED
DENSE_COMPLETE_BEST_TIER_POLICY = (
    SemanticTieBreakPolicy.DENSE_COMPLETE_BEST_TIER
)
DENSE_CONFIDENT_BEST_TIER_POLICY = (
    SemanticTieBreakPolicy.DENSE_CONFIDENT_BEST_TIER
)


class SemanticTieBreakStatus(str, Enum):
    """Mutually exclusive outcomes of the bounded semantic tie-break."""

    DISABLED = "disabled"
    NO_SUPPORT = "no_support"
    SINGLETON = "singleton"
    INCOMPLETE_DENSE_COVERAGE = "incomplete_dense_coverage"
    SCORES_UNAVAILABLE = "scores_unavailable"
    LOW_CONFIDENCE = "low_confidence"
    UNCHANGED = "unchanged"
    REORDERED = "reordered"


@dataclass(frozen=True, slots=True)
class CandidateBelief:
    """One candidate's normalized rank prior in the best evidence tier."""

    parent_asin: str
    weight: float


@dataclass(frozen=True, slots=True)
class CandidateDisclosure:
    """Candidate-card values that the current intent has already disclosed."""

    parent_asin: str
    disclosed_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactEvidenceTrace:
    """Aggregate-only evidence facts; deliberately contains no IDs or text."""

    candidate_count: int
    category_compatible_count: int
    strong_disclosed_value_count: int
    tentative_clue_count: int
    no_additional_observation_count: int
    reply_consistent_count: int
    exclusion_violation_count: int
    consistent_support_count: int
    best_tier_count: int
    exact_phrase_candidate_count: int
    budget_compatible_count: int


@dataclass(frozen=True, slots=True)
class ExactEvidenceResult:
    """Immutable ordering, support, beliefs, disclosures, and safe trace."""

    status: ExactEvidenceStatus
    ranked_ids: tuple[str, ...]
    consistent_support_ids: tuple[str, ...]
    beliefs: tuple[CandidateBelief, ...]
    disclosures: tuple[CandidateDisclosure, ...]
    trace: ExactEvidenceTrace


@dataclass(frozen=True, slots=True)
class SemanticTieBreakResult:
    """Exact ranking plus the outcome of one semantic best-tier pass."""

    ranking: ExactEvidenceResult
    status: SemanticTieBreakStatus
    eligible_count: int


@dataclass(frozen=True, slots=True)
class _Atom:
    forms: frozenset[str]
    phrase_forms: tuple[tuple[str, ...], ...]
    is_budget: bool


@dataclass(frozen=True, slots=True)
class _Assessment:
    evidence: ProductProtocolEvidence
    protocol_consistent: bool
    evidence_tier: tuple[int, ...]
    sort_key: tuple[int, ...]
    disclosed_values: tuple[str, ...]
    category_compatible: bool
    reply_consistent: bool
    exclusion_violation: bool
    exact_phrase_affinity: int
    budget_compatibility: int


@dataclass(frozen=True, slots=True)
class _RequirementReplayStep:
    kind: str
    attribute: str | None
    values: tuple[str, ...]


def rank_exact_evidence(
    candidate_ids: Sequence[str],
    evidence: Sequence[ProductProtocolEvidence],
    state: IntentState,
    *,
    protocol_events: Sequence[ObservedProtocolEvent] = (),
) -> ExactEvidenceResult:
    """Rank a bounded selected-route slate with label-free protocol evidence.

    ``candidate_ids`` is the existing selected base-route order: hybrid when
    dense executes, or BM25 when it is intentionally skipped. It must align
    positionally with ``evidence`` so a stale or partial metadata lookup cannot
    silently attach one product's card to another product ID.
    """

    _, candidates = _validated_inputs(candidate_ids, evidence, state)
    events = _validated_protocol_events(protocol_events)
    override_index = next(
        (
            index
            for index in range(len(events) - 1, -1, -1)
            if events[index].kind is ProtocolEventKind.OVERRIDE
        ),
        None,
    )
    if override_index is None:
        replay_events = events
        atom_events = events
        override_event = None
        pre_override_reply_events: tuple[ObservedProtocolEvent, ...] = ()
    else:
        override_event = events[override_index]
        pre_override_reply_events = tuple(
            event
            for event in events[:override_index]
            if event.kind
            in {ProtocolEventKind.DISCLOSURE, ProtocolEventKind.NO_ADDITIONAL}
        )
        replay_events = events[override_index + 1 :]
        atom_events = ()
    requirement_replay = (
        () if replay_events else _requirement_replay_steps(state)
    )
    strong_atoms, clue_atoms = _intent_atoms(state, atom_events)
    excluded_atoms = _excluded_atoms(state)
    category = _normalized_value(state.category or "")
    popularity_ordinals = _bounded_popularity_ordinals(candidates)
    strong_non_budget_atoms = tuple(
        atom for atom in strong_atoms if not atom.is_budget
    )
    non_budget_atoms = tuple(
        atom
        for atom in (*strong_atoms, *clue_atoms)
        if not atom.is_budget
    )
    strong_budget_predicates = _budget_predicates(strong_atoms)
    budget_predicates = _budget_predicates((*strong_atoms, *clue_atoms))
    strong_budget_atom_count = sum(atom.is_budget for atom in strong_atoms)
    has_product_evidence = bool(
        strong_atoms
        or clue_atoms
        or excluded_atoms
        or any(
            event.kind
            in {
                ProtocolEventKind.DISCLOSURE,
                ProtocolEventKind.NO_ADDITIONAL,
            }
            for event in replay_events
        )
    )

    assessments: list[_Assessment] = []
    for original_rank, candidate in enumerate(candidates):
        card_values = (
            *candidate.card.hard_constraints,
            *candidate.card.soft_preferences,
        )
        card_forms = tuple(_value_forms(value) for value in card_values)
        exact_constraint_coverage = sum(
            any(atom.forms.intersection(forms) for forms in card_forms)
            for atom in strong_non_budget_atoms
        )
        strong_budget_compatibility = _budget_compatibility(
            strong_budget_predicates,
            candidate.price,
        )
        exact_budget_coverage = (
            strong_budget_atom_count
            if strong_budget_compatibility == 2
            else 0
        )
        exact_strong_coverage = (
            exact_constraint_coverage + exact_budget_coverage
        )
        all_constraints_covered = (
            exact_constraint_coverage == len(strong_non_budget_atoms)
            and strong_budget_compatibility != 0
        )
        candidate_category = _normalized_value(candidate.coarse_category)
        category_compatible = not category or category == candidate_category

        text_token_string = (
            _token_string(candidate.text)
            if non_budget_atoms or excluded_atoms
            else ""
        )
        exclusion_violation = any(
            _atom_occurs_in_text(atom, text_token_string)
            or any(atom.forms.intersection(forms) for forms in card_forms)
            for atom in excluded_atoms
        )
        reply_consistent, modeled_disclosed = _modeled_disclosures(
            candidate,
            replay_events,
            requirement_replay,
            override_event=override_event,
            pre_override_reply_events=pre_override_reply_events,
        )
        text_matches = tuple(
            _atom_occurs_in_text(atom, text_token_string)
            for atom in non_budget_atoms
        )
        exact_phrase_affinity = sum(text_matches)
        multi_constraint_coverage = sum(
            text_match
            or any(atom.forms.intersection(forms) for forms in card_forms)
            for atom, text_match in zip(non_budget_atoms, text_matches)
        )
        budget_compatibility = _budget_compatibility(
            budget_predicates,
            candidate.price,
        )
        hard_constraint_compatible = (
            not exclusion_violation
            and reply_consistent
            and strong_budget_compatibility != 0
        )
        evidence_tier = (
            int(not exclusion_violation),
            int(reply_consistent),
            int(strong_budget_compatibility != 0),
            int(all_constraints_covered),
            int(exact_strong_coverage >= 2),
            exact_strong_coverage,
            int(category_compatible) if has_product_evidence else 0,
            exact_constraint_coverage,
            exact_phrase_affinity,
            multi_constraint_coverage,
            budget_compatibility,
        )
        sort_key = (
            *evidence_tier,
            -original_rank,
            popularity_ordinals[original_rank] if has_product_evidence else 0,
        )
        assessments.append(
            _Assessment(
                evidence=candidate,
                protocol_consistent=(
                    hard_constraint_compatible
                    and category_compatible
                    and all_constraints_covered
                ),
                evidence_tier=evidence_tier,
                sort_key=sort_key,
                disclosed_values=modeled_disclosed,
                category_compatible=category_compatible,
                reply_consistent=reply_consistent,
                exclusion_violation=exclusion_violation,
                exact_phrase_affinity=exact_phrase_affinity,
                budget_compatibility=budget_compatibility,
            )
        )

    consistent = tuple(item for item in assessments if item.protocol_consistent)
    if not consistent:
        ordered = tuple(assessments)
        status = ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT
        best_tier: tuple[_Assessment, ...] = ()
    else:
        ordered = tuple(
            sorted(assessments, key=lambda item: item.sort_key, reverse=True)
        )
        status = ExactEvidenceStatus.APPLIED
        best_key = max(item.evidence_tier for item in consistent)
        best_tier = tuple(
            item
            for item in ordered
            if item.protocol_consistent and item.evidence_tier == best_key
        )

    beliefs = _harmonic_beliefs(
        tuple(item.evidence.parent_asin for item in best_tier)
    )
    consistent_ids = tuple(
        item.evidence.parent_asin
        for item in ordered
        if item.protocol_consistent
    )
    disclosures = tuple(
        CandidateDisclosure(item.evidence.parent_asin, item.disclosed_values)
        for item in ordered
    )
    trace = ExactEvidenceTrace(
        candidate_count=len(assessments),
        category_compatible_count=sum(
            item.category_compatible for item in assessments
        ),
        strong_disclosed_value_count=len(strong_atoms),
        tentative_clue_count=len(clue_atoms),
        no_additional_observation_count=sum(
            event.kind is ProtocolEventKind.NO_ADDITIONAL for event in events
        ),
        reply_consistent_count=sum(
            item.reply_consistent for item in assessments
        ),
        exclusion_violation_count=sum(
            item.exclusion_violation for item in assessments
        ),
        consistent_support_count=len(consistent),
        best_tier_count=len(best_tier),
        exact_phrase_candidate_count=sum(
            item.exact_phrase_affinity > 0 for item in assessments
        ),
        budget_compatible_count=sum(
            item.budget_compatibility == 2 for item in assessments
        ),
    )
    return ExactEvidenceResult(
        status=status,
        ranked_ids=tuple(item.evidence.parent_asin for item in ordered),
        consistent_support_ids=consistent_ids,
        beliefs=beliefs,
        disclosures=disclosures,
        trace=trace,
    )


def apply_dense_best_tier_tiebreak(
    ranking: ExactEvidenceResult,
    dense_ids: Sequence[str],
    *,
    dense_scores: Sequence[float] = (),
    policy: SemanticTieBreakPolicy = DISABLED_SEMANTIC_TIEBREAK_POLICY,
) -> SemanticTieBreakResult:
    """Use dense rank only to order a fully observed exact-evidence tie.

    The candidate set and every exact-evidence tier remain immutable. Dense
    evidence is allowed to influence the result only when it covers every
    member of the exact best tier, preventing missing dense hits from becoming
    implicit negative evidence.
    """

    if not isinstance(ranking, ExactEvidenceResult):
        raise TypeError("ranking must be ExactEvidenceResult")
    if not isinstance(policy, SemanticTieBreakPolicy):
        raise TypeError("policy must be SemanticTieBreakPolicy")
    if isinstance(dense_ids, (str, bytes)):
        raise TypeError("dense_ids must be a sequence of product IDs")
    dense = tuple(dense_ids)
    if len(dense) > MAX_EXACT_EVIDENCE_CANDIDATES:
        raise ValueError("too many dense IDs for semantic tie-breaking")
    if any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        for parent_asin in dense
    ):
        raise ValueError("dense IDs must be non-empty normalized strings")
    if len(set(dense)) != len(dense):
        raise ValueError("dense IDs must be unique")
    if not set(dense).issubset(ranking.ranked_ids):
        raise ValueError("dense IDs must be inside the exact candidate pool")
    if isinstance(dense_scores, (str, bytes)):
        raise TypeError("dense_scores must be a sequence of cosine scores")
    scores = tuple(dense_scores)
    if scores and len(scores) != len(dense):
        raise ValueError("dense scores must align with dense IDs")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not -1.0 <= float(score) <= 1.0
        for score in scores
    ):
        raise ValueError("dense scores must be finite cosine values")

    if policy is DISABLED_SEMANTIC_TIEBREAK_POLICY:
        return SemanticTieBreakResult(
            ranking,
            SemanticTieBreakStatus.DISABLED,
            0,
        )

    best_ids = tuple(belief.parent_asin for belief in ranking.beliefs)
    if (
        ranking.status is ExactEvidenceStatus.FAIL_OPEN_ZERO_SUPPORT
        or not best_ids
    ):
        return SemanticTieBreakResult(
            ranking,
            SemanticTieBreakStatus.NO_SUPPORT,
            0,
        )
    if len(best_ids) != len(set(best_ids)) or not set(best_ids).issubset(
        ranking.ranked_ids
    ):
        raise ValueError("exact best-tier beliefs are malformed")
    if len(best_ids) == 1:
        return SemanticTieBreakResult(
            ranking,
            SemanticTieBreakStatus.SINGLETON,
            1,
        )

    dense_rank = {
        parent_asin: rank for rank, parent_asin in enumerate(dense)
    }
    if any(parent_asin not in dense_rank for parent_asin in best_ids):
        return SemanticTieBreakResult(
            ranking,
            SemanticTieBreakStatus.INCOMPLETE_DENSE_COVERAGE,
            len(best_ids),
        )
    if policy is DENSE_CONFIDENT_BEST_TIER_POLICY:
        if not scores:
            return SemanticTieBreakResult(
                ranking,
                SemanticTieBreakStatus.SCORES_UNAVAILABLE,
                len(best_ids),
            )
        score_by_id = {
            parent_asin: float(score)
            for parent_asin, score in zip(dense, scores)
        }
        semantic_top = max(
            best_ids,
            key=lambda parent_asin: (
                score_by_id[parent_asin],
                -dense_rank[parent_asin],
            ),
        )
        if semantic_top == best_ids[0]:
            return SemanticTieBreakResult(
                ranking,
                SemanticTieBreakStatus.UNCHANGED,
                len(best_ids),
            )
        margin = score_by_id[semantic_top] - score_by_id[best_ids[0]]
        if margin < MIN_DENSE_TIEBREAK_MARGIN:
            return SemanticTieBreakResult(
                ranking,
                SemanticTieBreakStatus.LOW_CONFIDENCE,
                len(best_ids),
            )
        reordered_best = (
            semantic_top,
            *(item for item in best_ids if item != semantic_top),
        )
    else:
        reordered_best = tuple(
            sorted(best_ids, key=dense_rank.__getitem__)
        )
    if reordered_best == best_ids:
        return SemanticTieBreakResult(
            ranking,
            SemanticTieBreakStatus.UNCHANGED,
            len(best_ids),
        )

    best_set = frozenset(best_ids)
    reordered_iterator = iter(reordered_best)
    ranked_ids = tuple(
        next(reordered_iterator) if item in best_set else item
        for item in ranking.ranked_ids
    )
    consistent_set = frozenset(ranking.consistent_support_ids)
    consistent_support_ids = tuple(
        item for item in ranked_ids if item in consistent_set
    )
    disclosures_by_id = {
        disclosure.parent_asin: disclosure
        for disclosure in ranking.disclosures
    }
    if set(disclosures_by_id) != set(ranking.ranked_ids):
        raise ValueError("exact candidate disclosures are malformed")
    updated = ExactEvidenceResult(
        status=ranking.status,
        ranked_ids=tuple(ranked_ids),
        consistent_support_ids=consistent_support_ids,
        beliefs=_harmonic_beliefs(reordered_best),
        disclosures=tuple(disclosures_by_id[item] for item in ranked_ids),
        trace=ranking.trace,
    )
    return SemanticTieBreakResult(
        updated,
        SemanticTieBreakStatus.REORDERED,
        len(best_ids),
    )


def _harmonic_beliefs(parent_asins: Sequence[str]) -> tuple[CandidateBelief, ...]:
    raw_weights = tuple(
        1.0 / rank for rank in range(1, len(parent_asins) + 1)
    )
    total_weight = sum(raw_weights)
    return tuple(
        CandidateBelief(parent_asin, weight / total_weight)
        for parent_asin, weight in zip(parent_asins, raw_weights)
    )


def _validated_inputs(
    candidate_ids: Sequence[str],
    evidence: Sequence[ProductProtocolEvidence],
    state: IntentState,
) -> tuple[tuple[str, ...], tuple[ProductProtocolEvidence, ...]]:
    if isinstance(candidate_ids, (str, bytes)):
        raise TypeError("candidate_ids must be a sequence of product IDs")
    if isinstance(evidence, (str, bytes)):
        raise TypeError("evidence must be a sequence of ProductProtocolEvidence")
    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")

    ids = tuple(candidate_ids)
    candidates = tuple(evidence)
    if len(ids) > MAX_EXACT_EVIDENCE_CANDIDATES:
        raise ValueError(
            "exact evidence ranking supports at most "
            f"{MAX_EXACT_EVIDENCE_CANDIDATES} candidates"
        )
    if len(ids) != len(candidates):
        raise ValueError("candidate IDs and evidence must have equal length")
    if any(
        not isinstance(parent_asin, str)
        or not parent_asin
        or parent_asin != parent_asin.strip()
        for parent_asin in ids
    ):
        raise ValueError("candidate IDs must be non-empty normalized strings")
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if any(not isinstance(item, ProductProtocolEvidence) for item in candidates):
        raise TypeError("evidence must contain ProductProtocolEvidence values")
    evidence_ids = tuple(item.parent_asin for item in candidates)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("evidence product IDs must be unique")
    if ids != evidence_ids:
        raise ValueError("candidate IDs and evidence must align positionally")
    return ids, candidates


def _validated_protocol_events(
    events: Sequence[ObservedProtocolEvent],
) -> tuple[ObservedProtocolEvent, ...]:
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence):
        raise TypeError("protocol_events must be a sequence")
    retained = tuple(events)
    if len(retained) > 10:
        raise ValueError("at most ten protocol events are supported")
    if any(not isinstance(item, ObservedProtocolEvent) for item in retained):
        raise TypeError("protocol_events must contain ObservedProtocolEvent values")
    turns = tuple(item.turn for item in retained)
    if turns != tuple(sorted(set(turns))):
        raise ValueError("protocol event turns must be unique and increasing")
    if retained and retained[0].kind not in {
        ProtocolEventKind.INITIAL_BROWSING,
        ProtocolEventKind.INITIAL_EXPLICIT,
        ProtocolEventKind.INITIAL_TENTATIVE,
    }:
        raise ValueError("a protocol transcript must begin with an initial event")
    return retained


def _intent_atoms(
    state: IntentState,
    protocol_events: tuple[ObservedProtocolEvent, ...] = (),
) -> tuple[tuple[_Atom, ...], tuple[_Atom, ...]]:
    strong: list[_Atom] = []
    clues: list[_Atom] = []
    seen_strong: set[frozenset[str]] = set()
    seen_clues: set[frozenset[str]] = set()

    if protocol_events:
        override_observed = any(
            event.kind is ProtocolEventKind.OVERRIDE
            for event in protocol_events
        )
        for event in protocol_events:
            if event.kind in {
                ProtocolEventKind.INITIAL_EXPLICIT,
                ProtocolEventKind.OVERRIDE,
            }:
                destination = strong
                seen = seen_strong
            elif event.kind is ProtocolEventKind.DISCLOSURE and event.values:
                destination = strong
                seen = seen_strong
            elif (
                event.kind is ProtocolEventKind.INITIAL_TENTATIVE
                and not override_observed
            ):
                destination = clues
                seen = seen_clues
            else:
                continue
            for value in event.values:
                forms = _value_forms(value)
                if not forms or forms in seen:
                    continue
                seen.add(forms)
                phrase_forms = tuple(
                    dict.fromkeys(
                        tokens
                        for form in forms
                        if (tokens := _tokens(form))
                    )
                )
                attribute = event.attribute or classify_constraint(value)
                destination.append(
                    _Atom(
                        forms=forms,
                        phrase_forms=phrase_forms,
                        is_budget=attribute == "budget",
                    )
                )
        clues = [atom for atom in clues if atom.forms not in seen_strong]
        return tuple(strong), tuple(clues)

    for requirement in state.requirements:
        if not isinstance(requirement, Requirement):
            raise TypeError("state requirements must contain Requirement values")
        if requirement.strength == "hard" and requirement.source != "free_text":
            destination = strong
            seen = seen_strong
        elif requirement.strength == "soft" and requirement.source != "free_text":
            destination = clues
            seen = seen_clues
        else:
            continue
        for raw_value in requirement.value.split(";"):
            forms = _value_forms(raw_value)
            if not forms or forms in seen:
                continue
            seen.add(forms)
            phrase_forms = tuple(
                dict.fromkeys(tokens for value in forms if (tokens := _tokens(value)))
            )
            destination.append(
                _Atom(
                    forms=forms,
                    phrase_forms=phrase_forms,
                    is_budget=requirement.attribute == "budget",
                )
            )
    clues = [atom for atom in clues if atom.forms not in seen_strong]
    return tuple(strong), tuple(clues)


def _excluded_atoms(state: IntentState) -> tuple[_Atom, ...]:
    atoms: list[_Atom] = []
    seen: set[frozenset[str]] = set()
    for excluded in state.excluded:
        if not isinstance(excluded, str):
            raise TypeError("state exclusions must contain strings")
        for raw_value in excluded.split(";"):
            forms = _value_forms(raw_value)
            if not forms or forms in seen:
                continue
            seen.add(forms)
            phrase_forms = tuple(
                dict.fromkeys(
                    tokens for value in forms if (tokens := _tokens(value))
                )
            )
            atoms.append(
                _Atom(
                    forms=forms,
                    phrase_forms=phrase_forms,
                    is_budget=False,
                )
            )
    return tuple(atoms)


def _requirement_replay_steps(
    state: IntentState,
) -> tuple[_RequirementReplayStep, ...]:
    events = sorted(
        (
            (index, requirement)
            for index, requirement in enumerate(state.requirements)
            if requirement.strength == "hard"
            and requirement.source in _STRONG_SOURCES
        ),
        key=lambda item: (item[1].turn, item[0]),
    )
    steps: list[_RequirementReplayStep] = []
    index = 0
    while index < len(events):
        _, requirement = events[index]
        if requirement.source in {"initial_explicit", "override"}:
            steps.append(
                _RequirementReplayStep(
                    "initial",
                    requirement.attribute,
                    _split_normalized_requirement(requirement.value),
                )
            )
            index += 1
            continue

        grouped = [requirement]
        next_index = index + 1
        while next_index < len(events):
            _, following = events[next_index]
            if (
                following.source != "answer"
                or following.turn != requirement.turn
            ):
                break
            grouped.append(following)
            next_index += 1
        if any(
            item.attribute != requirement.attribute for item in grouped[1:]
        ):
            steps.append(
                _RequirementReplayStep("invalid", requirement.attribute, ())
            )
            break
        steps.append(
            _RequirementReplayStep(
                "answer",
                requirement.attribute,
                tuple(
                    value
                    for item in grouped
                    for value in _split_normalized_requirement(item.value)
                ),
            )
        )
        index = next_index
    return tuple(steps)


def _modeled_disclosures(
    candidate: ProductProtocolEvidence,
    protocol_events: tuple[ObservedProtocolEvent, ...],
    requirement_replay: tuple[_RequirementReplayStep, ...],
    *,
    override_event: ObservedProtocolEvent | None = None,
    pre_override_reply_events: tuple[ObservedProtocolEvent, ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    """Replay active protocol-shaped requirements against one candidate card."""

    if override_event is not None:
        hard = candidate.card.hard_constraints
        actual = tuple(
            _normalized_value(value) for value in override_event.values
        )
        expected = (_normalized_value(hard[0]),) if hard else ()
        if not hard or actual != expected:
            return False, ()
        disclosed = [hard[0]]
        for event in pre_override_reply_events:
            if event.kind is ProtocolEventKind.DISCLOSURE:
                matched = _project_prior_disclosure(
                    candidate,
                    event,
                    disclosed,
                )
                if matched is None:
                    return False, tuple(disclosed)
                for value in matched:
                    if value not in disclosed:
                        disclosed.append(value)
            elif event.kind is ProtocolEventKind.NO_ADDITIONAL:
                signature = remaining_reply(
                    candidate.card,
                    event.attribute,
                    disclosed,
                )
                if signature.status is not CandidateReplyStatus.NO_ADDITIONAL:
                    return False, tuple(disclosed)
        return _replay_protocol_events(
            candidate,
            protocol_events,
            initial_disclosed=tuple(disclosed),
        )

    if protocol_events:
        return _replay_protocol_events(candidate, protocol_events)

    hard = candidate.card.hard_constraints
    disclosed: list[str] = []
    for step in requirement_replay:
        if step.kind == "invalid":
            return False, tuple(disclosed)
        if step.kind == "initial":
            expected = (_normalized_value(hard[0]),) if hard else ()
            if not hard or step.values != expected:
                return False, tuple(disclosed)
            if hard[0] not in disclosed:
                disclosed.append(hard[0])
            continue
        signature = remaining_reply(
            candidate.card,
            step.attribute,
            disclosed,
        )
        expected = tuple(_normalized_value(value) for value in signature.values)
        if (
            signature.status is not CandidateReplyStatus.DISCLOSURE
            or step.values != expected
        ):
            return False, tuple(disclosed)
        for value in signature.values:
            if value not in disclosed:
                disclosed.append(value)
    return True, tuple(disclosed)


def _project_prior_disclosure(
    candidate: ProductProtocolEvidence,
    event: ObservedProtocolEvent,
    disclosed: list[str],
) -> tuple[str, ...] | None:
    """Map one retained serialized reply onto exact values in the new card."""

    payload = _normalized_value(event.serialized_reply_values or "")
    if not payload:
        return None
    attribute = event.attribute
    values = tuple(
        value
        for value in (
            *candidate.card.hard_constraints,
            *candidate.card.soft_preferences,
        )
        if attribute == "other" or classify_constraint(value) == attribute
    )
    sequences = tuple((value,) for value in values) + tuple(
        (values[left], values[right])
        for left in range(len(values))
        for right in range(left + 1, len(values))
    )
    for sequence in sequences:
        if _normalized_value("; ".join(sequence)) == payload:
            return sequence
    return None


def _replay_protocol_events(
    candidate: ProductProtocolEvidence,
    events: tuple[ObservedProtocolEvent, ...],
    *,
    initial_disclosed: tuple[str, ...] = (),
) -> tuple[bool, tuple[str, ...]]:
    hard = candidate.card.hard_constraints
    disclosed = list(initial_disclosed)
    for event in events:
        if event.kind in {
            ProtocolEventKind.INITIAL_BROWSING,
            ProtocolEventKind.INITIAL_TENTATIVE,
            ProtocolEventKind.BOUNDARY_DECLINE,
            ProtocolEventKind.NEED_ATTRIBUTE,
        }:
            continue
        if event.kind in {
            ProtocolEventKind.INITIAL_EXPLICIT,
            ProtocolEventKind.OVERRIDE,
        }:
            expected = (_normalized_value(hard[0]),) if hard else ()
            actual = tuple(_normalized_value(value) for value in event.values)
            if not hard or actual != expected:
                return False, tuple(disclosed)
            if hard[0] not in disclosed:
                disclosed.append(hard[0])
            continue

        signature = remaining_reply(
            candidate.card,
            event.attribute,
            disclosed,
        )
        if event.kind is ProtocolEventKind.DISCLOSURE:
            actual = _normalized_value(
                event.serialized_reply_values or ""
            )
            expected = _normalized_value("; ".join(signature.values))
            if (
                signature.status is not CandidateReplyStatus.DISCLOSURE
                or actual != expected
            ):
                return False, tuple(disclosed)
            for value in signature.values:
                if value not in disclosed:
                    disclosed.append(value)
        elif (
            event.kind is ProtocolEventKind.NO_ADDITIONAL
            and signature.status is not CandidateReplyStatus.NO_ADDITIONAL
        ):
            return False, tuple(disclosed)
    return True, tuple(disclosed)


def _split_normalized_requirement(value: str) -> tuple[str, ...]:
    return tuple(
        normalized
        for raw in value.split(";")
        if (normalized := _normalized_value(raw))
    )


def _normalized_value(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip(" -;,.\t\n").casefold()


@lru_cache(maxsize=4_096)
def _value_forms(value: str) -> frozenset[str]:
    normalized = _normalized_value(value)
    if not normalized:
        return frozenset()
    forms = {normalized}
    without_label = _normalized_value(
        _LEADING_PROTOCOL_LABEL_RE.sub("", value, count=1)
    )
    if without_label:
        forms.add(without_label)
    return frozenset(forms)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(_normalized_value(value)))


@lru_cache(maxsize=256)
def _token_string(value: str) -> str:
    tokens = _tokens(value)
    return f" {' '.join(tokens)} " if tokens else ""


def _atom_occurs_in_text(atom: _Atom, text: str) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in atom.phrase_forms)


def _contains_phrase(text: str, phrase: tuple[str, ...]) -> bool:
    return bool(text and phrase and f" {' '.join(phrase)} " in text)


def _bounded_popularity_ordinals(
    candidates: tuple[ProductProtocolEvidence, ...],
) -> tuple[int, ...]:
    observed = sorted(
        {item.popularity for item in candidates if item.popularity is not None}
    )
    ordinal = {value: index + 1 for index, value in enumerate(observed)}
    return tuple(
        0 if item.popularity is None else ordinal[item.popularity]
        for item in candidates
    )


def _budget_predicates(
    atoms: tuple[_Atom, ...],
) -> tuple[Callable[[Decimal], bool] | None, ...]:
    return tuple(
        _budget_predicate(next(iter(sorted(atom.forms))))
        for atom in atoms
        if atom.is_budget
    )


def _budget_compatibility(
    predicates: tuple[Callable[[Decimal], bool] | None, ...],
    raw_price: str | None,
) -> int:
    if not predicates:
        return 1
    price = _decimal(raw_price or "")
    if price is None:
        return 1

    outcomes: list[bool] = []
    unknown = False
    for predicate in predicates:
        if predicate is None:
            unknown = True
            continue
        outcomes.append(predicate(price))
    if any(not outcome for outcome in outcomes):
        return 0
    if unknown or not outcomes:
        return 1
    return 2


def _budget_predicate(value: str):
    numbers = tuple(
        number
        for raw in _NUMBER_RE.findall(value)
        if (number := _decimal(raw)) is not None
    )
    if not numbers:
        return None
    lowered = value.casefold()
    if len(numbers) >= 2 and (
        "between" in lowered
        or " to " in lowered
        or re.search(r"\d\s*[-\u2013\u2014]\s*\$?\d", lowered)
    ):
        lower, upper = sorted(numbers[:2])
        return lambda price: lower <= price <= upper
    threshold = numbers[0]
    if "<=" in lowered or any(
        marker in lowered
        for marker in ("up to", "at most", "no more than", "maximum", "max ")
    ):
        return lambda price: price <= threshold
    if "<" in lowered or any(
        marker in lowered for marker in ("under", "below", "less than")
    ):
        return lambda price: price < threshold
    if ">=" in lowered or any(
        marker in lowered for marker in ("at least", "minimum", "min ")
    ):
        return lambda price: price >= threshold
    if ">" in lowered or any(
        marker in lowered for marker in ("over", "above", "more than")
    ):
        return lambda price: price > threshold
    return lambda price: price == threshold


def _decimal(value: str) -> Decimal | None:
    match = _NUMBER_RE.search(value)
    if match is None:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None
