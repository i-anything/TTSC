"""Ordinal importance-aware satisfaction ranking over bounded catalog evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum, IntEnum
from typing import Sequence

from conversational_search.intent import (
    IntentState,
    RequirementImportance,
    requirement_semantic_payload,
)
from conversational_search.profiles import NEUTRAL_PROFILE_PRIOR, ProductTheme, ProfilePrior
from conversational_search.protocol import ProductProtocolEvidence, classify_constraint
from conversational_search.ranking import CandidateDocument, recognize_candidate_themes


MAX_SATISFACTION_CANDIDATES = 200
MAX_SATISFACTION_REQUIREMENTS = 64
MAX_REQUIREMENT_CHARACTERS = 1_024

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:category|material|color|size|style|brand|budget|price|"
    r"feature|features|use[_ ]case|other)\s*:\s*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_MAXIMUM_BUDGET_RE = re.compile(
    r"(?P<operator>under|less\s+than|at\s+most|no\s+more\s+than|"
    r"maximum|max)\s*(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_MINIMUM_BUDGET_RE = re.compile(
    r"(?P<operator>over|more\s+than|at\s+least|minimum|min)\s*"
    r"(?P<amount>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_RANGE_BUDGET_RE = re.compile(
    r"(?:between|from)\s*(?P<lower>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?:and|to)\s*(?P<upper>\d+(?:,\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_EXCLUSIVE_ATTRIBUTES = frozenset({"brand"})
_PROTECTION_LEVELS = frozenset({"proof", "resistant", "resistance"})
_COMPOUND_PROTECTION_RE = re.compile(r"^([a-z0-9]+)(proof)$")
_NEGATION_TOKENS = frozenset({"no", "non", "not", "never", "without"})


class RequirementSatisfaction(IntEnum):
    """Ordered evidence state for one candidate/requirement pair."""

    VIOLATED = 0
    UNKNOWN = 1
    PARTIAL = 2
    FULL = 3


class ImportanceAwareStatus(str, Enum):
    APPLIED = "applied"
    NO_REQUIREMENTS = "no_requirements"


@dataclass(frozen=True, slots=True)
class RankedRequirement:
    """One deduplicated transient requirement used by the comparator."""

    kind: str
    attribute: str | None
    normalized_value: str
    importance: RequirementImportance
    profile_theme: ProductTheme = ProductTheme.NONE


@dataclass(frozen=True, slots=True)
class CandidateSatisfaction:
    parent_asin: str
    satisfactions: tuple[RequirementSatisfaction, ...]
    requirement_tier: tuple[int, ...]
    exact_affinity: int


@dataclass(frozen=True, slots=True)
class ImportanceAwareTrace:
    """Aggregate-only evidence counts safe for experiment diagnostics."""

    candidate_count: int
    requirement_count: int
    must_requirement_count: int
    should_requirement_count: int
    prefer_requirement_count: int
    exclusion_requirement_count: int
    budget_requirement_count: int
    profile_preference_count: int
    must_violation_candidate_count: int
    must_unknown_candidate_count: int
    all_must_full_candidate_count: int
    best_requirement_tier_count: int
    exact_affinity_candidate_count: int


@dataclass(frozen=True, slots=True)
class ImportanceAwareResult:
    status: ImportanceAwareStatus
    ranked_ids: tuple[str, ...]
    requirements: tuple[RankedRequirement, ...]
    assessments: tuple[CandidateSatisfaction, ...]
    fully_satisfied_best_ids: tuple[str, ...]
    trace: ImportanceAwareTrace


@dataclass(frozen=True, slots=True)
class _CandidateEvidence:
    protocol: ProductProtocolEvidence
    document_text: str
    normalized_document: str
    document_tokens: tuple[str, ...]
    structured_values: tuple[tuple[str, str, tuple[str, ...]], ...]
    profile_themes: ProductTheme


def rank_importance_aware_satisfaction(
    base_ranked_ids: Sequence[str],
    bm25_ids: Sequence[str],
    protocol_evidence: Sequence[ProductProtocolEvidence],
    candidate_documents: Sequence[CandidateDocument],
    state: IntentState,
    *,
    profile_prior: ProfilePrior = NEUTRAL_PROFILE_PRIOR,
) -> ImportanceAwareResult:
    """Rank without allowing lower-importance matches to compensate upward."""

    base, bm25, evidence, documents = _validated_inputs(
        base_ranked_ids,
        bm25_ids,
        protocol_evidence,
        candidate_documents,
        state,
        profile_prior,
    )
    requirements = _ranked_requirements(state, profile_prior)
    candidate_evidence = _candidate_evidence(evidence, documents, profile_prior)
    base_positions = {parent_asin: index for index, parent_asin in enumerate(base)}
    bm25_positions = {parent_asin: index for index, parent_asin in enumerate(bm25)}

    assessments: list[CandidateSatisfaction] = []
    sort_keys: dict[str, tuple[int, ...]] = {}
    for candidate in candidate_evidence:
        statuses = tuple(
            _satisfaction(requirement, candidate) for requirement in requirements
        )
        requirement_tier = _requirement_tier(requirements, statuses)
        exact_affinity = sum(
            status is RequirementSatisfaction.FULL
            and requirement.kind != "profile"
            for requirement, status in zip(requirements, statuses)
        )
        assessment = CandidateSatisfaction(
            candidate.protocol.parent_asin,
            statuses,
            requirement_tier,
            exact_affinity,
        )
        assessments.append(assessment)
        bm25_position = bm25_positions.get(candidate.protocol.parent_asin)
        sort_keys[candidate.protocol.parent_asin] = (
            *requirement_tier,
            exact_affinity,
            int(bm25_position is not None),
            -(bm25_position if bm25_position is not None else len(base)),
            -base_positions[candidate.protocol.parent_asin],
        )

    if not requirements:
        ordered_ids = base
        status = ImportanceAwareStatus.NO_REQUIREMENTS
        best_tier_count = len(base)
        fully_satisfied_best_ids = base
    else:
        ordered_ids = tuple(
            sorted(base, key=lambda parent_asin: sort_keys[parent_asin], reverse=True)
        )
        status = ImportanceAwareStatus.APPLIED
        best_requirement_tier = max(
            assessment.requirement_tier for assessment in assessments
        )
        best_tier_count = sum(
            assessment.requirement_tier == best_requirement_tier
            for assessment in assessments
        )
        assessment_by_id = {
            assessment.parent_asin: assessment for assessment in assessments
        }
        fully_satisfied_best_ids = tuple(
            parent_asin
            for parent_asin in ordered_ids
            if assessment_by_id[parent_asin].requirement_tier == best_requirement_tier
            and _all_must_full(
                requirements,
                assessment_by_id[parent_asin].satisfactions,
            )
        )

    must_indexes = tuple(
        index
        for index, requirement in enumerate(requirements)
        if requirement.importance is RequirementImportance.MUST
    )
    trace = ImportanceAwareTrace(
        candidate_count=len(base),
        requirement_count=len(requirements),
        must_requirement_count=len(must_indexes),
        should_requirement_count=sum(
            requirement.importance is RequirementImportance.SHOULD
            for requirement in requirements
        ),
        prefer_requirement_count=sum(
            requirement.importance is RequirementImportance.PREFER
            for requirement in requirements
        ),
        exclusion_requirement_count=sum(
            requirement.kind == "exclusion" for requirement in requirements
        ),
        budget_requirement_count=sum(
            requirement.attribute == "budget" for requirement in requirements
        ),
        profile_preference_count=sum(
            requirement.kind == "profile" for requirement in requirements
        ),
        must_violation_candidate_count=sum(
            any(
                assessment.satisfactions[index]
                is RequirementSatisfaction.VIOLATED
                for index in must_indexes
            )
            for assessment in assessments
        ),
        must_unknown_candidate_count=sum(
            any(
                assessment.satisfactions[index]
                is RequirementSatisfaction.UNKNOWN
                for index in must_indexes
            )
            for assessment in assessments
        ),
        all_must_full_candidate_count=sum(
            _all_must_full(requirements, assessment.satisfactions)
            for assessment in assessments
        ),
        best_requirement_tier_count=best_tier_count,
        exact_affinity_candidate_count=sum(
            assessment.exact_affinity > 0 for assessment in assessments
        ),
    )
    return ImportanceAwareResult(
        status,
        ordered_ids,
        requirements,
        tuple(assessments),
        fully_satisfied_best_ids,
        trace,
    )


def _validated_inputs(
    base_ranked_ids: Sequence[str],
    bm25_ids: Sequence[str],
    protocol_evidence: Sequence[ProductProtocolEvidence],
    candidate_documents: Sequence[CandidateDocument],
    state: IntentState,
    profile_prior: ProfilePrior,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[ProductProtocolEvidence, ...],
    tuple[CandidateDocument, ...],
]:
    if not isinstance(state, IntentState):
        raise TypeError("state must be an IntentState")
    if not isinstance(profile_prior, ProfilePrior):
        raise TypeError("profile_prior must be a ProfilePrior")
    for values, name in (
        (base_ranked_ids, "base_ranked_ids"),
        (bm25_ids, "bm25_ids"),
        (protocol_evidence, "protocol_evidence"),
        (candidate_documents, "candidate_documents"),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{name} must be a sequence")
    base = tuple(base_ranked_ids)
    bm25 = tuple(bm25_ids)
    evidence = tuple(protocol_evidence)
    documents = tuple(candidate_documents)
    if (
        not base
        or len(base) > MAX_SATISFACTION_CANDIDATES
        or len(base) != len(set(base))
        or any(not isinstance(value, str) or not value for value in base)
    ):
        raise ValueError("base candidate IDs are invalid")
    if len(bm25) != len(set(bm25)) or any(
        not isinstance(value, str) or not value for value in bm25
    ):
        raise ValueError("BM25 candidate IDs are invalid")
    if not set(bm25).issubset(base):
        raise ValueError("BM25 candidates must be inside the base pool")
    if len(evidence) != len(base) or any(
        not isinstance(value, ProductProtocolEvidence) for value in evidence
    ):
        raise ValueError("protocol evidence must align with the base pool")
    if tuple(value.parent_asin for value in evidence) != base:
        raise ValueError("protocol evidence is positionally misaligned")
    if len(documents) != len(base) or any(
        not isinstance(value, CandidateDocument) for value in documents
    ):
        raise ValueError("candidate documents must cover the base pool")
    document_ids = tuple(value.parent_asin for value in documents)
    if len(set(document_ids)) != len(base) or set(document_ids) != set(base):
        raise ValueError("candidate documents do not match the base pool")
    return base, bm25, evidence, documents


def _ranked_requirements(
    state: IntentState,
    profile_prior: ProfilePrior,
) -> tuple[RankedRequirement, ...]:
    ordered: list[RankedRequirement] = []
    positions: dict[tuple[str, str | None, str, int], int] = {}

    def add(requirement: RankedRequirement) -> None:
        key = (
            requirement.kind,
            requirement.attribute,
            requirement.normalized_value,
            int(requirement.profile_theme),
        )
        current_index = positions.get(key)
        if current_index is None:
            if len(ordered) >= MAX_SATISFACTION_REQUIREMENTS:
                raise ValueError("too many satisfaction requirements")
            positions[key] = len(ordered)
            ordered.append(requirement)
            return
        current = ordered[current_index]
        if _importance_rank(requirement.importance) > _importance_rank(
            current.importance
        ):
            ordered[current_index] = requirement

    if state.category:
        category = _normalized_value(state.category)
        if category:
            add(
                RankedRequirement(
                    "category",
                    "category",
                    category,
                    RequirementImportance.MUST,
                )
            )

    for requirement in state.requirements:
        if not isinstance(requirement.importance, RequirementImportance):
            raise ValueError("requirement importance is unavailable")
        parts = tuple(
            value.strip()
            for value in requirement.value.split(";")
            if value.strip()
        ) or (requirement.value,)
        for part in parts:
            attribute = requirement.attribute
            if attribute in {None, "other"}:
                attribute = classify_constraint(part)
            normalized = _normalized_value(part)
            if not normalized:
                continue
            add(
                RankedRequirement(
                    "positive",
                    attribute,
                    normalized,
                    requirement.importance,
                )
            )

    for raw_value in state.excluded:
        parts = tuple(
            value.strip() for value in raw_value.split(";") if value.strip()
        ) or (raw_value,)
        for part in parts:
            normalized = _normalized_value(part)
            if normalized:
                add(
                    RankedRequirement(
                        "exclusion",
                        classify_constraint(part),
                        normalized,
                        RequirementImportance.MUST,
                    )
                )

    for theme in ProductTheme:
        if (
            theme is not ProductTheme.NONE
            and profile_prior.theme_mask & theme
        ):
            add(
                RankedRequirement(
                    "profile",
                    "profile",
                    theme.name.casefold(),
                    RequirementImportance.PREFER,
                    theme,
                )
            )
    return tuple(ordered)


def _candidate_evidence(
    evidence: tuple[ProductProtocolEvidence, ...],
    documents: tuple[CandidateDocument, ...],
    profile_prior: ProfilePrior,
) -> tuple[_CandidateEvidence, ...]:
    document_by_id = {document.parent_asin: document for document in documents}
    result: list[_CandidateEvidence] = []
    for item in evidence:
        document = document_by_id[item.parent_asin]
        structured: list[tuple[str, str, tuple[str, ...]]] = []
        for value in (*item.card.hard_constraints, *item.card.soft_preferences):
            normalized = _normalized_value(value)
            if normalized:
                structured.append(
                    (classify_constraint(value), normalized, _tokens(normalized))
                )
        store_match = re.search(
            r"(?:^|\n)Store:\s*(?P<store>[^\n]+)",
            document.text,
            re.IGNORECASE,
        )
        if store_match is not None:
            store = _normalized_value(store_match.group("store"))
            if store:
                structured.append(("brand", store, _tokens(store)))
        normalized_document = _normalized_text(document.text)
        result.append(
            _CandidateEvidence(
                item,
                document.text,
                normalized_document,
                _tokens(normalized_document),
                tuple(structured),
                recognize_candidate_themes(
                    document.text,
                    profile_prior.theme_mask,
                ),
            )
        )
    return tuple(result)


def _satisfaction(
    requirement: RankedRequirement,
    candidate: _CandidateEvidence,
) -> RequirementSatisfaction:
    if requirement.kind == "profile":
        return (
            RequirementSatisfaction.FULL
            if candidate.profile_themes & requirement.profile_theme
            else RequirementSatisfaction.UNKNOWN
        )
    if requirement.kind == "category":
        candidate_category = _normalized_value(candidate.protocol.coarse_category)
        if candidate_category == requirement.normalized_value:
            return RequirementSatisfaction.FULL
        if _partially_compatible(
            _tokens(requirement.normalized_value),
            _tokens(candidate_category),
        ):
            return RequirementSatisfaction.PARTIAL
        return (
            RequirementSatisfaction.VIOLATED
            if candidate_category
            else RequirementSatisfaction.UNKNOWN
        )
    if requirement.kind == "exclusion":
        positive, negated = _phrase_evidence(
            _tokens(requirement.normalized_value),
            candidate.document_tokens,
        )
        if positive:
            return RequirementSatisfaction.VIOLATED
        if negated:
            return RequirementSatisfaction.FULL
        typed_values = tuple(
            (value, tokens)
            for attribute, value, tokens in candidate.structured_values
            if attribute == requirement.attribute
        )
        if any(
            value == requirement.normalized_value
            or _partially_compatible(
                _tokens(requirement.normalized_value),
                tokens,
            )
            for value, tokens in typed_values
        ):
            return RequirementSatisfaction.VIOLATED
        if requirement.attribute in _EXCLUSIVE_ATTRIBUTES and typed_values:
            return RequirementSatisfaction.FULL
        return RequirementSatisfaction.UNKNOWN
    if requirement.attribute == "budget":
        return _budget_satisfaction(
            requirement.normalized_value,
            candidate.protocol.price,
        )

    required_tokens = _tokens(requirement.normalized_value)
    positive, negated = _phrase_evidence(
        required_tokens,
        candidate.document_tokens,
    )
    if positive:
        return RequirementSatisfaction.FULL
    if negated:
        return RequirementSatisfaction.VIOLATED
    typed_values = tuple(
        (value, tokens)
        for attribute, value, tokens in candidate.structured_values
        if attribute == requirement.attribute
    )
    if any(value == requirement.normalized_value for value, _tokens_value in typed_values):
        return RequirementSatisfaction.FULL
    if any(
        _partially_compatible(required_tokens, candidate_tokens)
        for _value, candidate_tokens in typed_values
    ):
        return RequirementSatisfaction.PARTIAL
    if requirement.attribute in _EXCLUSIVE_ATTRIBUTES and typed_values:
        return RequirementSatisfaction.VIOLATED
    return RequirementSatisfaction.UNKNOWN


def _budget_satisfaction(
    requirement_value: str,
    raw_price: str | None,
) -> RequirementSatisfaction:
    if raw_price is None:
        return RequirementSatisfaction.UNKNOWN
    price = _first_decimal(raw_price)
    if price is None:
        return RequirementSatisfaction.UNKNOWN
    price_range = _RANGE_BUDGET_RE.search(requirement_value)
    if price_range is not None:
        lower = _decimal(price_range.group("lower"))
        upper = _decimal(price_range.group("upper"))
        if lower is None or upper is None or lower > upper:
            return RequirementSatisfaction.UNKNOWN
        return (
            RequirementSatisfaction.FULL
            if lower <= price <= upper
            else RequirementSatisfaction.VIOLATED
        )
    maximum = _MAXIMUM_BUDGET_RE.search(requirement_value)
    if maximum is not None:
        limit = _decimal(maximum.group("amount"))
        if limit is None:
            return RequirementSatisfaction.UNKNOWN
        strict = maximum.group("operator").casefold() in {"under", "less than"}
        satisfied = price < limit if strict else price <= limit
        return (
            RequirementSatisfaction.FULL
            if satisfied
            else RequirementSatisfaction.VIOLATED
        )
    minimum = _MINIMUM_BUDGET_RE.search(requirement_value)
    if minimum is not None:
        limit = _decimal(minimum.group("amount"))
        if limit is None:
            return RequirementSatisfaction.UNKNOWN
        strict = minimum.group("operator").casefold() in {"over", "more than"}
        satisfied = price > limit if strict else price >= limit
        return (
            RequirementSatisfaction.FULL
            if satisfied
            else RequirementSatisfaction.VIOLATED
        )
    requested = _first_decimal(requirement_value)
    if requested is None:
        return RequirementSatisfaction.UNKNOWN
    return (
        RequirementSatisfaction.FULL
        if price == requested
        else RequirementSatisfaction.UNKNOWN
    )


def _requirement_tier(
    requirements: tuple[RankedRequirement, ...],
    statuses: tuple[RequirementSatisfaction, ...],
) -> tuple[int, ...]:
    grouped = {
        importance: tuple(
            int(status)
            for requirement, status in zip(requirements, statuses)
            if requirement.importance is importance
        )
        for importance in RequirementImportance
    }
    must = grouped[RequirementImportance.MUST]
    should = grouped[RequirementImportance.SHOULD]
    prefer = grouped[RequirementImportance.PREFER]
    def tier(values: tuple[int, ...]) -> tuple[int, int, int]:
        return (
            int(
                all(
                    value > int(RequirementSatisfaction.VIOLATED)
                    for value in values
                )
            ),
            min(values, default=int(RequirementSatisfaction.FULL)),
            sum(values),
        )

    return (*tier(must), *tier(should), *tier(prefer))


def _all_must_full(
    requirements: tuple[RankedRequirement, ...],
    statuses: tuple[RequirementSatisfaction, ...],
) -> bool:
    return all(
        status is RequirementSatisfaction.FULL
        for requirement, status in zip(requirements, statuses)
        if requirement.importance is RequirementImportance.MUST
    )


def _importance_rank(value: RequirementImportance) -> int:
    return {
        RequirementImportance.PREFER: 1,
        RequirementImportance.SHOULD: 2,
        RequirementImportance.MUST: 3,
    }[value]


def _normalized_value(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("requirement values must be strings")
    if len(value) > MAX_REQUIREMENT_CHARACTERS:
        raise ValueError("requirement value exceeds the satisfaction limit")
    without_label = _LEADING_LABEL_RE.sub("", value, count=1)
    return _normalized_text(requirement_semantic_payload(without_label))


def _normalized_text(value: str) -> str:
    value = value.replace("<=", " at most ").replace(">=", " at least ")
    folded = (
        unicodedata.normalize("NFKD", value.casefold())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return " ".join(_TOKEN_RE.findall(folded))


def _tokens(value: str) -> tuple[str, ...]:
    expanded: list[str] = []
    for token in _TOKEN_RE.findall(value):
        compound = _COMPOUND_PROTECTION_RE.fullmatch(token)
        if compound is None:
            expanded.append(token)
        else:
            expanded.extend(compound.groups())
    return tuple(expanded)


def _phrase_evidence(
    phrase: tuple[str, ...],
    document: tuple[str, ...],
) -> tuple[bool, bool]:
    if not phrase or len(phrase) > len(document):
        return False, False
    width = len(phrase)
    positive = False
    negated = False
    for index in range(len(document) - width + 1):
        if document[index : index + width] != phrase:
            continue
        prior = document[max(0, index - 2) : index]
        is_negated = bool(
            set(prior).intersection(_NEGATION_TOKENS)
            or tuple(prior[-2:]) == ("free", "of")
        )
        negated = negated or is_negated
        positive = positive or not is_negated
    return positive, negated


def _partially_compatible(
    required: tuple[str, ...],
    candidate: tuple[str, ...],
) -> bool:
    required_set = frozenset(required)
    candidate_set = frozenset(candidate)
    if not required_set or not candidate_set or required_set == candidate_set:
        return False
    if required_set < candidate_set or candidate_set < required_set:
        return True
    shared_base = required_set.intersection(candidate_set) - _PROTECTION_LEVELS
    return bool(
        shared_base
        and required_set.intersection(_PROTECTION_LEVELS)
        and candidate_set.intersection(_PROTECTION_LEVELS)
    )


def _first_decimal(value: str) -> Decimal | None:
    match = _NUMBER_RE.search(value)
    return _decimal(match.group(0)) if match is not None else None


def _decimal(value: str) -> Decimal | None:
    try:
        parsed = Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None
