"""Deterministic, label-free Stage-A candidate reranking."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Sequence

from conversational_search.intent import IntentState
from conversational_search.profiles import (
    DEFAULT_PROFILE_RESIDUAL_WEIGHT,
    ProductTheme,
    ProfilePolicy,
    ProfilePrior,
)
from conversational_search.strategy import RouteWeights, intent_completeness


RRF_K = 60
MAX_CANDIDATE_TEXT_CHARACTERS = 32_768
MAX_CLAUSES = 32
MAX_CLAUSE_CHARACTERS = 1_024
MAX_CLAUSE_TOKENS = 64

_SIGNIFICANT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "some",
        "that",
        "the",
        "this",
        "to",
        "want",
        "with",
        "would",
        "you",
    }
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:category|material|color|size|style|brand|budget|price|"
    r"feature|features|use[_ ]case|other|search\s+clues?)\s*:\s*",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CandidateDocument:
    """One transient fused candidate and its bounded searchable document."""

    parent_asin: str
    text: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent_asin, str)
            or not self.parent_asin
            or self.parent_asin != self.parent_asin.strip()
        ):
            raise ValueError("parent_asin must be a non-empty normalized string")
        if not isinstance(self.text, str):
            raise TypeError("candidate text must be a string")
        if len(self.text) > MAX_CANDIDATE_TEXT_CHARACTERS:
            raise ValueError(
                "candidate text exceeds the Stage-A character limit"
            )


@dataclass(frozen=True, slots=True)
class RankingTrace:
    """Bounded Stage-A audit data without query, requirement, or document text."""

    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    beta: float
    observable_clause_count: int


@dataclass(frozen=True, slots=True)
class RankingResult:
    ranked_ids: tuple[str, ...]
    trace: RankingTrace


class ProfileResidualStatus(str, Enum):
    """Bounded result classes for aggregate-only profile health counters."""

    DISABLED = "disabled"
    NEUTRAL = "neutral"
    ACTIVE_REQUIREMENTS = "active_requirements"
    NO_REPRESENTED_THEME = "no_represented_theme"
    CONSTANT_SCORE = "constant_score"
    APPLIED = "applied"
    SCORING_FALLBACK = "scoring_fallback"


@dataclass(frozen=True, slots=True)
class ProfileRankingResult:
    """Transient profile decision wrapped around an ordinary Stage-A result."""

    ranking: RankingResult
    status: ProfileResidualStatus
    requested_theme_count: int
    represented_theme_count: int


class Bm25RescueStatus(str, Enum):
    """Mutually exclusive aggregate outcomes for the Phase 10 rescue."""

    ZERO_COMPLETENESS = "zero_completeness"
    EMPTY_BM25 = "bm25_unavailable_or_empty"
    NO_POSITIVE_UPLIFT = "no_positive_uplift"
    CONSTANT_UPLIFT = "constant_uplift"
    UNCHANGED_ORDER = "unchanged_order"
    REORDERED = "reordered"
    SCORING_FALLBACK = "scoring_fallback"


@dataclass(frozen=True, slots=True)
class Bm25RescueRankingResult:
    """One transient rescue decision with the composed profile outcome."""

    ranking: RankingResult
    status: Bm25RescueStatus
    profile_status: ProfileResidualStatus
    requested_theme_count: int
    represented_theme_count: int


class RouteRedundancyStatus(str, Enum):
    """Mutually exclusive outcomes for the Phase 12 route correction."""

    EMPTY = "empty_exact_baseline"
    SINGLE_ROUTE = "single_route_exact_baseline"
    DISJOINT = "disjoint_exact_baseline"
    IDENTICAL = "identical_order_exact_baseline"
    APPLIED = "redundancy_correction_applied"
    SCORING_FALLBACK = "scoring_fallback"


@dataclass(frozen=True, slots=True)
class RouteRedundancyRankingResult:
    """One transient Phase 12 result composed with the profile residual."""

    ranking: RankingResult
    status: RouteRedundancyStatus
    profile_status: ProfileResidualStatus
    requested_theme_count: int
    represented_theme_count: int


class RankingPolicy(Enum):
    """Supported immutable switches for a reversible Stage-A experiment."""

    FUSED_ONLY = "fused_only"
    STAGE_A = "stage_a"
    COMPLETENESS_BM25_RESCUE = "phase10-completeness-gated-bm25-rescue-v1"
    ROUTE_REDUNDANCY_CORRECTED = (
        "phase12-route-redundancy-corrected-stage-a-v1"
    )
    LEXICOGRAPHIC_EXACT_EVIDENCE = (
        "phase2-lexicographic-exact-evidence-v1"
    )
    IMPORTANCE_AWARE_SATISFACTION = (
        "importance-aware-satisfaction-lexicographic-v1"
    )


FUSED_ONLY_RANKING_POLICY = RankingPolicy.FUSED_ONLY
STAGE_A_RANKING_POLICY = RankingPolicy.STAGE_A
COMPLETENESS_BM25_RESCUE_RANKING_POLICY = (
    RankingPolicy.COMPLETENESS_BM25_RESCUE
)
ROUTE_REDUNDANCY_CORRECTED_RANKING_POLICY = (
    RankingPolicy.ROUTE_REDUNDANCY_CORRECTED
)
LEXICOGRAPHIC_EXACT_EVIDENCE_RANKING_POLICY = (
    RankingPolicy.LEXICOGRAPHIC_EXACT_EVIDENCE
)
IMPORTANCE_AWARE_SATISFACTION_RANKING_POLICY = (
    RankingPolicy.IMPORTANCE_AWARE_SATISFACTION
)


class _RouteEvidencePolicy(Enum):
    ADDITIVE_RRF = "additive_rrf"
    REDUNDANCY_CORRECTED = "redundancy_corrected"


@dataclass(frozen=True, slots=True)
class _AtomicClause:
    tokens: tuple[str, ...]
    provenance_weight: float


@dataclass(frozen=True, slots=True)
class _StageAComputation:
    """Internal Phase 7 result plus transient inputs needed by the residual."""

    ranking: RankingResult
    base_scores: tuple[float, ...]
    tokenized_documents: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _Bm25RescueComputation:
    """Phase 10-only inputs wrapped around an untouched Phase 9 result."""

    phase9: _StageAComputation
    completeness: float
    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    route_weights: RouteWeights


def _significant_tokens(value: str) -> tuple[str, ...]:
    folded = value.casefold()
    if folded.isascii():
        without_marks = folded
    else:
        # The token grammar is ASCII-only. Encoding the decomposed text in C
        # removes combining marks and other unsupported code points with the
        # same resulting token stream as a Python character-by-character pass.
        without_marks = (
            unicodedata.normalize("NFKD", folded)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    return tuple(
        token
        for token in _TOKEN_RE.findall(without_marks)
        if token not in _SIGNIFICANT_STOPWORDS
    )


_PROFILE_CUE_TEXT = MappingProxyType({
    ProductTheme.COMFORT: (
        "comfort",
        "comfortable",
        "cushioned",
        "cushioning",
        "ergonomic",
        "padded",
    ),
    ProductTheme.DURABILITY: (
        "durability",
        "durable",
        "rugged",
        "reinforced",
        "hard wearing",
        "long lasting",
    ),
    ProductTheme.PERFORMANCE: (
        "performance",
        "high performance",
        "performance focused",
        "technical performance",
    ),
    ProductTheme.WARMTH: (
        "warmth",
        "warm",
        "insulated",
        "insulation",
        "thermal",
        "fleece lined",
    ),
    ProductTheme.WEATHER_PROTECTION: (
        "weather protection",
        "weather resistant",
        "weatherproof",
        "water resistant",
        "waterproof",
        "wind resistant",
        "windproof",
        "rainproof",
    ),
    ProductTheme.LIGHTWEIGHT: (
        "lightweight",
        "light weight",
        "ultralight",
        "featherweight",
    ),
    ProductTheme.BREATHABILITY: (
        "breathability",
        "breathable",
        "ventilated",
        "ventilation",
        "airflow",
        "moisture wicking",
    ),
    ProductTheme.EASY_CARE: (
        "easy care",
        "low maintenance",
        "machine washable",
        "washable",
        "wrinkle resistant",
        "stain resistant",
    ),
    ProductTheme.VERSATILITY: (
        "versatility",
        "versatile",
        "multi purpose",
        "multipurpose",
        "all purpose",
        "convertible",
    ),
    ProductTheme.SUSTAINABILITY: (
        "sustainability",
        "sustainable",
        "eco friendly",
        "environmentally friendly",
        "recycled",
        "organic",
        "responsibly sourced",
    ),
})


def _profile_cue_masks() -> tuple[
    Mapping[str, int],
    Mapping[str, Mapping[str, int]],
]:
    single: dict[str, int] = {}
    paired: dict[str, dict[str, int]] = {}
    for theme, values in _PROFILE_CUE_TEXT.items():
        for value in values:
            cue = _significant_tokens(value)
            if not 1 <= len(cue) <= 2:
                raise RuntimeError("profile theme cues must contain one or two tokens")
            if len(cue) == 1:
                single[cue[0]] = single.get(cue[0], 0) | int(theme)
            else:
                second = paired.setdefault(cue[0], {})
                second[cue[1]] = second.get(cue[1], 0) | int(theme)
    return (
        MappingProxyType(single),
        MappingProxyType(
            {
                first: MappingProxyType(second)
                for first, second in paired.items()
            }
        ),
    )


_PROFILE_SINGLE_CUE_BITS, _PROFILE_PAIRED_CUE_BITS = _profile_cue_masks()
_ALL_PROFILE_THEME_BITS = sum(int(theme) for theme in ProductTheme)


def _candidate_theme_mask(
    tokens: tuple[str, ...],
    requested_themes: ProductTheme | None = None,
) -> ProductTheme:
    """Recognize fixed one- and two-token cues in one bounded document pass."""

    requested_bits = (
        _ALL_PROFILE_THEME_BITS
        if requested_themes is None
        else int(requested_themes)
    )
    mask = 0
    previous: str | None = None
    for token in tokens:
        mask |= _PROFILE_SINGLE_CUE_BITS.get(token, 0) & requested_bits
        if previous is not None:
            second = _PROFILE_PAIRED_CUE_BITS.get(previous)
            if second is not None:
                mask |= second.get(token, 0) & requested_bits
        if mask == requested_bits:
            break
        previous = token
    return ProductTheme(mask)


def recognize_candidate_themes(
    document_text: str,
    requested_themes: ProductTheme,
) -> ProductTheme:
    """Return bounded profile themes represented in one candidate document."""

    if not isinstance(document_text, str):
        raise TypeError("candidate document text must be a string")
    if len(document_text) > MAX_CANDIDATE_TEXT_CHARACTERS:
        raise ValueError("candidate document text exceeds the Stage-A limit")
    if not isinstance(requested_themes, ProductTheme):
        raise TypeError("requested_themes must be a ProductTheme")
    return _candidate_theme_mask(
        _significant_tokens(document_text),
        requested_themes,
    )


def _clauses(state: IntentState) -> tuple[_AtomicClause, ...]:
    clauses: list[_AtomicClause] = []

    def append_values(raw_value: str, provenance_weight: float) -> None:
        remaining = MAX_CLAUSES - len(clauses)
        if remaining <= 0:
            return
        values = raw_value.split(";", maxsplit=remaining)[:remaining]
        for value in values:
            normalized_value = _LEADING_LABEL_RE.sub("", value, count=1)
            tokens = _significant_tokens(
                normalized_value[:MAX_CLAUSE_CHARACTERS]
            )[:MAX_CLAUSE_TOKENS]
            if tokens:
                clauses.append(_AtomicClause(tokens, provenance_weight))

    if state.category:
        append_values(state.category, 0.5)

    for requirement in state.requirements:
        # CandidateDocument intentionally omits price. Scoring numeric budget
        # language against unrelated text (for example, "50 pieces") would be
        # a false signal, so typed budget clauses remain retrieval-only.
        if requirement.attribute == "budget":
            continue
        if requirement.strength == "hard":
            provenance_weight = 1.0
        elif requirement.strength == "soft":
            provenance_weight = 0.5
        else:
            raise ValueError(
                f"unsupported requirement strength: {requirement.strength!r}"
            )
        append_values(requirement.value, provenance_weight)
    return tuple(clauses)


def _contains_phrase(document: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    if width == 0:
        return False
    if width == 1:
        return phrase[0] in document
    first = phrase[0]
    for index in range(len(document) - width + 1):
        if document[index] != first:
            continue
        for offset in range(1, width):
            if document[index + offset] != phrase[offset]:
                break
        else:
            return True
    return False


def _clause_match(
    clause: _AtomicClause,
    document_tokens: tuple[str, ...],
    document_token_set: frozenset[str] | None = None,
    required_tokens: frozenset[str] | None = None,
) -> float:
    if _contains_phrase(document_tokens, clause.tokens):
        return 1.0
    required = (
        frozenset(clause.tokens) if required_tokens is None else required_tokens
    )
    if not required:
        return 0.0
    document_values = (
        frozenset(document_tokens)
        if document_token_set is None
        else document_token_set
    )
    present = required.intersection(document_values)
    coverage = len(present) / len(required)
    if coverage == 1.0:
        return 0.8
    return coverage * coverage


def _requirement_satisfaction(
    clauses: tuple[_AtomicClause, ...],
    tokenized_documents: tuple[tuple[str, ...], ...],
) -> tuple[tuple[float, ...], int]:
    represented: list[tuple[_AtomicClause, tuple[float, ...]]] = []
    document_token_sets: tuple[frozenset[str] | None, ...] = (
        tuple(frozenset(tokens) for tokens in tokenized_documents)
        if len(clauses) > 1
        else (None,) * len(tokenized_documents)
    )
    for clause in clauses:
        required_tokens = frozenset(clause.tokens)
        matches = tuple(
            _clause_match(
                clause,
                document_tokens,
                document_token_set,
                required_tokens,
            )
            for document_tokens, document_token_set in zip(
                tokenized_documents,
                document_token_sets,
            )
        )
        if any(match > 0.0 for match in matches):
            represented.append((clause, matches))

    denominator = sum(clause.provenance_weight for clause, _matches in represented)
    if denominator <= 0.0:
        return (0.0,) * len(tokenized_documents), 0
    return (
        tuple(
            sum(
                clause.provenance_weight * matches[index]
                for clause, matches in represented
            )
            / denominator
            for index in range(len(tokenized_documents))
        ),
        len(represented),
    )


def _ranked_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of product IDs")
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty string product IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicate product IDs")
    return result


def route_redundancy_coefficient(
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
) -> float:
    """Return bounded route overlap without scores, labels, or outcomes."""

    bm25 = _ranked_ids(bm25_ids, "bm25_ids")
    dense = _ranked_ids(dense_ids, "dense_ids")
    if len(bm25) > 100 or len(dense) > 100:
        raise ValueError("route redundancy supports at most 100 IDs per route")
    denominator = min(len(bm25), len(dense))
    if denominator == 0:
        return 0.0
    coefficient = len(set(bm25).intersection(dense)) / denominator
    if not math.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("route redundancy coefficient must be in [0, 1]")
    return coefficient


def _route_redundancy_status(
    bm25: tuple[str, ...],
    dense: tuple[str, ...],
) -> RouteRedundancyStatus:
    if not bm25 and not dense:
        return RouteRedundancyStatus.EMPTY
    if not bm25 or not dense:
        return RouteRedundancyStatus.SINGLE_ROUTE
    if set(bm25).isdisjoint(dense):
        return RouteRedundancyStatus.DISJOINT
    if bm25 == dense:
        return RouteRedundancyStatus.IDENTICAL
    return RouteRedundancyStatus.APPLIED


def _redundancy_corrected_route_scores(
    bm25: tuple[str, ...],
    dense: tuple[str, ...],
    fused: tuple[str, ...],
    route_weights: RouteWeights,
) -> dict[str, float]:
    """Compute the frozen rank-only submodular route-evidence sum."""

    if len(bm25) > 100 or len(dense) > 100 or len(fused) > 200:
        raise ValueError("route redundancy input exceeds its frozen bounds")
    if set(bm25) | set(dense) != set(fused):
        raise ValueError("fused_ids must equal the union of the two route lists")
    if route_weights.bm25 + route_weights.dense != 1.0:
        raise ValueError("route redundancy requires exactly normalized weights")

    coefficient = route_redundancy_coefficient(bm25, dense)
    bm25_ranks = {
        parent_asin: rank for rank, parent_asin in enumerate(bm25, 1)
    }
    dense_ranks = {
        parent_asin: rank for rank, parent_asin in enumerate(dense, 1)
    }
    rank_scale = RRF_K + 1
    scores: dict[str, float] = {}
    for parent_asin in fused:
        bm25_evidence = (
            route_weights.bm25
            * rank_scale
            / (RRF_K + bm25_ranks[parent_asin])
            if parent_asin in bm25_ranks
            else 0.0
        )
        dense_evidence = (
            route_weights.dense
            * rank_scale
            / (RRF_K + dense_ranks[parent_asin])
            if parent_asin in dense_ranks
            else 0.0
        )
        score = (
            bm25_evidence
            + dense_evidence
            - coefficient * min(bm25_evidence, dense_evidence)
        )
        if not math.isfinite(score) or not 0.0 < score <= 1.0:
            raise ValueError(
                "redundancy-corrected route evidence must be in (0, 1]"
            )
        scores[parent_asin] = score
    return scores


def _compute_stage_a(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
    route_evidence_policy: _RouteEvidencePolicy = (
        _RouteEvidencePolicy.ADDITIVE_RRF
    ),
) -> _StageAComputation:
    """Compute exact Phase 7 Stage-A scores and bounded transient tokens.

    Candidate documents must be aligned exactly to ``fused_ids``. Weighted RRF
    is reconstructed from the two route ranks and divided by the maximum score
    in this supplied candidate set.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if not isinstance(route_weights, RouteWeights):
        raise TypeError("route_weights must be RouteWeights")
    if not isinstance(route_evidence_policy, _RouteEvidencePolicy):
        raise TypeError("route_evidence_policy must be _RouteEvidencePolicy")
    if isinstance(candidate_documents, (str, bytes)):
        raise TypeError("candidate_documents must be a sequence")
    documents = tuple(candidate_documents)
    if any(not isinstance(item, CandidateDocument) for item in documents):
        raise TypeError("candidate_documents must contain CandidateDocument values")

    bm25 = _ranked_ids(bm25_ids, "bm25_ids")
    dense = _ranked_ids(dense_ids, "dense_ids")
    fused = _ranked_ids(fused_ids, "fused_ids")
    document_ids = tuple(item.parent_asin for item in documents)
    if document_ids != fused:
        raise ValueError("candidate documents must be aligned to fused_ids")
    if set(bm25) | set(dense) != set(fused):
        raise ValueError("fused_ids must equal the union of the two route lists")
    clauses = _clauses(state)
    beta = 0.20 + 0.25 * intent_completeness(state)
    if not documents:
        trace = RankingTrace(
            input_ids=(),
            output_ids=(),
            beta=beta,
            observable_clause_count=0,
        )
        return _StageAComputation(
            ranking=RankingResult(ranked_ids=(), trace=trace),
            base_scores=(),
            tokenized_documents=(),
        )

    if route_evidence_policy is _RouteEvidencePolicy.ADDITIVE_RRF:
        # This is the byte-for-byte Phase 9 arithmetic path. Keep it isolated
        # from the non-default Phase 12 formula so the protected policy does
        # not acquire new numerical behavior.
        bm25_ranks = {
            parent_asin: rank for rank, parent_asin in enumerate(bm25, 1)
        }
        dense_ranks = {
            parent_asin: rank for rank, parent_asin in enumerate(dense, 1)
        }
        raw_rrf: dict[str, float] = {}
        for parent_asin in fused:
            score = 0.0
            if parent_asin in bm25_ranks:
                score += route_weights.bm25 / (
                    RRF_K + bm25_ranks[parent_asin]
                )
            if parent_asin in dense_ranks:
                score += route_weights.dense / (
                    RRF_K + dense_ranks[parent_asin]
                )
            raw_rrf[parent_asin] = score
    else:
        raw_rrf = _redundancy_corrected_route_scores(
            bm25,
            dense,
            fused,
            route_weights,
        )
    maximum_rrf = max(raw_rrf.values())
    if maximum_rrf <= 0.0:
        raise ValueError("a non-empty fused union must have a positive RRF score")

    tokenized_documents = tuple(
        _significant_tokens(document.text) for document in documents
    )
    satisfaction, observable_clause_count = _requirement_satisfaction(
        clauses,
        tokenized_documents,
    )
    normalized_rrf = {
        parent_asin: score / maximum_rrf for parent_asin, score in raw_rrf.items()
    }
    scores = tuple(
        (
            document.parent_asin,
            (1.0 - beta) * normalized_rrf[document.parent_asin]
            + beta * satisfaction[index],
            index,
        )
        for index, document in enumerate(documents)
    )
    ranked_ids = tuple(
        parent_asin
        for parent_asin, _score, _index in sorted(
            scores,
            key=lambda item: (-item[1], item[2]),
        )
    )
    trace = RankingTrace(
        input_ids=fused,
        output_ids=ranked_ids,
        beta=beta,
        observable_clause_count=observable_clause_count,
    )
    return _StageAComputation(
        ranking=RankingResult(ranked_ids=ranked_ids, trace=trace),
        base_scores=tuple(score for _parent_asin, score, _index in scores),
        tokenized_documents=tokenized_documents,
    )


def rerank_stage_a(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
) -> RankingResult:
    """Rerank one fused union without model calls, labels, or retained text."""

    return _compute_stage_a(
        state,
        candidate_documents,
        bm25_ids=bm25_ids,
        dense_ids=dense_ids,
        fused_ids=fused_ids,
        route_weights=route_weights,
    ).ranking


def _profile_residual_scores(
    policy: ProfilePolicy,
    represented_prior: ProfilePrior,
    candidate_masks: tuple[ProductTheme, ...],
) -> tuple[float, ...]:
    """Keep residual calculation isolated so every fault fails closed."""

    return tuple(
        policy.residual(represented_prior, candidate_mask)
        for candidate_mask in candidate_masks
    )


def _apply_profile_residual(
    state: IntentState,
    computation: _StageAComputation,
    *,
    profile_prior: ProfilePrior,
    profile_policy: ProfilePolicy,
) -> ProfileRankingResult:
    """Apply the frozen profile residual to one supplied Stage-A computation.

    The caller-owned computation remains the exact fail-closed result.
    """
    base = computation.ranking

    if not isinstance(profile_policy, ProfilePolicy):
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.SCORING_FALLBACK,
            0,
            0,
        )
    if profile_policy is ProfilePolicy.DISABLED:
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.DISABLED,
            0,
            0,
        )
    if not isinstance(profile_prior, ProfilePrior):
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.SCORING_FALLBACK,
            0,
            0,
        )

    requested_theme_count = profile_prior.active_theme_count
    if profile_prior.is_neutral:
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.NEUTRAL,
            requested_theme_count,
            0,
        )
    if state.requirements:
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.ACTIVE_REQUIREMENTS,
            requested_theme_count,
            0,
        )

    try:
        candidate_masks = tuple(
            _candidate_theme_mask(tokens, profile_prior.theme_mask)
            for tokens in computation.tokenized_documents
        )
        if len(candidate_masks) != len(computation.base_scores):
            raise ValueError("profile candidate and base-score lengths differ")

        represented_mask = ProductTheme.NONE
        for candidate_mask in candidate_masks:
            represented_mask |= candidate_mask & profile_prior.theme_mask
        represented_theme_count = int(represented_mask).bit_count()
        if represented_theme_count == 0:
            return ProfileRankingResult(
                base,
                ProfileResidualStatus.NO_REPRESENTED_THEME,
                requested_theme_count,
                0,
            )

        represented_prior = ProfilePrior(represented_mask)
        profile_scores = _profile_residual_scores(
            profile_policy,
            represented_prior,
            candidate_masks,
        )
        if len(profile_scores) != len(computation.base_scores):
            raise ValueError("profile-score and base-score lengths differ")
        if any(
            not math.isfinite(score) or not 0.0 <= score <= 1.0
            for score in profile_scores
        ):
            raise ValueError("profile scores must be finite values in [0, 1]")
        if not profile_scores or all(
            score == profile_scores[0] for score in profile_scores[1:]
        ):
            return ProfileRankingResult(
                base,
                ProfileResidualStatus.CONSTANT_SCORE,
                requested_theme_count,
                represented_theme_count,
            )

        base_positions = {
            parent_asin: index
            for index, parent_asin in enumerate(base.ranked_ids)
        }
        final_scores = tuple(
            (1.0 - DEFAULT_PROFILE_RESIDUAL_WEIGHT) * base_score
            + DEFAULT_PROFILE_RESIDUAL_WEIGHT * profile_score
            for base_score, profile_score in zip(
                computation.base_scores,
                profile_scores,
            )
        )
        if any(not math.isfinite(score) for score in final_scores):
            raise ValueError("final profile-aware scores must be finite")

        document_ids = base.trace.input_ids
        if len(document_ids) != len(final_scores):
            raise ValueError("profile candidate and final-score lengths differ")
        ranked_ids = tuple(
            parent_asin
            for parent_asin, _score in sorted(
                zip(document_ids, final_scores),
                key=lambda item: (-item[1], base_positions[item[0]]),
            )
        )
        trace = RankingTrace(
            input_ids=base.trace.input_ids,
            output_ids=ranked_ids,
            beta=base.trace.beta,
            observable_clause_count=base.trace.observable_clause_count,
        )
        return ProfileRankingResult(
            RankingResult(ranked_ids=ranked_ids, trace=trace),
            ProfileResidualStatus.APPLIED,
            requested_theme_count,
            represented_theme_count,
        )
    except Exception:
        return ProfileRankingResult(
            base,
            ProfileResidualStatus.SCORING_FALLBACK,
            requested_theme_count,
            0,
        )


def rerank_stage_a_with_profile(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
    profile_prior: ProfilePrior,
    profile_policy: ProfilePolicy,
) -> ProfileRankingResult:
    """Apply the frozen bounded profile residual to exact Phase 7 Stage-A.

    Phase 7 is computed first and remains the fail-closed result. Profile
    evidence may only reorder that complete candidate set when the policy is
    enabled, the prior is recognized, and the conversation has no active
    requirements.
    """

    computation = _compute_stage_a(
        state,
        candidate_documents,
        bm25_ids=bm25_ids,
        dense_ids=dense_ids,
        fused_ids=fused_ids,
        route_weights=route_weights,
    )
    return _apply_profile_residual(
        state,
        computation,
        profile_prior=profile_prior,
        profile_policy=profile_policy,
    )


def rerank_stage_a_with_profile_and_route_redundancy(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
    profile_prior: ProfilePrior,
    profile_policy: ProfilePolicy,
) -> RouteRedundancyRankingResult:
    """Compose the frozen Phase 12 route correction with exact Phase 9.

    Exact Phase 9 arithmetic is used directly for empty, single-route,
    disjoint, and identical-order inputs. Any candidate-only validation or
    scoring failure also fails closed to a newly computed Phase 9 result.
    """

    phase9: _StageAComputation | None = None
    try:
        bm25 = _ranked_ids(bm25_ids, "bm25_ids")
        dense = _ranked_ids(dense_ids, "dense_ids")
        fused = _ranked_ids(fused_ids, "fused_ids")
        if len(bm25) > 100 or len(dense) > 100 or len(fused) > 200:
            raise ValueError("route redundancy input exceeds its frozen bounds")
        if set(bm25) | set(dense) != set(fused):
            raise ValueError(
                "fused_ids must equal the union of the two route lists"
            )
        status = _route_redundancy_status(bm25, dense)
        if status is RouteRedundancyStatus.APPLIED:
            candidate = _compute_stage_a(
                state,
                candidate_documents,
                bm25_ids=bm25,
                dense_ids=dense,
                fused_ids=fused,
                route_weights=route_weights,
                route_evidence_policy=(
                    _RouteEvidencePolicy.REDUNDANCY_CORRECTED
                ),
            )
        else:
            candidate = _compute_stage_a(
                state,
                candidate_documents,
                bm25_ids=bm25,
                dense_ids=dense,
                fused_ids=fused,
                route_weights=route_weights,
            )
            phase9 = candidate
    except Exception:
        phase9 = _compute_stage_a(
            state,
            candidate_documents,
            bm25_ids=bm25_ids,
            dense_ids=dense_ids,
            fused_ids=fused_ids,
            route_weights=route_weights,
        )
        candidate = phase9
        status = RouteRedundancyStatus.SCORING_FALLBACK

    profile_result = _apply_profile_residual(
        state,
        candidate,
        profile_prior=profile_prior,
        profile_policy=profile_policy,
    )
    if profile_result.status is ProfileResidualStatus.SCORING_FALLBACK:
        status = RouteRedundancyStatus.SCORING_FALLBACK
        if phase9 is None:
            phase9 = _compute_stage_a(
                state,
                candidate_documents,
                bm25_ids=bm25_ids,
                dense_ids=dense_ids,
                fused_ids=fused_ids,
                route_weights=route_weights,
            )
            profile_result = _apply_profile_residual(
                state,
                phase9,
                profile_prior=profile_prior,
                profile_policy=profile_policy,
            )

    return RouteRedundancyRankingResult(
        ranking=profile_result.ranking,
        status=status,
        profile_status=profile_result.status,
        requested_theme_count=profile_result.requested_theme_count,
        represented_theme_count=profile_result.represented_theme_count,
    )


def _apply_bm25_rescue(
    computation: _Bm25RescueComputation,
) -> tuple[_StageAComputation, Bm25RescueStatus]:
    """Apply the frozen one-sided rescue or return exact Phase 9.

    Route-derived rescue scores are reconstructed here rather than attached to
    the shared Stage-A computation.  The Phase 9 comparator therefore retains
    its original work path, and inconsistent-but-bounded synthetic vectors
    cannot enter the candidate scorer.
    """

    if not isinstance(computation, _Bm25RescueComputation):
        raise TypeError("computation must be _Bm25RescueComputation")
    phase9 = computation.phase9
    if not isinstance(phase9, _StageAComputation):
        raise TypeError("computation must contain exact Phase 9 Stage-A")
    base = phase9.ranking
    if not isinstance(base, RankingResult) or not isinstance(base.trace, RankingTrace):
        raise TypeError("Phase 9 computation must contain RankingResult and trace")

    completeness = computation.completeness
    beta = base.trace.beta
    if (
        isinstance(completeness, bool)
        or not isinstance(completeness, (int, float))
        or not math.isfinite(completeness)
        or not 0.0 <= completeness <= 1.0
        or isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or beta != 0.20 + 0.25 * completeness
        or type(base.trace.observable_clause_count) is not int
        or not 0 <= base.trace.observable_clause_count <= MAX_CLAUSES
    ):
        raise ValueError("rescue completeness, beta, or clause count is invalid")
    if completeness == 0.0:
        return phase9, Bm25RescueStatus.ZERO_COMPLETENESS

    bm25 = _ranked_ids(computation.bm25_ids, "bm25_ids")
    if not bm25:
        return phase9, Bm25RescueStatus.EMPTY_BM25

    if (
        type(base.ranked_ids) is not tuple
        or type(base.trace.input_ids) is not tuple
        or type(base.trace.output_ids) is not tuple
        or type(phase9.base_scores) is not tuple
        or type(phase9.tokenized_documents) is not tuple
    ):
        raise TypeError("Phase 9 rescue inputs must use immutable tuples")

    candidate_ids = _ranked_ids(base.trace.input_ids, "trace.input_ids")
    ranked_ids = _ranked_ids(base.ranked_ids, "ranking.ranked_ids")
    candidate_count = len(candidate_ids)
    if (
        base.trace.output_ids != ranked_ids
        or len(ranked_ids) != candidate_count
        or set(ranked_ids) != set(candidate_ids)
        or len(phase9.base_scores) != candidate_count
        or len(phase9.tokenized_documents) != candidate_count
    ):
        raise ValueError("Phase 9 computation is not aligned to the fused union")
    dense = _ranked_ids(computation.dense_ids, "dense_ids")
    if set(bm25) | set(dense) != set(candidate_ids):
        raise ValueError("rescue routes must equal the complete fused union")
    if not isinstance(computation.route_weights, RouteWeights):
        raise TypeError("rescue route_weights must be RouteWeights")

    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        for score in phase9.base_scores
    ):
        raise ValueError("Phase 9 scores must be finite values in [0, 1]")

    if not candidate_ids:
        raise ValueError("non-empty BM25 route requires a non-empty fused union")

    bm25_ranks = {parent_asin: rank for rank, parent_asin in enumerate(bm25, 1)}
    dense_ranks = {parent_asin: rank for rank, parent_asin in enumerate(dense, 1)}
    raw_rrf = tuple(
        (
            computation.route_weights.bm25
            / (RRF_K + bm25_ranks[parent_asin])
            if parent_asin in bm25_ranks
            else 0.0
        )
        + (
            computation.route_weights.dense
            / (RRF_K + dense_ranks[parent_asin])
            if parent_asin in dense_ranks
            else 0.0
        )
        for parent_asin in candidate_ids
    )
    maximum_rrf = max(raw_rrf)
    if not math.isfinite(maximum_rrf) or maximum_rrf <= 0.0:
        raise ValueError("rescue fused union must have a positive RRF score")
    input_positions = {
        parent_asin: index for index, parent_asin in enumerate(candidate_ids)
    }
    score_by_id = dict(zip(candidate_ids, phase9.base_scores))
    previous_score = math.inf
    previous_input_position = -1
    for parent_asin in ranked_ids:
        score = score_by_id[parent_asin]
        input_position = input_positions[parent_asin]
        if score > previous_score or (
            score == previous_score and input_position < previous_input_position
        ):
            raise ValueError("Phase 9 order is inconsistent with its base scores")
        previous_score = score
        previous_input_position = input_position

    final_scores: list[float] = []
    first_uplift: float | None = None
    any_positive_uplift = False
    constant_uplift = True
    for parent_asin, base_score, raw_score in zip(
        candidate_ids,
        phase9.base_scores,
        raw_rrf,
    ):
        rrf_score = raw_score / maximum_rrf
        retrieval_floor = (1.0 - beta) * rrf_score
        if not retrieval_floor <= base_score <= retrieval_floor + beta:
            raise ValueError(
                "Phase 9 score is inconsistent with bounded satisfaction"
            )
        bm25_score = (
            (RRF_K + 1) / (RRF_K + bm25_ranks[parent_asin])
            if parent_asin in bm25_ranks
            else 0.0
        )
        uplift = completeness * max(0.0, bm25_score - rrf_score)
        if uplift > 0.0:
            any_positive_uplift = True
        if first_uplift is None:
            first_uplift = uplift
        elif uplift != first_uplift:
            constant_uplift = False
        final_score = base_score + (1.0 - beta) * uplift
        if not math.isfinite(final_score) or not 0.0 <= final_score <= 1.0:
            raise ValueError(
                "rescued Stage-A scores must be finite values in [0, 1]"
            )
        final_scores.append(final_score)

    if not any_positive_uplift:
        return phase9, Bm25RescueStatus.NO_POSITIVE_UPLIFT
    if constant_uplift:
        return phase9, Bm25RescueStatus.CONSTANT_UPLIFT

    base_positions = {
        parent_asin: index for index, parent_asin in enumerate(ranked_ids)
    }
    rescued_ids = tuple(
        parent_asin
        for parent_asin, _score in sorted(
            zip(candidate_ids, final_scores),
            key=lambda item: (-item[1], base_positions[item[0]]),
        )
    )
    if rescued_ids == ranked_ids:
        return phase9, Bm25RescueStatus.UNCHANGED_ORDER

    trace = RankingTrace(
        input_ids=base.trace.input_ids,
        output_ids=rescued_ids,
        beta=base.trace.beta,
        observable_clause_count=base.trace.observable_clause_count,
    )
    return (
        _StageAComputation(
            ranking=RankingResult(ranked_ids=rescued_ids, trace=trace),
            base_scores=tuple(final_scores),
            tokenized_documents=phase9.tokenized_documents,
        ),
        Bm25RescueStatus.REORDERED,
    )


def rerank_stage_a_with_profile_and_bm25_rescue(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
    profile_prior: ProfilePrior,
    profile_policy: ProfilePolicy,
) -> Bm25RescueRankingResult:
    """Compose the Phase 10 rescue before the frozen Phase 9 profile residual."""

    phase7 = _compute_stage_a(
        state,
        candidate_documents,
        bm25_ids=bm25_ids,
        dense_ids=dense_ids,
        fused_ids=fused_ids,
        route_weights=route_weights,
    )
    try:
        rescue_computation = _Bm25RescueComputation(
            phase9=phase7,
            completeness=intent_completeness(state),
            bm25_ids=tuple(bm25_ids),
            dense_ids=tuple(dense_ids),
            route_weights=route_weights,
        )
        candidate, rescue_status = _apply_bm25_rescue(rescue_computation)
    except Exception:
        candidate = phase7
        rescue_status = Bm25RescueStatus.SCORING_FALLBACK

    profile_result = _apply_profile_residual(
        state,
        candidate,
        profile_prior=profile_prior,
        profile_policy=profile_policy,
    )
    if profile_result.status is ProfileResidualStatus.SCORING_FALLBACK:
        rescue_status = Bm25RescueStatus.SCORING_FALLBACK
        if candidate is not phase7:
            profile_result = _apply_profile_residual(
                state,
                phase7,
                profile_prior=profile_prior,
                profile_policy=profile_policy,
            )

    return Bm25RescueRankingResult(
        ranking=profile_result.ranking,
        status=rescue_status,
        profile_status=profile_result.status,
        requested_theme_count=profile_result.requested_theme_count,
        represented_theme_count=profile_result.represented_theme_count,
    )
