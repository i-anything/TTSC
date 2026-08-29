"""Deterministic, label-free Stage-A candidate reranking."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from conversational_search.intent import IntentState
from conversational_search.strategy import RouteWeights, intent_completeness


RRF_K = 60
MAX_CANDIDATE_TEXT_CHARACTERS = 32_768
MAX_CLAUSES = 32
MAX_CLAUSE_CHARACTERS = 1_024
MAX_CLAUSE_TOKENS = 64

_STRONG_SOURCES = frozenset({"initial_explicit", "answer", "override"})
_WEAK_SOURCES = frozenset({"initial_tentative", "free_text"})
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


class RankingPolicy(Enum):
    """Supported immutable switches for a reversible Stage-A experiment."""

    FUSED_ONLY = "fused_only"
    STAGE_A = "stage_a"


FUSED_ONLY_RANKING_POLICY = RankingPolicy.FUSED_ONLY
STAGE_A_RANKING_POLICY = RankingPolicy.STAGE_A


@dataclass(frozen=True, slots=True)
class _AtomicClause:
    tokens: tuple[str, ...]
    provenance_weight: float


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
        if requirement.source in _STRONG_SOURCES:
            provenance_weight = 1.0
        elif requirement.source in _WEAK_SOURCES:
            provenance_weight = 0.5
        else:
            raise ValueError(
                f"unsupported requirement source: {requirement.source!r}"
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


def rerank_stage_a(
    state: IntentState,
    candidate_documents: Sequence[CandidateDocument],
    *,
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    fused_ids: Sequence[str],
    route_weights: RouteWeights,
) -> RankingResult:
    """Rerank one fused union without model calls, labels, or retained text.

    Candidate documents must be aligned exactly to ``fused_ids``. Weighted RRF
    is reconstructed from the two route ranks and divided by the maximum score
    in this supplied candidate set.
    """

    if not isinstance(state, IntentState):
        raise TypeError("state must be IntentState")
    if not isinstance(route_weights, RouteWeights):
        raise TypeError("route_weights must be RouteWeights")
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
        return RankingResult(ranked_ids=(), trace=trace)

    bm25_ranks = {parent_asin: rank for rank, parent_asin in enumerate(bm25, 1)}
    dense_ranks = {parent_asin: rank for rank, parent_asin in enumerate(dense, 1)}
    raw_rrf: dict[str, float] = {}
    for parent_asin in fused:
        score = 0.0
        if parent_asin in bm25_ranks:
            score += route_weights.bm25 / (RRF_K + bm25_ranks[parent_asin])
        if parent_asin in dense_ranks:
            score += route_weights.dense / (RRF_K + dense_ranks[parent_asin])
        raw_rrf[parent_asin] = score
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
    return RankingResult(ranked_ids=ranked_ids, trace=trace)
