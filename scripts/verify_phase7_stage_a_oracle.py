"""Fixed-seed synthetic drift oracle for the frozen Phase 7 Stage-A ranker."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Iterator, Sequence, TypeVar

from conversational_search.intent import IntentState, Requirement
from conversational_search.ranking import (
    CandidateDocument,
    RankingResult,
    rerank_stage_a,
)
from conversational_search.strategy import RouteWeights


ORACLE_CASES = 1_000
ORACLE_SEED = 0x5A17A7E
MAX_CANDIDATES_PER_CASE = 12
MAX_REQUIREMENTS_PER_CASE = 6
EXPECTED_SHA256 = "ae216ca5267a3283fd95d7e1fac1aae060a89bef0ea43957d524f9011434a48a"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SOURCES = (
    "initial_explicit",
    "initial_tentative",
    "answer",
    "override",
    "free_text",
)
_CATEGORIES = (
    None,
    "trail shoes",
    "Travel Bags",
    "café gear",
    "home office; desk gear",
    "running apparel",
)
_REQUIREMENTS = (
    ("Color: deep red", "color"),
    ("Material: organic cotton", "material"),
    ("Feature: waterproof shell", "feature"),
    ("Use case: winter travel", "use_case"),
    ("Style: low profile", "style"),
    ("wide fit", "size"),
    ("noise cancelling", "feature"),
    ("rugged outdoor; easy care", "other"),
    ("Budget: under 50", "budget"),
    ("unrepresented quartz signal", "other"),
)
_DOCUMENT_FRAGMENTS = (
    "deep-red finish",
    "red finish with a deep tone",
    "organic cotton lining",
    "waterproof shell",
    "shell built for wet weather",
    "winter travel essential",
    "low profile design",
    "wide fit",
    "noise cancelling system",
    "rugged outdoor construction",
    "easy-care fabric",
    "café compatible",
    "plain synthetic item",
    "package of 50 pieces",
)
_ROUTE_WEIGHTS = (
    RouteWeights(bm25=0.5, dense=0.5),
    RouteWeights(bm25=0.6, dense=0.4),
    RouteWeights(bm25=0.4, dense=0.6),
    RouteWeights(bm25=0.7, dense=0.3),
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _SyntheticCase:
    state: IntentState
    documents: tuple[CandidateDocument, ...]
    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    route_weights: RouteWeights


class OracleDriftError(RuntimeError):
    """The current Stage-A output stream no longer matches the frozen oracle."""

    def __init__(self, *, cases: int, actual: str, expected: str) -> None:
        self.cases = cases
        self.actual = actual
        self.expected = expected
        super().__init__(
            "Phase 7 Stage-A oracle drift: "
            f"expected {expected}, observed {actual} across {cases} cases"
        )


def _choice(rng: random.Random, values: Sequence[_T]) -> _T:
    """Choose through Random.random(), whose seeded sequence is stable."""

    return values[int(rng.random() * len(values))]


def _shuffle(rng: random.Random, values: list[str]) -> None:
    """Use an explicit Fisher-Yates pass to avoid version-specific helpers."""

    for upper in range(len(values) - 1, 0, -1):
        selected = int(rng.random() * (upper + 1))
        values[upper], values[selected] = values[selected], values[upper]


def _synthetic_cases() -> Iterator[_SyntheticCase]:
    rng = random.Random(ORACLE_SEED)
    for case_index in range(ORACLE_CASES):
        requirement_count = case_index % (MAX_REQUIREMENTS_PER_CASE + 1)
        requirements = []
        for requirement_index in range(requirement_count):
            value, attribute = _choice(rng, _REQUIREMENTS)
            requirements.append(
                Requirement(
                    value=value,
                    source=_choice(rng, _SOURCES),
                    turn=requirement_index + 1,
                    attribute=attribute,
                )
            )
        state = IntentState(
            category=_choice(rng, _CATEGORIES),
            requirements=tuple(requirements),
        )

        candidate_count = case_index % (MAX_CANDIDATES_PER_CASE + 1)
        identifiers = [
            f"S{case_index:04d}P{candidate_index:02d}"
            for candidate_index in range(candidate_count)
        ]
        documents_by_id: dict[str, CandidateDocument] = {}
        for candidate_index, parent_asin in enumerate(identifiers):
            fragment_count = int(rng.random() * 4)
            fragments = [
                _choice(rng, _DOCUMENT_FRAGMENTS)
                for _ in range(fragment_count)
            ]
            if fragments and (case_index + candidate_index) % 5 == 0:
                fragments[0] = " ".join(str(fragments[0]).split()[::-1])
            text = " | ".join(
                [*(str(fragment) for fragment in fragments), "synthetic item"]
            )
            documents_by_id[parent_asin] = CandidateDocument(parent_asin, text)

        bm25_ids: list[str] = []
        dense_ids: list[str] = []
        for parent_asin in identifiers:
            membership = int(rng.random() * 3)
            if membership != 1:
                bm25_ids.append(parent_asin)
            if membership != 0:
                dense_ids.append(parent_asin)
        _shuffle(rng, bm25_ids)
        _shuffle(rng, dense_ids)
        fused_ids = list(identifiers)
        _shuffle(rng, fused_ids)

        yield _SyntheticCase(
            state=state,
            documents=tuple(documents_by_id[value] for value in fused_ids),
            bm25_ids=tuple(bm25_ids),
            dense_ids=tuple(dense_ids),
            fused_ids=tuple(fused_ids),
            route_weights=_choice(rng, _ROUTE_WEIGHTS),
        )


def _canonical_output(result: RankingResult) -> bytes:
    trace = result.trace
    payload = {
        "ranked_ids": list(result.ranked_ids),
        "trace": {
            "beta_hex": trace.beta.hex(),
            "input_ids": list(trace.input_ids),
            "observable_clause_count": trace.observable_clause_count,
            "output_ids": list(trace.output_ids),
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _compute_oracle_digest() -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(b"[")
    cases = 0
    for case in _synthetic_cases():
        if cases:
            digest.update(b",")
        result = rerank_stage_a(
            case.state,
            case.documents,
            bm25_ids=case.bm25_ids,
            dense_ids=case.dense_ids,
            fused_ids=case.fused_ids,
            route_weights=case.route_weights,
        )
        digest.update(_canonical_output(result))
        cases += 1
    digest.update(b"]")
    if cases != ORACLE_CASES:
        raise RuntimeError(f"oracle generated {cases} cases, expected {ORACLE_CASES}")
    return cases, digest.hexdigest()


def verify_oracle(
    expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, int | str]:
    """Return one aggregate verification record, or raise on output drift."""

    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    cases, actual = _compute_oracle_digest()
    if actual != expected_sha256:
        raise OracleDriftError(
            cases=cases,
            actual=actual,
            expected=expected_sha256,
        )
    return {"cases": cases, "digest": actual, "status": "ok"}


def main() -> int:
    try:
        aggregate = verify_oracle()
    except OracleDriftError as error:
        aggregate = {
            "cases": error.cases,
            "digest": error.actual,
            "status": "drift",
        }
        print(json.dumps(aggregate, separators=(",", ":"), sort_keys=True))
        return 1
    print(json.dumps(aggregate, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
