"""Fixed-seed synthetic drift oracle for the frozen Phase 9 ranker."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Iterator, Sequence, TypeVar

from conversational_search.intent import IntentState, Requirement
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    NEUTRAL_PROFILE_PRIOR,
    ProductTheme,
    ProfilePolicy,
    ProfilePrior,
)
from conversational_search.ranking import (
    CandidateDocument,
    ProfileRankingResult,
    rerank_stage_a_with_profile,
)
from conversational_search.strategy import RouteWeights


ORACLE_CASES = 1_200
ORACLE_SEED = 0x9A5E10
MAX_CANDIDATES_PER_CASE = 12
MAX_REQUIREMENTS_PER_CASE = 4
EXPECTED_SHA256 = "853f33454db9e3ce8c468a0b7ead525a174217c565e6a8a60ef65faf915476e1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_INTENT_KINDS = ("empty", "category", "weak", "strong", "mixed")
_ROUTE_KINDS = ("bm25_only", "dense_only", "overlap", "partial")
_PROFILE_KINDS = (
    "disabled",
    "neutral",
    "comfort",
    "durability",
    "comfort_durability",
    "weather_breathability_sustainability",
)
_CATEGORIES = (
    "synthetic trail shoes",
    "synthetic travel bags",
    "synthetic office gear",
    "synthetic running apparel",
)
_WEAK_REQUIREMENTS = (
    ("Material: organic cotton", "material", "initial_tentative"),
    ("Style: low profile", "style", "free_text"),
    ("wide fit", "size", "initial_tentative"),
)
_STRONG_REQUIREMENTS = (
    ("Feature: waterproof shell", "feature", "initial_explicit"),
    ("Use case: winter travel", "use_case", "answer"),
    ("noise cancelling", "feature", "override"),
)
_DOCUMENT_FRAGMENTS = (
    "ergonomic cushioned comfort",
    "durable reinforced rugged construction",
    "technical performance system",
    "thermal insulated warmth",
    "waterproof wind resistant shell",
    "ultralight lightweight frame",
    "breathable airflow panels",
    "machine washable easy care fabric",
    "convertible multipurpose design",
    "recycled organic material",
    "organic cotton lining",
    "low profile wide fit",
    "winter travel essential",
    "noise cancelling system",
    "plain synthetic construction",
)
_ROUTE_WEIGHTS = (
    RouteWeights(bm25=0.4, dense=0.6),
    RouteWeights(bm25=0.5, dense=0.5),
    RouteWeights(bm25=0.6, dense=0.4),
    RouteWeights(bm25=0.45, dense=0.55),
)
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _SyntheticCase:
    intent_kind: str
    route_kind: str
    profile_kind: str
    state: IntentState
    documents: tuple[CandidateDocument, ...]
    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    route_weights: RouteWeights
    profile_prior: ProfilePrior
    profile_policy: ProfilePolicy


class OracleDriftError(RuntimeError):
    """The current Phase 9 output stream no longer matches the frozen oracle."""

    def __init__(self, *, cases: int, actual: str, expected: str) -> None:
        self.cases = cases
        self.actual = actual
        self.expected = expected
        super().__init__(
            "Phase 9 ranking oracle drift: "
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


def _intent_state(
    rng: random.Random,
    case_index: int,
    intent_kind: str,
) -> IntentState:
    if intent_kind == "empty":
        return IntentState()
    category = _choice(rng, _CATEGORIES)
    if intent_kind == "category":
        return IntentState(category=category)
    if intent_kind == "weak":
        value, attribute, source = _choice(rng, _WEAK_REQUIREMENTS)
        requirements = (Requirement(value, source, 1, attribute),)
    elif intent_kind == "strong":
        value, attribute, source = _choice(rng, _STRONG_REQUIREMENTS)
        requirements = (Requirement(value, source, 1, attribute),)
    else:
        weak = _choice(rng, _WEAK_REQUIREMENTS)
        strong = _choice(rng, _STRONG_REQUIREMENTS)
        requirements = (
            Requirement(weak[0], weak[2], 1, weak[1]),
            Requirement(strong[0], strong[2], 2, strong[1]),
        )
        if case_index % 2:
            extra = _choice(rng, _STRONG_REQUIREMENTS)
            requirements += (Requirement(extra[0], extra[2], 3, extra[1]),)
    return IntentState(category=category, requirements=requirements)


def _profile(profile_kind: str) -> tuple[ProfilePrior, ProfilePolicy]:
    if profile_kind == "disabled":
        return (
            ProfilePrior(ProductTheme.COMFORT | ProductTheme.DURABILITY),
            DISABLED_PROFILE_POLICY,
        )
    if profile_kind == "neutral":
        return NEUTRAL_PROFILE_PRIOR, BOUNDED_RESIDUAL_PROFILE_POLICY
    if profile_kind == "comfort":
        mask = ProductTheme.COMFORT
    elif profile_kind == "durability":
        mask = ProductTheme.DURABILITY
    elif profile_kind == "comfort_durability":
        mask = ProductTheme.COMFORT | ProductTheme.DURABILITY
    else:
        mask = (
            ProductTheme.WEATHER_PROTECTION
            | ProductTheme.BREATHABILITY
            | ProductTheme.SUSTAINABILITY
        )
    return ProfilePrior(mask), BOUNDED_RESIDUAL_PROFILE_POLICY


def _route_ids(
    rng: random.Random,
    identifiers: list[str],
    route_kind: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if route_kind == "bm25_only":
        bm25_ids = list(identifiers)
        dense_ids: list[str] = []
    elif route_kind == "dense_only":
        bm25_ids = []
        dense_ids = list(identifiers)
    elif route_kind == "overlap":
        bm25_ids = list(identifiers)
        dense_ids = list(identifiers)
    else:
        bm25_ids = []
        dense_ids = []
        for index, parent_asin in enumerate(identifiers):
            membership = index % 3
            if membership != 1:
                bm25_ids.append(parent_asin)
            if membership != 0:
                dense_ids.append(parent_asin)
    _shuffle(rng, bm25_ids)
    _shuffle(rng, dense_ids)
    return tuple(bm25_ids), tuple(dense_ids)


def _synthetic_cases() -> Iterator[_SyntheticCase]:
    rng = random.Random(ORACLE_SEED)
    for case_index in range(ORACLE_CASES):
        intent_kind = _INTENT_KINDS[case_index % len(_INTENT_KINDS)]
        route_kind = _ROUTE_KINDS[case_index % len(_ROUTE_KINDS)]
        profile_kind = _PROFILE_KINDS[case_index % len(_PROFILE_KINDS)]
        state = _intent_state(rng, case_index, intent_kind)
        profile_prior, profile_policy = _profile(profile_kind)

        candidate_count = case_index % (MAX_CANDIDATES_PER_CASE + 1)
        identifiers = [
            f"P9O{case_index:04d}C{candidate_index:02d}"
            for candidate_index in range(candidate_count)
        ]
        documents_by_id: dict[str, CandidateDocument] = {}
        for candidate_index, parent_asin in enumerate(identifiers):
            fragment_count = (case_index + candidate_index) % 4
            fragments = [
                _choice(rng, _DOCUMENT_FRAGMENTS)
                for _ in range(fragment_count)
            ]
            if (case_index + candidate_index) % 3 == 0:
                fragments.append(
                    _DOCUMENT_FRAGMENTS[
                        (case_index + candidate_index) % len(_DOCUMENT_FRAGMENTS)
                    ]
                )
            text = " | ".join([*fragments, "fully synthetic phase nine item"])
            documents_by_id[parent_asin] = CandidateDocument(parent_asin, text)

        bm25_ids, dense_ids = _route_ids(rng, identifiers, route_kind)
        fused_ids = list(identifiers)
        _shuffle(rng, fused_ids)
        yield _SyntheticCase(
            intent_kind=intent_kind,
            route_kind=route_kind,
            profile_kind=profile_kind,
            state=state,
            documents=tuple(documents_by_id[value] for value in fused_ids),
            bm25_ids=bm25_ids,
            dense_ids=dense_ids,
            fused_ids=tuple(fused_ids),
            route_weights=_ROUTE_WEIGHTS[case_index % len(_ROUTE_WEIGHTS)],
            profile_prior=profile_prior,
            profile_policy=profile_policy,
        )


def _canonical_output(result: ProfileRankingResult) -> bytes:
    trace = result.ranking.trace
    payload = {
        "profile": {
            "represented_theme_count": result.represented_theme_count,
            "requested_theme_count": result.requested_theme_count,
            "status": result.status.value,
        },
        "ranked_ids": list(result.ranking.ranked_ids),
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
        result = rerank_stage_a_with_profile(
            case.state,
            case.documents,
            bm25_ids=case.bm25_ids,
            dense_ids=case.dense_ids,
            fused_ids=case.fused_ids,
            route_weights=case.route_weights,
            profile_prior=case.profile_prior,
            profile_policy=case.profile_policy,
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
        raise OracleDriftError(cases=cases, actual=actual, expected=expected_sha256)
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
