"""Synthetic exact-equivalence oracle for Phase 10 neutral/fallback paths."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Iterator
from unittest.mock import patch

from conversational_search.intent import IntentState, Requirement
from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    ProductTheme,
    ProfilePrior,
)
from conversational_search.ranking import (
    Bm25RescueRankingResult,
    Bm25RescueStatus,
    CandidateDocument,
    ProfileRankingResult,
    rerank_stage_a_with_profile,
    rerank_stage_a_with_profile_and_bm25_rescue,
)
from conversational_search.strategy import RouteWeights
from scripts import verify_phase9_ranking_oracle as phase9_oracle


ORACLE_CASES = 1_200
ORACLE_SEED = phase9_oracle.ORACLE_SEED
EXPECTED_SHA256 = "a45a56b15931ab5e203e2b06cfb8d7651bc85d274d886b8d9ac48218df9450f1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MODES = (
    "zero_completeness",
    "empty_bm25",
    "no_positive_uplift",
    "unchanged_order",
    "rescue_fault",
    "profile_fault",
)
_EXPECTED_STATUSES = {
    "zero_completeness": Bm25RescueStatus.ZERO_COMPLETENESS,
    "empty_bm25": Bm25RescueStatus.EMPTY_BM25,
    "no_positive_uplift": Bm25RescueStatus.NO_POSITIVE_UPLIFT,
    "unchanged_order": Bm25RescueStatus.UNCHANGED_ORDER,
    "rescue_fault": Bm25RescueStatus.SCORING_FALLBACK,
    "profile_fault": Bm25RescueStatus.SCORING_FALLBACK,
}
_RESCUE_FAULT = "rescue"
_PROFILE_SCORE_FAULT = "profile_score"
_INVALID_PROFILE = "invalid_profile"
_INVALID_PROFILE_POLICY = object()


@dataclass(frozen=True, slots=True)
class _ExactCase:
    case_index: int
    mode: str
    state: IntentState
    documents: tuple[CandidateDocument, ...]
    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    route_weights: RouteWeights
    profile_prior: object
    profile_policy: object
    injected_fault: str | None = None


class OracleExactnessError(RuntimeError):
    """A synthetic Phase 10 fallback diverged from exact Phase 9."""

    def __init__(self, *, cases: int, mode: str) -> None:
        self.cases = cases
        self.mode = mode
        super().__init__(
            f"Phase 10 exact fallback diverged after {cases} cases in {mode}"
        )


class OracleDriftError(RuntimeError):
    """The exact-equivalence output stream changed after it was frozen."""

    def __init__(self, *, cases: int, actual: str, expected: str) -> None:
        self.cases = cases
        self.actual = actual
        self.expected = expected
        super().__init__(
            "Phase 10/Phase 9 exact oracle drift: "
            f"expected {expected}, observed {actual} across {cases} cases"
        )


def _strong_state(category: str | None) -> IntentState:
    return IntentState(
        category=category,
        requirements=(
            Requirement(
                "Feature: exact oracle signal",
                "answer",
                1,
                "feature",
            ),
        ),
    )


def _one_candidate(case_index: int) -> tuple[
    tuple[CandidateDocument, ...],
    tuple[str, ...],
]:
    parent_asin = f"P10E{case_index:04d}A"
    return (
        (CandidateDocument(parent_asin, "fully synthetic exact oracle item"),),
        (parent_asin,),
    )


def _unchanged_candidates(case_index: int) -> tuple[
    tuple[CandidateDocument, ...],
    tuple[str, str],
]:
    identifiers = (f"P10E{case_index:04d}A", f"P10E{case_index:04d}B")
    return (
        tuple(
            CandidateDocument(
                parent_asin,
                "fully synthetic unmatched exact oracle item",
            )
            for parent_asin in identifiers
        ),
        identifiers,
    )


def _transform_case(
    case_index: int,
    base: phase9_oracle._SyntheticCase,
) -> _ExactCase:
    mode = _MODES[case_index % len(_MODES)]
    if mode == "zero_completeness":
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=IntentState(category=base.state.category),
            documents=base.documents,
            bm25_ids=base.bm25_ids,
            dense_ids=base.dense_ids,
            fused_ids=base.fused_ids,
            route_weights=base.route_weights,
            profile_prior=base.profile_prior,
            profile_policy=base.profile_policy,
        )
    if mode == "empty_bm25":
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=_strong_state(base.state.category),
            documents=base.documents,
            bm25_ids=(),
            dense_ids=base.fused_ids,
            fused_ids=base.fused_ids,
            route_weights=base.route_weights,
            profile_prior=base.profile_prior,
            profile_policy=base.profile_policy,
        )
    if mode == "no_positive_uplift":
        documents, identifiers = _one_candidate(case_index)
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=_strong_state(base.state.category),
            documents=documents,
            bm25_ids=identifiers,
            dense_ids=(),
            fused_ids=identifiers,
            route_weights=base.route_weights,
            profile_prior=base.profile_prior,
            profile_policy=base.profile_policy,
        )
    if mode == "unchanged_order":
        documents, identifiers = _unchanged_candidates(case_index)
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=_strong_state(base.state.category),
            documents=documents,
            bm25_ids=(identifiers[1],),
            dense_ids=(identifiers[0],),
            fused_ids=identifiers,
            route_weights=RouteWeights(bm25=0.4, dense=0.6),
            profile_prior=base.profile_prior,
            profile_policy=base.profile_policy,
        )
    if mode == "rescue_fault":
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=base.state,
            documents=base.documents,
            bm25_ids=base.bm25_ids,
            dense_ids=base.dense_ids,
            fused_ids=base.fused_ids,
            route_weights=base.route_weights,
            profile_prior=base.profile_prior,
            profile_policy=base.profile_policy,
            injected_fault=_RESCUE_FAULT,
        )

    if (case_index // len(_MODES)) % 2:
        documents, identifiers = _one_candidate(case_index)
        documents = (
            CandidateDocument(
                identifiers[0],
                "comfortable fully synthetic exact oracle item",
            ),
        )
        return _ExactCase(
            case_index=case_index,
            mode=mode,
            state=IntentState(category="synthetic profile oracle"),
            documents=documents,
            bm25_ids=identifiers,
            dense_ids=identifiers,
            fused_ids=identifiers,
            route_weights=base.route_weights,
            profile_prior=ProfilePrior(ProductTheme.COMFORT),
            profile_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            injected_fault=_PROFILE_SCORE_FAULT,
        )
    return _ExactCase(
        case_index=case_index,
        mode=mode,
        state=base.state,
        documents=base.documents,
        bm25_ids=base.bm25_ids,
        dense_ids=base.dense_ids,
        fused_ids=base.fused_ids,
        route_weights=base.route_weights,
        profile_prior=base.profile_prior,
        profile_policy=_INVALID_PROFILE_POLICY,
        injected_fault=_INVALID_PROFILE,
    )


def _synthetic_cases() -> Iterator[_ExactCase]:
    cases = 0
    for case_index, base in enumerate(phase9_oracle._synthetic_cases()):
        if case_index >= ORACLE_CASES:
            break
        yield _transform_case(case_index, base)
        cases += 1
    if cases != ORACLE_CASES:
        raise RuntimeError(f"oracle generated {cases} cases, expected {ORACLE_CASES}")


def _rank_phase9(case: _ExactCase) -> ProfileRankingResult:
    return rerank_stage_a_with_profile(
        case.state,
        case.documents,
        bm25_ids=case.bm25_ids,
        dense_ids=case.dense_ids,
        fused_ids=case.fused_ids,
        route_weights=case.route_weights,
        profile_prior=case.profile_prior,  # type: ignore[arg-type]
        profile_policy=case.profile_policy,  # type: ignore[arg-type]
    )


def _rank_phase10(case: _ExactCase) -> Bm25RescueRankingResult:
    return rerank_stage_a_with_profile_and_bm25_rescue(
        case.state,
        case.documents,
        bm25_ids=case.bm25_ids,
        dense_ids=case.dense_ids,
        fused_ids=case.fused_ids,
        route_weights=case.route_weights,
        profile_prior=case.profile_prior,  # type: ignore[arg-type]
        profile_policy=case.profile_policy,  # type: ignore[arg-type]
    )


def _evaluate_case(
    case: _ExactCase,
) -> tuple[ProfileRankingResult, Bm25RescueRankingResult]:
    if case.injected_fault == _PROFILE_SCORE_FAULT:
        with patch(
            "conversational_search.ranking._profile_residual_scores",
            side_effect=RuntimeError("synthetic profile scoring fault"),
        ):
            phase9 = _rank_phase9(case)
            phase10 = _rank_phase10(case)
    else:
        phase9 = _rank_phase9(case)
        if case.injected_fault == _RESCUE_FAULT:
            with patch(
                "conversational_search.ranking._apply_bm25_rescue",
                side_effect=RuntimeError("synthetic rescue fault"),
            ):
                phase10 = _rank_phase10(case)
        else:
            phase10 = _rank_phase10(case)

    phase10_profile = ProfileRankingResult(
        ranking=phase10.ranking,
        status=phase10.profile_status,
        requested_theme_count=phase10.requested_theme_count,
        represented_theme_count=phase10.represented_theme_count,
    )
    if phase10_profile != phase9:
        raise OracleExactnessError(cases=case.case_index + 1, mode=case.mode)
    if phase10.status is not _EXPECTED_STATUSES[case.mode]:
        raise OracleExactnessError(cases=case.case_index + 1, mode=case.mode)
    return phase9, phase10


def _canonical_output(
    case: _ExactCase,
    phase9: ProfileRankingResult,
    phase10: Bm25RescueRankingResult,
) -> bytes:
    payload = {
        "mode": case.mode,
        "phase9": json.loads(phase9_oracle._canonical_output(phase9)),
        "rescue_status": phase10.status.value,
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
        phase9, phase10 = _evaluate_case(case)
        digest.update(_canonical_output(case, phase9, phase10))
        cases += 1
    digest.update(b"]")
    if cases != ORACLE_CASES:
        raise RuntimeError(f"oracle evaluated {cases} cases, expected {ORACLE_CASES}")
    return cases, digest.hexdigest()


def verify_oracle(
    expected_sha256: str = EXPECTED_SHA256,
) -> dict[str, int | str]:
    """Return one aggregate verification record, or raise on drift."""

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
    except OracleExactnessError as error:
        aggregate = {"cases": error.cases, "status": "mismatch"}
        print(json.dumps(aggregate, separators=(",", ":"), sort_keys=True))
        return 1
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
