"""Independent synthetic oracle for the Phase 11 intent reducer.

The generator contains no catalog, evaluator, or released-set values.  Valid
cases carry an independently expressed operation ledger; fallback cases must
remain byte-for-byte equivalent to the protected robust reducer.  The CLI
publishes aggregate counts and one frozen digest only.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, replace
from itertools import product
from typing import Iterator, Literal

from conversational_search.intent import (
    LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    ROBUST_INTENT_POLICY,
    IntentReduction,
    IntentReductionStatus,
    IntentState,
    Requirement,
    apply_user_message,
    apply_user_message_with_trace,
    render_dense_query,
    render_lexical_query,
)


ORACLE_SEED = 20260830
VALID_ORACLE_CASES = 20_000
BASELINE_EQUIVALENCE_CASES = 10_000
ORACLE_CASES = VALID_ORACLE_CASES + BASELINE_EQUIVALENCE_CASES
EXPECTED_SHA256 = "dfa3286aaf4ddd5cb05b7db9d23bfd8b42961d34343754166a73278736f86901"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_Operation = Literal["add", "exclude", "clear"]
_Source = Literal[
    "initial_explicit",
    "initial_tentative",
    "answer",
    "override",
    "free_text",
]
_Mode = Literal["add", "full_override"]


@dataclass(frozen=True, slots=True)
class OracleAtom:
    operation: _Operation
    attribute: str | None
    value: str
    replacement: bool = False


@dataclass(frozen=True, slots=True)
class OracleUnit:
    text: str
    atoms: tuple[OracleAtom, ...]


@dataclass(frozen=True, slots=True)
class OracleCase:
    case_index: int
    family: Literal["valid", "fallback"]
    mode: str
    state: IntentState
    message: str
    turn: int
    expected_status: IntentReductionStatus
    source: _Source | None = None
    envelope_mode: _Mode = "add"
    category: str | None = None
    atoms: tuple[OracleAtom, ...] = ()


class OracleExactnessError(RuntimeError):
    """The implementation diverged without exposing the synthetic case."""

    def __init__(self, *, cases: int, family: str) -> None:
        self.cases = cases
        self.family = family
        super().__init__(
            f"Phase 11 intent oracle diverged after {cases} cases in {family}"
        )


class OracleDriftError(RuntimeError):
    """The frozen aggregate oracle stream changed."""

    def __init__(self, *, cases: int, actual: str, expected: str) -> None:
        self.cases = cases
        self.actual = actual
        self.expected = expected
        super().__init__(
            "Phase 11 intent oracle drift: "
            f"expected {expected}, observed {actual} across {cases} cases"
        )


_UNITS = (
    OracleUnit("color: blue", (OracleAtom("add", "color", "blue"),)),
    OracleUnit("material: wool", (OracleAtom("add", "material", "wool"),)),
    OracleUnit("budget: under $80", (OracleAtom("add", "budget", "under $80"),)),
    OracleUnit("weatherproof", (OracleAtom("add", None, "weatherproof"),)),
    OracleUnit(
        "not material: leather",
        (OracleAtom("exclude", "material", "leather"),),
    ),
    OracleUnit("Any color is fine", (OracleAtom("clear", "color", ""),)),
    OracleUnit(
        "now color: blue",
        (OracleAtom("add", "color", "blue", replacement=True),),
    ),
    OracleUnit(
        "replace color: red with color: blue",
        (
            OracleAtom("exclude", "color", "red"),
            OracleAtom("add", "color", "blue", replacement=True),
        ),
    ),
)


def _base_state(case_index: int) -> IntentState:
    """Small deterministic states with contradictions available to exercise."""

    requirements = (
        Requirement("red", "initial_explicit", 1, "color"),
        Requirement("cotton", "answer", 1, "material"),
        Requirement("comfortable", "free_text", 1, None),
    )[: case_index % 4]
    excluded = ("yellow",) if case_index % 5 == 0 else ()
    no_preference = frozenset({"size"}) if case_index % 7 == 0 else frozenset()
    asked = ("material", "color")[: case_index % 3]
    return IntentState(
        category="Synthetic apparel" if case_index % 2 else None,
        requirements=requirements,
        excluded=excluded,
        no_preference=no_preference,
        asked_attributes=asked,
        last_asked_attribute=asked[-1] if asked else None,
        intent_version=case_index % 3,
        last_turn=1,
    )


def _eligible_units(units: tuple[OracleUnit, ...]) -> bool:
    atoms = tuple(atom for unit in units for atom in unit.atoms)
    if not 2 <= len(atoms) <= 8:
        return False
    if any(atom.operation != "add" for atom in atoms):
        return True
    return len({atom.attribute for atom in atoms}) >= 2


def _valid_case(
    case_index: int,
    units: tuple[OracleUnit, ...],
    envelope_index: int,
) -> OracleCase:
    payload = "; ".join(unit.text for unit in units)
    atoms = tuple(atom for unit in units for atom in unit.atoms)
    state = _base_state(case_index)
    envelope = envelope_index % 5
    if envelope == 0:
        return OracleCase(
            case_index,
            "valid",
            "free_text",
            state,
            payload,
            2,
            IntentReductionStatus.APPLIED,
            "free_text",
            atoms=atoms,
        )
    if envelope == 1:
        return OracleCase(
            case_index,
            "valid",
            "answer",
            state,
            f"For that, what matters is: {payload}.",
            2,
            IntentReductionStatus.APPLIED,
            "answer",
            atoms=atoms,
        )
    if envelope == 2:
        return OracleCase(
            case_index,
            "valid",
            "full_override",
            state,
            (
                "Actually, ignore my earlier preference. "
                f"What I need is: {payload}."
            ),
            2,
            IntentReductionStatus.APPLIED,
            "override",
            "full_override",
            atoms=atoms,
        )
    if envelope == 3:
        return OracleCase(
            case_index,
            "valid",
            "initial_explicit",
            IntentState(),
            (
                "I'm looking for Synthetic Shoes. A key requirement is: "
                f"{payload}."
            ),
            1,
            IntentReductionStatus.APPLIED,
            "initial_explicit",
            category="Synthetic Shoes",
            atoms=atoms,
        )
    return OracleCase(
        case_index,
        "valid",
        "initial_tentative",
        IntentState(),
        f"Looking for Synthetic Shoes, maybe {payload}.",
        1,
        IntentReductionStatus.APPLIED,
        "initial_tentative",
        category="Synthetic Shoes",
        atoms=atoms,
    )


def _valid_cases() -> Iterator[OracleCase]:
    rng = random.Random(ORACLE_SEED)
    partitions = [
        units
        for width in (2, 3)
        for units in product(_UNITS, repeat=width)
        if _eligible_units(units)
    ]
    rng.shuffle(partitions)
    for case_index in range(VALID_ORACLE_CASES):
        units = partitions[case_index % len(partitions)]
        yield _valid_case(case_index, units, case_index // len(partitions))


def _fallback_case(case_index: int) -> OracleCase:
    state = _base_state(case_index)
    family = case_index % 7
    if family == 0:
        message = f"synthetic preference token {case_index}"
        status = IntentReductionStatus.SINGLE_SLOT
        mode = "single_residual"
    elif family == 1:
        message = "red and blue"
        status = IntentReductionStatus.SINGLE_SLOT
        mode = "same_slot"
    elif family == 2:
        message = "red leather"
        status = IntentReductionStatus.AMBIGUOUS
        mode = "ambiguous_slot"
    elif family == 3:
        message = "not sure about leather and red"
        status = IntentReductionStatus.AMBIGUOUS
        mode = "unsafe_negation"
    elif family == 4:
        message = "x" * 2049
        status = IntentReductionStatus.BOUNDS
        mode = "message_bound"
    elif family == 5:
        message = "; ".join(f"feature: token{index}" for index in range(9))
        status = IntentReductionStatus.BOUNDS
        mode = "atom_count_bound"
    else:
        message = f"feature: {'x' * 257}; material: wool"
        status = IntentReductionStatus.BOUNDS
        mode = "value_bound"
    return OracleCase(
        VALID_ORACLE_CASES + case_index,
        "fallback",
        mode,
        state,
        message,
        2,
        status,
    )


def synthetic_cases() -> Iterator[OracleCase]:
    yield from _valid_cases()
    for case_index in range(BASELINE_EQUIVALENCE_CASES):
        yield _fallback_case(case_index)


def _same_value(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _infer_reference_attribute(value: str) -> str | None:
    lowered = value.casefold()
    matches: set[str] = set()
    if any(word in lowered for word in ("leather", "wool", "cotton")):
        matches.add("material")
    if any(word in lowered for word in ("red", "blue", "yellow")):
        matches.add("color")
    if "under $" in lowered or "budget" in lowered:
        matches.add("budget")
    return next(iter(matches)) if len(matches) == 1 else None


def _append_reference(
    requirements: list[Requirement], requirement: Requirement
) -> None:
    requirements[:] = [
        current
        for current in requirements
        if not _same_value(current.value, requirement.value)
    ]
    requirements.append(requirement)


def _reference_apply(case: OracleCase) -> IntentState:
    if case.family != "valid" or case.source is None:
        raise ValueError("reference transition requires a valid oracle case")
    requirements = list(case.state.requirements)
    exclusions = list(case.state.excluded)
    no_preference = set(case.state.no_preference)
    destructive = case.envelope_mode != "add"
    if case.envelope_mode == "full_override":
        requirements = [
            requirement
            for requirement in requirements
            if requirement.source not in {"initial_explicit", "initial_tentative"}
        ]

    for atom in case.atoms:
        if atom.operation == "add":
            if atom.replacement:
                if atom.attribute is None:
                    raise ValueError("synthetic replacement must be typed")
                requirements = [
                    requirement
                    for requirement in requirements
                    if requirement.attribute != atom.attribute
                ]
                destructive = True
            exclusions = [
                value for value in exclusions if not _same_value(value, atom.value)
            ]
            if atom.attribute is not None:
                no_preference.discard(atom.attribute)
            _append_reference(
                requirements,
                Requirement(atom.value, case.source, case.turn, atom.attribute),
            )
        elif atom.operation == "exclude":
            requirements = [
                requirement
                for requirement in requirements
                if not _same_value(requirement.value, atom.value)
            ]
            exclusions = [
                value for value in exclusions if not _same_value(value, atom.value)
            ]
            exclusions.append(atom.value)
            destructive = True
        elif atom.operation == "clear":
            if atom.attribute is None:
                raise ValueError("synthetic clear must be typed")
            requirements = [
                requirement
                for requirement in requirements
                if requirement.attribute != atom.attribute
            ]
            exclusions = [
                value
                for value in exclusions
                if _infer_reference_attribute(value) != atom.attribute
            ]
            no_preference.add(atom.attribute)
            destructive = True
        else:  # pragma: no cover - OracleAtom's type and frozen table prevent this.
            raise ValueError("unsupported synthetic oracle operation")

    return replace(
        case.state,
        category=case.category or case.state.category,
        requirements=tuple(requirements),
        excluded=tuple(exclusions),
        no_preference=frozenset(no_preference),
        intent_version=case.state.intent_version + int(destructive),
        last_turn=case.turn,
    )


def _deduplicate(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _without_label(value: str, label: str) -> str:
    return re.sub(
        rf"^\s*{re.escape(label)}\s*:\s*",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    )


def _reference_dense_query(state: IntentState) -> str:
    lines: list[str] = []
    if state.category:
        lines.append(f"Category: {state.category}")
    clues = _deduplicate(
        [
            requirement.value
            for requirement in state.requirements
            if requirement.attribute in {None, "feature", "use_case", "other"}
        ]
    )
    if clues:
        lines.append("Search Clues: " + " | ".join(clues))
    brands = _deduplicate(
        [
            _without_label(requirement.value, "brand")
            for requirement in state.requirements
            if requirement.attribute == "brand"
        ]
    )
    if brands:
        lines.append("Brand: " + " | ".join(brands))
    attributes: list[str] = []
    for attribute, label in (
        ("material", "Material"),
        ("color", "Color"),
        ("size", "Size"),
        ("style", "Style"),
    ):
        attributes.extend(
            f"{label}: {_without_label(requirement.value, attribute)}"
            for requirement in state.requirements
            if requirement.attribute == attribute
        )
    if attributes:
        lines.append("Attributes: " + " | ".join(_deduplicate(attributes)))
    budgets = _deduplicate(
        [
            requirement.value
            for requirement in state.requirements
            if requirement.attribute == "budget"
        ]
    )
    if budgets:
        lines.append("Price: " + " | ".join(budgets))
    return "\n".join(lines)


def _reference_lexical_query(state: IntentState) -> str:
    return " ".join(
        _deduplicate(
            [
                value
                for value in (
                    state.category or "",
                    *(requirement.value for requirement in state.requirements),
                )
                if value
            ]
        )
    )


def evaluate_case(case: OracleCase) -> IntentReduction:
    first = apply_user_message_with_trace(
        case.state,
        case.message,
        case.turn,
        policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    )
    second = apply_user_message_with_trace(
        case.state,
        case.message,
        case.turn,
        policy=LOSSLESS_MULTI_SLOT_INTENT_POLICY,
    )
    if first != second or first.status is not case.expected_status:
        raise OracleExactnessError(cases=case.case_index + 1, family=case.family)

    if case.family == "fallback":
        expected = apply_user_message(
            case.state,
            case.message,
            case.turn,
            policy=ROBUST_INTENT_POLICY,
        )
    else:
        expected = _reference_apply(case)
        positive = sum(atom.operation == "add" for atom in case.atoms)
        exclusions = sum(atom.operation == "exclude" for atom in case.atoms)
        clears = sum(atom.operation == "clear" for atom in case.atoms)
        replacements = sum(atom.replacement for atom in case.atoms)
        residuals = sum(
            atom.operation == "add" and atom.attribute is None
            for atom in case.atoms
        )
        if (
            first.positive_atoms,
            first.exclusion_atoms,
            first.clear_atoms,
            first.replacement_atoms,
            first.residual_atoms,
        ) != (positive, exclusions, clears, replacements, residuals):
            raise OracleExactnessError(
                cases=case.case_index + 1,
                family=case.family,
            )

    if first.state != expected:
        raise OracleExactnessError(cases=case.case_index + 1, family=case.family)
    if render_dense_query(first.state) != _reference_dense_query(first.state):
        raise OracleExactnessError(cases=case.case_index + 1, family=case.family)
    if render_lexical_query(first.state) != _reference_lexical_query(first.state):
        raise OracleExactnessError(cases=case.case_index + 1, family=case.family)
    return first


def _state_payload(state: IntentState) -> dict:
    return {
        "category": state.category,
        "requirements": [
            {
                "value": requirement.value,
                "source": requirement.source,
                "turn": requirement.turn,
                "attribute": requirement.attribute,
            }
            for requirement in state.requirements
        ],
        "excluded": list(state.excluded),
        "no_preference": sorted(state.no_preference),
        "asked_attributes": list(state.asked_attributes),
        "last_asked_attribute": state.last_asked_attribute,
        "intent_version": state.intent_version,
        "last_turn": state.last_turn,
    }


def canonical_output(case: OracleCase, reduction: IntentReduction) -> bytes:
    payload = {
        "family": case.family,
        "mode": case.mode,
        "status": reduction.status.value,
        "state": _state_payload(reduction.state),
        "dense_query": render_dense_query(reduction.state),
        "lexical_query": render_lexical_query(reduction.state),
        "counts": [
            reduction.positive_atoms,
            reduction.exclusion_atoms,
            reduction.clear_atoms,
            reduction.replacement_atoms,
            reduction.residual_atoms,
        ],
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def compute_oracle_digest() -> tuple[str, int, int, int]:
    digest = hashlib.sha256()
    digest.update(b"[")
    cases = 0
    valid = 0
    fallback = 0
    for case in synthetic_cases():
        if cases:
            digest.update(b",")
        reduction = evaluate_case(case)
        digest.update(canonical_output(case, reduction))
        cases += 1
        valid += int(case.family == "valid")
        fallback += int(case.family == "fallback")
    digest.update(b"]")
    if (cases, valid, fallback) != (
        ORACLE_CASES,
        VALID_ORACLE_CASES,
        BASELINE_EQUIVALENCE_CASES,
    ):
        raise RuntimeError("Phase 11 oracle generated an incomplete case stream")
    return digest.hexdigest(), cases, valid, fallback


def verify_oracle(expected_sha256: str = EXPECTED_SHA256) -> dict:
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise ValueError("expected digest must be 64 lowercase hexadecimal characters")
    actual, cases, valid, fallback = compute_oracle_digest()
    if actual != expected_sha256:
        raise OracleDriftError(
            cases=cases,
            actual=actual,
            expected=expected_sha256,
        )
    return {
        "cases": cases,
        "valid_cases": valid,
        "baseline_equivalence_cases": fallback,
        "digest": actual,
        "status": "ok",
    }


def main() -> int:
    print(json.dumps(verify_oracle(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
