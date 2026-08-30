"""Fixed-seed differential oracle for Phase 14 requirement probes."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence

from conversational_search.retrieval import (
    MAX_CANDIDATE_DOCUMENTS,
    MAX_PROBE_CANDIDATES,
    MAX_PROBE_TEXT_CHARACTERS,
    MAX_REQUIREMENT_PROBES,
    ROUTE_LIMIT,
    HybridRetriever,
    RequirementProbeStatus,
)


ORACLE_SEED = 140260830
RANDOM_CASES = 30_000
EXPECTED_SHA256 = "79853931779412647e2adb4fce3b2b87ac3864b8a9cc5d260ad398cd8fbd2fde"

_MAX_CANDIDATE_DOCUMENTS = 200
_MAX_PROBE_CANDIDATES = 24
_MAX_PROBE_TEXT_CHARACTERS = 1024
_MAX_REQUIREMENT_PROBES = 2
_ROUTE_LIMIT = 100
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = frozenset(
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
        "looking",
    }
)
_WORDS = (
    "silk",
    "waterproof",
    "titanium",
    "cotton",
    "leather",
    "breathable",
    "running",
    "winter",
    "silver",
    "black",
    "wide",
    "casual",
    "formal",
    "zipper",
    "pockets",
    "lightweight",
    "stretch",
    "outdoor",
    "wool",
    "nylon",
    "the",
    "and",
    "with",
    "for",
)


class _VocabularyConnection:
    def __init__(self, frequencies: dict[str, int]) -> None:
        self._frequencies = frequencies

    def execute(
        self,
        statement: str,
        parameters: Sequence[str] = (),
    ) -> "_Rows":
        if not statement.startswith("SELECT term, doc FROM products_vocab"):
            raise AssertionError("oracle received an unexpected SQL statement")
        return _Rows(
            tuple(
                (term, self._frequencies[term])
                for term in parameters
                if term in self._frequencies
            )
        )


class _Rows:
    def __init__(self, values: tuple[tuple[str, int], ...]) -> None:
        self._values = values

    def fetchall(self) -> tuple[tuple[str, int], ...]:
        return self._values


class _RouteProvider:
    def __init__(
        self,
        routes: dict[str, tuple[str, ...]],
        errors: frozenset[str],
    ) -> None:
        self._routes = routes
        self._errors = errors

    def __call__(self, query: str) -> list[str]:
        if query in self._errors:
            raise RuntimeError("injected probe-route failure")
        return list(self._routes.get(query, ()))


def _terms(value: str) -> tuple[str, ...]:
    return tuple(
        token.lower()
        for token in _TOKEN_RE.findall(value)
        if len(token) > 1 and token.lower() not in _STOPWORDS
    )


def _validate_candidates(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("requirement probe candidates must be a sequence")
    candidates = tuple(values)
    if len(candidates) > _MAX_PROBE_CANDIDATES:
        raise ValueError("too many requirement probe candidates")
    if any(not isinstance(value, str) or not value for value in candidates):
        raise ValueError("requirement probe candidates must be non-empty strings")
    if sum(len(value) for value in candidates) > _MAX_PROBE_TEXT_CHARACTERS:
        raise ValueError("requirement probe candidates exceed the text bound")
    return candidates


def _reference_select(
    candidates: object,
    lexical_text: str,
    frequencies: dict[str, int],
) -> tuple[str, ...]:
    values = _validate_candidates(candidates)
    main_terms = tuple(dict.fromkeys(_terms(lexical_text)))[:40]
    signatures: list[tuple[int, tuple[str, ...]]] = []
    for index, value in enumerate(values):
        terms = tuple(dict.fromkeys(_terms(value)))[:40]
        if not terms:
            continue
        signatures.append((index, terms))

    ranked: list[tuple[int, int, str]] = []
    main_signature = frozenset(
        term for term in main_terms if frequencies.get(term, 0) > 0
    )
    seen: set[frozenset[str]] = set()
    for index, terms in signatures:
        known = tuple(
            term for term in terms if frequencies.get(term, 0) > 0
        )
        signature = frozenset(known)
        if not signature or signature == main_signature or signature in seen:
            continue
        seen.add(signature)
        ranked.append(
            (
                min(frequencies[term] for term in known),
                index,
                " ".join(known),
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[:_MAX_REQUIREMENT_PROBES])


def _reference_sanitize(
    values: Sequence[str],
    valid_ids: frozenset[str],
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in valid_ids and value not in seen:
            seen.add(value)
            result.append(value)
            if len(result) >= _ROUTE_LIMIT:
                break
    return tuple(result)


def _reference_supplements(
    candidates: object,
    lexical_text: str,
    incumbent_ids: set[str],
    capacity: int,
    *,
    frequencies: dict[str, int],
    vocabulary_available: bool,
    routes: dict[str, tuple[str, ...]],
    route_errors: frozenset[str],
    valid_ids: frozenset[str],
) -> tuple[tuple[str, ...], RequirementProbeStatus, int]:
    if capacity <= 0:
        return (), "capacity", 0
    if not vocabulary_available:
        return (), "unavailable", 0
    try:
        queries = _reference_select(candidates, lexical_text, frequencies)
    except Exception:
        return (), "error", 0
    if not queries:
        return (), "no_eligible", 0

    sanitized: list[tuple[str, ...]] = []
    attempted_queries = 0
    for query in queries:
        attempted_queries += 1
        if query in route_errors:
            return (), "error", attempted_queries
        sanitized.append(
            _reference_sanitize(routes.get(query, ()), valid_ids)
        )
    if not any(sanitized):
        return (), "empty", len(queries)

    additions: list[str] = []
    seen = set(incumbent_ids)
    for rank in range(_ROUTE_LIMIT):
        for route in sanitized:
            if rank >= len(route):
                continue
            parent_asin = route[rank]
            if parent_asin in seen:
                continue
            seen.add(parent_asin)
            additions.append(parent_asin)
            if len(additions) >= capacity:
                return tuple(additions), "ok", len(queries)
    status: RequirementProbeStatus = "ok" if additions else "no_additions"
    return tuple(additions), status, len(queries)


def _capture_selection(
    function: object,
    candidates: object,
    lexical_text: str,
) -> tuple[str, object]:
    try:
        result = function(candidates, lexical_text)  # type: ignore[operator]
    except Exception as error:
        return "error", type(error).__name__
    return "ok", tuple(result)


def _random_text(rng: random.Random, *, allow_empty: bool = True) -> str:
    count = rng.randrange(0 if allow_empty else 1, 7)
    words = [rng.choice(_WORDS) for _ in range(count)]
    if words and rng.randrange(4) == 0:
        words[rng.randrange(len(words))] = words[rng.randrange(len(words))].upper()
    separator = rng.choice((" ", " / ", ", ", " - "))
    value = separator.join(words)
    return value if value else "!!!"


def _random_candidates(rng: random.Random, case_index: int) -> object:
    malformed = case_index % 97
    if malformed == 0:
        return "silk waterproof"
    if malformed == 1:
        return ("",)
    if malformed == 2:
        return tuple(f"feature-{index}" for index in range(25))
    if malformed == 3:
        return ("x" * 1025,)
    if malformed == 4:
        return (123,)

    values = [_random_text(rng) for _ in range(rng.randrange(0, 9))]
    if values and rng.randrange(3) == 0:
        values.append(rng.choice(values))
    return tuple(values)


def _candidate_payload(value: object) -> object:
    if isinstance(value, tuple):
        return [
            item
            if isinstance(item, str)
            else {"type": type(item).__name__, "value": repr(item)}
            for item in value
        ]
    return {"type": type(value).__name__, "value": repr(value)}


def _run_case(
    digest: object,
    rng: random.Random,
    case_index: int,
) -> tuple[str, RequirementProbeStatus, int]:
    candidates = _random_candidates(rng, case_index)
    lexical_text = _random_text(rng)
    frequencies = {
        word: rng.randrange(0, 5001)
        for word in _WORDS
        if word not in _STOPWORDS and rng.randrange(5) != 0
    }
    valid_ids = frozenset(f"P{index:03d}" for index in range(120))
    incumbent_ids = set(rng.sample(tuple(sorted(valid_ids)), rng.randrange(0, 21)))
    vocabulary_available = rng.randrange(13) != 0
    capacity = rng.choice((-2, 0, 1, 2, 5, 20, 200))

    expected_selection = _capture_selection(
        lambda values, lexical: _reference_select(
            values,
            lexical,
            frequencies,
        ),
        candidates,
        lexical_text,
    )
    selected_queries = (
        expected_selection[1]
        if expected_selection[0] == "ok"
        else ()
    )
    routes: dict[str, tuple[str, ...]] = {}
    invalid_ids = ("UNKNOWN", "P999")
    for query in selected_queries:
        if case_index == 5:
            route_values = [*sorted(valid_ids), *invalid_ids]
        else:
            route_values = [
                rng.choice((*tuple(sorted(valid_ids)), *invalid_ids))
                for _ in range(rng.randrange(0, 18))
            ]
        if route_values and rng.randrange(3) == 0:
            route_values.append(rng.choice(route_values))
        routes[str(query)] = tuple(route_values)
    route_errors = frozenset(
        query
        for query in selected_queries
        if rng.randrange(29) == 0
    )

    retriever = object.__new__(HybridRetriever)
    retriever._connection = _VocabularyConnection(frequencies)
    retriever._valid_ids = valid_ids
    retriever.bm25_available = True
    retriever.requirement_probe_vocabulary_available = vocabulary_available
    retriever._requirement_probe_vocabulary_initialized = True
    retriever._bm25 = _RouteProvider(routes, route_errors)

    observed_selection = _capture_selection(
        retriever._select_requirement_probes,
        candidates,
        lexical_text,
    )
    if observed_selection != expected_selection:
        raise AssertionError("production requirement-probe selection drifted")

    expected_supplements = _reference_supplements(
        candidates,
        lexical_text,
        incumbent_ids,
        capacity,
        frequencies=frequencies,
        vocabulary_available=vocabulary_available,
        routes=routes,
        route_errors=route_errors,
        valid_ids=valid_ids,
    )
    observed_supplements = retriever._requirement_probe_supplements(
        candidates,  # type: ignore[arg-type]
        lexical_text,
        incumbent_ids,
        capacity,
    )
    if observed_supplements != expected_supplements:
        raise AssertionError("production requirement-probe supplement drifted")

    additions, status, query_count = observed_supplements
    if (
        query_count < 0
        or query_count > _MAX_REQUIREMENT_PROBES
        or len(additions) != len(set(additions))
        or not set(additions).issubset(valid_ids)
        or set(additions) & incumbent_ids
        or len(additions) > max(capacity, 0)
    ):
        raise AssertionError("requirement-probe result violated its bounds")

    payload = {
        "case": case_index,
        "candidates": _candidate_payload(candidates),
        "lexical": lexical_text,
        "frequencies": frequencies,
        "incumbent": sorted(incumbent_ids),
        "capacity": capacity,
        "vocabulary_available": vocabulary_available,
        "routes": routes,
        "route_errors": sorted(route_errors),
        "selection": expected_selection,
        "supplements": (additions, status, query_count),
    }
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )  # type: ignore[attr-defined]
    digest.update(b"\n")  # type: ignore[attr-defined]
    return expected_selection[0], status, query_count


def oracle_digest() -> tuple[int, str]:
    frozen_constants = (
        MAX_CANDIDATE_DOCUMENTS,
        MAX_PROBE_CANDIDATES,
        MAX_PROBE_TEXT_CHARACTERS,
        MAX_REQUIREMENT_PROBES,
        ROUTE_LIMIT,
    )
    if frozen_constants != (
        _MAX_CANDIDATE_DOCUMENTS,
        _MAX_PROBE_CANDIDATES,
        _MAX_PROBE_TEXT_CHARACTERS,
        _MAX_REQUIREMENT_PROBES,
        _ROUTE_LIMIT,
    ):
        raise AssertionError("Phase 14 requirement-probe bounds drifted")

    digest = hashlib.sha256()
    rng = random.Random(ORACLE_SEED)
    selection_outcomes: set[str] = set()
    supplement_statuses: set[RequirementProbeStatus] = set()
    query_counts: set[int] = set()
    for case_index in range(RANDOM_CASES):
        selection, status, query_count = _run_case(digest, rng, case_index)
        selection_outcomes.add(selection)
        supplement_statuses.add(status)
        query_counts.add(query_count)
    if selection_outcomes != {"ok", "error"}:
        raise AssertionError("oracle did not cover both selection outcomes")
    if supplement_statuses != {
        "capacity",
        "ok",
        "empty",
        "no_additions",
        "no_eligible",
        "unavailable",
        "error",
    }:
        raise AssertionError("oracle did not cover every supplement status")
    if query_counts != {0, 1, 2}:
        raise AssertionError("oracle did not cover every bounded query count")
    return RANDOM_CASES, digest.hexdigest()


def verify() -> dict[str, int | str]:
    cases, observed = oracle_digest()
    if observed != EXPECTED_SHA256:
        raise AssertionError(
            "Phase 14 requirement-probe oracle drifted: "
            f"expected {EXPECTED_SHA256}, observed {observed}"
        )
    return {"cases": cases, "digest": observed}


def main() -> None:
    print(json.dumps(verify(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
