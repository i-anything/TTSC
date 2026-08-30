"""Build sealed, target-disjoint Phase 15 protocol robustness suites.

The CLI is dry-run by default.  Materialization requires ``--write`` plus
explicit hashes for every catalog/session input and both design references.
Generated JSONL files are local research inputs.  Their published manifest is
aggregate-only and contains no target IDs, messages, cards, or case rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluator.local_evaluator import (
    behavior_for,
    coarse_category,
    intent_card,
)
from scripts.build_phase14_explicit_card_suite import _family as phase14_family


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SOURCE = REPOSITORY_ROOT / "evaluator/local_evaluator.py"
PHASE14_BUILDER_SOURCE = (
    REPOSITORY_ROOT / "scripts/build_phase14_explicit_card_suite.py"
)

SELECTION_SALT = "phase15-protocol-robustness-target-disjoint-v2"
CASES_PER_CELL = 4
QUALITY_CASES_PER_CELL = 12
SUITE_ORDER = (
    "fresh_exact",
    "paraphrase_fail_open",
    "card_perturbed",
    "scenario_balanced",
    "target_disjoint_development",
    "target_disjoint_validation",
)
SUITE_CASES_PER_CELL = {
    "fresh_exact": CASES_PER_CELL,
    "paraphrase_fail_open": CASES_PER_CELL,
    "card_perturbed": CASES_PER_CELL,
    "scenario_balanced": CASES_PER_CELL,
    "target_disjoint_development": QUALITY_CASES_PER_CELL,
    "target_disjoint_validation": QUALITY_CASES_PER_CELL,
}
FAMILY_ORDER = ("apparel", "footwear", "jewelry_and_accessories")
POPULARITY_ORDER = ("tail", "torso", "head")
SCENARIO_ORDER = ("buying", "browsing", "boundary", "intent_override")
PERTURBATION_ORDER = (
    "constraint_order",
    "optional_soft_absent",
    "opaque_semicolon",
)
PARAPHRASE_ORDER = (
    "searching_priority",
    "shopping_must_have",
    "after_specific",
    "seeking_essential",
)
FORBIDDEN_SOURCE_NAMES = (
    "public",
    "development",
    "validation",
    "phase14_fresh",
)
REFERENCE_SOURCE_NAMES = ("evaluator", "phase14_builder")
SOURCE_NAMES = ("catalog", *FORBIDDEN_SOURCE_NAMES, *REFERENCE_SOURCE_NAMES)

MAX_SOURCE_ROWS = 1_000_000
MAX_JSONL_CHARACTERS = 2_000_000
MAX_TARGET_CHARACTERS = 256
MAX_CARD_VALUE_CHARACTERS = 180
MAX_CATEGORY_CHARACTERS = 180
MAX_MESSAGE_CHARACTERS = 1_024
SHA256_RE = re.compile(r"[0-9a-f]{64}")

CANONICAL_REPLY_SHAPES = {
    "disclosure": "For that, what matters is: {values}.",
    "boundary_decline": (
        "I don't have a preference for {attribute}; please use your judgment."
    ),
    "no_additional": (
        "I don't have an additional preference for {attribute}."
    ),
    "need_attribute": (
        "Those options are not quite right yet. "
        "Ask me about one specific attribute."
    ),
    "override": (
        "Actually, ignore my earlier preference. What I need is: {value}."
    ),
}
PARAPHRASE_REPLY_SHAPES = {
    "disclosure": "For {attribute}, the details I care about are {values}.",
    "boundary_decline": (
        "I have no set preference on {attribute}; decide what fits best."
    ),
    "no_additional": (
        "Nothing else for {attribute}; keep the broader requirements."
    ),
    "need_attribute": (
        "Please ask one concrete product attribute before showing more."
    ),
    "override": "Replace my earlier preference with this requirement: {value}.",
}


@dataclass(frozen=True, slots=True)
class Candidate:
    parent_asin: str
    family: str
    popularity: int
    category: str
    card: dict[str, object]
    popularity_stratum: str = ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return value


def _source_paths(
    catalog: Path,
    forbidden_sources: Mapping[str, Path],
) -> dict[str, Path]:
    if tuple(sorted(forbidden_sources)) != tuple(sorted(FORBIDDEN_SOURCE_NAMES)):
        raise ValueError(
            "forbidden sources must be exactly public, development, "
            "validation, and phase14_fresh"
        )
    return {
        "catalog": catalog,
        **{name: forbidden_sources[name] for name in FORBIDDEN_SOURCE_NAMES},
        "evaluator": EVALUATOR_SOURCE,
        "phase14_builder": PHASE14_BUILDER_SOURCE,
    }


def _verify_sources(
    sources: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, str]:
    if tuple(sorted(sources)) != tuple(sorted(SOURCE_NAMES)):
        raise ValueError("source map is incomplete")
    if tuple(sorted(expected_sha256)) != tuple(sorted(SOURCE_NAMES)):
        raise ValueError("expected source hash map is incomplete")
    observed: dict[str, str] = {}
    for name in SOURCE_NAMES:
        expected = _validate_sha256(expected_sha256[name], name)
        path = sources[name]
        if not path.is_file():
            raise RuntimeError(f"frozen source is missing: {name}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen source hash mismatch: {name}")
        observed[name] = actual
    return observed


def _jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle, 1):
            if ordinal > MAX_SOURCE_ROWS:
                raise ValueError(f"{path.name} exceeds the row limit")
            if not line.strip():
                continue
            if len(line) > MAX_JSONL_CHARACTERS:
                raise ValueError(f"{path.name} contains an oversized JSONL row")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path.name} row {ordinal} is invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} row {ordinal} must be an object")
            yield value


def _normalized_target(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_TARGET_CHARACTERS
    ):
        raise ValueError(f"{label} contains an invalid target ID")
    return value


def _session_target(row: Mapping[str, object], label: str) -> str:
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError(f"{label} row has no ground_truth object")
    return _normalized_target(ground_truth.get("parent_asin"), label)


def _forbidden_targets(sources: Mapping[str, Path]) -> set[str]:
    targets: set[str] = set()
    for name in FORBIDDEN_SOURCE_NAMES:
        count = 0
        for row in _jsonl_rows(sources[name]):
            count += 1
            targets.add(_session_target(row, name))
        if count == 0:
            raise ValueError(f"{name} source is empty")
    return targets


def _normalized_card(product: Mapping[str, object]) -> dict[str, object] | None:
    raw = intent_card(dict(product))
    category = str(raw.get("target_category") or "")
    hard = tuple(str(value) for value in raw.get("hard_constraints") or ())
    soft = tuple(str(value) for value in raw.get("soft_preferences") or ())
    values = (category, *hard, *soft)
    if (
        not category
        or len(hard) < 2
        or not soft
        or any(not value or len(value) > MAX_CARD_VALUE_CHARACTERS for value in values)
        or len(f"{hard[0]}; {hard[1]}") > MAX_CARD_VALUE_CHARACTERS
    ):
        return None
    return {
        "target_category": category,
        "hard_constraints": list(hard[:2]),
        "soft_preferences": list(soft[:2]),
    }


def _catalog_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for row in _jsonl_rows(path):
        parent_asin = _normalized_target(row.get("parent_asin"), "catalog")
        if parent_asin in seen:
            raise ValueError("catalog contains a duplicate target ID")
        seen.add(parent_asin)
        raw_family = phase14_family(dict(row))
        family = (
            "jewelry_and_accessories"
            if raw_family == "jewelry_accessories"
            else raw_family
        )
        if family not in FAMILY_ORDER:
            continue
        popularity = row.get("rating_number")
        if (
            isinstance(popularity, bool)
            or not isinstance(popularity, int)
            or popularity < 0
        ):
            raise ValueError("catalog rating_number must be a non-negative integer")
        categories = row.get("categories")
        if not isinstance(categories, list):
            raise ValueError("catalog categories must be a list")
        card = _normalized_card(row)
        if card is None:
            continue
        category = coarse_category([str(value) for value in categories])
        if not category or len(category) > MAX_CATEGORY_CHARACTERS:
            continue
        candidates.append(
            Candidate(
                parent_asin=parent_asin,
                family=family,
                popularity=popularity,
                category=category,
                card=card,
            )
        )
    if len(candidates) < len(SUITE_ORDER) * len(FAMILY_ORDER):
        raise RuntimeError("catalog has too few eligible protocol candidates")
    return candidates


def _selection_digest(domain: str, value: str) -> bytes:
    return hashlib.sha256(
        f"{SELECTION_SALT}\0{domain}\0{value}".encode("utf-8")
    ).digest()


def _target_set_digest(domain: str, values: Sequence[str] | set[str]) -> str:
    fingerprints = sorted(_selection_digest(domain, value) for value in values)
    return hashlib.sha256(b"".join(fingerprints)).hexdigest()


def _case_fingerprint_set_digest(rows: Sequence[Mapping[str, object]]) -> str:
    fingerprints = sorted(hashlib.sha256(_canonical(row)).digest() for row in rows)
    return hashlib.sha256(b"".join(fingerprints)).hexdigest()


def _with_popularity_strata(candidates: Sequence[Candidate]) -> list[Candidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.popularity,
            _selection_digest("popularity-tie", item.parent_asin),
        ),
    )
    total = len(ordered)
    result: list[Candidate] = []
    for rank, candidate in enumerate(ordered):
        bucket = min(2, (rank * 3) // total)
        stratum = POPULARITY_ORDER[bucket]
        result.append(
            Candidate(
                parent_asin=candidate.parent_asin,
                family=candidate.family,
                popularity=candidate.popularity,
                category=candidate.category,
                card=candidate.card,
                popularity_stratum=stratum,
            )
        )
    return result


def _select_targets(
    candidates: Sequence[Candidate],
    forbidden: set[str],
) -> dict[str, list[Candidate]]:
    cells: dict[tuple[str, str], list[Candidate]] = {
        (family, popularity): []
        for family in FAMILY_ORDER
        for popularity in POPULARITY_ORDER
    }
    for candidate in candidates:
        if candidate.parent_asin not in forbidden:
            cells[(candidate.family, candidate.popularity_stratum)].append(candidate)

    required = sum(SUITE_CASES_PER_CELL.values())
    for cell, values in cells.items():
        if len(values) < required:
            raise RuntimeError(
                f"cell {cell[0]}/{cell[1]} has {len(values)} eligible targets; "
                f"requires {required}"
            )

    selected: dict[str, list[Candidate]] = {name: [] for name in SUITE_ORDER}
    used: set[str] = set()
    for suite in SUITE_ORDER:
        for family in FAMILY_ORDER:
            for popularity in POPULARITY_ORDER:
                available = [
                    item
                    for item in cells[(family, popularity)]
                    if item.parent_asin not in used
                ]
                available.sort(
                    key=lambda item: _selection_digest(
                        f"suite:{suite}", item.parent_asin
                    )
                )
                required_for_suite = SUITE_CASES_PER_CELL[suite]
                chosen = available[:required_for_suite]
                if len(chosen) != required_for_suite:
                    raise RuntimeError("target allocation became unbalanced")
                selected[suite].extend(chosen)
                used.update(item.parent_asin for item in chosen)

    if used & forbidden:
        raise RuntimeError("selected targets overlap a forbidden source")
    if len(used) != sum(len(values) for values in selected.values()):
        raise RuntimeError("Phase 15 suites are not mutually target-disjoint")
    return selected


def _copy_card(card: Mapping[str, object]) -> dict[str, object]:
    return {
        "target_category": str(card["target_category"]),
        "hard_constraints": [str(value) for value in card["hard_constraints"]],
        "soft_preferences": [
            str(value) for value in card["soft_preferences"]
        ],
    }


def _perturb_card(
    card: Mapping[str, object],
    mode: str,
) -> dict[str, object]:
    result = _copy_card(card)
    hard = list(result["hard_constraints"])
    soft = list(result["soft_preferences"])
    if mode == "constraint_order":
        result["hard_constraints"] = list(reversed(hard))
        result["soft_preferences"] = list(reversed(soft))
    elif mode == "optional_soft_absent":
        result["soft_preferences"] = []
    elif mode == "opaque_semicolon":
        result["hard_constraints"] = [f"{hard[0]}; {hard[1]}"]
    else:
        raise ValueError("unknown card perturbation")
    return result


def _paraphrased_initial(category: str, value: str, shape: str) -> str:
    templates = {
        "searching_priority": "I'm searching for {category}; prioritize {value}.",
        "shopping_must_have": "I'm shopping for {category}. My must-have is {value}.",
        "after_specific": "I'm after {category}, specifically with {value}.",
        "seeking_essential": "I'm seeking {category}; {value} is essential.",
    }
    try:
        return templates[shape].format(category=category, value=value)
    except KeyError as error:
        raise ValueError("unknown paraphrase shape") from error


def _canonical_initial(
    scenario: str,
    category: str,
    card: Mapping[str, object],
    behavior: Mapping[str, object],
) -> str:
    hard = [str(value) for value in card.get("hard_constraints") or ()]
    soft = [str(value) for value in card.get("soft_preferences") or ()]
    if scenario == "buying":
        return (
            f"I'm looking for {category}. A key requirement is: {hard[0]}."
        )
    if scenario == "intent_override":
        override = behavior.get("override")
        old = override.get("old_value") if isinstance(override, dict) else None
        value = str(old or (soft[-1] if soft else "a different style"))
        return f"I'm looking for {category}. {value}"
    return f"I'm looking for {category}, but I'm still exploring."


def _scenario_behavior(
    scenario: str,
    card: Mapping[str, object],
    case_key: str,
) -> dict[str, object]:
    seed = int.from_bytes(_selection_digest("behavior", case_key)[:8], "big")
    return behavior_for(scenario, dict(card), random.Random(seed))


def _dialog(
    mode: str,
    initial_message: str,
) -> dict[str, object]:
    if (
        mode not in {"canonical_v1", "paraphrase_fail_open_v1"}
        or not initial_message
        or len(initial_message) > MAX_MESSAGE_CHARACTERS
    ):
        raise ValueError("dialog mode or initial message is invalid")
    reply_shapes = (
        PARAPHRASE_REPLY_SHAPES
        if mode == "paraphrase_fail_open_v1"
        else CANONICAL_REPLY_SHAPES
    )
    return {
        "mode": mode,
        "initial_message": initial_message,
        "reply_shapes": dict(reply_shapes),
    }


def _base_row(
    suite: str,
    ordinal: int,
    candidate: Candidate,
    scenario: str,
    card: Mapping[str, object],
    behavior: Mapping[str, object],
    dialog: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "sample_id": f"phase15-{suite}-{ordinal:04d}",
        "suite_variant": suite,
        "scenario_type": scenario,
        "user_profile": {
            "purchase_frequency": "unknown",
            "average_prior_rating": None,
            "rating_style": "unknown",
            "preference_tags": [],
            "summary": "",
        },
        "ground_truth": {"parent_asin": candidate.parent_asin},
        "intent_card": _copy_card(card),
        "behavior": dict(behavior),
        "phase15_family": candidate.family,
        "phase15_popularity_stratum": candidate.popularity_stratum,
        "phase15_dialog": dict(dialog),
    }


def _materialize_rows(
    selected: Mapping[str, Sequence[Candidate]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for suite in SUITE_ORDER:
        ordered = sorted(
            selected[suite],
            key=lambda item: _selection_digest(
                f"case-order:{suite}", item.parent_asin
            ),
        )
        rows: list[dict[str, object]] = []
        for index, candidate in enumerate(ordered):
            ordinal = index + 1
            card = _copy_card(candidate.card)
            scenario = "buying"
            extra: dict[str, object] = {}
            mode = "canonical_v1"
            if suite == "paraphrase_fail_open":
                shape = PARAPHRASE_ORDER[index % len(PARAPHRASE_ORDER)]
                mode = "paraphrase_fail_open_v1"
                extra["phase15_paraphrase_shape"] = shape
            elif suite == "card_perturbed":
                perturbation = PERTURBATION_ORDER[index % len(PERTURBATION_ORDER)]
                original = _copy_card(card)
                card = _perturb_card(card, perturbation)
                extra["phase15_card_perturbation"] = {
                    "mode": perturbation,
                    "original_intent_card": original,
                }
            elif suite in {
                "scenario_balanced",
                "target_disjoint_development",
                "target_disjoint_validation",
            }:
                scenario = SCENARIO_ORDER[index % len(SCENARIO_ORDER)]

            case_key = f"{suite}:{candidate.parent_asin}"
            behavior = _scenario_behavior(scenario, card, case_key)
            hard = [str(value) for value in card["hard_constraints"]]
            if suite == "paraphrase_fail_open":
                initial = _paraphrased_initial(candidate.category, hard[0], shape)
            else:
                initial = _canonical_initial(
                    scenario,
                    candidate.category,
                    card,
                    behavior,
                )
            row = _base_row(
                suite,
                ordinal,
                candidate,
                scenario,
                card,
                behavior,
                _dialog(mode, initial),
            )
            row.update(extra)
            rows.append(row)
        result[suite] = rows
    return result


def _aggregate_counts(
    rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]],
    field: str,
    values: Sequence[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for suite in SUITE_ORDER:
        counts = Counter(str(row[field]) for row in rows_by_suite[suite])
        unknown = set(counts) - set(values)
        if unknown:
            raise RuntimeError(f"unexpected {field} values")
        result[suite] = {value: counts[value] for value in values}
    return result


def _variant_counts(
    rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for suite in SUITE_ORDER:
        counts: Counter[str] = Counter()
        for row in rows_by_suite[suite]:
            if suite == "paraphrase_fail_open":
                counts[str(row["phase15_paraphrase_shape"])] += 1
            elif suite == "card_perturbed":
                detail = row["phase15_card_perturbation"]
                if not isinstance(detail, dict):
                    raise RuntimeError("invalid card perturbation metadata")
                counts[str(detail["mode"])] += 1
            else:
                counts["canonical"] += 1
        result[suite] = dict(sorted(counts.items()))
    return result


def _publish_artifact_set(payloads: Mapping[Path, bytes]) -> None:
    """Publish a new artifact set without leaving partial destinations."""

    if not payloads:
        raise ValueError("at least one artifact is required")
    destinations = tuple(payloads)
    if len(set(destinations)) != len(destinations):
        raise ValueError("artifact destinations must be unique")
    if any(path.exists() for path in destinations):
        raise FileExistsError("a frozen Phase 15 artifact already exists")

    staged: dict[Path, str] = {}
    published: list[Path] = []
    try:
        for path, payload in payloads.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
            staged[path] = temporary_name
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for path in destinations:
            os.replace(staged[path], path)
            del staged[path]
            published.append(path)
    except Exception:
        for temporary_name in staged.values():
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _nested_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_keys(item)


def _validate_manifest_privacy(
    manifest: Mapping[str, object],
    selected_ids: set[str],
    rows_by_suite: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    forbidden_keys = {
        "parent_asin",
        "ground_truth",
        "sample_id",
        "message",
        "initial_message",
        "intent_card",
        "behavior",
    }
    if forbidden_keys & set(_nested_keys(manifest)):
        raise RuntimeError("published manifest contains a private field")
    serialized = _canonical(manifest).decode("utf-8")
    for target in selected_ids:
        if json.dumps(target, ensure_ascii=False) in serialized:
            raise RuntimeError("published manifest contains a target ID")
    for rows in rows_by_suite.values():
        for row in rows:
            dialog = row.get("phase15_dialog")
            if not isinstance(dialog, dict):
                raise RuntimeError("suite row has no dialog policy")
            message = dialog.get("initial_message")
            if isinstance(message, str) and json.dumps(
                message, ensure_ascii=False
            ) in serialized:
                raise RuntimeError("published manifest contains a message")


def _validate_output_paths(
    sources: Mapping[str, Path],
    output_directory: Path,
    manifest_path: Path,
) -> dict[str, Path]:
    outputs = {
        suite: output_directory / f"{suite}.jsonl" for suite in SUITE_ORDER
    }
    source_paths = {path.resolve() for path in sources.values()}
    destinations = [
        *(path.resolve() for path in outputs.values()),
        manifest_path.resolve(),
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("output destinations must be unique")
    if any(path in source_paths for path in destinations):
        raise ValueError("an output would overwrite a frozen source")
    if any(path.exists() for path in destinations):
        raise FileExistsError("a frozen Phase 15 suite output already exists")
    return outputs


def build(
    *,
    catalog: Path,
    forbidden_sources: Mapping[str, Path],
    expected_sha256: Mapping[str, str],
    output_directory: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Build four robustness and two quality suites from new targets."""

    sources = _source_paths(catalog.resolve(), forbidden_sources)
    source_hashes = _verify_sources(sources, expected_sha256)
    outputs = _validate_output_paths(
        sources,
        output_directory.resolve(),
        manifest_path.resolve(),
    )
    forbidden = _forbidden_targets(sources)
    catalog_candidates = _with_popularity_strata(
        _catalog_candidates(sources["catalog"])
    )
    selected = _select_targets(catalog_candidates, forbidden)
    rows_by_suite = _materialize_rows(selected)

    selected_ids = {
        candidate.parent_asin
        for values in selected.values()
        for candidate in values
    }
    expected_cases = {
        suite: SUITE_CASES_PER_CELL[suite]
        * len(FAMILY_ORDER)
        * len(POPULARITY_ORDER)
        for suite in SUITE_ORDER
    }
    if any(
        len(rows_by_suite[suite]) != expected_cases[suite]
        for suite in SUITE_ORDER
    ):
        raise RuntimeError("suite cardinality is invalid")

    payloads = {
        suite: _jsonl_bytes(rows_by_suite[suite]) for suite in SUITE_ORDER
    }
    output_metadata: dict[str, dict[str, object]] = {}
    for suite in SUITE_ORDER:
        suite_ids = {item.parent_asin for item in selected[suite]}
        output_metadata[suite] = {
            "filename": outputs[suite].name,
            "bytes": len(payloads[suite]),
            "sha256": hashlib.sha256(payloads[suite]).hexdigest(),
            "case_fingerprint_set_sha256": _case_fingerprint_set_digest(
                rows_by_suite[suite]
            ),
            "target_set_sha256": _target_set_digest(
                f"selected:{suite}", suite_ids
            ),
        }

    manifest: dict[str, object] = {
        "schema_version": 2,
        "lock_id": "phase15-protocol-robustness-target-disjoint-v2",
        "status": "generated_before_phase15_candidate_data_execution",
        "generator_sha256": _sha256(Path(__file__)),
        "input_source_hashes": {
            name: source_hashes[name]
            for name in ("catalog", *FORBIDDEN_SOURCE_NAMES)
        },
        "reference_source_hashes": {
            name: source_hashes[name] for name in REFERENCE_SOURCE_NAMES
        },
        "selection_salt_sha256": hashlib.sha256(
            SELECTION_SALT.encode("utf-8")
        ).hexdigest(),
        "selection_policy": {
            "cases_per_family_popularity_cell": dict(SUITE_CASES_PER_CELL),
            "family_order": list(FAMILY_ORDER),
            "popularity_order": list(POPULARITY_ORDER),
            "popularity_quantiles": (
                "catalog-only stable rank thirds; target exclusions applied later"
            ),
            "suite_order": list(SUITE_ORDER),
            "without_replacement_across_suites": True,
        },
        "forbidden_target_count": len(forbidden),
        "forbidden_target_set_sha256": _target_set_digest(
            "forbidden", forbidden
        ),
        "selected_target_count": len(selected_ids),
        "selected_target_set_sha256": _target_set_digest(
            "selected:all", selected_ids
        ),
        "case_counts": {
            suite: len(rows_by_suite[suite]) for suite in SUITE_ORDER
        },
        "family_counts": _aggregate_counts(
            rows_by_suite, "phase15_family", FAMILY_ORDER
        ),
        "popularity_counts": _aggregate_counts(
            rows_by_suite,
            "phase15_popularity_stratum",
            POPULARITY_ORDER,
        ),
        "scenario_counts": _aggregate_counts(
            rows_by_suite, "scenario_type", SCENARIO_ORDER
        ),
        "variant_counts": _variant_counts(rows_by_suite),
        "outputs": output_metadata,
        "overlap_proof": {
            "forbidden_overlap": 0,
            "inter_suite_overlap": 0,
            "all_selected_targets_unique": (
                len(selected_ids) == sum(len(values) for values in selected.values())
            ),
        },
        "privacy": {
            "aggregate_only": True,
            "target_ids_published": False,
            "messages_published": False,
            "cards_published": False,
            "case_records_published": False,
        },
        "runtime_agent_or_retriever_used": False,
    }
    _validate_manifest_privacy(manifest, selected_ids, rows_by_suite)

    manifest_payload = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _publish_artifact_set(
        {
            **{outputs[suite]: payloads[suite] for suite in SUITE_ORDER},
            manifest_path: manifest_payload,
        }
    )
    return manifest


def _dry_plan() -> dict[str, object]:
    per_suite = {
        suite: SUITE_CASES_PER_CELL[suite]
        * len(FAMILY_ORDER)
        * len(POPULARITY_ORDER)
        for suite in SUITE_ORDER
    }
    return {
        "dry_run": True,
        "writes_performed": False,
        "suite_order": list(SUITE_ORDER),
        "cases_per_suite": per_suite,
        "selected_targets_required": sum(per_suite.values()),
        "required_hashes": list(SOURCE_NAMES),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build sealed Phase 15 protocol robustness suites"
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--phase14-fresh", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--manifest", type=Path)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}-sha256")
    arguments = parser.parse_args(argv)
    if not arguments.write:
        print(json.dumps(_dry_plan(), sort_keys=True, separators=(",", ":")))
        return

    required_paths = {
        name: getattr(arguments, name)
        for name in (
            "catalog",
            "public",
            "development",
            "validation",
            "phase14_fresh",
            "output_directory",
            "manifest",
        )
    }
    if any(value is None for value in required_paths.values()):
        parser.error("--write requires every input and output path")
    hashes = {
        name: getattr(arguments, f"{name}_sha256") for name in SOURCE_NAMES
    }
    if any(value is None for value in hashes.values()):
        parser.error("--write requires every source SHA256")

    manifest = build(
        catalog=arguments.catalog,
        forbidden_sources={
            "public": arguments.public,
            "development": arguments.development,
            "validation": arguments.validation,
            "phase14_fresh": arguments.phase14_fresh,
        },
        expected_sha256=hashes,
        output_directory=arguments.output_directory,
        manifest_path=arguments.manifest,
    )
    print(
        json.dumps(
            {
                "case_counts": manifest["case_counts"],
                "selected_target_count": manifest["selected_target_count"],
                "selected_target_set_sha256": manifest[
                    "selected_target_set_sha256"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
