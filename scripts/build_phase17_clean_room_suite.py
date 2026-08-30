"""Build the sealed Phase 17 target-disjoint language-shift suite.

The builder is intentionally independent of ``starter`` and
``conversational_search``.  It reproduces only the public evaluator's pure
catalog-to-card helpers, selects targets without running retrieval, and writes
raw cases only to a private local path.  The published manifest is aggregate
and uses keyed commitments for enumerable product identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import random
import re
import string
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CASE_COUNT = 800
SCENARIO_COUNTS = {
    "buying": 320,
    "browsing": 320,
    "intent_override": 120,
    "boundary": 40,
}
SURFACE_SCENARIO_COUNTS = {
    "official_exact": {
        "buying": 160,
        "browsing": 160,
        "intent_override": 60,
        "boundary": 20,
    },
    "clean_room_language_shift": {
        "buying": 160,
        "browsing": 160,
        "intent_override": 60,
        "boundary": 20,
    },
}
FAMILIES = ("apparel", "footwear", "jewelry_accessories")
POPULARITY_STRATA = ("tail", "torso", "head")
EVENT_SLOTS = {
    "buying_open": ("category", "value"),
    "exploring_open": ("category",),
    "override_open": ("category", "value"),
    "override_event": ("value",),
    "need_attribute": (),
    "disclosure": ("attribute", "values"),
    "no_additional": ("attribute",),
    "boundary_indifference": ("attribute",),
}
OFFICIAL_TEMPLATE_NORMALIZATIONS = {
    "im looking for category a key requirement is value",
    "im looking for category but im still exploring",
    "im looking for category value",
    "actually ignore my earlier preference what i need is value",
    "those options are not quite right yet ask me about one specific attribute",
    "for that what matters is values",
    "i dont have an additional preference for attribute",
    "i dont have a preference for attribute please use your judgment",
}
SEARCH_FIELDS = (
    "title",
    "features",
    "details",
    "description",
    "categories",
    "store",
)
MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.I,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.I,
)
NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
HEX_64_RE = re.compile(r"[0-9a-f]{64}")


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


def _jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name} row {line_number} is not an object")
            yield value


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            f"{key}: {item}"
            for key, item in value.items()
            if item not in (None, "", [])
        ]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _searchable_text(product: Mapping[str, object]) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _intent_card(product: Mapping[str, object], limit: int = 180) -> dict[str, object]:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = _searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(
            _clean_constraint(item, limit)
            for item in candidates
            if _clean_constraint(item, limit)
        )
    )
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def _classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(
        word in lowered
        for word in ("color", "black", "white", "blue", "red", "pink", "green")
    ):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def _coarse_category(values: Sequence[str]) -> str:
    excluded = {
        "clothing",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def _behavior_for(
    scenario: str,
    card: Mapping[str, object],
    rng: random.Random,
) -> dict[str, object]:
    behavior: dict[str, object] = {"scenario_type": scenario}
    if scenario == "intent_override":
        hard = list(card["hard_constraints"])
        soft = list(card["soft_preferences"])
        old_value = soft[-1] if soft else "I prefer a different style."
        new_value = hard[0] if hard else "Please prioritize the target requirements."
        behavior["override"] = {
            "turn": rng.choice([3, 4]),
            "old_value": old_value,
            "new_value": new_value,
            "message": (
                "Actually, ignore my earlier preference. "
                f"What I need is: {new_value}."
            ),
        }
    return behavior


def _family(categories: Sequence[str]) -> str:
    excluded_roots = {
        "clothing",
        "shoes & jewelry",
        "clothing shoes & jewelry",
        "clothing, shoes & jewelry",
    }
    category_parts: list[str] = []
    for value in categories:
        normalized = re.sub(r"\s+", " ", str(value)).strip().casefold()
        if normalized in excluded_roots:
            continue
        for part in normalized.split(","):
            part = part.strip()
            if part and part not in excluded_roots:
                category_parts.append(part)
    text = " ".join(category_parts)
    footwear = (
        "shoe",
        "boot",
        "sandal",
        "slipper",
        "sneaker",
        "footwear",
        "loafer",
        "clog",
    )
    accessories = (
        "jewelry",
        "jewellery",
        "ring",
        "necklace",
        "bracelet",
        "earring",
        "watch",
        "handbag",
        "wallet",
        "accessor",
        "luggage",
        "backpack",
        "belt",
        "scarf",
        "tie",
    )
    if any(token in text for token in footwear):
        return "footwear"
    if any(token in text for token in accessories):
        return "jewelry_accessories"
    return "apparel"


def _target(row: Mapping[str, object]) -> str | None:
    truth = row.get("ground_truth")
    if not isinstance(truth, dict):
        return None
    value = truth.get("parent_asin")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _forbidden_targets(paths: Sequence[Path]) -> tuple[set[str], dict[str, int]]:
    combined: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        values = {target for row in _jsonl_rows(path) if (target := _target(row))}
        counts[path.name] = len(values)
        combined.update(values)
    return combined, counts


def _key(path: Path) -> bytes:
    raw = path.read_text(encoding="ascii").strip()
    if HEX_64_RE.fullmatch(raw) is None:
        raise ValueError("selection key must be exactly 32 bytes encoded as lowercase hex")
    return bytes.fromhex(raw)


def _digest(key: bytes, domain: str, value: str) -> bytes:
    return hmac.new(
        key,
        f"{domain}\0{value}".encode("utf-8"),
        hashlib.sha256,
    ).digest()


def _set_commitment(key: bytes, domain: str, values: set[str]) -> str:
    payload = b"\0".join(value.encode("utf-8") for value in sorted(values))
    return hmac.new(key, domain.encode("utf-8") + b"\0" + payload, hashlib.sha256).hexdigest()


def _normalize_template(value: str) -> str:
    without_slots = value.replace("{", "").replace("}", "").casefold()
    return NORMALIZE_RE.sub(" ", without_slots).strip()


def _template_fields(value: str) -> tuple[str, ...]:
    fields: list[str] = []
    for _, field, _, conversion in string.Formatter().parse(value):
        if conversion:
            raise ValueError("template conversions are forbidden")
        if field is not None:
            if not field or any(character in field for character in ".["):
                raise ValueError("template field syntax is invalid")
            fields.append(field)
    return tuple(fields)


def _templates(path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(EVENT_SLOTS):
        raise ValueError("template event set is invalid")
    result: dict[str, tuple[str, ...]] = {}
    for event, expected_slots in EVENT_SLOTS.items():
        values = raw[event]
        if not isinstance(values, list) or len(values) != 8:
            raise ValueError(f"{event} must contain exactly eight templates")
        normalized: set[str] = set()
        checked: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"{event} contains an invalid template")
            fields = _template_fields(value)
            if sorted(fields) != sorted(expected_slots) or len(fields) != len(expected_slots):
                raise ValueError(f"{event} template slots are invalid")
            skeleton = _normalize_template(value)
            if skeleton in OFFICIAL_TEMPLATE_NORMALIZATIONS or skeleton in normalized:
                raise ValueError(f"{event} contains a duplicate or official template")
            normalized.add(skeleton)
            checked.append(value)
        result[event] = tuple(checked)
    return result


def _catalog_candidates(catalog: Path, forbidden: set[str]) -> tuple[list[Candidate], set[str]]:
    candidates: list[Candidate] = []
    catalog_ids: set[str] = set()
    for product in _jsonl_rows(catalog):
        parent_asin = str(product.get("parent_asin") or "").strip()
        if not parent_asin or parent_asin in catalog_ids:
            raise ValueError("catalog identifiers are missing or duplicated")
        catalog_ids.add(parent_asin)
        if parent_asin in forbidden:
            continue
        categories = [str(value) for value in product.get("categories") or ()]
        card = _intent_card(product)
        hard = [str(value) for value in card["hard_constraints"]]
        if (
            len(hard) < 2
            or len(set(hard)) < 2
            or any(_classify_constraint(value) == "budget" for value in hard[:2])
        ):
            continue
        raw_popularity = product.get("rating_number")
        popularity = int(raw_popularity) if isinstance(raw_popularity, (int, float)) else 0
        candidates.append(
            Candidate(
                parent_asin=parent_asin,
                family=_family(categories),
                popularity=max(0, popularity),
                category=_coarse_category(categories),
                card=card,
            )
        )
    if not forbidden.issubset(catalog_ids):
        raise RuntimeError("a historical target is absent from the frozen catalog")
    return candidates, catalog_ids


def _stratify(candidates: Sequence[Candidate], key: bytes) -> list[Candidate]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.family].append(candidate)
    result: list[Candidate] = []
    for family in FAMILIES:
        values = sorted(
            grouped[family],
            key=lambda item: (
                item.popularity,
                _digest(key, "popularity-tie", item.parent_asin),
            ),
        )
        if len(values) < 3:
            raise RuntimeError(f"insufficient eligible targets in {family}")
        for index, item in enumerate(values):
            stratum_index = min(2, (index * 3) // len(values))
            result.append(
                Candidate(
                    item.parent_asin,
                    item.family,
                    item.popularity,
                    item.category,
                    item.card,
                    POPULARITY_STRATA[stratum_index],
                )
            )
    return result


def _select(candidates: Sequence[Candidate], key: bytes) -> list[Candidate]:
    cells: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        cells[(candidate.family, candidate.popularity_stratum)].append(candidate)
    cell_order = [
        (family, popularity)
        for family in FAMILIES
        for popularity in POPULARITY_STRATA
    ]
    base, remainder = divmod(CASE_COUNT, len(cell_order))
    selected: list[Candidate] = []
    for index, cell in enumerate(cell_order):
        required = base + int(index < remainder)
        ordered = sorted(
            cells[cell],
            key=lambda item: _digest(key, "target-selection", item.parent_asin),
        )
        if len(ordered) < required:
            raise RuntimeError(f"insufficient eligible targets in cell {cell}")
        selected.extend(ordered[:required])
    if len({item.parent_asin for item in selected}) != CASE_COUNT:
        raise RuntimeError("target selection is not unique")
    return sorted(
        selected,
        key=lambda item: _digest(key, "case-order", item.parent_asin),
    )


def _case_assignments(key: bytes) -> list[tuple[str, str]]:
    labels = [
        (surface, scenario)
        for surface, counts in SURFACE_SCENARIO_COUNTS.items()
        for scenario, count in counts.items()
        for _ in range(count)
    ]
    decorated = [
        (
            _digest(key, "surface-scenario-order", f"{index}:{surface}:{scenario}"),
            (surface, scenario),
        )
        for index, (surface, scenario) in enumerate(labels)
    ]
    return [assignment for _, assignment in sorted(decorated)]


def _render(template: str, **slots: str) -> str:
    try:
        result = template.format(**slots)
    except (IndexError, KeyError, ValueError) as error:
        raise RuntimeError("template rendering failed") from error
    if not result or len(result) > 2048:
        raise RuntimeError("rendered template is empty or overbound")
    return result


def _choose_template(
    templates: Mapping[str, Sequence[str]],
    key: bytes,
    sample_id: str,
    event: str,
) -> str:
    values = templates[event]
    index = int.from_bytes(_digest(key, f"template:{event}", sample_id)[:8], "big")
    return values[index % len(values)]


def _copy_card(card: Mapping[str, object]) -> dict[str, object]:
    return {
        "target_category": str(card["target_category"]),
        "hard_constraints": [str(value) for value in card["hard_constraints"]],
        "soft_preferences": [str(value) for value in card["soft_preferences"]],
    }


def _rows(
    selected: Sequence[Candidate],
    templates: Mapping[str, Sequence[str]],
    key: bytes,
) -> list[dict[str, object]]:
    assignments = _case_assignments(key)
    rows: list[dict[str, object]] = []
    for index, (candidate, assignment) in enumerate(zip(selected, assignments), 1):
        surface, scenario = assignment
        sample_id = f"phase17-clean-room-{index:04d}"
        card = _copy_card(candidate.card)
        hard = list(card["hard_constraints"])
        seed = int.from_bytes(_digest(key, "behavior", candidate.parent_asin)[:8], "big")
        behavior = _behavior_for(scenario, card, random.Random(seed))
        if surface == "official_exact" and scenario == "buying":
            initial = (
                f"I'm looking for {candidate.category}. "
                f"A key requirement is: {hard[0]}."
            )
        elif surface == "official_exact" and scenario == "intent_override":
            override = behavior["override"]
            if not isinstance(override, dict):
                raise RuntimeError("override behavior is invalid")
            initial = f"I'm looking for {candidate.category}. {override['old_value']}"
            override["message"] = (
                "Actually, ignore my earlier preference. "
                f"What I need is: {override['new_value']}."
            )
        elif surface == "official_exact":
            initial = (
                f"I'm looking for {candidate.category}, but I'm still exploring."
            )
        elif scenario == "buying":
            initial = _render(
                _choose_template(templates, key, sample_id, "buying_open"),
                category=candidate.category,
                value=str(hard[0]),
            )
        elif scenario == "intent_override":
            override = behavior["override"]
            if not isinstance(override, dict):
                raise RuntimeError("override behavior is invalid")
            initial = _render(
                _choose_template(templates, key, sample_id, "override_open"),
                category=candidate.category,
                value=str(override["old_value"]),
            )
            override["message"] = _render(
                _choose_template(templates, key, sample_id, "override_event"),
                value=str(override["new_value"]),
            )
        else:
            initial = _render(
                _choose_template(templates, key, sample_id, "exploring_open"),
                category=candidate.category,
            )
        dialog: dict[str, object]
        if surface == "official_exact":
            dialog = {"mode": "official_exact"}
        else:
            dialog = {
                "mode": "clean_room_language_shift",
                "initial_message": initial,
                "reply_templates": {
                    event: _choose_template(templates, key, sample_id, event)
                    for event in (
                        "need_attribute",
                        "disclosure",
                        "no_additional",
                        "boundary_indifference",
                    )
                },
            }
        rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "scenario_type": scenario,
                "user_profile": {
                    "purchase_frequency": "unknown",
                    "average_prior_rating": None,
                    "rating_style": "unknown",
                    "preference_tags": [],
                    "summary": "",
                },
                "ground_truth": {"parent_asin": candidate.parent_asin},
                "intent_card": card,
                "behavior": behavior,
                "phase17_family": candidate.family,
                "phase17_popularity_stratum": candidate.popularity_stratum,
                "phase17_surface": surface,
                "phase17_dialog": dialog,
            }
        )
    if Counter(str(row["scenario_type"]) for row in rows) != Counter(SCENARIO_COUNTS):
        raise RuntimeError("scenario assignment drifted")
    surface_scenarios = Counter(
        (str(row["phase17_surface"]), str(row["scenario_type"])) for row in rows
    )
    if any(
        surface_scenarios[(surface, scenario)] != count
        for surface, counts in SURFACE_SCENARIO_COUNTS.items()
        for scenario, count in counts.items()
    ):
        raise RuntimeError("surface/scenario assignment drifted")
    return rows


def _publish_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build(
    *,
    catalog: Path,
    exclusion_paths: Sequence[Path],
    template_path: Path,
    key_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("a Phase 17 suite artifact already exists")
    key = _key(key_path)
    templates = _templates(template_path)
    forbidden, source_target_counts = _forbidden_targets(exclusion_paths)
    candidates, catalog_ids = _catalog_candidates(catalog, forbidden)
    selected = _select(_stratify(candidates, key), key)
    selected_ids = {item.parent_asin for item in selected}
    if selected_ids & forbidden:
        raise RuntimeError("selected targets overlap historical targets")
    rows = _rows(selected, templates, key)
    suite_payload = b"".join(_canonical(row) + b"\n" for row in rows)
    family_counts = Counter(item.family for item in selected)
    popularity_counts = Counter(item.popularity_stratum for item in selected)
    surface_counts = Counter(str(row["phase17_surface"]) for row in rows)
    surface_scenario_counts = {
        surface: {
            scenario: sum(
                row["phase17_surface"] == surface
                and row["scenario_type"] == scenario
                for row in rows
            )
            for scenario in SCENARIO_COUNTS
        }
        for surface in SURFACE_SCENARIO_COUNTS
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "lock_id": "phase17-clean-room-hidden-generalization-suite-v1",
        "status": "generated_before_candidate_evaluation",
        "case_count": len(rows),
        "unique_target_count": len(selected_ids),
        "catalog_id_count": len(catalog_ids),
        "eligible_target_count": len(candidates),
        "forbidden_target_count": len(forbidden),
        "historical_source_unique_target_counts": dict(sorted(source_target_counts.items())),
        "scenario_counts": dict(sorted(Counter(str(row["scenario_type"]) for row in rows).items())),
        "surface_counts": dict(sorted(surface_counts.items())),
        "surface_scenario_counts": surface_scenario_counts,
        "family_counts": {name: family_counts[name] for name in FAMILIES},
        "popularity_counts": {name: popularity_counts[name] for name in POPULARITY_STRATA},
        "overlap_proof": {
            "historical_target_overlap": len(selected_ids & forbidden),
            "selected_targets_unique": len(selected_ids) == len(rows),
            "all_selected_targets_catalog_valid": selected_ids <= catalog_ids,
        },
        "key_commitment_sha256": hashlib.sha256(key).hexdigest(),
        "forbidden_target_set_hmac_sha256": _set_commitment(key, "forbidden", forbidden),
        "selected_target_set_hmac_sha256": _set_commitment(key, "selected", selected_ids),
        "case_fingerprint_set_hmac_sha256": _set_commitment(
            key,
            "cases",
            {hashlib.sha256(_canonical(row)).hexdigest() for row in rows},
        ),
        "inputs": {
            "catalog_sha256": _sha256(catalog),
            "exclusion_source_sha256": {
                path.name: _sha256(path) for path in exclusion_paths
            },
            "templates_sha256": _sha256(template_path),
            "generator_sha256": _sha256(Path(__file__)),
        },
        "suite": {
            "path": str(output_path),
            "bytes": len(suite_payload),
            "sha256": hashlib.sha256(suite_payload).hexdigest(),
        },
        "generation": {
            "agent_or_retriever_imported": False,
            "candidate_search_or_ranking_used": False,
            "external_model_or_api_calls": 0,
            "template_assignment_hidden": True,
            "neutral_profiles": True,
        },
        "privacy": {
            "aggregate_only": True,
            "target_ids_published": False,
            "messages_published": False,
            "cards_published": False,
            "selection_key_published": False,
        },
    }
    serialized_manifest = json.dumps(manifest, sort_keys=True, allow_nan=False)
    if re.search(r"(?<![A-Z0-9])B[A-Z0-9]{9}(?![A-Z0-9])", serialized_manifest):
        raise RuntimeError("manifest contains a product identifier")
    _publish_new(output_path, suite_payload)
    try:
        _publish_new(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--catalog", type=Path, default=REPOSITORY_ROOT / "data/catalog.jsonl")
    parser.add_argument("--public", type=Path, default=REPOSITORY_ROOT / "data/public_set.jsonl")
    parser.add_argument("--development", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--phase14", type=Path)
    parser.add_argument("--phase16-activation", type=Path, default=REPOSITORY_ROOT / "benchmarks/phase16b_semantic_rescue_activation.jsonl")
    parser.add_argument("--shadow", type=Path)
    parser.add_argument("--templates", type=Path, default=REPOSITORY_ROOT / "docs/phase17_language_templates.json")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    if not args.write:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "writes_performed": False,
                    "case_count": CASE_COUNT,
                    "scenario_counts": SCENARIO_COUNTS,
                    "surface_scenario_counts": SURFACE_SCENARIO_COUNTS,
                    "templates_per_event": 8,
                    "historical_source_count": 6,
                },
                sort_keys=True,
            )
        )
        return
    required = (
        args.development,
        args.validation,
        args.phase14,
        args.shadow,
        args.key_file,
        args.output,
        args.manifest,
    )
    if any(value is None for value in required):
        parser.error("--write requires every historical source, key, and output path")
    manifest = build(
        catalog=args.catalog.resolve(),
        exclusion_paths=(
            args.public.resolve(),
            args.development.resolve(),
            args.validation.resolve(),
            args.phase14.resolve(),
            args.phase16_activation.resolve(),
            args.shadow.resolve(),
        ),
        template_path=args.templates.resolve(),
        key_path=args.key_file.resolve(),
        output_path=args.output.resolve(),
        manifest_path=args.manifest.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
