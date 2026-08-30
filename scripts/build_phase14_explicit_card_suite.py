"""Build the sealed, target-disjoint Phase 14 mechanistic suite.

The generator never calls a retriever or Agent and publishes aggregate hashes
only.  The generated JSONL is a local research input, not a submission asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from evaluator.local_evaluator import (
    classify_constraint,
    coarse_category,
    intent_card,
    load_jsonl,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPOSITORY_ROOT / "data/catalog.jsonl"
PUBLIC = REPOSITORY_ROOT / "data/public_set.jsonl"
DEVELOPMENT = Path(
    "/Users/limzichao/Downloads/public_plus_synthetic_1200.jsonl"
)
VALIDATION = Path(
    "/Users/limzichao/Downloads/public_plus_synthetic_scenario_aware_1200.jsonl"
)
CONTRACT = REPOSITORY_ROOT / "docs/phase14_experiment_contract.json"
BASELINE_LOCK = REPOSITORY_ROOT / "docs/phase14_baseline_lock.json"

EXPECTED_SHA256 = {
    CATALOG: "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67",
    PUBLIC: "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579",
    DEVELOPMENT: "f2cdf94b8dbdf22373f42dd661f22372e92715d7bcbc924590db26cf824894db",
    VALIDATION: "78da5e8402bd7d6c7d9eee86de24eec9a13d3e433ed6f3cfe720bf85cb3319c9",
}

FAMILY_SIZE = 128
FAMILY_ORDER = ("apparel", "footwear", "jewelry_accessories")
SELECTION_SALT = "phase14-explicit-card-v1"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

FOOTWEAR_TERMS = frozenset(
    {
        "boot",
        "boots",
        "cleat",
        "cleats",
        "footwear",
        "loafer",
        "loafers",
        "sandal",
        "sandals",
        "shoe",
        "shoes",
        "slipper",
        "slippers",
        "sneaker",
        "sneakers",
    }
)
JEWELRY_ACCESSORY_TERMS = frozenset(
    {
        "accessories",
        "accessory",
        "bag",
        "bags",
        "belt",
        "belts",
        "bracelet",
        "bracelets",
        "earring",
        "earrings",
        "glove",
        "gloves",
        "handbag",
        "handbags",
        "hat",
        "hats",
        "jewelry",
        "necklace",
        "necklaces",
        "purse",
        "purses",
        "ring",
        "rings",
        "scarf",
        "scarves",
        "sunglasses",
        "wallet",
        "wallets",
        "watch",
        "watches",
    }
)
APPAREL_TERMS = frozenset(
    {
        "apparel",
        "blouse",
        "bra",
        "clothing",
        "coat",
        "dress",
        "hoodie",
        "jacket",
        "jeans",
        "pants",
        "shirt",
        "shirts",
        "shorts",
        "skirt",
        "socks",
        "sweater",
        "swimwear",
        "top",
        "tops",
        "underwear",
    }
)


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


def _target_id(row: object) -> str:
    if not isinstance(row, dict):
        raise ValueError("suite source rows must be objects")
    ground_truth = row.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise ValueError("suite source row has no ground_truth object")
    parent_asin = ground_truth.get("parent_asin")
    if not isinstance(parent_asin, str) or not parent_asin:
        raise ValueError("suite source row has an invalid target")
    return parent_asin


def _forbidden_targets() -> set[str]:
    result: set[str] = set()
    for path in (PUBLIC, DEVELOPMENT, VALIDATION):
        for row in load_jsonl(path):
            result.add(_target_id(row))
    return result


def _family(product: dict) -> str | None:
    raw_categories = product.get("categories") or []
    categories = (
        [str(item) for item in raw_categories]
        if isinstance(raw_categories, list)
        else [str(raw_categories)]
    )
    # The universal root label contains all three words (Clothing, Shoes, and
    # Jewelry), so it carries no family information and must not participate.
    informative_categories = [
        value
        for value in categories
        if value.casefold().strip() not in {
            "clothing, shoes & jewelry",
            "clothing shoes & jewelry",
        }
    ]
    corpus = " ".join(
        [*informative_categories, str(product.get("title") or "")]
    )
    tokens = frozenset(token.casefold() for token in TOKEN_RE.findall(corpus))
    if tokens & FOOTWEAR_TERMS:
        return "footwear"
    if tokens & JEWELRY_ACCESSORY_TERMS:
        return "jewelry_accessories"
    if tokens & APPAREL_TERMS:
        return "apparel"
    return None


def _searchable_constraint(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if classify_constraint(value) == "budget":
        return None
    tokens = [
        token.casefold()
        for token in TOKEN_RE.findall(value)
        if len(token) > 1 and any(character.isalpha() for character in token)
    ]
    return value if tokens else None


def _candidate(product: dict, forbidden: set[str]) -> tuple[str, dict] | None:
    parent_asin = product.get("parent_asin")
    if not isinstance(parent_asin, str) or not parent_asin or parent_asin in forbidden:
        return None
    family = _family(product)
    if family is None:
        return None
    card = intent_card(product)
    hard: list[str] = []
    seen: set[str] = set()
    for raw in card.get("hard_constraints") or []:
        value = _searchable_constraint(raw)
        if value is None or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        hard.append(value)
    if len(hard) < 2:
        return None
    normalized_card = {
        "target_category": str(card["target_category"]),
        "hard_constraints": hard[:2],
        "soft_preferences": [str(value) for value in card.get("soft_preferences") or []],
    }
    return family, normalized_card


def _selection_key(parent_asin: str) -> bytes:
    return hashlib.sha256(
        f"{SELECTION_SALT}\0{parent_asin}".encode("utf-8")
    ).digest()


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical(row).decode("utf-8") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(output: Path, manifest: Path) -> dict:
    for path, expected in EXPECTED_SHA256.items():
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input drifted: {path}")

    forbidden = _forbidden_targets()
    eligible: dict[str, list[tuple[str, dict, list[str]]]] = {
        family: [] for family in FAMILY_ORDER
    }
    with CATALOG.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            candidate = _candidate(product, forbidden)
            if candidate is None:
                continue
            family, card = candidate
            parent_asin = str(product["parent_asin"])
            categories = [str(value) for value in product.get("categories") or []]
            eligible[family].append((parent_asin, card, categories))

    rows: list[dict] = []
    selected_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    eligible_counts = {family: len(values) for family, values in eligible.items()}
    for family in FAMILY_ORDER:
        ordered = sorted(eligible[family], key=lambda item: _selection_key(item[0]))
        if len(ordered) < FAMILY_SIZE:
            raise RuntimeError(
                f"family {family} has {len(ordered)} eligible products; "
                f"requires {FAMILY_SIZE}"
            )
        selected = ordered[:FAMILY_SIZE]
        family_counts[family] = len(selected)
        for parent_asin, card, categories in selected:
            if parent_asin in selected_ids:
                raise RuntimeError("fresh suite repeats a target")
            selected_ids.add(parent_asin)
            first, second = card["hard_constraints"][:2]
            category = coarse_category(categories)
            rows.append(
                {
                    "sample_id": f"phase14-{family}-{len(rows) + 1:04d}",
                    "scenario_type": "buying",
                    "user_profile": {
                        "purchase_frequency": "unknown",
                        "average_prior_rating": None,
                        "rating_style": "unknown",
                        "preference_tags": [],
                        "summary": "",
                    },
                    "ground_truth": {"parent_asin": parent_asin},
                    "intent_card": card,
                    "behavior": {"scenario_type": "buying"},
                    "phase14_messages": [
                        f"I'm looking for {category}. A key requirement is: {first}.",
                        f"For that, what matters is: {second}.",
                    ],
                    "phase14_family": family,
                }
            )

    if len(rows) != FAMILY_SIZE * len(FAMILY_ORDER):
        raise RuntimeError("fresh suite cardinality is invalid")
    if selected_ids & forbidden:
        raise RuntimeError("fresh suite overlaps a prior target")
    rows.sort(key=lambda row: str(row["sample_id"]))
    _atomic_jsonl(output, rows)

    fingerprints = sorted(hashlib.sha256(_canonical(row)).digest() for row in rows)
    fingerprint_digest = hashlib.sha256(b"".join(fingerprints)).hexdigest()
    forbidden_digest = hashlib.sha256(
        b"\0".join(value.encode("utf-8") for value in sorted(forbidden))
    ).hexdigest()
    result = {
        "schema_version": 1,
        "lock_id": "phase14-explicit-card-target-disjoint-suite-v1",
        "status": "generated_before_phase14_candidate_implementation",
        "generator_sha256": _sha256(Path(__file__)),
        "contract_sha256": _sha256(CONTRACT),
        "baseline_lock_sha256": _sha256(BASELINE_LOCK),
        "inputs": {
            str(path): {"sha256": expected}
            for path, expected in EXPECTED_SHA256.items()
        },
        "forbidden_target_count": len(forbidden),
        "forbidden_target_set_sha256": forbidden_digest,
        "eligible_family_counts": eligible_counts,
        "selected_family_counts": family_counts,
        "selected_target_count": len(selected_ids),
        "selected_targets_unique": len(selected_ids) == len(rows),
        "prior_target_overlap": 0,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "row_fingerprint_set_sha256": fingerprint_digest,
        "row_values_or_targets_published": False,
        "candidate_or_baseline_retrieval_used": False,
    }
    _atomic_json(manifest, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/ttsc-phase14-explicit-card-v1.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "docs/phase14_fresh_suite_lock.json",
    )
    arguments = parser.parse_args()
    result = build(arguments.output.resolve(), arguments.manifest.resolve())
    print(
        json.dumps(
            {
                "selected_target_count": result["selected_target_count"],
                "selected_family_counts": result["selected_family_counts"],
                "prior_target_overlap": result["prior_target_overlap"],
                "output_sha256": result["output_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
