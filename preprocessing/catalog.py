"""Deterministic normalization of the frozen product catalog.

Raw JSONL rows are validated and reduced to ``NormalizedProduct`` records
with cleaned text fields, canonical price parsing, flattened details, and a
versioned canonical text template (``product-text-v2``) shared by the BM25
store and the dense encoder.  ``scan_catalog`` hashes both the input file
and the derived corpus so embedding builds can prove row-wise alignment
between the catalog and the vector shards.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import struct
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterator


CATALOG_FIELDS = {
    "parent_asin",
    "title",
    "features",
    "description",
    "price",
    "categories",
    "details",
    "average_rating",
    "rating_number",
    "store",
}
TEXT_TEMPLATE_VERSION = "product-text-v2"
MAX_LINE_BYTES = 1024 * 1024

ASIN_RE = re.compile(r"[A-Z0-9]{10}\Z")
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")
SPACE_RE = re.compile(r"\s+")
FROM_PRICE_RE = re.compile(r"from\s+\$?\s*([0-9]+(?:\.[0-9]+)?)\Z", re.IGNORECASE)
EXACT_PRICE_RE = re.compile(r"\$?\s*([0-9]+(?:\.[0-9]+)?)\Z")
MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b",
    re.IGNORECASE,
)

PRIMARY_BRAND_KEYS = {"brand", "brand name"}
MANUFACTURER_KEYS = {"manufacturer"}
PROMOTED_DETAIL_KEYS = (
    "department",
    "material",
    "material type",
    "fabric type",
    "color",
    "size",
    "style",
    "fit type",
    "pattern",
    "closure type",
    "closure",
    "outer material",
    "inner material",
    "sole material",
    "special feature",
    "special features",
    "sport type",
    "sport",
    "occasion",
    "theme",
    "neck style",
    "sleeve type",
    "age range (description)",
    "age range description",
    "target audience",
    "suggested users",
    "product care instructions",
    "care instructions",
    "recommended uses for product",
    "specific uses for product",
)
PROMOTED_DETAIL_KEY_SET = set(PROMOTED_DETAIL_KEYS)
EXCLUDED_DENSE_DETAIL_ROOTS = {"best sellers rank"}
EXCLUDED_DENSE_DETAIL_LEAVES = {
    "asin",
    "date first available",
    "is discontinued by manufacturer",
    "item model number",
    "package dimensions",
    "product dimensions",
}
ROOT_CATEGORIES = {
    "clothing",
    "clothing shoes & jewelry",
    "clothing, shoes & jewelry",
}


class CatalogError(ValueError):
    """Raised when catalog input or generated preprocessing data is invalid."""


@dataclass(frozen=True)
class NormalizedProduct:
    row_index: int
    parent_asin: str
    title: str
    features: tuple[str, ...]
    description: tuple[str, ...]
    categories: tuple[str, ...]
    details: tuple[tuple[str, str], ...]
    store: str
    price_text: str
    price_min_cents: int | None
    price_kind: str
    average_rating: float
    rating_number: int


@dataclass(frozen=True)
class ScanResult:
    path: Path
    row_count: int
    byte_count: int
    catalog_sha256: str
    canonical_text_sha256: str
    product_ids: tuple[str, ...]
    warning_counts: dict[str, int]
    document_character_count: int
    max_document_characters: int
    max_line_bytes: int

    @property
    def mean_document_characters(self) -> float:
        if not self.row_count:
            return 0.0
        return self.document_character_count / self.row_count


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    cleaned: list[str] = []
    for character in text:
        if character.isspace():
            cleaned.append(" ")
            continue
        category = unicodedata.category(character)
        if character == "\ufffd" or category.startswith("C"):
            cleaned.append(" ")
            continue
        cleaned.append(character)
    return SPACE_RE.sub(" ", "".join(cleaned)).strip()


def _clean_string_list(value: object, field: str, row_index: int) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CatalogError(f"row {row_index + 1}: {field} must be a list")
    result: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise CatalogError(
                f"row {row_index + 1}: {field}[{item_index}] must be a string"
            )
        cleaned = _clean_text(item)
        if cleaned:
            result.append(cleaned)
    return tuple(result)


def _deduplicate(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if not value or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return tuple(result)


def _flatten_details(
    value: object,
    row_index: int,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CatalogError(f"row {row_index + 1}: detail keys must be strings")
            cleaned_key = _clean_text(key)
            if cleaned_key:
                yield from _flatten_details(item, row_index, (*path, cleaned_key))
        return
    if isinstance(value, list):
        for item in value:
            yield from _flatten_details(item, row_index, path)
        return
    cleaned_value = _clean_text(value)
    if path and cleaned_value:
        yield " / ".join(path), cleaned_value


def _price(value: object, row_index: int) -> tuple[str, int | None, str]:
    if value is None:
        return "", None, "missing"
    if isinstance(value, bool):
        raise CatalogError(f"row {row_index + 1}: price must not be boolean")
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise CatalogError(f"row {row_index + 1}: price must be finite and non-negative")
        decimal_value = Decimal(str(value))
        try:
            cents = int(
                (decimal_value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            )
        except (InvalidOperation, OverflowError) as error:
            raise CatalogError(f"row {row_index + 1}: invalid numeric price {value!r}") from error
        return str(value), cents, "exact"
    if not isinstance(value, str):
        raise CatalogError(f"row {row_index + 1}: price has unsupported type")
    cleaned = _clean_text(value)
    if not cleaned or cleaned in {"—", "-"}:
        return "", None, "missing"
    kind = "from"
    match = FROM_PRICE_RE.fullmatch(cleaned)
    if match is None:
        kind = "exact"
        match = EXACT_PRICE_RE.fullmatch(cleaned)
    if match is None:
        return cleaned, None, "unparsed"
    try:
        decimal_value = Decimal(match.group(1))
        cents = int(
            (decimal_value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
    except (InvalidOperation, OverflowError) as error:
        raise CatalogError(f"row {row_index + 1}: invalid price {cleaned!r}") from error
    if decimal_value < 0:
        raise CatalogError(f"row {row_index + 1}: price must be non-negative")
    return cleaned, cents, kind


def normalize_product(raw: object, row_index: int) -> NormalizedProduct:
    if not isinstance(raw, dict):
        raise CatalogError(f"row {row_index + 1}: product must be an object")
    missing_fields = sorted(CATALOG_FIELDS.difference(raw))
    if missing_fields:
        raise CatalogError(
            f"row {row_index + 1}: missing required fields {', '.join(missing_fields)}"
        )
    unexpected_fields = sorted(set(raw).difference(CATALOG_FIELDS))
    if unexpected_fields:
        raise CatalogError(
            f"row {row_index + 1}: unexpected fields {', '.join(unexpected_fields)}"
        )

    parent_asin = raw["parent_asin"]
    if not isinstance(parent_asin, str) or ASIN_RE.fullmatch(parent_asin) is None:
        raise CatalogError(f"row {row_index + 1}: invalid parent_asin {parent_asin!r}")
    if not isinstance(raw["title"], str):
        raise CatalogError(f"row {row_index + 1}: title must be a string")
    if raw["store"] is not None and not isinstance(raw["store"], str):
        raise CatalogError(f"row {row_index + 1}: store must be a string or null")
    if not isinstance(raw["details"], dict):
        raise CatalogError(f"row {row_index + 1}: details must be an object")

    average_rating = raw["average_rating"]
    if isinstance(average_rating, bool) or not isinstance(average_rating, (int, float)):
        raise CatalogError(f"row {row_index + 1}: average_rating must be numeric")
    average_rating = float(average_rating)
    if not math.isfinite(average_rating) or not 0 <= average_rating <= 5:
        raise CatalogError(f"row {row_index + 1}: average_rating must be between 0 and 5")
    rating_number = raw["rating_number"]
    if isinstance(rating_number, bool) or not isinstance(rating_number, int) or rating_number < 0:
        raise CatalogError(f"row {row_index + 1}: rating_number must be non-negative integer")

    price_text, price_min_cents, price_kind = _price(raw["price"], row_index)
    return NormalizedProduct(
        row_index=row_index,
        parent_asin=parent_asin,
        title=_clean_text(raw["title"]),
        features=_clean_string_list(raw["features"], "features", row_index),
        description=_clean_string_list(raw["description"], "description", row_index),
        categories=_clean_string_list(raw["categories"], "categories", row_index),
        details=tuple(_flatten_details(raw["details"], row_index)),
        store=_clean_text(raw["store"]),
        price_text=price_text,
        price_min_cents=price_min_cents,
        price_kind=price_kind,
        average_rating=average_rating,
        rating_number=rating_number,
    )


def _brand_aliases(product: NormalizedProduct) -> tuple[str, ...]:
    values = [product.store]
    values.extend(
        value
        for key, value in product.details
        if key.casefold().split(" / ")[-1] in PRIMARY_BRAND_KEYS
    )
    brands = _deduplicate(values)
    if brands:
        return brands
    return _deduplicate(
        [
            value
            for key, value in product.details
            if key.casefold().split(" / ")[-1] in MANUFACTURER_KEYS
        ]
    )


def _evaluator_corpus(product: NormalizedProduct) -> str:
    return " ".join(
        (
            product.title,
            *product.features,
            *(f"{key} {value}" for key, value in product.details),
            *product.description,
            *product.categories,
            product.store,
        )
    )


def _detected_attributes(product: NormalizedProduct) -> tuple[tuple[str, str], ...]:
    corpus = _evaluator_corpus(product)
    result: list[tuple[str, str]] = []
    material = MATERIAL_RE.search(corpus)
    if material:
        result.append(("Detected Material", material.group(1).lower()))
    color = COLOR_RE.search(corpus)
    if color:
        result.append(("Detected Color", color.group(1).lower()))
    return tuple(result)


def _promoted_attributes(product: NormalizedProduct) -> tuple[tuple[str, str], ...]:
    by_key: dict[str, tuple[str, str]] = {}
    for key, value in product.details:
        leaf = key.casefold().split(" / ")[-1]
        if leaf in PROMOTED_DETAIL_KEY_SET and leaf not in by_key:
            by_key[leaf] = (key.split(" / ")[-1], value)

    result: list[tuple[str, str]] = []
    for key in PROMOTED_DETAIL_KEYS:
        if key in by_key:
            result.append(by_key[key])

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, value in result:
        identity = key.casefold(), value.casefold()
        if identity not in seen:
            seen.add(identity)
            unique.append((key, value))
    return tuple(unique)


def _coarse_category(categories: tuple[str, ...]) -> str:
    cleaned: list[str] = []
    for category in categories:
        for part in category.split(","):
            part = part.strip()
            if part and part.casefold() not in ROOT_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:])


def _clean_constraint(value: str, limit: int = 180) -> str:
    return SPACE_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _search_clues(product: NormalizedProduct) -> tuple[str, ...]:
    candidates: list[str] = []
    for key, value in _detected_attributes(product):
        candidates.append(value if key == "Detected Material" else f"color: {value}")
    candidates.extend(product.features)
    candidates.extend(f"{key}: {value}" for key, value in product.details)
    if product.price_text:
        candidates.append(f"budget around ${product.price_text}")
    return _deduplicate(
        [_clean_constraint(candidate) for candidate in candidates if _clean_constraint(candidate)]
    )[:4]


def _display_price(product: NormalizedProduct) -> str:
    if product.price_min_cents is None:
        return product.price_text
    amount = f"${product.price_min_cents / 100:.2f}"
    return f"from {amount}" if product.price_kind == "from" else amount


def canonical_product_text(product: NormalizedProduct) -> str:
    sections: list[str] = []
    if product.title:
        sections.append(f"Title: {product.title}")
    coarse_category = _coarse_category(product.categories)
    if coarse_category:
        sections.append(f"Category: {coarse_category}")
    clues = _search_clues(product)
    if clues:
        sections.append(f"Search Clues: {' | '.join(clues)}")
    if product.categories:
        sections.append(f"Category Path: {' > '.join(product.categories)}")

    brands = _brand_aliases(product)
    if brands:
        sections.append(f"Brand: {' | '.join(brands)}")

    attributes = (*_detected_attributes(product), *_promoted_attributes(product))
    if attributes:
        sections.append(
            "Attributes: " + " | ".join(f"{key}: {value}" for key, value in attributes)
        )

    features = _deduplicate(product.features)
    if features:
        sections.append(f"Features: {' | '.join(features)}")

    promoted_keys = {key.casefold() for key, _ in attributes}
    details: list[str] = []
    seen_details: set[tuple[str, str]] = set()
    for key, value in product.details:
        root = key.casefold().split(" / ")[0]
        leaf = key.casefold().split(" / ")[-1]
        if (
            root in EXCLUDED_DENSE_DETAIL_ROOTS
            or leaf in EXCLUDED_DENSE_DETAIL_LEAVES
            or leaf in PRIMARY_BRAND_KEYS
            or leaf in MANUFACTURER_KEYS
            or leaf in promoted_keys
        ):
            continue
        identity = key.casefold(), value.casefold()
        if identity in seen_details:
            continue
        seen_details.add(identity)
        details.append(f"{key}: {value}")
    descriptions = _deduplicate(product.description)
    if descriptions:
        sections.append(f"Description: {' | '.join(descriptions)}")
    if details:
        sections.append(f"Details: {' | '.join(details)}")
    price = _display_price(product)
    if price:
        sections.append(f"Price: {price}")

    document = "\n".join(sections)
    if not document:
        raise CatalogError(
            f"row {product.row_index + 1}: product has no searchable text after normalization"
        )
    return document


def _product_from_line(raw_line: bytes, row_index: int, path: Path) -> NormalizedProduct:
    if len(raw_line) > MAX_LINE_BYTES:
        raise CatalogError(
            f"row {row_index + 1}: line exceeds {MAX_LINE_BYTES} bytes in {path}"
        )
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CatalogError(f"row {row_index + 1}: invalid UTF-8 in {path}") from error
    if not text.strip():
        raise CatalogError(f"row {row_index + 1}: blank JSONL row in {path}")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise CatalogError(f"row {row_index + 1}: invalid JSON in {path}: {error.msg}") from error
    return normalize_product(raw, row_index)


def iter_normalized_products(path: str | Path) -> Iterator[NormalizedProduct]:
    catalog_path = Path(path)
    with catalog_path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            yield _product_from_line(raw_line, row_index, catalog_path)


def scan_catalog(path: str | Path, expected_rows: int | None = None) -> ScanResult:
    catalog_path = Path(path)
    catalog_hash = hashlib.sha256()
    canonical_hash = hashlib.sha256()
    product_ids: list[str] = []
    seen_ids: set[str] = set()
    warnings: Counter[str] = Counter()
    total_document_characters = 0
    max_document_characters = 0
    max_observed_line_bytes = 0

    with catalog_path.open("rb") as handle:
        for row_index, raw_line in enumerate(handle):
            catalog_hash.update(raw_line)
            max_observed_line_bytes = max(max_observed_line_bytes, len(raw_line))
            product = _product_from_line(raw_line, row_index, catalog_path)
            if product.parent_asin in seen_ids:
                raise CatalogError(
                    f"row {row_index + 1}: duplicate parent_asin {product.parent_asin}"
                )
            seen_ids.add(product.parent_asin)
            product_ids.append(product.parent_asin)

            if not product.title:
                warnings["empty_title"] += 1
            if not product.store:
                warnings["empty_store"] += 1
            if not product.features:
                warnings["empty_features"] += 1
            if not product.description:
                warnings["empty_description"] += 1
            if not product.details:
                warnings["empty_details"] += 1
            warnings[f"price_{product.price_kind}"] += 1

            document = canonical_product_text(product)
            document_bytes = document.encode("utf-8")
            canonical_hash.update(struct.pack(">Q", len(document_bytes)))
            canonical_hash.update(document_bytes)
            total_document_characters += len(document)
            max_document_characters = max(max_document_characters, len(document))

    row_count = len(product_ids)
    if row_count == 0:
        raise CatalogError(f"catalog is empty: {catalog_path}")
    if expected_rows is not None and row_count != expected_rows:
        raise CatalogError(
            f"expected {expected_rows} products but found {row_count} in {catalog_path}"
        )

    return ScanResult(
        path=catalog_path,
        row_count=row_count,
        byte_count=catalog_path.stat().st_size,
        catalog_sha256=catalog_hash.hexdigest(),
        canonical_text_sha256=canonical_hash.hexdigest(),
        product_ids=tuple(product_ids),
        warning_counts=dict(sorted(warnings.items())),
        document_character_count=total_document_characters,
        max_document_characters=max_document_characters,
        max_line_bytes=max_observed_line_bytes,
    )
