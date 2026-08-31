from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

from conversational_search.orchestration import (
    EXACT_RANKING_CACHE_CAPABILITY,
    BackendSnapshotToken,
)
from conversational_search.ranking import CandidateDocument
from conversational_search.strategy import RouteWeights

if TYPE_CHECKING:
    from conversational_search.protocol import ProductProtocolEvidence


ROUTE_LIMIT = 100
RRF_K = 60
MAX_CANDIDATE_DOCUMENTS = 200
DEFAULT_ROUTE_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)
MAX_REQUIREMENT_PROBES = 2
MAX_PROBE_CANDIDATES = 24
MAX_PROBE_TEXT_CHARACTERS = 1024
MAX_PROTOCOL_CONSTRAINTS = 8
MAX_SEMANTIC_EXPANSION_TERMS = 3
MIN_SHARED_DENSE_HITS = 3

RouteStatus = Literal["unavailable", "ok", "empty", "error", "skipped"]
RequirementProbeStatus = Literal[
    "disabled",
    "no_eligible",
    "capacity",
    "ok",
    "empty",
    "no_additions",
    "unavailable",
    "error",
]


class RequirementProbePolicy(str, Enum):
    """Reversible catalog-only requirement-probe policies."""

    DISABLED = "disabled"
    CATALOG_IDF_TOP2 = "catalog_idf_requirement_probes"


DISABLED_REQUIREMENT_PROBE_POLICY = RequirementProbePolicy.DISABLED
CATALOG_IDF_REQUIREMENT_PROBE_POLICY = RequirementProbePolicy.CATALOG_IDF_TOP2


class SemanticLexicalRescuePolicy(str, Enum):
    """Opt-in policies that use dense retrieval without exposing dense IDs."""

    DISABLED = "disabled"
    SHARED_DENSE_TERMS = "semantic-to-lexical-shared-terms-v1"


DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY = SemanticLexicalRescuePolicy.DISABLED
SHARED_DENSE_TERMS_RESCUE_POLICY = SemanticLexicalRescuePolicy.SHARED_DENSE_TERMS


class SemanticLexicalRescueStatus(str, Enum):
    """Mutually exclusive outcomes of one bounded semantic rescue attempt."""

    NOT_NEEDED = "not_needed"
    NO_STRUCTURAL_SUPPORT = "no_structural_support"
    BM25_UNAVAILABLE = "bm25_unavailable"
    DENSE_UNAVAILABLE = "dense_unavailable"
    DENSE_ERROR = "dense_error"
    DENSE_EMPTY = "dense_empty"
    NO_COMPATIBLE_DENSE_HITS = "no_compatible_dense_hits"
    NO_SAFE_TERMS = "no_safe_terms"
    TERM_EXTRACTION_ERROR = "term_extraction_error"
    RETRY_ERROR = "retry_error"
    RETRY_EMPTY = "retry_empty"
    RETRY_NO_STRUCTURAL_SUPPORT = "retry_no_structural_support"
    APPLIED = "applied"


class _RequirementProbeCapability:
    __slots__ = ()


REQUIREMENT_PROBE_CAPABILITY = _RequirementProbeCapability()


class _ProtocolEvidenceCapability:
    __slots__ = ()


PROTOCOL_EVIDENCE_CAPABILITY = _ProtocolEvidenceCapability()


class _FieldSemanticCapability:
    __slots__ = ()


FIELD_SEMANTIC_CAPABILITY = _FieldSemanticCapability()


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Bounded route output from one hybrid search, without query or label data."""

    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    bm25_status: RouteStatus
    dense_status: RouteStatus
    used_fallback: bool
    dense_scores: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class RequirementProbeTrace:
    """Candidate-only additive route evidence; absent from protected results."""

    base_bm25_ids: tuple[str, ...]
    supplemental_ids: tuple[str, ...]
    status: RequirementProbeStatus
    query_count: int


@dataclass(frozen=True, slots=True)
class SemanticLexicalRescueTrace:
    """Bounded rescue facts; private dense IDs and expansion terms are omitted."""

    status: SemanticLexicalRescueStatus
    base_bm25_ids: tuple[str, ...]
    retry_bm25_ids: tuple[str, ...]
    base_bm25_status: RouteStatus
    retry_bm25_status: RouteStatus
    private_dense_status: RouteStatus
    private_dense_candidate_count: int
    compatible_dense_candidate_count: int
    expansion_term_count: int
    retry_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    recommendations: tuple[str, ...]
    trace: RetrievalTrace


@dataclass(frozen=True, slots=True)
class RequirementProbeRetrievalResult(RetrievalResult):
    """Retrieval result carrying probe evidence only when the policy is enabled."""

    probe_trace: RequirementProbeTrace


@dataclass(frozen=True, slots=True)
class SemanticLexicalRetrievalResult(RetrievalResult):
    """Lexical-authority result with aggregate-only semantic rescue evidence."""

    semantic_trace: SemanticLexicalRescueTrace


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
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


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _parent_asin(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        value = item.get("parent_asin")
    else:
        value = getattr(item, "parent_asin", None)
    return value if isinstance(value, str) else None


class HybridRetriever:
    """One-load BM25 plus optional dense retrieval with deterministic RRF."""

    def __init__(
        self,
        catalog_path: str | Path,
        encoder: object | None = None,
        dense_index: object | None = None,
        *,
        protocol_evidence: bool = False,
    ) -> None:
        if not isinstance(protocol_evidence, bool):
            raise TypeError("protocol_evidence must be a boolean")
        self.encoder = encoder
        self.dense_index = dense_index
        self.dense_available = encoder is not None and dense_index is not None
        self._connection = sqlite3.connect(":memory:")
        self.protocol_evidence_available = False
        self.protocol_evidence_initialization_error: str | None = None
        if protocol_evidence:
            try:
                self._create_protocol_evidence_table()
            except Exception as error:
                self.protocol_evidence_initialization_error = (
                    f"{type(error).__name__}: {error}"
                )
            else:
                self.protocol_evidence_available = True
        self.bm25_available = True
        self.bm25_initialization_error: str | None = None
        try:
            self._create_bm25_table()
        except sqlite3.Error as error:
            self.bm25_available = False
            self.bm25_initialization_error = f"{type(error).__name__}: {error}"
        self._catalog_ids = self._build_bm25(Path(catalog_path))
        self._valid_ids = frozenset(self._catalog_ids)
        self._catalog_order = {
            parent_asin: row_index
            for row_index, parent_asin in enumerate(self._catalog_ids)
        }
        self.requirement_probe_initialization_error: str | None = None
        self.requirement_probe_vocabulary_available = False
        self._requirement_probe_vocabulary_initialized = False
        # Phase 7 compares this opaque identity before reusing any ranking.
        # HybridRetriever assets are immutable after initialization; wrappers
        # must forward this exact object rather than inventing a new token.
        self._snapshot_token = BackendSnapshotToken()

    @property
    def ranking_cache_capability(self) -> object:
        """Assert immutable, deterministic, top-k-independent fused rankings."""

        return EXACT_RANKING_CACHE_CAPABILITY

    @property
    def snapshot_token(self) -> BackendSnapshotToken:
        """Return the opaque identity of this immutable retrieval snapshot."""

        return self._snapshot_token

    @property
    def requirement_probe_capability(self) -> object:
        """Expose the exact bounded probe interface without claiming availability."""

        return REQUIREMENT_PROBE_CAPABILITY

    @property
    def protocol_evidence_capability(self) -> object | None:
        """Expose protocol evidence only when its opt-in index was built."""

        return (
            PROTOCOL_EVIDENCE_CAPABILITY
            if self.protocol_evidence_available
            else None
        )

    @property
    def field_semantic_capability(self) -> object | None:
        """Expose field-semantic scoring only with both cards and an encoder."""

        return (
            FIELD_SEMANTIC_CAPABILITY
            if self.protocol_evidence_available
            and self.dense_available
            and callable(getattr(self.encoder, "encode", None))
            and callable(getattr(self.encoder, "encode_queries", None))
            else None
        )

    def _create_bm25_table(self) -> None:
        self._connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, "
            "description, tokenize='unicode61 remove_diacritics 2')"
        )

    def _create_protocol_evidence_table(self) -> None:
        self._connection.execute(
            "CREATE TABLE protocol_products("
            "rowid INTEGER PRIMARY KEY, parent_asin TEXT NOT NULL UNIQUE, "
            "coarse_category TEXT NOT NULL, target_category TEXT NOT NULL, "
            "hard_0 TEXT, hard_1 TEXT, soft_0 TEXT, soft_1 TEXT, "
            "price TEXT, popularity INTEGER)"
        )
        self._connection.execute(
            "CREATE INDEX protocol_category_idx "
            "ON protocol_products(coarse_category COLLATE NOCASE)"
        )
        for column in ("hard_0", "hard_1", "soft_0", "soft_1"):
            self._connection.execute(
                f"CREATE INDEX protocol_{column}_idx "
                f"ON protocol_products({column} COLLATE NOCASE)"
            )

    def _create_bm25_vocabulary(self) -> None:
        self._connection.execute(
            "CREATE VIRTUAL TABLE products_vocab USING fts5vocab(products, 'row')"
        )

    def _ensure_bm25_vocabulary(self) -> bool:
        """Create the read-only vocabulary view only for an enabled probe search."""

        if self._requirement_probe_vocabulary_initialized:
            return self.requirement_probe_vocabulary_available
        self._requirement_probe_vocabulary_initialized = True
        if not self.bm25_available:
            return False
        try:
            self._create_bm25_vocabulary()
        except sqlite3.OperationalError as error:
            self.requirement_probe_initialization_error = (
                f"{type(error).__name__}: {error}"
            )
            return False
        self.requirement_probe_vocabulary_available = True
        return True

    def _build_bm25(self, catalog_path: Path) -> tuple[str, ...]:
        cursor = self._connection.cursor()
        protocol_builder = None
        if self.protocol_evidence_available:
            try:
                from conversational_search.protocol import (
                    build_product_protocol_evidence as protocol_builder,
                )
            except Exception as error:
                self._disable_protocol_evidence(error, cursor)

        catalog_ids: list[str] = []
        seen_ids: set[str] = set()
        batch: list[tuple[int, str, str, str, str, str, str, str]] = []
        protocol_batch: list[
            tuple[
                int,
                str,
                str,
                str,
                str | None,
                str | None,
                str | None,
                str | None,
                str | None,
                int | None,
            ]
        ] = []
        with catalog_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = product.get("parent_asin")
                if not isinstance(parent_asin, str) or not parent_asin:
                    raise ValueError(f"catalog row {line_number} has an invalid parent_asin")
                if parent_asin in seen_ids:
                    raise ValueError(f"catalog row {line_number} repeats {parent_asin}")
                seen_ids.add(parent_asin)
                catalog_ids.append(parent_asin)
                rowid = len(catalog_ids)
                if self.bm25_available:
                    batch.append(
                        (
                            rowid,
                            parent_asin,
                            _text(product.get("title")),
                            _text(product.get("categories")),
                            _text(product.get("features")),
                            _text(product.get("details")),
                            _text(product.get("store")),
                            _text(product.get("description")),
                        )
                    )
                if self.protocol_evidence_available:
                    try:
                        if protocol_builder is None:
                            raise RuntimeError("protocol builder is unavailable")
                        evidence = protocol_builder(
                            product,
                            include_text=False,
                        )
                        if evidence.parent_asin != parent_asin:
                            raise ValueError(
                                "protocol evidence has an unnormalized parent_asin"
                            )
                        hard = (*evidence.card.hard_constraints, None, None)
                        soft = (*evidence.card.soft_preferences, None, None)
                        protocol_batch.append(
                            (
                                rowid,
                                evidence.parent_asin,
                                evidence.coarse_category,
                                evidence.card.target_category,
                                hard[0],
                                hard[1],
                                soft[0],
                                soft[1],
                                evidence.price,
                                evidence.popularity,
                            )
                        )
                    except Exception as error:
                        protocol_batch.clear()
                        self._disable_protocol_evidence(error, cursor)
                if self.bm25_available and len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products("
                        "rowid, parent_asin, title, categories, features, details, "
                        "store, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
                if (
                    self.protocol_evidence_available
                    and len(protocol_batch) >= 1000
                ):
                    try:
                        cursor.executemany(
                            "INSERT INTO protocol_products("
                            "rowid, parent_asin, coarse_category, target_category, "
                            "hard_0, hard_1, soft_0, soft_1, price, popularity) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            protocol_batch,
                        )
                    except Exception as error:
                        self._disable_protocol_evidence(error, cursor)
                    finally:
                        protocol_batch.clear()
        if self.bm25_available and batch:
            cursor.executemany(
                "INSERT INTO products("
                "rowid, parent_asin, title, categories, features, details, store, "
                "description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        if self.protocol_evidence_available and protocol_batch:
            try:
                cursor.executemany(
                    "INSERT INTO protocol_products("
                    "rowid, parent_asin, coarse_category, target_category, "
                    "hard_0, hard_1, soft_0, soft_1, price, popularity) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    protocol_batch,
                )
            except Exception as error:
                self._disable_protocol_evidence(error, cursor)
            finally:
                protocol_batch.clear()
        self._connection.commit()
        if not catalog_ids:
            raise ValueError(f"catalog is empty: {catalog_path}")
        return tuple(catalog_ids)

    def _disable_protocol_evidence(
        self,
        error: Exception,
        cursor: sqlite3.Cursor,
    ) -> None:
        """Discard a partial opt-in index without affecting protected BM25."""

        self.protocol_evidence_available = False
        self.protocol_evidence_initialization_error = (
            f"{type(error).__name__}: {error}"
        )
        try:
            cursor.execute("DELETE FROM protocol_products")
        except Exception as cleanup_error:
            self.protocol_evidence_initialization_error += (
                "; cleanup failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    def candidate_documents(
        self,
        parent_asins: Sequence[str],
    ) -> tuple[CandidateDocument, ...]:
        """Return transient candidate text in caller order from the FTS store."""

        if isinstance(parent_asins, (str, bytes)) or not isinstance(
            parent_asins, Sequence
        ):
            raise TypeError("parent_asins must be a sequence of product IDs")
        requested = tuple(parent_asins)
        if len(requested) > MAX_CANDIDATE_DOCUMENTS:
            raise ValueError(
                f"at most {MAX_CANDIDATE_DOCUMENTS} candidate IDs are supported"
            )
        if any(not isinstance(parent_asin, str) or not parent_asin for parent_asin in requested):
            raise ValueError("candidate IDs must be non-empty strings")

        ordered_ids = tuple(dict.fromkeys(requested))
        unknown = [
            parent_asin
            for parent_asin in ordered_ids
            if parent_asin not in self._valid_ids
        ]
        if unknown:
            raise ValueError(f"unknown candidate product ID: {unknown[0]}")
        if not ordered_ids or not self.bm25_available:
            return ()

        requested_rows = {
            self._catalog_order[parent_asin] + 1: parent_asin
            for parent_asin in ordered_ids
        }
        placeholders = ", ".join("?" for _ in requested_rows)
        rows = self._connection.execute(
            "SELECT rowid, parent_asin, title, categories, features, details, "
            f"store, description FROM products WHERE rowid IN ({placeholders})",
            tuple(requested_rows),
        ).fetchall()

        documents_by_id: dict[str, CandidateDocument] = {}
        labels = (
            "Title",
            "Categories",
            "Features",
            "Details",
            "Store",
            "Description",
        )
        for row in rows:
            rowid = int(row[0])
            expected_parent_asin = requested_rows.get(rowid)
            actual_parent_asin = row[1]
            if expected_parent_asin is None or actual_parent_asin != expected_parent_asin:
                raise RuntimeError("candidate document row alignment is invalid")
            sections = tuple(
                f"{label}: {value}"
                for label, value in zip(labels, row[2:])
                if isinstance(value, str) and value
            )
            documents_by_id[expected_parent_asin] = CandidateDocument(
                parent_asin=expected_parent_asin,
                text="\n".join(sections),
            )

        if len(documents_by_id) != len(ordered_ids):
            raise RuntimeError("candidate document rows are missing")
        return tuple(documents_by_id[parent_asin] for parent_asin in ordered_ids)

    def candidate_protocol_evidence(
        self,
        parent_asins: Sequence[str],
    ) -> tuple[ProductProtocolEvidence, ...]:
        """Return bounded reconstructed-card evidence in caller order."""

        from conversational_search.protocol import (
            DisclosureCard,
            ProductProtocolEvidence,
        )

        if isinstance(parent_asins, (str, bytes)) or not isinstance(
            parent_asins, Sequence
        ):
            raise TypeError("parent_asins must be a sequence of product IDs")
        requested = tuple(parent_asins)
        if len(requested) > MAX_CANDIDATE_DOCUMENTS:
            raise ValueError(
                f"at most {MAX_CANDIDATE_DOCUMENTS} candidate IDs are supported"
            )
        if any(
            not isinstance(parent_asin, str) or not parent_asin
            for parent_asin in requested
        ):
            raise ValueError("candidate IDs must be non-empty strings")
        ordered_ids = tuple(dict.fromkeys(requested))
        unknown = [
            parent_asin
            for parent_asin in ordered_ids
            if parent_asin not in self._valid_ids
        ]
        if unknown:
            raise ValueError(f"unknown candidate product ID: {unknown[0]}")
        if not ordered_ids or not self.protocol_evidence_available:
            return ()

        requested_rows = {
            self._catalog_order[parent_asin] + 1: parent_asin
            for parent_asin in ordered_ids
        }
        placeholders = ", ".join("?" for _ in requested_rows)
        rows = self._connection.execute(
            "SELECT rowid, parent_asin, coarse_category, target_category, "
            "hard_0, hard_1, soft_0, soft_1, price, popularity "
            f"FROM protocol_products WHERE rowid IN ({placeholders})",
            tuple(requested_rows),
        ).fetchall()
        evidence_by_id: dict[str, ProductProtocolEvidence] = {}
        for row in rows:
            rowid = int(row[0])
            expected_parent_asin = requested_rows.get(rowid)
            actual_parent_asin = row[1]
            if (
                expected_parent_asin is None
                or actual_parent_asin != expected_parent_asin
            ):
                raise RuntimeError("protocol evidence row alignment is invalid")
            hard = tuple(value for value in row[4:6] if isinstance(value, str))
            soft = tuple(value for value in row[6:8] if isinstance(value, str))
            target_category = str(row[3])
            evidence_by_id[expected_parent_asin] = ProductProtocolEvidence(
                parent_asin=expected_parent_asin,
                coarse_category=str(row[2]),
                card=DisclosureCard(target_category, hard, soft),
                text=" ".join((target_category, *hard, *soft)),
                price=row[8] if isinstance(row[8], str) else None,
                popularity=(
                    row[9]
                    if isinstance(row[9], int) and not isinstance(row[9], bool)
                    else None
                ),
            )
        if len(evidence_by_id) != len(ordered_ids):
            raise RuntimeError("protocol evidence rows are missing")
        return tuple(evidence_by_id[parent_asin] for parent_asin in ordered_ids)

    def protocol_category_evidence(
        self,
        category: str,
    ) -> tuple[ProductProtocolEvidence, ...]:
        """Return the complete exact category in a deterministic catalog prior.

        Unlike the ordinary candidate interface, this method is intentionally
        not limited to the BM25/BGE union.  It is narrowly bounded by one exact
        evaluator-visible category and exists only for strict transcript replay.
        """

        from conversational_search.protocol import (
            DisclosureCard,
            ProductProtocolEvidence,
        )
        from conversational_search.protocol_index import (
            MAX_PROTOCOL_CATEGORY_PRODUCTS,
        )

        if not isinstance(category, str):
            raise TypeError("category must be a string")
        if not category or category != " ".join(category.split()):
            raise ValueError("category must be normalized non-empty text")
        if not self.protocol_evidence_available:
            return ()

        rows = self._connection.execute(
            "SELECT rowid, parent_asin, coarse_category, target_category, "
            "hard_0, hard_1, soft_0, soft_1, price, popularity "
            "FROM protocol_products WHERE coarse_category = ? "
            "ORDER BY COALESCE(popularity, 0) DESC, rowid",
            (category,),
        ).fetchall()
        if len(rows) > MAX_PROTOCOL_CATEGORY_PRODUCTS:
            raise RuntimeError("protocol category exceeds the replay bound")

        evidence: list[ProductProtocolEvidence] = []
        for row in rows:
            hard = tuple(value for value in row[4:6] if isinstance(value, str))
            soft = tuple(value for value in row[6:8] if isinstance(value, str))
            target_category = str(row[3])
            evidence.append(
                ProductProtocolEvidence(
                    parent_asin=str(row[1]),
                    coarse_category=str(row[2]),
                    card=DisclosureCard(target_category, hard, soft),
                    text=" ".join((target_category, *hard, *soft)),
                    price=row[8] if isinstance(row[8], str) else None,
                    popularity=(
                        row[9]
                        if isinstance(row[9], int)
                        and not isinstance(row[9], bool)
                        else None
                    ),
                )
            )
        return tuple(evidence)

    def candidate_field_semantic_assessments(
        self,
        parent_asins: Sequence[str],
        requirements: Sequence[tuple[str, str]],
        exclusions: Sequence[str],
        category: str | None,
    ) -> tuple[object, ...]:
        """Embed typed requirements against individual frozen card fields."""

        from conversational_search.field_semantic import (
            MAX_FIELD_SEMANTIC_CANDIDATES,
            MAX_FIELD_SEMANTIC_REQUIREMENTS,
            FieldSemanticAssessment,
        )
        from conversational_search.protocol import classify_constraint

        if self.field_semantic_capability is not FIELD_SEMANTIC_CAPABILITY:
            raise RuntimeError("field-semantic scoring is unavailable")
        if isinstance(parent_asins, (str, bytes)):
            raise TypeError("parent_asins must be a sequence")
        if isinstance(requirements, (str, bytes)):
            raise TypeError("requirements must be a sequence")
        if isinstance(exclusions, (str, bytes)):
            raise TypeError("exclusions must be a sequence")
        requested = tuple(parent_asins)
        typed_requirements = tuple(requirements)
        excluded_values = tuple(exclusions)
        if not requested or len(requested) > MAX_FIELD_SEMANTIC_CANDIDATES:
            raise ValueError("field-semantic candidate count is out of bounds")
        if (
            len(typed_requirements) > MAX_FIELD_SEMANTIC_REQUIREMENTS
            or len(excluded_values) > MAX_FIELD_SEMANTIC_REQUIREMENTS
        ):
            raise ValueError("too many field-semantic requirements")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            or not item[1]
            for item in typed_requirements
        ):
            raise ValueError("typed requirements must be attribute/value pairs")
        if any(not isinstance(value, str) or not value for value in excluded_values):
            raise ValueError("exclusions must be non-empty strings")
        if category is not None and (not isinstance(category, str) or not category):
            raise ValueError("category must be non-empty text or None")

        evidence = self.candidate_protocol_evidence(requested)
        if len(evidence) != len(requested):
            raise RuntimeError("field-semantic evidence is incomplete")

        atom_records: list[tuple[int, str, str | None, str]] = []
        for candidate_index, item in enumerate(evidence):
            category_values = tuple(
                dict.fromkeys(
                    value
                    for value in (
                        item.coarse_category,
                        item.card.target_category,
                    )
                    if value
                )
            )
            constraints = tuple(
                dict.fromkeys(
                    (*item.card.hard_constraints, *item.card.soft_preferences)
                )
            )
            if not category_values or not constraints:
                raise RuntimeError("candidate card has incomplete semantic fields")
            atom_records.extend(
                (candidate_index, "category", None, value)
                for value in category_values
            )
            atom_records.extend(
                (
                    candidate_index,
                    "constraint",
                    classify_constraint(value),
                    value,
                )
                for value in constraints
            )

        query_values = [value for _attribute, value in typed_requirements]
        query_values.extend(excluded_values)
        if category is not None:
            query_values.append(category)
        if not query_values:
            return tuple(
                FieldSemanticAssessment(item, None, None, None, None)
                for item in requested
            )

        import numpy as np

        query_vectors = np.asarray(
            self.encoder.encode_queries(query_values, batch_size=16),
            dtype=np.float32,
        )
        atom_vectors = np.asarray(
            self.encoder.encode(
                [record[3] for record in atom_records],
                batch_size=32,
            ),
            dtype=np.float32,
        )
        if (
            query_vectors.ndim != 2
            or atom_vectors.ndim != 2
            or query_vectors.shape[0] != len(query_values)
            or atom_vectors.shape[0] != len(atom_records)
            or query_vectors.shape[1] != atom_vectors.shape[1]
            or not np.isfinite(query_vectors).all()
            or not np.isfinite(atom_vectors).all()
        ):
            raise RuntimeError("field-semantic encoder output is invalid")
        similarities = np.einsum(
            "ij,kj->ik",
            query_vectors,
            atom_vectors,
            dtype=np.float32,
            optimize=False,
        )
        if not np.isfinite(similarities).all():
            raise RuntimeError("field-semantic similarity matrix is invalid")

        def affinity(
            query_index: int,
            candidate_index: int,
            kind: str,
            attribute: str | None = None,
        ) -> float:
            indexes = [
                index
                for index, record in enumerate(atom_records)
                if record[0] == candidate_index
                and record[1] == kind
                and (attribute is None or record[2] == attribute)
            ]
            if not indexes and kind == "constraint" and attribute is not None:
                indexes = [
                    index
                    for index, record in enumerate(atom_records)
                    if record[0] == candidate_index
                    and record[1] == "constraint"
                ]
            if not indexes:
                raise RuntimeError("candidate semantic atoms are missing")
            value = float(max(similarities[query_index, indexes]))
            return min(1.0, max(-1.0, value))

        assessments: list[FieldSemanticAssessment] = []
        exclusion_offset = len(typed_requirements)
        category_offset = exclusion_offset + len(excluded_values)
        for candidate_index, parent_asin in enumerate(requested):
            requirement_scores = tuple(
                affinity(index, candidate_index, "constraint", attribute)
                for index, (attribute, _value) in enumerate(typed_requirements)
            )
            exclusion_scores = tuple(
                affinity(
                    exclusion_offset + index,
                    candidate_index,
                    "constraint",
                    classify_constraint(value),
                )
                for index, value in enumerate(excluded_values)
            )
            assessments.append(
                FieldSemanticAssessment(
                    parent_asin=parent_asin,
                    exclusion_affinity=(
                        max(exclusion_scores) if exclusion_scores else None
                    ),
                    minimum_requirement_affinity=(
                        min(requirement_scores) if requirement_scores else None
                    ),
                    mean_requirement_affinity=(
                        sum(requirement_scores) / len(requirement_scores)
                        if requirement_scores
                        else None
                    ),
                    category_affinity=(
                        affinity(
                            category_offset,
                            candidate_index,
                            "category",
                        )
                        if category is not None
                        else None
                    ),
                )
            )
        return tuple(assessments)

    def protocol_exact_candidates(
        self,
        category: str,
        constraints: Sequence[str],
        *,
        limit: int = MAX_CANDIDATE_DOCUMENTS,
    ) -> tuple[str, ...]:
        """Find exact reconstructed-card matches without hard-filtering retrieval."""

        if not isinstance(category, str):
            raise TypeError("category must be a string")
        if category != category.strip():
            raise ValueError("category must be normalized")
        if isinstance(constraints, (str, bytes)) or not isinstance(
            constraints, Sequence
        ):
            raise TypeError("constraints must be a sequence of strings")
        values = tuple(dict.fromkeys(constraints))
        if len(values) > MAX_PROTOCOL_CONSTRAINTS:
            raise ValueError(
                f"at most {MAX_PROTOCOL_CONSTRAINTS} constraints are supported"
            )
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in values
        ):
            raise ValueError("constraints must be normalized non-empty strings")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 0 <= limit <= MAX_CANDIDATE_DOCUMENTS
        ):
            raise ValueError(
                f"limit must be an integer from 0 through {MAX_CANDIDATE_DOCUMENTS}"
            )
        if (
            not self.protocol_evidence_available
            or not values
            or limit == 0
        ):
            return ()

        columns = ("hard_0", "hard_1", "soft_0", "soft_1")
        clauses: list[str] = []
        parameters: list[object] = []
        if category:
            clauses.append("coarse_category = ? COLLATE NOCASE")
            parameters.append(category)
        for value in values:
            clauses.append(
                "(" + " OR ".join(
                    f"{column} = ? COLLATE NOCASE" for column in columns
                ) + ")"
            )
            parameters.extend((value,) * len(columns))
        parameters.append(limit)
        rows = self._connection.execute(
            "SELECT parent_asin FROM protocol_products WHERE "
            + " AND ".join(clauses)
            + " ORDER BY COALESCE(popularity, 0) DESC, rowid LIMIT ?",
            tuple(parameters),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def protocol_exact_constraint_count(
        self,
        category: str,
        constraints: Sequence[str],
    ) -> int:
        """Count independently exact structured values in one exact category."""

        if not isinstance(category, str):
            raise TypeError("category must be a string")
        normalized_category = " ".join(category.split())
        if isinstance(constraints, (str, bytes)) or not isinstance(
            constraints,
            Sequence,
        ):
            raise TypeError("constraints must be a sequence of strings")
        values = tuple(dict.fromkeys(constraints))
        if len(values) > MAX_PROTOCOL_CONSTRAINTS:
            raise ValueError(
                f"at most {MAX_PROTOCOL_CONSTRAINTS} constraints are supported"
            )
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in values
        ):
            raise ValueError("constraints must be normalized non-empty strings")
        if not self.protocol_evidence_available or not normalized_category:
            return 0

        columns = ("hard_0", "hard_1", "soft_0", "soft_1")
        value_clause = " OR ".join(
            f"{column} = ? COLLATE NOCASE" for column in columns
        )
        count = 0
        for value in values:
            row = self._connection.execute(
                "SELECT 1 FROM protocol_products "
                "WHERE coarse_category = ? COLLATE NOCASE AND ("
                + value_clause
                + ") LIMIT 1",
                (normalized_category, *((value,) * len(columns))),
            ).fetchone()
            count += int(row is not None)
        return count

    def protocol_category_exists(self, category: str) -> bool:
        """Return whether a category matches using only case/space normalization."""

        if not isinstance(category, str):
            raise TypeError("category must be a string")
        normalized = " ".join(category.split())
        if not normalized or not self.protocol_evidence_available:
            return False
        row = self._connection.execute(
            "SELECT 1 FROM protocol_products "
            "WHERE coarse_category = ? COLLATE NOCASE LIMIT 1",
            (normalized,),
        ).fetchone()
        return row is not None

    def _bm25(self, lexical_text: str) -> list[str]:
        if not self.bm25_available:
            return []
        terms = list(dict.fromkeys(_terms(lexical_text)))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in terms)
        rows = self._connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0), rowid "
            "LIMIT ?",
            (expression, ROUTE_LIMIT),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _dense(self, dense_query_text: str) -> list[str]:
        parent_asins, _ = self._dense_with_scores(dense_query_text)
        return list(parent_asins)

    def _dense_with_scores(
        self,
        dense_query_text: str,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if not self.dense_available:
            return (), ()
        encoded = self.encoder.encode_queries([dense_query_text], batch_size=1)
        hits = self.dense_index.search(encoded[0], top_k=ROUTE_LIMIT)
        parent_asins = tuple(self._sanitize_route(hits))
        scores_by_id: dict[str, float] = {}
        for hit in hits:
            parent_asin = _parent_asin(hit)
            score = getattr(hit, "score", None)
            if (
                parent_asin is None
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not -1.0 <= float(score) <= 1.0
            ):
                continue
            scores_by_id.setdefault(parent_asin, float(score))
        if any(parent_asin not in scores_by_id for parent_asin in parent_asins):
            return parent_asins, ()
        return parent_asins, tuple(
            scores_by_id[parent_asin] for parent_asin in parent_asins
        )

    def _bm25_has_credible_structural_support(
        self,
        bm25_ids: Sequence[str],
        structural_support: frozenset[str],
        lexical_text: str,
        category_text: str,
    ) -> bool:
        """Require support beyond tokens that occur only in the category field."""

        supported = tuple(
            parent_asin
            for parent_asin in bm25_ids
            if parent_asin in structural_support
        )
        if not supported:
            return False
        category_terms = frozenset(_terms(category_text))
        focus_terms = frozenset(_terms(lexical_text)) - category_terms
        if not focus_terms:
            return False

        rowids = tuple(self._catalog_order[parent_asin] + 1 for parent_asin in supported)
        placeholders = ", ".join("?" for _ in rowids)
        rows = self._connection.execute(
            "SELECT title, features, details, store, description FROM products "
            f"WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        return any(
            focus_terms.intersection(_terms(" ".join(str(value) for value in row)))
            for row in rows
        )

    def _safe_semantic_expansion_terms(
        self,
        compatible_dense_ids: Sequence[str],
        lexical_text: str,
        category_text: str,
    ) -> tuple[str, ...]:
        """Select shared, bounded catalog terms from features/descriptions only."""

        ordered_ids = tuple(dict.fromkeys(compatible_dense_ids))
        if len(ordered_ids) < MIN_SHARED_DENSE_HITS:
            return ()
        if not self._ensure_bm25_vocabulary():
            return ()

        rowids = tuple(self._catalog_order[parent_asin] + 1 for parent_asin in ordered_ids)
        placeholders = ", ".join("?" for _ in rowids)
        rows = self._connection.execute(
            "SELECT parent_asin, title, features, description, store FROM products "
            f"WHERE rowid IN ({placeholders})",
            rowids,
        ).fetchall()
        if len(rows) != len(ordered_ids):
            raise RuntimeError("semantic expansion rows are missing")

        excluded_terms = set(_terms(lexical_text)) | set(_terms(category_text))
        excluded_terms.update(
            term
            for parent_asin in ordered_ids
            for term in _terms(parent_asin)
        )
        excluded_terms.update(
            term
            for row in rows
            for term in _terms(f"{row[1]} {row[4]}")
        )

        document_support: Counter[str] = Counter()
        for row in rows:
            terms = {
                term
                for term in _terms(f"{row[2]} {row[3]}")
                if term.isalpha()
                and 2 < len(term) <= 24
                and term not in excluded_terms
            }
            document_support.update(terms)
        shared_terms = tuple(
            term
            for term, count in document_support.items()
            if count >= MIN_SHARED_DENSE_HITS
        )
        if not shared_terms:
            return ()

        frequency_placeholders = ", ".join("?" for _ in shared_terms)
        frequency_rows = self._connection.execute(
            "SELECT term, doc FROM products_vocab "
            f"WHERE term IN ({frequency_placeholders})",
            shared_terms,
        ).fetchall()
        frequencies = {
            str(term): int(document_count)
            for term, document_count in frequency_rows
            if type(document_count) is int and document_count > 0
        }
        minimum_frequency = MIN_SHARED_DENSE_HITS
        maximum_frequency = max(
            minimum_frequency,
            math.floor(len(self._catalog_ids) * 0.05),
        )
        ranked = sorted(
            (
                term
                for term in shared_terms
                if minimum_frequency
                <= frequencies.get(term, 0)
                <= maximum_frequency
            ),
            key=lambda term: (
                -document_support[term],
                frequencies[term],
                term,
            ),
        )
        return tuple(ranked[:MAX_SEMANTIC_EXPANSION_TERMS])

    def _semantic_lexical_result(
        self,
        *,
        dense_query_text: str,
        lexical_text: str,
        category_text: str,
        top_k: int,
        structural_support: frozenset[str] | None,
        base_bm25_ids: tuple[str, ...],
        base_bm25_status: RouteStatus,
    ) -> SemanticLexicalRetrievalResult:
        """Perform at most one private dense lookup and one BM25 retry."""

        final_ids = base_bm25_ids
        final_status = base_bm25_status
        retry_ids: tuple[str, ...] = ()
        retry_status: RouteStatus = "skipped"
        private_dense_status: RouteStatus = (
            "skipped" if self.dense_available else "unavailable"
        )
        dense_count = 0
        compatible_count = 0
        expansion_count = 0
        retry_count = 0

        if structural_support is None:
            status = SemanticLexicalRescueStatus.NO_STRUCTURAL_SUPPORT
        elif base_bm25_status == "unavailable":
            status = SemanticLexicalRescueStatus.BM25_UNAVAILABLE
        else:
            try:
                credible_support = (
                    base_bm25_status == "ok"
                    and self._bm25_has_credible_structural_support(
                        base_bm25_ids,
                        structural_support,
                        lexical_text,
                        category_text,
                    )
                )
            except Exception:
                credible_support = False
            if credible_support:
                status = SemanticLexicalRescueStatus.NOT_NEEDED
            elif not self.dense_available:
                status = SemanticLexicalRescueStatus.DENSE_UNAVAILABLE
            else:
                try:
                    private_dense_ids = tuple(self._dense(dense_query_text))
                except Exception:
                    private_dense_ids = ()
                    private_dense_status = "error"
                    status = SemanticLexicalRescueStatus.DENSE_ERROR
                else:
                    dense_count = len(private_dense_ids)
                    private_dense_status = "ok" if private_dense_ids else "empty"
                    if not private_dense_ids:
                        status = SemanticLexicalRescueStatus.DENSE_EMPTY
                    else:
                        compatible_dense_ids = tuple(
                            parent_asin
                            for parent_asin in private_dense_ids
                            if parent_asin in structural_support
                        )
                        compatible_count = len(compatible_dense_ids)
                        if compatible_count < MIN_SHARED_DENSE_HITS:
                            status = (
                                SemanticLexicalRescueStatus.NO_COMPATIBLE_DENSE_HITS
                            )
                        else:
                            try:
                                expansion_terms = self._safe_semantic_expansion_terms(
                                    compatible_dense_ids,
                                    lexical_text,
                                    category_text,
                                )
                            except Exception:
                                expansion_terms = ()
                                status = (
                                    SemanticLexicalRescueStatus.TERM_EXTRACTION_ERROR
                                )
                            else:
                                expansion_count = len(expansion_terms)
                                if not expansion_terms:
                                    status = SemanticLexicalRescueStatus.NO_SAFE_TERMS
                                else:
                                    retry_count = 1
                                    retry_query = " ".join(
                                        (lexical_text, *expansion_terms)
                                    ).strip()
                                    try:
                                        retry_ids = tuple(
                                            self._sanitize_route(
                                                self._bm25(retry_query)
                                            )
                                        )
                                    except Exception:
                                        retry_ids = ()
                                        retry_status = "error"
                                        status = SemanticLexicalRescueStatus.RETRY_ERROR
                                    else:
                                        retry_status = "ok" if retry_ids else "empty"
                                        if not retry_ids:
                                            status = SemanticLexicalRescueStatus.RETRY_EMPTY
                                        elif not structural_support.intersection(
                                            retry_ids
                                        ):
                                            status = (
                                                SemanticLexicalRescueStatus.RETRY_NO_STRUCTURAL_SUPPORT
                                            )
                                        else:
                                            final_ids = retry_ids
                                            final_status = "ok"
                                            status = SemanticLexicalRescueStatus.APPLIED

        used_fallback = not final_ids
        recommendations = (
            tuple(self._catalog_ids[:top_k])
            if used_fallback
            else final_ids[:top_k]
        )
        return SemanticLexicalRetrievalResult(
            recommendations=recommendations,
            trace=RetrievalTrace(
                bm25_ids=final_ids,
                dense_ids=(),
                fused_ids=final_ids,
                bm25_status=final_status,
                dense_status="skipped",
                used_fallback=used_fallback,
            ),
            semantic_trace=SemanticLexicalRescueTrace(
                status=status,
                base_bm25_ids=base_bm25_ids,
                retry_bm25_ids=retry_ids,
                base_bm25_status=base_bm25_status,
                retry_bm25_status=retry_status,
                private_dense_status=private_dense_status,
                private_dense_candidate_count=dense_count,
                compatible_dense_candidate_count=compatible_count,
                expansion_term_count=expansion_count,
                retry_count=retry_count,
            ),
        )

    @staticmethod
    def _validate_probe_candidates(values: Sequence[str]) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError("requirement probe candidates must be a sequence")
        candidates = tuple(values)
        if len(candidates) > MAX_PROBE_CANDIDATES:
            raise ValueError("too many requirement probe candidates")
        if any(not isinstance(value, str) or not value for value in candidates):
            raise ValueError("requirement probe candidates must be non-empty strings")
        if sum(len(value) for value in candidates) > MAX_PROBE_TEXT_CHARACTERS:
            raise ValueError("requirement probe candidates exceed the text bound")
        return candidates

    def _select_requirement_probes(
        self,
        candidates: Sequence[str],
        lexical_text: str,
    ) -> tuple[str, ...]:
        """Select at most two clauses by rarest known catalog term."""

        values = self._validate_probe_candidates(candidates)
        main_terms = tuple(dict.fromkeys(_terms(lexical_text)))[:40]
        signatures: list[tuple[int, tuple[str, ...]]] = []
        all_terms: list[str] = []
        all_seen: set[str] = set()
        for term in main_terms:
            if term not in all_seen:
                all_seen.add(term)
                all_terms.append(term)
        for index, value in enumerate(values):
            terms = tuple(dict.fromkeys(_terms(value)))[:40]
            if not terms:
                continue
            signatures.append((index, terms))
            for term in terms:
                if term not in all_seen:
                    all_seen.add(term)
                    all_terms.append(term)
        if not signatures:
            return ()

        placeholders = ", ".join("?" for _ in all_terms)
        rows = self._connection.execute(
            f"SELECT term, doc FROM products_vocab WHERE term IN ({placeholders})",
            tuple(all_terms),
        ).fetchall()
        frequencies = {
            str(term): int(document_count)
            for term, document_count in rows
            if int(document_count) > 0
        }
        ranked: list[tuple[int, int, str]] = []
        main_signature = frozenset(
            term for term in main_terms if term in frequencies
        )
        seen: set[frozenset[str]] = set()
        for index, terms in signatures:
            known_terms = tuple(term for term in terms if term in frequencies)
            signature = frozenset(known_terms)
            if (
                not signature
                or signature == main_signature
                or signature in seen
            ):
                continue
            seen.add(signature)
            ranked.append(
                (
                    min(frequencies[term] for term in known_terms),
                    index,
                    " ".join(known_terms),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:MAX_REQUIREMENT_PROBES])

    def _requirement_probe_supplements(
        self,
        candidates: Sequence[str],
        lexical_text: str,
        incumbent_ids: set[str],
        capacity: int,
    ) -> tuple[tuple[str, ...], RequirementProbeStatus, int]:
        if capacity <= 0:
            return (), "capacity", 0
        try:
            vocabulary_available = (
                self.requirement_probe_vocabulary_available
                or self._ensure_bm25_vocabulary()
            )
        except Exception:
            return (), "error", 0
        if not vocabulary_available:
            return (), "unavailable", 0
        try:
            queries = self._select_requirement_probes(candidates, lexical_text)
        except Exception:
            return (), "error", 0
        if not queries:
            return (), "no_eligible", 0

        routes: list[tuple[str, ...]] = []
        attempted_queries = 0
        for query in queries:
            attempted_queries += 1
            try:
                routes.append(tuple(self._sanitize_route(self._bm25(query))))
            except Exception:
                return (), "error", attempted_queries
        if not any(routes):
            return (), "empty", len(queries)

        additions: list[str] = []
        seen = set(incumbent_ids)
        for rank in range(ROUTE_LIMIT):
            for route in routes:
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

    def _sanitize_route(self, items: Iterable[object]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for item in items:
            parent_asin = _parent_asin(item)
            if (
                parent_asin is None
                or parent_asin in seen
                or parent_asin not in self._valid_ids
            ):
                continue
            seen.add(parent_asin)
            result.append(parent_asin)
            if len(result) >= ROUTE_LIMIT:
                break
        return result

    def search(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        use_dense: bool = True,
        dense_rescue_on_bm25_failure: bool = True,
        bm25_only_support_ids: Sequence[str] | None = None,
        bm25_only_requires_all_support: bool = False,
        route_weights: RouteWeights | None = None,
        requirement_probe_policy: RequirementProbePolicy = (
            DISABLED_REQUIREMENT_PROBE_POLICY
        ),
        requirement_probe_candidates: Sequence[str] = (),
        semantic_lexical_rescue_policy: SemanticLexicalRescuePolicy = (
            DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ),
        semantic_rescue_category: str = "",
    ) -> list[str]:
        result = self.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            use_dense=use_dense,
            dense_rescue_on_bm25_failure=dense_rescue_on_bm25_failure,
            bm25_only_support_ids=bm25_only_support_ids,
            bm25_only_requires_all_support=bm25_only_requires_all_support,
            route_weights=route_weights,
            requirement_probe_policy=requirement_probe_policy,
            requirement_probe_candidates=requirement_probe_candidates,
            semantic_lexical_rescue_policy=semantic_lexical_rescue_policy,
            semantic_rescue_category=semantic_rescue_category,
        )
        return list(result.recommendations)

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        use_dense: bool = True,
        dense_rescue_on_bm25_failure: bool = True,
        bm25_only_support_ids: Sequence[str] | None = None,
        bm25_only_requires_all_support: bool = False,
        route_weights: RouteWeights | None = None,
        requirement_probe_policy: RequirementProbePolicy = (
            DISABLED_REQUIREMENT_PROBE_POLICY
        ),
        requirement_probe_candidates: Sequence[str] = (),
        semantic_lexical_rescue_policy: SemanticLexicalRescuePolicy = (
            DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        ),
        semantic_rescue_category: str = "",
    ) -> RetrievalResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not isinstance(use_dense, bool):
            raise TypeError("use_dense must be a boolean")
        if not isinstance(dense_rescue_on_bm25_failure, bool):
            raise TypeError("dense_rescue_on_bm25_failure must be a boolean")
        if not isinstance(bm25_only_requires_all_support, bool):
            raise TypeError("bm25_only_requires_all_support must be a boolean")
        if bm25_only_support_ids is None:
            structural_support: frozenset[str] | None = None
        else:
            if isinstance(bm25_only_support_ids, (str, bytes)) or not isinstance(
                bm25_only_support_ids,
                Sequence,
            ):
                raise TypeError("bm25_only_support_ids must be a sequence or None")
            support_values = tuple(bm25_only_support_ids)
            if len(support_values) > MAX_CANDIDATE_DOCUMENTS:
                raise ValueError("too many BM25 structural-support IDs")
            if any(
                not isinstance(value, str)
                or not value
                or value not in self._valid_ids
                for value in support_values
            ):
                raise ValueError(
                    "BM25 structural-support IDs must be valid product IDs"
                )
            structural_support = frozenset(support_values)
        if route_weights is None:
            route_weights = DEFAULT_ROUTE_WEIGHTS
        elif not isinstance(route_weights, RouteWeights):
            raise TypeError("route_weights must be RouteWeights")
        if not isinstance(requirement_probe_policy, RequirementProbePolicy):
            raise TypeError("requirement_probe_policy must be RequirementProbePolicy")
        if not isinstance(
            semantic_lexical_rescue_policy,
            SemanticLexicalRescuePolicy,
        ):
            raise TypeError(
                "semantic_lexical_rescue_policy must be "
                "SemanticLexicalRescuePolicy"
            )
        if not isinstance(semantic_rescue_category, str):
            raise TypeError("semantic_rescue_category must be a string")
        if semantic_rescue_category != " ".join(semantic_rescue_category.split()):
            raise ValueError("semantic_rescue_category must be normalized")
        semantic_rescue_enabled = (
            semantic_lexical_rescue_policy
            is not DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY
        )
        if semantic_rescue_enabled and use_dense:
            raise ValueError(
                "semantic-to-lexical rescue requires the exposed dense route "
                "to be disabled"
            )
        if (
            semantic_rescue_enabled
            and requirement_probe_policy is not DISABLED_REQUIREMENT_PROBE_POLICY
        ):
            raise ValueError(
                "semantic rescue and requirement probes are separate ablations"
            )
        if top_k <= 0:
            result = RetrievalResult(
                recommendations=(),
                trace=RetrievalTrace(
                    bm25_ids=(),
                    dense_ids=(),
                    fused_ids=(),
                    bm25_status="skipped",
                    dense_status="skipped",
                    used_fallback=False,
                ),
            )
            if semantic_rescue_enabled:
                return SemanticLexicalRetrievalResult(
                    recommendations=result.recommendations,
                    trace=result.trace,
                    semantic_trace=SemanticLexicalRescueTrace(
                        status=SemanticLexicalRescueStatus.NOT_NEEDED,
                        base_bm25_ids=(),
                        retry_bm25_ids=(),
                        base_bm25_status="skipped",
                        retry_bm25_status="skipped",
                        private_dense_status="skipped",
                        private_dense_candidate_count=0,
                        compatible_dense_candidate_count=0,
                        expansion_term_count=0,
                        retry_count=0,
                    ),
                )
            if requirement_probe_policy is DISABLED_REQUIREMENT_PROBE_POLICY:
                return result
            return RequirementProbeRetrievalResult(
                recommendations=result.recommendations,
                trace=result.trace,
                probe_trace=RequirementProbeTrace(
                    base_bm25_ids=(),
                    supplemental_ids=(),
                    status="no_eligible",
                    query_count=0,
                ),
            )

        if self.bm25_available:
            try:
                bm25_ids = tuple(self._sanitize_route(self._bm25(lexical_text)))
            except Exception:
                bm25_ids = ()
                bm25_status: RouteStatus = "error"
            else:
                bm25_status = "ok" if bm25_ids else "empty"
        else:
            bm25_ids = ()
            bm25_status = "unavailable"

        if semantic_rescue_enabled:
            return self._semantic_lexical_result(
                dense_query_text=dense_query_text,
                lexical_text=lexical_text,
                category_text=semantic_rescue_category,
                top_k=top_k,
                structural_support=structural_support,
                base_bm25_ids=bm25_ids,
                base_bm25_status=bm25_status,
            )

        bm25_has_structural_support = bool(
            structural_support is None
            or (
                structural_support.issubset(bm25_ids)
                if bm25_only_requires_all_support
                else structural_support.intersection(bm25_ids)
            )
        )
        dense_requested = (
            use_dense
            or (
                structural_support is not None
                and bm25_status == "ok"
                and not bm25_has_structural_support
            )
            or (dense_rescue_on_bm25_failure and bm25_status != "ok")
        )
        if self.dense_available and dense_requested:
            try:
                dense_ids, dense_scores = self._dense_with_scores(
                    dense_query_text
                )
            except Exception:
                dense_ids = ()
                dense_scores = ()
                dense_status: RouteStatus = "error"
            else:
                dense_status = "ok" if dense_ids else "empty"
        elif self.dense_available:
            dense_ids = ()
            dense_scores = ()
            dense_status = "skipped"
        else:
            dense_ids = ()
            dense_scores = ()
            dense_status = "unavailable"

        base_bm25_ids = (
            ()
            if requirement_probe_policy is DISABLED_REQUIREMENT_PROBE_POLICY
            else bm25_ids
        )
        requirement_probe_ids: tuple[str, ...] = ()
        requirement_probe_queries = 0
        if requirement_probe_policy is DISABLED_REQUIREMENT_PROBE_POLICY:
            requirement_probe_status: RequirementProbeStatus = "disabled"
        elif bm25_status not in {"ok", "empty"}:
            requirement_probe_status = "unavailable"
        else:
            incumbent = set(bm25_ids) | set(dense_ids)
            capacity = MAX_CANDIDATE_DOCUMENTS - len(incumbent)
            (
                requirement_probe_ids,
                requirement_probe_status,
                requirement_probe_queries,
            ) = self._requirement_probe_supplements(
                requirement_probe_candidates,
                lexical_text,
                incumbent,
                capacity,
            )
            if requirement_probe_status == "ok":
                bm25_ids = (*bm25_ids, *requirement_probe_ids)
                bm25_status = "ok"

        if not bm25_ids and not dense_ids:
            result = RetrievalResult(
                recommendations=tuple(self._catalog_ids[:top_k]),
                trace=RetrievalTrace(
                    bm25_ids=bm25_ids,
                    dense_ids=dense_ids,
                    fused_ids=(),
                    bm25_status=bm25_status,
                    dense_status=dense_status,
                    used_fallback=True,
                ),
            )
            if requirement_probe_policy is DISABLED_REQUIREMENT_PROBE_POLICY:
                return result
            return RequirementProbeRetrievalResult(
                recommendations=result.recommendations,
                trace=result.trace,
                probe_trace=RequirementProbeTrace(
                    base_bm25_ids=base_bm25_ids,
                    supplemental_ids=requirement_probe_ids,
                    status=requirement_probe_status,
                    query_count=requirement_probe_queries,
                ),
            )

        scores: dict[str, float] = {}
        weighted_routes = (
            (bm25_ids, route_weights.bm25),
            (dense_ids, route_weights.dense),
        )
        for route, weight in weighted_routes:
            for rank, parent_asin in enumerate(route, start=1):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + weight / (
                    RRF_K + rank
                )
        ordered = sorted(
            scores,
            key=lambda parent_asin: (
                -scores[parent_asin],
                self._catalog_order[parent_asin],
            ),
        )
        fused_ids = tuple(ordered)
        result = RetrievalResult(
            recommendations=fused_ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=bm25_ids,
                dense_ids=dense_ids,
                fused_ids=fused_ids,
                bm25_status=bm25_status,
                dense_status=dense_status,
                used_fallback=False,
                dense_scores=dense_scores,
            ),
        )
        if requirement_probe_policy is DISABLED_REQUIREMENT_PROBE_POLICY:
            return result
        return RequirementProbeRetrievalResult(
            recommendations=result.recommendations,
            trace=result.trace,
            probe_trace=RequirementProbeTrace(
                base_bm25_ids=base_bm25_ids,
                supplemental_ids=requirement_probe_ids,
                status=requirement_probe_status,
                query_count=requirement_probe_queries,
            ),
        )
