from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from conversational_search.orchestration import (
    EXACT_RANKING_CACHE_CAPABILITY,
    BackendSnapshotToken,
)
from conversational_search.ranking import CandidateDocument
from conversational_search.strategy import RouteWeights


ROUTE_LIMIT = 100
RRF_K = 60
MAX_CANDIDATE_DOCUMENTS = 200
DEFAULT_ROUTE_WEIGHTS = RouteWeights(bm25=0.5, dense=0.5)

RouteStatus = Literal["unavailable", "ok", "empty", "error", "skipped"]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Bounded route output from one hybrid search, without query or label data."""

    bm25_ids: tuple[str, ...]
    dense_ids: tuple[str, ...]
    fused_ids: tuple[str, ...]
    bm25_status: RouteStatus
    dense_status: RouteStatus
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    recommendations: tuple[str, ...]
    trace: RetrievalTrace


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
    ) -> None:
        self.encoder = encoder
        self.dense_index = dense_index
        self.dense_available = encoder is not None and dense_index is not None
        self._connection = sqlite3.connect(":memory:")
        self.bm25_available = True
        self.bm25_initialization_error: str | None = None
        try:
            self._create_bm25_table()
        except sqlite3.OperationalError as error:
            self.bm25_available = False
            self.bm25_initialization_error = f"{type(error).__name__}: {error}"
        self._catalog_ids = self._build_bm25(Path(catalog_path))
        self._valid_ids = frozenset(self._catalog_ids)
        self._catalog_order = {
            parent_asin: row_index
            for row_index, parent_asin in enumerate(self._catalog_ids)
        }
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

    def _create_bm25_table(self) -> None:
        self._connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, "
            "description, tokenize='unicode61 remove_diacritics 2')"
        )

    def _build_bm25(self, catalog_path: Path) -> tuple[str, ...]:
        cursor = self._connection.cursor()
        catalog_ids: list[str] = []
        seen_ids: set[str] = set()
        batch: list[tuple[int, str, str, str, str, str, str, str]] = []
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
                if self.bm25_available:
                    batch.append(
                        (
                            len(catalog_ids),
                            parent_asin,
                            _text(product.get("title")),
                            _text(product.get("categories")),
                            _text(product.get("features")),
                            _text(product.get("details")),
                            _text(product.get("store")),
                            _text(product.get("description")),
                        )
                    )
                if self.bm25_available and len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products("
                        "rowid, parent_asin, title, categories, features, details, "
                        "store, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()
        if self.bm25_available and batch:
            cursor.executemany(
                "INSERT INTO products("
                "rowid, parent_asin, title, categories, features, details, store, "
                "description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        self._connection.commit()
        if not catalog_ids:
            raise ValueError(f"catalog is empty: {catalog_path}")
        return tuple(catalog_ids)

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
        if not self.dense_available:
            return []
        encoded = self.encoder.encode_queries([dense_query_text], batch_size=1)
        hits = self.dense_index.search(encoded[0], top_k=ROUTE_LIMIT)
        return self._sanitize_route(hits)

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
        route_weights: RouteWeights | None = None,
    ) -> list[str]:
        result = self.search_with_trace(
            dense_query_text,
            lexical_text,
            top_k=top_k,
            route_weights=route_weights,
        )
        return list(result.recommendations)

    def search_with_trace(
        self,
        dense_query_text: str,
        lexical_text: str,
        top_k: int = 10,
        *,
        route_weights: RouteWeights | None = None,
    ) -> RetrievalResult:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if route_weights is None:
            route_weights = DEFAULT_ROUTE_WEIGHTS
        elif not isinstance(route_weights, RouteWeights):
            raise TypeError("route_weights must be RouteWeights")
        if top_k <= 0:
            return RetrievalResult(
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

        if self.dense_available:
            try:
                dense_ids = tuple(self._dense(dense_query_text))
            except Exception:
                dense_ids = ()
                dense_status: RouteStatus = "error"
            else:
                dense_status = "ok" if dense_ids else "empty"
        else:
            dense_ids = ()
            dense_status = "unavailable"

        if not bm25_ids and not dense_ids:
            return RetrievalResult(
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
        return RetrievalResult(
            recommendations=fused_ids[:top_k],
            trace=RetrievalTrace(
                bm25_ids=bm25_ids,
                dense_ids=dense_ids,
                fused_ids=fused_ids,
                bm25_status=bm25_status,
                dense_status=dense_status,
                used_fallback=False,
            ),
        )
