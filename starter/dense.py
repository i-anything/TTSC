"""Validating loader for the sharded dense-vector artifact bundle.

``ShardedDenseIndex`` verifies the manifest, ``READY`` marker, and per-file
SHA-256 checksums, then memory-maps the contiguous float32 shards and
scores exact cosine similarity for query vectors, breaking ties on the
global row index.  Every contract violation raises ``DenseIndexError``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ARTIFACT_SCHEMA_VERSION = 2
EXPECTED_SHARD_COUNT = 4
ASIN_RE = re.compile(r"[A-Z0-9]{10}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class DenseIndexError(ValueError):
    """Raised when dense artifacts or a query vector violate the index contract."""


@dataclass(frozen=True, slots=True)
class DenseHit:
    row_index: int
    parent_asin: str
    score: float


@dataclass(frozen=True, slots=True)
class _Shard:
    row_start: int
    row_end: int
    matrix: np.memmap


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DenseIndexError(f"{name} must be an integer")
    if positive and value <= 0:
        raise DenseIndexError(f"{name} must be positive")
    return value


def _shape(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise DenseIndexError(f"{name} must contain [rows, dimension]")
    rows = _integer(value[0], f"{name}[0]", positive=True)
    dimension = _integer(value[1], f"{name}[1]", positive=True)
    return rows, dimension


def _artifact_file(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DenseIndexError(f"{name} must be a non-empty file name")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DenseIndexError(f"{name} must stay inside the artifact directory") from error
    if not path.is_file():
        raise DenseIndexError(f"missing artifact file: {path}")
    return path


def _verify_sha256(path: Path, expected: object, name: str) -> None:
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
        raise DenseIndexError(f"{name} must be a lowercase SHA-256 digest")
    if _sha256(path) != expected:
        raise DenseIndexError(f"{name} checksum mismatch: {path.name}")


def _load_memmap(path: Path, name: str) -> np.memmap:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise DenseIndexError(f"cannot load {name}: {path}") from error
    if not isinstance(value, np.memmap):
        raise DenseIndexError(f"{name} must be an uncompressed .npy array")
    return value


def _local_top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return an exact score-descending, row-ascending local top-k."""

    rows = scores.shape[0]
    if top_k >= rows:
        candidates = np.arange(rows, dtype=np.int64)
    else:
        partition = np.argpartition(scores, rows - top_k)[rows - top_k :]
        threshold = scores[partition].min()
        # Include all values tied at the boundary so argpartition cannot choose
        # an arbitrary row among equal scores.
        candidates = np.flatnonzero(scores >= threshold)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:top_k]]


class ShardedDenseIndex:
    """Exact cosine scorer over four contiguous memory-mapped row shards."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        verify_hashes: bool = True,
    ) -> None:
        if not isinstance(verify_hashes, bool):
            raise TypeError("verify_hashes must be a boolean")
        root = Path(artifact_dir).resolve()
        if not root.is_dir():
            raise DenseIndexError(f"artifact directory does not exist: {root}")

        manifest_path = root / "manifest.json"
        ready_path = root / "READY"
        if not manifest_path.is_file() or not ready_path.is_file():
            raise DenseIndexError(f"artifact directory is not finalized: {root}")
        if ready_path.read_text(encoding="ascii").strip() != _sha256(manifest_path):
            raise DenseIndexError("manifest checksum does not match READY marker")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DenseIndexError(f"invalid manifest: {manifest_path}") from error
        if not isinstance(manifest, dict):
            raise DenseIndexError("manifest must be a JSON object")
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise DenseIndexError(
                f"unsupported artifact schema: {manifest.get('schema_version')!r}"
            )

        embeddings = manifest.get("embeddings")
        ids_metadata = manifest.get("ids")
        if not isinstance(embeddings, dict) or not isinstance(ids_metadata, dict):
            raise DenseIndexError("manifest must contain embeddings and ids objects")
        self.row_count, self.dimension = _shape(
            embeddings.get("shape"), "embeddings.shape"
        )
        if embeddings.get("dtype") != "float32":
            raise DenseIndexError("embeddings.dtype must be float32")
        if embeddings.get("l2_normalized") is not True:
            raise DenseIndexError("embeddings must be L2-normalized")
        if embeddings.get("shard_count") != EXPECTED_SHARD_COUNT:
            raise DenseIndexError(
                f"embeddings.shard_count must be {EXPECTED_SHARD_COUNT}"
            )
        shard_metadata = embeddings.get("shards")
        if not isinstance(shard_metadata, list) or len(shard_metadata) != EXPECTED_SHARD_COUNT:
            raise DenseIndexError(f"manifest must contain {EXPECTED_SHARD_COUNT} shards")

        ids_shape = ids_metadata.get("shape")
        if not isinstance(ids_shape, list) or ids_shape != [self.row_count]:
            raise DenseIndexError("ids.shape must equal the global row count")
        try:
            ids_dtype = np.dtype(ids_metadata.get("dtype"))
        except (TypeError, ValueError) as error:
            raise DenseIndexError("invalid ids.dtype") from error
        if ids_dtype != np.dtype("S10"):
            raise DenseIndexError("ids.dtype must be S10")
        ids_path = _artifact_file(root, ids_metadata.get("file"), "ids.file")
        if verify_hashes:
            _verify_sha256(ids_path, ids_metadata.get("sha256"), "ids.sha256")
        ids = _load_memmap(ids_path, "product IDs")
        if ids.shape != (self.row_count,) or ids.dtype != np.dtype("S10"):
            raise DenseIndexError(
                f"invalid product ID array shape/dtype: {ids.shape}, {ids.dtype}"
            )

        shards: list[_Shard] = []
        expected_start = 0
        for position, metadata in enumerate(shard_metadata):
            if not isinstance(metadata, dict):
                raise DenseIndexError(f"embeddings.shards[{position}] must be an object")
            if metadata.get("index") != position:
                raise DenseIndexError("embedding shard indices must be ordered 0 through 3")
            row_start = _integer(
                metadata.get("row_start"), f"shard {position} row_start"
            )
            row_end = _integer(metadata.get("row_end"), f"shard {position} row_end")
            if row_start != expected_start or row_end <= row_start or row_end > self.row_count:
                raise DenseIndexError(
                    f"shard {position} has a gap, overlap, or invalid row range"
                )
            expected_shape = (row_end - row_start, self.dimension)
            if _shape(metadata.get("shape"), f"shard {position} shape") != expected_shape:
                raise DenseIndexError(f"shard {position} shape does not match its row range")
            shard_path = _artifact_file(
                root, metadata.get("file"), f"shard {position} file"
            )
            if verify_hashes:
                _verify_sha256(
                    shard_path,
                    metadata.get("sha256"),
                    f"shard {position} sha256",
                )
            matrix = _load_memmap(shard_path, f"embedding shard {position}")
            if matrix.shape != expected_shape or matrix.dtype != np.float32:
                raise DenseIndexError(
                    f"invalid shard {position} shape/dtype: {matrix.shape}, {matrix.dtype}"
                )
            if not matrix.flags.c_contiguous:
                raise DenseIndexError(f"embedding shard {position} must be C-contiguous")
            shards.append(_Shard(row_start=row_start, row_end=row_end, matrix=matrix))
            expected_start = row_end
        if expected_start != self.row_count:
            raise DenseIndexError("embedding shards do not cover every global row")

        self._ids = ids
        self._shards = tuple(shards)
        self.manifest = manifest

    def search(self, query_vector: object, top_k: int) -> list[DenseHit]:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k <= 0:
            return []

        query = np.array(query_vector, dtype=np.float32, copy=True, order="C")
        if query.shape != (self.dimension,):
            raise DenseIndexError(
                f"query vector has shape {query.shape}; expected {(self.dimension,)}"
            )
        if not np.isfinite(query).all():
            raise DenseIndexError("query vector contains non-finite values")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise DenseIndexError("query vector has zero length")
        query /= norm

        result_count = min(top_k, self.row_count)
        # Heap keys are ordered from worst to best: lower score is worse, and
        # for equal scores the larger global row is worse because -row is lower.
        best: list[tuple[float, int, int]] = []
        for shard in self._shards:
            scores = np.einsum(
                "ij,j->i",
                shard.matrix,
                query,
                dtype=np.float32,
                optimize=False,
            )
            if scores.shape != (shard.row_end - shard.row_start,):
                raise DenseIndexError("dense scorer produced an invalid score shape")
            if not np.isfinite(scores).all():
                raise DenseIndexError(
                    f"embedding shard starting at row {shard.row_start} produced non-finite scores"
                )
            local_count = min(result_count, scores.shape[0])
            for local_row in _local_top_indices(scores, local_count):
                global_row = shard.row_start + int(local_row)
                item = (float(scores[local_row]), -global_row, global_row)
                if len(best) < result_count:
                    heapq.heappush(best, item)
                elif item[:2] > best[0][:2]:
                    heapq.heapreplace(best, item)
            del scores

        ordered = sorted(best, key=lambda item: (-item[0], item[2]))
        hits: list[DenseHit] = []
        for score, _, row_index in ordered:
            try:
                parent_asin = bytes(self._ids[row_index]).decode("ascii")
            except UnicodeDecodeError as error:
                raise DenseIndexError(f"invalid product ID encoding at row {row_index}") from error
            if ASIN_RE.fullmatch(parent_asin) is None:
                raise DenseIndexError(f"invalid product ID at row {row_index}: {parent_asin!r}")
            hits.append(
                DenseHit(
                    row_index=row_index,
                    parent_asin=parent_asin,
                    score=score,
                )
            )
        return hits
