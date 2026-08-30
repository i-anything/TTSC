from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import struct
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from preprocessing.catalog import (
    ASIN_RE,
    TEXT_TEMPLATE_VERSION,
    CatalogError,
    NormalizedProduct,
    canonical_product_text,
    iter_normalized_products,
    scan_catalog,
)
from preprocessing.encoder import Embedder, EncoderMetadata, sha256_file


LOGGER = logging.getLogger(__name__)
ARTIFACT_SCHEMA_VERSION = 2
DEFAULT_SHARD_COUNT = 4


def _canonical_digest_update(digest: object, document: str) -> None:
    encoded = document.encode("utf-8")
    digest.update(struct.pack(">Q", len(encoded)))
    digest.update(encoded)


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _safe_file(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise CatalogError(f"invalid artifact filename: {name!r}")
    path = root / name
    if not path.is_file():
        raise CatalogError(f"missing artifact file: {path}")
    return path


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def shard_ranges(row_count: int, shard_count: int) -> tuple[tuple[int, int], ...]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if shard_count <= 0 or shard_count > row_count:
        raise ValueError("shard_count must be between 1 and row_count")
    return tuple(
        (
            index * row_count // shard_count,
            (index + 1) * row_count // shard_count,
        )
        for index in range(shard_count)
    )


class ShardedEmbeddingWriter:
    def __init__(
        self,
        root: Path,
        row_count: int,
        dimension: int,
        shard_count: int,
    ) -> None:
        self.ranges = shard_ranges(row_count, shard_count)
        self.paths: list[Path] = []
        self._arrays: list[np.memmap] = []
        self.next_row = 0
        for index, (start, end) in enumerate(self.ranges):
            path = root / f"product_embeddings-{index:03d}-of-{shard_count:03d}.npy"
            self.paths.append(path)
            self._arrays.append(
                np.lib.format.open_memmap(
                    path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(end - start, dimension),
                )
            )

    def write(self, global_start: int, values: np.ndarray) -> None:
        if global_start != self.next_row:
            raise CatalogError(
                f"non-contiguous embedding write: {global_start} != {self.next_row}"
            )
        source_start = 0
        global_end = global_start + len(values)
        for shard_index, (shard_start, shard_end) in enumerate(self.ranges):
            write_start = max(global_start, shard_start)
            write_end = min(global_end, shard_end)
            if write_start >= write_end:
                continue
            count = write_end - write_start
            local_start = write_start - shard_start
            self._arrays[shard_index][local_start : local_start + count] = values[
                source_start : source_start + count
            ]
            source_start += count
        if source_start != len(values):
            raise CatalogError("embedding write fell outside configured shard ranges")
        self.next_row = global_end

    def close(self) -> None:
        for array in self._arrays:
            array.flush()
        self._arrays.clear()
        for path in self.paths:
            _fsync_file(path)


def _validate_vectors(vectors: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    if vectors.shape != (rows, dimension):
        raise CatalogError(
            f"embedder returned shape {vectors.shape}; expected {(rows, dimension)}"
        )
    values = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(values).all():
        raise CatalogError("embedder returned non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise CatalogError("embedder returned a zero-length vector")
    values /= norms
    return values


def _verify_matrix(
    path: Path,
    rows: int,
    dimension: int,
    chunk_size: int = 4096,
) -> float:
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (rows, dimension) or matrix.dtype != np.float32:
        raise CatalogError(
            f"invalid embedding matrix shape/dtype: {matrix.shape}, {matrix.dtype}"
        )
    maximum_norm_error = 0.0
    for start in range(0, rows, chunk_size):
        values = np.asarray(matrix[start : start + chunk_size], dtype=np.float32)
        if not np.isfinite(values).all():
            raise CatalogError(f"non-finite embedding values near row {start}")
        norms = np.linalg.norm(values, axis=1)
        maximum_norm_error = max(
            maximum_norm_error,
            float(np.max(np.abs(norms - 1.0))),
        )
    if maximum_norm_error > 1e-4:
        raise CatalogError(f"embedding norm error exceeds tolerance: {maximum_norm_error}")
    return maximum_norm_error


def _logical_embedding_sha256(paths: Sequence[Path], chunk_size: int = 4096) -> str:
    digest = hashlib.sha256()
    for path in paths:
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(matrix), chunk_size):
            values = np.asarray(matrix[start : start + chunk_size], dtype="<f4")
            digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _build_identity(
    *,
    catalog_sha256: str,
    canonical_text_sha256: str,
    metadata: EncoderMetadata,
    shard_count: int,
    allow_truncation: bool,
    ids_sha256: str,
    logical_embedding_sha256: str,
) -> str:
    payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "catalog_sha256": catalog_sha256,
        "canonical_text_sha256": canonical_text_sha256,
        "text_template_version": TEXT_TEMPLATE_VERSION,
        "model": asdict(metadata),
        "dtype": "float32",
        "shard_count": shard_count,
        "allow_truncation": allow_truncation,
        "ids_sha256": ids_sha256,
        "logical_embedding_sha256": logical_embedding_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_embedding_artifacts(
    catalog_path: str | Path,
    output_dir: str | Path,
    embedder: Embedder,
    *,
    expected_rows: int | None = 50_000,
    batch_size: int = 32,
    chunk_size: int = 512,
    shard_count: int = DEFAULT_SHARD_COUNT,
    allow_truncation: bool = True,
    compute_threads: int | None = None,
) -> dict:
    if batch_size <= 0 or chunk_size <= 0 or chunk_size < batch_size:
        raise ValueError("batch_size and chunk_size must be positive; chunk_size >= batch_size")
    if compute_threads is not None and compute_threads <= 0:
        raise ValueError("compute_threads must be positive when provided")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("validating and hashing %s", catalog_path)
    scan = scan_catalog(catalog_path, expected_rows=expected_rows)
    metadata = embedder.metadata
    if metadata.dimension <= 0 or metadata.max_sequence_length <= 0:
        raise ValueError("embedder metadata has invalid dimension or sequence length")
    ranges = shard_ranges(scan.row_count, shard_count)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    ids_path = staging / "product_ids.npy"
    start_time = time.perf_counter()
    canonical_hash = hashlib.sha256()
    maximum_tokens = 0
    truncated_documents = 0

    try:
        writer = ShardedEmbeddingWriter(
            staging,
            scan.row_count,
            metadata.dimension,
            shard_count,
        )
        ids = np.lib.format.open_memmap(
            ids_path,
            mode="w+",
            dtype="S10",
            shape=(scan.row_count,),
        )

        chunk_products: list[NormalizedProduct] = []
        completed = 0
        for product in iter_normalized_products(catalog_path):
            expected_id = scan.product_ids[product.row_index]
            if product.parent_asin != expected_id:
                raise CatalogError(
                    f"catalog changed during build at row {product.row_index + 1}: "
                    f"{expected_id} != {product.parent_asin}"
                )
            chunk_products.append(product)
            if len(chunk_products) < chunk_size:
                continue
            completed, maximum_tokens, truncated_documents = _write_chunk(
                chunk_products,
                completed,
                writer,
                ids,
                embedder,
                batch_size,
                allow_truncation,
                canonical_hash,
                maximum_tokens,
                truncated_documents,
            )
            chunk_products.clear()
            LOGGER.info("embedded %d/%d products", completed, scan.row_count)

        if chunk_products:
            completed, maximum_tokens, truncated_documents = _write_chunk(
                chunk_products,
                completed,
                writer,
                ids,
                embedder,
                batch_size,
                allow_truncation,
                canonical_hash,
                maximum_tokens,
                truncated_documents,
            )

        if completed != scan.row_count or writer.next_row != scan.row_count:
            raise CatalogError(f"embedded {completed} rows; expected {scan.row_count}")
        if canonical_hash.hexdigest() != scan.canonical_text_sha256:
            raise CatalogError("canonical text changed between validation and embedding passes")
        if sha256_file(catalog_path) != scan.catalog_sha256:
            raise CatalogError("catalog changed during preprocessing")

        writer.close()
        ids.flush()
        del ids
        _fsync_file(ids_path)

        shard_specs: list[dict] = []
        for index, ((row_start, row_end), path) in enumerate(zip(ranges, writer.paths)):
            maximum_norm_error = _verify_matrix(
                path,
                row_end - row_start,
                metadata.dimension,
            )
            shard_specs.append(
                {
                    "index": index,
                    "row_start": row_start,
                    "row_end": row_end,
                    "file": path.name,
                    "shape": [row_end - row_start, metadata.dimension],
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "max_norm_error": maximum_norm_error,
                }
            )

        loaded_ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        if loaded_ids.shape != (scan.row_count,) or loaded_ids.dtype != np.dtype("S10"):
            raise CatalogError(f"invalid product ID array: {loaded_ids.shape}, {loaded_ids.dtype}")
        for start in range(0, scan.row_count, 4096):
            expected = np.asarray(
                [value.encode("ascii") for value in scan.product_ids[start : start + 4096]],
                dtype="S10",
            )
            if not np.array_equal(loaded_ids[start : start + len(expected)], expected):
                raise CatalogError(f"product ID alignment failed near row {start}")
        del loaded_ids

        ids_sha = sha256_file(ids_path)
        logical_sha = _logical_embedding_sha256(writer.paths)
        elapsed = time.perf_counter() - start_time
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "build_id": _build_identity(
                catalog_sha256=scan.catalog_sha256,
                canonical_text_sha256=scan.canonical_text_sha256,
                metadata=metadata,
                shard_count=shard_count,
                allow_truncation=allow_truncation,
                ids_sha256=ids_sha,
                logical_embedding_sha256=logical_sha,
            ),
            "catalog": {
                "bytes": scan.byte_count,
                "rows": scan.row_count,
                "sha256": scan.catalog_sha256,
                "warnings": scan.warning_counts,
            },
            "text": {
                "template_version": TEXT_TEMPLATE_VERSION,
                "canonical_sha256": scan.canonical_text_sha256,
                "mean_characters": round(scan.mean_document_characters, 3),
                "max_characters": scan.max_document_characters,
                "unicode_normalization": "NFC",
                "persisted": False,
            },
            "model": asdict(metadata),
            "embeddings": {
                "shape": [scan.row_count, metadata.dimension],
                "dtype": "float32",
                "l2_normalized": True,
                "shard_count": shard_count,
                "logical_data_sha256": logical_sha,
                "shards": shard_specs,
            },
            "ids": {
                "file": ids_path.name,
                "shape": [scan.row_count],
                "dtype": "S10",
                "bytes": ids_path.stat().st_size,
                "sha256": ids_sha,
                "alignment": "zero-based catalog row",
            },
            "build": {
                "allow_truncation": allow_truncation,
                "batch_size": batch_size,
                "chunk_size": chunk_size,
                "compute_threads": compute_threads,
                "duration_seconds": round(elapsed, 3),
                "documents_per_second": round(scan.row_count / elapsed, 3),
                "maximum_tokens": maximum_tokens,
                "truncated_documents": truncated_documents,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "libraries": {
                    "numpy": np.__version__,
                    "onnxruntime": _installed_version("onnxruntime"),
                    "tokenizers": _installed_version("tokenizers"),
                },
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _fsync_file(manifest_path)
        ready_path = staging / "READY"
        ready_path.write_text(sha256_file(manifest_path) + "\n", encoding="ascii")
        _fsync_file(ready_path)
        _fsync_directory(staging)
        if output.exists():
            raise FileExistsError(f"artifact output appeared during build: {output}")
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_chunk(
    products: Sequence[NormalizedProduct],
    start: int,
    writer: ShardedEmbeddingWriter,
    ids: np.memmap,
    embedder: Embedder,
    batch_size: int,
    allow_truncation: bool,
    canonical_hash: object,
    maximum_tokens: int,
    truncated_documents: int,
) -> tuple[int, int, int]:
    documents = [canonical_product_text(product) for product in products]
    for document in documents:
        _canonical_digest_update(canonical_hash, document)

    token_lengths = [int(value) for value in embedder.token_lengths(documents)]
    if len(token_lengths) != len(documents):
        raise CatalogError("embedder returned the wrong number of token lengths")
    if token_lengths:
        maximum_tokens = max(maximum_tokens, max(token_lengths))
    over_limit = [
        index
        for index, length in enumerate(token_lengths)
        if length > embedder.metadata.max_sequence_length
    ]
    truncated_documents += len(over_limit)
    if over_limit and not allow_truncation:
        product = products[over_limit[0]]
        raise CatalogError(
            f"row {product.row_index + 1} ({product.parent_asin}) needs "
            f"{token_lengths[over_limit[0]]} tokens but model limit is "
            f"{embedder.metadata.max_sequence_length}"
        )

    values = _validate_vectors(
        embedder.encode(documents, batch_size),
        len(documents),
        embedder.metadata.dimension,
    )
    end = start + len(documents)
    writer.write(start, values)
    ids[start:end] = np.asarray(
        [product.parent_asin.encode("ascii") for product in products],
        dtype="S10",
    )
    return end, maximum_tokens, truncated_documents


def verify_embedding_artifacts(path: str | Path, *, check_hashes: bool = True) -> dict:
    root = Path(path)
    if not (root / "READY").is_file():
        raise CatalogError(f"artifact directory is not finalized: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise CatalogError("artifact manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise CatalogError("unsupported embedding artifact schema")
    ready_hash = (root / "READY").read_text(encoding="ascii").strip()
    if ready_hash != sha256_file(manifest_path):
        raise CatalogError("manifest checksum does not match READY marker")

    embeddings = manifest.get("embeddings")
    ids_spec = manifest.get("ids")
    if not isinstance(embeddings, dict) or not isinstance(ids_spec, dict):
        raise CatalogError("artifact manifest is missing embeddings or IDs")
    shape = embeddings.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
    ):
        raise CatalogError("invalid logical embedding shape")
    rows, dimension = shape
    shard_count = embeddings.get("shard_count")
    shard_specs = embeddings.get("shards")
    if not isinstance(shard_count, int) or not isinstance(shard_specs, list):
        raise CatalogError("invalid embedding shard metadata")
    if len(shard_specs) != shard_count:
        raise CatalogError("embedding shard count mismatch")
    expected_ranges = shard_ranges(rows, shard_count)
    shard_paths: list[Path] = []
    for index, (spec, expected_range) in enumerate(zip(shard_specs, expected_ranges)):
        if not isinstance(spec, dict) or spec.get("index") != index:
            raise CatalogError(f"invalid embedding shard index {index}")
        row_start, row_end = expected_range
        if (spec.get("row_start"), spec.get("row_end")) != expected_range:
            raise CatalogError(f"embedding shard range mismatch at index {index}")
        if spec.get("shape") != [row_end - row_start, dimension]:
            raise CatalogError(f"embedding shard shape mismatch at index {index}")
        shard_path = _safe_file(root, spec.get("file"))
        if shard_path.stat().st_size != spec.get("bytes"):
            raise CatalogError(f"embedding shard size mismatch at index {index}")
        if check_hashes and sha256_file(shard_path) != spec.get("sha256"):
            raise CatalogError(f"embedding shard checksum mismatch at index {index}")
        _verify_matrix(shard_path, row_end - row_start, dimension)
        shard_paths.append(shard_path)

    if check_hashes:
        logical_hash = _logical_embedding_sha256(shard_paths)
        if logical_hash != embeddings.get("logical_data_sha256"):
            raise CatalogError("logical embedding checksum mismatch")

    ids_path = _safe_file(root, ids_spec.get("file"))
    if ids_path.stat().st_size != ids_spec.get("bytes"):
        raise CatalogError("product ID artifact size mismatch")
    if check_hashes and sha256_file(ids_path) != ids_spec.get("sha256"):
        raise CatalogError("product ID artifact checksum mismatch")
    ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
    if ids.shape != (rows,) or ids.dtype != np.dtype("S10"):
        raise CatalogError("product ID artifact shape or dtype mismatch")
    decoded = [value.decode("ascii") for value in ids]
    if len(set(decoded)) != rows or any(ASIN_RE.fullmatch(value) is None for value in decoded):
        raise CatalogError("product ID artifact contains invalid or duplicate ASINs")
    return manifest
