"""Pinned offline BGE text encoder behind a minimal ``Embedder`` protocol.

``OnnxTextEncoder`` loads the quantized BGE-small ONNX model and tokenizer
from a verified asset directory using only ONNX Runtime's CPU provider,
validating the manifest contract and SHA-256 checksums before use.
``EncoderMetadata`` records the provenance (model id, revision, checksums,
pooling, prefixes) carried into the embedding manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from preprocessing.catalog import CatalogError


BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_DIMENSION = 384
BGE_MAX_SEQUENCE_LENGTH = 512
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_SOURCE_ONNX_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EncoderMetadata:
    backend: str
    model_id: str
    revision: str
    model_file: str
    model_sha256: str
    source_model_sha256: str
    asset_manifest_sha256: str
    tokenizer_sha256: str
    dimension: int
    max_sequence_length: int
    pooling: str
    normalization: str
    document_prefix: str
    query_prefix: str
    license: str
    provider: str
    compute_dtype: str
    asset_identity_sha256: str = ""


class Embedder(Protocol):
    metadata: EncoderMetadata

    def token_lengths(self, texts: Sequence[str]) -> Sequence[int]: ...

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray: ...


@dataclass(frozen=True)
class _ModelContract:
    model_id: str
    revision: str
    source_model_sha256: str
    dimension: int
    max_sequence_length: int
    query_prefix: str
    license: str
    output_name: str
    output_is_sentence_embedding: bool
    requires_external_data: bool


def model_asset_identity_sha256(manifest: object) -> str:
    """Return a host-independent identity for runtime-relevant model assets."""

    if not isinstance(manifest, dict):
        raise CatalogError("model manifest must be an object")
    model = manifest.get("model")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(tokenizer, dict):
        raise CatalogError("model manifest is missing model or tokenizer metadata")
    payload = {
        "schema_version": manifest.get("schema_version"),
        "model": model,
        "tokenizer": tokenizer,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


_MODEL_CONTRACTS = {
    (BGE_MODEL_ID, BGE_MODEL_REVISION): _ModelContract(
        model_id=BGE_MODEL_ID,
        revision=BGE_MODEL_REVISION,
        source_model_sha256=BGE_SOURCE_ONNX_SHA256,
        dimension=BGE_DIMENSION,
        max_sequence_length=BGE_MAX_SEQUENCE_LENGTH,
        query_prefix=BGE_QUERY_PREFIX,
        license="MIT",
        output_name="last_hidden_state",
        output_is_sentence_embedding=False,
        requires_external_data=False,
    ),
}


class OnnxTextEncoder:
    """Pinned offline text encoder using only ONNX Runtime's CPU provider."""

    def __init__(
        self,
        asset_dir: str | Path,
        *,
        threads: int | None = None,
        verify_hashes: bool = True,
    ) -> None:
        # Runtime queries are batch-size one. Disabling the tokenizer's Rayon
        # pool avoids unnecessary threads and fork-safety warnings on macOS.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "install requirements-runtime.txt to use the local ONNX encoder"
            ) from error

        root = Path(asset_dir)
        manifest_path = root / "model_manifest.json"
        if not manifest_path.is_file():
            raise CatalogError(f"missing model manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = self._validate_manifest(manifest)

        model_path = self._asset_path(root, manifest["model"]["file"])
        tokenizer_path = self._asset_path(root, manifest["tokenizer"]["file"])
        if verify_hashes:
            if sha256_file(model_path) != manifest["model"]["sha256"]:
                raise CatalogError("ONNX model checksum mismatch")
            if sha256_file(tokenizer_path) != manifest["tokenizer"]["sha256"]:
                raise CatalogError("tokenizer checksum mismatch")
            self._verify_external_data(root, manifest["model"])

        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        resolved_threads = self._resolve_threads(threads)
        session_options.intra_op_num_threads = resolved_threads
        session_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise CatalogError(
                f"unexpected ONNX providers: {self._session.get_providers()}"
            )

        self._input_names = tuple(item.name for item in self._session.get_inputs())
        required_inputs = {"input_ids", "attention_mask"}
        if not required_inputs.issubset(self._input_names):
            raise CatalogError(f"unsupported ONNX inputs: {self._input_names}")
        if not set(self._input_names).issubset(
            {"input_ids", "attention_mask", "token_type_ids"}
        ):
            raise CatalogError(f"unexpected ONNX inputs: {self._input_names}")

        output_names = {item.name for item in self._session.get_outputs()}
        if contract.output_name not in output_names:
            raise CatalogError(
                f"missing {contract.output_name} output: {sorted(output_names)}"
            )
        self._output_name = contract.output_name
        self._output_is_sentence_embedding = (
            contract.output_is_sentence_embedding
        )

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._pad_id = self._tokenizer.token_to_id("[PAD]")
        if self._pad_id is None:
            raise CatalogError("tokenizer does not define [PAD]")
        self.metadata = EncoderMetadata(
            backend="onnxruntime",
            model_id=manifest["model"]["id"],
            revision=manifest["model"]["revision"],
            model_file=manifest["model"]["file"],
            model_sha256=manifest["model"]["sha256"],
            source_model_sha256=manifest["model"]["source_sha256"],
            asset_manifest_sha256=sha256_file(manifest_path),
            tokenizer_sha256=manifest["tokenizer"]["sha256"],
            dimension=contract.dimension,
            max_sequence_length=contract.max_sequence_length,
            pooling="cls",
            normalization="l2_float32",
            document_prefix="",
            query_prefix=contract.query_prefix,
            license=contract.license,
            provider="CPUExecutionProvider",
            compute_dtype="int8_weights_float32_output",
            asset_identity_sha256=model_asset_identity_sha256(manifest),
        )

    @staticmethod
    def _asset_path(root: Path, name: object) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise CatalogError(f"invalid model asset filename: {name!r}")
        path = root / name
        if not path.is_file():
            raise CatalogError(f"missing model asset: {path}")
        return path

    @staticmethod
    def _resolve_threads(value: int | None) -> int:
        if value is None:
            configured = os.environ.get("TTSC_ONNX_THREADS")
            value = int(configured) if configured else 1
        if value <= 0:
            raise ValueError("ONNX thread count must be positive")
        return value

    @classmethod
    def _validate_manifest(cls, manifest: object) -> _ModelContract:
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise CatalogError("unsupported model manifest schema")
        model = manifest.get("model")
        tokenizer = manifest.get("tokenizer")
        if not isinstance(model, dict) or not isinstance(tokenizer, dict):
            raise CatalogError("model manifest is missing model or tokenizer metadata")
        contract = _MODEL_CONTRACTS.get(
            (model.get("id"), model.get("revision"))
        )
        if contract is None:
            raise CatalogError("model manifest identifies an unsupported encoder")
        expected = {
            "id": contract.model_id,
            "revision": contract.revision,
            "source_sha256": contract.source_model_sha256,
            "dimension": contract.dimension,
            "max_sequence_length": contract.max_sequence_length,
            "pooling": "cls",
            "document_prefix": "",
            "query_prefix": contract.query_prefix,
        }
        for key, value in expected.items():
            if model.get(key) != value:
                raise CatalogError(
                    f"model manifest {key} mismatch: {model.get(key)!r} != {value!r}"
                )
        for section, key in (
            (model, "file"),
            (model, "sha256"),
            (tokenizer, "file"),
            (tokenizer, "sha256"),
        ):
            if not isinstance(section.get(key), str) or not section[key]:
                raise CatalogError(f"model manifest has invalid {key}")
        external_data = model.get("external_data", [])
        if not isinstance(external_data, list) or (
            contract.requires_external_data != bool(external_data)
        ):
            raise CatalogError("model manifest external-data contract mismatch")
        for record in external_data:
            if (
                not isinstance(record, dict)
                or set(record) != {"file", "bytes", "sha256"}
                or not isinstance(record.get("file"), str)
                or Path(record["file"]).name != record["file"]
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)
                or record["bytes"] <= 0
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
            ):
                raise CatalogError("model manifest has invalid external data")
        return contract

    @classmethod
    def _verify_external_data(cls, root: Path, model: dict) -> None:
        for record in model.get("external_data", []):
            path = cls._asset_path(root, record["file"])
            if path.stat().st_size != record["bytes"]:
                raise CatalogError("ONNX external-data size mismatch")
            if sha256_file(path) != record["sha256"]:
                raise CatalogError("ONNX external-data checksum mismatch")

    def token_lengths(self, texts: Sequence[str]) -> Sequence[int]:
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        return [
            len(encoding.ids)
            for encoding in self._tokenizer.encode_batch(
                [self.metadata.document_prefix + text for text in texts],
                add_special_tokens=True,
            )
        ]

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        return self._encode(texts, self.metadata.document_prefix, batch_size)

    def encode_queries(
        self,
        texts: Sequence[str],
        batch_size: int = 1,
    ) -> np.ndarray:
        return self._encode(texts, self.metadata.query_prefix, batch_size)

    def _encode(
        self,
        texts: Sequence[str],
        prefix: str,
        batch_size: int,
    ) -> np.ndarray:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return np.empty((0, self.metadata.dimension), dtype=np.float32)

        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        result = np.empty((len(texts), self.metadata.dimension), dtype=np.float32)
        self._tokenizer.enable_truncation(
            max_length=self.metadata.max_sequence_length,
            direction="right",
        )
        self._tokenizer.enable_padding(
            direction="right",
            pad_id=self._pad_id,
            pad_type_id=0,
            pad_token="[PAD]",
        )

        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            encodings = self._tokenizer.encode_batch(
                [prefix + texts[index] for index in indexes],
                add_special_tokens=True,
            )
            arrays = {
                "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encodings], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encodings], dtype=np.int64
                ),
            }
            feeds = {name: arrays[name] for name in self._input_names}
            output = self._session.run([self._output_name], feeds)[0]
            if self._output_is_sentence_embedding:
                pooled = np.asarray(output, dtype=np.float32)
            else:
                if output.ndim != 3 or output.shape[0] != len(indexes):
                    raise CatalogError(
                        f"unexpected ONNX output shape: {output.shape}"
                    )
                pooled = np.asarray(output[:, 0, :], dtype=np.float32)
            if pooled.shape != (len(indexes), self.metadata.dimension):
                raise CatalogError(f"unexpected pooled embedding shape: {pooled.shape}")
            if not np.isfinite(pooled).all():
                raise CatalogError("ONNX encoder returned non-finite values")
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            if np.any(norms <= 1e-12):
                raise CatalogError("ONNX encoder returned a zero-length vector")
            pooled /= norms
            result[indexes] = pooled
        return result


class OnnxBgeEncoder(OnnxTextEncoder):
    """Backward-compatible name for the manifest-selected ONNX encoder."""
