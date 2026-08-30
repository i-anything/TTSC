from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # Keep catalog-only test discovery usable without runtime extras.
    np = None

if np is not None:
    from preprocessing.encoder import (
        BGE_DIMENSION,
        BGE_MAX_SEQUENCE_LENGTH,
        BGE_MODEL_ID,
        BGE_MODEL_REVISION,
        BGE_QUERY_PREFIX,
        BGE_SOURCE_ONNX_SHA256,
        OnnxBgeEncoder,
        OnnxTextEncoder,
        model_asset_identity_sha256,
    )

from preprocessing.catalog import CatalogError


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _Encoding:
    def __init__(self, ids: list[int], attention_mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = [0] * len(ids)


def _fake_runtime(
    state: dict,
    reported_providers: list[str] | None = None,
    *,
    output_name: str = "last_hidden_state",
    dimension: int = BGE_DIMENSION,
) -> dict:
    ort = types.ModuleType("onnxruntime")
    tokenizers = types.ModuleType("tokenizers")

    class SessionOptions:
        def __init__(self) -> None:
            self.execution_mode = None
            self.graph_optimization_level = None
            self.intra_op_num_threads = None
            self.inter_op_num_threads = None

    class Node:
        def __init__(self, name: str) -> None:
            self.name = name

    class InferenceSession:
        def __init__(
            self,
            model_path: str,
            *,
            sess_options: SessionOptions,
            providers: list[str],
        ) -> None:
            state["session_inits"].append(
                {
                    "model_path": model_path,
                    "options": sess_options,
                    "providers": list(providers),
                }
            )
            self._providers = list(reported_providers or providers)

        def get_providers(self) -> list[str]:
            return list(self._providers)

        def get_inputs(self) -> list[Node]:
            return [Node("input_ids"), Node("attention_mask"), Node("token_type_ids")]

        def get_outputs(self) -> list[Node]:
            return [Node(output_name)]

        def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
            state["runs"].append({"output_names": list(output_names), "feeds": feeds})
            batch, sequence = feeds["input_ids"].shape
            if output_name == "sentence_embedding":
                output = np.zeros((batch, dimension), dtype=np.float64)
                output[:, 0] = 3.0
                output[:, 1] = 4.0
            else:
                output = np.zeros((batch, sequence, dimension), dtype=np.float64)
                output[:, 0, 0] = 3.0
                output[:, 0, 1] = 4.0
                if sequence > 1:
                    # Mean pooling would produce a visibly different result.
                    output[:, 1:, 2] = 100.0
            return [output]

    class Tokenizer:
        def __init__(self, path: str) -> None:
            self.path = path
            self.truncation: tuple[int, str] | None = None
            self.padding = False
            self.calls: list[dict] = []
            state["tokenizers"].append(self)

        @classmethod
        def from_file(cls, path: str) -> "Tokenizer":
            return cls(path)

        def token_to_id(self, token: str) -> int | None:
            return 0 if token == "[PAD]" else None

        def no_truncation(self) -> None:
            self.truncation = None
            state["no_truncation_calls"] += 1

        def no_padding(self) -> None:
            self.padding = False
            state["no_padding_calls"] += 1

        def enable_truncation(self, *, max_length: int, direction: str) -> None:
            self.truncation = (max_length, direction)
            state["truncation_calls"].append((max_length, direction))

        def enable_padding(self, **options: object) -> None:
            self.padding = True
            state["padding_calls"].append(dict(options))

        def encode_batch(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[_Encoding]:
            rows: list[list[int]] = []
            for text in texts:
                content = [200 + (ord(character) % 50) for character in text]
                if self.truncation is not None:
                    max_length, direction = self.truncation
                    available = max_length - (2 if add_special_tokens else 0)
                    content = content[:available] if direction == "right" else content[-available:]
                ids = ([101] + content + [102]) if add_special_tokens else content
                rows.append(ids)

            padded_length = max((len(row) for row in rows), default=0) if self.padding else None
            encodings: list[_Encoding] = []
            for row in rows:
                padding = 0 if padded_length is None else padded_length - len(row)
                encodings.append(_Encoding(row + [0] * padding, [1] * len(row) + [0] * padding))
            self.calls.append(
                {
                    "texts": list(texts),
                    "add_special_tokens": add_special_tokens,
                    "lengths": [len(item.ids) for item in encodings],
                }
            )
            return encodings

    ort.SessionOptions = SessionOptions
    ort.InferenceSession = InferenceSession
    ort.ExecutionMode = types.SimpleNamespace(ORT_SEQUENTIAL="sequential")
    ort.GraphOptimizationLevel = types.SimpleNamespace(ORT_ENABLE_ALL="all")
    tokenizers.Tokenizer = Tokenizer
    return {"onnxruntime": ort, "tokenizers": tokenizers}


@unittest.skipIf(np is None, "NumPy is not installed")
class OnnxBgeEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.asset_dir = Path(self.temporary_directory.name)
        self.model_bytes = b"fake-int8-onnx-model"
        self.tokenizer_bytes = b'{"fake": "tokenizer"}'
        (self.asset_dir / "model_int8.onnx").write_bytes(self.model_bytes)
        (self.asset_dir / "tokenizer.json").write_bytes(self.tokenizer_bytes)
        self.manifest = {
            "schema_version": 1,
            "model": {
                "id": BGE_MODEL_ID,
                "revision": BGE_MODEL_REVISION,
                "file": "model_int8.onnx",
                "sha256": _sha256(self.model_bytes),
                "source_sha256": BGE_SOURCE_ONNX_SHA256,
                "dimension": BGE_DIMENSION,
                "max_sequence_length": BGE_MAX_SEQUENCE_LENGTH,
                "pooling": "cls",
                "document_prefix": "",
                "query_prefix": BGE_QUERY_PREFIX,
            },
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": _sha256(self.tokenizer_bytes),
            },
        }
        self._write_manifest()

    def _write_manifest(self) -> None:
        (self.asset_dir / "model_manifest.json").write_text(
            json.dumps(self.manifest),
            encoding="utf-8",
        )

    @staticmethod
    def _state() -> dict:
        return {
            "session_inits": [],
            "runs": [],
            "tokenizers": [],
            "truncation_calls": [],
            "padding_calls": [],
            "no_truncation_calls": 0,
            "no_padding_calls": 0,
        }

    def _encoder(
        self,
        state: dict,
        *,
        reported_providers: list[str] | None = None,
    ) -> OnnxBgeEncoder:
        modules = _fake_runtime(state, reported_providers)
        with mock.patch.dict(sys.modules, modules):
            return OnnxBgeEncoder(self.asset_dir, threads=2)

    def test_manifest_and_asset_hashes_are_validated(self) -> None:
        state = self._state()
        encoder = self._encoder(state)
        self.assertEqual(encoder.metadata.model_id, BGE_MODEL_ID)
        self.assertEqual(encoder.metadata.revision, BGE_MODEL_REVISION)

        (self.asset_dir / "model_int8.onnx").write_bytes(b"corrupt")
        with self.assertRaisesRegex(CatalogError, "ONNX model checksum mismatch"):
            self._encoder(self._state())

        (self.asset_dir / "model_int8.onnx").write_bytes(self.model_bytes)
        (self.asset_dir / "tokenizer.json").write_bytes(b"corrupt")
        with self.assertRaisesRegex(CatalogError, "tokenizer checksum mismatch"):
            self._encoder(self._state())

        (self.asset_dir / "tokenizer.json").write_bytes(self.tokenizer_bytes)
        self.manifest["model"]["query_prefix"] = "query: "
        self._write_manifest()
        with self.assertRaisesRegex(CatalogError, "query_prefix mismatch"):
            self._encoder(self._state())

    def test_document_and_query_prefixes_and_right_truncation_are_exact(self) -> None:
        state = self._state()
        encoder = self._encoder(state)
        tokenizer = state["tokenizers"][0]

        self.assertEqual(encoder.token_lengths(["plain document"]), [16])
        self.assertEqual(tokenizer.calls[-1]["texts"], ["plain document"])
        self.assertEqual(state["no_truncation_calls"], 1)
        self.assertEqual(state["no_padding_calls"], 1)

        encoder.encode(["plain document"], batch_size=1)
        self.assertEqual(tokenizer.calls[-1]["texts"], ["plain document"])

        encoder.encode_queries(["blue work shoe"], batch_size=1)
        self.assertEqual(
            tokenizer.calls[-1]["texts"],
            [BGE_QUERY_PREFIX + "blue work shoe"],
        )

        encoder.encode(["x" * 700], batch_size=1)
        self.assertEqual(state["truncation_calls"][-1], (512, "right"))
        self.assertEqual(tokenizer.calls[-1]["lengths"], [512])
        self.assertTrue(tokenizer.calls[-1]["add_special_tokens"])

    def test_cls_pooling_float32_l2_normalization_and_batch_size_one(self) -> None:
        state = self._state()
        encoder = self._encoder(state)
        vectors = encoder.encode_queries(["shoe"], batch_size=1)

        self.assertEqual(vectors.shape, (1, BGE_DIMENSION))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(vectors[0, :3], [0.6, 0.8, 0.0], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0], atol=1e-7)

        run = state["runs"][-1]
        self.assertEqual(run["output_names"], ["last_hidden_state"])
        self.assertEqual(run["feeds"]["input_ids"].ndim, 2)
        self.assertEqual(run["feeds"]["input_ids"].shape[0], 1)
        for value in run["feeds"].values():
            self.assertEqual(value.dtype, np.int64)

        init = state["session_inits"][0]
        self.assertEqual(init["providers"], ["CPUExecutionProvider"])
        self.assertEqual(init["options"].intra_op_num_threads, 2)
        self.assertEqual(init["options"].inter_op_num_threads, 1)

    def test_non_cpu_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogError, "unexpected ONNX providers"):
            self._encoder(
                self._state(),
                reported_providers=["CPUExecutionProvider", "AzureExecutionProvider"],
            )


if __name__ == "__main__":
    unittest.main()
