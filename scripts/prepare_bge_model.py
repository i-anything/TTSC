"""Download, quantize, and verify the offline BGE-small model assets.

``prepare_model`` fetches the pinned official ``BAAI/bge-small-en-v1.5``
revision, validates the published metadata and source ONNX checksum,
quantizes the graph to int8, checks quantization fidelity (mean and worst
cosine against the float source), and writes the asset directory with an
atomic manifest recording every file's origin and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse


LOGGER = logging.getLogger(__name__)

MODEL_ID = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
SOURCE_ONNX_PATH = "onnx/model.onnx"
SOURCE_ONNX_SHA256 = "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35"
SOURCE_ONNX_BYTES = 133_093_490
DERIVED_MODEL_FILE = "model_int8.onnx"
ATTRIBUTION_FILE = "BGE_MODEL_ATTRIBUTION.md"
MANIFEST_FILE = "model_manifest.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
FIDELITY_MEAN_COSINE_MIN = 0.995
FIDELITY_WORST_COSINE_MIN = 0.980

EXPECTED_BUILD_VERSIONS = {
    "numpy": "2.2.6",
    "onnx": "1.19.1",
    "onnxruntime": "1.23.2",
    "tokenizers": "0.22.1",
}


@dataclass(frozen=True)
class OfficialAsset:
    source_path: str
    file: str


OFFICIAL_ASSETS = (
    OfficialAsset("config.json", "config.json"),
    OfficialAsset("config_sentence_transformers.json", "config_sentence_transformers.json"),
    OfficialAsset("modules.json", "modules.json"),
    OfficialAsset("sentence_bert_config.json", "sentence_bert_config.json"),
    OfficialAsset("special_tokens_map.json", "special_tokens_map.json"),
    OfficialAsset("tokenizer.json", "tokenizer.json"),
    OfficialAsset("tokenizer_config.json", "tokenizer_config.json"),
    OfficialAsset("vocab.txt", "vocab.txt"),
    OfficialAsset("1_Pooling/config.json", "pooling_config.json"),
    OfficialAsset("README.md", "MODEL_CARD.md"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _official_url(source_path: str) -> str:
    encoded_path = quote(source_path, safe="/")
    return (
        f"https://huggingface.co/{MODEL_ID}/resolve/"
        f"{MODEL_REVISION}/{encoded_path}?download=true"
    )


def _download(source_path: str, destination: Path, timeout: float) -> dict[str, object]:
    url = _official_url(source_path)
    LOGGER.info("downloading %s", source_path)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TTSC-model-vendor/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise RuntimeError(f"refusing non-HTTPS redirect for {source_path}")
            with destination.open("xb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
                handle.flush()
                os.fsync(handle.fileno())
    except Exception as error:
        raise RuntimeError(f"failed to download pinned asset {source_path}: {error}") from error
    return {
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
        "origin": "official_upstream",
        "source_path": source_path,
        "source_url": url,
    }


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON asset {path.name}: {error}") from error


def _validate_official_metadata(root: Path) -> None:
    config = _load_json(root / "config.json")
    if not isinstance(config, dict):
        raise RuntimeError("config.json must contain an object")
    expected_config = {
        "model_type": "bert",
        "hidden_size": 384,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30_522,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise RuntimeError(f"unexpected config.json {key}: {config.get(key)!r}")

    tokenizer = _load_json(root / "tokenizer_config.json")
    if not isinstance(tokenizer, dict):
        raise RuntimeError("tokenizer_config.json must contain an object")
    if (
        tokenizer.get("tokenizer_class") != "BertTokenizer"
        or tokenizer.get("do_lower_case") is not True
        or tokenizer.get("model_max_length") != 512
    ):
        raise RuntimeError("tokenizer configuration does not match pinned BGE contract")

    sentence_config = _load_json(root / "sentence_bert_config.json")
    if sentence_config != {"max_seq_length": 512, "do_lower_case": True}:
        raise RuntimeError("sentence-transformer sequence configuration changed upstream")

    pooling = _load_json(root / "pooling_config.json")
    expected_pooling = {
        "word_embedding_dimension": 384,
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }
    if pooling != expected_pooling:
        raise RuntimeError("pooling configuration does not match CLS-only BGE contract")

    modules = _load_json(root / "modules.json")
    if not isinstance(modules, list):
        raise RuntimeError("modules.json must contain a list")
    module_types = [item.get("type") for item in modules if isinstance(item, dict)]
    if module_types != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]:
        raise RuntimeError("sentence-transformer module pipeline changed upstream")


def _build_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    problems: list[str] = []
    for distribution, expected in EXPECTED_BUILD_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{distribution} is not installed")
            continue
        installed[distribution] = actual
        if actual != expected:
            problems.append(f"{distribution}=={actual}, expected {expected}")
    if problems:
        raise RuntimeError(
            "install the exact build environment from requirements-preprocessing.txt: "
            + "; ".join(problems)
        )
    return installed


def _quantize(source: Path, destination: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise RuntimeError(
            "install requirements-preprocessing.txt before preparing the model"
        ) from error

    LOGGER.info("quantizing pinned FP32 ONNX to dynamic QInt8")
    quantize_dynamic(
        model_input=source,
        model_output=destination,
        op_types_to_quantize=None,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=None,
        nodes_to_exclude=None,
        use_external_data_format=False,
        extra_options=None,
    )
    if not destination.is_file():
        raise RuntimeError("ONNX Runtime did not produce the quantized model")
    _fsync_file(destination)


def _validate_cpu_graph(
    model_path: Path,
    tokenizer_path: Path,
    reference_model_path: Path,
) -> dict[str, object]:
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError(
            "install requirements-preprocessing.txt before preparing the model"
        ) from error

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime does not provide CPUExecutionProvider")
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"derived model enabled unexpected providers: {session.get_providers()}")

    inputs = tuple(item.name for item in session.get_inputs())
    if not {"input_ids", "attention_mask"}.issubset(inputs) or not set(inputs).issubset(
        {"input_ids", "attention_mask", "token_type_ids"}
    ):
        raise RuntimeError(f"unsupported derived model inputs: {inputs}")
    outputs = tuple(item.name for item in session.get_outputs())
    if "last_hidden_state" not in outputs:
        raise RuntimeError(f"derived model lacks last_hidden_state output: {outputs}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoding = tokenizer.encode("offline model validation", add_special_tokens=True)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    if cls_id is None or sep_id is None or encoding.ids[:1] != [cls_id] or encoding.ids[-1:] != [sep_id]:
        raise RuntimeError("tokenizer does not add the expected BERT special tokens")
    arrays = {
        "input_ids": np.asarray([encoding.ids], dtype=np.int64),
        "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.asarray([encoding.type_ids], dtype=np.int64),
    }
    result = session.run(
        ["last_hidden_state"],
        {name: arrays[name] for name in inputs},
    )[0]
    expected_shape = (1, len(encoding.ids), 384)
    if result.shape != expected_shape or result.dtype != np.float32:
        raise RuntimeError(
            f"unexpected smoke-test output: shape={result.shape}, dtype={result.dtype}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("derived model smoke test returned non-finite values")

    # Quantization is only accepted when the normalized CLS vectors remain close
    # to the pinned FP32 graph.  The fixed cases cover document/query prefixes,
    # terse attributes, prose, numbers, and right truncation at 512 tokens.
    fidelity_texts = [
        "Wireless noise cancelling headphones for a long flight",
        "Title: Red cotton summer dress\nCategory: Clothing > Women > Dresses\n"
        "Attributes: material: cotton | color: red | size: medium",
        "Waterproof leather hiking boots, men's size 10, wide fit",
        "A compact stainless steel insulated bottle that keeps drinks cold for 24 hours.",
        "$19.99 black USB-C charger with two ports and 65 watt fast charging",
        QUERY_PREFIX + "comfortable office shoes with arch support",
        QUERY_PREFIX + "gift for a seven year old who likes dinosaurs",
        QUERY_PREFIX + "lightweight camera for travel under 500 dollars",
        " ".join(["breathable running shirt"] * 220),
    ]
    tokenizer.enable_truncation(max_length=512, direction="right")
    tokenizer.enable_padding(
        direction="right",
        pad_id=tokenizer.token_to_id("[PAD]"),
        pad_type_id=0,
        pad_token="[PAD]",
    )
    encodings = tokenizer.encode_batch(fidelity_texts, add_special_tokens=True)
    fidelity_arrays = {
        "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
        "attention_mask": np.asarray(
            [item.attention_mask for item in encodings], dtype=np.int64
        ),
        "token_type_ids": np.asarray(
            [item.type_ids for item in encodings], dtype=np.int64
        ),
    }

    reference_session = ort.InferenceSession(
        str(reference_model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    def normalized_cls(active_session: object) -> np.ndarray:
        active_inputs = tuple(item.name for item in active_session.get_inputs())
        hidden = active_session.run(
            ["last_hidden_state"],
            {name: fidelity_arrays[name] for name in active_inputs},
        )[0]
        pooled = np.asarray(hidden[:, 0, :], dtype=np.float32)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        if not np.isfinite(pooled).all() or np.any(norms <= 1e-12):
            raise RuntimeError("fidelity validation returned invalid embeddings")
        return pooled / norms

    reference_vectors = normalized_cls(reference_session)
    derived_vectors = normalized_cls(session)
    cosines = np.einsum("ij,ij->i", reference_vectors, derived_vectors)
    mean_cosine = float(np.mean(cosines))
    worst_cosine = float(np.min(cosines))
    if mean_cosine < FIDELITY_MEAN_COSINE_MIN or worst_cosine < FIDELITY_WORST_COSINE_MIN:
        raise RuntimeError(
            "derived model failed FP32 fidelity gate: "
            f"mean_cosine={mean_cosine:.8f}, worst_cosine={worst_cosine:.8f}"
        )
    return {
        "providers": list(session.get_providers()),
        "inputs": list(inputs),
        "outputs": list(outputs),
        "smoke_output_shape": list(result.shape),
        "smoke_output_dtype": str(result.dtype),
        "fidelity": {
            "sample_count": len(fidelity_texts),
            "mean_cosine": mean_cosine,
            "worst_cosine": worst_cosine,
            "required_mean_cosine": FIDELITY_MEAN_COSINE_MIN,
            "required_worst_cosine": FIDELITY_WORST_COSINE_MIN,
        },
    }


def _inventory_record(path: Path, origin: str, **extra: object) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "origin": origin,
        **extra,
    }


def _write_manifest_atomic(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def prepare_model(output_dir: str | Path, *, timeout: float = 120.0) -> dict[str, object]:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    versions = _build_versions()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    source_model = staging / ".source_model.onnx"
    try:
        file_records: list[dict[str, object]] = []
        for asset in OFFICIAL_ASSETS:
            destination = staging / asset.file
            record = _download(asset.source_path, destination, timeout)
            file_records.append(record)

        source_record = _download(SOURCE_ONNX_PATH, source_model, timeout)
        if source_record["sha256"] != SOURCE_ONNX_SHA256:
            raise RuntimeError(
                "official FP32 ONNX checksum mismatch: "
                f"{source_record['sha256']} != {SOURCE_ONNX_SHA256}"
            )
        if source_record["bytes"] != SOURCE_ONNX_BYTES:
            raise RuntimeError(
                "official FP32 ONNX size mismatch: "
                f"{source_record['bytes']} != {SOURCE_ONNX_BYTES}"
            )
        _validate_official_metadata(staging)

        derived_model = staging / DERIVED_MODEL_FILE
        _quantize(source_model, derived_model)
        validation = _validate_cpu_graph(
            derived_model,
            staging / "tokenizer.json",
            source_model,
        )
        source_model.unlink()

        attribution_source = Path(__file__).resolve().parents[1] / ATTRIBUTION_FILE
        if not attribution_source.is_file():
            raise RuntimeError(f"missing repository attribution file: {attribution_source}")
        attribution_destination = staging / ATTRIBUTION_FILE
        shutil.copyfile(attribution_source, attribution_destination)
        _fsync_file(attribution_destination)

        derived_record = _inventory_record(
            derived_model,
            "derived_dynamic_quantization",
            source_path=SOURCE_ONNX_PATH,
            source_sha256=SOURCE_ONNX_SHA256,
        )
        attribution_record = _inventory_record(
            attribution_destination,
            "repository_attribution",
        )
        file_records.extend((derived_record, attribution_record))
        file_records.sort(key=lambda item: str(item["file"]))

        expected_files = {str(item["file"]) for item in file_records}
        actual_files = {path.name for path in staging.iterdir() if path.is_file()}
        if actual_files != expected_files:
            raise RuntimeError(
                "staged asset inventory mismatch: "
                f"actual={sorted(actual_files)}, expected={sorted(expected_files)}"
            )

        tokenizer_record = next(
            item for item in file_records if item["file"] == "tokenizer.json"
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "file": DERIVED_MODEL_FILE,
                "sha256": derived_record["sha256"],
                "source_file": SOURCE_ONNX_PATH,
                "source_sha256": SOURCE_ONNX_SHA256,
                "source_bytes": SOURCE_ONNX_BYTES,
                "source_url": _official_url(SOURCE_ONNX_PATH),
                "dimension": 384,
                "max_sequence_length": 512,
                "pooling": "cls",
                "normalization": "l2_float32",
                "document_prefix": "",
                "query_prefix": QUERY_PREFIX,
                "license": "MIT",
            },
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": tokenizer_record["sha256"],
                "type": "BertTokenizer",
                "lowercase": True,
                "vocab_size": 30_522,
            },
            "quantization": {
                "api": "onnxruntime.quantization.quantize_dynamic",
                "activation_type": "QUInt8",
                "weight_type": "QInt8",
                "op_types_to_quantize": None,
                "per_channel": True,
                "reduce_range": False,
                "nodes_to_quantize": None,
                "nodes_to_exclude": None,
                "use_external_data_format": False,
                "extra_options": None,
            },
            "build_environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "packages": versions,
            },
            "validation": validation,
            "files": file_records,
        }
        _write_manifest_atomic(staging / MANIFEST_FILE, manifest)

        final_files = {path.name for path in staging.iterdir() if path.is_file()}
        if final_files != expected_files | {MANIFEST_FILE}:
            raise RuntimeError(f"final staged inventory mismatch: {sorted(final_files)}")
        _fsync_directory(staging)
        if os.path.lexists(output):
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        staging.rename(output)
        _fsync_directory(output.parent)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vendor pinned BAAI BGE-small assets and derive a CPU INT8 ONNX model"
    )
    parser.add_argument(
        "--output",
        default="assets/bge-small-en-v1.5-int8",
        help="new artifact directory; existing paths are never overwritten",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="per-request network timeout in seconds",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = prepare_model(args.output, timeout=args.timeout)
    print(
        json.dumps(
            {
                "output": str(Path(args.output)),
                "model_sha256": manifest["model"]["sha256"],
                "revision": MODEL_REVISION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
