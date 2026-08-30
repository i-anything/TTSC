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
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse


LOGGER = logging.getLogger(__name__)

MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v1.5"
MODEL_REVISION = "97eab2e17fcb7ccb8bb94d6e547898fa1a6a0f47"
SOURCE_ONNX_PATH = "onnx/model_int8.onnx"
SOURCE_ONNX_SHA256 = (
    "a18f437b2466863901a0bdc14904cf93246f5ecce0b656fc773bc2b7b2f84f6e"
)
SOURCE_ONNX_BYTES = 110_145_162
MODEL_FILE = "model_int8.onnx"
WEIGHT_FILE_TEMPLATE = "model_int8.weights-{index:03d}.bin"
MAX_WEIGHT_PART_BYTES = 64 * 1024 * 1024
MIN_EXTERNAL_TENSOR_BYTES = 1024
GITHUB_FILE_LIMIT_BYTES = 100_000_000
ATTRIBUTION_FILE = "ARCTIC_MODEL_ATTRIBUTION.md"
MANIFEST_FILE = "model_manifest.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

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
    bytes: int
    sha256: str


OFFICIAL_ASSETS = (
    OfficialAsset(
        "config.json",
        "config.json",
        738,
        "6cd4a7ca9b3e037db490c6791b98bcb6c63d3209e4cd00d20179ef0093443e4e",
    ),
    OfficialAsset(
        "config_sentence_transformers.json",
        "config_sentence_transformers.json",
        253,
        "705887b2d56911fa93be916a2457bcc38d289740c630e06b3684ba3abf61a071",
    ),
    OfficialAsset(
        "modules.json",
        "modules.json",
        350,
        "21ddb3037a55ebf4549eb5c1dc44c3f28f10b56ec7f1335c92fd90e5b30d88ac",
    ),
    OfficialAsset(
        "sentence_bert_config.json",
        "sentence_bert_config.json",
        54,
        "ed20daeb5b882c3e1f116ce7b3d53e6643d11063aa9baaa31f5ae91770cfad8f",
    ),
    OfficialAsset(
        "special_tokens_map.json",
        "special_tokens_map.json",
        695,
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    OfficialAsset(
        "tokenizer.json",
        "tokenizer.json",
        711_649,
        "91f1def9b9391fdabe028cd3f3fcc4efd34e5d1f08c3bf2de513ebb5911a1854",
    ),
    OfficialAsset(
        "tokenizer_config.json",
        "tokenizer_config.json",
        1_381,
        "0e83e9d7206b3ade43f8f2aeef523cf5d5b4a25b67af21b273de21972c0f58b7",
    ),
    OfficialAsset(
        "vocab.txt",
        "vocab.txt",
        231_508,
        "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
    OfficialAsset(
        "1_Pooling/config.json",
        "pooling_config.json",
        297,
        "4bf279dd9204db673304cab6e7db8448f427412a0d87f04c3e19f1de55dd98cd",
    ),
    OfficialAsset(
        "README.md",
        "MODEL_CARD.md",
        243_150,
        "751e5f790cdd8915ded71c20c2a20f07c525bc08b6b92881390745177ed13cad",
    ),
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


def _download(
    source_path: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    timeout: float,
) -> dict[str, object]:
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
        raise RuntimeError(
            f"failed to download pinned asset {source_path}: {error}"
        ) from error
    observed_bytes = destination.stat().st_size
    observed_sha256 = _sha256(destination)
    if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"pinned asset drifted: {source_path} "
            f"bytes={observed_bytes}, sha256={observed_sha256}"
        )
    return {
        "file": destination.name,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
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
    expected_config = {
        "model_type": "bert",
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30_522,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected_config.items()
    ):
        raise RuntimeError("config.json does not match the pinned Arctic contract")

    tokenizer = _load_json(root / "tokenizer_config.json")
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("tokenizer_class") != "BertTokenizer"
        or tokenizer.get("do_lower_case") is not True
        or tokenizer.get("model_max_length") != 512
        or tokenizer.get("padding_side") != "right"
        or tokenizer.get("truncation_side") != "right"
    ):
        raise RuntimeError("tokenizer configuration drifted")

    if _load_json(root / "sentence_bert_config.json") != {
        "max_seq_length": 512,
        "do_lower_case": False,
    }:
        raise RuntimeError("sentence-transformer sequence contract drifted")

    pooling = _load_json(root / "pooling_config.json")
    if not isinstance(pooling, dict) or pooling != {
        "word_embedding_dimension": 768,
        "pooling_mode_cls_token": True,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": True,
    }:
        raise RuntimeError("pooling configuration drifted")

    sentence_config = _load_json(root / "config_sentence_transformers.json")
    if (
        not isinstance(sentence_config, dict)
        or sentence_config.get("prompts") != {"query": QUERY_PREFIX}
    ):
        raise RuntimeError("query prompt configuration drifted")

    modules = _load_json(root / "modules.json")
    module_types = (
        [item.get("type") for item in modules if isinstance(item, dict)]
        if isinstance(modules, list)
        else []
    )
    if module_types != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]:
        raise RuntimeError("sentence-transformer module pipeline drifted")


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
            "install requirements-preprocessing.txt exactly: " + "; ".join(problems)
        )
    return installed


def _split_external_weights(source: Path, destination: Path) -> list[dict]:
    try:
        import onnx
        from onnx.external_data_helper import set_external_data
    except ImportError as error:
        raise RuntimeError(
            "install requirements-preprocessing.txt before preparing the model"
        ) from error

    model = onnx.load(source, load_external_data=True)
    records: list[dict] = []
    part_index = -1
    part_path: Path | None = None
    part_handle = None
    part_bytes = 0
    try:
        for tensor in model.graph.initializer:
            raw = tensor.raw_data
            if len(raw) < MIN_EXTERNAL_TENSOR_BYTES:
                continue
            if len(raw) > MAX_WEIGHT_PART_BYTES:
                raise RuntimeError(
                    "an ONNX initializer exceeds the configured external-weight "
                    f"part limit: {tensor.name!r} has {len(raw)} bytes"
                )
            if part_handle is None or (
                part_bytes and part_bytes + len(raw) > MAX_WEIGHT_PART_BYTES
            ):
                if part_handle is not None and part_path is not None:
                    part_handle.flush()
                    os.fsync(part_handle.fileno())
                    part_handle.close()
                    records.append(_weight_record(part_path))
                part_index += 1
                part_path = destination.parent / WEIGHT_FILE_TEMPLATE.format(
                    index=part_index
                )
                part_handle = part_path.open("xb")
                part_bytes = 0
            set_external_data(
                tensor,
                location=part_path.name,
                offset=part_bytes,
                length=len(raw),
            )
            part_handle.write(raw)
            part_bytes += len(raw)
            tensor.ClearField("raw_data")
        if part_handle is not None and part_path is not None:
            part_handle.flush()
            os.fsync(part_handle.fileno())
            part_handle.close()
            part_handle = None
            records.append(_weight_record(part_path))
    finally:
        if part_handle is not None:
            part_handle.close()

    if not records or any(
        record["bytes"] > MAX_WEIGHT_PART_BYTES
        or record["bytes"] >= GITHUB_FILE_LIMIT_BYTES
        for record in records
    ):
        raise RuntimeError("external ONNX weights do not satisfy packaged file limits")
    onnx.save_model(model, destination)
    _fsync_file(destination)
    if destination.stat().st_size >= GITHUB_FILE_LIMIT_BYTES:
        raise RuntimeError("externalized ONNX graph exceeds GitHub file limits")
    return records


def _weight_record(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_cpu_graph(
    model_path: Path,
    tokenizer_path: Path,
    source_model_path: Path,
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
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    started = time.perf_counter()
    derived = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    startup_seconds = time.perf_counter() - started
    source = ort.InferenceSession(
        str(source_model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    for session in (derived, source):
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("model enabled a non-CPU execution provider")
        inputs = {item.name for item in session.get_inputs()}
        outputs = {item.name for item in session.get_outputs()}
        if inputs != {"input_ids", "attention_mask"}:
            raise RuntimeError(f"model input contract drifted: {sorted(inputs)}")
        if not {"token_embeddings", "sentence_embedding"}.issubset(outputs):
            raise RuntimeError(f"model output contract drifted: {sorted(outputs)}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_padding(
        direction="right",
        pad_id=0,
        pad_type_id=0,
        pad_token="[PAD]",
    )
    encodings = tokenizer.encode_batch(
        [
            "offline product embedding validation",
            QUERY_PREFIX + "comfortable waterproof work shoes",
        ],
        add_special_tokens=True,
    )
    feeds = {
        "input_ids": np.asarray([item.ids for item in encodings], dtype=np.int64),
        "attention_mask": np.asarray(
            [item.attention_mask for item in encodings], dtype=np.int64
        ),
    }
    source_vectors = source.run(["sentence_embedding"], feeds)[0]
    started = time.perf_counter()
    derived_vectors = derived.run(["sentence_embedding"], feeds)[0]
    inference_seconds = time.perf_counter() - started
    if not np.array_equal(source_vectors, derived_vectors):
        raise RuntimeError("externalized ONNX graph is not bit-exact to its source")
    if derived_vectors.shape != (2, 768) or derived_vectors.dtype != np.float32:
        raise RuntimeError("model returned an invalid sentence embedding")
    norms = np.linalg.norm(derived_vectors, axis=1)
    if not np.isfinite(derived_vectors).all() or np.max(np.abs(norms - 1.0)) > 1e-5:
        raise RuntimeError("model output is not finite and L2-normalized")
    return {
        "providers": list(derived.get_providers()),
        "inputs": [item.name for item in derived.get_inputs()],
        "outputs": [item.name for item in derived.get_outputs()],
        "smoke_output_shape": list(derived_vectors.shape),
        "smoke_output_dtype": str(derived_vectors.dtype),
        "source_externalization_bit_exact": True,
        "maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
        "single_thread_startup_seconds": round(startup_seconds, 6),
        "single_thread_batch2_inference_seconds": round(inference_seconds, 6),
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
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def prepare_model(output_dir: str | Path, *, timeout: float = 120.0) -> dict:
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    output = Path(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    versions = _build_versions()
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    source_model = staging / ".source_model_int8.onnx"
    try:
        file_records: list[dict[str, object]] = []
        for asset in OFFICIAL_ASSETS:
            destination = staging / asset.file
            file_records.append(
                _download(
                    asset.source_path,
                    destination,
                    expected_bytes=asset.bytes,
                    expected_sha256=asset.sha256,
                    timeout=timeout,
                )
            )
        _download(
            SOURCE_ONNX_PATH,
            source_model,
            expected_bytes=SOURCE_ONNX_BYTES,
            expected_sha256=SOURCE_ONNX_SHA256,
            timeout=timeout,
        )
        _validate_official_metadata(staging)

        model_path = staging / MODEL_FILE
        external_records = _split_external_weights(source_model, model_path)
        validation = _validate_cpu_graph(
            model_path,
            staging / "tokenizer.json",
            source_model,
        )
        source_model.unlink()

        attribution_source = Path(__file__).resolve().parents[1] / ATTRIBUTION_FILE
        if not attribution_source.is_file():
            raise RuntimeError(f"missing repository attribution: {attribution_source}")
        attribution_destination = staging / ATTRIBUTION_FILE
        shutil.copyfile(attribution_source, attribution_destination)
        _fsync_file(attribution_destination)

        model_record = _inventory_record(
            model_path,
            "lossless_externalization_of_official_int8_onnx",
            source_path=SOURCE_ONNX_PATH,
            source_sha256=SOURCE_ONNX_SHA256,
        )
        file_records.extend(
            [
                model_record,
                *(
                    {
                        **record,
                        "origin": "lossless_externalized_official_int8_weights",
                    }
                    for record in external_records
                ),
                _inventory_record(
                    attribution_destination,
                    "repository_attribution",
                ),
            ]
        )
        file_records.sort(key=lambda item: str(item["file"]))
        tokenizer_record = next(
            item for item in file_records if item["file"] == "tokenizer.json"
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "file": MODEL_FILE,
                "sha256": model_record["sha256"],
                "source_file": SOURCE_ONNX_PATH,
                "source_sha256": SOURCE_ONNX_SHA256,
                "source_bytes": SOURCE_ONNX_BYTES,
                "source_url": _official_url(SOURCE_ONNX_PATH),
                "dimension": 768,
                "max_sequence_length": 512,
                "pooling": "cls",
                "normalization": "l2_float32",
                "document_prefix": "",
                "query_prefix": QUERY_PREFIX,
                "output_name": "sentence_embedding",
                "license": "Apache-2.0",
                "external_data": external_records,
            },
            "tokenizer": {
                "file": "tokenizer.json",
                "sha256": tokenizer_record["sha256"],
                "type": "BertTokenizer",
                "lowercase": True,
                "vocab_size": 30_522,
            },
            "packaging": {
                "method": "lossless ONNX external data",
                "minimum_external_tensor_bytes": MIN_EXTERNAL_TENSOR_BYTES,
                "maximum_part_bytes": MAX_WEIGHT_PART_BYTES,
                "github_file_limit_bytes": GITHUB_FILE_LIMIT_BYTES,
                "runtime_reassembly": False,
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

        expected_files = {str(record["file"]) for record in file_records}
        actual_files = {path.name for path in staging.iterdir() if path.is_file()}
        if actual_files != expected_files | {MANIFEST_FILE}:
            raise RuntimeError("final Arctic asset inventory is incomplete")
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
        description="Vendor pinned Arctic Embed M v1.5 INT8 assets for CPU inference"
    )
    parser.add_argument(
        "--output",
        default="assets/snowflake-arctic-embed-m-v1.5-int8",
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
                "dimension": 768,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
