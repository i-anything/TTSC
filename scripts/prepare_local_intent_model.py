"""Download and verify the optional local intent-parser GGUF model."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "assets"
    / "qwen3-1.7b-intent"
    / "Qwen3-1.7B-Q4_K_M.gguf"
)
MODEL_URL = (
    "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/"
    "daeb8e2d528a760970442092f6bf1e55c3b659eb/"
    "Qwen3-1.7B-Q4_K_M.gguf?download=true"
)
MODEL_SHA256 = "d2387ca2dbfee2ffabce7120d3770dadca0b293052bc2f0e138fdc940d9bc7b5"
CHUNK_SIZE = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download(output: Path) -> None:
    if output.is_file():
        actual = sha256(output)
        if actual == MODEL_SHA256:
            print(f"Verified existing model: {output}")
            return
        raise ValueError(
            f"existing model checksum mismatch: {actual} != {MODEL_SHA256}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with partial.open("wb") as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
        actual = sha256(partial)
        if actual != MODEL_SHA256:
            raise ValueError(
                f"downloaded model checksum mismatch: {actual} != {MODEL_SHA256}"
            )
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    print(f"Downloaded and verified model: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the optional grounded Qwen intent parser."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination GGUF path",
    )
    args = parser.parse_args()
    download(args.output.resolve())


if __name__ == "__main__":
    main()
