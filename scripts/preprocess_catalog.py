from __future__ import annotations

import argparse
import json
import logging

from preprocessing.catalog import TEXT_TEMPLATE_VERSION, scan_catalog


DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_MODEL_ASSETS = "assets/bge-small-en-v1.5-int8"
DEFAULT_OUTPUT = "assets/search-index-bge-small-en-v1.5-v2"


def _scan_payload(path: str, expected_rows: int | None) -> dict:
    result = scan_catalog(path, expected_rows=expected_rows)
    return {
        "path": str(result.path),
        "rows": result.row_count,
        "bytes": result.byte_count,
        "catalog_sha256": result.catalog_sha256,
        "canonical_text_sha256": result.canonical_text_sha256,
        "text_template_version": TEXT_TEMPLATE_VERSION,
        "warnings": result.warning_counts,
        "mean_document_characters": round(result.mean_document_characters, 3),
        "max_document_characters": result.max_document_characters,
        "max_line_bytes": result.max_line_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic BGE-small catalog search artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser(
        "scan",
        help="validate and hash the catalog without ML dependencies",
    )
    scan.add_argument("--catalog", default=DEFAULT_CATALOG)
    scan.add_argument("--expected-rows", type=int, default=50_000)

    build = subparsers.add_parser(
        "build",
        help="encode the catalog into four row-aligned float32 shards",
    )
    build.add_argument("--catalog", default=DEFAULT_CATALOG)
    build.add_argument("--model-assets", default=DEFAULT_MODEL_ASSETS)
    build.add_argument("--output", default=DEFAULT_OUTPUT)
    build.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="products encoded sequentially per batch (default: 4 for low thermal load)",
    )
    build.add_argument("--chunk-size", type=int, default=512)
    build.add_argument(
        "--threads",
        type=int,
        default=1,
        help="ONNX CPU compute threads (default: 1 for low thermal load)",
    )
    build.add_argument("--expected-rows", type=int, default=50_000)
    build.add_argument(
        "--fail-on-truncation",
        action="store_true",
        help="diagnostic mode: reject documents longer than BGE's 512-token limit",
    )

    verify = subparsers.add_parser("verify", help="verify an existing artifact bundle")
    verify.add_argument("artifact_dir")
    verify.add_argument("--skip-checksums", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "scan":
        payload = _scan_payload(args.catalog, args.expected_rows)
    elif args.command == "verify":
        from preprocessing.embeddings import verify_embedding_artifacts

        payload = verify_embedding_artifacts(
            args.artifact_dir,
            check_hashes=not args.skip_checksums,
        )
    else:
        from preprocessing.embeddings import build_embedding_artifacts
        from preprocessing.encoder import OnnxBgeEncoder

        encoder = OnnxBgeEncoder(args.model_assets, threads=args.threads)
        payload = build_embedding_artifacts(
            args.catalog,
            args.output,
            encoder,
            expected_rows=args.expected_rows,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            shard_count=4,
            allow_truncation=not args.fail_on_truncation,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
