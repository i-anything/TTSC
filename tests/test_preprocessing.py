from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # The catalog-only tests still run before optional dependencies are installed.
    np = None

from preprocessing.catalog import (
    CatalogError,
    canonical_product_text,
    normalize_product,
    scan_catalog,
)


def product(**overrides: object) -> dict:
    value = {
        "parent_asin": "B000000001",
        "title": "Example Shoe",
        "features": ["Breathable", "Rubber sole"],
        "description": ["Comfortable for work."],
        "price": 19.99,
        "categories": ["Clothing, Shoes & Jewelry", "Shoes", "Work Shoes"],
        "details": {
            "Department": "Womens",
            "Material": "Leather",
            "Best Sellers Rank": {"Work Shoes": "42"},
        },
        "average_rating": 4.5,
        "rating_number": 12,
        "store": "Example Brand",
    }
    value.update(overrides)
    return value


class CatalogNormalizationTest(unittest.TestCase):
    def test_canonical_text_is_clean_prioritized_and_deterministic(self) -> None:
        raw = product(
            title="  Example\u00a0Shoe  ",
            features=["<p>Breathable &amp; light</p>", "Breathable & light"],
            description=["Cafe\u0301 style", "Cafe\u0301 style"],
            details={
                "Date First Available": "January 1, 2020",
                "Material": "Leather",
                "Best Sellers Rank": {"Work Shoes": "42"},
            },
        )

        normalized = normalize_product(raw, row_index=0)
        text = canonical_product_text(normalized)

        self.assertIn("Title: Example Shoe", text)
        self.assertIn("Category: Shoes Work Shoes", text)
        self.assertIn("Category Path: Clothing, Shoes & Jewelry > Shoes > Work Shoes", text)
        self.assertIn("Brand: Example Brand", text)
        self.assertIn("Detected Material: leather", text)
        self.assertIn("Material: Leather", text)
        self.assertIn("Features: Breathable & light", text)
        self.assertIn("Description: Café style", text)
        self.assertIn("Price: $19.99", text)
        self.assertNotIn("Best Sellers Rank", text)
        self.assertEqual(text.count("Features: Breathable & light"), 1)
        self.assertEqual(text, canonical_product_text(normalized))

    def test_price_variants_and_empty_title_are_supported(self) -> None:
        cases = [
            (None, None, "missing"),
            ("—", None, "missing"),
            ("from 12.99", 1299, "from"),
            (12.99, 1299, "exact"),
            (0, 0, "exact"),
        ]
        for price, expected_cents, expected_kind in cases:
            with self.subTest(price=price):
                normalized = normalize_product(product(title="", price=price), row_index=0)
                self.assertEqual(normalized.price_min_cents, expected_cents)
                self.assertEqual(normalized.price_kind, expected_kind)
                self.assertTrue(canonical_product_text(normalized))

    def test_evaluator_ordered_clues_keep_detected_and_explicit_attributes(self) -> None:
        normalized = normalize_product(
            product(
                title="Red Work Shirt",
                features=["77% Polyester", "Breathable uniform"],
                details={
                    "Material": "Spandex",
                    "Color": "1-Wax-Red-2",
                    "Manufacturer": "Unrelated Factory",
                },
                store="Actual Brand",
            ),
            row_index=0,
        )
        text = canonical_product_text(normalized)

        self.assertIn("Search Clues: polyester | color: red", text)
        self.assertIn("Detected Material: polyester", text)
        self.assertIn("Detected Color: red", text)
        self.assertIn("Material: Spandex", text)
        self.assertIn("Color: 1-Wax-Red-2", text)
        self.assertIn("Brand: Actual Brand", text)
        self.assertNotIn("Unrelated Factory", text)

    def test_operational_details_are_not_dense_text(self) -> None:
        normalized = normalize_product(
            product(
                features=[],
                details={
                    "Item model number": "ABC-123",
                    "Date First Available": "January 1, 2020",
                    "Department": "Women",
                },
            ),
            row_index=0,
        )
        text = canonical_product_text(normalized)

        self.assertIn("Search Clues: Item model number: ABC-123", text)
        self.assertIn("Department: Women", text)
        self.assertNotIn("Details: Item model number", text)

    def test_unexpected_fields_are_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogError, "unexpected fields"):
            normalize_product(product(extra="not-in-frozen-schema"), row_index=0)

    def test_invalid_product_types_are_rejected(self) -> None:
        with self.assertRaisesRegex(CatalogError, "features"):
            normalize_product(product(features="not-a-list"), row_index=0)
        with self.assertRaisesRegex(CatalogError, "parent_asin"):
            normalize_product(product(parent_asin="bad"), row_index=0)
        with self.assertRaisesRegex(CatalogError, "rating_number"):
            normalize_product(product(rating_number=-1), row_index=0)


class CatalogScanTest(unittest.TestCase):
    def test_scan_reports_stable_hashes_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            rows = [product(), product(parent_asin="B000000002", title="Second")]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            first = scan_catalog(path, expected_rows=2)
            second = scan_catalog(path, expected_rows=2)

            self.assertEqual(first.row_count, 2)
            self.assertEqual(first.product_ids, ("B000000001", "B000000002"))
            self.assertEqual(first.catalog_sha256, second.catalog_sha256)
            self.assertEqual(first.canonical_text_sha256, second.canonical_text_sha256)

    def test_duplicate_ids_and_malformed_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                json.dumps(product()) + "\n" + json.dumps(product(title="Duplicate")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CatalogError, "duplicate parent_asin"):
                scan_catalog(duplicate)

            malformed = root / "malformed.jsonl"
            malformed.write_text("{not-json}\n", encoding="utf-8")
            with self.assertRaisesRegex(CatalogError, "invalid JSON"):
                scan_catalog(malformed)


@unittest.skipIf(np is None, "NumPy is not installed")
class EmbeddingArtifactTest(unittest.TestCase):
    class FakeEmbedder:
        def __init__(self, invalid_shape: bool = False) -> None:
            from preprocessing.embeddings import EncoderMetadata

            self.invalid_shape = invalid_shape
            self.metadata = EncoderMetadata(
                backend="fake",
                model_id="fake/test-model",
                revision="a" * 40,
                model_file="fake.bin",
                model_sha256="b" * 64,
                source_model_sha256="c" * 64,
                asset_manifest_sha256="d" * 64,
                tokenizer_sha256="e" * 64,
                dimension=4,
                max_sequence_length=1024,
                pooling="mean",
                normalization="l2_float32",
                document_prefix="",
                query_prefix="query: ",
                license="test-only",
                provider="cpu",
                compute_dtype="float32",
            )

        def token_lengths(self, texts: list[str]) -> list[int]:
            return [len(text.split()) + 2 for text in texts]

        def encode(self, texts: list[str], batch_size: int) -> object:
            rows = []
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                rows.append([float(value + 1) for value in digest[:4]])
            result = np.asarray(rows, dtype=np.float32)
            return result[:, :3] if self.invalid_shape else result

    def test_build_is_aligned_atomic_and_contains_no_text_copy(self) -> None:
        from preprocessing.embeddings import (
            build_embedding_artifacts,
            verify_embedding_artifacts,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            rows = [
                product(),
                product(parent_asin="B000000002", title="Second"),
                product(parent_asin="B000000003", title="Third"),
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "artifacts"

            manifest = build_embedding_artifacts(
                catalog,
                output,
                self.FakeEmbedder(),
                expected_rows=3,
                batch_size=1,
                chunk_size=2,
                shard_count=2,
            )

            self.assertTrue((output / "READY").is_file())
            self.assertEqual(manifest["embeddings"]["shape"], [3, 4])
            self.assertFalse(manifest["text"]["persisted"])
            self.assertFalse(any("text" in item.name for item in output.iterdir()))
            ids = np.load(output / "product_ids.npy", allow_pickle=False)
            self.assertEqual(ids.tolist(), [b"B000000001", b"B000000002", b"B000000003"])
            vectors = np.vstack(
                [
                    np.load(output / shard["file"], allow_pickle=False)
                    for shard in manifest["embeddings"]["shards"]
                ]
            )
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
            self.assertEqual(verify_embedding_artifacts(output)["build_id"], manifest["build_id"])

    def test_shards_cover_rows_and_chunk_can_cross_boundaries(self) -> None:
        from preprocessing.embeddings import build_embedding_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            rows = [
                product(parent_asin=f"B{index:09d}", title=f"Product {index}")
                for index in range(8)
            ]
            catalog.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "artifacts"
            manifest = build_embedding_artifacts(
                catalog,
                output,
                self.FakeEmbedder(),
                expected_rows=8,
                batch_size=1,
                chunk_size=3,
                shard_count=4,
            )

            ranges = [
                (item["row_start"], item["row_end"])
                for item in manifest["embeddings"]["shards"]
            ]
            self.assertEqual(ranges, [(0, 2), (2, 4), (4, 6), (6, 8)])
            self.assertEqual(
                np.load(output / "product_ids.npy", allow_pickle=False).tolist(),
                [f"B{index:09d}".encode("ascii") for index in range(8)],
            )

    def test_failed_build_leaves_no_final_directory(self) -> None:
        from preprocessing.embeddings import build_embedding_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            catalog.write_text(json.dumps(product()) + "\n", encoding="utf-8")
            output = root / "artifacts"
            with self.assertRaisesRegex(CatalogError, "shape"):
                build_embedding_artifacts(
                    catalog,
                    output,
                    self.FakeEmbedder(invalid_shape=True),
                    expected_rows=1,
                    batch_size=1,
                    chunk_size=1,
                    shard_count=1,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
