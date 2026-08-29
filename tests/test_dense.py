from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ImportError:  # Dense retrieval has an optional preprocessing/runtime dependency.
    np = None


@unittest.skipIf(np is None, "NumPy is not installed")
class ShardedDenseIndexTest(unittest.TestCase):
    def _artifacts(self, root: Path, matrix: np.ndarray) -> Path:
        rows, dimension = matrix.shape
        ids = np.asarray(
            [f"B{index:09d}".encode("ascii") for index in range(rows)],
            dtype="S10",
        )
        np.save(root / "product_ids.npy", ids, allow_pickle=False)

        shards = []
        for index in range(4):
            row_start = index * rows // 4
            row_end = (index + 1) * rows // 4
            file_name = f"product_embeddings-{index:03d}-of-004.npy"
            np.save(root / file_name, matrix[row_start:row_end], allow_pickle=False)
            shard_path = root / file_name
            shards.append(
                {
                    "index": index,
                    "row_start": row_start,
                    "row_end": row_end,
                    "shape": [row_end - row_start, dimension],
                    "file": file_name,
                    "sha256": hashlib.sha256(shard_path.read_bytes()).hexdigest(),
                }
            )

        manifest = {
            "schema_version": 2,
            "embeddings": {
                "shape": [rows, dimension],
                "dtype": "float32",
                "l2_normalized": True,
                "shard_count": 4,
                "shards": shards,
            },
            "ids": {
                "file": "product_ids.npy",
                "shape": [rows],
                "dtype": "S10",
                "sha256": hashlib.sha256(
                    (root / "product_ids.npy").read_bytes()
                ).hexdigest(),
            },
        }
        self._write_manifest(root, manifest)
        return root

    @staticmethod
    def _write_manifest(root: Path, manifest: dict) -> None:
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (root / "READY").write_text(digest + "\n", encoding="ascii")

    def test_search_matches_brute_force_and_uses_memory_maps(self) -> None:
        from starter import dense

        matrix = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.6, 0.8, 0.0],
                [-1.0, 0.0, 0.0],
                [0.5, 0.0, 0.8660254],
                [0.2, 0.0, 0.9797959],
                [0.7, 0.0, 0.7141428],
            ],
            dtype=np.float32,
        )
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        query = np.asarray([3.0, 0.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            index = dense.ShardedDenseIndex(self._artifacts(Path(directory), matrix))
            self.assertIsInstance(index._ids, np.memmap)
            self.assertTrue(all(isinstance(shard.matrix, np.memmap) for shard in index._shards))

            with mock.patch.object(
                dense.np,
                "concatenate",
                side_effect=AssertionError("full shards must not be concatenated"),
            ):
                hits = index.search(query, top_k=5)

            scores = matrix @ (query / np.linalg.norm(query))
            rows = np.arange(matrix.shape[0])
            expected = np.lexsort((rows, -scores))[:5].tolist()
            self.assertEqual([hit.row_index for hit in hits], expected)
            self.assertEqual(
                [hit.parent_asin for hit in hits],
                [f"B{row:09d}" for row in expected],
            )
            np.testing.assert_allclose(
                [hit.score for hit in hits],
                scores[expected],
                atol=1e-7,
            )

    def test_equal_scores_use_global_row_as_tie_break(self) -> None:
        from starter.dense import ShardedDenseIndex

        matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        )
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        with tempfile.TemporaryDirectory() as directory:
            index = ShardedDenseIndex(self._artifacts(Path(directory), matrix))
            hits = index.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=3)
            self.assertEqual([hit.row_index for hit in hits], [0, 2, 4])

    def test_rejects_invalid_ranges_shapes_and_queries(self) -> None:
        from starter.dense import DenseIndexError, ShardedDenseIndex

        matrix = np.eye(8, 2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = self._artifacts(Path(directory), matrix)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["embeddings"]["shards"][1]["row_start"] += 1
            self._write_manifest(root, manifest)
            with self.assertRaisesRegex(DenseIndexError, "gap, overlap"):
                ShardedDenseIndex(root)

        with tempfile.TemporaryDirectory() as directory:
            index = ShardedDenseIndex(self._artifacts(Path(directory), matrix))
            with self.assertRaisesRegex(DenseIndexError, "query vector has shape"):
                index.search(np.ones((1, 2), dtype=np.float32), top_k=1)
            with self.assertRaisesRegex(DenseIndexError, "zero length"):
                index.search(np.zeros(2, dtype=np.float32), top_k=1)
            self.assertEqual(index.search(np.ones(2, dtype=np.float32), top_k=0), [])

    def test_requires_finalized_schema_v2_bundle(self) -> None:
        from starter.dense import DenseIndexError, ShardedDenseIndex

        matrix = np.eye(8, 2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = self._artifacts(Path(directory), matrix)
            (root / "READY").write_text("bad checksum\n", encoding="ascii")
            with self.assertRaisesRegex(DenseIndexError, "READY"):
                ShardedDenseIndex(root)

    def test_rejects_tampered_ids_and_embedding_shards(self) -> None:
        from starter.dense import DenseIndexError, ShardedDenseIndex

        matrix = np.eye(8, 2, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = self._artifacts(Path(directory), matrix)
            ids_path = root / "product_ids.npy"
            ids_path.write_bytes(ids_path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(DenseIndexError, "ids.sha256 checksum mismatch"):
                ShardedDenseIndex(root)

        with tempfile.TemporaryDirectory() as directory:
            root = self._artifacts(Path(directory), matrix)
            shard_path = root / "product_embeddings-002-of-004.npy"
            payload = bytearray(shard_path.read_bytes())
            payload[-1] ^= 1
            shard_path.write_bytes(payload)
            with self.assertRaisesRegex(DenseIndexError, "shard 2 sha256 checksum mismatch"):
                ShardedDenseIndex(root)


if __name__ == "__main__":
    unittest.main()
