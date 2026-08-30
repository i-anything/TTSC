from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
except ImportError:
    np = None
    onnx = None

from scripts import prepare_arctic_model as arctic


@unittest.skipIf(np is None or onnx is None, "ONNX preprocessing extras unavailable")
class ArcticModelPackagingTest(unittest.TestCase):
    def test_external_weight_split_is_lossless_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            destination = root / "model_int8.onnx"
            expected = {
                "first": np.arange(64, dtype=np.float32),
                "second": np.arange(64, 128, dtype=np.float32),
            }
            graph = helper.make_graph(
                nodes=[],
                name="external-data-test",
                inputs=[],
                outputs=[],
                initializer=[
                    numpy_helper.from_array(value, name=name)
                    for name, value in expected.items()
                ],
            )
            onnx.save_model(helper.make_model(graph), source)

            with (
                mock.patch.object(arctic, "MIN_EXTERNAL_TENSOR_BYTES", 1),
                mock.patch.object(arctic, "MAX_WEIGHT_PART_BYTES", 300),
            ):
                records = arctic._split_external_weights(source, destination)

            self.assertEqual(len(records), 2)
            self.assertTrue(all(record["bytes"] <= 300 for record in records))
            self.assertTrue(
                all((root / str(record["file"])).is_file() for record in records)
            )
            loaded = onnx.load(destination, load_external_data=True)
            observed = {
                tensor.name: numpy_helper.to_array(tensor)
                for tensor in loaded.graph.initializer
            }
            self.assertEqual(set(observed), set(expected))
            for name, value in expected.items():
                np.testing.assert_array_equal(observed[name], value)

    def test_pinned_source_contract_matches_downloaded_model(self) -> None:
        self.assertEqual(arctic.MODEL_ID, "Snowflake/snowflake-arctic-embed-m-v1.5")
        self.assertEqual(len(arctic.MODEL_REVISION), 40)
        self.assertEqual(arctic.SOURCE_ONNX_BYTES, 110_145_162)
        self.assertEqual(len(arctic.SOURCE_ONNX_SHA256), 64)
        self.assertLess(arctic.MAX_WEIGHT_PART_BYTES, arctic.GITHUB_FILE_LIMIT_BYTES)

    def test_single_initializer_larger_than_part_cap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.onnx"
            destination = root / "model_int8.onnx"
            graph = helper.make_graph(
                nodes=[],
                name="oversized-external-data-test",
                inputs=[],
                outputs=[],
                initializer=[
                    numpy_helper.from_array(
                        np.arange(64, dtype=np.float32),
                        name="too-large",
                    )
                ],
            )
            onnx.save_model(helper.make_model(graph), source)

            with (
                mock.patch.object(arctic, "MIN_EXTERNAL_TENSOR_BYTES", 1),
                mock.patch.object(arctic, "MAX_WEIGHT_PART_BYTES", 128),
            ):
                with self.assertRaisesRegex(RuntimeError, "initializer exceeds"):
                    arctic._split_external_weights(source, destination)


if __name__ == "__main__":
    unittest.main()
