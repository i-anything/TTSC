import gzip
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_submission import file_sha256, prepare_catalog


class PrepareSubmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.archive = self.root / "catalog.jsonl.gz"
        self.catalog = self.root / "catalog.jsonl"
        self.catalog.write_bytes(b'{"parent_asin":"example"}\n')
        self.expected = file_sha256(self.catalog)
        self.archive.write_bytes(gzip.compress(self.catalog.read_bytes()))
        self.archive_hash = file_sha256(self.archive)
        self.catalog.unlink()

    def prepare(self, **overrides):
        options = dict(archive_sha256=self.archive_hash, catalog_sha256=self.expected)
        options.update(overrides)
        return prepare_catalog(self.archive, self.catalog, **options)

    def test_preparation_is_verified_idempotent_and_preserves_archive(self) -> None:
        self.assertEqual(self.prepare(), self.catalog)
        self.assertEqual(file_sha256(self.catalog), self.expected)
        modified = self.catalog.stat().st_mtime_ns
        self.prepare()
        self.assertEqual(self.catalog.stat().st_mtime_ns, modified)
        self.assertEqual(file_sha256(self.archive), self.archive_hash)

    def test_corrupt_archive_never_creates_a_catalog(self) -> None:
        self.archive.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "archive checksum mismatch"):
            self.prepare()
        self.assertFalse(self.catalog.exists())

    def test_wrong_expanded_checksum_cleans_up_partial_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "expanded catalog checksum mismatch"):
            self.prepare(catalog_sha256="0" * 64)
        self.assertFalse(self.catalog.exists())
        self.assertEqual(list(self.root.glob(".catalog-*.tmp")), [])

    def test_existing_different_catalog_is_preserved(self) -> None:
        self.catalog.write_bytes(b"my catalog")
        with self.assertRaisesRegex(ValueError, "existing catalog checksum mismatch"):
            self.prepare()
        self.assertEqual(self.catalog.read_bytes(), b"my catalog")
