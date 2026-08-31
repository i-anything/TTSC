from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Byte-level identities from the organizer's participant-kit release.
# README.md and starter/agent.py are intentionally excluded because teams are
# expected to replace them with their own documentation and agent.
FROZEN_PARTICIPANT_KIT_FILES = {
    "DATA_ATTRIBUTION.md": (
        "23e9f3077536c7e2456fa4df95cd11a8be29e371dd4d102a3afe21ffa3b5db0f"
    ),
    "data/README.md": (
        "a5667489e4b25d473d468ff7a98cfd7ebf301cadb99ad4cb5cd777c3af901049"
    ),
    "data/public_set.jsonl": (
        "857259f7a438e6188ac63e18995b6ff4489bfcfc4a716a798b9a2aa0ee8f7579"
    ),
    "docs/agent_api_contract.json": (
        "635563741dd71c273d540722913eccdc595b4af9b47ade79f38cd42ae45c8822"
    ),
    "docs/baseline_results.json": (
        "70834f738918191e4108f89ea66bc502db521bccce5ed95d6ccc73563fba5a96"
    ),
    "docs/competition_specification.md": (
        "408e264acbd1e4567b98038d448fd23c9e9b51705149ca7da1bcd1571e10d001"
    ),
    "docs/evaluation_config.json": (
        "8ee0c899ddc68d521754cf9d2f239a8bc09851fb37c5872567160c30d431aa53"
    ),
    "docs/submission_rules.md": (
        "2d312636ceda4576f95e45d5f56f8ac9e106023d3028081a5d59dd274f385b16"
    ),
    "evaluator/__init__.py": (
        "c597e982409b24fe5411298cfe033aeb287eafcf26c33e34b8c43294cff0a917"
    ),
    "evaluator/local_evaluator.py": (
        "79a5ea06f9a1b8c5036f30efa85dc1f36b8f6b06eb8feb8f545dfa767bc45564"
    ),
    "starter/__init__.py": (
        "03c49004df458e7fb767d172cc896fb5dd08a2aa00686d322248befdc2d7f5d4"
    ),
}

OPTIONAL_CATALOG_FILES = {
    "data/catalog.jsonl": (
        "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"
    ),
    "data/catalog.jsonl.gz": (
        "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OfficialArtifactIntegrityTest(unittest.TestCase):
    def test_frozen_participant_kit_files_are_byte_identical(self) -> None:
        for relative_path, expected_sha256 in FROZEN_PARTICIPANT_KIT_FILES.items():
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                self.assertTrue(path.is_file(), f"missing official file: {relative_path}")
                self.assertEqual(file_sha256(path), expected_sha256)

    def test_downloaded_catalog_files_match_official_release(self) -> None:
        for relative_path, expected_sha256 in OPTIONAL_CATALOG_FILES.items():
            with self.subTest(path=relative_path):
                path = REPOSITORY_ROOT / relative_path
                if path.exists():
                    self.assertTrue(path.is_file())
                    self.assertEqual(file_sha256(path), expected_sha256)


if __name__ == "__main__":
    unittest.main()
