from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from conversational_search.exposure_policy import (
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
    TOP3_STRUCTURAL_EXPOSURE_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    SHARED_DENSE_TERMS_RESCUE_POLICY,
)
from scripts.run_semantic_exposure_ablations import (
    ARM_CONFIGS,
    ARM_ORDER,
    _fixed_architecture_contract,
    _load_contract,
    _validate_locked_paths,
    validate_publication,
)


def _safe_publication() -> dict:
    return {
        "privacy": {
            "aggregate_only": True,
            "labels_used_only_after_agent_replay": True,
            "runtime_received_evaluation_labels": False,
            "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
        },
        "contract_promotion_authority": {
            "automatic_promotion_allowed": False,
            "starter_may_be_changed_by_this_run": False,
        },
    }


class SemanticExposureAblationTests(unittest.TestCase):
    def test_four_arms_change_only_the_two_requested_axes(self) -> None:
        self.assertEqual(
            ARM_ORDER,
            ("baseline", "rescue_only", "gate_only", "combined"),
        )
        self.assertEqual(
            [
                (config.semantic_policy, config.exposure_policy)
                for config in ARM_CONFIGS
            ],
            [
                (
                    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
                    DISABLED_EVIDENCE_EXPOSURE_POLICY,
                ),
                (
                    SHARED_DENSE_TERMS_RESCUE_POLICY,
                    DISABLED_EVIDENCE_EXPOSURE_POLICY,
                ),
                (
                    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
                    TOP3_STRUCTURAL_EXPOSURE_POLICY,
                ),
                (
                    SHARED_DENSE_TERMS_RESCUE_POLICY,
                    TOP3_STRUCTURAL_EXPOSURE_POLICY,
                ),
            ],
        )

    def test_frozen_contract_matches_runtime_policy_objects(self) -> None:
        contract = _load_contract()
        self.assertEqual(
            contract["arms"],
            [config.public_contract() for config in ARM_CONFIGS],
        )
        self.assertEqual(
            contract["fixed_shared_architecture"],
            _fixed_architecture_contract(),
        )
        self.assertFalse(
            contract["promotion_authority"]["automatic_promotion_allowed"]
        )
        self.assertFalse(
            contract["promotion_authority"]["starter_may_be_changed_by_this_run"]
        )

    def test_publication_validator_rejects_raw_keys_and_product_ids(self) -> None:
        safe = _safe_publication()
        validate_publication(safe)

        raw = copy.deepcopy(safe)
        raw["sessions"] = []
        with self.assertRaisesRegex(ValueError, "forbidden raw-data key"):
            validate_publication(raw)

        identifier = copy.deepcopy(safe)
        identifier["opaque"] = "B012345678"
        with self.assertRaisesRegex(ValueError, "product identifier"):
            validate_publication(identifier)

    def test_locked_path_validation_detects_post_lock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "component.py"
            source.write_text("before\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            _validate_locked_paths(root, {"component.py": digest})

            source.write_text("after\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "locked path drifted"):
                _validate_locked_paths(root, {"component.py": digest})


if __name__ == "__main__":
    unittest.main()
