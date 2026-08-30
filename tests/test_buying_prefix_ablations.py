from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
)
from scripts.run_buying_prefix_ablations import (
    ARM_CONFIGS,
    ARM_ORDER,
    _arm_contract,
    _fixed_architecture_contract,
    _load_contract,
    _validate_hashes,
    validate_publication,
)


def _publication() -> dict:
    return {
        "privacy": {
            "aggregate_only": True,
            "labels_used_only_after_agent_replay": True,
            "runtime_received_evaluation_labels": False,
            "contains_identifiers_messages_queries_profiles_or_candidate_lists": False,
        },
        "promotion_authority": {
            "automatic_promotion_allowed": False,
            "starter_may_be_changed_by_this_run": False,
        },
    }


class BuyingPrefixAblationTests(unittest.TestCase):
    def test_arm_order_and_single_varying_axis_are_frozen(self) -> None:
        self.assertEqual(
            ARM_ORDER,
            ("baseline", "buying_withhold_v2", "buying_prefix_v3"),
        )
        self.assertEqual(
            [config.semantic_policy for config in ARM_CONFIGS],
            [DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY] * 3,
        )
        self.assertEqual(
            [config.exposure_policy for config in ARM_CONFIGS],
            [
                DISABLED_EVIDENCE_EXPOSURE_POLICY,
                BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
                BUYING_ONLY_TOP3_PREFIX_EXPOSURE_POLICY,
            ],
        )

    def test_contract_matches_runtime_and_has_no_promotion_authority(self) -> None:
        contract = _load_contract()

        self.assertEqual(
            contract["arms"],
            [_arm_contract(config) for config in ARM_CONFIGS],
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

    def test_publication_rejects_raw_data_and_product_ids(self) -> None:
        safe = _publication()
        validate_publication(safe)

        raw = copy.deepcopy(safe)
        raw["sessions"] = []
        with self.assertRaisesRegex(ValueError, "forbidden raw-data key"):
            validate_publication(raw)

        identifier = copy.deepcopy(safe)
        identifier["opaque"] = "B012345678"
        with self.assertRaisesRegex(ValueError, "product identifier"):
            validate_publication(identifier)

    def test_hash_validation_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("before\n", encoding="utf-8")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            _validate_hashes(root, {"source.py": digest})

            source.write_text("after\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "locked path drifted"):
                _validate_hashes(root, {"source.py": digest})


if __name__ == "__main__":
    unittest.main()
