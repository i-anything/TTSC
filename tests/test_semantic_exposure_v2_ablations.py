from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from conversational_search.exposure_policy import (
    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
    DISABLED_EVIDENCE_EXPOSURE_POLICY,
)
from conversational_search.retrieval import (
    DISABLED_SEMANTIC_LEXICAL_RESCUE_POLICY,
    SHARED_DENSE_TERMS_RESCUE_POLICY,
    SemanticLexicalRescueStatus,
)
from scripts.run_semantic_exposure_v2_ablations import (
    ARM_CONFIGS,
    ARM_ORDER,
    ActivationRun,
    _fixed_architecture_contract,
    _load_contract,
    _technical_faults_are_zero,
    _validate_hashes,
    validate_publication,
)


def _diagnostics() -> dict:
    semantic = {
        status.value: 0 for status in SemanticLexicalRescueStatus
    }
    semantic["validation_or_execution_fallbacks"] = 0
    return {
        "route_health": {"fallback_turns": 0},
        "ranking_health": {"failures": 0, "unavailable_skips": 0},
        "exact_evidence_health": {
            "capability_unavailable": 0,
            "evidence_errors": 0,
            "validation_errors": 0,
        },
        "semantic_lexical_rescue_health": semantic,
        "evidence_exposure_health": {
            "retrieval_fail_open": 7,
            "evidence_fail_open": 3,
            "validation_fallbacks": 0,
        },
        "slate_health": {"failures": 0},
        "orchestration_health": {
            "fault_invalidations": 0,
            "store_rejections": 0,
        },
        "response_audit": {
            "response_exceptions": 0,
            "invalid_api_responses": 0,
        },
        "runtime_network_attempts": 0,
    }


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


class SemanticExposureV2AblationTests(unittest.TestCase):
    def test_arm_order_and_only_two_varying_axes_are_frozen(self) -> None:
        self.assertEqual(
            ARM_ORDER,
            ("baseline", "rescue_only", "buying_gate_only", "combined_v2"),
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
                    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
                ),
                (
                    SHARED_DENSE_TERMS_RESCUE_POLICY,
                    BUYING_ONLY_TOP3_STRUCTURAL_EXPOSURE_POLICY,
                ),
            ],
        )

    def test_contract_matches_runtime_and_has_no_promotion_authority(self) -> None:
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

    def test_safe_policy_fail_open_is_not_a_technical_fault(self) -> None:
        diagnostics = _diagnostics()
        run = ActivationRun(diagnostics, "digest", (), {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })

        self.assertTrue(_technical_faults_are_zero(run))

        diagnostics["evidence_exposure_health"]["validation_fallbacks"] = 1
        self.assertFalse(_technical_faults_are_zero(run))

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
