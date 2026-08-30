from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from conversational_search.decision import (
    ProtocolObservation,
    recognize_protocol_observation,
)
from evaluator.local_evaluator import evaluate
from scripts import build_phase15_protocol_robustness_suites as suites
from scripts import run_protocol_utility_ablations as ablations


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _product(
    parent_asin: str,
    family: str,
    popularity: int,
) -> dict:
    family_fields = {
        "apparel": ("cotton dress", ["Women's Clothing", "Dresses"]),
        "footwear": ("cotton running shoe", ["Shoes", "Running Shoes"]),
        "jewelry_and_accessories": (
            "cotton necklace accessory",
            ["Jewelry", "Necklaces"],
        ),
    }
    title, categories = family_fields[family]
    return {
        "parent_asin": parent_asin,
        "title": f"{title} {parent_asin}",
        "features": ["color: blue", "packs flat for work"],
        "details": {"Care": "easy clean"},
        "description": "durable everyday option",
        "categories": ["Clothing, Shoes & Jewelry", *categories],
        "store": "Synthetic Store",
        "price": 20.0,
        "rating_number": popularity,
    }


def _independent_target_digest(domain: str, targets: set[str]) -> str:
    members = sorted(
        hashlib.sha256(
            (
                f"{suites.SELECTION_SALT}\0{domain}\0{target}"
            ).encode("utf-8")
        ).digest()
        for target in targets
    )
    return hashlib.sha256(b"".join(members)).hexdigest()


def _independent_case_digest(rows: list[dict]) -> str:
    members = sorted(
        hashlib.sha256(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        for row in rows
    )
    return hashlib.sha256(b"".join(members)).hexdigest()


class Phase15ProtocolRobustnessSuiteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.catalog = cls.root / "catalog.jsonl"
        cls.public = cls.root / "public.jsonl"
        cls.development = cls.root / "development.jsonl"
        cls.validation = cls.root / "validation.jsonl"
        cls.phase14_fresh = cls.root / "phase14-fresh.jsonl"

        rows: list[dict] = []
        popularity_values = {"tail": 1, "torso": 100, "head": 10_000}
        ids: dict[tuple[str, str], list[str]] = {}
        for family in suites.FAMILY_ORDER:
            for popularity_name in suites.POPULARITY_ORDER:
                cell_ids: list[str] = []
                for ordinal in range(45):
                    parent_asin = (
                        f"synthetic-{family}-{popularity_name}-{ordinal:02d}"
                    )
                    cell_ids.append(parent_asin)
                    rows.append(
                        _product(
                            parent_asin,
                            family,
                            popularity_values[popularity_name],
                        )
                    )
                ids[(family, popularity_name)] = cell_ids
        _write_jsonl(cls.catalog, rows)
        cls.catalog_rows = rows

        forbidden_by_source = {
            "public": ids[("apparel", "tail")][0],
            "development": ids[("footwear", "torso")][0],
            "validation": ids[("jewelry_and_accessories", "head")][0],
            "phase14_fresh": ids[("apparel", "head")][0],
        }
        cls.forbidden = set(forbidden_by_source.values())
        cls.forbidden_sources = {
            "public": cls.public,
            "development": cls.development,
            "validation": cls.validation,
            "phase14_fresh": cls.phase14_fresh,
        }
        for name, path in cls.forbidden_sources.items():
            _write_jsonl(
                path,
                [
                    {
                        "sample_id": f"forbidden-{name}",
                        "ground_truth": {
                            "parent_asin": forbidden_by_source[name]
                        },
                    }
                ],
            )

        cls.expected_hashes = {
            "catalog": _sha256(cls.catalog),
            "public": _sha256(cls.public),
            "development": _sha256(cls.development),
            "validation": _sha256(cls.validation),
            "phase14_fresh": _sha256(cls.phase14_fresh),
            "evaluator": _sha256(suites.EVALUATOR_SOURCE),
            "phase14_builder": _sha256(suites.PHASE14_BUILDER_SOURCE),
        }
        cls.first_output = cls.root / "first"
        cls.first_manifest = cls.root / "first-manifest.json"
        cls.second_output = cls.root / "second"
        cls.second_manifest = cls.root / "second-manifest.json"
        cls.first = cls._build(cls.first_output, cls.first_manifest)
        cls.second = cls._build(cls.second_output, cls.second_manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    @classmethod
    def _build(cls, output: Path, manifest: Path) -> dict[str, object]:
        return suites.build(
            catalog=cls.catalog,
            forbidden_sources=cls.forbidden_sources,
            expected_sha256=cls.expected_hashes,
            output_directory=output,
            manifest_path=manifest,
        )

    @classmethod
    def _rows(cls, suite: str) -> list[dict]:
        path = cls.first_output / f"{suite}.jsonl"
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def test_deterministic_disjoint_balanced_and_private(self) -> None:
        self.assertEqual(self.first, self.second)
        self.assertEqual(
            self.first_manifest.read_bytes(),
            self.second_manifest.read_bytes(),
        )
        selected: set[str] = set()
        all_messages: list[str] = []
        expected_total = 0
        for suite in suites.SUITE_ORDER:
            first_path = self.first_output / f"{suite}.jsonl"
            second_path = self.second_output / f"{suite}.jsonl"
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            rows = self._rows(suite)
            cases_per_cell = suites.SUITE_CASES_PER_CELL[suite]
            expected_count = (
                cases_per_cell
                * len(suites.FAMILY_ORDER)
                * len(suites.POPULARITY_ORDER)
            )
            expected_total += expected_count
            self.assertEqual(len(rows), expected_count)
            targets = {row["ground_truth"]["parent_asin"] for row in rows}
            self.assertEqual(len(targets), expected_count)
            self.assertFalse(targets & self.forbidden)
            self.assertFalse(targets & selected)
            selected.update(targets)
            all_messages.extend(
                row["phase15_dialog"]["initial_message"] for row in rows
            )

            family_counts = self.first["family_counts"][suite]
            popularity_counts = self.first["popularity_counts"][suite]
            self.assertEqual(
                set(family_counts.values()),
                {cases_per_cell * len(suites.POPULARITY_ORDER)},
            )
            self.assertEqual(
                set(popularity_counts.values()),
                {cases_per_cell * len(suites.FAMILY_ORDER)},
            )
            metadata = self.first["outputs"][suite]
            self.assertEqual(metadata["sha256"], _sha256(first_path))
            self.assertEqual(
                metadata["case_fingerprint_set_sha256"],
                _independent_case_digest(rows),
            )
            self.assertEqual(
                metadata["target_set_sha256"],
                _independent_target_digest(f"selected:{suite}", targets),
            )

        self.assertEqual(len(selected), expected_total)
        self.assertEqual(
            self.first["selected_target_set_sha256"],
            _independent_target_digest("selected:all", selected),
        )
        self.assertEqual(
            self.first["overlap_proof"],
            {
                "forbidden_overlap": 0,
                "inter_suite_overlap": 0,
                "all_selected_targets_unique": True,
            },
        )
        manifest_text = self.first_manifest.read_text(encoding="utf-8")
        for private_value in (*selected, *all_messages):
            self.assertNotIn(
                json.dumps(private_value, ensure_ascii=False),
                manifest_text,
            )
        for private_key in (
            '"parent_asin"',
            '"ground_truth"',
            '"sample_id"',
            '"initial_message"',
            '"intent_card"',
        ):
            self.assertNotIn(private_key, manifest_text)
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_suites_cover_fail_open_perturbations_and_scenarios(self) -> None:
        exact_rows = self._rows("fresh_exact")
        for row in exact_rows:
            self.assertEqual(
                recognize_protocol_observation(
                    row["phase15_dialog"]["initial_message"], 1
                ),
                ProtocolObservation.INITIAL,
            )

        paraphrase_rows = self._rows("paraphrase_fail_open")
        self.assertEqual(
            set(self.first["variant_counts"]["paraphrase_fail_open"].values()),
            {9},
        )
        for row in paraphrase_rows:
            dialog = row["phase15_dialog"]
            self.assertEqual(
                recognize_protocol_observation(dialog["initial_message"], 1),
                ProtocolObservation.UNSUPPORTED,
            )
            shapes = dialog["reply_shapes"]
            examples = (
                shapes["disclosure"].format(
                    attribute="material", values="cotton; color: blue"
                ),
                shapes["boundary_decline"].format(attribute="material"),
                shapes["no_additional"].format(attribute="material"),
                shapes["need_attribute"],
                shapes["override"].format(value="cotton"),
            )
            for message in examples:
                self.assertEqual(
                    recognize_protocol_observation(message, 2),
                    ProtocolObservation.UNSUPPORTED,
                )

        perturbation_rows = self._rows("card_perturbed")
        self.assertEqual(
            self.first["variant_counts"]["card_perturbed"],
            {mode: 12 for mode in suites.PERTURBATION_ORDER},
        )
        for row in perturbation_rows:
            detail = row["phase15_card_perturbation"]
            original = detail["original_intent_card"]
            card = row["intent_card"]
            mode = detail["mode"]
            if mode == "constraint_order":
                self.assertEqual(
                    set(card["hard_constraints"]),
                    set(original["hard_constraints"]),
                )
            elif mode == "optional_soft_absent":
                self.assertEqual(
                    card["hard_constraints"], original["hard_constraints"]
                )
                self.assertEqual(card["soft_preferences"], [])
            else:
                self.assertEqual(
                    card["hard_constraints"],
                    ["; ".join(original["hard_constraints"])],
                )
                self.assertLessEqual(
                    len(card["hard_constraints"][0]),
                    suites.MAX_CARD_VALUE_CHARACTERS,
                )

        scenario_rows = self._rows("scenario_balanced")
        self.assertEqual(
            self.first["scenario_counts"]["scenario_balanced"],
            {scenario: 9 for scenario in suites.SCENARIO_ORDER},
        )
        override_rows = [
            row for row in scenario_rows if row["scenario_type"] == "intent_override"
        ]
        boundary_rows = [
            row for row in scenario_rows if row["scenario_type"] == "boundary"
        ]
        self.assertEqual(len(override_rows), 9)
        self.assertEqual(len(boundary_rows), 9)
        for row in override_rows:
            self.assertIn(row["behavior"]["override"]["turn"], (3, 4))

        for suite in (
            "target_disjoint_development",
            "target_disjoint_validation",
        ):
            self.assertEqual(
                self.first["scenario_counts"][suite],
                {scenario: 27 for scenario in suites.SCENARIO_ORDER},
            )

    def test_hash_mismatch_fails_before_any_output(self) -> None:
        output = self.root / "drift-output"
        manifest = self.root / "drift-manifest.json"
        drifted = {**self.expected_hashes, "catalog": "0" * 64}
        with self.assertRaisesRegex(RuntimeError, "hash mismatch: catalog"):
            suites.build(
                catalog=self.catalog,
                forbidden_sources=self.forbidden_sources,
                expected_sha256=drifted,
                output_directory=output,
                manifest_path=manifest,
            )
        self.assertFalse(output.exists())
        self.assertFalse(manifest.exists())

    def test_generated_rows_are_official_evaluator_compatible(self) -> None:
        sample = self._rows("fresh_exact")[0]
        target = sample["ground_truth"]["parent_asin"]

        class _ImmediateTargetAgent:
            def reset(self, session_id: str, user_profile: dict) -> None:
                del session_id, user_profile

            def respond(
                self,
                session_id: str,
                user_message: str,
                turn: int,
                top_k: int,
            ) -> dict:
                del session_id, user_message, turn, top_k
                return {
                    "message": "",
                    "ask_attribute": None,
                    "recommendations": [{"parent_asin": target}],
                }

        products = {
            row["parent_asin"]: row for row in self.catalog_rows
        }
        categories = {
            parent_asin: [str(value) for value in row["categories"]]
            for parent_asin, row in products.items()
        }
        result = evaluate(
            _ImmediateTargetAgent(),
            [sample],
            set(products),
            categories,
            products,
        )

        self.assertEqual(result["sample_count"], 1)
        self.assertEqual(
            result["sessions"][0]["sample_id"], sample["sample_id"]
        )

    def test_execution_harness_revalidates_v2_cardinality_and_balance(self) -> None:
        def load(path: Path) -> list[dict]:
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        def locked(path: Path, rows: list[dict]) -> dict[str, object]:
            targets = {
                row["ground_truth"]["parent_asin"] for row in rows
            }
            return {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": len(rows),
                "case_fingerprint_set_sha256": (
                    ablations._case_fingerprint_set_digest(rows)
                ),
                "target_fingerprint_set_sha256": (
                    ablations._target_set_digest(targets)
                ),
            }

        rows_by_source = {
            name: self._rows(name) for name in suites.SUITE_ORDER
        }
        source_paths = {
            "public_confirmation": self.public,
            "legacy_development": self.development,
            "legacy_validation": self.validation,
            "phase14_fresh": self.phase14_fresh,
        }
        for name, path in source_paths.items():
            rows_by_source[name] = load(path)
        targets_by_source = {
            name: {
                row["ground_truth"]["parent_asin"] for row in rows
            }
            for name, rows in rows_by_source.items()
        }
        suite_lock = {
            "catalog_sha256": self.expected_hashes["catalog"],
            "generator_source_sha256": {
                ablations.ROBUSTNESS_GENERATOR_RELATIVE: _sha256(
                    Path(suites.__file__)
                ),
                ablations.ROBUSTNESS_REFERENCE_RELATIVES["evaluator"]: (
                    self.expected_hashes["evaluator"]
                ),
                ablations.ROBUSTNESS_REFERENCE_RELATIVES["phase14_builder"]: (
                    self.expected_hashes["phase14_builder"]
                ),
            },
            "sources": {
                **{
                    name: locked(
                        self.first_output / f"{name}.jsonl",
                        rows_by_source[name],
                    )
                    for name in suites.SUITE_ORDER
                },
                "public_confirmation": locked(
                    self.public,
                    rows_by_source["public_confirmation"],
                ),
            },
            "prior_sources": {
                name: locked(path, rows_by_source[name])
                for name, path in source_paths.items()
                if name != "public_confirmation"
            },
        }

        ablations._validate_robustness_manifest(
            self.first,
            suite_lock,
            rows_by_source,
            targets_by_source,
            self.root,
        )

        bad_selection = copy.deepcopy(self.first)
        bad_selection["selection_policy"][
            "cases_per_family_popularity_cell"
        ]["target_disjoint_validation"] = 4
        with self.assertRaisesRegex(RuntimeError, "selection policy"):
            ablations._validate_robustness_manifest(
                bad_selection,
                suite_lock,
                rows_by_source,
                targets_by_source,
                self.root,
            )

        bad_balance = copy.deepcopy(self.first)
        bad_balance["family_counts"]["target_disjoint_development"][
            "apparel"
        ] -= 1
        with self.assertRaisesRegex(RuntimeError, "balance proof"):
            ablations._validate_robustness_manifest(
                bad_balance,
                suite_lock,
                rows_by_source,
                targets_by_source,
                self.root,
            )

    def test_distribution_rejects_joint_skew_hidden_by_valid_marginals(
        self,
    ) -> None:
        rows = {
            name: self._rows(name) for name in suites.SUITE_ORDER
        }
        expected_counts = {name: len(value) for name, value in rows.items()}
        manifest = copy.deepcopy(self.first)
        ablations._validate_robustness_distribution(
            rows,
            manifest,
            expected_counts,
        )

        skewed = copy.deepcopy(rows)
        suite_rows = skewed["fresh_exact"]
        apparel_tail = next(
            row
            for row in suite_rows
            if row["phase15_family"] == "apparel"
            and row["phase15_popularity_stratum"] == "tail"
        )
        footwear_torso = next(
            row
            for row in suite_rows
            if row["phase15_family"] == "footwear"
            and row["phase15_popularity_stratum"] == "torso"
        )
        apparel_tail["phase15_popularity_stratum"] = "torso"
        footwear_torso["phase15_popularity_stratum"] = "tail"

        self.assertEqual(
            suites._aggregate_counts(
                skewed,
                "phase15_family",
                suites.FAMILY_ORDER,
            ),
            manifest["family_counts"],
        )
        self.assertEqual(
            suites._aggregate_counts(
                skewed,
                "phase15_popularity_stratum",
                suites.POPULARITY_ORDER,
            ),
            manifest["popularity_counts"],
        )
        with self.assertRaisesRegex(RuntimeError, "joint strata"):
            ablations._validate_robustness_distribution(
                skewed,
                manifest,
                expected_counts,
            )

    def test_distribution_rejects_unknown_variant_even_when_echo_matches(
        self,
    ) -> None:
        rows = {
            name: self._rows(name) for name in suites.SUITE_ORDER
        }
        expected_counts = {name: len(value) for name, value in rows.items()}
        mutated = copy.deepcopy(rows)
        mutated["paraphrase_fail_open"][0][
            "phase15_paraphrase_shape"
        ] = "unknown_shape"
        manifest = copy.deepcopy(self.first)
        manifest["variant_counts"] = suites._variant_counts(mutated)

        with self.assertRaisesRegex(RuntimeError, "variant coverage"):
            ablations._validate_robustness_distribution(
                mutated,
                manifest,
                expected_counts,
            )

    def test_publication_failure_leaves_no_partial_artifact_set(self) -> None:
        output = self.root / "partial-output"
        manifest = self.root / "partial-manifest.json"
        real_replace = suites.os.replace
        calls = 0

        def fail_second_replace(source: str, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic publication failure")
            real_replace(source, destination)

        with mock.patch.object(
            suites.os,
            "replace",
            side_effect=fail_second_replace,
        ):
            with self.assertRaisesRegex(
                OSError, "synthetic publication failure"
            ):
                suites.build(
                    catalog=self.catalog,
                    forbidden_sources=self.forbidden_sources,
                    expected_sha256=self.expected_hashes,
                    output_directory=output,
                    manifest_path=manifest,
                )

        self.assertFalse(manifest.exists())
        self.assertEqual(list(output.glob("*.jsonl")), [])
        self.assertFalse(list(self.root.rglob("*.tmp")))

    def test_cli_is_dry_run_by_default(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            suites.main([])
        plan = json.loads(output.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["writes_performed"])
        self.assertEqual(plan["suite_order"], list(suites.SUITE_ORDER))
        self.assertEqual(
            plan["cases_per_suite"],
            {
                suite: suites.SUITE_CASES_PER_CELL[suite]
                * len(suites.FAMILY_ORDER)
                * len(suites.POPULARITY_ORDER)
                for suite in suites.SUITE_ORDER
            },
        )


if __name__ == "__main__":
    unittest.main()
