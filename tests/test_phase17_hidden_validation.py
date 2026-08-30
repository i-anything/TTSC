from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import evaluator.local_evaluator as evaluator_module

from scripts import build_phase17_clean_room_suite as builder
from scripts import run_phase17_hidden_validation as runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Phase17BuilderTests(unittest.TestCase):
    def test_builder_import_does_not_import_agent_runtime(self) -> None:
        command = (
            "import sys; "
            "import scripts.build_phase17_clean_room_suite; "
            "assert 'starter.agent' not in sys.modules; "
            "assert not any(name == 'conversational_search' or "
            "name.startswith('conversational_search.') for name in sys.modules)"
        )
        subprocess.run(
            [sys.executable, "-c", command],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    def test_pure_evaluator_helpers_are_exact(self) -> None:
        products = [
            {
                "title": "Trail Jacket",
                "features": ["water resistant", "regular fit"],
                "details": {"Material": "nylon", "Color": "blue"},
                "description": ["outdoor work layer"],
                "categories": ["Clothing, Shoes & Jewelry", "Men", "Jackets"],
                "store": "Example",
                "price": 79.5,
            },
            {
                "title": "Silver Ring",
                "features": [],
                "details": {},
                "description": [],
                "categories": ["Jewelry", "Rings"],
                "store": "",
                "price": None,
            },
        ]
        for product in products:
            self.assertEqual(builder._intent_card(product), evaluator_module.intent_card(product))

        categories = ["Clothing, Shoes & Jewelry", "Women", "Running Shoes"]
        self.assertEqual(
            builder._coarse_category(categories),
            evaluator_module.coarse_category(categories),
        )
        for value in (
            "cotton",
            "color: blue",
            "wide size",
            "regular fit",
            "outdoor work",
            "budget around $20",
            "zip closure",
        ):
            self.assertEqual(
                builder._classify_constraint(value),
                evaluator_module.classify_constraint(value),
            )

        card = builder._intent_card(products[0])
        for scenario in runner.SCENARIO_ORDER:
            self.assertEqual(
                builder._behavior_for(scenario, card, random.Random(17)),
                evaluator_module.behavior_for(scenario, card, random.Random(17)),
            )

    def test_template_bank_is_locked_and_valid(self) -> None:
        path = REPOSITORY_ROOT / runner.TEMPLATE_RELATIVE
        templates = builder._templates(path)
        self.assertEqual(set(templates), set(builder.EVENT_SLOTS))
        self.assertTrue(all(len(values) == 8 for values in templates.values()))

    def test_surface_and_scenario_assignment_is_exact(self) -> None:
        assignments = builder._case_assignments(bytes(range(32)))
        self.assertEqual(len(assignments), builder.CASE_COUNT)
        observed = Counter(assignments)
        for surface, counts in builder.SURFACE_SCENARIO_COUNTS.items():
            for scenario, count in counts.items():
                self.assertEqual(observed[(surface, scenario)], count)

    def test_family_classifier_ignores_catalog_root(self) -> None:
        root = "Clothing, Shoes & Jewelry"
        self.assertEqual(builder._family([root, "Women", "Dresses"]), "apparel")
        self.assertEqual(builder._family([root, "Men", "Running Shoes"]), "footwear")
        self.assertEqual(
            builder._family([root, "Women", "Necklaces"]),
            "jewelry_accessories",
        )


class Phase17RunnerTests(unittest.TestCase):
    def _sample(self, mode: str) -> dict:
        dialog = {"mode": "official_exact"}
        if mode == "clean_room_language_shift":
            dialog = {
                "mode": mode,
                "initial_message": "Shifted opening",
                "reply_templates": {
                    "need_attribute": "Ask one detail.",
                    "disclosure": "For {attribute}, use {values}.",
                    "no_additional": "Nothing else for {attribute}.",
                    "boundary_indifference": "Choose {attribute} for me.",
                },
            }
        return {
            "sample_id": "case",
            "scenario_type": "buying",
            "intent_card": {
                "target_category": "Trail Jacket",
                "hard_constraints": ["nylon", "color: blue"],
                "soft_preferences": ["regular fit"],
            },
            "behavior": {"scenario_type": "buying"},
            "phase17_surface": mode,
            "phase17_dialog": dialog,
        }

    def test_shifted_surface_preserves_disclosure_semantics(self) -> None:
        sample = self._sample("clean_room_language_shift")
        disclosed: set[str] = set()
        with runner._language_surface():
            initial = evaluator_module.initial_message(sample, "Jackets", disclosed)
            reply, boundary = evaluator_module.customer_reply(
                sample,
                "color",
                disclosed,
                False,
            )
        self.assertEqual(initial, "Shifted opening")
        self.assertIn("nylon", disclosed)
        self.assertIn("For color", reply)
        self.assertIn("color: blue", disclosed)
        self.assertFalse(boundary)

    def test_official_surface_delegates_exactly(self) -> None:
        sample = self._sample("official_exact")
        expected_disclosed: set[str] = set()
        expected = evaluator_module.initial_message(sample, "Jackets", expected_disclosed)
        actual_disclosed: set[str] = set()
        with runner._language_surface():
            actual = evaluator_module.initial_message(sample, "Jackets", actual_disclosed)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_disclosed, expected_disclosed)

    def test_zero_delta_bootstrap_is_exact(self) -> None:
        outcomes: list[dict] = []
        ordinal = 0
        for scenario, count in builder.SCENARIO_COUNTS.items():
            for _ in range(count):
                ordinal += 1
                outcomes.append(
                    {
                        "sample_id": f"case-{ordinal}",
                        "scenario_type": scenario,
                        "hit": True,
                        "first_hit_turn": 2,
                        "best_rank": 1,
                        "reciprocal_rank": 1.0,
                    }
                )
        strata, transitions = runner._paired_deltas(outcomes, outcomes)
        result = runner._bootstrap(strata)
        self.assertEqual(transitions["both_hit"], builder.CASE_COUNT)
        self.assertTrue(all(value == 0.0 for value in result["mean_delta"].values()))
        self.assertTrue(
            all(value == 0.0 for value in result["lower_95_one_sided"].values())
        )
        self.assertTrue(
            all(value == 0.0 for value in result["upper_95_one_sided"].values())
        )

    def test_publication_rejects_row_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(RuntimeError):
                runner._publish(path, {"sample_id": "private"})
            self.assertFalse(path.exists())

    def test_contract_contains_predeclared_mixed_surface_gate(self) -> None:
        contract = json.loads(
            (REPOSITORY_ROOT / runner.CONTRACT_RELATIVE).read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["suite"]["surface_counts"],
            {"official_exact": 400, "clean_room_language_shift": 400},
        )
        self.assertEqual(
            contract["promotion_gates"]["surface_hit_rate_maximum_point_decline"],
            {"official_exact": 0.05, "clean_room_language_shift": 0.05},
        )


if __name__ == "__main__":
    unittest.main()
