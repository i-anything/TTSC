from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.record_benchmark import (
    append_record,
    deterministic_json,
    validate_record,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIRECTORY = REPOSITORY_ROOT / "benchmarks"
DOCUMENT_DIRECTORY = REPOSITORY_ROOT / "docs"


def valid_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "record_id": "candidate-v1",
        "source_document": "docs/candidate_results.json",
        "system": "candidate-v1",
        "sample_count": 3,
        "metrics": {
            "hit_rate_at_10": 0.5,
            "mrr": 0.25,
            "mttc": 4.0,
            "efficiency": 0.6,
            "recommended_technical_score": 0.45,
        },
        "scenario_metrics": {
            "boundary": {
                "sample_count": 3,
                "hit_rate_at_10": 0.5,
                "mrr": 0.25,
                "mttc": 4.0,
            }
        },
    }


class BenchmarkRecorderTest(unittest.TestCase):
    def test_serialization_is_deterministic(self) -> None:
        first = valid_record()
        second = {
            key: first[key]
            for key in reversed(tuple(first))
        }
        second["metrics"] = {
            key: first["metrics"][key]  # type: ignore[index]
            for key in reversed(tuple(first["metrics"]))  # type: ignore[arg-type]
        }

        self.assertEqual(deterministic_json(first), deterministic_json(second))
        self.assertTrue(deterministic_json(first).endswith(b"\n"))

    def test_append_is_atomic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = append_record(valid_record(), root)
            original = destination.read_bytes()

            self.assertEqual(original, deterministic_json(valid_record()))
            self.assertEqual(list(root.glob(".*.tmp")), [])
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                append_record(valid_record(), root)
            self.assertEqual(destination.read_bytes(), original)

    def test_failed_publication_leaves_no_partial_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "scripts.record_benchmark.os.link",
                side_effect=OSError("injected publication failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected publication failure"):
                    append_record(valid_record(), root)

            self.assertFalse((root / "candidate-v1.json").exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_racing_writer_is_reported_as_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def publish_competing_record(_source: object, destination: object) -> None:
                Path(destination).write_bytes(b"competing-writer\n")
                raise FileExistsError("race")

            with mock.patch(
                "scripts.record_benchmark.os.link",
                side_effect=publish_competing_record,
            ):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    append_record(valid_record(), root)

            self.assertEqual(
                (root / "candidate-v1.json").read_bytes(),
                b"competing-writer\n",
            )
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_unknown_or_sensitive_fields_are_rejected(self) -> None:
        for field in (
            "sessions",
            "sample_id",
            "user_message",
            "user_profile",
            "ground_truth",
            "scenario_type",
            "label",
        ):
            with self.subTest(field=field):
                record = valid_record()
                record[field] = "must-not-be-recorded"
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    validate_record(record)

    def test_invalid_numbers_identifiers_and_scenario_totals_are_rejected(self) -> None:
        invalid_records = []

        nan_record = valid_record()
        nan_record["metrics"] = dict(nan_record["metrics"])  # type: ignore[arg-type]
        nan_record["metrics"]["mrr"] = float("nan")  # type: ignore[index]
        invalid_records.append(nan_record)

        huge_number = valid_record()
        huge_number["metrics"] = dict(huge_number["metrics"])  # type: ignore[arg-type]
        huge_number["metrics"]["mrr"] = 10**10000  # type: ignore[index]
        invalid_records.append(huge_number)

        boolean_version = valid_record()
        boolean_version["schema_version"] = True
        invalid_records.append(boolean_version)

        raw_system = valid_record()
        raw_system["system"] = "show me red shoes"
        invalid_records.append(raw_system)

        traversal = valid_record()
        traversal["source_document"] = "../private.json"
        invalid_records.append(traversal)

        bad_total = valid_record()
        bad_total["sample_count"] = 4
        invalid_records.append(bad_total)

        for record in invalid_records:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    validate_record(record)


class BackfilledBenchmarkTest(unittest.TestCase):
    def _load(self, path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_schema_is_strict_and_parseable(self) -> None:
        schema = self._load(BENCHMARK_DIRECTORY / "schema.json")

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)  # type: ignore[index]

    def test_backfills_match_source_documents_exactly(self) -> None:
        baseline = self._load(BENCHMARK_DIRECTORY / "baseline.json")
        baseline_source = self._load(DOCUMENT_DIRECTORY / "baseline_results.json")
        validate_record(baseline)

        self.assertEqual(baseline["system"], baseline_source["baseline"])
        self.assertEqual(baseline["sample_count"], baseline_source["sample_count"])
        baseline_metrics = baseline["metrics"]
        for metric in ("hit_rate_at_10", "mrr", "mttc", "efficiency"):
            self.assertEqual(baseline_metrics[metric], baseline_source[metric])  # type: ignore[index]
        self.assertEqual(
            baseline_metrics["recommended_technical_score"],  # type: ignore[index]
            baseline_source["technical_score"],
        )
        self.assertNotIn("scenario_metrics", baseline)

        for record_id in (
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "phase5",
            "phase6",
            "phase7",
        ):
            with self.subTest(record_id=record_id):
                record = self._load(BENCHMARK_DIRECTORY / f"{record_id}.json")
                source = self._load(DOCUMENT_DIRECTORY / f"{record_id}_results.json")
                validate_record(record)

                self.assertEqual(record["system"], source["agent_version"])
                self.assertEqual(
                    record["sample_count"],
                    source["evaluation"]["public_samples"],  # type: ignore[index]
                )
                self.assertEqual(record["metrics"], source["metrics"])
                self.assertEqual(record["scenario_metrics"], source["scenario_metrics"])

    def test_records_contain_no_per_sample_or_sensitive_keys(self) -> None:
        forbidden = {
            "ground_truth",
            "label",
            "profile",
            "sample_id",
            "scenario_type",
            "sessions",
            "target",
            "user_message",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value).union(
                    *(keys(item) for item in value.values()),
                )
            if isinstance(value, list):
                return set().union(*(keys(item) for item in value))
            return set()

        for record_id in (
            "baseline",
            "phase1",
            "phase2",
            "phase3",
            "phase4",
            "phase5",
            "phase6",
            "phase7",
        ):
            with self.subTest(record_id=record_id):
                record = self._load(BENCHMARK_DIRECTORY / f"{record_id}.json")
                self.assertEqual(
                    (BENCHMARK_DIRECTORY / f"{record_id}.json").read_bytes(),
                    deterministic_json(record),
                )
                self.assertTrue(forbidden.isdisjoint(keys(record)))


if __name__ == "__main__":
    unittest.main()
