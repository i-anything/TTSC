from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from conversational_search.profiles import (
    BOUNDED_RESIDUAL_PROFILE_POLICY,
    DISABLED_PROFILE_POLICY,
    ProductTheme,
    ProfilePrior,
)
from conversational_search.service import ConversationalSearchAgent
from scripts.run_profile_ablations import (
    CONTRACT_RELATIVE,
    FOCUSED_SUITE_COMMAND,
    FULL_SUITE_COMMAND,
    IMPLEMENTATION_LOCK_RELATIVE,
    PHASE7_DIFFERENTIAL_COMMAND,
    PHASE7_ORACLE_CASES,
    PHASE7_ORACLE_SHA256,
    SOURCE_PATHS,
    _CallAuditRetriever,
    _build_decision_gates,
    _build_implementation_lock,
    _claim_run_output,
    _collect_prelock_verification,
    _core_health,
    _faults_are_zero,
    _inspect_retained_profile_state,
    _overall_official_summary,
    _paired_hit_to_miss_count,
    _project_profile_health,
    _publication_privacy_is_valid,
    _ten_bit_profile_contract_holds,
    _validate_implementation_lock,
    _validate_output,
    _validate_publication_privacy,
    _write_implementation_lock,
)


def _profile_health(*, policy: str, sessions: int, applications: int = 0) -> dict:
    nonzero = min(sessions, applications)
    return {
        "policy": policy,
        "session_entries": sessions,
        "logical_profile_bytes": sessions * 2,
        "profiles_reset": sessions,
        "zero_mask_profiles": sessions - nonzero,
        "nonzero_mask_profiles": nonzero,
        "recognized_theme_count": nonzero,
        "turns_disabled_by_active_requirements": 0,
        "eligible_stage_a_attempts": applications,
        "empty_represented_theme_fallbacks": 0,
        "constant_score_neutral_fallbacks": 0,
        "successful_residual_applications": applications,
        "parsing_or_scoring_fallbacks": 0,
    }


def _result(*, mrr: float, mttc: float, score: float) -> dict:
    return {
        "sample_count": 200,
        "hit_rate_at_10": 0.99,
        "mrr": mrr,
        "mttc": mttc,
        "efficiency": 0.8,
        "recommended_technical_score": score,
        "reported_token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "scenario_metrics": {},
        "sessions": [],
    }


def _publication() -> dict:
    baseline = _overall_official_summary(
        _result(mrr=0.52223, mttc=3.07, score=0.810269)
    )
    candidate = _overall_official_summary(
        _result(mrr=0.53, mttc=3.0, score=0.82)
    )
    return {
        "schema_version": 1,
        "experiment_id": "phase9-bounded-profile-aware-stage-a-v1",
        "candidate": "phase9-bounded-profile-residual-v1",
        "baseline": "phase7-exact-ranking-reuse-v1",
        "run_configuration": {},
        "official_metrics": {
            "baseline": baseline,
            "candidate": candidate,
            "delta": {
                "hit_rate_at_10": 0.0,
                "mrr": 0.00777,
                "mttc": -0.07,
                "efficiency": 0.0,
                "recommended_technical_score": 0.009731,
            },
        },
        "paired_quality": {},
        "health": {},
        "latency": {},
        "exactness": {},
        "privacy": {},
        "reproducibility": {},
        "decision_gate": {},
    }


def _verification(*, focused: int = 10, complete: int = 20) -> dict:
    return {
        "focused_suite_command": FOCUSED_SUITE_COMMAND,
        "focused_tests_passed": focused,
        "complete_suite_command": FULL_SUITE_COMMAND,
        "complete_unit_tests_passed": complete,
        "phase7_exact_differential_command": PHASE7_DIFFERENTIAL_COMMAND,
        "phase7_exact_differential_cases": PHASE7_ORACLE_CASES,
        "phase7_exact_oracle_sha256": PHASE7_ORACLE_SHA256,
        "completed_before_lock": True,
    }


def _diagnostics(
    *,
    policy: str,
    route_calls: int,
    applications: int,
    wall: float,
    p95: float,
) -> dict:
    return {
        "expected_turns": 500,
        "route_health": {
            "bm25": {"ok": route_calls},
            "dense": {"ok": route_calls},
            "fallback_turns": 0,
            "candidate_document_calls": route_calls,
        },
        "ranking_health": {
            "policy": "stage_a",
            "attempts": route_calls,
            "successes": route_calls,
            "failures": 0,
            "unavailable_skips": 0,
        },
        "slate_health": {
            "policy": "stagnation_aware",
            "attempts": 500,
            "successes": 500,
            "failures": 0,
        },
        "orchestration_health": {
            "searches": route_calls,
            "fault_invalidations": 0,
            "store_rejections": 0,
        },
        "profile_health": _profile_health(
            policy=policy,
            sessions=200,
            applications=applications,
        ),
        "retained_profile_state_valid": True,
        "evaluation_wall_seconds": wall,
        "respond_latency_ms": {"warm_p95": p95},
    }


class ProfileAblationPrivacyTests(unittest.TestCase):
    def test_publication_accepts_aggregate_only_payload(self) -> None:
        payload = _publication()
        payload["health"] = {
            "profiles_reset": 200,
            "successful_residual_applications": 14,
        }
        payload["reproducibility"] = {
            "baseline_record": "benchmarks/phase7.json"
        }

        self.assertTrue(_publication_privacy_is_valid(payload))
        _validate_publication_privacy(payload)

    def test_overall_metrics_drop_scenario_cells_and_labels(self) -> None:
        result = _result(mrr=0.53, mttc=3.0, score=0.82)
        result["scenario_metrics"] = {
            "browsing": {"sample_count": 1, "mrr": 1.0}
        }

        summary = _overall_official_summary(result)

        self.assertNotIn("scenario_metrics", summary)
        self.assertNotIn("browsing", json.dumps(summary, sort_keys=True))

    def test_publication_rejects_private_fields_and_product_ids(self) -> None:
        mutations = (
            lambda payload: payload.update({"sessions": []}),
            lambda payload: payload["health"].update(
                {"preference_tags": ["private"]}
            ),
            lambda payload: payload["health"].update({"profile_mask": 3}),
            lambda payload: payload["health"].update(
                {"theme_frequencies": {"private": 1}}
            ),
            lambda payload: payload["health"].update(
                {"parent_asin": "synthetic"}
            ),
            lambda payload: payload["health"].update(
                {"value": "B012345678"}
            ),
            lambda payload: payload["health"].update(
                {"aggregate_label": "browsing"}
            ),
            lambda payload: payload["official_metrics"]["candidate"].update(
                {"scenario_metrics": {}}
            ),
        )
        for mutate in mutations:
            payload = _publication()
            mutate(payload)
            with self.subTest(payload=payload):
                self.assertFalse(_publication_privacy_is_valid(payload))
                with self.assertRaises(RuntimeError):
                    _validate_publication_privacy(payload)

    def test_paired_hit_to_miss_publishes_only_a_count(self) -> None:
        baseline = {
            "sessions": [
                {"sample_id": "one", "scenario_type": "a", "hit": True},
                {"sample_id": "two", "scenario_type": "b", "hit": False},
            ]
        }
        candidate = {
            "sessions": [
                {"sample_id": "one", "scenario_type": "a", "hit": False},
                {"sample_id": "two", "scenario_type": "b", "hit": True},
            ]
        }

        self.assertEqual(_paired_hit_to_miss_count(baseline, candidate), 1)

    def test_paired_hit_to_miss_rejects_alignment_drift(self) -> None:
        baseline = {
            "sessions": [
                {"sample_id": "one", "scenario_type": "a", "hit": True}
            ]
        }
        candidate = {
            "sessions": [
                {"sample_id": "different", "scenario_type": "a", "hit": True}
            ]
        }
        with self.assertRaises(RuntimeError):
            _paired_hit_to_miss_count(baseline, candidate)


class ProfileAblationHealthTests(unittest.TestCase):
    @staticmethod
    def _retained_agent() -> object:
        class SyntheticAgent:
            pass

        agent = SyntheticAgent()
        agent._profile_policy = BOUNDED_RESIDUAL_PROFILE_POLICY
        agent._profile_priors = {
            b"x" * 32: ProfilePrior(ProductTheme.COMFORT),
        }
        agent._profiles_reset = 1
        agent._zero_mask_profiles = 0
        agent._nonzero_mask_profiles = 1
        agent._recognized_theme_count = 1
        agent._turns_disabled_by_active_requirements = 0
        agent._eligible_stage_a_attempts = 1
        agent._empty_represented_theme_fallbacks = 0
        agent._constant_score_neutral_fallbacks = 0
        agent._successful_residual_applications = 1
        agent._parsing_or_scoring_fallbacks = 0
        return agent

    def test_candidate_document_audit_counts_calls_without_payloads(self) -> None:
        class Backend:
            def candidate_documents(self, parent_asins: tuple[str, ...]) -> tuple:
                return tuple(parent_asins)

        audit = _CallAuditRetriever(Backend())
        self.assertEqual(audit.candidate_documents(("synthetic",)), ("synthetic",))
        self.assertEqual(audit.summary()["candidate_document_calls"], 1)

    def test_retained_profile_inspector_accepts_only_hashed_bounded_priors(self) -> None:
        self.assertTrue(
            _inspect_retained_profile_state(
                self._retained_agent(),  # type: ignore[arg-type]
                expected_sessions=1,
            )
        )

    def test_retained_profile_inspector_accepts_clean_real_agent(self) -> None:
        agent = ConversationalSearchAgent(
            "unused-synthetic-catalog.jsonl",
            retriever=object(),
        )
        agent.reset("synthetic-session", {"preference_tags": ["comfort"]})

        self.assertTrue(
            _inspect_retained_profile_state(agent, expected_sessions=1)
        )

    def test_retained_profile_inspector_rejects_raw_or_extra_state(self) -> None:
        bad_key = self._retained_agent()
        bad_key._profile_priors = {"raw-session": ProfilePrior()}  # type: ignore[attr-defined]
        raw_mapping = self._retained_agent()
        raw_mapping._raw_data = {  # type: ignore[attr-defined]
            "user_profile": {"preference_tags": ["private"]}
        }
        extra_profile_attribute = self._retained_agent()
        extra_profile_attribute._profile_tags = ("private",)  # type: ignore[attr-defined]

        for agent in (bad_key, raw_mapping, extra_profile_attribute):
            with self.subTest(agent=agent):
                self.assertFalse(
                    _inspect_retained_profile_state(
                        agent,  # type: ignore[arg-type]
                        expected_sessions=1,
                    )
                )

    def test_profile_health_is_exact_bounded_aggregate_schema(self) -> None:
        health = _profile_health(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            sessions=3,
            applications=1,
        )
        self.assertEqual(
            _project_profile_health(
                health,
                expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
                expected_sessions=3,
            ),
            health,
        )
        self.assertTrue(_ten_bit_profile_contract_holds())

    def test_profile_health_rejects_memory_or_schema_drift(self) -> None:
        health = _profile_health(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            sessions=2,
            applications=1,
        )
        for invalid in (
            {**health, "logical_profile_bytes": 5},
            {**health, "unexpected_dimension": 1},
            {**health, "recognized_theme_count": 11},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RuntimeError):
                    _project_profile_health(
                        invalid,
                        expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
                        expected_sessions=2,
                    )

    def test_disabled_comparator_cannot_perform_profile_work(self) -> None:
        health = _profile_health(
            policy=DISABLED_PROFILE_POLICY.value,
            sessions=1,
            applications=1,
        )
        with self.assertRaises(RuntimeError):
            _project_profile_health(
                health,
                expected_policy=DISABLED_PROFILE_POLICY,
                expected_sessions=1,
            )

    def test_combined_fallback_counter_is_a_gate_failure_not_projection_abort(self) -> None:
        health = _profile_health(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            sessions=1,
            applications=0,
        )
        health.update(
            {
                "zero_mask_profiles": 0,
                "nonzero_mask_profiles": 1,
                "recognized_theme_count": 1,
                "eligible_stage_a_attempts": 1,
                "parsing_or_scoring_fallbacks": 2,
            }
        )
        projected = _project_profile_health(
            health,
            expected_policy=BOUNDED_RESIDUAL_PROFILE_POLICY,
            expected_sessions=1,
        )
        diagnostics = _diagnostics(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            route_calls=1,
            applications=0,
            wall=1.0,
            p95=1.0,
        )
        diagnostics["profile_health"] = projected

        self.assertFalse(_faults_are_zero(diagnostics))


class ProfileImplementationLockTests(unittest.TestCase):
    def _repository(self, directory: str) -> tuple[Path, tuple[str, ...]]:
        root = Path(directory)
        contract = root / CONTRACT_RELATIVE
        contract.parent.mkdir(parents=True)
        contract.write_text('{"frozen":true}\n', encoding="utf-8")
        source = root / "candidate.py"
        source.write_text("VALUE = 1\n", encoding="utf-8")
        return root, ("candidate.py",)

    def test_lock_build_is_deterministic_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, sources = self._repository(directory)
            first = _build_implementation_lock(
                root,
                verification=_verification(),
                source_paths=sources,
            )
            second = _build_implementation_lock(
                root,
                verification=_verification(),
                source_paths=sources,
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first["verification"]["focused_suite_command"],
                FOCUSED_SUITE_COMMAND,
            )
            self.assertEqual(
                first["verification"]["complete_suite_command"],
                FULL_SUITE_COMMAND,
            )

            lock_path = root / IMPLEMENTATION_LOCK_RELATIVE
            _write_implementation_lock(lock_path, first)
            self.assertEqual(
                _validate_implementation_lock(
                    root,
                    source_paths=sources,
                ),
                first,
            )

    def test_lock_write_refuses_overwrite_and_detects_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, sources = self._repository(directory)
            lock = _build_implementation_lock(
                root,
                verification=_verification(focused=1, complete=1),
                source_paths=sources,
            )
            lock_path = root / IMPLEMENTATION_LOCK_RELATIVE
            _write_implementation_lock(lock_path, lock)
            with self.assertRaises(FileExistsError):
                _write_implementation_lock(lock_path, lock)

            (root / sources[0]).write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                _validate_implementation_lock(root, source_paths=sources)

    def test_lock_rejects_unverified_or_drifted_pre_lock_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, sources = self._repository(directory)
            zero_tests = _verification(focused=0)
            with self.assertRaises(ValueError):
                _build_implementation_lock(
                    root,
                    verification=zero_tests,
                    source_paths=sources,
                )
            wrong_oracle = {
                **_verification(),
                "phase7_exact_oracle_sha256": "0" * 64,
            }
            with self.assertRaises(ValueError):
                _build_implementation_lock(
                    root,
                    verification=wrong_oracle,
                    source_paths=sources,
                )

    def test_prelock_verification_executes_and_parses_frozen_commands(self) -> None:
        oracle = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "cases": PHASE7_ORACLE_CASES,
                    "digest": PHASE7_ORACLE_SHA256,
                    "status": "ok",
                }
            ),
            stderr="",
        )
        focused = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="Ran 151 tests in 0.2s\n\nOK\n",
        )
        complete = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="Ran 260 tests in 1.0s\n\nOK\n",
        )
        repository_root = Path(__file__).resolve().parents[1]

        with patch(
            "scripts.run_profile_ablations.subprocess.run",
            side_effect=(oracle, focused, complete),
        ) as run:
            verification = _collect_prelock_verification(repository_root)

        self.assertEqual(verification, _verification(focused=151, complete=260))
        self.assertEqual(run.call_count, 3)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands[0][1:],
            ["-m", "scripts.verify_phase7_stage_a_oracle"],
        )
        self.assertIn("tests.test_profile_ablations", commands[1])
        self.assertEqual(commands[2][1:], ["-m", "unittest", "discover", "-s", "tests", "-q"])

    def test_prelock_verification_rejects_failed_or_unparseable_commands(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        failed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="synthetic failure",
        )
        with patch(
            "scripts.run_profile_ablations.subprocess.run",
            return_value=failed,
        ):
            with self.assertRaises(RuntimeError):
                _collect_prelock_verification(repository_root)

        oracle = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "cases": PHASE7_ORACLE_CASES,
                    "digest": PHASE7_ORACLE_SHA256,
                    "status": "ok",
                }
            ),
            stderr="",
        )
        unparseable = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="OK without an executed-test count",
        )
        with patch(
            "scripts.run_profile_ablations.subprocess.run",
            side_effect=(oracle, unparseable),
        ):
            with self.assertRaises(RuntimeError):
                _collect_prelock_verification(repository_root)


class ProfileDecisionGateTests(unittest.TestCase):
    def test_all_frozen_promotion_gates_can_pass(self) -> None:
        baseline = _result(mrr=0.52223, mttc=3.07, score=0.810269)
        candidate = _result(mrr=0.53, mttc=3.0, score=0.82)
        baseline_diagnostics = _diagnostics(
            policy=DISABLED_PROFILE_POLICY.value,
            route_calls=400,
            applications=0,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            route_calls=390,
            applications=10,
            wall=10.4,
            p95=20.8,
        )
        replay_diagnostics = {
            **candidate_diagnostics,
            "evaluation_wall_seconds": 10.2,
            "respond_latency_ms": {"warm_p95": 20.6},
        }
        independent_diagnostics = _core_health(candidate_diagnostics)
        trace = [("aggregate-private-trace",)]
        lock = {"verification": _verification()}

        gates, latency = _build_decision_gates(
            baseline=baseline,
            candidate=candidate,
            replay=candidate,
            independent=candidate,
            baseline_diagnostics=baseline_diagnostics,
            candidate_diagnostics=candidate_diagnostics,
            replay_diagnostics=replay_diagnostics,
            independent_diagnostics=independent_diagnostics,
            candidate_trace=trace,
            replay_trace=trace,
            independent_trace=trace,
            hit_to_miss_count=0,
            implementation_lock=lock,
            publication_privacy_valid=True,
        )

        self.assertTrue(gates["adopt"])
        self.assertEqual(latency["candidate_wall_time_ratio"], 1.04)
        self.assertEqual(latency["candidate_warm_p95_ratio"], 1.04)

    def test_one_hit_to_miss_forces_rejection(self) -> None:
        baseline = _result(mrr=0.52223, mttc=3.07, score=0.810269)
        candidate = _result(mrr=0.53, mttc=3.0, score=0.82)
        baseline_diagnostics = _diagnostics(
            policy=DISABLED_PROFILE_POLICY.value,
            route_calls=400,
            applications=0,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics = _diagnostics(
            policy=BOUNDED_RESIDUAL_PROFILE_POLICY.value,
            route_calls=390,
            applications=10,
            wall=10.0,
            p95=20.0,
        )
        candidate_diagnostics["retained_profile_state_valid"] = False
        trace: list[tuple[object, ...]] = []
        gates, _latency = _build_decision_gates(
            baseline=baseline,
            candidate=candidate,
            replay=candidate,
            independent=candidate,
            baseline_diagnostics=baseline_diagnostics,
            candidate_diagnostics=candidate_diagnostics,
            replay_diagnostics=candidate_diagnostics,
            independent_diagnostics=_core_health(candidate_diagnostics),
            candidate_trace=trace,
            replay_trace=trace,
            independent_trace=trace,
            hit_to_miss_count=1,
            implementation_lock={
                "verification": _verification(focused=1, complete=1)
            },
            publication_privacy_valid=False,
        )

        self.assertFalse(gates["phase7_hit_to_phase9_miss_count_is_zero"])
        self.assertFalse(gates["aggregate_publication_privacy_valid"])
        self.assertFalse(
            gates["retained_profile_state_is_only_valid_ten_bit_mask"]
        )
        self.assertFalse(gates["adopt"])


class ProfileAblationOutputGuardTests(unittest.TestCase):
    def test_exclusive_run_claim_survives_and_refuses_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            _claim_run_output(output)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "experiment_id": "phase9-bounded-profile-aware-stage-a-v1",
                    "status": "run_started_no_retry",
                },
            )
            with self.assertRaises(FileExistsError):
                _claim_run_output(output)

    def test_output_guard_rejects_inputs_sources_lock_and_publication_paths(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        catalog = repository_root / "data" / "catalog.jsonl"
        dataset = repository_root / "data" / "public_set.jsonl"
        rejected = (
            catalog,
            dataset,
            repository_root / SOURCE_PATHS[0],
            repository_root / IMPLEMENTATION_LOCK_RELATIVE,
            repository_root / "docs" / "phase9_results.json",
            repository_root / "benchmarks" / "phase9.json",
        )
        for output in rejected:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    _validate_output(output, catalog, dataset)

    def test_output_guard_accepts_gitignored_style_root_result(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with patch.object(Path, "exists", return_value=False):
            _validate_output(
                repository_root / "results-phase9-profile-aware-ranking.json",
                repository_root / "data" / "catalog.jsonl",
                repository_root / "data" / "public_set.jsonl",
            )

    def test_output_guard_refuses_a_second_frozen_run(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaises(FileExistsError):
                _validate_output(
                    repository_root / "results-phase9-profile-aware-ranking.json",
                    repository_root / "data" / "catalog.jsonl",
                    repository_root / "data" / "public_set.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
