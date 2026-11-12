import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from negsim.run_sweep import (
    _count_successful_game_end_runs_by_combo,
    _find_existing_sweep_logs,
    _prepare_existing_sweep_logs,
    _validate_continue_proposal_schema_versions,
    run_sweep,
)
from negsim.log_paths import (
    combo_folder_name,
    combo_id_from_config,
    iter_results_paths,
    run_log_dir,
    sweep_folder_name,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _yaml_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_result_log(
    log_root: Path,
    name: str,
    *,
    sweep_id: str,
    combo_id: str | None = None,
    outcome: str = "deal",
    game_end: bool = True,
    config_name: str = "",
    folder_combo_id: str = "",
) -> Path:
    folder_combo = folder_combo_id or combo_id or "unknown_combo"
    run_dir = (
        log_root
        / sweep_folder_name(sweep_id)
        / combo_folder_name(folder_combo)
        / name
    )
    run_dir.mkdir(parents=True)
    results = {
        "sweep_id": sweep_id,
        "outcome": outcome,
    }
    if combo_id is not None:
        results["combo_id"] = combo_id
    if config_name:
        results["config_name"] = config_name
    (run_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    if game_end:
        (run_dir / "summary.log").write_text(
            "**Game end reason:** deal\n", encoding="utf-8"
        )
    return run_dir


def _fake_run_game(outcomes: list[object]):
    """Return a main() fake that persists one realistic log per invocation."""

    remaining = iter(outcomes)

    def fake(*, config: dict, config_name: str, game_id: str):
        outcome = next(remaining)
        run_dir = run_log_dir(
            config["log_dir"],
            config["sweep_id"],
            combo_id_from_config(config),
            game_id,
        )
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        if isinstance(outcome, BaseException):
            raise outcome
        results = {
            "game_id": game_id,
            "config_name": config_name,
            "sweep_id": config["sweep_id"],
            "combo_id": combo_id_from_config(config),
            "outcome": str(outcome),
            "result_metadata": config.get("result_metadata", {}),
        }
        (run_dir / "results.json").write_text(
            json.dumps(results),
            encoding="utf-8",
        )
        (run_dir / "summary.log").write_text(
            f"**Game end reason:** {outcome}\n",
            encoding="utf-8",
        )
        return outcome

    return fake


class ExistingSweepLogsTests(unittest.TestCase):
    def test_versioned_schema_requires_matching_sweep_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(
                f"""
log_dir: {_yaml_path(log_root)}
game_params:
  proposal_schema_version: explicit_gives_json_v1
""".lstrip(),
                encoding="utf-8",
            )
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"""
sweep_id: versioned_test
n_runs: 1
combos:
  - id: c000
    base: {_yaml_path(base)}
    metadata: {{}}
""".lstrip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Use a new sweep_id"):
                run_sweep(sweep_path, dry_run=True)

            sweep_path.write_text(
                f"""
sweep_id: versioned_test_v2
n_runs: 1
combos:
  - id: c000
    base: {_yaml_path(base)}
    metadata:
      proposal_schema_version: explicit_gives_json_v1
""".lstrip(),
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_sweep(sweep_path, dry_run=True)

    def test_continue_rejects_mixed_proposal_schema_versions(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()
            run_dir = _write_result_log(
                log_root,
                "existing-run",
                sweep_id="target_sweep",
                combo_id="c000",
            )
            results_path = run_dir / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["game_params"] = {}
            results_path.write_text(json.dumps(results), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Use a new sweep_id"):
                _validate_continue_proposal_schema_versions(
                    [run_dir],
                    "target_sweep",
                    {"c000": "explicit_gives_json_v1"},
                )

            results["game_params"] = {
                "proposal_schema_version": "explicit_gives_json_v1"
            }
            results_path.write_text(json.dumps(results), encoding="utf-8")
            _validate_continue_proposal_schema_versions(
                [run_dir],
                "target_sweep",
                {"c000": "explicit_gives_json_v1"},
            )

    def test_finds_matching_sweep_logs_from_results_and_config(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()

            results_match = log_root / "target_sweep" / "c000" / "results-match"
            results_match.mkdir(parents=True)
            (results_match / "results.json").write_text(
                json.dumps({"sweep_id": "target_sweep"}), encoding="utf-8"
            )

            config_match = log_root / "target_sweep" / "c001" / "config-match"
            config_match.mkdir(parents=True)
            (config_match / "config.yaml").write_text(
                "sweep_id: target_sweep\n", encoding="utf-8"
            )

            other = log_root / "other_sweep" / "c000" / "other"
            other.mkdir(parents=True)
            (other / "results.json").write_text(
                json.dumps({"sweep_id": "other_sweep"}), encoding="utf-8"
            )

            matches = _find_existing_sweep_logs([log_root], "target_sweep")

            self.assertEqual(
                {path.name for path in matches},
                {"results-match", "config-match"},
            )

    def test_existing_sweep_logs_raise_without_delete_flag(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()
            existing = log_root / "target_sweep" / "unknown_combo" / "existing"
            existing.mkdir(parents=True)
            (existing / "config.yaml").write_text(
                "sweep_id: target_sweep\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, "--delete-existing-sweep"):
                _prepare_existing_sweep_logs(
                    "target_sweep",
                    [log_root],
                    delete_existing_sweep=False,
                )

            self.assertTrue(existing.exists())

    def test_continue_flag_allows_existing_logs_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()
            existing = log_root / "target_sweep" / "unknown_combo" / "existing"
            existing.mkdir(parents=True)
            (existing / "config.yaml").write_text(
                "sweep_id: target_sweep\n", encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                matches = _prepare_existing_sweep_logs(
                    "target_sweep",
                    [log_root],
                    delete_existing_sweep=False,
                    continue_sweep=True,
                )

            self.assertEqual(matches, [existing])
            self.assertTrue(existing.exists())

    def test_delete_flag_removes_only_matching_sweep_logs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()

            existing = log_root / "target_sweep" / "unknown_combo" / "existing"
            existing.mkdir(parents=True)
            (existing / "config.yaml").write_text(
                "sweep_id: target_sweep\n", encoding="utf-8"
            )

            other = log_root / "other_sweep" / "unknown_combo" / "other"
            other.mkdir(parents=True)
            (other / "config.yaml").write_text(
                "sweep_id: other_sweep\n", encoding="utf-8"
            )

            with contextlib.redirect_stdout(io.StringIO()):
                _prepare_existing_sweep_logs(
                    "target_sweep",
                    [log_root],
                    delete_existing_sweep=True,
                )

            self.assertFalse(existing.exists())
            self.assertTrue(other.exists())

    def test_counts_successful_game_end_runs_by_combo(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            log_root = Path(tmp) / "logs"
            log_root.mkdir()

            _write_result_log(log_root, "success-c000", sweep_id="target_sweep", combo_id="c000")
            _write_result_log(
                log_root,
                "success-c001-config-name",
                sweep_id="target_sweep",
                combo_id=None,
                config_name="target_sweep__c001__label",
                folder_combo_id="c001",
            )
            _write_result_log(
                log_root,
                "error-c000",
                sweep_id="target_sweep",
                combo_id="c000",
                outcome="error",
                game_end=False,
            )
            _write_result_log(
                log_root,
                "invalid-model-output-c000",
                sweep_id="target_sweep",
                combo_id="c000",
                outcome="invalid_model_output",
            )
            _write_result_log(
                log_root,
                "invalid-c000",
                sweep_id="target_sweep",
                combo_id="c000",
                outcome="invalid",
            )
            _write_result_log(
                log_root,
                "interrupted-c000",
                sweep_id="target_sweep",
                combo_id="c000",
                outcome="interrupted",
            )
            _write_result_log(
                log_root,
                "no-game-end-c000",
                sweep_id="target_sweep",
                combo_id="c000",
                game_end=False,
            )
            _write_result_log(log_root, "other-sweep", sweep_id="other_sweep", combo_id="c000")

            matches = _find_existing_sweep_logs([log_root], "target_sweep")
            counts = _count_successful_game_end_runs_by_combo(matches, "target_sweep")

            self.assertEqual(counts["c000"], 1)
            self.assertEqual(counts["c001"], 1)
            self.assertEqual(counts["missing"], 0)

    def test_continue_dry_run_schedules_only_missing_combo_runs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"""
sweep_id: resume_test
n_runs: 2
combos:
  - id: c000
    n_runs: 3
    combo_label: first
    label: Combo 0
    base: {_yaml_path(base)}
    overrides: {{}}
  - id: c001
    n_runs: 1
    combo_label: second
    label: Combo 1
    base: {_yaml_path(base)}
    overrides: {{}}
""".lstrip(),
                encoding="utf-8",
            )
            _write_result_log(log_root, "success-c000", sweep_id="resume_test", combo_id="c000")
            existing_failed = _write_result_log(
                log_root,
                "existing-invalid-c000",
                sweep_id="resume_test",
                combo_id="c000",
                outcome="invalid",
            )
            _write_result_log(
                log_root,
                "archived-failed-attempt",
                sweep_id="resume_test_failed",
                combo_id="c000",
                outcome="invalid_model_output",
            )
            for i in range(3):
                _write_result_log(
                    log_root,
                    f"success-c001-{i}",
                    sweep_id="resume_test",
                    combo_id="c001",
                )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_sweep(sweep_path, dry_run=True, continue_sweep=True)

            text = output.getvalue()
            run_lines = [
                line.strip()
                for line in text.splitlines()
                if line.startswith("  resume_test__")
            ]
            self.assertIn("scheduling 2 missing run(s)", text)
            self.assertIn("Would archive 1 existing failed log record(s)", text)
            self.assertIn("Runs per combo: c000=3, c001=1", text)
            self.assertIn("c001=3>1", text)
            self.assertEqual(
                run_lines,
                [
                    "resume_test__c000__first [Combo 0]",
                    "resume_test__c000__first [Combo 0]",
                ],
            )
            self.assertTrue(existing_failed.exists())

    def test_continue_archives_all_existing_retryable_outcomes(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: archive_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )
            _write_result_log(
                log_root,
                "deal",
                sweep_id="archive_test",
                combo_id="c000",
            )
            _write_result_log(
                log_root,
                "error",
                sweep_id="archive_test",
                combo_id="c000",
                outcome="error",
            )
            _write_result_log(
                log_root,
                "invalid-output",
                sweep_id="archive_test",
                combo_id="c000",
                outcome=" Invalid_Model_Output ",
            )
            _write_result_log(
                log_root,
                "invalid",
                sweep_id="archive_test",
                combo_id="c000",
                outcome="INVALID",
            )
            _write_result_log(
                log_root,
                "interrupted",
                sweep_id="archive_test",
                combo_id="c000",
                outcome=" Interrupted ",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_sweep(
                    sweep_path,
                    continue_sweep=True,
                )

            self.assertIn("Archived 4 existing failed log record(s)", output.getvalue())
            normal_paths = list(iter_results_paths(log_root, sweep_id="archive_test"))
            failed_paths = list(
                iter_results_paths(log_root, sweep_id="archive_test_failed")
            )
            self.assertEqual([path.parent.name for path in normal_paths], ["deal"])
            self.assertEqual(len(failed_paths), 4)
            for path in failed_paths:
                result = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(result["sweep_id"], "archive_test_failed")
                self.assertEqual(
                    result["result_metadata"]["original_sweep_id"],
                    "archive_test",
                )
                self.assertIn(
                    "archive_test_failed",
                    result["result_metadata"]["failed_log_dir"],
                )

    def test_continue_archives_existing_log_without_results(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: archive_missing_results\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )
            _write_result_log(
                log_root,
                "deal",
                sweep_id="archive_missing_results",
                combo_id="c000",
            )
            partial = (
                log_root
                / "archive_missing_results"
                / "c000"
                / "partial-run"
            )
            partial.mkdir(parents=True)
            (partial / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "sweep_id": "archive_missing_results",
                        "combo_id": "c000",
                        "result_metadata": {
                            "combo_id": "c000",
                            "sweep_attempt_number": 1,
                            "sweep_retry_number": 0,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_sweep(sweep_path, continue_sweep=True)

            archived = (
                log_root
                / "archive_missing_results_failed"
                / "c000"
                / "partial-run"
            )
            self.assertIn("Archived 1 existing failed log record(s)", output.getvalue())
            self.assertFalse(partial.exists())
            self.assertTrue(archived.is_dir())
            self.assertFalse((archived / "results.json").exists())
            config = yaml.safe_load((archived / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["sweep_id"], "archive_missing_results_failed")
            self.assertEqual(
                config["result_metadata"]["sweep_attempt_failure"],
                "outcome=interrupted (missing results.json)",
            )

    def test_dry_run_plans_existing_failure_archive_without_mutating(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: archive_dry_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )
            source = _write_result_log(
                log_root,
                "invalid-output",
                sweep_id="archive_dry_test",
                combo_id="c000",
                outcome="invalid_model_output",
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_sweep(sweep_path, dry_run=True)

            text = output.getvalue()
            self.assertIn("Would archive 1 existing failed log record(s)", text)
            self.assertIn("archive_dry_test_failed", text)
            self.assertIn("[dry-run] Would run", text)
            self.assertTrue(source.exists())
            self.assertFalse((log_root / "archive_dry_test_failed").exists())

    def test_existing_failure_archive_detects_destination_collision_in_dry_run(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: collision_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )
            source = _write_result_log(
                log_root,
                "same-run",
                sweep_id="collision_test",
                combo_id="c000",
                outcome="error",
            )
            _write_result_log(
                log_root,
                "same-run",
                sweep_id="collision_test_failed",
                combo_id="c000",
                outcome="error",
            )

            with self.assertRaisesRegex(FileExistsError, "destination already exists"):
                run_sweep(
                    sweep_path,
                    dry_run=True,
                )
            self.assertTrue(source.exists())

    def test_failed_suffix_is_reserved_for_archive_companions(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(tmp_path / 'logs')}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: 'reserved_failed '\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reserved for failed logs"):
                run_sweep(sweep_path, dry_run=True)

    def test_returned_error_outcome_is_reported_as_failed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"""
sweep_id: returned_error_test
n_runs: 1
base: {_yaml_path(base)}
combos:
  - combo_id: c000
""".lstrip(),
                encoding="utf-8",
            )

            output = io.StringIO()
            fake_main = _fake_run_game(
                ["error", "invalid_model_output", "error"]
            )
            with patch("negsim.main.main", side_effect=fake_main) as mocked_main:
                with self.assertLogs("negsim.run_sweep", level="WARNING") as captured:
                    with contextlib.redirect_stdout(output):
                        with self.assertRaisesRegex(RuntimeError, "1 game\\(s\\) failed"):
                            run_sweep(sweep_path, workers=1)

            text = output.getvalue()
            warning_text = "\n".join(captured.output)
            self.assertEqual(mocked_main.call_count, 3)
            self.assertIn("ERROR - outcome=error", text)
            self.assertIn("Done. 0 completed, 1 failed.", text)
            self.assertIn("starting retry 1/2", warning_text)
            self.assertIn("starting retry 2/2", warning_text)
            self.assertIn("retry 2/2 failed", warning_text)
            self.assertIn("no retries remain", warning_text)

            self.assertEqual(
                list(iter_results_paths(log_root, sweep_id="returned_error_test")),
                [],
            )
            failed_paths = list(
                iter_results_paths(
                    log_root,
                    sweep_id="returned_error_test_failed",
                )
            )
            self.assertEqual(len(failed_paths), 3)
            failed_results = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in failed_paths
            ]
            self.assertEqual(
                {result["outcome"] for result in failed_results},
                {"error", "invalid_model_output"},
            )
            self.assertEqual(
                {result["sweep_id"] for result in failed_results},
                {"returned_error_test_failed"},
            )
            self.assertEqual(
                {
                    result["result_metadata"]["sweep_retry_number"]
                    for result in failed_results
                },
                {0, 1, 2},
            )
            self.assertEqual(
                {
                    result["result_metadata"]["original_sweep_id"]
                    for result in failed_results
                },
                {"returned_error_test"},
            )

    def test_second_retry_can_succeed_and_keeps_failed_attempt_logs_separate(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: returned_invalid_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )

            output = io.StringIO()
            fake_main = _fake_run_game(["invalid", "interrupted", "deal"])
            with patch("negsim.main.main", side_effect=fake_main) as mocked_main:
                with self.assertLogs("negsim.run_sweep", level="WARNING") as captured:
                    with contextlib.redirect_stdout(output):
                        run_sweep(sweep_path, workers=1)

            text = output.getvalue()
            warning_text = "\n".join(captured.output)
            self.assertEqual(mocked_main.call_count, 3)
            self.assertIn(": deal", text)
            self.assertIn("Done. 1 completed, 0 failed.", text)
            self.assertIn("starting retry 1/2", warning_text)
            self.assertIn("starting retry 2/2", warning_text)
            self.assertNotIn("no retries remain", warning_text)

            successful_paths = list(
                iter_results_paths(log_root, sweep_id="returned_invalid_test")
            )
            failed_paths = list(
                iter_results_paths(
                    log_root,
                    sweep_id="returned_invalid_test_failed",
                )
            )
            self.assertEqual(len(successful_paths), 1)
            self.assertEqual(len(failed_paths), 2)
            successful = json.loads(
                successful_paths[0].read_text(encoding="utf-8")
            )
            self.assertEqual(successful["outcome"], "deal")
            self.assertEqual(successful["sweep_id"], "returned_invalid_test")
            self.assertEqual(
                successful["result_metadata"]["sweep_retry_number"],
                2,
            )
            self.assertEqual(
                {
                    json.loads(path.read_text(encoding="utf-8"))["outcome"]
                    for path in failed_paths
                },
                {"invalid", "interrupted"},
            )
            logical_run_ids = {
                successful["result_metadata"]["sweep_logical_run_id"],
                *(
                    json.loads(path.read_text(encoding="utf-8"))[
                        "result_metadata"
                    ]["sweep_logical_run_id"]
                    for path in failed_paths
                ),
            }
            self.assertEqual(len(logical_run_ids), 1)

    def test_exception_retries_the_entire_run_and_archives_partial_log(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: exception_retry_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )

            fake_main = _fake_run_game([RuntimeError("boom"), "deal"])
            with patch("negsim.main.main", side_effect=fake_main) as mocked_main:
                with self.assertLogs("negsim.run_sweep", level="WARNING") as captured:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_sweep(sweep_path, workers=1)

            self.assertEqual(mocked_main.call_count, 2)
            warning_text = "\n".join(captured.output)
            self.assertIn("RuntimeError: boom", warning_text)
            self.assertIn("starting retry 1/2", warning_text)

            successful_paths = list(
                iter_results_paths(log_root, sweep_id="exception_retry_test")
            )
            self.assertEqual(len(successful_paths), 1)
            failed_sweep_dir = log_root / "exception_retry_test_failed"
            partial_dirs = [
                path
                for combo_dir in failed_sweep_dir.iterdir()
                for path in combo_dir.iterdir()
                if path.is_dir()
            ]
            self.assertEqual(len(partial_dirs), 1)
            failed_results = json.loads(
                (partial_dirs[0] / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_results["outcome"], "error")
            self.assertEqual(failed_results["sweep_id"], "exception_retry_test_failed")
            self.assertEqual(failed_results["error"]["type"], "RuntimeError")
            failed_config = yaml.safe_load(
                (partial_dirs[0] / "config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_config["sweep_id"], "exception_retry_test_failed")
            self.assertEqual(
                failed_config["result_metadata"]["original_sweep_id"],
                "exception_retry_test",
            )

    def test_worker_keyboard_interrupt_is_archived_and_retried(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            log_root = tmp_path / "logs"
            log_root.mkdir()
            base = tmp_path / "base.yaml"
            base.write_text(f"log_dir: {_yaml_path(log_root)}\n", encoding="utf-8")
            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"sweep_id: interrupted_retry_test\nn_runs: 1\nbase: {_yaml_path(base)}\ncombos:\n  - combo_id: c000\n",
                encoding="utf-8",
            )

            fake_main = _fake_run_game([KeyboardInterrupt(), "deal"])
            with patch("negsim.main.main", side_effect=fake_main) as mocked_main:
                with self.assertLogs("negsim.run_sweep", level="WARNING") as captured:
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_sweep(sweep_path, workers=1)

            self.assertEqual(mocked_main.call_count, 2)
            self.assertIn("outcome=interrupted", "\n".join(captured.output))
            successful_paths = list(
                iter_results_paths(log_root, sweep_id="interrupted_retry_test")
            )
            interrupted_paths = list(
                iter_results_paths(
                    log_root,
                    sweep_id="interrupted_retry_test_failed",
                )
            )
            self.assertEqual(len(successful_paths), 1)
            self.assertEqual(len(interrupted_paths), 1)
            interrupted = json.loads(
                interrupted_paths[0].read_text(encoding="utf-8")
            )
            self.assertEqual(interrupted["outcome"], "interrupted")
            self.assertEqual(interrupted["error"]["type"], "KeyboardInterrupt")


if __name__ == "__main__":
    unittest.main()
