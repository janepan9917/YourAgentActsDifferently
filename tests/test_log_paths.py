import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from negsim.log_paths import (
    canonical_logs_root,
    combo_folder_name,
    iter_results_paths,
    run_log_dir,
    sweep_folder_name,
)
from negsim.main import setup_log_dir


REPO_ROOT = Path(__file__).resolve().parents[1]


class LogPathTests(unittest.TestCase):
    def test_unsafe_group_ids_have_stable_non_colliding_names(self) -> None:
        self.assertNotEqual(sweep_folder_name("a:b"), sweep_folder_name("a?b"))
        self.assertNotEqual(combo_folder_name("a:b"), combo_folder_name("a?b"))
        self.assertEqual(sweep_folder_name("plain_sweep"), "plain_sweep")

    def test_logs_root_defaults_to_logs(self) -> None:
        self.assertEqual(canonical_logs_root(None), Path("logs"))
        self.assertEqual(canonical_logs_root("logs/single_user"), Path("logs/single_user"))

    def test_sweep_lookup_reads_only_the_requested_bounded_depth(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp) / "logs"
            wanted = run_log_dir(root, "wanted", "c000", "run-1")
            other = run_log_dir(root, "other", "c000", "run-2")
            wanted.mkdir(parents=True)
            other.mkdir(parents=True)
            (wanted / "results.json").write_text(
                json.dumps({"sweep_id": "wanted"}), encoding="utf-8"
            )
            (other / "results.json").write_text(
                json.dumps({"sweep_id": "other"}), encoding="utf-8"
            )

            with patch.object(Path, "rglob", side_effect=AssertionError("recursive scan")):
                paths = list(iter_results_paths(root, sweep_id="wanted"))

            self.assertEqual(paths, [wanted / "results.json"])

    def test_setup_log_dir_writes_directly_to_sweep_and_combo(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp) / "logs"
            config = {
                "log_dir": str(root),
                "sweep_id": "sweep-a",
                "result_metadata": {"combo_id": "c007"},
            }

            with patch("negsim.main.make_game_id", return_value="run-123"):
                actual, game_id = setup_log_dir(config)

            expected = root / "sweep-a" / "c007" / "run-123"
            self.assertEqual(game_id, "run-123")
            self.assertEqual(actual, expected)
            self.assertTrue((expected / "config.yaml").exists())
            self.assertTrue((expected / "agent").is_dir())
            self.assertTrue((expected / "user").is_dir())

    def test_setup_log_dir_accepts_a_preallocated_game_id(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp) / "logs"
            config = {
                "log_dir": str(root),
                "sweep_id": "sweep-a",
                "result_metadata": {"combo_id": "c007"},
            }

            with patch(
                "negsim.main.make_game_id",
                side_effect=AssertionError("must not generate another game id"),
            ):
                actual, game_id = setup_log_dir(config, game_id="preallocated-123")

            expected = root / "sweep-a" / "c007" / "preallocated-123"
            self.assertEqual(game_id, "preallocated-123")
            self.assertEqual(actual, expected)
            self.assertTrue((expected / "config.yaml").is_file())

    def test_setup_log_dir_rejects_unsafe_or_reused_game_id(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            root = Path(tmp) / "logs"
            config = {
                "log_dir": str(root),
                "sweep_id": "sweep-a",
                "result_metadata": {"combo_id": "c007"},
            }

            with self.assertRaisesRegex(ValueError, "safe path component"):
                setup_log_dir(config.copy(), game_id="../outside")

            setup_log_dir(config.copy(), game_id="preallocated-123")
            with self.assertRaises(FileExistsError):
                setup_log_dir(config.copy(), game_id="preallocated-123")


if __name__ == "__main__":
    unittest.main()
