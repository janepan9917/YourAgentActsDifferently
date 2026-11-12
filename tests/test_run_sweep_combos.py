import tempfile
import unittest
from pathlib import Path

from negsim.run_sweep import load_sweep


REPO_ROOT = Path(__file__).resolve().parents[1]


def _yaml_path(path: Path) -> str:
    return str(path).replace("\\", "/")


class SweepComboLoadingTests(unittest.TestCase):
    def test_explicit_combos_can_use_different_base_configs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            base_a = tmp_path / "base_a.yaml"
            base_b = tmp_path / "base_b.yaml"
            base_a.write_text("marker: a\nnested:\n  value: from-a\n", encoding="utf-8")
            base_b.write_text("marker: b\nnested:\n  value: from-b\n", encoding="utf-8")

            sweep_path = tmp_path / "sweep.yaml"
            sweep_path.write_text(
                f"""
sweep_id: explicit_combo_test
n_runs: 3
combos:
  - id: combo_a
    n_runs: 5
    base: {_yaml_path(base_a)}
    base_id: base-a
    base_label: Base A
    label: Combo A
    metadata:
      source_sweep_id: source-a
    overrides:
      nested.value: overridden-a
  - combo_id: combo_b
    combo_label: custom_b
    base: {_yaml_path(base_b)}
    label: Combo B
    overrides: {{}}
""".lstrip(),
                encoding="utf-8",
            )

            sweep_id, combos, n_runs = load_sweep(sweep_path)

            self.assertEqual(sweep_id, "explicit_combo_test")
            self.assertEqual(n_runs, 3)
            self.assertEqual([combo.combo_id for combo in combos], ["combo_a", "combo_b"])
            self.assertEqual([combo.n_runs for combo in combos], [5, 3])
            self.assertEqual([combo.base_config["marker"] for combo in combos], ["a", "b"])
            self.assertEqual(combos[0].overrides, {"nested.value": "overridden-a"})
            self.assertEqual(combos[0].base_id, "base-a")
            self.assertEqual(combos[0].base_label, "Base A")
            self.assertEqual(combos[0].metadata["source_sweep_id"], "source-a")
            self.assertEqual(combos[1].combo_label, "custom_b")

    def test_unknown_n_runs_by_combo_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp_path = Path(tmp)
            base = tmp_path / "base.yaml"
            base.write_text("marker: base\n", encoding="utf-8")

            sweep_path = tmp_path / "bad_sweep.yaml"
            sweep_path.write_text(
                f"""
sweep_id: bad_n_runs_key
n_runs: 2
n_runs_by_combo:
  missing: 4
combos:
  - id: c000
    base: {_yaml_path(base)}
    overrides:
      nested.value: one
""".lstrip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown combo_id"):
                load_sweep(sweep_path)


if __name__ == "__main__":
    unittest.main()
