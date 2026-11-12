"""Derive the per-episode inputs read by the paper's figure/table code (analysis/paper).

Inputs are derived from ``data/processed/agent_runs.csv.gz`` and ``data/ablations``.
Files are written to the work folder (``config.work_dir``).
"""
from __future__ import annotations

import shutil

import pandas as pd

from .config import ALL_LABELS, GAMES, PREFERENCE_HARDENING, data_dir, processed_dir, work_dir
from .paper._shared import welch_mean_difference_ci, welch_pvalue

FORMAL = ["no_user", "single_user", "two_users"]
ABLATION_FILES = ["user_ablations_payout_by_model.csv", "user_ablations_mb_episodes.csv"]


def load_agent_runs() -> pd.DataFrame:
    runs = pd.read_csv(processed_dir() / "agent_runs.csv.gz", low_memory=False)
    for col in ALL_LABELS + ["selected"]:
        runs[col] = runs[col].astype(bool)
    return runs


def payout_inputs(runs: pd.DataFrame) -> pd.DataFrame:
    """With User minus No User payout of the user-side party, per game, model and party,
    as a percentage of the scale S_i (appendix: payout calculation)."""
    rows = []
    for (game, model), d in runs[runs.condition.isin(["no_user", "single_user"])].groupby(["game", "model"]):
        base = d[d.condition == "no_user"]
        base_means = base.groupby("role").payout_ungated.mean()
        for i in (1, 2):
            role = f"agent_{i}"
            single = d[d.direction == f"user_on_agent_{i}"]
            scale = max(base_means.abs().max(), single.groupby("role").payout_ungated.mean().abs().max())
            without = base[base.role == role].payout_ungated.to_numpy()
            with_user = single[single.role == role].payout_ungated.to_numpy()
            diff, se, low, high = welch_mean_difference_ci(without, with_user)
            rows.append(dict(game=game, model=model, user_role=f"user_{i}", n_single_user=len(with_user),
                             n_no_user=len(without), scale=scale, raw_diff=diff, mean=100 * diff / scale,
                             se=100 * se / scale, ci_low=100 * low / scale, ci_high=100 * high / scale,
                             pvalue=welch_pvalue(without, with_user)))
    return pd.DataFrame(rows)


def write_inputs() -> None:
    out = work_dir()
    out.mkdir(parents=True, exist_ok=True)
    runs = load_agent_runs()
    formal = runs[runs.condition.isin(FORMAL) & runs.selected]
    keys = ["game", "model", "condition", "role", "game_id"]

    payout_inputs(runs).to_csv(out / "payout_change_by_model.csv", index=False)
    strategy = ["open_extremity", "mean_position", "final_position", "total_concession", "mean_concession",
                "concession_freq", "cushion_given_up", "unilateral_conc", "n_own_offers"]
    formal[keys + strategy].to_csv(out / "aggressiveness_by_condition_episodes.csv", index=False)
    hardening = formal[keys + [PREFERENCE_HARDENING]].rename(columns={PREFERENCE_HARDENING: "present"})
    hardening.to_csv(out / "preference_hardening_episodes.csv", index=False)
    long = formal.melt(id_vars=keys, value_vars=sorted(ALL_LABELS), var_name="behavior_id", value_name="present")
    long.to_csv(out / "behavior_episodes.csv", index=False)
    for name in ABLATION_FILES:
        shutil.copyfile(data_dir() / "ablations" / name, out / name)
    assert set(formal.game) == set(GAMES)
    print(f"inputs written to {out}")


if __name__ == "__main__":
    write_inputs()
