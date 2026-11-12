"""Figure 5: F1 of the candidate judges at the turn and trajectory level, misaligned behaviors (top) and
Preference Hardening (bottom).

Builds both panels from the human and judge labels using the two panel scripts
(figure5_misaligned_behavior.py, figure5_preference_hardening.py). Preference Hardening is not MB.
  Turn level      : every manually annotated turn x behavior (MB: 229 turns x 6 categories; PH: 229 turns); a judge
                    label counts for a turn when the judge lists that turn as an occurrence.
  Trajectory level: OR over the annotated turns of each trajectory x behavior, as in the two panel scripts, whose
                    metrics this script must match.
The figure shows point estimates; the metrics CSV also records a 95% percentile bootstrap CI for each F1, resampling the 20 trajectories
(all their turns move together), 2,000 draws, fixed seed.

Run by `python -m analysis all` (step figure5), after the two panel scripts.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "judge"))  # MB taxonomy and label parsing
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))  # paper_style
import paper_style as ps
import figure5_misaligned_behavior as mb
import figure5_preference_hardening as ph

OUT = ps.DATA
STEM = "appendix_judge_validation_combined"
PAPER_PDF = "appendix/judge_validation_combined"
MEASURES = ["precision", "recall", "f1"]
METRIC_COLUMNS = ["tp", "fp", "fn", "tn", "n", "positives", *MEASURES]
LEVELS = ["Turn level", "Trajectory level"]
N_BOOT, SEED = 2000, 0


def verify_saved_metrics(current: pd.DataFrame, path: Path, *, mb_panel: bool) -> None:
    """The trajectory-level values must equal the standalone panels' saved metrics."""
    saved = pd.read_csv(path)
    if mb_panel:
        saved = saved[saved.category.eq(mb.MICRO)]
    saved = saved.set_index("method").loc[list(mb.METHODS), METRIC_COLUMNS]
    np.testing.assert_allclose(current.loc[list(mb.METHODS), METRIC_COLUMNS].to_numpy(float),
                               saved.to_numpy(float), rtol=1e-12, atol=1e-12)


def counts(manual: np.ndarray, auto: np.ndarray) -> dict:
    tp, fp = int((manual & auto).sum()), int((~manual & auto).sum())
    fn, tn = int((manual & ~auto).sum()), int((~manual & ~auto).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=tp + fp + fn + tn, positives=tp + fn,
                precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0)


def level_frames(aligned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Turn-level rows as aligned; trajectory-level rows = OR over each trajectory's annotated turns."""
    turn = aligned[["method", "game_id", "behavior_id", "manual_present", "auto_present"]].copy()
    traj = turn.groupby(["method", "game_id", "behavior_id"], as_index=False)[["manual_present", "auto_present"]].any()
    return {"Turn level": turn, "Trajectory level": traj}


def scores(aligned: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """F1 (and counts) per method and level, with a trajectory bootstrap CI for F1."""
    games = np.array(sorted(aligned.game_id.unique()))
    draws = [rng.choice(games, len(games)) for _ in range(N_BOOT)]
    rows = []
    for level, frame in level_frames(aligned).items():
        for method in mb.METHODS:
            d = frame[frame.method.eq(method)]
            by_game = {g: (x.manual_present.to_numpy(bool), x.auto_present.to_numpy(bool)) for g, x in d.groupby("game_id")}
            point = counts(d.manual_present.to_numpy(bool), d.auto_present.to_numpy(bool))
            boot = [counts(np.concatenate([by_game[g][0] for g in draw]), np.concatenate([by_game[g][1] for g in draw]))["f1"]
                    for draw in draws]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(method=method, judge=mb.METHODS[method], level=level, **point, f1_ci_low=lo, f1_ci_high=hi))
    return pd.DataFrame(rows)


def build_figure(panels: list[tuple[str, pd.DataFrame]]):
    ps.apply()
    fig, axes = plt.subplots(2, 1, figsize=(ps.COLUMN_WIDTH, 3.6), sharex=True, layout="constrained")
    fig.get_layout_engine().set(hspace=.06)
    x, width = np.arange(len(LEVELS)), .2
    for ax, (title, data) in zip(axes, panels):
        for i, (method, color) in enumerate(zip(mb.METHODS, ph.COLORS)):
            d = data[data.method.eq(method)].set_index("level").loc[LEVELS]
            positions = x + (i - 1.5) * width
            ax.bar(positions, d.f1, width * .92, color=color, label=mb.METHODS[method], zorder=3)
            for xx, yy in zip(positions, d.f1):
                if yy == 0:
                    ax.annotate("0", (xx, 0), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color=ps.MUTED)
        ax.set_title(title, pad=5)
        ax.set_xticks(x, LEVELS)
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, .25))
        ps.ygrid(ax)
    axes[0].tick_params(axis="x", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [0, 2, 1, 3]  # legend fills column-wise; this reads Mini, Luna / Terra, Sol row by row
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="outside upper center", ncol=2)
    return fig


def main() -> None:
    mb_aligned, mb_result = mb.load_data(mb.SOURCE)
    ph_aligned, _, ph_result = ph.load_data()
    if set(mb_aligned.game_id) != set(ph_aligned.game_id):
        raise ValueError("Panels must use the same validation trajectories")
    if mb.METHODS != ph.METHODS:
        raise ValueError("Judge labels/order differ between panels")
    mb_aligned = mb_aligned[mb_aligned.method.isin(mb.METHODS)]
    rng = np.random.default_rng(SEED)
    left, right = scores(mb_aligned, rng), scores(ph_aligned, rng)
    for data, path, is_mb in ((left, OUT / f"{mb.STEM}_metrics.csv", True), (right, OUT / f"{ph.STEM}_metrics.csv", False)):
        verify_saved_metrics(data[data.level.eq("Trajectory level")].set_index("method"), path, mb_panel=is_mb)
    combined = pd.concat([left.assign(panel="a", behavior_group="Six MB categories (micro)"),
                          right.assign(panel="b", behavior_group="Preference Hardening (not MB)")], ignore_index=True)
    combined = combined[["panel", "behavior_group", "level", "method", "judge", *METRIC_COLUMNS, "f1_ci_low", "f1_ci_high"]]
    OUT.mkdir(parents=True, exist_ok=True)
    fig = build_figure([("(a) Misaligned behaviors", left), ("(b) Preference Hardening", right)])
    ps.save(fig, PAPER_PDF, OUT / f"{STEM}.png")
    plt.close(fig)
    combined.to_csv(OUT / f"{STEM}_metrics.csv", index=False, encoding="utf-8-sig")
    print(combined[["panel", "level", "judge", "n", "positives", *MEASURES, "f1_ci_low", "f1_ci_high"]].round(3).to_string(index=False))
    print("Trajectory-level metrics match the standalone panels.")


if __name__ == "__main__":
    main()
