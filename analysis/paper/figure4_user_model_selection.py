"""Figure 4: judge scores of candidate user models in Dual User Ultimatum.

Reads the runs in data/raw/user_model_selection and the user prompt-following
scores in data/labels/user_prompt_following (judges: GPT-5.6 Sol and Claude Opus 5).
Writes outputs/figures/appendix/user_model_selection.pdf and the score tables in
the work folder. Run by `python -m analysis all` (step figure4).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402
import user_prompt_following_scores as source  # noqa: E402

from analysis.config import data_dir  # noqa: E402

OUT = ps.DATA
STEM = "appendix_user_model_selection"
MODELS = ("gpt-5.6-terra", "gpt-5.6-sol", "claude-sonnet-5", "claude-opus-5")
# GPT family in the accent hue (selected Terra darkest), Claude family in greys; markers repeat the split for CVD.
COLORS = (ps.ACCENT, "#F2A07B", "#A3A29B", "#6E6D68")
MARKERS = ("o", "o", "s", "s")
MODEL_LABELS = dict(zip(MODELS, ("GPT-5.6 Terra", "GPT-5.6 Sol", "Claude Sonnet 5", "Claude Opus 5")))
JUDGE_LABELS = {"gpt56sol_medium": "GPT-5.6 Sol judge", "claudeopus5_default": "Claude Opus 5 judge"}
DIMENSIONS = (("overall", "Overall"), ("UF_RULE_FOLLOWING", "Adherence to\nrequirements"),
              ("UF_TERMS_UNDERSTANDING", "Terms\nunderstanding"))
PAPER_PDF = "appendix/user_model_selection"
YLIM = (4.5, 5.0)


def validate_summary(summary: pd.DataFrame) -> None:
    required = {"user_model", "judge_id", "dimension_id", "n_runs", "n_agent_models_pooled",
                "mean_score", "ci95_low", "ci95_high"}
    if not required <= set(summary.columns):
        raise ValueError(f"Missing summary columns: {required - set(summary.columns)}")
    keys = ["user_model", "judge_id", "dimension_id"]
    expected = {(m, j, d) for m in MODELS for j in JUDGE_LABELS for d, _ in DIMENSIONS}
    if summary.duplicated(keys).any() or set(summary[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Expected exactly four user models, two judges and three plotted dimensions")
    numeric = summary[["mean_score", "ci95_low", "ci95_high"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not ((numeric[:, 1] <= numeric[:, 0]) &
            (numeric[:, 0] <= numeric[:, 2]) & (numeric[:, 1] >= 4) & (numeric[:, 2] <= 5)).all():
        raise ValueError("Scores/intervals outside the figure's 4-5 axis")
    if not summary.n_runs.eq(150).all() or not summary.n_agent_models_pooled.eq(3).all():
        raise ValueError("Each user-model/judge cell must pool 150 runs across three agent models")


def rebuild_data(logs_root: Path, eval_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roles = source.load_role_scores(logs_root=logs_root, eval_root=eval_root, expected_runs=50)
    runs = source.build_run_scores(roles)
    summary = source.build_summary(runs)
    if roles.game_id.nunique() != 600 or len(roles) != 4800 or len(runs) != 1200:
        raise ValueError("Expected 600 trajectories, 4,800 role/criterion scores and 1,200 run/judge rows")
    validate_summary(summary)
    return roles, runs, summary


def build_figure(summary: pd.DataFrame):
    """Two judge panels; per dimension, one point (mean) with a 95% CI per user model."""
    validate_summary(summary)
    if summary.ci95_low.min() < YLIM[0]:
        raise ValueError("A confidence interval falls below the plotted range")
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(ps.TEXT_WIDTH, 2.1), sharey=True, layout="constrained")
    x, step = np.arange(len(DIMENSIONS)), .16
    for ax, judge in zip(axes, JUDGE_LABELS):
        for m, (model, color, marker) in enumerate(zip(MODELS, COLORS, MARKERS)):
            rows = summary.loc[summary.user_model.eq(model) & summary.judge_id.eq(judge)].set_index("dimension_id")
            rows = rows.loc[[dim for dim, _ in DIMENSIONS]]
            values = rows.mean_score.to_numpy(float)
            ax.errorbar(x+(m-1.5)*step, values, yerr=np.vstack([values-rows.ci95_low, rows.ci95_high-values]),
                        fmt=marker, ms=4, color=color, mec=color, ecolor=color, elinewidth=.9, capsize=1.8,
                        capthick=.9, label=MODEL_LABELS[model], zorder=3)
        ax.set_xticks(x, [title for _, title in DIMENSIONS])
        ax.set_xlim(-.5, len(DIMENSIONS)-.5)
        ax.set_ylim(YLIM[0], YLIM[1]+.015)  # headroom so points and caps at 5.0 are not clipped
        ps.ygrid(ax)
        ax.set_yticks(np.arange(YLIM[0], YLIM[1]+1e-9, .1))
        ax.set_title(JUDGE_LABELS[judge])
    axes[0].set_ylabel("Mean score (1–5)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4)
    return fig


def main() -> None:
    logs_root = data_dir() / "raw" / "user_model_selection"
    eval_root = data_dir() / "labels" / "user_prompt_following"
    out = OUT.resolve()
    if out.is_relative_to(logs_root.resolve()) or out.is_relative_to(eval_root.resolve()):
        raise ValueError("Output directory must not be inside the logs or judge scores")
    roles, runs, summary = rebuild_data(logs_root, eval_root)
    fig = build_figure(summary)
    out.mkdir(parents=True, exist_ok=True)
    roles.to_csv(out / f"{STEM}_role_scores.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(out / f"{STEM}_run_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / f"{STEM}_score_summary.csv", index=False, encoding="utf-8-sig")
    pdf, png = ps.PAPER_FIGS / f"{PAPER_PDF}.pdf", out / f"{STEM}.png"
    ps.save(fig, PAPER_PDF, png)
    plt.close(fig)
    print(json.dumps(dict(figure=str(pdf), summary_cells=len(summary)), ensure_ascii=False))


if __name__ == "__main__":
    main()
