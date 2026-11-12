"""Figure 6: Pearson r between an agent's own misaligned behavior and its own payout.

For each game and setting, r is the point-biserial correlation between one behavior label
and the (ungated) payout over agent-runs (No User: both agents; With User: the agent
whose party has the user). Per-game z-transformed correlations are pooled with a
random-effects model. Reads data/processed/agent_runs.csv.gz.
"""
from __future__ import annotations

import json

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

from analysis.inputs import load_agent_runs
from analysis.config import out_dir, work_dir
from analysis.paper import paper_style as ps

OUTPUT = out_dir() / "figures" / "misbehavior_correlations.pdf"
VALUES = work_dir() / "figure6_mb_payout_correlations.json"


CONDITIONS = ["no_user", "single_user"]


BEHAVIORS = {
    "MB_OFFER_STATE_MISREPORTING": "Misreporting",
    "MB_HALLUCINATED_ACTION": "Hallucinated action",
    "MB_UNSUPPORTED_URGENCY": "User-facing\nunsupported urgency",
    "MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING": "Misunderstanding",
    "MB_OTHER_AGENT_UNSUPPORTED_URGENCY": "Counterparty-facing\nunsupported urgency",
    "MB_EXTRA_PROTOCOL_CONTRACTING": "Out-of-scope terms",
}


def meta(z_values, errors):
    z_values, errors = np.asarray(z_values), np.asarray(errors)
    variances = errors**2
    w = 1 / variances
    q = np.sum(w * (z_values - np.sum(w*z_values)/w.sum())**2)
    c = w.sum() - np.sum(w*w) / w.sum()
    tau = max(0, (q - len(w) + 1) / c) if c > 0 else 0
    w = 1 / (variances + tau)
    mean, se = np.sum(w*z_values)/w.sum(), np.sqrt(1/w.sum())
    df = max(1, len(w)-1)
    bound = stats.t.ppf(.975, df) * se
    return dict(r=float(np.tanh(mean)), ci_low=float(np.tanh(mean-bound)),
                ci_high=float(np.tanh(mean+bound)), p=float(2*stats.t.sf(abs(mean/se), df)), k=len(w))


def correlations(raw, payout):
    per_game = []
    for (game, condition), group in raw.groupby(["game", "condition"]):
        for behavior in BEHAVIORS:
            if group[behavior].nunique() < 2 or group[payout].nunique() < 2:
                continue
            r = float(stats.pearsonr(group[behavior].astype(float), group[payout]).statistic)
            if not np.isfinite(r) or abs(r) >= 1:
                continue
            n = len(group)
            per_game.append(dict(game=game, condition=condition, behavior=behavior,
                                 r=r, z=float(np.arctanh(r)), se=float(1/np.sqrt(n-3)), n=n))
    pooled = []
    for (condition, behavior), group in pd.DataFrame(per_game).groupby(["condition", "behavior"]):
        pooled.append(dict(condition=condition, behavior=behavior, payout=payout,
                           **meta(group.z, group.se), n=int(group.n.sum())))
    return per_game, pooled


def plot(pooled):
    ps.apply()
    matrix = np.full((6, 2), np.nan)
    pmatrix = np.full((6, 2), np.nan)
    indexed = {(r["behavior"], r["condition"]): r for r in pooled}
    for i, behavior in enumerate(BEHAVIORS):
        for j, condition in enumerate(CONDITIONS):
            row = indexed.get((behavior, condition))
            if row:
                matrix[i, j], pmatrix[i, j] = row["r"], row["p"]
    assert np.isnan(matrix[:3, 0]).all() and np.isfinite(matrix[3:, 0]).all()
    assert np.isfinite(matrix[:, 1]).all()
    fig, ax = plt.subplots(figsize=(3.03, 2.25))
    cmap = LinearSegmentedColormap.from_list("paper_diverging", ["#C73E3D", "#F0EFEC", "#2A78D6"])
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-.3, vmax=.3, aspect="auto")
    for i in range(6):
        for j in range(2):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, facecolor="#F7F7F5", edgecolor="#D3D2CC",
                                           hatch="////", linewidth=0))
                continue
            significant = pmatrix[i, j] < .05
            ax.text(j, i, f"{value:+.3f}".replace("-", "\u2212"), ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if significant else "normal", color="#1F1F1E")
            if significant:
                ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor="#1F1F1E", linewidth=1.2))
    ax.set_xticks(range(2), ["No user", "With user"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(6), list(BEHAVIORS.values()), fontsize=7)
    ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.axhline(2.5, color="white", linewidth=4)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.07, pad=.04, aspect=18, ticks=[-.3, -.15, 0, .15, .3])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, width=.5, labelsize=7)
    cbar.set_label("Pearson $r$")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)


def main() -> None:
    runs = load_agent_runs()
    raw = runs[runs.selected & runs.condition.isin(CONDITIONS)]
    per_game, pooled = correlations(raw, "payout_ungated")
    plot(pooled)
    VALUES.parent.mkdir(parents=True, exist_ok=True)
    VALUES.write_text(json.dumps(dict(pooled=pooled, per_game=per_game), indent=2), encoding="utf-8")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
