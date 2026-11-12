"""Figure 3: change in opening offers, mean concession and preference hardening with a user, one row per game.

Effect = single user (the agent that has the user) minus no user, per (game, model), pooled over the 8 models per game with the same
DerSimonian-Laird random-effects meta used elsewhere. Dot = pooled effect, line = 95% CI. Bottom row: mean over the 7 games
with a t CI across games.
  opening offer   : aggressiveness_by_condition_episodes.csv (open_extremity), unclipped, SD units of each game (SD over no-user + single-user episodes only); roles averaged; Welch se
  hardening       : preference_hardening_episodes.csv, percentage points of episodes flagged; binomial se with (x+0.5)/(n+1) adjustment
  mean concession : aggressiveness_by_condition_episodes.csv (mean_concession), with-user minus no-user change
                    (- = smaller concessions); SD units; not adjusted for the number of own offers
Panels: opening offer, mean concession, preference hardening; titles "Δ <measure>" with Δ = with user minus no user, x axes
carry the unit only. The mean-concession axis is reversed, so in every panel more aggressive is to the right. Colors follow
aggressiveness: a grey-to-red ramp (symmetric around 0, scale = the panel's largest |effect|), greyer = less aggressive
(left), redder = more aggressive (right). Large dot = CI excludes 0, small faded dot = CI includes 0.
Drawn at printed size in the shared paper font. Rows follow the paper's game order.
Writes figures/agent_aggression.pdf.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_style as ps  # noqa: E402

from _shared import random_effects_meta  # noqa: E402

D = ps.DATA
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

RAMP = LinearSegmentedColormap.from_list("grey_red", ["#6E6D68", "#D8D5CF", "#C0392B"])  # left grey -> right red


def value_color(x, scale):
    """Aggressiveness color: grey (less aggressive) -> pale (0) -> red (more aggressive). A negative `scale` marks a panel
    where smaller values are more aggressive (mean concession), so its colors flip together with its x axis."""
    t = float(np.clip(x / scale, -1, 1)) if scale else 0.0
    return RAMP((t + 1) / 2)


def gradient_line(ax, lo, hi, y, scale, lw, zorder=2):
    xs = np.linspace(lo, hi, 120)
    mids = (xs[:-1] + xs[1:]) / 2
    segs = [[(xs[i], y), (xs[i + 1], y)] for i in range(len(xs) - 1)]
    ax.add_collection(LineCollection(segs, colors=[value_color(x, scale) for x in mids], linewidths=lw, capstyle="butt", zorder=zorder))


def pool(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, d in cells.groupby("game"):
        r = random_effects_meta(d.est.to_numpy(float), d.se.to_numpy(float))
        rows.append(dict(game=g, est=r["estimate"], lo=r["ci_low"], hi=r["ci_high"], p=r["pvalue"], k_models=r["k"]))
    return pd.DataFrame(rows).set_index("game")


def episode_metric_cells(ep: pd.DataFrame, metric: str, sign: float, sd: pd.Series) -> pd.DataFrame:
    """Per (game, model): role-averaged (single - no user) effect of `metric` in SD units of the game, times `sign`."""
    d0 = ep.dropna(subset=[metric])
    rows = []
    for (g, m, role), d in d0.groupby(["game", "model", "role"]):
        a, b = d[d.condition == "no_user"][metric], d[d.condition == "single_user"][metric]
        if len(a) < 2 or len(b) < 2:
            continue
        rows.append(dict(game=g, model=m, est=sign * (b.mean() - a.mean()) / sd[g], se=np.sqrt(b.var() / len(b) + a.var() / len(a)) / sd[g]))
    return (pd.DataFrame(rows).groupby(["game", "model"]).apply(lambda d: pd.Series({"est": d.est.mean(), "se": np.sqrt((d.se ** 2).sum()) / len(d)}),
                                                                 include_groups=False).reset_index())


def main() -> None:
    ps.apply()
    ep = pd.read_csv(D / "aggressiveness_by_condition_episodes.csv")
    ep = ep[ep.condition.isin(["no_user", "single_user"])]  # two-user episodes are not used here
    # SD of each measure within a game (the unit of the effects) is taken over the no-user and single-user episodes only
    sds = {m: ep.dropna(subset=[m]).groupby("game")[m].std() for m in ("open_extremity", "mean_concession")}
    op = episode_metric_cells(ep, "open_extremity", +1.0, sds["open_extremity"])
    conc = episode_metric_cells(ep, "mean_concession", +1.0, sds["mean_concession"])     # - = smaller concessions
    h = pd.read_csv(D / "preference_hardening_episodes.csv")
    h = h[h.condition.isin(["no_user", "single_user"])]
    rows = []
    for (g, m), d in h.groupby(["game", "model"]):
        a, b = d[d.condition == "no_user"].present, d[d.condition == "single_user"].present
        p0, p1 = (a.sum() + 0.5) / (len(a) + 1), (b.sum() + 0.5) / (len(b) + 1)
        rows.append(dict(game=g, model=m, est=100 * (b.mean() - a.mean()), se=100 * np.sqrt(p0 * (1 - p0) / len(a) + p1 * (1 - p1) / len(b))))
    hard = pd.DataFrame(rows)
    pooled = {"opening": pool(op), "hardening": pool(hard), "concession": pool(conc)}
    games = ps.GAMES  # paper game order
    assert set(pooled["opening"].index) == set(games), sorted(pooled["opening"].index)

    fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH, 2.45), sharey=True)
    ypos = {g: i for i, g in enumerate(games)}
    ymean = len(games) + 0.7
    csv_rows = []
    panels = [
        ("opening", "Δ Opening Offer", "Standard deviations", "{:+.2f}", +1),
        ("concession", "Δ Mean Concession", "Standard deviations", "{:+.2f}", -1),  # smaller concessions = more aggressive
        ("hardening", "Δ Preference Hardening", "Percentage points", "{:+.1f} pp", +1),
    ]
    for ax, (key, title, xlab, fmt, direction) in zip(axes, panels):
        P = pooled[key]
        # color scale: the panel's largest point estimate, signed so that red always means more aggressive
        scale = direction * float(P.loc[games, "est"].abs().max())
        for g in games:
            r = P.loc[g]
            sig = not (r.lo <= 0 <= r.hi)
            gradient_line(ax, r.lo, r.hi, ypos[g], scale, 2.0 if sig else 1.5)
            ax.plot(r.est, ypos[g], "o", ms=6.0 if sig else 4.3, color=value_color(r.est, scale), mec="#111827" if sig else "#6B7280",
                    mew=0.9 if sig else 0.55, alpha=1.0 if sig else 0.75, zorder=3)
            csv_rows.append(dict(measure=key, game=g, est=r.est, ci_low=r.lo, ci_high=r.hi, p=r.p, k_models=r.k_models))
        v = P.loc[games, "est"].to_numpy(float)
        m = v.mean()
        half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        gradient_line(ax, m - half, m + half, ymean, scale, 2.8)
        ax.plot(m, ymean, "D", ms=6.3, color=value_color(m, scale), mec="#111827", mew=0.9, zorder=3)
        ax.text(m, ymean + 0.42, fmt.format(m).replace("-", "−"), ha="center", va="top", fontsize=6.5)
        csv_rows.append(dict(measure=key, game="MEAN_OF_7_GAMES", est=m, ci_low=m - half, ci_high=m + half, p=np.nan, k_models=len(v)))
        ax.axvline(0, color="#6B7280", lw=0.6, zorder=1)
        ax.axhline(ymean - 0.65, color="#D1D5DB", lw=0.5)
        ax.set_title(title)
        ax.set_xlabel(xlab, fontsize=7)
        ax.grid(axis="x", color="#E5E7EB", lw=0.4, zorder=0)
        ax.autoscale_view()
        if direction < 0:
            ax.invert_xaxis()  # more aggressive (smaller concessions) to the right, as in the other panels
        for side in ax.spines.values():          # a clear frame around each panel
            side.set_visible(True)
            side.set_color("#374151")
            side.set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[g] for g in games] + [ymean])
    axes[0].set_yticklabels([ps.GAME_NAMES[g] for g in games] + ["Mean"])
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    axes[0].set_ylim(ymean + 1.1, -0.7)
    fig.tight_layout(w_pad=1.2)
    ps.save(fig, "agent_aggression", D / "figure3_agent_aggression.png")
    out = pd.DataFrame(csv_rows)
    out.to_csv(D / "figure3_agent_aggression_values.csv", index=False)
    print(out.round(2).pivot_table(index="game", columns="measure", values="est", sort=False).to_string())


if __name__ == "__main__":
    main()
