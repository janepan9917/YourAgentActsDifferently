"""Table 3: payout change of the user-side agent, by game and model.

Input: payout_change_by_model.csv (analysis/inputs.py), the With User minus No User payout of the user-side
party per (game, model, role) in % of the scale S_i, with its Welch SE and p-value. The two roles are averaged
within each (game, model), SE = sqrt(se1^2 + se2^2)/2.
  Pooled column: per game, DerSimonian-Laird random-effects pool over the 8 models (payout_pooling.py).
  Mean row: per model, equal-weight mean over the 7 games, SE = sqrt(sum_g se_g^2)/G (paper_stats.combine).
Stars mark two-sided p < .05. Writes tables/user_outcomes_body.tex (layout: table_templates/user_outcomes_body.tex).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_stats as st  # noqa: E402
import paper_style as ps  # noqa: E402
import paper_tables as pt  # noqa: E402

from payout_pooling import role_averaged_pooled  # noqa: E402

DATA = ps.DATA
PB_PREFIX = "principal_bench_stripped_"


HEATMAP_MIN_TINT = 0.20


def heatmap_limit(values) -> float:
    """One symmetric color range for all estimates, rounded outward to tens."""
    values = np.asarray(list(values), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Heatmap estimates must be nonempty and finite.")
    return max(10.0, float(np.ceil(np.abs(values).max() / 10) * 10))


def heatmap_color(value: float, limit: float) -> str:
    """Linear intensity above a 20% tint floor for displayed nonzero values.

    Nonzero tint = 0.20 + 0.80 * abs(value) / limit, on one shared scale.
    Values rounding to 0.0 use white, matching the printed table.
    """
    if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0 or abs(value) > limit:
        raise ValueError("Heatmap value must lie within a finite, positive color limit.")
    if pt.number(value) == "0.0":
        return "white"
    color = "PayoutIncrease" if value > 0 else "PayoutDecrease"
    intensity = HEATMAP_MIN_TINT + (1 - HEATMAP_MIN_TINT) * abs(value) / limit
    return f"{color}!{100 * intensity:.2f}!white"


def write_table(left: pd.DataFrame, right: pd.DataFrame, gm: pd.DataFrame) -> dict:
    """Write the payout table body.

    Each model cell uses its role-averaged z-test. Pooled uses the game's
    random-effects estimate; Mean and the corner use equal-weight means.
    Layout lives in table_templates/user_outcomes_body.tex. Stars mark two-sided p < .05.
    Cell backgrounds use one red--white--green scale with a 20% floor and linear increments,
    including Pooled and Mean estimates; CI-only cells remain uncolored.
    """
    games = list(left.index)
    models = sorted(right.model, key=list(ps.MODEL_NAMES).index)
    pt.require_model_order(models)
    corner = st.combine(gm.m, gm.se)
    cells = gm.set_index(["model", "game"])
    by_model = right.set_index("model")
    limit = heatmap_limit([*gm.m, *left["mean"], *right["mean"], corner["mean"]])
    backgrounds = []
    for row_index, game in enumerate(games, start=3):
        estimates = [cells.loc[(model, game), "m"] for model in models] + [left.loc[game, "mean"]]
        for col_index, estimate in enumerate(estimates, start=2):
            backgrounds.append(
                rf"        cell{{{row_index}}}{{{col_index}}}={{bg={heatmap_color(estimate, limit)}}},"
            )
    for col_index, estimate in enumerate(
        [by_model.loc[model, "mean"] for model in models] + [corner["mean"]], start=2,
    ):
        backgrounds.append(
            rf"        cell{{{3 + len(games)}}}{{{col_index}}}={{bg={heatmap_color(estimate, limit)}}},"
        )
    patterns = {
        m: pt.payout_pattern(
            [cells.loc[(m, g), "m"] for g in games] + [by_model.loc[m, "mean"]],
            by_model.loc[m, "ci_high"] - by_model.loc[m, "mean"],
        )
        for m in models
    }
    lines = []
    for g in games:
        row = [ps.GAME_NAMES[g]]
        for m in models:
            c = cells.loc[(m, g)]
            row.append(pt.payout_value(c.m, st.z_summary(c.m, c.se)["pvalue"], pattern=patterns[m]))
        row.append(pt.payout_ci(
            left.loc[g, "mean"], left.loc[g, "ci_high"] - left.loc[g, "mean"], left.loc[g, "pvalue"],
        ))
        lines.append(pt.row(row))
    lines += [
        r"        \midrule",
        pt.row([r"\textit{Mean}"] + [
            pt.payout_value(by_model.loc[m, "mean"], by_model.loc[m, "pvalue"], pattern=patterns[m])
            for m in models
        ] + [pt.payout_ci(corner["mean"], corner["ci_high"] - corner["mean"], corner["pvalue"])]),
        pt.row([r"\textbf{95\% CI}"] + [
            pt.payout_value(by_model.loc[m, "ci_high"] - by_model.loc[m, "mean"], pattern=patterns[m], ci=True)
            for m in models
        ] + [""]),
    ]
    ps.save_table(pt.render_template(
        "user_outcomes_body", ROWS="\n".join(lines),
        CELL_COLORS="\n".join(backgrounds),
    ), "user_outcomes_body")
    return corner


def main() -> None:
    by_model = pd.read_csv(DATA / "payout_change_by_model.csv")
    by_model["game"] = by_model.game.str.replace(PB_PREFIX, "pb_", regex=False)
    games = ps.GAMES
    assert set(by_model.game) == set(games), sorted(by_model.game.unique())
    # per (game, model): role-averaged change (both roles have the same n, so this is the cell's mean change)
    gm = (by_model.groupby(["game", "model"])
          .apply(lambda d: pd.Series(dict(m=d["mean"].mean(), se=np.sqrt((d["se"] ** 2).sum()) / 2)), include_groups=False)
          .reset_index())
    assert gm.groupby("model").game.nunique().eq(len(games)).all(), gm.groupby("model").game.nunique()
    left = role_averaged_pooled(by_model).reindex(games)
    right = pd.DataFrame([dict(model=m, **st.combine(d.m, d.se), n_games=len(d), games_up=int((d.m > 0).sum()))
                          for m, d in gm.groupby("model")])
    corner = write_table(left, right, gm)
    print(f"overall (equal-weight mean of all game x model cells): {corner['mean']:.3f}, p={corner['pvalue']:.2g}")
    print(left.round(3).to_string())
    print(right.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
