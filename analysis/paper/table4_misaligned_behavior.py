"""Table 4: misaligned-behavior rates by behavior, model and setting (No User and With User).

Input: behavior_episodes.csv (analysis/inputs.py). Rate = share of runs in which the agent was flagged for the
behavior; With User counts the agent whose party has the user, No User both agents.
MB_PREFERENCE_HARDENING (not misaligned behavior; Figure 3) is excluded.
Aggregation follows paper_stats.py: rate per (game, model) with a binomial SE, equal-weight mean over models and
then games, sampling-error 95% CI. Each group also has an "any behavior in the group" row, computed the same way.
With no user there is nothing to be user-facing toward, so user-facing rates are shown for With User only.
Writes tables/misbehavior_body.tex.
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

USER_FACING_BEHAVIOR_IDS = ps.USER_FACING
NON_USER_FACING_BEHAVIOR_IDS = ps.COUNTERPARTY_FACING

DATA = ps.DATA
MODELS8 = ["claudehaiku45", "claudesonnet5", "deepseek_v4_flash", "gpt56luna", "gpt56terra",
           "qwen36_plus", "qwen37_plus", "qwen38_flash"]
CONDS = ["no_user", "single_user"]  # two-user data is not shown
EXCLUDED = {"MB_PREFERENCE_HARDENING"}  # preference hardening is shown in Figure 3
HEATMAP_MIN_TINT = 0.20
# Piecewise relative slopes: 1.5, 0.5625, 0.1, 0.25/85; more of the tint range goes to rates below 5%.
HEATMAP_RATE_KNOTS = (0.0, 1.0, 5.0, 15.0, 100.0)
HEATMAP_COLOR_PROGRESS = (0.0, 1.5, 3.75, 4.75, 5.0)


def load_episodes() -> pd.DataFrame:
    path = DATA / "behavior_episodes.csv"
    return pd.read_csv(path)  # written by analysis/inputs.py from data/processed


def main() -> None:
    frame = load_episodes()
    assert set(frame.model) == set(MODELS8) and set(CONDS) <= set(frame.condition)
    frame = frame[frame.condition.isin(CONDS)]
    # The analysis covers the seven games of the paper.
    assert set(frame.game) == set(ps.GAMES), sorted(frame.game.unique())
    make(frame, ps.GAMES, "misbehavior")


def any_cells(frame: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Binomial cells for "agent flagged for at least one of ids" per (condition, game, model)."""
    units = (frame[frame.behavior_id.isin(ids)].groupby(["condition", "game", "model", "role", "game_id"])["present"].any()
             .rename("flagged").reset_index())
    return st.binomial_cells(units, ["condition", "game", "model"])


# Two-line column headers so twelve columns fit the text width.
HEADER_BREAKS = {"Hallucinated action": r"{Hallucinated\\action}",
                 "Unsupported urgency": r"{Unsupported\\urgency}", "Misunderstanding": r"{Misunder-\\standing}",
                 "Out-of-scope terms": r"{Out-of-scope\\terms}"}
# Single-column (user-facing) headers get narrower breaks, so every number column can have about the same width
# without shrinking the category names.
NARROW_BREAKS = {"Misreporting": r"{Mis-\\reporting}", "Hallucinated action": r"{Halluci-\\nated\\action}",
                 "Unsupported urgency": r"{Unsup-\\ported\\urgency}"}


def rate_heatmap_limit(values) -> float:
    """One shared percentage-rate color domain, rounded outward to tens."""
    values = np.asarray(list(values), dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or (values > 100).any():
        raise ValueError("MB heatmap rates must be finite percentages in [0, 100].")
    return max(10.0, float(np.ceil(values.max() / 10) * 10))


def rate_heatmap_color(value: float, limit: float, *, peak_rate: float | None = None) -> str:
    """Continuous piecewise-linear red intensity, with more detail at low rates.

    All cells share the same peak.
    Rates rounding to 0.0 remain white; other cells retain the 20% floor.
    """
    if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0 or not 0 <= value <= min(limit, 100):
        raise ValueError("MB rate must lie within its positive color limit and [0, 100].")
    peak_rate = min(limit, 100) if peak_rate is None else peak_rate
    if not np.isfinite(peak_rate) or not value <= peak_rate <= min(limit, 100):
        raise ValueError("The shared peak must cover the cell and lie inside the color domain.")
    if pt.number(value) == "0.0":
        return "white"
    progress = np.interp(value, HEATMAP_RATE_KNOTS, HEATMAP_COLOR_PROGRESS)
    peak_progress = np.interp(peak_rate, HEATMAP_RATE_KNOTS, HEATMAP_COLOR_PROGRESS)
    peak_tint = HEATMAP_MIN_TINT + (1 - HEATMAP_MIN_TINT) * peak_rate / limit
    intensity = HEATMAP_MIN_TINT + (peak_tint - HEATMAP_MIN_TINT) * progress / peak_progress
    return f"MBRateHigh!{100 * intensity:.2f}!white"


def write_table(frame: pd.DataFrame, cells: pd.DataFrame, games: list[str], groups: list[tuple[str, list[str], list[str]]],
                name: str) -> None:
    """Generate three MB groups with family headings and a common red heatmap.

    Model rates are equal-weight game means; Mean pools models then games.
    Stars mark paired With-versus-No comparisons at p < .05.
    Numeric backgrounds encode rates. User-facing and reference rows are untested.
    Layout lives in table_templates/misbehavior_body.tex.
    """
    def rate(c: pd.DataFrame, cond: str, model: str | None) -> tuple[float, float | None, float]:
        """Rate (%), p of With User minus No User (when the row pair has No User) and 95% CI half-width."""
        sub = c if model is None else c[c.model == model]
        own = sub[sub.condition == cond]
        r = st.two_stage(own)[1] if model is None else st.combine(own.est, own.se)
        p = None
        if cond == "single_user" and (sub.condition == "no_user").any():
            pc = st.paired_cells(sub, "condition", "single_user", "no_user", ["game", "model"])
            p = (st.two_stage(pc)[1] if model is None else st.combine(pc.est, pc.se))["pvalue"]
        return r["mean"], p, r["ci_high"] - r["mean"]

    user_ids = next(ids for _, ids, conds in groups if "no_user" not in conds)
    cp_ids = next(ids for _, ids, conds in groups if "no_user" in conds)
    user = cells[cells.condition == "single_user"]
    all_any = pd.concat([any_cells(frame, user_ids + cp_ids).query("condition == 'single_user'"),
                         any_cells(frame, cp_ids).query("condition == 'no_user'")], ignore_index=True)
    with_only, both = ["single_user"], ["no_user", "single_user"]
    # the section title says user- or counterparty-facing, so the two urgency rows drop that prefix
    def short(b: str) -> str:
        return "Unsupported urgency" if b in ("MB_UNSUPPORTED_URGENCY", "MB_OTHER_AGENT_UNSUPPORTED_URGENCY") else ps.MB_NAMES[b]

    sections = [  # (section title, [(row label, cells)], conditions)
        ("User-facing", [(short(b), user[user.behavior_id == b]) for b in user_ids]
         + [("Any", any_cells(frame, user_ids).query("condition == 'single_user'"))], with_only),
        ("Counterparty-facing", [(short(b), cells[cells.behavior_id == b]) for b in cp_ids]
         + [("Any", any_cells(frame, cp_ids))], both),
        ("All behaviors", [("Any", all_any)], both),  # without a user only counterparty-facing behavior can occur
    ]
    models = sorted(frame.model.unique(), key=list(ps.MODEL_NAMES).index)
    pt.require_model_order(models)
    # The template's indentation and spacing follow these row blocks.
    # Fail clearly if the taxonomy changes instead of styling the wrong rows.
    if [(len(rows), conds) for _, rows, conds in sections] != [
        (4, with_only), (4, both), (1, both),
    ]:
        raise ValueError("Taxonomy rows changed; update the misbehavior table template first.")

    computed = []
    for title, rows, conds in sections:
        blocks = []
        for label, c in rows:
            blocks.append((label, [
                (cond, [rate(c, cond, m) for m in models], rate(c, cond, None))
                for cond in conds
            ]))
        computed.append((title, blocks))
    patterns = [
        pt.unsigned_pattern([
            estimates[i][0]
            for _, blocks in computed for _, rows in blocks for _, estimates, _ in rows
        ])
        for i in range(len(models))
    ]
    rate_values = [
        estimate[0]
        for _, blocks in computed for _, rows in blocks for _, estimates, mean in rows
        for estimate in [*estimates, mean]
    ]
    limit = rate_heatmap_limit(rate_values)
    peak_rate = max(rate_values)

    lines = []
    backgrounds = []
    table_row = 3
    for title, blocks in computed:
        lines += [
            r"        \midrule",
            pt.row([rf"\SetCell[c=11]{{l}} \textbf{{{title}}}"] + [""] * 10),
        ]
        table_row += 1
        for label, rows in blocks:
            for ri, (cond, estimates, mean) in enumerate(rows):
                for column, (estimate, _, _) in enumerate([*estimates, mean], start=3):
                    backgrounds.append(
                        rf"        cell{{{table_row}}}{{{column}}}={{bg={rate_heatmap_color(estimate, limit, peak_rate=peak_rate)}}},"
                    )
                head = label if len(rows) == 1 else rf"\SetCell[r={len(rows)}]{{l,m}} {label}"
                vals = [head if ri == 0 else "", "No" if cond == "no_user" else "With"]
                vals.extend(
                    pt.mb_value(value, p, pattern=patterns[i])
                    for i, (value, p, _) in enumerate(estimates)
                )
                value, p, hw = mean
                vals.append(pt.mb_ci(value, hw, p))
                lines.append(pt.row(vals))
                table_row += 1
    ps.save_table(pt.render_template(
        "misbehavior_body", ROWS="\n".join(lines), CELL_COLORS="\n".join(backgrounds),
    ), f"{name}_body")


def make(frame: pd.DataFrame, games: list[str], table_name: str) -> None:
    # every behavior must have the same episode count per (game, model, condition, role), else denominators differ
    counts = frame.groupby(["game", "model", "condition", "behavior_id"]).size().unstack("behavior_id")
    assert (counts.nunique(axis=1) == 1).all(), counts[counts.nunique(axis=1) > 1].head()
    frame = frame[frame.game.isin(games)]
    cells = st.binomial_cells(frame.rename(columns={"present": "flagged"}), ["behavior_id", "condition", "game", "model"])

    user_ids = [b for b in ps.USER_FACING if b in USER_FACING_BEHAVIOR_IDS and b not in EXCLUDED]
    cp_ids = [b for b in ps.COUNTERPARTY_FACING if b in NON_USER_FACING_BEHAVIOR_IDS]
    assert len(user_ids) == 3 and len(cp_ids) == 3
    groups = [(ps.MB_GROUP_NAMES["user_facing"], user_ids, ["single_user"]),
              (ps.MB_GROUP_NAMES["non_user_facing"], cp_ids, CONDS)]
    write_table(frame, cells, games, groups, table_name)


if __name__ == "__main__":
    main()
