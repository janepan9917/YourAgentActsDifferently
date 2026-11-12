"""Table 5: user ablations -- With User vs. Passive User vs. Opening-Only User.
Payout change vs. No User and the has-user agent's misbehavior rate, user-facing vs. counterparty-facing.

Inputs:
  payout      : user_ablations_payout_by_model.csv (plain-formula PrincipalBench payouts, no eligibility gate)
                -- per (game, role, model, condition) payout means and
                the Welch difference vs. no user. The input expresses that difference in % of a per-condition scale;
                it is converted back to payout units and re-expressed in % of Table 3's payout scale
                (payout_change_by_model.csv: the largest |mean payout| of both parties in the no-user
                runs and in the runs where this role has the user), so every condition shares one denominator per
                (game, model, role) and the with-user bar equals Table 3's effect. Averaged over the two roles and
                over models, one value per game.
  misbehavior : user_ablations_mb_episodes.csv -- has-user agent's own flags; preference hardening is not
                misaligned behavior and is excluded. Rate = share of episodes flagged; game value = mean over models.
Aggregation follows paper_stats.py: bar = equal-weight mean over models then games; error bar = sampling-error 95% CI
(payout: per-(game, role, model) Welch SE, roles averaged; misbehavior: binomial SE per (game, model)). The values CSV also
records paired comparisons (condition minus reference within each cell, aggregated the same way).
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

OUT = ps.DATA
CONDS = ["with_user_llm", "initial_consult_only", "silent"]  # decreasing user involvement
VIEWS = ["user_facing", "non_user_facing"]


def table3_scaled_payout(pay: pd.DataFrame) -> pd.DataFrame:
    """Re-express each (game, role, model, condition) payout change in % of Table 3's per-(game, model, role) scale.

    The input's percent uses scale_c = max(no-user mean A1, A2, condition mean A1, A2) for that condition, so the payout
    difference is diff * scale_c / 100 (likewise the SE); dividing by Table 3's scale puts all conditions on one
    denominator."""
    t3 = pd.read_csv(OUT / "payout_change_by_model.csv")
    t3["game"] = t3.game.str.replace("principal_bench_stripped_", "pb_", regex=False)
    t3["role"] = t3.user_role.map({"user_1": "agent_1", "user_2": "agent_2"})
    t3_scale = t3.set_index(["game", "model", "role"]).scale
    means = pay.pivot_table(index=["game", "model"], columns=["condition", "role"], values="payout_mean")
    rows = []
    for _, r in pay[pay.condition != "no_user"].iterrows():
        m = means.loc[(r.game, r.model)]
        scale_c = max(m[("no_user", "agent_1")], m[("no_user", "agent_2")], m[(r.condition, "agent_1")], m[(r.condition, "agent_2")])
        scale = t3_scale.loc[(r.game, r.model, r.role)]
        rows.append(dict(game=r.game, model=r.model, role=r.role, condition=r.condition,
                         diff=r["diff"] * scale_c / scale, se=r["se"] * scale_c / scale))
    out = pd.DataFrame(rows)
    assert out.notna().all().all(), "every payout cell needs a Table 3 scale"
    return out


def half(r: dict) -> float:
    return r["ci_high"] - r["mean"]


def make(games: list[str], stem: str) -> None:
    mb = pd.read_csv(OUT / "user_ablations_mb_episodes.csv")
    models = sorted(mb.model.unique())
    pay = pd.read_csv(OUT / "user_ablations_payout_by_model.csv")
    pay = pay[pay.model.isin(models) & pay.condition.isin(CONDS + ["no_user"]) & pay.game.isin(games)]
    pay = table3_scaled_payout(pay)
    assert pay.groupby(["game", "condition"]).model.nunique().eq(len(models)).all()
    pay_cells = (pay.groupby(["condition", "game", "model"])      # role-averaged change and its SE per (game, model)
                 .apply(lambda d: pd.Series(dict(est=d["diff"].mean(), se=np.sqrt((d["se"] ** 2).sum()) / len(d))),
                        include_groups=False).reset_index())

    mb = mb[mb.game.isin(games) & mb.condition.isin(CONDS + ["no_user"]) & mb.view.isin(VIEWS)]
    mb_cells = st.binomial_cells(mb.rename(columns={"present": "flagged"}), ["view", "condition", "game", "model"])

    rows = []
    for cond in CONDS:
        per_game, r = st.two_stage(pay_cells[pay_cells.condition == cond])
        rows.append(dict(panel="payout_pct", group="payout", condition=cond, mean=r["mean"], ci_half=half(r),
                         reference="no_user", diff_p=r["pvalue"], **per_game["mean"].to_dict()))
    for view in VIEWS:
        conds = CONDS + (["no_user"] if view == "non_user_facing" else [])  # user-facing is 0 with no user by construction
        vc = mb_cells[mb_cells.view == view]
        for cond in conds:
            per_game, r = st.two_stage(vc[vc.condition == cond])
            ref = "no_user" if view == "non_user_facing" and cond != "no_user" else ("with_user_llm" if cond != "with_user_llm" else None)
            diff_p = (st.two_stage(st.paired_cells(vc, "condition", cond, ref, ["game", "model"]))[1]["pvalue"]
                      if ref else np.nan)
            rows.append(dict(panel="misbehavior_pct", group=view, condition=cond, mean=r["mean"], ci_half=half(r),
                             reference=ref, diff_p=diff_p, **per_game["mean"].to_dict()))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"{stem}_values.csv", index=False)
    write_table(out)
    print(stem); print(out[["panel", "group", "condition", "mean", "ci_half"]].round(2).to_string(index=False))


def write_table(values: pd.DataFrame) -> None:
    """Fill the 75%-width table layout.

    A star means a two-sided p < .05 comparison against No User
    (payout/counterparty-facing) or With User (user-facing). Reference cells
    are unstarred.
    """
    conds = ["with_user_llm", "initial_consult_only", "silent", "no_user"]
    macro = {"with_user_llm": r"\oneuser{}", "initial_consult_only": r"\suserwopening{}", "silent": r"\suser{}",
             "no_user": r"\nouser{}"}
    cols = [("payout", r"$-$0.0"), ("user_facing", "00.0"), ("non_user_facing", "0.0")]
    v = values.set_index(["group", "condition"])
    lines = []
    for cond in conds:
        cells = [macro[cond]]
        for group, pattern in cols:
            if (group, cond) not in v.index:
                cells.append("---")
                continue
            r = v.loc[(group, cond)]
            p = None if cond == "no_user" or pd.isna(r.diff_p) else float(r.diff_p)
            cells.append(pt.ablation_value(r["mean"], r.ci_half, p, pattern=pattern))
        lines.append(pt.row(cells))
    ps.save_table(pt.render_template("user_ablations_body", ROWS="\n".join(lines)), "user_ablations_body")


def main() -> None:
    make(ps.GAMES, "table5_user_ablations")


if __name__ == "__main__":
    main()
