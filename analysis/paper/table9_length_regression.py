"""Table 9: linear probability models of counterparty-facing misaligned behavior.

Outcome: 100 x (the agent showed any counterparty-facing MB). Agents: No User, both
agents; With User, the agent whose party has the user (16,800 agent-runs). Predictors:
With User indicator, centered log(turns), and their interaction; HC1 standard errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.inputs import load_agent_runs
from analysis.config import COUNTERPARTY_FACING, out_dir, work_dir

SPECS = {"(1)": "y ~ with_user", "(2)": "y ~ with_user + log_turns", "(3)": "y ~ with_user * log_turns"}
ROWS = [("with_user", r"\oneuser{}"), ("log_turns", r"$\log$(turns)"),
        ("with_user:log_turns", r"\oneuser{} $\times$ $\log$(turns)"), ("Intercept", "Intercept")]


def fit() -> pd.DataFrame:
    runs = load_agent_runs()
    d = runs[runs.selected & runs.condition.isin(["no_user", "single_user"])].copy()
    d["y"] = 100 * d[COUNTERPARTY_FACING].any(axis=1).astype(float)
    d["with_user"] = (d.condition == "single_user").astype(float)
    d["log_turns"] = np.log(d.turns)
    d["log_turns"] -= d.log_turns.mean()
    rows = []
    for model, spec in SPECS.items():
        result = smf.ols(spec, d).fit(cov_type="HC1")
        for term in result.params.index:
            rows.append(dict(model=model, term=term, coef=result.params[term], se=result.bse[term],
                             p=result.pvalues[term], r2=result.rsquared, n=int(result.nobs)))
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def latex(values: pd.DataFrame) -> str:
    cell = {(r.model, r.term): r for r in values.itertuples()}
    lines = []
    for term, label in ROWS:
        coefs, ses = [], []
        for model in SPECS:
            r = cell.get((model, term))
            if r is None:
                coefs.append(""); ses.append("")
                continue
            s = stars(r.p)
            coefs.append(f"{r.coef:.2f}" + (rf"\rlap{{$^{{{s}}}$}}" if s else ""))
            ses.append(f"({r.se:.2f})")
        lines.append(f"        {label} & " + " & ".join(coefs) + r" \\")
        lines.append("        & " + " & ".join(ses) + r" \\")
    first = values.groupby("model").first()
    n = " & ".join(f"{int(first.loc[m, 'n']):,}" for m in SPECS)
    r2 = " & ".join(f"{first.loc[m, 'r2']:.3f}" for m in SPECS)
    return "\n".join([
        r"    \begin{tblr}{",
        r"        colspec={X[l] Q[c] Q[c] Q[c]},",
        r"        colsep=5pt, rowsep=1.5pt, column{1}={leftsep=0pt}, column{Z}={rightsep=7pt},",
        r"    }",
        r"        \toprule",
        r"        \SetCell[c=4]{c}{Dependent variable:\\ counterparty-facing misaligned-behavior rate (\%)} & & & \\",
        r"        \midrule",
        r"        & (1) & (2) & (3) \\",
        r"        \midrule",
        *lines,
        r"        \midrule",
        f"        Observations (agents) & {n} \\\\",
        f"        $R^2$ & {r2} \\\\",
        r"        \bottomrule",
        r"    \end{tblr}",
        "",
    ])


def main() -> None:
    values = fit()
    work_dir().mkdir(parents=True, exist_ok=True)
    (out_dir() / "tables").mkdir(parents=True, exist_ok=True)
    values.to_csv(work_dir() / "table9_length_regression_values.csv", index=False)
    (out_dir() / "tables" / "length_regression_body.tex").write_text(latex(values), encoding="utf-8")
    print("Saved table 9")


if __name__ == "__main__":
    main()
