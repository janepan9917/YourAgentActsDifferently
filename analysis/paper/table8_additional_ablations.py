"""Table 8: additional user-condition ablations.

Writes the table body from data/ablations/additional_ablations_summary.csv.
"""
from __future__ import annotations

import re

import pandas as pd

from analysis.config import data_dir, out_dir, work_dir

ROWS = [  # summary-table condition -> paper row label
    ("no user", r"\textsc{No User}"),
    ("standard single-user", r"\textsc{With User}"),
    ("silent user", r"\textsc{Passive User}"),
    ("silent + opening consult", r"\textsc{Opening-Only User}"),
    ("user sees transcript", r"\textsc{User Sees Agent-Agent Transcript}"),
    ("no final veto", r"\textsc{No Consent Step}"),
    ("agent sees private info", r"\textsc{Agent Sees Private Info}"),
]
NUMBER = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)")

HEADER = r"""\begingroup
\microtypesetup{protrusion=false}
\settowidth{\AblationCIWidth}{\footnotesize0.0}
\newcommand{\AblationValue}[4][00.0]{\begingroup\settowidth{\AblationValueWidth}{#1}\makebox[\AblationValueWidth][r]{#2}\,$\pm$\,\makebox[\AblationCIWidth][l]{#3\rlap{\textsuperscript{#4}}}\endgroup}
    \begin{tblr}{
        width=0.75\linewidth, colsep=1.8pt, rowsep=0.8pt,
        colspec={Q[l,m] *{3}{X[c,m]}},
        column{1}={leftsep=0pt}, column{Z}={rightsep=0pt},
        cells={font=\footnotesize}, row{1-2}={c,font=\footnotesize\bfseries}, cell{1-2}{1}={l},
    }
        \toprule
        \SetCell[r=2]{l,m} Condition & \SetCell[r=2]{c,m} Payout vs.\ no user (\%) & \SetCell[c=2]{c} Misaligned behavior rate (\%) & \\
        \cmidrule[lr]{3-4}
        & & User-facing & Counterparty-facing \\
        \midrule"""
FOOTER = r"""        \bottomrule
    \end{tblr}
\endgroup
"""


def parse(text: str) -> tuple[float, float] | None:
    match = NUMBER.search(str(text))
    return (float(match.group(1)), float(match.group(2))) if match else None


def value(cell, width: str = "") -> str:
    mean, half = cell
    option = f"[{width}]" if width else ""
    return rf"\AblationValue{option}{{{abs(mean):.1f}}}{{{half:.1f}}}{{}}" if mean >= 0 else \
        rf"\AblationValue{option}{{$-${abs(mean):.1f}}}{{{half:.1f}}}{{}}"


def main() -> None:
    summary = pd.read_csv(data_dir() / "ablations" / "additional_ablations_summary.csv",
                          encoding="utf-8").set_index("condition")
    lines, values = [HEADER], []
    for key, label in ROWS:
        row = summary.loc[key]
        payout, user, cp = parse(row.payout), parse(row.user_facing), parse(row.non_user_facing)
        values.append(dict(condition=label, payout=payout, user_facing=user, counterparty_facing=cp))
        if key == "no user":  # reference row: no payout change and no user to mislead
            cells = ["---", "---", value(cp, "0.0")]
        else:
            cells = [value(payout, "$-$0.0"), value(user), value(cp, "0.0")]
        lines.append(f"        {label} & " + " & ".join(cells) + r" \\")
    lines.append(FOOTER)
    (out_dir() / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir() / "tables" / "user_ablation_additional_variants_body.tex").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(values).to_csv(work_dir() / "table8_additional_ablations_values.csv", index=False)
    print("Saved table 8")


if __name__ == "__main__":
    main()
