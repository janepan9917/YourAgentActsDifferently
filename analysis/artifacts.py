"""Write the paper's figures and tables from the data in data/.

Each step runs in its own Python process (separate matplotlib state). Figures and tables are
written under ``outputs/``; intermediate files and step logs under ``work/`` (``ANALYSIS_WORK``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from .config import REPO, out_dir, work_dir

PAPER = "analysis/paper"
STEPS = [  # (name, paper artifact, command)
    ("inputs", "per-episode inputs of the figure and table code", ["-m", "analysis.inputs"]),
    ("table3", "Table 3 (payout change by game and model)", [f"{PAPER}/table3_payout.py"]),
    ("figure3", "Figure 3 (opening offers, concession, preference hardening)", [f"{PAPER}/figure3_agent_aggression.py"]),
    ("table4", "Table 4 (misaligned-behavior rates)", [f"{PAPER}/table4_misaligned_behavior.py"]),
    ("table5", "Table 5 (user condition ablations)", [f"{PAPER}/table5_user_ablations.py"]),
    ("table7", "Table 7 (different user models)", [f"{PAPER}/table7_user_model_robustness.py"]),
    ("table8", "Table 8 (additional ablations)", ["-m", "analysis.paper.table8_additional_ablations"]),
    ("table9", "Table 9 (length-controlled regression)", ["-m", "analysis.paper.table9_length_regression"]),
    ("figure4", "Figure 4 (user model selection)", [f"{PAPER}/figure4_user_model_selection.py"]),
    ("figure5a", "Figure 5 panel inputs (misaligned behaviors)", [f"{PAPER}/judge_validation/figure5_misaligned_behavior.py"]),
    ("figure5b", "Figure 5 panel inputs (preference hardening)",
     [f"{PAPER}/judge_validation/figure5_preference_hardening.py"]),
    ("figure5", "Figure 5 (judge validation)", [f"{PAPER}/judge_validation/figure5_judge_validation.py"]),
    ("figure6", "Figure 6 (misaligned behavior vs. payout)", ["-m", "analysis.paper.figure6_mb_payout_correlations"]),
]


def run_all(only: list[str] | None = None) -> None:
    out_dir().mkdir(parents=True, exist_ok=True)
    work = work_dir()
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep),
               MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", ANALYSIS_WORK=str(work))
    log_dir = work / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name, label, args in STEPS:
        if only and name not in only and not (name == "inputs") and not any(name.startswith(o) for o in only):
            continue
        start = time.time()
        with (log_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, *args], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if result.returncode == 0 else f"FAILED (see {log_dir / (name + '.log')})"
        print(f"{name:9s} {label:62s} {time.time() - start:5.1f}s  {status}", flush=True)
        if result.returncode != 0:
            raise SystemExit(1)
    print(f"figures and tables written to {out_dir()}")
