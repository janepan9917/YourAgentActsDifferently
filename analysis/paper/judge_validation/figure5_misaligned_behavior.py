"""Figure 5, panel (a): F1 of candidate judges for the six misaligned behaviors.

Reads data/labels/judge_validation (human ground truth and four candidate judges) and writes
the panel's metrics to the work folder. Run by `python -m analysis all` (step figure5a).
"""
from __future__ import annotations

import argparse
import json
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
from analysis.config import data_dir  # noqa: E402
from turn_labels import load_auto_labels, load_manual_labels  # noqa: E402
from misaligned_behavior_f1 import build_f1_tables  # noqa: E402

OUT = ps.DATA
SOURCE = data_dir() / "labels" / "judge_validation"
STEM = "appendix_judge_validation"
CATEGORIES = (
    "MB_OFFER_STATE_MISREPORTING", "MB_HALLUCINATED_ACTION", "MB_UNSUPPORTED_URGENCY",
    "MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING", "MB_OTHER_AGENT_UNSUPPORTED_URGENCY",
    "MB_EXTRA_PROTOCOL_CONTRACTING",
)
METHODS = {"single_gpt54mini": "GPT-5.4 Mini", "single_gpt56luna": "GPT-5.6 Luna",
           "single_gpt56terra": "GPT-5.6 Terra", "single_gpt56sol": "GPT-5.6 Sol"}
COLORS = ("#D3D2CC", "#A3A29B", "#6E6D68", ps.ACCENT)  # grey ramp by judge size; the selected judge (Sol) in the accent
MICRO = "Micro (six categories)"


def metrics(frame: pd.DataFrame) -> dict:
    truth, pred = frame.manual_present.to_numpy(bool), frame.auto_present.to_numpy(bool)
    tp, fp, fn, tn = [int(a.sum()) for a in (truth & pred, ~truth & pred, truth & ~pred, ~truth & ~pred)]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=len(frame), positives=tp+fn,
                precision=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)


def trajectory_metrics(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["method", "game_id", "turn_number", "behavior_id"]
    data = aligned[aligned.behavior_id.isin(CATEGORIES)].copy()
    if not data.scored.all() or data.duplicated(keys).any():
        raise ValueError("Incomplete or duplicate human/judge alignment; missing predictions are not negative")
    if not set(data.manual_label) <= {"present", "absent"} or not set(data.auto_label) <= {"present", "absent"}:
        raise ValueError("Unscorable labels in the aligned six-category subset")
    data["manual_present"] = data.manual_label.eq("present")
    data["auto_present"] = data.auto_label.eq("present")
    pooled = data.groupby(["method", "game_id", "behavior_id"], as_index=False)[["manual_present", "auto_present"]].any()
    rows = []
    for method, group in pooled.groupby("method"):
        rows.append(dict(level="Trajectory", method=method, category=MICRO, **metrics(group)))
        for category, cells in group.groupby("behavior_id"):
            rows.append(dict(level="Trajectory", method=method, category=category, **metrics(cells)))
        union = group.groupby("game_id", as_index=False)[["manual_present", "auto_present"]].any()
        rows.append(dict(level="Trajectory", method=method, category="Any MB", **metrics(union)))
    return data, pd.DataFrame(rows)


def load_data(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manual = load_manual_labels("manual", source / "ground_truth")
    autos = []
    for method in METHODS:
        path = source / "automatic" / method / "misbehavior_eval.jsonl"
        raw = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if any(row.get("error") or str(row.get("input_truncated", False)).lower() in {"true", "1"} for row in raw):
            raise ValueError(f"Failed or truncated automatic judgment: {method}")
        if not set(row["label_mb"] for row in raw) <= {"present", "absent", "not_applicable", "indeterminate"}:
            raise ValueError(f"Unknown automatic label: {method}")
        if len({row["game_id"] for row in raw}) != 20:
            raise ValueError(f"Expected 20 evaluated trajectories: {method}")
        autos.append(load_auto_labels(method, source / "automatic" / method))
    _, _, aligned = build_f1_tables(manual=manual, autos=autos)
    data, result = trajectory_metrics(pd.DataFrame(aligned))
    if len(data) != len(METHODS) * 229 * 6:
        raise ValueError("Expected 229 annotated turns, six categories and four methods")
    validate_metrics(result)
    return data, result


def validate_metrics(result: pd.DataFrame) -> pd.DataFrame:
    if set(result.level) != {"Trajectory"} or not set(result.method) <= set(METHODS):
        raise ValueError("Only four single-judge trajectory-level methods are allowed")
    overall = result[result.category.eq(MICRO)].copy()
    if overall.method.duplicated().any() or set(overall.method) != set(METHODS):
        raise ValueError("Expected one six-category micro result for each of the four judges")
    if not overall.n.eq(120).all():
        raise ValueError("Micro denominator must be 20 trajectories x six categories")
    numeric = overall[["precision", "recall", "f1"]].to_numpy(float)
    if not np.isfinite(numeric).all() or not ((numeric >= 0) & (numeric <= 1)).all():
        raise ValueError("Invalid pooled precision/recall/F1")
    return overall.set_index("method").loc[list(METHODS)]


def build_figure(result: pd.DataFrame):
    overall = validate_metrics(result)
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 1.95), layout="constrained")
    x, width = np.arange(3), .2
    for m, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = overall.loc[method, ["precision", "recall", "f1"]].to_numpy(float)
        ax.bar(x+(m-1.5)*width, values, width*.92, color=color, label=METHODS[method], zorder=3)
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ps.ygrid(ax)
    ax.set_yticks(np.arange(0, 1.01, .25))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--plot-only", action="store_true", help="Use previously rebuilt trajectory metrics")
    parser.add_argument("--metrics", type=Path, help="Metrics CSV for --plot-only")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.is_relative_to(args.source_root.resolve()):
        raise ValueError("Output directory must not be inside the label folder")
    if args.plot_only:
        path = args.metrics or out / f"{STEM}_metrics.csv"
        result = pd.read_csv(path)
        validate_metrics(result)
    else:
        aligned, result = load_data(args.source_root)
    fig = build_figure(result)
    out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        aligned.to_csv(out / f"{STEM}_aligned_labels.csv", index=False, encoding="utf-8-sig")
        result.to_csv(out / f"{STEM}_metrics.csv", index=False, encoding="utf-8-sig")
    pdf, png = out / f"{STEM}.pdf", out / f"{STEM}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    ps.save(fig, None, png)
    plt.close(fig)
    print(result[result.category.eq(MICRO)][["method", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
