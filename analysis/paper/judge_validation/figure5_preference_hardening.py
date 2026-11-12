"""Figure 5, panel (b): F1 of candidate judges for preference hardening.

Reads data/labels/judge_validation and writes the panel's metrics to the work folder.
Run by `python -m analysis all` (step figure5b).
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
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "judge"))  # MB taxonomy and label parsing
sys.path.insert(2, str(Path(__file__).resolve().parents[1]))  # paper_style
from turn_labels import load_auto_labels, load_manual_labels
from misaligned_behavior_f1 import build_f1_tables
from evaluate_misaligned_behavior import _parse_turn_numbers
import paper_style as ps
from analysis.config import data_dir

SOURCE = data_dir() / "labels" / "judge_validation"
OUT = ps.DATA
STEM = "appendix_preference_hardening_validation"
BEHAVIOR = "MB_PREFERENCE_HARDENING"
METHODS = {"single_gpt54mini": "GPT-5.4 Mini", "single_gpt56luna": "GPT-5.6 Luna",
           "single_gpt56terra": "GPT-5.6 Terra", "single_gpt56sol": "GPT-5.6 Sol"}
COLORS = ("#D3D2CC", "#A3A29B", "#6E6D68", "#EB6834")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn, tn = [int(a.sum()) for a in (truth & pred, ~truth & pred, truth & ~pred, ~truth & ~pred)]
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=len(truth), positives=tp+fn,
                precision=tp/(tp+fp) if tp+fp else np.nan,
                recall=tp/(tp+fn) if tp+fn else np.nan,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual_path = SOURCE / "ground_truth/misbehavior_manual_annotations.jsonl"
    manual_rows = [r for r in read_rows(manual_path) if r["behavior_id"] == BEHAVIOR]
    if len(manual_rows) != 229 or {r["label_mb"] for r in manual_rows} != {"present", "absent"}:
        raise ValueError("Expected 229 explicitly positive/negative PH annotations")
    if {r["agent_id"] for r in manual_rows} != {"any_agent"}:
        raise ValueError("Expected any-agent human labels")
    manual_map = {(r["game_id"], int(r["turn"])): r["label_mb"] == "present" for r in manual_rows}
    games = {g for g, _ in manual_map}
    if len(manual_map) != 229 or len(games) != 20:
        raise ValueError("Duplicate annotations or unexpected trajectory coverage")

    manual = load_manual_labels("manual", SOURCE / "ground_truth")
    autos, raw_positive_keys = [], {}
    for method in METHODS:
        path = SOURCE / "automatic" / method / "misbehavior_eval.jsonl"
        rows = read_rows(path)
        if any(r.get("error") or str(r.get("input_truncated", False)).lower() in {"true", "1"} for r in rows):
            raise ValueError(f"Failed or truncated evaluation: {method}")
        ph = [r for r in rows if r["behavior_id"] == BEHAVIOR]
        expected = {(g, a) for g in games for a in ("agent_1", "agent_2")}
        if len(ph) != 40 or {(r["game_id"], r["agent_id"]) for r in ph} != expected:
            raise ValueError(f"Missing/duplicate PH agent-trajectory labels: {method}")
        if not {r["label_mb"] for r in ph} <= {"present", "absent", "not_applicable", "indeterminate"}:
            raise ValueError(f"Unknown label: {method}")
        if any(r.get("judge_mode") not in {"single_model", "single_model_extracted_ensemble_member"} for r in ph):
            raise ValueError(f"Non-single-judge records: {method}")
        for row in ph:
            if row["judge_mode"] == "single_model_extracted_ensemble_member":
                member = row.get("extracted_from_judge_id")
                votes = json.loads(row["ensemble_votes"])
                turn_votes = json.loads(row["ensemble_turn_votes"])
                if member != "judge_1" or row["label_mb"] != votes[member] or _parse_turn_numbers(row["turn_numbers"]) != _parse_turn_numbers(turn_votes[member]):
                    raise ValueError(f"Extracted row is not the first judge's own vote: {method}")
        keys = set()
        for row in ph:
            turns = _parse_turn_numbers(row.get("turn_numbers"))
            if row["label_mb"] == "present":
                if not turns:
                    raise ValueError(f"Positive label without occurrence turns: {method}")
                keys.update((row["game_id"], int(t)) for t in turns)
            elif turns:
                raise ValueError(f"Non-positive label with occurrence turns: {method}")
        raw_positive_keys[method] = keys
        autos.append(load_auto_labels(method, path.parent))

    _, _, rows = build_f1_tables(manual=manual, autos=autos)
    aligned = pd.DataFrame(rows)
    aligned = aligned[aligned.behavior_id.eq(BEHAVIOR)].copy()
    keys = ["method", "game_id", "turn_number", "behavior_id"]
    if len(aligned) != 4*229 or not aligned.scored.all() or aligned.duplicated(keys).any():
        raise ValueError("Incomplete or duplicate aligned PH labels")
    if not set(aligned.auto_label) <= {"present", "absent"}:
        raise ValueError("Unscorable predictions are not negative labels")
    aligned["manual_present"] = aligned.manual_label.eq("present")
    aligned["auto_present"] = aligned.auto_label.eq("present")
    # Check every loader-derived label against the label JSONL.
    for row in aligned.itertuples():
        key = (row.game_id, int(row.turn_number))
        if row.manual_present != manual_map[key] or row.auto_present != (key in raw_positive_keys[row.method]):
            raise ValueError(f"Raw JSONL/loader disagreement: {row.method}, {key}")
    trajectory = aligned.groupby(["method", "game_id"], as_index=False)[["manual_present", "auto_present"]].any()
    result = pd.DataFrame([
        dict(method=m, judge=METHODS[m], behavior_id=BEHAVIOR, level="Trajectory",
             **metrics(g.manual_present.to_numpy(bool), g.auto_present.to_numpy(bool)))
        for m, g in trajectory.groupby("method")
    ]).set_index("method").loc[list(METHODS)].reset_index()
    if not result.n.eq(20).all() or not result.positives.eq(8).all():
        raise ValueError("Expected 20 trajectories with eight human-positive trajectories")
    values = result[["precision", "recall", "f1"]].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Invalid PH performance metrics")
    return aligned, trajectory, result


def build_figure(result: pd.DataFrame):
    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 1.95), layout="constrained")
    overall = result.set_index("method")
    x, width = np.arange(3), .2
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        values = overall.loc[method, ["precision", "recall", "f1"]].to_numpy(float)
        positions = x+(i-1.5)*width
        ax.bar(positions, values, width*.92, color=color, label=METHODS[method], zorder=3)
        for xx, yy in zip(positions, values):
            if yy == 0:
                ax.annotate("0", (xx, 0), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color=ps.MUTED)
    ax.set_xticks(x, ["Precision", "Recall", "F1"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ps.ygrid(ax)
    ax.set_yticks(np.arange(0, 1.01, .25))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
    return fig


def main() -> None:
    aligned, trajectory, result = load_data()
    fig = build_figure(result)
    OUT.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(OUT/f"{STEM}_aligned_labels.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(OUT/f"{STEM}_trajectory_labels.csv", index=False, encoding="utf-8-sig")
    result.to_csv(OUT/f"{STEM}_metrics.csv", index=False, encoding="utf-8-sig")
    fig.savefig(OUT/f"{STEM}.pdf", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    ps.save(fig, None, OUT/f"{STEM}.png")
    plt.close(fig)
    print(result[["judge", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
