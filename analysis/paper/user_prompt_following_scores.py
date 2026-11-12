"""Prompt-following scores of the candidate user models in Dual User Ultimatum, per role, run and model (used by Figure 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


from negsim.log_paths import iter_results_paths  # noqa: E402
from negsim.run_outcomes import is_evaluation_excluded_outcome  # noqa: E402


EVAL_STEM = "user_prompt_following_eval.csv"
EXTRA_SWEEP_IDS = ("n100_ultimatum_gpt56terra",)
JUDGE_SPECS = (
    {
        "judge_id": "gpt56sol_medium",
        "judge_label": "GPT-5.6 Sol judge",
        "suffix": "_user_prompt_following_single_gpt56sol_medium",
    },
    {
        "judge_id": "claudeopus5_default",
        "judge_label": "Claude Opus 5 judge",
        "suffix": "_user_prompt_following_single_claudeopus5_default",
    },
)
CRITERIA = (
    ("UF_RULE_FOLLOWING", "Rule following"),
    ("UF_TERMS_UNDERSTANDING", "Terms understanding"),
)
CRITERION_IDS = tuple(item[0] for item in CRITERIA)
CRITERION_LABELS = dict(CRITERIA)
USER_MODEL_LABELS = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-5.6-terra": "GPT-5.6 Terra",
}


def _model_label(model: str) -> str:
    return USER_MODEL_LABELS.get(model, model)


def _read_result_index(logs_root: Path, sweep_id: str) -> dict[str, dict[str, Any]]:
    result_by_game: dict[str, dict[str, Any]] = {}
    for path in iter_results_paths(logs_root, sweep_id=sweep_id):
        result = json.loads(path.read_text(encoding="utf-8"))
        if is_evaluation_excluded_outcome(result.get("outcome")):
            continue
        game_id = str(result.get("game_id") or path.parent.name)
        if game_id in result_by_game:
            raise ValueError(f"Duplicate game_id {game_id!r} in {sweep_id}")
        result_by_game[game_id] = result
    return result_by_game


def _discover_eval_dirs(eval_root: Path) -> list[tuple[Path, dict[str, str]]]:
    discovered: list[tuple[Path, dict[str, str]]] = []
    for judge in JUDGE_SPECS:
        suffix = judge["suffix"]
        paths = {
            path
            for path in eval_root.glob(f"n100_ultimatum_*_as_user{suffix}")
            if path.is_dir() and "_failed" not in path.name
        }
        for sweep_id in EXTRA_SWEEP_IDS:
            path = eval_root / f"{sweep_id}{suffix}"
            if path.is_dir():
                paths.add(path)
        discovered.extend((path, judge) for path in sorted(paths))
    return discovered


def load_role_scores(
    *,
    logs_root: Path,
    eval_root: Path,
    expected_runs: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_dirs = _discover_eval_dirs(eval_root)
    if not eval_dirs:
        raise FileNotFoundError(f"No Ultimatum mixed-user evaluations under {eval_root}")

    for eval_dir, judge in eval_dirs:
        sweep_id = eval_dir.name[: -len(judge["suffix"])]
        csv_path = eval_dir / EVAL_STEM
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        frame = pd.read_csv(csv_path)
        required = {"game_id", "user_id", "criterion_id", "score_1_to_5"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        if "error" in frame.columns:
            errors = frame["error"].fillna("").astype(str).str.strip()
            if errors.ne("").any():
                raise ValueError(f"{csv_path}: main output contains error rows")
        frame["game_id"] = frame["game_id"].astype(str)
        frame["user_id"] = frame["user_id"].astype(str)
        frame["criterion_id"] = frame["criterion_id"].astype(str)
        frame["score_1_to_5"] = pd.to_numeric(
            frame["score_1_to_5"], errors="raise"
        )
        result_by_game = _read_result_index(logs_root, sweep_id)
        observed_game_ids = set(frame["game_id"])
        unknown_game_ids = observed_game_ids.difference(result_by_game)
        if unknown_game_ids:
            raise ValueError(
                f"{sweep_id}: evaluation contains {len(unknown_game_ids)} "
                "game_id(s) absent from eligible results"
            )
        if len(observed_game_ids) != expected_runs:
            raise ValueError(
                f"{sweep_id}: expected {expected_runs} runs, found "
                f"{len(observed_game_ids)}"
            )
        identities = frame[["game_id", "user_id", "criterion_id"]]
        if identities.duplicated().any():
            raise ValueError(f"{csv_path}: duplicate game/user/criterion rows")
        expected_rows = expected_runs * 2 * len(CRITERION_IDS)
        if len(frame) != expected_rows:
            raise ValueError(
                f"{csv_path}: expected {expected_rows} rows, found {len(frame)}"
            )
        if set(frame["user_id"]) != {"user_1", "user_2"}:
            raise ValueError(f"{csv_path}: expected user_1 and user_2")
        if set(frame["criterion_id"]) != set(CRITERION_IDS):
            raise ValueError(f"{csv_path}: unexpected criterion inventory")
        if not frame["score_1_to_5"].between(1, 5).all():
            raise ValueError(f"{csv_path}: scores must be in [1, 5]")

        for row in frame.to_dict(orient="records"):
            result = result_by_game[row["game_id"]]
            models = result.get("models") or {}
            user_models = {
                str(models.get("user_1") or ""),
                str(models.get("user_2") or ""),
            }
            agent_models = {
                str(models.get("agent_1") or ""),
                str(models.get("agent_2") or ""),
            }
            if len(user_models) != 1 or "" in user_models:
                raise ValueError(
                    f"{sweep_id}/{row['game_id']}: users do not share one model"
                )
            if len(agent_models) != 1 or "" in agent_models:
                raise ValueError(
                    f"{sweep_id}/{row['game_id']}: agents do not share one model"
                )
            rows.append(
                {
                    "sweep_id": sweep_id,
                    "game_id": row["game_id"],
                    "agent_model": next(iter(agent_models)),
                    "judge_id": judge["judge_id"],
                    "judge_label": judge["judge_label"],
                    "user_model": next(iter(user_models)),
                    "user_model_label": _model_label(next(iter(user_models))),
                    "user_id": row["user_id"],
                    "criterion_id": row["criterion_id"],
                    "criterion_label": CRITERION_LABELS[row["criterion_id"]],
                    "score_1_to_5": float(row["score_1_to_5"]),
                }
            )
    return pd.DataFrame(rows)


def build_run_scores(role_scores: pd.DataFrame) -> pd.DataFrame:
    role_counts = role_scores.groupby(
        ["sweep_id", "game_id", "agent_model", "judge_id", "judge_label", "user_model", "criterion_id"]
    )["user_id"].nunique()
    if not role_counts.eq(2).all():
        raise ValueError("Every run/criterion must contain both user roles")
    dimension = (
        role_scores.groupby(
            ["sweep_id", "game_id", "agent_model", "judge_id", "judge_label", "user_model", "criterion_id"],
            as_index=False,
        )["score_1_to_5"]
        .mean()
        .rename(columns={"score_1_to_5": "role_averaged_score"})
    )
    pivot = dimension.pivot(
        index=["sweep_id", "game_id", "agent_model", "judge_id", "judge_label", "user_model"],
        columns="criterion_id",
        values="role_averaged_score",
    ).reset_index()
    if any(criterion_id not in pivot for criterion_id in CRITERION_IDS):
        raise ValueError("Missing criterion after role averaging")
    pivot["overall_score"] = pivot[list(CRITERION_IDS)].mean(axis=1)
    pivot["user_model_label"] = pivot["user_model"].map(_model_label)
    return pivot


def _mean_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    array = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    n = len(array)
    value = float(array.mean())
    if n < 2 or np.isclose(array.std(ddof=1), 0.0):
        return n, value, value, value, float(array.std(ddof=1)) if n > 1 else 0.0
    sd = float(array.std(ddof=1))
    margin = float(student_t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return n, value, max(1.0, value - margin), min(5.0, value + margin), sd


def build_summary(run_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (user_model, judge_id, judge_label), group in run_scores.groupby(
        ["user_model", "judge_id", "judge_label"], sort=False
    ):
        dimensions = [("overall", "Overall", group["overall_score"])] + [
            (criterion_id, label, group[criterion_id])
            for criterion_id, label in CRITERIA
        ]
        for dimension_id, dimension_label, values in dimensions:
            n, value, low, high, sd = _mean_ci(values)
            rows.append(
                {
                    "user_model": user_model,
                    "user_model_label": _model_label(user_model),
                    "judge_id": judge_id,
                    "judge_label": judge_label,
                    "dimension_id": dimension_id,
                    "dimension_label": dimension_label,
                    "n_runs": n,
                    "n_agent_models_pooled": int(group["agent_model"].nunique()),
                    "mean_score": value,
                    "ci95_low": low,
                    "ci95_high": high,
                    "sd": sd,
                }
            )
    return pd.DataFrame(rows)
