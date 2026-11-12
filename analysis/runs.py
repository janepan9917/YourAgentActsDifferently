"""Read the runs and judge labels into one table with a row per agent per run.

This is the only place that touches ``data/raw`` and ``data/labels``; everything else
starts from the table built here (saved as ``data/processed/agent_runs.csv.gz``).
"""
from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ALL_LABELS, FAILED_OUTCOMES, MODELS, REPO, data_dir

# condition in data/manifest.csv -> condition name used by the analysis
CONDITIONS = {"no_user": "no_user", "single_user": "single_user", "dual_user": "two_users",
              "passive_user": "silent", "user_model_robustness": "user_model_robustness"}
ONE_SIDED = {"single_user", "silent", "user_model_robustness"}
STRATEGY = ["open_extremity", "mean_position", "final_position", "total_concession", "mean_concession",
            "concession_freq", "cushion_given_up", "unilateral_conc", "n_own_offers"]
RED, BLUE = "Player RED Gives", "Player BLUE Gives"
PB_GAMES = {"pb-capit-anchored-01": "pb_capit_anchored_01", "pb-leak-price-01": "pb_leak_price_01",
            "pb-sanity-negot-01": "pb_sanity_negot_01"}
CASINO_POINTS = {"high": 5, "medium": 4, "low": 3}


def game_key(result: dict) -> str:
    name = result["game_name"]
    if name == "principal_bench_stripped":
        return PB_GAMES[result["game_params"]["setting_id"]]
    return {"buy_sell_harness_fixed": "buy_sell", "ultimatum_harness_fixed": "ultimatum"}.get(name, name)


@lru_cache(maxsize=None)
def pb_sidecar(setting_id: str) -> dict:
    path = REPO / "configs" / "principal_bench" / "proposals" / f"{setting_id}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def pb_value(setting_id: str, role_id: str, proposal: dict) -> float:
    """Linear PrincipalBench payout without the eligibility gates."""
    formula = pb_sidecar(setting_id)["payout"]["formulas"][role_id]
    return float(formula.get("constant", 0)) + sum(
        float(coef) * float(proposal[field]) for field, coef in formula.get("terms", {}).items())


def values(result: dict, proposal: dict) -> tuple[float, float]:
    """Payouts (user_1, user_2) of a proposal in this run's game."""
    game = result["game_name"]
    if game == "ultimatum_harness_fixed":
        n = float(proposal[RED]["Dollars"])
        return float(result["game_params"].get("player_1_dollars", 100)) - n, n
    if game == "buy_sell_harness_fixed":
        seller, buyer = result["private_info"]["user_1"], result["private_info"]["user_2"]
        n, qty = float(proposal[BLUE]["ZUP"]), float(seller.get("quantity", 1))
        return n - float(seller["cost"]) * qty, float(buyer["willingness_to_pay"]) * qty - n
    if game == "principal_bench_stripped":
        setting = result["game_params"]["setting_id"]
        return pb_value(setting, "party_1", proposal), pb_value(setting, "party_2", proposal)
    private = result["private_info"]
    if game == "casino":
        vals = [{k: CASINO_POINTS[v] for k, v in private[f"user_{i}"]["priorities"].items()} for i in (1, 2)]
    elif game == "deal_or_no_deal":
        vals = [private[f"user_{i}"]["item_values"] for i in (1, 2)]
    else:
        raise ValueError(game)
    return tuple(sum(float(proposal[f"user_{i}"][k]) * float(v) for k, v in vals[i - 1].items())
                 for i in (1, 2))


def payouts(result: dict) -> tuple[list[float], list[float]]:
    """(recorded payouts, ungated payouts). Only PrincipalBench deals differ: the paper
    reports the linear payout without the setting's eligibility gate."""
    recorded = [float(result["payouts"][f"user_{i}"]) for i in (1, 2)]
    if result["outcome"] != "deal" or result["game_name"] != "principal_bench_stripped":
        return recorded, recorded
    return recorded, list(values(result, result["final_deal"]))


def reservation(result: dict, role: str) -> float:
    """No-deal payoff of ``role``: 0 in Buy-Sell, Ultimatum and Deal or No Deal, 5 in
    CaSiNo, and the sidecar's no-deal value in PrincipalBench."""
    game = result["game_name"]
    if game == "principal_bench_stripped":
        sidecar = pb_sidecar(result["game_params"]["setting_id"])
        return float(sidecar["payout"]["no_deal"]["party_1" if role == "agent_1" else "party_2"])
    return 5.0 if game == "casino" else 0.0


def strategy_metrics(result: dict, role: str) -> dict:
    """Opening offer and concession measures from the agent's own proposals.

    Each proposal is scored by its share of the joint value that goes to ``role``."""
    moves = []
    for turn in result.get("negotiation_trajectory") or []:
        proposal = turn.get("proposal")
        if proposal is None:
            continue
        u1, u2 = values(result, proposal)
        if u1 + u2 <= 1e-9:
            continue
        own = u1 if role == "agent_1" else u2
        moves.append((turn["speaker"], own / (u1 + u2), own))
    p = np.array([v for s, v, _ in moves if s == role])
    u_own = np.array([u for s, _, u in moves if s == role])
    row = {m: np.nan for m in STRATEGY}
    row["n_own_offers"] = len(p)
    if len(p) == 0:
        return row
    row["open_extremity"], row["mean_position"], row["final_position"] = p[0], p.mean(), p[-1]
    if len(p) >= 2:
        c = p[:-1] - p[1:]
        row["total_concession"] = p[0] - p[-1]
        row["mean_concession"] = (p[0] - p[-1]) / (len(p) - 1)
        row["concession_freq"] = float((c > 0).mean())
        room = u_own[0] - reservation(result, role)
        if room > 1e-9:
            row["cushion_given_up"] = (u_own[0] - u_own[-1]) / room
        own_idx = [i for i, (s, _, _) in enumerate(moves) if s == role]
        opp_idx = [i for i, (s, _, _) in enumerate(moves) if s != role]
        opp_vals = np.array([moves[i][1] for i in opp_idx])
        unilateral = 0
        for k in range(1, len(p)):
            if c[k - 1] <= 0:
                continue
            prior = [j for j, oi in enumerate(opp_idx) if oi < own_idx[k]]
            if len(prior) < 2 or not (opp_vals[prior[-1]] > opp_vals[prior[-2]]):
                unilateral += 1
        row["unilateral_conc"] = unilateral
    return row


def _run_rows(job: tuple[dict, Path]) -> list[dict]:
    entry, root = job
    result = json.loads((root / entry["path"] / "results.json").read_text(encoding="utf-8-sig"))
    if result.get("outcome") in FAILED_OUTCOMES:
        return []
    condition = CONDITIONS[entry["condition"]]
    combo = entry["combo"]
    direction = combo if combo.startswith("user_on_agent_") else (
        "no_user" if condition == "no_user" else "two_users")
    gated, ungated = payouts(result)
    models = result["models"]
    rows = []
    for i in (1, 2):
        role = f"agent_{i}"
        user_side = int(combo[-1]) if condition in ONE_SIDED else None
        rows.append(dict(
            condition=condition, direction=direction, game=game_key(result),
            model=MODELS[models["agent_1"]], agent_model=models["agent_1"],
            user_model=models.get(f"user_{user_side}") if user_side else None,
            sweep_id=entry["sweep_id"], game_id=result["game_id"], role=role,
            selected=user_side is None or user_side == i,
            turns=result["turns"], outcome=result["outcome"],
            payout_gated=gated[i - 1], payout_ungated=ungated[i - 1],
            **strategy_metrics(result, role)))
    return rows


def _read_labels(path: Path) -> list[dict]:
    raw = path.read_bytes().decode("utf-8-sig")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    return pd.read_csv(io.StringIO(raw), keep_default_na=False).to_dict("records")


def _label_flags(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Judge labels (present/absent) for every agent of every run, one column per behavior."""
    index = {(r["condition"], r["sweep_id"]): root / r["path"]
             for r in csv.DictReader((root / "labels_manifest.csv").open(encoding="utf-8"))}
    reverse = {v: k for k, v in CONDITIONS.items()}
    flags = []
    for (condition, sweep_id), runs in frame.groupby(["condition", "sweep_id"]):
        ids = set(runs.game_id)
        seen = set()
        for rec in _read_labels(index[(reverse[condition], sweep_id)]):
            if rec["game_id"] not in ids or rec["behavior_id"] not in ALL_LABELS:
                continue
            key = (rec["game_id"], rec["agent_id"], rec["behavior_id"])
            assert key not in seen, key
            assert not rec.get("error") and str(rec.get("input_truncated", False)).lower() != "true", key
            seen.add(key)
            flags.append(dict(condition=condition, game_id=key[0], role=key[1], behavior=key[2],
                              present=rec["label_mb"] == "present"))
        assert len(seen) == 14 * len(ids), (condition, sweep_id, len(seen), len(ids))
    table = pd.DataFrame(flags).pivot_table(index=["condition", "game_id", "role"], columns="behavior",
                                            values="present", aggfunc="first")
    return frame.join(table, on=["condition", "game_id", "role"])


def build_agent_runs(root: Path | None = None, workers: int = 8) -> pd.DataFrame:
    root = root or data_dir()
    manifest = [r for r in csv.DictReader((root / "manifest.csv").open(encoding="utf-8"))
                if r["condition"] in CONDITIONS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [row for chunk in pool.map(_run_rows, [(e, root) for e in manifest], chunksize=64)
                for row in chunk]
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["condition", "game_id", "role"]).any()
    frame = _label_flags(frame, root)
    for behavior in ALL_LABELS:
        frame[behavior] = frame[behavior].astype(bool)
    return frame
