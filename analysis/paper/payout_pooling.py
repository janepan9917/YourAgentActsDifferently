"""Role-averaged, random-effects pooled With User payout change per game."""
import numpy as np
import pandas as pd

from _shared import random_effects_meta


def role_averaged_pooled(by_model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for game, g in by_model.groupby("game"):
        per_model = []
        for _, m in g.groupby("model"):
            assert len(m) == 2, (game, len(m))
            per_model.append((m["mean"].mean(), np.sqrt((m["se"] ** 2).sum()) / 2))
        est = random_effects_meta(np.array([a for a, _ in per_model]), np.array([b for _, b in per_model]))
        rows.append({"game": game, "mean": est["estimate"], "ci_low": est["ci_low"],
                     "ci_high": est["ci_high"], "pvalue": est["pvalue"], "k_models": est["k"]})
    return pd.DataFrame(rows).set_index("game")
