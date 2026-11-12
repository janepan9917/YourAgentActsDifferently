"""The equal-weight estimator shared by the paper tables (misaligned-behavior rates and mean payout changes), so the
same quantity gets the same CI wherever it appears. Other results use the methods described in their own scripts.

Two-stage, equal-weight means; uncertainty is run-to-run sampling error only (games and models are fixed: the results
describe these 7 games and 8 models, not a population of games or models):

  stage 1  one value per (game, model) cell with its sampling SE: a rate (binomial SE), a mean or a difference of
           means (Welch SE), or a difference between conditions (SEs combined in quadrature)
  stage 2  per game  = equal-weight mean over the models, SE = sqrt(sum SE^2) / k
  stage 3  overall   = equal-weight mean over the games,  SE = sqrt(sum SE^2) / k
  per model          = equal-weight mean over the games,  SE = sqrt(sum SE^2) / k

95% CIs and two-sided p-values are normal (z) based. Comparisons are made on paired differences computed within the
same cell and then averaged the same way; "significant" means the 95% CI of the difference excludes 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

Z = float(stats.norm.ppf(0.975))


def z_summary(mean: float, se: float, k: int = 1) -> dict:
    """95% normal CI and two-sided p for an estimate with standard error se."""
    if not np.isfinite(mean):
        return dict(mean=np.nan, se=np.nan, ci_low=np.nan, ci_high=np.nan, pvalue=np.nan, k=0)
    if not np.isfinite(se) or se <= 0:
        return dict(mean=float(mean), se=0.0, ci_low=float(mean), ci_high=float(mean), pvalue=float(mean == 0), k=k)
    return dict(mean=float(mean), se=float(se), ci_low=float(mean - Z * se), ci_high=float(mean + Z * se),
                pvalue=float(2 * stats.norm.sf(abs(mean) / se)), k=k)


def combine(estimates, ses) -> dict:
    """Equal-weight mean of independent estimates; SE = sqrt(sum SE^2) / k. Missing estimates are skipped."""
    e = pd.Series(estimates, dtype=float).reset_index(drop=True)
    s = pd.Series(ses, dtype=float).reset_index(drop=True)
    ok = e.notna() & s.notna()
    e, s = e[ok], s[ok]
    k = len(e)
    if k == 0:
        return z_summary(np.nan, np.nan, 0)
    return z_summary(float(e.mean()), float(np.sqrt((s ** 2).sum()) / k), k)


def two_stage(cells: pd.DataFrame, est: str = "est", se: str = "se") -> tuple[pd.DataFrame, dict]:
    """cells: one row per (game, model) with est/se -> (per-game summaries indexed by game, overall summary)."""
    per_game = pd.DataFrame({g: combine(d[est], d[se]) for g, d in cells.groupby("game")}).T
    return per_game, combine(per_game["mean"], per_game["se"])


def binomial_cells(units: pd.DataFrame, keys: list[str], flag: str = "flagged") -> pd.DataFrame:
    """Per cell: rate (%) and its binomial SE (%), with (x + 0.5) / (n + 1) in the SE so a 0% cell keeps a nonzero SE."""
    c = units.groupby(keys)[flag].agg(["sum", "count"])
    p_adj = (c["sum"] + 0.5) / (c["count"] + 1)
    return pd.DataFrame(dict(est=c["sum"] / c["count"] * 100, se=np.sqrt(p_adj * (1 - p_adj) / c["count"]) * 100,
                             n=c["count"])).reset_index()


def paired_cells(cells: pd.DataFrame, by: str, a: str, b: str, keys: list[str]) -> pd.DataFrame:
    """Within each cell (keys), estimate of `a` minus `b` (values of column `by`), SEs combined in quadrature."""
    present = set(cells[by])
    if a not in present or b not in present:
        return pd.DataFrame(columns=[*keys, "est", "se"])
    w = cells.pivot_table(index=keys, columns=by, values=["est", "se"])
    out = pd.DataFrame(dict(est=w[("est", a)] - w[("est", b)], se=np.sqrt(w[("se", a)] ** 2 + w[("se", b)] ** 2)))
    return out.dropna().reset_index()


def is_significant(summary: dict) -> bool:
    return bool(np.isfinite(summary["ci_low"]) and not (summary["ci_low"] <= 0 <= summary["ci_high"]))
