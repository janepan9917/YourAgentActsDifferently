"""Statistics helpers (Welch intervals, random-effects pooling) shared by the figure and table scripts."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t


# ---------------------------------------------------------------------------
# Welch confidence interval / p-value helpers
# ---------------------------------------------------------------------------


def welch_mean_difference_ci(
    without_user: np.ndarray,
    with_user: np.ndarray,
    *,
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Return difference, standard error, and Welch CI for with-minus-without."""
    left = np.asarray(without_user, dtype=float)
    right = np.asarray(with_user, dtype=float)
    if len(left) < 2 or len(right) < 2:
        raise ValueError("Welch confidence intervals require at least two runs per group")
    difference = float(right.mean() - left.mean())
    left_term = float(left.var(ddof=1) / len(left))
    right_term = float(right.var(ddof=1) / len(right))
    standard_error = float(np.sqrt(left_term + right_term))
    if np.isclose(standard_error, 0.0):
        return difference, standard_error, difference, difference
    denominator = left_term**2 / (len(left) - 1) + right_term**2 / (len(right) - 1)
    degrees_of_freedom = (
        (left_term + right_term) ** 2 / denominator
        if denominator > 0
        else float(len(left) + len(right) - 2)
    )
    critical = float(student_t.ppf(0.5 + confidence / 2.0, degrees_of_freedom))
    margin = critical * standard_error
    return difference, standard_error, difference - margin, difference + margin


def welch_pvalue(without_user: np.ndarray, with_user: np.ndarray) -> float:
    """Return a stable two-sided Welch p-value for two samples."""
    left = np.asarray(without_user, dtype=float)
    right = np.asarray(with_user, dtype=float)
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    difference = float(right.mean() - left.mean())
    left_term = float(left.var(ddof=1) / len(left))
    right_term = float(right.var(ddof=1) / len(right))
    variance = left_term + right_term
    if np.isclose(variance, 0.0):
        return 1.0 if np.isclose(difference, 0.0) else 0.0
    denominator = left_term**2 / (len(left) - 1) + right_term**2 / (len(right) - 1)
    degrees_of_freedom = (
        variance**2 / denominator if denominator > 0 else float(len(left) + len(right) - 2)
    )
    statistic = difference / np.sqrt(variance)
    value = float(2.0 * student_t.sf(abs(statistic), degrees_of_freedom))
    return value if np.isfinite(value) else float("nan")


def random_effects_meta(
    estimates: np.ndarray, standard_errors: np.ndarray
) -> dict[str, float]:
    """DerSimonian-Laird random-effects meta-analysis across k models' own
    with-user-vs-no-user estimates for one game.

    Random effects are used because models are expected to differ in how they
    respond to a user, not only by sampling noise around one shared effect:
    tau2 estimates the between-model variance and widens the pooled CI.
    """
    theta = np.asarray(estimates, dtype=float)
    se = np.asarray(standard_errors, dtype=float)
    k = len(theta)
    if k == 0:
        return {
            "estimate": float("nan"), "se": float("nan"), "ci_low": float("nan"),
            "ci_high": float("nan"), "pvalue": float("nan"), "tau2": float("nan"),
            "i2": float("nan"), "k": 0,
        }
    if k == 1:
        # Nothing to combine -- the single model's own estimate/SE stands,
        # with no between-model variance to speak of.
        pooled, pooled_se = float(theta[0]), float(se[0])
        if pooled_se <= 0 or not np.isfinite(pooled_se):
            ci_low = ci_high = pooled
            pvalue = 1.0 if np.isclose(pooled, 0.0) else 0.0
        else:
            critical = float(student_t.ppf(0.975, 1))
            ci_low, ci_high = pooled - critical * pooled_se, pooled + critical * pooled_se
            pvalue = float(2.0 * student_t.sf(abs(pooled / pooled_se), 1))
        return {
            "estimate": pooled, "se": pooled_se, "ci_low": ci_low, "ci_high": ci_high,
            "pvalue": pvalue, "tau2": 0.0, "i2": 0.0, "k": 1,
        }

    # A model whose metric is constant across all of its runs reports SE=0.
    # Variances are floored so that no study's weight exceeds ~10x that of the
    # most precise nonzero-variance study.
    variances = _floored_variances(se)
    fixed_weights = 1.0 / variances
    theta_fixed = float(np.sum(fixed_weights * theta) / np.sum(fixed_weights))

    q_statistic = float(np.sum(fixed_weights * (theta - theta_fixed) ** 2))
    degrees_of_freedom = k - 1
    c = float(np.sum(fixed_weights) - np.sum(fixed_weights**2) / np.sum(fixed_weights))
    tau2 = max(0.0, (q_statistic - degrees_of_freedom) / c) if c > 0 else 0.0
    i2 = max(0.0, (q_statistic - degrees_of_freedom) / q_statistic) * 100.0 if q_statistic > 0 else 0.0

    random_weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(random_weights * theta) / np.sum(random_weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(random_weights)))

    # t-distribution with k-1 degrees of freedom.
    critical = float(student_t.ppf(0.975, degrees_of_freedom))
    ci_low, ci_high = pooled - critical * pooled_se, pooled + critical * pooled_se
    pvalue = float(2.0 * student_t.sf(abs(pooled / pooled_se), degrees_of_freedom)) if pooled_se > 0 else (
        1.0 if np.isclose(pooled, 0.0) else 0.0
    )

    return {
        "estimate": pooled, "se": pooled_se, "ci_low": ci_low, "ci_high": ci_high,
        "pvalue": pvalue, "tau2": tau2, "i2": i2, "k": k,
    }


def _floored_variances(standard_errors: np.ndarray) -> np.ndarray:
    """SE=0 guard for random_effects_meta (see the comment there)."""
    variances = np.asarray(standard_errors, dtype=float) ** 2
    nonzero = variances[variances > 0]
    floor = float(nonzero.min() / 10.0) if len(nonzero) else 1.0
    return np.maximum(variances, floor)
