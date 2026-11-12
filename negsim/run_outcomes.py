"""Canonical terminal-outcome classification shared across run tooling."""

from __future__ import annotations


FAILED_RUN_OUTCOMES = frozenset({"error", "invalid_model_output"})
RETRYABLE_RUN_OUTCOMES = FAILED_RUN_OUTCOMES | frozenset(
    {"invalid", "interrupted"}
)
EVALUATION_EXCLUDED_OUTCOMES = RETRYABLE_RUN_OUTCOMES


def normalize_run_outcome(outcome: object) -> str:
    """Return the normalized string used for terminal-outcome comparisons."""
    if outcome is None:
        return ""
    return str(outcome).strip().lower()


def is_failed_run_outcome(outcome: object) -> bool:
    """Return whether *outcome* represents an experiment-execution failure."""
    return normalize_run_outcome(outcome) in FAILED_RUN_OUTCOMES


def is_retryable_run_outcome(outcome: object) -> bool:
    """Return whether a sweep should archive and retry the complete run."""
    return normalize_run_outcome(outcome) in RETRYABLE_RUN_OUTCOMES


def is_evaluation_excluded_outcome(outcome: object) -> bool:
    """Return whether evaluator and analysis inputs must exclude the run."""
    return normalize_run_outcome(outcome) in EVALUATION_EXCLUDED_OUTCOMES
