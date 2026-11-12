from negsim.run_outcomes import (
    EVALUATION_EXCLUDED_OUTCOMES,
    FAILED_RUN_OUTCOMES,
    RETRYABLE_RUN_OUTCOMES,
    is_evaluation_excluded_outcome,
    is_failed_run_outcome,
    is_retryable_run_outcome,
    normalize_run_outcome,
)


def test_normalize_run_outcome() -> None:
    assert normalize_run_outcome(None) == ""
    assert normalize_run_outcome(" Invalid_Model_Output ") == "invalid_model_output"


def test_failed_run_outcomes_are_canonical() -> None:
    assert FAILED_RUN_OUTCOMES == frozenset({"error", "invalid_model_output"})
    assert is_failed_run_outcome(" ERROR ")
    assert is_failed_run_outcome("Invalid_Model_Output")
    assert not is_failed_run_outcome("limit_exceeded")
    assert not is_failed_run_outcome("invalid")


def test_retry_and_evaluation_exclusion_outcomes_include_invalid_and_interrupted() -> None:
    expected = frozenset(
        {"error", "invalid_model_output", "invalid", "interrupted"}
    )
    assert RETRYABLE_RUN_OUTCOMES == expected
    assert EVALUATION_EXCLUDED_OUTCOMES == expected
    for outcome in expected:
        assert is_retryable_run_outcome(outcome)
        assert is_evaluation_excluded_outcome(outcome)
    assert not is_retryable_run_outcome("limit_exceeded")
    assert not is_evaluation_excluded_outcome("quit")
