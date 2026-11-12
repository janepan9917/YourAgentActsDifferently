"""Typed model-output retry state shared by agents, users, and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import uuid


class RetryMode(str, Enum):
    INITIAL = "initial"
    RESAMPLE = "resample"
    FORMAT_REPAIR = "format_repair"
    FRESH_DECISION = "fresh_decision"
    PROTOCOL_CORRECTION = "protocol_correction"
    PROPOSAL_CORRECTION = "proposal_correction"


class ModelOutputError(Exception):
    """Base class for errors attributable to a model-generated response."""

    def __init__(
        self,
        message: str,
        *,
        retry_mode: RetryMode,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_mode = retry_mode
        self.raw_response = raw_response


class UnrecoverableModelFormatError(ModelOutputError):
    """A malformed response whose original semantic action cannot be preserved."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(
            message,
            retry_mode=RetryMode.RESAMPLE,
            raw_response=raw_response,
        )


class ModelOutputParseError(ModelOutputError):
    """A parse/validation error caused by the current model response."""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(
            message,
            retry_mode=RetryMode.RESAMPLE,
            raw_response=raw_response,
        )


class ModelOutputAttemptsExceeded(ModelOutputError):
    """The model failed to produce a valid action within one decision cycle."""

    def __init__(self, budget: "ModelOutputAttemptBudget") -> None:
        super().__init__(
            (
                "Model output attempt limit "
                f"({budget.max_attempts}) exceeded for decision cycle "
                f"{budget.decision_cycle_id}."
            ),
            retry_mode=budget.next_retry_mode,
            raw_response=budget.last_raw_response,
        )
        self.budget = budget


@dataclass
class ModelOutputAttemptBudget:
    """Shared logical-completion budget for one expected model action."""

    max_attempts: int = 8
    used_attempts: int = 0
    decision_cycle_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    actor_id: str = ""
    next_retry_mode: RetryMode = RetryMode.INITIAL
    last_retry_mode: RetryMode | None = None
    last_error_message: str = ""
    last_raw_response: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")

    @property
    def remaining_attempts(self) -> int:
        return max(0, self.max_attempts - self.used_attempts)

    @property
    def exhausted(self) -> bool:
        return self.used_attempts >= self.max_attempts

    def set_next_retry_mode(self, retry_mode: RetryMode) -> None:
        self.next_retry_mode = retry_mode

    def record_failure(
        self,
        error: BaseException | str,
        *,
        raw_response: str | None,
        next_retry_mode: RetryMode,
    ) -> None:
        self.last_error_message = str(error)
        self.last_raw_response = raw_response
        self.next_retry_mode = next_retry_mode

    def consume(self, retry_mode: RetryMode | None = None) -> int:
        if self.exhausted:
            raise ModelOutputAttemptsExceeded(self)
        resolved_mode = retry_mode or self.next_retry_mode
        self.used_attempts += 1
        self.last_retry_mode = resolved_mode
        self.next_retry_mode = RetryMode.INITIAL
        return self.used_attempts

    def log_context(self) -> dict[str, str | int]:
        return {
            "decision_cycle_id": self.decision_cycle_id,
            "model_output_attempt": self.used_attempts,
            "max_model_output_attempts": self.max_attempts,
            "retry_mode": (
                self.last_retry_mode.value
                if self.last_retry_mode is not None
                else RetryMode.INITIAL.value
            ),
            "decision_actor_id": self.actor_id,
        }
