"""
Action: abstract base for conversation actions. AgentAction and UserAction derive from it.
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Any, Union


class Action(ABC):
    """Abstract base class for actions in the conversation."""

    def __init__(
        self,
        action_type: Union["AgentActionType", "UserActionType"],
        output: str = "",
        thinking: str = "",
        action_details: Any = None,
    ) -> None:
        self.action_type = action_type
        self.output = output
        self.thinking = thinking
        self.action_details = action_details


class AgentActionType(str, Enum):
    QUIT = "QUIT"
    PROPOSE = "PROPOSE"
    MESSAGE_USER = "MESSAGE_USER"
    MESSAGE_USER_AND_CONTINUE_NEGOTIATION = (
        "MESSAGE_USER_AND_CONTINUE_NEGOTIATION"
    )
    ACCEPT = "ACCEPT"
    ASK_USER = "ASK USER"
    ACCEPT_AND_CONFIRM = "ACCEPT AND CONFIRM WITH USER"
    REQUEST_USER_CONFIRMATION = "REQUEST_USER_CONFIRMATION"


class AgentAction(Action):
    """Action from an agent."""

    def __init__(
        self,
        action_type: AgentActionType,
        output: str = "",
        thinking: str = "",
        action_details: Any = None,
        missing_details: bool = False,
    ) -> None:
        super().__init__(action_type, output, thinking, action_details)
        self.missing_details = missing_details


class UserActionType(str, Enum):
    CONSENT = "CONSENT"
    FURTHER_INSTRUCT = "FURTHER_INSTRUCT"
    DONT_CONSENT = "DONT_CONSENT"


class UserAction(Action):
    """Action from a user: consent, further instruct, or don't consent."""

    def __init__(
        self,
        action_type: UserActionType,
        output: str = "",
        thinking: str = "",
        action_details: Any = None,
    ) -> None:
        super().__init__(action_type, output, thinking, action_details)
