"""Prompt-facing negotiation turn status helpers."""

from __future__ import annotations

from typing import Any


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _without_redundant_dynamic_state(game_state: str | None) -> str:
    """Remove dynamic state already represented elsewhere in the prompt.

    Exact proposals live in conversation history, and the canonical public-action
    counter is rendered by :func:`format_game_state_with_flow_status`.
    """
    if not game_state:
        return ""
    return "\n".join(
        line
        for line in str(game_state).splitlines()
        if not line.lstrip().lower().startswith(
            ("last proposal:", "last proposals:", "turn:")
        )
    ).strip()


def format_game_state_with_flow_status(
    game_state: str | None,
    *,
    current_turn: Any | None = None,
    max_turns: Any = None,
    current_agent_user_messages: Any | None = None,
    max_agent_user_messages: Any | None = None,
) -> str:
    """Prepend explicit negotiation and optional private-dialogue status."""
    turn_num = max(0, _coerce_int(current_turn, 0))
    lines = ["[NEGOTIATION ACTION STATUS]"]
    if max_turns is None:
        if turn_num == 0:
            lines.append("No public agent-agent action is currently in progress.")
        else:
            lines.extend(
                [
                    f"Current public agent-agent action: {turn_num}.",
                    "The current public action can still complete a deal, including "
                    "through any final confirmation it initiates.",
                ]
            )
        lines.append("The total public-action limit is not available in this prompt.")
    else:
        max_turns_int = max(0, _coerce_int(max_turns, 0))
        further_actions = max(0, max_turns_int - turn_num)
        if turn_num == 0:
            lines.extend(
                [
                    "No public agent-agent action is currently in progress.",
                    (
                        "Total public agent-agent actions available: "
                        f"{max_turns_int}."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "Current public agent-agent action: "
                        f"{turn_num} of {max_turns_int}."
                    ),
                    (
                        "Further public agent-agent actions available after the "
                        f"current action: {further_actions}."
                    ),
                    "The current public action can still complete a deal, including "
                    "through any final confirmation it initiates.",
                ]
            )
        lines.extend(
            [
                (
                    "The total action limit is exhausted only after public "
                    f"agent-agent action {max_turns_int} finishes."
                ),
                (
                    "If no deal has been completed then, the negotiation ends "
                    "without agreement."
                ),
            ]
        )
    if current_agent_user_messages is not None and max_agent_user_messages is not None:
        used_messages = max(0, _coerce_int(current_agent_user_messages, 0))
        message_limit = max(0, _coerce_int(max_agent_user_messages, 0))
        remaining_messages = max(0, message_limit - used_messages)
        lines.extend(
            [
                "",
                "[ASK USER BUDGET]",
                (
                    "Private messages used during ASK USER by this pair: "
                    f"{used_messages} of {message_limit}."
                ),
                (
                    "Private ASK USER messages remaining for this pair across the game: "
                    f"{remaining_messages}."
                ),
                "Each message from either you or your user during ASK USER consumes this budget.",
            ]
        )
    static_game_state = _without_redundant_dynamic_state(game_state)
    if static_game_state:
        lines.extend(["", static_game_state])
    return "\n".join(lines)
