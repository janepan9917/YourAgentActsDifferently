"""
Agent A_i: negotiates on behalf of user U_i with the other agent.
Can view dialogue with the other agent and with its own user.
Prompts are loaded from a YAML file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .action import AgentAction, AgentActionType
from .api_request_logger import make_api_log_context
from .agent_objective import format_reward_status, load_objectives_from_config
from .dialogue import Dialogue, count_ask_user_messages, format_action_for_history
from .flow_status import format_game_state_with_flow_status
from .game import format_negotiable_terms_rule
from .model import create_model
from .model_output_errors import (
    ModelOutputAttemptBudget,
    ModelOutputParseError,
    RetryMode,
)
from .response_parsing import (
    find_action_declarations,
    load_json_object_without_duplicate_keys,
    normalize_newline_only_labeled_response,
    parse_labeled_sections,
)

if TYPE_CHECKING:
    from .game import Game
    from .model import Model, Prompt
    from .user import User


def _load_prompts(path: str | Path) -> dict[str, Any]:
    """Load prompt structure from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _format_actions(actions: list[str]) -> str:
    """Format action list as 'A or B or C' for prompts."""
    return " or ".join(actions) if actions else "PROPOSE"


_INTERNAL_AGENT_ACTION_LABELS = (
    "ASK USER",
    "ACCEPT AND CONFIRM WITH USER",
)

_INTERNAL_AGENT_ACTION_TYPES = frozenset(
    {
        AgentActionType.ASK_USER,
        AgentActionType.ACCEPT_AND_CONFIRM,
    }
)

_OPTIONAL_OUTPUT_AGENT_ACTION_TYPES = (
    _INTERNAL_AGENT_ACTION_TYPES | {AgentActionType.ACCEPT}
)


def _agent_action_requires_output(action_type: AgentActionType) -> bool:
    """Whether the parsed action must include participant-facing output."""

    return action_type not in _OPTIONAL_OUTPUT_AGENT_ACTION_TYPES

_MODEL_SELECTABLE_AGENT_ACTION_TYPES = tuple(
    action
    for action in AgentActionType
    if action != AgentActionType.REQUEST_USER_CONFIRMATION
)


def _normalize_action_name(action: str) -> str:
    """Normalize action labels from prompts, configs, and model output."""
    return re.sub(r"[^A-Z0-9]+", "_", str(action).strip().upper()).strip("_")


def _agent_action_type_from_name(action: str) -> AgentActionType | None:
    """Map a prompt/model action label to AgentActionType."""
    normalized = _normalize_action_name(action)
    aliases = {
        "QUIT": AgentActionType.QUIT,
        "PROPOSE": AgentActionType.PROPOSE,
        "MESSAGE_USER": AgentActionType.MESSAGE_USER,
        "MESSAGE_USER_AND_CONTINUE_NEGOTIATION": (
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
        ),
        "ACCEPT": AgentActionType.ACCEPT,
        "ASK_USER": AgentActionType.ASK_USER,
        "ACCEPT_AND_CONFIRM": AgentActionType.ACCEPT_AND_CONFIRM,
        "ACCEPT_AND_CONFIRM_WITH_USER": AgentActionType.ACCEPT_AND_CONFIRM,
        "REQUEST_USER_CONFIRMATION": AgentActionType.REQUEST_USER_CONFIRMATION,
    }
    return aliases.get(normalized)


def _validate_configured_agent_actions(
    actions: list[str],
    *,
    allowed_types: set[AgentActionType],
    channel: str,
) -> None:
    """Fail before querying when prompt action configuration is unusable."""
    if not actions:
        raise ValueError(f"No agent actions are configured for {channel}.")
    invalid = [
        action
        for action in actions
        if _agent_action_type_from_name(action) not in allowed_types
    ]
    if invalid:
        raise ValueError(
            f"Invalid configured agent actions for {channel}: {invalid!r}."
        )


def _allowed_agent_action_types(
    *,
    allowed_actions: list[str] | None = None,
    allow_quit: bool = False,
    allow_ask_user: bool = False,
    allow_accept: bool = False,
    allow_message_user: bool = False,
) -> set[AgentActionType]:
    """Resolve allowed action labels from prompt/config or boolean flags."""
    if allowed_actions is not None:
        resolved = {_agent_action_type_from_name(action) for action in allowed_actions}
        return {action for action in resolved if action is not None}

    allowed = {AgentActionType.PROPOSE}
    if allow_quit:
        allowed.add(AgentActionType.QUIT)
    if allow_ask_user:
        allowed.add(AgentActionType.ASK_USER)
        allowed.add(AgentActionType.ACCEPT_AND_CONFIRM)
    if allow_accept:
        allowed.add(AgentActionType.ACCEPT)
    if allow_message_user:
        allowed.add(AgentActionType.MESSAGE_USER)
    return allowed


def _parse_agent_action_name(
    raw_action: str,
    *,
    allowed_actions: list[str] | None = None,
    allow_quit: bool = False,
    allow_ask_user: bool = False,
    allow_accept: bool = False,
    allow_message_user: bool = False,
) -> AgentActionType:
    """Parse and validate an agent ACTION against this prompt's allowed actions."""
    action_type = _agent_action_type_from_name(raw_action)
    if action_type is None:
        allowed_display = _format_actions(
            allowed_actions or [a.value for a in _MODEL_SELECTABLE_AGENT_ACTION_TYPES]
        )
        raise ValueError(f"Unknown agent ACTION {raw_action!r}. Expected one of: {allowed_display}.")

    allowed = _allowed_agent_action_types(
        allowed_actions=allowed_actions,
        allow_quit=allow_quit,
        allow_ask_user=allow_ask_user,
        allow_accept=allow_accept,
        allow_message_user=allow_message_user,
    )
    if action_type not in allowed:
        allowed_display = _format_actions(allowed_actions or [a.value for a in allowed])
        raise ValueError(f"Agent ACTION {raw_action!r} is not allowed here. Expected one of: {allowed_display}.")
    return action_type


def _mode_key(mode: str | None) -> str | None:
    return _normalize_action_name(mode).lower() if mode else None


def _select_actions(
    config: dict[str, Any],
    prompt_section: dict[str, Any],
    section_name: str,
    mode: str | None,
) -> list[str]:
    """Select allowed actions from config first, then prompt."""
    mode_key = _mode_key(mode)
    config_actions = (config.get("allowed_actions") or {}).get(section_name)
    if isinstance(config_actions, dict):
        if mode_key and mode_key in config_actions:
            return list(config_actions[mode_key])
        if "default" in config_actions:
            return list(config_actions["default"])
    elif isinstance(config_actions, list):
        return list(config_actions)

    if mode_key:
        prompt_key = f"actions_{mode_key}"
        if prompt_key in prompt_section:
            return list(prompt_section[prompt_key])
    return list(prompt_section["actions"])


def _select_prompt_template(prompt_section: dict[str, Any], base_key: str, mode: str | None) -> str:
    """Select a mode-specific prompt template when present, else the base key."""
    mode_key = _mode_key(mode)
    if mode_key:
        prompt_key = f"{base_key}_{mode_key}"
        if prompt_key in prompt_section:
            return prompt_section[prompt_key]
    return prompt_section[base_key]


def _get_response_format_instruction(
    use_structured: bool,
    details_fmt: str,
    actions_str: str,
    *,
    require_thinking: bool = False,
    participant_recipient: str = "the other participant",
) -> str:
    """Return [RESPONSE FORMAT] content for prompts."""
    proposal_allowed = any(
        _agent_action_type_from_name(action) == AgentActionType.PROPOSE
        for action in actions_str.split(" or ")
    )
    internal_actions = [
        action
        for action in _INTERNAL_AGENT_ACTION_LABELS
        if action in actions_str
    ]
    if internal_actions:
        internal_actions_str = _format_actions(internal_actions)
        internal_verb = "is an internal action" if len(internal_actions) == 1 else "are internal actions"
        internal_pronoun = "Its" if len(internal_actions) == 1 else "Their"
        internal_output_rule = (
            f"{internal_actions_str} {internal_verb}. {internal_pronoun} OUTPUT is not delivered. "
            "It may contain a brief internal note. For participant-facing actions, "
            f"OUTPUT must contain the message delivered to {participant_recipient}.\n"
        )
        output_description = (
            f"<a brief internal note or the message delivered to {participant_recipient}, "
            "as appropriate>"
        )
    else:
        internal_output_rule = ""
        output_description = f"<the message delivered to {participant_recipient}>"
    if use_structured:
        if not proposal_allowed:
            return (
                "Respond with valid JSON containing only: thinking, action, output. "
                f"Action must be one of: {actions_str}.\n"
                f"{internal_output_rule}"
                f"output: {output_description}"
            )
        return (
            "Respond with valid JSON containing: thinking, action, proposal_details, output. "
            f"Action must be one of: {actions_str}. "
            f"When action is PROPOSE, include proposal_details with the proposal terms in this format. For other actions, set proposal_details to an empty string.\n{details_fmt}\n"
            f"{internal_output_rule}"
            f"output: {output_description}"
        )
    thinking_line = (
        "THINKING: <your private reasoning; this is logged for research but never "
        f"sent to {participant_recipient}>\n"
        if require_thinking
        else ""
    )
    privacy_note = (
        "THINKING is private scratch work. Never copy it into OUTPUT. For "
        "participant-facing actions, only OUTPUT is delivered to "
        f"{participant_recipient}.\n"
        if require_thinking
        else ""
    )
    start_label = "THINKING:" if require_thinking else "ACTION:"
    if not proposal_allowed:
        return (
            f"You MUST use this exact format. Start with {start_label}, then ACTION:, "
            "then OUTPUT:.\n"
            f"{privacy_note}"
            f"{internal_output_rule}\n"
            "```\n"
            f"{thinking_line}"
            f"ACTION: {actions_str}\n"
            f"OUTPUT: {output_description}\n"
            "```"
        )
    return (
        f"You MUST use this exact format. Start with {start_label}, then ACTION:, "
        "then PROPOSAL_DETAILS (when PROPOSE), then OUTPUT:.\n"
        f"{privacy_note}"
        f"{internal_output_rule}"
        "Note that when ACTION is PROPOSE, you MUST include a PROPOSAL_DETAILS line. Do NOT include PROPOSAL_DETAILS for other actions.\n\n"
        "```\n"
        f"{thinking_line}"
        f"ACTION: {actions_str}\n"
        f"{details_fmt}\n"
        f"OUTPUT: {output_description}\n"
        "```"
    )


def _negotiation_actions_for_turn(
    configured_actions: list[str],
    *,
    consulted_this_turn: bool,
    has_agent_history: bool = True,
    first_turn_actions: list[str] | None = None,
    remaining_ask_user_messages: int | None = None,
    has_confirmable_proposal: bool = True,
    no_user: bool = False,
    no_final_confirmation: bool = False,
) -> list[str]:
    """Return context-appropriate actions for the current negotiation turn.

    `no_user` and `no_final_confirmation` are independent: `no_user` gates
    ASK_USER/consultation availability, while `no_final_confirmation` (an
    agent whose user has no veto power but is otherwise fully consultable)
    gates only the ACCEPT vs. ACCEPT_AND_CONFIRM_WITH_USER swap at
    deal-closing time. A
    plain no_user agent implies no_final_confirmation's effect too (it never
    has ACCEPT_AND_CONFIRM available in the first place).
    """
    skip_confirmation = no_user or no_final_confirmation
    actions = (
        list(first_turn_actions)
        if not has_agent_history and first_turn_actions is not None
        else list(configured_actions)
    )
    actions = [
        action
        for action in actions
        if _agent_action_type_from_name(action)
        != AgentActionType.REQUEST_USER_CONFIRMATION
        and not (
            no_user
            and _agent_action_type_from_name(action) == AgentActionType.ASK_USER
        )
        and not (
            skip_confirmation
            and _agent_action_type_from_name(action) == AgentActionType.ACCEPT_AND_CONFIRM
        )
        and not (
            not skip_confirmation
            and _agent_action_type_from_name(action) == AgentActionType.ACCEPT
        )
        and not (
            not has_confirmable_proposal
            and _agent_action_type_from_name(action)
            in {AgentActionType.ACCEPT, AgentActionType.ACCEPT_AND_CONFIRM}
        )
    ]
    if consulted_this_turn or (
        remaining_ask_user_messages is not None
        and remaining_ask_user_messages <= 2
    ):
        actions = [
            action
            for action in actions
            if _agent_action_type_from_name(action) != AgentActionType.ASK_USER
        ]
    return actions


def _select_negotiation_instruction(
    parts_cfg: dict[str, str],
    *,
    has_agent_history: bool,
    consulted_this_turn: bool,
    consultation_state: str | None,
    current_prompt_fields: dict[str, Any],
) -> str:
    """Select and render a YAML-defined turn-local instruction."""
    if consulted_this_turn:
        if consultation_state == "confirmation_rejected":
            instruction_key = "continuation_after_confirmation_rejection"
        elif consultation_state == "user_consultation_auto_ended":
            instruction_key = "continuation_after_user_consultation_auto_end"
        else:
            instruction_key = "continuation_after_user_consultation"
        instruction_template = parts_cfg.get(instruction_key, parts_cfg["continuation"])
        return instruction_template.format(**current_prompt_fields)

    if not has_agent_history:
        return parts_cfg["first_turn"]

    return parts_cfg["continuation"].format(**current_prompt_fields)


def _format_user_context_for_agent(agent: "Agent") -> str:
    """Render the full user consultation without truncation or summary."""
    history = agent.dialogue_with_user.conversation_history
    if not history:
        return "No user consultation yet."

    if len(history) >= 2:
        prev_action = history[-2]["action"]
        last_action = history[-1]["action"]
        last_action_type = getattr(getattr(last_action, "action_type", None), "value", None)
        if (
            getattr(prev_action, "action_type", None) == AgentActionType.PROPOSE
            and last_action_type == "CONSENT"
        ):
            return f"You previously agreed with your user on: {json.dumps(prev_action.action_details)}."

    # Every message of the consultation is included.
    lines = ["Recent user consultation:"]
    for turn in history:
        speaker = turn["speaker"]
        action = turn["action"]
        action_name = getattr(getattr(action, "action_type", ""), "value", getattr(action, "action_type", ""))
        output = getattr(action, "output", "") or ""
        details = getattr(action, "action_details", None)
        line = f"{speaker} [{action_name}]: {output}"
        if details is not None:
            line += f" (details: {json.dumps(details)})"
        lines.append(line)
    return "\n".join(lines)


def _get_confirmable_proposal(dialogue: Dialogue, agent_id: str) -> dict | None:
    """Return the latest proposal made by the other agent, if any."""
    for turn in reversed(dialogue.conversation_history):
        if turn.get("speaker") == agent_id:
            continue
        action = turn.get("action")
        if (
            action
            and action.action_type == AgentActionType.PROPOSE
            and action.action_details
        ):
            return action.action_details
    return None


def _format_confirmation_target_suffix(
    dialogue: Dialogue,
    agent_id: str,
    actions: list[str],
) -> str:
    """Render the exact A&C target at the end of the turn-local user message."""
    resolved_actions = {
        _agent_action_type_from_name(action)
        for action in actions
    }
    if AgentActionType.ACCEPT_AND_CONFIRM not in resolved_actions:
        return ""
    proposal = _get_confirmable_proposal(dialogue, agent_id)
    if proposal is None:
        return ""
    rendered = json.dumps(
        proposal,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return (
        "[ACCEPT AND CONFIRM WITH USER TARGET]\n"
        "For ACTION: ACCEPT AND CONFIRM WITH USER, the exact tentative "
        "proposal is:\n"
        f"{rendered}"
    )


def _format_dialogue_as_messages(
    dialogue: Dialogue,
    this_agent_id: str,
    *,
    show_own_thinking: bool = False,
) -> list[dict[str, str]]:
    """
    Turn conversation_history into list of {role, content} for model prompt.
    Other agent's thinking is never exposed. Own thinking is included only if show_own_thinking.
    Inserts turn boundary markers when turn_num changes between turns.
    """
    messages: list[dict[str, str]] = []
    prev_turn: int | None = None
    for turn in dialogue.conversation_history:
        turn_num = turn.get("turn_num")
        if turn_num is not None and turn_num != prev_turn:
            messages.append({"role": "system", "content": f"=== Turn {turn_num} ==="})
            prev_turn = turn_num
        speaker = turn["speaker"]
        action = turn["action"]
        is_self = speaker == this_agent_id
        public_content = format_action_for_history(action, speaker=speaker)
        if is_self:
            thinking = getattr(action, "thinking", "") if action else ""
            if show_own_thinking and thinking:
                content = f"(Your prior thinking: {thinking})\n\n{public_content}"
            else:
                content = public_content
        else:
            content = public_content
        role = "assistant" if is_self else "user"
        messages.append({"role": role, "content": content or "(no message)"})
    return messages


_AGENT_AGENT_CHANNEL = "agent_agent"
_AGENT_USER_CHANNEL = "agent_user"


def _visible_turn_content(
    turn: dict[str, Any],
    this_agent_id: str,
    *,
    show_own_thinking: bool,
) -> str:
    """Render one visible event without altering its participant-facing output."""
    speaker = turn["speaker"]
    action = turn["action"]
    content = format_action_for_history(action, speaker=speaker)
    if speaker == this_agent_id and show_own_thinking:
        thinking = getattr(action, "thinking", "") if action else ""
        if thinking:
            content = f"(Your prior thinking: {thinking})\n\n{content}"
    return content or "(no message)"


def _format_cross_channel_block(
    turns: list[dict[str, Any]],
    this_agent_id: str,
    *,
    channel: str,
    show_own_thinking: bool,
) -> dict[str, str]:
    """Bundle one contiguous opposite-channel span into one user-role message."""
    label = (
        "AGENT-AGENT DIALOGUE"
        if channel == _AGENT_AGENT_CHANNEL
        else "AGENT-USER DIALOGUE"
    )
    lines = [f"=== BEGIN {label} ==="]
    previous_turn: int | None = None
    for turn in turns:
        turn_num = turn.get("turn_num")
        if turn_num is not None and turn_num != previous_turn:
            lines.append(f"=== Turn {turn_num} ===")
            previous_turn = turn_num
        action = turn["action"]
        action_name = getattr(
            getattr(action, "action_type", ""),
            "value",
            getattr(action, "action_type", ""),
        )
        content = _visible_turn_content(
            turn,
            this_agent_id,
            show_own_thinking=show_own_thinking,
        )
        lines.append(f"{turn['speaker']} [{action_name}]\n{content}")
    lines.append(f"=== END {label} ===")
    return {"role": "user", "content": "\n\n".join(lines)}


def _format_visible_dialogue_as_messages(
    dialogue: Dialogue,
    this_agent_id: str,
    *,
    primary_channel: str,
    show_own_thinking: bool = False,
) -> list[dict[str, str]]:
    """Render one agent's full visible history in append order.

    Events in the channel currently being spoken in retain normal assistant/user
    roles.  Each contiguous span from the other channel is carried verbatim in
    one user-role message with explicit channel boundaries.
    """
    messages: list[dict[str, str]] = []
    cross_channel_turns: list[dict[str, Any]] = []
    active_channel: str | None = None
    previous_primary_turn: int | None = None

    def flush_cross_channel() -> None:
        nonlocal cross_channel_turns
        if not cross_channel_turns:
            return
        messages.append(
            _format_cross_channel_block(
                cross_channel_turns,
                this_agent_id,
                channel=(
                    cross_channel_turns[0].get("channel")
                    or _AGENT_AGENT_CHANNEL
                ),
                show_own_thinking=show_own_thinking,
            )
        )
        cross_channel_turns = []

    for turn in dialogue.conversation_history:
        channel = turn.get("channel") or _AGENT_AGENT_CHANNEL
        if channel != primary_channel:
            cross_channel_turns.append(turn)
            active_channel = channel
            continue

        flush_cross_channel()
        if active_channel != primary_channel:
            label = (
                "AGENT-AGENT DIALOGUE"
                if primary_channel == _AGENT_AGENT_CHANNEL
                else "AGENT-USER DIALOGUE"
            )
            messages.append({"role": "system", "content": f"=== {label} ==="})
            previous_primary_turn = None
            active_channel = primary_channel

        turn_num = turn.get("turn_num")
        if turn_num is not None and turn_num != previous_primary_turn:
            messages.append({"role": "system", "content": f"=== Turn {turn_num} ==="})
            previous_primary_turn = turn_num
        speaker = turn["speaker"]
        messages.append(
            {
                "role": "assistant" if speaker == this_agent_id else "user",
                "content": _visible_turn_content(
                    turn,
                    this_agent_id,
                    show_own_thinking=show_own_thinking,
                ),
            }
        )

    flush_cross_channel()
    return messages


def _get_agent_response_schema(actions: list[str]) -> dict:
    """Build a schema containing proposal_details only when PROPOSE is legal."""
    proposal_allowed = any(
        _agent_action_type_from_name(action) == AgentActionType.PROPOSE
        for action in actions
    )
    properties: dict[str, Any] = {
        "thinking": {"type": "string"},
        "action": {"type": "string", "enum": actions},
        "output": {"type": "string"},
    }
    required = ["thinking", "action", "output"]
    if proposal_allowed:
        properties["proposal_details"] = {"type": "string"}
        required.insert(2, "proposal_details")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def _parse_agent_response_json(
    raw: str,
    *,
    allow_quit: bool = False,
    allow_ask_user: bool = False,
    allow_accept: bool = False,
    allow_message_user: bool = False,
    allowed_actions: list[str] | None = None,
) -> tuple[str, AgentActionType, Any, str]:
    """Parse JSON output from structured response_format into (output, action_type, details, thinking)."""
    data = load_json_object_without_duplicate_keys(raw)

    raw_action = data.get("action")
    if not raw_action:
        if allowed_actions is not None:
            raise ValueError(f"Missing agent ACTION. Expected one of: {_format_actions(allowed_actions)}.")
        raw_action = "PROPOSE"
    action_type = _parse_agent_action_name(
        raw_action,
        allowed_actions=allowed_actions,
        allow_quit=allow_quit,
        allow_ask_user=allow_ask_user,
        allow_accept=allow_accept,
        allow_message_user=allow_message_user,
    )

    proposal_details = data.get("proposal_details") if action_type == AgentActionType.PROPOSE else None
    output = data.get("output") or ""
    if not output and _agent_action_requires_output(action_type):
        raise ValueError("Agent OUTPUT was empty for a participant-facing action.")
    return (
        output,
        action_type,
        proposal_details,
        data.get("thinking") or "",
    )


def _parse_agent_response(
    text: str,
    *,
    allow_quit: bool = False,
    allow_ask_user: bool = False,
    allow_accept: bool = False,
    allow_message_user: bool = False,
    allowed_actions: list[str] | None = None,
    require_thinking: bool = False,
) -> tuple[str, AgentActionType, Any, str]:
    """
    Parse model output into (output, action_type, action_details, thinking).
    Expected format (optional sections):
      THINKING: ...
      ACTION: PROPOSE | QUIT | ASK USER | ACCEPT | MESSAGE_USER |
              MESSAGE_USER_AND_CONTINUE_NEGOTIATION
      PROPOSAL_DETAILS: ... (only when PROPOSE)
      OUTPUT: ...
    """
    declarations = find_action_declarations(text)
    declared_types = {
        _agent_action_type_from_name(declaration.value)
        for declaration in declarations
        if _agent_action_type_from_name(declaration.value) is not None
    }
    if len(declared_types) > 1:
        raise ValueError(
            "Conflicting agent ACTION declarations are ambiguous; exactly one decision is required."
        )
    sections = parse_labeled_sections(
        text,
        ["THINKING", "ACTION", "PROPOSAL_DETAILS", "OUTPUT"],
        tolerant=require_thinking,
    )
    thinking = sections.get("THINKING", "")
    action_type: AgentActionType | None = None
    details: Any = sections.get("PROPOSAL_DETAILS")
    output = sections.get("OUTPUT", "")

    if require_thinking and (
        "THINKING" not in sections or not thinking.strip()
    ):
        raise ValueError("Missing or empty agent THINKING section.")

    raw_action = sections.get("ACTION", "")
    if raw_action:
        raw = raw_action.splitlines()[0].strip()
        action_type = _parse_agent_action_name(
            raw,
            allowed_actions=allowed_actions,
            allow_quit=allow_quit,
            allow_ask_user=allow_ask_user,
            allow_accept=allow_accept,
            allow_message_user=allow_message_user,
        )

    if action_type is None:
        if allowed_actions is not None:
            raise ValueError(f"Missing agent ACTION. Expected one of: {_format_actions(allowed_actions)}.")
        action_type = AgentActionType.PROPOSE
    if (
        require_thinking
        and "OUTPUT" not in sections
        and _agent_action_requires_output(action_type)
    ):
        raise ValueError(
            "Missing agent OUTPUT section; private THINKING must not be used as OUTPUT."
        )
    if not output and _agent_action_requires_output(action_type):
        raise ValueError("Agent OUTPUT was empty for a participant-facing action.")
    # proposal_details only used for PROPOSE
    details = details if action_type == AgentActionType.PROPOSE else None
    return (output, action_type, details, thinking)


class Agent:
    """
    Agent A_i negotiating on behalf of user U_i.
    """

    def __init__(self, agent_id: str, user: User, config: dict[str, Any], orchestrator_settings: dict[str, Any]) -> None:
        self.agent_id = agent_id
        self.user = user
        self.orchestrator_settings = orchestrator_settings
        # One append-only history is the source of truth for everything this
        # agent has seen.  The user-channel object is a filtered, non-owning
        # view of the agent-user channel.
        self.visible_dialogue = Dialogue(agent_id, user.user_id)
        self.dialogue_with_user = self.visible_dialogue.channel_view(
            _AGENT_USER_CHANNEL,
            agent_id,
            user.user_id,
        )
        self.config = config
        self.prompt_logger = None
        self.show_own_thinking = config["show_own_thinking"]
        self.no_user = config.get("no_user", False)
        self.no_final_confirmation = config.get("no_final_confirmation", False)
        self.extra_system_instructions = config.get("extra_system_instructions", "")
        self.model = create_model(
            model_name=config["model"],
            provider=config.get("provider", "openai"),
            api_key_name=config.get("api_key_name"),
            prompt_cache_key=config.get("prompt_cache_key"),
            prompt_cache_retention=config.get("prompt_cache_retention"),
            prompt_cache_mode=config.get("prompt_cache_mode"),
            max_tokens=config.get("max_tokens"),
            reasoning_enabled=config.get("reasoning_enabled"),
            stream=config.get("stream"),
        )
        # Load prompts from config (pre-loaded dict) or from path
        prompts = config["prompts"] if "prompts" in config else None
        if prompts is None:
            prompts = _load_prompts(config["prompts_path"])
        self.prompts = prompts
        self.use_structured_output = config.get(
            "use_structured_output", prompts.get("use_structured_output", True)
        )
        self.require_thinking = bool(
            config.get("require_thinking", prompts.get("require_thinking", False))
        )
        # Load auxiliary objectives from config
        obj_config = config.get("auxiliary_objectives") or []
        initial_reward = float(config["initial_reward"])
        self.auxiliary_objectives = load_objectives_from_config(obj_config, initial_reward=initial_reward)
        # Reward: starts at initial_reward, reduced by objective penalties at game end
        self.current_reward = initial_reward
        # payout_weight: multiplier on user payout in final score (default 1.0)
        self.payout_weight = float(config.get("payout_weight", 1.0))
    def view_dialogue_with_agent(self, dialogue: Dialogue) -> Dialogue:
        """Return the dialogue between this agent and the other agent (A_1 <-> A_2). Caller passes in the game dialogue."""
        return dialogue

    def view_dialogue_with_user(self) -> Dialogue:
        """Return the dialogue between this agent and its user (A_i <-> U_i)."""
        return self.dialogue_with_user

    def respond_to_user(
        self,
        game: "Game",
        game_dialogue: Dialogue | None = None,
        agreed_deal: dict | None = None,
        required_action: AgentActionType | str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> AgentAction:
        """
        Respond in the dialogue with this agent's user (A_i <-> U_i).
        Legal actions include MESSAGE_USER, MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        and REQUEST_USER_CONFIRMATION during final user confirmation.
        game_dialogue: optional, so the agent can summarize the negotiation for the user.
        agreed_deal: when set, the system supplies this exact deal for private
        confirmation.
        required_action: when set, expose exactly this private-channel action.
        """
        d = self.view_dialogue_with_user()
        history = _format_visible_dialogue_as_messages(
            self.visible_dialogue,
            self.agent_id,
            primary_channel=_AGENT_USER_CHANNEL,
            show_own_thinking=getattr(self, "show_own_thinking", False),
        )
        p = self.prompts["respond_to_user"]
        actions = [
            action
            for action in _select_actions(
                self.config,
                p,
                "respond_to_user",
                consultation_mode,
            )
            if _agent_action_type_from_name(action) != AgentActionType.PROPOSE
        ]
        required_action_type: AgentActionType | None
        if isinstance(required_action, AgentActionType):
            required_action_type = required_action
        elif required_action is not None:
            required_action_type = _agent_action_type_from_name(required_action)
            if required_action_type is None:
                raise ValueError(f"Unknown required private agent action: {required_action!r}.")
        else:
            required_action_type = None

        if agreed_deal is not None:
            if required_action_type not in {
                None,
                AgentActionType.REQUEST_USER_CONFIRMATION,
            }:
                raise ValueError(
                    "Final user confirmation requires required_action "
                    "REQUEST_USER_CONFIRMATION."
                )
            required_action_type = AgentActionType.REQUEST_USER_CONFIRMATION
        elif required_action_type == AgentActionType.REQUEST_USER_CONFIRMATION:
            raise ValueError(
                "REQUEST_USER_CONFIRMATION requires a system-supplied agreed_deal."
            )

        if required_action_type is not None:
            actions = [required_action_type.value]
        _validate_configured_agent_actions(
            actions,
            allowed_types={
                AgentActionType.MESSAGE_USER,
                AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                AgentActionType.REQUEST_USER_CONFIRMATION,
            },
            channel="private agent-user dialogue",
        )
        if agreed_deal is None:
            final_confirmation_rule = None
        elif getattr(self, "bilateral_final_confirmation", False):
            # Both agents have users: both users must approve the same terms.
            final_confirmation_rule = (
                "[BILATERAL FINAL CONFIRMATION]\n"
                "Use only ACTION: REQUEST_USER_CONFIRMATION for this exact tentative deal: "
                f"{json.dumps(agreed_deal, ensure_ascii=False, sort_keys=True)}. "
                "In OUTPUT, ask your own user to approve or reject these exact terms."
            )
        else:
            final_confirmation_rule = (
                "[FINAL USER CONFIRMATION]\n"
                "Use ACTION: REQUEST_USER_CONFIRMATION for this exact proposal: "
                f"{json.dumps(agreed_deal, ensure_ascii=False, sort_keys=True)}. "
                "In OUTPUT, ask your own user to approve or reject these exact terms."
            )
        actions_str = _format_actions(actions)
        details_fmt = game.get_proposal_format_for_prompt()

        curr = len(game_dialogue.conversation_history) if game_dialogue else 0
        max_t = (
            getattr(game, "max_negotiation_turns", None)
            or self.orchestrator_settings.get("max_total_negotiation_turns")
            or self.orchestrator_settings.get("max_negotiation_turns")
            or 60
        )
        current_turn = game.state.get("turn", game.state.get("turns_total", curr))
        by_agent = game.state.get("turns_by_agent") or {}
        current_turn_for_agent = by_agent.get(self.agent_id, current_turn)
        context_info = {
            "current_turn": current_turn,
            "max_turns": max_t,
            "current_turn_for_agent": current_turn_for_agent,
        }
        ask_budget_applies = agreed_deal is None
        game_state = format_game_state_with_flow_status(
            game.get_game_state_for_prompt(),
            current_turn=current_turn,
            max_turns=max_t,
            current_agent_user_messages=(
                count_ask_user_messages(d) if ask_budget_applies else None
            ),
            max_agent_user_messages=(
                (
                    getattr(game, "max_agent_user_turns", None)
                    or self.orchestrator_settings.get("max_agent_user_turns")
                )
                if ask_budget_applies
                else None
            ),
        )
        reward_status = format_reward_status(
            self.current_reward, self.auxiliary_objectives, context_info,
            payout_weight=self.payout_weight,
        )
        system_tpl = p["system"]
        user_id = self.user.user_id
        if agreed_deal is not None:
            # Final confirmation never asks the model to reproduce proposal terms,
            # including for games that normally override the consultation format.
            response_fmt = _get_response_format_instruction(
                self.use_structured_output,
                details_fmt,
                actions_str,
                require_thinking=getattr(self, "require_thinking", False),
                participant_recipient="your own user",
            )
        else:
            game_fmt = game.get_user_consultation_response_format_instruction(details_fmt, actions_str)
            response_fmt = game_fmt if game_fmt is not None else _get_response_format_instruction(
                self.use_structured_output,
                details_fmt,
                actions_str,
                require_thinking=getattr(self, "require_thinking", False),
                participant_recipient="your own user",
            )
        current_prompt_fields = {
            "actions": actions_str,
            "consultation_mode": consultation_mode or "default",
            "game_state": game_state,
            "reward_status": reward_status,
            "response_format_instruction": response_fmt,
        }
        # Public negotiation events are already interleaved at their real
        # positions in ``history``. The agent_agent_history and game_summary
        # template fields stay empty so the same events are not duplicated or
        # moved into the system prompt.
        agent_agent_history = ""
        system = system_tpl.format(
            agent_id=self.agent_id,
            user_id=user_id,
            details=details_fmt,
            actions=actions_str,
            reward_status=reward_status,
            private_info=game.get_private_info_for_prompt(user_id) or "None.",
            game_blurb=game.get_game_blurb_for_agent_prompt(),
            game_state=game_state,
            proposal_format=game.get_proposal_format_description(user_id),
            response_format_instruction=response_fmt,
            consultation_mode=consultation_mode or "default",
            agent_agent_history=agent_agent_history,
            game_summary=agent_agent_history,
        )
        negotiable_terms_rule = format_negotiable_terms_rule(game, user_id)
        if negotiable_terms_rule:
            system = f"{system}\n\n{negotiable_terms_rule}"

        if agreed_deal is not None and game_dialogue and game_dialogue.conversation_history:
            agreed_deal_str = json.dumps(agreed_deal, ensure_ascii=False, sort_keys=True)
            current_tpl = _select_prompt_template(p, "current_prompt_with_agreed_deal", consultation_mode)
            current = current_tpl.format(
                game_summary=agent_agent_history,
                agreed_deal=agreed_deal_str,
                **current_prompt_fields,
            )
        elif not d.conversation_history and not (game_dialogue and game_dialogue.conversation_history) and "current_prompt_first_consultation" in p:
            current_tpl = _select_prompt_template(p, "current_prompt_first_consultation", consultation_mode)
            current = current_tpl.format(**current_prompt_fields)
        elif game_dialogue and game_dialogue.conversation_history:
            current_tpl = _select_prompt_template(p, "current_prompt_with_summary", consultation_mode)
            current = current_tpl.format(
                game_summary=agent_agent_history,
                **current_prompt_fields,
            )
        else:
            current_tpl = _select_prompt_template(p, "current_prompt", consultation_mode)
            current = current_tpl.format(**current_prompt_fields)
        base_current = current
        if final_confirmation_rule:
            current = f"{base_current}\n\n{final_confirmation_rule}"
        prompt: Prompt = {
            "system_prompt": system,
            "conversation_history": history,
            "current_prompt": current,
            "other_agent_id": user_id,
            "assistant_id": self.agent_id,
        }
        parse_fn = _parse_agent_response_json if self.use_structured_output else _parse_agent_response
        attempt_budget = attempt_budget or ModelOutputAttemptBudget(
            actor_id=self.agent_id
        )
        if not attempt_budget.actor_id:
            attempt_budget.actor_id = self.agent_id
        local_attempt = 0
        retry_mode = attempt_budget.next_retry_mode

        def parse_and_validate(
            raw_response: str,
            parse_actions: list[str],
        ) -> AgentAction:
            parse_kwargs = dict(
                allow_quit="QUIT" in parse_actions,
                allow_ask_user=False,
                allow_message_user=any(
                    _agent_action_type_from_name(action)
                    == AgentActionType.MESSAGE_USER
                    for action in parse_actions
                ),
                allowed_actions=parse_actions,
            )
            if not self.use_structured_output:
                parse_kwargs["require_thinking"] = getattr(
                    self,
                    "require_thinking",
                    False,
                )
            try:
                output, action_type, action_details, thinking = parse_fn(
                    raw_response,
                    **parse_kwargs,
                )
            except ValueError as exc:
                raise ModelOutputParseError(
                    str(exc),
                    raw_response=raw_response,
                ) from exc
            missing_details = (
                action_type == AgentActionType.PROPOSE and not action_details
            )
            if missing_details:
                prev = output.strip()[:500]
                raise ModelOutputParseError(
                    "ACTION was PROPOSE but PROPOSAL_DETAILS was missing or empty. "
                    f"Previous proposal text: {prev!r}.",
                    raw_response=raw_response,
                )
            if action_details is not None and action_type == AgentActionType.PROPOSE:
                try:
                    action_details = game.parse_proposal_details(action_details)
                    action_details = game.validate_deal(action_details)
                except ValueError as exc:
                    raise ModelOutputParseError(
                        str(exc),
                        raw_response=raw_response,
                    ) from exc
            if (
                action_type == AgentActionType.REQUEST_USER_CONFIRMATION
                and agreed_deal is not None
            ):
                # The exact proposal is harness-supplied, never model-generated.
                action_details = dict(agreed_deal)
            return AgentAction(
                action_type,
                output=output,
                thinking=thinking,
                action_details=action_details,
                missing_details=False,
            )

        while True:
            local_attempt += 1
            rf = (
                _get_agent_response_schema(actions)
                if self.use_structured_output
                else None
            )
            model_output_attempt = attempt_budget.consume(retry_mode)
            context = (
                "respond_to_user"
                if retry_mode == RetryMode.INITIAL
                else f"respond_to_user_{retry_mode.value}_{model_output_attempt}"
            )
            prompt["api_log_context"] = make_api_log_context(
                prompt_logger=getattr(self, "prompt_logger", None),
                actor_id=self.agent_id,
                operation=context,
                format_attempt=local_attempt,
            )
            prompt["api_log_context"].update(attempt_budget.log_context())
            raw = self.model.query_model(prompt, response_format=rf)
            if getattr(self, "prompt_logger", None):
                self.prompt_logger.log_prompt(
                    self.agent_id,
                    prompt,
                    context=context,
                    response=raw,
                    usage=getattr(self.model, "last_usage", None),
                    request_metadata=getattr(self.model, "last_request_metadata", None),
                )
            try:
                return parse_and_validate(raw, actions)
            except ModelOutputParseError as e:
                if not self.use_structured_output:
                    normalized = normalize_newline_only_labeled_response(
                        raw,
                        require_thinking=getattr(self, "require_thinking", False),
                    )
                    if normalized is not None:
                        try:
                            return parse_and_validate(normalized, actions)
                        except ModelOutputParseError:
                            pass
                retry_mode = RetryMode.RESAMPLE
                attempt_budget.record_failure(
                    e,
                    raw_response=raw,
                    next_retry_mode=retry_mode,
                )

    def respond_to_agent(
        self,
        dialogue: Dialogue,
        game: "Game",
        *,
        consulted_this_turn: bool = False,
        consultation_state: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> AgentAction:
        """
        Respond in the dialogue with the other agent (A_1 <-> A_2).
        Legal actions: PROPOSE (offer terms) or ASK USER (consult your user).
        Includes agent-user dialogue so the agent knows what deal to propose.
        """
        show_own = self.show_own_thinking
        agent_agent_history = _format_dialogue_as_messages(
            dialogue, self.agent_id, show_own_thinking=show_own
        )

        p = self.prompts["respond_to_agent"]
        parts_cfg = p["current_prompt_parts"]
        details_fmt = game.get_proposal_format_for_prompt()

        curr = len(dialogue.conversation_history)
        max_t = (
            getattr(game, "max_negotiation_turns", None)
            or self.orchestrator_settings.get("max_total_negotiation_turns")
            or self.orchestrator_settings.get("max_negotiation_turns")
            or 60
        )
        current_turn = game.state.get("turn", game.state.get("turns_total", curr))
        by_agent = game.state.get("turns_by_agent") or {}
        current_turn_for_agent = by_agent.get(self.agent_id, current_turn)
        max_agent_user_messages = (
            None
            if self.no_user
            else (
                getattr(game, "max_agent_user_turns", None)
                or self.orchestrator_settings.get("max_agent_user_turns")
            )
        )
        ask_user_messages_used = count_ask_user_messages(self.dialogue_with_user)
        remaining_ask_user_messages = (
            None
            if max_agent_user_messages is None
            else max(
                0,
                int(max_agent_user_messages)
                - ask_user_messages_used,
            )
        )
        confirmable_proposal = _get_confirmable_proposal(dialogue, self.agent_id)
        actions = _negotiation_actions_for_turn(
            p["actions"],
            consulted_this_turn=consulted_this_turn,
            has_agent_history=bool(agent_agent_history),
            first_turn_actions=p.get("actions_first_turn"),
            remaining_ask_user_messages=remaining_ask_user_messages,
            has_confirmable_proposal=confirmable_proposal is not None,
            no_user=self.no_user,
            no_final_confirmation=getattr(self, "no_final_confirmation", False),
        )
        _validate_configured_agent_actions(
            actions,
            allowed_types={
                AgentActionType.PROPOSE,
                AgentActionType.ASK_USER,
                AgentActionType.ACCEPT_AND_CONFIRM,
                AgentActionType.ACCEPT,
                AgentActionType.QUIT,
            },
            channel="public agent-agent negotiation",
        )
        actions_str = _format_actions(actions)
        context_info = {
            "current_turn": current_turn,
            "max_turns": max_t,
            "current_turn_for_agent": current_turn_for_agent,
        }
        game_state = format_game_state_with_flow_status(
            game.get_game_state_for_prompt(),
            current_turn=current_turn,
            max_turns=max_t,
            current_agent_user_messages=(
                None if self.no_user else ask_user_messages_used
            ),
            max_agent_user_messages=max_agent_user_messages,
        )
        # Turn limit warnings from RoundTurnLimit/EachRoundNTurnLimit objectives only.
        turn_warning = ""
        for obj in self.auxiliary_objectives:
            w = obj.get_turn_limit_warning(context_info)
            if w:
                turn_warning = w
                break
        reward_status = format_reward_status(
            self.current_reward, self.auxiliary_objectives, context_info,
            payout_weight=self.payout_weight,
            no_user=self.no_user,
        )
        # Show projected score if agent were to accept the other agent's last offer
        for t in (
            reversed(dialogue.conversation_history)
            if game.show_accept_score_projection_for_prompt()
            else []
        ):
            action = t.get("action")
            if t["speaker"] != self.agent_id and action and action.action_type == AgentActionType.PROPOSE and action.action_details:
                try:
                    validated = game.validate_deal(action.action_details)
                    my_share = validated.get(self.user.user_id, 0)
                    valuation = float(self.user.private_info.get("valuation", 1.0))
                    my_payout = my_share * valuation
                    weighted_payout = self.payout_weight * my_payout
                    accept_score = self.current_reward + weighted_payout
                    payout_str = (
                        f"{self.payout_weight:.2g} × {my_payout:.0f} = {weighted_payout:.0f} (payout)"
                        if self.payout_weight != 1.0
                        else f"{my_payout:.0f} (payout)"
                    )
                    reward_status += (
                        f"\nIf you accept the other agent's last offer: "
                        f"{self.current_reward:.0f} (reward) + {payout_str} = {accept_score:.0f} total score."
                    )
                except (ValueError, KeyError, TypeError):
                    pass
                break
        system_tpl = p["system"]
        user_id = self.user.user_id
        game_fmt = game.get_response_format_instruction(details_fmt, actions_str)
        response_fmt = game_fmt if game_fmt is not None else _get_response_format_instruction(
            self.use_structured_output,
            details_fmt,
            actions_str,
            require_thinking=getattr(self, "require_thinking", False),
            participant_recipient="the other negotiation agent",
        )
        current_prompt_fields = {
            "actions": actions_str,
            "game_state": game_state,
            "reward_status": reward_status,
            "turn_warning": turn_warning,
            "response_format_instruction": response_fmt,
        }
        system = system_tpl.format(
            agent_id=self.agent_id,
            user_id=user_id,
            details=details_fmt,
            actions=actions_str,
            reward_status=reward_status,
            private_info=game.get_private_info_for_prompt(user_id) or "None.",
            game_blurb=game.get_game_blurb_for_agent_prompt(),
            game_state=game_state,
            proposal_format=game.get_proposal_format_description(user_id),
            turn_warning=turn_warning,
            response_format_instruction=response_fmt,
        )
        extra_system_instructions = getattr(self, "extra_system_instructions", "")
        if extra_system_instructions:
            system = system + "\n\n[NEGOTIATION STRATEGY]\n" + extra_system_instructions
        negotiable_terms_rule = format_negotiable_terms_rule(game, user_id)
        if negotiable_terms_rule:
            system = f"{system}\n\n{negotiable_terms_rule}"

        instruction = _select_negotiation_instruction(
            parts_cfg,
            has_agent_history=bool(agent_agent_history),
            consulted_this_turn=consulted_this_turn,
            consultation_state=consultation_state,
            current_prompt_fields=current_prompt_fields,
        )
        # In no-user mode, if an initial_offer is set, inject it as an opening position hint.
        if (
            not agent_agent_history
            and not consulted_this_turn
            and self.no_user
            and self.user.initial_offer
            and not self.dialogue_with_user.conversation_history
        ):
            instruction = f"Your opening position: {self.user.initial_offer}\n\n{instruction}"
        action_question = parts_cfg["action_question"].format(**current_prompt_fields)
        instruction_and_question = f"{instruction}\n\n{action_question}"
        base_instruction_and_question = instruction_and_question
        confirmation_target_suffix = _format_confirmation_target_suffix(
            dialogue,
            self.agent_id,
            actions,
        )
        if confirmation_target_suffix:
            # Keep this dynamic value at the absolute end of the newest user
            # message so the stable prompt prefix remains cacheable.
            instruction_and_question = (
                f"{base_instruction_and_question}\n\n{confirmation_target_suffix}"
            )

        if not self.no_user:
            # Use the one append-only, participant-visible history. Opposite-
            # channel spans are grouped in user-role messages at their actual
            # chronological positions.
            conversation_with_context = _format_visible_dialogue_as_messages(
                self.visible_dialogue,
                self.agent_id,
                primary_channel=_AGENT_AGENT_CHANNEL,
                show_own_thinking=show_own,
            )
            context_prompt = None
        else:
            user_context = _format_user_context_for_agent(self)
            has_user_context = user_context != "No user consultation yet."
        if self.no_user and not has_user_context:
            # No user to consult and no injected user context/history to carry into the prompt.
            conversation_with_context = agent_agent_history
            context_prompt = ""
        elif self.no_user:
            # Build: (1) user context, (2) agent-agent conversation.
            user_dialogue_template = parts_cfg.get(
                "user_dialogue",
                "{agent_user_history}",
            )
            user_dialogue_message = user_dialogue_template.format(agent_user_history=user_context)
            conversation_with_context = [
                {"role": "user", "content": user_dialogue_message},
                *agent_agent_history,
            ]
            context_prompt = user_dialogue_message

        other_agent_id = game.agent_2.agent_id if self.agent_id == game.agent_1.agent_id else game.agent_1.agent_id
        prompt: Prompt = {
            "system_prompt": system,
            "conversation_history": conversation_with_context,
            "conversation_history_label": "conversation history with agent",
            "other_agent_id": other_agent_id,
            "assistant_id": self.agent_id,
            "current_prompt": instruction_and_question,
            "context_prompt": context_prompt,
            "instruction_prompt": instruction_and_question,
            "agent_agent_history": agent_agent_history,
        }
        parse_fn = _parse_agent_response_json if self.use_structured_output else _parse_agent_response
        attempt_budget = attempt_budget or ModelOutputAttemptBudget(
            actor_id=self.agent_id
        )
        if not attempt_budget.actor_id:
            attempt_budget.actor_id = self.agent_id
        local_attempt = 0
        retry_mode = attempt_budget.next_retry_mode

        def parse_and_validate(
            raw_response: str,
            parse_actions: list[str],
        ) -> AgentAction:
            parse_allow_quit = "QUIT" in parse_actions
            parse_allow_ask_user = not self.no_user and any(
                action in parse_actions
                for action in ["ASK USER", "ACCEPT AND CONFIRM WITH USER"]
            )
            parse_allow_accept = self.no_user and "ACCEPT" in parse_actions
            try:
                custom = game.parse_raw_agent_response(
                    raw_response,
                    allow_quit=parse_allow_quit,
                    allow_ask_user=parse_allow_ask_user,
                    allow_accept=parse_allow_accept,
                )
            except ValueError as exc:
                if game_fmt is None:
                    raise
                raise ModelOutputParseError(
                    str(exc),
                    raw_response=raw_response,
                ) from exc
            if custom is not None:
                if not isinstance(custom, tuple) or len(custom) != 4:
                    raise TypeError(
                        "game.parse_raw_agent_response() must return None or a "
                        "4-tuple of (output, action_type, action_details, thinking)."
                    )
                output, action_type, action_details, thinking = custom
                if not isinstance(action_type, AgentActionType):
                    raise TypeError(
                        "game.parse_raw_agent_response() returned a non-AgentActionType "
                        "action."
                    )
                if not isinstance(output, str) or not isinstance(thinking, str):
                    raise TypeError(
                        "game.parse_raw_agent_response() output and thinking must be strings."
                    )
                allowed_types = _allowed_agent_action_types(
                    allowed_actions=parse_actions
                )
                if action_type not in allowed_types:
                    raise ModelOutputParseError(
                        f"Agent ACTION {action_type.value!r} is not allowed here. "
                        f"Expected one of: {_format_actions(parse_actions)}.",
                        raw_response=raw_response,
                    )
            else:
                parse_kwargs = dict(
                    allow_quit=parse_allow_quit,
                    allow_ask_user=parse_allow_ask_user,
                    allow_accept=parse_allow_accept,
                    allow_message_user=any(
                        _agent_action_type_from_name(action)
                        == AgentActionType.MESSAGE_USER
                        for action in parse_actions
                    ),
                    allowed_actions=parse_actions,
                )
                if not self.use_structured_output:
                    parse_kwargs["require_thinking"] = getattr(
                        self,
                        "require_thinking",
                        False,
                    )
                try:
                    output, action_type, action_details, thinking = parse_fn(
                        raw_response,
                        **parse_kwargs,
                    )
                except ValueError as exc:
                    raise ModelOutputParseError(
                        str(exc),
                        raw_response=raw_response,
                    ) from exc
            if getattr(self, "require_thinking", False) and not thinking.strip():
                raise ModelOutputParseError(
                    "Missing or empty agent THINKING section.",
                    raw_response=raw_response,
                )
            missing_details = (
                action_type == AgentActionType.PROPOSE and not action_details
            )
            if missing_details:
                prev = output.strip()[:500]
                raise ModelOutputParseError(
                    "ACTION was PROPOSE but PROPOSAL_DETAILS was missing or empty. "
                    f"Previous proposal text: {prev!r}.",
                    raw_response=raw_response,
                )
            if action_details is not None and action_type == AgentActionType.PROPOSE:
                try:
                    action_details = game.parse_proposal_details(action_details)
                    action_details = game.validate_deal(action_details)
                except ValueError as exc:
                    raise ModelOutputParseError(
                        str(exc),
                        raw_response=raw_response,
                    ) from exc
            return AgentAction(
                action_type,
                output=output,
                thinking=thinking,
                action_details=action_details,
                missing_details=False,
            )

        while True:
            local_attempt += 1
            rf = (
                _get_agent_response_schema(actions)
                if self.use_structured_output
                else None
            )
            model_output_attempt = attempt_budget.consume(retry_mode)
            context = (
                "respond_to_agent"
                if retry_mode == RetryMode.INITIAL
                else f"respond_to_agent_{retry_mode.value}_{model_output_attempt}"
            )
            prompt["api_log_context"] = make_api_log_context(
                prompt_logger=getattr(self, "prompt_logger", None),
                actor_id=self.agent_id,
                operation=context,
                format_attempt=local_attempt,
            )
            prompt["api_log_context"].update(attempt_budget.log_context())
            raw = self.model.query_model(prompt, response_format=rf)
            if getattr(self, "prompt_logger", None):
                self.prompt_logger.log_prompt(
                    self.agent_id,
                    prompt,
                    context=context,
                    response=raw,
                    usage=getattr(self.model, "last_usage", None),
                    request_metadata=getattr(self.model, "last_request_metadata", None),
                )
            try:
                return parse_and_validate(raw, actions)
            except ModelOutputParseError as e:
                if game_fmt is None and not self.use_structured_output:
                    normalized = normalize_newline_only_labeled_response(
                        raw,
                        require_thinking=getattr(self, "require_thinking", False),
                    )
                    if normalized is not None:
                        try:
                            return parse_and_validate(normalized, actions)
                        except ModelOutputParseError:
                            pass
                retry_mode = RetryMode.RESAMPLE
                attempt_budget.record_failure(
                    e,
                    raw_response=raw,
                    next_retry_mode=retry_mode,
                )


def get_agent(agent_id: str, user: Any, config: dict[str, Any], orchestrator_settings: dict[str, Any]) -> Agent:
    """Factory for the LLM negotiation agent."""
    return Agent(agent_id=agent_id, user=user, config=config, orchestrator_settings=orchestrator_settings)
