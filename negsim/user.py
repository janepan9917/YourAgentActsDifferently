"""
User U_i: represented by an agent A_i in the negotiation.
DummyUser: always consents. LLMUser: uses an LLM to respond.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .action import UserAction, UserActionType
from .api_request_logger import make_api_log_context
from .dialogue import format_action_for_history
from .flow_status import format_game_state_with_flow_status
from .game import format_negotiable_terms_rule
from .model import create_model
from .model_output_errors import (
    ModelOutputAttemptBudget,
    ModelOutputParseError,
    RetryMode,
    UnrecoverableModelFormatError,
)
from .response_parsing import (
    extract_unique_labeled_section,
    find_action_declarations,
    find_duplicate_labeled_sections,
    normalize_newline_only_labeled_response,
    normalize_single_action_declaration,
    parse_labeled_sections,
)

if TYPE_CHECKING:
    from .dialogue import Dialogue
    from .model import Prompt


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
    """Format action list as 'A or B' for prompts."""
    return " or ".join(actions) if actions else "CONSENT"


def _normalize_action_name(action: str) -> str:
    """Normalize action labels from prompts, configs, and model output."""
    return re.sub(r"[^A-Z0-9]+", "_", str(action).strip().upper()).strip("_")


def _user_action_type_from_name(action: str) -> UserActionType | None:
    normalized = _normalize_action_name(action)
    aliases = {
        "CONSENT": UserActionType.CONSENT,
        "FURTHER_INSTRUCT": UserActionType.FURTHER_INSTRUCT,
        "FURTHER_INSTRUCTION": UserActionType.FURTHER_INSTRUCT,
        "DONT_CONSENT": UserActionType.DONT_CONSENT,
        "DO_NOT_CONSENT": UserActionType.DONT_CONSENT,
    }
    return aliases.get(normalized)


def _validate_configured_user_actions(
    actions: list[str],
    *,
    allowed_types: set[UserActionType],
    consultation_mode: str | None,
) -> None:
    """Fail before querying when prompt action configuration is unusable."""
    if not actions:
        raise ValueError(
            f"No user actions are configured for {consultation_mode or 'default'} mode."
        )
    invalid = [
        action
        for action in actions
        if _user_action_type_from_name(action) not in allowed_types
    ]
    if invalid:
        raise ValueError(
            "Invalid configured user actions for "
            f"{consultation_mode or 'default'} mode: {invalid!r}."
        )


_NO_USER_OUTPUT_LOCK = object()


@dataclass(frozen=True)
class _UserActionDeclaration:
    status: str
    raw_value: str | None = None
    action_type: UserActionType | None = None
    allowed_label: str | None = None
    normalized_response: str | None = None


@dataclass(frozen=True)
class _UserFormatRepairLock:
    action_type: UserActionType
    action_label: str
    output: Any = _NO_USER_OUTPUT_LOCK


def _inspect_user_action_declaration(
    raw: str,
    actions: list[str],
) -> _UserActionDeclaration:
    declarations = find_action_declarations(raw)
    if not declarations:
        return _UserActionDeclaration(status="missing")
    values = [declaration.value.strip() for declaration in declarations]
    resolved_types = [_user_action_type_from_name(value) for value in values]
    recognized = [
        (value, action_type)
        for value, action_type in zip(values, resolved_types)
        if action_type is not None
    ]
    if len({action_type for _, action_type in recognized}) > 1:
        return _UserActionDeclaration(status="ambiguous")
    if recognized:
        raw_value, action_type = recognized[0]
    else:
        raw_value, action_type = values[0], None
    if action_type is None:
        return _UserActionDeclaration(status="unknown", raw_value=raw_value)
    allowed_label = next(
        (
            label
            for label in actions
            if _user_action_type_from_name(label) == action_type
        ),
        None,
    )
    normalized = (
        normalize_single_action_declaration(raw)
        if len(declarations) == 1
        else None
    )
    return _UserActionDeclaration(
        status="allowed" if allowed_label is not None else "disallowed",
        raw_value=raw_value,
        action_type=action_type,
        allowed_label=allowed_label,
        normalized_response=normalized[0] if normalized is not None else None,
    )


def _prepare_user_repair(
    raw: str,
    actions: list[str],
) -> tuple[_UserFormatRepairLock | None, bool, str | None]:
    duplicate_sections = find_duplicate_labeled_sections(
        raw,
        ["THINKING", "ACTION", "OUTPUT"],
        tolerant=True,
    )
    conflicting_content = duplicate_sections - {"ACTION"}
    if conflicting_content:
        labels = ", ".join(sorted(conflicting_content))
        raise UnrecoverableModelFormatError(
            "Unrecoverable format error: duplicate content sections "
            f"({labels}) would require the harness to choose model content.",
            raw_response=raw,
        )

    declaration = _inspect_user_action_declaration(raw, actions)
    if declaration.status == "disallowed":
        return None, True, declaration.normalized_response
    if declaration.status != "allowed":
        reason = {
            "ambiguous": "multiple conflicting ACTION declarations",
            "unknown": f"unknown ACTION {declaration.raw_value!r}",
            "missing": "no unambiguous ACTION declaration",
        }.get(declaration.status, "no recoverable ACTION declaration")
        raise UnrecoverableModelFormatError(
            "Unrecoverable format error: "
            f"{reason}. The harness will request a fresh model decision.",
            raw_response=raw,
        )
    assert declaration.allowed_label is not None
    assert declaration.action_type is not None
    try:
        output = extract_unique_labeled_section(
            declaration.normalized_response or raw,
            "OUTPUT",
            all_labels=["THINKING", "ACTION", "OUTPUT"],
            tolerant=True,
        )
    except ValueError:
        output = None
    return (
        _UserFormatRepairLock(
            action_type=declaration.action_type,
            action_label=declaration.allowed_label,
            output=output if output else _NO_USER_OUTPUT_LOCK,
        ),
        False,
        declaration.normalized_response,
    )


def _format_agent_agent_transcript_for_user(
    user_config: dict[str, Any],
    agent_agent_history: str,
) -> str:
    """Return the agent-agent negotiation transcript block, only when this user
    is configured to see it (user_config["give_transcript_access"]). Off by
    default: the user has no visibility into the agent-agent negotiation
    unless this ablation flag is explicitly set."""
    if not user_config.get("give_transcript_access"):
        return ""
    return (
        "[AGENT-AGENT NEGOTIATION TRANSCRIPT]\n"
        "You can see the full negotiation transcript between the two agents so far:\n"
        f"{agent_agent_history}"
    )


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


def _format_dialogue_as_messages(
    dialogue: "Dialogue",
    agent_id: str,
) -> list[dict[str, str]]:
    """
    Turn conversation_history into list of {role, content} for model prompt.
    From the user's perspective: agent messages are 'user' role, user messages are 'assistant' role.
    """
    messages: list[dict[str, str]] = []
    for turn in dialogue.conversation_history:
        speaker = turn["speaker"]
        action = turn["action"]
        content = format_action_for_history(action, speaker=speaker)
        if speaker == agent_id:
            messages.append({"role": "user", "content": content or "(no message)"})
        else:
            messages.append({"role": "assistant", "content": content or "(no message)"})
    return messages


def _format_game_dialogue_as_text(dialogue: "Dialogue") -> str:
    """Render the agent-agent negotiation transcript as plain text for the user's system prompt."""
    lines: list[str] = []
    prev_turn: int | None = None
    for turn in dialogue.conversation_history:
        turn_num = turn.get("turn_num")
        if turn_num is not None and turn_num != prev_turn:
            lines.append(f"\n=== Turn {turn_num} ===")
            prev_turn = turn_num
        speaker = turn["speaker"]
        action = turn.get("action")
        content = format_action_for_history(action, speaker=speaker)
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines).strip() or "(no negotiation yet)"


def _format_user_retry_prompt(
    actions: list[str],
    parse_error: str,
    previous_response: str,
    *,
    require_thinking: bool = False,
    locked_action: str | None = None,
    locked_output: Any = _NO_USER_OUTPUT_LOCK,
    protocol_correction: bool = False,
    retry_mode: RetryMode | None = None,
) -> str:
    """Build a constrained repair/correction addendum for an invalid response."""
    valid_actions = ", ".join(actions)
    actions_str = _format_actions(actions)
    thinking_line = (
        "THINKING: <your private reasoning; never copy it into OUTPUT>\n"
        if require_thinking
        else ""
    )
    if retry_mode == RetryMode.FRESH_DECISION:
        repair_directive = (
            "The previous response did not contain one unambiguous executable decision. "
            "Make a fresh decision from the current context using one of the currently "
            "allowed actions."
        )
    elif protocol_correction or retry_mode == RetryMode.PROTOCOL_CORRECTION:
        repair_directive = (
            "Your declared action is not legal in this state. Reconsider only the action "
            "as needed to choose one of the currently allowed actions."
        )
    elif locked_action is not None:
        repair_directive = (
            "This is a format-only repair. Do not reconsider your decision. "
            f"ACTION is locked to exactly {locked_action}."
        )
        if locked_output is not _NO_USER_OUTPUT_LOCK:
            repair_directive += (
                " OUTPUT is locked to this exact text: "
                f"{json.dumps(locked_output, ensure_ascii=False)}. "
                "Do not paraphrase or otherwise change it."
            )
    else:
        repair_directive = (
            "Keep the same negotiation intent if possible; only fix the format."
        )

    return (
        f"Your previous response could not be parsed: {parse_error}\n\n"
        "Here is that previous invalid response verbatim:\n"
        "<previous_invalid_response>\n"
        f"{previous_response}\n"
        "</previous_invalid_response>\n\n"
        "Respond again using exactly this format and no extra text:\n\n"
        "```\n"
        f"{thinking_line}"
        f"ACTION: {actions_str}\n"
        "OUTPUT: <natural-language message to your agent>\n"
        "```\n\n"
        f"The ACTION value must be exactly one of: {valid_actions}. "
        f"{repair_directive}"
    )


def _parse_user_response(
    text: str,
    allowed_actions: list[str] | None = None,
    *,
    require_thinking: bool = False,
) -> tuple[str, UserActionType, str]:
    """
    Parse model output into (output, action_type, thinking).
    Expected format (optional sections):
      THINKING: ...
      ACTION: CONSENT | FURTHER_INSTRUCT | DONT_CONSENT
      OUTPUT: ...
    """
    declarations = find_action_declarations(text)
    declared_types = {
        _user_action_type_from_name(declaration.value)
        for declaration in declarations
        if _user_action_type_from_name(declaration.value) is not None
    }
    if len(declared_types) > 1:
        raise ValueError(
            "Conflicting user ACTION declarations are ambiguous; exactly one decision is required."
        )
    sections = parse_labeled_sections(
        text,
        ["THINKING", "ACTION", "OUTPUT"],
        tolerant=require_thinking,
    )
    thinking = sections.get("THINKING", "")
    action_type: UserActionType | None = None
    output = sections.get("OUTPUT", "")

    if require_thinking:
        if "THINKING" not in sections or not thinking.strip():
            raise ValueError("Missing or empty user THINKING section.")
        if "OUTPUT" not in sections:
            raise ValueError("Missing user OUTPUT section; private THINKING must not be used as OUTPUT.")

    raw_action = sections.get("ACTION", "")
    if raw_action:
        raw = raw_action.splitlines()[0].strip()
        action_type = _user_action_type_from_name(raw)
        if action_type is None:
            valid = ", ".join(a.value for a in UserActionType)
            raise ValueError(f"Unknown user ACTION {raw!r}. Expected exactly one of: {valid}.")
        if allowed_actions is not None:
            allowed = {_user_action_type_from_name(action) for action in allowed_actions}
            if action_type not in allowed:
                raise ValueError(
                    f"User ACTION {raw!r} is not allowed here. Expected exactly one of: {_format_actions(allowed_actions)}."
                )

    if action_type is None:
        valid = _format_actions(allowed_actions) if allowed_actions is not None else ", ".join(a.value for a in UserActionType)
        raise ValueError(f"Missing user ACTION. Expected exactly one of: {valid}.")
    if require_thinking and not output:
        raise ValueError("User OUTPUT was empty.")
    if not output:
        if action_type == UserActionType.CONSENT:
            output = "Sounds good, go ahead"
        elif action_type == UserActionType.FURTHER_INSTRUCT:
            output = "Here is what I want you to do next."
        else:
            output = "No, I don't agree."

    return (output, action_type, thinking)


class User(ABC):
    """
    User U_i. Base class for user implementations.
    """

    def __init__(self, config: dict, user_id: str = "") -> None:
        self.config = config
        self.initial_offer = config.get("initial_offer", "")
        self.user_id = user_id
        self.private_info: dict = config.get("private_info") or {}
        self.prompt_logger = None

    @abstractmethod
    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """
        Respond to the agent's proposal. Called after the agent has spoken.
        dialogue_with_agent: full conversation between this user and their agent.
        agent_id: the agent's id (for formatting).
        game_dialogue: agent-agent negotiation (for context); None during initiation.
        game: Game instance (for game_blurb and other game context).
        """
        ...


class DummyYesUser(User):
    """
    Dummy user that always consents. Use for testing or when user input is not needed.
    """

    def __init__(self, config: dict, user_id: str = "") -> None:
        super().__init__(config, user_id)

    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """Always consent."""
        if consultation_mode == "guidance":
            return UserAction(UserActionType.FURTHER_INSTRUCT, output="Continue with your best judgment.")
        return UserAction(UserActionType.CONSENT, output="Sounds good, go ahead")

class DummyNoUser(User):
    """
    Dummy user that always does not consent. Use for testing or when user input is not needed.
    """

    def __init__(self, config: dict, user_id: str = "") -> None:
        super().__init__(config, user_id)

    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """Always do not consent."""
        if consultation_mode == "guidance":
            return UserAction(UserActionType.FURTHER_INSTRUCT, output="Please keep negotiating for a better option.")
        return UserAction(UserActionType.DONT_CONSENT, output="No, I don't agree. Please try again.")


class SilentUser(User):
    """
    User that never queries an API. Gives minimal, uninformative responses:
    states its private info once at the start, then always defers to the agent.
    """

    def __init__(self, config: dict, user_id: str = "") -> None:
        super().__init__(config, user_id)

    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """State private info once at initiation, otherwise always defer."""
        if consultation_mode == "initial_guidance":
            private_info = game.get_private_info_for_user_prompt(self.user_id) or "None."
            return UserAction(UserActionType.FURTHER_INSTRUCT, output=private_info)
        if consultation_mode == "guidance":
            return UserAction(UserActionType.FURTHER_INSTRUCT, output="I'll leave it to you.")
        return UserAction(UserActionType.CONSENT, output="I'll leave it to you, go ahead.")


class LLMUser(User):
    """
    User that uses an LLM to respond. Loads prompts from YAML and queries the model.
    """

    def __init__(self, config: dict, user_id: str = "") -> None:
        super().__init__(config, user_id)

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
        self.prompts = config["prompts"] if "prompts" in config else _load_prompts(config["prompts_path"])
        self.require_thinking = bool(
            config.get("require_thinking", self.prompts.get("require_thinking", False))
        )

    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """Build prompt from dialogue, query model, parse response."""
        p = self.prompts["respond_to_agent"]
        actions = _select_actions(self.config, p, "respond_to_agent", consultation_mode)
        mode = _mode_key(consultation_mode)
        if mode == "final_decision":
            actions = ["CONSENT", "DONT_CONSENT"]
        if mode in {"guidance", "initial_guidance"}:
            configured_user_types = {UserActionType.FURTHER_INSTRUCT}
        elif mode == "final_decision":
            configured_user_types = {
                UserActionType.CONSENT,
                UserActionType.DONT_CONSENT,
            }
        else:
            configured_user_types = set(UserActionType)
        _validate_configured_user_actions(
            actions,
            allowed_types=configured_user_types,
            consultation_mode=consultation_mode,
        )
        actions_str = _format_actions(actions)
        game_blurb = game.get_game_blurb_for_user_prompt()
        public_info = game.get_public_info_for_user_prompt(self.user_id) or "None."
        private_info = game.get_private_info_for_user_prompt(self.user_id) or "None."
        current_turn = game.state.get("turn", game.state.get("turns_total", 0))
        game_state = format_game_state_with_flow_status(
            game.get_game_state_for_prompt(),
            current_turn=current_turn,
            max_turns=getattr(game, "max_negotiation_turns", None),
        )
        system_tpl = p["system"]
        agent_agent_history = (
            _format_game_dialogue_as_text(game_dialogue)
            if game_dialogue and game_dialogue.conversation_history
            else "(no negotiation yet)"
        )
        agent_reward_status_text = agent_reward_status or "(not available)"
        fmt_kwargs: dict[str, Any] = dict(
            user_id=self.user_id,
            agent_id=agent_id,
            actions=actions_str,
            public_info=public_info,
            private_info=private_info,
            game_blurb=game_blurb,
            game_state=game_state,
            consultation_mode=consultation_mode or "default",
            agent_agent_history=agent_agent_history,
            agent_reward_status=agent_reward_status_text,
        )
        system = system_tpl.format(**fmt_kwargs)
        negotiable_terms_rule = format_negotiable_terms_rule(game, self.user_id)
        if negotiable_terms_rule:
            system = f"{system}\n\n{negotiable_terms_rule}"
        transcript_block = _format_agent_agent_transcript_for_user(
            self.config, agent_agent_history
        )
        if transcript_block:
            system = f"{system}\n\n{transcript_block}"

        history = _format_dialogue_as_messages(dialogue_with_agent, agent_id)

        current_tpl = _select_prompt_template(p, "current_prompt", consultation_mode)
        current = current_tpl.format(**fmt_kwargs)
        base_current = current
        if _mode_key(consultation_mode) != "final_decision":
            final_confirmation_rule = None
        elif getattr(self, "bilateral_final_confirmation", False):
            # Both agents have users: the deal closes only if both users consent.
            final_confirmation_rule = (
                "[BILATERAL FINAL CONFIRMATION]\n"
                "Your agent's latest action is REQUEST_USER_CONFIRMATION. The exact "
                "proposal shown in that action was supplied by the system and cannot "
                "be changed here. CONSENT approves those exact terms; DONT_CONSENT "
                "rejects those exact terms. If either user rejects them, this "
                "confirmation action does not complete the deal."
            )
        else:
            final_confirmation_rule = (
                "[FINAL USER CONFIRMATION]\n"
                "Your agent is requesting your decision on the exact proposal shown "
                "in REQUEST_USER_CONFIRMATION. CONSENT approves those exact terms "
                "and completes the deal. DONT_CONSENT rejects those exact terms "
                "and leaves any remaining turns available for negotiation."
            )
        if final_confirmation_rule:
            current = f"{base_current}\n\n{final_confirmation_rule}"
        user_id = self.user_id
        prompt: Prompt = {
            "system_prompt": system,
            "conversation_history": history,
            "current_prompt": current,
            "conversation_history_label": f"conversation history between {user_id} and {agent_id}",
            "other_agent_id": agent_id,
            "assistant_id": user_id,
        }
        attempt_budget = attempt_budget or ModelOutputAttemptBudget(
            actor_id=self.user_id
        )
        if not attempt_budget.actor_id:
            attempt_budget.actor_id = self.user_id
        local_attempt = 0
        retry_mode = attempt_budget.next_retry_mode

        def parse_and_validate(
            raw_response: str,
            parse_actions: list[str],
        ) -> UserAction:
            try:
                output, action_type, thinking = _parse_user_response(
                    raw_response,
                    allowed_actions=parse_actions,
                    require_thinking=getattr(self, "require_thinking", False),
                )
            except ValueError as exc:
                raise ModelOutputParseError(
                    str(exc),
                    raw_response=raw_response,
                ) from exc
            return UserAction(action_type, output=output, thinking=thinking)

        while True:
            local_attempt += 1
            model_output_attempt = attempt_budget.consume(retry_mode)
            context = (
                "respond_to_agent"
                if retry_mode == RetryMode.INITIAL
                else f"respond_to_agent_{retry_mode.value}_{model_output_attempt}"
            )
            prompt["api_log_context"] = make_api_log_context(
                prompt_logger=getattr(self, "prompt_logger", None),
                actor_id=self.user_id,
                operation=context,
                format_attempt=local_attempt,
            )
            prompt["api_log_context"].update(attempt_budget.log_context())
            raw = self.model.query_model(prompt)
            if getattr(self, "prompt_logger", None) and self.user_id:
                self.prompt_logger.log_prompt(
                    self.user_id,
                    prompt,
                    context=context,
                    response=raw,
                    usage=getattr(self.model, "last_usage", None),
                    request_metadata=getattr(self.model, "last_request_metadata", None),
            )
            try:
                return parse_and_validate(raw, actions)
            except ModelOutputParseError as e:
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


class InitialConsultOnlyUser(LLMUser):
    """
    User that gives one real LLM-generated initial consultation, then always
    defers with a canned response for every later consultation (same canned
    strings as SilentUser), without ever querying the model again.
    """

    def respond_to_agent(
        self,
        dialogue_with_agent: "Dialogue",
        agent_id: str,
        game_dialogue: "Dialogue | None",
        game: Any,
        agent_reward_status: str | None = None,
        consultation_mode: str | None = None,
        attempt_budget: ModelOutputAttemptBudget | None = None,
    ) -> UserAction:
        """Real LLM call at initiation only; canned deferral after."""
        if consultation_mode == "initial_guidance":
            return super().respond_to_agent(
                dialogue_with_agent,
                agent_id,
                game_dialogue,
                game,
                agent_reward_status=agent_reward_status,
                consultation_mode=consultation_mode,
                attempt_budget=attempt_budget,
            )
        if consultation_mode == "guidance":
            return UserAction(UserActionType.FURTHER_INSTRUCT, output="I'll leave it to you.")
        return UserAction(UserActionType.CONSENT, output="I'll leave it to you, go ahead.")


def get_user_by_type(user_id: str, config: dict) -> User:
    """Factory: return User instance based on type ('dummy_yes', 'dummy_no', 'llm', 'silent', 'initial_consult_only', or 'none')."""
    user_type = config["type"]
    if user_type in ("dummy_yes", "none"):
        return DummyYesUser(config, user_id)
    if user_type == "dummy_no":
        return DummyNoUser(config, user_id)
    if user_type == "llm":
        return LLMUser(config, user_id)
    if user_type == "silent":
        return SilentUser(config, user_id)
    if user_type == "initial_consult_only":
        return InitialConsultOnlyUser(config, user_id)
    raise ValueError(
        f"Unknown user type: {user_type!r}. Use 'dummy_yes', 'dummy_no', 'llm', "
        "'silent', 'initial_consult_only', or 'none'."
    )
