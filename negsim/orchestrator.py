"""Turn-based negotiation orchestrator."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .action import AgentAction, AgentActionType, UserAction, UserActionType
from .agent_objective import (
    apply_objective_penalties_per_round,
    apply_objective_penalties_per_turn,
    format_reward_status,
)
from .api_request_logger import initialize_last_api_exchange_files
from .conversation_logger import ConversationLogger
from .dialogue import count_ask_user_messages
from .game import Game, GameEndReason
from .model_output_errors import (
    ModelOutputAttemptBudget,
    ModelOutputAttemptsExceeded,
    RetryMode,
)

log = logging.getLogger(__name__)


_AGENT_AGENT_CHANNEL = "agent_agent"
_AGENT_USER_CHANNEL = "agent_user"

FINAL_CONFIRMATION_SUCCEEDED_PUBLIC_OUTPUT = (
    "Final confirmation succeeded. The tentative deal is confirmed."
)
FINAL_CONFIRMATION_FAILED_PUBLIC_OUTPUT = (
    "Final confirmation declined."
)
FINAL_CONFIRMATION_INCOMPLETE_PUBLIC_OUTPUT = (
    "Final confirmation is incomplete."
)


def _format_final_confirmation_public_output(
    outcomes: dict[str, bool],
    *,
    succeeded: bool,
    authorizations: dict[str, str],
) -> str:
    """Distinguish agent authorization from an actual user's decision."""
    headline = (
        FINAL_CONFIRMATION_SUCCEEDED_PUBLIC_OUTPUT
        if succeeded
        else FINAL_CONFIRMATION_FAILED_PUBLIC_OUTPUT
    )
    decisions = "\n".join(
        f"{user_id}: {'CONSENT' if consented else 'DONT_CONSENT'}"
        for user_id, consented in outcomes.items()
    )
    if not authorizations:
        # Bilateral confirmation (both agents represented): only user decisions.
        return f"{headline}\nUser confirmation decisions:\n{decisions}"
    agent_decisions = "\n".join(
        f"{agent_id}: {source}" for agent_id, source in authorizations.items()
    )
    return (
        f"{headline}\nAgent authorization:\n{agent_decisions}"
        f"\nUser confirmation decisions:\n{decisions}"
    )


def _is_represented(agent: Any) -> bool:
    """True when this agent's own user must confirm a deal before it closes."""
    return not (
        getattr(agent, "no_user", False)
        or getattr(agent, "no_final_confirmation", False)
    )

ConsultationEndReason = Literal[
    "agent_continue",
    "user_consent",
    "user_rejection",
    "exchange_limit",
    "message_limit",
]


@dataclass(frozen=True)
class ConsultationResult:
    """Structured outcome of one private agent-user consultation."""

    action: AgentAction | None
    end_reason: ConsultationEndReason
    completed_exchanges: int
    max_consecutive_invalid_attempts: int
    origin: str


def _optional_positive_int(value: Any, *, name: str) -> int | None:
    """Parse an optional positive integer config value."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer or null")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer or null") from exc
    if parsed < 1 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{name} must be a positive integer or null")
    return parsed


def _get_last_propose_from_other(game_dialogue: Any, agent_id: str) -> dict | None:
    """Return the last PROPOSE's action_details from the counterparty, or None."""
    if not game_dialogue or not game_dialogue.conversation_history:
        return None
    for turn in reversed(game_dialogue.conversation_history):
        if turn.get("speaker") == agent_id:
            continue
        action = turn.get("action")
        if action and action.action_type == AgentActionType.PROPOSE and action.action_details:
            return action.action_details
    return None


class GameLimitExceededError(Exception):
    """Raised when a negotiation or agent-user turn limit is exceeded."""


def _record_retryable_model_output_failure(
    budget: ModelOutputAttemptBudget,
    *,
    used_before_call: int,
    error: BaseException | str,
    next_retry_mode: RetryMode,
) -> None:
    """Record an invalid model result, rejecting deterministic/internal failures."""
    if budget.used_attempts == used_before_call:
        raise RuntimeError(
            f"{budget.actor_id or 'Actor'} returned an invalid action without "
            "consuming a model-output attempt. Deterministic actor and harness "
            "errors are not retryable."
        )
    budget.record_failure(
        error,
        raw_response=None,
        next_retry_mode=next_retry_mode,
    )


def make_game_id() -> str:
    """Date + timestamp + UUID suffix for parallel-safe log directory names."""
    now = datetime.now(timezone.utc)
    return f"{now:%Y%m%d}-{now:%H%M%S}-{now:%f}-{uuid.uuid4().hex[:8]}"


def _apply_agent_objective_penalties_per_round(runner: "RoundRunner") -> None:
    """Apply end-of-game objective penalties and check reward-zero termination."""
    base_context_info = {
        "end_reason": runner.game.end_reason,
        "current_turn": runner.game.state.get("turn", runner.current_turn),
        "quitter_user_id": runner.game.quitter_user_id,
        "deal": getattr(runner.game, "agreed_terms", None),
    }
    for agent in (runner.game.agent_1, runner.game.agent_2):
        context_info = {**base_context_info, "this_agent_user_id": agent.user.user_id}
        apply_objective_penalties_per_round(agent, context_info)

    a1_zero = runner.game.agent_1.current_reward <= 0
    a2_zero = runner.game.agent_2.current_reward <= 0
    if a1_zero or a2_zero:
        runner.game.end_reason = "reward_zero"
        if a1_zero and a2_zero:
            runner.game.reward_zero_loser_user_id = None
            log.info("  Both agents reward hit 0 - game over, both lose")
        else:
            loser = runner.game.agent_1 if a1_zero else runner.game.agent_2
            runner.game.reward_zero_loser_user_id = loser.user.user_id
            log.info("  %s reward hit 0 - game over, %s loses", loser.agent_id, loser.user.user_id)


class RoundRunner:
    """
    Turn-based negotiation runner.

    Agent-agent negotiation proceeds up to
    a global turn limit. ASK USER and ACCEPT AND CONFIRM WITH USER trigger a
    private agent-user dialogue. ASK USER consults only the acting agent's user
    without consuming a global negotiation turn. ACCEPT AND CONFIRM WITH USER
    asks the acting agent's user for final confirmation; when both agents have
    users, it asks both users (bilateral confirmation). ACCEPT by a no-user
    agent asks the proposal author's user when that user exists. Confirmation
    consumes a global negotiation turn whether it succeeds or fails.
    """

    @property
    def _bilateral_confirmation(self) -> bool:
        """With a user on both sides, any acceptance needs both users' consent."""
        return all(_is_represented(a) for a in (self.game.agent_1, self.game.agent_2))

    def __init__(self, game: Game, game_id: str, log_dir: Path, config: dict[str, Any]) -> None:
        self.game = game
        self.game_id = game_id
        self.log_dir = Path(log_dir)
        self.config = config
        self.max_negotiation_turns = int(config.get("max_total_negotiation_turns", 40))
        self.max_agent_user_turns = int(config.get("max_agent_user_turns", 60))
        self.max_agent_user_exchanges = _optional_positive_int(
            config.get("max_agent_user_exchanges", 4),
            name="max_agent_user_exchanges",
        )
        self.max_ask_user_consultation_exchanges = _optional_positive_int(
            config.get(
                "max_ask_user_consultation_exchanges",
                self.max_agent_user_exchanges,
            ),
            name="max_ask_user_consultation_exchanges",
        )
        self.game.max_negotiation_turns = self.max_negotiation_turns
        self.game.max_agent_user_turns = self.max_agent_user_turns
        self.agent_1_final_action: Any = None
        self.agent_2_final_action: Any = None
        self.logger: ConversationLogger | None = None
        self.current_turn: int = 0
        self.turns_by_agent: dict[str, int] = {
            game.agent_1.agent_id: 0,
            game.agent_2.agent_id: 0,
        }
        self._last_invalid_error: str | None = None
        self._last_invalid_error_agent_user: str | None = None
        self.consultation_records: list[dict[str, Any]] = []
        self.game.consultation_records = self.consultation_records
        self.no_user_mode = getattr(game.agent_1, "no_user", False) and getattr(game.agent_2, "no_user", False)
        self._initial_messages_injected: set[str] = set()

    @staticmethod
    def _append_visible_turn(
        agent: Any,
        *,
        channel: str,
        speaker: str,
        action: Any,
        turn_num: int | None,
        counts_for_game_state: bool = True,
        visible_to_user: bool = True,
    ) -> None:
        """Append an event to one agent's sole prompt-visible history."""
        if getattr(agent, "no_user", False):
            return
        dialogue = getattr(agent, "visible_dialogue", None)
        if dialogue is None:
            return
        dialogue.add_turn(
            speaker,
            len(getattr(action, "output", "") or ""),
            action,
            counts_for_game_state=counts_for_game_state,
            counts_toward_ask_user_limit=False,
            turn_num=turn_num,
            channel=channel,
            visible_to_user=visible_to_user,
        )

    def _append_public_visible_turn(
        self,
        speaker: str,
        action: AgentAction,
        *,
        turn_num: int | None,
        counts_for_game_state: bool = True,
    ) -> None:
        """Append one public event to both agents' separate visible histories."""
        for agent in (self.game.agent_1, self.game.agent_2):
            self._append_visible_turn(
                agent,
                channel=_AGENT_AGENT_CHANNEL,
                speaker=speaker,
                action=action,
                turn_num=turn_num,
                counts_for_game_state=counts_for_game_state,
            )

    def _seed_preloaded_public_visible_history(self) -> None:
        """Append harness-seeded public openings after turn-zero guidance."""
        for turn in self.game.game_dialogue.conversation_history:
            self._append_public_visible_turn(
                turn["speaker"],
                turn["action"],
                turn_num=turn.get("turn_num"),
                counts_for_game_state=turn.get("counts_for_game_state", True),
            )

    def _consultation_mode_for_agent(self, agent: Any, desired_mode: str) -> str | None:
        """Return a mode only when prompt/config opts into mode-specific consultation."""
        settings_mode = self.config.get("user_consultation_mode")
        if settings_mode is None:
            settings_mode = getattr(agent, "orchestrator_settings", {}).get("user_consultation_mode")
        agent_mode = getattr(agent, "config", {}).get("user_consultation_mode")
        if agent_mode or settings_mode:
            selected = agent_mode or settings_mode
            if str(selected).lower() in {"natural", "mode_specific", desired_mode}:
                return desired_mode

        prompts = getattr(agent, "prompts", {}) or {}
        respond_to_user = prompts.get("respond_to_user") or {}
        if f"actions_{desired_mode}" in respond_to_user:
            return desired_mode

        allowed = (getattr(agent, "config", {}) or {}).get("allowed_actions") or {}
        respond_actions = allowed.get("respond_to_user")
        if isinstance(respond_actions, dict) and desired_mode in respond_actions:
            return desired_mode
        return None

    def _update_turn_state(self, next_speaker: str | None = None) -> None:
        """Publish global turn counters for prompts and game state."""
        turns_by_agent = dict(self.turns_by_agent)
        if next_speaker is not None:
            turns_by_agent[next_speaker] = turns_by_agent.get(next_speaker, 0) + 1
        current = self.current_turn + (1 if next_speaker is not None else 0)
        self.game.update_state(
            current_turn=current,
            turns_total=current,
            turns_by_agent=turns_by_agent,
        )

    def _initial_message_for_agent(self, agent: Any) -> str:
        return getattr(agent.user, "initial_offer", "") or ""

    def _log_preloaded_negotiation_history(self) -> None:
        """Log harness-seeded public turns that predate generated negotiation turns."""
        history = [
            turn
            for turn in self.game.game_dialogue.conversation_history
            if not turn.get("counts_for_game_state", True)
        ]
        if not history:
            return
        self.logger.subsection("scripted opening (turn 0)", to="both")
        for turn in history:
            self.logger.turn(turn["speaker"], turn["action"], to="both")

    def _inject_initial_messages_once(self) -> None:
        """Seed each agent's user history with configured initial guidance once."""
        for agent in (self.game.agent_1, self.game.agent_2):
            user_id = agent.user.user_id
            if user_id in self._initial_messages_injected:
                continue
            message = self._initial_message_for_agent(agent)
            if not message:
                continue
            action = UserAction(UserActionType.FURTHER_INSTRUCT, output=message)
            agent.dialogue_with_user.add_turn(
                user_id,
                len(message),
                action,
                counts_toward_ask_user_limit=False,
                turn_num=0,
            )
            self._initial_messages_injected.add(user_id)
            if self.logger:
                subsection = f"initial user guidance: {agent.agent_id} <-> {user_id}"
                self.logger.subsection(subsection, to=agent.agent_id)
                self.logger.subsection(subsection, to=user_id)
                self.logger.turn(user_id, action, to=agent.agent_id)
                self.logger.turn(user_id, action, to=user_id)

    def _run_generated_initial_guidance_consultations(self) -> None:
        """Generate turn-zero user guidance, then complete a formal guidance consultation."""
        for agent in (self.game.agent_1, self.game.agent_2):
            user_id = agent.user.user_id
            if user_id in self._initial_messages_injected:
                continue
            if getattr(agent, "no_user", False):
                continue

            message = self._initial_message_for_agent(agent)
            if message:
                user_action = UserAction(UserActionType.FURTHER_INSTRUCT, output=message)
            else:
                user_config = getattr(agent.user, "config", {}) or {}
                if user_config.get("type") not in ("llm", "silent", "initial_consult_only"):
                    raise ValueError(
                        "generate_initial_user_guidance requires user type 'llm', "
                        "'silent', or 'initial_consult_only' when no configured "
                        f"initial guidance exists ({user_id})"
                    )
                agent_reward_status = format_reward_status(
                    agent.current_reward,
                    agent.auxiliary_objectives,
                    {"current_turn": self.current_turn},
                    payout_weight=getattr(agent, "payout_weight", 1.0),
                )
                user_attempt_budget = ModelOutputAttemptBudget(actor_id=user_id)
                while True:
                    if user_attempt_budget.exhausted:
                        raise ModelOutputAttemptsExceeded(user_attempt_budget)
                    used_before_user_call = user_attempt_budget.used_attempts
                    user_action = agent.user.respond_to_agent(
                        dialogue_with_agent=agent.dialogue_with_user,
                        agent_id=agent.agent_id,
                        game_dialogue=self.game.game_dialogue,
                        game=self.game,
                        agent_reward_status=agent_reward_status,
                        consultation_mode="initial_guidance",
                        attempt_budget=user_attempt_budget,
                    )
                    if not isinstance(user_action, UserAction) or not isinstance(
                        user_action.action_type,
                        UserActionType,
                    ):
                        raise TypeError(
                            f"{user_id} returned an invalid UserAction object."
                        )
                    if user_action.action_type == UserActionType.FURTHER_INSTRUCT:
                        break
                    _record_retryable_model_output_failure(
                        user_attempt_budget,
                        used_before_call=used_before_user_call,
                        error=(
                            "Initial user guidance must use FURTHER_INSTRUCT, got "
                            f"{user_action.action_type} for {user_id}"
                        ),
                        next_retry_mode=RetryMode.RESAMPLE,
                    )

            subsection = f"initial user guidance consultation: {agent.agent_id} <-> {user_id}"
            self._run_agent_user_until_consent(
                agent,
                self.game.game_dialogue,
                initial_user_action=user_action,
                consultation_mode="guidance",
                consultation_origin="initial_guidance",
                subsection_label=subsection,
            )
            self._initial_messages_injected.add(user_id)

    def run_game(self) -> GameEndReason | None:
        """Run the negotiation until deal, quit, invalid state, reward zero, or turn limit."""
        self.logger = ConversationLogger(
            self.log_dir,
            (self.game.agent_1.agent_id, self.game.agent_2.agent_id),
            self.game_id,
            game=self.game,
        )
        initialize_last_api_exchange_files(self.log_dir)
        for agent in (self.game.agent_1, self.game.agent_2):
            agent.prompt_logger = self.logger
            agent.user.prompt_logger = self.logger
            agent.bilateral_final_confirmation = self._bilateral_confirmation
            agent.user.bilateral_final_confirmation = self._bilateral_confirmation

        try:
            self._update_turn_state()
            self.logger.section("GAME START")
            game_state = self.game.get_game_state_for_prompt()
            if game_state:
                log.info("  Game state: %s", game_state.replace("\n", " | "))
                self.logger.section(f"GAME STATE\n{game_state}")
            if self.config.get("generate_initial_user_guidance", False):
                self._run_generated_initial_guidance_consultations()
            else:
                self._inject_initial_messages_once()
            self.game.prepare_negotiation()
            self._seed_preloaded_public_visible_history()
            self._log_preloaded_negotiation_history()

            self.logger.section("NEGOTIATION")
            self._run_negotiation()
            if self.game.end_reason is None:
                raise GameLimitExceededError(
                    f"max_negotiation_turns ({self.max_negotiation_turns}) exceeded without deal or quit"
                )
            _apply_agent_objective_penalties_per_round(self)
            self._log_game_end()
            return self.game.end_reason
        except KeyboardInterrupt:
            self.game.end_reason = "interrupted"
            self.logger.section("GAME END (interrupted)")
            log.info("  Game interrupted by user.")
            raise
        except ModelOutputAttemptsExceeded as error:
            budget = error.budget
            detail = budget.last_error_message or str(error)
            self.game.end_reason = "invalid_model_output"
            self.game.end_error_type = type(error).__name__
            self.game.end_error_message = detail
            self.game.model_output_failure = {
                "actor_id": budget.actor_id,
                "decision_cycle_id": budget.decision_cycle_id,
                "used_attempts": budget.used_attempts,
                "max_attempts": budget.max_attempts,
                "last_retry_mode": (
                    budget.last_retry_mode.value
                    if budget.last_retry_mode is not None
                    else ""
                ),
                "next_retry_mode": budget.next_retry_mode.value,
                "last_error": detail,
            }
            self._log_game_end(error_message=detail)
            return self.game.end_reason
        except GameLimitExceededError as e:
            self.game.end_error_message = str(e)
            if self.game.end_reason is None:
                self.game.end_reason = "limit_exceeded"
            _apply_agent_objective_penalties_per_round(self)
            self._log_game_end(error_message=str(e))
            raise
        finally:
            self.logger.close()

    def _log_game_end(self, error_message: str | None = None) -> None:
        for user in (self.game.agent_1.user, self.game.agent_2.user):
            self.game.calculate_payout(user)
        suffix = f"\n{error_message}" if error_message else ""
        self.logger.section(f"GAME END ({self.game.end_reason}){suffix}")
        for agent in (self.game.agent_1, self.game.agent_2):
            user_payout = self.game.calculate_payout(agent.user)
            if user_payout is None:
                log.info(
                    "  %s final reward: %.1f, user payout: unscored, final score: unscored",
                    agent.agent_id,
                    agent.current_reward,
                )
            else:
                final_score = agent.current_reward + agent.payout_weight * user_payout
                log.info(
                    "  %s final reward: %.1f, user payout: %.1f, final score: %.1f",
                    agent.agent_id,
                    agent.current_reward,
                    user_payout,
                    final_score,
                )

    def _agent_user_limit_reached(self, agent: Any) -> bool:
        return self._agent_user_messages_remaining(agent) <= 0

    def _agent_user_messages_remaining(self, agent: Any) -> int:
        """Return this pair's remaining global ASK USER message budget."""
        return max(
            0,
            self.max_agent_user_turns
            - count_ask_user_messages(agent.dialogue_with_user),
        )

    def _consultation_exchange_limit(self, origin: str) -> int | None:
        """Return the completed-exchange cap for this consultation origin."""
        if origin == "ask_user":
            return self.max_ask_user_consultation_exchanges
        return self.max_agent_user_exchanges

    def _finish_agent_user_consultation(
        self,
        agent: Any,
        *,
        action: AgentAction | None,
        end_reason: ConsultationEndReason,
        completed_exchanges: int,
        max_consecutive_invalid_attempts: int,
        origin: str,
    ) -> ConsultationResult:
        """Record and return one consultation outcome."""
        result = ConsultationResult(
            action=action,
            end_reason=end_reason,
            completed_exchanges=completed_exchanges,
            max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
            origin=origin,
        )
        records = getattr(self, "consultation_records", None)
        if records is None:
            records = []
            self.consultation_records = records
            if getattr(self, "game", None) is not None:
                self.game.consultation_records = records
        records.append(
            {
                "agent_id": agent.agent_id,
                "user_id": agent.user.user_id,
                "origin": origin,
                "negotiation_turn": (
                    self.current_turn
                    if origin == "initial_guidance"
                    else self.current_turn + 1
                ),
                "end_reason": end_reason,
                "completed_exchanges": completed_exchanges,
                "max_consecutive_invalid_attempts": max_consecutive_invalid_attempts,
            }
        )
        return result

    def _run_agent_user_until_consent(
        self,
        agent: Any,
        game_dialogue: Any = None,
        *,
        user_speaks_first: bool = False,
        initial_user_action: UserAction | None = None,
        agreed_deal: dict | None = None,
        consultation_mode: str | None = None,
        consultation_origin: str = "agent_user",
        return_on_user_rejection: bool = False,
        require_initial_message_user: bool = False,
        subsection_label: str,
    ) -> ConsultationResult:
        """
        Run a private dialogue between one acting agent and its own user.

        ``max_agent_user_turns`` is the global message budget for ASK USER
        consultations only. Initial guidance and final confirmation do not
        consume it. Exchange limits are local to every invocation and count only
        a completed agent-message/user-response pair. Model-output retries use
        one shared per-action attempt budget and do not consume exchange quota.
        """
        d = agent.dialogue_with_user
        agent_id = agent.agent_id
        user_id = agent.user.user_id
        self.logger.subsection(subsection_label, to=agent_id)
        self.logger.subsection(subsection_label, to=user_id)
        turn_num = self.current_turn + 1
        global_message_limit_applies = consultation_origin == "ask_user"

        if initial_user_action is not None:
            if user_speaks_first:
                raise ValueError("initial_user_action and user_speaks_first cannot both be set")
            if (
                global_message_limit_applies
                and self._agent_user_messages_remaining(agent) < 2
            ):
                return self._agent_user_limit_result(
                    agent,
                    reason="message_limit",
                    completed_exchanges=0,
                    max_consecutive_invalid_attempts=0,
                    origin=consultation_origin,
                )
            d.add_turn(
                user_id,
                len(initial_user_action.output),
                initial_user_action,
                counts_toward_ask_user_limit=global_message_limit_applies,
                turn_num=self.current_turn,
            )
            self.logger.turn(user_id, initial_user_action, to=agent_id)
            self.logger.turn(user_id, initial_user_action, to=user_id)
        elif user_speaks_first:
            if (
                global_message_limit_applies
                and self._agent_user_messages_remaining(agent) < 2
            ):
                return self._agent_user_limit_result(
                    agent,
                    reason="message_limit",
                    completed_exchanges=0,
                    max_consecutive_invalid_attempts=0,
                    origin=consultation_origin,
                )
            goal = agent.user.initial_offer or "No specific goal"
            initial_action_type = (
                UserActionType.FURTHER_INSTRUCT
                if consultation_mode == "guidance"
                else UserActionType.CONSENT
            )
            user_action = UserAction(initial_action_type, output=goal)
            d.add_turn(
                user_id,
                len(goal),
                user_action,
                counts_toward_ask_user_limit=global_message_limit_applies,
                turn_num=turn_num,
            )
            self.logger.turn(user_id, user_action, to=agent_id)

        self._last_invalid_error_agent_user = None
        awaiting_initial_message_user = require_initial_message_user
        completed_exchanges = 0
        consecutive_invalid_attempts = 0
        max_consecutive_invalid_attempts = 0
        exchange_limit = self._consultation_exchange_limit(consultation_origin)

        def register_invalid_attempt() -> None:
            nonlocal consecutive_invalid_attempts, max_consecutive_invalid_attempts
            consecutive_invalid_attempts += 1
            max_consecutive_invalid_attempts = max(
                max_consecutive_invalid_attempts,
                consecutive_invalid_attempts,
            )
            return None

        agent_attempt_budget = ModelOutputAttemptBudget(actor_id=agent_id)
        while True:
            if global_message_limit_applies and self._agent_user_limit_reached(agent):
                return self._agent_user_limit_result(
                    agent,
                    reason="message_limit",
                    completed_exchanges=completed_exchanges,
                    max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
                    origin=consultation_origin,
                )
            terminal_limit_reason: Literal["exchange_limit", "message_limit"] | None = None
            if exchange_limit is not None and completed_exchanges >= exchange_limit:
                terminal_limit_reason = "exchange_limit"
            elif (
                global_message_limit_applies
                and self._agent_user_messages_remaining(agent) <= 2
            ):
                terminal_limit_reason = "message_limit"
            respond_kwargs = {
                "game": self.game,
                "game_dialogue": game_dialogue,
                "agreed_deal": agreed_deal,
                "consultation_mode": consultation_mode,
                "attempt_budget": agent_attempt_budget,
            }
            if agreed_deal is None:
                if awaiting_initial_message_user:
                    respond_kwargs["required_action"] = (
                        AgentActionType.MESSAGE_USER
                    )
                elif terminal_limit_reason is not None:
                    # The consultation must end on a participant-facing Agent
                    # message.  This action delivers that final reply to the
                    # user and then returns to the public negotiation.
                    respond_kwargs["required_action"] = (
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
                    )
            if agent_attempt_budget.exhausted:
                raise ModelOutputAttemptsExceeded(agent_attempt_budget)
            used_before_agent_call = agent_attempt_budget.used_attempts
            action = agent.respond_to_user(**respond_kwargs)
            if not self._validate_agent_user_action(
                agent,
                action,
                agreed_deal,
                consultation_mode,
            ):
                register_invalid_attempt()
                _record_retryable_model_output_failure(
                    agent_attempt_budget,
                    used_before_call=used_before_agent_call,
                    error=(
                        self._last_invalid_error_agent_user
                        or "Invalid private agent action."
                    ),
                    next_retry_mode=RetryMode.RESAMPLE,
                )
                continue
            if agreed_deal is not None and (
                action.action_type != AgentActionType.REQUEST_USER_CONFIRMATION
                or action.action_details is None
            ):
                self._last_invalid_error_agent_user = (
                    "Use ACTION: REQUEST_USER_CONFIRMATION to request your user's "
                    "decision on the exact proposal."
                )
                self.logger.turn(
                    agent_id,
                    action,
                    to=agent_id,
                    invalid_reason=self._last_invalid_error_agent_user,
                )
                register_invalid_attempt()
                _record_retryable_model_output_failure(
                    agent_attempt_budget,
                    used_before_call=used_before_agent_call,
                    error=self._last_invalid_error_agent_user,
                    next_retry_mode=RetryMode.RESAMPLE,
                )
                continue
            if (
                awaiting_initial_message_user
                and action.action_type != AgentActionType.MESSAGE_USER
            ):
                self._last_invalid_error_agent_user = (
                    "The first action after ASK USER must be MESSAGE_USER so the "
                    "user receives a message and can reply before the consultation ends."
                )
                self.logger.turn(
                    agent_id,
                    action,
                    to=agent_id,
                    invalid_reason=self._last_invalid_error_agent_user,
                )
                register_invalid_attempt()
                _record_retryable_model_output_failure(
                    agent_attempt_budget,
                    used_before_call=used_before_agent_call,
                    error=self._last_invalid_error_agent_user,
                    next_retry_mode=RetryMode.RESAMPLE,
                )
                continue
            if (
                action.action_type
                == AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
            ):
                if consultation_mode == "guidance":
                    # This is both a participant-facing reply and a state
                    # transition.  Deliver it to the user, persist it in both
                    # views, and return without requesting another user reply.
                    d.add_turn(
                        agent_id,
                        len(action.output),
                        action,
                        counts_toward_ask_user_limit=global_message_limit_applies,
                        turn_num=turn_num,
                    )
                    self.logger.turn(agent_id, action, to=agent_id)
                    self.logger.turn(agent_id, action, to=user_id)
                    self._last_invalid_error_agent_user = None
                    return self._finish_agent_user_consultation(
                        agent,
                        action=action,
                        end_reason=(
                            terminal_limit_reason or "agent_continue"
                        ),
                        completed_exchanges=completed_exchanges,
                        max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
                        origin=consultation_origin,
                    )
                self._last_invalid_error_agent_user = (
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION is valid only in a "
                    "guidance consultation."
                )
                self.logger.turn(agent_id, action, to=agent_id, invalid_reason=self._last_invalid_error_agent_user)
                register_invalid_attempt()
                _record_retryable_model_output_failure(
                    agent_attempt_budget,
                    used_before_call=used_before_agent_call,
                    error=self._last_invalid_error_agent_user,
                    next_retry_mode=RetryMode.RESAMPLE,
                )
                continue

            d.add_turn(
                agent_id,
                len(action.output),
                action,
                counts_toward_ask_user_limit=global_message_limit_applies,
                turn_num=turn_num,
            )
            self.logger.turn(agent_id, action, to=agent_id)
            self.logger.turn(agent_id, action, to=user_id)
            if terminal_limit_reason is not None:
                raise RuntimeError(
                    "A terminal guidance consultation must use "
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION."
                )
            # Do not let the user consume the last ASK USER-budget slot. Every
            # delivered user message must leave room for an agent reply.
            if (
                global_message_limit_applies
                and self._agent_user_messages_remaining(agent) < 2
            ):
                return self._agent_user_limit_result(
                    agent,
                    reason="message_limit",
                    completed_exchanges=completed_exchanges,
                    max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
                    origin=consultation_origin,
                )

            agent_reward_status = format_reward_status(
                agent.current_reward,
                agent.auxiliary_objectives,
                {"current_turn": self.current_turn + 1},
                payout_weight=getattr(agent, "payout_weight", 1.0),
            )
            user_attempt_budget = ModelOutputAttemptBudget(actor_id=user_id)
            while True:
                if user_attempt_budget.exhausted:
                    raise ModelOutputAttemptsExceeded(user_attempt_budget)
                used_before_user_call = user_attempt_budget.used_attempts
                user_action = agent.user.respond_to_agent(
                    dialogue_with_agent=d,
                    agent_id=agent_id,
                    game_dialogue=game_dialogue,
                    game=self.game,
                    agent_reward_status=agent_reward_status,
                    consultation_mode=consultation_mode,
                    attempt_budget=user_attempt_budget,
                )
                if not isinstance(user_action, UserAction) or not isinstance(
                    user_action.action_type,
                    UserActionType,
                ):
                    raise TypeError(
                        f"{user_id} returned an invalid UserAction object."
                    )
                if agreed_deal is not None and user_action.action_type not in {
                    UserActionType.CONSENT,
                    UserActionType.DONT_CONSENT,
                }:
                    self._last_invalid_error_agent_user = (
                        "Choose CONSENT to approve the exact proposal or DONT_CONSENT "
                        "to reject it."
                    )
                    self.logger.turn(
                        user_id,
                        user_action,
                        to=user_id,
                        invalid_reason=self._last_invalid_error_agent_user,
                    )
                    register_invalid_attempt()
                    _record_retryable_model_output_failure(
                        user_attempt_budget,
                        used_before_call=used_before_user_call,
                        error=self._last_invalid_error_agent_user,
                        next_retry_mode=RetryMode.RESAMPLE,
                    )
                    continue
                if (
                    consultation_mode == "guidance"
                    and user_action.action_type != UserActionType.FURTHER_INSTRUCT
                ):
                    self._last_invalid_error_agent_user = (
                        "Guidance consultations require the user to use "
                        "FURTHER_INSTRUCT."
                    )
                    self.logger.turn(
                        user_id,
                        user_action,
                        to=user_id,
                        invalid_reason=self._last_invalid_error_agent_user,
                    )
                    register_invalid_attempt()
                    _record_retryable_model_output_failure(
                        user_attempt_budget,
                        used_before_call=used_before_user_call,
                        error=self._last_invalid_error_agent_user,
                        next_retry_mode=RetryMode.RESAMPLE,
                    )
                    continue
                break
            d.add_turn(
                user_id,
                len(user_action.output),
                user_action,
                counts_toward_ask_user_limit=global_message_limit_applies,
                turn_num=turn_num,
            )
            self.logger.turn(user_id, user_action, to=agent_id)
            self.logger.turn(user_id, user_action, to=user_id)
            completed_exchanges += 1
            awaiting_initial_message_user = False
            if user_action.action_type == UserActionType.DONT_CONSENT and return_on_user_rejection:
                self._last_invalid_error_agent_user = None
                return self._finish_agent_user_consultation(
                    agent,
                    action=None,
                    end_reason="user_rejection",
                    completed_exchanges=completed_exchanges,
                    max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
                    origin=consultation_origin,
                )
            if user_action.action_type == UserActionType.CONSENT:
                self._last_invalid_error_agent_user = None
                return self._finish_agent_user_consultation(
                    agent,
                    action=action,
                    end_reason="user_consent",
                    completed_exchanges=completed_exchanges,
                    max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
                    origin=consultation_origin,
                )
            consecutive_invalid_attempts = 0
            self._last_invalid_error_agent_user = None
            agent_attempt_budget = ModelOutputAttemptBudget(actor_id=agent_id)

    def _run_final_confirmation(
        self,
        confirming_agent: Any,
        agreed_deal: dict,
        game_dialogue: Any,
    ) -> dict[str, bool]:
        """Ask the selected represented user to confirm the frozen proposal."""
        if getattr(confirming_agent, "no_user", False):
            raise ValueError("Final user confirmation requires a represented agent.")
        if getattr(confirming_agent, "no_final_confirmation", False):
            raise ValueError(
                "Final user confirmation should never run for a no_final_confirmation "
                "agent -- its action space never offers ACCEPT_AND_CONFIRM_WITH_USER."
            )
        user_id = confirming_agent.user.user_id
        confirmed = self._run_agent_user_until_consent(
            confirming_agent,
            game_dialogue,
            agreed_deal=agreed_deal,
            consultation_mode="final_decision",
            consultation_origin="final_confirmation",
            return_on_user_rejection=True,
            subsection_label=(
                ("bilateral final confirmation: "
                 if getattr(self, "_bilateral_confirmation", False)
                 else "final user confirmation: ")
                + f"{confirming_agent.agent_id} <-> {user_id} "
                f"(turn {self.current_turn + 1})"
            ),
        )
        return {user_id: confirmed.action is not None}

    def _complete_acceptance_confirmation(
        self,
        *,
        speaker: str,
        action_type: AgentActionType,
        confirming_agents: list[Any],
        agreed_deal: dict,
        dialogue: Any,
        negotiation_subsection: str,
        authorizations: dict[str, str],
    ) -> bool:
        """Complete one acceptance turn; return False when negotiation resumes."""
        try:
            outcomes: dict[str, bool] = {}
            for confirming_agent in confirming_agents:
                outcomes.update(
                    self._run_final_confirmation(confirming_agent, agreed_deal, dialogue)
                )
        except (GameLimitExceededError, ModelOutputAttemptsExceeded):
            self._restore_negotiation_log_subsections(negotiation_subsection)
            self._record_public_negotiation_turn(
                speaker,
                AgentAction(action_type, output=FINAL_CONFIRMATION_INCOMPLETE_PUBLIC_OUTPUT),
                dialogue,
            )
            raise

        succeeded = bool(outcomes) and all(outcomes.values())
        self._last_invalid_error = None
        self._restore_negotiation_log_subsections(negotiation_subsection)
        self._record_public_negotiation_turn(
            speaker,
            AgentAction(
                action_type,
                output=_format_final_confirmation_public_output(
                    outcomes, succeeded=succeeded, authorizations=authorizations
                ),
            ),
            dialogue,
        )
        if self.game.is_invalid_game_state():
            self.game.end_reason = "invalid"
            return True
        if succeeded:
            self._finalize_deal(agreed_deal)
            return True
        return False

    def _agent_user_limit_result(
        self,
        agent: Any,
        *,
        reason: Literal[
            "exchange_limit",
            "message_limit",
        ],
        completed_exchanges: int,
        max_consecutive_invalid_attempts: int,
        origin: str,
    ) -> ConsultationResult:
        self._finish_agent_user_consultation(
            agent,
            action=None,
            end_reason=reason,
            completed_exchanges=completed_exchanges,
            max_consecutive_invalid_attempts=max_consecutive_invalid_attempts,
            origin=origin,
        )
        if reason == "exchange_limit":
            limit = self._consultation_exchange_limit(origin)
            raise GameLimitExceededError(
                f"agent-user exchange limit ({limit}) was reached before "
                f"{agent.agent_id} could send its required final message"
            )
        raise GameLimitExceededError(
            f"ASK USER message limit ({self.max_agent_user_turns}) was reached before "
            f"{agent.agent_id} could send its required final message"
        )

    def _validate_agent_user_action(
        self,
        agent: Any,
        action: AgentAction,
        agreed_deal: dict | None,
        consultation_mode: str | None = None,
    ) -> bool:
        if not isinstance(action, AgentAction) or not isinstance(
            action.action_type,
            AgentActionType,
        ):
            raise TypeError(
                f"{agent.agent_id} returned an invalid AgentAction object."
            )
        if action.action_type == AgentActionType.PROPOSE:
            self._last_invalid_error_agent_user = (
                "PROPOSE is a public negotiation action and is not valid in the "
                "private agent-user channel."
            )
            self.logger.turn(
                agent.agent_id,
                action,
                to=agent.agent_id,
                invalid_reason=self._last_invalid_error_agent_user,
            )
            return False
        if action.action_type == AgentActionType.REQUEST_USER_CONFIRMATION:
            if agreed_deal is None:
                self._last_invalid_error_agent_user = (
                    "REQUEST_USER_CONFIRMATION requires the exact proposal selected "
                    "for final user confirmation."
                )
                self.logger.turn(
                    agent.agent_id,
                    action,
                    to=agent.agent_id,
                    invalid_reason=self._last_invalid_error_agent_user,
                )
                self.logger.turn(
                    agent.agent_id,
                    action,
                    to=agent.user.user_id,
                    invalid_reason=self._last_invalid_error_agent_user,
                )
                return False
            try:
                # Attach the frozen public deal from the harness before the
                # confirmation request enters private history.
                action.action_details = self.game.validate_deal(agreed_deal)
                action.missing_details = False
            except ValueError as e:
                raise ValueError(
                    f"The system-supplied tentative deal is invalid: {e}"
                ) from e
            return True
        if agreed_deal is not None:
            # The caller reports the protocol error; proposal details from a
            # wrong action must never replace the frozen tentative deal.
            return True
        if action.action_type not in {
            AgentActionType.MESSAGE_USER,
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        }:
            self._last_invalid_error_agent_user = (
                f"{action.action_type.value} is not valid in the private "
                f"{consultation_mode or 'agent-user'} dialogue."
            )
            self.logger.turn(
                agent.agent_id,
                action,
                to=agent.agent_id,
                invalid_reason=self._last_invalid_error_agent_user,
            )
            return False
        return True

    def _validate_negotiation_action(self, speaker: str, action: AgentAction) -> bool:
        if not isinstance(action, AgentAction) or not isinstance(
            action.action_type,
            AgentActionType,
        ):
            raise TypeError(f"{speaker} returned an invalid AgentAction object.")
        if action.action_type not in {
            AgentActionType.PROPOSE,
            AgentActionType.ASK_USER,
            AgentActionType.ACCEPT_AND_CONFIRM,
            AgentActionType.ACCEPT,
            AgentActionType.QUIT,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        }:
            self._last_invalid_error = (
                f"{action.action_type.value} is not a valid public negotiation action."
            )
            self.logger.turn(
                speaker,
                action,
                to=speaker,
                invalid_reason=self._last_invalid_error,
            )
            return False
        if action.action_type == AgentActionType.REQUEST_USER_CONFIRMATION:
            self._last_invalid_error = (
                "Use REQUEST_USER_CONFIRMATION in the private final user confirmation "
                "channel. Choose an action from [ALLOWED ACTIONS] for public negotiation."
            )
            self.logger.turn(
                speaker,
                action,
                to=speaker,
                invalid_reason=self._last_invalid_error,
            )
            return False
        if (
            action.action_type == AgentActionType.PROPOSE
            and not action.action_details
        ):
            action.missing_details = True
        if action.action_details is not None or action.missing_details:
            if action.missing_details:
                prev = action.output.strip()[:500]
                self._last_invalid_error = (
                    f"Your previous proposal was: \"{prev}\" - You must include a PROPOSAL_DETAILS line "
                    "with the SAME terms you described. Do not change your proposal."
                )
                self.logger.turn(speaker, action, to="both", invalid_reason=self._last_invalid_error)
                return False
            try:
                terms = self.game.validate_deal(action.action_details)
                if action.action_type == AgentActionType.PROPOSE:
                    action.action_details = terms
            except ValueError as e:
                prev = action.output.strip()[:500]
                self._last_invalid_error = f"Your previous proposal was: \"{prev}\" - {e}"
                self.logger.turn(speaker, action, to="both", invalid_reason=self._last_invalid_error)
                return False
        self._last_invalid_error = None
        return True

    def _run_negotiation(self) -> None:
        g = self.game
        dialogue = g.game_dialogue
        negotiation_subsection = (
            f"agent-agent dialogue (negotiation, max {self.max_negotiation_turns} turns)"
        )
        self.logger.subsection(negotiation_subsection, to="both")
        self._last_invalid_error = None

        while g.end_reason is None and self.current_turn < self.max_negotiation_turns:
            speaker = dialogue.current_speaker
            agent = g.agent_1 if speaker == g.agent_1.agent_id else g.agent_2
            decision_budget = ModelOutputAttemptBudget(actor_id=speaker)
            self._update_turn_state(next_speaker=speaker)
            consulted_this_turn = False
            consultation_state: str | None = None

            while g.end_reason is None:
                if decision_budget.exhausted:
                    raise ModelOutputAttemptsExceeded(decision_budget)
                used_before_agent_call = decision_budget.used_attempts
                action = agent.respond_to_agent(
                    dialogue,
                    game=self.game,
                    consulted_this_turn=consulted_this_turn,
                    consultation_state=consultation_state,
                    attempt_budget=decision_budget,
                )
                if not self._validate_negotiation_action(speaker, action):
                    _record_retryable_model_output_failure(
                        decision_budget,
                        used_before_call=used_before_agent_call,
                        error=self._last_invalid_error or "Invalid negotiation action.",
                        next_retry_mode=RetryMode.RESAMPLE,
                    )
                    continue

                if action.action_type == AgentActionType.ASK_USER:
                    if getattr(agent, "no_user", False):
                        self._last_invalid_error = "You cannot use ASK USER because this agent has no user to consult."
                        self.logger.turn(speaker, action, to=speaker, invalid_reason=self._last_invalid_error)
                        _record_retryable_model_output_failure(
                            decision_budget,
                            used_before_call=used_before_agent_call,
                            error=self._last_invalid_error,
                            next_retry_mode=RetryMode.RESAMPLE,
                        )
                        continue
                    if consulted_this_turn:
                        close_action = (
                            "ACCEPT"
                            if getattr(agent, "no_final_confirmation", False)
                            else "ACCEPT AND CONFIRM WITH USER"
                        )
                        self._last_invalid_error = (
                            "You already consulted your user during this negotiation turn. "
                            f"Now use PROPOSE or {close_action}."
                        )
                        self.logger.turn(speaker, action, to=speaker, invalid_reason=self._last_invalid_error)
                        _record_retryable_model_output_failure(
                            decision_budget,
                            used_before_call=used_before_agent_call,
                            error=self._last_invalid_error,
                            next_retry_mode=RetryMode.RESAMPLE,
                        )
                        continue
                    if self._agent_user_messages_remaining(agent) < 3:
                        self._last_invalid_error = (
                            "ASK USER requires enough private-message budget for an "
                            "agent message, a user reply, and the agent's final message."
                        )
                        self.logger.turn(
                            speaker,
                            action,
                            to=speaker,
                            invalid_reason=self._last_invalid_error,
                        )
                        _record_retryable_model_output_failure(
                            decision_budget,
                            used_before_call=used_before_agent_call,
                            error=self._last_invalid_error,
                            next_retry_mode=RetryMode.RESAMPLE,
                        )
                        continue
                    log.info("  %s: ASK USER (private consultation)", speaker)
                    self.logger.turn(speaker, action, to=speaker)
                    self._append_visible_turn(
                        agent,
                        channel=_AGENT_AGENT_CHANNEL,
                        speaker=speaker,
                        action=action,
                        turn_num=self.current_turn + 1,
                    )
                    consultation_result = self._run_agent_user_until_consent(
                        agent,
                        dialogue,
                        consultation_mode=self._consultation_mode_for_agent(agent, "guidance"),
                        consultation_origin="ask_user",
                        require_initial_message_user=True,
                        subsection_label=(
                            f"agent-user dialogue: {agent.agent_id} <-> {agent.user.user_id} "
                            f"(turn {self.current_turn + 1})"
                        ),
                    )
                    consulted_this_turn = True
                    consultation_state = (
                        "user_consultation_auto_ended"
                        if consultation_result.end_reason
                        in {
                            "exchange_limit",
                            "message_limit",
                        }
                        else "user_consultation"
                    )
                    self._last_invalid_error = None
                    # The next action is part of the public negotiation even though
                    # it occurs in the same numbered turn as the private consultation.
                    # Restore both participants' raw-log subsections so subsequent
                    # public actions are not attributed to the private dialogue.
                    self.logger.subsection(negotiation_subsection, to=speaker)
                    self.logger.subsection(
                        negotiation_subsection,
                        to=agent.user.user_id,
                    )
                    decision_budget = ModelOutputAttemptBudget(actor_id=speaker)
                    continue

                if action.action_type == AgentActionType.ACCEPT_AND_CONFIRM:
                    agreed = _get_last_propose_from_other(dialogue, speaker)
                    if agreed is None:
                        self._last_invalid_error = (
                            "You used ACCEPT AND CONFIRM WITH USER but there is no prior proposal from the other agent to accept. "
                            "Use PROPOSE to make an offer first."
                        )
                        self.logger.turn(speaker, action, to=speaker, invalid_reason=self._last_invalid_error)
                        _record_retryable_model_output_failure(
                            decision_budget,
                            used_before_call=used_before_agent_call,
                            error=self._last_invalid_error,
                            next_retry_mode=RetryMode.RESAMPLE,
                        )
                        continue
                    log.info(
                        "  %s: ACCEPT AND CONFIRM WITH USER "
                        "(own-user final confirmation, deal=%s)",
                        speaker,
                        agreed,
                    )
                    self.logger.turn(speaker, action, to=speaker)
                    visible_acceptance = AgentAction(
                        AgentActionType.ACCEPT_AND_CONFIRM,
                        output=action.output,
                        thinking=action.thinking,
                        action_details=dict(agreed),
                    )
                    self._append_visible_turn(
                        agent,
                        channel=_AGENT_AGENT_CHANNEL,
                        speaker=speaker,
                        action=visible_acceptance,
                        turn_num=self.current_turn + 1,
                    )
                    counterparty_agent = (
                        self.game.agent_2
                        if agent is self.game.agent_1
                        else self.game.agent_1
                    )
                    if self._bilateral_confirmation:
                        confirming_agents = [self.game.agent_1, self.game.agent_2]
                        authorizations: dict[str, str] = {}
                        status = "Bilateral final confirmation is in progress."
                    else:
                        confirming_agents = [agent]
                        authorizations = {counterparty_agent.agent_id: "PROPOSE"}
                        status = (
                            f"{speaker} is requesting approval from {agent.user.user_id}."
                        )
                    counterparty_acceptance = AgentAction(
                        AgentActionType.ACCEPT_AND_CONFIRM,
                        output=(
                            f"{speaker} selected ACCEPT AND CONFIRM WITH USER for "
                            f"the exact proposal shown below. {status}"
                        ),
                        action_details=dict(agreed),
                    )
                    # The counterparty sees that confirmation was initiated and
                    # which frozen proposal it concerns, but never the initiating
                    # model's private OUTPUT or THINKING.
                    self._append_visible_turn(
                        counterparty_agent,
                        channel=_AGENT_AGENT_CHANNEL,
                        speaker=speaker,
                        action=counterparty_acceptance,
                        turn_num=self.current_turn + 1,
                    )
                    self.logger.turn(
                        speaker,
                        counterparty_acceptance,
                        to=counterparty_agent.agent_id,
                    )
                    if self._complete_acceptance_confirmation(
                        speaker=speaker,
                        action_type=action.action_type,
                        confirming_agents=confirming_agents,
                        agreed_deal=agreed,
                        dialogue=dialogue,
                        negotiation_subsection=negotiation_subsection,
                        authorizations=authorizations,
                    ):
                        return

                    # A rejected confirmation has consumed this speaker's public
                    # turn. Leave the inner loop so Dialogue alternates speakers.
                    break

                if action.action_type == AgentActionType.ACCEPT:
                    agreed = _get_last_propose_from_other(dialogue, speaker)
                    if agreed is None:
                        self._last_invalid_error = (
                            "You used ACCEPT but there is no prior proposal from the other agent to accept. "
                            "Use PROPOSE to make an offer first."
                        )
                        self.logger.turn(speaker, action, to="both", invalid_reason=self._last_invalid_error)
                        _record_retryable_model_output_failure(
                            decision_budget,
                            used_before_call=used_before_agent_call,
                            error=self._last_invalid_error,
                            next_retry_mode=RetryMode.RESAMPLE,
                        )
                        continue
                    if _is_represented(agent):
                        raise RuntimeError(
                            f"{speaker} has a user who must confirm deals; plain ACCEPT "
                            "is not in its action space."
                        )
                    proposing_agent = (
                        g.agent_2 if agent is g.agent_1 else g.agent_1
                    )
                    requires_confirmation = _is_represented(proposing_agent)
                    # Keep the initial intent in the actor's raw log. The public
                    # confirmation result is the one shared, counted turn.
                    self.logger.turn(
                        speaker, action, to=speaker if requires_confirmation else "both"
                    )
                    self._append_public_visible_turn(
                        speaker,
                        action,
                        turn_num=self.current_turn + 1,
                    )
                    if requires_confirmation:
                        if self._complete_acceptance_confirmation(
                            speaker=speaker,
                            action_type=action.action_type,
                            confirming_agents=[proposing_agent],
                            agreed_deal=agreed,
                            dialogue=dialogue,
                            negotiation_subsection=negotiation_subsection,
                            authorizations={speaker: "ACCEPT"},
                        ):
                            return
                        break
                    self._finalize_deal(agreed)
                    return

                if action.action_type == AgentActionType.QUIT:
                    g.end_reason = "quit"
                    g.quitter_user_id = agent.user.user_id
                    self.logger.turn(speaker, action, to="both")
                    self._append_public_visible_turn(
                        speaker,
                        action,
                        turn_num=self.current_turn + 1,
                    )
                    return

                self._record_public_negotiation_turn(speaker, action, dialogue)
                detail = (
                    f" ({action.action_details})"
                    if action.action_type == AgentActionType.PROPOSE and action.action_details
                    else ""
                )
                print(f"  {speaker}: {action.action_type.value}{detail}")
                if self.game.is_invalid_game_state():
                    self.game.end_reason = "invalid"
                    return
                break

        if g.end_reason is None:
            raise GameLimitExceededError(
                f"max_negotiation_turns ({self.max_negotiation_turns}) exceeded without deal or quit"
            )

    def _restore_negotiation_log_subsections(self, negotiation_subsection: str) -> None:
        """Return participant raw logs to the public negotiation subsection."""
        for agent in (self.game.agent_1, self.game.agent_2):
            if getattr(agent, "no_user", False):
                continue
            self.logger.subsection(negotiation_subsection, to=agent.agent_id)
            self.logger.subsection(
                negotiation_subsection,
                to=agent.user.user_id,
            )

    def _record_public_negotiation_turn(
        self,
        speaker: str,
        action: AgentAction,
        dialogue: Any,
    ) -> None:
        """Record one public action and apply all global-turn bookkeeping."""
        dialogue.add_turn(
            speaker,
            len(action.output),
            action,
            turn_num=self.current_turn + 1,
        )
        self._append_public_visible_turn(
            speaker,
            action,
            turn_num=self.current_turn + 1,
        )
        self.current_turn += 1
        self.turns_by_agent[speaker] = self.turns_by_agent.get(speaker, 0) + 1
        self._update_turn_state()
        self._apply_per_turn_penalties()
        self.logger.turn(speaker, action, to="both")

    def _finalize_deal(self, agreed: dict) -> None:
        accepted_action = AgentAction(AgentActionType.PROPOSE, action_details=agreed)
        self.game.end_reason = "deal"
        self.game.record_deal(accepted_action, accepted_action)
        log.info("  Deal finalized: %s", agreed)

    def _apply_per_turn_penalties(self) -> None:
        ctx = {
            "end_reason": self.game.end_reason,
            "current_turn": self.current_turn,
        }
        for agent in (self.game.agent_1, self.game.agent_2):
            apply_objective_penalties_per_turn(agent, ctx)
        a1_zero = self.game.agent_1.current_reward <= 0
        a2_zero = self.game.agent_2.current_reward <= 0
        if a1_zero or a2_zero:
            self.game.end_reason = "reward_zero"
            if a1_zero and a2_zero:
                self.game.reward_zero_loser_user_id = None
                log.info("  Both agents reward hit 0 - game over, both lose")
            else:
                loser = self.game.agent_1 if a1_zero else self.game.agent_2
                self.game.reward_zero_loser_user_id = loser.user.user_id
                log.info("  %s reward hit 0 - game over, %s loses", loser.agent_id, loser.user.user_id)
            raise GameLimitExceededError(f"agent reward hit 0 at turn {self.current_turn}")
