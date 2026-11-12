"""
Game: abstract base. Holds the two agents and the agent-to-agent dialogue. Actual games inherit and implement payout/verification.
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from typing import Any, Literal, TYPE_CHECKING

from .action import AgentAction, AgentActionType
from .dialogue import Dialogue

if TYPE_CHECKING:
    from .agent import Agent
    from .user import User

GameEndReason = Literal[
    "quit",
    "deal",
    "invalid",
    "reward_zero",
    "limit_exceeded",
    "invalid_model_output",
    "interrupted",
    "error",
]

# Name -> Game class; use get_game_by_name to look up.
_GAME_REGISTRY: dict[str, type["Game"]] = {}


def get_game_by_name(name: str) -> type["Game"]:
    """Return the Game class registered for this name. Raises ValueError if unknown."""
    if name not in _GAME_REGISTRY:
        raise ValueError(f"Unknown game: {name}")
    return _GAME_REGISTRY[name]


def register_game(name: str):
    """Decorator: register a Game subclass under the given name.

    Raises ValueError if the name is already registered to a different class.
    """
    def _register(cls: type["Game"]) -> type["Game"]:
        if name in _GAME_REGISTRY and _GAME_REGISTRY[name] is not cls:
            raise ValueError(
                f"Game name {name!r} is already registered to "
                f"{_GAME_REGISTRY[name].__name__}; cannot also register {cls.__name__}."
            )
        _GAME_REGISTRY[name] = cls
        return cls
    return _register


def _parse_leading_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object at the start of a proposal details string."""
    decoder = json.JSONDecoder()
    try:
        parsed, _ = decoder.raw_decode(raw.strip())
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid proposal JSON: {raw!r}") from e
    if not isinstance(parsed, dict):
        raise ValueError("Proposal details must be a JSON object.")
    return parsed


def format_negotiable_terms_rule(game: Any, user_id: str) -> str:
    """Return the shared system rule that makes a game's proposal schema exhaustive."""
    description_fn = getattr(game, "get_proposal_format_description", None)
    format_fn = getattr(game, "get_proposal_format_for_prompt", None)
    description = description_fn(user_id) if callable(description_fn) else ""
    raw_format = format_fn() if callable(format_fn) else ""
    schema = str(description or raw_format or "").strip()
    if not schema:
        return ""
    return (
        "[NEGOTIABLE TERMS]\n"
        "The only terms that may be negotiated, proposed, accepted, or "
        "confirmed are the fields defined by this structured proposal schema:\n"
        f"{schema}\n"
        "Do not introduce or treat any other condition as part of the "
        "negotiation or deal."
    )


class Game(ABC):
    """
    Abstract base for a negotiation game. Owns the agent-agent dialogue and the two agents.
    Subclasses implement calculate_payout, check_for_same_proposed_deal, and either_agent_quit_first_turn.
    """

    def __init__(
        self,
        name: str,
        agent_1: Agent,
        agent_2: Agent,
        game_params: dict = {},
        state: Any = None,
    ) -> None:
        self.name = name
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        forced_first = game_params.get("first_speaker")
        if forced_first == "agent_1":
            first, second = agent_1.agent_id, agent_2.agent_id
        elif forced_first == "agent_2":
            first, second = agent_2.agent_id, agent_1.agent_id
        else:
            first, second = (agent_1.agent_id, agent_2.agent_id) if random.random() < 0.5 else (agent_2.agent_id, agent_1.agent_id)
        self.game_dialogue = Dialogue(first, second)
        self.game_params = game_params
        self.state = state if state is not None else {}
        self.max_negotiation_turns: int | None = None  # set by orchestrator after construction
        self.max_agent_user_turns: int | None = None
        self.consultation_records: list[dict[str, Any]] = []
        self.end_reason: GameEndReason | None = None
        self.reward_zero_loser_user_id: str | None = None  # set when end_reason is reward_zero
        self.quitter_user_id: str | None = None  # set when end_reason is quit

        self._validate_private_info()
        self.initialize_game_state()

    @property
    def agents(self) -> tuple["Agent", "Agent"]:
        """All agents in this game"""
        return (self.agent_1, self.agent_2)

    @abstractmethod
    def initialize_game_state(self) -> None:
        """Initialize the game state."""
        ...

    def prepare_negotiation(self) -> None:
        """Insert any harness-owned public history after turn-zero user guidance."""
        return None

    @abstractmethod
    def calculate_payout(self, user: User) -> float | None:
        """Return a payout, or None when the source does not define one."""
        ...

    @abstractmethod
    def get_proposal_format_for_prompt(self) -> str:
        """Return a short description of the expected DETAILS format for agent prompts."""
        ...

    @abstractmethod
    def parse_proposal_details(self, raw_details: Any) -> dict[str, Any]:
        """Parse model output of DETAILS into a standard format. Raises ValueError if invalid."""
        ...

    @abstractmethod
    def validate_deal(self, action_details: Any) -> dict[str, Any]:
        """Validate parsed action_details. Returns validated terms. Raises ValueError if invalid."""
        ...

    @abstractmethod
    def check_for_same_proposed_deal(self, action_1: AgentAction, action_2: AgentAction) -> bool:
        """Return True if the two Propose actions represent the same deal."""
        ...

    @abstractmethod
    def record_deal(self, action_1: AgentAction, action_2: AgentAction) -> None:
        """Called when deal is reached. Subclasses store agreed terms for calculate_payout."""
        pass

    @abstractmethod
    def either_agent_quit_first_turn(self) -> bool:
        """Return True if either agent submitted QUIT in the first turn of the game dialogue."""
        ...

    @abstractmethod
    def get_game_blurb_for_agent_prompt(self) -> str:
        """Return a short description of the game for the agent prompt (no current state)."""
        ...

    def get_game_state_for_prompt(self) -> str:
        """Return current game state for the prompt (turn status, assets, offers, etc.). Empty by default."""
        return ""

    def _turn_status_line(self, no_deal_suffix: str | None = None) -> str:
        """Return raw game state with turn and optional outcome on separate lines.

        The prompt wrapper removes the raw ``Turn:`` line and supplies the
        canonical public-action status.  A game-specific no-deal outcome remains
        as static state and deliberately does not restate when the limit applies.
        """
        turn_num = int(self.state.get("turn", self.state.get("turns_total", 0)) or 0)
        if self.max_negotiation_turns is not None:
            line = f"Turn: {turn_num} of {self.max_negotiation_turns}"
        else:
            line = f"Turn: {turn_num}"
        if no_deal_suffix:
            return f"{line}\nNo-deal outcome: {no_deal_suffix.strip()}"
        return line

    def get_proposal_format_description(self, user_id: str) -> str:
        """Return the [PROPOSAL FORMAT] section text for agent prompts. Override in subclasses."""
        return ""

    def get_response_format_instruction(self, details_fmt: str, actions_str: str) -> str | None:
        """Return a custom response format instruction for agent-agent turns. None = use default."""
        return None

    def get_user_consultation_response_format_instruction(self, details_fmt: str, actions_str: str) -> str | None:
        """Return a custom response format instruction for agent-user consultation turns. None = use default."""
        return None

    def show_accept_score_projection_for_prompt(self) -> bool:
        """Whether agents should see a projected score for accepting the previous offer."""
        return False

    def parse_raw_agent_response(
        self, raw: str, *, allow_quit: bool, allow_ask_user: bool, allow_accept: bool
    ) -> tuple | None:
        """Parse raw model output for agent-agent turns.

        Return ``None`` to use the default parser. ``ValueError`` is reserved for
        invalid model-supplied content; configuration and implementation errors
        must use their natural non-``ValueError`` exception type.
        """
        return None

    def extract_declared_action_for_repair(
        self,
        raw: str,
        *,
        allow_quit: bool,
        allow_ask_user: bool,
        allow_accept: bool,
    ) -> tuple[AgentActionType, Any] | tuple[AgentActionType, Any, str] | None:
        """Return an explicitly declared custom-protocol action, if unambiguous.

        Standard THINKING/ACTION responses are handled by ``Agent``. Games with
        another wire format may override this hook so format repair can freeze
        the model's declared action, proposal details, and optional output text
        instead of guessing from a parser fallback. A model-content ambiguity may
        raise ``ValueError`` with an ``Unrecoverable format error:`` prefix;
        other exceptions are treated as implementation failures.
        """
        return None

    # Subclasses set this to require specific private_info keys on each user.
    required_private_info: tuple[str, ...] = ()

    def _validate_private_info(self) -> None:
        """Raise ValueError if any required private_info field is missing from either user."""
        if not self.required_private_info:
            return
        for agent in (self.agent_1, self.agent_2):
            user = agent.user
            missing = [k for k in self.required_private_info if not user.private_info.get(k)]
            if missing:
                raise ValueError(
                    f"User {user.user_id!r} is missing required private_info fields: {missing}. "
                    f"Set them in the config under user_{user.user_id[-1]}.private_info."
                )

    def get_private_info_for_prompt(self, user_id: str) -> str:
        """Return private information for a specific user's agent prompt. Override in subclasses."""
        return ""

    def get_public_info_for_user_prompt(self, user_id: str) -> str:
        """Return non-private role/task context for a user's own prompt."""
        return ""

    def get_private_info_for_user_prompt(self, user_id: str) -> str:
        """Return private information for a user's own prompt. Defaults to agent-visible private info."""
        return self.get_private_info_for_prompt(user_id)

    @abstractmethod
    def get_game_blurb_for_user_prompt(self) -> str:
        """Return a simplified game description for the user prompt. Override in subclasses."""
        ...

    def update_state(
        self,
        *,
        post_negotiation_no_deal: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Update game state based on the global negotiation turn.
        """
        if "current_turn" in kwargs:
            self.state["turn"] = kwargs["current_turn"]
            self.state["turns_total"] = kwargs["current_turn"]
        if "turns_total" in kwargs:
            self.state["turns_total"] = kwargs["turns_total"]
            self.state.setdefault("turn", kwargs["turns_total"])
        if "turns_by_agent" in kwargs:
            self.state["turns_by_agent"] = kwargs["turns_by_agent"]

    def is_invalid_game_state(self) -> bool:
        """
        Return True if game state is invalid and negotiation cannot continue.
        Override in subclasses. Default: False.
        """
        return False

    def _no_deal_payout(self, user: "User") -> float:
        """Return payout when no deal is reached. Uses private_info['batna'] if set, else 0."""
        return float(user.private_info.get("batna", 0.0))

    def _batna_line(self, user_id: str) -> str:
        """Return a prompt line describing the user's BATNA, if nonzero."""
        agent = self.agent_1 if self.agent_1.user.user_id == user_id else self.agent_2
        batna = float(agent.user.private_info.get("batna", 0.0))
        if batna > 0:
            return (
                f"Your user's outside option (BATNA): ${batna:,.0f} — "
                f"they receive this if no deal is reached, instead of $0."
            )
        return ""
