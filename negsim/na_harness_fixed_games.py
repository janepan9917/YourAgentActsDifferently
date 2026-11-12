"""Standalone, JSON-native Buy-Sell and Ultimatum games inspired by NegotiationArena.

Proposal dictionaries use explicit contribution labels (``Player RED Gives``
and ``Player BLUE Gives``), so dialogue history, confirmation prompts, and
results remain semantically self-describing.
"""

from __future__ import annotations

import json
import math
from typing import Any

from .action import AgentAction, AgentActionType
from .game import Game, register_game
from .response_parsing import load_json_object_without_duplicate_keys


PLAYER_RED = "Player RED"
PLAYER_BLUE = "Player BLUE"
RED_GIVES = f"{PLAYER_RED} Gives"
BLUE_GIVES = f"{PLAYER_BLUE} Gives"
MONEY_TOKEN = "ZUP"
NO_DEAL_RULE = (
    "If a deal is not reached by the end of the available turns, "
    "each party's payout is automatically zero."
)


def _require_exact_keys(value: Any, expected: set[str], *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{label} must contain exactly {sorted(expected)}; "
            f"missing={missing}, extra={extra}."
        )


def _require_plain_integer(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    return value


def _require_non_negative_integer(value: Any, *, label: str) -> int:
    amount = _require_plain_integer(value, label=label)
    if amount < 0:
        raise ValueError(f"{label} must be non-negative.")
    return amount


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _resource_dict(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty JSON object.")
    result: dict[str, int] = {}
    for resource, raw_amount in value.items():
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError(f"{label} resource names must be non-empty strings.")
        result[resource] = _require_non_negative_integer(
            raw_amount, label=f"{label}.{resource}"
        )
    return result


def _json_template(red: dict[str, Any], blue: dict[str, Any]) -> str:
    return json.dumps({RED_GIVES: red, BLUE_GIVES: blue}, ensure_ascii=False)


class _StandaloneHarnessFixedGame(Game):
    """Shared JSON protocol and deal lifecycle for the Buy-Sell and Ultimatum games."""

    PROPOSAL_SCHEMA_VERSION = "explicit_gives_json_v1"
    agreed_terms: dict[str, dict[str, int]] | None

    def __init__(
        self,
        name: str,
        agent_1: Any,
        agent_2: Any,
        game_params: dict[str, Any] | None = None,
        state: Any = None,
    ) -> None:
        params = dict(game_params or {})
        version = params.get("proposal_schema_version")
        if version != self.PROPOSAL_SCHEMA_VERSION:
            raise ValueError(
                "Harness-fixed games require game_params.proposal_schema_version="
                f"{self.PROPOSAL_SCHEMA_VERSION!r}; got {version!r}."
            )
        super().__init__(name, agent_1, agent_2, params, state)

    def parse_proposal_details(self, raw_details: Any) -> dict[str, Any]:
        if raw_details is None:
            raise ValueError("Proposal details cannot be None.")
        if isinstance(raw_details, str):
            text = raw_details.strip()
            if not text:
                raise ValueError("Proposal details cannot be empty.")
            raw_details = load_json_object_without_duplicate_keys(text)
        if not isinstance(raw_details, dict):
            raise ValueError("Proposal details must be a JSON object.")
        return raw_details

    def _parse_contributions(
        self, action_details: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data = self.parse_proposal_details(action_details)
        _require_exact_keys(data, {RED_GIVES, BLUE_GIVES}, label="Proposal")
        red = data[RED_GIVES]
        blue = data[BLUE_GIVES]
        if not isinstance(red, dict) or not isinstance(blue, dict):
            raise ValueError(
                f"{RED_GIVES!r} and {BLUE_GIVES!r} must each map to a JSON object."
            )
        return red, blue

    def _latest_confirmable_proposal(self) -> dict[str, Any] | None:
        current_speaker = self.game_dialogue.current_speaker
        for turn in reversed(self.game_dialogue.conversation_history):
            if turn.get("speaker") == current_speaker:
                continue
            action = turn.get("action")
            if (
                action
                and action.action_type == AgentActionType.PROPOSE
                and action.action_details
            ):
                return action.action_details
        return None

    def get_game_state_for_prompt(self) -> str:
        lines = [self._turn_status_line()]
        proposal = self._latest_confirmable_proposal()
        if proposal is not None:
            lines.append(
                "Latest proposal available for confirmation (contributions): "
                f"{json.dumps(proposal, ensure_ascii=False, sort_keys=True)}"
            )
        return "\n".join(lines)

    def _proposal_details_from_actions(
        self, action_1: AgentAction, action_2: AgentAction
    ) -> dict[str, dict[str, int]]:
        details = action_1.action_details or action_2.action_details
        return self.validate_deal(details)

    def check_for_same_proposed_deal(
        self, action_1: AgentAction, action_2: AgentAction
    ) -> bool:
        if (
            action_1.action_type != AgentActionType.PROPOSE
            or action_2.action_type != AgentActionType.PROPOSE
        ):
            return False
        try:
            return self.validate_deal(action_1.action_details) == self.validate_deal(
                action_2.action_details
            )
        except (TypeError, ValueError):
            return False

    def either_agent_quit_first_turn(self) -> bool:
        if not self.game_dialogue.conversation_history:
            return False
        action = self.game_dialogue.conversation_history[0].get("action")
        return bool(action and action.action_type == AgentActionType.QUIT)

    def _agent_for_user_id(self, user_id: str) -> Any:
        if user_id == self.agent_1.user.user_id:
            return self.agent_1
        if user_id == self.agent_2.user.user_id:
            return self.agent_2
        raise ValueError(f"Unknown user id: {user_id!r}.")

    def _full_private_info_for_prompt(self, user_id: str) -> str:
        """Return role-private information for the user or an autonomous agent."""
        raise NotImplementedError

    def get_private_info_for_prompt(self, user_id: str) -> str:
        """Keep exact private information from delegated agents."""
        agent = self._agent_for_user_id(user_id)
        if getattr(agent, "no_user", False):
            return self._full_private_info_for_prompt(user_id)
        role = PLAYER_RED if agent is self.agent_1 else PLAYER_BLUE
        return (
            f"You represent {role}. Your user's exact private information is visible "
            "only to them. Ask your user for preferences and guidance before "
            "making or accepting proposals; do not infer undisclosed values or constraints."
        )

    def get_private_info_for_user_prompt(self, user_id: str) -> str:
        """Users always receive their own complete private information."""
        return self._full_private_info_for_prompt(user_id)

    def _no_deal_payout(self, user: Any) -> float:
        """These games always pay zero when no deal is reached."""
        self._is_red_user(user)
        return 0.0

    def _is_red_user(self, user: Any) -> bool:
        if user is self.agent_1.user:
            return True
        if user is self.agent_2.user:
            return False
        raise ValueError("Payout requested for a user who is not a game participant.")


@register_game("buy_sell_harness_fixed")
class BuySellHarnessFixedGame(_StandaloneHarnessFixedGame):
    """One item is exchanged for an integer price under role-private values."""

    def initialize_game_state(self) -> None:
        seller_info = self.agent_1.user.private_info
        buyer_info = self.agent_2.user.private_info
        if seller_info.get("cost") is None:
            raise ValueError("Player RED seller private_info is missing 'cost'.")
        if buyer_info.get("willingness_to_pay") is None:
            raise ValueError(
                "Player BLUE buyer private_info is missing 'willingness_to_pay'."
            )
        self._commodity = str(self.game_params.get("commodity", "Widget")).strip()
        if not self._commodity or self._commodity == MONEY_TOKEN:
            raise ValueError(
                f"Commodity must be a non-empty name different from {MONEY_TOKEN!r}."
            )
        self._quantity = _require_non_negative_integer(
            seller_info.get("quantity", 1), label="Seller quantity"
        )
        if self._quantity < 1:
            raise ValueError("Seller quantity must be at least 1.")
        self._seller_cost = _finite_number(seller_info["cost"], label="Seller cost")
        self._buyer_wtp = _finite_number(
            buyer_info["willingness_to_pay"], label="Buyer willingness_to_pay"
        )
        self._buyer_budget = _require_non_negative_integer(
            buyer_info.get("budget", 1000), label="Buyer budget"
        )
        self.agreed_terms = None

    def get_game_blurb_for_agent_prompt(self) -> str:
        return (
            "Game: A bilateral sale. "
            f"{PLAYER_RED} sells exactly {self._quantity} unit(s) of "
            f"{self._commodity}; {PLAYER_BLUE} pays in {MONEY_TOKEN}. "
            "The parties negotiate a non-negative integer price. Each party's "
            "starting resources and valuation are private unless disclosed. "
            f"{NO_DEAL_RULE}"
        )

    def get_game_blurb_for_user_prompt(self) -> str:
        return (
            f"You are negotiating the sale of {self._quantity} unit(s) of "
            f"{self._commodity!r}. {PLAYER_RED} is the seller and {PLAYER_BLUE} "
            f"is the buyer, who pays in {MONEY_TOKEN}. {NO_DEAL_RULE}"
        )

    def _full_private_info_for_prompt(self, user_id: str) -> str:
        if user_id == self.agent_1.user.user_id:
            return (
                f"You are {PLAYER_RED}, the seller. Producing each unit of "
                f"{self._commodity} costs {self._seller_cost:g} {MONEY_TOKEN}. "
                f"You own {self._quantity} unit(s) and start with 0 {MONEY_TOKEN}. "
                "Higher sale profit is better. The buyer's valuation and budget "
                "are private."
            )
        if user_id == self.agent_2.user.user_id:
            return (
                f"You are {PLAYER_BLUE}, the buyer. You value each unit of "
                f"{self._commodity} at {self._buyer_wtp:g} {MONEY_TOKEN} and "
                f"have a budget of {self._buyer_budget} {MONEY_TOKEN}. Lower purchase "
                f"cost is better, conditional on buying the item. "
                "The seller's cost is private."
            )
        raise ValueError(f"Unknown user id: {user_id!r}.")

    def get_proposal_format_for_prompt(self) -> str:
        template = _json_template(
            {self._commodity: self._quantity},
            {MONEY_TOKEN: "<non-negative integer price>"},
        )
        return (
            f"PROPOSAL_DETAILS: {template}\n"
            "Output one JSON object. Replace the price placeholder with a JSON integer."
        )

    def get_proposal_format_description(self, user_id: str) -> str:
        return (
            f"A proposal is a JSON object with exactly two contribution labels: "
            f"{RED_GIVES!r}, whose only field is {self._commodity!r} fixed at "
            f"{self._quantity}, and {BLUE_GIVES!r}, whose only field is "
            f"{MONEY_TOKEN!r} with a non-negative integer price."
        )

    def validate_deal(self, action_details: Any) -> dict[str, dict[str, int]]:
        red, blue = self._parse_contributions(action_details)
        _require_exact_keys(red, {self._commodity}, label=RED_GIVES)
        _require_exact_keys(blue, {MONEY_TOKEN}, label=BLUE_GIVES)
        quantity = _require_plain_integer(
            red[self._commodity], label=f"{RED_GIVES}.{self._commodity}"
        )
        if quantity != self._quantity:
            raise ValueError(
                f"{RED_GIVES} must contain exactly {self._quantity} unit(s) of "
                f"{self._commodity!r}."
            )
        price = _require_non_negative_integer(
            blue[MONEY_TOKEN], label=f"{BLUE_GIVES}.{MONEY_TOKEN}"
        )
        return {
            RED_GIVES: {self._commodity: self._quantity},
            BLUE_GIVES: {MONEY_TOKEN: price},
        }

    def record_deal(self, action_1: AgentAction, action_2: AgentAction) -> None:
        self.agreed_terms = self._proposal_details_from_actions(action_1, action_2)

    def calculate_payout(self, user: Any) -> float:
        is_red = self._is_red_user(user)
        if self.end_reason != "deal" or self.agreed_terms is None:
            return self._no_deal_payout(user)
        price = self.agreed_terms[BLUE_GIVES][MONEY_TOKEN]
        if is_red:
            return float(price - self._seller_cost * self._quantity)
        return float(self._buyer_wtp * self._quantity - price)


@register_game("ultimatum_harness_fixed")
class UltimatumGameHarnessFixed(_StandaloneHarnessFixedGame):
    """Multi-turn ultimatum bargaining with an explicit JSON transfer schema."""

    def initialize_game_state(self) -> None:
        self._total_dollars = _require_non_negative_integer(
            self.game_params.get("player_1_dollars", 100), label="Total Dollars"
        )
        self.agreed_terms = None
        self.agreed_split: int | None = None

    def get_game_blurb_for_agent_prompt(self) -> str:
        return (
            "Game: Multi-turn ultimatum bargaining. "
            f"{PLAYER_RED} starts with {self._total_dollars} Dollars and "
            f"{PLAYER_BLUE} starts with none. Proposal JSON records contributions: "
            f"the value under {RED_GIVES!r} is transferred to {PLAYER_BLUE}, while "
            f"{BLUE_GIVES!r} must be 0. RED keeps the remainder. Either party may "
            f"counter-propose. {NO_DEAL_RULE}"
        )

    def get_game_blurb_for_user_prompt(self) -> str:
        return (
            "You are playing a multi-turn ultimatum bargaining game. "
            f"{PLAYER_RED} initially holds {self._total_dollars} Dollars. Proposal "
            f"labels explicitly state how many Dollars each party gives. {NO_DEAL_RULE}"
        )

    def _full_private_info_for_prompt(self, user_id: str) -> str:
        if user_id == self.agent_1.user.user_id:
            return (
                f"You are {PLAYER_RED}. When {RED_GIVES!r} contains Dollars: N, "
                f"{PLAYER_BLUE} receives N and you retain {self._total_dollars}-N. "
                "Your deal payout is the Dollars you retain, so lower N is better for "
                "you, conditional on reaching agreement."
            )
        if user_id == self.agent_2.user.user_id:
            return (
                f"You are {PLAYER_BLUE}. When {RED_GIVES!r} contains Dollars: N, you "
                "receive N and RED keeps the remainder. Your deal payout is N, so higher "
                "N is better for you. You may accept or counter-propose on your turn."
            )
        raise ValueError(f"Unknown user id: {user_id!r}.")

    def get_proposal_format_for_prompt(self) -> str:
        template = _json_template(
            {"Dollars": "<integer from 0 through total Dollars>"},
            {"Dollars": 0},
        )
        return (
            f"PROPOSAL_DETAILS: {template}\n"
            "Output one JSON object. Replace RED's placeholder with a JSON integer "
            f"from 0 through {self._total_dollars}; BLUE's contribution remains 0."
        )

    def get_proposal_format_description(self, user_id: str) -> str:
        return (
            f"A proposal is a JSON object with exactly two top-level labels: "
            f"{RED_GIVES!r}, whose only field is 'Dollars' with integer N, and "
            f"{BLUE_GIVES!r}, whose only field is 'Dollars' fixed at 0. N ranges "
            f"from 0 through {self._total_dollars} and is transferred from RED to BLUE."
        )

    def validate_deal(self, action_details: Any) -> dict[str, dict[str, int]]:
        red, blue = self._parse_contributions(action_details)
        _require_exact_keys(red, {"Dollars"}, label=RED_GIVES)
        _require_exact_keys(blue, {"Dollars"}, label=BLUE_GIVES)
        amount = _require_plain_integer(
            red["Dollars"], label=f"{RED_GIVES}.Dollars"
        )
        if amount < 0 or amount > self._total_dollars:
            raise ValueError(
                f"{RED_GIVES}.Dollars must be between 0 and "
                f"{self._total_dollars}, got {amount}."
            )
        blue_amount = _require_plain_integer(
            blue["Dollars"], label=f"{BLUE_GIVES}.Dollars"
        )
        if blue_amount != 0:
            raise ValueError(f"{BLUE_GIVES}.Dollars must be 0.")
        return {
            RED_GIVES: {"Dollars": amount},
            BLUE_GIVES: {"Dollars": 0},
        }

    def get_game_state_for_prompt(self) -> str:
        lines = [super().get_game_state_for_prompt()]
        proposal = self._latest_confirmable_proposal()
        if proposal is not None:
            try:
                amount = int(proposal[RED_GIVES]["Dollars"])
                lines.append(
                    "Plain outcome for that proposal: "
                    f"{PLAYER_BLUE} receives {amount} Dollars; {PLAYER_RED} retains "
                    f"{self._total_dollars - amount} Dollars."
                )
            except (KeyError, TypeError, ValueError):
                pass
        return "\n".join(line for line in lines if line)

    def record_deal(self, action_1: AgentAction, action_2: AgentAction) -> None:
        self.agreed_terms = self._proposal_details_from_actions(action_1, action_2)
        self.agreed_split = self.agreed_terms[RED_GIVES]["Dollars"]

    def calculate_payout(self, user: Any) -> float:
        is_red = self._is_red_user(user)
        if self.end_reason != "deal" or self.agreed_split is None:
            return self._no_deal_payout(user)
        if is_red:
            return float(self._total_dollars - self.agreed_split)
        return float(self.agreed_split)
