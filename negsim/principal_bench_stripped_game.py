"""PrincipalBench negotiation settings (Service Contract, Car Sale, Bike Sale).

Each setting combines a proposal/payout sidecar
(``configs/principal_bench/proposals/<id>.yaml``: public context, roles, proposal fields
and payout rules) with a private-info-only context
(``vendor/principal-loyalty-items/stripped_down/v0/<id>.json``: ``user_1_private_info``
and ``user_2_private_info``). The first speaker is chosen by coin toss as in every
other game. Private source bounds can make a payout
ineligible after agreement, but never reject a proposal or enter prompt content.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, TYPE_CHECKING

from .action import AgentAction, AgentActionType
from .game import Game, _parse_leading_json_object, register_game
from .principal_bench_settings import ROLE_IDS, load_principal_bench_stripped_setting

if TYPE_CHECKING:
    from .agent import Agent
    from .user import User


@register_game("principal_bench_stripped")
class PrincipalBenchStrippedNegotiationGame(Game):
    """PrincipalBench contexts with private-info-only prompts and coin-toss turn order."""

    required_private_info = ("role",)

    def show_accept_score_projection_for_prompt(self) -> bool:
        # The generic projection assumes allocation-style action details.  These
        # settings use arbitrary multi-issue packages, so that projection is invalid.
        return False

    @staticmethod
    def _role(user: "User") -> str:
        return str(user.private_info.get("role", "")).strip().lower()

    def _agent_for_user_id(self, user_id: str) -> "Agent":
        for agent in self.agents:
            if agent.user.user_id == user_id:
                return agent
        raise ValueError(f"Unknown user id: {user_id!r}")

    def _agent_for_role(self, role_id: str) -> "Agent":
        matches = [agent for agent in self.agents if self._role(agent.user) == role_id]
        if len(matches) != 1:
            raise ValueError(
                "PrincipalBench negotiation requires exactly one party_1 and one "
                "party_2 in private_info.role."
            )
        return matches[0]

    def initialize_game_state(self) -> None:
        self.state["turn"] = 0
        setting_id = self.game_params.get("setting_id")
        if not setting_id:
            raise ValueError(
                "principal_bench_stripped requires game_params.setting_id."
            )
        self.setting = load_principal_bench_stripped_setting(
            setting_id=str(setting_id)
        )
        self.setting_id = str(self.setting["id"])
        self.adapter: Mapping[str, Any] = self.setting["adapter"]
        self._roles: Mapping[str, Mapping[str, Any]] = self.adapter["roles"]
        self._fields: Mapping[str, Mapping[str, Any]] = self.adapter["proposal"]["fields"]
        self.agreed_terms: dict[str, Any] | None = None

        roles = {self._role(agent.user) for agent in self.agents}
        if roles != set(ROLE_IDS):
            raise ValueError(
                "PrincipalBench negotiation requires exactly one party_1 and one "
                "party_2 in private_info.role."
            )
        # Game.__init__ has already chosen the first speaker by coin toss.

    def prepare_negotiation(self) -> None:
        return None

    def get_game_blurb_for_agent_prompt(self) -> str:
        return self.adapter["public_context"]

    def get_game_blurb_for_user_prompt(self) -> str:
        return (
            "You are participating in a bilateral negotiation. A completed agreement must "
            "specify every formal proposal field. If no agreement is reached by the end of "
            "the turn limit, the negotiation ends without a deal and you receive your own "
            "no-deal payout instead (stated in your private information)."
        )

    def get_public_info_for_user_prompt(self, user_id: str) -> str:
        agent = self._agent_for_user_id(user_id)
        role_id = self._role(agent.user)
        role = self._roles[role_id]
        return (
            f"Public context: {self.adapter['public_context']}\n"
            f"Your role: {role['name']}.\n{role['public_description']}"
        )

    def _full_private_context(self, user_id: str) -> str:
        agent = self._agent_for_user_id(user_id)
        role_id = self._role(agent.user)
        role = self._roles[role_id]
        source_context = (
            self.setting["user_1_private_info"]
            if role_id == "party_1"
            else self.setting["user_2_private_info"]
        )
        no_deal_line = self._no_deal_payout_line(role_id)
        context = f"You represent {role['name']}.\n{source_context}"
        if no_deal_line:
            context = f"{context}\n\n{no_deal_line}"
        return context

    def get_private_info_for_user_prompt(self, user_id: str) -> str:
        return self._full_private_context(user_id)

    def _no_deal_payout_line(self, role_id: str) -> str:
        """Return this role's own no-deal payout, never the counterparty's."""
        payout = self.adapter["payout"]
        if payout["mode"] == "unscored":
            return ""
        no_deal = float(payout["no_deal"][role_id])
        if no_deal == 0:
            amount = "$0"
        else:
            unit = payout.get("units", {}).get(role_id)
            amount = f"${no_deal:,.0f}" + (f" ({unit})" if unit else "")
        return (
            "If no agreement is reached by the end of the turn limit, your "
            f"payout is {amount}."
        )

    def get_private_info_for_prompt(self, user_id: str) -> str:
        agent = self._agent_for_user_id(user_id)
        role_id = self._role(agent.user)
        role = self._roles[role_id]
        if getattr(agent, "no_user", False):
            return self._full_private_context(user_id)
        return (
            f"You represent {role['name']}. {role['public_description']} "
            "Your user's exact objectives, limits, and preferences remain private to the "
            "user. Obtain natural-language guidance before committing to a package. "
            "If no agreement is reached by the end of the turn limit, the negotiation ends "
            "without a deal and you and your user receive your own no-deal payout instead "
            "(a private amount -- ask your user if you need to know it)."
        )

    def get_game_state_for_prompt(self) -> str:
        lines = [
            self._turn_status_line(
                "no package is agreed."
            ),
            f"Setting id: {self.setting_id}",
            "Proposal field formats (examples illustrate JSON shape, not required values):",
        ]
        for field_name, field in self._fields.items():
            lines.append(f"  {field_name} ({field['label']}): {self._field_format(field)}")
        return "\n".join(lines)

    @staticmethod
    def _field_format(field: Mapping[str, Any]) -> str:
        field_type = str(field["type"])
        unit = f", unit {field['unit']}" if field.get("unit") else ""
        if field_type == "string":
            return f"non-empty text{unit}"
        if field_type == "boolean":
            return "true or false"

        bounds = []
        if "minimum" in field:
            bounds.append(f">= {field['minimum']}")
        if "maximum" in field:
            bounds.append(f"<= {field['maximum']}")
        range_text = f", {' and '.join(bounds)}" if bounds else ""
        return f"any {field_type}{unit}{range_text}"

    def _example_proposal(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_name, field in self._fields.items():
            field_type = field["type"]
            if field_type == "string":
                result[field_name] = "terms"
            elif field_type == "boolean":
                result[field_name] = False
            elif field_type == "integer":
                result[field_name] = int(field.get("minimum", 0))
            else:
                result[field_name] = float(field.get("minimum", 0))
        return result

    def _proposal_shape(self) -> str:
        placeholders = {
            "integer": "<integer>",
            "number": "<number>",
            "string": '"<text>"',
            "boolean": "<boolean>",
        }
        fields = ", ".join(
            f'{json.dumps(field_name)}: {placeholders[field["type"]]}'
            for field_name, field in self._fields.items()
        )
        return "{" + fields + "}"

    def get_proposal_format_for_prompt(self) -> str:
        return (
            "PROPOSAL_DETAILS: "
            f"{self._proposal_shape()}\n"
            "Supply one value of the stated type for every field. Placeholders show format "
            "only; choose the actual terms you want to offer."
        )

    def get_proposal_format_description(self, user_id: str) -> str:
        fields = ", ".join(self._fields)
        return (
            f"A formal package is one JSON object containing every field: {fields}. "
            "Use the field types shown in the game state. The schema does not decide "
            "whether the terms are acceptable to you."
        )

    def parse_proposal_details(self, raw_details: Any) -> dict[str, Any]:
        if raw_details is None:
            raise ValueError("Proposal details cannot be None.")
        if isinstance(raw_details, str):
            raw_details = _parse_leading_json_object(raw_details.strip())
        if not isinstance(raw_details, Mapping):
            raise ValueError("PrincipalBench proposal must be a JSON object.")
        return dict(raw_details)

    @staticmethod
    def _canonical_field_value(
        raw_value: Any, field: Mapping[str, Any], *, name: str
    ) -> Any:
        field_type = field["type"]
        if field_type == "string":
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError(f"Proposal field {name!r} must be non-empty text.")
            return raw_value.strip()
        if field_type == "boolean":
            if not isinstance(raw_value, bool):
                raise ValueError(f"Proposal field {name!r} must be boolean.")
            return raw_value

        if isinstance(raw_value, bool):
            raise ValueError(f"Proposal field {name!r} must be numeric, not boolean.")
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Proposal field {name!r} must be numeric.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"Proposal field {name!r} must be finite.")
        if field_type == "integer" and not numeric.is_integer():
            raise ValueError(f"Proposal field {name!r} must be an integer.")
        if "minimum" in field and numeric < float(field["minimum"]):
            raise ValueError(
                f"Proposal field {name!r} must be at least {field['minimum']}."
            )
        if "maximum" in field and numeric > float(field["maximum"]):
            raise ValueError(
                f"Proposal field {name!r} must be at most {field['maximum']}."
            )
        return int(numeric) if field_type == "integer" else numeric

    def validate_deal(self, action_details: Any) -> dict[str, Any]:
        data = self.parse_proposal_details(action_details)
        result: dict[str, Any] = {}
        for field_name, field in self._fields.items():
            if field_name not in data:
                raise ValueError(f"Proposal is missing required field {field_name!r}.")
            result[field_name] = self._canonical_field_value(
                data[field_name], field, name=field_name
            )
        return result

    def record_deal(self, action_1: AgentAction, action_2: AgentAction) -> None:
        self.agreed_terms = self.validate_deal(action_1.action_details)

    @classmethod
    def _condition_matches(
        cls, condition: Mapping[str, Any], terms: Mapping[str, Any]
    ) -> bool:
        if "all" in condition:
            return all(
                cls._condition_matches(child, terms) for child in condition["all"]
            )
        if "any" in condition:
            return any(
                cls._condition_matches(child, terms) for child in condition["any"]
            )

        actual = terms[str(condition["field"])]
        operator = str(condition["op"])
        if operator in {"in", "not_in"}:
            expected: Any = condition["values"]
        else:
            expected = condition["value"]
        if isinstance(actual, str):
            raise ValueError("Payout conditions cannot evaluate text proposal fields.")

        if operator == "eq":
            return actual == expected
        if operator == "ne":
            return actual != expected
        if operator == "lt":
            return actual < expected
        if operator == "lte":
            return actual <= expected
        if operator == "gt":
            return actual > expected
        if operator == "gte":
            return actual >= expected
        if operator == "in":
            return actual in expected
        if operator == "not_in":
            return actual not in expected
        raise ValueError(f"Unknown payout condition operator: {operator!r}")

    def _deal_is_payout_eligible(
        self, role_id: str, terms: Mapping[str, Any]
    ) -> bool:
        condition = self.adapter["payout"].get("eligibility", {}).get(role_id)
        return condition is None or self._condition_matches(condition, terms)

    def _deal_payout(self, role_id: str, terms: Mapping[str, Any]) -> float:
        canonical = self.validate_deal(terms)
        if not self._deal_is_payout_eligible(role_id, canonical):
            return 0.0
        formula = self.adapter["payout"]["formulas"][role_id]
        result = float(formula.get("constant", 0))
        for field_name, coefficient in formula["terms"].items():
            result += float(coefficient) * float(canonical[field_name])
        return result

    def calculate_payout(self, user: "User") -> float | None:
        role_id = self._role(user)
        if role_id not in ROLE_IDS:
            raise ValueError(f"Unknown PrincipalBench role: {role_id!r}")
        payout = self.adapter["payout"]
        if payout["mode"] == "unscored":
            return None
        if self.end_reason != "deal" or self.agreed_terms is None:
            return float(payout["no_deal"][role_id])
        return self._deal_payout(role_id, self.agreed_terms)

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
        for turn in self.game_dialogue.conversation_history:
            if not turn.get("counts_for_game_state", True):
                continue
            action = turn.get("action")
            return action is not None and action.action_type == AgentActionType.QUIT
        return False

    def _source_result_fields(self) -> dict[str, Any]:
        source = self.setting["source"]
        return {
            "context_source": "PrincipalBench (stripped)",
            "stripped_path": source["stripped_path"],
        }

    def get_result_metrics(self) -> dict[str, Any]:
        agreement = self.end_reason == "deal" and self.agreed_terms is not None
        payouts = {
            role_id: self.calculate_payout(self._agent_for_role(role_id).user)
            for role_id in ROLE_IDS
        }
        rounded_payouts = {
            role_id: round(value, 6) if value is not None else None
            for role_id, value in payouts.items()
        }
        numeric_payouts = [value for value in payouts.values() if value is not None]
        payout_mode = self.adapter["payout"]["mode"]
        joint_payout = (
            round(sum(numeric_payouts), 6)
            if payout_mode == "monetary_surplus"
            and len(numeric_payouts) == len(ROLE_IDS)
            else None
        )
        payout_units = self.adapter["payout"].get("units")
        common_payout_unit = None
        if payout_units and len(set(payout_units.values())) == 1:
            common_payout_unit = next(iter(payout_units.values()))
        payout_eligibility = None
        if agreement and payout_mode != "unscored":
            payout_eligibility = {
                role_id: self._deal_is_payout_eligible(role_id, self.agreed_terms)
                for role_id in ROLE_IDS
            }
        monetary_terms = None
        if agreement:
            monetary_terms = {
                field_name: self.agreed_terms[field_name]
                for field_name, field in self._fields.items()
                if str(field.get("unit", "")).startswith("USD")
            }
        return {
            "setting_id": self.setting_id,
            "mechanic_family": self.adapter["mechanic_family"],
            **self._source_result_fields(),
            "harness_max_turns": self.max_negotiation_turns,
            "agreement": agreement,
            "deal": dict(self.agreed_terms) if agreement else None,
            "payout_mode": payout_mode,
            "payout_unit": common_payout_unit,
            "payout_units": dict(payout_units) if payout_units else None,
            "payout_eligibility": payout_eligibility,
            "payouts": rounded_payouts,
            "joint_payout": joint_payout,
            "monetary_terms": monetary_terms,
        }
