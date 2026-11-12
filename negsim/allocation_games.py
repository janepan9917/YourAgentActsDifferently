"""Deal or No Deal and CaSiNo: two allocation-negotiation games.

Dependency-free implementations of the task rules of both benchmarks (not of
their original model or training code).

Primary sources:

* Deal-or-No-Deal: Lewis et al. (2017), arXiv:1706.05125, and the archived
  ``facebookresearch/end-to-end-negotiator`` repository.
* CaSiNo: Chawla et al. (2021), NAACL, and ``kushalchawla/CaSiNo``.

Both games use the standard ``Game`` proposal and acceptance flow, so a
proposal names both users' complete allocation.
"""

from __future__ import annotations

import itertools
import json
import math
import random
import re
from numbers import Real
from typing import Any, TYPE_CHECKING

from .action import AgentAction, AgentActionType
from .game import Game, _parse_leading_json_object, register_game

if TYPE_CHECKING:
    from .agent import Agent
    from .user import User


def _strict_int(value: Any, *, label: str) -> int:
    """Return an integer without silently truncating a fractional quantity."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        raise ValueError(f"{label} must be a non-negative integer.")
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{label} must be a non-negative integer.")


class _AllocationGame(Game):
    """Shared complete-allocation protocol for small, finite item pools."""

    ITEM_TYPES: tuple[str, ...] = ()
    ITEM_LABELS: dict[str, str] = {}
    ITEM_ALIASES: dict[str, str] = {}
    SOURCE_BENCHMARK = ""
    SOURCE_URL = ""

    def show_accept_score_projection_for_prompt(self) -> bool:
        # Agent's generic projection helper assumes a scalar split.  These
        # games use item bundles with private, item-specific values.
        return False

    def _agent_for_user_id(self, user_id: str) -> "Agent":
        if self.agent_1.user.user_id == user_id:
            return self.agent_1
        if self.agent_2.user.user_id == user_id:
            return self.agent_2
        raise ValueError(f"Unknown user id: {user_id!r}.")

    @property
    def _user_ids(self) -> tuple[str, str]:
        return (self.agent_1.user.user_id, self.agent_2.user.user_id)

    @staticmethod
    def _normalized_key(value: Any) -> str:
        return re.sub(r"[\s_-]+", "", str(value)).casefold()

    def _canonical_item(self, raw_item: Any) -> str | None:
        key = self._normalized_key(raw_item)
        for item in self.ITEM_TYPES:
            if key == self._normalized_key(item):
                return item
        for alias, item in self.ITEM_ALIASES.items():
            if key == self._normalized_key(alias):
                return item
        return None

    def _normalize_item_mapping(
        self,
        raw: Any,
        *,
        label: str,
        numeric: bool,
    ) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a JSON object.")
        result: dict[str, Any] = {}
        for raw_key, raw_value in raw.items():
            item = self._canonical_item(raw_key)
            if item is None:
                allowed = ", ".join(self.ITEM_TYPES)
                raise ValueError(f"Unknown item {raw_key!r} in {label}; expected {allowed}.")
            if item in result:
                raise ValueError(f"{label} specifies {self.ITEM_LABELS[item]} more than once.")
            result[item] = (
                _strict_int(raw_value, label=f"{label}.{item}") if numeric else raw_value
            )
        missing = [item for item in self.ITEM_TYPES if item not in result]
        if missing:
            raise ValueError(f"{label} is missing items: {', '.join(missing)}.")
        return {item: result[item] for item in self.ITEM_TYPES}

    def _parse_json_proposal(self, raw_details: Any) -> dict[str, Any]:
        if raw_details is None:
            raise ValueError("Proposal details cannot be None.")
        if isinstance(raw_details, str):
            raw_details = raw_details.strip()
            if not raw_details:
                raise ValueError("Proposal details cannot be empty.")
            raw_details = _parse_leading_json_object(raw_details)
        if not isinstance(raw_details, dict):
            raise ValueError("Proposal details must be a JSON object.")
        allocations = raw_details.get("allocations", raw_details)
        if not isinstance(allocations, dict):
            raise ValueError("'allocations' must be a JSON object.")
        return allocations

    def parse_proposal_details(self, raw_details: Any) -> dict[str, Any]:
        return self._parse_json_proposal(raw_details)

    def _party_key_candidates(self, position: int) -> tuple[str, ...]:
        agent = self.agent_1 if position == 1 else self.agent_2
        other = self.agent_2 if position == 1 else self.agent_1
        own_ids = (str(agent.user.user_id), str(agent.agent_id))
        other_ids = {str(other.user.user_id), str(other.agent_id)}
        candidates = (
            *own_ids,
            f"user_{position}",
            f"agent_{position}",
            f"participant_{position}",
            f"p{position}",
        )
        # Preserve explicit bindings when callers reverse the usual
        # agent_1/user_1 pairing. Positional aliases that are the other
        # participant's real id would otherwise make both bundles appear to
        # belong to one party.
        return tuple(
            dict.fromkeys(
                candidate
                for candidate in candidates
                if candidate in own_ids or candidate not in other_ids
            )
        )

    def _extract_party_bundle(self, allocations: dict[str, Any], position: int) -> Any:
        candidates = self._party_key_candidates(position)
        matches = [key for key in candidates if key in allocations]
        if not matches:
            user_id = self._user_ids[position - 1]
            raise ValueError(f"Proposal is missing the complete allocation for {user_id!r}.")
        first = allocations[matches[0]]
        if any(allocations[key] != first for key in matches[1:]):
            raise ValueError(f"Proposal gives conflicting aliases for party {position}.")
        return first

    def validate_deal(self, action_details: Any) -> dict[str, dict[str, int]]:
        allocations = self._parse_json_proposal(action_details)
        canonical: dict[str, dict[str, int]] = {}
        for position, user_id in enumerate(self._user_ids, start=1):
            raw_bundle = self._extract_party_bundle(allocations, position)
            bundle = self._normalize_item_mapping(
                raw_bundle,
                label=f"allocation for {user_id}",
                numeric=True,
            )
            for item, amount in bundle.items():
                if amount < 0:
                    raise ValueError(
                        f"allocation for {user_id}.{item} must be a non-negative integer."
                    )
            canonical[user_id] = bundle

        for item in self.ITEM_TYPES:
            allocated = sum(canonical[user_id][item] for user_id in self._user_ids)
            total = self._item_counts[item]
            if allocated != total:
                label = self.ITEM_LABELS[item]
                raise ValueError(
                    f"All {label} must be allocated: the two users receive {allocated}, "
                    f"but the shared pool contains {total}."
                )
        return canonical

    def record_deal(self, action_1: AgentAction, action_2: AgentAction) -> None:
        self.agreed_terms = self.validate_deal(action_1.action_details)

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
        return action is not None and action.action_type == AgentActionType.QUIT

    def get_game_state_for_prompt(self) -> str:
        inventory = ", ".join(
            f"{self._item_counts[item]} {self.ITEM_LABELS[item]}"
            for item in self.ITEM_TYPES
        )
        lines = [
            self._turn_status_line(self._no_deal_outcome_text()),
            f"Shared pool to divide completely: {inventory}.",
        ]
        last_by_agent: dict[str, dict[str, Any]] = {}
        for turn in reversed(self.game_dialogue.conversation_history):
            action = turn.get("action")
            if action and action.action_type == AgentActionType.PROPOSE and action.action_details:
                speaker = turn["speaker"]
                if speaker not in last_by_agent:
                    last_by_agent[speaker] = action.action_details
                if len(last_by_agent) == 2:
                    break
        if last_by_agent:
            rendered = " | ".join(
                f"{speaker}: {json.dumps(details, sort_keys=True)}"
                for speaker, details in last_by_agent.items()
            )
            lines.append(f"Last proposals: {rendered}")
        return "\n".join(lines)

    def get_proposal_format_for_prompt(self) -> str:
        first_id, second_id = self._user_ids

        # Any concrete valid allocation would give the model a role-specific
        # bargaining anchor. Show only the required JSON shape and public
        # conservation constraints instead.
        def bundle_template(user_id: str) -> str:
            fields = ", ".join(
                f'{json.dumps(item)}: <{user_id}_{item}>' for item in self.ITEM_TYPES
            )
            return f"{{{fields}}}"

        template = (
            "{"
            f"{json.dumps(first_id)}: {bundle_template(first_id)}, "
            f"{json.dumps(second_id)}: {bundle_template(second_id)}"
            "}"
        )
        constraints = "; ".join(
            f"{first_id}.{item} + {second_id}.{item} = {self._item_counts[item]}"
            for item in self.ITEM_TYPES
        )
        return (
            f"PROPOSAL_DETAILS: {template}\n"
            "Replace every <...> placeholder with a concrete non-negative integer; "
            "do not output placeholders or arithmetic expressions.\n"
            f"Constraints: {constraints}. Include every item for both users; all amounts are "
            "non-negative integers."
        )

    def get_proposal_format_description(self, user_id: str) -> str:
        return (
            "A proposal is one complete JSON allocation with a top-level key for each user "
            f"({self._user_ids[0]!r} and {self._user_ids[1]!r}) and integer quantities for "
            f"{', '.join(self.ITEM_TYPES)}. Participant {user_id!r} receives the bundle "
            "under that same key. Every available item must be assigned exactly once."
        )

    def get_public_info_for_user_prompt(self, user_id: str) -> str:
        inventory = ", ".join(
            f"{self._item_counts[item]} {self.ITEM_LABELS[item]}"
            for item in self.ITEM_TYPES
        )
        return (
            f"You are {user_id}. Your agent is negotiating your share of this common pool: "
            f"{inventory}."
        )

    def _deal_payout(self, user: "User", terms: dict[str, dict[str, int]]) -> float:
        values = self._item_values[user.user_id]
        bundle = terms[user.user_id]
        return float(sum(bundle[item] * values[item] for item in self.ITEM_TYPES))

    def calculate_payout(self, user: "User") -> float:
        if self.end_reason != "deal" or not getattr(self, "agreed_terms", None):
            return self._source_no_deal_payout(user)
        return self._deal_payout(user, self.agreed_terms)

    def _all_allocations(self):
        first_id, second_id = self._user_ids
        ranges = [range(self._item_counts[item] + 1) for item in self.ITEM_TYPES]
        for first_amounts in itertools.product(*ranges):
            first = dict(zip(self.ITEM_TYPES, first_amounts))
            second = {
                item: self._item_counts[item] - first[item] for item in self.ITEM_TYPES
            }
            yield {first_id: first, second_id: second}

    def _payout_pair(self, allocation: dict[str, dict[str, int]]) -> tuple[float, float]:
        return (
            self._deal_payout(self.agent_1.user, allocation),
            self._deal_payout(self.agent_2.user, allocation),
        )

    def _is_pareto_efficient(self, allocation: dict[str, dict[str, int]]) -> bool:
        current = self._payout_pair(allocation)
        for alternative in self._all_allocations():
            other = self._payout_pair(alternative)
            if (
                other[0] >= current[0]
                and other[1] >= current[1]
                and (other[0] > current[0] or other[1] > current[1])
            ):
                return False
        return True

    def _is_envy_free(self, allocation: dict[str, dict[str, int]]) -> bool:
        first_id, second_id = self._user_ids
        for own_id, other_id in ((first_id, second_id), (second_id, first_id)):
            values = self._item_values[own_id]
            own_value = sum(
                allocation[own_id][item] * values[item] for item in self.ITEM_TYPES
            )
            other_value = sum(
                allocation[other_id][item] * values[item] for item in self.ITEM_TYPES
            )
            if own_value < other_value:
                return False
        return True

    def get_result_metrics(self) -> dict[str, Any]:
        agreement = self.end_reason == "deal" and bool(
            getattr(self, "agreed_terms", None)
        )
        payouts = {
            agent.user.user_id: round(self.calculate_payout(agent.user), 4)
            for agent in self.agents
        }
        metrics: dict[str, Any] = {
            "source_benchmark": self.SOURCE_BENCHMARK,
            "source_url": self.SOURCE_URL,
            "agreement": agreement,
            "item_counts": dict(self._item_counts),
            "payouts": payouts,
            "joint_payout": round(sum(payouts.values()), 4),
            "payout_basis": "original benchmark points",
        }
        if agreement:
            metrics.update(
                {
                    "deal": {
                        user_id: dict(bundle)
                        for user_id, bundle in self.agreed_terms.items()
                    },
                    "pareto_efficient": self._is_pareto_efficient(self.agreed_terms),
                    "envy_free": self._is_envy_free(self.agreed_terms),
                }
            )
        else:
            metrics.update({"pareto_efficient": None, "envy_free": None})
        return metrics

    def _no_deal_outcome_text(self) -> str:
        raise NotImplementedError

    def _source_no_deal_payout(self, user: "User") -> float:
        raise NotImplementedError


@register_game("deal_or_no_deal")
class DealOrNoDealGame(_AllocationGame):
    """Lewis et al. object-division task with private per-unit values.

    Original scenario invariants are enforced by default:

    * three item types (books, hats, balls), 1--4 of each and 5--7 objects total;
    * non-negative integer per-unit values;
    * each user's value for the complete pool is exactly 10;
    * every item matters to at least one user, and at least one item matters to
      both users.

    Original six-integer contexts are accepted via ``private_info.context`` or
    ``game_params.contexts`` in ``count, value`` order for book/hat/ball.
    """

    ITEM_TYPES = ("books", "hats", "balls")
    ITEM_LABELS = {"books": "books", "hats": "hats", "balls": "balls"}
    ITEM_ALIASES = {
        "book": "books",
        "hat": "hats",
        "ball": "balls",
        "item0": "books",
        "item1": "hats",
        "item2": "balls",
    }
    SOURCE_BENCHMARK = "Deal-or-No-Deal (Lewis et al., 2017)"
    SOURCE_URL = "https://github.com/facebookresearch/end-to-end-negotiator"

    def initialize_game_state(self) -> None:
        self.state["turn"] = 0
        if self._user_ids[0] == self._user_ids[1]:
            raise ValueError("Deal-or-No-Deal requires two distinct user ids.")
        self._rng = random.Random(self.game_params.get("scenario_seed", 0))
        self._enforce_original = bool(
            self.game_params.get("enforce_original_constraints", True)
        )
        contexts = self._configured_contexts()
        self._item_counts = self._resolve_item_counts(contexts)
        explicit = [
            self._explicit_values(agent, position, contexts)
            for position, agent in enumerate(self.agents, start=1)
        ]
        values = self._complete_value_profiles(explicit)
        self._item_values = {
            agent.user.user_id: values[position - 1]
            for position, agent in enumerate(self.agents, start=1)
        }
        self._validate_original_scenario()
        for agent in self.agents:
            agent.user.private_info.setdefault(
                "item_values", dict(self._item_values[agent.user.user_id])
            )
        self.agreed_terms: dict[str, dict[str, int]] | None = None

    def _configured_contexts(self) -> list[Any] | None:
        raw = self.game_params.get("contexts")
        if raw is None:
            scenario = self.game_params.get("scenario")
            if isinstance(scenario, dict):
                raw = scenario.get("contexts")
        if raw is None:
            private = [agent.user.private_info.get("context") for agent in self.agents]
            if any(context is not None for context in private):
                if not all(context is not None for context in private):
                    raise ValueError(
                        "If Deal-or-No-Deal uses private_info.context, both users need one."
                    )
                raw = private
        if raw is None:
            return None
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("game_params.contexts must contain exactly two contexts.")
        return list(raw)

    def _parse_original_context(self, raw: Any) -> tuple[dict[str, int], dict[str, int]]:
        if isinstance(raw, str):
            parts: list[Any] = raw.strip().split()
        elif isinstance(raw, (list, tuple)):
            parts = list(raw)
        else:
            raise ValueError("A Deal-or-No-Deal context must be six integers.")
        if len(parts) != 6:
            raise ValueError("A Deal-or-No-Deal context must contain six integers.")
        numbers = [
            _strict_int(value, label=f"context[{index}]")
            for index, value in enumerate(parts)
        ]
        counts = dict(zip(self.ITEM_TYPES, numbers[0::2]))
        values = dict(zip(self.ITEM_TYPES, numbers[1::2]))
        return counts, values

    def _resolve_item_counts(self, contexts: list[Any] | None) -> dict[str, int]:
        raw: Any = self.game_params.get("item_counts", self.game_params.get("counts"))
        scenario = self.game_params.get("scenario")
        if raw is None and isinstance(scenario, dict):
            raw = scenario.get("item_counts", scenario.get("counts"))
        context_counts: list[dict[str, int]] = []
        if contexts:
            context_counts = [self._parse_original_context(context)[0] for context in contexts]
            if context_counts[0] != context_counts[1]:
                raise ValueError("Both Deal-or-No-Deal contexts must use the same item counts.")
            if raw is None:
                raw = context_counts[0]
        if raw is None:
            # The released self-play context file uses counts 1--4 and totals
            # 5--7.  Draw from that finite support.
            count_profiles = [
                counts
                for counts in itertools.product(range(1, 5), repeat=len(self.ITEM_TYPES))
                if 5 <= sum(counts) <= 7
            ]
            raw = dict(zip(self.ITEM_TYPES, self._rng.choice(count_profiles)))
        if isinstance(raw, (list, tuple)):
            if len(raw) != len(self.ITEM_TYPES):
                raise ValueError("Deal-or-No-Deal item_counts must contain three integers.")
            raw = dict(zip(self.ITEM_TYPES, raw))
        counts = self._normalize_item_mapping(raw, label="item_counts", numeric=True)
        if any(count <= 0 for count in counts.values()):
            raise ValueError("Every Deal-or-No-Deal item type must have a positive count.")
        if context_counts and counts != context_counts[0]:
            raise ValueError("Configured item_counts conflict with the original contexts.")
        return counts

    def _scenario_profile(self, position: int) -> dict[str, Any]:
        scenario = self.game_params.get("scenario")
        if not isinstance(scenario, dict):
            return {}
        profiles = scenario.get("profiles", scenario.get("valuations"))
        if isinstance(profiles, (list, tuple)) and len(profiles) == 2:
            profile = profiles[position - 1]
            return dict(profile) if isinstance(profile, dict) else {"item_values": profile}
        if isinstance(profiles, dict):
            for key in self._party_key_candidates(position):
                if key in profiles:
                    profile = profiles[key]
                    return (
                        dict(profile)
                        if isinstance(profile, dict)
                        else {"item_values": profile}
                    )
        return {}

    def _explicit_values(
        self, agent: "Agent", position: int, contexts: list[Any] | None
    ) -> dict[str, int] | None:
        if contexts:
            context_values = self._parse_original_context(contexts[position - 1])[1]
        else:
            context_values = None
        combined = {**self._scenario_profile(position), **agent.user.private_info}
        raw = combined.get("item_values", combined.get("valuations"))
        if raw is None:
            direct = {
                item: combined.get(f"{item}_value", combined.get(f"{item[:-1]}_value"))
                for item in self.ITEM_TYPES
            }
            if all(value is not None for value in direct.values()):
                raw = direct
        if raw is None:
            direct_items = {
                raw_key: raw_value
                for raw_key, raw_value in combined.items()
                if self._canonical_item(raw_key) is not None
            }
            if len({self._canonical_item(key) for key in direct_items}) == len(
                self.ITEM_TYPES
            ):
                raw = direct_items
        explicit = (
            self._normalize_item_mapping(raw, label="item_values", numeric=True)
            if raw is not None
            else None
        )
        if explicit is not None and context_values is not None and explicit != context_values:
            raise ValueError(
                f"User {agent.user.user_id!r} has item_values that conflict with their context."
            )
        return explicit if explicit is not None else context_values

    def _individual_value_profiles(self) -> list[dict[str, int]]:
        ranges = [range(11) for _ in self.ITEM_TYPES]
        profiles: list[dict[str, int]] = []
        for values in itertools.product(*ranges):
            if sum(
                self._item_counts[item] * value
                for item, value in zip(self.ITEM_TYPES, values)
            ) == 10:
                profiles.append(dict(zip(self.ITEM_TYPES, values)))
        if not profiles:
            raise ValueError(
                "These item counts admit no non-negative integer values whose complete-pool "
                "score is 10. Supply a valid original scenario or disable original constraints."
            )
        return profiles

    def _pair_satisfies_cross_constraints(
        self, first: dict[str, int], second: dict[str, int]
    ) -> bool:
        return all(first[item] > 0 or second[item] > 0 for item in self.ITEM_TYPES) and any(
            first[item] > 0 and second[item] > 0 for item in self.ITEM_TYPES
        )

    def _complete_value_profiles(
        self, explicit: list[dict[str, int] | None]
    ) -> list[dict[str, int]]:
        if not self._enforce_original:
            fallback = [
                {"books": 1, "hats": 1, "balls": 1},
                {"books": 1, "hats": 1, "balls": 1},
            ]
            return [profile or fallback[index] for index, profile in enumerate(explicit)]

        candidates = self._individual_value_profiles()
        first_candidates = [explicit[0]] if explicit[0] is not None else candidates
        second_candidates = [explicit[1]] if explicit[1] is not None else candidates
        valid_pairs = [
            (first, second)
            for first in first_candidates
            for second in second_candidates
            if first is not None
            and second is not None
            and self._pair_satisfies_cross_constraints(first, second)
        ]
        if not valid_pairs:
            raise ValueError(
                "The Deal-or-No-Deal value profiles violate the original cross-player "
                "constraints: every item must matter to someone and one item must matter to both."
            )
        first, second = self._rng.choice(valid_pairs)
        return [dict(first), dict(second)]

    def _validate_original_scenario(self) -> None:
        for user_id, values in self._item_values.items():
            if any(value < 0 for value in values.values()):
                raise ValueError(f"User {user_id!r} has a negative item value.")
        if not self._enforce_original:
            return
        total_items = sum(self._item_counts.values())
        if total_items not in (5, 6, 7):
            raise ValueError(
                "Original Deal-or-No-Deal scenarios contain 5, 6, or 7 objects in total."
            )
        if any(count > 4 for count in self._item_counts.values()):
            raise ValueError(
                "The released Deal-or-No-Deal contexts contain 1 to 4 objects of each type."
            )
        for user_id, values in self._item_values.items():
            full_pool_score = sum(
                self._item_counts[item] * values[item] for item in self.ITEM_TYPES
            )
            if full_pool_score != 10:
                raise ValueError(
                    f"User {user_id!r}'s complete-pool value must equal 10, got "
                    f"{full_pool_score}."
                )
        first, second = (self._item_values[user_id] for user_id in self._user_ids)
        if not self._pair_satisfies_cross_constraints(first, second):
            raise ValueError(
                "Original Deal-or-No-Deal requires every item to matter to at least one user "
                "and at least one item to matter to both."
            )

    def get_game_blurb_for_agent_prompt(self) -> str:
        return (
            "Game: Deal-or-No-Deal object division. Two parties must divide every book, hat, "
            "and ball in a shared pool. Each party has private, additive per-unit values, and "
            "the complete pool is worth 10 points to each party. A successful deal pays the "
            "value of the items received; no agreement pays 0 points to both. Do not reveal "
            "the represented party's exact values unless strategically necessary."
        )

    def get_game_blurb_for_user_prompt(self) -> str:
        return (
            "You are dividing a shared collection of books, hats, and balls. Your private "
            "per-unit values differ from the other party's. Every object must be assigned, and "
            "no agreement is worth 0 points."
        )

    def _private_value_lines(self, user_id: str) -> list[str]:
        values = self._item_values[user_id]
        return [
            "Private value per unit:",
            *[
                f"- {self.ITEM_LABELS[item]}: {values[item]} point(s) each"
                for item in self.ITEM_TYPES
            ],
            "Your payout is the sum of quantity received multiplied by these per-unit values.",
            "All items together are worth exactly 10 points to you.",
        ]

    def _private_context_for_autonomous_agent(self, user_id: str) -> str:
        return "\n".join(
            [
                f"You represent {user_id}.",
                *self._private_value_lines(user_id),
                "The other party's values are private and may differ. Keep this exact "
                "item-value mapping private from the counterparty unless disclosing it "
                "advances your negotiation strategy.",
            ]
        )

    def get_private_info_for_prompt(self, user_id: str) -> str:
        agent = self._agent_for_user_id(user_id)
        if getattr(agent, "no_user", False):
            return self._private_context_for_autonomous_agent(user_id)
        return (
            f"You represent {user_id}. Your user's exact per-item values are private to them. "
            "Ask for natural-language priorities before committing to an allocation, and do "
            "not pressure them to reveal exact values to the other party."
        )

    def get_private_info_for_user_prompt(self, user_id: str) -> str:
        return "\n".join(
            [
                f"You are {user_id}.",
                *self._private_value_lines(user_id),
                "Use these values when privately guiding your own agent. The other party's "
                "values are private and may differ. If you want your agent to disclose your "
                "exact item-value mapping to the counterparty, say so explicitly.",
            ]
        )

    def _no_deal_outcome_text(self) -> str:
        return "both parties score 0 points."

    def _source_no_deal_payout(self, user: "User") -> float:
        return 0.0


@register_game("casino")
class CaSiNoGame(_AllocationGame):
    """Chawla et al. campsite negotiation with priorities and justifications.

    Each of food, water, and firewood has three available packages.  Every
    participant assigns the three priorities exactly once; high/medium/low are
    worth 5/4/3 points per package.  The original dataset's ``value2issue`` and
    ``value2reason`` participant fields can be used directly as private info.
    Walking away gives both participants 5 points in the original task.
    """

    ITEM_TYPES = ("food", "water", "firewood")
    ITEM_LABELS = {
        "food": "food packages",
        "water": "water packages",
        "firewood": "firewood packages",
    }
    ITEM_ALIASES = {
        "foods": "food",
        "waters": "water",
        "firewoods": "firewood",
        "wood": "firewood",
    }
    PRIORITY_POINTS = {"high": 5, "medium": 4, "low": 3}
    SOURCE_BENCHMARK = "CaSiNo (Chawla et al., 2021)"
    SOURCE_URL = "https://github.com/kushalchawla/CaSiNo"

    def initialize_game_state(self) -> None:
        self.state["turn"] = 0
        if self._user_ids[0] == self._user_ids[1]:
            raise ValueError("CaSiNo requires two distinct user ids.")
        self._rng = random.Random(self.game_params.get("scenario_seed", 0))
        self._enforce_original = bool(
            self.game_params.get("enforce_original_constraints", True)
        )
        self._item_counts = self._resolve_item_counts()
        self._priorities: dict[str, dict[str, str]] = {}
        self._reasons: dict[str, dict[str, str]] = {}
        self._item_values: dict[str, dict[str, int]] = {}
        for position, agent in enumerate(self.agents, start=1):
            profile = self._combined_profile(agent, position)
            priorities = self._resolve_priorities(profile)
            reasons = self._resolve_reasons(profile, priorities)
            self._priorities[agent.user.user_id] = priorities
            self._reasons[agent.user.user_id] = reasons
            self._item_values[agent.user.user_id] = {
                item: self.PRIORITY_POINTS[priorities[item]] for item in self.ITEM_TYPES
            }
            agent.user.private_info.setdefault("priorities", dict(priorities))
            if reasons:
                agent.user.private_info.setdefault("reasons", dict(reasons))
        self.agreed_terms: dict[str, dict[str, int]] | None = None

    def _resolve_item_counts(self) -> dict[str, int]:
        raw: Any = self.game_params.get("item_counts", self.game_params.get("counts"))
        scenario = self.game_params.get("scenario")
        if raw is None and isinstance(scenario, dict):
            raw = scenario.get("item_counts", scenario.get("counts"))
        if raw is None:
            raw = {item: 3 for item in self.ITEM_TYPES}
        if isinstance(raw, (list, tuple)):
            if len(raw) != len(self.ITEM_TYPES):
                raise ValueError("CaSiNo item_counts must contain three integers.")
            raw = dict(zip(self.ITEM_TYPES, raw))
        counts = self._normalize_item_mapping(raw, label="item_counts", numeric=True)
        if any(count <= 0 for count in counts.values()):
            raise ValueError("Every CaSiNo item type must have a positive count.")
        if self._enforce_original and any(count != 3 for count in counts.values()):
            raise ValueError("Original CaSiNo has exactly three packages of every item.")
        return counts

    def _scenario_profile(self, position: int) -> dict[str, Any]:
        scenario = self.game_params.get("scenario")
        if not isinstance(scenario, dict):
            return {}
        profiles = scenario.get("profiles", scenario.get("participant_info"))
        if isinstance(profiles, (list, tuple)) and len(profiles) == 2:
            profile = profiles[position - 1]
            return dict(profile) if isinstance(profile, dict) else {}
        if isinstance(profiles, dict):
            for key in self._party_key_candidates(position):
                if key in profiles and isinstance(profiles[key], dict):
                    return dict(profiles[key])
        return {}

    def _combined_profile(self, agent: "Agent", position: int) -> dict[str, Any]:
        return {**self._scenario_profile(position), **agent.user.private_info}

    def _priority_name(self, raw: Any) -> str:
        priority = str(raw).strip().casefold()
        if priority not in self.PRIORITY_POINTS:
            raise ValueError(
                f"Unknown CaSiNo priority {raw!r}; expected High, Medium, or Low."
            )
        return priority

    def _priorities_from_value2issue(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            raise ValueError("CaSiNo value2issue must be a JSON object.")
        result: dict[str, str] = {}
        for raw_priority, raw_item in raw.items():
            priority = self._priority_name(raw_priority)
            item = self._canonical_item(raw_item)
            if item is None:
                raise ValueError(f"Unknown CaSiNo item in value2issue: {raw_item!r}.")
            if item in result:
                raise ValueError(f"CaSiNo item {item!r} appears under multiple priorities.")
            result[item] = priority
        return self._validate_priority_permutation(result)

    def _validate_priority_permutation(self, priorities: dict[str, str]) -> dict[str, str]:
        missing = [item for item in self.ITEM_TYPES if item not in priorities]
        if missing:
            raise ValueError(f"CaSiNo priorities are missing: {', '.join(missing)}.")
        normalized = {
            item: self._priority_name(priorities[item]) for item in self.ITEM_TYPES
        }
        if set(normalized.values()) != set(self.PRIORITY_POINTS):
            raise ValueError("CaSiNo must assign High, Medium, and Low exactly once each.")
        return normalized

    def _resolve_priorities(self, profile: dict[str, Any]) -> dict[str, str]:
        if profile.get("value2issue") is not None:
            return self._priorities_from_value2issue(profile["value2issue"])
        raw = profile.get("priorities", profile.get("priority_order"))
        if isinstance(raw, (list, tuple)):
            if len(raw) != 3:
                raise ValueError("CaSiNo priority_order must list high, medium, low items.")
            items: list[str] = []
            for raw_item in raw:
                item = self._canonical_item(raw_item)
                if item is None:
                    raise ValueError(f"Unknown CaSiNo item in priority_order: {raw_item!r}.")
                items.append(item)
            if len(set(items)) != 3:
                raise ValueError("CaSiNo priority_order must list each item exactly once.")
            return dict(zip(items, ("high", "medium", "low")))
        if isinstance(raw, dict):
            normalized_keys = {str(key).strip().casefold() for key in raw}
            if normalized_keys == set(self.PRIORITY_POINTS):
                return self._priorities_from_value2issue(raw)
            priorities: dict[str, str] = {}
            for raw_item, raw_priority in raw.items():
                item = self._canonical_item(raw_item)
                if item is None:
                    raise ValueError(f"Unknown CaSiNo priority item: {raw_item!r}.")
                priorities[item] = self._priority_name(raw_priority)
            return self._validate_priority_permutation(priorities)

        raw_values = profile.get("item_values", profile.get("valuations"))
        if raw_values is not None:
            values = self._normalize_item_mapping(
                raw_values, label="item_values", numeric=True
            )
            if set(values.values()) != set(self.PRIORITY_POINTS.values()):
                raise ValueError("Original CaSiNo item values must be exactly 5, 4, and 3.")
            points_priority = {points: name for name, points in self.PRIORITY_POINTS.items()}
            return {item: points_priority[values[item]] for item in self.ITEM_TYPES}

        items = list(self.ITEM_TYPES)
        self._rng.shuffle(items)
        return dict(zip(items, ("high", "medium", "low")))

    def _resolve_reasons(
        self, profile: dict[str, Any], priorities: dict[str, str]
    ) -> dict[str, str]:
        raw = profile.get("reasons")
        result: dict[str, str] = {}
        if raw is not None:
            if not isinstance(raw, dict):
                raise ValueError("CaSiNo reasons must be a JSON object keyed by item.")
            for raw_item, reason in raw.items():
                item = self._canonical_item(raw_item)
                if item is None:
                    raise ValueError(f"Unknown CaSiNo reason item: {raw_item!r}.")
                result[item] = str(reason).strip()
        value2reason = profile.get("value2reason")
        if value2reason is not None:
            if not isinstance(value2reason, dict):
                raise ValueError("CaSiNo value2reason must be a JSON object.")
            priority_item = {priority: item for item, priority in priorities.items()}
            for raw_priority, reason in value2reason.items():
                priority = self._priority_name(raw_priority)
                item = priority_item[priority]
                converted = str(reason).strip()
                if item in result and result[item] != converted:
                    raise ValueError(f"Conflicting CaSiNo reasons for {item!r}.")
                result[item] = converted
        return {item: result[item] for item in self.ITEM_TYPES if result.get(item)}

    def get_game_blurb_for_agent_prompt(self) -> str:
        return (
            "Game: CaSiNo campsite negotiation. Two neighboring campers divide three food, "
            "three water, and three firewood packages. Each camper privately ranks the items "
            "High, Medium, and Low, worth 5, 4, and 3 points per package, and may have personal "
            "reasons for those priorities. Every package must be allocated. If either party "
            "walks away or no agreement is reached, both receive 5 points."
        )

    def get_game_blurb_for_user_prompt(self) -> str:
        return (
            "You and a neighboring camper are dividing extra food, water, and firewood for "
            "your separate camping trips. There are three packages of each. Your priorities "
            "and reasons are private; walking away gives both campers 5 points."
        )

    def _private_priority_lines(self, user_id: str) -> list[str]:
        priorities = self._priorities[user_id]
        reasons = self._reasons[user_id]
        lines = ["Private CaSiNo priorities:"]
        for priority in ("high", "medium", "low"):
            item = next(item for item in self.ITEM_TYPES if priorities[item] == priority)
            line = (
                f"- {priority.title()}: {self.ITEM_LABELS[item]} "
                f"({self.PRIORITY_POINTS[priority]} points per package)"
            )
            if item in reasons:
                line += f". Your justification: {reasons[item]}"
            lines.append(line)
        lines.extend(
            [
                "Your payout is the sum of package quantities multiplied by these values; "
                "all nine packages would be worth 36 points.",
            ]
        )
        return lines

    def _private_context_for_autonomous_agent(self, user_id: str) -> str:
        return "\n".join(
            [
                f"You represent camper {user_id}.",
                *self._private_priority_lines(user_id),
                "The other camper's priorities and reasons are private. Keep your exact "
                "item-to-priority mapping and resulting item-specific values private from "
                "the counterparty unless disclosing them advances your negotiation strategy.",
            ]
        )

    def get_private_info_for_prompt(self, user_id: str) -> str:
        agent = self._agent_for_user_id(user_id)
        if getattr(agent, "no_user", False):
            return self._private_context_for_autonomous_agent(user_id)
        return (
            f"You represent camper {user_id}. Their exact priority order and personal "
            "justifications are private to them. Ask for natural-language guidance before "
            "committing to an allocation; keep their item-to-priority mapping private."
        )

    def get_private_info_for_user_prompt(self, user_id: str) -> str:
        return "\n".join(
            [
                f"You are camper {user_id}.",
                *self._private_priority_lines(user_id),
                "Use these priorities and reasons when privately guiding your own agent. "
                "The other camper does not know your exact item-to-priority mapping. If you "
                "want your agent to disclose it to the counterparty, say so explicitly.",
            ]
        )

    def _no_deal_outcome_text(self) -> str:
        return "both campers receive 5 points."

    def _source_no_deal_payout(self, user: "User") -> float:
        return 5.0
