"""
Writes results.json to log_dir after each game.
Used for cross-run analysis and visualization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .action import AgentActionType, UserActionType
from .agent_objective import LinearRoundPenalty, LinearTurnPenalty

if TYPE_CHECKING:
    from .game import Game


def _json_safe_copy(value: Any) -> Any:
    """Return a JSON-safe copy of nested metadata."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        return json.loads(json.dumps(value, default=str))


def _result_metadata_from_config(config: dict) -> dict[str, Any]:
    """Return optional metadata supplied by the sweep runner."""
    raw_metadata = config.get("result_metadata") or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}

    def _has_value(value: Any) -> bool:
        return value not in (None, "")

    passthrough_fields = (
        "base_config_path",
        "base_id",
        "base_label",
        "sweep_config_path",
        "label",
        "combo_id",
        "combo_index",
        "combo_label",
        "combo_overrides",
    )
    for field in passthrough_fields:
        value = config.get(field)
        if _has_value(value) and not _has_value(metadata.get(field)):
            metadata[field] = value

    return metadata


def write_results(
    log_dir: Path,
    game: "Game",
    game_id: str,
    config: dict,
    config_name: str = "",
) -> None:
    """Write results.json to log_dir with structured game outcome data."""
    log_dir = Path(log_dir)

    # Agent-agent negotiation trajectory
    trajectory = []
    for turn in game.game_dialogue.conversation_history:
        action = turn.get("action")
        if action and action.action_type == AgentActionType.PROPOSE and action.action_details:
            trajectory.append({
                "turn": turn.get("turn_num"),
                "round": turn.get("round_num"),
                "speaker": turn["speaker"],
                "proposal": action.action_details,
            })

    # Agent-user trajectories: proposal/consent summaries grouped by global negotiation turn.
    user_trajectories: dict[str, list] = {}
    for agent in game.agents:
        history = agent.dialogue_with_user.conversation_history
        # Group turns by negotiation turn; find last PROPOSE and rejection count per turn.
        from collections import defaultdict
        by_turn: dict = defaultdict(list)
        for turn in history:
            tn = turn.get("turn_num")
            if tn is not None:
                by_turn[tn].append(turn)
        mandates = []
        for tn in sorted(by_turn):
            turn_group = by_turn[tn]
            last_propose = None
            n_rejections = 0
            for turn in turn_group:
                action = turn.get("action")
                if not action:
                    continue
                if action.action_type in {
                    AgentActionType.PROPOSE,
                    AgentActionType.REQUEST_USER_CONFIRMATION,
                } and action.action_details:
                    last_propose = action.action_details
                elif action.action_type == UserActionType.DONT_CONSENT:
                    n_rejections += 1
            if last_propose is not None:
                mandates.append({"turn": tn, "proposal": last_propose, "rejections": n_rejections})
        user_trajectories[agent.agent_id] = mandates

    # Payouts and scores
    payouts = {}
    scores = {}
    for agent in game.agents:
        user = agent.user
        payout = game.calculate_payout(user)
        payouts[user.user_id] = payout
        scores[agent.agent_id] = (
            round(agent.current_reward + agent.payout_weight * payout, 4)
            if payout is not None
            else None
        )

    # Disabled user configs still identify payout parties, not active models.
    user_enabled = {
        agent.user.user_id: (
            not getattr(
                agent, "no_user", config.get(agent.agent_id, {}).get("no_user", False)
            )
            and config.get(agent.user.user_id, {}).get("type") != "none"
        )
        for agent in game.agents
    }

    # Models
    models = {}
    for key in ("agent_1", "agent_2", "user_1", "user_2"):
        if key in user_enabled and not user_enabled[key]:
            continue
        cfg = config.get(key, {})
        if "model" in cfg:
            models[key] = cfg["model"]

    # Record the response protocol of each agent and enabled user.
    response_protocols: dict[str, str] = {}
    thinking_required: dict[str, bool] = {}
    for agent in game.agents:
        agent_prompts = getattr(agent, "prompts", {}) or {}
        response_protocols[agent.agent_id] = agent_prompts.get(
            "response_protocol", "legacy"
        )
        thinking_required[agent.agent_id] = bool(
            getattr(agent, "require_thinking", False)
        )
        user = agent.user
        user_prompts = getattr(user, "prompts", {}) or {}
        if not user_enabled[user.user_id]:
            continue
        response_protocols[user.user_id] = user_prompts.get(
            "response_protocol", "legacy"
        )
        thinking_required[user.user_id] = bool(
            getattr(user, "require_thinking", False)
        )

    # Private info per user
    private_info: dict[str, Any] = {}
    for agent in game.agents:
        private_info[agent.user.user_id] = dict(agent.user.private_info)

    # Initial rewards and per-negotiation-turn penalty values.
    initial_rewards = {}
    turn_penalties = {}
    for agent in game.agents:
        cfg = config.get(agent.agent_id, {})
        initial_rewards[agent.agent_id] = float(cfg.get("initial_reward", 100))
        penalty = 0.0
        for obj in agent.auxiliary_objectives:
            if isinstance(obj, LinearRoundPenalty):
                penalty = obj.penalty_per_round
                break
            if isinstance(obj, LinearTurnPenalty):
                penalty = obj.penalty_per_turn
                break
        turn_penalties[agent.agent_id] = penalty

    result_metadata = _result_metadata_from_config(config)

    results = {
        "game_id": game_id,
        "config_name": config_name,
        "sweep_id": config.get("sweep_id", ""),
        "label": result_metadata.get("label", ""),
        "base_config_path": result_metadata.get("base_config_path", ""),
        "base_id": result_metadata.get("base_id", ""),
        "base_label": result_metadata.get("base_label", ""),
        "sweep_config_path": result_metadata.get("sweep_config_path", ""),
        "combo_id": result_metadata.get("combo_id", ""),
        "combo_index": result_metadata.get("combo_index", ""),
        "combo_label": result_metadata.get("combo_label", ""),
        "combo_overrides": result_metadata.get("combo_overrides", {}),
        "result_metadata": result_metadata,
        "game_name": config.get("game_name", ""),
        "first_speaker": game.game_dialogue.speaker_1,
        "outcome": game.end_reason,
        "turns": game.state.get("turn", game.state.get("turns_total", 0)),
        "rounds": 1,
        "final_deal": getattr(game, "agreed_terms", None),
        "payouts": payouts,
        "rewards": {a.agent_id: round(a.current_reward, 4) for a in game.agents},
        "final_scores": scores,
        "initial_rewards": initial_rewards,
        "turn_penalties": turn_penalties,
        "round_penalties": turn_penalties,
        "models": models,
        "user_enabled": user_enabled,
        "response_protocols": response_protocols,
        "thinking_required": thinking_required,
        "private_info": private_info,
        "game_params": dict(config.get("game_params", {})),
        "negotiation_trajectory": trajectory,
        "user_trajectories": user_trajectories,
        "agent_user_consultations": list(
            getattr(game, "consultation_records", [])
        ),
    }
    model_output_failure = getattr(game, "model_output_failure", None)
    if model_output_failure:
        results["model_output_failure"] = _json_safe_copy(model_output_failure)

    metrics_fn = getattr(game, "get_result_metrics", None)
    if callable(metrics_fn):
        results["game_metrics"] = metrics_fn()

    error = {
        "type": getattr(game, "end_error_type", ""),
        "message": getattr(game, "end_error_message", ""),
        "traceback": getattr(game, "end_error_traceback", ""),
    }
    if any(error.values()):
        results["error"] = error

    (log_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
