"""Scripted (offline) episodes for ``tests/test_smoke.py``.

Every model call is answered from a fixed script; no network access is used.
"""
from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path

import negsim.agent as agent_mod
import negsim.conversation_logger as logger_mod
import negsim.game as game_mod
import negsim.orchestrator as orch
import negsim.results_writer as writer
import negsim.user as user_mod

DEALS = {
    "buy_sell_harness_fixed": {"Player RED Gives": {"Widget": 1}, "Player BLUE Gives": {"ZUP": 68}},
    "casino": {"user_1": {"food": 2, "water": 2, "firewood": 1}, "user_2": {"food": 1, "water": 1, "firewood": 2}},
    "deal_or_no_deal": {"user_1": {"books": 1, "hats": 2, "balls": 0}, "user_2": {"books": 2, "hats": 0, "balls": 1}},
    "ultimatum_harness_fixed": {"Player RED Gives": {"Dollars": 35}, "Player BLUE Gives": {"Dollars": 0}},
    "pb-capit-anchored-01": {"hourly_rate_usd": 105, "contract_term": "4 weeks"},
    "pb-leak-price-01": {"price_usd": 12000},
    "pb-sanity-negot-01": {"price_usd": 725},
}


def response(action, output="Scripted response.", details=None):
    text = f"THINKING: Follow the test scenario.\nACTION: {action}\n"
    if details is not None:
        text += f"PROPOSAL_DETAILS: {json.dumps(details)}\n"
    return text + f"OUTPUT: {output}"


class ScriptedModel:
    def __init__(self):
        self.public, self.private = [], []

    def query_model(self, prompt, **kwargs):
        private = str(prompt.get("other_agent_id", "")).startswith("user_")
        queue = self.private if private else self.public
        if not queue:
            raise RuntimeError("script exhausted")
        return queue.pop(0)


def run_one(cfg: dict, scenario: dict) -> dict:
    agent_mod.create_model = lambda *a, **kw: ScriptedModel()
    user_mod.create_model = lambda *a, **kw: ScriptedModel()

    cfg = deepcopy(cfg)
    accepting = scenario["accepting"]
    proposing = 3 - accepting
    cfg["game_params"]["first_speaker"] = f"agent_{proposing}"
    cfg["orchestrator_settings"]["max_total_negotiation_turns"] = 2 * len(scenario["decisions"])
    users = [user_mod.get_user_by_type(f"user_{s}", cfg[f"user_{s}"]) for s in (1, 2)]
    agents = [agent_mod.get_agent(agent_id=f"agent_{s}", user=users[s - 1], config=cfg[f"agent_{s}"],
                                  orchestrator_settings=cfg["orchestrator_settings"]) for s in (1, 2)]
    game = game_mod.get_game_by_name(cfg["game_name"])(cfg["game_name"], *agents, game_params=cfg["game_params"])
    deal = game.validate_deal(DEALS[cfg["game_params"].get("setting_id", cfg["game_name"])])
    acc, prop = agents[accepting - 1], agents[proposing - 1]
    for agent in agents:
        if agent.no_user:
            continue
        model = getattr(agent.user, "model", None)
        if isinstance(model, ScriptedModel):
            model.public.append(response("FURTHER_INSTRUCT", "Opening preferences."))
        agent.model.private.append(response("MESSAGE_USER_AND_CONTINUE_NEGOTIATION", "Ready to propose."))
    action = scenario["action"]
    confirming = prop if action == "ACCEPT" else acc
    for consent in scenario["decisions"]:
        if not confirming.no_user:
            confirming.model.private.append(response("REQUEST_USER_CONFIRMATION", "Approve these terms?"))
            model = getattr(confirming.user, "model", None)
            if isinstance(model, ScriptedModel):
                model.public.append(response("CONSENT" if consent else "DONT_CONSENT"))
    prop.model.public = [response("PROPOSE", "Proposed terms.", deal) for _ in scenario["decisions"]]
    acc.model.public = [response(action, "Acceptance intent.") for _ in scenario["decisions"]]

    with tempfile.TemporaryDirectory() as tmp:
        runner = orch.RoundRunner(game, "synthetic", Path(tmp), cfg["orchestrator_settings"])
        try:
            end = runner.run_game()
        except Exception as error:  # limit reached etc.
            end = f"raised {type(error).__name__}"
        writer.write_results(Path(tmp), game, "synthetic", cfg)
        logger_mod.write_summary_log(Path(tmp), game=game)
        results = json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))
        summary = (Path(tmp) / "summary.log").read_text(encoding="utf-8")
    return dict(end=str(end), results=results, summary=summary)
