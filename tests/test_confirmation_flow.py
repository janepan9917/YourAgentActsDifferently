"""Acceptance and user-confirmation protocol, with scripted models and zero API calls.

With User (one represented agent): the represented side's user confirms every deal.
Dual User (both agents represented): any acceptance needs both users' consent.
No User: plain ACCEPT closes the deal directly.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

import negsim.main  # noqa: F401 - registers the game implementations
from negsim.agent import get_agent
from negsim.conversation_logger import write_summary_log
from negsim.game import get_game_by_name
from negsim.main import validate_user_agent_pairing
from negsim.orchestrator import GameLimitExceededError, RoundRunner
from negsim.results_writer import write_results
from negsim.run_sweep import _make_task_config, load_sweep
from negsim.user import get_user_by_type

GAMES = ["buy_sell", "ultimatum", "casino", "deal_or_no_deal", "service_contract", "car_sale", "bike_sale"]
SINGLE = {g: Path(f"sweeps/single_user/{g}/claude_haiku45.yaml") for g in GAMES}
DUAL = {g: Path(f"sweeps/dual_user/{g}/claude_haiku45.yaml") for g in GAMES}
NO_USER = {g: Path(f"sweeps/no_user/{g}/claude_haiku45.yaml") for g in GAMES}
DEALS = {
    "buy_sell_harness_fixed": {
        "Player RED Gives": {"Widget": 1}, "Player BLUE Gives": {"ZUP": 68}
    },
    "casino": {
        "user_1": {"food": 2, "water": 2, "firewood": 1},
        "user_2": {"food": 1, "water": 1, "firewood": 2},
    },
    "deal_or_no_deal": {
        "user_1": {"books": 1, "hats": 2, "balls": 0},
        "user_2": {"books": 2, "hats": 0, "balls": 1},
    },
    "ultimatum_harness_fixed": {
        "Player RED Gives": {"Dollars": 35}, "Player BLUE Gives": {"Dollars": 0}
    },
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
        self.public = []
        self.private = []
        self.prompts = []

    def query_model(self, prompt, **kwargs):
        self.prompts.append(deepcopy(prompt))
        private = str(prompt.get("other_agent_id", "")).startswith("user_")
        queue = self.private if private else self.public
        assert queue, "Unexpected model call (no network backend is installed)."
        return queue.pop(0)


def resolved_config(path, combo_id):
    sweep_id, combos, _ = load_sweep(path)
    combo = next(c for c in combos if c.combo_id == combo_id)
    cfg, _ = _make_task_config(sweep_path=path, sweep_id=sweep_id, combo=combo, target_n_runs=1)
    return cfg


def build(monkeypatch, tmp_path, cfg, proposing_slot, turn_limit):
    # Patch factories before constructing any production Agent or LLMUser.
    monkeypatch.setattr("negsim.agent.create_model", lambda *a, **kw: ScriptedModel())
    monkeypatch.setattr("negsim.user.create_model", lambda *a, **kw: ScriptedModel())
    cfg["game_params"]["first_speaker"] = f"agent_{proposing_slot}"
    cfg["orchestrator_settings"]["max_total_negotiation_turns"] = turn_limit
    users = [get_user_by_type(f"user_{s}", cfg[f"user_{s}"]) for s in (1, 2)]
    agents = [get_agent(agent_id=f"agent_{s}", user=users[s - 1], config=cfg[f"agent_{s}"],
                        orchestrator_settings=cfg["orchestrator_settings"]) for s in (1, 2)]
    game = get_game_by_name(cfg["game_name"])(cfg["game_name"], *agents, game_params=cfg["game_params"])
    key = cfg["game_params"].get("setting_id", cfg["game_name"])
    deal = game.validate_deal(DEALS[key])
    for agent in agents:
        if agent.no_user:
            def unexpected_dummy_call(*args, **kwargs):
                raise AssertionError("An inactive user was consulted")
            agent.user.respond_to_agent = unexpected_dummy_call
        else:
            agent.user.model.public = [response("FURTHER_INSTRUCT", "Opening preferences.")]
            agent.model.private = [response("MESSAGE_USER_AND_CONTINUE_NEGOTIATION", "Ready to propose.")]
    runner = RoundRunner(game, "synthetic", tmp_path, cfg["orchestrator_settings"])
    return game, agents, runner, deal


def single_user_scenario(monkeypatch, tmp_path, path, user_slot, action, decisions=(True,), turn_limit=None):
    cfg = resolved_config(path, f"user_on_agent_{user_slot}")
    accepting_slot = 3 - user_slot if action == "ACCEPT" else user_slot
    proposing_slot = 3 - accepting_slot
    game, agents, runner, deal = build(monkeypatch, tmp_path, cfg, proposing_slot,
                                       turn_limit or 2 * len(decisions))
    accepting, proposing = agents[accepting_slot - 1], agents[proposing_slot - 1]
    confirming = proposing if action == "ACCEPT" else accepting
    for consent in decisions:
        confirming.model.private.append(response("REQUEST_USER_CONFIRMATION", "Approve these terms?"))
        confirming.user.model.public.append(response("CONSENT" if consent else "DONT_CONSENT"))
    proposing.model.public = [response("PROPOSE", "Proposed terms.", deal) for _ in decisions]
    accepting.model.public = [response(action, "Acceptance intent.") for _ in decisions]
    return cfg, game, runner, accepting, proposing, confirming, deal


def dual_user_scenario(monkeypatch, tmp_path, path, accepting_slot, decisions):
    """decisions: one (user_1 consents, user_2 consents) pair per acceptance attempt."""
    cfg = resolved_config(path, "w_user_wo_instructions")
    game, agents, runner, deal = build(monkeypatch, tmp_path, cfg, 3 - accepting_slot, 2 * len(decisions))
    accepting, proposing = agents[accepting_slot - 1], agents[2 - accepting_slot]
    for pair in decisions:
        for agent, consent in zip(agents, pair):
            agent.model.private.append(response("REQUEST_USER_CONFIRMATION", "Approve these terms?"))
            agent.user.model.public.append(response("CONSENT" if consent else "DONT_CONSENT"))
    proposing.model.public = [response("PROPOSE", "Proposed terms.", deal) for _ in decisions]
    accepting.model.public = [response("ACCEPT AND CONFIRM WITH USER", "Acceptance intent.")
                              for _ in decisions]
    return cfg, game, runner, agents, deal


def final_confirmations(game):
    return [r for r in game.consultation_records if r["origin"] == "final_confirmation"]


def assert_completed(tmp_path, cfg, game, runner, confirming, deal, turns):
    assert runner.run_game() == "deal"
    assert game.agreed_terms == deal
    assert game.state["turn"] == turns
    assert runner.current_turn == turns
    confirmations = final_confirmations(game)
    assert {r["user_id"] for r in confirmations} == {confirming.user.user_id}
    assert len(confirmations) == turns // 2
    assert confirmations[-1]["end_reason"] == "user_consent"
    for agent in game.agents:
        if not agent.no_user:
            public_prompts = [p for p in agent.model.prompts
                              if p.get("other_agent_id", "").startswith("agent_")]
            assert "Opening preferences." in str(public_prompts[0]["conversation_history"])
            # The harness-added confirmation rule is the single-user one.
            for prompt in agent.model.prompts + agent.user.model.prompts:
                assert "[BILATERAL FINAL CONFIRMATION]" not in str(prompt)
    write_results(tmp_path, game, "synthetic", cfg)
    write_summary_log(tmp_path, game=game)
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    summary = (tmp_path / "summary.log").read_text(encoding="utf-8")
    assert re.findall(r"^### Turn (\d+)$", summary, re.M) == [str(n) for n in range(turns + 1)]
    assert f"{confirming.user.user_id}: CONSENT" in summary
    for agent in game.agents:
        if agent.no_user:
            user_id = agent.user.user_id
            assert not saved["user_enabled"][user_id]
            assert user_id not in saved["models"]
            assert f"{user_id}: CONSENT" not in summary
            assert user_id in saved["payouts"]


# --- With User -------------------------------------------------------------------

@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("user_slot", [1, 2])
@pytest.mark.parametrize("action", ["ACCEPT", "ACCEPT AND CONFIRM WITH USER"])
def test_single_user_every_game_confirms_the_existing_user(monkeypatch, tmp_path, game, user_slot, action):
    cfg, g, runner, _, _, confirming, deal = single_user_scenario(
        monkeypatch, tmp_path, SINGLE[game], user_slot, action)
    assert_completed(tmp_path, cfg, g, runner, confirming, deal, 2)


@pytest.mark.parametrize("user_slot", [1, 2])
@pytest.mark.parametrize("action", ["ACCEPT", "ACCEPT AND CONFIRM WITH USER"])
def test_single_user_rejection_resumes_then_new_acceptance_can_close(monkeypatch, tmp_path, user_slot, action):
    cfg, game, runner, _, _, confirming, deal = single_user_scenario(
        monkeypatch, tmp_path, SINGLE["buy_sell"], user_slot, action, decisions=(False, True))
    assert_completed(tmp_path, cfg, game, runner, confirming, deal, 4)
    confirmations = final_confirmations(game)
    assert [r["end_reason"] for r in confirmations] == ["user_rejection", "user_consent"]
    assert [r["negotiation_turn"] for r in confirmations] == [2, 4]


@pytest.mark.parametrize("user_slot", [1, 2])
@pytest.mark.parametrize("action", ["ACCEPT", "ACCEPT AND CONFIRM WITH USER"])
def test_single_user_rejection_on_last_turn_preserves_no_deal(monkeypatch, tmp_path, user_slot, action):
    _, game, runner, _, _, _, _ = single_user_scenario(
        monkeypatch, tmp_path, SINGLE["buy_sell"], user_slot, action, decisions=(False,))
    with pytest.raises(GameLimitExceededError, match="max_negotiation_turns"):
        runner.run_game()
    assert game.end_reason == "limit_exceeded"
    assert game.agreed_terms is None
    assert runner.current_turn == 2


@pytest.mark.parametrize("user_slot", [1, 2])
@pytest.mark.parametrize("action", ["ACCEPT", "ACCEPT AND CONFIRM WITH USER"])
def test_confirmation_output_exhaustion_cannot_finalize(monkeypatch, tmp_path, user_slot, action):
    _, game, runner, _, _, confirming, _ = single_user_scenario(
        monkeypatch, tmp_path, SINGLE["buy_sell"], user_slot, action)
    confirming.user.model.public = [response("FURTHER_INSTRUCT", "Opening preferences.")]
    confirming.user.model.public += [response("FURTHER_INSTRUCT") for _ in range(8)]
    assert runner.run_game() == "invalid_model_output"
    assert game.agreed_terms is None
    assert runner.current_turn == 2
    assert game.model_output_failure["actor_id"] == confirming.user.user_id
    assert game.model_output_failure["used_attempts"] == 8


@pytest.mark.parametrize("user_slot", [1, 2])
@pytest.mark.parametrize("action", ["ACCEPT", "ACCEPT AND CONFIRM WITH USER"])
def test_acceptance_without_a_proposal_never_confirms(monkeypatch, tmp_path, user_slot, action):
    from negsim.dialogue import Dialogue

    _, game, runner, accepting, proposing, _, _ = single_user_scenario(
        monkeypatch, tmp_path, SINGLE["buy_sell"], user_slot, action)
    game.game_dialogue = Dialogue(accepting.agent_id, proposing.agent_id)
    accepting.model.public = [response(action) for _ in range(8)]
    assert runner.run_game() == "invalid_model_output"
    assert game.agreed_terms is None
    assert all(r["origin"] == "initial_guidance" for r in game.consultation_records)


@pytest.mark.parametrize("user_slot", [1, 2])
def test_represented_agent_cannot_close_with_plain_accept(monkeypatch, tmp_path, user_slot):
    """Plain ACCEPT is outside a represented agent's action space; it never finalizes."""
    _, game, runner, accepting, _, _, _ = single_user_scenario(
        monkeypatch, tmp_path, SINGLE["buy_sell"], user_slot, "ACCEPT AND CONFIRM WITH USER")
    accepting.model.public = [response("ACCEPT") for _ in range(8)]
    assert runner.run_game() == "invalid_model_output"
    assert game.agreed_terms is None
    assert final_confirmations(game) == []


# --- Dual User (bilateral confirmation) ----------------------------------------------

@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("accepting_slot", [1, 2])
def test_dual_user_acceptance_needs_both_users(monkeypatch, tmp_path, game, accepting_slot):
    cfg, g, runner, agents, deal = dual_user_scenario(
        monkeypatch, tmp_path, DUAL[game], accepting_slot, decisions=[(True, True)])
    assert runner.run_game() == "deal"
    assert g.agreed_terms == deal
    confirmations = final_confirmations(g)
    # Both users are asked, in agent order, about the same frozen terms.
    assert [r["user_id"] for r in confirmations] == ["user_1", "user_2"]
    assert all(r["end_reason"] == "user_consent" for r in confirmations)
    for agent in agents:
        confirm_prompts = [p for p in agent.model.prompts + agent.user.model.prompts
                           if "BILATERAL FINAL CONFIRMATION" in str(p)]
        assert confirm_prompts, agent.agent_id
        assert not any("[FINAL USER CONFIRMATION]" in str(p)
                       for p in agent.model.prompts + agent.user.model.prompts)
    write_results(tmp_path, g, "synthetic", cfg)
    write_summary_log(tmp_path, game=g)
    summary = (tmp_path / "summary.log").read_text(encoding="utf-8")
    assert "user_1: CONSENT" in summary and "user_2: CONSENT" in summary
    assert "Agent authorization" not in summary
    assert "Bilateral final confirmation is in progress." in summary


@pytest.mark.parametrize("rejecting", [(False, True), (True, False), (False, False)])
def test_dual_user_any_rejection_resumes_negotiation(monkeypatch, tmp_path, rejecting):
    cfg, game, runner, agents, deal = dual_user_scenario(
        monkeypatch, tmp_path, DUAL["buy_sell"], 2, decisions=[rejecting, (True, True)])
    assert runner.run_game() == "deal"
    assert game.agreed_terms == deal
    assert runner.current_turn == 4
    confirmations = final_confirmations(game)
    assert [r["user_id"] for r in confirmations] == ["user_1", "user_2", "user_1", "user_2"]
    expected = ["user_consent" if c else "user_rejection" for c in rejecting]
    assert [r["end_reason"] for r in confirmations[:2]] == expected
    assert [r["negotiation_turn"] for r in confirmations] == [2, 2, 4, 4]


def test_dual_user_rejection_on_last_turn_preserves_no_deal(monkeypatch, tmp_path):
    _, game, runner, _, _ = dual_user_scenario(
        monkeypatch, tmp_path, DUAL["buy_sell"], 1, decisions=[(True, False)])
    with pytest.raises(GameLimitExceededError, match="max_negotiation_turns"):
        runner.run_game()
    assert game.agreed_terms is None


# --- No User ---------------------------------------------------------------------

@pytest.mark.parametrize("game", GAMES)
def test_no_user_agents_accept_directly(monkeypatch, tmp_path, game):
    cfg = resolved_config(NO_USER[game], "wo_user_wo_instructions")
    g, agents, runner, deal = build(monkeypatch, tmp_path, cfg, 1, 2)
    agents[0].model.public = [response("PROPOSE", "Proposed terms.", deal)]
    agents[1].model.public = [response("ACCEPT", "Accepted.")]
    assert runner.run_game() == "deal"
    assert g.agreed_terms == deal
    assert g.consultation_records == []
    # The direct-ACCEPT path does not add a counted turn.
    assert runner.current_turn == 1


# --- Configuration checks ----------------------------------------------------------

def test_disabled_user_requires_a_no_user_agent():
    cfg = resolved_config(SINGLE["buy_sell"], "user_on_agent_1")
    validate_user_agent_pairing(cfg)  # the paper's configuration is valid
    broken = deepcopy(cfg)
    broken["agent_2"]["no_user"] = False
    with pytest.raises(ValueError, match="user_2.type is 'none'"):
        validate_user_agent_pairing(broken)


@pytest.mark.parametrize("condition", ["single_user", "passive_user", "opening_only_user"])
def test_one_sided_sweeps_give_exactly_one_agent_a_user(condition):
    paths = sorted(Path("sweeps", condition).glob("*/*.yaml"))
    assert len(paths) == 56
    for path in paths:
        for user_slot in (1, 2):
            cfg = resolved_config(path, f"user_on_agent_{user_slot}")
            assert cfg["orchestrator_settings"]["generate_initial_user_guidance"] is True
            assert cfg[f"agent_{user_slot}"].get("no_user", False) is False
            assert cfg[f"agent_{3 - user_slot}"]["no_user"] is True
            assert cfg[f"user_{3 - user_slot}"]["type"] == "none"
            validate_user_agent_pairing(cfg)
