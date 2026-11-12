"""Smoke test: one scripted (offline) episode per game and user condition.

Run with ``pytest -m smoke``. Every model call is answered by a fixed script, so no API
key or network access is needed; the episode goes through the real game, orchestrator,
results writer and summary log.
"""
import importlib.util
from pathlib import Path

import pytest

from negsim.run_sweep import _make_task_config, load_sweep

spec = importlib.util.spec_from_file_location("run_scenarios", Path(__file__).parent / "scripted" / "run_scenarios.py")
run_scenarios = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_scenarios)

GAMES = ["buy_sell", "ultimatum", "casino", "deal_or_no_deal", "service_contract", "car_sale", "bike_sale"]
CONDITIONS = {
    "no_user": ("wo_user_wo_instructions", dict(accepting=2, action="ACCEPT", decisions=[True])),
    "single_user": ("user_on_agent_1", dict(accepting=1, action="ACCEPT AND CONFIRM WITH USER",
                                            decisions=[False, True])),
    "dual_user": ("w_user_wo_instructions", None),
    "passive_user": ("user_on_agent_2", dict(accepting=1, action="ACCEPT", decisions=[True])),
    "opening_only_user": ("user_on_agent_1", dict(accepting=2, action="ACCEPT", decisions=[True])),
}


@pytest.mark.smoke
@pytest.mark.parametrize("game", GAMES)
@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_one_episode_per_game_and_condition(game, condition):
    path = Path(f"sweeps/{condition}/{game}/claude_haiku45.yaml")
    combo_id, scenario = CONDITIONS[condition]
    sweep_id, combos, _ = load_sweep(path)
    combo = next(c for c in combos if c.combo_id == combo_id)
    cfg, _ = _make_task_config(sweep_path=path, sweep_id=sweep_id, combo=combo, target_n_runs=1)
    if condition == "dual_user":
        out = run_dual_user(cfg)
    else:
        out = run_scenarios.run_one(cfg, scenario)
    assert out["end"] == "deal"
    assert out["results"]["outcome"] == "deal"
    assert set(out["results"]["payouts"]) == {"user_1", "user_2"}
    assert "### Turn 1" in out["summary"]


def run_dual_user(cfg):
    """Dual User: agent_2 accepts agent_1's proposal; both users are asked and consent."""
    import negsim.agent as agent_mod
    import negsim.user as user_mod
    original = (agent_mod.create_model, user_mod.create_model)
    models = []

    def factory(*args, **kwargs):
        models.append(run_scenarios.ScriptedModel())
        return models[-1]

    agent_mod.create_model = user_mod.create_model = factory
    try:
        from copy import deepcopy
        import tempfile
        from negsim.agent import get_agent
        from negsim.conversation_logger import write_summary_log
        from negsim.game import get_game_by_name
        from negsim.orchestrator import RoundRunner
        from negsim.results_writer import write_results
        from negsim.user import get_user_by_type
        import json

        cfg = deepcopy(cfg)
        cfg["game_params"]["first_speaker"] = "agent_1"
        users = [get_user_by_type(f"user_{s}", cfg[f"user_{s}"]) for s in (1, 2)]
        agents = [get_agent(agent_id=f"agent_{s}", user=users[s - 1], config=cfg[f"agent_{s}"],
                            orchestrator_settings=cfg["orchestrator_settings"]) for s in (1, 2)]
        game = get_game_by_name(cfg["game_name"])(cfg["game_name"], *agents, game_params=cfg["game_params"])
        deal = game.validate_deal(run_scenarios.DEALS[cfg["game_params"].get("setting_id", cfg["game_name"])])
        r = run_scenarios.response
        for agent in agents:
            agent.user.model.public = [r("FURTHER_INSTRUCT", "Opening preferences."), r("CONSENT")]
            agent.model.private = [r("MESSAGE_USER_AND_CONTINUE_NEGOTIATION", "Ready."),
                                   r("REQUEST_USER_CONFIRMATION", "Approve these terms?")]
        agents[0].model.public = [r("PROPOSE", "Proposed terms.", deal)]
        agents[1].model.public = [r("ACCEPT AND CONFIRM WITH USER", "Accepting.")]
        with tempfile.TemporaryDirectory() as tmp:
            end = RoundRunner(game, "synthetic", Path(tmp), cfg["orchestrator_settings"]).run_game()
            write_results(Path(tmp), game, "synthetic", cfg)
            write_summary_log(Path(tmp), game=game)
            results = json.loads((Path(tmp) / "results.json").read_text(encoding="utf-8"))
            summary = (Path(tmp) / "summary.log").read_text(encoding="utf-8")
        assert "user_1: CONSENT" in summary and "user_2: CONSENT" in summary
        return dict(end=str(end), results=results, summary=summary)
    finally:
        agent_mod.create_model, user_mod.create_model = original
