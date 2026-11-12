from pathlib import Path
from types import SimpleNamespace

import pytest

import negsim.principal_bench_stripped_game  # noqa: F401 - registers the game
from negsim.action import AgentAction, AgentActionType
from negsim.dialogue import Dialogue
from negsim.game import get_game_by_name
from negsim.principal_bench_settings import (
    DEFAULT_STRIPPED_DATA_ROOT,
    load_principal_bench_stripped_setting,
    resolve_principal_bench_stripped_context_path,
)

ALL_STRIPPED_IDS = sorted(
    path.stem for path in Path(DEFAULT_STRIPPED_DATA_ROOT).glob("*/*.json")
)


def _user(user_id: str, role: str) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, private_info={"role": role})


def _agent(agent_id: str, user: SimpleNamespace, *, no_user: bool) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=agent_id,
        user=user,
        no_user=no_user,
        dialogue_with_user=Dialogue(agent_id, user.user_id),
        current_reward=100.0,
        payout_weight=1.0,
        auxiliary_objectives=[],
    )


def _game(setting_id: str = "pb-sanity-negot-01", *, no_user: bool = True):
    party_1 = _agent("agent_1", _user("user_1", "party_1"), no_user=no_user)
    party_2 = _agent("agent_2", _user("user_2", "party_2"), no_user=no_user)
    game_class = get_game_by_name("principal_bench_stripped")
    return game_class(
        name="principal_bench_stripped",
        agent_1=party_1,
        agent_2=party_2,
        game_params={"setting_id": setting_id},
    )


def test_game_is_registered() -> None:
    assert (
        get_game_by_name("principal_bench_stripped").__name__
        == "PrincipalBenchStrippedNegotiationGame"
    )




@pytest.mark.parametrize("setting_id", ALL_STRIPPED_IDS)
def test_every_stripped_setting_constructs_and_exposes_native_prompt_contract(
    setting_id: str,
) -> None:
    game = _game(setting_id)
    example = game._example_proposal()
    assert game.setting_id == setting_id
    assert game.validate_deal(example) == example
    assert game.get_game_blurb_for_agent_prompt()
    assert game.get_game_blurb_for_user_prompt()
    assert "PROPOSAL_DETAILS" in game.get_proposal_format_for_prompt()
    assert set(example) == set(game.setting["adapter"]["proposal"]["fields"])
    # No opening, ever, for any stripped setting.
    assert game.game_dialogue.conversation_history == []
    game.prepare_negotiation()
    assert game.game_dialogue.conversation_history == []
    # Private text reaches the user prompt but not the live (with-user) agent prompt.
    with_user_game = _game(setting_id, no_user=False)
    party_1_private = with_user_game.setting["user_1_private_info"]
    assert party_1_private in with_user_game.get_private_info_for_user_prompt("user_1")
    assert party_1_private not in with_user_game.get_private_info_for_prompt("user_1")


def test_stripped_context_has_only_id_and_two_private_info_fields() -> None:
    path = resolve_principal_bench_stripped_context_path("pb-sanity-negot-01")
    setting = load_principal_bench_stripped_setting(setting_id="pb-sanity-negot-01")
    assert path.name == "pb-sanity-negot-01.json"
    assert setting["id"] == "pb-sanity-negot-01"
    assert "$800" in setting["user_1_private_info"]
    assert "$720" in setting["user_2_private_info"]
    assert "counterparty_strategy" not in setting
    assert "counterparty_opening" not in setting


def test_no_opening_is_ever_inserted() -> None:
    game = _game()
    assert game.game_dialogue.conversation_history == []
    game.prepare_negotiation()
    assert game.game_dialogue.conversation_history == []


def test_turn_order_is_not_forced_to_counterparty_first() -> None:
    # Coin toss is random.random() < 0.5 in Game.__init__; run enough trials that
    # both orderings would appear if turn order really is left to chance.
    first_speakers = set()
    for _ in range(40):
        game = _game()
        first_speakers.add(game.game_dialogue.current_speaker)
    assert first_speakers == {"agent_1", "agent_2"}


def test_private_info_is_stripped_text_only() -> None:
    game = _game(no_user=False)
    party_1_context = game.get_private_info_for_user_prompt("user_1")
    party_2_context = game.get_private_info_for_user_prompt("user_2")
    assert "$650" in party_1_context
    assert "don't explicitly name" not in party_1_context
    assert "Be straightforward and cooperative" not in party_2_context

    # Live agent (has a user) never sees the private numbers directly.
    live_agent_context = game.get_private_info_for_prompt("user_1")
    assert "$650" not in live_agent_context

    autonomous_game = _game(no_user=True)
    autonomous_context = autonomous_game.get_private_info_for_prompt("user_1")
    assert "$650" in autonomous_context


def test_payout_uses_the_proposal_sidecar() -> None:
    game = _game("pb-sanity-negot-01")
    proposal = AgentAction(AgentActionType.PROPOSE, action_details={"price_usd": 720})
    game.record_deal(proposal, proposal)
    game.end_reason = "deal"
    party_1 = game._agent_for_role("party_1").user
    party_2 = game._agent_for_role("party_2").user
    assert game.calculate_payout(party_1) == pytest.approx(70.0)
    assert game.calculate_payout(party_2) == pytest.approx(30.0)


def test_result_metrics_report_stripped_source() -> None:
    game = _game("pb-sanity-negot-01")
    proposal = AgentAction(AgentActionType.PROPOSE, action_details={"price_usd": 720})
    game.record_deal(proposal, proposal)
    game.end_reason = "deal"
    metrics = game.get_result_metrics()
    assert metrics["context_source"] == "PrincipalBench (stripped)"
    assert metrics["stripped_path"].endswith("pb-sanity-negot-01.json")




