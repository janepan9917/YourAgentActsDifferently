from types import SimpleNamespace

import pytest

import negsim  # noqa: F401 - registers the game implementations
from negsim.game import get_game_by_name
from negsim.action import AgentAction, AgentActionType
from negsim.dialogue import format_action_for_history
from negsim.na_harness_fixed_games import (
    BLUE_GIVES,
    NO_DEAL_RULE,
    RED_GIVES,
    UltimatumGameHarnessFixed,
)


def _user(user_id: str, **private_info):
    return SimpleNamespace(user_id=user_id, private_info=private_info)


def _agent(agent_id: str, user, *, no_user: bool = False):
    return SimpleNamespace(agent_id=agent_id, user=user, no_user=no_user)


def _make_game(
    name: str,
    *,
    user_1_info: dict,
    user_2_info: dict,
    game_params: dict,
    no_user: bool = False,
):
    game_cls = get_game_by_name(name)
    game_params = dict(game_params)
    if name.endswith("_harness_fixed"):
        game_params.setdefault(
            "proposal_schema_version",
            "explicit_gives_json_v1",
        )
    return game_cls(
        name=name,
        agent_1=_agent(
            "agent_1", _user("user_1", **user_1_info), no_user=no_user
        ),
        agent_2=_agent(
            "agent_2", _user("user_2", **user_2_info), no_user=no_user
        ),
        game_params=game_params,
    )


PRIVATE_VISIBILITY_CASES = (
    pytest.param(
        "buy_sell_harness_fixed",
        {"cost": 40123, "quantity": 1},
        {"willingness_to_pay": 90234, "budget": 987654},
        {"commodity": "Widget"},
        "costs 40123 ZUP",
        "budget of 987654 ZUP",
        id="buy-sell",
    ),
    pytest.param(
        "ultimatum_harness_fixed",
        {},
        {},
        {"player_1_dollars": 100},
        "lower N is better",
        "higher N is better",
        id="ultimatum",
    ),
)


@pytest.mark.parametrize(
    "name,user_1_info,user_2_info,game_params,first_marker,second_marker",
    PRIVATE_VISIBILITY_CASES,
)
def test_private_information_visibility_and_public_no_deal_rule(
    name,
    user_1_info,
    user_2_info,
    game_params,
    first_marker,
    second_marker,
):
    delegated = _make_game(
        name,
        user_1_info=user_1_info,
        user_2_info=user_2_info,
        game_params=game_params,
    )
    autonomous = _make_game(
        name,
        user_1_info=user_1_info,
        user_2_info=user_2_info,
        game_params=game_params,
        no_user=True,
    )

    for blurb in (
        delegated.get_game_blurb_for_agent_prompt(),
        delegated.get_game_blurb_for_user_prompt(),
    ):
        assert blurb.count(NO_DEAL_RULE) == 1

    for user_id, own_marker, other_marker in (
        ("user_1", first_marker, second_marker),
        ("user_2", second_marker, first_marker),
    ):
        delegated_agent_context = delegated.get_private_info_for_prompt(user_id)
        user_context = delegated.get_private_info_for_user_prompt(user_id)
        autonomous_agent_context = autonomous.get_private_info_for_prompt(user_id)

        assert delegated_agent_context.startswith("You represent Player ")
        assert "visible only to them" in delegated_agent_context
        assert own_marker not in delegated_agent_context
        assert other_marker not in delegated_agent_context

        assert user_context.startswith("You are Player ")
        assert own_marker in user_context
        assert other_marker not in user_context
        assert autonomous_agent_context.startswith("You are Player ")
        assert own_marker in autonomous_agent_context
        assert other_marker not in autonomous_agent_context

        for private_context in (
            delegated_agent_context,
            user_context,
            autonomous_agent_context,
        ):
            assert NO_DEAL_RULE not in private_context
            assert "no-deal payout" not in private_context.lower()


@pytest.mark.parametrize(
    "name,user_1_info,user_2_info,game_params,first_marker,second_marker",
    PRIVATE_VISIBILITY_CASES,
)
def test_no_deal_payout_is_fixed_at_zero_and_ignores_configured_batna(
    name,
    user_1_info,
    user_2_info,
    game_params,
    first_marker,
    second_marker,
):
    del first_marker, second_marker
    fixed = _make_game(
        name,
        user_1_info={**user_1_info, "batna": 123},
        user_2_info={**user_2_info, "batna": 456},
        game_params=game_params,
    )
    fixed.end_reason = "limit_exceeded"

    assert fixed.calculate_payout(fixed.agent_1.user) == 0.0
    assert fixed.calculate_payout(fixed.agent_2.user) == 0.0




def test_buy_sell_keeps_starting_resources_role_local():
    fixed = _make_game(
        "buy_sell_harness_fixed",
        user_1_info={"cost": 40, "quantity": 1},
        user_2_info={"willingness_to_pay": 90, "budget": 987654},
        game_params={"commodity": "Widget"},
        no_user=True,
    )
    shared = fixed.get_game_blurb_for_agent_prompt()
    seller_private = fixed.get_private_info_for_prompt("user_1")
    buyer_private = fixed.get_private_info_for_prompt("user_2")

    assert "987654" not in shared
    assert "987654" not in seller_private
    assert "987654" in buyer_private
    assert "own 1 unit(s)" in seller_private
    assert "987654" not in fixed.get_game_blurb_for_user_prompt()
    assert "987654" not in fixed.get_public_info_for_user_prompt("user_1")
    assert "987654" not in fixed.get_private_info_for_user_prompt("user_1")
    assert "987654" in fixed.get_private_info_for_user_prompt("user_2")

    valid = {
        RED_GIVES: {"Widget": 1},
        BLUE_GIVES: {"ZUP": 987655},
    }
    assert fixed.validate_deal(valid) == valid
    assert '"Player RED Gives"' in fixed.get_proposal_format_for_prompt()
    assert '"Player BLUE Gives"' in fixed.get_proposal_format_for_prompt()




def test_ultimatum_validates_transfer_schema():
    fixed = _make_game(
        "ultimatum_harness_fixed",
        user_1_info={},
        user_2_info={},
        game_params={
            "player_1_dollars": 100,
            "first_speaker": "agent_2",
        },
        no_user=True,
    )
    assert "<player answer>" not in fixed.get_game_blurb_for_agent_prompt().lower()
    assert "Proposal JSON records contributions" in fixed.get_game_blurb_for_agent_prompt()
    assert "counter-propose" in fixed.get_private_info_for_prompt("user_2")
    assert "deal payout is N" in fixed.get_private_info_for_prompt("user_2")

    valid = {
        RED_GIVES: {"Dollars": 40},
        BLUE_GIVES: {"Dollars": 0},
    }
    assert fixed.validate_deal(valid) == valid
    parsed = fixed.parse_proposal_details(
        '{"Player RED Gives": {"Dollars": 40}, '
        '"Player BLUE Gives": {"Dollars": 0}}'
    )
    assert fixed.validate_deal(parsed) == valid

    counterparty_offer = {
        RED_GIVES: {"Dollars": 55},
        BLUE_GIVES: {"Dollars": 0},
    }
    own_later_offer = {
        RED_GIVES: {"Dollars": 60},
        BLUE_GIVES: {"Dollars": 0},
    }
    fixed.game_dialogue.add_turn(
        "agent_2",
        0,
        AgentAction(AgentActionType.PROPOSE, action_details=counterparty_offer),
    )
    fixed.game_dialogue.add_turn(
        "agent_1",
        0,
        AgentAction(AgentActionType.PROPOSE, action_details=own_later_offer),
    )
    fixed.game_dialogue.add_turn(
        "agent_2",
        0,
        AgentAction(AgentActionType.ACCEPT_AND_CONFIRM),
    )
    assert fixed.game_dialogue.current_speaker == "agent_1"
    game_state = fixed.get_game_state_for_prompt()
    assert "Player BLUE receives 55 Dollars" in game_state
    assert "Player RED retains 45 Dollars" in game_state
    assert "Player BLUE receives 60 Dollars" not in game_state

    with pytest.raises(ValueError, match="must be 0"):
        fixed.validate_deal(
            {
                RED_GIVES: {"Dollars": 40},
                BLUE_GIVES: {"Dollars": 1},
            }
        )
    with pytest.raises(ValueError, match="exactly"):
        fixed.validate_deal({RED_GIVES: {"Dollars": 40}})
    with pytest.raises(ValueError, match="extra"):
        fixed.validate_deal(
            {
                RED_GIVES: {"Dollars": 40, "Tokens": 0},
                BLUE_GIVES: {"Dollars": 0},
            }
        )


def test_harness_games_require_unambiguous_json_and_preserve_gives_in_history():
    fixed = _make_game(
        "ultimatum_harness_fixed",
        user_1_info={},
        user_2_info={},
        game_params={"player_1_dollars": 100},
    )
    with pytest.raises(ValueError, match="Invalid JSON"):
        fixed.parse_proposal_details(
            "Player RED Gives Dollars: 40 | Player BLUE Gives Dollars: 0"
        )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        fixed.parse_proposal_details(
            '{"Player RED Gives": {"Dollars": 40}, '
            '"Player RED Gives": {"Dollars": 41}, '
            '"Player BLUE Gives": {"Dollars": 0}}'
        )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        fixed.parse_proposal_details(
            '{"Player RED Gives": {"Dollars": 40, "Dollars": 41}, '
            '"Player BLUE Gives": {"Dollars": 0}}'
        )

    details = {
        RED_GIVES: {"Dollars": 40},
        BLUE_GIVES: {"Dollars": 0},
    }
    rendered = format_action_for_history(
        AgentAction(AgentActionType.PROPOSE, action_details=details),
        speaker="agent_1",
    )
    assert '"Player RED Gives"' in rendered
    assert '"Player BLUE Gives"' in rendered

    with pytest.raises(ValueError, match="proposal_schema_version"):
        UltimatumGameHarnessFixed(
            "ultimatum_harness_fixed",
            _agent("agent_1", _user("user_1")),
            _agent("agent_2", _user("user_2")),
            {"player_1_dollars": 100},
        )
    with pytest.raises(ValueError, match="not a game participant"):
        fixed.calculate_payout(_user("unknown"))


