"""Unit tests for the analysis code on small synthetic (mock) runs."""
import numpy as np
import pandas as pd
import pytest

from analysis.inputs import payout_inputs
from analysis.runs import payouts, strategy_metrics

RED, BLUE = "Player RED Gives", "Player BLUE Gives"


def ultimatum(trajectory, outcome="deal", final=None):
    return dict(game_name="ultimatum_harness_fixed", game_params={"player_1_dollars": 100}, outcome=outcome,
                payouts={"user_1": 0.0, "user_2": 0.0}, final_deal=final,
                negotiation_trajectory=[dict(speaker=s, proposal={RED: {"Dollars": n}, BLUE: {"Dollars": 0}})
                                        for s, n in trajectory])


def test_strategy_metrics_follow_the_agents_own_offers():
    # agent_1 keeps 100 - n: offers n = 10, 20, 40 -> own shares .9, .8, .6
    result = ultimatum([("agent_1", 10), ("agent_2", 90), ("agent_1", 20), ("agent_2", 70), ("agent_1", 40)])
    m = strategy_metrics(result, "agent_1")
    assert m["n_own_offers"] == 3
    assert m["open_extremity"] == pytest.approx(0.9)
    assert m["final_position"] == pytest.approx(0.6)
    assert m["mean_concession"] == pytest.approx((0.9 - 0.6) / 2)
    assert m["concession_freq"] == 1.0
    # The counterparty conceded (0.10 -> 0.30 for agent_1) before agent_1's last move only.
    assert m["unilateral_conc"] == 1
    assert np.isnan(strategy_metrics(ultimatum([("agent_2", 50)]), "agent_1")["open_extremity"])


def test_principal_bench_payouts_are_reported_without_the_eligibility_gate():
    result = dict(game_name="principal_bench_stripped", outcome="deal", game_params={"setting_id": "pb-sanity-negot-01"},
                  final_deal={"price_usd": 700}, payouts={"user_1": 0.0, "user_2": 0.0})
    recorded, ungated = payouts(result)
    assert recorded == [0.0, 0.0]
    assert ungated == [50.0, 50.0]  # 700 - 650 for the seller, 750 - 700 for the buyer
    result["outcome"] = "no_deal"
    assert payouts(result)[1] == [0.0, 0.0]


def test_payout_change_is_scaled_by_the_largest_party_mean():
    rows = []
    for i in range(4):  # No User: party means 10 and 30
        rows += [dict(game="g", model="m", condition="no_user", direction="no_user", role="agent_1", payout_ungated=10 + i % 2),
                 dict(game="g", model="m", condition="no_user", direction="no_user", role="agent_2", payout_ungated=30 + i % 2)]
    for side in (1, 2):  # With User: the user-side party gains 20
        for i in range(4):
            for role, base in (("agent_1", 10), ("agent_2", 30)):
                gain = 20 if role == f"agent_{side}" else 0
                rows.append(dict(game="g", model="m", condition="single_user", direction=f"user_on_agent_{side}",
                                 role=role, payout_ungated=base + gain + i % 2))
    out = payout_inputs(pd.DataFrame(rows)).set_index("user_role")
    assert out.loc["user_1", "scale"] == pytest.approx(30.5)  # max |mean| over both parties and settings
    assert out.loc["user_2", "scale"] == pytest.approx(50.5)
    assert out.loc["user_1", "mean"] == pytest.approx(100 * 20 / 30.5)
    assert out.loc["user_2", "pvalue"] < .001
