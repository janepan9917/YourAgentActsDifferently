import unittest
from types import SimpleNamespace

import negsim.allocation_games  # noqa: F401 - registers the games
from negsim.action import AgentAction, AgentActionType
from negsim.game import get_game_by_name


def _user(user_id, **private_info):
    return SimpleNamespace(user_id=user_id, private_info=private_info)


def _agent(agent_id, user, *, no_user=False):
    return SimpleNamespace(agent_id=agent_id, user=user, no_user=no_user)


class DealOrNoDealGameTests(unittest.TestCase):
    # Both complete pools are worth 10.
    CONTEXT_1 = "3 1 2 3 1 1"
    CONTEXT_2 = "3 2 2 1 1 2"

    def _game(self, *, no_user=False, **game_params):
        first = _user("user_1", context=self.CONTEXT_1)
        second = _user("user_2", context=self.CONTEXT_2)
        game_cls = get_game_by_name("deal_or_no_deal")
        return game_cls(
            name="deal_or_no_deal",
            agent_1=_agent("agent_1", first, no_user=no_user),
            agent_2=_agent("agent_2", second, no_user=no_user),
            game_params=game_params,
        )

    def test_registration_and_original_context_format(self):
        game = self._game()
        self.assertEqual(game._item_counts, {"books": 3, "hats": 2, "balls": 1})
        self.assertEqual(game._item_values["user_1"], {"books": 1, "hats": 3, "balls": 1})
        self.assertEqual(game._item_values["user_2"], {"books": 2, "hats": 1, "balls": 2})
        for user_id in ("user_1", "user_2"):
            full_pool = sum(
                game._item_counts[item] * game._item_values[user_id][item]
                for item in game.ITEM_TYPES
            )
            self.assertEqual(full_pool, 10)
        self.assertIn(sum(game._item_counts.values()), (5, 6, 7))
        self.assertTrue(all(1 <= count <= 4 for count in game._item_counts.values()))
        self.assertFalse(game.show_accept_score_projection_for_prompt())

    def test_complete_allocation_is_conserved_and_scored_additively(self):
        game = self._game()
        deal = game.validate_deal(
            {
                "user_1": {"books": 2, "hats": 2, "balls": 0},
                "user_2": {"books": 1, "hats": 0, "balls": 1},
            }
        )
        game.agreed_terms = deal
        game.end_reason = "deal"
        self.assertEqual(game.calculate_payout(game.agent_1.user), 8.0)
        self.assertEqual(game.calculate_payout(game.agent_2.user), 4.0)

        with self.assertRaisesRegex(ValueError, "must be allocated"):
            game.validate_deal(
                {
                    "user_1": {"books": 3, "hats": 2, "balls": 0},
                    "user_2": {"books": 1, "hats": 0, "balls": 1},
                }
            )
        with self.assertRaisesRegex(ValueError, "integer"):
            game.validate_deal(
                {
                    "user_1": {"books": 1.5, "hats": 2, "balls": 0},
                    "user_2": {"books": 1.5, "hats": 0, "balls": 1},
                }
            )

    def test_original_item_aliases_and_wrapped_allocations_are_accepted(self):
        game = self._game()
        terms = game.validate_deal(
            {
                "allocations": {
                    "agent_1": {"item0": "2", "item1": 2, "item2": 0},
                    "agent_2": {"book": 1, "hat": 0, "ball": 1},
                }
            }
        )
        self.assertEqual(terms["user_1"], {"books": 2, "hats": 2, "balls": 0})
        self.assertEqual(terms["user_2"], {"books": 1, "hats": 0, "balls": 1})

    def test_no_agreement_and_quit_are_zero_for_both(self):
        game = self._game()
        for end_reason in (None, "quit", "invalid", "reward_zero"):
            game.end_reason = end_reason
            for agent in game.agents:
                self.assertEqual(game.calculate_payout(agent.user), 0.0)

    def test_bad_original_scenarios_are_rejected(self):
        game_cls = get_game_by_name("deal_or_no_deal")
        with self.assertRaisesRegex(ValueError, "complete-pool value must equal 10"):
            game_cls(
                name="deal_or_no_deal",
                agent_1=_agent("agent_1", _user("user_1", context="3 1 2 3 1 2")),
                agent_2=_agent("agent_2", _user("user_2", context=self.CONTEXT_2)),
                game_params={},
            )
        with self.assertRaisesRegex(ValueError, "5, 6, or 7"):
            game_cls(
                name="deal_or_no_deal",
                agent_1=_agent("agent_1", _user("user_1", item_values={"books": 2, "hats": 2, "balls": 1})),
                agent_2=_agent("agent_2", _user("user_2", item_values={"books": 1, "hats": 1, "balls": 3})),
                game_params={"item_counts": {"books": 2, "hats": 2, "balls": 4}},
            )
        with self.assertRaisesRegex(ValueError, "1 to 4"):
            game_cls(
                name="deal_or_no_deal",
                agent_1=_agent(
                    "agent_1",
                    _user("user_1", item_values={"books": 2, "hats": 0, "balls": 0}),
                ),
                agent_2=_agent(
                    "agent_2",
                    _user("user_2", item_values={"books": 1, "hats": 2, "balls": 3}),
                ),
                game_params={"item_counts": {"books": 5, "hats": 1, "balls": 1}},
            )

    def test_seeded_generation_respects_original_constraints(self):
        game_cls = get_game_by_name("deal_or_no_deal")
        game = game_cls(
            name="deal_or_no_deal",
            agent_1=_agent("agent_1", _user("user_1")),
            agent_2=_agent("agent_2", _user("user_2")),
            game_params={"scenario_seed": 19},
        )
        self.assertIn(sum(game._item_counts.values()), (5, 6, 7))
        self.assertTrue(all(1 <= count <= 4 for count in game._item_counts.values()))
        for values in game._item_values.values():
            self.assertEqual(
                sum(game._item_counts[item] * values[item] for item in game.ITEM_TYPES),
                10,
            )
        self.assertTrue(
            all(
                game._item_values["user_1"][item] > 0
                or game._item_values["user_2"][item] > 0
                for item in game.ITEM_TYPES
            )
        )
        self.assertTrue(
            any(
                game._item_values["user_1"][item] > 0
                and game._item_values["user_2"][item] > 0
                for item in game.ITEM_TYPES
            )
        )

    def test_user_effect_private_information_boundary(self):
        with_user = self._game(no_user=False)
        no_user = self._game(no_user=True)
        delegated = with_user.get_private_info_for_prompt("user_1")
        autonomous = no_user.get_private_info_for_prompt("user_1")
        user_context = with_user.get_private_info_for_user_prompt("user_1")

        self.assertNotIn("hats: 3 point", delegated)
        self.assertIn("hats: 3 point", autonomous)
        self.assertIn("hats: 3 point", user_context)
        self.assertTrue(autonomous.startswith("You represent user_1."))
        self.assertTrue(user_context.startswith("You are user_1."))
        self.assertNotIn("your user's", autonomous.lower())
        self.assertNotIn("your user's", user_context.lower())
        self.assertIn("privately guiding your own agent", user_context)
        self.assertNotIn(
            "Your user",
            with_user.get_proposal_format_description("user_1"),
        )

    def test_same_deal_compares_canonical_allocations(self):
        game = self._game()
        first = AgentAction(
            AgentActionType.PROPOSE,
            action_details={
                "user_1": {"books": 2, "hats": 2, "balls": 0},
                "user_2": {"books": 1, "hats": 0, "balls": 1},
            },
        )
        second = AgentAction(
            AgentActionType.PROPOSE,
            action_details={
                "agent_1": {"book": 2, "hat": 2, "ball": 0},
                "agent_2": {"item0": 1, "item1": 0, "item2": 1},
            },
        )
        self.assertTrue(game.check_for_same_proposed_deal(first, second))

    def test_allocation_keys_follow_actual_bindings_when_agent_order_is_reversed(self):
        game_cls = get_game_by_name("deal_or_no_deal")
        game = game_cls(
            name="deal_or_no_deal",
            agent_1=_agent("agent_2", _user("user_2", context=self.CONTEXT_2)),
            agent_2=_agent("agent_1", _user("user_1", context=self.CONTEXT_1)),
            game_params={},
        )
        terms = game.validate_deal(
            {
                "user_1": {"books": 2, "hats": 2, "balls": 0},
                "user_2": {"books": 1, "hats": 0, "balls": 1},
            }
        )
        self.assertEqual(terms["user_1"]["books"], 2)
        self.assertEqual(terms["user_2"]["books"], 1)


class CaSiNoGameTests(unittest.TestCase):
    def _game(self, *, no_user=False, **game_params):
        # Dialogue 0 from the released casino.json participant fields.
        first = _user(
            "user_1",
            value2issue={"Medium": "Food", "Low": "Water", "High": "Firewood"},
            value2reason={
                "High": "We have a large group and need warmth.",
                "Medium": "Food supports our group.",
                "Low": "We already have enough water.",
            },
        )
        second = _user(
            "user_2",
            value2issue={"Low": "Food", "Medium": "Water", "High": "Firewood"},
            value2reason={
                "High": "My dog needs a warm fire.",
                "Medium": "I need to stay hydrated.",
                "Low": "I need little extra food.",
            },
        )
        game_cls = get_game_by_name("casino")
        return game_cls(
            name="casino",
            agent_1=_agent("agent_1", first, no_user=no_user),
            agent_2=_agent("agent_2", second, no_user=no_user),
            game_params=game_params,
        )

    def test_original_stock_and_priority_points(self):
        game = self._game()
        self.assertEqual(game._item_counts, {"food": 3, "water": 3, "firewood": 3})
        self.assertEqual(game.PRIORITY_POINTS, {"high": 5, "medium": 4, "low": 3})
        self.assertEqual(game._item_values["user_1"], {"food": 4, "water": 3, "firewood": 5})
        self.assertEqual(game._item_values["user_2"], {"food": 3, "water": 4, "firewood": 5})
        self.assertFalse(game.show_accept_score_projection_for_prompt())
        for values in game._item_values.values():
            self.assertEqual(set(values.values()), {3, 4, 5})
            self.assertEqual(sum(3 * value for value in values.values()), 36)

    def test_released_dataset_deal_reproduces_points(self):
        game = self._game()
        deal = game.validate_deal(
            {
                "user_1": {"Firewood": "3", "Water": "0", "Food": "1"},
                "user_2": {"Firewood": "0", "Water": "3", "Food": "2"},
            }
        )
        game.agreed_terms = deal
        game.end_reason = "deal"
        self.assertEqual(game.calculate_payout(game.agent_1.user), 19.0)
        self.assertEqual(game.calculate_payout(game.agent_2.user), 18.0)
        metrics = game.get_result_metrics()
        self.assertEqual(metrics["joint_payout"], 37.0)
        self.assertTrue(metrics["envy_free"])

    def test_walk_away_or_no_agreement_gives_both_five(self):
        game = self._game()
        for end_reason in (None, "quit", "invalid", "reward_zero"):
            game.end_reason = end_reason
            for agent in game.agents:
                self.assertEqual(game.calculate_payout(agent.user), 5.0)

    def test_all_nine_packages_must_be_allocated(self):
        game = self._game()
        with self.assertRaisesRegex(ValueError, "All food packages must be allocated"):
            game.validate_deal(
                {
                    "user_1": {"food": 1, "water": 0, "firewood": 3},
                    "user_2": {"food": 1, "water": 3, "firewood": 0},
                }
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            game.validate_deal(
                {
                    "user_1": {"food": -1, "water": 0, "firewood": 3},
                    "user_2": {"food": 4, "water": 3, "firewood": 0},
                }
            )

    def test_original_stock_and_priority_permutation_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "exactly three"):
            self._game(item_counts={"food": 2, "water": 3, "firewood": 3})

        game_cls = get_game_by_name("casino")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            game_cls(
                name="casino",
                agent_1=_agent(
                    "agent_1",
                    _user(
                        "user_1",
                        priorities={"food": "high", "water": "high", "firewood": "low"},
                    ),
                ),
                agent_2=_agent(
                    "agent_2",
                    _user(
                        "user_2",
                        priorities={"food": "high", "water": "medium", "firewood": "low"},
                    ),
                ),
                game_params={},
            )

    def test_user_effect_private_information_boundary_includes_backstory(self):
        with_user = self._game(no_user=False)
        no_user = self._game(no_user=True)
        live_agent_context = with_user.get_private_info_for_prompt("user_1")
        autonomous_context = no_user.get_private_info_for_prompt("user_1")
        user_context = with_user.get_private_info_for_user_prompt("user_1")
        self.assertNotIn("large group and need warmth", live_agent_context)
        self.assertNotIn("High: firewood", live_agent_context)
        self.assertIn("High: firewood", autonomous_context)
        self.assertIn("large group and need warmth", autonomous_context)
        self.assertIn("large group and need warmth", user_context)
        self.assertTrue(autonomous_context.startswith("You represent camper user_1."))
        self.assertTrue(user_context.startswith("You are camper user_1."))
        self.assertNotIn("your user's", autonomous_context.lower())
        self.assertNotIn("your user's", user_context.lower())
        self.assertIn("privately guiding your own agent", user_context)
        self.assertIn("item-to-priority mapping", autonomous_context)

    def test_seeded_missing_profiles_are_valid_and_reproducible(self):
        game_cls = get_game_by_name("casino")

        def make_game():
            return game_cls(
                name="casino",
                agent_1=_agent("agent_1", _user("user_1")),
                agent_2=_agent("agent_2", _user("user_2")),
                game_params={"scenario_seed": 23},
            )

        first = make_game()
        second = make_game()
        self.assertEqual(first._priorities, second._priorities)
        for priorities in first._priorities.values():
            self.assertEqual(set(priorities.values()), {"high", "medium", "low"})


if __name__ == "__main__":
    unittest.main()
