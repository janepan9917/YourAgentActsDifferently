import unittest

from negsim.action import AgentAction, AgentActionType, UserAction, UserActionType
from negsim.dialogue import Dialogue
from negsim.orchestrator import RoundRunner
from negsim.user import LLMUser, _load_prompts


class InitialUserGuidanceTests(unittest.TestCase):
    def test_principal_bench_user_prompt_includes_public_info_in_initial_guidance(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []

            def query_model(self, prompt):
                self.prompts.append(prompt)
                return (
                    "THINKING: I should state my priorities before negotiation.\n"
                    "ACTION: FURTHER_INSTRUCT\n"
                    "OUTPUT: Prioritize the house while preserving a fair overall split."
                )

        class FakeGame:
            state = {"turn": 0, "turns_total": 0}
            max_negotiation_turns = 60

            def get_game_blurb_for_user_prompt(self):
                return "Split an inheritance."

            def get_public_info_for_user_prompt(self, user_id):
                return "Three assets are available."

            def get_private_info_for_user_prompt(self, user_id):
                return "The house is most valuable to you."

            def get_game_state_for_prompt(self):
                return "Negotiation has not started."

        user = LLMUser.__new__(LLMUser)
        user.config = {"type": "llm"}
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.model = FakeModel()
        user.prompts = _load_prompts(
            "prompts/thinking/user_principal_bench_natural_thinking.yaml"
        )
        user.require_thinking = True

        action = user.respond_to_agent(
            Dialogue("agent_1", "user_1"),
            "agent_1",
            Dialogue("agent_1", "agent_2"),
            FakeGame(),
            consultation_mode="initial_guidance",
        )

        self.assertEqual(action.action_type, UserActionType.FURTHER_INSTRUCT)
        prompt = user.model.prompts[0]
        self.assertIn("Negotiation has not started yet", prompt["current_prompt"])
        self.assertIn("FURTHER_INSTRUCT", prompt["current_prompt"])
        self.assertIn("[PUBLIC INFORMATION]:", prompt["system_prompt"])
        self.assertIn("Three assets are available.", prompt["system_prompt"])

    def test_generated_guidance_requires_agent_continue_before_negotiation(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.turns = []

            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action.action_type, to))

        class FakeUser:
            def __init__(self, user_id: str) -> None:
                self.user_id = user_id
                self.initial_offer = ""
                self.config = {"type": "llm"}
                self.calls = []

            def respond_to_agent(
                self,
                dialogue_with_agent,
                agent_id,
                game_dialogue,
                game,
                agent_reward_status=None,
                consultation_mode=None,
                attempt_budget=None,
            ):
                self.calls.append(consultation_mode)
                return UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output=f"Guidance for {agent_id} in {consultation_mode} mode.",
                )

        class FakeAgent:
            def __init__(self, agent_id: str, user_id: str) -> None:
                self.agent_id = agent_id
                self.user = FakeUser(user_id)
                self.dialogue_with_user = Dialogue(agent_id, user_id)
                self.no_user = False
                self.current_reward = 100.0
                self.auxiliary_objectives = []
                self.payout_weight = 1.0
                self.calls = []
                self.actions = [
                    AgentAction(
                        AgentActionType.MESSAGE_USER,
                        output="Can you clarify your preferred tradeoff?",
                    ),
                    AgentAction(
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                        output="Thanks, I have enough guidance and will continue negotiating.",
                    ),
                ]

            def respond_to_user(
                self,
                game,
                game_dialogue=None,
                agreed_deal=None,
                consultation_mode=None,
                attempt_budget=None,
            ):
                self.calls.append(consultation_mode)
                return self.actions.pop(0)

        class FakeGame:
            def __init__(self) -> None:
                self.agent_1 = FakeAgent("agent_1", "user_1")
                self.agent_2 = FakeAgent("agent_2", "user_2")
                self.game_dialogue = Dialogue("agent_1", "agent_2")
                self.state = {}

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        runner.config = {"generate_initial_user_guidance": True}
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 5
        runner._last_invalid_error_agent_user = None
        runner._initial_messages_injected = set()

        runner._run_generated_initial_guidance_consultations()

        for agent in (runner.game.agent_1, runner.game.agent_2):
            history = agent.dialogue_with_user.conversation_history
            self.assertEqual(agent.user.calls, ["initial_guidance", "guidance"])
            self.assertEqual(agent.calls, ["guidance", "guidance"])
            self.assertEqual(
                [turn["action"].action_type for turn in history],
                [
                    UserActionType.FURTHER_INSTRUCT,
                    AgentActionType.MESSAGE_USER,
                    UserActionType.FURTHER_INSTRUCT,
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                ],
            )
            self.assertTrue(
                all(not turn["counts_toward_ask_user_limit"] for turn in history)
            )
        continue_logs = [
            (speaker, to)
            for speaker, action_type, to in runner.logger.turns
            if action_type
            == AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
        ]
        self.assertEqual(
            continue_logs,
            [
                ("agent_1", "agent_1"),
                ("agent_1", "user_1"),
                ("agent_2", "agent_2"),
                ("agent_2", "user_2"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
