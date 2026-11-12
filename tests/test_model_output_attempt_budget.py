import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from negsim.action import (
    AgentAction,
    AgentActionType,
    UserAction,
    UserActionType,
)
from negsim.agent import Agent
from negsim.dialogue import Dialogue
from negsim.model_output_errors import (
    ModelOutputAttemptBudget,
    ModelOutputAttemptsExceeded,
    RetryMode,
)
from negsim.model import AnthropicModel
from negsim.orchestrator import RoundRunner
from negsim.results_writer import write_results
from negsim.user import DummyYesUser, LLMUser


class _ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def query_model(self, prompt, response_format=None):
        self.prompts.append(dict(prompt))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _AgentGame:
    state = {"turn": 2}
    max_negotiation_turns = 10

    def __init__(self, agent):
        self.agent_1 = agent
        self.agent_2 = SimpleNamespace(agent_id="agent_2")
        self.extractor_error = None

    @staticmethod
    def get_proposal_format_for_prompt():
        return 'PROPOSAL_DETAILS: {"terms":"..."}'

    @staticmethod
    def get_response_format_instruction(details_fmt, actions_str):
        return None

    @staticmethod
    def get_user_consultation_response_format_instruction(details_fmt, actions_str):
        return None

    @staticmethod
    def get_private_info_for_prompt(user_id):
        return "Private context."

    @staticmethod
    def get_game_blurb_for_agent_prompt():
        return "Split assets."

    @staticmethod
    def get_game_state_for_prompt():
        return "Current state."

    @staticmethod
    def get_proposal_format_description(user_id):
        return "Use terms."

    @staticmethod
    def parse_proposal_details(details):
        return json.loads(details) if isinstance(details, str) else details

    def validate_deal(self, details):
        if details == {"terms": "invalid"}:
            raise ValueError("Invalid proposal terms.")
        return details

    @staticmethod
    def parse_raw_agent_response(raw, **kwargs):
        return None

    @staticmethod
    def show_accept_score_projection_for_prompt():
        return False

    def extract_declared_action_for_repair(self, raw, **kwargs):
        if self.extractor_error is not None:
            raise self.extractor_error
        return None


def _make_agent(responses):
    agent = Agent.__new__(Agent)
    agent.agent_id = "agent_1"
    agent.user = DummyYesUser({}, "user_1")
    agent.orchestrator_settings = {"max_negotiation_turns": 10}
    agent.visible_dialogue = Dialogue("agent_1", "user_1")
    agent.dialogue_with_user = agent.visible_dialogue.channel_view(
        "agent_user",
        "agent_1",
        "user_1",
    )
    agent.config = {}
    agent.prompt_logger = None
    agent.show_own_thinking = False
    agent.require_thinking = True
    agent.use_structured_output = False
    agent.no_user = True
    agent.model = _ScriptedModel(responses)
    agent.prompts = {
        "respond_to_user": {
            "system": (
                "{response_format_instruction}\n{private_info}\n{game_blurb}\n"
                "{game_state}\n{proposal_format}\n{reward_status}\n"
                "{consultation_mode}\n{agent_agent_history}\n{game_summary}"
            ),
            "current_prompt": "Choose {actions}.",
            "actions": [
                "MESSAGE_USER",
                "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            ],
        },
        "respond_to_agent": {
            "system": (
                "{response_format_instruction}\n{private_info}\n{game_blurb}\n"
                "{game_state}\n{proposal_format}\n{reward_status}\n{turn_warning}"
            ),
            "current_prompt_parts": {
                "first_turn": "Open.",
                "continuation": "Choose an action.",
                "action_question": "Choose {actions}.",
            },
            "actions": ["PROPOSE", "ACCEPT"],
        }
    }
    agent.auxiliary_objectives = []
    agent.current_reward = 100.0
    agent.payout_weight = 1.0

    game = _AgentGame(agent)
    dialogue = Dialogue("agent_1", "agent_2")
    dialogue.add_turn(
        "agent_2",
        1,
        AgentAction(
            AgentActionType.PROPOSE,
            output="Prior offer.",
            action_details={"terms": "40-60"},
        ),
    )
    return agent, game, dialogue


class ModelOutputAttemptBudgetTests(unittest.TestCase):
    def test_budget_defaults_to_eight_and_never_executes_a_ninth_attempt(self):
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        self.assertEqual(budget.max_attempts, 8)
        for expected in range(1, 9):
            self.assertEqual(budget.consume(RetryMode.RESAMPLE), expected)
        self.assertEqual(budget.remaining_attempts, 0)

        with self.assertRaises(ModelOutputAttemptsExceeded):
            budget.consume(RetryMode.RESAMPLE)
        self.assertEqual(budget.used_attempts, 8)

        for invalid in (0, -1, 1.5, True, "8"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    ModelOutputAttemptBudget(max_attempts=invalid)

    def test_real_agent_repair_fresh_decision_exhaustion_and_log_context(self):
        cases = [
            (
                "deterministic_normalization",
                [
                    "THINKING: preserve this ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose 50-50."
                ],
                AgentActionType.PROPOSE,
                1,
                [RetryMode.INITIAL.value],
            ),
            (
                "resample_after_missing_action",
                [
                    "THINKING: undecided\nOUTPUT: I have not chosen.",
                    "THINKING: accept\nACTION: ACCEPT\nOUTPUT: I accept.",
                ],
                AgentActionType.ACCEPT,
                2,
                [RetryMode.INITIAL.value, RetryMode.RESAMPLE.value],
            ),
            (
                "resample_after_missing_output",
                [
                    "THINKING: propose\nACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}',
                    "THINKING: repaired\nACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose 50-50.",
                ],
                AgentActionType.PROPOSE,
                2,
                [RetryMode.INITIAL.value, RetryMode.RESAMPLE.value],
            ),
        ]

        for name, responses, expected_action, attempts, expected_modes in cases:
            with self.subTest(name=name):
                agent, game, dialogue = _make_agent(responses)
                budget = ModelOutputAttemptBudget(actor_id="agent_1")
                action = agent.respond_to_agent(
                    dialogue,
                    game,
                    attempt_budget=budget,
                )

                self.assertEqual(action.action_type, expected_action)
                self.assertEqual(budget.used_attempts, attempts)
                contexts = [p["api_log_context"] for p in agent.model.prompts]
                self.assertEqual(
                    [context["model_output_attempt"] for context in contexts],
                    list(range(1, attempts + 1)),
                )
                self.assertEqual(
                    [context["retry_mode"] for context in contexts],
                    expected_modes,
                )
                self.assertEqual(
                    {context["decision_cycle_id"] for context in contexts},
                    {budget.decision_cycle_id},
                )
                self.assertTrue(
                    all(context["max_model_output_attempts"] == 8 for context in contexts)
                )

        malformed = "THINKING: undecided\nOUTPUT: Still no action."
        agent, game, dialogue = _make_agent([malformed] * 8)
        budget = ModelOutputAttemptBudget(actor_id="agent_1")
        with self.assertRaises(ModelOutputAttemptsExceeded) as raised:
            agent.respond_to_agent(dialogue, game, attempt_budget=budget)

        self.assertIs(raised.exception.budget, budget)
        self.assertEqual(len(agent.model.prompts), 8)
        self.assertEqual(budget.used_attempts, 8)
        self.assertEqual(budget.next_retry_mode, RetryMode.RESAMPLE)
        self.assertEqual(budget.last_raw_response, malformed)

    def test_real_agent_accept_without_output_does_not_resample(self):
        agent, game, dialogue = _make_agent(
            ["THINKING: The offer is acceptable.\nACTION: ACCEPT"]
        )
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        action = agent.respond_to_agent(
            dialogue,
            game,
            attempt_budget=budget,
        )

        self.assertEqual(action.action_type, AgentActionType.ACCEPT)
        self.assertEqual(action.output, "")
        self.assertEqual(budget.used_attempts, 1)
        self.assertEqual(len(agent.model.prompts), 1)
        self.assertEqual(
            agent.model.prompts[0]["api_log_context"]["retry_mode"],
            RetryMode.INITIAL.value,
        )

    def test_real_private_agent_has_the_same_eight_attempt_budget(self):
        malformed = "THINKING: undecided\nOUTPUT: Still no action."
        agent, game, _ = _make_agent([malformed] * 8)
        agent.no_user = False
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        with self.assertRaises(ModelOutputAttemptsExceeded):
            agent.respond_to_user(
                game,
                consultation_mode="guidance",
                attempt_budget=budget,
            )

        self.assertEqual(len(agent.model.prompts), 8)
        self.assertEqual(budget.used_attempts, 8)
        self.assertEqual(
            [
                prompt["api_log_context"]["model_output_attempt"]
                for prompt in agent.model.prompts
            ],
            list(range(1, 9)),
        )
        self.assertEqual(
            {prompt["api_log_context"]["decision_cycle_id"] for prompt in agent.model.prompts},
            {budget.decision_cycle_id},
        )

    def test_provider_retries_do_not_consume_additional_output_attempts(self):
        class FakeMessages:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("transient provider timeout")
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            type="text",
                            text=(
                                "THINKING: accept\nACTION: ACCEPT\n"
                                "OUTPUT: I accept."
                            ),
                        )
                    ],
                    usage=None,
                )

        agent, game, dialogue = _make_agent([])
        messages = FakeMessages()
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(messages=messages)
        model.request_timeout_seconds = None
        agent.model = model
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        with patch.dict("os.environ", {"ANTHROPIC_API_MAX_RETRIES": "2"}), patch(
            "negsim.model.random.uniform",
            return_value=0,
        ), patch("negsim.model.time.sleep"), patch(
            "negsim.model.log_api_event"
        ) as log_event:
            action = agent.respond_to_agent(
                dialogue,
                game,
                attempt_budget=budget,
            )

        self.assertEqual(action.action_type, AgentActionType.ACCEPT)
        self.assertEqual(messages.calls, 2)
        self.assertEqual(budget.used_attempts, 1)
        request_starts = [
            call.kwargs
            for call in log_event.call_args_list
            if call.kwargs["event"] == "request_start"
        ]
        self.assertEqual(len(request_starts), 2)
        self.assertTrue(
            all(
                event["context"]["model_output_attempt"] == 1
                for event in request_starts
            )
        )
        self.assertTrue(
            all(
                event["context"]["decision_cycle_id"]
                == budget.decision_cycle_id
                for event in request_starts
            )
        )

    def test_invalid_proposal_resamples_without_semantic_lock(self):
        agent, game, dialogue = _make_agent(
            [
                "THINKING: propose\nACTION: PROPOSE\nOUTPUT: Terms omitted.",
                "ACTION: PROPOSE\n"
                'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                "OUTPUT: I propose 50-50.",
                "THINKING: drift\nACTION: PROPOSE\n"
                'PROPOSAL_DETAILS: {"terms":"60-40"}\n'
                "OUTPUT: I propose 60-40.",
            ]
        )
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        action = agent.respond_to_agent(dialogue, game, attempt_budget=budget)

        self.assertEqual(action.action_details, {"terms": "60-40"})
        self.assertEqual(budget.used_attempts, 3)
        self.assertEqual(
            [p["api_log_context"]["retry_mode"] for p in agent.model.prompts],
            [
                RetryMode.INITIAL.value,
                RetryMode.RESAMPLE.value,
                RetryMode.RESAMPLE.value,
            ],
        )
        for prompt in agent.model.prompts[1:]:
            self.assertEqual(prompt["system_prompt"], agent.model.prompts[0]["system_prompt"])
            self.assertEqual(
                prompt["conversation_history"],
                agent.model.prompts[0]["conversation_history"],
            )
            self.assertEqual(prompt["current_prompt"], agent.model.prompts[0]["current_prompt"])

        drift_agent, drift_game, drift_dialogue = _make_agent(
            [
                "THINKING: propose\nACTION: PROPOSE\nOUTPUT: Terms omitted.",
                "THINKING: switch\nACTION: ACCEPT\nOUTPUT: Accept instead.",
            ]
        )
        drift_budget = ModelOutputAttemptBudget(actor_id="agent_1")

        drift_action = drift_agent.respond_to_agent(
            drift_dialogue,
            drift_game,
            attempt_budget=drift_budget,
        )

        self.assertEqual(drift_action.action_type, AgentActionType.ACCEPT)
        self.assertEqual(drift_budget.used_attempts, 2)

    def test_game_semantic_invalid_resamples_the_frozen_request(self):
        agent, game, dialogue = _make_agent(
            [
                "THINKING: invalid terms\nACTION: PROPOSE\n"
                'PROPOSAL_DETAILS: {"terms":"invalid"}\n'
                "OUTPUT: Invalid proposal.",
                "THINKING: choose another action\n"
                "ACTION: ACCEPT\nOUTPUT: Accept instead.",
            ]
        )
        budget = ModelOutputAttemptBudget(actor_id="agent_1")

        action = agent.respond_to_agent(
            dialogue,
            game,
            attempt_budget=budget,
        )

        self.assertEqual(action.action_type, AgentActionType.ACCEPT)
        self.assertEqual(budget.used_attempts, 2)
        self.assertEqual(
            agent.model.prompts[1]["system_prompt"],
            agent.model.prompts[0]["system_prompt"],
        )
        self.assertEqual(
            agent.model.prompts[1]["conversation_history"],
            agent.model.prompts[0]["conversation_history"],
        )
        self.assertEqual(
            agent.model.prompts[1]["current_prompt"],
            agent.model.prompts[0]["current_prompt"],
        )
        self.assertEqual(
            agent.model.prompts[1]["api_log_context"]["retry_mode"],
            RetryMode.RESAMPLE.value,
        )

    def test_internal_exceptions_are_not_converted_into_output_retries(self):
        exceptions = [
            ValueError("model client configuration bug"),
            KeyError("missing_internal_key"),
            TypeError("internal type bug"),
            AssertionError("internal invariant bug"),
        ]
        for sentinel in exceptions:
            with self.subTest(exception=type(sentinel).__name__):
                agent, game, dialogue = _make_agent([sentinel])
                budget = ModelOutputAttemptBudget(actor_id="agent_1")
                with self.assertRaises(type(sentinel)) as raised:
                    agent.respond_to_agent(dialogue, game, attempt_budget=budget)
                self.assertIs(raised.exception, sentinel)
                self.assertEqual(len(agent.model.prompts), 1)
                self.assertEqual(budget.used_attempts, 1)

        agent, game, dialogue = _make_agent(
            ["THINKING: valid\nACTION: ACCEPT\nOUTPUT: Accept."]
        )
        game.parse_raw_agent_response = lambda *args, **kwargs: ("wrong shape",)
        budget = ModelOutputAttemptBudget(actor_id="agent_1")
        with self.assertRaisesRegex(TypeError, "must return None or a 4-tuple"):
            agent.respond_to_agent(dialogue, game, attempt_budget=budget)
        self.assertEqual(len(agent.model.prompts), 1)
        self.assertEqual(budget.used_attempts, 1)

        sentinel = ValueError("internal raw-response hook bug")
        agent, game, dialogue = _make_agent(
            ["THINKING: valid\nACTION: ACCEPT\nOUTPUT: Accept."]
        )

        def fail_raw_response_hook(*args, **kwargs):
            raise sentinel

        game.parse_raw_agent_response = fail_raw_response_hook
        budget = ModelOutputAttemptBudget(actor_id="agent_1")
        with self.assertRaises(ValueError) as raised:
            agent.respond_to_agent(dialogue, game, attempt_budget=budget)
        self.assertIs(raised.exception, sentinel)
        self.assertEqual(len(agent.model.prompts), 1)
        self.assertEqual(budget.used_attempts, 1)

    def test_invalid_action_configuration_fails_before_querying_a_model(self):
        agent, game, dialogue = _make_agent(
            ["THINKING: valid\nACTION: ACCEPT\nOUTPUT: Accept."]
        )
        agent.prompts["respond_to_agent"]["actions"] = ["BOGUS"]
        budget = ModelOutputAttemptBudget(actor_id="agent_1")
        with self.assertRaisesRegex(ValueError, "Invalid configured agent actions"):
            agent.respond_to_agent(dialogue, game, attempt_budget=budget)
        self.assertEqual(agent.model.prompts, [])
        self.assertEqual(budget.used_attempts, 0)

        private_agent, private_game, _ = _make_agent(
            ["THINKING: valid\nACTION: MESSAGE_USER\nOUTPUT: Update."]
        )
        private_agent.no_user = False
        private_agent.prompts["respond_to_user"]["actions"] = ["PROPOSE"]
        private_budget = ModelOutputAttemptBudget(actor_id="agent_1")
        with self.assertRaisesRegex(ValueError, "No agent actions are configured"):
            private_agent.respond_to_user(
                private_game,
                consultation_mode="guidance",
                attempt_budget=private_budget,
            )
        self.assertEqual(private_agent.model.prompts, [])
        self.assertEqual(private_budget.used_attempts, 0)

        user = LLMUser.__new__(LLMUser)
        user.config = {}
        user.user_id = "user_1"
        user.prompts = {
            "respond_to_agent": {
                "system": "unused",
                "current_prompt": "unused",
                "actions": ["BOGUS"],
            }
        }
        user.model = _ScriptedModel(["unused"])
        user_budget = ModelOutputAttemptBudget(actor_id="user_1")
        with self.assertRaisesRegex(ValueError, "Invalid configured user actions"):
            user.respond_to_agent(
                Dialogue("agent_1", "user_1"),
                "agent_1",
                None,
                SimpleNamespace(),
                attempt_budget=user_budget,
            )
        self.assertEqual(user.model.prompts, [])
        self.assertEqual(user_budget.used_attempts, 0)

    def test_llm_user_has_its_own_eight_attempt_cycle(self):
        class FakeGame:
            state = {"turn": 0}
            max_negotiation_turns = 10

            @staticmethod
            def get_game_blurb_for_user_prompt():
                return "Negotiate."

            @staticmethod
            def get_public_info_for_user_prompt(user_id):
                return "Public."

            @staticmethod
            def get_private_info_for_user_prompt(user_id):
                return "Private."

            @staticmethod
            def get_game_state_for_prompt():
                return "State."

        malformed = "THINKING: undecided\nOUTPUT: No action."
        user = LLMUser.__new__(LLMUser)
        user.config = {}
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.require_thinking = True
        user.model = _ScriptedModel([malformed] * 8)
        user.prompts = {
            "respond_to_agent": {
                "system": (
                    "{public_info}\n{private_info}\n{game_blurb}\n{game_state}\n"
                    "{agent_agent_history}\n{agent_reward_status}\n{actions}"
                ),
                "current_prompt": "Choose {actions}.",
                "actions": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
            }
        }
        budget = ModelOutputAttemptBudget(actor_id="user_1")

        with self.assertRaises(ModelOutputAttemptsExceeded):
            user.respond_to_agent(
                Dialogue("agent_1", "user_1"),
                "agent_1",
                Dialogue("agent_1", "agent_2"),
                FakeGame(),
                attempt_budget=budget,
            )

        self.assertEqual(len(user.model.prompts), 8)
        self.assertEqual(budget.used_attempts, 8)
        self.assertEqual(
            {p["api_log_context"]["decision_cycle_id"] for p in user.model.prompts},
            {budget.decision_cycle_id},
        )
        user_contexts = [p["api_log_context"] for p in user.model.prompts]
        self.assertEqual(
            [context["model_output_attempt"] for context in user_contexts],
            list(range(1, 9)),
        )
        self.assertEqual(
            [context["retry_mode"] for context in user_contexts],
            [RetryMode.INITIAL.value] + [RetryMode.RESAMPLE.value] * 7,
        )
        self.assertTrue(
            all(context["max_model_output_attempts"] == 8 for context in user_contexts)
        )

    def test_deterministic_actor_errors_fail_without_model_output_retries(self):
        budget = ModelOutputAttemptBudget(actor_id="user_1")
        dummy = DummyYesUser({}, "user_1")
        dummy.respond_to_agent(
            Dialogue("agent_1", "user_1"),
            "agent_1",
            None,
            SimpleNamespace(),
            attempt_budget=budget,
        )
        self.assertEqual(budget.used_attempts, 0)

        class FakeLogger:
            @staticmethod
            def subsection(message, to="both"):
                return None

            @staticmethod
            def turn(speaker, action, to="both", invalid_reason=None):
                return None

        class DeterministicAgent:
            agent_id = "agent_1"
            no_user = False
            user = SimpleNamespace(user_id="user_1")

            def __init__(self, action):
                self.calls = 0
                self.action = action

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return self.action

        invalid_actions = [
            AgentAction(
                AgentActionType.REQUEST_USER_CONFIRMATION,
                output="Invalid public confirmation request.",
            ),
            AgentAction(
                AgentActionType.MESSAGE_USER,
                output="Invalid public report.",
            ),
            AgentAction(
                AgentActionType.PROPOSE,
                output="Proposal with no details.",
            ),
        ]
        for invalid_action in invalid_actions:
            with self.subTest(action=invalid_action.action_type):
                agent = DeterministicAgent(invalid_action)
                game = SimpleNamespace(
                    agent_1=agent,
                    agent_2=SimpleNamespace(
                        agent_id="agent_2",
                        user=SimpleNamespace(user_id="user_2"),
                    ),
                    game_dialogue=Dialogue("agent_1", "agent_2"),
                    end_reason=None,
                )
                runner = RoundRunner.__new__(RoundRunner)
                runner.game = game
                runner.logger = FakeLogger()
                runner.current_turn = 0
                runner.max_negotiation_turns = 10
                runner.turns_by_agent = {"agent_1": 0, "agent_2": 0}
                runner._update_turn_state = lambda next_speaker=None: None

                with self.assertRaisesRegex(
                    RuntimeError,
                    "without consuming a model-output attempt",
                ):
                    runner._run_negotiation()
                self.assertEqual(agent.calls, 1)

    def test_runner_semantic_invalid_resamples_without_action_lock(self):
        class FakeLogger:
            @staticmethod
            def subsection(message, to="both"):
                return None

            @staticmethod
            def turn(speaker, action, to="both", invalid_reason=None):
                return None

        class BudgetAwareAgent:
            agent_id = "agent_1"
            no_user = True
            user = SimpleNamespace(user_id="user_1")

            def __init__(self):
                self.observations = []

            def respond_to_agent(self, *args, attempt_budget=None, **kwargs):
                self.observations.append(
                    (
                        attempt_budget.decision_cycle_id,
                        attempt_budget.used_attempts,
                        attempt_budget.next_retry_mode,
                    )
                )
                attempt_budget.consume()
                if len(self.observations) == 1:
                    return AgentAction(
                        AgentActionType.PROPOSE,
                        output="Invalid terms.",
                        action_details={"terms": "invalid"},
                    )
                if len(self.observations) == 2:
                    return AgentAction(AgentActionType.QUIT, output="Try to escape.")
                return AgentAction(
                    AgentActionType.PROPOSE,
                    output="Corrected terms.",
                    action_details={"terms": "corrected"},
                )

        def validate_deal(details):
            if details == {"terms": "invalid"}:
                raise ValueError("Invalid proposal terms.")
            return details

        agent = BudgetAwareAgent()
        game = SimpleNamespace(
            agent_1=agent,
            agent_2=SimpleNamespace(
                agent_id="agent_2",
                user=SimpleNamespace(user_id="user_2"),
            ),
            game_dialogue=Dialogue("agent_1", "agent_2"),
            end_reason=None,
            quitter_user_id=None,
            validate_deal=validate_deal,
            is_invalid_game_state=lambda: False,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = game
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_negotiation_turns = 10
        runner.turns_by_agent = {"agent_1": 0, "agent_2": 0}
        runner._update_turn_state = lambda next_speaker=None: None
        runner._record_public_negotiation_turn = (
            lambda speaker, action, dialogue: setattr(game, "end_reason", "quit")
        )

        runner._run_negotiation()

        self.assertEqual(game.end_reason, "quit")
        self.assertEqual(len(agent.observations), 2)
        self.assertEqual(len({observation[0] for observation in agent.observations}), 1)
        self.assertEqual(
            [observation[1] for observation in agent.observations],
            [0, 1],
        )
        self.assertEqual(
            [observation[2] for observation in agent.observations],
            [
                RetryMode.INITIAL,
                RetryMode.RESAMPLE,
            ],
        )

    def test_private_channel_allowlists_fail_fast_for_deterministic_actors(self):
        class FakeLogger:
            @staticmethod
            def subsection(message, to="both"):
                return None

            @staticmethod
            def turn(speaker, action, to="both", invalid_reason=None):
                return None

        class GuidanceUser:
            user_id = "user_1"

            def __init__(self, action_type=UserActionType.FURTHER_INSTRUCT):
                self.action_type = action_type
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return UserAction(self.action_type, output="User response.")

        class PrivateAgent:
            agent_id = "agent_1"
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self, action, user):
                self.action = action
                self.user = user
                self.calls = 0
                self.dialogue_with_user = Dialogue("agent_1", "user_1")

            def respond_to_user(self, *args, **kwargs):
                self.calls += 1
                return self.action

        def make_runner():
            runner = RoundRunner.__new__(RoundRunner)
            runner.game = SimpleNamespace(state={})
            runner.logger = FakeLogger()
            runner.current_turn = 0
            runner.max_agent_user_turns = 60
            runner.max_agent_user_exchanges = 4
            runner._last_invalid_error_agent_user = None
            return runner

        invalid_agent = PrivateAgent(
            AgentAction(AgentActionType.ACCEPT, output="Invalid private accept."),
            GuidanceUser(),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "without consuming a model-output attempt",
        ):
            make_runner()._run_agent_user_until_consent(
                invalid_agent,
                consultation_mode="guidance",
                subsection_label="guidance",
            )
        self.assertEqual(invalid_agent.calls, 1)
        self.assertEqual(invalid_agent.user.calls, 0)

        invalid_user = GuidanceUser(UserActionType.CONSENT)
        reporting_agent = PrivateAgent(
            AgentAction(AgentActionType.MESSAGE_USER, output="Guidance question."),
            invalid_user,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "without consuming a model-output attempt",
        ):
            make_runner()._run_agent_user_until_consent(
                reporting_agent,
                consultation_mode="guidance",
                subsection_label="guidance",
            )
        self.assertEqual(reporting_agent.calls, 1)
        self.assertEqual(invalid_user.calls, 1)

    def test_generated_initial_guidance_uses_one_shared_user_budget(self):
        class InvalidGuidanceUser:
            def __init__(self, user_id):
                self.user_id = user_id
                self.config = {"type": "llm"}
                self.calls = []

            def respond_to_agent(self, *args, attempt_budget=None, **kwargs):
                self.calls.append(
                    (
                        attempt_budget.decision_cycle_id,
                        attempt_budget.used_attempts,
                        attempt_budget.next_retry_mode,
                    )
                )
                attempt_budget.consume()
                return UserAction(UserActionType.CONSENT, output="Wrong initial action.")

        def participant(agent_id, user_id):
            return SimpleNamespace(
                agent_id=agent_id,
                user=InvalidGuidanceUser(user_id),
                no_user=False,
                current_reward=100.0,
                payout_weight=1.0,
                auxiliary_objectives=[],
                dialogue_with_user=Dialogue(agent_id, user_id),
            )

        agent_1 = participant("agent_1", "user_1")
        agent_2 = participant("agent_2", "user_2")
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = SimpleNamespace(
            agent_1=agent_1,
            agent_2=agent_2,
            game_dialogue=Dialogue("agent_1", "agent_2"),
            state={},
        )
        runner.current_turn = 0
        runner._initial_messages_injected = set()
        runner._initial_message_for_agent = lambda agent: ""

        with self.assertRaises(ModelOutputAttemptsExceeded) as raised:
            runner._run_generated_initial_guidance_consultations()

        self.assertEqual(raised.exception.budget.actor_id, "user_1")
        self.assertEqual(len(agent_1.user.calls), 8)
        self.assertEqual(
            len({call[0] for call in agent_1.user.calls}),
            1,
        )
        self.assertEqual(
            [call[1] for call in agent_1.user.calls],
            list(range(8)),
        )
        self.assertEqual(
            [call[2] for call in agent_1.user.calls],
            [RetryMode.INITIAL] + [RetryMode.RESAMPLE] * 7,
        )
        self.assertEqual(agent_2.user.calls, [])

    def test_exhaustion_becomes_persisted_invalid_model_output(self):
        class FakeLogger:
            def __init__(self):
                self.sections = []
                self.closed = False

            def section(self, message):
                self.sections.append(message)

            def close(self):
                self.closed = True

        def participant(agent_id, user_id):
            user = SimpleNamespace(
                user_id=user_id,
                private_info={},
                prompt_logger=None,
                prompts={},
            )
            return SimpleNamespace(
                agent_id=agent_id,
                user=user,
                current_reward=100.0,
                payout_weight=1.0,
                auxiliary_objectives=[],
                dialogue_with_user=Dialogue(agent_id, user_id),
                prompt_logger=None,
                prompts={},
                require_thinking=False,
            )

        agent_1 = participant("agent_1", "user_1")
        agent_2 = participant("agent_2", "user_2")
        game = SimpleNamespace(
            agent_1=agent_1,
            agent_2=agent_2,
            agents=[agent_1, agent_2],
            game_dialogue=Dialogue("agent_1", "agent_2"),
            state={},
            end_reason=None,
            quitter_user_id=None,
            agreed_terms=None,
            prepare_negotiation=lambda: None,
            get_game_state_for_prompt=lambda: "",
            calculate_payout=lambda user: 0.0,
        )
        budget = ModelOutputAttemptBudget(actor_id="user_1")
        for _ in range(8):
            budget.consume(RetryMode.RESAMPLE)
            budget.record_failure(
                "Missing user ACTION.",
                raw_response="malformed user response",
                next_retry_mode=RetryMode.RESAMPLE,
            )

        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            logger = FakeLogger()
            runner = RoundRunner.__new__(RoundRunner)
            runner.game = game
            runner.game_id = "test-game"
            runner.log_dir = log_dir
            runner.config = {}
            runner.current_turn = 0
            runner._update_turn_state = lambda next_speaker=None: None
            runner._inject_initial_messages_once = lambda: None
            runner._seed_preloaded_public_visible_history = lambda: None
            runner._log_preloaded_negotiation_history = lambda: None

            def fail_negotiation():
                raise ModelOutputAttemptsExceeded(budget)

            runner._run_negotiation = fail_negotiation
            with patch("negsim.orchestrator.ConversationLogger", return_value=logger), patch(
                "negsim.orchestrator.initialize_last_api_exchange_files"
            ):
                outcome = runner.run_game()

            self.assertEqual(outcome, "invalid_model_output")
            self.assertEqual(game.end_reason, "invalid_model_output")
            self.assertTrue(logger.closed)
            self.assertEqual(
                game.model_output_failure["decision_cycle_id"],
                budget.decision_cycle_id,
            )
            self.assertEqual(game.model_output_failure["actor_id"], "user_1")
            self.assertEqual(game.model_output_failure["used_attempts"], 8)
            self.assertEqual(game.model_output_failure["max_attempts"], 8)
            self.assertEqual(
                game.model_output_failure["last_retry_mode"],
                RetryMode.RESAMPLE.value,
            )
            self.assertEqual(
                game.model_output_failure["next_retry_mode"],
                RetryMode.RESAMPLE.value,
            )
            self.assertEqual(
                game.model_output_failure["last_error"],
                "Missing user ACTION.",
            )

            write_results(log_dir, game, "test-game", {})
            persisted = json.loads(
                (log_dir / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["outcome"], "invalid_model_output")
            self.assertEqual(persisted["first_speaker"], "agent_1")
            self.assertEqual(
                persisted["model_output_failure"],
                game.model_output_failure,
            )


if __name__ == "__main__":
    unittest.main()
