import unittest
import json
from types import SimpleNamespace

from negsim.action import AgentAction, AgentActionType, UserAction, UserActionType
from negsim.agent import (
    Agent,
    _format_dialogue_as_messages as _format_agent_dialogue,
    _format_visible_dialogue_as_messages,
    _get_agent_response_schema,
    _get_response_format_instruction,
    _negotiation_actions_for_turn,
    _parse_agent_response,
    _parse_agent_response_json,
    _select_negotiation_instruction,
)
from negsim.dialogue import Dialogue, count_ask_user_messages
from negsim.flow_status import format_game_state_with_flow_status
from negsim.model_output_errors import (
    ModelOutputAttemptsExceeded,
    RetryMode,
)
from negsim.orchestrator import ConsultationResult, GameLimitExceededError, RoundRunner
from negsim.response_parsing import normalize_newline_only_labeled_response
from negsim.user import DummyYesUser, LLMUser
from negsim.user import (
    _format_dialogue_as_messages as _format_user_dialogue,
    _format_game_dialogue_as_text,
    _format_user_retry_prompt,
    _parse_user_response,
    _prepare_user_repair,
)


class ResponseParsingTests(unittest.TestCase):
    def test_newline_only_normalizer_never_infers_missing_labels(self) -> None:
        raw = (
            "THINKING: keep negotiating ACTION: FURTHER_INSTRUCT "
            "OUTPUT: Please counter at $64."
        )
        normalized = normalize_newline_only_labeled_response(
            raw,
            require_thinking=True,
        )

        self.assertIsNotNone(normalized)
        self.assertEqual(
            normalized.replace("\n", ""),
            raw,
        )
        output, action, thinking = _parse_user_response(
            normalized,
            allowed_actions=["FURTHER_INSTRUCT"],
            require_thinking=True,
        )
        self.assertEqual(action, UserActionType.FURTHER_INSTRUCT)
        self.assertEqual(output, "Please counter at $64.")
        self.assertEqual(thinking, "keep negotiating")

        rejected = [
            "FURTHER_INSTRUCT\nOUTPUT: Please counter at $64.",
            "ACTION: FURTHER_INSTRUCT\nOUTPUT: Please counter at $64.",
            "THINKING: x ACTION: FURTHER_INSTRUCT OUTPUT: one OUTPUT: two",
            "THINKING: x OUTPUT: one ACTION: FURTHER_INSTRUCT",
            "**THINKING:** x **ACTION:** FURTHER_INSTRUCT **OUTPUT:** one",
            "THINKING: x NOTACTION: FURTHER_INSTRUCT OUTPUT: one",
            "THINKING: x ACTION: FURTHER_INSTRUCT MY_OUTPUT: one",
        ]
        for candidate in rejected:
            with self.subTest(candidate=candidate):
                self.assertIsNone(
                    normalize_newline_only_labeled_response(
                        candidate,
                        require_thinking=True,
                    )
                )

        embedded_details = (
            "THINKING: x ACTION: PROPOSE MY_PROPOSAL_DETAILS: {} OUTPUT: one"
        )
        normalized_embedded = normalize_newline_only_labeled_response(
            embedded_details,
            require_thinking=True,
        )
        self.assertIsNotNone(normalized_embedded)
        with self.assertRaises(ValueError):
            _parse_agent_response(
                normalized_embedded,
                allowed_actions=["PROPOSE"],
                require_thinking=True,
            )

    def test_round_runner_reads_consultation_limits(self) -> None:
        def game():
            return SimpleNamespace(
                agent_1=SimpleNamespace(agent_id="agent_1", no_user=True),
                agent_2=SimpleNamespace(agent_id="agent_2", no_user=True),
            )

        shared_runner = RoundRunner(
            game(),
            game_id="shared",
            log_dir=".",
            config={"max_agent_user_exchanges": 7},
        )
        self.assertEqual(shared_runner.max_agent_user_exchanges, 7)
        self.assertEqual(shared_runner.max_ask_user_consultation_exchanges, 7)

        default_runner = RoundRunner(
            game(),
            game_id="default",
            log_dir=".",
            config={},
        )
        self.assertEqual(default_runner.max_negotiation_turns, 40)
        self.assertEqual(default_runner.max_agent_user_exchanges, 4)
        self.assertEqual(default_runner.max_ask_user_consultation_exchanges, 4)

        separate_runner = RoundRunner(
            game(),
            game_id="separate",
            log_dir=".",
            config={
                "max_agent_user_exchanges": 20,
                "max_ask_user_consultation_exchanges": 8,
            },
        )
        self.assertEqual(separate_runner.max_agent_user_exchanges, 20)
        self.assertEqual(separate_runner.max_ask_user_consultation_exchanges, 8)

        with self.assertRaisesRegex(ValueError, "positive integer"):
            RoundRunner(
                game(),
                game_id="invalid",
                log_dir=".",
                config={"max_ask_user_consultation_exchanges": 0},
            )

    def test_parser_tolerates_markdown_labels_full_width_colon_and_fence(self) -> None:
        raw = """```text
**THINKING:** Keep the reasoning private.
- **ACTION:** FURTHER_INSTRUCT
### OUTPUT： Please continue, but do not exceed the limit.
```"""

        output, action_type, thinking = _parse_user_response(
            raw,
            allowed_actions=["FURTHER_INSTRUCT"],
            require_thinking=True,
        )

        self.assertEqual(action_type, UserActionType.FURTHER_INSTRUCT)
        self.assertEqual(thinking, "Keep the reasoning private.")
        self.assertEqual(output, "Please continue, but do not exceed the limit.")

    def test_markdown_labels_are_rejected_without_thinking_protocol(self) -> None:
        raw = "**ACTION:** FURTHER_INSTRUCT\n**OUTPUT:** Continue."

        with self.assertRaisesRegex(ValueError, "Missing user ACTION"):
            _parse_user_response(
                raw,
                allowed_actions=["FURTHER_INSTRUCT"],
                require_thinking=False,
            )

    def test_parser_accepts_proposal_details_with_spaces_in_label(self) -> None:
        raw = """THINKING: Make a valid offer.
ACTION: PROPOSE
PROPOSAL DETAILS: {"price": 25}
OUTPUT: I can offer 25.
"""

        output, action_type, details, thinking = _parse_agent_response(
            raw,
            allowed_actions=["PROPOSE"],
            require_thinking=True,
        )

        self.assertEqual(action_type, AgentActionType.PROPOSE)
        self.assertEqual(details, '{"price": 25}')
        self.assertEqual(thinking, "Make a valid offer.")
        self.assertEqual(output, "I can offer 25.")

    def test_request_user_confirmation_is_private_and_system_supplied(self) -> None:
        labeled = (
            "THINKING: Ask my user for an independent decision.\n"
            "ACTION: REQUEST_USER_CONFIRMATION\n"
            "OUTPUT: Please approve or reject these exact terms."
        )
        output, action_type, details, thinking = _parse_agent_response(
            labeled,
            allowed_actions=["REQUEST_USER_CONFIRMATION"],
            require_thinking=True,
        )
        self.assertEqual(action_type, AgentActionType.REQUEST_USER_CONFIRMATION)
        self.assertIsNone(details)
        self.assertIn("approve or reject", output)
        self.assertIn("independent decision", thinking)

        structured = json.dumps(
            {
                "thinking": "Ask for a decision.",
                "action": "REQUEST_USER_CONFIRMATION",
                "output": "Please decide.",
            }
        )
        _, structured_type, structured_details, _ = _parse_agent_response_json(
            structured,
            allowed_actions=["REQUEST_USER_CONFIRMATION"],
        )
        self.assertEqual(
            structured_type,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        )
        self.assertIsNone(structured_details)

        _, structured_type, structured_details, _ = _parse_agent_response_json(
            json.dumps(
                {
                    "thinking": "Ask for a decision.",
                    "action": "REQUEST_USER_CONFIRMATION",
                    "proposal_details": '{"terms": "model-authored"}',
                    "output": "Please decide.",
                }
            ),
            allowed_actions=["REQUEST_USER_CONFIRMATION"],
        )
        self.assertEqual(
            structured_type,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        )
        self.assertIsNone(structured_details)

        _, labeled_type, labeled_details, _ = _parse_agent_response(
            labeled.replace(
                "OUTPUT:",
                'PROPOSAL_DETAILS: {"terms": "model-authored"}\nOUTPUT:',
            ),
            allowed_actions=["REQUEST_USER_CONFIRMATION"],
            require_thinking=True,
        )
        self.assertEqual(
            labeled_type,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        )
        self.assertIsNone(labeled_details)
        with self.assertRaisesRegex(ValueError, "participant-facing"):
            _parse_agent_response_json(
                json.dumps(
                    {
                        "thinking": "Ask for a decision.",
                        "action": "REQUEST_USER_CONFIRMATION",
                        "output": "",
                    }
                ),
                allowed_actions=["REQUEST_USER_CONFIRMATION"],
            )
        with self.assertRaisesRegex(ValueError, "not allowed here"):
            _parse_agent_response(
                labeled,
                allowed_actions=["PROPOSE", "ACCEPT"],
                require_thinking=True,
            )

        response_format = _get_response_format_instruction(
            False,
            'PROPOSAL_DETAILS: {"terms": "..."}',
            "REQUEST_USER_CONFIRMATION",
            require_thinking=True,
        )
        self.assertNotIn("PROPOSAL_DETAILS", response_format)
        self.assertNotIn("proposal_details", response_format)
        self.assertNotIn("OUTPUT is not delivered", response_format)
        schema = _get_agent_response_schema(["REQUEST_USER_CONFIRMATION"])
        schema_body = schema["json_schema"]["schema"]
        self.assertNotIn("proposal_details", schema_body["properties"])
        self.assertNotIn("proposal_details", schema_body["required"])

        guidance_response_format = _get_response_format_instruction(
            False,
            'PROPOSAL_DETAILS: {"terms": "..."}',
            "MESSAGE_USER or MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            require_thinking=True,
        )
        structured_guidance_response_format = _get_response_format_instruction(
            True,
            'PROPOSAL_DETAILS: {"terms": "..."}',
            "MESSAGE_USER or MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            require_thinking=True,
        )
        for guidance_format in (
            guidance_response_format,
            structured_guidance_response_format,
        ):
            self.assertNotIn("PROPOSAL_DETAILS", guidance_format)
            self.assertNotIn("proposal_details", guidance_format)
        guidance_schema = _get_agent_response_schema(
            ["MESSAGE_USER", "MESSAGE_USER_AND_CONTINUE_NEGOTIATION"]
        )["json_schema"]["schema"]
        self.assertNotIn("proposal_details", guidance_schema["properties"])
        self.assertNotIn("proposal_details", guidance_schema["required"])

    def test_thinking_protocol_rejects_missing_thinking(self) -> None:
        raw = "ACTION: MESSAGE_USER\nOUTPUT: Here is the status."

        with self.assertRaisesRegex(ValueError, "THINKING"):
            _parse_agent_response(
                raw,
                allowed_actions=["MESSAGE_USER"],
                require_thinking=True,
            )

    def test_thinking_protocol_never_falls_back_to_raw_text_as_output(self) -> None:
        raw = "THINKING: secret strategy\nACTION: MESSAGE_USER"

        with self.assertRaisesRegex(ValueError, "OUTPUT"):
            _parse_agent_response(
                raw,
                allowed_actions=["MESSAGE_USER"],
                require_thinking=True,
            )

    def test_accept_allows_missing_or_empty_output(self) -> None:
        labeled_cases = (
            "THINKING: The offer is acceptable.\nACTION: ACCEPT",
            "THINKING: The offer is acceptable.\nACTION: ACCEPT\nOUTPUT:",
        )
        for raw in labeled_cases:
            with self.subTest(protocol="labeled", raw=raw):
                output, action_type, details, thinking = _parse_agent_response(
                    raw,
                    allowed_actions=["PROPOSE", "ACCEPT"],
                    require_thinking=True,
                )
                self.assertEqual(output, "")
                self.assertEqual(action_type, AgentActionType.ACCEPT)
                self.assertIsNone(details)
                self.assertEqual(thinking, "The offer is acceptable.")

        structured_cases = (
            {
                "thinking": "The offer is acceptable.",
                "action": "ACCEPT",
                "proposal_details": "",
            },
            {
                "thinking": "The offer is acceptable.",
                "action": "ACCEPT",
                "proposal_details": "",
                "output": "",
            },
        )
        for payload in structured_cases:
            with self.subTest(protocol="structured", payload=payload):
                output, action_type, details, thinking = _parse_agent_response_json(
                    json.dumps(payload),
                    allowed_actions=["PROPOSE", "ACCEPT"],
                )
                self.assertEqual(output, "")
                self.assertEqual(action_type, AgentActionType.ACCEPT)
                self.assertIsNone(details)
                self.assertEqual(thinking, "The offer is acceptable.")

    def test_non_accept_participant_actions_still_require_output(self) -> None:
        labeled = (
            "THINKING: Make the offer.\n"
            "ACTION: PROPOSE\n"
            'PROPOSAL_DETAILS: {"terms":"50-50"}'
        )
        with self.assertRaisesRegex(ValueError, "OUTPUT"):
            _parse_agent_response(
                labeled,
                allowed_actions=["PROPOSE", "ACCEPT"],
                require_thinking=True,
            )

        with self.assertRaisesRegex(ValueError, "participant-facing"):
            _parse_agent_response_json(
                json.dumps(
                    {
                        "thinking": "Make the offer.",
                        "action": "PROPOSE",
                        "proposal_details": '{"terms":"50-50"}',
                    }
                ),
                allowed_actions=["PROPOSE", "ACCEPT"],
            )

    def test_structured_internal_action_preserves_empty_output(self) -> None:
        cases = (("ASK USER", AgentActionType.ASK_USER),)
        for action, expected_type in cases:
            with self.subTest(action=action):
                output, action_type, details, thinking = _parse_agent_response_json(
                    json.dumps(
                        {
                            "thinking": "I have finished this internal step.",
                            "action": action,
                            "proposal_details": "",
                            "output": "",
                        }
                    ),
                    allowed_actions=[action],
                )

                self.assertEqual(output, "")
                self.assertEqual(action_type, expected_type)
                self.assertIsNone(details)
                self.assertEqual(thinking, "I have finished this internal step.")

    def test_structured_message_user_rejects_empty_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "participant-facing"):
            _parse_agent_response_json(
                json.dumps(
                    {
                        "thinking": "I should tell the user what happened.",
                        "action": "MESSAGE_USER",
                        "proposal_details": "",
                        "output": "",
                    }
                ),
                allowed_actions=[
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
            )

    def test_thinking_response_format_is_opt_in(self) -> None:
        without_thinking = _get_response_format_instruction(
            False, 'PROPOSAL_DETAILS: {"price": X}', "PROPOSE"
        )
        thinking = _get_response_format_instruction(
            False,
            'PROPOSAL_DETAILS: {"price": X}',
            "PROPOSE",
            require_thinking=True,
        )
        internal = _get_response_format_instruction(
            False,
            'PROPOSAL_DETAILS: {"price": X}',
            "PROPOSE or ASK USER or ACCEPT AND CONFIRM WITH USER",
            require_thinking=True,
        )
        structured_internal = _get_response_format_instruction(
            True,
            'PROPOSAL_DETAILS: {"price": X}',
            "PROPOSE or ASK USER",
            require_thinking=True,
        )
        combined_user_message = _get_response_format_instruction(
            False,
            'PROPOSAL_DETAILS: {"price": X}',
            "MESSAGE_USER or MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            require_thinking=True,
        )

        self.assertNotIn("THINKING:", without_thinking)
        self.assertIn("THINKING:", thinking)
        self.assertIn("never sent", thinking)
        self.assertIn("For participant-facing actions, only OUTPUT is delivered", internal)
        self.assertIn("ASK USER or ACCEPT AND CONFIRM WITH USER are internal actions", internal)
        self.assertIn("Their OUTPUT is not delivered", internal)
        self.assertIn("It may contain a brief internal note", internal)
        self.assertIn("For participant-facing actions, OUTPUT must contain", internal)
        self.assertNotIn("leave empty", internal)
        self.assertIn("ASK USER is an internal action", structured_internal)
        self.assertIn("Its OUTPUT is not delivered", structured_internal)
        self.assertIn("It may contain a brief internal note", structured_internal)
        self.assertIn(
            "For participant-facing actions, OUTPUT must contain",
            structured_internal,
        )
        self.assertNotIn("leave empty", structured_internal)
        self.assertNotIn("internal action", combined_user_message)
        self.assertNotIn("OUTPUT is not delivered", combined_user_message)
        self.assertIn(
            "OUTPUT: <the message delivered to the other participant>",
            combined_user_message,
        )



    def test_private_thinking_is_not_rendered_for_other_participant(self) -> None:
        dialogue = Dialogue("agent_1", "agent_2")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.MESSAGE_USER,
                output="public agent message",
                thinking="agent secret sentinel",
            ),
        )
        dialogue.add_turn(
            "agent_2",
            1,
            AgentAction(
                AgentActionType.MESSAGE_USER,
                output="public peer message",
                thinking="peer secret sentinel",
            ),
        )

        agent_messages = _format_agent_dialogue(
            dialogue, "agent_1", show_own_thinking=False
        )
        agent_text = "\n".join(message["content"] for message in agent_messages)
        self.assertNotIn("agent secret sentinel", agent_text)
        self.assertNotIn("peer secret sentinel", agent_text)

        user_dialogue = Dialogue("agent_1", "user_1")
        user_dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.MESSAGE_USER,
                output="public consultation",
                thinking="consultation secret sentinel",
            ),
        )
        user_messages = _format_user_dialogue(user_dialogue, "agent_1")
        user_text = "\n".join(message["content"] for message in user_messages)
        self.assertNotIn("consultation secret sentinel", user_text)

    def test_unified_visible_dialogue_preserves_cross_channel_order_without_windows(self) -> None:
        dialogue = Dialogue("agent_1", "user_1")
        dialogue.add_turn(
            "agent_2",
            1,
            AgentAction(AgentActionType.PROPOSE, output="public before"),
            channel="agent_agent",
            turn_num=1,
        )
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(AgentActionType.ASK_USER, output="persistent ask note"),
            channel="agent_agent",
            turn_num=2,
        )
        for index in range(12):
            if index % 2 == 0:
                speaker = "agent_1"
                action = AgentAction(
                    AgentActionType.MESSAGE_USER,
                    output=f"private message {index}",
                )
            else:
                speaker = "user_1"
                action = UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output=f"private message {index}",
                )
            dialogue.add_turn(
                speaker,
                1,
                action,
                channel="agent_user",
                turn_num=2,
            )
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                output="persistent continue note",
            ),
            channel="agent_user",
            turn_num=2,
            visible_to_user=True,
        )
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(AgentActionType.PROPOSE, output="public after"),
            channel="agent_agent",
            turn_num=2,
        )

        public_view = _format_visible_dialogue_as_messages(
            dialogue,
            "agent_1",
            primary_channel="agent_agent",
        )
        public_text = "\n".join(message["content"] for message in public_view)
        for index in range(12):
            self.assertIn(f"private message {index}", public_text)
        self.assertIn("persistent continue note", public_text)
        positions = [
            public_text.index("public before"),
            public_text.index("persistent ask note"),
            public_text.index("private message 0"),
            public_text.index("private message 11"),
            public_text.index("persistent continue note"),
            public_text.index("public after"),
        ]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(public_text.count("=== BEGIN AGENT-USER DIALOGUE ==="), 1)
        self.assertNotIn("event_sequence", public_text)

        private_view = _format_visible_dialogue_as_messages(
            dialogue,
            "agent_1",
            primary_channel="agent_user",
        )
        private_text = "\n".join(message["content"] for message in private_view)
        self.assertLess(
            private_text.index("persistent ask note"),
            private_text.index("private message 0"),
        )
        self.assertLess(
            private_text.index("persistent continue note"),
            private_text.index("public after"),
        )
        self.assertEqual(private_text.count("=== BEGIN AGENT-AGENT DIALOGUE ==="), 2)

    def test_user_output_keeps_multiline_ordered_list(self) -> None:
        raw = """THINKING: I prefer early morning.
ACTION: FURTHER_INSTRUCT
OUTPUT: Please do not accept 2026-07-10 15:00-16:00. Offer these:
1) 2026-07-10 08:00-09:00
2) 2026-07-13 08:00-09:00
3) 2026-07-10 13:00-14:00

Fallback:
- 2026-07-10 09:00-10:00
- 2026-07-13 09:00-10:00
"""
        output, action_type, thinking = _parse_user_response(raw)

        self.assertEqual(action_type, UserActionType.FURTHER_INSTRUCT)
        self.assertEqual(thinking, "I prefer early morning.")
        self.assertIn("1) 2026-07-10 08:00-09:00", output)
        self.assertIn("3) 2026-07-10 13:00-14:00", output)
        self.assertIn("- 2026-07-13 09:00-10:00", output)

    def test_agent_output_keeps_multiline_user_message(self) -> None:
        raw = """THINKING: Need more guidance.
ACTION: MESSAGE_USER
OUTPUT: Please choose:
- first option
- second option
"""
        output, action_type, details, thinking = _parse_agent_response(
            raw, allow_message_user=True
        )

        self.assertEqual(action_type, AgentActionType.MESSAGE_USER)
        self.assertIsNone(details)
        self.assertEqual(thinking, "Need more guidance.")
        self.assertEqual(output, "Please choose:\n- first option\n- second option")

    def test_agent_combined_user_message_and_continue_is_prompt_controlled(self) -> None:
        raw = """THINKING: I have enough guidance.
ACTION: MESSAGE_USER_AND_CONTINUE_NEGOTIATION
OUTPUT: Thanks, I will continue negotiating with that in mind.
"""
        output, action_type, details, thinking = _parse_agent_response(
            raw,
            allowed_actions=[
                "MESSAGE_USER",
                "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            ],
        )

        self.assertEqual(
            action_type,
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        )
        self.assertIsNone(details)
        self.assertEqual(thinking, "I have enough guidance.")
        self.assertIn("continue negotiating", output)

    def test_agent_combined_user_message_and_continue_rejects_empty_output(self) -> None:
        raw = """THINKING: I have enough guidance.
ACTION: MESSAGE_USER_AND_CONTINUE_NEGOTIATION
OUTPUT:
"""
        with self.assertRaisesRegex(ValueError, "participant-facing"):
            _parse_agent_response(
                raw,
                allowed_actions=[
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
                require_thinking=True,
            )

    def test_agent_message_user_rejects_empty_output(self) -> None:
        raw = """THINKING: I should tell the user what happened.
ACTION: MESSAGE_USER
OUTPUT:
"""
        with self.assertRaisesRegex(ValueError, "participant-facing"):
            _parse_agent_response(
                raw,
                allowed_actions=[
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
                require_thinking=True,
            )

    def test_agent_rejects_action_not_allowed_by_prompt(self) -> None:
        raw = """THINKING: I should make a concrete offer.
ACTION: PROPOSE
PROPOSAL_DETAILS: {"terms":"50-50"}
OUTPUT: I propose 50-50.
"""
        with self.assertRaises(ValueError):
            _parse_agent_response(
                raw,
                allowed_actions=[
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
            )

    def test_user_rejects_action_not_allowed_by_prompt(self) -> None:
        raw = """THINKING: I agree.
ACTION: CONSENT
OUTPUT: Sounds good.
"""
        with self.assertRaises(ValueError):
            _parse_user_response(raw, allowed_actions=["FURTHER_INSTRUCT"])

    def test_private_channel_rejects_propose(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.turns = []

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action, to, invalid_reason))

        runner = RoundRunner.__new__(RoundRunner)
        runner.logger = FakeLogger()
        runner._last_invalid_error_agent_user = None
        agent = SimpleNamespace(
            agent_id="agent_1",
            user=SimpleNamespace(user_id="user_1"),
        )
        action = AgentAction(
            AgentActionType.PROPOSE,
            output="I offer 57.75.",
            action_details={"price": 57.75},
        )

        self.assertFalse(runner._validate_agent_user_action(agent, action, None))
        self.assertIn("public negotiation action", runner._last_invalid_error_agent_user)
        self.assertEqual(len(runner.logger.turns), 1)
        self.assertEqual(runner.logger.turns[0][2], "agent_1")

    def test_required_message_user_rejects_combined_action_and_retries(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []
                self.responses = [
                    (
                        "THINKING: I can return immediately.\n"
                        "ACTION: MESSAGE_USER_AND_CONTINUE_NEGOTIATION\n"
                        "OUTPUT: Return to the negotiation."
                    ),
                    (
                        "THINKING: I need to ask my user first.\n"
                        "ACTION: MESSAGE_USER\n"
                        "OUTPUT: Which scheduling preference should I prioritize?"
                    ),
                ]

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                return self.responses.pop(0)

        class FakeGame:
            state = {}

            def get_proposal_format_for_prompt(self):
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            def get_user_consultation_response_format_instruction(self, details_fmt, actions_str):
                return None

            def get_private_info_for_prompt(self, user_id):
                return ""

            def get_game_blurb_for_agent_prompt(self):
                return "Schedule a meeting."

            def get_game_state_for_prompt(self):
                return "Negotiation in progress."

            def get_proposal_format_description(self, user_id):
                return "Use concrete scheduling terms."

        agent = Agent.__new__(Agent)
        agent.agent_id = "agent_1"
        agent.user = DummyYesUser({}, "user_1")
        agent.orchestrator_settings = {"max_negotiation_turns": 20}
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
        agent.no_user = False
        agent.model = FakeModel()
        agent.prompts = {
            "respond_to_user": {
                "system": (
                    "{response_format_instruction}\n{private_info}\n{game_blurb}\n"
                    "{game_state}\n{proposal_format}\n{agent_agent_history}"
                ),
                "current_prompt_guidance": "Choose from {actions}.\n{response_format_instruction}",
                "current_prompt_after_invalid_guidance": (
                    "{invalid_error}\nChoose from {actions}.\n{response_format_instruction}"
                ),
                "actions_guidance": [
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
            }
        }
        agent.auxiliary_objectives = []
        agent.current_reward = 100.0
        agent.payout_weight = 1.0

        action = agent.respond_to_user(
            game=FakeGame(),
            game_dialogue=Dialogue("agent_1", "agent_2"),
            required_action=AgentActionType.MESSAGE_USER,
            consultation_mode="guidance",
        )

        self.assertEqual(action.action_type, AgentActionType.MESSAGE_USER)
        self.assertEqual(len(agent.model.prompts), 2)
        self.assertIn(
            "Choose from MESSAGE_USER.", agent.model.prompts[0]["current_prompt"]
        )
        self.assertNotIn(
            "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
            agent.model.prompts[0]["current_prompt"],
        )
        self.assertEqual(
            agent.model.prompts[1]["current_prompt"],
            agent.model.prompts[0]["current_prompt"],
        )

    def test_guidance_consultation_can_end_without_propose_or_consent(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.turns = []

            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action.action_type, action.output, to))

        class FakeGame:
            state = {}

        class FakeUser:
            user_id = "user_1"

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
                self.last_mode = consultation_mode
                return UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output="Please prioritize a fair split but avoid unnecessary risk.",
                )

        class FakeAgent:
            agent_id = "agent_1"
            user = FakeUser()
            dialogue_with_user = Dialogue("agent_1", "user_1")
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.calls = []
                self.actions = [
                    AgentAction(
                        AgentActionType.MESSAGE_USER,
                        output="What matters most to you before I negotiate?",
                    ),
                    AgentAction(
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                        output=(
                            "Thanks, I have enough guidance and will continue negotiating."
                        ),
                    ),
                ]

            def respond_to_user(
                self,
                game,
                game_dialogue=None,
                agreed_deal=None,
                required_action=None,
                consultation_mode=None,
                attempt_budget=None,
            ):
                self.calls.append((consultation_mode, required_action))
                return self.actions.pop(0)

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        logger = FakeLogger()
        runner.logger = logger
        runner.current_turn = 0
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 5
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        final_action = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            consultation_mode="guidance",
            require_initial_message_user=True,
            subsection_label="guidance",
        )

        self.assertEqual(
            final_action.action.action_type,
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        )
        self.assertEqual(final_action.end_reason, "agent_continue")
        self.assertEqual(final_action.completed_exchanges, 1)
        self.assertEqual(
            agent.calls,
            [("guidance", AgentActionType.MESSAGE_USER), ("guidance", None)],
        )
        self.assertEqual(agent.user.last_mode, "guidance")
        history_actions = [
            turn["action"].action_type
            for turn in agent.dialogue_with_user.conversation_history
        ]
        self.assertEqual(
            history_actions,
            [
                AgentActionType.MESSAGE_USER,
                UserActionType.FURTHER_INSTRUCT,
                AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
            ],
        )
        self.assertEqual(
            [
                turn
                for turn in logger.turns
                if turn[1]
                == AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
            ],
            [
                (
                    "agent_1",
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                    "Thanks, I have enough guidance and will continue negotiating.",
                    "agent_1",
                ),
                (
                    "agent_1",
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                    "Thanks, I have enough guidance and will continue negotiating.",
                    "user_1",
                ),
            ],
        )

    def test_consultation_auto_ends_after_agent_at_exchange_and_message_limits(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.turns = []

            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action.action_type, to))

        class FakeGame:
            state = {}

        class FakeUser:
            user_id = "user_1"

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
                return UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output="Keep pushing for my priorities.",
                )

        class FakeAgent:
            agent_id = "agent_1"
            user = FakeUser()
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.dialogue_with_user = Dialogue("agent_1", "user_1")
                self.calls = 0

            def respond_to_user(
                self,
                game,
                game_dialogue=None,
                agreed_deal=None,
                required_action=None,
                consultation_mode=None,
                attempt_budget=None,
            ):
                self.calls += 1
                action_type = (
                    required_action
                    or AgentActionType.MESSAGE_USER
                )
                return AgentAction(
                    action_type,
                    output=(
                        "I understand. I will return to the negotiation with that guidance."
                        if self.calls == 2
                        else "What should I prioritize next?"
                    ),
                )

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        logger = FakeLogger()
        runner.logger = logger
        runner.current_turn = 1
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 1
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        final_action = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            consultation_mode="guidance",
            subsection_label="guidance",
        )

        self.assertEqual(
            final_action.action.action_type,
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        )
        self.assertEqual(final_action.end_reason, "exchange_limit")
        self.assertEqual(final_action.completed_exchanges, 1)
        self.assertIn("return to the negotiation", final_action.action.output)
        self.assertIn(
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
            [
                turn["action"].action_type
                for turn in agent.dialogue_with_user.conversation_history
            ],
        )
        self.assertEqual(
            [
                turn
                for turn in logger.turns
                if turn[1]
                == AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION
            ],
            [
                (
                    "agent_1",
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                    "agent_1",
                ),
                (
                    "agent_1",
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                    "user_1",
                ),
            ],
        )
        self.assertEqual(
            [turn["speaker"] for turn in agent.dialogue_with_user.conversation_history],
            ["agent_1", "user_1", "agent_1"],
        )
        self.assertEqual(agent.calls, 2)
        self.assertIn(
            "return to the negotiation",
            agent.dialogue_with_user.conversation_history[-1]["action"].output,
        )

        runner.max_agent_user_turns = 3
        runner.max_agent_user_exchanges = 20
        runner.max_ask_user_consultation_exchanges = 20
        message_limited_agent = FakeAgent()
        message_limit_result = runner._run_agent_user_until_consent(
            message_limited_agent,
            Dialogue("agent_1", "agent_2"),
            consultation_origin="ask_user",
            consultation_mode="guidance",
            subsection_label="message-limited guidance",
        )

        self.assertEqual(message_limit_result.end_reason, "message_limit")
        self.assertEqual(message_limit_result.completed_exchanges, 1)
        self.assertEqual(message_limited_agent.calls, 2)
        self.assertEqual(count_ask_user_messages(message_limited_agent.dialogue_with_user), 3)
        self.assertEqual(
            [
                turn["speaker"]
                for turn in message_limited_agent.dialogue_with_user.conversation_history
            ],
            ["agent_1", "user_1", "agent_1"],
        )

    def test_agent_user_model_output_budget_is_eight_and_shared(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.turns = []

            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action.action_type, to, invalid_reason))

        class FakeGame:
            state = {}

        class FakeUser:
            user_id = "user_1"

            def __init__(self):
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return UserAction(UserActionType.FURTHER_INSTRUCT, output="Guidance.")

        class FakeAgent:
            agent_id = "agent_1"
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.calls = 0
                self.user = FakeUser()
                self.dialogue_with_user = Dialogue("agent_1", "user_1")

            def respond_to_user(self, *args, **kwargs):
                self.calls += 1
                kwargs["attempt_budget"].consume()
                return AgentAction(
                    AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                    output="Trying to leave before the required user message.",
                )

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 20
        runner.max_ask_user_consultation_exchanges = 8
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        with self.assertRaises(ModelOutputAttemptsExceeded) as raised:
            runner._run_agent_user_until_consent(
                agent,
                Dialogue("agent_1", "agent_2"),
                consultation_mode="guidance",
                consultation_origin="ask_user",
                require_initial_message_user=True,
                subsection_label="guidance",
            )

        self.assertEqual(agent.calls, 8)
        self.assertEqual(agent.user.calls, 0)
        self.assertEqual(raised.exception.budget.used_attempts, 8)
        self.assertEqual(
            raised.exception.budget.next_retry_mode,
            RetryMode.RESAMPLE,
        )
        self.assertEqual(agent.dialogue_with_user.conversation_history, [])

    def test_agent_user_invalid_attempt_counter_resets_after_valid_exchange(self) -> None:
        class FakeLogger:
            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class FakeGame:
            state = {}

        class FakeUser:
            user_id = "user_1"

            def __init__(self):
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output="Use this guidance.",
                )

        invalid = lambda: AgentAction(
            AgentActionType.PROPOSE,
            output="Malformed proposal.",
            missing_details=True,
        )

        class FakeAgent:
            agent_id = "agent_1"
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.user = FakeUser()
                self.dialogue_with_user = Dialogue("agent_1", "user_1")
                self.actions = [
                    *(invalid() for _ in range(4)),
                    AgentAction(
                        AgentActionType.MESSAGE_USER,
                        output="What should I do?",
                    ),
                    *(invalid() for _ in range(4)),
                    AgentAction(
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                        output="Thanks, I will continue negotiating.",
                    ),
                ]

            def respond_to_user(self, *args, **kwargs):
                kwargs["attempt_budget"].consume()
                return self.actions.pop(0)

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 20
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        result = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            consultation_mode="guidance",
            require_initial_message_user=True,
            subsection_label="guidance",
        )

        self.assertEqual(result.end_reason, "agent_continue")
        self.assertEqual(result.completed_exchanges, 1)
        self.assertEqual(result.max_consecutive_invalid_attempts, 4)
        self.assertEqual(agent.user.calls, 1)

    def test_ask_user_consultation_requires_message_user_on_first_call(self) -> None:
        class FakeLogger:
            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class FakeGame:
            state = {}

        class GuidingUser:
            user_id = "user_1"

            def __init__(self):
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output="Continue with this guidance.",
                )

        class FakeAgent:
            agent_id = "agent_1"
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.user = GuidingUser()
                self.dialogue_with_user = Dialogue("agent_1", "user_1")
                self.calls = []
                self.actions = [
                    AgentAction(AgentActionType.MESSAGE_USER, output="First message."),
                    AgentAction(
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                        output="Thanks, I will continue negotiating.",
                    ),
                ]

            def respond_to_user(self, *args, **kwargs):
                self.calls.append(kwargs)
                return self.actions.pop(0)

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 20
        runner.max_ask_user_consultation_exchanges = 4
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        result = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            consultation_mode="guidance",
            consultation_origin="ask_user",
            require_initial_message_user=True,
            subsection_label="guidance",
        )

        self.assertEqual(result.end_reason, "agent_continue")
        self.assertEqual(len(agent.calls), 2)
        self.assertEqual(
            agent.calls[0]["required_action"], AgentActionType.MESSAGE_USER
        )
        self.assertNotIn("required_action", agent.calls[1])
        self.assertEqual(agent.user.calls, 1)

    def test_public_negotiation_protocol_error_uses_shared_output_budget(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.turns = []

            def subsection(self, message, to="both"):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action, to, invalid_reason))

        class FakeAgent:
            agent_id = "agent_1"
            user = SimpleNamespace(user_id="user_1")

            def __init__(self):
                self.calls = 0
                self.budget_observations = []

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                budget = kwargs["attempt_budget"]
                self.budget_observations.append(
                    (
                        budget.decision_cycle_id,
                        budget.used_attempts,
                        budget.next_retry_mode,
                    )
                )
                budget.consume()
                return AgentAction(
                    AgentActionType.REQUEST_USER_CONFIRMATION,
                    output="Ask my user privately.",
                    action_details={"terms": "must stay private"},
                )

        agent = FakeAgent()
        dialogue = Dialogue("agent_1", "agent_2")
        game = SimpleNamespace(
            agent_1=agent,
            agent_2=SimpleNamespace(
                agent_id="agent_2",
                user=SimpleNamespace(user_id="user_2"),
            ),
            game_dialogue=dialogue,
            end_reason=None,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = game
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_negotiation_turns = 10
        runner.turns_by_agent = {"agent_1": 0, "agent_2": 0}
        runner._last_invalid_error = None
        runner._update_turn_state = lambda next_speaker=None: None

        with self.assertRaises(ModelOutputAttemptsExceeded):
            runner._run_negotiation()

        self.assertEqual(agent.calls, 8)
        self.assertEqual(
            len({item[0] for item in agent.budget_observations}),
            1,
        )
        self.assertEqual(
            [item[1] for item in agent.budget_observations],
            list(range(8)),
        )
        self.assertEqual(
            [item[2] for item in agent.budget_observations],
            [RetryMode.INITIAL] + [RetryMode.RESAMPLE] * 7,
        )
        self.assertIn("private final user confirmation", runner._last_invalid_error)
        self.assertEqual(dialogue.conversation_history, [])
        self.assertEqual(len(runner.logger.turns), 8)
        self.assertTrue(
            all(turn[2] == "agent_1" for turn in runner.logger.turns)
        )
        self.assertTrue(
            all(turn[3] == runner._last_invalid_error for turn in runner.logger.turns)
        )

    def test_negotiation_protocol_errors_share_eight_attempt_budget(self) -> None:
        class FakeLogger:
            def subsection(self, msg, to="both"):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class FakeAgent:
            agent_id = "agent_1"
            no_user = True
            user = SimpleNamespace(user_id="user_1")

            def __init__(self):
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                kwargs["attempt_budget"].consume()
                return AgentAction(AgentActionType.ASK_USER, output="Ask anyway.")

        agent = FakeAgent()
        other_agent = SimpleNamespace(
            agent_id="agent_2",
            user=SimpleNamespace(user_id="user_2"),
        )
        game = SimpleNamespace(
            agent_1=agent,
            agent_2=other_agent,
            game_dialogue=Dialogue("agent_1", "agent_2"),
            end_reason=None,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = game
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_negotiation_turns = 10
        runner.turns_by_agent = {"agent_1": 0, "agent_2": 0}
        runner._last_invalid_error = None
        runner._update_turn_state = lambda next_speaker=None: None

        with self.assertRaises(ModelOutputAttemptsExceeded):
            runner._run_negotiation()

        self.assertEqual(agent.calls, 8)
        self.assertEqual(runner.current_turn, 0)

    def test_post_negotiation_user_rejection_returns_to_negotiation(self) -> None:
        class FakeLogger:
            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class FakeGame:
            state = {}

            def validate_deal(self, details):
                return details

            def check_for_same_proposed_deal(self, action_1, action_2):
                return action_1.action_details == action_2.action_details

        class FakeUser:
            user_id = "user_1"

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
                self.last_mode = consultation_mode
                return UserAction(
                    UserActionType.DONT_CONSENT,
                    output="No, do not accept this. Go back and negotiate a better option.",
                )

        class FakeAgent:
            agent_id = "agent_1"
            user = FakeUser()
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.calls = 0
                self.dialogue_with_user = Dialogue("agent_1", "user_1")

            def respond_to_user(
                self,
                game,
                game_dialogue=None,
                agreed_deal=None,
                consultation_mode=None,
                attempt_budget=None,
            ):
                self.calls += 1
                return AgentAction(
                    AgentActionType.REQUEST_USER_CONFIRMATION,
                    output="Do you approve this tentative deal?",
                )

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        runner.logger = FakeLogger()
        runner.current_turn = 1
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 5
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        for index in range(runner.max_agent_user_turns):
            if index % 2 == 0:
                speaker = agent.agent_id
                prior_action = AgentAction(
                    AgentActionType.MESSAGE_USER,
                    output=f"Prior ASK USER agent message {index}.",
                )
            else:
                speaker = agent.user.user_id
                prior_action = UserAction(
                    UserActionType.FURTHER_INSTRUCT,
                    output=f"Prior ASK USER user message {index}.",
                )
            agent.dialogue_with_user.add_turn(
                speaker,
                1,
                prior_action,
                counts_toward_ask_user_limit=True,
            )
        final_action = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            agreed_deal={"terms": "tentative"},
            consultation_mode="final_decision",
            return_on_user_rejection=True,
            subsection_label="final confirmation",
        )

        self.assertIsNone(final_action.action)
        self.assertEqual(final_action.end_reason, "user_rejection")
        self.assertEqual(final_action.completed_exchanges, 1)
        self.assertEqual(agent.calls, 1)
        self.assertEqual(agent.user.last_mode, "final_decision")
        self.assertEqual(len(agent.dialogue_with_user.conversation_history), 62)
        self.assertEqual(count_ask_user_messages(agent.dialogue_with_user), 60)
        self.assertTrue(
            all(
                not turn["counts_toward_ask_user_limit"]
                for turn in agent.dialogue_with_user.conversation_history[-2:]
            )
        )
        self.assertEqual(
            agent.dialogue_with_user.conversation_history[-1]["action"].action_type,
            UserActionType.DONT_CONSENT,
        )

    def test_final_confirmation_rejects_non_decision_actions_at_runtime(self) -> None:
        class FakeLogger:
            def __init__(self):
                self.turns = []

            def subsection(self, msg, to):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action.action_type, to, invalid_reason))

        class FakeGame:
            state = {}

            def validate_deal(self, details):
                return details

            def check_for_same_proposed_deal(self, action_1, action_2):
                return action_1.action_details == action_2.action_details

        class FakeUser:
            user_id = "user_1"

            def __init__(self):
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                kwargs["attempt_budget"].consume()
                if self.calls == 1:
                    return UserAction(
                        UserActionType.FURTHER_INSTRUCT,
                        output="Please clarify before I decide.",
                    )
                return UserAction(UserActionType.CONSENT, output="I approve.")

        class FakeAgent:
            agent_id = "agent_1"
            current_reward = 100.0
            auxiliary_objectives = []
            payout_weight = 1.0

            def __init__(self):
                self.calls = 0
                self.user = FakeUser()
                self.dialogue_with_user = Dialogue("agent_1", "user_1")

            def respond_to_user(self, *args, **kwargs):
                self.calls += 1
                kwargs["attempt_budget"].consume()
                if self.calls == 1:
                    return AgentAction(AgentActionType.QUIT, output="End without a deal.")
                if self.calls == 2:
                    return AgentAction(
                        AgentActionType.PROPOSE,
                        output="Do you approve?",
                    )
                return AgentAction(
                    AgentActionType.REQUEST_USER_CONFIRMATION,
                    output="Do you approve the exact tentative deal?",
                )

        runner = RoundRunner.__new__(RoundRunner)
        runner.game = FakeGame()
        logger = FakeLogger()
        runner.logger = logger
        runner.current_turn = 1
        runner.max_agent_user_turns = 60
        runner.max_agent_user_exchanges = 5
        runner._last_invalid_error_agent_user = None

        agent = FakeAgent()
        final_action = runner._run_agent_user_until_consent(
            agent,
            Dialogue("agent_1", "agent_2"),
            agreed_deal={"terms": "tentative"},
            consultation_mode="final_decision",
            return_on_user_rejection=True,
            subsection_label="final confirmation",
        )

        self.assertEqual(
            final_action.action.action_type,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        )
        self.assertEqual(final_action.action.action_details, {"terms": "tentative"})
        self.assertEqual(final_action.end_reason, "user_consent")
        self.assertEqual(agent.calls, 3)
        self.assertEqual(agent.user.calls, 2)
        self.assertEqual(
            [turn["action"].action_type for turn in agent.dialogue_with_user.conversation_history],
            [
                AgentActionType.REQUEST_USER_CONFIRMATION,
                UserActionType.CONSENT,
            ],
        )
        invalid_logs = [turn for turn in logger.turns if turn[3]]
        self.assertEqual(
            [(speaker, action_type, to) for speaker, action_type, to, _ in invalid_logs],
            [
                ("agent_1", AgentActionType.QUIT, "agent_1"),
                ("agent_1", AgentActionType.PROPOSE, "agent_1"),
                ("user_1", UserActionType.FURTHER_INSTRUCT, "user_1"),
            ],
        )

    def test_agent_prompt_includes_turn_and_remaining_steps_without_llm(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []
                self.calls = 0

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                self.calls += 1
                if self.calls == 1:
                    return (
                        "THINKING: use an invalid action first\n"
                        "ACTION: MESSAGE_USER\n"
                        "OUTPUT: This should trigger a format retry."
                    )
                return (
                    "THINKING: make an offer\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose a 50-50 split."
                )

        class FakeGame:
            def __init__(self, agent, other_agent) -> None:
                self.state = {"turn": 3, "turns_total": 3}
                self.max_negotiation_turns = 7
                self.max_agent_user_turns = 3
                self.agent_1 = agent
                self.agent_2 = other_agent

            def get_proposal_format_for_prompt(self):
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            def get_response_format_instruction(self, details_fmt, actions_str):
                return None

            def get_private_info_for_prompt(self, user_id):
                return "Private context."

            def get_game_blurb_for_agent_prompt(self):
                return "Split assets."

            def get_game_state_for_prompt(self):
                return "Original game state."

            def get_proposal_format_description(self, user_id):
                return "Use terms."

            def parse_proposal_details(self, action_details):
                return json.loads(action_details) if isinstance(action_details, str) else action_details

            def validate_deal(self, action_details):
                return action_details

            def parse_raw_agent_response(self, raw, allow_quit=False, allow_ask_user=False, allow_accept=False):
                return None

            def show_accept_score_projection_for_prompt(self):
                return False

        class OtherAgent:
            agent_id = "agent_2"

        agent = Agent.__new__(Agent)
        agent.agent_id = "agent_1"
        agent.user = DummyYesUser({}, "user_1")
        agent.orchestrator_settings = {"max_negotiation_turns": 7}
        agent.visible_dialogue = Dialogue("agent_1", "user_1")
        agent.dialogue_with_user = agent.visible_dialogue.channel_view(
            "agent_user",
            "agent_1",
            "user_1",
        )
        agent.dialogue_with_user.add_turn(
            "user_1",
            1,
            UserAction(UserActionType.FURTHER_INSTRUCT, output="Fair split, but move quickly."),
            counts_toward_ask_user_limit=True,
        )
        agent.dialogue_with_user.add_turn(
            "agent_1",
            1,
                AgentAction(
                    AgentActionType.REQUEST_USER_CONFIRMATION,
                    output="Tentative final terms.",
                    action_details={"terms": "50-50"},
                ),
            counts_toward_ask_user_limit=False,
        )
        agent.dialogue_with_user.add_turn(
            "user_1",
            1,
            UserAction(UserActionType.CONSENT, output="I confirm."),
            counts_toward_ask_user_limit=False,
        )
        agent.config = {}
        agent.prompt_logger = None
        agent.show_own_thinking = False
        agent.use_structured_output = False
        agent.no_user = False
        agent.model = FakeModel()
        agent.prompts = {
            "respond_to_agent": {
                "system": (
                    "{response_format_instruction}\n"
                    "[CURRENT GAME STATE]\n{game_state}\n"
                    "{private_info}\n{game_blurb}\n{proposal_format}\n{reward_status}\n{turn_warning}"
                ),
                "current_prompt_parts": {
                    "user_dialogue": "{agent_user_history}",
                    "first_turn": "Open.",
                    "start_of_round": "Start.",
                    "continuation": "Continue.",
                    "continuation_after_invalid": "{invalid_error}",
                    "action_question": "Action ({actions})?",
                },
                "actions": ["PROPOSE", "ASK USER", "ACCEPT AND CONFIRM WITH USER"],
                "actions_first_turn": ["PROPOSE"],
            }
        }
        agent.auxiliary_objectives = []
        agent.current_reward = 100.0
        agent.payout_weight = 1.0

        game = FakeGame(agent, OtherAgent())
        negotiation = Dialogue("agent_1", "agent_2")
        negotiation.add_turn(
            "agent_2",
            1,
            AgentAction(
                AgentActionType.PROPOSE,
                output="I propose a 50-50 split.",
                action_details={"terms": "50-50"},
            ),
        )
        agent.respond_to_agent(negotiation, game=game)

        system_prompt = agent.model.prompts[0]["system_prompt"]
        current_prompt = agent.model.prompts[0]["current_prompt"]
        self.assertIn("Current public agent-agent action: 3 of 7.", system_prompt)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 4.",
            system_prompt,
        )
        self.assertIn("Private messages used during ASK USER by this pair: 1 of 3.", system_prompt)
        self.assertIn("Private ASK USER messages remaining for this pair across the game: 2.", system_prompt)
        self.assertIn("Original game state.", system_prompt)
        self.assertIn(
            "Action (PROPOSE or ACCEPT AND CONFIRM WITH USER)?",
            current_prompt,
        )
        self.assertNotIn("ASK USER", current_prompt)
        self.assertIn("ACCEPT AND CONFIRM WITH USER", current_prompt)
        target_marker = "[ACCEPT AND CONFIRM WITH USER TARGET]"
        self.assertIn(target_marker, current_prompt)
        self.assertTrue(current_prompt.endswith('{"terms": "50-50"}'))
        self.assertNotIn(target_marker, system_prompt)
        self.assertEqual(len(negotiation.conversation_history), 1)
        retry_prompt = agent.model.prompts[1]["current_prompt"]
        self.assertEqual(retry_prompt, current_prompt)

    def test_agent_prompt_context_placeholders_keep_system_prompt_stable(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                return (
                    "THINKING: make an offer\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose a 50-50 split."
                )

        class FakeGame:
            def __init__(self, agent, other_agent) -> None:
                self.state = {"turn": 3, "turns_total": 3}
                self.max_negotiation_turns = 7
                self.agent_1 = agent
                self.agent_2 = other_agent

            def get_proposal_format_for_prompt(self):
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            def get_response_format_instruction(self, details_fmt, actions_str):
                return None

            def get_private_info_for_prompt(self, user_id):
                return "Private context."

            def get_game_blurb_for_agent_prompt(self):
                return "Split assets."

            def get_game_state_for_prompt(self):
                return "Original game state."

            def get_proposal_format_description(self, user_id):
                return "Use terms."

            def parse_proposal_details(self, action_details):
                return json.loads(action_details) if isinstance(action_details, str) else action_details

            def validate_deal(self, action_details):
                return action_details

            def parse_raw_agent_response(self, raw, allow_quit=False, allow_ask_user=False, allow_accept=False):
                return None

            def show_accept_score_projection_for_prompt(self):
                return False

        class OtherAgent:
            agent_id = "agent_2"

        agent = Agent.__new__(Agent)
        agent.agent_id = "agent_1"
        agent.user = DummyYesUser({}, "user_1")
        agent.orchestrator_settings = {"max_negotiation_turns": 7}
        agent.visible_dialogue = Dialogue("agent_1", "user_1")
        agent.dialogue_with_user = agent.visible_dialogue.channel_view(
            "agent_user",
            "agent_1",
            "user_1",
        )
        agent.config = {}
        agent.prompt_logger = None
        agent.show_own_thinking = False
        agent.use_structured_output = False
        agent.no_user = False
        agent.model = FakeModel()
        agent.prompts = {
            "respond_to_agent": {
                "system": "{private_info}\n{game_blurb}\n{proposal_format}",
                "current_prompt_parts": {
                    "user_dialogue": "{agent_user_history}",
                    "first_turn": "Open.",
                    "start_of_round": "Start.",
                    "continuation": "Continue.",
                    "continuation_after_invalid": "{invalid_error}",
                    "action_question": (
                        "Action ({actions})?\n\n"
                        "[CURRENT GAME STATE]\n{game_state}\n\n"
                        "[REWARD]\n{reward_status}\n\n"
                        "[RESPONSE FORMAT]\n{response_format_instruction}"
                    ),
                },
                "actions": ["PROPOSE", "ASK USER", "ACCEPT AND CONFIRM WITH USER"],
                "actions_first_turn": ["PROPOSE"],
            }
        }
        agent.auxiliary_objectives = []
        agent.current_reward = 100.0
        agent.payout_weight = 1.0

        game = FakeGame(agent, OtherAgent())
        agent.respond_to_agent(Dialogue("agent_1", "agent_2"), game=game)

        system_prompt = agent.model.prompts[0]["system_prompt"]
        current_prompt = agent.model.prompts[0]["current_prompt"]
        self.assertNotIn("Current public agent-agent action: 3 of 7.", system_prompt)
        self.assertNotIn("Original game state.", system_prompt)
        self.assertIn("[CURRENT GAME STATE]", current_prompt)
        self.assertIn("[REWARD]", current_prompt)
        self.assertIn("[RESPONSE FORMAT]", current_prompt)
        self.assertIn("Current public agent-agent action: 3 of 7.", current_prompt)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 4.",
            current_prompt,
        )
        self.assertIn("Original game state.", current_prompt)

    def test_agent_user_prompt_interleaves_complete_public_history(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                return (
                    "THINKING: enough guidance\n"
                    "ACTION: MESSAGE_USER_AND_CONTINUE_NEGOTIATION\n"
                    "OUTPUT: Thanks, I have enough guidance and will continue negotiating."
                )

        class FakeGame:
            state = {"turn": 4, "turns_total": 4}
            max_negotiation_turns = 10

            def get_proposal_format_for_prompt(self):
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            def get_user_consultation_response_format_instruction(self, details_fmt, actions_str):
                return None

            def get_private_info_for_prompt(self, user_id):
                return "Private context."

            def get_game_blurb_for_agent_prompt(self):
                return "Split assets."

            def get_game_state_for_prompt(self):
                return "Original consultation state."

            def get_proposal_format_description(self, user_id):
                return "Use terms."

            def parse_proposal_details(self, action_details):
                return json.loads(action_details) if isinstance(action_details, str) else action_details

        agent = Agent.__new__(Agent)
        agent.agent_id = "agent_1"
        agent.user = DummyYesUser({}, "user_1")
        agent.orchestrator_settings = {"max_negotiation_turns": 10}
        agent.visible_dialogue = Dialogue("agent_1", "user_1")
        agent.dialogue_with_user = agent.visible_dialogue.channel_view(
            "agent_user", "agent_1", "user_1"
        )
        agent.config = {}
        agent.prompt_logger = None
        agent.show_own_thinking = False
        agent.require_thinking = True
        agent.use_structured_output = False
        agent.no_user = False
        agent.model = FakeModel()
        agent.prompts = {
            "respond_to_user": {
                "system": (
                    "{response_format_instruction}\n"
                    "[RECENT NEGOTIATION]\n{agent_agent_history}\n"
                    "[CURRENT GAME STATE]\n{game_state}\n"
                    "{private_info}\n{game_blurb}\n{proposal_format}\n{reward_status}"
                ),
                "current_prompt": "Reply ({actions}).",
                "current_prompt_guidance": "Guidance reply ({actions}).",
                "current_prompt_with_summary": "Reply after negotiating ({actions}).",
                "current_prompt_after_invalid": "{invalid_error}",
                "actions": ["PROPOSE"],
                "actions_guidance": [
                    "MESSAGE_USER",
                    "MESSAGE_USER_AND_CONTINUE_NEGOTIATION",
                ],
            }
        }
        agent.auxiliary_objectives = []
        agent.current_reward = 100.0
        agent.payout_weight = 1.0

        game_dialogue = Dialogue("agent_1", "agent_2")
        for turn in range(8):
            speaker = "agent_1" if turn % 2 == 0 else "agent_2"
            public_action = AgentAction(
                AgentActionType.PROPOSE,
                output=f"history turn {turn}",
                action_details={"terms": f"turn-{turn}"},
            )
            game_dialogue.add_turn(
                speaker,
                1,
                public_action,
                turn_num=turn,
            )
            agent.visible_dialogue.add_turn(
                speaker,
                1,
                public_action,
                turn_num=turn,
                channel="agent_agent",
            )
            if turn == 3:
                ask_action = AgentAction(
                    AgentActionType.ASK_USER,
                    output="ASK output sentinel: compare the live offer.",
                )
                agent.visible_dialogue.add_turn(
                    "agent_1",
                    1,
                    ask_action,
                    turn_num=turn,
                    channel="agent_agent",
                )
                agent.dialogue_with_user.add_turn(
                    "agent_1",
                    1,
                    AgentAction(
                        AgentActionType.MESSAGE_USER,
                        output="Private message.",
                    ),
                    turn_num=turn,
                )
                agent.dialogue_with_user.add_turn(
                    "user_1",
                    1,
                    UserAction(
                        UserActionType.FURTHER_INSTRUCT,
                        output="Private guidance.",
                    ),
                    turn_num=turn,
                )
                agent.visible_dialogue.add_turn(
                    "agent_1",
                    1,
                    AgentAction(
                        AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
                        output="Return to negotiation.",
                    ),
                    turn_num=turn,
                    channel="agent_user",
                    visible_to_user=True,
                )

        action = agent.respond_to_user(
            game=FakeGame(),
            game_dialogue=game_dialogue,
            consultation_mode="guidance",
        )

        self.assertEqual(
            action.action_type,
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
        )
        self.assertIn("continue negotiating", action.output)
        self.assertEqual(len(agent.model.prompts), 1)
        system_prompt = agent.model.prompts[0]["system_prompt"]
        self.assertIn("Current public agent-agent action: 4 of 10.", system_prompt)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 6.",
            system_prompt,
        )
        self.assertIn("Original consultation state.", system_prompt)
        self.assertNotIn("history turn", system_prompt)
        self.assertNotIn("ASK output sentinel", system_prompt)
        history_text = "\n".join(
            message["content"]
            for message in agent.model.prompts[0]["conversation_history"]
        )
        for turn in range(8):
            self.assertIn(f"history turn {turn}", history_text)
        self.assertIn("ASK output sentinel: compare the live offer.", history_text)
        self.assertIn("Private message.", history_text)
        self.assertIn("Private guidance.", history_text)
        self.assertIn("Return to negotiation.", history_text)
        self.assertGreaterEqual(
            history_text.count("=== BEGIN AGENT-AGENT DIALOGUE ==="),
            2,
        )
        self.assertIn("=== AGENT-USER DIALOGUE ===", history_text)
        ordered = [
            history_text.index("history turn 3"),
            history_text.index("ASK output sentinel"),
            history_text.index("Private message."),
            history_text.index("Private guidance."),
            history_text.index("Return to negotiation."),
            history_text.index("history turn 4"),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertNotIn("event_sequence", history_text)
        current_prompt = agent.model.prompts[0]["current_prompt"]
        self.assertNotIn("history turn", current_prompt)
        self.assertNotIn("ASK output sentinel", current_prompt)
        self.assertTrue(
            all(
                "ASK output sentinel" not in turn["action"].output
                for turn in game_dialogue.conversation_history
            )
        )
        self.assertIn(
            AgentActionType.MESSAGE_USER_AND_CONTINUE_NEGOTIATION,
            [
                turn["action"].action_type
                for turn in agent.dialogue_with_user.conversation_history
            ],
        )

    def test_agent_final_confirmation_uses_system_supplied_request(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []
                self.responses = [
                    (
                        "THINKING: I want to clarify first.\n"
                        "ACTION: MESSAGE_USER\n"
                        "OUTPUT: Can we discuss this?"
                    ),
                    (
                        "THINKING: I must present the exact deal.\n"
                        "ACTION: REQUEST_USER_CONFIRMATION\n"
                        "OUTPUT: Do you approve these exact terms?"
                    ),
                ]

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                return self.responses.pop(0)

        class FakeGame:
            state = {"turn": 4}
            max_negotiation_turns = 10

            def get_proposal_format_for_prompt(self):
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            def get_user_consultation_response_format_instruction(self, details_fmt, actions_str):
                return None

            def get_private_info_for_prompt(self, user_id):
                return "Private context."

            def get_game_blurb_for_agent_prompt(self):
                return "Negotiate exact terms."

            def get_game_state_for_prompt(self):
                return "Current state."

            def get_proposal_format_description(self, user_id):
                return "Use terms."

            def parse_proposal_details(self, action_details):
                return json.loads(action_details)

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
        agent.config = {
            "allowed_actions": {
                "respond_to_user": {
                    "final_decision": ["MESSAGE_USER", "PROPOSE", "QUIT"],
                }
            }
        }
        agent.prompt_logger = None
        agent.show_own_thinking = False
        agent.require_thinking = True
        agent.use_structured_output = False
        agent.no_user = False
        agent.model = FakeModel()
        agent.prompts = {
            "respond_to_user": {
                "system": "Allowed response:\n{response_format_instruction}",
                "current_prompt": "Reply ({actions}).",
                "current_prompt_with_agreed_deal_final_decision": (
                    "Confirm {agreed_deal} using ({actions})."
                ),
                "current_prompt_after_invalid": "{invalid_error}",
                "actions": ["MESSAGE_USER", "PROPOSE", "QUIT"],
                "actions_final_decision": ["MESSAGE_USER", "PROPOSE", "QUIT"],
            }
        }
        agent.auxiliary_objectives = []
        agent.current_reward = 100.0
        agent.payout_weight = 1.0

        game_dialogue = Dialogue("agent_1", "agent_2")
        game_dialogue.add_turn(
            "agent_2",
            1,
            AgentAction(
                AgentActionType.PROPOSE,
                output="Tentative deal.",
                action_details={"terms": "tentative"},
            ),
        )
        action = agent.respond_to_user(
            game=FakeGame(),
            game_dialogue=game_dialogue,
            agreed_deal={"terms": "tentative"},
            consultation_mode="final_decision",
        )

        self.assertEqual(
            action.action_type,
            AgentActionType.REQUEST_USER_CONFIRMATION,
        )
        self.assertEqual(action.action_details, {"terms": "tentative"})
        self.assertEqual(len(agent.model.prompts), 2)
        first_prompt = agent.model.prompts[0]
        self.assertIn(
            "using (REQUEST_USER_CONFIRMATION)",
            first_prompt["current_prompt"],
        )
        self.assertNotIn("MESSAGE_USER", first_prompt["system_prompt"])
        self.assertNotIn("QUIT", first_prompt["system_prompt"])
        self.assertNotIn("PROPOSAL_DETAILS", first_prompt["current_prompt"])
        self.assertNotIn("proposal_details", first_prompt["current_prompt"])
        self.assertNotIn(
            'PROPOSAL_DETAILS: {"terms":"..."}',
            first_prompt["system_prompt"],
        )
        self.assertNotIn("proposal_details", first_prompt["system_prompt"])
        retry_prompt = agent.model.prompts[1]["current_prompt"]
        self.assertIn("[FINAL USER CONFIRMATION]", retry_prompt)
        self.assertEqual(
            retry_prompt.count("[FINAL USER CONFIRMATION]"),
            1,
        )
        self.assertIn('{"terms": "tentative"}', retry_prompt)
        self.assertNotIn("PROPOSAL_DETAILS", retry_prompt)
        self.assertNotIn("proposal_details", retry_prompt)

    def test_user_final_confirmation_allows_only_binary_decision(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []
                self.responses = [
                    (
                        "THINKING: I want different terms.\n"
                        "ACTION: FURTHER_INSTRUCT\n"
                        "OUTPUT: Please keep negotiating."
                    ),
                    (
                        "THINKING: I reject these exact terms.\n"
                        "ACTION: DONT_CONSENT\n"
                        "OUTPUT: Please negotiate different terms."
                    ),
                ]

            def query_model(self, prompt):
                self.prompts.append(dict(prompt))
                return self.responses.pop(0)

        class FakeGame:
            state = {"turn": 4}
            max_negotiation_turns = 10

            def get_game_blurb_for_user_prompt(self):
                return "Decide whether to approve."

            def get_public_info_for_user_prompt(self, user_id):
                return "Public context."

            def get_private_info_for_user_prompt(self, user_id):
                return "Private preferences."

            def get_game_state_for_prompt(self):
                return "Current state."

        user = LLMUser.__new__(LLMUser)
        user.config = {
            "allowed_actions": {
                "respond_to_agent": {
                    "final_decision": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
                }
            }
        }
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.require_thinking = True
        user.model = FakeModel()
        user.prompts = {
            "respond_to_agent": {
                "system": "Allowed actions: {actions}",
                "current_prompt": "Choose ({actions}).",
                "current_prompt_final_decision": "Decide using ({actions}).",
                "actions": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
                "actions_final_decision": [
                    "CONSENT",
                    "FURTHER_INSTRUCT",
                    "DONT_CONSENT",
                ],
            }
        }
        dialogue = Dialogue("agent_1", "user_1")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.REQUEST_USER_CONFIRMATION,
                output="Do you approve?",
                action_details={"terms": "tentative"},
            ),
        )

        action = user.respond_to_agent(
            dialogue_with_agent=dialogue,
            agent_id="agent_1",
            game_dialogue=Dialogue("agent_1", "agent_2"),
            game=FakeGame(),
            consultation_mode="final_decision",
        )

        self.assertEqual(action.action_type, UserActionType.DONT_CONSENT)
        self.assertEqual(len(user.model.prompts), 2)
        first_prompt = user.model.prompts[0]
        self.assertIn("CONSENT or DONT_CONSENT", first_prompt["system_prompt"])
        self.assertNotIn("FURTHER_INSTRUCT", first_prompt["system_prompt"])
        confirmation_message = first_prompt["conversation_history"][-1]["content"]
        self.assertIn("SYSTEM-SUPPLIED CONFIRMATION PROPOSAL", confirmation_message)
        self.assertIn('{"terms": "tentative"}', confirmation_message)
        retry_prompt = user.model.prompts[1]["current_prompt"]
        self.assertIn("[FINAL USER CONFIRMATION]", retry_prompt)
        self.assertEqual(
            retry_prompt.count("[FINAL USER CONFIRMATION]"),
            1,
        )
        self.assertIn("DONT_CONSENT rejects those exact terms", retry_prompt)
        self.assertIn(
            "leaves any remaining turns available for negotiation",
            retry_prompt,
        )
        self.assertNotIn("If you need clarification", retry_prompt)
        self.assertNotIn("use DONT_CONSENT", retry_prompt)

    def test_user_prompt_includes_turn_and_remaining_steps_without_llm(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []

            def query_model(self, prompt):
                self.prompts.append(dict(prompt))
                return (
                    "THINKING: give guidance\n"
                    "ACTION: FURTHER_INSTRUCT\n"
                    "OUTPUT: Keep trying for a fairer deal."
                )

        class FakeGame:
            state = {"turn": 2, "turns_total": 2}
            max_negotiation_turns = 9

            def get_game_blurb_for_user_prompt(self):
                return "Negotiate a deal."

            def get_public_info_for_user_prompt(self, user_id):
                return "Public context."

            def get_private_info_for_user_prompt(self, user_id):
                return "Private preferences."

            def get_game_state_for_prompt(self):
                return "Original user-visible state."

        user = LLMUser.__new__(LLMUser)
        user.config = {}
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.model = FakeModel()
        user.prompts = {
            "respond_to_agent": {
                "system": (
                    "{consultation_mode}\n"
                    "[CURRENT GAME STATE]\n{game_state}\n"
                    "{public_info}\n{private_info}\n{game_blurb}\n{actions}"
                ),
                "current_prompt": "Reply ({actions}).",
                "current_prompt_guidance": "Guidance reply ({actions}).",
                "actions": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
                "actions_guidance": ["FURTHER_INSTRUCT"],
            }
        }
        dialogue = Dialogue("agent_1", "user_1")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(AgentActionType.MESSAGE_USER, output="What should I do?"),
        )

        action = user.respond_to_agent(
            dialogue_with_agent=dialogue,
            agent_id="agent_1",
            game_dialogue=Dialogue("agent_1", "agent_2"),
            game=FakeGame(),
            consultation_mode="guidance",
        )

        self.assertEqual(action.action_type, UserActionType.FURTHER_INSTRUCT)
        system_prompt = user.model.prompts[0]["system_prompt"]
        self.assertIn("Current public agent-agent action: 2 of 9.", system_prompt)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 7.",
            system_prompt,
        )
        self.assertIn("Original user-visible state.", system_prompt)

    def test_user_prompt_context_placeholders_keep_system_prompt_stable(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []

            def query_model(self, prompt):
                self.prompts.append(dict(prompt))
                return (
                    "THINKING: give guidance\n"
                    "ACTION: FURTHER_INSTRUCT\n"
                    "OUTPUT: Keep trying for a fairer deal."
                )

        class FakeGame:
            state = {"turn": 2, "turns_total": 2}
            max_negotiation_turns = 9

            def get_game_blurb_for_user_prompt(self):
                return "Negotiate a deal."

            def get_public_info_for_user_prompt(self, user_id):
                return "Public context."

            def get_private_info_for_user_prompt(self, user_id):
                return "Private preferences."

            def get_game_state_for_prompt(self):
                return "Original user-visible state."

        user = LLMUser.__new__(LLMUser)
        user.config = {}
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.model = FakeModel()
        user.prompts = {
            "respond_to_agent": {
                "system": "{public_info}\n{private_info}\n{game_blurb}",
                "current_prompt": (
                    "Reply ({actions}).\n\n"
                    "[CONSULTATION MODE]\n{consultation_mode}\n\n"
                    "[CURRENT GAME STATE]\n{game_state}\n\n"
                    "[AGENT REWARD STATUS]\n{agent_reward_status}"
                ),
                "current_prompt_guidance": (
                    "Guidance reply ({actions}).\n\n"
                    "[CONSULTATION MODE]\n{consultation_mode}\n\n"
                    "[CURRENT GAME STATE]\n{game_state}\n\n"
                    "[AGENT REWARD STATUS]\n{agent_reward_status}"
                ),
                "actions": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
                "actions_guidance": ["FURTHER_INSTRUCT"],
            }
        }
        dialogue = Dialogue("agent_1", "user_1")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(AgentActionType.MESSAGE_USER, output="What should I do?"),
        )

        action = user.respond_to_agent(
            dialogue_with_agent=dialogue,
            agent_id="agent_1",
            game_dialogue=Dialogue("agent_1", "agent_2"),
            game=FakeGame(),
            agent_reward_status="Reward is 97.",
            consultation_mode="guidance",
        )

        self.assertEqual(action.action_type, UserActionType.FURTHER_INSTRUCT)
        system_prompt = user.model.prompts[0]["system_prompt"]
        current_prompt = user.model.prompts[0]["current_prompt"]
        self.assertNotIn("Current public agent-agent action: 2 of 9.", system_prompt)
        self.assertNotIn("Original user-visible state.", system_prompt)
        self.assertIn("[CURRENT GAME STATE]", current_prompt)
        self.assertIn("Current public agent-agent action: 2 of 9.", current_prompt)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 7.",
            current_prompt,
        )
        self.assertIn("Original user-visible state.", current_prompt)
        self.assertIn("Reward is 97.", current_prompt)

    def test_user_retry_prompt_includes_previous_invalid_response_verbatim(self) -> None:
        invalid_user_response = "ACTION CONSENT\nOUTPUT: Yes, I approve."
        user_retry = _format_user_retry_prompt(
            ["CONSENT", "DONT_CONSENT"],
            "Missing user ACTION",
            invalid_user_response,
        )
        self.assertIn(invalid_user_response, user_retry)

    def test_agent_parser_reads_normalized_inline_action(self) -> None:
        raw = (
            "THINKING: preserve this proposal ACTION: PROPOSE\n"
            'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
            "OUTPUT: I propose a 50-50 split."
        )
        normalized = normalize_newline_only_labeled_response(
            raw,
            require_thinking=True,
        )
        self.assertIsNotNone(normalized)
        output, action_type, details, thinking = _parse_agent_response(
            normalized,
            allowed_actions=["PROPOSE"],
            require_thinking=True,
        )
        self.assertEqual(action_type, AgentActionType.PROPOSE)
        self.assertEqual(json.loads(details), {"terms": "50-50"})
        self.assertEqual(output, "I propose a 50-50 split.")
        self.assertEqual(thinking, "preserve this proposal")

    def test_agent_parser_rejects_duplicate_actions(self) -> None:
        repeated = (
            "THINKING: one decision, duplicated label.\n"
            "ACTION: PROPOSE\n"
            "ACTION: PROPOSE\n"
            'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
            "OUTPUT: Same decision."
        )
        with self.assertRaisesRegex(ValueError, "Multiple ACTION sections"):
            _parse_agent_response(
                repeated,
                allowed_actions=["PROPOSE", "ACCEPT"],
                require_thinking=True,
            )

        structured_repeated = (
            '{"thinking":"x","action":"PROPOSE","action":"PROPOSE",'
            '"proposal_details":"{\\"terms\\":\\"50-50\\"}","output":"offer"}'
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field 'action'"):
            _parse_agent_response_json(
                structured_repeated,
                allowed_actions=["PROPOSE", "ACCEPT"],
            )

    def test_agent_parser_ignores_action_words_in_prose(self) -> None:
        raw = (
            "THINKING: I considered the phrase ACTION: PROPOSE later, but chose a report.\n"
            "ACTION: MESSAGE_USER\n"
            "OUTPUT: Here is the update.\n"
            "Action items:\n"
            "- wait for the other side"
        )
        output, action_type, _, thinking = _parse_agent_response(
            raw,
            allowed_actions=["MESSAGE_USER", "PROPOSE"],
            require_thinking=True,
        )
        self.assertEqual(action_type, AgentActionType.MESSAGE_USER)
        self.assertIn("Action items", output)
        self.assertIn("ACTION: PROPOSE later", thinking)

    def test_user_format_repair_locks_declared_action(self) -> None:
        raw = "THINKING: continue guidance\nACTION: FURTHER_INSTRUCT"
        repair_lock, protocol_correction, normalized = _prepare_user_repair(
            raw,
            ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
        )
        self.assertEqual(repair_lock.action_label, "FURTHER_INSTRUCT")
        self.assertFalse(protocol_correction)
        self.assertEqual(normalized, raw)

        retry = _format_user_retry_prompt(
            [repair_lock.action_label],
            "Missing user OUTPUT",
            raw,
            require_thinking=True,
            locked_action=repair_lock.action_label,
            locked_output=repair_lock.output,
        )
        self.assertIn("ACTION is locked to exactly FURTHER_INSTRUCT", retry)
        self.assertNotIn("ACTION: CONSENT or", retry)

        with_output = (
            "ACTION: FURTHER_INSTRUCT\n"
            "OUTPUT: Preserve this exact user instruction."
        )
        output_lock, _, _ = _prepare_user_repair(
            with_output,
            ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
        )
        self.assertEqual(
            output_lock.output,
            "Preserve this exact user instruction.",
        )
        output_retry = _format_user_retry_prompt(
            [output_lock.action_label],
            "Missing user THINKING",
            with_output,
            require_thinking=True,
            locked_action=output_lock.action_label,
            locked_output=output_lock.output,
        )
        self.assertIn(
            'OUTPUT is locked to this exact text: "Preserve this exact user instruction."',
            output_retry,
        )

    def test_user_resample_can_change_action_without_prompt_feedback(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prompts = []
                self.responses = [
                    "THINKING: continue guidance\nACTION: FURTHER_INSTRUCT",
                    (
                        "THINKING: switch decisions\n"
                        "ACTION: CONSENT\n"
                        "OUTPUT: I attempted consent."
                    ),
                    (
                        "THINKING: repair only\n"
                        "ACTION: FURTHER_INSTRUCT\n"
                        "OUTPUT: Keep negotiating."
                    ),
                ]

            def query_model(self, prompt):
                self.prompts.append(dict(prompt))
                return self.responses.pop(0)

        class FakeGame:
            state = {"turn": 2}
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

        user = LLMUser.__new__(LLMUser)
        user.config = {}
        user.initial_offer = ""
        user.user_id = "user_1"
        user.private_info = {}
        user.prompt_logger = None
        user.require_thinking = True
        user.model = FakeModel()
        user.prompts = {
            "respond_to_agent": {
                "system": (
                    "{public_info}\n{private_info}\n{game_blurb}\n{game_state}\n"
                    "{agent_agent_history}\n{agent_reward_status}\n{actions}"
                ),
                "current_prompt": "Original user decision prompt: {actions}.",
                "actions": ["CONSENT", "FURTHER_INSTRUCT", "DONT_CONSENT"],
            }
        }

        action = user.respond_to_agent(
            dialogue_with_agent=Dialogue("agent_1", "user_1"),
            agent_id="agent_1",
            game_dialogue=Dialogue("agent_1", "agent_2"),
            game=FakeGame(),
        )

        self.assertEqual(action.action_type, UserActionType.CONSENT)
        self.assertEqual(action.output, "I attempted consent.")
        self.assertEqual(len(user.model.prompts), 2)
        for retry_prompt in user.model.prompts[1:]:
            self.assertEqual(
                retry_prompt["current_prompt"],
                user.model.prompts[0]["current_prompt"],
            )
            self.assertEqual(
                retry_prompt["conversation_history"],
                user.model.prompts[0]["conversation_history"],
            )

    def test_agent_resample_keeps_provider_prompt_but_not_semantic_locks(self) -> None:
        class FakeModel:
            def __init__(self, responses) -> None:
                self.prompts = []
                self.responses = list(responses)

            def query_model(self, prompt, response_format=None):
                self.prompts.append(dict(prompt))
                return self.responses.pop(0)

        class OtherAgent:
            agent_id = "agent_2"

        class FakeGame:
            def __init__(self, agent) -> None:
                self.state = {"turn": 2}
                self.max_negotiation_turns = 10
                self.agent_1 = agent
                self.agent_2 = OtherAgent()

            @staticmethod
            def get_proposal_format_for_prompt():
                return 'PROPOSAL_DETAILS: {"terms":"..."}'

            @staticmethod
            def get_response_format_instruction(details_fmt, actions_str):
                return None

            @staticmethod
            def get_private_info_for_prompt(user_id):
                return "Private context."

            @staticmethod
            def get_game_blurb_for_agent_prompt():
                return "Split assets."

            @staticmethod
            def get_game_state_for_prompt():
                return "Original state."

            @staticmethod
            def get_proposal_format_description(user_id):
                return "Use terms."

            @staticmethod
            def parse_proposal_details(details):
                return json.loads(details) if isinstance(details, str) else details

            @staticmethod
            def validate_deal(details):
                return details

            @staticmethod
            def parse_raw_agent_response(raw, **kwargs):
                return None

            @staticmethod
            def show_accept_score_projection_for_prompt():
                return False

        def run_agent(responses):
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
            agent.model = FakeModel(responses)
            agent.prompts = {
                "respond_to_agent": {
                    "system": (
                        "{response_format_instruction}\n{private_info}\n{game_blurb}\n"
                        "{game_state}\n{proposal_format}\n{reward_status}\n{turn_warning}"
                    ),
                    "current_prompt_parts": {
                        "first_turn": "Open.",
                        "continuation": "Original decision prompt.",
                        "action_question": "Choose {actions}.",
                    },
                    "actions": ["PROPOSE", "ACCEPT"],
                }
            }
            agent.auxiliary_objectives = []
            agent.current_reward = 100.0
            agent.payout_weight = 1.0

            game = FakeGame(agent)
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
            action = agent.respond_to_agent(dialogue, game)
            return agent, action

        agent, action = run_agent(
            [
                (
                    "THINKING: keep the 50-50 proposal\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}'
                ),
                (
                    "THINKING: try changing the terms\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"60-40"}\n'
                    "OUTPUT: I propose 60-40."
                ),
                (
                    "THINKING: repair format only\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose 50-50."
                ),
            ]
        )

        self.assertEqual(action.action_type, AgentActionType.PROPOSE)
        self.assertEqual(action.action_details, {"terms": "60-40"})
        self.assertEqual(len(agent.model.prompts), 2)
        for retry_prompt in agent.model.prompts[1:]:
            self.assertEqual(
                retry_prompt["current_prompt"],
                agent.model.prompts[0]["current_prompt"],
            )
            self.assertEqual(
                retry_prompt["conversation_history"],
                agent.model.prompts[0]["conversation_history"],
            )

        drift_agent, drift_action = run_agent(
            [
                (
                    "THINKING: keep proposing\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}'
                ),
                (
                    "THINKING: switch the action\n"
                    "ACTION: ACCEPT\n"
                    "OUTPUT: Accept instead."
                ),
                (
                    "THINKING: preserve the action\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose 50-50."
                ),
            ]
        )
        self.assertEqual(drift_action.action_type, AgentActionType.ACCEPT)
        self.assertEqual(len(drift_agent.model.prompts), 2)

        recovered_agent, recovered_action = run_agent(
            [
                (
                    "THINKING: keep the exact offer ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: I propose 50-50."
                )
            ]
        )
        self.assertEqual(recovered_action.action_type, AgentActionType.PROPOSE)
        self.assertEqual(recovered_action.action_details, {"terms": "50-50"})
        self.assertEqual(len(recovered_agent.model.prompts), 1)

        output_agent, output_action = run_agent(
            [
                (
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: Original participant-facing promise."
                ),
                (
                    "THINKING: paraphrase it\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: Changed participant-facing promise."
                ),
                (
                    "THINKING: preserve it exactly\n"
                    "ACTION: PROPOSE\n"
                    'PROPOSAL_DETAILS: {"terms":"50-50"}\n'
                    "OUTPUT: Original participant-facing promise."
                ),
            ]
        )
        self.assertEqual(
            output_action.output,
            "Changed participant-facing promise.",
        )
        self.assertEqual(len(output_agent.model.prompts), 2)
        self.assertEqual(
            output_agent.model.prompts[1]["current_prompt"],
            output_agent.model.prompts[0]["current_prompt"],
        )

    def test_negotiation_continuation_uses_yaml_user_instruction(self) -> None:
        yaml_instruction = "YAML instruction: use ASK USER when guidance is incomplete."
        instruction = _select_negotiation_instruction(
            {
                "first_turn": "Open.",
                "continuation": yaml_instruction,
                "continuation_after_invalid": "Invalid: {invalid_error}",
            },
            has_agent_history=True,
            consulted_this_turn=False,
            consultation_state=None,
            current_prompt_fields={},
        )

        self.assertEqual(instruction, yaml_instruction)

    def test_first_negotiation_turn_uses_first_turn_actions(self) -> None:
        configured_actions = [
            "PROPOSE",
            "ASK USER",
            "ACCEPT AND CONFIRM WITH USER",
        ]

        actions = _negotiation_actions_for_turn(
            configured_actions,
            consulted_this_turn=False,
            has_agent_history=False,
            first_turn_actions=["PROPOSE"],
        )

        self.assertEqual(actions, ["PROPOSE"])
        self.assertEqual(
            configured_actions,
            ["PROPOSE", "ASK USER", "ACCEPT AND CONFIRM WITH USER"],
        )

    def test_public_actions_match_runtime_state_before_query(self) -> None:
        with_user = ["PROPOSE", "ASK USER", "ACCEPT AND CONFIRM WITH USER"]
        self.assertEqual(
            _negotiation_actions_for_turn(
                with_user,
                consulted_this_turn=False,
                has_confirmable_proposal=False,
            ),
            ["PROPOSE", "ASK USER"],
        )
        self.assertEqual(
            _negotiation_actions_for_turn(
                with_user,
                consulted_this_turn=True,
                has_confirmable_proposal=True,
            ),
            ["PROPOSE", "ACCEPT AND CONFIRM WITH USER"],
        )
        self.assertEqual(
            _negotiation_actions_for_turn(
                ["PROPOSE", "ACCEPT"],
                consulted_this_turn=False,
                has_confirmable_proposal=False,
                no_user=True,
            ),
            ["PROPOSE"],
        )
        self.assertEqual(
            _negotiation_actions_for_turn(
                ["PROPOSE", "ACCEPT"],
                consulted_this_turn=False,
                has_confirmable_proposal=True,
                no_user=True,
            ),
            ["PROPOSE", "ACCEPT"],
        )

    def test_post_consultation_has_its_own_instruction_and_removes_ask_user(self) -> None:
        configured_actions = [
            "PROPOSE",
            "ASK USER",
            "ACCEPT AND CONFIRM WITH USER",
        ]
        actions = _negotiation_actions_for_turn(
            configured_actions,
            consulted_this_turn=True,
        )
        instruction = _select_negotiation_instruction(
            {
                "first_turn": "Open.",
                "continuation": "Continue.",
                "continuation_after_invalid": "Proposal rejected: {invalid_error}",
                "continuation_after_user_consultation": (
                    "YAML instruction after user consultation."
                ),
            },
            has_agent_history=True,
            consulted_this_turn=True,
            consultation_state="user_consultation",
            current_prompt_fields={},
        )

        self.assertEqual(
            actions,
            ["PROPOSE", "ACCEPT AND CONFIRM WITH USER"],
        )
        self.assertEqual(
            configured_actions,
            ["PROPOSE", "ASK USER", "ACCEPT AND CONFIRM WITH USER"],
        )
        self.assertEqual(instruction, "YAML instruction after user consultation.")

    def test_two_remaining_private_messages_remove_ask_user(self) -> None:
        configured_actions = [
            "PROPOSE",
            "ASK USER",
            "ACCEPT AND CONFIRM WITH USER",
        ]

        with_three_remaining = _negotiation_actions_for_turn(
            configured_actions,
            consulted_this_turn=False,
            remaining_ask_user_messages=3,
        )
        with_two_remaining = _negotiation_actions_for_turn(
            configured_actions,
            consulted_this_turn=False,
            remaining_ask_user_messages=2,
        )
        with_no_messages_remaining = _negotiation_actions_for_turn(
            configured_actions,
            consulted_this_turn=False,
            remaining_ask_user_messages=0,
        )

        self.assertEqual(with_three_remaining, configured_actions)
        self.assertEqual(
            with_two_remaining,
            ["PROPOSE", "ACCEPT AND CONFIRM WITH USER"],
        )
        self.assertEqual(with_no_messages_remaining, with_two_remaining)

    def test_auto_ended_consultation_uses_safety_limit_instruction(self) -> None:
        instruction = _select_negotiation_instruction(
            {
                "first_turn": "Open.",
                "continuation": "Continue.",
                "continuation_after_user_consultation": "Normal return.",
                "continuation_after_user_consultation_auto_end": (
                    "The safety limit ended the consultation."
                ),
            },
            has_agent_history=True,
            consulted_this_turn=True,
            consultation_state="user_consultation_auto_ended",
            current_prompt_fields={},
        )

        self.assertEqual(
            instruction,
            "The safety limit ended the consultation.",
        )

    def test_orchestrator_returns_from_ask_user_without_invalid_retry(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.subsections = []

            def subsection(self, message, to="both"):
                self.subsections.append((message, to))

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class FakeAgent:
            agent_id = "agent_1"
            no_user = False
            config = {}
            orchestrator_settings = {}
            prompts = {}
            user = SimpleNamespace(user_id="user_1")

            def __init__(self) -> None:
                self.calls = []
                self.visible_dialogue = Dialogue("agent_1", "user_1")
                self.dialogue_with_user = self.visible_dialogue.channel_view(
                    "agent_user",
                    "agent_1",
                    "user_1",
                )

            def respond_to_agent(
                self,
                dialogue,
                game,
                *,
                consulted_this_turn=False,
                consultation_state=None,
                attempt_budget=None,
            ):
                self.calls.append(
                    {
                        "consulted_this_turn": consulted_this_turn,
                        "consultation_state": consultation_state,
                        "decision_cycle_id": attempt_budget.decision_cycle_id,
                        "attempts_before_call": attempt_budget.used_attempts,
                    }
                )
                if len(self.calls) == 1:
                    return AgentAction(AgentActionType.ASK_USER, output="Let me check.")
                return AgentAction(AgentActionType.QUIT, output="Stop after the check.")

        agent = FakeAgent()
        other_visible_dialogue = Dialogue("agent_2", "user_2")
        other_agent = SimpleNamespace(
            agent_id="agent_2",
            user=SimpleNamespace(user_id="user_2"),
            no_user=False,
            visible_dialogue=other_visible_dialogue,
            dialogue_with_user=other_visible_dialogue.channel_view(
                "agent_user",
                "agent_2",
                "user_2",
            ),
        )
        game = SimpleNamespace(
            agent_1=agent,
            agent_2=other_agent,
            game_dialogue=Dialogue("agent_1", "agent_2"),
            state={},
            end_reason=None,
            quitter_user_id=None,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = game
        runner.config = {}
        runner.logger = FakeLogger()
        runner.current_turn = 0
        runner.max_negotiation_turns = 5
        runner.max_agent_user_turns = 60
        runner.turns_by_agent = {"agent_1": 0, "agent_2": 0}
        runner._last_invalid_error = None
        runner._last_invalid_error_agent_user = None
        runner._update_turn_state = lambda next_speaker=None: None
        consultation_calls = []

        def run_consultation(*args, **kwargs):
            consultation_calls.append(kwargs)
            return ConsultationResult(
                action=None,
                end_reason="agent_continue",
                completed_exchanges=1,
                max_consecutive_invalid_attempts=0,
                origin="ask_user",
            )

        runner._run_agent_user_until_consent = run_consultation

        runner._run_negotiation()

        self.assertEqual(len(agent.calls), 2)
        self.assertFalse(agent.calls[0]["consulted_this_turn"])
        self.assertIsNone(agent.calls[0]["consultation_state"])
        self.assertTrue(agent.calls[1]["consulted_this_turn"])
        self.assertEqual(agent.calls[1]["consultation_state"], "user_consultation")
        self.assertNotEqual(
            agent.calls[0]["decision_cycle_id"],
            agent.calls[1]["decision_cycle_id"],
        )
        self.assertEqual(
            [call["attempts_before_call"] for call in agent.calls],
            [0, 0],
        )
        self.assertEqual(len(consultation_calls), 1)
        self.assertTrue(consultation_calls[0]["require_initial_message_user"])
        self.assertEqual(
            [
                turn["action"].action_type
                for turn in agent.visible_dialogue.conversation_history
            ],
            [AgentActionType.ASK_USER, AgentActionType.QUIT],
        )
        self.assertEqual(
            [
                turn["action"].action_type
                for turn in other_visible_dialogue.conversation_history
            ],
            [AgentActionType.QUIT],
        )
        self.assertEqual(
            agent.visible_dialogue.conversation_history[0]["action"].output,
            "Let me check.",
        )
        self.assertTrue(
            all(
                turn["channel"] == "agent_agent"
                for turn in agent.visible_dialogue.conversation_history
            )
        )
        self.assertEqual(
            runner.logger.subsections[-2:],
            [
                ("agent-agent dialogue (negotiation, max 5 turns)", "agent_1"),
                ("agent-agent dialogue (negotiation, max 5 turns)", "user_1"),
            ],
        )

    def test_confirmation_rejection_consumes_turn_and_switches_speaker(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.subsections = []
                self.turns = []

            def subsection(self, message, to="both"):
                self.subsections.append((message, to))

            def turn(self, speaker, action, to="both", invalid_reason=None):
                self.turns.append((speaker, action, to, invalid_reason))

        class ConfirmingAgent:
            agent_id = "agent_2"
            no_user = False
            config = {}
            orchestrator_settings = {}
            prompts = {}
            user = SimpleNamespace(user_id="user_2")

            def __init__(self) -> None:
                self.calls = []
                self.visible_dialogue = Dialogue(self.agent_id, self.user.user_id)

            def respond_to_agent(
                self,
                dialogue,
                game,
                *,
                consulted_this_turn=False,
                consultation_state=None,
                attempt_budget=None,
            ):
                self.calls.append((consulted_this_turn, consultation_state))
                if len(self.calls) == 1:
                    return AgentAction(
                        AgentActionType.ACCEPT_AND_CONFIRM,
                        output="Internal note that must not reach the counterparty.",
                    )

                raise AssertionError("The confirming agent must not act twice in one turn")

        class FollowingAgent:
            agent_id = "agent_1"
            no_user = False
            # Only the confirming agent's user decides (not a Dual User game).
            no_final_confirmation = True
            config = {}
            orchestrator_settings = {}
            prompts = {}
            user = SimpleNamespace(user_id="user_1")

            def __init__(self) -> None:
                self.calls = []
                self.visible_dialogue = Dialogue(self.agent_id, self.user.user_id)

            def respond_to_agent(
                self,
                dialogue,
                game,
                *,
                consulted_this_turn=False,
                consultation_state=None,
                attempt_budget=None,
            ):
                self.calls.append((consulted_this_turn, consultation_state))
                return AgentAction(AgentActionType.QUIT, output="Stop after rejection.")

        accepting_agent = FollowingAgent()
        confirming_agent = ConfirmingAgent()
        dialogue = Dialogue("agent_1", "agent_2")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.PROPOSE,
                output="Proposed terms.",
                action_details={"terms": "tentative"},
            ),
        )
        game = SimpleNamespace(
            agent_1=accepting_agent,
            agent_2=confirming_agent,
            game_dialogue=dialogue,
            state={},
            end_reason=None,
            quitter_user_id=None,
            is_invalid_game_state=lambda: False,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = game
        runner.config = {}
        runner.logger = FakeLogger()
        runner.current_turn = 1
        runner.max_negotiation_turns = 5
        runner.turns_by_agent = {"agent_1": 1, "agent_2": 0}
        runner._last_invalid_error = None
        runner._last_invalid_error_agent_user = None
        runner._update_turn_state = lambda next_speaker=None: None
        penalty_calls = []
        runner._apply_per_turn_penalties = lambda: penalty_calls.append(True)
        def confirm_after_visibility_update(selected_agent, deal, history):
            self.assertIs(selected_agent, confirming_agent)
            initiator_turn = confirming_agent.visible_dialogue.conversation_history[-1]
            counterparty_turn = accepting_agent.visible_dialogue.conversation_history[-1]
            self.assertIn("Internal note", initiator_turn["action"].output)
            self.assertNotIn("Internal note", counterparty_turn["action"].output)
            self.assertIn("selected ACCEPT AND CONFIRM WITH USER", counterparty_turn["action"].output)
            self.assertEqual(counterparty_turn["action"].action_details, deal)
            self.assertEqual(counterparty_turn["speaker"], "agent_2")
            return {"user_2": False}

        runner._run_final_confirmation = confirm_after_visibility_update

        runner._run_negotiation()

        negotiation_subsection = "agent-agent dialogue (negotiation, max 5 turns)"
        self.assertEqual(
            runner.logger.subsections,
            [
                (negotiation_subsection, "both"),
                (negotiation_subsection, "agent_1"),
                (negotiation_subsection, "user_1"),
                (negotiation_subsection, "agent_2"),
                (negotiation_subsection, "user_2"),
            ],
        )
        self.assertEqual(
            confirming_agent.calls,
            [(False, None)],
        )
        self.assertEqual(accepting_agent.calls, [(False, None)])
        self.assertEqual(runner.current_turn, 2)
        self.assertEqual(runner.turns_by_agent, {"agent_1": 1, "agent_2": 1})
        self.assertEqual(dialogue.current_speaker, "agent_1")
        self.assertEqual(len(penalty_calls), 1)

        confirmation_turn = dialogue.conversation_history[-1]
        self.assertEqual(confirmation_turn["speaker"], "agent_2")
        self.assertEqual(confirmation_turn["turn_num"], 2)
        self.assertEqual(
            confirmation_turn["action"].action_type,
            AgentActionType.ACCEPT_AND_CONFIRM,
        )
        self.assertIn("Final confirmation declined", confirmation_turn["action"].output)
        self.assertNotIn("Internal note", confirmation_turn["action"].output)
        self.assertNotIn("Negotiation continues", confirmation_turn["action"].output)
        self.assertNotIn("user_1:", confirmation_turn["action"].output)
        self.assertIn("user_2: DONT_CONSENT", confirmation_turn["action"].output)
        self.assertIn("agent_1: PROPOSE", confirmation_turn["action"].output)

        public_confirmation_turns = [
            action
            for _, action, target, _ in runner.logger.turns
            if target == "both"
            and action.action_type == AgentActionType.ACCEPT_AND_CONFIRM
        ]
        self.assertEqual(public_confirmation_turns, [confirmation_turn["action"]])
        self.assertTrue(
            any(
                target == "agent_2"
                and action.output.startswith("Internal note")
                for _, action, target, _ in runner.logger.turns
            )
        )
        sanitized_events = [
            action
            for _, action, target, _ in runner.logger.turns
            if target == "agent_1"
            and action.action_type == AgentActionType.ACCEPT_AND_CONFIRM
            and "selected ACCEPT AND CONFIRM WITH USER" in action.output
        ]
        self.assertEqual(len(sanitized_events), 1)
        self.assertNotIn("Internal note", sanitized_events[0].output)
        self.assertEqual(sanitized_events[0].action_details, {"terms": "tentative"})

    def test_confirmation_on_last_available_turn_runs_before_limit_check(self) -> None:
        class FakeLogger:
            def subsection(self, message, to="both"):
                pass

            def turn(self, speaker, action, to="both", invalid_reason=None):
                pass

        class ConfirmingAgent:
            agent_id = "agent_2"
            no_user = False
            config = {}
            orchestrator_settings = {}
            prompts = {}
            user = SimpleNamespace(user_id="user_2")

            def __init__(self) -> None:
                self.calls = 0

            def respond_to_agent(self, *args, **kwargs):
                self.calls += 1
                return AgentAction(AgentActionType.ACCEPT_AND_CONFIRM)

        for confirmation_outcome in ("success", "rejection", "incomplete"):
            with self.subTest(confirmation_outcome=confirmation_outcome):
                confirmation_succeeds = confirmation_outcome == "success"
                accepting_agent = SimpleNamespace(
                    agent_id="agent_1",
                    no_user=False,
                    no_final_confirmation=True,  # single confirmation, not Dual User
                    user=SimpleNamespace(user_id="user_1"),
                )
                confirming_agent = ConfirmingAgent()
                dialogue = Dialogue("agent_1", "agent_2")
                agreed = {"terms": "tentative"}
                dialogue.add_turn(
                    "agent_1",
                    1,
                    AgentAction(
                        AgentActionType.PROPOSE,
                        output="Proposed terms.",
                        action_details=agreed,
                    ),
                    turn_num=1,
                )
                game = SimpleNamespace(
                    agent_1=accepting_agent,
                    agent_2=confirming_agent,
                    game_dialogue=dialogue,
                    state={},
                    end_reason=None,
                    quitter_user_id=None,
                    is_invalid_game_state=lambda: False,
                )
                runner = RoundRunner.__new__(RoundRunner)
                runner.game = game
                runner.config = {}
                runner.logger = FakeLogger()
                runner.current_turn = 1
                runner.max_negotiation_turns = 2
                runner.turns_by_agent = {"agent_1": 1, "agent_2": 0}
                runner._last_invalid_error = None
                runner._last_invalid_error_agent_user = None
                runner._update_turn_state = lambda next_speaker=None: None
                penalty_calls = []
                runner._apply_per_turn_penalties = lambda: penalty_calls.append(True)
                def run_final_confirmation(selected_agent, deal, history):
                    self.assertIs(selected_agent, confirming_agent)
                    if confirmation_outcome == "incomplete":
                        raise GameLimitExceededError(
                            "final-confirmation exchange limit exceeded"
                        )
                    return {"user_2": confirmation_succeeds}

                runner._run_final_confirmation = run_final_confirmation
                finalized = []
                runner._finalize_deal = lambda deal: (
                    finalized.append(deal),
                    setattr(game, "end_reason", "deal"),
                )

                if confirmation_succeeds:
                    runner._run_negotiation()
                else:
                    expected_error = (
                        "final-confirmation exchange limit exceeded"
                        if confirmation_outcome == "incomplete"
                        else r"max_negotiation_turns \(2\) exceeded"
                    )
                    with self.assertRaisesRegex(
                        GameLimitExceededError,
                        expected_error,
                    ):
                        runner._run_negotiation()

                self.assertEqual(confirming_agent.calls, 1)
                self.assertEqual(runner.current_turn, 2)
                self.assertEqual(
                    runner.turns_by_agent,
                    {"agent_1": 1, "agent_2": 1},
                )
                self.assertEqual(len(penalty_calls), 1)
                self.assertEqual(finalized, [agreed] if confirmation_succeeds else [])
                self.assertEqual(
                    game.end_reason,
                    "deal" if confirmation_succeeds else None,
                )
                public_output = dialogue.conversation_history[-1]["action"].output
                expected_fragment = {
                    "success": "tentative deal is confirmed",
                    "rejection": "Final confirmation declined",
                    "incomplete": "Final confirmation is incomplete",
                }[confirmation_outcome]
                self.assertIn(expected_fragment, public_output)
                if confirmation_outcome == "success":
                    self.assertNotIn("user_1:", public_output)
                    self.assertIn("user_2: CONSENT", public_output)
                elif confirmation_outcome == "rejection":
                    self.assertNotIn("user_1:", public_output)
                    self.assertIn("user_2: DONT_CONSENT", public_output)
                else:
                    self.assertNotIn("User confirmation decisions:", public_output)

    def test_confirmation_rejection_uses_yaml_instruction(self) -> None:
        instruction = _select_negotiation_instruction(
            {
                "first_turn": "Open.",
                "continuation": "Continue.",
                "continuation_after_confirmation_rejection": (
                    "YAML confirmation rejection."
                ),
            },
            has_agent_history=True,
            consulted_this_turn=True,
            consultation_state="confirmation_rejected",
            current_prompt_fields={},
        )

        self.assertEqual(instruction, "YAML confirmation rejection.")

    def test_structured_proposals_are_preserved_in_conversation_history(self) -> None:
        proposal = {
            "user_1": {"house": 0.63},
            "user_2": {"house": 0.37},
        }
        dialogue = Dialogue("agent_1", "agent_2")
        dialogue.add_turn(
            "agent_1",
            1,
            AgentAction(
                AgentActionType.PROPOSE,
                output="I propose this division.",
                action_details=proposal,
            ),
        )

        agent_text = "\n".join(
            message["content"]
            for message in _format_agent_dialogue(dialogue, "agent_2")
        )
        user_history_text = _format_game_dialogue_as_text(dialogue)

        for rendered in (agent_text, user_history_text):
            self.assertIn("[STRUCTURED PROPOSAL FROM agent_1]", rendered)
            self.assertIn('"user_1": {"house": 0.63}', rendered)
            self.assertIn('"user_2": {"house": 0.37}', rendered)

    def test_prompt_game_state_drops_last_proposal_snapshot(self) -> None:
        rendered = format_game_state_with_flow_status(
            "Turn: 4\n"
            "Last proposals: agent_1: {'user_1': 0.63}\n"
            "No-deal outcome: the government seizes the assets.\n"
            "Assets: House, investments, land",
            current_turn=4,
            max_turns=10,
        )

        self.assertNotIn("Last proposal", rendered)
        self.assertNotIn("Turn: 4", rendered)
        self.assertEqual(rendered.count("Current public agent-agent action:"), 1)
        self.assertIn(
            "No-deal outcome: the government seizes the assets.", rendered
        )
        self.assertIn("Assets: House, investments, land", rendered)

    def test_final_public_action_is_still_available_before_limit_exhaustion(self) -> None:
        rendered = format_game_state_with_flow_status(
            "No-deal outcome: no sale is completed.",
            current_turn=40,
            max_turns=40,
        )

        self.assertIn("[NEGOTIATION ACTION STATUS]", rendered)
        self.assertIn("Current public agent-agent action: 40 of 40.", rendered)
        self.assertIn(
            "Further public agent-agent actions available after the current action: 0.",
            rendered,
        )
        self.assertIn("The current public action can still complete a deal", rendered)
        self.assertIn(
            "The total action limit is exhausted only after public agent-agent "
            "action 40 finishes.",
            rendered,
        )
        self.assertIn(
            "If no deal has been completed then, the negotiation ends without agreement.",
            rendered,
        )


    def test_final_confirmation_asks_only_selected_user_after_rejection(self) -> None:
        agent_1 = SimpleNamespace(
            agent_id="agent_1",
            user=SimpleNamespace(user_id="user_1"),
            no_user=False,
        )
        agent_2 = SimpleNamespace(
            agent_id="agent_2",
            user=SimpleNamespace(user_id="user_2"),
            no_user=False,
        )
        runner = RoundRunner.__new__(RoundRunner)
        runner.game = SimpleNamespace(agent_1=agent_1, agent_2=agent_2)
        runner.current_turn = 6
        runner._consultation_mode_for_agent = lambda agent, mode: None

        calls = []

        def confirm(agent, dialogue, **kwargs):
            calls.append((agent.agent_id, kwargs))
            return ConsultationResult(
                action=(
                    None
                    if agent.agent_id == "agent_1"
                    else AgentAction(AgentActionType.REQUEST_USER_CONFIRMATION)
                ),
                end_reason="user_rejection" if agent.agent_id == "agent_1" else "user_consent",
                completed_exchanges=1,
                max_consecutive_invalid_attempts=0,
                origin="final_confirmation",
            )

        runner._run_agent_user_until_consent = confirm
        deal = {"user_1": {"house": 0.63}, "user_2": {"house": 0.37}}
        dialogue = Dialogue("agent_1", "agent_2")

        outcomes = runner._run_final_confirmation(agent_1, deal, dialogue)

        self.assertEqual([agent_id for agent_id, _ in calls], ["agent_1"])
        self.assertEqual(outcomes, {"user_1": False})
        self.assertTrue(all(call[1]["agreed_deal"] == deal for call in calls))
        self.assertTrue(all(call[1]["return_on_user_rejection"] for call in calls))
        self.assertTrue(all(call[1]["consultation_mode"] == "final_decision" for call in calls))
        self.assertTrue(
            all(
                call[1]["consultation_origin"] == "final_confirmation"
                for call in calls
            )
        )


if __name__ == "__main__":
    unittest.main()
