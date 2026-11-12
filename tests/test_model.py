from types import SimpleNamespace
import os
import threading
import unittest
from unittest.mock import Mock, patch

import anthropic
import httpx
from openai import BadRequestError

from negsim.model import (
    AnthropicModel,
    OpenAIModel,
    TogetherAIModel,
    create_model,
)


class OpenAIModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_log_env = patch.dict(
            "os.environ",
            {"RAW_API_LOG_ENABLED": "0"},
        )
        self.raw_log_env.start()

    def tearDown(self) -> None:
        self.raw_log_env.stop()

    @staticmethod
    def _bad_request(code: str) -> BadRequestError:
        response = httpx.Response(
            400,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        return BadRequestError(
            "Invalid prompt",
            response=response,
            body={
                "error": {
                    "message": "Invalid prompt",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": code,
                }
            },
        )

    def test_prompt_cache_metadata_and_usage_are_recorded(self) -> None:
        class FakeCompletions:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                usage = SimpleNamespace(
                    model_dump=lambda: {
                        "prompt_tokens": 1200,
                        "prompt_tokens_details": {"cached_tokens": 768},
                    }
                )
                message = SimpleNamespace(content="ok")
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice], usage=usage)

        completions = FakeCompletions()
        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        model = OpenAIModel(
            "gpt-test",
            prompt_cache_key="cache-key",
            prompt_cache_retention="24h",
        )
        model._client = fake_client

        result = model.query_model({"system_prompt": "system", "current_prompt": "current"})

        self.assertEqual(result, "ok")
        self.assertEqual(completions.kwargs["prompt_cache_key"], "cache-key")
        self.assertEqual(completions.kwargs["prompt_cache_retention"], "24h")
        self.assertEqual(model.last_usage["prompt_tokens_details"]["cached_tokens"], 768)
        self.assertEqual(model.last_request_metadata["prompt_cache_key"], "cache-key")
        self.assertIn("latency_seconds", model.last_request_metadata)

    def test_gpt56_disables_implicit_prompt_cache_writes_by_default(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        model = OpenAIModel("gpt-5.6-luna")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")
        self.assertEqual(
            captured["extra_body"],
            {"prompt_cache_options": {"mode": "explicit"}},
        )
        self.assertEqual(model.last_request_metadata["prompt_cache_mode"], "explicit")

    def test_gpt56_can_explicitly_opt_into_implicit_prompt_cache_writes(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        model = OpenAIModel("gpt-5.6-sol", prompt_cache_mode="implicit")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")
        self.assertEqual(
            captured["extra_body"],
            {"prompt_cache_options": {"mode": "implicit"}},
        )

    def test_older_gpt_model_does_not_receive_prompt_cache_options(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        model = OpenAIModel("gpt-5.4-mini-2026-03-17")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")
        self.assertNotIn("extra_body", captured)

    def test_invalid_prompt_cache_mode_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "prompt_cache_mode"):
            OpenAIModel("gpt-5.6-luna", prompt_cache_mode="automatic")

    def test_default_api_key_environment_variable_and_timeout_are_used(self) -> None:
        model = OpenAIModel("gpt-test")

        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_API_KEY": "openai-key",
                    "OPENAI_API_TIMEOUT_SECONDS": "45",
                },
            ),
            patch("negsim.model.OpenAI") as client_class,
        ):
            model.initialize_model()

        client_class.assert_called_once_with(
            api_key="openai-key",
            timeout=45,
            max_retries=0,
        )
        self.assertEqual(model.request_timeout_seconds, 45)

    def test_raw_request_and_response_events_are_emitted(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
            _request_id="openai-provider-request",
        )
        completions = SimpleNamespace(create=lambda **kwargs: response)
        model = OpenAIModel("gpt-test")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with (
            patch("negsim.model.new_api_trace_id", return_value="openai-trace"),
            patch("negsim.model.log_api_event") as log_event,
        ):
            result = model.query_model(
                {
                    "system_prompt": "system",
                    "current_prompt": "current",
                    "api_log_context": {"game_id": "game-1", "actor_id": "agent_1"},
                }
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_success"],
        )
        start = log_event.call_args_list[0].kwargs
        self.assertEqual(start["trace_id"], "openai-trace")
        self.assertEqual(start["request"]["messages"][0]["role"], "system")
        self.assertEqual(start["context"]["game_id"], "game-1")
        self.assertIs(log_event.call_args_list[1].kwargs["response"], response)
        self.assertEqual(model.last_request_metadata["trace_id"], "openai-trace")
        self.assertEqual(
            model.last_request_metadata["provider_request_id"],
            "openai-provider-request",
        )
        self.assertEqual(model.last_request_metadata["attempt"], 1)

    def test_first_query_loads_env_before_cache_and_retry_settings(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=None,
            )

        order: list[str] = []

        def load_env() -> None:
            order.append("load_env")
            os.environ["OPENAI_PROMPT_CACHE_KEY"] = "from-dotenv"
            os.environ["OPENAI_PROMPT_CACHE_RETENTION"] = "24h"
            os.environ["OPENAI_API_MAX_RETRIES"] = "2"

        def log_event(**kwargs) -> None:
            order.append(kwargs["event"])

        model = OpenAIModel("gpt-test")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENAI_PROMPT_CACHE_KEY": "",
                    "OPENAI_PROMPT_CACHE_RETENTION": "",
                    "OPENAI_API_MAX_RETRIES": "",
                },
            ),
            patch("negsim.model._load_simple_env", side_effect=load_env),
            patch("negsim.model.log_api_event", side_effect=log_event),
        ):
            self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")

        self.assertEqual(order[:2], ["load_env", "request_start"])
        self.assertEqual(captured["prompt_cache_key"], "from-dotenv")
        self.assertEqual(captured["prompt_cache_retention"], "24h")

    def test_response_processing_error_is_not_logged_as_request_error(self) -> None:
        response = SimpleNamespace(choices=[], usage=None)
        completions = SimpleNamespace(create=lambda **kwargs: response)
        model = OpenAIModel("gpt-test")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )

        with (
            patch("negsim.model.log_api_event") as log_event,
            self.assertRaises(IndexError),
        ):
            model.query_model({"current_prompt": "current"})

        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_success", "response_processing_error"],
        )

    def test_client_initialization_error_is_logged_and_preserved(self) -> None:
        model = OpenAIModel("gpt-test")
        initialization_error = RuntimeError("missing OpenAI credentials")

        with (
            patch.object(
                model,
                "initialize_model",
                side_effect=initialization_error,
            ),
            patch("negsim.model.log_api_event") as log_event,
            self.assertRaises(RuntimeError) as raised,
        ):
            model.query_model({"current_prompt": "current"})

        self.assertIs(raised.exception, initialization_error)
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_error"],
        )
        error_event = log_event.call_args_list[-1].kwargs
        self.assertFalse(error_event["retryable"])
        self.assertFalse(error_event["will_retry"])

    def test_invalid_prompt_is_retried(self) -> None:
        class FakeCompletions:
            def __init__(self, error: Exception) -> None:
                self.error = error
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise self.error
                message = SimpleNamespace(content="ok")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=message)],
                    usage=None,
                )

        completions = FakeCompletions(self._bad_request("invalid_prompt"))
        model = OpenAIModel("gpt-test")
        model._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with (
            patch.dict("os.environ", {"OPENAI_API_MAX_RETRIES": "2"}),
            patch("negsim.model.random.uniform", return_value=0),
            patch("negsim.model.time.sleep") as sleep,
            patch("negsim.model.log_api_event") as log_event,
        ):
            result = model.query_model({"current_prompt": "current"})

        self.assertEqual(result, "ok")
        self.assertEqual(completions.calls, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            [
                "request_start",
                "request_error",
                "request_start",
                "request_success",
            ],
        )
        self.assertEqual(log_event.call_args_list[1].kwargs["attempt"], 1)
        self.assertEqual(log_event.call_args_list[3].kwargs["attempt"], 2)
        self.assertTrue(log_event.call_args_list[1].kwargs["retryable"])
        self.assertTrue(log_event.call_args_list[1].kwargs["will_retry"])

    def test_other_bad_request_is_not_retried(self) -> None:
        class FakeCompletions:
            def __init__(self, error: Exception) -> None:
                self.error = error
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                raise self.error

        completions = FakeCompletions(self._bad_request("invalid_parameter"))
        model = OpenAIModel("gpt-test")
        model._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        with (
            patch.dict("os.environ", {"OPENAI_API_MAX_RETRIES": "2"}),
            patch("negsim.model.time.sleep") as sleep,
            self.assertRaises(BadRequestError),
        ):
            model.query_model({"current_prompt": "current"})

        self.assertEqual(completions.calls, 1)
        sleep.assert_not_called()


class TogetherAIModelTests(unittest.TestCase):
    PROVIDERS = (
        (
            TogetherAIModel,
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "togetherai",
            "TOGETHERAI",
            "https://api.together.ai/v1",
        ),
    )

    def setUp(self) -> None:
        self.raw_log_env = patch.dict(
            "os.environ",
            {"RAW_API_LOG_ENABLED": "0"},
        )
        self.raw_log_env.start()

    def tearDown(self) -> None:
        self.raw_log_env.stop()

    def test_clients_use_provider_key_timeout_and_base_url(self) -> None:
        for model_class, model_name, _, env_prefix, base_url in self.PROVIDERS:
            with self.subTest(provider=env_prefix.lower()):
                api_key_name = f"{env_prefix}_API_KEY"
                model = model_class(model_name)

                with (
                    patch.dict(
                        "os.environ",
                        {
                            api_key_name: f"{env_prefix.lower()}-key",
                            f"{env_prefix}_API_TIMEOUT_SECONDS": "45",
                        },
                    ),
                    patch("negsim.model.OpenAI") as client_class,
                ):
                    model.initialize_model()

                client_class.assert_called_once_with(
                    api_key=f"{env_prefix.lower()}-key",
                    timeout=45,
                    max_retries=0,
                    base_url=base_url,
                )
                self.assertEqual(model.api_key_name, api_key_name)
                self.assertEqual(model.request_timeout_seconds, 45)

    def test_missing_provider_key_does_not_fall_back_to_openai_key(self) -> None:
        for model_class, model_name, _, env_prefix, _ in self.PROVIDERS:
            with self.subTest(provider=env_prefix.lower()):
                api_key_name = f"{env_prefix}_API_KEY"
                model = model_class(model_name)

                with (
                    patch.dict(
                        "os.environ",
                        {
                            api_key_name: "",
                            "OPENAI_API_KEY": "must-not-be-used",
                        },
                    ),
                    patch("negsim.model.OpenAI") as client_class,
                    self.assertRaisesRegex(RuntimeError, f"{api_key_name} is not set"),
                ):
                    model.initialize_model()

                client_class.assert_not_called()

    def test_queries_use_provider_logging_without_openai_cache_fields(self) -> None:
        for model_class, model_name, provider_name, env_prefix, _ in self.PROVIDERS:
            with self.subTest(provider=provider_name):
                response = SimpleNamespace(
                    id=f"{provider_name}-request",
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content="ok"),
                        )
                    ],
                    usage=None,
                )
                captured: dict = {}

                def create(**kwargs):
                    captured.update(kwargs)
                    return response

                model = model_class(
                    model_name,
                    prompt_cache_key="must-not-be-sent",
                    prompt_cache_retention="must-not-be-sent",
                    max_tokens=4096,
                    reasoning_enabled=False,
                )
                model._client = SimpleNamespace(
                    chat=SimpleNamespace(
                        completions=SimpleNamespace(create=create)
                    )
                )
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": {
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                }

                with (
                    patch.dict(
                        "os.environ",
                        {f"{env_prefix}_API_MAX_RETRIES": "1"},
                    ),
                    patch("negsim.model.log_api_event") as log_event,
                ):
                    result = model.query_model(
                        {
                            "system_prompt": "system",
                            "current_prompt": "current",
                        },
                        response_format=response_format,
                    )

                self.assertEqual(result, "ok")
                self.assertEqual(captured["response_format"], response_format)
                self.assertEqual(captured["max_tokens"], 4096)
                self.assertEqual(
                    captured["extra_body"],
                    {"reasoning": {"enabled": False}},
                )
                self.assertNotIn("prompt_cache_key", captured)
                self.assertNotIn("prompt_cache_retention", captured)
                self.assertTrue(
                    all(
                        call.kwargs["provider"] == provider_name
                        for call in log_event.call_args_list
                    )
                )
                self.assertEqual(
                    model.last_request_metadata["provider"], provider_name
                )
                self.assertEqual(
                    model.last_request_metadata["provider_request_id"],
                    f"{provider_name}-request",
                )
                self.assertEqual(model.last_request_metadata["max_tokens"], 4096)
                self.assertFalse(
                    model.last_request_metadata["reasoning_enabled"]
                )

    def test_generation_controls_are_omitted_by_default(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="together-defaults",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok"),
                    )
                ],
                usage=None,
            )

        model = TogetherAIModel("model")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")

        self.assertNotIn("max_tokens", captured)
        self.assertNotIn("extra_body", captured)
        self.assertNotIn("stream", captured)
        self.assertIsNone(model.last_request_metadata["max_tokens"])
        self.assertIsNone(model.last_request_metadata["reasoning_enabled"])
        self.assertIsNone(model.last_request_metadata["stream"])

    def test_messages_are_normalized_for_together_chat_templates(self) -> None:
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="together-normalized",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok"),
                    )
                ],
                usage=None,
            )

        model = TogetherAIModel("model")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = model.query_model(
            {
                "system_prompt": "top-level system",
                "conversation_history": [
                    {"role": "system", "content": "opening context"},
                    {"role": "system", "content": "more context"},
                    {"role": "user", "content": "user one"},
                    {"role": "user", "content": "user two"},
                    {"role": "assistant", "content": "model reply"},
                    {"role": "system", "content": "turn boundary"},
                ],
                "current_prompt": "current",
            }
        )

        self.assertEqual(result, "ok")
        self.assertEqual(
            captured["messages"],
            [
                {
                    "role": "system",
                    "content": (
                        "top-level system\n\nopening context\n\nmore context"
                    ),
                },
                {"role": "user", "content": "user one\n\nuser two"},
                {"role": "assistant", "content": "model reply"},
                {"role": "user", "content": "turn boundary\n\ncurrent"},
            ],
        )

    def test_invalid_together_message_fails_before_api_call(self) -> None:
        create = Mock()
        model = TogetherAIModel("model")
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        invalid_prompts = (
            ({"conversation_history": [{"role": "tool", "content": "x"}]}, "role"),
            (
                {"conversation_history": [{"role": "user", "content": 1}]},
                "content",
            ),
        )
        for prompt, message in invalid_prompts:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    model.query_model(prompt)

        create.assert_not_called()

    def test_provider_retry_env_is_used_for_empty_success_response(self) -> None:
        for model_class, model_name, provider_name, env_prefix, _ in self.PROVIDERS:
            with self.subTest(provider=provider_name):
                responses = [
                    SimpleNamespace(
                        id=f"{provider_name}-empty",
                        choices=[],
                        usage=None,
                    ),
                    SimpleNamespace(
                        id=f"{provider_name}-good",
                        choices=[
                            SimpleNamespace(
                                finish_reason="stop",
                                message=SimpleNamespace(content="ok"),
                            )
                        ],
                        usage=None,
                    ),
                ]

                class FakeCompletions:
                    def __init__(self) -> None:
                        self.calls = 0

                    def create(self, **kwargs):
                        self.calls += 1
                        return responses.pop(0)

                completions = FakeCompletions()
                model = model_class(model_name)
                model._client = SimpleNamespace(
                    chat=SimpleNamespace(completions=completions)
                )

                with (
                    patch.dict(
                        "os.environ",
                        {f"{env_prefix}_API_MAX_RETRIES": "2"},
                    ),
                    patch("negsim.model.random.uniform", return_value=0),
                    patch("negsim.model.time.sleep") as sleep,
                ):
                    result = model.query_model({"current_prompt": "current"})

                self.assertEqual(result, "ok")
                self.assertEqual(completions.calls, 2)
                expected_delay = 10.0 if provider_name == "togetherai" else 1.0
                sleep.assert_called_once_with(expected_delay)

    def test_invalid_together_generation_options_fail_fast(self) -> None:
        for value in (0, -1, True, "4096"):
            with self.subTest(max_tokens=value):
                with self.assertRaisesRegex(ValueError, "max_tokens"):
                    TogetherAIModel("model", max_tokens=value)
        for value in ("false", 0, 1):
            with self.subTest(reasoning_enabled=value):
                with self.assertRaisesRegex(ValueError, "reasoning_enabled"):
                    TogetherAIModel("model", reasoning_enabled=value)
        with self.assertRaisesRegex(ValueError, "stream"):
            TogetherAIModel("model", stream="true")

    def test_streaming_response_is_assembled_and_closed(self) -> None:
        usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
        )
        chunks = [
            SimpleNamespace(
                id="stream-request",
                model="Qwen/Qwen3.6-Plus",
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content="O",
                            reasoning="private",
                        ),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="stream-request",
                model="Qwen/Qwen3.6-Plus",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        delta=SimpleNamespace(content="K", reasoning=None),
                    )
                ],
                usage=None,
            ),
            SimpleNamespace(
                id="stream-request",
                model="Qwen/Qwen3.6-Plus",
                choices=[],
                usage=usage,
            ),
        ]

        class FakeStream:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return iter(chunks)

            def close(self) -> None:
                self.closed = True

        stream = FakeStream()
        captured: dict = {}

        def create(**kwargs):
            captured.update(kwargs)
            return stream

        model = TogetherAIModel("Qwen/Qwen3.6-Plus", stream=True)
        model._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )

        result = model.query_model({"current_prompt": "current"})

        self.assertEqual(result, "OK")
        self.assertTrue(captured["stream"])
        self.assertTrue(stream.closed)
        self.assertEqual(model.last_usage["total_tokens"], 12)
        self.assertEqual(
            model.last_request_metadata["provider_request_id"],
            "stream-request",
        )
        self.assertTrue(model.last_request_metadata["stream"])


class AnthropicModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_log_env = patch.dict(
            "os.environ",
            {"RAW_API_LOG_ENABLED": "0"},
        )
        self.raw_log_env.start()

    def tearDown(self) -> None:
        self.raw_log_env.stop()

    def test_default_api_key_environment_variable_is_used(self) -> None:
        model = AnthropicModel("claude-test")
        self.assertEqual(model.max_tokens, 8192)

        with (
            patch.dict(
                "os.environ",
                {
                    "ANTHROPIC_API_KEY": "anthropic-key",
                    "ANTHROPIC_API_TIMEOUT_SECONDS": "45",
                },
            ),
            patch("negsim.model.anthropic.Anthropic") as client_class,
        ):
            model.initialize_model()

        client_class.assert_called_once_with(
            api_key="anthropic-key",
            timeout=45,
            max_retries=0,
        )
        self.assertEqual(model.request_timeout_seconds, 45)

    def test_messages_schema_usage_and_text_are_translated(self) -> None:
        class FakeMessages:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                usage = SimpleNamespace(
                    model_dump=lambda: {
                        "input_tokens": 120,
                        "output_tokens": 30,
                    }
                )
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(type="thinking", thinking="private"),
                        SimpleNamespace(type="text", text='{"action":"PROPOSE"}'),
                    ],
                    usage=usage,
                )

        messages = FakeMessages()
        model = AnthropicModel("claude-test", max_tokens=4096)
        model._client = SimpleNamespace(messages=messages)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "agent_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
            },
        }

        result = model.query_model(
            {
                "system_prompt": "system",
                "conversation_history": [
                    {"role": "user", "content": "earlier"},
                    {"role": "assistant", "content": "reply"},
                ],
                "current_prompt": "current",
            },
            response_format=response_format,
        )

        self.assertEqual(result, '{"action":"PROPOSE"}')
        self.assertEqual(messages.kwargs["system"], "system")
        self.assertEqual(
            messages.kwargs["messages"],
            [
                {"role": "user", "content": "earlier"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "current"},
            ],
        )
        self.assertEqual(messages.kwargs["max_tokens"], 4096)
        self.assertEqual(
            messages.kwargs["output_config"],
            {
                "format": {
                    "type": "json_schema",
                    "schema": response_format["json_schema"]["schema"],
                }
            },
        )
        self.assertEqual(model.last_usage["input_tokens"], 120)
        self.assertEqual(model.last_request_metadata["provider"], "anthropic")
        self.assertIsNone(model.last_request_metadata["request_timeout_seconds"])
        self.assertIn("latency_seconds", model.last_request_metadata)

    def test_history_system_messages_are_converted_and_adjacent_roles_coalesced(
        self,
    ) -> None:
        class FakeMessages:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    usage=None,
                )

        messages = FakeMessages()
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(messages=messages)

        result = model.query_model(
            {
                "system_prompt": "top-level system",
                "conversation_history": [
                    {"role": "system", "content": "=== Turn 1 ==="},
                    {"role": "assistant", "content": "prior reply"},
                    {"role": "system", "content": "=== Turn 2 ==="},
                    {"role": "user", "content": "new offer"},
                ],
                "current_prompt": "respond now",
            }
        )

        self.assertEqual(result, "ok")
        self.assertEqual(messages.kwargs["system"], "top-level system")
        self.assertEqual(
            messages.kwargs["messages"],
            [
                {"role": "user", "content": "=== Turn 1 ==="},
                {"role": "assistant", "content": "prior reply"},
                {
                    "role": "user",
                    "content": "=== Turn 2 ===\n\nnew offer\n\nrespond now",
                },
            ],
        )
        self.assertNotIn(
            "system",
            {message["role"] for message in messages.kwargs["messages"]},
        )

    def test_raw_request_and_response_events_are_emitted(self) -> None:
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=None,
            _request_id="anthropic-provider-request",
        )
        messages = SimpleNamespace(create=lambda **kwargs: response)
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(messages=messages)

        with (
            patch("negsim.model.new_api_trace_id", return_value="anthropic-trace"),
            patch("negsim.model.log_api_event") as log_event,
        ):
            result = model.query_model(
                {
                    "system_prompt": "system",
                    "current_prompt": "current",
                    "api_log_context": {"game_id": "game-2", "actor_id": "user_2"},
                }
            )

        self.assertEqual(result, "ok")
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_success"],
        )
        start = log_event.call_args_list[0].kwargs
        self.assertEqual(start["trace_id"], "anthropic-trace")
        self.assertEqual(start["request"]["messages"][0]["content"], "current")
        self.assertEqual(start["context"]["actor_id"], "user_2")
        self.assertIs(log_event.call_args_list[1].kwargs["response"], response)
        self.assertEqual(
            model.last_request_metadata["trace_id"],
            "anthropic-trace",
        )
        self.assertEqual(
            model.last_request_metadata["provider_request_id"],
            "anthropic-provider-request",
        )
        self.assertEqual(model.last_request_metadata["attempt"], 1)

    def test_first_query_loads_env_before_retry_settings(self) -> None:
        order: list[str] = []

        def load_env() -> None:
            order.append("load_env")
            os.environ["ANTHROPIC_API_MAX_RETRIES"] = "4"

        def log_event(**kwargs) -> None:
            order.append(kwargs["event"])
            if kwargs["event"] == "request_start":
                self.assertEqual(kwargs["max_attempts"], 4)

        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=None,
        )
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: response)
        )
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_MAX_RETRIES": ""}),
            patch("negsim.model._load_simple_env", side_effect=load_env),
            patch("negsim.model.log_api_event", side_effect=log_event),
        ):
            self.assertEqual(model.query_model({"current_prompt": "current"}), "ok")

        self.assertEqual(order[:2], ["load_env", "request_start"])

    def test_response_processing_error_is_not_logged_as_request_error(self) -> None:
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=None,
        )
        messages = SimpleNamespace(create=lambda **kwargs: response)
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(messages=messages)

        with (
            patch.object(model, "_extract_text", side_effect=ValueError("bad blocks")),
            patch("negsim.model.log_api_event") as log_event,
            self.assertRaises(ValueError),
        ):
            model.query_model({"current_prompt": "current"})

        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_success", "response_processing_error"],
        )

    def test_client_initialization_error_is_logged_and_preserved(self) -> None:
        model = AnthropicModel("claude-test")
        initialization_error = RuntimeError("missing Anthropic credentials")

        with (
            patch.object(
                model,
                "initialize_model",
                side_effect=initialization_error,
            ),
            patch("negsim.model.log_api_event") as log_event,
            self.assertRaises(RuntimeError) as raised,
        ):
            model.query_model({"current_prompt": "current"})

        self.assertIs(raised.exception, initialization_error)
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            ["request_start", "request_error"],
        )
        error_event = log_event.call_args_list[-1].kwargs
        self.assertFalse(error_event["retryable"])
        self.assertFalse(error_event["will_retry"])

    def test_wall_clock_deadline_closes_stuck_client(self) -> None:
        request_released = threading.Event()
        close_started = threading.Event()
        close_released = threading.Event()

        class FakeMessages:
            def create(self, **kwargs):
                request_released.wait(timeout=1)
                return SimpleNamespace(content=[], usage=None)

        class FakeClient:
            def __init__(self) -> None:
                self.messages = FakeMessages()
                self.closed = False

            def close(self) -> None:
                self.closed = True
                close_started.set()
                close_released.wait(timeout=1)

        client = FakeClient()
        model = AnthropicModel("claude-test")
        model._client = client
        model.request_timeout_seconds = 0.01

        with self.assertRaisesRegex(TimeoutError, "wall-clock deadline"):
            model._create_message_with_deadline(
                {
                    "model": "claude-test",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hello"}],
                }
            )

        self.assertTrue(close_started.wait(timeout=0.2))
        self.assertTrue(client.closed)
        self.assertIsNone(model._client)
        close_released.set()
        request_released.set()

    def test_transient_api_error_is_retried(self) -> None:
        class FakeMessages:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise anthropic.APIConnectionError(
                        request=httpx.Request(
                            "POST", "https://api.anthropic.com/v1/messages"
                        )
                    )
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    usage=None,
                )

        messages = FakeMessages()
        model = AnthropicModel("claude-test")
        model._client = SimpleNamespace(messages=messages)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_MAX_RETRIES": "2"}),
            patch("negsim.model.random.uniform", return_value=0),
            patch("negsim.model.time.sleep") as sleep,
            patch("negsim.model.log_api_event") as log_event,
        ):
            result = model.query_model({"current_prompt": "current"})

        self.assertEqual(result, "ok")
        self.assertEqual(messages.calls, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(
            [call.kwargs["event"] for call in log_event.call_args_list],
            [
                "request_start",
                "request_error",
                "request_start",
                "request_success",
            ],
        )
        self.assertTrue(log_event.call_args_list[1].kwargs["retryable"])
        self.assertTrue(log_event.call_args_list[1].kwargs["will_retry"])


class ModelFactoryTests(unittest.TestCase):
    def test_openai_is_default_provider(self) -> None:
        model = create_model("gpt-test")

        self.assertIsInstance(model, OpenAIModel)
        self.assertEqual(model.api_key_name, "OPENAI_API_KEY")

    def test_openai_factory_forwards_prompt_cache_mode_override(self) -> None:
        model = create_model("gpt-5.6-luna", prompt_cache_mode="implicit")

        self.assertIsInstance(model, OpenAIModel)
        self.assertEqual(model.prompt_cache_mode, "implicit")

    def test_anthropic_provider_uses_anthropic_defaults(self) -> None:
        model = create_model(
            "claude-test",
            provider="Anthropic",
        )

        self.assertIsInstance(model, AnthropicModel)
        self.assertEqual(model.api_key_name, "ANTHROPIC_API_KEY")
        self.assertEqual(model.max_tokens, 8192)

    def test_anthropic_provider_forwards_max_tokens_override(self) -> None:
        model = create_model(
            "claude-test",
            provider="anthropic",
            max_tokens=2048,
        )

        self.assertIsInstance(model, AnthropicModel)
        self.assertEqual(model.max_tokens, 2048)

    def test_togetherai_provider_uses_togetherai_defaults(self) -> None:
        model = create_model(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            provider="TogetherAI",
            max_tokens=8192,
            reasoning_enabled=False,
            stream=True,
        )

        self.assertIsInstance(model, TogetherAIModel)
        self.assertEqual(model.api_key_name, "TOGETHERAI_API_KEY")
        self.assertEqual(model.base_url, "https://api.together.ai/v1")
        self.assertEqual(model.max_tokens, 8192)
        self.assertFalse(model.reasoning_enabled)
        self.assertTrue(model.stream)

    def test_unknown_provider_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as raised:
            create_model("model-test", provider="unknown")

        message = str(raised.exception)
        self.assertIn("Unsupported model provider", message)
        self.assertIn("anthropic", message)
        self.assertIn("togetherai", message)


if __name__ == "__main__":
    unittest.main()
