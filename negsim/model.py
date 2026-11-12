"""
Model: abstract base for LLM/API backends. Subclasses implement initialize_model and query_model.
"""

from __future__ import annotations

import random
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import anthropic
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from .api_request_logger import (
    get_provider_request_id,
    log_api_event,
    new_api_trace_id,
)
from .usage_tracker import record_usage


_ENV_LOADED = False
_ENV_LOCK = threading.Lock()
_API_SEMAPHORES: dict[str, tuple[int, threading.BoundedSemaphore]] = {}
_API_SEMAPHORE_LOCK = threading.Lock()
_OPENAI_PROMPT_CACHE_MODES = frozenset({"explicit", "implicit"})
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192


def _normalize_openai_prompt_cache_mode(mode: str | None) -> str:
    """Resolve the repository-wide OpenAI cache-write policy."""
    normalized = "explicit" if mode is None else str(mode).strip().lower()
    if normalized not in _OPENAI_PROMPT_CACHE_MODES:
        allowed = ", ".join(sorted(_OPENAI_PROMPT_CACHE_MODES))
        raise ValueError(
            f"Unsupported OpenAI prompt_cache_mode {mode!r}; expected one of: {allowed}."
        )
    return normalized


def openai_prompt_cache_control_kwargs(
    model_name: str,
    prompt_cache_mode: str | None = None,
) -> dict[str, Any]:
    """Return GPT-5.6+ cache controls; cache writes are off by default.

    GPT-5.6 and later default to implicit prompt-cache breakpoints.  Explicit
    mode without a breakpoint disables those implicit cache reads and writes.
    Earlier OpenAI models do not support this request field.
    """
    mode = _normalize_openai_prompt_cache_mode(prompt_cache_mode)
    match = re.match(r"^gpt-(\d+)(?:\.(\d+))?(?:-|$)", str(model_name).lower())
    if match is None:
        return {}
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    if (major, minor) < (5, 6):
        return {}
    return {
        "extra_body": {
            "prompt_cache_options": {
                "mode": mode,
            }
        }
    }


def _load_simple_env(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing env vars."""
    global _ENV_LOADED
    with _ENV_LOCK:
        if _ENV_LOADED:
            return

        env_path = path or Path(".env")
        if not env_path.exists():
            repo_env = Path(__file__).resolve().parents[1] / ".env"
            env_path = repo_env if repo_env.exists() else env_path

        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

        _ENV_LOADED = True


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def _get_api_semaphore(provider: str = "openai") -> threading.BoundedSemaphore | None:
    """Limit simultaneous API socket creation while allowing many game workers."""
    normalized_provider = provider.strip().upper()
    limit = _env_int(
        f"{normalized_provider}_API_MAX_CONCURRENT_REQUESTS", 20, minimum=0
    )
    if limit <= 0:
        return None

    with _API_SEMAPHORE_LOCK:
        cached = _API_SEMAPHORES.get(normalized_provider)
        if cached is None or cached[0] != limit:
            cached = (limit, threading.BoundedSemaphore(limit))
            _API_SEMAPHORES[normalized_provider] = cached
        return cached[1]


def _to_plain_data(value: Any) -> Any:
    """Convert SDK response objects into JSON-serializable primitives."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


class Prompt(TypedDict, total=False):
    """Prompt payload: system prompt, conversation history, and current user prompt."""
    system_prompt: str
    conversation_history: list[dict[str, str]]
    current_prompt: str
    api_log_context: dict[str, Any]


class Model(ABC):
    """Abstract base class for a model (e.g. LLM API)."""

    def __init__(self, model_name: str, api_key_name: str | None = None) -> None:
        self.model_name = model_name
        self.api_key_name = api_key_name

    @abstractmethod
    def initialize_model(self) -> None:
        """Load or connect to the model (e.g. set up client, load API key)."""
        ...

    @abstractmethod
    def query_model(
        self, prompt: Prompt, *, response_format: dict | None = None
    ) -> str:
        """Send a prompt to the model and return the response.
        If response_format is provided (e.g. json_schema), the API enforces that structure."""
        ...


class ProviderResponseError(RuntimeError):
    """The provider returned HTTP success without usable model text."""


class OpenAIModel(Model):
    """Model implementation that queries the OpenAI API."""

    DEFAULT_API_MAX_ATTEMPTS = 3
    RETRY_BASE_DELAY_SECONDS = 1.0

    def __init__(
        self,
        model_name: str,
        api_key_name: str | None = "OPENAI_API_KEY",
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        prompt_cache_mode: str | None = None,
        base_url: str | None = None,
        provider_name: str = "openai",
        env_prefix: str = "OPENAI",
        send_prompt_cache_fields: bool = True,
        require_explicit_api_key: bool = False,
    ) -> None:
        super().__init__(model_name, api_key_name)
        self._client: OpenAI | None = None
        self.prompt_cache_key = prompt_cache_key
        self.prompt_cache_retention = prompt_cache_retention
        self.prompt_cache_mode = _normalize_openai_prompt_cache_mode(
            prompt_cache_mode
        )
        self.base_url = base_url
        self.provider_name = provider_name
        self.env_prefix = env_prefix
        self.send_prompt_cache_fields = send_prompt_cache_fields
        self.require_explicit_api_key = require_explicit_api_key
        self.last_usage: dict[str, Any] | None = None
        self.last_request_metadata: dict[str, Any] | None = None
        self.request_timeout_seconds: int | None = None

    def initialize_model(self) -> None:
        _load_simple_env()
        api_key = os.environ.get(
            self.api_key_name or f"{self.env_prefix}_API_KEY"
        )
        if self.require_explicit_api_key and not api_key:
            raise RuntimeError(
                f"{self.api_key_name or f'{self.env_prefix}_API_KEY'} is not set."
            )
        self.request_timeout_seconds = _env_int(
            f"{self.env_prefix}_API_TIMEOUT_SECONDS", 120, minimum=1
        )
        # Keep retries in this class so one configured timeout is not multiplied
        # by the SDK's internal retry policy.
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.request_timeout_seconds,
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(
            **client_kwargs,
        )

    @classmethod
    def _is_invalid_prompt_error(cls, error: Exception) -> bool:
        """Return True only for the policy classifier's retryable prompt error."""
        if not isinstance(error, BadRequestError):
            return False

        code = getattr(error, "code", None)
        body = getattr(error, "body", None)
        if code is None and isinstance(body, dict):
            error_body = body.get("error", body)
            if isinstance(error_body, dict):
                code = error_body.get("code")
        return code == "invalid_prompt"

    @classmethod
    def _is_retryable_api_error(cls, error: Exception) -> bool:
        """Return True for transient OpenAI API failures worth retrying."""
        if isinstance(error, ProviderResponseError):
            return True
        if cls._is_invalid_prompt_error(error):
            return True
        if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        return False

    @staticmethod
    def _response_error(response: Any) -> ProviderResponseError | None:
        """Return a provider-specific error for an otherwise successful response."""
        return None

    @staticmethod
    def _openai_compatible_text_response_error(
        response: Any,
    ) -> ProviderResponseError | None:
        """Classify Chat Completions responses that have no usable text output."""
        choices = getattr(response, "choices", None)
        if not choices:
            return ProviderResponseError("Provider response contained no choices.")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None)
        if finish_reason == "error":
            return ProviderResponseError(
                "Provider returned finish_reason='error'."
            )
        if not isinstance(content, str) or not content.strip():
            return ProviderResponseError(
                "Provider returned no usable assistant text."
            )
        return None

    @staticmethod
    def _to_plain_data(value: Any) -> Any:
        """Convert SDK response objects into JSON-serializable primitives."""
        return _to_plain_data(value)

    def _messages_for_prompt(self, prompt: Prompt) -> list[dict[str, str]]:
        """Build the shared OpenAI-style message list for one request."""
        messages: list[dict[str, str]] = []
        if system := prompt.get("system_prompt"):
            messages.append({"role": "system", "content": system})
        if history := prompt.get("conversation_history"):
            messages.extend(history)
        if current := prompt.get("current_prompt"):
            messages.append({"role": "user", "content": current})
        return messages

    def _provider_request_kwargs(self) -> dict[str, Any]:
        """Return provider-specific Chat Completions request fields."""
        return {}

    def _provider_request_metadata(self) -> dict[str, Any]:
        """Return provider-specific fields for request diagnostics."""
        return {}

    def _normalize_provider_response(self, response: Any) -> Any:
        """Convert provider-specific response containers to one completion."""
        return response

    def query_model(
        self, prompt: Prompt, *, response_format: dict | None = None
    ) -> str:
        # Request policy and cache settings may themselves be configured in .env.
        # Load it before constructing/logging the first provider request.
        _load_simple_env()
        messages = self._messages_for_prompt(prompt)

        kwargs: dict = {"model": self.model_name, "messages": messages}
        kwargs.update(self._provider_request_kwargs())
        if response_format is not None:
            kwargs["response_format"] = response_format
        prompt_cache_key = None
        prompt_cache_retention = None
        if self.send_prompt_cache_fields:
            prompt_cache_key = self.prompt_cache_key or os.environ.get(
                f"{self.env_prefix}_PROMPT_CACHE_KEY"
            )
            prompt_cache_retention = self.prompt_cache_retention or os.environ.get(
                f"{self.env_prefix}_PROMPT_CACHE_RETENTION"
            )
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention:
            kwargs["prompt_cache_retention"] = prompt_cache_retention
        if self.provider_name == "openai":
            kwargs.update(
                openai_prompt_cache_control_kwargs(
                    self.model_name,
                    self.prompt_cache_mode,
                )
            )
        trace_id = new_api_trace_id()
        api_log_context = prompt.get("api_log_context")
        max_retries = _env_int(
            f"{self.env_prefix}_API_MAX_RETRIES",
            self.DEFAULT_API_MAX_ATTEMPTS,
            minimum=1,
        )
        self.last_usage = None
        self.last_request_metadata = {
            "provider": self.provider_name,
            "model": self.model_name,
            "trace_id": trace_id,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "prompt_cache_mode": self.prompt_cache_mode,
            "request_timeout_seconds": self.request_timeout_seconds,
            **self._provider_request_metadata(),
        }

        base_delay = self.RETRY_BASE_DELAY_SECONDS
        for attempt in range(max_retries):
            attempt_number = attempt + 1
            log_api_event(
                event="request_start",
                provider=self.provider_name,
                model=self.model_name,
                trace_id=trace_id,
                attempt=attempt_number,
                max_attempts=max_retries,
                request=kwargs,
                context=api_log_context,
            )
            started = time.monotonic()
            try:
                if self._client is None:
                    self.initialize_model()
                    if self.last_request_metadata is not None:
                        self.last_request_metadata["request_timeout_seconds"] = (
                            self.request_timeout_seconds
                        )
                assert self._client is not None
                semaphore = _get_api_semaphore(self.provider_name)
                if semaphore is None:
                    response = self._normalize_provider_response(
                        self._client.chat.completions.create(**kwargs)
                    )
                else:
                    with semaphore:
                        response = self._normalize_provider_response(
                            self._client.chat.completions.create(**kwargs)
                        )
            except Exception as e:
                latency_seconds = time.monotonic() - started
                retryable = self._is_retryable_api_error(e)
                will_retry = attempt < max_retries - 1 and retryable
                log_api_event(
                    event="request_error",
                    provider=self.provider_name,
                    model=self.model_name,
                    trace_id=trace_id,
                    attempt=attempt_number,
                    max_attempts=max_retries,
                    error=e,
                    latency_seconds=latency_seconds,
                    retryable=retryable,
                    will_retry=will_retry,
                    context=api_log_context,
                )
                if not will_retry:
                    raise
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2 + attempt)
                time.sleep(delay)
                continue

            latency_seconds = time.monotonic() - started
            response_error = self._response_error(response)
            if response_error is not None:
                will_retry = attempt < max_retries - 1
                log_api_event(
                    event="request_error",
                    provider=self.provider_name,
                    model=self.model_name,
                    trace_id=trace_id,
                    attempt=attempt_number,
                    max_attempts=max_retries,
                    response=response,
                    error=response_error,
                    latency_seconds=latency_seconds,
                    retryable=True,
                    will_retry=will_retry,
                    context=api_log_context,
                )
                if not will_retry:
                    raise response_error
                delay = base_delay * (2 ** attempt) + random.uniform(0, 2 + attempt)
                time.sleep(delay)
                continue
            log_api_event(
                event="request_success",
                provider=self.provider_name,
                model=self.model_name,
                trace_id=trace_id,
                attempt=attempt_number,
                max_attempts=max_retries,
                response=response,
                latency_seconds=latency_seconds,
                context=api_log_context,
            )
            provider_request_id = get_provider_request_id(response=response)
            try:
                self.last_usage = self._to_plain_data(getattr(response, "usage", None))
                record_usage(self.model_name, self.last_usage)
                content = response.choices[0].message.content
            except Exception as error:
                log_api_event(
                    event="response_processing_error",
                    provider=self.provider_name,
                    model=self.model_name,
                    trace_id=trace_id,
                    attempt=attempt_number,
                    max_attempts=max_retries,
                    response=response,
                    error=error,
                    latency_seconds=latency_seconds,
                    retryable=False,
                    will_retry=False,
                    context=api_log_context,
                )
                raise
            if self.last_request_metadata is not None:
                self.last_request_metadata["attempt"] = attempt_number
                self.last_request_metadata["latency_seconds"] = round(latency_seconds, 3)
                if provider_request_id is not None:
                    self.last_request_metadata["provider_request_id"] = provider_request_id
            return content or ""


class _OpenAICompatibleModel(OpenAIModel):
    """Shared behavior for non-OpenAI Chat Completions endpoints."""

    @classmethod
    def _is_invalid_prompt_error(cls, error: Exception) -> bool:
        """Third-party 4xx prompt/configuration errors are not transient."""
        return False

    @staticmethod
    def _response_error(response: Any) -> ProviderResponseError | None:
        return OpenAIModel._openai_compatible_text_response_error(response)


class TogetherAIModel(_OpenAICompatibleModel):
    """Together AI backend using its OpenAI-compatible endpoint."""

    BASE_URL = "https://api.together.ai/v1"
    DEFAULT_API_MAX_ATTEMPTS = 6
    RETRY_BASE_DELAY_SECONDS = 10.0

    def __init__(
        self,
        model_name: str,
        api_key_name: str | None = "TOGETHERAI_API_KEY",
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        max_tokens: int | None = None,
        reasoning_enabled: bool | None = None,
        stream: bool | None = None,
    ) -> None:
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens < 1
        ):
            raise ValueError("Together AI max_tokens must be a positive integer")
        if reasoning_enabled is not None and not isinstance(
            reasoning_enabled, bool
        ):
            raise ValueError("Together AI reasoning_enabled must be a boolean")
        if stream is not None and not isinstance(stream, bool):
            raise ValueError("Together AI stream must be a boolean")
        super().__init__(
            model_name=model_name,
            api_key_name=api_key_name,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            base_url=self.BASE_URL,
            provider_name="togetherai",
            env_prefix="TOGETHERAI",
            send_prompt_cache_fields=False,
            require_explicit_api_key=True,
        )
        self.max_tokens = max_tokens
        self.reasoning_enabled = reasoning_enabled
        self.stream = stream

    def _messages_for_prompt(self, prompt: Prompt) -> list[dict[str, str]]:
        """Normalize roles for Together chat templates that require alternation."""
        raw_messages = super()._messages_for_prompt(prompt)
        messages: list[dict[str, str]] = []
        seen_non_system = False
        for index, message in enumerate(raw_messages):
            role = str(message.get("role") or "").strip().lower()
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"Together AI message {index} content must be a string"
                )
            if role == "system":
                if seen_non_system:
                    role = "user"
            elif role in {"user", "assistant"}:
                seen_non_system = True
            else:
                raise ValueError(
                    f"Together AI message {index} has unsupported role {role!r}"
                )

            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = (
                    f"{messages[-1]['content']}\n\n{content}"
                )
            else:
                messages.append({"role": role, "content": content})
        return messages

    def _provider_request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_enabled is not None:
            kwargs["extra_body"] = {
                "reasoning": {"enabled": self.reasoning_enabled}
            }
        if self.stream is not None:
            kwargs["stream"] = self.stream
        return kwargs

    def _provider_request_metadata(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens,
            "reasoning_enabled": self.reasoning_enabled,
            "stream": self.stream,
        }

    def _normalize_provider_response(self, response: Any) -> Any:
        if not self.stream:
            return response

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        response_id: str | None = None
        response_model: str | None = None
        finish_reason: str | None = None
        usage: Any = None
        close = getattr(response, "close", None)
        try:
            for chunk in response:
                chunk_id = getattr(chunk, "id", None)
                if response_id is None and chunk_id is not None:
                    response_id = str(chunk_id)
                chunk_model = getattr(chunk, "model", None)
                if response_model is None and chunk_model is not None:
                    response_model = str(chunk_model)
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    content_parts.append(content)
                reasoning = getattr(delta, "reasoning", None)
                if not isinstance(reasoning, str):
                    reasoning = getattr(delta, "reasoning_content", None)
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
        finally:
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        message = SimpleNamespace(
            role="assistant",
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) or None,
        )
        return SimpleNamespace(
            id=response_id,
            model=response_model or self.model_name,
            object="chat.completion",
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason=finish_reason,
                    message=message,
                )
            ],
            usage=usage,
        )


class AnthropicModel(Model):
    """Model implementation that queries the Anthropic Messages API."""

    def __init__(
        self,
        model_name: str,
        api_key_name: str | None = "ANTHROPIC_API_KEY",
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
    ) -> None:
        super().__init__(model_name, api_key_name)
        if max_tokens < 1:
            raise ValueError("Anthropic max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_request_metadata: dict[str, Any] | None = None
        self.request_timeout_seconds: int | None = None

    def initialize_model(self) -> None:
        _load_simple_env()
        api_key = os.environ.get(self.api_key_name or "ANTHROPIC_API_KEY")
        self.request_timeout_seconds = _env_int(
            "ANTHROPIC_API_TIMEOUT_SECONDS", 120, minimum=1
        )
        # Retry in query_model only. Leaving the SDK's own retry loop enabled
        # multiplies its long read timeout by both retry policies.
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=self.request_timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _is_retryable_api_error(error: Exception) -> bool:
        """Return True for transient Anthropic API failures worth retrying."""
        if isinstance(
            error,
            (
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
            ),
        ):
            return True
        if isinstance(error, anthropic.APIStatusError):
            return error.status_code in {408, 409, 429} or error.status_code >= 500
        return False

    @staticmethod
    def _to_output_config(response_format: dict) -> dict:
        """Translate an OpenAI-style JSON schema request for Anthropic."""
        if response_format.get("type") != "json_schema":
            raise ValueError(
                "Anthropic response_format currently supports only type='json_schema'"
            )
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict) or not isinstance(
            json_schema.get("schema"), dict
        ):
            raise ValueError(
                "Anthropic json_schema response_format requires json_schema.schema"
            )
        return {
            "format": {
                "type": "json_schema",
                "schema": json_schema["schema"],
            }
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Return all text blocks while excluding thinking and other block types."""
        text_blocks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        ]
        return "\n".join(text_blocks)

    def _create_message_with_deadline(self, kwargs: dict[str, Any]) -> Any:
        """Call the sync SDK with a hard wall-clock deadline.

        HTTP read timeouts reset whenever the server sends another chunk, so a
        response that keeps the connection alive can otherwise exceed the
        configured timeout indefinitely. Closing this model's client interrupts
        the in-flight request; the next retry creates a fresh client.
        """
        assert self._client is not None
        if self.request_timeout_seconds is None:
            return self._client.messages.create(**kwargs)

        client = self._client
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def make_request() -> None:
            try:
                result_queue.put((True, client.messages.create(**kwargs)))
            except BaseException as error:
                result_queue.put((False, error))

        request_thread = threading.Thread(target=make_request, daemon=True)
        request_thread.start()
        request_thread.join(self.request_timeout_seconds)
        if request_thread.is_alive():
            self._client = None

            def close_client() -> None:
                try:
                    client.close()
                except Exception:
                    pass

            # Some transports serialize close() behind the active request.
            # Do not let cleanup defeat the wall-clock deadline.
            threading.Thread(target=close_client, daemon=True).start()
            raise TimeoutError(
                "Anthropic request exceeded the "
                f"{self.request_timeout_seconds}-second wall-clock deadline"
            )

        succeeded, value = result_queue.get_nowait()
        if succeeded:
            return value
        raise value

    @staticmethod
    def _normalize_messages(prompt: Prompt) -> list[dict[str, str]]:
        """Convert shared prompt history into Anthropic-compatible messages.

        Anthropic requires the initial system prompt in the top-level ``system``
        parameter rather than as a message. The shared dialogue formatter also
        emits in-history system messages for turn-boundary context, so represent
        those as user context and coalesce adjacent messages with the same role.
        """
        raw_messages = list(prompt.get("conversation_history") or [])
        if current := prompt.get("current_prompt"):
            raw_messages.append({"role": "user", "content": current})

        messages: list[dict[str, str]] = []
        for index, message in enumerate(raw_messages):
            role = str(message.get("role") or "").strip().lower()
            content = message.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"Anthropic message {index} content must be a string"
                )
            if role == "system":
                role = "user"
            elif role not in {"user", "assistant"}:
                raise ValueError(
                    f"Anthropic message {index} has unsupported role {role!r}"
                )

            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = (
                    f"{messages[-1]['content']}\n\n{content}"
                )
            else:
                messages.append({"role": role, "content": content})
        return messages

    def query_model(
        self, prompt: Prompt, *, response_format: dict | None = None
    ) -> str:
        # Retry policy may be configured only in .env on the first query.
        _load_simple_env()
        messages = self._normalize_messages(prompt)
        if not messages:
            raise ValueError("Anthropic requests require at least one user or assistant message")

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system := prompt.get("system_prompt"):
            kwargs["system"] = system
        if response_format is not None:
            kwargs["output_config"] = self._to_output_config(response_format)

        trace_id = new_api_trace_id()
        api_log_context = prompt.get("api_log_context")
        max_retries = _env_int("ANTHROPIC_API_MAX_RETRIES", 3, minimum=1)
        self.last_usage = None
        self.last_request_metadata = {
            "provider": "anthropic",
            "model": self.model_name,
            "trace_id": trace_id,
            "max_tokens": self.max_tokens,
            "request_timeout_seconds": self.request_timeout_seconds,
        }

        base_delay = 1.0
        for attempt in range(max_retries):
            attempt_number = attempt + 1
            log_api_event(
                event="request_start",
                provider="anthropic",
                model=self.model_name,
                trace_id=trace_id,
                attempt=attempt_number,
                max_attempts=max_retries,
                request=kwargs,
                context=api_log_context,
            )
            started = time.monotonic()
            try:
                if self._client is None:
                    self.initialize_model()
                    if self.last_request_metadata is not None:
                        self.last_request_metadata["request_timeout_seconds"] = (
                            self.request_timeout_seconds
                        )
                assert self._client is not None
                semaphore = _get_api_semaphore("anthropic")
                if semaphore is None:
                    response = self._create_message_with_deadline(kwargs)
                else:
                    with semaphore:
                        response = self._create_message_with_deadline(kwargs)
            except Exception as error:
                latency_seconds = time.monotonic() - started
                retryable = isinstance(error, TimeoutError) or self._is_retryable_api_error(
                    error
                )
                will_retry = attempt < max_retries - 1 and retryable
                log_api_event(
                    event="request_error",
                    provider="anthropic",
                    model=self.model_name,
                    trace_id=trace_id,
                    attempt=attempt_number,
                    max_attempts=max_retries,
                    error=error,
                    latency_seconds=latency_seconds,
                    retryable=retryable,
                    will_retry=will_retry,
                    context=api_log_context,
                )
                if not will_retry:
                    raise
                delay = base_delay * (2**attempt) + random.uniform(0, 2 + attempt)
                time.sleep(delay)
                continue

            latency_seconds = time.monotonic() - started
            log_api_event(
                event="request_success",
                provider="anthropic",
                model=self.model_name,
                trace_id=trace_id,
                attempt=attempt_number,
                max_attempts=max_retries,
                response=response,
                latency_seconds=latency_seconds,
                context=api_log_context,
            )
            provider_request_id = get_provider_request_id(response=response)
            try:
                self.last_usage = _to_plain_data(getattr(response, "usage", None))
                content = self._extract_text(response)
            except Exception as error:
                log_api_event(
                    event="response_processing_error",
                    provider="anthropic",
                    model=self.model_name,
                    trace_id=trace_id,
                    attempt=attempt_number,
                    max_attempts=max_retries,
                    response=response,
                    error=error,
                    latency_seconds=latency_seconds,
                    retryable=False,
                    will_retry=False,
                    context=api_log_context,
                )
                raise
            if self.last_request_metadata is not None:
                self.last_request_metadata["attempt"] = attempt_number
                self.last_request_metadata["latency_seconds"] = round(
                    latency_seconds, 3
                )
                if provider_request_id is not None:
                    self.last_request_metadata["provider_request_id"] = provider_request_id
            return content

        raise RuntimeError("Anthropic retry loop exited without returning or raising")


def create_model(
    model_name: str,
    *,
    provider: str = "openai",
    api_key_name: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    prompt_cache_mode: str | None = None,
    max_tokens: int | None = None,
    reasoning_enabled: bool | None = None,
    stream: bool | None = None,
) -> Model:
    """Create a model backend for a provider (default: OpenAI)."""
    normalized_provider = (provider or "openai").strip().lower()
    if normalized_provider == "openai":
        return OpenAIModel(
            model_name=model_name,
            api_key_name=api_key_name or "OPENAI_API_KEY",
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            prompt_cache_mode=prompt_cache_mode,
        )
    if normalized_provider == "togetherai":
        return TogetherAIModel(
            model_name=model_name,
            api_key_name=api_key_name or "TOGETHERAI_API_KEY",
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            stream=stream,
        )
    if normalized_provider == "anthropic":
        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "api_key_name": api_key_name or "ANTHROPIC_API_KEY",
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AnthropicModel(**kwargs)
    raise ValueError(
        f"Unsupported model provider {provider!r}. Use 'openai', 'togetherai', or 'anthropic'."
    )
