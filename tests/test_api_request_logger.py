from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from negsim import api_request_logger


def _write_from_spawned_process(log_path: str, index: int) -> None:
    os.environ["RAW_API_LOG_ENABLED"] = "1"
    os.environ["RAW_API_LOG_PATH"] = log_path
    api_request_logger.log_api_event(
        event="request_start",
        provider="anthropic",
        model="claude-test",
        trace_id=f"spawn-trace-{index}",
        attempt=1,
        max_attempts=1,
        request={"index": index},
    )
    if not api_request_logger.flush_api_log():
        raise RuntimeError("child raw API log did not flush")


class RawApiRequestLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        api_request_logger.flush_api_log()
        self.temp_dir = TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "raw_api_requests.jsonl"
        self.env = patch.dict(
            "os.environ",
            {
                "RAW_API_LOG_ENABLED": "1",
                "RAW_API_LOG_PATH": str(self.log_path),
                "RAW_API_LOG_RETENTION_DAYS": "7",
            },
        )
        self.env.start()
        api_request_logger._LAST_PRUNE_MONOTONIC.clear()
        api_request_logger._LAST_EXCHANGES.clear()

    def tearDown(self) -> None:
        api_request_logger.flush_api_log()
        self.env.stop()
        api_request_logger._LAST_PRUNE_MONOTONIC.clear()
        api_request_logger._LAST_EXCHANGES.clear()
        self.temp_dir.cleanup()

    def _records(self) -> list[dict]:
        self.assertTrue(api_request_logger.flush_api_log())
        return [
            json.loads(line)
            for line in self.log_path.read_text(encoding="utf-8").splitlines()
        ]

    def test_writes_raw_request_and_response_verbatim(self) -> None:
        api_request_logger.log_api_event(
            event="request_start",
            provider="openai",
            model="gpt-test",
            trace_id="trace-1",
            attempt=1,
            max_attempts=3,
            request={
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
                "api_key": "raw-api-key-value",
                "headers": {"authorization": "Bearer raw-header-value"},
            },
        )
        api_request_logger.log_api_event(
            event="request_success",
            provider="openai",
            model="gpt-test",
            trace_id="trace-1",
            attempt=1,
            max_attempts=3,
            response=SimpleNamespace(
                id="response-1",
                _request_id="provider-request-1",
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            ),
            latency_seconds=1.25,
        )

        records = self._records()
        self.assertEqual([record["event"] for record in records], [
            "request_start",
            "request_success",
        ])
        self.assertEqual(records[0]["trace_id"], "trace-1")
        self.assertEqual(records[0]["request"]["messages"][0]["content"], "hello")
        self.assertEqual(records[0]["schema_version"], 2)
        self.assertEqual(records[0]["request"]["api_key"], "raw-api-key-value")
        self.assertEqual(
            records[0]["request"]["headers"]["authorization"],
            "Bearer raw-header-value",
        )
        self.assertEqual(records[1]["response"]["id"], "response-1")
        self.assertEqual(records[1]["provider_request_id"], "provider-request-1")
        self.assertEqual(records[1]["latency_seconds"], 1.25)

    def test_writes_four_separate_readable_last_exchange_files(self) -> None:
        run_dir = Path(self.temp_dir.name) / "one-run"
        api_request_logger.initialize_last_api_exchange_files(run_dir)
        exchange_dir = run_dir / "last_api_exchanges"
        self.assertEqual(
            {path.name for path in exchange_dir.iterdir()},
            {"agent_1.log", "user_1.log", "agent_2.log", "user_2.log"},
        )

        context = {
            "run_log_dir": str(run_dir),
            "actor_id": "agent_1",
            "operation": "respond_to_agent",
        }
        exact_system = "system first line\nsystem second line"
        exact_input = "input first line\n\ninput final line"
        exact_reply = "reply first line\n\nreply final line"
        api_request_logger.log_api_event(
            event="request_start",
            provider="anthropic",
            model="claude-test",
            trace_id="trace-last-file",
            attempt=1,
            max_attempts=1,
            request={
                "model": "claude-test",
                "max_tokens": 123,
                "system": exact_system,
                "messages": [{"role": "user", "content": exact_input}],
            },
            context=context,
        )
        pending = (exchange_dir / "agent_1.log").read_text(encoding="utf-8")
        self.assertIn(exact_system, pending)
        self.assertIn(exact_input, pending)
        self.assertIn("No API response has been recorded", pending)

        api_request_logger.log_api_event(
            event="request_success",
            provider="anthropic",
            model="claude-test",
            trace_id="trace-last-file",
            attempt=1,
            max_attempts=1,
            response={
                "id": "response-last-file",
                "content": [{"type": "text", "text": exact_reply}],
            },
            context=context,
        )
        completed = (exchange_dir / "agent_1.log").read_text(encoding="utf-8")
        self.assertIn("EVENT: request_success", completed)
        self.assertIn(exact_system, completed)
        self.assertIn(exact_input, completed)
        self.assertIn(exact_reply, completed)
        self.assertIn('"id": "response-last-file"', completed)
        self.assertIn("No API request has been recorded", (exchange_dir / "user_1.log").read_text(encoding="utf-8"))

    def test_processing_error_retains_provider_request_id_from_response(self) -> None:
        api_request_logger.log_api_event(
            event="response_processing_error",
            provider="anthropic",
            model="claude-test",
            trace_id="trace-processing",
            attempt=1,
            max_attempts=1,
            response=SimpleNamespace(_request_id="provider-processing-request"),
            error=ValueError("bad response blocks"),
        )

        record = self._records()[0]
        self.assertEqual(
            record["provider_request_id"],
            "provider-processing-request",
        )

    def test_preserves_serializable_content_and_uses_safe_object_fallback(self) -> None:
        class UnsafeObject:
            def model_dump(self):
                raise RuntimeError("cannot serialize")

            def __repr__(self) -> str:
                return "unserializable-repr-value"

        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "raw-environment-value"},
        ):
            api_request_logger.log_api_event(
                event="request_error",
                provider="openai",
                model="gpt-test",
                trace_id="trace-redaction",
                attempt=1,
                max_attempts=1,
                request={
                    "proxy-authorization": "Basic raw-header-value",
                    "client_secret": "raw-client-secret-value",
                    "message": (
                        "Authorization: Bearer raw-environment-value"
                    ),
                    "unsafe": UnsafeObject(),
                },
                error=RuntimeError(
                    "failed with Authorization: Bearer raw-error-value\n"
                    "Authorization: Basic raw-basic-value\n"
                    "Cookie: session=abc; auth=raw-cookie-value"
                ),
            )

        raw = self.log_path.read_text(encoding="utf-8") if self.log_path.exists() else ""
        if not raw:
            self.assertTrue(api_request_logger.flush_api_log())
            raw = self.log_path.read_text(encoding="utf-8")
        for preserved_content in (
            "raw-environment-value",
            "raw-header-value",
            "raw-client-secret-value",
            "raw-error-value",
            "raw-basic-value",
            "raw-cookie-value",
        ):
            self.assertIn(preserved_content, raw)
        self.assertNotIn("unserializable-repr-value", raw)
        record = json.loads(raw)
        self.assertEqual(
            record["request"]["proxy-authorization"],
            "Basic raw-header-value",
        )
        self.assertEqual(
            record["request"]["unsafe"],
            {"unserializable_type": "UnsafeObject"},
        )

    def test_preserves_plain_language_authorization_text(self) -> None:
        api_request_logger.log_api_event(
            event="request_success",
            provider="anthropic",
            model="claude-test",
            trace_id="trace-plain-authorization",
            attempt=1,
            max_attempts=1,
            response={
                "content": (
                    "Counter-proposal authorization: send the exact $24/3 terms."
                )
            },
        )

        self.assertTrue(api_request_logger.flush_api_log())
        record = self._records()[0]
        self.assertEqual(
            record["response"]["content"],
            "Counter-proposal authorization: send the exact $24/3 terms.",
        )

    def test_prunes_records_older_than_seven_days_in_the_same_file(self) -> None:
        now = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            {
                "timestamp": (now - timedelta(days=8)).isoformat(),
                "event": "old",
            },
            {
                "timestamp": (now - timedelta(days=6)).isoformat(),
                "event": "recent",
            },
        ]
        self.log_path.write_text(
            "".join(json.dumps(record) + "\n" for record in existing),
            encoding="utf-8",
        )

        with patch("negsim.api_request_logger._utc_now", return_value=now):
            api_request_logger.log_api_event(
                event="request_start",
                provider="anthropic",
                model="claude-test",
                trace_id="trace-2",
                attempt=1,
                max_attempts=1,
                request={"messages": [{"role": "user", "content": "hello"}]},
            )

        records = self._records()
        self.assertEqual([record["event"] for record in records], [
            "recent",
            "request_start",
        ])

    def test_concurrent_writes_remain_valid_json_lines(self) -> None:
        def write(index: int) -> None:
            api_request_logger.log_api_event(
                event="request_start",
                provider="anthropic",
                model="claude-test",
                trace_id=f"trace-{index}",
                attempt=1,
                max_attempts=3,
                request={"index": index},
            )

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(write, range(50)))

        records = self._records()
        self.assertEqual(len(records), 50)
        self.assertEqual(
            {record["request"]["index"] for record in records},
            set(range(50)),
        )

    def test_spawned_processes_share_one_valid_jsonl_file(self) -> None:
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(
                target=_write_from_spawned_process,
                args=(str(self.log_path), index),
            )
            for index in range(4)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=15)
            self.assertFalse(process.is_alive())
            self.assertEqual(process.exitcode, 0)

        records = self._records()
        self.assertEqual(len(records), 4)
        self.assertEqual(
            {record["request"]["index"] for record in records},
            set(range(4)),
        )

    def test_after_fork_reset_replaces_an_inherited_locked_start_lock(self) -> None:
        old_queue = api_request_logger._EVENT_QUEUE
        old_start_lock = api_request_logger._WRITER_START_LOCK
        old_writer = api_request_logger._WRITER_THREAD
        old_health_lock = api_request_logger._HEALTH_LOCK
        old_dropped = api_request_logger._DROPPED_EVENTS
        old_failures = api_request_logger._WRITE_FAILURES
        old_warning = api_request_logger._LAST_HEALTH_WARNING_MONOTONIC
        old_start_lock.acquire()
        try:
            api_request_logger._reset_after_fork()
            self.assertIsNot(api_request_logger._WRITER_START_LOCK, old_start_lock)
            self.assertTrue(api_request_logger._WRITER_START_LOCK.acquire(blocking=False))
            api_request_logger._WRITER_START_LOCK.release()
        finally:
            old_start_lock.release()
            api_request_logger._EVENT_QUEUE = old_queue
            api_request_logger._WRITER_START_LOCK = old_start_lock
            api_request_logger._WRITER_THREAD = old_writer
            api_request_logger._HEALTH_LOCK = old_health_lock
            api_request_logger._DROPPED_EVENTS = old_dropped
            api_request_logger._WRITE_FAILURES = old_failures
            api_request_logger._LAST_HEALTH_WARNING_MONOTONIC = old_warning

    def test_slow_writer_does_not_block_api_threads(self) -> None:
        writer_started = threading.Event()
        release_writer = threading.Event()
        original_append = api_request_logger._append_serialized_batch

        def slow_append(*args, **kwargs):
            writer_started.set()
            release_writer.wait(timeout=2)
            return original_append(*args, **kwargs)

        with patch(
            "negsim.api_request_logger._append_serialized_batch",
            side_effect=slow_append,
        ):
            api_request_logger.log_api_event(
                event="request_start",
                provider="openai",
                model="gpt-test",
                trace_id="trace-slow-1",
                attempt=1,
                max_attempts=1,
                request={"messages": []},
            )
            self.assertTrue(writer_started.wait(timeout=1))
            started = time.monotonic()
            api_request_logger.log_api_event(
                event="request_start",
                provider="openai",
                model="gpt-test",
                trace_id="trace-slow-2",
                attempt=1,
                max_attempts=1,
                request={"messages": []},
            )
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.25)
            release_writer.set()
            self.assertTrue(api_request_logger.flush_api_log())

    def test_failed_retention_compaction_still_appends_current_event(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        before = api_request_logger.get_api_log_health()["write_failures"]
        with patch(
            "negsim.api_request_logger._prune_locked",
            side_effect=OSError("replace unavailable"),
        ):
            api_request_logger.log_api_event(
                event="request_start",
                provider="anthropic",
                model="claude-test",
                trace_id="trace-prune-failure",
                attempt=1,
                max_attempts=1,
                request={"messages": []},
            )
            records = self._records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["trace_id"], "trace-prune-failure")
        after = api_request_logger.get_api_log_health()["write_failures"]
        self.assertEqual(after, before + 1)

    def test_logging_failures_never_escape(self) -> None:
        before = api_request_logger.get_api_log_health()["write_failures"]
        with patch(
            "negsim.api_request_logger._append_serialized_batch",
            side_effect=OSError("disk unavailable"),
        ):
            api_request_logger.log_api_event(
                event="request_start",
                provider="openai",
                model="gpt-test",
                trace_id="trace-3",
                attempt=1,
                max_attempts=1,
                request={"messages": []},
            )
            self.assertTrue(api_request_logger.flush_api_log())
        after = api_request_logger.get_api_log_health()["write_failures"]
        self.assertEqual(after, before + 1)

    def test_queue_overflow_is_counted_without_escaping(self) -> None:
        old_queue = api_request_logger._EVENT_QUEUE
        full_queue = api_request_logger.queue.Queue(maxsize=1)
        full_queue.put_nowait((self.log_path, "{}", datetime.now(timezone.utc)))
        before = api_request_logger.get_api_log_health()["dropped_events"]
        api_request_logger._EVENT_QUEUE = full_queue
        try:
            with patch("negsim.api_request_logger._ensure_writer"):
                api_request_logger.log_api_event(
                    event="request_start",
                    provider="openai",
                    model="gpt-test",
                    trace_id="trace-overflow",
                    attempt=1,
                    max_attempts=1,
                    request={"messages": []},
                )
        finally:
            api_request_logger._EVENT_QUEUE = old_queue
        after = api_request_logger.get_api_log_health()["dropped_events"]
        self.assertEqual(after, before + 1)

    def test_logging_can_be_disabled(self) -> None:
        with patch.dict("os.environ", {"RAW_API_LOG_ENABLED": "0"}):
            api_request_logger.log_api_event(
                event="request_start",
                provider="openai",
                model="gpt-test",
                trace_id="trace-4",
                attempt=1,
                max_attempts=1,
                request={"messages": []},
            )
        self.assertFalse(self.log_path.exists())


if __name__ == "__main__":
    unittest.main()
