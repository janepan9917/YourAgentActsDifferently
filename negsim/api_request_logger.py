"""Central, best-effort raw API request/response logging.

Every API attempt is written to one JSONL file. Request, response, and error
content is preserved as supplied; callers are responsible for not passing
credentials in those objects. Logging failures never affect an experiment.
Records older than the configured retention window are pruned under the same
cross-process lock used for appends.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Iterator
import uuid


_EVENT_QUEUE: queue.Queue[tuple[Path, str, datetime]] = queue.Queue(maxsize=10_000)
_WRITER_START_LOCK = threading.Lock()
_WRITER_THREAD: threading.Thread | None = None
_HEALTH_LOCK = threading.Lock()
_DROPPED_EVENTS = 0
_WRITE_FAILURES = 0
_LAST_HEALTH_WARNING_MONOTONIC = 0.0
_HEALTH_WARNING_INTERVAL_SECONDS = 60.0
_LAST_PRUNE_MONOTONIC: dict[Path, float] = {}
_PRUNE_INTERVAL_SECONDS = 60 * 60
_LOCK_TIMEOUT_SECONDS = 10.0
_LAST_EXCHANGE_LOCK = threading.Lock()
_LAST_EXCHANGES: dict[tuple[Path, str], dict[str, Any]] = {}
_LAST_EXCHANGE_ACTORS = ("agent_1", "user_1", "agent_2", "user_2")


def new_api_trace_id() -> str:
    """Return a local trace ID shared by all retries for one model query."""
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _logging_enabled() -> bool:
    value = os.environ.get("RAW_API_LOG_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _retention_days() -> int:
    raw = os.environ.get("RAW_API_LOG_RETENTION_DAYS", "7")
    try:
        return max(1, int(raw))
    except ValueError:
        return 7


def get_raw_api_log_path() -> Path:
    """Return the central log path, resolving relative paths from the repo root."""
    configured = Path(
        os.environ.get("RAW_API_LOG_PATH", "logs/raw_api_requests.jsonl")
    ).expanduser()
    if configured.is_absolute():
        return configured
    return Path(__file__).resolve().parents[1] / configured


def make_api_log_context(
    *,
    prompt_logger: Any,
    actor_id: str,
    operation: str,
    format_attempt: int,
) -> dict[str, Any]:
    """Build enough local context to locate a request even when it never returns."""
    context: dict[str, Any] = {
        "actor_id": actor_id,
        "operation": operation,
        "format_attempt": format_attempt,
    }
    if prompt_logger is not None:
        game_id = getattr(prompt_logger, "game_id", None)
        if game_id:
            context["game_id"] = str(game_id)
        log_dir = getattr(prompt_logger, "log_dir", None)
        if log_dir is not None:
            try:
                context["run_log_dir"] = str(Path(log_dir).resolve())
            except Exception:
                context["run_log_dir"] = str(log_dir)
    return context


def get_api_log_health() -> dict[str, int]:
    """Return process-local counts of events the best-effort logger could not persist."""
    with _HEALTH_LOCK:
        return {
            "dropped_events": _DROPPED_EVENTS,
            "write_failures": _WRITE_FAILURES,
        }


def _record_logging_failure(kind: str, error: BaseException | None = None) -> None:
    global _DROPPED_EVENTS, _LAST_HEALTH_WARNING_MONOTONIC, _WRITE_FAILURES
    now = time.monotonic()
    should_warn = False
    with _HEALTH_LOCK:
        if kind == "queue_full":
            _DROPPED_EVENTS += 1
        else:
            _WRITE_FAILURES += 1
        if now - _LAST_HEALTH_WARNING_MONOTONIC >= _HEALTH_WARNING_INTERVAL_SECONDS:
            _LAST_HEALTH_WARNING_MONOTONIC = now
            should_warn = True
    if should_warn:
        try:
            detail = ""
            if error is not None:
                detail = f": {error}"
            print(
                f"[raw-api-log warning] {kind}{detail}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass


def _unserializable(value: Any) -> dict[str, str]:
    return {"unserializable_type": type(value).__name__}


def _plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _plain_data(value.model_dump())
        except Exception:
            return _unserializable(value)
    if isinstance(value, dict):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain_data(item) for item in value]
    if hasattr(value, "__dict__"):
        try:
            return _plain_data(vars(value))
        except Exception:
            return _unserializable(value)
    return _unserializable(value)


def _write_verbatim_text(buffer: io.StringIO, value: str) -> None:
    """Write a string without stripping, wrapping, or normalizing its content."""
    buffer.write(value)
    if not value.endswith(("\n", "\r")):
        buffer.write("\n")


def _pretty_json(value: Any) -> str:
    return json.dumps(
        _plain_data(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _response_texts(response: Any) -> list[str]:
    """Extract provider text fields for verbatim human-readable display."""
    if not isinstance(response, dict):
        return []
    texts: list[str] = []
    choices = response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                texts.append(message["content"])
    content = response.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
    return texts


def _render_last_exchange(actor_id: str, record: dict[str, Any]) -> str:
    """Render one actor's last actual provider attempt without changing values."""
    buffer = io.StringIO(newline="")
    buffer.write(f"ACTOR: {actor_id}\n")
    for field in (
        "event",
        "provider",
        "model",
        "trace_id",
        "attempt",
        "max_attempts",
        "timestamp",
        "provider_request_id",
    ):
        if field in record:
            buffer.write(f"{field.upper()}: {record[field]}\n")
    context = record.get("context")
    if isinstance(context, dict):
        for field in ("game_id", "operation", "format_attempt"):
            if field in context:
                buffer.write(f"{field.upper()}: {context[field]}\n")

    request = record.get("request")
    buffer.write("\n=== API REQUEST ===\n")
    if not isinstance(request, dict):
        buffer.write("No API request has been recorded for this participant.\n")
    else:
        parameters = {
            key: value
            for key, value in request.items()
            if key not in {"system", "messages"}
        }
        buffer.write("\n--- REQUEST PARAMETERS ---\n")
        _write_verbatim_text(buffer, _pretty_json(parameters))
        if "system" in request:
            buffer.write("\n--- SYSTEM ---\n")
            system = request["system"]
            if isinstance(system, str):
                _write_verbatim_text(buffer, system)
            else:
                _write_verbatim_text(buffer, _pretty_json(system))
        messages = request.get("messages")
        if isinstance(messages, list):
            for index, message in enumerate(messages):
                role = message.get("role") if isinstance(message, dict) else None
                buffer.write(f"\n--- MESSAGE {index} (role={role}) ---\n")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    _write_verbatim_text(buffer, message["content"])
                    metadata = {
                        key: value
                        for key, value in message.items()
                        if key != "content"
                    }
                    buffer.write(f"\n--- MESSAGE {index} METADATA ---\n")
                    _write_verbatim_text(buffer, _pretty_json(metadata))
                else:
                    _write_verbatim_text(buffer, _pretty_json(message))

    response = record.get("response")
    error = record.get("error")
    buffer.write("\n=== API RESPONSE ===\n")
    if response is not None:
        texts = _response_texts(response)
        for index, text in enumerate(texts):
            buffer.write(f"\n--- RESPONSE TEXT {index} ---\n")
            _write_verbatim_text(buffer, text)
        buffer.write("\n--- FULL RESPONSE OBJECT ---\n")
        _write_verbatim_text(buffer, _pretty_json(response))
    elif error is not None:
        buffer.write("\n--- ERROR RETURNED/RAISED ---\n")
        _write_verbatim_text(buffer, _pretty_json(error))
    else:
        buffer.write("No API response has been recorded for this provider attempt.\n")
    return buffer.getvalue()


def _write_last_exchange_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        os.replace(temp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def initialize_last_api_exchange_files(run_log_dir: str | Path) -> None:
    """Create the four per-participant last-exchange files for one run."""
    root = Path(run_log_dir).resolve() / "last_api_exchanges"
    try:
        with _LAST_EXCHANGE_LOCK:
            for actor_id in _LAST_EXCHANGE_ACTORS:
                path = root / f"{actor_id}.log"
                if path.exists():
                    continue
                _write_last_exchange_file(
                    path,
                    _render_last_exchange(actor_id, {"event": "no_request"}),
                )
    except Exception as error:
        _record_logging_failure("last_exchange_initialization_failure", error)


def _update_last_api_exchange(record: dict[str, Any]) -> None:
    context = record.get("context")
    if not isinstance(context, dict):
        return
    actor_id = context.get("actor_id")
    run_log_dir = context.get("run_log_dir")
    if actor_id not in _LAST_EXCHANGE_ACTORS or not isinstance(run_log_dir, str):
        return

    run_dir = Path(run_log_dir).resolve()
    key = (run_dir, actor_id)
    with _LAST_EXCHANGE_LOCK:
        if record.get("event") == "request_start":
            snapshot = {
                field: record[field]
                for field in (
                    "timestamp",
                    "event",
                    "provider",
                    "model",
                    "trace_id",
                    "attempt",
                    "max_attempts",
                    "request",
                    "context",
                )
                if field in record
            }
        else:
            previous = _LAST_EXCHANGES.get(key, {})
            same_attempt = (
                previous.get("trace_id") == record.get("trace_id")
                and previous.get("attempt") == record.get("attempt")
            )
            snapshot = dict(previous) if same_attempt else {}
            for field in (
                "timestamp",
                "event",
                "provider",
                "model",
                "trace_id",
                "attempt",
                "max_attempts",
                "provider_request_id",
                "response",
                "error",
                "context",
            ):
                if field in record:
                    snapshot[field] = record[field]
        _LAST_EXCHANGES[key] = snapshot
        path = run_dir / "last_api_exchanges" / f"{actor_id}.log"
        _write_last_exchange_file(path, _render_last_exchange(actor_id, snapshot))


def _error_data(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    for attribute in ("status_code", "code", "body"):
        value = getattr(error, attribute, None)
        if value is not None:
            payload[attribute] = _plain_data(value)
    return payload


def get_provider_request_id(
    response: Any = None,
    error: BaseException | None = None,
) -> str | None:
    candidates = (error, response)
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("request_id", "_request_id", "id"):
            try:
                value = getattr(candidate, attribute, None)
            except Exception:
                continue
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value)
    return None


@contextmanager
def _exclusive_process_lock(log_path: Path) -> Iterator[None]:
    """Serialize appends/pruning across threads and independent sweep processes."""
    lock_path = log_path.with_name(f"{log_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    acquired = False
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)

        if os.name == "nt":
            import msvcrt

            while not acquired:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out locking raw API log {log_path}")
                    time.sleep(0.05)
        else:
            import fcntl

            while not acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out locking raw API log {log_path}")
                    time.sleep(0.05)

        yield
    finally:
        if acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        handle.close()


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _prune_locked(log_path: Path, *, now: datetime) -> None:
    if not log_path.exists():
        return
    cutoff = now - timedelta(days=_retention_days())
    temp_path = log_path.with_name(
        f".{log_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with (
            open(log_path, "r", encoding="utf-8") as source,
            open(temp_path, "w", encoding="utf-8", newline="\n") as destination,
        ):
            try:
                os.chmod(temp_path, 0o600)
            except OSError:
                pass
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                timestamp = _parse_timestamp(record.get("timestamp"))
                if timestamp is not None and timestamp >= cutoff:
                    destination.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    destination.write("\n")
        os.replace(temp_path, log_path)
        try:
            os.chmod(log_path, 0o600)
        except OSError:
            pass
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _append_serialized_batch(
    log_path: Path,
    serialized_records: list[str],
    *,
    now: datetime,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path = log_path.resolve()

    with _exclusive_process_lock(log_path):
        monotonic_now = time.monotonic()
        last_prune = _LAST_PRUNE_MONOTONIC.get(resolved_path)
        if (
            last_prune is None
            or monotonic_now - last_prune >= _PRUNE_INTERVAL_SECONDS
        ):
            marker_path = log_path.with_name(f"{log_path.name}.retention")
            marker_timestamp: float | None = None
            try:
                marker_timestamp = float(marker_path.read_text(encoding="ascii"))
            except (OSError, ValueError):
                pass
            if (
                marker_timestamp is None
                or now.timestamp() - marker_timestamp >= _PRUNE_INTERVAL_SECONDS
            ):
                try:
                    _prune_locked(log_path, now=now)
                    marker_path.write_text(str(now.timestamp()), encoding="ascii")
                except Exception as error:
                    # A failed compaction must not discard the current event.
                    _record_logging_failure("retention_compaction_failure", error)
            _LAST_PRUNE_MONOTONIC[resolved_path] = monotonic_now

        created = not log_path.exists()
        with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
            for serialized in serialized_records:
                handle.write(serialized)
                handle.write("\n")
            handle.flush()
        if created:
            try:
                os.chmod(log_path, 0o600)
            except OSError:
                pass


def _writer_main() -> None:
    while True:
        try:
            first = _EVENT_QUEUE.get(timeout=_PRUNE_INTERVAL_SECONDS)
        except queue.Empty:
            try:
                log_path = get_raw_api_log_path()
                if log_path.exists():
                    _append_serialized_batch(log_path, [], now=_utc_now())
            except Exception as error:
                _record_logging_failure("retention_maintenance_failure", error)
            continue
        batch = [first]
        while len(batch) < 100:
            try:
                batch.append(_EVENT_QUEUE.get_nowait())
            except queue.Empty:
                break

        try:
            grouped: dict[Path, tuple[list[str], datetime]] = {}
            for log_path, serialized, now in batch:
                records, latest_now = grouped.setdefault(log_path, ([], now))
                records.append(serialized)
                if now > latest_now:
                    grouped[log_path] = (records, now)

            for log_path, (records, latest_now) in grouped.items():
                try:
                    _append_serialized_batch(log_path, records, now=latest_now)
                except Exception as error:
                    _record_logging_failure("write_failure", error)
        except Exception as error:
            _record_logging_failure("writer_batch_failure", error)
        finally:
            for _ in batch:
                _EVENT_QUEUE.task_done()


def _ensure_writer() -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD is not None and _WRITER_THREAD.is_alive():
        return
    with _WRITER_START_LOCK:
        if _WRITER_THREAD is not None and _WRITER_THREAD.is_alive():
            return
        _WRITER_THREAD = threading.Thread(
            target=_writer_main,
            name="raw-api-log-writer",
            daemon=True,
        )
        _WRITER_THREAD.start()


def flush_api_log(timeout_seconds: float = 10.0) -> bool:
    """Wait briefly for queued events; primarily used at process exit and in tests."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    with _EVENT_QUEUE.all_tasks_done:
        while _EVENT_QUEUE.unfinished_tasks:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _EVENT_QUEUE.all_tasks_done.wait(timeout=remaining)
    return True


def log_api_event(
    *,
    event: str,
    provider: str,
    model: str,
    trace_id: str,
    attempt: int,
    max_attempts: int,
    request: dict[str, Any] | None = None,
    response: Any = None,
    error: BaseException | None = None,
    latency_seconds: float | None = None,
    retryable: bool | None = None,
    will_retry: bool | None = None,
    context: dict[str, Any] | None = None,
) -> None:
    """Append one raw API event without ever interrupting the caller."""
    if not _logging_enabled():
        return
    try:
        now = _utc_now()
        record: dict[str, Any] = {
            "schema_version": 2,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
            "event": event,
            "provider": provider,
            "model": model,
            "trace_id": trace_id,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
        }
        if request is not None:
            record["request"] = request
        if response is not None:
            record["response"] = response
        if error is not None:
            record["error"] = _error_data(error)
        provider_request_id = get_provider_request_id(response=response, error=error)
        if provider_request_id is not None:
            record["provider_request_id"] = provider_request_id
        if context is not None:
            record["context"] = context
        if latency_seconds is not None:
            record["latency_seconds"] = round(latency_seconds, 6)
        if retryable is not None:
            record["retryable"] = retryable
        if will_retry is not None:
            record["will_retry"] = will_retry
        plain_record = _plain_data(record)
        try:
            _update_last_api_exchange(plain_record)
        except Exception as error:
            _record_logging_failure("last_exchange_write_failure", error)
        serialized = json.dumps(
            plain_record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        _ensure_writer()
        try:
            _EVENT_QUEUE.put_nowait((get_raw_api_log_path(), serialized, now))
        except queue.Full:
            _record_logging_failure("queue_full")
    except Exception as error:
        # Observability must never change an experiment's outcome.
        _record_logging_failure("serialization_or_enqueue_failure", error)
        return


def _reset_after_fork() -> None:
    """Discard inherited thread/queue state in a POSIX fork child."""
    global _DROPPED_EVENTS, _EVENT_QUEUE, _HEALTH_LOCK
    global _LAST_EXCHANGE_LOCK
    global _LAST_HEALTH_WARNING_MONOTONIC, _WRITE_FAILURES
    global _WRITER_START_LOCK, _WRITER_THREAD
    _EVENT_QUEUE = queue.Queue(maxsize=10_000)
    _HEALTH_LOCK = threading.Lock()
    _DROPPED_EVENTS = 0
    _WRITE_FAILURES = 0
    _LAST_HEALTH_WARNING_MONOTONIC = 0.0
    _WRITER_START_LOCK = threading.Lock()
    _WRITER_THREAD = None
    _LAST_EXCHANGE_LOCK = threading.Lock()
    _LAST_EXCHANGES.clear()
    _LAST_PRUNE_MONOTONIC.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
atexit.register(flush_api_log)
