"""Shared helpers for post-hoc evaluator scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Hashable, Iterable, Mapping

import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from negsim.log_paths import iter_run_dirs  # noqa: E402
from negsim.run_outcomes import is_evaluation_excluded_outcome  # noqa: E402
from negsim.model import openai_prompt_cache_control_kwargs  # noqa: E402


THREAD_LOCAL = threading.local()
API_SEMAPHORE: threading.BoundedSemaphore | None = None
API_SEMAPHORE_LIMIT: int | None = None
API_SEMAPHORE_LOCK = threading.Lock()
JUDGE_API_USAGE_LOG_LOCK = threading.Lock()


class JudgeOutputError(RuntimeError):
    """A completed judge call whose model output cannot be used."""

    def __init__(
        self,
        message: str,
        *,
        error_kind: str,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.raw_response = raw_response


def judge_cache_control_kwargs(model: str) -> dict[str, Any]:
    """Disable GPT-5.6+ prompt-cache writes for one-off judge requests.

    The installed OpenAI SDK predates the typed ``prompt_cache_options``
    parameter, so send the supported top-level API field through ``extra_body``.
    Explicit mode without any explicit breakpoint disables both implicit cache
    reads and writes. Earlier models do not support this control and do not
    charge an additional cache-write rate.
    """
    return openai_prompt_cache_control_kwargs(model)


def env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


def get_api_semaphore() -> threading.BoundedSemaphore | None:
    """Limit concurrent judge API calls while allowing high worker counts."""
    global API_SEMAPHORE, API_SEMAPHORE_LIMIT

    limit = env_int("OPENAI_API_MAX_CONCURRENT_REQUESTS", 20, minimum=0)
    if limit <= 0:
        return None

    with API_SEMAPHORE_LOCK:
        if API_SEMAPHORE is None or API_SEMAPHORE_LIMIT != limit:
            API_SEMAPHORE = threading.BoundedSemaphore(limit)
            API_SEMAPHORE_LIMIT = limit
        return API_SEMAPHORE


@dataclass
class RunBundle:
    log_dir: Path
    summary_log: str
    results: dict[str, Any]
    config: dict[str, Any]

    @property
    def game_id(self) -> str:
        return str(self.results.get("game_id") or self.log_dir.name)

    @property
    def game_name(self) -> str:
        return str(self.results.get("game_name") or self.config.get("game_name") or "")


def load_simple_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_text(path: Path) -> str:
    if not path.exists():
        return f"[missing file: {path}]"
    return path.read_text(encoding="utf-8", errors="replace")


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_run_bundle(log_dir: Path) -> RunBundle:
    results_path = log_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.json in {log_dir}")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    config = load_yaml(log_dir / "config.yaml")
    summary = read_text(log_dir / "summary.log")
    return RunBundle(log_dir=log_dir, summary_log=summary, results=results, config=config)


def _is_evaluation_excluded_run(log_dir: Path) -> bool:
    """Return True for readable runs excluded from evaluator inputs."""

    results_path = log_dir / "results.json"
    if not results_path.exists():
        return False
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(results, dict):
        return False
    return is_evaluation_excluded_outcome(results.get("outcome"))


def game_ids_for_log_dirs(log_dirs: Iterable[Path]) -> set[str]:
    """Read stable game IDs without loading large summaries or run configs."""
    game_ids: set[str] = set()
    for log_dir in log_dirs:
        results = json.loads((log_dir / "results.json").read_text(encoding="utf-8"))
        game_ids.add(str(results.get("game_id") or log_dir.name))
    return game_ids


def discover_log_dirs(
    *,
    logs: Path,
    sweep: str = "",
    limit: int | None = None,
) -> list[Path]:
    if limit is not None and limit <= 0:
        return []
    candidates: list[Path] = []
    for log_dir in iter_run_dirs(logs, sweep_id=sweep):
        if _is_evaluation_excluded_run(log_dir):
            continue
        candidates.append(log_dir)
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def protocol_summary(game_name: str) -> str:
    summaries = {
        "buy_sell_harness_fixed": (
            "Standalone JSON-native buy/sell game. Final agreement has exact top-level "
            "contribution labels 'Player RED Gives' and 'Player BLUE Gives'. RED gives the "
            "configured commodity quantity; BLUE gives a non-negative integer ZUP price. "
            "These fields are contributions, not final holdings. Seller payout is "
            "price minus production cost; buyer payout is willingness-to-pay minus price "
            "(scaled by quantity). No deal pays zero to both parties."
        ),
        "ultimatum_harness_fixed": (
            "Standalone JSON-native multi-turn ultimatum. Final agreement uses exact "
            "contribution labels 'Player RED Gives' and 'Player BLUE Gives'. If RED Gives "
            "Dollars N and BLUE Gives Dollars 0, BLUE receives N and RED retains total-N. "
            "The dict records transfers, not final holdings. No deal pays zero to both parties."
        ),
    }
    return summaries.get(game_name, "Use the game config and final_deal fields to infer the protocol schema.")


def common_metadata(bundle: RunBundle) -> dict[str, Any]:
    results = bundle.results
    result_metadata = results.get("result_metadata") or {}
    return {
        "game_id": results.get("game_id", bundle.log_dir.name),
        "game_name": bundle.game_name,
        "config": results.get("config_name", ""),
        "label": results.get("label") or result_metadata.get("label", ""),
        "base_id": results.get("base_id") or result_metadata.get("base_id", ""),
        "base_label": results.get("base_label") or result_metadata.get("base_label", ""),
        "combo_id": results.get("combo_id") or result_metadata.get("combo_id", ""),
        "source_summary_chars": len(bundle.summary_log),
        "input_truncated": False,
        "evaluation_input_policy": "complete_summary_log",
    }


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def get_api_key(api_key_name: str) -> str:
    load_simple_env()
    api_key = os.environ.get(api_key_name)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {api_key_name!r}.")
    return api_key


def thread_client(api_key: str):
    client = getattr(THREAD_LOCAL, "openai_client", None)
    if client is None:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            timeout=env_int("OPENAI_API_TIMEOUT_SECONDS", 120, minimum=1),
            max_retries=0,
        )
        THREAD_LOCAL.openai_client = client
    return client


def is_retryable(error: Exception) -> bool:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
    except Exception:
        return False
    if isinstance(error, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in {408, 409, 429} or error.status_code >= 500
    return False


def _append_judge_api_usage(
    path: str,
    *,
    call_id: str,
    prompt_sha256: str,
    model: str,
    attempt: int,
    started_at: str,
    elapsed_seconds: float,
    system_chars: int,
    user_chars: int,
    response: Any,
    error: Exception | None,
) -> None:
    """Append opt-in accounting without leaking request/response text or secrets.

    Each entry describes one SDK ``create`` attempt (SDK retries are disabled).
    A completed response is recorded even if its output cannot be parsed or used.
    Logging must never retry a successful, already billable API request.
    """
    try:
        usage = getattr(response, "usage", None)
        if usage is not None and callable(getattr(usage, "model_dump", None)):
            usage = usage.model_dump(mode="json")
        elif isinstance(usage, Mapping):
            usage = dict(usage)
        elif usage is not None:
            raise TypeError("Unsupported judge usage object")
        choices = getattr(response, "choices", None) or []
        row = {
            "timestamp": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "call_id": call_id,
            "prompt_sha256": prompt_sha256,
            "model": model,
            "attempt": attempt,
            "latency_seconds": elapsed_seconds,
            "status": "error" if error is not None else "success",
            "error_type": type(error).__name__ if error is not None else None,
            "error_kind": getattr(error, "error_kind", None),
            "status_code": getattr(error, "status_code", None),
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "finish_reason": getattr(choices[0], "finish_reason", None) if choices else None,
            "service_tier": getattr(response, "service_tier", None),
            "usage": usage,
            "system_prompt_chars": system_chars,
            "user_prompt_chars": user_chars,
            "messages_input_chars": system_chars + user_chars,
        }
        line = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with JUDGE_API_USAGE_LOG_LOCK:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
    except Exception as log_error:
        # Do not log exception bodies: a provider exception may contain prompts
        # or authentication data. A failed sidecar does not alter API behavior.
        try:
            print(
                "WARNING: JUDGE_API_USAGE_LOG_PATH accounting append failed "
                f"({type(log_error).__name__}); accounting may be incomplete. "
                "The judge request will not be retried because of this logging failure.",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            # Even a closed stderr must not turn accounting into a paid retry.
            pass


def query_llm_json(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format or {"type": "json_object"},
        **judge_cache_control_kwargs(model),
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if max_completion_tokens is not None:
        request["max_completion_tokens"] = max_completion_tokens
    usage_path = os.environ.get("JUDGE_API_USAGE_LOG_PATH", "").strip()
    # Disabled by default: no hashing, file creation, or additional API fields.
    call_id = uuid.uuid4().hex if usage_path else ""
    prompt_sha256 = (
        hashlib.sha256(
            json.dumps([system_prompt, user_prompt], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if usage_path else ""
    )
    max_retries = env_int("OPENAI_API_MAX_RETRIES", 6, minimum=1)
    for attempt in range(max_retries):
        response = None
        started_at = ""
        started = None
        try:
            semaphore = get_api_semaphore()
            with semaphore if semaphore is not None else nullcontext():
                client = thread_client(api_key)
                if usage_path:
                    started_at = datetime.now(timezone.utc).isoformat()
                    started = time.perf_counter()
                response = client.chat.completions.create(**request)
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise JudgeOutputError(
                    "LLM judge response was truncated at the token limit.",
                    error_kind="truncated",
                )
            message = choice.message
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise JudgeOutputError(
                    f"LLM judge refused the request: {refusal}",
                    error_kind="refusal",
                )
            content = message.content
            if not content:
                raise JudgeOutputError(
                    "LLM judge returned an empty response.",
                    error_kind="empty",
                )
            try:
                result = extract_json_object(content)
            except (json.JSONDecodeError, ValueError) as error:
                raise JudgeOutputError(
                    f"LLM judge returned invalid JSON: {error}",
                    error_kind="malformed_json",
                    raw_response=content,
                ) from error
        except Exception as exc:
            if usage_path and started is not None:
                _append_judge_api_usage(
                    usage_path, call_id=call_id, prompt_sha256=prompt_sha256,
                    model=model, attempt=attempt + 1, started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    system_chars=len(system_prompt), user_chars=len(user_prompt),
                    response=response, error=exc,
                )
            if attempt == max_retries - 1 or not is_retryable(exc):
                raise
            time.sleep(1.0 * (2**attempt) + random.uniform(0, 2 + attempt))
        else:
            if usage_path and started is not None:
                _append_judge_api_usage(
                    usage_path, call_id=call_id, prompt_sha256=prompt_sha256,
                    model=model, attempt=attempt + 1, started_at=started_at,
                    elapsed_seconds=time.perf_counter() - started,
                    system_chars=len(system_prompt), user_chars=len(user_prompt),
                    response=response, error=None,
                )
            return result
    return {}


def load_existing_outputs(out_dir: Path, stem: str) -> list[dict[str, Any]]:
    """Load prior evaluator outputs, preferring JSONL over CSV."""
    jsonl_path = out_dir / f"{stem}.jsonl"
    csv_path = out_dir / f"{stem}.csv"
    if jsonl_path.exists():
        rows = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {jsonl_path}:{line_no}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if csv_path.exists():
        return pd.read_csv(csv_path).to_dict(orient="records")
    return []


def _row_game_id(row: Mapping[str, Any]) -> str:
    value = row.get("game_id", "")
    if value is None:
        return ""
    text = str(value)
    return "" if text == "nan" else text


def completed_game_ids(
    existing_rows: list[dict[str, Any]],
    expected_keys_by_game_id: Mapping[str, set[Hashable]],
    *,
    row_key: Callable[[dict[str, Any]], Hashable],
) -> set[str]:
    """Return game_ids whose existing rows fully cover expected evaluator keys.

    Rows with an ``error`` field never count as complete. Empty expected-key sets
    are considered complete so resume mode can skip runs with no applicable work.
    """
    rows_by_game_id: dict[str, list[dict[str, Any]]] = {}
    for row in existing_rows:
        game_id = _row_game_id(row)
        if game_id:
            rows_by_game_id.setdefault(game_id, []).append(row)

    complete: set[str] = set()
    for game_id, expected_keys in expected_keys_by_game_id.items():
        game_id = str(game_id)
        if not expected_keys:
            complete.add(game_id)
            continue
        rows = rows_by_game_id.get(game_id, [])
        if not rows or any("error" in row and row.get("error") for row in rows):
            continue
        actual_keys = {row_key(row) for row in rows}
        if expected_keys.issubset(actual_keys):
            complete.add(game_id)
    return complete


def merge_existing_and_new_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    *,
    eligible_game_ids: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Merge rows and optionally discard outputs outside the current input set."""
    if eligible_game_ids is not None:
        eligible = {str(game_id) for game_id in eligible_game_ids}
        existing_rows = [
            row for row in existing_rows if _row_game_id(row) in eligible
        ]
        new_rows = [row for row in new_rows if _row_game_id(row) in eligible]
    new_game_ids = {_row_game_id(row) for row in new_rows}
    new_game_ids.discard("")
    if not new_game_ids:
        return list(existing_rows)
    kept = [row for row in existing_rows if _row_game_id(row) not in new_game_ids]
    return kept + new_rows


def failed_companion_output_dir(out_dir: Path, sweep_id: str = "") -> Path:
    """Return the sibling directory used to archive excluded evaluator rows."""
    name = out_dir.name or "evaluation"
    normalized_sweep_id = str(sweep_id or "")
    if normalized_sweep_id and name.startswith(normalized_sweep_id):
        failed_name = (
            f"{normalized_sweep_id}_failed"
            f"{name[len(normalized_sweep_id):]}"
        )
    else:
        failed_name = f"{name}_failed"
    return out_dir.with_name(failed_name)


def _deduplicate_output_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        identity = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(row)
    return deduplicated


def archive_ineligible_existing_rows(
    existing_rows: list[dict[str, Any]],
    *,
    eligible_game_ids: Collection[str],
    out_dir: Path,
    stem: str,
    sweep_id: str = "",
) -> list[dict[str, Any]]:
    """Archive prior rows outside current eligibility and return retained rows.

    Incoming rows replace every already-archived row for the same game ID.
    Both persisted formats are rewritten from the same deduplicated row list.
    """
    eligible = {str(game_id) for game_id in eligible_game_ids}
    retained = [
        row for row in existing_rows if _row_game_id(row) in eligible
    ]
    ineligible = [
        row for row in existing_rows if _row_game_id(row) not in eligible
    ]
    failed_dir = failed_companion_output_dir(out_dir, sweep_id)
    failed_output_exists = any(
        (failed_dir / f"{stem}{suffix}").exists()
        for suffix in (".jsonl", ".csv")
    )
    if not ineligible and not failed_output_exists:
        return retained

    archived = load_existing_outputs(failed_dir, stem)
    archived = [
        row
        for row in archived
        if _row_game_id(row) not in eligible or row.get("error")
    ]
    archived = merge_existing_and_new_rows(archived, ineligible)
    archived = _deduplicate_output_rows(archived)
    write_outputs(archived, failed_dir, stem)
    return retained


def archive_failed_output_rows(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    stem: str,
    sweep_id: str = "",
) -> list[dict[str, Any]]:
    """Archive evaluator error rows and return only successful output rows.

    Failed judge attempts remain append-only (with exact duplicate rows removed)
    in the failed companion directory. A later successful ``--continue`` run
    therefore replaces the main-output error with complete result rows without
    erasing the earlier judge failure record.
    """

    successful = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    if not failed:
        return successful

    failed_dir = failed_companion_output_dir(out_dir, sweep_id)
    archived = load_existing_outputs(failed_dir, stem)
    archived.extend(failed)
    archived = _deduplicate_output_rows(archived)
    write_outputs(archived, failed_dir, stem)
    return successful


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{stem}.jsonl"
    csv_path = out_dir / f"{stem}.csv"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  saved {jsonl_path}")
    print(f"  saved {csv_path}")
    return jsonl_path, csv_path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--logs",
        default="logs",
        help="Logs root containing sweep/combo/run directories.",
    )
    parser.add_argument("--sweep", default="", help="Only evaluate this sweep_id.")
    parser.add_argument("--out", default="judge_labels", help="Output directory.")
    parser.add_argument("--limit", type=int, help="Maximum number of runs to evaluate.")
    parser.add_argument("--model", help="Judge model.")
    parser.add_argument("--api-key-name", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument(
        "--continue",
        dest="continue_eval",
        action="store_true",
        help=(
            "Resume an evaluator output directory by skipping runs that already "
            "have complete non-error rows and appending only missing runs."
        ),
    )


def run_parallel(
    bundles: list[RunBundle],
    worker: Callable[[RunBundle], list[dict[str, Any]]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not bundles:
        return rows
    max_workers = max(1, min(workers, len(bundles)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, bundle): bundle for bundle in bundles}
        completed = 0
        for future in as_completed(futures):
            bundle = futures[future]
            completed += 1
            try:
                part = future.result()
                rows.extend(part)
                print(f"[{completed}/{len(bundles)}] done: {bundle.game_id}")
            except Exception as exc:
                print(f"[{completed}/{len(bundles)}] error: {bundle.game_id}: {exc}")
                rows.append({**common_metadata(bundle), "error": repr(exc)})
    return rows
