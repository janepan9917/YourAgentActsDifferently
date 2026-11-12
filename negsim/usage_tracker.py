"""
Thread-safe global token-usage accumulator, keyed by model name.

Usage:
    from negsim.usage_tracker import record_usage, get_usage_summary, reset_usage

    record_usage(model_name, usage)   # usage = OpenAI SDK response.usage (or dict)
    ...
    print(get_usage_summary())
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_USAGE: dict[str, dict[str, int]] = {}


def _as_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {}


def record_usage(model_name: str, usage: Any) -> None:
    """Add one API call's usage to the running total for model_name."""
    u = _as_dict(usage)
    if not u:
        return
    prompt_tokens = u.get("prompt_tokens", 0) or 0
    completion_tokens = u.get("completion_tokens", 0) or 0
    total_tokens = u.get("total_tokens", prompt_tokens + completion_tokens)
    details = u.get("completion_tokens_details") or {}
    if hasattr(details, "model_dump"):
        details = details.model_dump()
    reasoning_tokens = (details or {}).get("reasoning_tokens", 0) or 0

    with _LOCK:
        entry = _USAGE.setdefault(model_name, {
            "calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0,
        })
        entry["calls"] += 1
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["reasoning_tokens"] += reasoning_tokens
        entry["total_tokens"] += total_tokens


def get_usage_summary() -> dict[str, dict[str, int]]:
    with _LOCK:
        return {k: dict(v) for k, v in _USAGE.items()}


def reset_usage() -> None:
    with _LOCK:
        _USAGE.clear()
