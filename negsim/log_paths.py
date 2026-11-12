"""Log paths shared by the sweep runner and the judges.

Each run is written to::

    logs/<sweep>/<combo>/<run_id>/
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LOGS_DIR = Path("logs")
UNGROUPED_SWEEP_ID = "_ungrouped"
UNKNOWN_COMBO_ID = "unknown_combo"
MAX_GROUP_DIR_NAME_LENGTH = 160

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_WINDOWS_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def sanitize_sweep_id(
    sweep_id: str,
    *,
    max_length: int = MAX_GROUP_DIR_NAME_LENGTH,
) -> str:
    """Return a readable, filesystem-safe directory component."""
    safe = str(sweep_id).strip()
    safe = INVALID_FILENAME_CHARS.sub("_", safe)
    safe = re.sub(r"\s+", "_", safe)
    safe = safe.rstrip(" .")

    if safe in {"", ".", ".."}:
        safe = "unknown_sweep"

    stem = safe.split(".", 1)[0].upper()
    if stem in RESERVED_WINDOWS_NAMES:
        safe = f"_{safe}"

    if len(safe) > max_length:
        digest = _short_hash(str(sweep_id))
        keep = max(1, max_length - len(digest) - 2)
        safe = safe[:keep].rstrip(" ._") or "group"
        safe = f"{safe}__{digest}"

    return safe


def group_folder_name(group_id: object, *, fallback: str) -> str:
    """Return the deterministic directory name for one sweep or combo id.

    Unsafe ids receive a stable hash suffix. Therefore ``a:b`` and ``a?b``
    cannot collide even when each run is created independently.
    """
    raw = str(group_id).strip() if group_id not in (None, "") else fallback
    safe = sanitize_sweep_id(raw)
    if safe == raw:
        return safe

    digest = _short_hash(raw)
    if safe.endswith(f"__{digest}"):
        return safe
    keep = max(1, MAX_GROUP_DIR_NAME_LENGTH - len(digest) - 2)
    stem = safe[:keep].rstrip(" ._") or "group"
    return f"{stem}__{digest}"


def sweep_folder_name(sweep_id: object) -> str:
    return group_folder_name(sweep_id, fallback=UNGROUPED_SWEEP_ID)


def combo_folder_name(combo_id: object) -> str:
    return group_folder_name(combo_id, fallback=UNKNOWN_COMBO_ID)


def combo_id_from_config(config: dict[str, Any]) -> str:
    metadata = config.get("result_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    value = config.get("combo_id") or metadata.get("combo_id")
    return str(value).strip() if value not in (None, "") else UNKNOWN_COMBO_ID


def canonical_logs_root(configured: str | Path | None = None) -> Path:
    """Resolve the log root (``logs`` when none is configured)."""
    if configured in (None, ""):
        return DEFAULT_LOGS_DIR
    return Path(configured)


def run_log_dir(root: str | Path, sweep_id: object, combo_id: object, run_id: str) -> Path:
    """Build the canonical directory for a run without touching the filesystem."""
    return (
        canonical_logs_root(root)
        / sweep_folder_name(sweep_id)
        / combo_folder_name(combo_id)
        / run_id
    )


def resolve_sweep_dir(root: str | Path, sweep_id: object) -> Path:
    """Resolve a sweep directory without scanning sibling sweeps."""
    root = canonical_logs_root(root)
    raw = str(sweep_id).strip() if sweep_id not in (None, "") else UNGROUPED_SWEEP_ID
    return root / sweep_folder_name(raw)


def iter_results_paths(root: str | Path, *, sweep_id: object = "") -> Iterable[Path]:
    """Iterate canonical results files, using a bounded-depth sweep lookup."""
    root = canonical_logs_root(root)
    if sweep_id not in (None, ""):
        root_is_sweep = root.name == sweep_folder_name(str(sweep_id).strip())
        sweep_dir = root if root_is_sweep else resolve_sweep_dir(root, sweep_id)
        if not sweep_dir.is_dir():
            return iter(())
        return iter(sorted(sweep_dir.glob("*/*/results.json")))

    if not root.is_dir():
        return iter(())
    paths = sorted(root.glob("*/*/*/results.json"))
    if not paths:
        # Also accept a caller that already supplied one sweep directory.
        paths = sorted(root.glob("*/*/results.json"))
    return iter(paths)


def iter_run_dirs(root: str | Path, *, sweep_id: object = "") -> Iterable[Path]:
    return (path.parent for path in iter_results_paths(root, sweep_id=sweep_id))
