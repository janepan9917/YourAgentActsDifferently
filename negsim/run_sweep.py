"""
Sweep runner: reads a sweep config, generates all parameter combinations, and runs
games in parallel. Outcomes ``error``, ``invalid_model_output``, ``invalid``,
and ``interrupted`` are retryable attempts. Their logs are archived under
``<sweep_id>_failed`` and the complete run is retried up to two times before
its logical task is reported as failed.

Usage:
    python -m negsim.run_sweep sweeps/my_sweep.yaml
    python -m negsim.run_sweep sweeps/my_sweep.yaml --workers 8
    python -m negsim.run_sweep sweeps/my_sweep.yaml --dry-run   # print combos without running
    python -m negsim.run_sweep sweeps/my_sweep.yaml --continue  # resume missing runs
    python -m negsim.run_sweep sweeps/my_sweep.yaml --delete-existing-sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import traceback
from dotenv import load_dotenv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .log_paths import (
    canonical_logs_root,
    combo_id_from_config,
    resolve_sweep_dir,
    run_log_dir,
)
from .run_outcomes import is_retryable_run_outcome, normalize_run_outcome

log = logging.getLogger(__name__)

load_dotenv()

MAX_RUN_RETRIES = 2
FAILED_SWEEP_SUFFIX = "_failed"


@dataclass(frozen=True)
class SweepCombo:
    """One resolved sweep condition, including its base config."""

    combo_id: str
    combo_index: int
    combo_label: str
    label: str
    base_config: dict[str, Any]
    base_path: Path
    overrides: dict[str, Any]
    base_id: str
    base_label: str
    n_runs: int
    metadata: dict[str, Any]


# ── config manipulation ───────────────────────────────────────────────────────

def _set_nested(d: dict, dotpath: str, value: Any) -> None:
    """Set a value at a dot-path in a nested dict/list.

    Supports dict keys and list indices, e.g.:
      'agent_1.initial_reward'
      'agent_1.auxiliary_objectives[0].penalty_per_round'
    """
    import re
    # Build key list: split on '.' then handle '[n]' within each segment
    keys: list[str | int] = []
    for segment in dotpath.split("."):
        parts = re.split(r"\[(\d+)\]", segment)
        if parts[0]:
            keys.append(parts[0])
        for i in range(1, len(parts)):
            if parts[i] != "":
                keys.append(int(parts[i]))

    node: Any = d
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value


# ── sweep loading ─────────────────────────────────────────────────────────────

def _load_yaml_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{kind} must be a YAML mapping: {path}")
    return data


def _load_base_config(base_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(base_path)
    if not path.exists():
        raise FileNotFoundError(f"Base config not found: {path}")
    return path, _load_yaml_mapping(path, kind="Base config")


def _mapping_or_empty(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping, got {type(value).__name__}")
    return dict(value)


def _parse_n_runs(value: Any, *, name: str) -> int:
    try:
        n_runs = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if n_runs < 0:
        raise ValueError(f"{name} must be non-negative, got {n_runs}")
    return n_runs


def _load_n_runs_by_combo(sweep: dict[str, Any]) -> dict[str, int]:
    raw = sweep.get("n_runs_by_combo") or {}
    if not isinstance(raw, dict):
        raise ValueError("'n_runs_by_combo' must be a mapping of combo_id to run count")
    return {
        str(combo_id): _parse_n_runs(value, name=f"n_runs_by_combo[{combo_id!r}]")
        for combo_id, value in raw.items()
    }


def _base_id_from_path(base_path: Path) -> str:
    stem = base_path.stem
    return stem.removeprefix("config_") or stem


def _make_sweep_combo(
    *,
    combo_id: str,
    combo_index: int,
    combo_label: str,
    label: str,
    base_config: dict[str, Any],
    base_path: Path,
    overrides: dict[str, Any],
    base_id: str,
    base_label: str,
    n_runs: int,
    metadata: dict[str, Any] | None = None,
) -> SweepCombo:
    return SweepCombo(
        combo_id=combo_id,
        combo_index=combo_index,
        combo_label=combo_label,
        label=label,
        base_config=base_config,
        base_path=base_path,
        overrides=overrides,
        base_id=base_id,
        base_label=base_label,
        n_runs=n_runs,
        metadata=metadata or {},
    )


def _load_explicit_combos(
    sweep: dict[str, Any],
    *,
    combo_start: int,
    default_n_runs: int,
    n_runs_by_combo: dict[str, int],
    labels: dict[str, str],
) -> list[SweepCombo]:
    raw_combos = sweep.get("combos")
    if not isinstance(raw_combos, list) or not raw_combos:
        raise ValueError("'combos' must be a non-empty list")

    top_level_base = sweep.get("base")
    base_cache: dict[str, tuple[Path, dict[str, Any]]] = {}
    combos: list[SweepCombo] = []
    seen_combo_ids: set[str] = set()

    for offset, raw_combo in enumerate(raw_combos):
        if not isinstance(raw_combo, dict):
            raise ValueError(f"Each explicit combo must be a mapping, got {type(raw_combo).__name__}")

        base_value = raw_combo.get("base", top_level_base)
        if not base_value:
            raise ValueError(f"Explicit combo at index {offset} is missing 'base'")

        base_key = str(base_value)
        if base_key not in base_cache:
            base_cache[base_key] = _load_base_config(base_value)
        base_path, base_config = base_cache[base_key]

        combo_index = combo_start + offset
        combo_id = str(raw_combo.get("combo_id") or raw_combo.get("id") or f"c{combo_index:03d}")
        if combo_id in seen_combo_ids:
            raise ValueError(f"Duplicate combo id in explicit combos: {combo_id}")
        seen_combo_ids.add(combo_id)

        overrides = _mapping_or_empty(raw_combo.get("overrides"), name=f"combos[{offset}].overrides")
        combo_label = str(raw_combo.get("combo_label") or raw_combo.get("id") or _combo_label(overrides))
        label = str(raw_combo.get("label") or labels.get(combo_id) or labels.get(combo_label) or combo_label)
        base_id = str(raw_combo.get("base_id") or _base_id_from_path(base_path))
        base_label = str(raw_combo.get("base_label") or base_id)
        combo_n_runs = _parse_n_runs(
            raw_combo.get("n_runs", n_runs_by_combo.get(combo_id, default_n_runs)),
            name=f"combos[{offset}].n_runs",
        )
        metadata = _mapping_or_empty(raw_combo.get("metadata"), name=f"combos[{offset}].metadata")

        combos.append(
            _make_sweep_combo(
                combo_id=combo_id,
                combo_index=combo_index,
                combo_label=combo_label,
                label=label,
                base_config=base_config,
                base_path=base_path,
                overrides=overrides,
                base_id=base_id,
                base_label=base_label,
                n_runs=combo_n_runs,
                metadata=metadata,
            )
        )

    return combos


def load_sweep(sweep_path: Path) -> tuple[str, list[SweepCombo], int]:
    """
    Parse a sweep YAML.

    Returns (sweep_id, resolved_combos, default_n_runs). Each combo includes its
    own base config, flat {dotpath: value} overrides, and resolved n_runs.
    """
    sweep_path = Path(sweep_path)
    sweep = _load_yaml_mapping(sweep_path, kind="Sweep config")

    sweep_id: str = sweep["sweep_id"]
    combo_start: int = int(sweep.get("combo_start", 0))
    n_runs: int = _parse_n_runs(sweep.get("n_runs", 1), name="n_runs")
    n_runs_by_combo = _load_n_runs_by_combo(sweep)
    labels = {str(k): str(v) for k, v in (sweep.get("labels") or {}).items()}

    combos = _load_explicit_combos(
        sweep,
        combo_start=combo_start,
        default_n_runs=n_runs,
        n_runs_by_combo=n_runs_by_combo,
        labels=labels,
    )

    unknown_n_run_keys = set(n_runs_by_combo) - {combo.combo_id for combo in combos}
    if unknown_n_run_keys:
        unknown = ", ".join(sorted(unknown_n_run_keys))
        raise ValueError(f"n_runs_by_combo contains unknown combo_id(s): {unknown}")

    return sweep_id, combos, n_runs


def _combo_label(combo: dict[str, Any]) -> str:
    """Short human-readable label for a parameter combo."""
    if not combo:
        return "default"
    parts = []
    for dotpath, value in sorted(combo.items()):
        short_key = dotpath.split(".")[-1].replace("[", "").replace("]", "")
        parts.append(f"{short_key}={value}")
    return "_".join(parts)


def _read_log_results(log_dir: Path) -> dict[str, Any]:
    """Read results.json from a game log directory, returning an empty dict on parse errors."""
    results_path = log_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return results if isinstance(results, dict) else {}


def _read_log_config(log_dir: Path) -> dict[str, Any]:
    """Read config.yaml from a game log directory, returning an empty dict on parse errors."""
    config_path = log_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return config if isinstance(config, dict) else {}


def _combo_id_from_mapping(mapping: dict[str, Any]) -> str:
    combo_id = mapping.get("combo_id")
    if combo_id not in (None, ""):
        return str(combo_id)

    metadata = mapping.get("result_metadata")
    if isinstance(metadata, dict):
        combo_id = metadata.get("combo_id")
        if combo_id not in (None, ""):
            return str(combo_id)

    return ""


def _read_log_combo_id(
    log_dir: Path,
    *,
    sweep_id: str,
    results: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Read a sweep combo id from results/config, with a config_name fallback."""
    results = results if results is not None else _read_log_results(log_dir)
    config = config if config is not None else _read_log_config(log_dir)

    for mapping in (results, config):
        combo_id = _combo_id_from_mapping(mapping)
        if combo_id:
            return combo_id

    config_name = results.get("config_name")
    if isinstance(config_name, str):
        prefix = f"{sweep_id}__"
        if config_name.startswith(prefix):
            combo_id = config_name[len(prefix):].split("__", 1)[0]
            if combo_id:
                return combo_id

    return ""


def _log_has_game_end(log_dir: Path) -> bool:
    """Return True if logs show the game reached the GAME END phase."""
    summary_path = log_dir / "summary.log"
    if summary_path.exists():
        try:
            text = summary_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if "Game end reason:" in text or "GAME END" in text:
            return True

    agent_dir = log_dir / "agent"
    if not agent_dir.exists() or not agent_dir.is_dir():
        return False
    for path in agent_dir.glob("*.log"):
        if path.name.endswith("_prompts.log"):
            continue
        try:
            if "GAME END" in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            continue
    return False


def _is_successful_game_end_log(log_dir: Path, results: dict[str, Any] | None = None) -> bool:
    """Return True when a non-failed log has structured results and reached game end."""
    results = results if results is not None else _read_log_results(log_dir)
    outcome = results.get("outcome")
    if outcome in (None, "") or is_retryable_run_outcome(outcome):
        return False
    return _log_has_game_end(log_dir)


def _count_successful_game_end_runs_by_combo(
    log_dirs: list[Path],
    sweep_id: str,
) -> Counter[str]:
    """Count existing successful game-end runs for each combo in this sweep."""
    counts: Counter[str] = Counter()
    for log_dir in log_dirs:
        results = _read_log_results(log_dir)
        if str(results.get("sweep_id") or "") != sweep_id:
            continue
        combo_id = _read_log_combo_id(log_dir, sweep_id=sweep_id, results=results)
        if not combo_id:
            continue
        if _is_successful_game_end_log(log_dir, results=results):
            counts[combo_id] += 1
    return counts


def _validate_continue_proposal_schema_versions(
    log_dirs: list[Path],
    sweep_id: str,
    expected_by_combo: dict[str, str | None],
) -> None:
    """Refuse to continue a sweep across incompatible proposal schemas."""
    mismatches: list[str] = []
    for log_dir in log_dirs:
        results = _read_log_results(log_dir)
        if str(results.get("sweep_id") or "") != sweep_id:
            continue
        config = _read_log_config(log_dir)
        combo_id = _read_log_combo_id(
            log_dir,
            sweep_id=sweep_id,
            results=results,
            config=config,
        )
        expected = expected_by_combo.get(combo_id)
        if expected in (None, ""):
            continue
        result_params = results.get("game_params")
        config_params = config.get("game_params")
        actual = None
        if isinstance(result_params, dict):
            actual = result_params.get("proposal_schema_version")
        if actual in (None, "") and isinstance(config_params, dict):
            actual = config_params.get("proposal_schema_version")
        if actual != expected:
            mismatches.append(
                f"{log_dir} (combo={combo_id!r}, existing={actual!r}, expected={expected!r})"
            )
    if mismatches:
        preview = "; ".join(mismatches[:5])
        suffix = "" if len(mismatches) <= 5 else f"; ... ({len(mismatches)} total)"
        raise RuntimeError(
            "Cannot --continue a sweep across proposal schema versions. "
            f"Use a new sweep_id. Mismatches: {preview}{suffix}"
        )


def _find_existing_sweep_logs(log_dirs: list[Path], sweep_id: str) -> list[Path]:
    """Return run directories from this sweep's log folder."""
    matches: list[Path] = []
    seen_roots: set[Path] = set()

    for log_root in log_dirs:
        log_root = canonical_logs_root(log_root)
        try:
            resolved_root = log_root.resolve()
        except OSError:
            resolved_root = log_root.absolute()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        if not log_root.exists():
            continue
        if not log_root.is_dir():
            raise NotADirectoryError(f"log_dir is not a directory: {log_root}")

        sweep_dir = resolve_sweep_dir(log_root, sweep_id)
        if not sweep_dir.is_dir():
            continue
        for combo_dir in sweep_dir.iterdir():
            if not combo_dir.is_dir():
                continue
            matches.extend(path for path in combo_dir.iterdir() if path.is_dir())

    return sorted(matches, key=lambda path: str(path))


def _delete_sweep_log_dirs(log_dirs: list[Path], paths: list[Path]) -> None:
    """Delete matched log directories after verifying they live under log_dirs."""
    resolved_roots = [
        canonical_logs_root(log_dir).resolve()
        for log_dir in log_dirs
        if canonical_logs_root(log_dir).exists()
    ]
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"Refusing to delete symlinked log directory: {path}")

        resolved_path = path.resolve()
        contained = False
        for root in resolved_roots:
            try:
                relative = resolved_path.relative_to(root)
            except ValueError:
                continue
            # A canonical run is exactly sweep/combo/run below the root. Never
            # allow this helper to delete the root, a sweep, or a combo folder.
            if len(relative.parts) == 3:
                contained = True
                break
        if not contained:
            raise RuntimeError(f"Refusing to delete log directory outside configured log_dir: {path}")

        shutil.rmtree(path)


def _preview_paths(paths: list[Path], *, limit: int = 5) -> str:
    preview = ", ".join(str(path) for path in paths[:limit])
    suffix = "" if len(paths) <= limit else f", ... ({len(paths)} total)"
    return f"{preview}{suffix}"


def _prepare_existing_sweep_logs(
    sweep_id: str,
    log_dirs: list[Path],
    *,
    delete_existing_sweep: bool,
    continue_sweep: bool = False,
    existing_logs: list[Path] | None = None,
) -> list[Path]:
    if delete_existing_sweep and continue_sweep:
        raise ValueError("--continue cannot be used with --delete-existing-sweep")

    if existing_logs is None:
        existing_logs = _find_existing_sweep_logs(log_dirs, sweep_id)
    if not existing_logs:
        return []

    preview = _preview_paths(existing_logs)
    if delete_existing_sweep:
        _delete_sweep_log_dirs(log_dirs, existing_logs)
        print(f"Deleted {len(existing_logs)} existing log record(s) for sweep '{sweep_id}'.")
        return []

    if continue_sweep:
        print(
            f"Found {len(existing_logs)} existing log record(s) for sweep_id '{sweep_id}': "
            f"{preview}. Continuing with --continue."
        )
        return existing_logs

    if not delete_existing_sweep:
        raise RuntimeError(
            f"Found {len(existing_logs)} existing log record(s) for sweep_id '{sweep_id}': "
            f"{preview}. Re-run with --continue to resume missing runs, or "
            f"--delete-existing-sweep to remove them before running."
        )
    return existing_logs


# ── run ───────────────────────────────────────────────────────────────────────

def _failed_sweep_id(sweep_id: str) -> str:
    """Return the companion sweep id used only for failed run attempts."""

    return f"{sweep_id}{FAILED_SWEEP_SUFFIX}"


def _attempt_config(
    cfg: dict[str, Any],
    *,
    retry_number: int,
    logical_run_id: str,
) -> dict[str, Any]:
    """Return a pristine per-attempt config with auditable retry metadata."""

    attempt_cfg = copy.deepcopy(cfg)
    metadata = attempt_cfg.get("result_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        attempt_cfg["result_metadata"] = metadata
    metadata.update(
        {
            "sweep_attempt_number": retry_number + 1,
            "sweep_retry_number": retry_number,
            "sweep_max_retries": MAX_RUN_RETRIES,
            "sweep_logical_run_id": logical_run_id,
        }
    )
    return attempt_cfg


def _set_failed_sweep_metadata(
    mapping: dict[str, Any],
    *,
    original_sweep_id: str,
    failed_sweep_id: str,
    retry_number: int,
    failure_description: str,
    failed_log_dir: Path,
) -> None:
    """Make one persisted config/results mapping match its failed-log path."""

    mapping["sweep_id"] = failed_sweep_id
    metadata = mapping.get("result_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        mapping["result_metadata"] = metadata
    metadata.update(
        {
            "original_sweep_id": original_sweep_id,
            "failed_sweep_id": failed_sweep_id,
            "sweep_attempt_number": retry_number + 1,
            "sweep_retry_number": retry_number,
            "sweep_max_retries": MAX_RUN_RETRIES,
            "sweep_attempt_failure": failure_description,
            "failed_log_dir": str(failed_log_dir),
        }
    )


def _rewrite_failed_run_metadata(
    run_dir: Path,
    *,
    original_sweep_id: str,
    failed_sweep_id: str,
    retry_number: int,
    failure_description: str,
    failed_log_dir: Path,
) -> dict[Path, str]:
    """Atomically rewrite readable metadata and return its original text."""

    originals: dict[Path, str] = {}
    updates: dict[Path, str] = {}
    results_path = run_dir / "results.json"
    if results_path.exists():
        try:
            original_results = results_path.read_text(encoding="utf-8")
            results = json.loads(original_results)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot archive unreadable results metadata: {results_path}"
            ) from exc
        if not isinstance(results, dict):
            raise RuntimeError(f"Cannot archive non-object results metadata: {results_path}")
        _set_failed_sweep_metadata(
            results,
            original_sweep_id=original_sweep_id,
            failed_sweep_id=failed_sweep_id,
            retry_number=retry_number,
            failure_description=failure_description,
            failed_log_dir=failed_log_dir,
        )
        try:
            updated_results = json.dumps(results, indent=2)
        except TypeError as exc:
            raise RuntimeError(
                f"Cannot serialize failed results metadata: {results_path}"
            ) from exc
        originals[results_path] = original_results
        updates[results_path] = updated_results

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            original_config = config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(original_config) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RuntimeError(
                f"Cannot archive unreadable config metadata: {config_path}"
            ) from exc
        if not isinstance(config, dict):
            raise RuntimeError(f"Cannot archive non-mapping config metadata: {config_path}")
        _set_failed_sweep_metadata(
            config,
            original_sweep_id=original_sweep_id,
            failed_sweep_id=failed_sweep_id,
            retry_number=retry_number,
            failure_description=failure_description,
            failed_log_dir=failed_log_dir,
        )
        try:
            updated_config = yaml.safe_dump(
                config,
                sort_keys=False,
                allow_unicode=True,
            )
        except (yaml.YAMLError, TypeError) as exc:
            raise RuntimeError(
                f"Cannot serialize failed config metadata: {config_path}"
            ) from exc
        originals[config_path] = original_config
        updates[config_path] = updated_config

    temp_paths: dict[Path, Path] = {}
    try:
        for path, text in updates.items():
            temp_path = path.with_name(f"{path.name}.failed-archive.tmp")
            temp_path.write_text(text, encoding="utf-8")
            temp_paths[path] = temp_path
        for path, temp_path in temp_paths.items():
            temp_path.replace(path)
    except OSError as exc:
        restore_errors: list[str] = []
        for path, original in originals.items():
            try:
                path.write_text(original, encoding="utf-8")
            except OSError as restore_exc:
                restore_errors.append(f"{path}: {restore_exc}")
        for temp_path in temp_paths.values():
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        detail = ""
        if restore_errors:
            detail = f"; rollback also failed: {'; '.join(restore_errors)}"
        raise RuntimeError(
            f"Cannot persist failed run metadata in {run_dir}{detail}"
        ) from exc
    return originals


def _restore_run_metadata(originals: dict[Path, str]) -> None:
    """Best-effort rollback when a metadata rewrite succeeded but move failed."""

    errors: list[str] = []
    for path, original in originals.items():
        try:
            path.write_text(original, encoding="utf-8")
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    if errors:
        raise RuntimeError(f"Could not roll back run metadata: {'; '.join(errors)}")


def _ensure_exception_attempt_log(
    run_dir: Path,
    *,
    attempt_cfg: dict[str, Any],
    config_name: str,
    game_id: str,
    error: BaseException,
    outcome: str = "error",
) -> None:
    """Persist or repair an error record when main() raised unexpectedly."""

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_is_valid = False
    if config_path.exists():
        try:
            existing_config = yaml.safe_load(
                config_path.read_text(encoding="utf-8")
            )
            config_is_valid = isinstance(existing_config, dict)
        except (OSError, yaml.YAMLError):
            config_is_valid = False
    if not config_is_valid:
        temp_path = config_path.with_name(f"{config_path.name}.tmp")
        temp_path.write_text(
            yaml.safe_dump(attempt_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        temp_path.replace(config_path)

    results_path = run_dir / "results.json"
    results: dict[str, Any] = {}
    previous_outcome = ""
    if results_path.exists():
        try:
            existing_results = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(existing_results, dict):
                results = existing_results
                previous_outcome = normalize_run_outcome(results.get("outcome"))
        except (OSError, json.JSONDecodeError):
            results = {}
    metadata = attempt_cfg.get("result_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    persisted_metadata = results.get("result_metadata")
    if not isinstance(persisted_metadata, dict):
        persisted_metadata = {}
    persisted_metadata.update(copy.deepcopy(metadata))
    results.update(
        {
            "game_id": game_id,
            "config_name": config_name,
            "sweep_id": str(attempt_cfg.get("sweep_id") or ""),
            "combo_id": combo_id_from_config(attempt_cfg),
            "outcome": outcome,
            "result_metadata": persisted_metadata,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        }
    )
    if previous_outcome and previous_outcome != outcome:
        results["pre_exception_outcome"] = previous_outcome
    temp_path = results_path.with_name(f"{results_path.name}.tmp")
    temp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temp_path.replace(results_path)


def _archive_failed_attempt_log(
    run_dir: Path,
    *,
    logs_root: Path,
    original_sweep_id: str,
    combo_id: str,
    retry_number: int,
    failure_description: str,
) -> Path | None:
    """Move one failed attempt out of the successful sweep, if it made a log."""

    if not run_dir.exists():
        return None
    if run_dir.is_symlink():
        raise RuntimeError(f"Refusing to archive symlinked run log: {run_dir}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run log path is not a directory: {run_dir}")

    resolved_root = canonical_logs_root(logs_root).resolve()
    resolved_run = run_dir.resolve()
    try:
        relative = resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to archive run outside configured log root: {run_dir}"
        ) from exc
    if len(relative.parts) != 3:
        raise RuntimeError(
            f"Refusing to archive non-canonical run path: {run_dir}"
        )
    expected_sweep_dir = resolve_sweep_dir(logs_root, original_sweep_id)
    if run_dir.parent.parent.resolve() != expected_sweep_dir.resolve():
        raise RuntimeError(
            f"Run log is not under sweep {original_sweep_id!r}: {run_dir}"
        )

    failed_sweep_id = _failed_sweep_id(original_sweep_id)
    target = run_log_dir(
        logs_root,
        failed_sweep_id,
        combo_id,
        run_dir.name,
    )
    if target.exists():
        raise FileExistsError(f"Failed-log destination already exists: {target}")

    originals = _rewrite_failed_run_metadata(
        run_dir,
        original_sweep_id=original_sweep_id,
        failed_sweep_id=failed_sweep_id,
        retry_number=retry_number,
        failure_description=failure_description,
        failed_log_dir=target,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.rename(target)
    except Exception:
        _restore_run_metadata(originals)
        raise

    return target


def _retry_number_from_log(
    results: dict[str, Any],
    config: dict[str, Any],
) -> int:
    """Recover a non-negative retry number from an existing log, if present."""

    for mapping in (results, config):
        metadata = mapping.get("result_metadata")
        if not isinstance(metadata, dict):
            continue
        value = metadata.get("sweep_retry_number")
        try:
            retry_number = int(value)
        except (TypeError, ValueError):
            continue
        if retry_number >= 0:
            return retry_number
    return 0


def _log_root_containing_run(run_dir: Path, log_dirs: list[Path]) -> Path:
    """Return the configured log root that directly contains a run."""

    resolved_run = run_dir.resolve()
    for log_root in log_dirs:
        canonical_root = canonical_logs_root(log_root)
        try:
            relative = resolved_run.relative_to(canonical_root.resolve())
        except (OSError, ValueError):
            continue
        if len(relative.parts) == 3:
            return canonical_root
    raise RuntimeError(f"Existing run is outside configured log roots: {run_dir}")


def _archive_existing_failed_logs(
    *,
    sweep_id: str,
    log_dirs: list[Path],
    existing_logs: list[Path],
    dry_run: bool,
) -> tuple[list[Path], list[Path]]:
    """Archive existing failed logs and return (failed, remaining) sources."""

    failed_logs: list[Path] = []
    remaining_logs: list[Path] = []
    records: list[tuple[Path, Path, Path, str, int, str]] = []
    seen_targets: set[Path] = set()

    for run_dir in existing_logs:
        if run_dir.is_symlink():
            raise RuntimeError(f"Refusing to inspect symlinked run log: {run_dir}")
        results = _read_log_results(run_dir)
        missing_results = not (run_dir / "results.json").exists()
        outcome = (
            "interrupted"
            if missing_results
            else normalize_run_outcome(results.get("outcome"))
        )
        if not is_retryable_run_outcome(outcome):
            remaining_logs.append(run_dir)
            continue

        config = _read_log_config(run_dir)
        combo_id = _read_log_combo_id(
            run_dir,
            sweep_id=sweep_id,
            results=results,
            config=config,
        ) or run_dir.parent.name
        retry_number = _retry_number_from_log(results, config)
        logs_root = _log_root_containing_run(run_dir, log_dirs)
        target = run_log_dir(
            logs_root,
            _failed_sweep_id(sweep_id),
            combo_id,
            run_dir.name,
        )
        if target in seen_targets:
            raise RuntimeError(f"Multiple failed logs map to one destination: {target}")
        if target.exists():
            raise FileExistsError(f"Failed-log destination already exists: {target}")
        seen_targets.add(target)
        records.append(
            (
                run_dir,
                logs_root,
                target,
                combo_id,
                retry_number,
                (
                    "outcome=interrupted (missing results.json)"
                    if missing_results
                    else f"outcome={outcome}"
                ),
            )
        )
        failed_logs.append(run_dir)

    if not failed_logs:
        return [], remaining_logs

    failed_id = _failed_sweep_id(sweep_id)
    if dry_run:
        mappings = [f"{source} -> {target}" for source, _, target, _, _, _ in records]
        preview = ", ".join(mappings[:5])
        suffix = "" if len(mappings) <= 5 else f", ... ({len(mappings)} total)"
        print(
            f"[dry-run] Would archive {len(failed_logs)} existing failed log "
            f"record(s) from sweep '{sweep_id}' under '{failed_id}': "
            f"{preview}{suffix}"
        )
        return failed_logs, remaining_logs

    archived_targets: list[Path] = []
    for (
        run_dir,
        logs_root,
        _target,
        combo_id,
        retry_number,
        failure_description,
    ) in records:
        target = _archive_failed_attempt_log(
            run_dir,
            logs_root=logs_root,
            original_sweep_id=sweep_id,
            combo_id=combo_id,
            retry_number=retry_number,
            failure_description=failure_description,
        )
        if target is not None:
            archived_targets.append(target)
    print(
        f"Archived {len(archived_targets)} existing failed log record(s) "
        f"from sweep '{sweep_id}' under '{failed_id}'."
    )
    return failed_logs, remaining_logs


def _retry_warning(
    config_name: str,
    *,
    logical_run_id: str,
    retry_number: int,
    failure_description: str,
) -> None:
    log.warning(
        "WARNING: %s [%s] failed (%s); starting retry %d/%d of the entire run.",
        config_name,
        logical_run_id,
        failure_description,
        retry_number,
        MAX_RUN_RETRIES,
    )


def _final_retry_failure_warning(
    config_name: str,
    *,
    logical_run_id: str,
    failure_description: str,
) -> None:
    log.warning(
        "WARNING: %s [%s] retry %d/%d failed (%s); no retries remain.",
        config_name,
        logical_run_id,
        MAX_RUN_RETRIES,
        MAX_RUN_RETRIES,
        failure_description,
    )

def _make_task_config(
    *,
    sweep_path: Path,
    sweep_id: str,
    combo: SweepCombo,
    target_n_runs: int | None = None,
) -> tuple[dict[str, Any], str]:
    cfg = copy.deepcopy(combo.base_config)
    for dotpath, value in combo.overrides.items():
        _set_nested(cfg, dotpath, value)
    game_params = cfg.get("game_params")
    schema_version = (
        game_params.get("proposal_schema_version")
        if isinstance(game_params, dict)
        else None
    )
    metadata_schema_version = combo.metadata.get("proposal_schema_version")
    if schema_version not in (None, "") and metadata_schema_version != schema_version:
        raise ValueError(
            "A versioned proposal schema requires matching combo metadata. "
            f"Config has {schema_version!r}, metadata has {metadata_schema_version!r}. "
            "Use a new sweep_id and set metadata.proposal_schema_version explicitly."
        )
    cfg["sweep_id"] = sweep_id
    result_metadata = copy.deepcopy(combo.metadata)
    result_metadata.update({
        "base_config_path": str(combo.base_path),
        "base_id": combo.base_id,
        "base_label": combo.base_label,
        "sweep_config_path": str(sweep_path),
        "label": combo.label,
        "combo_id": combo.combo_id,
        "combo_index": combo.combo_index,
        "combo_label": combo.combo_label,
        "combo_overrides": copy.deepcopy(combo.overrides),
        "combo_n_runs": combo.n_runs,
    })
    if target_n_runs is not None:
        result_metadata["target_n_runs"] = target_n_runs
    cfg["result_metadata"] = result_metadata
    cfg["log_dir"] = str(canonical_logs_root(cfg.get("log_dir")))
    config_name = f"{sweep_id}__{combo.combo_id}__{combo.combo_label}"
    return cfg, config_name


def _preview_combo_run_counts(
    combos: list[SweepCombo],
    target_runs_by_combo: dict[str, int],
    *,
    limit: int = 8,
) -> str:
    parts = [f"{combo.combo_id}={target_runs_by_combo[combo.combo_id]}" for combo in combos[:limit]]
    suffix = "" if len(combos) <= limit else f", ... ({len(combos)} total)"
    return ", ".join(parts) + suffix


def run_sweep(
    sweep_path: str | Path,
    workers: int = 40,
    dry_run: bool = False,
    n_runs_override: int | None = None,
    delete_existing_sweep: bool = False,
    continue_sweep: bool = False,
) -> None:
    if delete_existing_sweep and continue_sweep:
        raise ValueError("--continue cannot be used with --delete-existing-sweep")
    sweep_path = Path(sweep_path)
    sweep_id, combos, default_n_runs = load_sweep(sweep_path)
    if sweep_id.strip().lower().endswith(FAILED_SWEEP_SUFFIX):
        raise ValueError(
            f"sweep_id suffix {FAILED_SWEEP_SUFFIX!r} is reserved for failed logs: "
            f"{sweep_id!r}"
        )
    if n_runs_override is not None:
        n_runs_override = _parse_n_runs(n_runs_override, name="--n-runs")
    target_runs_by_combo = {
        combo.combo_id: (n_runs_override if n_runs_override is not None else combo.n_runs)
        for combo in combos
    }

    combo_specs: list[tuple[SweepCombo, dict[str, Any], str]] = []
    for combo in combos:
        target_n_runs = target_runs_by_combo[combo.combo_id]
        cfg, config_name = _make_task_config(
            sweep_path=sweep_path,
            sweep_id=sweep_id,
            combo=combo,
            target_n_runs=target_n_runs,
        )
        combo_specs.append((combo, cfg, config_name))

    log_dirs = sorted(
        {canonical_logs_root(cfg.get("log_dir")) for _, cfg, _ in combo_specs},
        key=lambda path: str(path),
    )
    existing_logs = _find_existing_sweep_logs(log_dirs, sweep_id)
    _, existing_logs = _archive_existing_failed_logs(
        sweep_id=sweep_id,
        log_dirs=log_dirs,
        existing_logs=existing_logs,
        dry_run=dry_run,
    )
    if dry_run and delete_existing_sweep:
        if existing_logs:
            print(
                f"[dry-run] Would delete {len(existing_logs)} existing log record(s) "
                f"for sweep '{sweep_id}': {_preview_paths(existing_logs)}"
            )
        existing_logs = []
    else:
        existing_logs = _prepare_existing_sweep_logs(
            sweep_id,
            log_dirs,
            delete_existing_sweep=delete_existing_sweep,
            continue_sweep=continue_sweep,
            existing_logs=existing_logs,
        )

    successful_by_combo: Counter[str] = Counter()
    if continue_sweep and existing_logs:
        expected_schema_versions = {
            combo.combo_id: (
                (cfg.get("game_params") or {}).get("proposal_schema_version")
                if isinstance(cfg.get("game_params"), dict)
                else None
            )
            for combo, cfg, _ in combo_specs
        }
        _validate_continue_proposal_schema_versions(
            existing_logs,
            sweep_id,
            expected_schema_versions,
        )
        successful_by_combo = _count_successful_game_end_runs_by_combo(existing_logs, sweep_id)

    runs_to_schedule: dict[str, int] = {}
    for combo, _, _ in combo_specs:
        successful = successful_by_combo.get(combo.combo_id, 0) if continue_sweep else 0
        target_n_runs = target_runs_by_combo[combo.combo_id]
        runs_to_schedule[combo.combo_id] = max(target_n_runs - successful, 0)

    # Build task list: (config_dict, config_name)
    tasks: list[tuple[dict, str]] = []
    for combo, _, _ in combo_specs:
        target_n_runs = target_runs_by_combo[combo.combo_id]
        for _ in range(runs_to_schedule[combo.combo_id]):
            cfg, config_name = _make_task_config(
                sweep_path=sweep_path,
                sweep_id=sweep_id,
                combo=combo,
                target_n_runs=target_n_runs,
            )
            tasks.append((cfg, config_name))

    target_total = sum(target_runs_by_combo.values())
    scheduled_total = len(tasks)
    unique_targets = set(target_runs_by_combo.values())
    if len(unique_targets) == 1:
        target_n_runs = next(iter(unique_targets), default_n_runs)
        print(f"Sweep '{sweep_id}': {len(combos)} combo(s) x {target_n_runs} run(s) = {target_total} target game(s)")
    else:
        print(f"Sweep '{sweep_id}': {len(combos)} combo(s), {target_total} target game(s)")
        print(f"  Runs per combo: {_preview_combo_run_counts(combos, target_runs_by_combo)}")
    if continue_sweep:
        current_successes = sum(
            min(successful_by_combo.get(combo.combo_id, 0), target_runs_by_combo[combo.combo_id])
            for combo in combos
        )
        print(
            f"  Continue mode: {current_successes}/{target_total} successful existing game-end run(s); "
            f"scheduling {scheduled_total} missing run(s)."
        )
        over_target = {
            combo.combo_id: (successful_by_combo[combo.combo_id], target_runs_by_combo[combo.combo_id])
            for combo in combos
            if successful_by_combo.get(combo.combo_id, 0) > target_runs_by_combo[combo.combo_id]
        }
        if over_target:
            items = list(over_target.items())
            preview = ", ".join(f"{combo_id}={count}>{target}" for combo_id, (count, target) in items[:5])
            suffix = "" if len(items) <= 5 else f", ... ({len(items)} total)"
            print(
                f"  {len(over_target)} combo(s) already exceed configured n_runs; "
                f"no additional runs scheduled for them: {preview}{suffix}"
            )
    base_paths = sorted({str(combo.base_path) for combo in combos})
    if len(base_paths) > 1:
        print("  Base configs:")
        for base_path in base_paths:
            print(f"    {base_path}")
    swept_keys = sorted({dotpath for combo in combos for dotpath in combo.overrides})
    if swept_keys:
        print("  Parameters swept:")
        for dotpath in swept_keys:
            vals = sorted({str(combo.overrides[dotpath]) for combo in combos if dotpath in combo.overrides})
            print(f"    {dotpath}: {vals}")

    if dry_run:
        print("\n[dry-run] Would run:")
        if not tasks:
            print("  (nothing)")
        for cfg, name in tasks:
            label = (cfg.get("result_metadata") or {}).get("label")
            suffix = f" [{label}]" if label else ""
            print(f"  {name}{suffix}")
        return

    if not tasks:
        print("\nDone. No missing runs to schedule.")
        return

    # Lazy import to avoid loading the whole negsim package until we actually run
    from .main import main as run_game
    from .orchestrator import make_game_id

    completed = 0
    failed = 0
    failures: list[str] = []

    def _run(cfg: dict, config_name: str) -> str:
        original_sweep_id = str(cfg.get("sweep_id") or "")
        logs_root = canonical_logs_root(cfg.get("log_dir"))
        combo_id = combo_id_from_config(cfg)
        logical_run_id = make_game_id()

        for retry_number in range(MAX_RUN_RETRIES + 1):
            attempt_cfg = _attempt_config(
                cfg,
                retry_number=retry_number,
                logical_run_id=logical_run_id,
            )
            game_id = make_game_id()
            attempt_log_dir = run_log_dir(
                logs_root,
                original_sweep_id,
                combo_id,
                game_id,
            )
            try:
                result = str(
                    run_game(
                        config=attempt_cfg,
                        config_name=config_name,
                        game_id=game_id,
                    )
                )
            except KeyboardInterrupt as exc:
                result = "interrupted"
                failure_description = "outcome=interrupted (KeyboardInterrupt)"
                try:
                    _ensure_exception_attempt_log(
                        attempt_log_dir,
                        attempt_cfg=attempt_cfg,
                        config_name=config_name,
                        game_id=game_id,
                        error=exc,
                        outcome=result,
                    )
                    _archive_failed_attempt_log(
                        attempt_log_dir,
                        logs_root=logs_root,
                        original_sweep_id=original_sweep_id,
                        combo_id=combo_id,
                        retry_number=retry_number,
                        failure_description=failure_description,
                    )
                except Exception as archive_exc:
                    message = (
                        f"Could not isolate interrupted attempt log for {config_name}; "
                        "refusing to retry"
                    )
                    log.error("ERROR: %s: %s", message, archive_exc)
                    raise RuntimeError(message) from archive_exc
                if retry_number < MAX_RUN_RETRIES:
                    _retry_warning(
                        config_name,
                        logical_run_id=logical_run_id,
                        retry_number=retry_number + 1,
                        failure_description=failure_description,
                    )
                    continue
                _final_retry_failure_warning(
                    config_name,
                    logical_run_id=logical_run_id,
                    failure_description=failure_description,
                )
                return result
            except Exception as exc:
                failure_description = f"{type(exc).__name__}: {exc}"
                try:
                    _ensure_exception_attempt_log(
                        attempt_log_dir,
                        attempt_cfg=attempt_cfg,
                        config_name=config_name,
                        game_id=game_id,
                        error=exc,
                    )
                    _archive_failed_attempt_log(
                        attempt_log_dir,
                        logs_root=logs_root,
                        original_sweep_id=original_sweep_id,
                        combo_id=combo_id,
                        retry_number=retry_number,
                        failure_description=failure_description,
                    )
                except Exception as archive_exc:
                    message = (
                        f"Could not isolate failed attempt log for {config_name}; "
                        "refusing to retry"
                    )
                    log.error("ERROR: %s: %s", message, archive_exc)
                    raise RuntimeError(message) from archive_exc
                if retry_number < MAX_RUN_RETRIES:
                    _retry_warning(
                        config_name,
                        logical_run_id=logical_run_id,
                        retry_number=retry_number + 1,
                        failure_description=failure_description,
                    )
                    continue
                _final_retry_failure_warning(
                    config_name,
                    logical_run_id=logical_run_id,
                    failure_description=failure_description,
                )
                raise

            if not is_retryable_run_outcome(result):
                return result

            failure_description = f"outcome={result}"
            try:
                _archive_failed_attempt_log(
                    attempt_log_dir,
                    logs_root=logs_root,
                    original_sweep_id=original_sweep_id,
                    combo_id=combo_id,
                    retry_number=retry_number,
                    failure_description=failure_description,
                )
            except Exception as archive_exc:
                message = (
                    f"Could not isolate failed attempt log for {config_name}; "
                    "refusing to retry"
                )
                log.error("ERROR: %s: %s", message, archive_exc)
                raise RuntimeError(message) from archive_exc
            if retry_number < MAX_RUN_RETRIES:
                _retry_warning(
                    config_name,
                    logical_run_id=logical_run_id,
                    retry_number=retry_number + 1,
                    failure_description=failure_description,
                )
                continue

            _final_retry_failure_warning(
                config_name,
                logical_run_id=logical_run_id,
                failure_description=failure_description,
            )
            return result

        raise AssertionError("unreachable retry loop exit")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, cfg, name): name for cfg, name in tasks}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                result = fut.result()
                if is_retryable_run_outcome(result):
                    failed += 1
                    failure = f"{name} (outcome={result})"
                    failures.append(failure)
                    print(
                        f"  [{completed + failed}/{scheduled_total}] "
                        f"{name}: ERROR - outcome={result}"
                    )
                    log.error(
                        "Game exhausted retries with outcome: %s: %s",
                        name,
                        result,
                    )
                else:
                    completed += 1
                    print(f"  [{completed + failed}/{scheduled_total}] {name}: {result}")
            except Exception as e:
                failed += 1
                failures.append(name)
                print(f"  [{completed + failed}/{scheduled_total}] {name}: ERROR — {e}")
                log.exception("Game failed: %s", name)

    print(f"\nDone. {completed} completed, {failed} failed.")
    if failures:
        preview = ", ".join(failures[:5])
        suffix = "" if len(failures) <= 5 else f", ... ({len(failures)} total)"
        raise RuntimeError(f"{failed} game(s) failed in sweep '{sweep_id}': {preview}{suffix}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run a parameter sweep of negotiation games.")
    parser.add_argument("sweep", help="Path to sweep YAML config (e.g. sweeps/my_sweep.yaml)")
    parser.add_argument("--workers", type=int, default=40, help="Parallel games (default: 40)")
    parser.add_argument("--dry-run", action="store_true", help="Print combos without running")
    parser.add_argument("--n-runs", type=int, default=None, help="Override n_runs from sweep config")
    parser.add_argument(
        "--delete-existing-sweep",
        action="store_true",
        help="Delete existing logs with the same sweep_id before running",
    )
    parser.add_argument(
        "--continue",
        dest="continue_sweep",
        action="store_true",
        help="Resume a sweep_id by running only combo runs missing successful GAME END logs",
    )
    args = parser.parse_args()
    run_sweep(
        args.sweep,
        workers=args.workers,
        dry_run=args.dry_run,
        n_runs_override=args.n_runs,
        delete_existing_sweep=args.delete_existing_sweep,
        continue_sweep=args.continue_sweep,
    )
