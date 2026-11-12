"""Turn-level misaligned-behavior labels of the human annotators and the automatic judges (used by Figure 5)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluate_misaligned_behavior import (
    BEHAVIORS,
    _parse_turn_numbers,
)


AUTO_BASENAME = "misbehavior_eval"
MANUAL_BASENAME = "misbehavior_manual_annotations"
ANY_BEHAVIOR_ID = "MB_ANY"
ANY_TAXONOMY_GROUP = "MB_TURN_GATE"
PRESENT = "present"
ABSENT = "absent"


BEHAVIOR_GROUPS = {
    str(behavior["behavior_id"]): str(behavior.get("taxonomy_group", ""))
    for behavior in BEHAVIORS
}
BEHAVIOR_GROUPS[ANY_BEHAVIOR_ID] = ANY_TAXONOMY_GROUP


@dataclass
class LabelSet:
    name: str
    positives: set[tuple[str, int, str]] = field(default_factory=set)
    coverage: set[tuple[str, str]] = field(default_factory=set)
    explicit: dict[tuple[str, int, str], str] | None = None

    def label_for(self, key: tuple[str, int, str]) -> str:
        if self.explicit is not None:
            return self.explicit.get(key, "unannotated")
        game_id, _turn, behavior_id = key
        if key in self.positives:
            return PRESENT
        if (game_id, behavior_id) in self.coverage:
            return ABSENT
        return "unavailable"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported table format: {path}")


def _resolve_table_path(path: Path, basename: str) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    for suffix in (".jsonl", ".csv"):
        candidate = path / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {basename}.jsonl or {basename}.csv in {path}")


def _is_present(row: dict[str, Any]) -> bool:
    label = str(row.get("label_mb", "")).strip().lower()
    if label:
        return label == PRESENT
    return str(row.get("present", "")).strip().lower() in {"1", "true", "yes", "present"}


def _parse_turn(value: Any) -> int | None:
    turns = _parse_turn_numbers(value)
    return turns[0] if len(turns) == 1 else None


def load_auto_labels(name: str, path: Path) -> LabelSet:
    rows = _read_rows(_resolve_table_path(path, AUTO_BASENAME))
    labels = LabelSet(name=name)
    for row in rows:
        game_id = str(row.get("game_id", "") or "").strip()
        behavior_id = str(row.get("behavior_id", "") or "").strip()
        if not game_id or not behavior_id:
            continue
        labels.coverage.add((game_id, behavior_id))
        labels.coverage.add((game_id, ANY_BEHAVIOR_ID))
        if not _is_present(row):
            continue
        for turn in _parse_turn_numbers(row.get("turn_numbers")):
            labels.positives.add((game_id, turn, behavior_id))
            labels.positives.add((game_id, turn, ANY_BEHAVIOR_ID))
    return labels


def load_manual_labels(name: str, path: Path) -> LabelSet:
    rows = _read_rows(_resolve_table_path(path, MANUAL_BASENAME))
    explicit: dict[tuple[str, int, str], str] = {}
    for row in rows:
        game_id = str(row.get("game_id", "") or "").strip()
        behavior_id = str(row.get("behavior_id", "") or "").strip()
        turn = _parse_turn(row.get("turn"))
        if not game_id or not behavior_id or turn is None:
            continue
        key = (game_id, turn, behavior_id)
        label = PRESENT if _is_present(row) else ABSENT
        if explicit.get(key) == PRESENT:
            continue
        explicit[key] = label
    positives = {key for key, label in explicit.items() if label == PRESENT}
    coverage = {(game_id, behavior_id) for game_id, _turn, behavior_id in explicit}
    return LabelSet(
        name=name,
        positives=positives,
        coverage=coverage,
        explicit=explicit,
    )
