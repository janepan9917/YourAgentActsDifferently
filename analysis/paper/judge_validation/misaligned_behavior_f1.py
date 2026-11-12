"""Turn-level F1 of the automatic MB judges against the human labels (used by Figure 5)."""

from __future__ import annotations

import math
from typing import Any


from evaluate_misaligned_behavior import BEHAVIORS
from turn_labels import ANY_BEHAVIOR_ID, BEHAVIOR_GROUPS, LabelSet


PRESENT = "present"
ABSENT = "absent"
SCORABLE_LABELS = {PRESENT, ABSENT}


BEHAVIOR_ORDER = [ANY_BEHAVIOR_ID, *[str(behavior["behavior_id"]) for behavior in BEHAVIORS]]


def _metric_row(
    *,
    method: str,
    behavior_id: str,
    taxonomy_group: str,
    pairs: list[tuple[bool, bool]],
    unavailable_items: int,
) -> dict[str, Any]:
    tp = sum(1 for truth, pred in pairs if truth and pred)
    fp = sum(1 for truth, pred in pairs if not truth and pred)
    fn = sum(1 for truth, pred in pairs if truth and not pred)
    tn = sum(1 for truth, pred in pairs if not truth and not pred)
    predicted_positive = tp + fp
    support_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else (math.nan if not support_positive else 0.0)
    recall = tp / support_positive if support_positive else (math.nan if not predicted_positive else 0.0)
    if not support_positive and not predicted_positive:
        f1 = math.nan
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "method": method,
        "behavior_id": behavior_id,
        "taxonomy_group": taxonomy_group,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support_positive": support_positive,
        "support_negative": tn + fp,
        "predicted_positive": predicted_positive,
        "scored_items": len(pairs),
        "unavailable_items": unavailable_items,
    }


def build_f1_tables(
    *,
    manual: LabelSet,
    autos: list[LabelSet],
    exclude_turns: set[tuple[str, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if manual.explicit is None:
        raise ValueError("Manual label set must contain explicit turn-level labels")

    excluded = exclude_turns or set()
    manual_items = {
        key: label
        for key, label in manual.explicit.items()
        if label in SCORABLE_LABELS and (key[0], key[1]) not in excluded
    }
    aligned_rows: list[dict[str, Any]] = []
    total_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []

    for auto in autos:
        method_pairs: list[tuple[bool, bool]] = []
        method_unavailable = 0
        pairs_by_behavior: dict[str, list[tuple[bool, bool]]] = {}
        unavailable_by_behavior: dict[str, int] = {}
        for key, truth_label in sorted(manual_items.items()):
            pred_label = auto.label_for(key)
            game_id, turn_number, behavior_id = key
            if pred_label not in SCORABLE_LABELS:
                method_unavailable += 1
                unavailable_by_behavior[behavior_id] = unavailable_by_behavior.get(behavior_id, 0) + 1
                aligned_rows.append(
                    {
                        "method": auto.name,
                        "run_id": game_id,
                        "game_id": game_id,
                        "turn_number": turn_number,
                        "behavior_id": behavior_id,
                        "taxonomy_group": BEHAVIOR_GROUPS.get(behavior_id, ""),
                        "manual_label": truth_label,
                        "auto_label": pred_label,
                        "scored": False,
                    }
                )
                continue
            pair = (truth_label == PRESENT, pred_label == PRESENT)
            method_pairs.append(pair)
            pairs_by_behavior.setdefault(behavior_id, []).append(pair)
            aligned_rows.append(
                {
                    "method": auto.name,
                    "run_id": game_id,
                    "game_id": game_id,
                    "turn_number": turn_number,
                    "behavior_id": behavior_id,
                    "taxonomy_group": BEHAVIOR_GROUPS.get(behavior_id, ""),
                    "manual_label": truth_label,
                    "auto_label": pred_label,
                    "scored": True,
                }
            )

        total_rows.append(
            _metric_row(
                method=auto.name,
                behavior_id="ALL_BEHAVIORS",
                taxonomy_group="ALL",
                pairs=method_pairs,
                unavailable_items=method_unavailable,
            )
        )
        for behavior_id in BEHAVIOR_ORDER:
            pairs = pairs_by_behavior.get(behavior_id, [])
            missing = unavailable_by_behavior.get(behavior_id, 0)
            if not pairs and not missing:
                continue
            behavior_rows.append(
                _metric_row(
                    method=auto.name,
                    behavior_id=behavior_id,
                    taxonomy_group=BEHAVIOR_GROUPS.get(behavior_id, ""),
                    pairs=pairs,
                    unavailable_items=missing,
                )
            )
    return total_rows, behavior_rows, aligned_rows
