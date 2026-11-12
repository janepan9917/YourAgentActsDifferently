"""PrincipalBench (stripped) settings: proposal sidecars and private-info-only contexts.

The paper uses three stripped PrincipalBench settings (pb-capit-anchored-01,
pb-leak-price-01, pb-sanity-negot-01). Each setting combines

* a proposal/payout sidecar: ``configs/principal_bench/proposals/<id>.yaml``
* a private-info-only context: ``vendor/principal-loyalty-items/stripped_down/v0/<id>.json``
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRINCIPAL_BENCH_CONFIG_DIR = REPO_ROOT / "configs" / "principal_bench"
DEFAULT_PROPOSALS_DIR = PRINCIPAL_BENCH_CONFIG_DIR / "proposals"
DEFAULT_STRIPPED_DATA_ROOT = REPO_ROOT / "vendor" / "principal-loyalty-items" / "stripped_down"
PROPOSAL_SCHEMA_VERSION = 4
ROLE_IDS = ("party_1", "party_2")
FIELD_TYPES = frozenset({"boolean", "integer", "number", "string"})
PAYOUT_MODES = frozenset({"monetary_surplus", "source_metric", "unscored"})
CONDITION_OPERATORS = frozenset(
    {
        "eq",
        "gt",
        "gte",
        "in",
        "lt",
        "lte",
        "ne",
        "not_in",
    }
)
CONTEXT_KEYS = (
    "id",
    "user_1_private_info",
    "user_2_private_info",
)


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a YAML mapping: {path}")
    return data


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must contain valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean.")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}.") from exc
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return score


def _validate_condition_value(value: Any, *, field_type: str, label: str) -> None:
    if field_type in {"integer", "number"}:
        numeric = _finite_score(value, label=label)
        if field_type == "integer" and not numeric.is_integer():
            raise ValueError(f"{label} must be an integer.")
        return
    if field_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{label} must be boolean.")
        return
    _nonempty_string(value, label=label)


def _validate_payout_condition(
    condition: Any,
    *,
    field_types: Mapping[str, str],
    label: str,
) -> None:
    if not isinstance(condition, Mapping):
        raise ValueError(f"{label} must be a mapping.")

    logical_keys = {"all", "any"} & set(condition)
    if logical_keys:
        if len(condition) != 1 or len(logical_keys) != 1:
            raise ValueError(f"{label} must contain exactly one logical operator.")
        operator = next(iter(logical_keys))
        children = condition[operator]
        if not isinstance(children, list) or not children:
            raise ValueError(f"{label}.{operator} must be a non-empty list.")
        for index, child in enumerate(children):
            _validate_payout_condition(
                child,
                field_types=field_types,
                label=f"{label}.{operator}[{index}]",
            )
        return

    operator = condition.get("op")
    if operator not in CONDITION_OPERATORS:
        raise ValueError(
            f"{label}.op must be one of {sorted(CONDITION_OPERATORS)}."
        )
    field_name = condition.get("field")
    if field_name not in field_types:
        raise ValueError(f"{label}.field references unknown field {field_name!r}.")
    field_type = field_types[str(field_name)]
    if field_type == "string":
        raise ValueError(f"{label} cannot reference a text proposal field.")
    if operator in {"lt", "lte", "gt", "gte"} and field_type not in {
        "integer",
        "number",
    }:
        raise ValueError(f"{label}.{operator} requires a numeric field.")

    if operator in {"in", "not_in"}:
        if set(condition) != {"field", "op", "values"}:
            raise ValueError(f"{label} membership condition requires only values.")
        values = condition["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{label}.values must be a non-empty list.")
        for index, value in enumerate(values):
            _validate_condition_value(
                value,
                field_type=field_type,
                label=f"{label}.values[{index}]",
            )
        return

    if set(condition) != {"field", "op", "value"}:
        raise ValueError(f"{label} comparison condition requires only value.")
    _validate_condition_value(
        condition["value"], field_type=field_type, label=f"{label}.value"
    )


def validate_principal_bench_proposal(
    raw: Mapping[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    """Validate one source-faithful proposal and post-deal payout specification."""
    if not isinstance(raw, Mapping):
        raise ValueError("PrincipalBench proposal sidecar must be a mapping.")
    sidecar = dict(raw)
    location = f" in {path}" if path is not None else ""
    if sidecar.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported PrincipalBench proposal schema_version{location}: "
            f"{sidecar.get('schema_version')!r}."
        )

    setting_id = _nonempty_string(sidecar.get("id"), label=f"proposal id{location}")
    if path is not None and path.stem != setting_id:
        raise ValueError(f"Proposal id {setting_id!r} must match {path.name!r}.")
    for key in ("mechanic_family", "public_context"):
        _nonempty_string(sidecar.get(key), label=f"{setting_id}.{key}")

    duplicated_context_keys = (set(CONTEXT_KEYS) - {"id"}) & set(sidecar)
    if duplicated_context_keys:
        raise ValueError(
            f"{setting_id} proposal sidecar duplicates context fields: "
            f"{sorted(duplicated_context_keys)}."
        )

    proposal = sidecar.get("proposal")
    if not isinstance(proposal, Mapping):
        raise ValueError(f"{setting_id}.proposal must be a mapping.")
    _nonempty_string(
        proposal.get("description"), label=f"{setting_id}.proposal.description"
    )
    fields = proposal.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError(f"{setting_id}.proposal.fields must be a non-empty mapping.")

    field_types: dict[str, str] = {}
    numeric_fields: set[str] = set()
    for field_name, raw_field in fields.items():
        field_label = f"{setting_id}.proposal.fields.{field_name}"
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError(f"{setting_id} has an invalid proposal field name.")
        if not isinstance(raw_field, Mapping):
            raise ValueError(f"{field_label} must be a mapping.")
        _nonempty_string(raw_field.get("label"), label=f"{field_label}.label")
        field_type = raw_field.get("type")
        if field_type not in FIELD_TYPES:
            raise ValueError(
                f"{field_label}.type must be one of {sorted(FIELD_TYPES)}, "
                f"got {field_type!r}."
            )
        field_types[field_name] = str(field_type)
        if "description" in raw_field:
            _nonempty_string(
                raw_field["description"], label=f"{field_label}.description"
            )
        if "unit" in raw_field:
            _nonempty_string(raw_field["unit"], label=f"{field_label}.unit")
        if field_type in {"string", "boolean"}:
            if "values" in raw_field or any(
                key in raw_field for key in ("minimum", "maximum")
            ):
                raise ValueError(
                    f"{field_label} cannot enumerate values or define numeric bounds."
                )
        else:
            numeric_fields.add(field_name)
            if "values" in raw_field:
                raise ValueError(
                    f"{field_label} numeric fields must use ranges, not enumerated values."
                )
            minimum = (
                _finite_score(raw_field["minimum"], label=f"{field_label}.minimum")
                if "minimum" in raw_field
                else None
            )
            maximum = (
                _finite_score(raw_field["maximum"], label=f"{field_label}.maximum")
                if "maximum" in raw_field
                else None
            )
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{field_label}.minimum cannot exceed maximum.")

    roles = sidecar.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(ROLE_IDS):
        raise ValueError(f"{setting_id}.roles must define exactly {list(ROLE_IDS)}.")
    for role_id in ROLE_IDS:
        role = roles[role_id]
        role_label = f"{setting_id}.roles.{role_id}"
        if not isinstance(role, Mapping):
            raise ValueError(f"{role_label} must be a mapping.")
        for key in ("name", "public_description"):
            _nonempty_string(role.get(key), label=f"{role_label}.{key}")

    payout = sidecar.get("payout")
    if not isinstance(payout, Mapping):
        raise ValueError(f"{setting_id}.payout must be a mapping.")
    payout_mode = payout.get("mode")
    if payout_mode not in PAYOUT_MODES:
        raise ValueError(
            f"{setting_id}.payout.mode must be one of {sorted(PAYOUT_MODES)}."
        )
    if payout_mode == "unscored":
        if set(payout) != {"mode"}:
            raise ValueError(
                f"{setting_id}.payout unscored mode cannot contain invented values."
            )
    else:
        allowed_payout_keys = {
            "mode",
            "units",
            "no_deal",
            "formulas",
            "eligibility",
        }
        unexpected_payout_keys = set(payout) - allowed_payout_keys
        if unexpected_payout_keys:
            raise ValueError(
                f"{setting_id}.payout has unexpected keys: "
                f"{sorted(unexpected_payout_keys)}."
            )
        units = payout.get("units")
        if not isinstance(units, Mapping) or set(units) != set(ROLE_IDS):
            raise ValueError(
                f"{setting_id}.payout.units must define exactly {list(ROLE_IDS)}."
            )
        for role_id in ROLE_IDS:
            _nonempty_string(
                units[role_id], label=f"{setting_id}.payout.units.{role_id}"
            )
        if payout_mode == "monetary_surplus" and len(set(units.values())) != 1:
            raise ValueError(
                f"{setting_id}.payout monetary surplus units must match."
            )

        no_deal = payout.get("no_deal")
        if not isinstance(no_deal, Mapping) or set(no_deal) != set(ROLE_IDS):
            raise ValueError(
                f"{setting_id}.payout.no_deal must define exactly {list(ROLE_IDS)}."
            )
        for role_id in ROLE_IDS:
            _finite_score(
                no_deal[role_id], label=f"{setting_id}.payout.no_deal.{role_id}"
            )

        formulas = payout.get("formulas")
        if not isinstance(formulas, Mapping) or set(formulas) != set(ROLE_IDS):
            raise ValueError(
                f"{setting_id}.payout.formulas must define exactly {list(ROLE_IDS)}."
            )
        for role_id in ROLE_IDS:
            formula = formulas[role_id]
            formula_label = f"{setting_id}.payout.formulas.{role_id}"
            if not isinstance(formula, Mapping):
                raise ValueError(f"{formula_label} must be a mapping.")
            _finite_score(formula.get("constant", 0), label=f"{formula_label}.constant")
            if set(formula) - {"constant", "terms"}:
                raise ValueError(
                    f"{formula_label} may contain only constant and terms."
                )
            terms = formula.get("terms")
            if not isinstance(terms, Mapping):
                raise ValueError(f"{formula_label}.terms must be a mapping.")
            unknown_terms = set(terms) - set(numeric_fields)
            if unknown_terms:
                raise ValueError(
                    f"{formula_label}.terms references non-numeric fields: "
                    f"{sorted(unknown_terms)}."
                )
            for field_name, coefficient in terms.items():
                _finite_score(
                    coefficient, label=f"{formula_label}.terms.{field_name}"
                )

        eligibility = payout.get("eligibility", {})
        if not isinstance(eligibility, Mapping):
            raise ValueError(f"{setting_id}.payout.eligibility must be a mapping.")
        unknown_eligibility_roles = set(eligibility) - set(ROLE_IDS)
        if unknown_eligibility_roles:
            raise ValueError(
                f"{setting_id}.payout.eligibility has unknown roles: "
                f"{sorted(unknown_eligibility_roles)}."
            )
        for role_id, condition in eligibility.items():
            _validate_payout_condition(
                condition,
                field_types=field_types,
                label=f"{setting_id}.payout.eligibility.{role_id}",
            )

    return sidecar


def validate_principal_bench_stripped_context(
    raw: Mapping[str, Any], *, path: Path | None = None
) -> dict[str, Any]:
    """Validate a private-info-only context: ``id``, ``user_1_private_info``, ``user_2_private_info``."""
    if not isinstance(raw, Mapping):
        raise ValueError("PrincipalBench stripped context must be a mapping.")
    context = dict(raw)
    location = f" in {path}" if path is not None else ""
    missing = [key for key in CONTEXT_KEYS if key not in context]
    if missing:
        raise ValueError(
            f"PrincipalBench stripped context{location} is missing {missing}."
        )
    unexpected = sorted(set(context) - set(CONTEXT_KEYS))
    if unexpected:
        raise ValueError(
            f"PrincipalBench stripped context{location} contains non-context fields: "
            f"{unexpected}."
        )

    item_id = _nonempty_string(context.get("id"), label=f"context id{location}")
    if path is not None and path.stem != item_id:
        raise ValueError(f"Context id {item_id!r} must match {path.name!r}.")
    for key in ("user_1_private_info", "user_2_private_info"):
        _nonempty_string(context.get(key), label=f"{item_id}.{key}")
    return context


def resolve_principal_bench_stripped_context_path(
    setting_id: str, *, stripped_root: str | Path = DEFAULT_STRIPPED_DATA_ROOT
) -> Path:
    """Find one setting's private-info-only context file by id under ``stripped_root``
    (any version subfolder)."""
    root = Path(stripped_root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    matches = sorted(root.glob(f"*/{setting_id}.json"))
    if not matches:
        raise FileNotFoundError(
            f"No stripped-down PrincipalBench context found for {setting_id!r} "
            f"under {root}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple stripped-down PrincipalBench contexts found for "
            f"{setting_id!r}: {[str(m) for m in matches]}."
        )
    return matches[0]


def load_principal_bench_stripped_setting(*, setting_id: str) -> dict[str, Any]:
    """Load one private-info-only PrincipalBench context with its proposal sidecar.

    The sidecar defines the public context, roles, proposal fields and payout
    rules; the stripped context holds only ``user_1_private_info`` and
    ``user_2_private_info``. Turn order is left to the harness's coin toss.
    """
    proposal_path = DEFAULT_PROPOSALS_DIR / f"{setting_id}.yaml"
    if not proposal_path.exists():
        raise FileNotFoundError(f"No PrincipalBench proposal sidecar for {setting_id!r}: {proposal_path}")
    sidecar = validate_principal_bench_proposal(
        _load_yaml_mapping(proposal_path, label="PrincipalBench proposal sidecar"),
        path=proposal_path,
    )

    stripped_path = resolve_principal_bench_stripped_context_path(setting_id)
    context = validate_principal_bench_stripped_context(
        _load_json_mapping(stripped_path, label="PrincipalBench stripped context"),
        path=stripped_path,
    )
    if context["id"] != sidecar["id"]:
        raise ValueError(
            f"Context/proposal id mismatch: {context['id']!r} != {sidecar['id']!r}."
        )

    setting = dict(context)
    setting["source"] = {
        "stripped_path": str(stripped_path.relative_to(REPO_ROOT)),
    }
    setting["adapter"] = sidecar
    return setting
