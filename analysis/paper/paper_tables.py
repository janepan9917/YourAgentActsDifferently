"""Presentation-only helpers for the generated paper tables.

Statistics are computed in the table scripts; LaTeX layout lives in
table_templates.
"""
import math
import re
from pathlib import Path


TEMPLATES = Path(__file__).resolve().parent / "table_templates"
# The templates contain the two-level headings for exactly these columns.
MODEL_ORDER = (
    "claudehaiku45", "claudesonnet5", "deepseek_v4_flash", "gpt56luna",
    "gpt56terra", "qwen36_plus", "qwen37_plus", "qwen38_flash",
)


def require_model_order(models) -> None:
    """Check that the model columns match the table template headings."""
    if tuple(models) != MODEL_ORDER:
        raise ValueError("Model columns do not match the table template headings.")


def render_template(name: str, **fields: str) -> str:
    """Fill explicit markers without interpreting LaTeX braces or commands."""
    template = (TEMPLATES / f"{name}.tex").read_text(encoding="utf-8")
    required = set(re.findall(r"@@([A-Z_]+)@@", template))
    if required != set(fields):
        raise ValueError(f"{name}: expected fields {sorted(required)}, got {sorted(fields)}")
    rendered = re.sub(r"@@([A-Z_]+)@@", lambda match: fields[match[1]], template)
    if re.search(r"@@[A-Z_]+@@", rendered):
        raise ValueError(f"Unfilled template marker in {name}")
    return rendered


def number(value: float) -> str:
    """One decimal, no positive sign, and no negative rounded zero."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("A displayed estimate or CI must be finite.")
    result = f"{value:.1f}"
    return "0.0" if result == "-0.0" else result


def star(pvalue: float | None) -> str:
    """Star for two-sided p < .05; no test means no star."""
    if pvalue is None or math.isnan(float(pvalue)):
        return ""
    if not math.isfinite(float(pvalue)) or not 0 <= float(pvalue) <= 1:
        raise ValueError("A p-value must be in [0, 1], or missing for an untested cell.")
    return "*" if float(pvalue) < .05 else ""


def signed_parts(value: float) -> tuple[str, str]:
    text = number(value)
    return (r"$-$", text[1:]) if text.startswith("-") else ("", text)


def tex_number(value: float) -> str:
    sign, magnitude = signed_parts(value)
    return sign + magnitude


def optional_pattern(pattern: str, default: str) -> str:
    return "" if pattern == default else f"[{pattern}]"


def unsigned_pattern(values) -> str:
    digits = max(len(number(abs(float(value))).split(".")[0]) for value in values)
    return "0" * digits + ".0"


def payout_pattern(values, ci_half: float) -> str:
    """Center the column's signed envelope, including its bottom +/- CI cell.

    A minus plus one digit is at least as wide as two unsigned digits in the
    paper font. Two-digit negatives require the wider envelope used by Sonnet.
    """
    displayed = [number(value) for value in values]
    digits = max([1, len(number(ci_half).split(".")[0])] + [
        len(value.lstrip("-").split(".")[0]) - (not value.startswith("-"))
        for value in displayed
    ])
    return r"$-$" + "0" * digits + ".0"


def payout_value(value: float, pvalue=None, *, pattern=r"$-$0.0", ci=False) -> str:
    sign, magnitude = signed_parts(value)
    if ci:
        if float(value) < 0:
            raise ValueError("CI half-width must be nonnegative.")
        sign = r"$\pm$"
    option = optional_pattern(pattern, r"$-$0.0")
    return rf"\PayoutValue{option}{{{sign}}}{{{magnitude}}}{{{'' if ci else star(pvalue)}}}"


def payout_ci(value: float, ci_half: float, pvalue=None) -> str:
    sign, magnitude = signed_parts(value)
    if float(ci_half) < 0:
        raise ValueError("CI half-width must be nonnegative.")
    return rf"\PayoutCI{{{sign}}}{{{magnitude}}}{{{number(ci_half)}}}{{{star(pvalue)}}}"


def mb_value(value: float, pvalue=None, *, pattern="00.0") -> str:
    option = optional_pattern(pattern, "00.0")
    return rf"\MBValue{option}{{{number(value)}}}{{{star(pvalue)}}}"


def mb_ci(value: float, ci_half: float, pvalue=None) -> str:
    if float(ci_half) < 0:
        raise ValueError("CI half-width must be nonnegative.")
    return rf"\MBCI{{{number(value)}}}{{{number(ci_half)}}}{{{star(pvalue)}}}"


def ablation_value(value: float, ci_half: float, pvalue=None, *, pattern="00.0") -> str:
    if float(ci_half) < 0:
        raise ValueError("CI half-width must be nonnegative.")
    option = optional_pattern(pattern, "00.0")
    return rf"\AblationValue{option}{{{tex_number(value)}}}{{{number(ci_half)}}}{{{star(pvalue)}}}"


def row(cells) -> str:
    return "        " + " & ".join(cells) + r" \\"
