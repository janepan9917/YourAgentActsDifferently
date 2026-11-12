"""Helpers for parsing labeled plain-text model responses."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass


_STRICT_SECTION_RE = re.compile(
    r"(?im)^[ \t]*(THINKING|ACTION|PROPOSAL_DETAILS|OUTPUT):[ \t]*"
)

_TOLERANT_SECTION_RE = re.compile(
    r"""(?imx)
    ^[ \t]*
    (?:[-*+]\s+|\d+[.)]\s+)?       # optional Markdown list marker
    (?:\#{1,6}[ \t]+)?              # optional Markdown heading
    (?:\*\*|__)?                    # optional opening emphasis
    (THINKING|ACTION|PROPOSAL[_ -]DETAILS|OUTPUT)
    (?:\*\*|__)?                    # **ACTION**: value
    [ \t]*[:：][ \t]*               # ASCII or full-width colon
    (?:\*\*|__)?                    # **ACTION:** value
    """
)

_PROTOCOL_LABEL_ORDER = {
    "THINKING": 0,
    "ACTION": 1,
    "PROPOSAL_DETAILS": 2,
    "OUTPUT": 3,
}
_EXACT_PROTOCOL_LABEL_RE = re.compile(
    r"(?<!\S)(?:THINKING:|ACTION:|PROPOSAL_DETAILS:|OUTPUT:)"
)


def normalize_newline_only_labeled_response(
    text: str,
    *,
    require_thinking: bool,
) -> str | None:
    """Insert only missing newlines before exact protocol labels.

    This intentionally does not infer missing labels, accept aliases, repair
    punctuation, or scan arbitrary prose for an action.  A candidate is
    eligible only when exact ``ACTION:`` and ``OUTPUT:`` labels are present
    once, labels are unique and ordered, and ``THINKING:`` is present when the
    active protocol requires it.  The canonical parser remains the authority
    after normalization.
    """
    matches = list(_EXACT_PROTOCOL_LABEL_RE.finditer(text))
    if not matches:
        return None
    if text[: matches[0].start()].strip():
        return None

    labels = [match.group(0)[:-1] for match in matches]
    if len(labels) != len(set(labels)):
        return None
    if "ACTION" not in labels or "OUTPUT" not in labels:
        return None
    if require_thinking and "THINKING" not in labels:
        return None
    if labels != sorted(labels, key=_PROTOCOL_LABEL_ORDER.__getitem__):
        return None

    insertions: list[int] = []
    for match in matches:
        line_start = (
            max(
                text.rfind("\n", 0, match.start()),
                text.rfind("\r", 0, match.start()),
            )
            + 1
        )
        if text[line_start : match.start()].strip():
            insertions.append(match.start())

    if not insertions:
        return None

    normalized = text
    for index in reversed(insertions):
        normalized = normalized[:index] + "\n" + normalized[index:]
    return normalized


@dataclass(frozen=True)
class ActionDeclaration:
    """One syntactically recognizable ACTION declaration in model text."""

    start: int
    end: int
    value: str
    inline_after_thinking: bool


_ACTION_LINE_RE = re.compile(
    r"""(?imx)
    ^[ \t]*(?:[-*+]\s+|\d+[.)]\s+)?(?:\#{1,6}[ \t]+)?
    (?P<marker>(?:\*\*|__)?ACTION(?:\*\*|__)?(?=[ \t:：]))
    [ \t]*(?P<colon>[:：]?)[ \t]*(?P<value>[^\r\n]*?)
    [ \t]*$
    """
)

_INLINE_ACTION_AFTER_THINKING_RE = re.compile(
    r"""(?imx)
    ^[ \t]*(?:[-*+]\s+|\d+[.)]\s+)?(?:\#{1,6}[ \t]+)?
    (?:\*\*|__)?THINKING(?:\*\*|__)?[ \t]*[:：]
    [^\r\n]*?
    (?P<marker>(?:\*\*|__)?ACTION(?:\*\*|__)?(?=[ \t:：]))
    [ \t]*(?P<colon>[:：])[ \t]*(?P<value>[^\r\n]*?)
    [ \t]*$
    """
)


def _clean_action_value(value: str) -> str:
    """Remove only wrappers that can surround a one-line ACTION value."""
    cleaned = value.strip().strip("`").strip()
    for marker in ("**", "__"):
        if cleaned.startswith(marker) and cleaned.endswith(marker) and len(cleaned) >= 4:
            cleaned = cleaned[len(marker) : -len(marker)].strip()
    return cleaned


def find_action_declarations(text: str) -> list[ActionDeclaration]:
    """Find standalone ACTION lines and the constrained THINKING-inline error.

    We intentionally do not scan arbitrary prose.  The only inline form accepted
    is an ACTION declaration at the end of a line that begins with THINKING.
    This covers the observed formatting failure without treating quoted action
    names in ordinary OUTPUT as executable declarations.
    """
    declarations: list[ActionDeclaration] = []
    occupied: list[tuple[int, int]] = []

    for match in _ACTION_LINE_RE.finditer(text):
        marker_start = match.start("marker")
        end = match.end()
        declarations.append(
            ActionDeclaration(
                start=marker_start,
                end=end,
                value=_clean_action_value(match.group("value")),
                inline_after_thinking=False,
            )
        )
        occupied.append((match.start(), match.end()))

    for match in _INLINE_ACTION_AFTER_THINKING_RE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        declarations.append(
            ActionDeclaration(
                start=match.start("marker"),
                end=match.end(),
                value=_clean_action_value(match.group("value")),
                inline_after_thinking=True,
            )
        )

    return sorted(declarations, key=lambda declaration: declaration.start)


def normalize_single_action_declaration(text: str) -> tuple[str, str] | None:
    """Canonicalize one unambiguous ACTION declaration without changing content."""
    declarations = find_action_declarations(text)
    if len(declarations) != 1:
        return None
    declaration = declarations[0]
    if not declaration.value:
        return None
    prefix = text[: declaration.start]
    if declaration.inline_after_thinking:
        prefix = prefix.rstrip() + "\n"
    normalized = (
        prefix
        + f"ACTION: {declaration.value}"
        + text[declaration.end :]
    )
    return normalized, declaration.value


def find_duplicate_labeled_sections(
    text: str,
    labels: Iterable[str],
    *,
    tolerant: bool = False,
) -> set[str]:
    """Return requested protocol labels that occur more than once."""
    wanted = {_canonical_label(label) for label in labels}
    section_re = _TOLERANT_SECTION_RE if tolerant else _STRICT_SECTION_RE
    counts: dict[str, int] = {}
    for match in section_re.finditer(text):
        label = _canonical_label(match.group(1))
        if label not in wanted:
            continue
        counts[label] = counts.get(label, 0) + 1
    return {label for label, count in counts.items() if count > 1}


def load_json_object_without_duplicate_keys(raw: str) -> dict:
    """Decode a JSON object while rejecting duplicate keys at every level."""
    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field {key!r} is ambiguous.")
            result[key] = value
        return result

    try:
        decoded = json.loads(raw, object_pairs_hook=reject_duplicate_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {raw!r}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("Model response JSON must be an object.")
    return decoded


def extract_unique_labeled_section(
    text: str,
    label: str,
    *,
    all_labels: Iterable[str],
    tolerant: bool = False,
) -> str | None:
    """Extract one section while using every protocol label as a boundary."""
    wanted = {_canonical_label(item) for item in all_labels}
    target = _canonical_label(label)
    section_re = _TOLERANT_SECTION_RE if tolerant else _STRICT_SECTION_RE
    matches = [
        match
        for match in section_re.finditer(text)
        if _canonical_label(match.group(1)) in wanted
    ]
    target_indexes = [
        index
        for index, match in enumerate(matches)
        if _canonical_label(match.group(1)) == target
    ]
    if not target_indexes:
        return None
    if len(target_indexes) > 1:
        raise ValueError(
            f"Multiple {target} sections are ambiguous; exactly one is allowed."
        )
    index = target_indexes[0]
    match = matches[index]
    end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
    value = text[match.end() : end].strip()
    return _strip_trailing_markdown_fence(value) if tolerant else value


def _canonical_label(label: str) -> str:
    """Normalize accepted label spelling to the keys used by callers."""
    return re.sub(r"[ -]+", "_", label.upper())


def _strip_trailing_markdown_fence(value: str) -> str:
    """Remove a fence that only closes a model-wrapped response."""
    return re.sub(r"\n?[ \t]*```[ \t]*\Z", "", value).strip()


def parse_labeled_sections(
    text: str, labels: Iterable[str], *, tolerant: bool = False
) -> dict[str, str]:
    """Extract labeled sections; tolerant mode is opt-in for versioned protocols."""
    wanted = {_canonical_label(label) for label in labels}
    section_re = _TOLERANT_SECTION_RE if tolerant else _STRICT_SECTION_RE
    matches = [
        match
        for match in section_re.finditer(text)
        if _canonical_label(match.group(1)) in wanted
    ]
    sections: dict[str, str] = {}

    for i, match in enumerate(matches):
        label = _canonical_label(match.group(1))
        if label in sections:
            raise ValueError(
                f"Multiple {label} sections are ambiguous; exactly one is allowed."
            )
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        sections[label] = _strip_trailing_markdown_fence(value) if tolerant else value

    return sections
