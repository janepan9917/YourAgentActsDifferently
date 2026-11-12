"""
ConversationLogger: writes per-agent and per-user logs to log_dir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

from .action import Action


def _format_action(action: Action) -> str:
    """One-line summary of action type and output; optional thinking/details on following lines."""
    parts = [f"[{getattr(action, 'action_type', action.action_type)}]", action.output or ""]
    line = " ".join(p for p in parts if p).strip()
    out = [line]
    if getattr(action, "thinking", None) and action.thinking:
        out.append(f"  (thinking) {action.thinking}")
    if getattr(action, "action_details", None) is not None:
        out.append(f"  (proposal_details) {action.action_details}")
    return "\n".join(out)


LogTarget = Literal["agent_1", "agent_2", "user_1", "user_2", "both"]


class ConversationLogger:
    """
    Logs conversation to log_dir.
    Per agent: <agent_id>.log (conversation), <agent_id>_prompts.log (prompts).
    Per user (LLMUser): <user_id>.log (conversation), <user_id>_prompts.log (prompts).
    Safe to use from multiple threads.
    """

    def __init__(
        self,
        log_dir: Path,
        agent_ids: tuple[str, ...],
        game_id: str,
        user_ids: tuple[str, ...] = ("user_1", "user_2"),
        game: Any = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.agent_dir = self.log_dir / "agent"
        self.user_dir = self.log_dir / "user"
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.agent_ids = agent_ids
        self.user_ids = user_ids
        self.game_id = game_id
        self.game = game
        self._event_sequence = 0
        self._files: dict[str, TextIO] = {}
        self._prompt_files: dict[str, TextIO] = {}
        for agent_id in agent_ids:
            f = open(self.agent_dir / f"{agent_id}.log", "w", encoding="utf-8")
            f.write(f"game_id: {game_id}\n")
            f.flush()
            self._files[agent_id] = f
            pf = open(self.agent_dir / f"{agent_id}_prompts.log", "w", encoding="utf-8")
            pf.write(f"game_id: {game_id}\n")
            pf.flush()
            self._prompt_files[agent_id] = pf
        for user_id in user_ids:
            f = open(self.user_dir / f"{user_id}.log", "w", encoding="utf-8")
            f.write(f"game_id: {game_id}\n")
            f.flush()
            self._files[user_id] = f
            pf = open(self.user_dir / f"{user_id}_prompts.log", "w", encoding="utf-8")
            pf.write(f"game_id: {game_id}\n")
            pf.flush()
            self._prompt_files[user_id] = pf

    def _resolve_target(self, to: str) -> list[TextIO]:
        if to == "both":
            return list(self._files.values())
        if to in self._files:
            return [self._files[to]]
        return []

    def section(self, msg: str, to: LogTarget = "both") -> None:
        """Write a section header: === msg ==="""
        line = f"\n=== {msg} ===\n"
        for f in self._resolve_target(to):
            f.write(line)
            f.flush()

    def subsection(self, msg: str, to: LogTarget) -> None:
        """Write a subsection line: --- msg ---"""
        line = f"\n--- {msg} ---\n"
        for f in self._resolve_target(to):
            f.write(line)
            f.flush()

    def turn(
        self,
        speaker: str,
        action: Action,
        to: LogTarget = "both",
        *,
        invalid_reason: str | None = None,
    ) -> None:
        """Write a turn: speaker: [action_type] output (and optional thinking/details/invalid)."""
        self._event_sequence += 1
        line = f"  {speaker}: {_format_action(action)}\n"
        line += f"  (event_sequence) {self._event_sequence}\n"
        if invalid_reason:
            line += f"  (invalid) {invalid_reason}\n"
        for f in self._resolve_target(to):
            f.write(line)
            f.flush()

    def log_prompt(
        self,
        target_id: str,
        prompt: dict,
        context: str = "",
        response: str | None = None,
        usage: dict[str, Any] | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write the full prompt to the _prompts.log file for this agent or user. Optionally include the model's raw response."""
        if target_id not in self._prompt_files:
            return
        f = self._prompt_files[target_id]
        ctx = f" ({context})" if context else ""
        f.write(f"\n<<< PROMPT{ctx} >>>\n")
        if sys := prompt.get("system_prompt"):
            f.write("=== SYSTEM ===\n")
            f.write(sys)
            f.write("\n\n")
        # For respond_to_agent: user dialogue first, then agent-agent history, then instruction + question
        if prompt.get("context_prompt") is not None and prompt.get("agent_agent_history") is not None:
            f.write("=== CURRENT PROMPT ===\n")
            f.write(prompt["context_prompt"])
            f.write("\n\n")
            f.write("=== CONVERSATION HISTORY WITH AGENT ===\n")
            other_speaker = prompt.get("other_agent_id")
            assistant_speaker = prompt.get("assistant_id") or target_id
            for m in prompt["agent_agent_history"]:
                role = m.get("role", "?")
                content = m.get("content", "")
                if role == "system":
                    f.write(f"{content}\n\n")
                    continue
                if role == "user" and other_speaker:
                    role = other_speaker
                elif role == "assistant" and assistant_speaker:
                    role = assistant_speaker
                f.write(f"[{role}]\n{content}\n\n")
            if instr := prompt.get("instruction_prompt"):
                f.write(instr)
                f.write("\n")
        else:
            if hist := prompt.get("conversation_history"):
                header = prompt.get("conversation_history_label", "CONVERSATION HISTORY")
                other_speaker = prompt.get("other_agent_id")
                assistant_speaker = prompt.get("assistant_id") or target_id
                f.write(f"=== {header.upper()} ===\n")
                for m in hist:
                    role = m.get("role", "?")
                    if role == "user" and other_speaker:
                        role = other_speaker
                    elif role == "assistant" and assistant_speaker:
                        role = assistant_speaker
                    content = m.get("content", "")
                    f.write(f"[{role}]\n{content}\n\n")
            if curr := prompt.get("current_prompt"):
                f.write("=== CURRENT PROMPT ===\n")
                f.write(curr)
                f.write("\n")
        if response is not None:
            f.write("=== MODEL RESPONSE ===\n")
            f.write(response)
            if response and not response.endswith("\n"):
                f.write("\n")
            f.write("\n")
        if usage is not None or request_metadata is not None:
            f.write("=== API METADATA ===\n")
            f.write(
                json.dumps(
                    {
                        "request": request_metadata or {},
                        "usage": usage or {},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            f.write("\n\n")
        f.write("<<< END PROMPT >>>\n\n")
        f.flush()

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
        for f in self._prompt_files.values():
            f.close()
        self._prompt_files.clear()
        write_summary_log(self.log_dir, self.agent_ids, self.user_ids, self.game_id, self.game)


@dataclass
class _SummaryEvent:
    source_agent: str
    source_index: int
    section: str
    subsection: str
    speaker: str
    lines: list[str]
    explicit_turn: int | None
    sequence: int | None = None
    turn: int | None = None
    shared: bool = False
    display_label: str = ""


@dataclass
class _ParsedConversationLog:
    events: list[_SummaryEvent]
    section_headers: list[str]


_TURN_LINE_RE = re.compile(r"^  ([A-Za-z0-9_.-]+):\s")
_TURN_LABEL_RE = re.compile(r"\(turn\s+(\d+)\)", flags=re.I)
_EVENT_SEQUENCE_RE = re.compile(r"^  \(event_sequence\) (\d+)$")
_SECTION_PREFIXES = ("=== GAME START", "=== GAME STATE", "=== NEGOTIATION", "=== GAME END")


def _clean_summary_lines(lines: list[str]) -> list[str]:
    cleaned = [line.rstrip() for line in lines]
    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned


def _read_marker(lines: list[str], start: int, marker: str) -> tuple[str, int]:
    """Read a possibly multi-line section or subsection marker."""
    parts = [lines[start][len(marker) :].lstrip()]
    index = start
    while not lines[index].rstrip().endswith(marker):
        index += 1
        if index >= len(lines):
            break
        parts.append(lines[index].rstrip())
    if parts and parts[-1].rstrip().endswith(marker):
        parts[-1] = parts[-1].rstrip()[: -len(marker)].rstrip()
    return "\n".join(parts).strip(), index + 1


def _is_section_marker(line: str) -> bool:
    return line.startswith(_SECTION_PREFIXES)


def _is_subsection_marker(line: str) -> bool:
    return line.startswith("--- ") and line.endswith(" ---")


def _parse_conversation_log(path: Path, source_agent: str) -> _ParsedConversationLog:
    lines = path.read_text(encoding="utf-8").splitlines()
    events: list[_SummaryEvent] = []
    section_headers: list[str] = []
    section = ""
    subsection = ""
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if _is_section_marker(line):
            section, index = _read_marker(lines, index, "===")
            subsection = ""
            section_headers.append(section)
            continue
        if _is_subsection_marker(line):
            subsection, index = _read_marker(lines, index, "---")
            continue
        match = _TURN_LINE_RE.match(line)
        if not match:
            index += 1
            continue

        event_lines = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].rstrip()
            if (
                _is_section_marker(next_line)
                or _is_subsection_marker(next_line)
                or _TURN_LINE_RE.match(next_line)
            ):
                break
            event_lines.append(next_line)
            index += 1
        sequence = next(
            (
                int(sequence_match.group(1))
                for event_line in event_lines
                if (sequence_match := _EVENT_SEQUENCE_RE.match(event_line))
            ),
            None,
        )
        event_lines = _clean_summary_lines(
            [event_line for event_line in event_lines if not _EVENT_SEQUENCE_RE.match(event_line)]
        )
        turn_match = _TURN_LABEL_RE.search(subsection)
        events.append(
            _SummaryEvent(
                source_agent=source_agent,
                source_index=len(events),
                section=section.split("\n", 1)[0].strip(),
                subsection=subsection,
                speaker=match.group(1),
                lines=event_lines,
                explicit_turn=int(turn_match.group(1)) if turn_match else None,
                sequence=sequence,
            )
        )
    return _ParsedConversationLog(events=events, section_headers=section_headers)


def _event_signature(event: _SummaryEvent) -> str:
    return "\n".join(event.lines).strip()


def _shared_event_pairs(
    left: list[_SummaryEvent],
    right: list[_SummaryEvent],
) -> list[tuple[int, int]]:
    """Align events copied to both agent logs while excluding private user dialogue."""
    left_signatures = [_event_signature(event) for event in left]
    right_signatures = [_event_signature(event) for event in right]
    rows = len(left_signatures)
    cols = len(right_signatures)
    lengths = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(cols - 1, -1, -1):
            if left_signatures[i] == right_signatures[j]:
                lengths[i][j] = lengths[i + 1][j + 1] + 1
            else:
                lengths[i][j] = max(lengths[i + 1][j], lengths[i][j + 1])

    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < rows and j < cols:
        if left_signatures[i] == right_signatures[j]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _is_invalid_event(event: _SummaryEvent) -> bool:
    return any(line.lstrip().startswith("(invalid)") for line in event.lines)


def _is_agent_user_label(subsection: str) -> bool:
    lowered = subsection.lower()
    return "agent-user" in lowered or "user guidance" in lowered


def _merge_initial_events(
    events_by_agent: dict[str, list[_SummaryEvent]],
    agent_ids: tuple[str, ...],
) -> list[_SummaryEvent]:
    """Merge turn-zero logs while deduplicating harness events copied to both agents."""
    if not agent_ids:
        return []
    left_id = agent_ids[0]
    right_id = agent_ids[1] if len(agent_ids) > 1 else ""
    left = events_by_agent.get(left_id, [])
    right = events_by_agent.get(right_id, []) if right_id else []
    if not right:
        return left

    all_events = left + right
    if all_events and all(event.sequence is not None for event in all_events):
        by_sequence: dict[int, _SummaryEvent] = {}
        for event in all_events:
            assert event.sequence is not None
            by_sequence.setdefault(event.sequence, event)
        return [by_sequence[sequence] for sequence in sorted(by_sequence)]

    pairs = _shared_event_pairs(left, right)
    rank = {agent_id: index for index, agent_id in enumerate(agent_ids)}

    def pending_events(items: list[_SummaryEvent]) -> list[_SummaryEvent]:
        return sorted(
            items,
            key=lambda event: (
                rank.get(event.source_agent, len(rank)),
                event.source_index,
            ),
        )

    merged: list[_SummaryEvent] = []
    left_start = right_start = 0
    for left_index, right_index in pairs:
        merged.extend(
            pending_events(left[left_start:left_index] + right[right_start:right_index])
        )
        merged.append(left[left_index])
        left_start = left_index + 1
        right_start = right_index + 1
    merged.extend(pending_events(left[left_start:] + right[right_start:]))
    return merged


def _infer_private_event_metadata(events: list[_SummaryEvent], event_index: int) -> None:
    event = events[event_index]
    previous_shared = next(
        (earlier for earlier in reversed(events[:event_index]) if earlier.shared and earlier.turn is not None),
        None,
    )
    minimum_turn = 1
    if previous_shared is not None:
        minimum_turn = previous_shared.turn + (0 if _is_invalid_event(previous_shared) else 1)

    if event.explicit_turn is not None and event.explicit_turn >= minimum_turn:
        event.turn = event.explicit_turn
    else:
        for later in events[event_index + 1 :]:
            if later.explicit_turn is not None and later.explicit_turn >= minimum_turn:
                event.turn = later.explicit_turn
                break
            if later.shared and later.turn is not None:
                event.turn = later.turn
                break
        if event.turn is None:
            event.turn = minimum_turn

    if _is_agent_user_label(event.subsection) and event.explicit_turn == event.turn:
        event.display_label = event.subsection
        return
    for later in events[event_index + 1 :]:
        if later.explicit_turn is not None and later.explicit_turn != event.turn:
            break
        if later.shared and later.turn == event.turn:
            break
        if later.turn in {None, event.turn} and _is_agent_user_label(later.subsection):
            event.display_label = later.subsection
            return
    event.display_label = event.subsection or f"private dialogue for {event.source_agent}"


def _merge_negotiation_events(
    events_by_agent: dict[str, list[_SummaryEvent]],
    agent_ids: tuple[str, ...],
) -> list[_SummaryEvent]:
    if not agent_ids:
        return []
    left_id = agent_ids[0]
    right_id = agent_ids[1] if len(agent_ids) > 1 else ""
    left = events_by_agent.get(left_id, [])
    right = events_by_agent.get(right_id, []) if right_id else []
    pairs = _shared_event_pairs(left, right) if right else []

    next_turn = 1
    for left_index, right_index in pairs:
        left_event = left[left_index]
        right_event = right[right_index]
        left_event.shared = right_event.shared = True
        left_event.turn = right_event.turn = next_turn
        left_event.display_label = right_event.display_label = "agent-agent negotiation"
        if not _is_invalid_event(left_event):
            next_turn += 1

    if not right:
        for event in left:
            if event.subsection.lower().startswith("agent-agent dialogue"):
                event.shared = True
                event.turn = next_turn
                event.display_label = "agent-agent negotiation"
                if not _is_invalid_event(event):
                    next_turn += 1

    for source_events in events_by_agent.values():
        for event_index, event in enumerate(source_events):
            if not event.shared:
                _infer_private_event_metadata(source_events, event_index)

    all_events = [event for source_events in events_by_agent.values() for event in source_events]
    if all_events and all(event.sequence is not None for event in all_events):
        by_sequence: dict[int, _SummaryEvent] = {}
        for event in all_events:
            assert event.sequence is not None
            by_sequence.setdefault(event.sequence, event)
        return [by_sequence[sequence] for sequence in sorted(by_sequence)]

    rank = {agent_id: index for index, agent_id in enumerate(agent_ids)}

    def pending_events(items: list[_SummaryEvent]) -> list[_SummaryEvent]:
        return sorted(
            (event for event in items if not event.shared),
            key=lambda event: (
                event.turn if event.turn is not None else 10**9,
                rank.get(event.source_agent, len(rank)),
                event.source_index,
            ),
        )

    if not right:
        return left

    merged: list[_SummaryEvent] = []
    left_start = right_start = 0
    for left_index, right_index in pairs:
        merged.extend(pending_events(left[left_start:left_index] + right[right_start:right_index]))
        merged.append(left[left_index])
        left_start = left_index + 1
        right_start = right_index + 1
    merged.extend(pending_events(left[left_start:] + right[right_start:]))
    return merged


def _load_summary_results(log_dir: Path) -> dict[str, Any]:
    results_path = log_dir / "results.json"
    if not results_path.exists():
        return {}
    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_summary_log(
    log_dir: Path,
    agent_ids: tuple[str, ...] = ("agent_1", "agent_2"),
    user_ids: tuple[str, ...] = ("user_1", "user_2"),
    game_id: str = "",
    game: Any = None,
) -> None:
    """Write a deduplicated, turn-ordered summary from the conversation logs."""
    log_dir = Path(log_dir)
    parsed_by_agent: dict[str, _ParsedConversationLog] = {}
    for agent_id in agent_ids:
        path = log_dir / "agent" / f"{agent_id}.log"
        if path.exists():
            parsed_by_agent[agent_id] = _parse_conversation_log(path, agent_id)

    results = _load_summary_results(log_dir)
    resolved_game_id = str(game_id or results.get("game_id") or log_dir.name)
    out: list[str] = [f"# Game Summary — game_id: {resolved_game_id}", ""]

    all_section_headers = [
        header
        for parsed in parsed_by_agent.values()
        for header in parsed.section_headers
    ]
    game_state = next(
        (
            header.split("\n", 1)[1]
            for header in all_section_headers
            if header.startswith("GAME STATE\n")
        ),
        "",
    )
    initial_by_agent = {
        agent_id: [
            event
            for event in parsed.events
            if event.section in {"GAME START", "GAME STATE"}
        ]
        for agent_id, parsed in parsed_by_agent.items()
    }
    initial_events = _merge_initial_events(initial_by_agent, agent_ids)

    out.extend(["## GAME START", "", "### Turn 0", ""])
    if game_state:
        out.extend(["#### GAME STATE", "", game_state, ""])
    current_label = ""
    for event in initial_events:
        if _is_invalid_event(event):
            continue
        label = event.subsection or "initial dialogue"
        if label != current_label:
            out.extend([f"#### {label}", ""])
            current_label = label
        out.extend(event.lines)
        out.append("")

    negotiation_by_agent = {
        agent_id: [event for event in parsed.events if event.section == "NEGOTIATION"]
        for agent_id, parsed in parsed_by_agent.items()
    }
    merged_events = _merge_negotiation_events(negotiation_by_agent, agent_ids)
    out.extend(["-------------", "## NEGOTIATION", ""])
    current_turn: int | None = None
    current_label = ""
    for event in merged_events:
        if _is_invalid_event(event):
            continue
        turn = event.turn if event.turn is not None else 1
        if turn != current_turn:
            out.extend([f"### Turn {turn}", ""])
            current_turn = turn
            current_label = ""
        label = event.display_label or event.subsection or "agent-agent negotiation"
        if label != current_label:
            out.extend([f"#### {label}", ""])
            current_label = label
        out.extend(event.lines)
        out.append("")

    end_reason = getattr(game, "end_reason", None) if game is not None else None
    end_reason = end_reason or results.get("outcome")
    if not end_reason:
        for header in all_section_headers:
            match = re.match(r"GAME END \(([^)]+)\)", header)
            if match:
                end_reason = match.group(1)
                break
    error_message = getattr(game, "end_error_message", None) if game is not None else None
    result_error = results.get("error") if isinstance(results.get("error"), dict) else {}
    error_message = error_message or result_error.get("message")
    out.extend(["-------------", "## GAME END", ""])
    if end_reason:
        out.append(f"**Reason:** {end_reason}")
    if error_message:
        out.append(f"**Error:** {error_message}")
    out.append("")

    if game is not None:
        payouts = {agent.user.user_id: game.calculate_payout(agent.user) for agent in game.agents}
        rewards = {agent.agent_id: agent.current_reward for agent in game.agents}
    else:
        payouts = results.get("payouts") if isinstance(results.get("payouts"), dict) else {}
        rewards = results.get("rewards") if isinstance(results.get("rewards"), dict) else {}
    if payouts:
        out.extend(["## PAYOUTS", ""])
        for user_id in user_ids:
            if user_id in payouts:
                out.append(f"  {user_id}: {payouts[user_id]}")
        out.append("")
    if rewards:
        out.extend(["## AGENT REWARDS", ""])
        for agent_id in agent_ids:
            if agent_id in rewards:
                reward = rewards[agent_id]
                reward_text = f"{reward:.0f}" if isinstance(reward, (int, float)) else str(reward)
                out.append(f"  {agent_id}: {reward_text}")
        out.append("")

    (log_dir / "summary.log").write_text("\n".join(out).strip() + "\n", encoding="utf-8")
