"""
Dialogue: conversation history between two speakers (e.g. A_1 + A_2, A_1 + U_1, A_2 + U_2).
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from .action import Action


SpeakerKind = Literal["user", "agent"]


def format_action_for_history(action: Action | None, *, speaker: str | None = None) -> str:
    """Render public action content, including the exact terms of a proposal."""
    if action is None:
        return ""

    output = getattr(action, "output", "") or ""
    action_type = getattr(getattr(action, "action_type", None), "value", "")
    details = getattr(action, "action_details", None)
    detail_bearing_actions = {
        "PROPOSE",
        "REQUEST_USER_CONFIRMATION",
        "ACCEPT AND CONFIRM WITH USER",
    }
    if action_type not in detail_bearing_actions or details is None:
        return output

    source = f" FROM {speaker}" if speaker else ""
    rendered_details = json.dumps(
        details,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    detail_label = {
        "PROPOSE": "STRUCTURED PROPOSAL",
        "REQUEST_USER_CONFIRMATION": "SYSTEM-SUPPLIED CONFIRMATION PROPOSAL",
        "ACCEPT AND CONFIRM WITH USER": "SYSTEM-SUPPLIED ACCEPTANCE TARGET",
    }[action_type]
    proposal = (
        f"[{detail_label}{source}]\n"
        f"PROPOSAL_DETAILS: {rendered_details}"
    )
    return f"{output}\n\n{proposal}" if output else proposal


class Turn(TypedDict):
    """One turn in the conversation."""
    speaker: SpeakerKind | str
    num_tokens: int
    action: Action
    counts_toward_ask_user_limit: bool


class DialogueChannelView:
    """A non-owning channel view over one chronologically ordered Dialogue."""

    def __init__(
        self,
        dialogue: "Dialogue",
        channel: str,
        speaker_1: SpeakerKind | str,
        speaker_2: SpeakerKind | str,
    ) -> None:
        self._dialogue = dialogue
        self.channel = channel
        self.speaker_1 = speaker_1
        self.speaker_2 = speaker_2

    @property
    def conversation_history(self) -> list[dict[str, Any]]:
        return [
            turn
            for turn in self._dialogue.conversation_history
            if turn.get("channel") == self.channel
            and turn.get("visible_to_user", True)
        ]

    @property
    def current_turn(self) -> int:
        return len(self.conversation_history)

    @property
    def current_speaker(self) -> SpeakerKind | str:
        return self.speaker_1 if self.current_turn % 2 == 0 else self.speaker_2

    def add_turn(
        self,
        speaker: SpeakerKind | str,
        num_tokens: int,
        action: Action,
        *,
        counts_for_game_state: bool = True,
        counts_toward_ask_user_limit: bool = True,
        round_num: int | None = None,
        turn_num: int | None = None,
        visible_to_user: bool = True,
    ) -> None:
        self._dialogue.add_turn(
            speaker,
            num_tokens,
            action,
            counts_for_game_state=counts_for_game_state,
            counts_toward_ask_user_limit=counts_toward_ask_user_limit,
            round_num=round_num,
            turn_num=turn_num,
            channel=self.channel,
            visible_to_user=visible_to_user,
        )


def count_ask_user_messages(dialogue: "Dialogue") -> int:
    """Count private messages that consume the global ASK USER budget."""
    return sum(
        1
        for turn in dialogue.conversation_history
        if turn.get("counts_toward_ask_user_limit", True)
    )


class Dialogue:
    """
    Conversation between two speakers. Speakers take turns.
    """

    def __init__(
        self,
        speaker_1: SpeakerKind | str,
        speaker_2: SpeakerKind | str,
    ) -> None:
        self.speaker_1 = speaker_1
        self.speaker_2 = speaker_2
        self.current_turn: int = 0
        self.conversation_history: list[dict[str, Any]] = []

    @property
    def current_speaker(self) -> SpeakerKind | str:
        """Who speaks next: speaker_1 on even turns, speaker_2 on odd."""
        return self.speaker_1 if self.current_turn % 2 == 0 else self.speaker_2

    def add_turn(
        self,
        speaker: SpeakerKind | str,
        num_tokens: int,
        action: Action,
        *,
        counts_for_game_state: bool = True,
        counts_toward_ask_user_limit: bool = True,
        round_num: int | None = None,
        turn_num: int | None = None,
        channel: str | None = None,
        visible_to_user: bool = True,
    ) -> None:
        """Append a turn and advance current_turn."""
        self.conversation_history.append({
            "speaker": speaker,
            "num_tokens": num_tokens,
            "action": action,
            "counts_for_game_state": counts_for_game_state,
            "counts_toward_ask_user_limit": counts_toward_ask_user_limit,
            "round_num": round_num,
            "turn_num": turn_num,
            "channel": channel,
            "visible_to_user": visible_to_user,
        })
        self.current_turn += 1

    def channel_view(
        self,
        channel: str,
        speaker_1: SpeakerKind | str,
        speaker_2: SpeakerKind | str,
    ) -> DialogueChannelView:
        """Return a filtered view that appends into this same underlying list."""
        return DialogueChannelView(self, channel, speaker_1, speaker_2)
