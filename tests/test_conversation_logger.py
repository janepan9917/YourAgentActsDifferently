import tempfile
import unittest
from pathlib import Path

from negsim.conversation_logger import write_summary_log


class SummaryLogTests(unittest.TestCase):
    def test_summary_preserves_raw_multiline_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            agent_dir = log_dir / "agent"
            agent_dir.mkdir()

            log_text = """game_id: game-x

=== NEGOTIATION ===

--- agent-agent dialogue (negotiation, round 1) ---
  agent_1: [MESSAGE_USER] Please choose:
- first option
- second option
  (thinking) first line
second line
  (proposal_details) {"price": 10,
"date": "2026-07-10"}
"""
            (agent_dir / "agent_1.log").write_text(log_text, encoding="utf-8")
            (agent_dir / "agent_2.log").write_text(log_text, encoding="utf-8")

            write_summary_log(log_dir, agent_ids=("agent_1", "agent_2"), game_id="game-x")

            summary = (log_dir / "summary.log").read_text(encoding="utf-8")
            self.assertIn("Please choose:\n- first option\n- second option", summary)
            self.assertIn("(thinking) first line\nsecond line", summary)
            self.assertIn('(proposal_details) {"price": 10,\n"date": "2026-07-10"}', summary)
            self.assertIn("### Turn 1", summary)
            self.assertEqual(summary.count("#### agent-agent negotiation"), 1)
            self.assertEqual(
                summary.count("agent_1: [MESSAGE_USER] Please choose:"), 1
            )

    def test_summary_merges_agent_and_user_dialogue_by_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            agent_dir = log_dir / "agent"
            agent_dir.mkdir()

            agent_1_log = """game_id: game-x

=== GAME START ===

=== GAME STATE
Turn: 0 of 4 ===

--- initial user guidance: agent_1 <-> user_1 ---
  agent_2: [OPENING] shared-scripted-opening
  user_1: [FURTHER_INSTRUCT] initial-agent-1

=== NEGOTIATION ===

--- agent-user final confirmation: agent_1 <-> user_1 (turn 1) ---
  agent_1: [MESSAGE_USER] private-agent-1-turn-1
  user_1: [FURTHER_INSTRUCT] private-user-1-turn-1
  agent_1: [PROPOSE] shared-turn-1
  agent_1: [ASK_USER] ask-agent-1

--- agent-user dialogue: agent_1 <-> user_1 (turn 2) ---
  agent_1: [MESSAGE_USER] agent-1-user-turn-2
  user_1: [FURTHER_INSTRUCT] user-1-turn-2
  agent_2: [PROPOSE] shared-turn-2
"""
            agent_2_log = """game_id: game-x

=== GAME START ===

=== GAME STATE
Turn: 0 of 4 ===

--- initial user guidance: agent_2 <-> user_2 ---
  agent_2: [OPENING] shared-scripted-opening
  user_2: [FURTHER_INSTRUCT] initial-agent-2

=== NEGOTIATION ===

--- agent-agent dialogue (negotiation, max 4 turns) ---
  agent_1: [PROPOSE] shared-turn-1
  agent_2: [ASK_USER] ask-agent-2

--- agent-user dialogue: agent_2 <-> user_2 (turn 2) ---
  agent_2: [MESSAGE_USER] agent-2-user-turn-2
  user_2: [FURTHER_INSTRUCT] user-2-turn-2
  agent_2: [PROPOSE] shared-turn-2
"""
            (agent_dir / "agent_1.log").write_text(agent_1_log, encoding="utf-8")
            (agent_dir / "agent_2.log").write_text(agent_2_log, encoding="utf-8")

            write_summary_log(log_dir, agent_ids=("agent_1", "agent_2"), game_id="game-x")

            summary = (log_dir / "summary.log").read_text(encoding="utf-8")
            self.assertEqual(summary.count("shared-turn-1"), 1)
            self.assertEqual(summary.count("shared-turn-2"), 1)
            self.assertEqual(summary.count("shared-scripted-opening"), 1)
            self.assertLess(summary.index("shared-scripted-opening"), summary.index("initial-agent-1"))
            self.assertLess(summary.index("initial-agent-1"), summary.index("initial-agent-2"))
            self.assertLess(summary.index("agent-1-user-turn-2"), summary.index("agent-2-user-turn-2"))
            self.assertLess(summary.index("agent-2-user-turn-2"), summary.index("shared-turn-2"))
            self.assertEqual(summary.count("### Turn 1"), 1)
            self.assertEqual(summary.count("### Turn 2"), 1)
            turn_2_block = summary.split("### Turn 2", 1)[1]
            self.assertLess(
                turn_2_block.index("#### agent-user dialogue: agent_1 <-> user_1 (turn 2)"),
                turn_2_block.index("ask-agent-1"),
            )
            self.assertNotIn("agent-user final confirmation: agent_1 <-> user_1 (turn 1)", turn_2_block)

    def test_summary_omits_invalid_messages_without_advancing_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            agent_dir = log_dir / "agent"
            agent_dir.mkdir()

            shared_log = """game_id: game-x

=== GAME START ===

--- initial user guidance ---
  user_1: [FURTHER_INSTRUCT] invalid-initial-message
  (invalid) malformed initial guidance

=== NEGOTIATION ===

--- agent-agent dialogue (negotiation, max 4 turns) ---
  agent_1: [PROPOSE] invalid-shared-message
  (invalid) malformed proposal
  agent_1: [PROPOSE] valid-turn-1
  agent_2: [MESSAGE_USER] valid-turn-2
"""
            agent_1_log = shared_log.replace(
                "  agent_2: [MESSAGE_USER] valid-turn-2",
                """  agent_1: [ASK_USER] invalid-private-message
  (invalid) ASK USER is not available
  agent_2: [MESSAGE_USER] valid-turn-2""",
            )
            (agent_dir / "agent_1.log").write_text(agent_1_log, encoding="utf-8")
            (agent_dir / "agent_2.log").write_text(shared_log, encoding="utf-8")

            write_summary_log(log_dir, agent_ids=("agent_1", "agent_2"), game_id="game-x")

            summary = (log_dir / "summary.log").read_text(encoding="utf-8")
            self.assertNotIn("(invalid)", summary)
            self.assertNotIn("invalid-initial-message", summary)
            self.assertNotIn("invalid-shared-message", summary)
            self.assertNotIn("invalid-private-message", summary)
            self.assertIn("valid-turn-1", summary)
            self.assertIn("valid-turn-2", summary)
            self.assertEqual(summary.count("### Turn 1"), 1)
            self.assertEqual(summary.count("### Turn 2"), 1)

    def test_summary_uses_global_event_sequence_within_the_same_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            agent_dir = log_dir / "agent"
            agent_dir.mkdir()

            agent_1_log = """game_id: game-x

=== NEGOTIATION ===

--- bilateral final confirmation: agent_1 <-> user_1 (turn 2) ---
  agent_1: [REQUEST_USER_CONFIRMATION] confirm-after-accept
  (event_sequence) 6
  user_1: [CONSENT] approved-after-accept
  (event_sequence) 7
"""
            agent_2_log = """game_id: game-x

=== NEGOTIATION ===

--- agent-user dialogue: agent_2 <-> user_2 (turn 2) ---
  agent_2: [ASK USER] ask-first
  (event_sequence) 1
  agent_2: [MESSAGE_USER] report-second
  (event_sequence) 2
  user_2: [FURTHER_INSTRUCT] reply-third
  (event_sequence) 3
  agent_2: [MESSAGE_USER_AND_CONTINUE_NEGOTIATION] continue-fourth
  (event_sequence) 4
  agent_2: [ACCEPT_AND_CONFIRM] accept-fifth
  (event_sequence) 5
"""
            (agent_dir / "agent_1.log").write_text(agent_1_log, encoding="utf-8")
            (agent_dir / "agent_2.log").write_text(agent_2_log, encoding="utf-8")

            write_summary_log(log_dir, agent_ids=("agent_1", "agent_2"), game_id="game-x")

            summary = (log_dir / "summary.log").read_text(encoding="utf-8")
            expected = [
                "ask-first",
                "report-second",
                "reply-third",
                "continue-fourth",
                "accept-fifth",
                "confirm-after-accept",
                "approved-after-accept",
            ]
            positions = [summary.index(value) for value in expected]
            self.assertEqual(positions, sorted(positions))
            self.assertNotIn("event_sequence", summary)


if __name__ == "__main__":
    unittest.main()
