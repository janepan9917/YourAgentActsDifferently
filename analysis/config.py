"""Paths and constants shared by the analysis code."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Data folder (raw runs, labels, ablation results, processed tables)."""
    return Path(os.environ.get("ANALYSIS_DATA", REPO / "data")).resolve()


def processed_dir() -> Path:
    """Processed tables: the default inputs of the figures and tables."""
    return Path(os.environ.get("ANALYSIS_PROCESSED", data_dir() / "processed")).resolve()


def out_dir() -> Path:
    """The paper's figures and tables."""
    return Path(os.environ.get("ANALYSIS_OUT", REPO / "outputs")).resolve()


def work_dir() -> Path:
    """Intermediate files passed between the steps of ``python -m analysis all``."""
    return Path(os.environ.get("ANALYSIS_WORK", REPO / "work")).resolve()


# Runs with these outcomes are execution failures: excluded from every analysis.
FAILED_OUTCOMES = {"error", "invalid_model_output", "invalid", "interrupted"}

# The seven games, in the keys used by the processed table.
GAMES = ["buy_sell", "ultimatum", "casino", "deal_or_no_deal",
         "pb_capit_anchored_01", "pb_leak_price_01", "pb_sanity_negot_01"]
GAME_LABELS = {"buy_sell": "Buy-Sell", "ultimatum": "Ultimatum", "casino": "CaSiNo",
               "deal_or_no_deal": "Deal or No Deal", "pb_capit_anchored_01": "Service Contract",
               "pb_leak_price_01": "Car Sale", "pb_sanity_negot_01": "Bike Sale"}

MODELS = {  # model id -> short key used by the processed table
    "claude-haiku-4-5": "claudehaiku45", "claude-sonnet-5": "claudesonnet5",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek_v4_flash", "gpt-5.6-luna": "gpt56luna",
    "gpt-5.6-terra": "gpt56terra", "Qwen/Qwen3.6-Plus": "qwen36_plus",
    "Qwen/Qwen3.7-Plus": "qwen37_plus", "Qwen/Qwen3.8-Flash": "qwen38_flash",
}

USER_FACING = ["MB_OFFER_STATE_MISREPORTING", "MB_HALLUCINATED_ACTION", "MB_UNSUPPORTED_URGENCY"]
COUNTERPARTY_FACING = ["MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING", "MB_OTHER_AGENT_UNSUPPORTED_URGENCY",
                       "MB_EXTRA_PROTOCOL_CONTRACTING"]
MB_BEHAVIORS = USER_FACING + COUNTERPARTY_FACING
# Preference hardening is labeled by the same judge but is a negotiation tactic, not
# misaligned behavior: it never enters MB rates or counts.
PREFERENCE_HARDENING = "MB_PREFERENCE_HARDENING"
ALL_LABELS = MB_BEHAVIORS + [PREFERENCE_HARDENING]
