"""Run a negotiation using a thinking-enabled base config by default."""

from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path

from .log_paths import (
    canonical_logs_root,
    combo_id_from_config,
    run_log_dir,
    sanitize_sweep_id,
)

from .config import load_config, save_config
from .results_writer import write_results

logging.basicConfig(level=logging.INFO, format="%(message)s")
from .game import GameEndReason, get_game_by_name
from .orchestrator import RoundRunner, make_game_id, GameLimitExceededError
from .user import get_user_by_type
from .agent import get_agent


DEFAULT_CONFIG_PATH = Path("configs/buy_sell/think_harness_fixed.yaml")


def setup_log_dir(
    config: dict,
    *,
    game_id: str | None = None,
) -> tuple[Path, str]:
    """Create the log directory for one game run.

    ``game_id`` may be supplied by an orchestrating caller that needs to know
    the exact run directory before the game starts; otherwise a fresh id is
    generated here.
    """
    game_id = game_id or make_game_id()
    if sanitize_sweep_id(game_id) != game_id:
        raise ValueError(f"game_id must be one safe path component, got {game_id!r}")
    logs_root = canonical_logs_root(config.get("log_dir"))
    # Record the log root actually used (the default when the config sets none).
    config["log_dir"] = str(logs_root)
    log_dir = run_log_dir(
        logs_root,
        config.get("sweep_id"),
        combo_id_from_config(config),
        game_id,
    )
    log_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=False)
    (log_dir / "agent").mkdir(exist_ok=True)
    (log_dir / "user").mkdir(exist_ok=True)
    save_config(config, log_dir / "config.yaml")
    prompts_dir = log_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    seen: set[str] = set()
    for key in ("agent_1", "agent_2", "user_1", "user_2"):
        if key not in config:
            continue
        p = config[key].get("prompts_path")
        if not p or p in seen:
            continue
        seen.add(p)
        src = Path(p)
        if src.exists():
            shutil.copy2(src, prompts_dir / src.name)
    return log_dir, game_id


def validate_user_agent_pairing(config: dict) -> None:
    """A disabled user (type: none) always consents, so its agent must be a no-user agent."""
    for slot in (1, 2):
        user_type = (config.get(f"user_{slot}") or {}).get("type")
        if user_type == "none" and not (config.get(f"agent_{slot}") or {}).get("no_user", False):
            raise ValueError(
                f"user_{slot}.type is 'none' but agent_{slot}.no_user is not true; "
                f"set agent_{slot}.no_user: true or give user_{slot} a real type."
            )


def main(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    config: dict | None = None,
    config_name: str = "",
    game_id: str | None = None,
) -> GameEndReason:
    """Run the game to completion.

    Config can be supplied either as a path (config_path) or as a pre-built dict (config).
    config_name overrides the name written to results.json; defaults to the config file stem.
    """
    if config is None:
        config = load_config(config_path)
    if not config_name:
        config_name = config.get("config_name") or Path(config_path).stem
    validate_user_agent_pairing(config)

    log_dir, game_id = setup_log_dir(config, game_id=game_id)

    game_cls = get_game_by_name(config["game_name"])

    user_1 = get_user_by_type("user_1", config["user_1"])
    user_2 = get_user_by_type("user_2", config["user_2"])
    orchestrator_settings = config["orchestrator_settings"]
    agent_1 = get_agent(agent_id="agent_1", user=user_1, config=config["agent_1"], orchestrator_settings=orchestrator_settings)
    agent_2 = get_agent(agent_id="agent_2", user=user_2, config=config["agent_2"], orchestrator_settings=orchestrator_settings)

    game = game_cls(
        name=config["game_name"],
        agent_1=agent_1,
        agent_2=agent_2,
        game_params=config["game_params"],
    )

    runner = RoundRunner(
        game,
        game_id=game_id,
        log_dir=log_dir,
        config=orchestrator_settings,
    )
    try:
        end_reason = runner.run_game()
    except GameLimitExceededError:
        end_reason = game.end_reason
    except Exception as e:
        logging.exception("Game crashed")
        game.end_reason = game.end_reason or "error"
        game.end_error_type = type(e).__name__
        game.end_error_message = str(e)
        game.end_error_traceback = traceback.format_exc()
        end_reason = game.end_reason
    if game.end_reason is None:
        game.end_reason = end_reason
    write_results(log_dir, game, game_id, config, config_name=config_name)
    print(f"Game ended: {end_reason}")
    return end_reason


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the negotiation game.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to a game config (default: {DEFAULT_CONFIG_PATH.as_posix()})",
    )
    args = parser.parse_args()
    main(config_path=args.config)
