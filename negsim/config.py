"""
Config: load YAML config file for the negotiation game.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load config from YAML file. Raises if PyYAML not installed or file missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if yaml is None:
        raise ImportError("PyYAML required for config. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save config as YAML to path. Raises if PyYAML not installed."""
    if yaml is None:
        raise ImportError("PyYAML required for config. Install with: pip install pyyaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
