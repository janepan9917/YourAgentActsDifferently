"""Build ``data/processed`` from the raw runs and judge labels."""
from __future__ import annotations

import time
from pathlib import Path

from .config import processed_dir
from .runs import build_agent_runs

NAME = "agent_runs.csv.gz"


def write(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    runs = build_agent_runs()
    path = out / NAME
    # mtime 0 keeps the gzip file byte-identical across rebuilds.
    runs.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"{NAME}: {len(runs)} agent-runs ({time.time() - start:.0f}s)")
    return path


def build_processed() -> Path:
    return write(processed_dir())
