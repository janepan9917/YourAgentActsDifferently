"""Command line entry point: ``python -m analysis <command>``.

    build-processed   build data/processed from data/raw and data/labels
    all               write the paper's figures and tables to outputs/
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m analysis")
    parser.add_argument("command", choices=["build-processed", "all"])
    parser.add_argument("--only", nargs="*", help="(all) run only these steps, e.g. table3 figure6")
    args = parser.parse_args(argv)
    start = time.time()
    if args.command == "build-processed":
        from .processed import build_processed
        build_processed()
    else:
        from .artifacts import run_all
        run_all(args.only)
    print(f"done in {time.time() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
