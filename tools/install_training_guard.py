#!/usr/bin/env python3
"""Install or verify the removable training guard in the existing .venv."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "verify"))
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("run with the existing LLM-From-Scratch/.venv/bin/python")
    site = next((p for p in (ROOT / ".venv/lib").glob("python*/site-packages") if p.is_dir()), None)
    if site is None:
        parser.error("site-packages not found")
    target = site / "codexa_training_guard.pth"
    wanted = f"{ROOT / 'tools'}\nimport training_environment_guard; training_environment_guard.install()\n"
    if args.command == "install":
        if target.exists() and target.read_text() != wanted:
            parser.error(f"refusing to overwrite an unexpected guard at {target}")
        if not target.exists():
            target.write_text(wanted)
    if not target.exists() or target.read_text() != wanted:
        parser.error(f"training guard missing or stale: {target}")
    print(f"training guard installed and verified: {target}")


if __name__ == "__main__":
    main()
