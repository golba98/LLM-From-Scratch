#!/usr/bin/env python3
"""Verify or locally restore the pinned workspace infrastructure checkout."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(path: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(path), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Protected local clone or bundle, used only when checkout is absent")
    args = parser.parse_args()
    record = json.loads((ROOT / "compatibility.json").read_text())["tooling"]["workspace-infrastructure"]
    target = (ROOT / record["path"]).resolve()
    expected = record["commit"]
    if not target.exists():
        if args.source is None:
            parser.error(f"{target} is absent; supply --source LOCAL_GIT_CLONE_OR_BUNDLE")
        source = args.source.resolve()
        if not source.exists():
            parser.error(f"local source does not exist: {source}")
        subprocess.run(["git", "clone", "--no-checkout", str(source), str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", expected], check=True)
    actual = git(target, "rev-parse", "HEAD")
    pinned = git(target, "rev-parse", expected)
    if actual != pinned:
        raise SystemExit(f"Infrastructure revision mismatch: expected {pinned}, found {actual}")
    if git(target, "status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("Infrastructure checkout has uncommitted or untracked source")
    print(f"workspace-infrastructure verified at {actual}")


if __name__ == "__main__":
    main()
