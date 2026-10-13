#!/usr/bin/env python3
"""Run sibling packages with an existing, explicitly selected environment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from codexa_workspace.assets import workspace_root, asset_root, output_root, normalize_paths



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Canonical checkout directory.")
    parser.add_argument("--profile", choices=("generative", "specialist"), default="generative")
    parser.add_argument("--python", type=Path, help="Override the existing profile interpreter.")
    parser.add_argument("--repo", default="LLM-From-Scratch", help="Repository working directory.")
    parser.add_argument("command", help="test, module, or an existing scripts/<name>.py command.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    ROOT = workspace_root(args.workspace)
    WORKSPACE = ROOT
    manifest = json.loads((ROOT / "compatibility.json").read_text())
    env_name = ".venv-specialist" if args.profile == "specialist" else ".venv"
    candidate = ROOT / env_name / "bin/python"
    python = args.python or (candidate if candidate.is_file() else Path(sys.executable))
    allowed = {name: ROOT / details["path"] for name, details in manifest["components"].items()}
    allowed.update({"LLM-From-Scratch": ROOT, "integration": ROOT})
    if args.repo not in allowed:
        parser.error("--repo must identify a repository in this workspace.")
    cwd = allowed[args.repo]
    sources = [str(p / "src") for p in allowed.values() if p != ROOT and (p / "src").is_dir()]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([*dict.fromkeys(sources), str(ROOT / "src"), str(ROOT)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CODEXA_WORKSPACE_ROOT"] = str(ROOT)
    try:
        env["CODEXA_ASSET_ROOT"] = str(asset_root(ROOT))
        env["CODEXA_OUTPUT_ROOT"] = str(output_root(ROOT))
    except (ValueError, FileNotFoundError):
        pass  # Synthetic tests and help work without private assets.
    env["LLM_INTEGRATION_ROOT"] = str(ROOT)
    env["LLM_SPECIALIST_ROOT"] = str(ROOT)
    specialist = ROOT / ".venv-specialist/bin/python"
    if specialist.is_file():
        env.setdefault("LLM_MEMORY_WORKER_PYTHON", str(specialist))
    if args.command == "test":
        command = [str(python), "-m", "pytest", "-p", "no:cacheprovider", *args.arguments]
    elif args.command == "module":
        if not args.arguments:
            parser.error("module requires a module name.")
        command = [str(python), "-m", *args.arguments]
    else:
        name = args.command.removesuffix(".py") + ".py"
        script = ROOT / "scripts" / name
        if not script.is_file():
            # Helpers are grouped in scripts/<area>/; operator commands stay at the top level.
            grouped = sorted((ROOT / "scripts").glob(f"*/{name}"))
            if len(grouped) != 1:
                parser.error(f"Unknown command: {args.command}")
            script = grouped[0]
        arguments = normalize_paths(ROOT, args.arguments, training=args.command in {"train", "train_conversational_sft"})
        command = [str(python), str(script), *arguments]
    return subprocess.call(command, cwd=cwd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
