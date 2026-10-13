"""Deliver a run's terminal notification/AI wakeup exactly once, (re)open the interactive agent
window, or test a provider CLI."""

import argparse
import json
from pathlib import Path

from codexa_workspace.notifications.agent import test_provider
from codexa_workspace.notifications.notifier import deliver, open_agent_window
from codexa_workspace.paths import REPO_ROOT
from codexa_workspace.training.runs import resolve_run_dir


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("deliver", "open-agent", "test-provider"))
    parser.add_argument("--run-dir", type=Path, help="run directory (default: the most recently active run)")
    parser.add_argument("--provider", choices=("codex", "claude"))
    parser.add_argument("--dry-run", action="store_true", help="resolve commands; spend nothing")
    parser.add_argument("--output", type=Path, help="test-provider evidence directory")
    args = parser.parse_args(arguments)
    if args.command == "test-provider":
        if not args.provider or not args.output:
            parser.error("test-provider requires --provider and --output")
        result = test_provider(args.provider, args.output.resolve(), REPO_ROOT)
        print(json.dumps({key: result[key] for key in ("provider", "passed", "exit_code", "seconds", "session_id", "output")},
                         indent=2))
        return 0 if result["passed"] else 1
    run_dir = resolve_run_dir(args.run_dir)
    if args.command == "deliver":
        state = deliver(run_dir, REPO_ROOT, provider=args.provider, dry_run=args.dry_run)
        print(json.dumps(dict(execution_class=state["execution_class"], dry_run=state.get("dry_run", False),
                              **{name: value["status"] for name, value in state["channels"].items()}), indent=2))
        return 0
    result = open_agent_window(run_dir, REPO_ROOT, provider=args.provider)
    print(json.dumps({key: result.get(key) for key in ("success", "backend", "unit", "command")}, indent=2))
    return 0 if result.get("success") else 1
