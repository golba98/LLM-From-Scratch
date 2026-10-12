"""Deliver a run's terminal notification/AI wakeup exactly once, (re)open the interactive agent window,
or test a provider CLI."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codexa_workspace.foundation_training import run_directory
from codexa_workspace.training_notifications import deliver, open_agent_window
from codexa_workspace.training_wakeup import test_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("deliver", "open-agent", "test-provider"))
    parser.add_argument("--run-dir", type=Path, default=run_directory(ROOT))
    parser.add_argument("--provider", choices=("codex", "claude"))
    parser.add_argument("--dry-run", action="store_true", help="resolve commands; spend nothing")
    parser.add_argument("--output", type=Path, help="test-provider evidence directory")
    args = parser.parse_args()
    if args.command == "deliver":
        state = deliver(args.run_dir, ROOT, provider=args.provider, dry_run=args.dry_run)
        print(json.dumps(dict(execution_class=state["execution_class"], dry_run=state.get("dry_run", False),
                              **{name: value["status"] for name, value in state["channels"].items()}), indent=2))
    elif args.command == "open-agent":
        result = open_agent_window(args.run_dir, ROOT, provider=args.provider)
        print(json.dumps({key: result.get(key) for key in ("success", "backend", "unit", "command")}, indent=2))
        return 0 if result.get("success") else 1
    else:
        if not args.provider or not args.output:
            parser.error("test-provider requires --provider and --output")
        result = test_provider(args.provider, args.output.resolve(), ROOT)
        print(json.dumps({key: result[key] for key in ("provider", "passed", "exit_code", "seconds", "session_id", "output")}, indent=2))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
