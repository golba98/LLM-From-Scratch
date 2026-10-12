"""Show Codexa training status instantly from local files (no AI service is contacted)."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codexa_workspace.foundation_training import run_directory
from codexa_workspace.training_status import collect, render


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=run_directory(ROOT))
    parser.add_argument("--json", action="store_true", help="compact JSON")
    parser.add_argument("--watch", type=float, nargs="?", const=10.0, metavar="SECONDS",
                        help="refresh locally every SECONDS (default 10)")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    while True:
        status = collect(run_dir)
        text = json.dumps(status, separators=(",", ":")) if args.json else render(status)
        print(("\033[2J\033[H" if args.watch else "") + text, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
