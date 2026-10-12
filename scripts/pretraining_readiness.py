"""Prepare CPU-only foundational proposals; no GPU or training action exists."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from codexa_workspace.pretraining_readiness import build


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.output if args.output.is_absolute() else root/args.output
    print(json.dumps(build(root, directory), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
