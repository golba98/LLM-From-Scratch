"""Create a deterministic, corpus-wide sample from a conversation JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the sampling command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(arguments: argparse.Namespace) -> None:
    """Select records by stable hash so the sample spans the whole corpus."""

    if arguments.limit <= 0:
        raise ValueError("--limit must be positive.")
    selected: list[tuple[int, str]] = []
    with arguments.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source):
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError(f"Malformed conversation at line {line_number + 1}.")
            identity = str(row.get("conversation_id", line_number))
            digest = hashlib.sha256(
                f"{arguments.seed}:{identity}".encode("utf-8")
            ).digest()
            score = int.from_bytes(digest[:8], "big")
            item = (-score, line)
            if len(selected) < arguments.limit:
                heapq.heappush(selected, item)
            elif item > selected[0]:
                heapq.heapreplace(selected, item)
    records = [line for _score, line in sorted(selected, reverse=True)]
    if len(records) < arguments.limit:
        raise ValueError(
            f"Input contains only {len(records)} records; {arguments.limit} required."
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("".join(records), encoding="utf-8", newline="\n")
    print(json.dumps({"input": str(arguments.input), "output": str(arguments.output), "records": len(records), "seed": arguments.seed}))


def main() -> None:
    """Run the installed console command and module entry point."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
