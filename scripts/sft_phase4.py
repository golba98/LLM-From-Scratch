#!/usr/bin/env python3
"""Phase 4 conversational SFT preparation. No training, no optimizer, no launch.

Run through tools/codexa-dev (source-first components). There is deliberately no
launch subcommand: training requires a separate, explicit future authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from codexa_workspace import sft_phase4_data as data  # noqa: E402


def directory_argument(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(data.BASE.resolve()) or path == data.BASE.resolve() or not path.is_dir():
        raise argparse.ArgumentTypeError('Use an existing Phase 4 preparation directory.')
    return path


def batch_plan(values: list[str]) -> list[tuple[str, str, int, int]]:
    plan = []
    for value in values:
        split, source, count, size = value.split(':')
        plan.append((split, source, int(count), int(size)))
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    batches = commands.add_parser('batches')
    batches.add_argument('--directory', type=directory_argument, required=True)
    batches.add_argument('--prefilter', action='store_true', help='skip obvious non-English (logged, never admitted)')
    batches.add_argument('plan', nargs='+', help='split:source:records:batch_size')
    for name in ('ledger', 'pending', 'select', 'freeze', 'verify', 'step-zero', 'prepare',
                 'preflight', 'job', 'budget', 'preservation', 'report'):
        sub = commands.add_parser(name)
        sub.add_argument('--directory', type=directory_argument, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        print(data.initialize())
        return 0
    directory = args.directory
    if args.command == 'batches':
        print(json.dumps(data.make_batches(directory, batch_plan(args.plan), prefilter=args.prefilter)))
    elif args.command == 'ledger':
        print(json.dumps(data.review_statistics(data.ledger(directory)), indent=2))
    elif args.command == 'pending':
        print(json.dumps(data.pending_second_pass(directory)))
    elif args.command == 'select':
        result = data.select(directory)
        print(json.dumps({k: result[k] for k in ('complete', 'shortfall', 'counts', 'categories', 'admitted_both_passes')}, indent=2))
    elif args.command == 'freeze':
        manifest = data.freeze(directory)
        print(json.dumps(dict(manifest_sha256=manifest['manifest_sha256'], accounting=manifest['token_accounting']), indent=2))
    elif args.command == 'verify':
        print(json.dumps(data.verify_manifest(directory / 'manifest/phase4-manifest.json'), indent=2))
    else:
        from codexa_workspace import sft_phase4_operations as operations
        result = getattr(operations, args.command.replace('-', '_'))(directory)
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
