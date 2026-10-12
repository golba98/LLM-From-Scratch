"""Bounded CPU V3 qualification; writes only new disposable fixture namespaces."""

import argparse
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from codexa_workspace.checkpoint_benchmark_v3 import benchmark, memory_worker, restore_existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--memory-worker', choices=('legacy', 'v2', 'v3'))
    parser.add_argument('--checkpoint-directory', type=Path)
    parser.add_argument('--restore-existing', type=Path,
                        help='Read existing fixtures; write a new corrected measurement report only.')
    parser.add_argument('--report-name', default='restore-corrected.json')
    args = parser.parse_args()
    if args.memory_worker:
        if args.checkpoint_directory is None:
            parser.error('--memory-worker requires --checkpoint-directory')
        result = memory_worker(args.memory_worker, args.checkpoint_directory, args.baseline)
    elif args.restore_existing:
        result = restore_existing(root, args.restore_existing, args.baseline,args.report_name)
        result = {k: v for k,v in result.items() if k != 'cases'}
    else:
        if args.output is None:
            parser.error('--output is required')
        result = benchmark(root, args.output, args.baseline)
        result = {k: v for k, v in result.items() if k not in ('cases', 'rotation')}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
