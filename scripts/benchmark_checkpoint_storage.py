"""Benchmark bounded checkpoint fixtures without training or historical writes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['legacy', 'optimized', 'copy-manager'])
    args = parser.parse_args()
    from codexa_workspace.checkpoint_benchmark import benchmark, benchmark_backend
    result = benchmark_backend(args.output, args.worker) if args.worker else benchmark(ROOT, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
