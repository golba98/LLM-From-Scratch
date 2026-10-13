"""Read-only checkpoint evidence and V3 capacity projections; no training or cleanup."""

import argparse
import json
from pathlib import Path
import shutil

from llm_architecture.checkpoint_inspection import inspect_checkpoint_storage
from llm_training.preflight import project_continuous_storage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--filesystem-path', type=Path, default=Path('.'))
    parser.add_argument('--retained-states', type=int, default=6)
    parser.add_argument('--active-runs', type=int, default=1)
    args = parser.parse_args()
    evidence = inspect_checkpoint_storage(args.checkpoint)
    plan = project_continuous_storage(checkpoint_bytes=evidence['file_bytes'],
        retained_states=args.retained_states, active_runs=args.active_runs)
    free = shutil.disk_usage(args.filesystem_path).free
    print(json.dumps(dict(evidence=evidence, projection=plan, actual_free_bytes=free,
        storage_preflight_passed=free >= plan['required_free_bytes'],
        deficit_bytes=max(0,plan['required_free_bytes']-free)), indent=2))


if __name__ == '__main__':
    main()
