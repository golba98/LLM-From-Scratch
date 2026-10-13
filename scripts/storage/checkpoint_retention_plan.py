"""Print a read-only dependency and retention plan; no deletion capability."""

import argparse
import json
from pathlib import Path

from llm_training.recovery import RecoveryStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-directory', type=Path, required=True)
    parser.add_argument('--store-root', type=Path)
    parser.add_argument('--reclamation', action='store_true',
                        help='Print the V3 immutable candidate plan; never execute it.')
    args = parser.parse_args()
    store = RecoveryStore(args.checkpoint_directory, backend='optimized',
                          store_root=args.store_root)
    if args.reclamation:
        from llm_training.checkpoint_reclamation import plan_reclamation
        plan = plan_reclamation(store)
    else:
        plan = store.plan_retention()
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
