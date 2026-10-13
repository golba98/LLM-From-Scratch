#!/usr/bin/env python3
"""GO / NO-GO readiness for the frozen conversational SFT pilot. Read-only; never launches.

  tools/codexa-dev scripts/sft_readiness.py --experiment <step5a-dir>/config/job.json
  .venv/bin/codexa sft_readiness --experiment <step5a-dir>/config/job.json

Exit status: 0 READY FOR AUTHORIZATION, 3 CONDITIONALLY READY, 2 NOT READY.
Readiness is never authorization: the user grants that separately (scripts/sft/sft_pilot.py approve).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', type=Path, required=True, help='frozen job.json of the experiment')
    parser.add_argument('--write', type=Path, help='also write the full readiness record to this path')
    args = parser.parse_args()
    from codexa_workspace import sft_readiness
    result = sft_readiness.evaluate(args.experiment.resolve())
    if args.write:
        args.write.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'job_sha256', 'blocking', 'engineering_failures',
                                             'launch_time_conditions', 'authorization')}, indent=2))
    return {sft_readiness.READY: 0, sft_readiness.CONDITIONAL: 3}.get(result['verdict'], 2)


if __name__ == '__main__':
    raise SystemExit(main())
