#!/usr/bin/env python3
"""Step 5A: integration verification for the conversational SFT pilot. No training, no launch.

Run through tools/codexa-dev (source-first components). Every subcommand is read-only with
respect to checkpoints, datasets and Phase 4 artifacts; outputs go to the Step 5A directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from codexa_workspace import sft_phase4_data as data  # noqa: E402
from codexa_workspace import sft_step5a_verify as verify  # noqa: E402

EXTENSION_RECORD = '0c903975-f9da-4938-b897-07a6f1766390'


def directory_argument(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_relative_to(verify.STEP5A_BASE.resolve()) or path == verify.STEP5A_BASE.resolve() or not path.is_dir():
        raise argparse.ArgumentTypeError('Use an existing Step 5A output directory.')
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('snapshot', 'preservation', 'dataset', 'lineage', 'binding', 'reconfirm', 'budget',
                 'resources', 'freeze', 'readiness', 'report'):
        sub = commands.add_parser(name)
        sub.add_argument('--directory', type=directory_argument, required=True)
        if name in ('snapshot', 'preservation'):
            sub.add_argument('--label', required=True)
    args = parser.parse_args()
    directory = args.directory
    manifest = verify.PHASE4 / 'manifest/phase4-manifest.json'
    summary = verify.PHASE4 / 'evaluation/step-zero/summary.json'
    evidence = directory / 'evidence'
    if args.command == 'snapshot':
        result = verify.repository_snapshot()
        data.write(evidence / f'coordination-{args.label}.json', result)
        result = {name: dict(head=r['head'], dirty=r['dirty']) for name, r in result['repositories'].items()}
    elif args.command == 'preservation':
        expected = None
        if args.label != 'before':
            before = data.read(evidence / 'preservation-before.json')
            expected = {c['path']: c['expected'] for c in before['checks'] if str(verify.PHASE4) in c['path']}
        result = verify.preservation(directory, args.label, phase4_expected=expected)
    elif args.command == 'dataset':
        result = verify.verify_dataset(manifest)
        data.write(evidence / 'dataset-verification.json', result)
        exception = verify.validation_exception(manifest, EXTENSION_RECORD)
        data.write(evidence / 'validation-exception.json', exception)
        result = dict(verified=result['verified'], checks=result['checks'], train=result['train'],
                      validation=result['validation'], isolation=result['isolation'],
                      exception_isolation=exception['isolation_policy_satisfied'])
    elif args.command == 'lineage':
        result = verify.lineage_exposure(manifest)
        data.write(directory / 'manifest/lineage-exposure.json', result)
    elif args.command == 'binding':
        job = directory / 'config/job.json'
        result = verify.step_zero_binding(summary, manifest, data.read(job)['spec'] if job.exists() else None)
        data.write(evidence / 'step-zero-binding.json', result)
    elif args.command == 'reconfirm':
        target = evidence / 'step-zero-reconfirmation.json'
        if target.exists():
            raise SystemExit('Reconfirmation already recorded; never overwrite evidence.')
        result = verify.reconfirm_step_zero(manifest, summary)
        result['code_identity'] = verify.step_zero_binding(summary, manifest)['source_code']
        data.write(target, result)
    elif args.command == 'budget':
        result = verify.budget_analysis(manifest, summary)
        data.write(directory / 'config/budget-analysis.json', result)
    elif args.command == 'resources':
        result = verify.resources()
        data.write(evidence / 'resources.json', result)
        result = {k: result[k] for k in ('startup_headroom_met', 'host_checks_met', 'gpu_processes', 'supervisor')}
    elif args.command == 'freeze':
        result = verify.freeze_job(directory, manifest, summary)
    elif args.command == 'readiness':
        from codexa_workspace import sft_readiness
        result = sft_readiness.evaluate(directory / 'config/job.json')
        data.write(evidence / 'readiness.json', result)
        result = {k: result[k] for k in ('verdict', 'blocking', 'engineering_failures', 'launch_time_conditions',
                                         'authorization')}
    else:
        from codexa_workspace.sft_step5a_report import write_report
        result = write_report(directory)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
