#!/usr/bin/env python3
"""Codexa 10-hour foundation continuation: preparation and the user's authorization-time commands.

Preparation (inference only, no optimizer):
  tools/codexa-dev scripts/foundation_continuation.py baseline  --directory <prep-dir>
  tools/codexa-dev scripts/foundation_continuation.py freeze    --directory <prep-dir>
  tools/codexa-dev scripts/foundation_continuation.py readiness --experiment <prep-dir>/config/job.json [--run-tests]

User only (never run by preparation or by an AI agent; readiness is not authorization):
  tools/codexa-dev scripts/foundation_continuation.py approve --experiment <job.json> --job-sha256 <hash> --statement '<words>'
  tools/codexa-dev scripts/foundation_continuation.py launch  --experiment <job.json> --provider claude
  tools/codexa-dev scripts/foundation_continuation.py resume  --experiment <job.json> --confirm-run <run-id> --provider claude
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def measured_inputs(baseline: dict) -> dict:
    from codexa_workspace import foundation_continuation as job
    timings = baseline['timings_seconds']
    report = job.read(job.PREDECESSOR / 'metrics/report.json')
    checkpoint = max(c['write_verify_seconds'] for c in report['checkpoints'])
    return dict(steps=job.step_time_statistics(), startup_seconds=99.0,
                startup_basis='predecessor: supervisor spawn to first update, including model load, baseline '
                              'validation and a 48-case generation pass this job skips (conservative)',
                validation_seconds=timings['validation'], periodic_seconds=timings['periodic_scoring'],
                full_seconds=timings['historical_48'] + timings['full_scoring_and_greedy'],
                checkpoint_seconds=checkpoint, checkpoint_basis='predecessor maximum write+verify')


def freeze(directory: Path) -> dict:
    from codexa_workspace import foundation_continuation as job
    from codexa_workspace import capability_benchmark as bench
    baseline_path = directory / 'baseline/summary.json'
    baseline = json.loads(baseline_path.read_text())
    measured = measured_inputs(baseline)
    final_needed = measured['validation_seconds'] + 2 * measured['checkpoint_seconds'] + measured['full_seconds']
    reserve = max(180, 60 * math.ceil(final_needed * 1.25 / 60))
    generation_reserve = int(math.ceil(measured['full_seconds'] * 1.25))
    max_steps = job.choose_max_steps(measured, reserve_seconds=reserve)
    plan = dict(job.timing_model(measured, max_steps=max_steps, reserve_seconds=reserve), measured=measured,
                reserve_basis='1.25 x (final validation + final save/verify x2 + final full evaluation), >= 180 s',
                cap_basis='largest multiple of 224 updates whose p95 plan leaves >= 10% of 36,000 s unused')
    order = job.build_order(directory / 'config', max_steps=max_steps)
    data = job.verify_payload_and_sources(Path(order['path']), directory / 'evidence/data-verification.json')
    if not data['passed']:
        raise SystemExit(f"Data verification failed: {data['checks']}")
    suites = bench.load_suites(('v1', 'v2'))
    environment = job.environment_record()
    spec = job.build_spec(directory, max_steps=max_steps, reserve_seconds=reserve,
                          generation_reserve_seconds=generation_reserve, order=order,
                          baseline=dict(path=str(baseline_path), summary_sha256=baseline['summary_sha256']),
                          suites=suites, environment=environment, plan=plan)
    evidence = dict(data_verification=str(directory / 'evidence/data-verification.json'),
                    data_verification_sha256=job.file_sha256(directory / 'evidence/data-verification.json'),
                    baseline_summary_sha256=baseline['summary_sha256'], suites=suites['identity'])
    frozen = job.freeze(directory, spec, evidence=evidence)
    return dict(job_sha256=frozen['job_sha256'], spec_sha256=frozen['spec_sha256'], max_steps=max_steps,
                tokens=max_steps * job.UPDATE_TOKENS, reserve_seconds=reserve, plan_total=plan['planned_total_seconds'],
                path=str(directory / 'config/job.json'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('baseline', 'freeze', 'readiness', 'approve', 'launch', 'resume'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--job-sha256')
    parser.add_argument('--statement')
    parser.add_argument('--provider', choices=('codex', 'claude'), default='claude')
    parser.add_argument('--confirm-run')
    parser.add_argument('--run-tests', action='store_true')
    parser.add_argument('--write', type=Path, help='readiness: also write the full record (new file)')
    args = parser.parse_args()
    if args.directory is not None:
        args.directory = args.directory.resolve()        # every frozen path must be absolute
    from codexa_workspace import foundation_continuation as job
    stamp = datetime.now(timezone.utc).isoformat()
    if args.command == 'baseline':
        document = job.step_zero_baseline(args.directory / 'baseline')
        print(json.dumps(dict(summary_sha256=document['summary_sha256'], timings=document['timings_seconds'],
                              validation=document['validation']['sources']), indent=2))
        return 0
    if args.command == 'freeze':
        print(json.dumps(freeze(args.directory), indent=2))
        return 0
    experiment = args.experiment.resolve()
    frozen = job.read(experiment)
    directory = experiment.parent.parent
    if args.command == 'readiness':
        result = job.evaluate_readiness(experiment, run_tests=args.run_tests)
        if args.write:
            args.write.write_text(json.dumps(result, indent=1, default=str) + '\n')
        print(json.dumps({k: result[k] for k in ('verdict', 'job_sha256', 'run_id', 'blocking', 'engineering_failures',
                                                 'launch_time_conditions', 'authorization')}, indent=2))
        return {job.READY: 0, job.CONDITIONAL: 3}.get(result['verdict'], 2)
    if args.command == 'approve':
        if args.job_sha256 != frozen['job_sha256']:
            raise SystemExit('The hash does not name this frozen job; nothing was approved.')
        print(job.write_approval(directory, frozen, statement=args.statement or '', written_utc=stamp))
        return 0
    if args.command == 'launch':
        result = job.launch_authorized(directory, frozen, provider=args.provider, written_utc=stamp,
                                       readiness=lambda: job.evaluate_readiness(experiment, run_tests=True))
        print(json.dumps({k: result[k] for k in ('launched', 'run_dir')}, indent=2))
        print(f"Monitor (read-only): {ROOT / '.venv/bin/python'} {ROOT / 'scripts/training_dashboard.py'} --run-dir {result['run_dir']}")
        return 0
    if args.command == 'resume':
        from codexa_workspace.sft_pilot_runtime import measure_resources
        result = job.resume_authorized(frozen, confirm=args.confirm_run or '', provider=args.provider,
                                       resources=lambda: measure_resources(ROOT), written_utc=stamp)
        print(json.dumps({k: result[k] for k in ('resumed', 'run_dir', 'from_step')}, indent=2))
        return 0
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
