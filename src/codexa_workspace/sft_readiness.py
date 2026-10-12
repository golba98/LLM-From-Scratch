"""Deterministic GO / NO-GO readiness for the frozen conversational SFT pilot. Never launches.

Verdicts:
  READY FOR AUTHORIZATION  every engineering and launch-time check passes;
  CONDITIONALLY READY      only launch-time conditions fail (e.g. free VRAM at this moment);
  NOT READY                any material integration, reproducibility, data, evaluation or safety defect.

Authorization is reported separately: a missing user grant is the deliberate execution gate,
never an engineering failure, and a READY verdict is never treated as authorization.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

from codexa_workspace import sft_phase4_data as data
from codexa_workspace import sft_pilot_launch as launch
from codexa_workspace import sft_pilot_runtime as runtime
from codexa_workspace import sft_step5a_verify as verify

READY, CONDITIONAL, NOT_READY = 'READY FOR AUTHORIZATION', 'CONDITIONALLY READY', 'NOT READY'
LAUNCH_TIME = 'launch-time'
ENGINEERING = 'engineering'


def classify(checks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Engineering failures dominate; launch-time conditions alone give CONDITIONALLY READY."""
    if not checks:
        return NOT_READY, [dict(name='no checks ran')], []
    engineering = [c for c in checks if c['kind'] != LAUNCH_TIME and not c['passed']]
    launch_time = [c for c in checks if c['kind'] == LAUNCH_TIME and not c['passed']]
    return (NOT_READY if engineering else CONDITIONAL if launch_time else READY), engineering, launch_time


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def git_reproducibility(job: dict) -> dict:
    """Pinned component HEADs, clean executed trees, and the job's code identity equal to committed bytes."""
    compatibility = data.read(runtime.ROOT / 'compatibility.json')
    pins = {name: dict(pinned=record['commit'], head=_git((runtime.ROOT / record['path']).resolve(), 'rev-parse', 'HEAD'))
            for name, record in compatibility['components'].items()}
    dirty = {name: _git((runtime.ROOT / record['path']).resolve(), 'status', '--porcelain', '--untracked-files=all', '--', 'src')
             for name, record in compatibility['components'].items()}
    dirty['LLM-From-Scratch'] = _git(runtime.ROOT, 'status', '--porcelain', '--untracked-files=all', '--',
                                     'src', 'scripts', 'tools')
    untracked_code = [name for name in runtime.CODE_FILES
                      if not _git(runtime.ROOT, 'ls-files', '--', name)
                      or _git(runtime.ROOT, 'status', '--porcelain', '--', name)]
    current = runtime.code_identity()
    return dict(pins=pins, pins_match=all(p['pinned'] == p['head'] for p in pins.values()),
                dirty={k: v.splitlines() for k, v in dirty.items() if v}, clean=not any(dirty.values()),
                integration_head=_git(runtime.ROOT, 'rev-parse', 'HEAD'),
                uncommitted_code_files=untracked_code,
                code_identity_matches_job=current == job['code_identity'],
                code_identity_differences=sorted(k for k in set(current) | set(job['code_identity'])
                                                 if current.get(k) != job['code_identity'].get(k)))


def evaluate(job_path: Path, *, environment: runtime.EnvironmentProvider | None = None,
             resource_snapshot: Callable[[], dict] | None = None,
             dataset: Callable[[Path], dict] | None = None, support: Callable[[], dict] | None = None) -> dict:
    """Run every check; each failure is recorded with its category, none is skipped."""
    job_path = Path(job_path)
    directory = job_path.parent.parent
    job = data.read(job_path)
    spec = job['spec']
    checks = []

    def check(category, name, function, kind=ENGINEERING):
        try:
            result = function()
            passed = bool(result if isinstance(result, bool) else (result or {}).get('passed', True))
            detail = result
        except Exception as error:  # every failure is reported
            passed, detail = False, f'{type(error).__name__}: {error}'
        checks.append(dict(category=category, name=name, kind=kind, passed=passed, detail=detail))
        return detail

    check('configuration', 'job and spec hashes', lambda: bool(runtime.verify_job(job)))
    check('configuration', 'non-runnable frozen draft',
          lambda: spec['runnable'] is False and spec['authorization_required'] is True and job['launch_command'] is None)
    check('configuration', 'approved pilot configuration', lambda: launch.validate_training_config(spec))
    check('configuration', 'runtime supports every setting (signatures and capability only)',
          lambda: (support or launch.runtime_support)())
    check('configuration', 'budget is the recommended bounded cap',
          lambda: spec['budget']['total_seconds'] == data.read(directory / 'config/budget-analysis.json')
          ['recommendation']['total_seconds'] and spec['budget']['finalization_reserve_seconds'] == 180
          and spec['budget']['automatic_extension'] is False)
    guard = check('source identity', 'Codex guard: development, operation new',
                  lambda: (lambda r: dict(r, passed=r['context'] == 'current'))(
                      (environment or runtime.TrainingGuardProvider('new')).resolve()))
    check('source identity', 'frozen component revisions are current HEADs',
          lambda: isinstance(guard, dict) and {k: v.get('detected') for k, v in (guard.get('component_revisions') or {}).items()}
          == spec['environment']['component_revisions'] and bool(spec['environment']['component_revisions']))
    check('git reproducibility', 'pins, clean executed trees, committed code identity',
                lambda: (lambda g: dict(g, passed=g['pins_match'] and g['clean'] and not g['uncommitted_code_files']
                                        and g['code_identity_matches_job']))(git_reproducibility(job)))
    check('environment', 'repaired source-first architecture and tokenizer',
          lambda: isinstance(guard, dict) and (guard.get('architecture') or {}).get('repaired') is True
          and (guard.get('architecture') or {}).get('sha256') == spec['environment']['architecture_sha256']
          and guard.get('tokenizer_sha256') == spec['tokenizer']['sha256'])
    check('checkpoint preservation', 'initialization checkpoint bytes',
          lambda: data.digest(spec['initialization']['path']) == runtime.INITIALIZATION_SHA256 == spec['initialization']['sha256'])
    check('tokenizer', 'tokenizer bytes match the frozen manifest and job',
          lambda: data.digest(spec['tokenizer']['path']) == spec['tokenizer']['sha256']
          == data.read(spec['data']['manifest'])['tokenizer']['sha256'])
    check('dataset', 'Phase 4 manifest reproduces (hash, data, orders, tokenizer)',
          lambda: data.verify_manifest(Path(spec['data']['manifest']))['verified'])
    check('dataset', 'independent re-derivation (source content, masks, END, shift, counts, isolation)',
          lambda: (lambda r: dict(passed=r['verified'], checks=r['checks'], train=r['train']))(
              (dataset or verify.verify_dataset)(Path(spec['data']['manifest']))))
    check('dataset', 'validation-source exception satisfies isolation',
          lambda: (lambda r: dict(r, passed=r['isolation_policy_satisfied']))(
              data.read(directory / 'evidence/validation-exception.json')))
    check('evaluation', 'sealed step zero bound to checkpoint, tokenizer, prompts, decoding, environment',
          lambda: (lambda r: dict(passed=r['bound'], checks=r['checks']))(
              verify.step_zero_binding(Path(spec['evaluation']['step_zero_path']), Path(spec['data']['manifest']), spec)))
    check('evaluation', 'step zero reconfirmed token-identical under the committed identity',
          lambda: (lambda r: dict(passed=r['reconfirmed'], identical=r['greedy_identical_tokens'], gold=r['gold_match']))(
              data.read(directory / 'evidence/step-zero-reconfirmation.json')))
    check('evaluation', 'acceptance gates and human review pending (never fabricated)',
          lambda: spec['acceptance'] and spec['evaluation']['updates'] == [0, 32, 64, 128])
    check('safety', 'canonical resource gates (never lowered)', lambda: bool(runtime.validate_gates(spec['guards']['gates'])))
    check('safety', 'stop policy (nonfinite, OOM, headroom, deterioration, retention, cap, deadline)',
          lambda: spec['guards']['update_cap'] == 128 and runtime.StopPolicy().minimum_headroom_bytes
          == spec['optimization']['minimum_headroom_bytes'] == runtime.canonical_gates()['in_run_free_vram_bytes'])
    check('safety', 'fresh isolated output namespace', lambda: runtime._fresh_namespace(spec['outputs']))
    check('supervisor', 'OS supervisor, linger and read-only dashboard',
                     lambda: (lambda s: dict(s, passed=s['available']))(launch.supervisor_available()))
    resources = (resource_snapshot or verify.resources)()
    for item in resources['checks']:
        check('resources', f"launch-time {item['name']}", lambda item=item: dict(item, passed=item['passed']), LAUNCH_TIME)
    verdict, engineering, launch_time = classify(checks)
    approval = directory / 'config/approval.json'
    try:
        granted = bool(runtime.approval_valid(directory, job))
    except runtime.LaunchRefused:
        granted = False
    return dict(format='codexa-sft-pilot-readiness-v1', checked_utc=verify.now(), experiment=str(job_path),
                job_sha256=job['job_sha256'], verdict=verdict,
                blocking=[f"{c['category']}: {c['name']}" for c in engineering + launch_time],
                engineering_failures=len(engineering), launch_time_conditions=len(launch_time),
                authorization=dict(granted=granted, path=str(approval),
                                   note='Deliberate execution gate, separate from technical readiness. '
                                        'Readiness is never authorization.'),
                gpu_processes=resources.get('gpu_processes'), checks=checks)
