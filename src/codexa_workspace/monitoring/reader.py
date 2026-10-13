"""Run data for the dashboard: byte-offset JSONL tails, the run reader, host probes and derived values.

Reads only small atomic documents plus tails of two JSONL files; never writes into the run,
never signals a process and never notifies.
"""

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from codexa_workspace.training.supervisor.state import read_json, TERMINAL


GIB = 1024**3


class JsonlTail:
    """Parse only lines appended since the last read (byte offset), tolerating partial writes."""

    def __init__(self, path: Path, keep: int) -> None:
        self.path, self.offset, self.rows = Path(path), 0, deque(maxlen=keep)

    def read(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset, self.rows = 0, deque(maxlen=self.rows.maxlen)
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                chunk = stream.read()
        except FileNotFoundError:
            return list(self.rows)
        complete = chunk[:chunk.rfind(b'\n') + 1]
        self.offset += len(complete)
        for line in complete.splitlines():
            try:
                self.rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return list(self.rows)


class RunReader:
    def __init__(self, run_dir: Path, *, gpu: bool = True) -> None:
        self.run_dir, self.gpu_enabled = Path(run_dir), gpu
        self.metrics = JsonlTail(self.run_dir / 'metrics/train_metrics.jsonl', 240)
        self.validations = JsonlTail(self.run_dir / 'metrics/validation-strata.jsonl', 400)
        self.events = JsonlTail(self.run_dir / 'supervisor/events.jsonl', 8)
        self.job = None
        self.gpu_cache, self.gpu_at = None, 0.0

    def gpu(self) -> dict | None:
        if not self.gpu_enabled:
            return None
        if time.monotonic() - self.gpu_at > 10:
            self.gpu_at = time.monotonic()
            try:
                line = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                      timeout=5).stdout.strip().splitlines()[0]
                name, used, total, utilization = [value.strip() for value in line.split(',')]
                self.gpu_cache = dict(name=name, used_mib=int(used), total_mib=int(total), utilization=int(utilization))
            except Exception:
                self.gpu_cache = None
        return self.gpu_cache

    def read(self) -> dict:
        r = self.run_dir
        if self.job is None:
            self.job = read_json(r / 'config/job.json') or {}
        legacy = read_json(r / 'supervisor/notification-state.json')
        return dict(run_dir=str(r), run_id=r.name, now=time.time(), job=self.job,
            progress=read_json(r / 'metrics/progress.json') or {}, heartbeat=read_json(r / 'metrics/heartbeat.json') or {},
            budget=read_json(r / 'metrics/budget.json') or {}, report=read_json(r / 'metrics/report.json') if self._terminal() else None,
            state=read_json(r / 'supervisor/state.json') or {}, completion=read_json(r / 'supervisor/completion.json'),
            desktop=read_json(r / 'supervisor/desktop-notification.json'), ai=read_json(r / 'supervisor/ai-wakeup.json'),
            window=read_json(r / 'supervisor/agent-window.json'),
            legacy_notifications=legacy, gates=read_json(r / 'config/launch-gates.json') or {},
            recovery=read_json(r / 'checkpoints/recovery-index.json') or {},
            metrics=self.metrics.read(), validations=self.validations.read(), events=self.events.read(),
            gpu=self.gpu(), host_available=host_available(), disk_free=disk_free(r))

    def _terminal(self) -> bool:
        return (read_json(self.run_dir / 'supervisor/state.json') or {}).get('state') in TERMINAL


def host_available() -> int | None:
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def disk_free(path: Path) -> int | None:
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def epoch(value: str | None) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp() if value else None
    except ValueError:
        return None


def derive(s: dict) -> dict:
    """Pure calculations from a snapshot; every field tolerates missing inputs."""
    p, hb, budget, spec = s['progress'], s['heartbeat'], s['budget'], (s['job'].get('spec') or {})
    training, evaluation = spec.get('training') or {}, spec.get('evaluation') or {}
    state = s['state'].get('state') or ('RUNNING' if p.get('state') == 'RUNNING' else 'NOT_PREPARED')
    terminal = state in TERMINAL
    step = (s['completion'] or {}).get('optimizer_steps') if terminal else None
    step = step if step is not None else hb.get('optimizer_step', p.get('step', 0)) or 0
    total = p.get('total_steps') or training.get('max_steps') or 0
    per_step = None
    stamps = [(m['optimizer_step'], epoch(m.get('timestamp_utc'))) for m in s['metrics'][-65:]]
    stamps = [(a, b) for a, b in stamps if b is not None]
    # Wall time per update including validation/checkpoint/generation overhead, but excluding
    # downtime gaps (> 10 min between consecutive updates, e.g. a stop followed by a resume).
    gaps = [(t2 - t1) / (s2 - s1) for (s1, t1), (s2, t2) in zip(stamps, stamps[1:]) if s2 > s1 and t2 - t1 <= 600]
    if len(gaps) >= 2:
        per_step = statistics.fmean(gaps)
    elif s['metrics']:
        per_step = s['metrics'][-1].get('step_time_seconds')
    started, deadline = budget.get('started_utc'), budget.get('deadline_utc')
    seconds = budget.get('seconds')
    if terminal:
        elapsed = (s['completion'] or {}).get('elapsed_seconds') or budget.get('elapsed_seconds')
        remaining = None
    else:
        elapsed = s['now'] - started if started else None
        remaining = max(0., deadline - s['now']) if deadline else None
    eta = per_step * (total - step) if per_step and total and not terminal else None
    reserve = budget.get('reserve_seconds') or 0
    projected_stop = None
    if eta is not None and remaining is not None and eta > remaining - reserve and per_step:
        projected_stop = min(total, step + int(max(0., remaining - reserve) / per_step))
    tokens_per_update = (training.get('gradient_accumulation_steps', 0) * training.get('micro_batch_size', 0)
                         * (spec.get('model') or {}).get('context_length', 0))
    losses = [m['training_loss'] for m in s['metrics'] if isinstance(m.get('training_loss'), (int, float))]
    speeds = [m['tokens_per_second'] for m in s['metrics'] if isinstance(m.get('tokens_per_second'), (int, float))]
    validations = s['validations']
    baseline = validations[0] if validations and validations[0].get('optimizer_step') == 0 else None
    latest = validations[-1] if validations else None
    best = min((v['weighted_loss'] for v in validations), default=None)
    cadence = evaluation.get('cadence_updates') or training.get('evaluation_interval')
    phase = hb.get('phase') or p.get('activity')
    display = state
    if state in ('RUNNING', 'EARLY_EVALUATION', 'CHECKPOINTING'):
        display = {'VALIDATING': 'VALIDATING', 'BASELINE': 'VALIDATING', 'GENERATION_EVAL': 'GENERATING',
                   'CHECKPOINTING': 'CHECKPOINTING', 'FINALIZING': 'CHECKPOINTING'}.get(phase, 'RUNNING')
    heartbeat_age = s['now'] - hb['ts_utc'] if hb.get('ts_utc') else None
    if not terminal and state in ('RUNNING', 'EARLY_EVALUATION', 'CHECKPOINTING') and heartbeat_age and heartbeat_age > 60:
        display = 'STALE'
    latest_ckpt = s['recovery'].get('latest') or {}
    ckpt_path = Path(s['run_dir']) / 'checkpoints' / latest_ckpt['path'] if latest_ckpt.get('path') else None
    try:
        ckpt_age = s['now'] - ckpt_path.stat().st_mtime if ckpt_path else None
    except OSError:
        ckpt_age = None
    warmup = training.get('warmup_steps')
    schedule = None
    if warmup is not None and total:
        schedule = 'warmup' if step <= warmup else f'cosine {100 * (step - warmup) / max(1, total - warmup):.0f}%'
    return dict(state=state, display=display, terminal=terminal, phase=phase, step=step, total=total,
                fraction=min(1., step / total) if total else 0., per_step=per_step, elapsed=elapsed,
                remaining=remaining, eta=eta, finish=s['now'] + eta if eta else None, projected_stop=projected_stop,
                seconds=seconds, tokens=p.get('total_tokens_seen') or (s['completion'] or {}).get('processed_tokens'),
                token_target=tokens_per_update * total if tokens_per_update and total else None,
                losses=losses, speeds=speeds, baseline=baseline, latest=latest, best=best, cadence=cadence,
                next_eval=(step // cadence + 1) * cadence if cadence and not terminal else None,
                heartbeat_age=heartbeat_age, checkpoint=latest_ckpt, checkpoint_age=ckpt_age, schedule=schedule,
                guard=(spec.get('optimization') or {}).get('minimum_headroom_bytes', int(1.5 * GIB)),
                guards=spec.get('guards') or {})
