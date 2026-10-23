"""Append-only, crash-recoverable per-update training telemetry (``metrics/telemetry.jsonl``).

One JSON object per line. Existing bytes are never rewritten:

- a torn final line from a crash is left in place; the writer starts its next record on a
  fresh line, and the reader counts the torn line as ``corrupt`` instead of failing;
- every process that opens the file starts a new ``attempt`` (a ``session`` record carrying
  the recovery state: resumed or fresh, the step it restarted from, remaining budget);
- records carry a deterministic ``record_id`` (run, attempt, kind, step) and the writer
  refuses to append an id it has already written, so a repeated callback cannot duplicate;
- after a resume, the new session names the step it restarted from. ``read_telemetry``
  drops earlier attempts' update records beyond that step (their effects were rolled back)
  and keeps one record per (kind, step), so a resumed run never reports duplicate updates;
- a measurement that was not taken is ``null``, never ``0``.

Writes are one ``write`` + ``flush`` per record; ``fsync`` runs every ``fsync_every``
updates and at evaluation, checkpoint and stop boundaries, so collection stays cheap.
The reported gradient norm from ``clip_grad_norm_`` is the norm *before* clipping; the
post-clip norm is derived exactly as torch applies it (``norm * min(1, clip / (norm + 1e-6))``).

Observability failures never stop training, but they are never silent either: ``safe``
records each failure, and ``status`` reports ``complete`` only when every write succeeded.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Callable

FORMAT = 'codexa-training-telemetry-v1'
FILENAME = 'telemetry.jsonl'
CLIP_EPSILON = 1e-6          # torch.nn.utils.clip_grad_norm_ uses max_norm / (total_norm + 1e-6)
ROLLING_UPDATES = 20


def post_clip_norm(pre_clip: float | None, clip: float | None) -> float | None:
    """The total gradient norm after ``clip_grad_norm_(max_norm=clip)``; None when unknown."""
    if pre_clip is None or not math.isfinite(pre_clip):
        return None
    if not clip or clip <= 0:
        return pre_clip
    return pre_clip * min(1.0, clip / (pre_clip + CLIP_EPSILON))


def record_id(run_id: str, attempt: int, kind: str, step: int | None) -> str:
    return hashlib.sha256(f'{run_id}\0{attempt}\0{kind}\0{step}'.encode()).hexdigest()[:24]


def _clean(value):
    """JSON-safe copy: non-finite floats become None (strict JSON), dataclasses become dicts."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _field(metrics, name, default=None):
    if metrics is None:
        return default
    if isinstance(metrics, dict):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


def scan(path: Path) -> dict:
    """Read every parseable line; a torn or invalid line is counted, never raised."""
    records, corrupt = [], 0
    path = Path(path)
    if not path.exists():
        return dict(records=records, corrupt=0, torn_tail=False)
    data = path.read_bytes()
    torn_tail = bool(data) and not data.endswith(b'\n')
    for line in data.split(b'\n'):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            corrupt += 1
            continue
        if isinstance(record, dict) and record.get('format') == FORMAT:
            records.append(record)
        else:
            corrupt += 1
    return dict(records=records, corrupt=corrupt, torn_tail=torn_tail)


class TelemetryWriter:
    """Writer for one process (one attempt). Not thread-safe; the trainer calls it from one thread."""

    def __init__(self, path: Path, *, run_id: str, resumed_from_step: int = 0, fsync_every: int = 16,
                 clip: float | None = None, clock: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic, job_kind: str | None = None,
                 recovery: dict | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.clip, self.fsync_every = run_id, clip, max(1, int(fsync_every))
        self.clock, self.monotonic = clock, monotonic
        existing = scan(self.path)
        self.attempt = 1 + max((r.get('attempt', 0) for r in existing['records']), default=0)
        self.seen = {r['record_id'] for r in existing['records'] if r.get('record_id')}
        self.seq = 0
        self.started = monotonic()
        self.optimizer_seconds = 0.0
        self.evaluation_seconds = 0.0
        self.checkpoint_seconds = 0.0
        self.rolling = deque(maxlen=ROLLING_UPDATES)
        self.failures: list[str] = []
        self.unsynced = 0
        self.closed = False
        self._stream = open(self.path, 'ab')
        if existing['torn_tail']:
            self._stream.write(b'\n')          # start on a fresh line; the torn bytes stay as evidence
        self._append('session', None, dict(resumed_from_step=int(resumed_from_step), resumed=resumed_from_step > 0,
                                           job_kind=job_kind, previous_corrupt_lines=existing['corrupt'],
                                           pid=os.getpid(), clip_threshold=clip, fsync_every=self.fsync_every,
                                           recovery=recovery), sync=True)

    # ------------------------------------------------------------------ failure accounting

    @property
    def status(self) -> str:
        return 'complete' if not self.failures else 'degraded'

    def safe(self, method: str, *args, **values) -> bool:
        """Call a record method; a failure is recorded (and reported), never raised into training."""
        try:
            return bool(getattr(self, method)(*args, **values))
        except Exception as error:  # noqa: BLE001 - observability must never stop a run
            self.failures.append(f'{method}: {type(error).__name__}: {error}'[:500])
            return False

    # ------------------------------------------------------------------ core append

    def _append(self, kind: str, step: int | None, body: dict, *, sync: bool = False) -> bool:
        if self.closed:
            raise RuntimeError('telemetry writer is closed')
        identifier = record_id(self.run_id, self.attempt, kind, step)
        if identifier in self.seen:
            return False
        self.seq += 1
        record = dict(format=FORMAT, run_id=self.run_id, attempt=self.attempt, seq=self.seq, kind=kind, step=step,
                      record_id=identifier, time_utc=self.clock(),
                      elapsed_seconds=round(self.monotonic() - self.started, 6), **_clean(body))
        self._stream.write(json.dumps(record, allow_nan=False, separators=(',', ':')).encode() + b'\n')
        self._stream.flush()
        self.seen.add(identifier)
        self.unsynced += 1
        if sync or self.unsynced >= self.fsync_every:
            os.fsync(self._stream.fileno())
            self.unsynced = 0
        return True

    # ------------------------------------------------------------------ record kinds

    def update(self, metrics, *, free_vram_bytes: int | None = None, gates: dict | None = None, **extra) -> bool:
        """One optimizer update. ``metrics`` is a TrainingMetrics (or a dict of its fields)."""
        step = int(_field(metrics, 'optimizer_step'))
        pre = _field(metrics, 'gradient_norm')
        step_seconds = _field(metrics, 'step_time_seconds')
        tokens = _field(metrics, 'tokens_processed')
        if step_seconds:
            self.optimizer_seconds += float(step_seconds)
            if tokens:
                self.rolling.append((float(tokens), float(step_seconds)))
        total_tokens = _field(metrics, 'total_tokens_seen')
        post = post_clip_norm(pre, self.clip)
        now = self.monotonic()
        # Off-CUDA the metrics helper reports VRAM as 0; a measurement that was not taken is null, not zero.
        device = str(_field(metrics, 'device', '') or '')
        measured = device.startswith('cuda')
        vram = {name: (_field(metrics, field) if measured else None) for name, field in (
            ('vram_allocated_bytes', 'allocated_vram_bytes'), ('vram_reserved_bytes', 'reserved_vram_bytes'),
            ('vram_peak_allocated_bytes', 'peak_allocated_vram_bytes'),
            ('vram_peak_reserved_bytes', 'peak_reserved_vram_bytes'))}
        rolling = (sum(t for t, _ in self.rolling) / sum(s for _, s in self.rolling)
                   if self.rolling and sum(s for _, s in self.rolling) > 0 else None)
        body = dict(
            optimizer_step=step, micro_step=_field(metrics, 'micro_step'),
            training_loss=_field(metrics, 'training_loss'), learning_rate=_field(metrics, 'learning_rate'),
            gradient_norm_pre_clip=pre, gradient_norm_post_clip=post, gradient_norm_post_clip_source='derived',
            clip_threshold=self.clip, clipped=(pre is not None and self.clip is not None and post is not None
                                               and post < pre),
            supervised_tokens=tokens, supervised_tokens_total=total_tokens,
            context_tokens=_field(metrics, 'context_tokens_processed'),
            padded_tokens=_field(metrics, 'padded_tokens_processed'),
            step_seconds=step_seconds, forward_backward_seconds=_field(metrics, 'forward_backward_seconds'),
            optimizer_update_seconds=_field(metrics, 'optimizer_update_seconds'),
            step_tokens_per_second=_field(metrics, 'tokens_per_second'),
            rolling_tokens_per_second=rolling, rolling_window_updates=len(self.rolling),
            effective_tokens_per_second=(total_tokens / (now - self.started)
                                         if total_tokens and now > self.started else None),
            optimizer_seconds_total=self.optimizer_seconds, evaluation_seconds_total=self.evaluation_seconds,
            checkpoint_seconds_total=self.checkpoint_seconds,
            **vram, vram_measured=measured, device=device or None,
            vram_driver_free_bytes=free_vram_bytes, precision=_field(metrics, 'precision'),
            validation_loss=_field(metrics, 'validation_loss'), gates=gates, **extra)
        return self._append('update', step, body)

    def evaluation(self, step: int, summary: dict, *, seconds: float, schedule: str | None = None) -> bool:
        """An evaluation boundary. Raw generation rows stay in the evaluation file, not here."""
        self.evaluation_seconds += float(seconds)
        compact = {k: v for k, v in (summary or {}).items() if k != 'rows'}
        schedule = schedule or compact.get('schedule')
        return self._append(f'evaluation:{schedule}' if schedule else 'evaluation', int(step),
                            dict(seconds=seconds, schedule=schedule, evaluation_seconds_total=self.evaluation_seconds,
                                 metrics=compact), sync=True)

    def checkpoint(self, step: int, *, role: str, record: dict | None, seconds: float, error: str | None = None) -> bool:
        self.checkpoint_seconds += float(seconds)
        record = record or {}
        return self._append(f'checkpoint:{role}', int(step), dict(
            role=role, seconds=seconds, published=error is None and bool(record),
            path=record.get('path'), sha256=record.get('sha256'), bytes=record.get('bytes'),
            verified=record.get('verified'), error=error), sync=True)

    def gate(self, step: int | None, *, name: str, decision: str, detail: dict | None = None) -> bool:
        """A resource or quality gate that changed the run's course (e.g. a stop decision)."""
        return self._append(f'gate:{name}', step, dict(name=name, decision=decision, detail=detail), sync=True)

    def stop(self, *, state: str, reason: str | None, step: int | None, error: str | None = None,
             error_type: str | None = None, **extra) -> bool:
        return self._append('stop', step, dict(state=state, reason=reason, error=error, error_type=error_type,
                                               optimizer_seconds_total=self.optimizer_seconds,
                                               evaluation_seconds_total=self.evaluation_seconds,
                                               checkpoint_seconds_total=self.checkpoint_seconds,
                                               telemetry_failures=list(self.failures), **extra), sync=True)

    def close(self) -> None:
        if not self.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self.closed = True


def read_telemetry(path: Path) -> dict:
    """The effective record set after crashes and resumes (see module docstring)."""
    scanned = scan(path)
    records = scanned['records']
    sessions = {r['attempt']: r for r in records if r['kind'] == 'session'}
    effective = {}
    for record in sorted(records, key=lambda r: (r.get('attempt', 0), r.get('seq', 0))):
        if record['kind'] == 'session':
            # Rolled back: earlier attempts' records beyond the resume step no longer describe the run.
            restart = record.get('resumed_from_step') or 0
            for key in [k for k in effective if k[1] is not None and k[1] > restart
                        and effective[k]['attempt'] < record['attempt']]:
                del effective[key]
            continue
        effective[(record['kind'], record.get('step'))] = record   # latest attempt wins
    ordered = sorted(effective.values(), key=lambda r: (r.get('attempt', 0), r.get('seq', 0)))
    return dict(records=ordered, attempts=sorted(sessions), sessions=list(sessions.values()),
                corrupt_lines=scanned['corrupt'], torn_tail=scanned['torn_tail'], raw_records=len(records))


def summarize(path: Path) -> dict:
    """Compact statistics for reports; every field is None when the evidence is absent."""
    effective = read_telemetry(path)
    updates = [r for r in effective['records'] if r['kind'] == 'update']

    def values(name):
        return [r[name] for r in updates if r.get(name) is not None]
    peak_reserved, free = values('vram_peak_reserved_bytes'), values('vram_driver_free_bytes')
    clipped = [r['clipped'] for r in updates if r.get('clipped') is not None]
    stop = next((r for r in reversed(effective['records']) if r['kind'] == 'stop'), None)
    failures = (stop or {}).get('telemetry_failures') or []
    return dict(
        available=bool(updates), updates=len(updates), last_step=updates[-1]['optimizer_step'] if updates else None,
        first_loss=updates[0]['training_loss'] if updates else None,
        last_loss=updates[-1]['training_loss'] if updates else None,
        min_loss=min(values('training_loss'), default=None),
        supervised_tokens_total=updates[-1].get('supervised_tokens_total') if updates else None,
        clipped_updates=sum(clipped) if clipped else None,
        clip_threshold=updates[-1].get('clip_threshold') if updates else None,
        max_gradient_norm_pre_clip=max(values('gradient_norm_pre_clip'), default=None),
        last_learning_rate=updates[-1].get('learning_rate') if updates else None,
        peak_reserved_vram_bytes=max(peak_reserved, default=None),
        peak_allocated_vram_bytes=max(values('vram_peak_allocated_bytes'), default=None),
        min_driver_free_vram_bytes=min(free, default=None),
        mean_step_tokens_per_second=(sum(values('step_tokens_per_second')) / len(values('step_tokens_per_second'))
                                     if values('step_tokens_per_second') else None),
        last_rolling_tokens_per_second=updates[-1].get('rolling_tokens_per_second') if updates else None,
        optimizer_seconds_total=updates[-1].get('optimizer_seconds_total') if updates else None,
        evaluations=[dict(step=r['step'], seconds=r.get('seconds'), schedule=r.get('schedule'))
                     for r in effective['records'] if r['kind'].startswith('evaluation')],
        checkpoints=[dict(step=r['step'], role=r.get('role'), published=r.get('published'), sha256=r.get('sha256'),
                          seconds=r.get('seconds'), error=r.get('error'))
                     for r in effective['records'] if r['kind'].startswith('checkpoint:')],
        gates=[dict(step=r['step'], name=r.get('name'), decision=r.get('decision'))
               for r in effective['records'] if r['kind'].startswith('gate:')],
        stop=dict(state=stop.get('state'), reason=stop.get('reason'), error=stop.get('error')) if stop else None,
        # unavailable: nothing was recorded; degraded: any write failed or a line is corrupt; complete otherwise.
        status=('degraded' if failures or effective['corrupt_lines']
                else ('unavailable' if not effective['records'] else 'complete')),
        telemetry_failures=failures, attempts=effective['attempts'], corrupt_lines=effective['corrupt_lines'])
