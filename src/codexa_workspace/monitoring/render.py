"""Rich layout for the 120x30 training dashboard: panels, alerts, compact view and plain-text frames."""

import io
from pathlib import Path
import statistics
import time

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from codexa_workspace.monitoring.reader import GIB, RunReader, derive


SPARK = '▁▂▃▄▅▆▇█'
BADGES = {'RUNNING': ('RUNNING', 'black on green3'), 'VALIDATING': ('VALIDATING', 'black on cyan'),
          'GENERATING': ('GENERATING', 'black on cyan'), 'CHECKPOINTING': ('CHECKPOINTING', 'white on dodger_blue2'),
          'EARLY_STOPPED': ('EARLY STOP', 'black on gold1'), 'COMPLETED': ('COMPLETED', 'black on bright_green'),
          'FAILED': ('FAILED', 'white on red3'), 'INTERRUPTED': ('INTERRUPTED', 'white on magenta'),
          'BLOCKED': ('BLOCKED', 'white on red3'), 'READY': ('READY', 'black on grey70'),
          'VALIDATING_GATES': ('VALIDATING GATES', 'black on grey70'), 'NOT_PREPARED': ('NOT PREPARED', 'black on grey50'),
          'STALE': ('NO HEARTBEAT', 'black on gold1')}
LABEL, VALUE, DIM, GOOD, BAD, WARN, ACCENT, BORDER = ('grey62', 'bold white', 'grey50', 'green3', 'red1',
                                                      'gold1', 'deep_sky_blue1', 'grey37')
PHASE_NAMES = {'TRAINING': 'Training', 'VALIDATING': 'Validating', 'BASELINE': 'Baseline evaluation',
               'GENERATION_EVAL': 'Generation evaluation', 'CHECKPOINTING': 'Saving checkpoint',
               'FINALIZING': 'Finalizing', 'STARTING': 'Starting'}


def alerts(s: dict, d: dict) -> list[tuple[str, str]]:
    """Deterministic read-only rules. ('crit'|'warn'|'info', text)."""
    found, p = [], s['progress']
    if d['terminal']:
        c = s['completion'] or {}
        level = 'info' if d['state'] == 'COMPLETED' else ('crit' if d['state'] == 'FAILED' else 'warn')
        found.append((level, f"Run ended: {d['state']} — {c.get('reason') or s['state'].get('failure_reason')}"))
        if c.get('final_checkpoint_verification') not in (None, 'verified'):
            found.append(('warn', f"Final checkpoint verification: {c.get('final_checkpoint_verification')}"))
        for error in (c.get('errors') or [])[-2:]:
            found.append(('crit', str(error)[:160]))
    elif d['state'] == 'BLOCKED':
        found.append(('crit', 'Launch blocked: ' + ', '.join(s['gates'].get('failed') or [s['state'].get('failure_reason') or '?'])))
    else:
        age = d['heartbeat_age']
        if age is not None and age > 180:
            found.append(('crit', f'Heartbeat stale for {age:.0f}s'))
        elif age is not None and age > 60:
            found.append(('warn', f'Heartbeat delayed {age:.0f}s'))
        last_step = s['heartbeat'].get('last_step_utc')
        typical = statistics.median([m['step_time_seconds'] for m in s['metrics'][-30:]]) if s['metrics'] else None
        if d['phase'] == 'TRAINING' and last_step and typical and s['now'] - last_step > max(60, 3 * typical):
            found.append(('warn', f"No optimizer update for {s['now'] - last_step:.0f}s (typical {typical:.1f}s)"))
        if d['projected_stop'] is not None and d['projected_stop'] < d['total']:
            found.append(('info', f"Deadline projection: run will stop near update {d['projected_stop']:,} "
                                  f"of {d['total']:,} (TRAINING_DEADLINE)"))
        if len(d['speeds']) >= 10:
            median = statistics.median(d['speeds'][-60:])
            if d['speeds'][-1] < 0.85 * median:
                found.append(('warn', f"Throughput slowdown: {d['speeds'][-1]:,.0f} tok/s vs median {median:,.0f}"))
        if isinstance(p.get('gradient_norm'), (int, float)) and p['gradient_norm'] > d['guards'].get('gradient_norm_threshold', 10):
            found.append(('crit', f"Gradient norm {p['gradient_norm']:.2f} above guard"))
        next_ckpt = p.get('next_checkpoint_step')
        if next_ckpt and d['step'] > next_ckpt + 2 and d['phase'] != 'CHECKPOINTING':
            found.append(('warn', f'Checkpoint overdue (expected at update {next_ckpt})'))
        free_gpu = p.get('free_gpu_bytes')
        if isinstance(free_gpu, (int, float)) and free_gpu < d['guard'] + 0.5 * GIB:
            level = 'crit' if free_gpu < d['guard'] else 'warn'
            found.append((level, f"Free VRAM {free_gpu / GIB:.2f} GiB near the {d['guard'] / GIB:.1f} GiB stop guard — "
                                 'close GPU-heavy apps'))
    if d['baseline'] and d['latest'] and d['latest'] is not d['baseline']:
        threshold = d['guards'].get('source_validation_CE_increase', 0.1)
        for source, loss in (d['latest'].get('sources') or {}).items():
            delta = loss - (d['baseline'].get('sources') or {}).get(source, loss)
            if delta > threshold:
                found.append(('crit', f'{source} validation +{delta:.3f} vs baseline (guard {threshold})'))
            elif delta > threshold / 2:
                found.append(('warn', f'{source} validation +{delta:.3f} vs baseline'))
    free = s['disk_free']
    if free is not None and not d['terminal']:
        if free < 110 * GIB:
            found.append(('crit', f'Disk free {free / GIB:.0f} GiB near the 100 GiB recovery reserve'))
        elif free < 190 * GIB:
            found.append(('warn', f'Disk free {free / GIB:.0f} GiB below the 180 GiB launch floor margin'))
    if not s['progress'] and not d['terminal'] and d['state'] not in ('BLOCKED', 'NOT_PREPARED', 'READY'):
        found.append(('warn', 'No trainer progress file yet (data may be stale)'))
    return found


def notification_lines(s: dict, d: dict) -> list[tuple[str, str]]:
    """What the single owner (supervisor notifier) has recorded — the viewer never sends anything."""
    desktop, ai, legacy = s['desktop'], s['ai'], (s['legacy_notifications'] or {}).get('channels') or {}
    if not d['terminal']:
        return [('info', 'Desktop: one final-outcome notification will be sent by the supervisor at the end')]
    rows = []
    state = (desktop or {}).get('notification_state') or (legacy.get('desktop') or {}).get('status')
    backend = (desktop or {}).get('backend') or ('legacy' if legacy.get('desktop') else None)
    note = {'mock': ' (test run: mock only, nothing shown)',
            'legacy': ' (legacy notifier: outcome + analysis-ready popups)'}.get(backend, ' once')
    labels = {'delivered': ('ok', f'Desktop notification delivered{note}'),
              'failed': ('warn', 'Desktop notification failed (recorded; not retried)'),
              'indeterminate': ('warn', 'Desktop notification indeterminate (interrupted; not resent)'),
              'backend_unavailable': ('warn', 'Desktop notification pending: no session bus yet'),
              'claimed': ('info', 'Desktop notification being sent')}
    rows.append(labels.get(state, ('info', 'Desktop notification pending')))
    status = (ai or {}).get('status') or (legacy.get('ai') or {}).get('status')
    provider = (ai or {}).get('provider') or (s['job'].get('wakeup') or {}).get('provider') or 'AI'
    ai_labels = {'delivered': ('ok', f'{provider} analysis complete → final-report/ai-analysis.md'),
                 'submitted': ('info', f'{provider} analysis requested (running)'),
                 'deferred': ('info', 'AI analysis deferred until next start'),
                 'failed_retryable': ('warn', 'AI analysis failed; one retry remains'),
                 'failed_final': ('crit', 'AI analysis failed → final-report/RESUME_MANUALLY.md'),
                 'disabled': ('info', 'AI wakeup disabled')}
    if status:
        rows.append(ai_labels.get(status, ('info', f'AI wakeup: {status}')))
    window = s.get('window') or {}
    if window.get('terminal_event_id') == (s.get('completion') or {}).get('event_id'):
        reopen = 'reopen: scripts/training_wakeup.py open-agent'
        window_labels = {'opened': ('ok', f"{window.get('provider') or provider} agent window opened — {reopen}"),
                         'failed': ('warn', f'Agent window failed to open — {reopen}'),
                         'indeterminate': ('warn', f'Agent window interrupted while opening — {reopen}'),
                         'backend_unavailable': ('info', 'Agent window waits for a graphical session'),
                         'disabled': ('info', 'Agent window disabled for this run')}
        if window.get('status') in window_labels:
            rows.append(window_labels[window['status']])
    return rows


def duration(value) -> str:
    if value is None:
        return '—'
    value = max(0, int(value))
    return f'{value // 3600}h {value // 60 % 60:02d}m' if value >= 3600 else f'{value // 60}m {value % 60:02d}s'


def clock(value) -> str:
    return time.strftime('%H:%M', time.localtime(value)) if value else '—'


def spark(values: list[float], width: int = 24) -> str:
    values = values[-width:]
    if len(values) < 2:
        return ''
    low, high = min(values), max(values)
    return ''.join(SPARK[int((v - low) / (high - low) * (len(SPARK) - 1)) if high > low else 3] for v in values)


def delta(value: float | None, reference: float | None, *, lower_is_better: bool = True, digits: int = 4) -> Text:
    if value is None or reference is None:
        return Text('')
    change = value - reference
    good = change < 0 if lower_is_better else change > 0
    arrow = '▼' if change < 0 else ('▲' if change > 0 else '•')
    return Text(f'{arrow} {abs(change):.{digits}f}', style=GOOD if good else (BAD if change else DIM))


def number(value, digits=4, suffix='') -> str:
    return '—' if value is None else f'{value:,.{digits}f}{suffix}'


def bar(fraction: float, width: int = 12, *, style: str = ACCENT) -> Text:
    fraction = max(0., min(1., fraction or 0.))
    filled = round(fraction * width)
    return Text('━' * filled, style=style) + Text('━' * (width - filled), style='grey23')


def grid() -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style=LABEL, no_wrap=True, min_width=16)
    table.add_column(style=VALUE, no_wrap=True, min_width=14)
    table.add_column(no_wrap=True, overflow='ellipsis', ratio=1)
    return table


def panel(body, title: str, *, style: str = BORDER) -> Panel:
    return Panel(body, title=f'[bold {ACCENT}]{title}[/]', title_align='left', border_style=style,
                 box=box.ROUNDED, padding=(0, 1))


def header(s: dict, d: dict, width: int) -> Panel:
    label, style = BADGES.get(d['display'], (d['display'], 'black on grey70'))
    title = Text.assemble(('CODEXA 934M', f'bold {ACCENT}'), ('  ·  Foundational continued pretraining', 'grey70'))
    top = Table.grid(expand=True)
    top.add_column(ratio=1)
    top.add_column(justify='right', no_wrap=True)
    top.add_row(title, Text(f' {label} ', style=f'bold {style}'))
    phase = PHASE_NAMES.get(d['phase'], d['phase'] or '—') if not d['terminal'] else 'Run finished'
    top.add_row(Text(s['run_id'], style=DIM), Text(phase, style='grey70'))
    progress = Table.grid(expand=True, padding=(0, 1))
    progress.add_column(ratio=1)
    progress.add_column(justify='right', no_wrap=True)
    bar_style = 'green3' if d['state'] in ('RUNNING', 'COMPLETED') else ('red3' if d['state'] == 'FAILED' else 'gold1')
    progress.add_row(ProgressBar(total=max(1, d['total']), completed=d['step'], complete_style=bar_style,
                                 finished_style=bar_style, pulse=False),
                     Text.assemble((f"{d['step']:,}", VALUE), (f" / {d['total']:,}", DIM), (f"  {100 * d['fraction']:5.1f}%", VALUE)))
    timing = Text.assemble(('Elapsed ', LABEL), (duration(d['elapsed']), VALUE))
    if d['terminal']:
        timing += Text.assemble(('   of ', LABEL), (duration(d['seconds']), VALUE), (' budget', LABEL))
    else:
        timing += Text.assemble(('   Budget left ', LABEL), (duration(d['remaining']), VALUE),
                                ('   ETA ', LABEL), (duration(d['eta']), VALUE),
                                ('   Finish ≈ ', LABEL), (clock(d['finish']), VALUE))
    return Panel(Group(top, Text(''), progress, timing), border_style=ACCENT if not d['terminal'] else bar_style,
                 box=box.HEAVY if width >= 70 else box.SQUARE, padding=(0, 1))


def training_panel(s: dict, d: dict) -> Panel:
    p, t = s['progress'], grid()
    losses = d['losses']
    recent = statistics.fmean(losses[-16:]) if len(losses) >= 32 else None
    earlier = statistics.fmean(losses[-32:-16]) if len(losses) >= 32 else None
    trend = delta(recent, earlier) + Text(' 16-step avg  ', style=DIM) if recent is not None else Text('')
    t.add_row('Train loss', number(p.get('training_loss') if not d['terminal'] else (losses[-1] if losses else None)),
              trend + Text(spark(losses), style=ACCENT))
    latest = d['latest']['weighted_loss'] if d['latest'] else None
    t.add_row('Validation loss', number(latest),
              delta(latest, d['baseline']['weighted_loss'] if d['baseline'] else None) + Text(' vs baseline', style=DIM))
    t.add_row('Best validation', number(d['best']), Text(''))
    lr = p.get('learning_rate')
    t.add_row('Learning rate', '—' if lr is None else f'{lr:.2e}', Text(d['schedule'] or '', style=DIM))
    speeds = d['speeds']
    median = statistics.median(speeds[-60:]) if speeds else None
    current = speeds[-1] if speeds else p.get('tokens_per_second')
    ratio = Text(f'{100 * current / median:.0f}% of median  ', style=DIM) if current and median else Text('')
    t.add_row('Throughput', number(current, 0, ' tok/s'), ratio + Text(spark(speeds), style='cyan'))
    target = f" / {d['token_target'] / 1e6:,.1f}M" if d['token_target'] else ''
    t.add_row('Processed tokens', '—' if d['tokens'] is None else f"{d['tokens'] / 1e6:,.2f}M{target}", Text(''))
    t.add_row('Coverage', f"{100 * d['fraction']:.1f}%", Text('of the planned window order', style=DIM))
    if p.get('gradient_norm') is not None:
        t.add_row('Gradient norm', f"{p['gradient_norm']:.3f}", Text(f"guard {d['guards'].get('gradient_norm_threshold', 10)}", style=DIM))
    return panel(t, 'Training')


def validation_panel(s: dict, d: dict) -> Panel:
    t = Table(box=None, padding=(0, 2), show_edge=False, header_style=LABEL)
    t.add_column('Source', style=LABEL, no_wrap=True)
    t.add_column('Baseline', justify='right', style='grey70')
    t.add_column('Latest', justify='right', style=VALUE)
    t.add_column('Change', no_wrap=True)
    base = (d['baseline'] or {}).get('sources') or {}
    latest = (d['latest'] or {}).get('sources') or {}
    for source in sorted(set(base) | set(latest)):
        name = {'fineweb_edu': 'FineWeb-Edu', 'wikipedia': 'Wikipedia'}.get(source, source)
        t.add_row(name, number(base.get(source)), number(latest.get(source)), delta(latest.get(source), base.get(source)))
    footer = Text.assemble(('Evaluations ', LABEL), (str(len(s['validations'])), VALUE))
    if d['latest']:
        footer += Text.assemble(('   last at update ', LABEL), (f"{d['latest']['optimizer_step']:,}", VALUE))
    if d['next_eval']:
        footer += Text.assemble(('   next at ', LABEL), (f"{d['next_eval']:,}", VALUE))
    return panel(Group(t if latest or base else Text('No validation yet', style=DIM), footer), 'Validation by source')


def checkpoint_panel(s: dict, d: dict) -> Panel:
    t, c, p = grid(), d['checkpoint'], s['progress']
    if not d['terminal']:
        upcoming = p.get('next_checkpoint_step')
        when = f"≈ {duration((upcoming - d['step']) * d['per_step'])}" if upcoming and d['per_step'] else ''
        t.add_row('Next checkpoint', f'update {upcoming:,}' if upcoming else '—', Text(when, style=DIM))
    if c:
        final = (s['completion'] or {}).get('final_checkpoint') or {}
        failed_final = ((s['completion'] or {}).get('final_checkpoint_verification') == 'failed'
                        and str(final.get('path', '')).endswith(c.get('path', '?')))
        health = (Text('✗ failed final integrity check', style=BAD) if failed_final else
                  Text('✓ verified at save', style=GOOD) if c.get('verified') else Text('✗ unverified', style=BAD))
        t.add_row('Last verified', f"update {c.get('optimizer_step'):,}", health)
        detail = f"{(c.get('bytes') or 0) / GIB:.1f} GiB · {c.get('write_verify_seconds', 0):.0f}s save"
        t.add_row('Saved', duration(d['checkpoint_age']) + ' ago' if d['checkpoint_age'] is not None else '—', Text(detail, style=DIM))
    else:
        t.add_row('Last verified', '—', Text('none yet', style=DIM))
    best = s['recovery'].get('best_validation')
    if best:
        t.add_row('Best-val role', f"update {best.get('optimizer_step'):,}", Text(''))
    final = (s['completion'] or {}).get('final_checkpoint')
    if d['terminal']:
        verification = (s['completion'] or {}).get('final_checkpoint_verification')
        style = GOOD if verification == 'verified' else WARN
        t.add_row('Final checkpoint', f"update {final.get('optimizer_step'):,}" if final else '—',
                  Text(verification or '—', style=style))
    t.add_row('Backend', 'legacy', Text(f"6 slots · {len(s['recovery'])} roles", style=DIM))
    return panel(t, 'Checkpoints')


def resources_panel(s: dict, d: dict) -> Panel:
    t, gpu, p = grid(), s['gpu'], s['progress']
    if gpu:
        fraction = gpu['used_mib'] / gpu['total_mib']
        t.add_row('GPU memory', f"{gpu['used_mib'] / 1024:.1f} / {gpu['total_mib'] / 1024:.1f} GiB",
                  bar(fraction, style=BAD if fraction > .92 else ACCENT) + Text(f"  util {gpu['utilization']}%", style=DIM))
    if p.get('peak_vram_bytes'):
        t.add_row('Trainer peak', f"{p['peak_vram_bytes'] / GIB:.2f} GiB", Text('reserved', style=DIM))
    if isinstance(p.get('free_gpu_bytes'), (int, float)) and not d['terminal']:
        free = p['free_gpu_bytes']
        t.add_row('Free VRAM', f'{free / GIB:.2f} GiB', Text(f"stop guard {d['guard'] / GIB:.1f} GiB",
                  style=BAD if free < d['guard'] + 0.5 * GIB else DIM))
    if s['host_available'] is not None:
        t.add_row('Host RAM free', f"{s['host_available'] / GIB:.1f} GiB", Text('launch needs 16', style=DIM))
    if s['disk_free'] is not None:
        t.add_row('Disk free', f"{s['disk_free'] / GIB:.0f} GiB", Text('floor 180 · reserve 100', style=DIM))
    return panel(t, 'Resources')


STYLE = {'crit': ('✖', BAD), 'warn': ('▲', WARN), 'info': ('●', ACCENT), 'ok': ('✓', GOOD)}


def alerts_panel(s: dict, d: dict) -> Panel:
    found = alerts(s, d)
    rows = Table.grid(padding=(0, 1))
    rows.add_column(no_wrap=True)
    rows.add_column()
    if not found:
        rows.add_row(Text('✓', style=GOOD), Text('No alerts — training healthy', style=GOOD))
    for level, message in found:
        icon, style = STYLE[level]
        rows.add_row(Text(icon, style=style), Text(message, style=style if level != 'info' else 'grey85'))
    for level, message in notification_lines(s, d):
        icon, style = STYLE[level]
        rows.add_row(Text(icon, style=style), Text(message, style='grey70'))
    worst = BAD if any(l == 'crit' for l, _ in found) else (WARN if any(l == 'warn' for l, _ in found) else BORDER)
    return panel(rows, 'Alerts & notifications', style=worst)


def banner(s: dict, d: dict) -> Panel:
    c = s['completion'] or {}
    v = c.get('validation') or {}
    icon = {'COMPLETED': '✔', 'EARLY_STOPPED': '■', 'FAILED': '✖', 'INTERRUPTED': '❚❚'}.get(d['state'], '•')
    colour = {'COMPLETED': GOOD, 'EARLY_STOPPED': WARN, 'FAILED': BAD, 'INTERRUPTED': 'magenta'}.get(d['state'], ACCENT)
    t = grid()
    t.add_row('Outcome', Text(f"{icon} {d['state'].replace('_', ' ')}", style=f'bold {colour}'), Text(str(c.get('reason') or ''), style=colour))
    t.add_row('Updates', f"{d['step']:,} / {d['total']:,}", Text(f"{(c.get('processed_tokens') or 0) / 1e6:,.2f}M tokens", style=DIM))
    t.add_row('Validation', f"{number(v.get('baseline'))} → {number(v.get('final'))}",
              delta(v.get('final'), v.get('baseline')) + Text(f"  best {number(v.get('best'))}", style=DIM))
    final = c.get('final_checkpoint') or {}
    if final:
        t.add_row('Final checkpoint', Path(final['path']).parent.name + '/' + Path(final['path']).name,
                  Text(c.get('final_checkpoint_verification') or '', style=GOOD if c.get('final_checkpoint_verification') == 'verified' else WARN))
    ai = s['ai'] or {}
    legacy_ai = ((s['legacy_notifications'] or {}).get('channels') or {}).get('ai') or {}
    report = Path(s['run_dir']) / 'final-report/ai-analysis.md'
    if report.exists():
        t.add_row('AI analysis', 'final-report/ai-analysis.md', Text(ai.get('resume_command') or legacy_ai.get('resume_command') or '', style=DIM))
    note = Text('Next: human review of the analysis — no automatic training follows.', style='grey85')
    return panel(Group(t, note), 'Run finished', style=colour)


def details_panel(s: dict, d: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=LABEL, no_wrap=True)
    t.add_column(style='grey85', overflow='fold')
    job, spec = s['job'], s['job'].get('spec') or {}
    st, hb = s['state'], s['heartbeat']
    rows = [('Run directory', s['run_dir']), ('Job SHA256', job.get('job_sha256')),
            ('Trainer run_id', hb.get('run_id') or (s['completion'] or {}).get('trainer_run_id')),
            ('Initializer', f"{(spec.get('initialization') or {}).get('path')}  sha {((spec.get('initialization') or {}).get('sha256') or '')[:16]}"),
            ('Tokenizer sha', ((spec.get('tokenizer') or {}).get('sha256') or '')[:16]),
            ('Supervisor', f"state {st.get('state')} · pid {(st.get('supervisor') or {}).get('pid')} · unit {(st.get('supervisor') or {}).get('unit')}"),
            ('Trainer', f"pid {(st.get('trainer') or {}).get('pid')} · activity {s['progress'].get('activity')} · stop {s['progress'].get('stop_reason')}"),
            ('Budget', f"start {clock(s['budget'].get('started_utc'))} · deadline {clock(s['budget'].get('deadline_utc'))} · reserve {s['budget'].get('reserve_seconds')}s"),
            ('Failure reason', st.get('failure_reason') or s['progress'].get('error') or '—')]
    for role, record in sorted(s['recovery'].items()):
        rows.append((f'Role {role}', f"update {record.get('optimizer_step')} · {record.get('path')} · sha {str(record.get('sha256'))[:12]}"))
    for event in s['events'][-5:]:
        rows.append(('Event', f"{event.get('at')} {event.get('event')} {event.get('target', event.get('reason', ''))}"))
    for label, value in rows:
        t.add_row(label, str(value))
    return panel(t, 'Details (raw)')


def footer(run_dir: Path, *, details: bool, compact: bool) -> Text:
    keys = Text.assemble(('q', f'bold {ACCENT}'), (' quit  ', DIM), ('r', f'bold {ACCENT}'), (' refresh  ', DIM),
                         ('d', f'bold {ACCENT}'), (' details' + (' ✓' if details else '') + '  ', DIM),
                         ('c', f'bold {ACCENT}'), (' compact' + (' ✓' if compact else '') + '  ', DIM),
                         ('?', f'bold {ACCENT}'), (' help', DIM))
    return Text.assemble(('Closing this viewer never stops training.  ', 'grey70'), keys)


def compact_view(s: dict, d: dict) -> Panel:
    t = grid()
    latest = d['latest']['weighted_loss'] if d['latest'] else None
    t.add_row('Train loss', number(d['losses'][-1] if d['losses'] else s['progress'].get('training_loss')), Text(spark(d['losses'], 12), style=ACCENT))
    t.add_row('Val loss', number(latest), delta(latest, d['baseline']['weighted_loss'] if d['baseline'] else None))
    t.add_row('Throughput', number(d['speeds'][-1] if d['speeds'] else None, 0, ' tok/s'), Text(''))
    c = d['checkpoint']
    t.add_row('Checkpoint', f"update {c.get('optimizer_step')}" if c else '—', Text('✓' if c.get('verified') else '', style=GOOD))
    return panel(t, 'Key metrics')


def frame(s: dict, *, width: int, height: int | None = None, details: bool = False, compact: bool = False,
          measure: Console | None = None):
    """Build a renderable that fits the terminal; secondary panels are dropped first."""
    d = derive(s)
    compact = compact or width < 70
    top = [header(s, d, width)]
    if d['terminal']:
        top.append(banner(s, d))
    if compact:
        body = [compact_view(s, d), alerts_panel(s, d)]
        optional = []
    else:
        primary = [training_panel(s, d), checkpoint_panel(s, d)]
        secondary = [validation_panel(s, d), resources_panel(s, d)]
        body, optional = [], secondary
        if width >= 110:
            columns = Table.grid(expand=True, padding=(0, 1))
            columns.add_column(ratio=1)
            columns.add_column(ratio=1)
            columns.add_row(Group(primary[0], secondary[0]), Group(primary[1], secondary[1]))
            body = [columns, alerts_panel(s, d)]
            optional = []
        else:
            body = [primary[0], alerts_panel(s, d), primary[1]]
    tail = [details_panel(s, d)] if details else []
    parts = top + body + optional + tail + [footer(Path(s['run_dir']), details=details, compact=compact)]
    if height and measure is not None:
        def lines(renderables):
            return len(measure.render_lines(Group(*renderables), measure.options.update(width=width), pad=False))
        if lines(parts) > height and width >= 110 and not compact:
            # Two columns too tall: keep the primary panels stacked, drop the secondary ones.
            body = [training_panel(s, d), alerts_panel(s, d), checkpoint_panel(s, d)]
            parts = top + body + tail + [footer(Path(s['run_dir']), details=details, compact=compact)]
        while optional and lines(parts) > height:
            optional.pop()
            parts = top + body + optional + tail + [footer(Path(s['run_dir']), details=details, compact=compact)]
        if lines(parts) > height and not compact:
            parts = top + [compact_view(s, d), alerts_panel(s, d)] + tail + [footer(Path(s['run_dir']), details=details, compact=True)]
    return Group(*parts)


def render_text(run_dir: Path, *, width: int = 120, height: int | None = None, details: bool = False,
                compact: bool = False, gpu: bool = True, color: bool = False, reader: RunReader | None = None) -> str:
    """One frame as text (plain, or ANSI when ``color``); used by --once and tests."""
    reader = reader or RunReader(run_dir, gpu=gpu)
    buffer = io.StringIO()
    console = Console(file=buffer, width=width, height=height or 200, force_terminal=color,
                      color_system='truecolor' if color else None, record=not color)
    renderable = frame(reader.read(), width=width, height=height, details=details, compact=compact,
                       measure=Console(file=io.StringIO(), width=width))
    console.print(renderable)
    return buffer.getvalue() if color else console.export_text()
