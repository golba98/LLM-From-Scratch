# Codexa 934M — Training supervisor, status and AI wakeup

Date: 11 October 2026. Run: `codexa-934m-foundation-5h-v1`.

## Workflow

Prepare → validate → start training → exit AI session → monitor locally → detect completion → notify → re-invoke AI once.

**Who does what**
- The AI prepared and froze the job.
- The systemd user manager owns the process lifetime. `Linger=yes`, so the run survives logout.
- The Python supervisor waits on the trainer's pidfd. It wakes every 60 s to read small atomic JSON files.
- The trainer enforces in-loop guards.
- Exactly one fresh agent run (Codex) is invoked when the run reaches a terminal state.

No LLM is consulted during monitoring.

## Components

| Path | Owner | Role |
|---|---|---|
| `LLM-Training/src/llm_training/pretrain_run.py` | Training | Indexed-window continued pretraining. Covers the frozen order sampler, source-weighted validation, phase-aware heartbeat thread and in-loop guards. Uses legacy six-slot `RecoveryStore` and independently reloads the final checkpoint. |
| `src/codexa_workspace/foundation_training.py` | Integration | Derives the 5h spec from the pilot proposal and freezes `job.json` (job SHA). Also holds the approval binding, the launch gates, the trainer entry point and the 48-case neutral/direct generation evaluator. |
| `src/codexa_workspace/training_supervisor.py` | Integration | Explicit state machine, spawn/re-attach, phase-aware stall detection, deadline backstop, disk watchdog, signal forwarding, terminal classification, `completion.json`, boot reconcile and systemd launch. |
| `src/codexa_workspace/training_wakeup.py` | Integration | Durable exactly-once outbox (`notification-state.json`), compact `ai-handoff.md`, `notify-send`, Codex/Claude adapters, manual fallback. |
| `src/codexa_workspace/training_status.py` | Integration | Status reader. Never contacts an AI. |
| `scripts/training_{supervisor,status,wakeup}.py` | Integration | CLIs. |

## Authority boundary

**Trainer:**
- Nonfinite loss or gradient norm, and CUDA OOM: `FAILED`; uncertain weights are never saved.
- Gradient norm >10 for 3 consecutive updates: `EARLY_STOPPED`.
- Any source CE more than 0.1 above baseline for 2 consecutive evaluations: `EARLY_STOPPED`.
- Free VRAM below 1.5 GiB at an optimizer boundary: `EARLY_STOPPED`.
- The pre-save disk check refuses a save that would breach the 100 GiB recovery reserve.
- A checkpoint publication failure gives `FAILED`; the prior verified state is retained.
- The deadline has a 600 s finalization reserve.
- Generation evaluations are diagnostic only. Their errors are recorded and do not stop the run; nonfinite logits and OOM still do.

**Supervisor:**
- Phase-aware stalls give `FAILED`. The thresholds without step progress are:

  | Phase | Threshold |
  |---|---|
  | TRAINING | 15 min |
  | Validation / generation | 20 min |
  | Checkpointing | 15 min |
  | Heartbeat older than 3 min | (any phase) |

- Budget deadline + 10 min hard backstop gives `FAILED`.
- Disk below the 100 GiB reserve: SIGTERM, giving `EARLY_STOPPED`.
- The termination sequence is SIGTERM, then SIGKILL after the grace period.
- `COMPLETED` additionally requires exit 0 and a SHA-verified `final` recovery role. Exit 0 alone yields `FAILED / FINAL_CHECKPOINT_UNVERIFIED`.

**Restart policy:** the supervisor never restarts training, never changes hyperparameters, never deletes anything and never resets the budget.

## Commands

```sh
cd "/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch"
.venv/bin/python scripts/training_status.py            # instant status
.venv/bin/python scripts/training_status.py --json     # compact JSON
.venv/bin/python scripts/training_status.py --watch 10 # local refreshing view
systemctl --user status codexa-train-codexa-934m-foundation-5h-v1
systemctl --user stop codexa-train-codexa-934m-foundation-5h-v1   # graceful: stops at an optimizer boundary, saves, INTERRUPTED
```

## Failure, interruption and restart

| Event | Response |
|---|---|
| CUDA OOM or trainer crash | `FAILED`. The last verified checkpoint is preserved. One notification and one Codex diagnosis follow. |
| Logout | The unit continues (linger enabled). |
| `systemctl stop` or host shutdown | The supervisor forwards SIGTERM. The trainer stops at a boundary, saves if the reserve allows and reports `INTERRUPTED`. The AI wakeup is deferred. |
| Reboot or a lost supervisor | The persistent oneshot `codexa-train-reconcile.service` (user `default.target`) marks the run `INTERRUPTED (HOST_RESTART)` and delivers the deferred notification and wakeup once. It never resumes training. |

Manual resume:

```sh
training_supervisor.py resume --confirm-run codexa-934m-foundation-5h-v1
```

- It accepts only `INTERRUPTED` runs with an SHA-verified latest checkpoint.
- It carries the original deadline forward; downtime counts against the budget.

## AI wakeup

- **Codex:** `codex exec --skip-git-repo-check -s read-only -C <repo> --json -o final-report/ai-analysis.md -`
  - The prompt is passed on stdin.
  - The configured model is `gpt-6.1-sol`.
- **Claude:** `claude -p --output-format json --permission-mode dontAsk --max-budget-usd 5 --add-dir <run> --allowedTools Read Glob Grep Edit(final-report/**) Write(final-report/**)`
  - It runs in the run directory.

Rules for both:
- Every invocation is a **fresh** session, not a resumption of the launching conversation. The new session id is recorded, along with a `resume_command`.
- The agent environment is filtered and sets `CODEXA_WAKEUP_AGENT=1`, which launch, supervise and trainer all refuse.
- There are at most 2 AI attempts per event.
- Failures are recorded and produce `final-report/RESUME_MANUALLY.md` plus a critical desktop notice.

## Verification

| Check | Result |
|---|---|
| `LLM-Training`: `tests/test_pretrain_run.py tests/test_readiness_recovery.py` (CPU) | 12 passed. Covers bitwise interrupt/resume equality, budget carry-forward, guards, disk reserve, checkpoint failure, nonfinite, order drift, wakeup-agent refusal and non-fatal diagnostic errors. |
| Integration: `tests/test_training_supervisor.py` (fake trainer, stub CLIs) | 18 passed. Covers success, crash, nonfinite, checkpoint failure, early stop, corrupt final, missing report, stall, heartbeat loss, deadline backstop, disk watchdog, stop-signal deferral, supervisor restart re-attach, host-restart reconcile, failed-wakeup retry bound, Claude adapter, illegal transitions and a detached `systemd-run` unit. |
| Launch-gate test rerun | LLM-Training 12 passed; integration 36 passed. |
| Real CLI connectivity, harmless prompt | Codex passed (11.0 s); Claude passed (7.8 s, wrote via restricted `Write(final-report/**)`). |
| Bounded full-shape GPU smoke (separate `-smoke` namespace, no checkpoint publication) | Baseline validation and 48 generations completed. 2 AdamW8bit updates at ~7.74k tok/s, 8.47 s/update. Peak reserved 12.13 GiB. Loss 3.083 → 3.027. |
| First smoke attempt (`-smoke-attempt1`) | Exposed a real defect before launch: the hash-frozen `evaluate_model` aggregate divides by zero for a neutral/direct subset. The trainer now has its own loop reusing `scoring`/`likelihood_probe`. |

**Incident during testing.** The first integration test run's stub CLIs had a shebang containing spaces. `execvp` therefore fell through to the real CLIs:
- About 16 read-only `codex exec` runs and 1 `claude -p` run ($0.34) analysed fake runs.
- Desktop notifications were sent.

Their outputs stayed in pytest temp directories and no repository files changed. The tests now use a fail-closed PATH that excludes the agent CLI directory, plus a stub sanity check.

## Launch record

| Item | Value |
|---|---|
| Job SHA256 | `d7aa275e8904549ab799e283a0ef48d422bc5cc2393af544caf429a5c6eb01e6` |
| Approval | `config/approval.json` (plan approval; neutral-evidence blocker overridden by the user) |
| Launch gates | All 17 passed (`config/launch-gates.json`) |
| Disk / host RAM / free VRAM | 235.894 GiB / 22.02 GiB / 14,863 MiB |
| Unit | `codexa-train-codexa-934m-foundation-5h-v1` |
| Supervisor / trainer PIDs | 32436 / 32460 |
| Trainer run_id | `d4080f1c7db045d89943c12c4c89ced7` |
| Start / deadline | 16:20:35 UTC / 21:20:35 UTC |
| Baseline weighted validation CE | 3.11296 (Wikipedia 3.03582, FineWeb-Edu 3.16207; 524,288 targets) |
| Early training | Updates 1–3 completed at ~8.0k tok/s |

At the measured speed, all 1,792 updates plus validation, saves and generation approach the 5h budget. The deadline may end the run slightly before step 1,792; that would be recorded as `TRAINING_DEADLINE`.

## Run outcome (2026-10-11 16:43 UTC)

The run stopped itself at update 129 of 1,792, after 22.4 minutes, as `EARLY_STOPPED / MEMORY_HEADROOM_GATE`. Free VRAM measured at an optimizer boundary was 0.96 GiB, below the frozen 1.5 GiB guard.

- The trainer's reserved memory stayed at 12,420 MiB throughout. Other GPU clients took roughly 3 GB at that moment, against about 1 GB normally.
- The final step-129 checkpoint was published and SHA-verified.
- Weighted validation CE went 3.11296 → 3.11140.
- The supervisor's (legacy) notifier sent one outcome popup and one analysis-ready popup, and woke Codex once. The Codex analysis is in `final-report/ai-analysis.md`.
- The user chose to leave training stopped. Any continuation needs a new frozen job and grant.

## Notifications v2 (owner: `src/codexa_workspace/training_notifications.py`)

**Root causes of the popup spam**
1. The first integration-test runs had stub CLIs whose space-containing shebang failed. `execvp` fell through to the real `notify-send`/`codex`, so every fake scenario showed real popups: `fake-run: RuntimeError`, `DEADLINE_BACKSTOP`, and so on.
2. Production delivery had no execution-class guard.
3. The legacy design sent two popups per event without a replace id. It also allowed a desktop retry after a reported failure.

The viewer, status tool, trainer and reconcile code never notified.

**Policy**
- **Classification.** Execution class comes only from trusted configuration:
  - `launch.json.execution_class`, which only the production launcher writes as `production`;
  - otherwise, a self-verified approved job grant.

  Run-id text plays no part.
- **Mock-only test runs.** `integration_test`, `unit_test`, `simulation` and `unclassified` runs use recording mocks (`supervisor/notification-mock.jsonl`); no popup and no agent. `CODEXA_EXECUTION_CLASS` can only downgrade a classification, and the integration `conftest.py` sets it to `unit_test`.
- **Separate durable records.** Desktop and AI each have their own record, `supervisor/desktop-notification.json` and `supervisor/ai-wakeup.json`. Both are written under `flock(notification.lock)` with fsynced atomic replace.
- **Desktop delivery is at-most-once per terminal event:**
  1. A durable `claimed` record is written before `notify-send -p`.
  2. The result is then recorded as `delivered`, `failed` or `indeterminate`.
  3. A crash after the claim becomes `indeterminate` and is never resent. A failed send is never retried.
  4. Only `backend_unavailable` may be retried, because nothing was sent.
  5. Replace ids are reused only for the same run.
- **One popup per event.** The AI is submitted first, so the single popup says "Analysis requested" only when that is true. AI completion goes to the record and the dashboard, not to a second popup. AI gets at most 2 submissions, and success requires exit 0 plus non-empty output.
- **Compatibility.** `notification-state.json` remains as a summary for the hash-bound supervisor and reconcile code. Legacy state written by the old in-memory notifier is imported as already delivered.
- **Where the policy applies.** Only the unhashed `scripts/training_supervisor.py` and `scripts/training_wakeup.py` were re-pointed. The hash-bound `training_wakeup.py`/`training_supervisor.py` are byte-identical, so `verify_job` and manual resume still pass. A supervisor that is already running keeps the code it imported. The new policy therefore applies to every fresh process: reconcile, CLI delivery and future launches.

## Interactive agent window (2026-10-11)

When the AI outcome for a terminal event is final, `deliver` opens one interactive agent window.
- **When it opens:** after the analysis succeeds, or when the AI status is `failed_final`.
- **Window:** a 120x30 Kitty window titled `Codexa agent — <run>: <STATE>`.
- **What it runs:**

| Situation | Command |
|---|---|
| Analysis succeeded (Codex) | `codex resume <session> -s read-only -a on-request -C <repo>` |
| Analysis succeeded (Claude) | `claude --resume <session> --permission-mode default` |
| Analysis failed | A fresh session given the analysis prompt |

- **Permissions.** The window starts read-only and asks you before anything more. It overrides the user's permissive Codex defaults.
  - It does not set `CODEXA_WAKEUP_AGENT`. The user may therefore direct a resume from the window, approving each command.
  - The headless analysis agent still may not launch training.
- **Survives the supervisor.** It is started with `systemd-run --user --collect --unit codexa-agent-<run>-<event>`. Its own unit means it is not killed when the supervisor's `KillMode=mixed` unit ends.
- **Environment.** Only non-secret variables (PATH, HOME, display/session-bus) are passed on the command line. The agent CLIs use their stored logins.
- **Record.** `supervisor/agent-window.json`, using claim → open → result. Repeats are silent.
  - `backend_unavailable` (no display) is retried at the next delivery.
  - A failed or interrupted open is never retried automatically.
  - The record is archived by `resume`.
- **Reopen or disable.** Reopen with `scripts/training_wakeup.py open-agent [--provider]`. Disable with `--no-agent-window` on `launch`/`resume`, which writes `launch.json` `agent_window: false`.
- **Tests and other classes.** Non-production runs, and any delivery with injected backends, use `RecordingWindowBackend` only.
- **Late-binding notifier.** `scripts/training_supervisor.py` reloads `training_notifications` at the terminal event and falls back to the copy it loaded at startup. The resume at 18:37 UTC therefore gets this window when the run ends.

## Start-time viewer = Rich dashboard (2026-10-11)

**Cause of the old window.**
- `llm_training.pretrain_run` (hash-bound) calls `conversational_training.start_chat_viewer` when `viewer_required` is set.
- That function hard-coded `kitty 100x22 … -m llm_training.conversational_training --view`.
- The window was a trainer child inside the `KillMode=mixed` unit, so it died when the run ended.

**Fix.**
- `start_chat_viewer` (not hash-bound) honours `LLM_TRAINING_VIEWER_COMMAND`, a JSON argv. The supplied command gets `--status`/`--marker` and must attach within 15 s, otherwise the start fails exactly as before.
- `write_launch` sets it to `training_dashboard.py --attach`.
- `codexa_workspace/dashboard_window.py` then either reuses a live dashboard registered for the run in `$XDG_RUNTIME_DIR` or opens `systemd-run --user … kitty 120x30 training_dashboard.py --marker`.
- The dashboard writes the marker after its first frame.
- The attach process lives while the window does, so the trainer's `viewer_closed` stays truthful.
- `--kitty` uses the same opener and reuse check.

**Hash binding.** The rebuilt `llm_training` wheel differs only in `conversational_training.py`. Every hash-bound module is byte-identical, so `verify_job` is unaffected.

## Live dashboard

Requires `rich` in `.venv`: optional extra `viewer` in `pyproject.toml`. It added `rich==15.0.0`, `markdown-it-py==4.2.0` and `mdurl==0.1.2`; nothing else changed. `requirements-generative.lock` was not regenerated.

```sh
.venv/bin/python scripts/training_dashboard.py            # live (q quit · r refresh · d details · c compact · ? help)
.venv/bin/python scripts/training_dashboard.py --kitty    # separate Kitty window
.venv/bin/python scripts/training_dashboard.py --compact | --details | --once [--plain --width N]
```

**Panels:**
- Header with status badge.
- Progress bar with elapsed time, budget left, ETA and finish time.
- Training, with smoothed loss delta and sparklines.
- Validation by source.
- Checkpoints, showing verified-at-save versus failed-final.
- Resources.
- Alerts and notifications.
- A terminal banner, or a raw details panel.

**Layout:** two columns at 110 columns and wider, stacked at 70–109, compact below 70. Secondary panels are dropped first when height is short.

**How it reads data:** only small JSON documents, plus byte-offset tails of the metrics and validation JSONL files. `nvidia-smi` is cached for 10 s. It never writes, signals or notifies.

**Measured cost:** 1.7% CPU at a 1 s refresh and 40 MiB RSS.

## Verification (notifications + dashboard)

| Check | Result |
|---|---|
| `tests/test_training_notifications.py` | passed — all 14 required behaviours, including competing processes, crash-after-claim, simulated `DEADLINE_BACKSTOP`/`RuntimeError` silence, honest AI failure and legacy import |
| `tests/test_training_dashboard.py` | passed — 8 states × widths 50/80/120, height cap, ETA/deadline maths, alerts, details, torn-write tail |
| `tests/test_training_supervisor.py` | 17 passed — now asserts zero stub CLI calls and exactly one mock popup and one mock AI record per event |
| Full integration suite | 278 passed, 5 optional skips |
| Live checks | Real run classified `production` via its grant. Dry-run delivery on a copy: truthful text, nothing sent, real run dir unchanged. Pty session: keys work and the terminal is restored. |
| Safety snapshot before/after | Identical — hash-bound files, `verify_job`, pids and notification records |
