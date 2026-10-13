# Training operations

How a training run is prepared, launched, watched and finished. Related guides:
[NOTIFICATIONS.md](NOTIFICATIONS.md) (what happens when a run ends), [CODE_IDENTITY.md](CODE_IDENTITY.md)
(what a frozen job binds), [STORAGE_AND_ASSETS.md](STORAGE_AND_ASSETS.md) (where outputs go).
The supervisor's design and the 934M run history are in [the supervisor report](../reports/CODEXA_934M_TRAINING_SUPERVISOR.md).

## Who does what

| Piece | Responsibility |
| --- | --- |
| **The job** (e.g. `scripts/foundation_continuation.py`) | Prepares and freezes a job, checks readiness, records the user's approval, then launches or resumes it |
| **The supervisor** (`scripts/training_supervisor.py run`) | Runs one prepared run in a `systemd-run --user` unit (`KillMode=mixed`), watches the trainer, classifies the end, triggers notifications |
| **Reconcile** (`scripts/training_supervisor.py reconcile`) | Runs at login from `codexa-train-reconcile.service`. Finalizes runs whose supervisor was lost and delivers deferred notifications. **Never** resumes training |
| **The dashboard** (`scripts/training_dashboard.py`) | Read-only view of a run. It can never stop a run or notify |

Never poll a run with an AI, and never stop or restart a live supervisor: its exit kills the trainer.

## The foundation continuation job

```bash
# Preparation (inference only; no optimizer is ever built)
tools/codexa-dev scripts/foundation_continuation.py baseline  --directory <prep-dir>
tools/codexa-dev scripts/foundation_continuation.py freeze    --directory <prep-dir>
tools/codexa-dev scripts/foundation_continuation.py readiness --experiment <prep-dir>/config/job.json --run-tests

# The user only (readiness is not authorization)
tools/codexa-dev scripts/foundation_continuation.py approve --experiment <job.json> --job-sha256 <hash> --statement '<words>'
tools/codexa-dev scripts/foundation_continuation.py launch  --experiment <job.json> --provider claude
tools/codexa-dev scripts/foundation_continuation.py resume  --experiment <job.json> --confirm-run <run-id> --provider claude
```

- `launch` refuses unless the frozen job and its code identity verify, the approval names this exact job hash, readiness is `READY FOR AUTHORIZATION` and every gate passes at that moment.
- `resume` is allowed only after an interruption or a resource-guard stop (`MEMORY_HEADROOM_GATE`, `DISK_HEADROOM`, `DISK_RESERVE_WATCHDOG`). It re-checks every gate and keeps the original deadline.
- The job prepared on 2026-10-12 can no longer launch, because its code identity predates the 2026-10-13 restructure and the LLM-Architecture refactor. Prepare a fresh one.

## Gates (never lowered)

The values live in `codexa_workspace/training/gates.py`. Jobs may only raise them.

| Gate | Value |
| --- | --- |
| Free disk | 180 GiB, plus a 100 GiB recovery reserve |
| Available host RAM | 16 GiB |
| Free VRAM at start | 13,928 MiB |
| In-run VRAM guard | 1.5 GiB |

Never silently reset budgets, and never auto-start follow-up training.

## Monitoring

- `scripts/training_dashboard.py` shows the most recently active run by default; pass `--run-dir` for a specific one.
  - Keys: `q` quit, `r` refresh, `d` details, `c` compact, `?` help.
  - Modes: `--once`, `--compact`, `--details`, and `--kitty` for its own 120x30 window.
- `scripts/training_status.py [--json] [--watch]` gives an instant text status.
- Every supervised start shows the dashboard automatically:
  - The job's `launch.json` sets `LLM_TRAINING_VIEWER_COMMAND` to `training_dashboard.py --attach`.
  - That opens one 120x30 Kitty window in its own user unit (it survives the run's end), or reuses an open one.
- Direct `llm_training` launches outside the supervisor show `llm_training`'s own 100x22 viewer instead.
- During training, avoid `--kitty` and any other new GPU window, because each one takes VRAM from the trainer.

## GPU headroom

The 934M trainer reserves about 12.4–12.9 GiB of the RTX 4080's 16 GiB. Before and during a run:
- Quit Discord, Firefox, Zed and other GPU apps.
- Keep everything else under about 2 GiB in total.

## Interpreters

- Development and tests: `tools/codexa-dev` (component sources first).
- Operator scripts add the same source-first path themselves, so plain `.venv/bin/python scripts/<name>.py` works too.
- The startup training guard (`tools/training_environment_guard.py`, loaded by a `.pth` hook in `.venv`) refuses training unless an explicit `CODEXA_TRAINING_ENVIRONMENT` is set and the workspace matches its pins. Inspect it with `scripts/environment_identity.py --preflight-environment development --operation new`.
