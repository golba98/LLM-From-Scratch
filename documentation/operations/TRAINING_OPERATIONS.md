# Training operations

Full supervisor design and run history: [CODEXA_934M_TRAINING_SUPERVISOR.md](CODEXA_934M_TRAINING_SUPERVISOR.md).
Notifications: [NOTIFICATIONS.md](NOTIFICATIONS.md). Files you must not touch while a job can resume: [FROZEN_CODE.md](FROZEN_CODE.md).

## Monitoring: the Rich dashboard

- Run `.venv/bin/python scripts/training_dashboard.py` in an existing terminal, or add `--kitty` to give it its own 120x30 window. The layout is designed for 120x30.
- Keys: `q` quit, `r` refresh, `d` details, `c` compact, `?` help. Other modes: `--once`, `--compact`, `--details`.
- Instant status: `scripts/training_status.py [--json]`.
- The dashboard is read-only. It never stops a run or sends a notification.
- During training, avoid `--kitty` and any other new GPU window, because every window takes VRAM from the trainer's headroom.
- **Every start (`launch`/`resume`) shows this dashboard** as the trainer's required viewer:
  - `launch.json` sets `LLM_TRAINING_VIEWER_COMMAND` to `training_dashboard.py --attach`.
  - It opens one 120x30 window in its own `systemd-run --user` unit, so the window survives the run's end and shows the final banner.
  - It reuses an already-open dashboard for the run (registered in `$XDG_RUNTIME_DIR`).
  - The old 100x22 viewer (`llm_training.viewer`) appears when `LLM_TRAINING_VIEWER_COMMAND` is unset, as in direct/recipe launches outside the supervisor, or when the installed `llm_training` predates the hook.

## Long runs are OS-supervised — never poll them with an AI

- Path: `scripts/training_supervisor.py` with `prepare` → `approve` → `launch`. `launch` runs every gate itself, then starts a `systemd-run --user` unit (`KillMode=mixed`).
- For new SSD output roots use `scripts/training_supervisor_ssd.py` (see [STORAGE_AND_ASSETS.md](STORAGE_AND_ASSETS.md)).
- Never stop or restart a live supervisor, because its exit kills the trainer.
- **Resume:** `resume --confirm-run <run>`. It is allowed only after an interruption or a resource-guard stop (VRAM or disk). It re-checks every gate and keeps the original deadline.
- **Extend:** `extend --confirm-run <run> --extend-seconds N --statement "<user's words>"`.
  - Only for a verified `TRAINING_DEADLINE` stop before the step cap, and only when the user explicitly grants the extra time.
  - The grant is recorded in `config/budget-extensions.jsonl`. It is the same exact resume; only the deadline moves.
- Never silently reset budgets, never lower gates (180 GiB disk, 16 GiB RAM, 13,928 MiB free VRAM, 1.5 GiB in-run VRAM guard), and never auto-start follow-up training.
- The `codexa-train-reconcile.service` user unit runs `scripts/training_supervisor.py reconcile` at login. It finalizes interrupted runs and never resumes training.

## GPU headroom

The 934M trainer reserves about 12.4–12.9 GiB of the RTX 4080's 16 GiB. Before and during a run:
- Quit Discord, Firefox, Zed and other GPU apps.
- Keep everything else under about 2 GiB in total.

## Development vs. production interpreters

- Development and tests run through `tools/codexa-dev` (source-first components). The plain `.venv` interpreter's training guard rejects a dirty integration tree.
- Historical reproduction uses `.venv` without the launcher. See [../workspace/ENVIRONMENTS.md](../workspace/ENVIRONMENTS.md).
