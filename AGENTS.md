# Canonical PyTorch workspace guidelines

This integration Git repository pins seven independent sibling component repositories through compatibility.json. Keep reusable implementations in their owners and integration/evaluation orchestration in `src/codexa_workspace`. Generic `src` and script adapters are temporary compatibility interfaces. Do not import the integration package from a component or introduce reverse/circular package dependencies.

Use `.venv` for generative checks and `.venv-specialist` for the frozen encoder/heads. Assets resolve through CODEXA_ASSET_ROOT or ignored artifacts.local.json; never fall back to retired project directories. Never commit weights, datasets, private logs, environments, source snapshots, credentials or recovery bundles. New experiments require distinct output paths beneath this workspace or an explicitly configured output root; historical assets are read-only inputs.

Preserve state-dictionary keys, NPZ/native checkpoint dialects, tokenizer IDs and hashes, model lineage, RNG/optimizer restoration and memory scopes. Base/SFT operator launches require the automatically attached visible 100x22 Kitty viewer. Use bounded synthetic tests; no expensive training runs during refactors.

Append dated decisions/evidence to documentation/training/SESSION_DECISIONS.md and milestones to documentation/training/100M_PROGRESS_LOG.md (never a repository-root copy). Run the relevant regression suites, wheel and CLI checks before PRs. Stage source changes and intentional sibling compatibility pins only. Do not force-push, merge PRs, alter repository permissions/protection or remove pending-retirement directories without satisfying the documented gates.

## Consolidated local storage — 2026-10-09

Saved PyTorch weights, datasets, logs, training records, and recovery archives are
physically stored under `3-Python/31-LLM Source/`, in `checkpoints/`, `exports/`,
`data/`, `logs/`, `training/`, and `recovery/`. Shared recovery archives also preserve
NumPy history; NumPy recovery links resolve to this location.

`Development/LLM-Assets-Staging` is now only a compatibility symlink to
`31-LLM Source/compatibility/LLM-Assets-Staging`. Its entries link to canonical
31/32 Source locations. Retain these legacy asset-path links for existing tools;
there is no second physical asset store. Active local settings use the direct
Source folders. Keep all saved contents out of Git and preserve every model,
tokenizer, dataset, log, and recovery artifact.

The deprecated and local-artifact NumPy folders are preserved inside
`32-LLM Source/legacy/`. Their two former `3-Python` shortcut entries were removed
at the user's request. The remaining 32-prefixed top-level entries are
`32-LLM (NumPy)` and `32-LLM Source`; active NumPy work is in workspace 38.

## Training operations — 2026-10-11

**Monitoring: use the Rich dashboard.** Run `.venv/bin/python scripts/training_dashboard.py` inside an existing terminal, or `--kitty` for its own 120x30 window (the layout is designed for 120x30).
- Keys: `q`, `r`, `d` details, `c` compact, `?` help.
- Other modes: `--once`, `--compact`, `--details`.
- Prefer it over the raw text viewer.
- The dashboard is read-only: it never stops a run or sends a notification.
- During training, avoid `--kitty` or any other new GPU window. Every window takes VRAM from the trainer's headroom.
- Instant status: `scripts/training_status.py [--json]`.
- **Every start (`launch`/`resume`) shows this dashboard** as the trainer's required viewer.
  - `launch.json` sets `LLM_TRAINING_VIEWER_COMMAND` → `training_dashboard.py --attach`.
  - It opens one 120x30 window in its own `systemd-run --user` unit, so the window survives the run's end and shows the final banner.
  - It reuses an already-open dashboard for the run, registered in `$XDG_RUNTIME_DIR`.
  - The old 100x22 viewer appears only if the installed `llm_training` predates the hook.

**Long runs are OS-supervised; never poll them with an AI.**
- Path: `scripts/training_supervisor.py` with `prepare` → `approve` → `launch`.
- `launch` runs every gate itself and then starts a `systemd-run --user` unit (`KillMode=mixed`).
- Never stop or restart a live supervisor: its exit kills the trainer.
- Resume: `resume --confirm-run <run>`. It is allowed only after an interruption or a resource-guard stop (VRAM or disk). It re-checks every gate and keeps the original deadline.
- Extend: `extend --confirm-run <run> --extend-seconds N --statement "<user's words>"`.
  - Only for a verified `TRAINING_DEADLINE` stop before the step cap, and only when the user explicitly grants the extra time.
  - The grant is recorded in `config/budget-extensions.jsonl`. It is the same exact resume; only the deadline moves.
- Never silently reset budgets, lower gates (180 GiB disk, 16 GiB RAM, 13,928 MiB free VRAM, 1.5 GiB in-run VRAM guard), or auto-start follow-up training.

**Frozen jobs hash-bind code.** These files are in the frozen job's `code_identity`:
- `src/codexa_workspace/{foundation_training,training_supervisor,training_wakeup,foundation_diagnosis}.py`
- the installed `llm_training`, `llm_data` and `llm_architecture` modules

While a job may still resume, put changes in new modules or the unhashed `scripts/`.

**Notifications.**
- `src/codexa_workspace/training_notifications.py` is the sole owner. It sends one at-most-once desktop popup per real terminal event, plus one AI wakeup (Codex by default) as a fresh session.
- Production status comes only from trusted configuration: `launch.json` `execution_class` or an approved job grant. Never from run-id text.
- Tests, simulations and dry runs must never reach the real `notify-send`, `codex` or `claude`. Use injected mocks and a fail-closed PATH. `conftest.py` sets `CODEXA_EXECUTION_CLASS=unit_test`.
- **Agent window at run end.** Once the AI analysis finishes, it opens one 120x30 Kitty window with an interactive session.
  - Codex: `codex resume <analysis session> -s read-only -a on-request`. Claude: `--resume <id> --permission-mode default`.
  - If the analysis failed, it starts a fresh session with the prompt.
  - The window runs in its own `systemd-run --user` unit, so it outlives the supervisor.
  - At most once per terminal event; record in `supervisor/agent-window.json`.
  - Reopen: `scripts/training_wakeup.py open-agent`. Disable: `launch`/`resume --no-agent-window`.
  - Never use a permissive sandbox for it. The user's `~/.codex/config.toml` defaults to full access.
- `scripts/training_supervisor.py` reloads `training_notifications` from disk at the terminal event, falling back to the copy it loaded at startup. Fixes to the notifier therefore reach a running supervisor, so keep that module importable.

**GPU headroom.** The 934M trainer reserves about 12.4–12.9 GiB of the RTX 4080's 16 GiB. Before and during a run:
- Quit Discord, Firefox, Zed and other GPU apps.
- Keep everything else under about 2 GiB total.

Run details are in `documentation/training/CODEXA_934M_TRAINING_SUPERVISOR.md`.
