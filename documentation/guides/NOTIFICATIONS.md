# Training notifications

When a supervised run reaches a real terminal state, `codexa_workspace/notifications/notifier.py` (the sole owner) sends, exactly once per terminal event:
- one desktop popup;
- one AI wakeup (Codex by default): a fresh, non-interactive analysis session given the run summary (`notifications/reporting.py`);
- one interactive agent window, opened after the analysis finishes.

A durable outbox (`supervisor/notification-state.json`) records every attempt, so a crash or reboot never sends twice. Login `reconcile` delivers anything deferred.

The supervisor CLI reloads `notifications/notifier.py` from disk at the terminal event, falling back to the copy loaded at startup. Fixes to the notifier therefore reach a running supervisor, so keep that module importable.

## Production vs. test

- Production status comes only from trusted configuration: `launch.json` `execution_class`, or (for the 5h foundation run, which predates that field) an approved job grant. Never from run-id text.
- Tests, simulations and dry runs must never reach the real `notify-send`, `codex` or `claude`. Use injected mock backends and a fail-closed `PATH`. `conftest.py` sets `CODEXA_EXECUTION_CLASS=unit_test`.

## Agent window at run end

- One 100x22 Kitty window with an interactive session:
  - Codex: `codex resume <analysis session> -s read-only -a on-request`.
  - Claude: `--resume <id> --permission-mode default`.
- If the analysis failed, it starts a fresh session with the prompt.
- It runs in its own `systemd-run --user` unit, so it outlives the supervisor.
- At most once per terminal event, recorded in `supervisor/agent-window.json`.
- Disable it for a run with `"agent_window": false` in that run's `supervisor/launch.json`.
- Never use a permissive sandbox for it. The user's `~/.codex/config.toml` defaults to full access.

## Commands

```bash
scripts/training_wakeup.py open-agent [--run-dir <run>]          # reopen the agent window
scripts/training_wakeup.py deliver --dry-run [--run-dir <run>]   # show what would be sent; sends nothing
scripts/training_wakeup.py test-provider --provider codex --output <new-dir>
```

Without `--run-dir`, commands use the most recently active run.
