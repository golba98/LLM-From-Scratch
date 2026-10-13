# Training notifications

- `src/codexa_workspace/training_notifications.py` is the sole owner. Per real terminal event it sends:
  - one at-most-once desktop popup;
  - one AI wakeup (Codex by default), as a fresh session.
- `scripts/training_supervisor.py` reloads `training_notifications` from disk at the terminal event. If that fails it falls back to the copy it loaded at startup. Fixes to the notifier therefore reach a running supervisor, so keep that module importable.

## Production vs. test

- Production status comes only from trusted configuration: `launch.json` `execution_class` or an approved job grant. Never from run-id text.
- Tests, simulations and dry runs must never reach the real `notify-send`, `codex` or `claude`. Use injected mocks and a fail-closed PATH. `conftest.py` sets `CODEXA_EXECUTION_CLASS=unit_test`.

## Agent window at run end

Once the AI analysis finishes, the notifier opens one 100x22 Kitty window with an interactive session:
- Codex: `codex resume <analysis session> -s read-only -a on-request`. Claude: `--resume <id> --permission-mode default`.
- If the analysis failed, it starts a fresh session with the prompt.
- The window runs in its own `systemd-run --user` unit, so it outlives the supervisor.
- At most once per terminal event, recorded in `supervisor/agent-window.json`.
- Reopen: `scripts/training_wakeup.py open-agent`. Disable: `launch`/`resume --no-agent-window`.
- Never use a permissive sandbox for it. The user's `~/.codex/config.toml` defaults to full access.
