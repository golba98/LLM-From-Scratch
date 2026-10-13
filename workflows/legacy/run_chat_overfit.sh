#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="chat-overfit-diagnostic-v1"
log_dir="$repo/logs/$run"
mkdir -p "$log_dir"
setsid "$repo/.venv/bin/python" -u "$repo/scripts/run_chat_overfit.py" \
  --device auto --run-name "$run" --log-dir "$repo/logs" \
  >"$log_dir/console.log" 2>&1 &
pid=$!
if command -v kitty >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  kitty --title "Codexa Chat Overfit Diagnostic" --hold sh -lc \
    "'$repo/scripts/legacy/training_viewer.sh' '$run' 300 '$repo/logs/$run' '$repo/checkpoints/$run/latest.pt' '$pid' 'CHAT OVERFIT DIAGNOSTIC'"
else
  "$repo/scripts/legacy/training_viewer.sh" "$run" 300 "$repo/logs/$run" \
    "checkpoints/$run/latest.pt" "$pid" "CHAT OVERFIT DIAGNOSTIC"
fi
wait "$pid"
