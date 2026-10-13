#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run='codexa-100m-sft-pilot'
run_dir="$repo/logs/$run"
pid=$(pgrep -f "[s]cripts/train_conversational_sft.py.*${run}" | head -n 1 || true)
exec "$repo/scripts/legacy/training_viewer.sh" "$run" 2000 "$run_dir" "checkpoints/$run/latest.pt" "$pid" "CONVERSATIONAL SFT"
