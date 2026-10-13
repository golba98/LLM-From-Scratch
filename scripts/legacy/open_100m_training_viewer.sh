#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
run="${CODEXA_TRAINING_RUN:-codexa-100m-native-v1}"
run_dir="$repo/logs/$run"
pid=$(pgrep -f "[s]cripts/train.py.*${run}" | head -n 1 || true)
exec "$repo/scripts/legacy/training_viewer.sh" "$run" 10000 "$run_dir" "checkpoints/$run/latest.pt" "$pid" "NATIVE BASE TRAINING"
