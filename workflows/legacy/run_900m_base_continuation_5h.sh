#!/usr/bin/env bash
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="codexa-900m-base-continuation-5h-v1"
log="$repo/logs/$run"
checkpoint="$repo/checkpoints/$run/latest.pt"
cd "$repo" || exit 1
mkdir -p "$log"

setsid "$repo/.venv/bin/python" -u scripts/train.py \
  --config configs/900m-base-continuation-5h.yaml \
  --train-token-file data/tokenized/base-v1/mixed/train.bin \
  --validation-token-file data/tokenized/base-v1/mixed/train.bin \
  --token-manifest data/tokenized/base-v1/continuation-5h/token_data_manifest.json \
  --train-token-offset 0 \
  --train-token-count 12780548882 \
  --validation-token-offset 12780548882 \
  --validation-token-count 39298352 \
  --init-checkpoint checkpoints/codexa-900m-base-v1/latest.pt \
  --device cuda \
  --precision bf16 \
  --max-steps 2000 \
  --max-validation-batches 16 \
  --num-workers 0 \
  --run-name "$run" \
  --gradient-checkpointing \
  --optimizer adamw8bit \
  >"/tmp/$run.log" 2>&1 &
pid=$!

if command -v kitty >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  kitty --title "Codexa 900M Base Continuation" --hold sh -lc \
    "'$repo/scripts/legacy/training_viewer.sh' '$run' 2000 '$log' '$checkpoint' '$pid' '900M BASE CONTINUATION'"
else
  "$repo/scripts/legacy/training_viewer.sh" "$run" 2000 "$log" "$checkpoint" "$pid" "900M BASE CONTINUATION"
fi
wait "$pid"
