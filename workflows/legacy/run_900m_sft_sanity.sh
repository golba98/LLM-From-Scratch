#!/usr/bin/env bash
set -u

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run="codexa-900m-sft-sanity-v1"
log="$repo/logs/$run"
checkpoint="$repo/checkpoints/$run/latest.pt"
cd "$repo" || exit 1
mkdir -p "$log"

setsid "$repo/.venv/bin/python" -u scripts/train_conversational_sft.py \
  --config configs/1b.yaml \
  --checkpoint checkpoints/codexa-900m-base-v1/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --train-jsonl data/processed/chat-sft-v1/ultrachat/train.jsonl \
  --train-jsonl data/processed/chat-sft-v1/oasst1/train.jsonl \
  --validation-jsonl data/processed/chat-sft-v1/ultrachat/validation.jsonl \
  --validation-jsonl data/processed/chat-sft-v1/oasst1/validation.jsonl \
  --base-token-file data/tokenized/base-v1/mixed/train.bin \
  --run-name "$run" \
  --max-steps 205 \
  --train-limit 500 \
  --validation-limit 100 \
  --optimizer adamw8bit \
  >"/tmp/$run.log" 2>&1 &
pid=$!

if command -v kitty >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  kitty --title "Codexa 900M SFT Sanity" --hold sh -lc \
    "'$repo/scripts/legacy/training_viewer.sh' '$run' 205 '$log' '$checkpoint' '$pid' '900M SFT SANITY'"
else
  "$repo/scripts/legacy/training_viewer.sh" "$run" 205 "$log" "$checkpoint" "$pid" "900M SFT SANITY"
fi
wait "$pid"
