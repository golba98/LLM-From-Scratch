#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run='codexa-900m-sft-v2'
log="$repo/logs/$run"
cd "$repo" || exit 1
setsid .venv/bin/python -u scripts/train_conversational_sft.py \
  --config configs/1b.yaml \
  --checkpoint checkpoints/codexa-900m-base-v1/latest.pt \
  --resume-checkpoint checkpoints/codexa-900m-sft-v2/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --train-jsonl data/processed/chat-sft-v1/ultrachat/train.jsonl \
  --train-jsonl data/processed/chat-sft-v1/oasst1/train.jsonl \
  --validation-jsonl data/processed/chat-sft-v1/ultrachat/validation.jsonl \
  --validation-jsonl data/processed/chat-sft-v1/oasst1/validation.jsonl \
  --base-token-file data/tokenized/base-v1/mixed/train.bin \
  --run-name "$run" --max-steps 6000 --train-limit 30000 --validation-limit 500 \
  --optimizer adamw8bit \
  >"/tmp/$run.log" 2>&1 &
pid=$!
"$repo/scripts/legacy/training_viewer.sh" "$run" 6000 "$log" "checkpoints/$run/latest.pt" "$pid" "CONVERSATIONAL SFT"
wait "$pid"
