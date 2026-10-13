#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run='codexa-100m-sft-pilot'
log="$repo/logs/$run"
cd "$repo" || exit 1
setsid .venv/bin/python -u scripts/train_conversational_sft.py \
  --checkpoint checkpoints/codexa-100m-native-v1/latest.pt \
  --resume-checkpoint checkpoints/codexa-100m-sft-pilot/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --train-jsonl data/processed/chat-sft-v1/ultrachat/train.jsonl \
  --validation-jsonl data/processed/chat-sft-v1/ultrachat/validation.jsonl \
  --base-token-file data/tokenized/base-v1/mixed/train.bin \
  --run-name "$run" --max-steps 2000 --train-limit 60000 --validation-limit 1000 \
  >/tmp/codexa-100m-sft-pilot.log 2>&1 &
pid=$!
"$repo/scripts/legacy/training_viewer.sh" "$run" 2000 "$log" "checkpoints/$run/latest.pt" "$pid" "CONVERSATIONAL SFT"
wait "$pid"
