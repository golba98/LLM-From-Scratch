#!/usr/bin/env bash
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run='codexa-100m-native-v1'
log="$repo/logs/$run"
cd "$repo" || exit 1
setsid .venv/bin/python -u scripts/train.py \
  --config configs/100m-native.yaml \
  --train-token-file data/tokenized/base-v1/mixed/train.bin \
  --validation-token-file data/tokenized/base-v1/fineweb_edu/validation.bin \
  --token-manifest data/tokenized/base-v1/production/token_data_manifest.json \
  --run-name "$run" --log-dir logs --checkpoint-dir checkpoints \
  --optimizer adamw --overwrite-log >/tmp/codexa-100m-native-training.log 2>&1 &
pid=$!
"$repo/scripts/legacy/training_viewer.sh" "$run" 10000 "$log" "checkpoints/$run/latest.pt" "$pid" "NATIVE BASE TRAINING"
wait "$pid"
