# Codexa training viewer style

This is the approved visual shape for the 100×22 Kitty training monitor. Keep
the layout minimal, left-aligned, and vertically scannable. Do not replace it
with split metric cards, dense bordered tables, or Unicode box art.

```text
  CODEXA V1   CONVERSATIONAL SFT
  ------------------------------------------------------------------------

  ############################################------   88%    ETA 00:10:54    RUNNING

  STEPS       1766 / 2000
  SPEED       23426 tok/s
  TRAIN LOSS  1.997
  VAL LOSS    --
  TOKENS      8.4M
  ELAPSED     00:00:00

  CHECKPOINT
  checkpoints/codexa-100m-sft-pilot/latest.pt

  LOG
  logs/codexa-100m-sft-pilot/train_metrics.jsonl

  BF16  |  CUDA  |  refresh 2s
```

The implementation is `scripts/training_viewer.sh`, shared by every training
launcher and completed-run viewer.
