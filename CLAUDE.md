# Integration guidance

Follow AGENTS.md (including its "Training operations" section: use `scripts/training_dashboard.py` for monitoring, OS-supervised runs, notification ownership, hash-bound files and GPU headroom), README.md, MIGRATION.md and compatibility.json. The preserved source snapshot contains historical guidance. Do not restore stale model dimensions, merge environments, promote the pilot or modify original assets.

## Consolidated local storage — 2026-10-09

Saved PyTorch weights, datasets, logs, training records, and recovery archives are
physically stored under `3-Python/31-LLM Source/`, in `checkpoints/`, `exports/`,
`data/`, `logs/`, `training/`, and `recovery/`. Shared recovery archives also preserve
NumPy history; NumPy recovery links resolve to this location.

`Development/LLM-Assets-Staging` is now only a compatibility symlink to
`31-LLM Source/compatibility/LLM-Assets-Staging`. Its entries link to canonical
31/32 Source locations. Retain these legacy asset-path links for existing tools;
there is no second physical asset store. Active local settings use the direct
Source folders. Keep all saved contents out of Git and preserve every model,
tokenizer, dataset, log, and recovery artifact.

The deprecated and local-artifact NumPy folders are preserved inside
`32-LLM Source/legacy/`. Their two former `3-Python` shortcut entries were removed
at the user's request. The remaining 32-prefixed top-level entries are
`32-LLM (NumPy)` and `32-LLM Source`; active NumPy work is in workspace 38.
