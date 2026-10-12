# Masked KV-cache repair (2026-10-12)

## Defect

The post-training generation investigation (private report `outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256/REPORT.md`, finding 5) confirmed a latent defect in `LanguageModel.forward`. It has two coupled causes:

- **A — position reset.** With `attention_mask` and cached keys and values, default position IDs were `cumsum(mask) - 1` over the current tokens only. A cached decode step therefore restarted at position 0.
- **B — cached padding attended.** `CausalSelfAttention` prepended an all-ones mask for cached keys, so padding stored in the cache was always attended to. The cache tuple stores no mask, so an offset alone could not fix padded caches.

The historical generator (`llm_inference.generate`) compacts every prompt and passes no model mask, so historical generations never used this path.

## Correction

The correction is LLM-Architecture `bdf3683`, which changes `src/llm_architecture/model.py` (`b7f857ad…` → `3d4e9d1a…`).

- `attention_mask` may be `[batch, past_length + sequence]` (full mask). This form is required for padded caches, and the caller keeps it across steps.
- `attention_mask` may also be `[batch, sequence]` (legacy). The cached prefix is then defined as fully valid.
- Default positions are `cumsum(full_mask) - 1` over the current tokens.
- Other mask shapes and negative `position_ids` raise `ValueError`.
- Unmasked paths, parameters, state-dict keys, the cache tuple and the checkpoint format are unchanged.

## Validation

FP32 maximum absolute logit error on the real checkpoints, with read-only loading, no optimizer, and checkpoint SHA-256 unchanged:

| Measurement | Final `ca779a16…` before → after | Baseline `808afec9…` before → after |
|---|---|---|
| Masked cached step | 5.794949 → 1.907e-05 | 6.122253 → 1.431e-05 |
| Full-mask cached step | rejected → 1.907e-05 | rejected → 1.431e-05 |
| Left-padded batch, full mask | rejected → ≤ 2.34e-05 | rejected → ≤ 1.24e-05 |

The residual ~1e-5 equals the unmasked cached path's own FP32 difference.

The 48-prompt historical BF16 greedy benchmark matches 48/48 historical token arrays before and after, on both checkpoints. Unmasked logits digests are bit-identical. A current-token mask over a padded cache still attends to that padding, by contract.

Tests:

- `LLM-Architecture/tests/test_kv_cache_masking.py` (57 cases, CPU and CUDA/BF16);
- `LLM-Inference/tests/test_cached_generation_regression.py` (16; `4758dee`);
- the integration `tests/test_post_training_generation_diagnostic.py`;
- the checkpoint validator `scripts/masked_cache_repair_validation.py` (`--historical-context` runs the installed `.venv` wheels).

Private evidence:

- `outputs/diagnostics/masked-kv-cache-repair/20261012T060900Z/` (before and after the source edit);
- `…/20261012T095454Z-environments/` (after the commits). Here "before" means the historical environment (installed wheels, `--historical-context`) and "after" means the current, source-first environment. Both give 48/48 historical matches on both checkpoints, 48/48 identical generations across the two environments, and identical unmasked logits digests.

## Historical identity

`llm_architecture.model` is hash-bound into the completed run's `code_identity`. The frozen metadata was not changed. The historical environment (`.venv` without the bootstrap) still reproduces it exactly; the current, source-first environment intentionally differs. See `documentation/workspace/ENVIRONMENTS.md`.

The historical source can be recovered in two ways:

- `git -C ../LLM-Architecture show c334620:src/llm_architecture/model.py`;
- the read-only artifact `outputs/preservation/historical-source/codexa-934m-foundation-5h-v1/`, which includes 10 hash-bound files never committed to Git.
