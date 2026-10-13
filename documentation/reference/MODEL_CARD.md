---
model_name: Codexa v1
architecture: decoder-only Transformer
parameters: 934356480
context_length: 2048
vocabulary_size: 16384
language: en
license: other
---

# Codexa v1 Model Card

Codexa v1 is a 934,356,480-parameter decoder-only Transformer built from scratch
with PyTorch. It uses 24 blocks, hidden size 1,536, 24 attention heads, SwiGLU
feed-forward layers, RMSNorm, learned position embeddings, and tied token/output
embeddings.

## Current status (2026-10-13)

- Foundation run `codexa-934m-foundation-5h-v1`: completed at its 1,792-update cap
  (117,440,512 tokens); final and best validation cross-entropy 3.0909. See the
  [capability diagnosis](../reports/CODEXA_934M_FOUNDATIONAL_CAPABILITY_DIAGNOSIS.md) and
  [qualification](../reports/CODEXA_934M_FINAL_QUALIFICATION_MODEL_SELECTION.md) reports.
- SFT feasibility pilot `codexa-934m-sft-feasibility-pilot-v1`: completed (128 updates); not promoted.
- Native chat and optional retrieval are implemented, but conversational quality remains
  inadequate. The Stage 2 pilot is experimental-not-promoted
  ([results](../reports/STAGE2_RESULTS.md)).
- No checkpoint has been promoted.

## Earlier rebuild status

The rebuild has not produced an accepted checkpoint. The retained local
FineWeb-Edu artifact contains 883,814,184 training tokens from one pinned
shard. It is pipeline evidence and initial smoke data, not an adequate final
training budget for this model.

## Intended use

The first target is English text completion for educational model-development
research. It is not currently a chat assistant and must not be represented as
one. A future conversational stage is blocked until the base model passes
fixed coherence, repetition, termination, and held-out-loss gates.

## Limitations and risks

- No accepted model weights currently exist.
- FineWeb-Edu can contain errors, bias, private information, unsafe text,
  copyrighted material, and duplicates missed by exact matching.
- The tokenizer and corpus are English-focused and not specialized for code.
- Future outputs may be false, inconsistent, biased, repetitive, or malformed.
- No safety alignment or instruction-following ability is assumed.

## License

Repository source code is MIT licensed. Model-weight licensing remains `other`
until a release review accounts for all upstream dataset terms. FineWeb-Edu
requires ODC-By attribution.
