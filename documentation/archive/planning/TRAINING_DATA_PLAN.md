# Codexa v1 Training Data Operations

This document explains what will be trained, what the repository can execute
now, and the evidence required before the next gate. `PHASE_PLAN.md` is the
authoritative 18-stage checklist.

## Will it converse better?

Base pretraining on FineWeb-Edu and Wikipedia should improve English, general
information, explanations, and sustained text. It does not by itself teach a
user/assistant protocol. Later role-aware SFT on UltraChat and OASST1 teaches
turn taking, direct answers, follow-ups, and stopping. A weak base cannot be
repaired reliably by a short SFT run, so both independently evaluated stages
are required.

This is not a promise of factuality, advanced reasoning, or frontier-model
quality. Acceptance comes from frozen measurements and blind human scores.

> Codexa v1 is initially a general-language research model. FineWeb-Edu,
> Wikipedia, UltraChat and OASST1 may contain incidental programming material,
> but this training plan does not establish reliable code-generation
> capability.

## Approved local sources

| Role | Source | Local selection | Verified records | Raw Parquet size |
| --- | --- | --- | ---: | ---: |
| Base | FineWeb-Edu | ten pinned `sample-10BT` shards | 7,293,000 documents | 21,520,368,509 bytes |
| Base | English Wikipedia | all 41 pinned `20231101.en` shards | 6,407,814 articles | 11,630,929,031 bytes |
| Future SFT | UltraChat 200k | `train_sft`, untouched `test_sft` | 230,975 conversations | 813,204,672 bytes |
| Future SFT | OASST1 | train, untouched official validation | 88,838 message rows | 41,596,430 bytes |

Revisions and licenses live in `configs/data_preparation.yaml` and
`documentation/reference/DATASET.md`. Each raw directory contains a checksummed
download manifest. This task neither replaces nor redownloads those sources.

## Base data contract

### Deterministic cleaning order

1. Normalize Unicode, line endings, control characters, and whitespace.
2. Reject empty, invalid, or malformed documents with source/reason counters.
3. Exact-deduplicate FineWeb-Edu internally.
4. Exact-deduplicate Wikipedia internally.
5. Exact-deduplicate across both sources to remove direct Wikipedia copies from
   FineWeb-Edu.
6. Record near-duplicate fingerprints and run the staged similarity audit.
7. Assign complete duplicate groups to a split using the seed and stable group
   identity—not row order, worker count, or Parquet shard order.
8. Tokenize and pack only after split assignment.

`src/data/governance.py` implements the in-memory contract and focused leakage
tests. `scripts/prepare_base_corpus.py` applies the same digest-based split and
cross-source exact-deduplication semantics at full-corpus scale using SQLite.
Wikipedia is processed first and deterministically owns identical cross-source
text; split assignment still uses the shared text digest. Full near-duplicate
similarity clustering remains deferred and is an explicit production blocker.

Every chunk from a document inherits that document's split. Every document in a
canonical/duplicate cluster shares one split. FineWeb and Wikipedia validation
remain separate even when a weighted combined metric is reported.

### Split ownership

- FineWeb-Edu: independent train and validation artifacts.
- Wikipedia: independent train and validation artifacts.
- Combined training: deterministic interleaving of training documents only.
- Combined validation: a token-weighted reported value plus both source losses;
  the combined value never replaces the source values.
- UltraChat: `train_sft` is the only future preparation input; `test_sft` stays
  untouched until final evaluation.
- OASST1: reconstruct paths by root `message_tree_id`; official validation is
  final evaluation. Roots, messages, prompts, paths, and duplicate clusters may
  not cross splits.

## Tokenizer decision

The current production candidate is byte-level BPE with an 8,192-entry model
vocabulary. It is not frozen until the bake-off report is approved.

Stable reserved IDs are:

| ID | Token |
| ---: | --- |
| 0 | `<pad>` |
| 1 | `<bos>` |
| 2 | `<eos>` |
| 3 | `<unk>` |
| 4 | `<|system|>` |
| 5 | `<|user|>` |
| 6 | `<|assistant|>` |
| 7 | `<|end|>` |

`scripts/compare_tokenizers.py` trains 8,192 and 16,384 candidates from the same
stable, source-stratified cleaned sample. It reports bytes/token,
characters/token, tokens/document, total tokens, unknowns, round-trip failures,
encoding/decoding speed, per-source compression, sequence inflation, projected
corpus tokens/runtime, checksums, and embedding parameter impact. It never edits
`configs/1b.yaml` or selects a winner automatically.

## Packing contract

The initial base policy is versioned by `production_packing_policy()`:

- concatenate unique documents efficiently with one `<eos>` after each;
- allow multiple documents in a 2,048-token sequence;
- predict the next document's first token from the preceding EOS;
- allow causal attention across document boundaries;
- do not reset position IDs at document boundaries;
- use no padding for complete base sequences and therefore mask no positions;
- discard and count the one final incomplete sequence per split;
- retain every document's token start, end, EOS position, source, and ID;
- use identical semantics for training and validation.

This simple policy avoids custom block-diagonal attention while preserving
auditable boundaries. If padding is introduced later, its token ID is 0 and all
padding labels must be `-100`.

## Source mixture and token budget

The production percentage is deliberately unset. It must be chosen from the
post-cleaning, selected-tokenizer accounting report—not raw bytes or raw rows.

`src/data/mixture.py` orders documents by a seeded stable hash and repeatedly
selects the source furthest below its requested token share. It never repeats or
oversamples documents. It reports requested/achieved shares, documents, source
tokens, last-document truncation, exhausted sources, and repetition count.

The final token budget is also unset. The full corpus is not automatically one
training epoch. The production-shape benchmark estimates 1B, 3B, 6B, and 10B
token runtimes from sustained post-warmup throughput. The operator records the
chosen budget in the production manifest after reviewing wall time and storage.

## Checkpoint and resume contract

Existing checkpoints contain model, optimizer, scheduler, GradScaler, global
microstep/optimizer step/tokens, epoch and batch cursor, best validation, Python,
NumPy, CPU RNG, and CUDA RNG states. Production checkpoints now also record:

- microstep within gradient accumulation;
- data-manifest and configuration checksums;
- Git revision and run UUID;
- epoch/batch data cursor and mixture-sampler state;
- Python/PyTorch/CUDA/OS/GPU environment information;
- tokenizer checksum.

Writes use a temporary file, atomic replacement, SHA-256 sidecar, verification,
and an atomic `.complete.json` marker containing size, checksum, step, and write
duration. Resume rejects config or data-manifest mismatch. The exact CPU resume
test compares uninterrupted and interrupted data position, loss trajectory,
parameters, and optimizer state bit for bit. Bitwise identity across different
GPU/driver/PyTorch environments is not promised.

Limitation: checkpoint payload persistence is atomic at file level, not yet an
atomically renamed multi-file directory bundle. The completion marker prevents
an unverified file being treated as complete, but directory-bundle promotion is
deferred before declaring the recovery design final.

## Full production-shape benchmark

`scripts/benchmark_production_shape.py` refuses non-CUDA, non-BF16, non-2,048
context, or a parameter count other than 921,773,568. It runs the actual memmap
loader, microbatch 1, accumulation 32, `adamw8bit`, activation checkpointing,
validation, checkpoint save, and fixed prompt generation.

It excludes warmup measurements and reports median sustained tokens/s,
sequences/s, step duration, forward/backward duration, optimizer duration, peak
allocated/reserved memory, required free headroom, checkpoint-write duration,
validation/generation overhead, token-budget ETA, and checkpoint storage.

The benchmark fails on command failure, probable OOM, non-finite loss, corrupt
output, wrong shape, or less than the requested 1.5 GB VRAM headroom. It creates
a benchmark-only schedule copy; it does not change production configuration.

## Numeric quality gates

Proposed thresholds are in `configs/base_acceptance_thresholds.yaml`. They must
be reviewed and frozen by checksum before production outputs are inspected.

### Tiny overfit

- fixture and token count recorded;
- final loss at most 0.05;
- target-token accuracy at least 0.99;
- at most 500 optimizer steps;
- deterministic completion recorded;
- interrupted/resumed checkpoint completes successfully.

### Smoke training

- finite losses and gradients throughout;
- zero invalid IDs, malformed labels, or split overlap;
- at least 5% loss improvement across a predeclared 1,048,576-token interval;
- exact checkpoint save/resume test passes;
- deterministic greedy output;
- zero unexplained skipped optimizer steps or corrupt checkpoints.

### Base checkpoint

Report FineWeb loss, Wikipedia loss, weighted loss, completion length,
premature-EOS rate, repeated 4-gram rate, longest repeated span, distinct-2,
distinct-3, prompt-copy rate, Unicode/token errors, and maximum-length rate.
Human reviewers add blind 1–5 grammar, topic-continuity, and prompt-relevance
scores. Greedy decoding is the deterministic health check; one fixed sampling
configuration is used only for readability. Sampling cannot be tuned per
checkpoint.

## Conversational preparation and SFT

`src/sft.py` defines the initial `chat-v1` format. System/user text and all role
tokens are masked. Assistant text and its ending `<|end|>` are supervised.
Conversation examples must end in a complete assistant turn; overlength examples
are discarded rather than training on a half-truncated assistant response.
Multiple conversations are not packed together in the initial policy.

Future preparation must remove deleted/broken messages, malformed paths, spam,
clear PII, exact duplicates, defective answers, and examples below documented
rank/review/quality cutoffs. It must retain good refusals and boundary-setting
answers, genuine multi-turn paths, and rejection counts. Samples immediately on
both sides of every cutoff require manual audit.

The immutable base will produce two controlled variants: pure SFT, and SFT with
a small recorded percentage of base-language replay. Selection uses chat,
multi-turn memory, correction, formatting, stopping, repetition, both source
validation losses, and base prompts. Replay is not assumed beneficial.

## Operator commands

These commands produce ignored artifacts. Read each `--help` before use.

### Repository and parameter gate

```bash
source .venv/bin/activate
python -m pytest
python tests/test_model.py
python - <<'PY'
from src.config import load_config
from src.model import LanguageModel, count_parameters
config = load_config("configs/1b.yaml")
print(count_parameters(LanguageModel(config.model)))
PY
```

Expected parameter count: `921773568`.

### Manifest/schema inspection

```bash
python scripts/download_fineweb_edu.py --help
python scripts/download_language_corpora.py --help
python scripts/prepare_fineweb_edu.py --help
```

Do not invoke the download commands merely to validate existing data. Verify the
local manifests and their recorded checksums during the preparation integration.

### Tokenizer bake-off

```bash
python -m scripts.prepare_base_corpus \
  --fineweb-root data/raw/fineweb-edu-10bt/sample/10BT \
  --wikipedia-root data/raw/wikipedia/20231101.en \
  --output-dir data/processed/base-v1 \
  --validation-ratio 0.005 \
  --seed 42

python -m scripts.compare_tokenizers \
  --config configs/tokenizer_bakeoff.yaml \
  --fineweb-jsonl data/processed/base/fineweb-clean.jsonl \
  --wikipedia-jsonl data/processed/base/wikipedia-clean.jsonl \
  --output-dir logs/tokenizer-bakeoff-v1
```

### Token build and inspection

```bash
python -m scripts.train_tokenizer --help
python -m scripts.tokenize_dataset --help
python -m scripts.inspect_token_data --help
python -m scripts.benchmark_token_data --help
```

### Tiny-overfit gate

```bash
python -m scripts.run_tiny_overfit \
  --config configs/tiny_overfit.yaml \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --target-loss 0.05 \
  --target-accuracy 0.99 \
  --run-name codexa-base-tiny-overfit-v1
```

### Production-shape benchmark

```bash
python -m scripts.benchmark_production_shape \
  --config configs/1b.yaml \
  --train-token-file data/tokenized/base-v1/train.bin \
  --validation-token-file data/tokenized/base-v1/validation.bin \
  --token-manifest data/tokenized/base-v1/token_data_manifest.json \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --steps 12 \
  --warmup-steps 2 \
  --output-dir logs/production-shape-v1
```

Twelve steps are a functional minimum, not automatically a stable benchmark.
Increase sustained steps if throughput remains variable.

### Preflight and production launch shape

```bash
python -m scripts.preflight_full_run \
  --config configs/1b.yaml \
  --token-manifest data/tokenized/base-v1/token_data_manifest.json \
  --train-token-file data/tokenized/base-v1/train.bin \
  --validation-token-file data/tokenized/base-v1/validation.bin \
  --checkpoint-dir checkpoints \
  --output logs/preflight-base-v1.json

python -m scripts.train --help
```

The second command is intentionally `--help`. Do not launch production until
the manifest contains a selected tokenizer, mixture, token budget, frozen
threshold checksum, and evidence that stages 1–11 passed.

### Source-specific checkpoint evaluation

```bash
python -m scripts.evaluate_checkpoint \
  --checkpoint checkpoints/codexa-base-v1/milestones/step_XXXXXXXXX.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --prompts configs/evaluation_prompts.json \
  --source-validation fineweb_edu data/tokenized/fineweb-v1/validation.bin data/tokenized/fineweb-v1/token_data_manifest.json \
  --source-validation wikipedia data/tokenized/wikipedia-v1/validation.bin data/tokenized/wikipedia-v1/token_data_manifest.json \
  --device cuda \
  --output logs/evaluations/checkpoint-XXXXXXXXX-greedy.json
```

## Readiness statement

- Ready for repository-level deterministic unit validation: **yes**.
- Ready for full exact-deduplicated data preparation: **yes**. Full
  near-duplicate similarity clustering remains a separate unresolved production
  decision.
- Ready for the production-shape smoke: **no**—the approved mixed token artifact
  and tokenizer decision do not exist yet.
- Ready for the full base-training launch: **no**—do not launch until every gate
  through stage 11 passes and the production manifest is complete.
