# Codexa v1 — Gated Base-to-Conversation Rebuild

This is the authoritative stage order. A checked implementation box means the
repository contains an executable mechanism; it does not mean a production
artifact has passed. Evidence from every exit gate is stored in ignored output
directories and referenced by checksum from the production run manifest.

The architecture remains `configs/1b.yaml`: 921,773,568 parameters, 2,048-token
context, and an initially configured 8,192-entry vocabulary. The production
model starts from random weights on one RTX 4080. No stage below authorizes the
long training run by itself.

## Capability boundary

> Codexa v1 is initially a general-language research model. FineWeb-Edu,
> Wikipedia, UltraChat and OASST1 may contain incidental programming material,
> but this training plan does not establish reliable code-generation
> capability.

FineWeb-Edu and Wikipedia are the only base-pretraining sources. UltraChat and
OASST1 are isolated future SFT sources. No code corpus is approved here.

## 1. Freeze architecture and initial constraints

- [x] Keep the existing decoder-only architecture and exact parameter count.
- [x] Keep context length 2,048, BF16, microbatch 1, gradient accumulation 32,
  activation checkpointing, and intended `adamw8bit` production path.
- [x] Define `schemas/production_run_manifest.schema.json`.
- [ ] Fill and freeze a production manifest; null decisions are launch blockers.

Gate: config checksum, parameter count, run UUID, Git revision, optimizer,
token milestones, initialization, environment, and every input checksum are
recorded before weights are initialized.

## 2. Verify downloaded manifests and schemas

- [x] Download and manifest ten pinned FineWeb-Edu shards.
- [x] Download and manifest all 41 English Wikipedia shards.
- [x] Download and manifest UltraChat SFT and OASST1 without flattening them.
- [ ] Re-run manifest checks immediately before preparation and save the report.

Gate: no missing file, size mismatch, checksum mismatch, unexpected schema, or
unapproved revision. Source roles match `configs/data_preparation.yaml`.

## 3. Clean, deduplicate, and assign stable splits

- [x] Implement order-independent normalization, stable identity, exact
  within-source and cross-source deduplication, rejection accounting, grouped
  seeded splits, and leakage assertions in `src/data/governance.py`.
- [x] Record a staged word-shingle fingerprint for near-duplicate audits.
- [x] Connect the governed semantics to the full SQLite-backed streaming Parquet
  preparation CLI in `scripts/prepare_base_corpus.py`.
- [ ] Run scalable near-duplicate clustering, or explicitly approve exact-only
  deduplication as a recorded production limitation.

Required order: normalize; reject invalid; exact-deduplicate inside FineWeb;
exact-deduplicate inside Wikipedia; exact-deduplicate across both; near-duplicate
audit; assign complete clusters to splits; tokenize; pack.

Gate: zero stable-document or duplicate-cluster overlap; source/reason counters
sum exactly; results do not change with input order, shard order, or worker
count. Wikipedia-derived FineWeb copies cannot cross splits.

## 4. Benchmark tokenizer candidates

- [x] Reserve stable IDs 4–7 for `<|system|>`, `<|user|>`, `<|assistant|>`, and
  `<|end|>` while retaining `<pad>=0`, `<bos>=1`, `<eos>=2`, `<unk>=3`.
- [x] Add a deterministic stratified 8,192-versus-16,384 bake-off.
- [ ] Run it on the cleaned two-source sample and review the evidence.
- [ ] Record the explicit production selection; do not infer it from defaults.

Gate: per-source compression, unknowns, round trip, speed, sequence inflation,
projected tokens/runtime, tokenizer checksum, and embedding parameter impact are
reported. The 16,384 candidate adds 12,582,912 tied-embedding parameters versus
8,192; an untied design would add twice that, but this architecture is tied.

## 5. Perform final token accounting

- [x] Define `schemas/token_accounting_report.schema.json`.
- [ ] Tokenize all accepted documents for accounting without building the final
  mixed stream.
- [ ] Report source-specific document-length percentiles, stored/content tokens,
  truncation, and trailing discard.

Gate: FineWeb-Edu and Wikipedia counts reconcile from raw rows through cleaned
documents to tokens. The report checksum is frozen.

## 6. Select and record the source mixture

- [x] Implement deterministic token-deficit interleaving without oversampling.
- [ ] Select percentages from post-cleaning token measurements and compute the
  feasible token budget; `requested_token_percentages: null` is a blocker.

Gate: requested and achieved percentages, documents, tokens, truncation,
discard, repetition count, seed, exhaustion, and sampler state are recorded.
The production stream is not a FineWeb block followed by a Wikipedia block.

## 7. Build deterministic token artifacts

- [x] Insert exactly one EOS after every document and retain boundary offsets.
- [x] Pack multiple documents efficiently; causal attention crosses boundaries,
  positions do not reset, the token after EOS is predicted, and no padding is
  used in complete sequences.
- [x] Discard the final incomplete sequence and report its tokens.
- [ ] Build separate FineWeb train/validation and Wikipedia train/validation
  artifacts plus the deterministic mixed training stream.

Gate: repeated builds have identical checksums; no validation document enters
training; boundary tests inspect IDs and labels; source validation remains
independent.

## 8. Run tiny-overfit validation

- [x] Report fixture token count, initial/final loss, target-token accuracy,
  steps, generated text, checkpoint interruption, and resume.
- [ ] Run against the frozen tokenizer and meet the predeclared thresholds.

Gate: final loss at most 0.05 and target-token accuracy at least 0.99 within 500
steps, unless `configs/base_acceptance_thresholds.yaml` is versioned and frozen
before the result is inspected.

## 9. Run full-context production-shape smoke validation

- [x] Add `scripts/benchmark_production_shape.py` using the real 1B model,
  context 2,048, memmap loader, microbatch 1, accumulation 32, BF16,
  `adamw8bit`, activation checkpointing, validation, checkpointing, and fixed
  generation.
- [ ] Run it on the RTX 4080 with at least 1.5 GB reserved-memory headroom.

Gate: no OOM, non-finite loss/gradient, invalid IDs/labels, unexplained skipped
step, corrupt checkpoint, or nondeterministic greedy output. A short-context
smoke result cannot satisfy this gate.

## 10. Measure sustained throughput and choose token budget

- [x] Record forward/backward, optimizer, sustained step, validation,
  generation, checkpoint-write, allocated/reserved VRAM, and wall durations.
- [x] Estimate runtime for 1B, 3B, 6B, and 10B processed-token candidates and
  storage for recovery plus milestones.
- [ ] Run enough post-warmup steps to obtain stable median throughput.
- [ ] Select and record a production token budget that fits the operator's time
  and storage constraints.

Gate: the budget is an explicit manifest decision. Corpus size does not imply
that one full pass is affordable.

## 11. Freeze numeric acceptance thresholds

- [x] Add proposed thresholds in `configs/base_acceptance_thresholds.yaml`.
- [x] Version the fixed base prompt suite and machine-readable evaluation.
- [ ] Review and freeze thresholds before inspecting production checkpoints.

Gate: threshold checksum is in the production manifest. Later threshold changes
require a versioned decision and cannot retroactively rescue a weak checkpoint.

## 12. Train the 1B base from random weights

- [ ] Start only after stages 1–11 pass.
- [ ] Reject old run directories, mismatched tokenizers/manifests/configs, and
  unrelated checkpoints.
- [ ] Schedule warmup, decay, evaluation, generation, and checkpoints by
  processed tokens wherever practical.

Gate: the selected budget completes without integrity failure. FineWeb-Edu and
Wikipedia validation losses remain separately visible.

## 13. Compare and preserve milestone checkpoints

- [x] Produce deterministic greedy and fixed-sampling evaluation reports.
- [x] Report mechanical generation metrics and weighted/source validation.
- [ ] Add human 1–5 grammar, continuity, and relevance scores blind to step.

Gate: milestone comparison uses frozen thresholds, not final-step preference or
sampling changes.

## 14. Select an immutable base checkpoint

- [ ] Copy the accepted checkpoint, tokenizer, manifests, environment, reports,
  and checksums into a versioned immutable bundle.
- [ ] Prohibit SFT from overwriting this bundle.

Gate: recovery and checksum verification pass from the preserved copy.

## 15. Build tree-safe conversational datasets

- [x] Implement one `chat-v1` serialization contract with reserved role tokens,
  assistant-only targets, and supervised assistant `<|end|>`.
- [ ] Use UltraChat `train_sft` only for preparation; preserve `test_sft` as
  untouched final evaluation.
- [ ] Reconstruct OASST1 paths by `message_tree_id`; preserve official
  validation and split no messages or overlapping paths independently.
- [ ] Filter with reason counters: structural defects, deleted messages, spam,
  clear PII, duplicates, defective answers, and documented quality thresholds.
  Retain high-quality refusals and audit samples around every cutoff.

Gate: no root, tree, prompt identity, path, or duplicate cluster crosses a
split. No incomplete assistant response is supervised.

## 16. Fine-tune controlled conversational variants

- [ ] Fine-tune copies of the immutable base using (A) pure chat SFT and (B)
  chat SFT plus a small recorded base-language replay percentage.
- [ ] Never assume replay helps; keep all other settings controlled.

Gate: both variants finish with manifests and supervised-token accounting.

## 17. Evaluate chat and base-capability regression

- [ ] Compare instruction following, multi-turn recall, correction handling,
  formatting, stopping, repetition, FineWeb validation, Wikipedia validation,
  and base fixed prompts.
- [ ] Evaluate UltraChat `test_sft` and OASST1 official validation only at the
  final selection boundary.

Gate: selected SFT passes chat thresholds without unacceptable base regression.

## 18. Expose only an accepted conversational checkpoint

- [ ] Add serving only after stage 17 passes.
- [ ] Keep rejected and experimental checkpoints unavailable by default.

Gate: the served checkpoint checksum exactly matches the accepted SFT report.

## Current readiness

- Ready to finish full-corpus data-preparation integration: **yes**.
- Ready to execute a production-shape smoke test: **not yet**; production mixed
  token artifacts and the tokenizer decision are missing.
- Ready for the full base-training launch: **no**; stages 1–11 have not passed.

Detailed data semantics, commands, and limitations are in
`TRAINING_DATA_PLAN.md`.
