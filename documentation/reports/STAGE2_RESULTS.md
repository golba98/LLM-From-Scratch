# Stage 2 completion report — 2026-10-08

The main milestone is complete: native Codexa chat optionally retrieves scoped conversation history with frozen EmbeddingGemma and supplies it to the existing generative decoder. The 200-step pilot completed, was evaluated, and was exported separately. **The pilot did not demonstrate better conversational quality and is not promoted.** Nothing was committed, pushed or cleaned up; existing checkpoints, exports, datasets, classifier functionality and unrelated Git changes were preserved.

## Implementation

Memory is off by default. Explicit ephemeral/persistent modes use a local specialist subprocess, SQLite original turns and exact cosine vectors, normalized 768-dimensional SearchQuery/Document embeddings, CPU FP32 or CUDA BF16, bounded caching, explicit user/conversation scopes, chronological excerpts, deduplication, deletion, safe parallel index revisions and encoder-failure fallback. Native prompts reserve response space inside 2,048 tokens, cap memory overhead at 256 tokens, protect role boundaries and keep history transactional. The default model remains the original repair checkpoint; memory never replaces Codexa generation.

No dependency packages were upgraded. Existing environments remain separate: generative PyTorch 2.13 / Transformers 4.57.6; specialist PyTorch 2.13 / Transformers 5.19 / Sentence Transformers 6.1. SQLite is stdlib; NumPy was already installed. The pilot used existing bitsandbytes 0.50.0.

## Files created

- `src/memory/__init__.py`, `client.py`, `context.py`, `service.py`, `store.py`
- `src/data/general_chat.py`, `src/conversational_training.py`
- `scripts/inference/memory_worker.py`, `prepare_general_chat.py`, `evaluate_memory.py`, `evaluate_chat_memory.py`, `train_chat_stage2.py`, `export_chat_native.py`, `probe_embedding_states.py`, `summarize_chat_memory.py`, `evaluate_chat_decoding.py`
- `tests/test_memory.py`, `test_native_memory.py`, `test_general_chat.py`, `tests/fixtures/memory_retrieval.json`
- `documentation/reference/CONVERSATIONAL_MEMORY.md`, `ENCODER_CONDITIONING.md`, and this report

## Files modified

`.gitignore`, `README.md`, `src/specialist/encoder.py`, `src/native_chat.py`, `src/training.py`, `scripts/inference/chat_native.py`, `scripts/inference/native_chat_bridge.py`, `scripts/train_conversational_sft.py`, `scripts/legacy/validate_run_manifest.py`, `tests/test_config.py`, `tests/test_training.py`, `documentation/reference/SPECIALIST.md`, and both required session/progress journals. This inventory describes Stage 2 edits, not the repository's unrelated pre-existing changes.

Fixes include incorrect history trimming without a system prompt, rollback after failed generation, reserved-token injection into role boundaries, accidental base replay at ratio zero, stale parameter counts/configuration fixtures, and tokenizer compatibility checks. The new SFT launcher uses separate owned artifact namespaces and recoverable atomic checkpoints. Future launches additionally perform two-update steady-state preflight and reject another active CUDA compute process; the completed run used a one-update preflight and empirically stayed below the gate through all 200 steps.

## Retrieval evaluation

The 60-query evaluation fixture is synthetic and explicitly excluded from training. Thirty development queries selected threshold 0.68; thirty different-family test queries produced:

| Metric | Measured result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.50 / 0.90 / 1.00 |
| Precision@1 / @3 / @5 | 1.00 / 0.60 / 0.40 |
| MRR | 1.00 |
| No-context false positives | 0 / 5 |
| CPU embedding median / p95 | 29.56 / 31.26 ms |
| Search median | 0.264 ms |
| Vector payload | 491,520 bytes |
| Raw excerpt token median | 18.5 |

Precision uses fixed K; recall counts both relevant speaker records. The small negative sample does not establish a real-world false-positive guarantee. Actual chat overhead reached 103 tokens, including wrappers, below its 256-token cap. Relevant excerpts were supplied on all eight distant-reference/correction scenarios, even when generation ignored them.

Report: `logs/memory/retrieval-stage2-cpu-v1/report.json`.

## Three-system generation comparison

All systems use the same 24 authored scenarios, seed 42, temperature 0.7, top-p 0.9, 64-token cap and KV caching. Memory inputs use stable source labels. Each scenario has equivalent history/settings; the new pilot checkpoint is the only generative weight change in System 3.

| Automated diagnostic | Original, no memory | Original, memory | Pilot, memory |
| --- | ---: | ---: | ---: |
| Expected whole value present, distant/correction questions | 0 / 8 | 2 / 8 | 0 / 8 |
| Expected names attributed to the assistant itself | 0 | 2 | 0 |
| Exact READY instruction | 0 / 1 | 0 / 1 | 0 / 1 |
| END termination | 13 / 24 | 14 / 24 | 10 / 24 |
| Mean repeated-trigram fraction | 0.153 | 0.122 | 0.185 |
| Median full reply latency | 0.428 s | 0.500 s | 0.528 s |

Literal matches are not semantic accuracy: both original-with-memory name matches said **my name**, not the user's name. There is no demonstrated correct distant fact retention in this small sample under a strict speaker interpretation. Ordinary conversation, instructions and knowledge also remain weak. These qualitative conclusions are agent assessments, not human ratings. No human quality judgments were collected.

Example real outputs:

- Query: `What name did I tell you earlier?`, stored user name Leon.
  - Original without memory: a repetitive generic list of names, without Leon.
  - Original with memory: `My name is Leon.` — recalled value, wrong speaker attribution.
  - Pilot with memory: a repetitive explanation about remembering dates, without Leon.
- Query: `What is the capital of Japan?`
  - Original: `Tokyo is the capital of Japan.`
  - Pilot: `The capital of Japan is in Tokyo.`
- Query: `Reply with only the word READY.`
  - Original: `READY, READY is for a variety of word problems...` (continued beyond the requested word).
  - Pilot: `Certainly! But what about "Error"?`

Ellipses mark shortened quotations; full real outputs are retained in JSONL. Final reports: `logs/memory/chat-original-final-v2/report.json`, `logs/memory/chat-pilot-final-v1/report.json`; aggregate: `logs/memory/system-comparison-final-v2.json`. Repeated original and pilot evaluations reproduced every response exactly.

The source/root and duplicate-group audit found zero split violations. Twenty-one exact diagnostic-prompt occurrences were found in the full corpus, involving Hello, Thanks and the Japan-capital question. Those cases cannot establish unseen generalization. Other semantic overlap remains possible. The separate retrieval fixture was never included in SFT.

## Data and training

Existing local UltraChat/OpenAssistant derivatives yielded 182,892 train, 10,135 validation and 9,874 test conversations in `data/processed/general-chat-sft-v2/`. Source/root identities, complete multi-turn messages, normalized duplicate groups and historical exposure remain recorded. Repeated repair anchors and overlapping balanced versions were not concatenated. The preparation rejected 26,497 role/overlength records, 42 normalized duplicates, 56 severe repetitions and 5 control-marker records. Structural/duplicate control does not certify factual quality or semantic paraphrase isolation.

The source was the checksum- and tokenizer-fingerprint-verified base checkpoint. The pilot selected 20,000 seeded source-balanced conversations and trained 6,400 microbatches: microbatch 1, accumulation 32, BF16 autocast, gradient checkpointing, 8-bit AdamW, peak LR 1e-5, 20-step warmup, weight decay 0.1, clipping 1.0, zero replay. Targets contain only shifted assistant content and END. A visible Kitty viewer at 100x22 attached before preflight and training.

| Pilot measurement | Result |
| --- | ---: |
| Optimizer steps | 200 |
| Supervised assistant tokens | 4,029,761 |
| Initial validation loss | 2.652032 |
| Final validation loss | 2.379369 |
| Held-out test loss | 2.270733 |
| Final step training loss | 2.316029 |
| Pilot including preflight/automatic chat evaluation | 1,861.85 s (31.03 min) |
| Peak training reserved VRAM | 13,103,005,696 bytes (12.20 GiB) |

The held-out losses are from up to 32 fixed source-balanced validation/test examples and are token-weighted; they are not full-corpus scores. Old repair losses use a different dataset and must not be numerically ranked against these losses. Lower NLL did not translate into demonstrated conversational improvement. No additional base run or encoder-decoder training was launched.

Checkpoint: `checkpoints/codexa-900m-chat-stage2-pilot-v1/latest.pt`; full logs/report: `logs/codexa-900m-chat-stage2-pilot-v1/`. Export: `exports/codexa-900m-chat-stage2-pilot-v1/model.pt`, with tokenizer and `export_report.json`. Checksum verification passed; source/export smoke token IDs were identical. The export is marked experimental-not-promoted. All checkpoints, including previous/best recovery artifacts, remain retained.

## Hardware and decoding

Preliminary simultaneous-model measurement reached 7,647 MiB total GPU usage including desktop/driver memory; CUDA encoder peak allocator use was 737 MiB allocated / 796 MiB reserved. CPU encoder worker peak RSS was approximately 2.44 GiB. Both models fit in measured inference workloads. CPU is the isolation default, CUDA BF16 optional; no encoder was resident on CUDA during SFT. Final comparison GPU totals varied with desktop activity, so use recorded workload-specific allocator/global figures rather than extrapolating to arbitrary context lengths.

A real CPU client cache smoke measured 50.75 ms for its first embedding request, then 0.058 / 0.037 ms for cached requests, with identical vectors and normal worker exit code 0. Document vectors are also reused from the persistent index. Report: `logs/memory/worker-cache-smoke-v1.json`.

An eight-prompt decoding ablation measured repeated-trigram fractions of 0.123 (sampling), 0.138 (greedy), and 0.0022 (penalty 1.15 plus blocked trigrams). Loop controls still failed READY and produced other malformed text. They suppress a symptom without supplying knowledge; defaults remain unchanged. Report: `logs/memory/decoding-ablation-v1.json`.

## Tests and limitations

- Generative environment: **114 passed, 5 dependency-related specialist skips, 0 failed**; 8 existing pytest return-value warnings.
- Specialist plus memory/integration tests: **58 passed, 0 failed**, including all original **39 specialist tests**.
- Real frozen encoder, persistence/search, native chat comparisons, token-state probe, cache/shutdown, checksum and native/export equivalence checks passed.
- `git diff --check` passed. Original base and repair checksums were reverified unchanged.

The prototype uses exact search, explicit trusted local identities and bounded excerpts. It has no external knowledge database, automatic factual reconciliation or semantic paraphrase deduplication. Index revisions consume additional storage. Automatic resume is not exposed by the bounded launcher despite retained recovery state. The generator's instruction following, factual reliability, speaker understanding and end-of-turn behavior remain inadequate for claiming capable general-purpose conversation.

## Encoder-decoder feasibility and next stage

Real token-level states are accessible: 512-wide intermediate tensors and projected 768-wide final states. Architecture B needs trainable projection/key/value cross-attention, separate tokenizer handling, masked assistant next-token training and measured activation memory. Architecture C needs a separately trained generative decoder and substantially more migration/compute. Neither was trained; 16 GB feasibility for B/C remains an estimate requiring its own preflight. See `documentation/reference/ENCODER_CONDITIONING.md` and its primary-source research links.

Recommended next stage: keep retrieval A opt-in; obtain broader independently labelled memory and human conversation evaluations; derive provenance-preserving memory-format supervision from real multi-turn examples; verify a tiny overfit diagnostic with correct speaker attribution and END before longer SFT. Investigate the base model's low token exposure with controlled language/behavior learning curves, rather than assuming more SFT or cross-attention solves it. Do not promote the current pilot or launch unrestricted pretraining.

## Exact commands

```bash
# Working conversational system using the existing checkpoint and opt-in memory.
.venv/bin/python scripts/inference/chat_native.py \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --memory ephemeral

# Experimental pilot export, explicitly not the recommended checkpoint.
.venv/bin/python scripts/inference/chat_native.py \
  --checkpoint exports/codexa-900m-chat-stage2-pilot-v1/model.pt \
  --tokenizer exports/codexa-900m-chat-stage2-pilot-v1/tokenizer.json \
  --memory ephemeral

# A separately named future bounded pilot; never silently extends this experiment.
.venv/bin/python scripts/legacy/train_chat_stage2.py \
  --run-name codexa-900m-chat-stage2-pilot-new-run --max-steps 200 --minutes 60
```

For persistence, source selection, deletion, bridge operations, retrieval evaluation and data preparation commands, see `documentation/reference/CONVERSATIONAL_MEMORY.md`.

Inclusive acceptance time measured here was 58.25 minutes from pilot launch, within the agreed 60-minute ceiling. Acceptance metadata: `logs/memory/stage2-acceptance.json`.
