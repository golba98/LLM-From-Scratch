# CODEXA_900M_CHAT_10H_READINESS

Prepared 10 October 2026 from current local execution. Decision: **BLOCKED**.

## A. Executive Summary

Implemented a modular conversational successor, lazy audited chat loading, deterministic full-pool sampling,
assistant-token-weighted loss, transactional native recovery checkpoints, original-deadline accounting,
an independent production watchdog, signal handling, a detached visible Kitty viewer, structured logs,
reproducible development/protected acceptance evaluation, gated operator commands and this generated handoff.
No ten-hour production job or model promotion occurred. All five existing candidates fail minimum instruction
and arithmetic compliance. Lower loss during the corrective pilot did not establish a better assistant.

Failed/unproven mandatory gates: **initialization_quality, behavioral_pilot, held_out_exposure_provenance, strict_live_memory_headroom**. Preparation is implemented and tested; approval is not being
requested for a blocked launch. Resolve these gates before a separate production approval.

## B. Architecture and Initialization

934,356,480 trainable parameters; decoder-only, 24 layers, hidden 1,536, 24 heads × 64, FFN 6,144,
context 2,048, vocabulary 16,384, learned positional embeddings, RMSNorm, SwiGLU, causal SDPA, tied token/output
embeddings. Model dimensions, positional scheme, state keys and tokenizer are unchanged. Strict restores,
tied Parameter identity and finite forward/loss checks were performed on the main model.

**No production initialization is accepted.** Provisional benchmark/corrective weights initialization:
`/home/k9-vortex/Development/3-Python/31-LLM Source/checkpoints/codexa-900m-sft-repair-v1/latest.pt`.

SHA256: `c55f4edd1477a8ffefd03ce63d033a291eb261f6047b129baa47e10756899a4f`.
Repair ranked highest on the fixed literal task aggregate, despite worse repetition than balanced/v2.
Its weights-only file starts a new optimizer/scheduler phase. This is not an exact historical optimizer resume.
Base and Stage 2 have full native state; balanced/v2/repair do not. All five sidecar digests were verified and
actual strict loads performed. Missing embedded tokenizer hashes in older weights-only files are disclosed:
compatibility is supported by architecture, verified tokenizer identity and local lineage, not an embedded binding.

Tokenizer: `/home/k9-vortex/Development/3-Python/31-LLM Source/checkpoints/tokenizer-base-v1/tokenizer.json`; SHA256 `a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e`.
IDs pad/BOS/EOS/UNK/system/user/assistant/end = 0/1/2/3/4/5/6/7. END (7) is supervised assistant termination;
EOS (2) remains an alternate inference stop. Existing chat serialization does not insert BOS/EOS. One causal
target shift is applied by the dataset; assistant content and END are supervised, role/user/system/padding ignored.

## C. Dataset Strategy

Physical asset root remains `/home/k9-vortex/Development/3-Python/31-LLM Source`.
Dataset: `/home/k9-vortex/Development/3-Python/31-LLM Source/data/processed/general-chat-sft-v2`; manifest SHA256 `3b1741ae312316172a1cb0f8a6457cebf2603fcbf1f7ba8896a56fcc4ac60f0d`.
All 202,901 JSONL conversations were read-only audited against manifest digests, normalized content hashes,
reserialized boundaries, token lengths and shifted response masks. No retokenized corpus or duplicate store was created.

| Split | Conversations | Context tokens | Assistant tokens | Potential historical exposure | Group-filtered possibly clean |
|---|---:|---:|---:|---:|---:|
| train | 182,892 | 190,640,354 | 147,584,268 | 181,120 | 1759 |
| validation | 10,135 | 10,569,914 | 8,178,234 | 10,039 | 96 |
| test | 9,874 | 10,277,831 | 7,974,145 | 9,776 | 98 |

Training sources: OASST1 27,034 conversations / 10,250,006 assistant tokens; UltraChat 155,858 /
137,334,262. Assistant totals count repeated shared turns/prefixes across distinct paths, not unique linguistic tokens.
There are 163,889 training duplicate/root groups. No train/validation/test root, group or exact-content overlap was found.
Train SHA256 `15d29ea23ddb57c70067643e9feb641f03b9cd42bdecace7cb710eab784d61e0`; validation SHA256 `30f0fcde04cf4f9b34f4e42f5c7941036a8509e81a8d9e1ad7cbc49fd823b074`.
Test SHA256 `c27f379e49397f5a581351203d03283d0febfcbffb951cb134a9b7771f33e50b`.

Use seeded no-replacement ordering across the full 182,892 training pool, with local length bucketing; natural
source proportions are the provisional choice, not a proven optimum. Balanced interleaving is supported but was
not selected through an accepted mixture experiment. At most one epoch; partial final accumulation is discarded.
No base replay is enabled. Lazy serialization, 64 MiB cache, right dynamic padding, workers 0 and pinned transfers
avoid storing the entire padded pool in host RAM. Complete-conversation truncation is rejected, empty targets rejected,
padding labels -100, causal right padding validated. Conversations are not packed together.

Validation uses 96 records from 92 groups whose entire group lacks a positive historical exposure flag.
The pilot used a fixed first 32 of those; benchmark probes used first 8. These losses are different populations.
Common 96-record validation appears separately below. The 98 similarly flagged test records in 86 groups remain
unused in training and preparation evaluation. Flags cannot prove access history; earlier SFT validation exposure
and previous Stage 2 test use prevent a claim of clean historical holdout. Only 1,759 training records remain
possibly clean after group propagation (1,772 individual flags were false).

Five exact normalized training matches concern one development case, `development-002` (Tokyo).
The archived 63-case exact-match-excluded comparison preserves the strongest provisional candidate and all failures;
it is not semantic decontamination. No exact protected prompt match was found; paraphrase exposure is unproven.
The protected suite was not generated against during preparation. Missing original base validation binary/index
was not replaced by continuation validation or these replay windows. **Proven generalization is blocked.**

## D. Final Training Configuration

Prepared configuration: `configs/chat-10h-v1.yaml`; resolved proposed specification:
`/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/launch-manifest.json`. These values are prepared, not accepted production hyperparameters.

```json
{
  "purpose": "production",
  "model": {
    "vocab_size": 16384,
    "context_length": 2048,
    "num_layers": 24,
    "hidden_size": 1536,
    "num_heads": 24,
    "intermediate_size": 6144,
    "dropout": 0.0,
    "tie_embeddings": true,
    "position_embedding_type": "learned",
    "rope_theta": 10000.0
  },
  "training": {
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 1e-05,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_steps": 5715,
    "gradient_clip": 1.0,
    "precision": "bf16",
    "checkpoint_interval": 259,
    "evaluation_interval": 259,
    "seed": 42
  },
  "initialization": {
    "path": "/home/k9-vortex/Development/3-Python/31-LLM Source/checkpoints/codexa-900m-sft-repair-v1/latest.pt",
    "sha256": "c55f4edd1477a8ffefd03ce63d033a291eb261f6047b129baa47e10756899a4f",
    "semantics": "weights_initialization"
  },
  "tokenizer": {
    "path": "/home/k9-vortex/Development/3-Python/31-LLM Source/checkpoints/tokenizer-base-v1/tokenizer.json",
    "sha256": "a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e"
  },
  "data": {
    "train": "/home/k9-vortex/Development/3-Python/31-LLM Source/data/processed/general-chat-sft-v2/train.jsonl",
    "validation": "/home/k9-vortex/Development/3-Python/31-LLM Source/data/processed/general-chat-sft-v2/validation.jsonl",
    "train_sha256": "15d29ea23ddb57c70067643e9feb641f03b9cd42bdecace7cb710eab784d61e0",
    "validation_sha256": "30f0fcde04cf4f9b34f4e42f5c7941036a8509e81a8d9e1ad7cbc49fd823b074",
    "manifest_sha256": "3b1741ae312316172a1cb0f8a6457cebf2603fcbf1f7ba8896a56fcc4ac60f0d",
    "maximum_epochs": 1,
    "workers": 0,
    "balanced": false,
    "bucket_size": 128,
    "sampling": "seeded_no_replacement_epoch_shuffle",
    "packing": false,
    "sequence_policy": "complete_conversation_dynamic_right_padding",
    "validation_indices": [
      10039,
      10040,
      10041,
      10042,
      10043,
      10044,
      10045,
      10046,
      10047,
      10048,
      10049,
      10050,
      10051,
      10052,
      10053,
      10054,
      10055,
      10056,
      10057,
      10058,
      10059,
      10060,
      10061,
      10062,
      10063,
      10064,
      10065,
      10066,
      10067,
      10068,
      10069,
      10070,
      10071,
      10072,
      10073,
      10074,
      10075,
      10076,
      10077,
      10078,
      10079,
      10080,
      10081,
      10082,
      10083,
      10084,
      10085,
      10086,
      10087,
      10088,
      10089,
      10090,
      10091,
      10092,
      10093,
      10094,
      10095,
      10096,
      10097,
      10098,
      10099,
      10100,
      10101,
      10102,
      10103,
      10104,
      10105,
      10106,
      10107,
      10108,
      10109,
      10110,
      10111,
      10112,
      10113,
      10114,
      10115,
      10116,
      10117,
      10118,
      10119,
      10120,
      10121,
      10122,
      10123,
      10124,
      10125,
      10126,
      10127,
      10128,
      10129,
      10130,
      10131,
      10132,
      10133,
      10134
    ],
    "cache_bytes": 67108864
  },
  "optimization": {
    "optimizer": "adamw8bit",
    "gradient_checkpointing": true,
    "compile": false,
    "tf32": false,
    "loss_normalization": "supervised_tokens",
    "betas": [
      0.9,
      0.95
    ],
    "epsilon": 1e-08,
    "minimum_headroom_bytes": 1610612736,
    "deterministic_algorithms": true,
    "cublas_workspace_config": ":4096:8",
    "allocator": "expandable_segments:True",
    "release_cached_segments_at_boundary": true
  },
  "evaluation": {
    "validation_batches": 96,
    "quality_reserve_seconds": 120,
    "validation_reserve_seconds": 120,
    "development_suite_sha256": "1e9765e6dfa67fcc94a13df242a13e0aae7149ed7ab592337b525c19c55e13bc",
    "acceptance_suite_sha256": "4c8679b9288e13f764c3cc643e8b6ebb2e70af8a64a28f254a4099f6ed9b125c"
  },
  "checkpoint": {
    "minimum_save_seconds": 50.105593050004245,
    "slots": 6
  },
  "baseline_quality": {
    "cases": 64,
    "category_pass_rates": {
      "arithmetic": 0.0,
      "coherence_proxy": 0.0,
      "factual": 0.375,
      "instruction": 0.0,
      "length": 0.0,
      "logic": 0.0,
      "multi_turn": 0.625,
      "uncertainty": 0.0
    },
    "macro_task_accuracy": 0.14285714285714285,
    "END_rate": 0.6875,
    "EOS_rate": 0.0,
    "length_limit_rate": 0.3125,
    "severe_degeneration_rate": 0.359375,
    "empty_rate": 0.0,
    "role_switch_rate": 0.0,
    "mean_generated_tokens": 59.578125,
    "special_token_dominated_rate": 0.0,
    "mean_repeated_4gram_rate": 0.2503361911389545,
    "entry_gate_passed": false
  },
  "wall_clock_seconds": 36000,
  "finalization_reserve_seconds": 773.4717048840394,
  "evaluation_suite_version": "codexa-chat-local-v1",
  "architecture_sha256": "4f4739f333cd67efd382942b44bac689332762231e2d1b558f4e637f242b5545"
}
```

Conservative proposed peak LR 1e-5 (20× below historical base peak), cosine to 1e-6, measured-derived warmup and
step/interval caps above. Weight decay 0.1 with bias/norm exclusions and tied-parameter deduplication;
gradient clip 1.0; BF16 autocast with FP32 parameters; no AMP scaler. Microbatch 1 × accumulation 32 gives
32 conversations per complete optimizer update. Loss normalization sums assistant losses then divides the
accumulated gradients by the actual supervised token count, rather than averaging uneven microbatch losses.
Seed 42. No architecture, tokenizer, special ID or context-limit change.

The failed behavioral entry gate led to one LR corrective experiment, not an accepted three-LR search. Its separate
memory configurations preserve batch semantics and all artifacts. LR/mixture optimality is therefore unproven.
One-epoch stopping prevents time-budget-driven repetition. No new complex objective, base replay or external evaluator.

## E. GPU Performance

Fresh preflight: GPU `NVIDIA GeForce RTX 4080, 16376, 12648, 615.71.09`; host memory/disk snapshots in preflight JSON.
Generative Python 3.14.7 / PyTorch 2.13.0+cu130 / CUDA 13.0, BF16 supported; unrelated desktop processes retained.
Existing host swap use was observed. Free memory is a launch-time gate, not a promised constant.

Full 934M controlled probes: six complete optimizer updates each, first two excluded from steady-state throughput.
All seven variants passed finite loss/update checks. Per-probe sequence populations differ; short microbatch 2 is
not evidence that long microbatch 2 fits. Validation differences across probes are reported, not assumed identical.

| Variant | Assistant tok/s | Context tok/s | Median update seconds | Peak reserved GiB |
|---|---:|---:|---:|---:|
| fixed2048-long-m1 | 5919 | 6618 | 0.310 | 9.84 |
| dynamic-long-m1 | 5984 | 6691 | 0.306 | 9.95 |
| dynamic-short-m1 | 3157 | 3339 | 0.146 | 9.12 |
| dynamic-short-m2 | 4615 | 5005 | 0.193 | 9.30 |
| dynamic-short-workers2 | 3143 | 3324 | 0.145 | 9.12 |
| dynamic-short-no-checkpointing | 3651 | 3861 | 0.126 | 10.25 |
| dynamic-short-unpinned | 3172 | 3355 | 0.145 | 9.12 |

Final deterministic-setting confirmation (including complete accumulation-32 mixed-length updates):

| Variant | Assistant tok/s | Context tok/s | Median update seconds | Peak reserved GiB |
|---|---:|---:|---:|---:|
| dynamic-long-m1 | 5631 | 6297 | 0.325 | 9.46 |
| dynamic-short-m1 | 3046 | 3221 | 0.151 | 9.09 |
| dynamic-mixed-accum32 | 6305 | 6902 | 5.870 | 11.38 |

The long dynamic probe retained about 2.65 GiB free VRAM. Accumulation-32 variable-length pilots initially exceeded
the 1.5 GiB reserve after two updates; both stopped safely. Expandable segments alone was insufficient. The explicit
third configuration releases unused CUDA cache at optimizer boundaries; sampler/optimizer/batch semantics remain fixed.
Cold start includes indexing, model restore, viewer attachment and initial save. The pilot tests this realistic shape. A later strict allocator-headroom test failed its accumulation-32 probe under the then-current desktop load. That memory gate remains BLOCKED; the failed cap configuration is not adopted.

Pilot optimizer-only aggregate: **5237 assistant tok/s**, **6725 context tok/s**;
median complete update **5.100s**. This excludes cache-release, journaling, validation, quality generation
and checkpoint overhead; session budget includes all of them. Initial full-state recovery save/verify
10.85s; final 16.77s;
restore 8.15s. Development 64-case generation for repair took 37.71s.
Pilot evaluation duration and every validation/checkpoint duration remain in the JSON/JSONL artifacts.

**Projection, not a training result:** optimizer throughput × remaining budget × 90% overhead factor gives
approximately 147,584,268 assistant tokens (1.00
of the recorded one-epoch pool), capped by available examples and the resolved step cap. Length mix, desktop load,
evaluation and serialization can change this. No promise of 2,425 historical tok/s or guaranteed full-pool coverage.
BF16/SDPA/checkpointing/length padding/microbatch/workers/pinning were measured. Full FP32 AdamW would require about
14.95 GB just for weights, gradients and two moments, before activations/desktop; it was not deliberately OOM-tested.
torch.compile/TF32 and more aggressive optimizers remain unbenchmarked and disabled; no speedup claim.

## F. Checkpoint and Recovery

Native version-1 payload retained; six slots under each isolated run, atomic verified index references latest,
previous, best and final. Native staged serialization, SHA sidecars, deserialize/counter verification and publication
are reused. New finite saved-tensor checks and directory fsync protect publication. Interrupted writes never change
the published index. Changed run/config/tokenizer identity is rejected. Historical runs/exports are untouched.

Full checkpoints preserve model, optimizer, scheduler, RNG, step/microstep/token/epoch cursor, configuration,
tokenizer/data identity, initialization provenance, source/coverage counters and original budget. No partial accumulation
is serialized: interruption finishes the current complete window; deadline discards partial gradients and restores counters/RNG.
Reconstruction uses seeded epoch ordering and committed batch position. Logs beyond recovered state are archived before trimming.
Hard kill/power loss recovers only published state; unsaved steps cannot be guaranteed. Host/CUDA kernel execution and
opaque optimizer runtime behavior are not claimed bitwise reproducible unless validated below.

Original full-model recovery evidence:

```json
{
  "weights_exact_after_restore": true,
  "optimizer_exact_after_restore": true,
  "scheduler_exact": true,
  "counters_exact": true,
  "continuation_weights_bitwise_equal": false,
  "continuation_optimizer_bitwise_equal": false,
  "losses_reference": [
    2.3028125762939453,
    2.069218158721924
  ],
  "losses_resumed": [
    2.3028125762939453,
    2.068998098373413
  ],
  "restore_seconds": 8.15294083399931,
  "save": {
    "path": "slot-0/checkpoint.pt",
    "sha256": "3e89b037f180ef88144cb31344db1b9dcfc2fda10407fec3f8256e3192e8178e",
    "bytes": 5636069073,
    "optimizer_step": 2,
    "verified": true,
    "write_verify_seconds": 10.846284313003707
  },
  "finite_gradients": true,
  "passed": false,
  "final_save": {
    "path": "slot-1/checkpoint.pt",
    "sha256": "b750d41b1607be77b21187ad946d6a004f9d09220f14ac8832a383ab7d157d92",
    "bytes": 5636388920,
    "optimizer_step": 4,
    "verified": true,
    "write_verify_seconds": 16.774205948997405
  },
  "seconds": 69.19565463999606
}
```

Deterministic-algorithm follow-up:

```json
{
  "weights_exact_after_restore": true,
  "optimizer_exact_after_restore": true,
  "scheduler_exact": true,
  "counters_exact": true,
  "continuation_weights_bitwise_equal": true,
  "continuation_optimizer_bitwise_equal": true,
  "losses_reference": [
    2.301283597946167,
    2.0665042400360107
  ],
  "losses_resumed": [
    2.301283597946167,
    2.0665042400360107
  ],
  "restore_seconds": 9.646470717001648,
  "save": {
    "path": "slot-0/checkpoint.pt",
    "sha256": "d39ad459f32455764e8ca5bfd660606ab81d62a7ebc204093ebd4b8aae475785",
    "bytes": 5636069073,
    "optimizer_step": 2,
    "run_name": "934M-recovery-smoke",
    "run_id": "isolated-recovery",
    "verified": true,
    "write_verify_seconds": 13.436210738000227
  },
  "finite_gradients": true,
  "passed": true,
  "final_save": {
    "path": "slot-1/checkpoint.pt",
    "sha256": "0155949bf69814a96700e76fcc7d70d62bdd7c149aa6cb0dee5579e3eecfe0ee",
    "bytes": 5636388920,
    "optimizer_step": 4,
    "run_name": "934M-recovery-smoke",
    "run_id": "isolated-recovery",
    "verified": true,
    "write_verify_seconds": 23.40372870000283
  },
  "seconds": 87.94409386500047,
  "deterministic_algorithms": true,
  "cublas_workspace_config": ":4096:8"
}
```

Model/optimizer bytes, scheduler and counters restore exactly in the original test; finite continuation succeeds.
Bitwise continuation initially failed; the deterministic follow-up passed and its settings are required in the
prepared specification. This is tested on this stack, not a cross-driver bitwise guarantee. The real pilot SIGTERM
stopped after step 32, saved, then resumed its full state without repeated sample identities or reset deadline.
Viewer closure after step 5 did not stop training. Tiny tests also deliver actual SIGINT/SIGTERM, inject pre-update OOM,
interrupt save publication, discard partial deadline accumulation and preserve committed updates across validation timeout.
A separate allocator-limit test produced a controlled CUDA OOM during accumulation 32, safely recorded without promoting weights. Physical VRAM exhaustion and power loss were not deliberately induced.

Storage plan: 6 slots × 5.636 GB,
one temporary serialization generation, small logs/evaluation and optional separate final export; roughly 45 GB budget.
Preflight requires 100 GiB safe disk headroom plus an 80 GiB working reserve and 12 GiB available host RAM.
Smoke artifacts are retained and separately consume disk. No old-run cleanup occurred.
No independent historical backup was verified; originals remain irreplaceable. Large artifacts remain ignored by Git.

## G. Quality Baseline

64 fixed development cases, eight each in instruction/arithmetic/factual/multi-turn/uncertainty/logic/length/coherence-proxy.
Literal compliance checks, speaker attribution and mechanical repetition metrics; no external paid judge or subjective score
presented as objective truth. Greedy BF16 max-new-tokens 128, seed 42, KV cache, penalty 1.0, no blocked n-grams.
Sampled secondary comparisons use temperature 0.7/top-p 0.9 with the same prompts/seed; stored separately.
END/EOS model stops are distinguished from length limits; generation cannot continue past configured END/EOS stop.
Macro excludes the keyword coherence proxy. Original and speaker-aware scored outputs are preserved separately.
Primary suite SHA256 `1e9765e6dfa67fcc94a13df242a13e0aae7149ed7ab592337b525c19c55e13bc`; decoder/tokenizer/checkpoint fingerprints appear in every report.
The final comparison repeats every candidate with deterministic algorithms and CUBLAS workspace :4096:8;
primary aggregate results match the original comparison. The corrective pilot used the original numerical settings.
Unscored human-review templates define relevance/coherence/context/length ratings and require verification evidence
for checkable unsupported claims. No human rating is claimed.

| Candidate | Literal macro | Factual | Multi-turn | END stop | Length limit | Severe degeneration |
|---|---:|---:|---:|---:|---:|---:|
| codexa-900m-base-v1 | 0.00% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| codexa-900m-sft-v2 | 5.36% | 37.50% | 0.00% | 75.00% | 25.00% | 15.62% |
| codexa-900m-sft-balanced-v1 | 3.57% | 25.00% | 0.00% | 75.00% | 25.00% | 17.19% |
| codexa-900m-sft-repair-v1 | 14.29% | 37.50% | 62.50% | 68.75% | 31.25% | 35.94% |
| codexa-900m-chat-stage2-pilot-v1 | 0.00% | 0.00% | 0.00% | 40.62% | 59.38% | 60.94% |

Every candidate scores zero on exact instruction, arithmetic, logic, line-format length and uncertainty checks.
Entry floor: instruction ≥25%, arithmetic ≥25%, END ≥50%, severe degeneration ≤25%, no empty or unexpected role cases.
These deliberately low entry thresholds are not promotion criteria. Factual keyword rates do not measure unrestricted
hallucination frequency, contradiction or semantic coherence. No reliable broad hallucination percentage is claimed.
All raw outputs are available, including failures; the following consecutive cases were not selected for impressiveness:

**development-000 — instruction**

Prompt: Reply with only the word READY.

```text
READY is a word that is not commonly used in English. It is a word that is not commonly used in English.
```
Stop: stop_token; task pass: False.

**development-001 — arithmetic**

Prompt: What is 11 plus 7? Answer with just the number.

```text
12 plus 7 = 13.
```
Stop: stop_token; task pass: False.

**development-002 — factual**

Prompt: What is the capital of Japan?

```text
Tokyo is the capital of Japan.
```
Stop: stop_token; task pass: True.

**development-003 — multi_turn**

Prompt: What is my updated number?

```text
Your new number is 40.
```
Stop: stop_token; task pass: True.

**development-004 — uncertainty**

Prompt: What is the name of my neighbour? If you do not know, say so.

```text
My neighbour is called "My neighbour" by my friends.
```
Stop: stop_token; task pass: False.

**development-005 — logic**

Prompt: A box holds 2 pencils. I add 1 pencils. How many pencils are in the box? Reply with only the number.

```text
1. A box holds 2 pencils.

2. A box holds 2 pencils.

3. A box holds 2 pencils.

4. A box holds 2 pencils.

5. A box holds 2 pencils.

6. A box holds 2 pencils.

7. A box holds 2 pencils.

8. A box holds 2 pencils.

9. A box holds 2 pencils.

10. A box holds 2 pencils.

11. A box holds 2 pencils.

12. A box holds 2 pencil
```
Stop: length; task pass: False.

**development-006 — length**

Prompt: Write 4, 5, 6, one number per line. Do not add other text.

```text
Sure, here is a simple and straightforward way to add text to a 3rd-5th grade text-to-speech task:

```python
import numpy

#!/bin/bash

# Create a new line
python

# Create a new line
python

# Create a new line
python

# Create a new line
python

# Create a new line
python

# Create a new line
python

#
```
Stop: length; task pass: False.

**development-007 — coherence_proxy**

Prompt: Why does ice float on liquid water? Keep the explanation brief.

```text
The liquid water is a liquid that is not liquid and is not a liquid. It is not a liquid, and it is not a liquid. It is not a liquid, and it is not a liquid.

The liquid water is a liquid, and it is not a liquid. It is not a liquid, and it is not a liquid.
```
Stop: stop_token; task pass: False.

Corrective results:

| Pilot | Final step | Stop reason | Literal macro | END | Severe degeneration |
|---|---:|---|---:|---:|---:|
| corrective-00  | 2 | MEMORY_HEADROOM_GATE | 14.29% | 62.50% | 35.94% |
| corrective-01  | 2 | MEMORY_HEADROOM_GATE | 14.29% | 59.38% | 39.06% |
| corrective-02  | 32 | INTERRUPTED_SIGNAL_15 | 14.29% | 60.94% | 37.50% |
| corrective-02 resumed | 128 | STEP_CAP | 16.07% | 50.00% | 53.12% |

Paired cases are selected deterministically by first case ID satisfying the stated transition, not by impressive
appearance. Literal status changes do not establish broad semantic improvement:

**Literal check improved: first qualifying case development-019**

What is my updated number?

Initialization (stop_token):

```text
Your new number is 22.
```

Pilot (stop_token):

```text
Your new number is 42.
```

**Literal check regressed: first qualifying case development-026**

How many sides does a triangle have?

Initialization (length):

```text
There are seven sides in a triangle, of course.

The first three sides are:

1. The first three sides are:

1. The second three sides are:

2. The third side is:

3. The fourth side is:

4. The fourth side is:

5. The fourth side is:

6. The fourth side is:

7. The fourth side is:

8. The fourth side is:

9. The fourth side is:

10. The fourth side is:

11. The fifth side
```

Pilot (stop_token):

```text
The triangle is a triangle that is used to represent the length of a triangle. It is used to represent the length of a triangle, and it is used to represent the length of a triangle.
```

**Failure unchanged: first qualifying case development-000**

Reply with only the word READY.

Initialization (stop_token):

```text
READY is a word that is not commonly used in English. It is a word that is not commonly used in English.
```

Pilot (length):

```text
READY is a word that is used to describe a type of language that is not spoken by humans. It is a language that is not spoken by humans, but rather by humans. READY is a language that is not spoken by humans, but rather by humans. READY is a language that is not spoken by humans, but rather by humans. READY is a language that is not spoken by humans, but rather by humans. READY is a language that is not spoken by humans, but rather by humans. REA
```

Retention and common validation (same population/settings per checkpoint):

| Checkpoint folder | Exposed base replay loss | Common chat validation loss | Common chat target tokens |
|---|---:|---:|---:|
| codexa-900m-base-v1 | 3.12479 | 2.38119 | 74,708 |
| codexa-900m-sft-v2 | 3.41792 | 1.84100 | 74,708 |
| codexa-900m-sft-balanced-v1 | 3.18536 | 1.83780 | 74,708 |
| codexa-900m-sft-repair-v1 | 3.27385 | 2.22954 | 74,708 |
| codexa-900m-chat-stage2-pilot-v1 | 3.15909 | 2.14312 | 74,708 |
| slot-3 | 3.21068 | 2.08100 | 74,708 |

Replay is eight fixed 512-target-token windows from base TRAINING, not missing held-out base validation.
Offsets/digest are in `retention-diagnostic.json`. Development coding/factual/arithmetic tasks supply bounded capability
checks; they cannot certify catastrophic forgetting absence. Protected final acceptance is still sealed.

## H. Implemented Changes

- **LLM-Data:** `src/llm_data/chat_dataset.py`, `tests/test_chat_dataset.py` — lazy indexing, masks, collate,
  deterministic source-aware sampler, split isolation and restoration coverage.
- **LLM-Training:** `src/llm_training/training.py`, `session.py`, `recovery.py`, `chat_run.py`,
  `conversational_training.py`, `viewer.py`, `tests/test_chat_runtime.py` — runtime, recovery, deadline, signals,
  supervised normalization and conversational viewer. Existing training defaults remain compatible.
- **LLM-From-Scratch:** `configs/chat-10h-v1.yaml`, `scripts/chat_training.py`,
  `src/codexa_workspace/chat_preparation.py`, `chat_evaluation.py`, `chat_operations.py`, `chat_validation.py`,
  `chat_report.py`, `tests/test_chat_preparation.py`, this report and appended central session/progress records.
  Pre-existing dirty AGENTS/CLAUDE/CHATGPT_CURRENT_STATE files were preserved, not authored by this preparation.

Offline build via `tools/build_packages.py`; force-reinstalled only llm-data, llm-training and codexa-workspace wheels
with `--no-deps`, no dependency upgrades. Imported locations, source byte comparison and tree hashes are frozen in
the launch manifest. All component revisions and dirty state:

| Repository | Commit | Working tree | Installed Python source |
|---|---|---|---|
| LLM-Architecture | `c33462047ad75b67d39ffbaf4a95e064fbbe5be4` | clean | MATCH |
| LLM-Tokenizer | `cb9230544faf5d954bfe78f3c58c57d5ec27820d` | clean | MATCH |
| LLM-Data | `86d2b93944a4f47c14d8bb273f3bd0925780d6ec` | dirty | MATCH |
| LLM-Training | `208cede89176f9bdcca193b666f33d06155ef65c` | dirty | MATCH |
| LLM-Inference | `48a9f4b7fb5b4eef20fe9ce61f2089396b1fc244` | clean | MATCH |
| LLM-Memory | `b5f41b76debcd8733e2efa0793df29ca5964fc26` | clean | MATCH |
| LLM-Specialist | `8f5ca3a34cde70bd7b357b52cd8ee9e44e21527b` | clean | DIFFERENT/UNAVAILABLE |
| LLM-From-Scratch | `c568f9e2174e15c307f734c5bd8b004cbe78311f` | dirty | MATCH |

## I. Testing Results

`/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/bin/python -m pytest -q -rs`

Exit 0; 10.96s. Evidence: `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/regression-00.txt`.

```text
-none for more information.
    warnings.warn(

tests/test_token_data.py::test_memmap_dataset_and_loader
  /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/lib64/python3.14/site-packages/_pytest/python.py:171: PytestReturnNotNoneWarning: Test functions should return None, but tests/test_token_data.py::test_memmap_dataset_and_loader returned <class 'float'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

tests/test_tokenizer.py::test_training_and_round_trip
  /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/lib64/python3.14/site-packages/_pytest/python.py:171: PytestReturnNotNoneWarning: Test functions should return None, but tests/test_tokenizer.py::test_training_and_round_trip returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

tests/test_training.py::test_training_and_cycling
  /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/lib64/python3.14/site-packages/_pytest/python.py:171: PytestReturnNotNoneWarning: Test functions should return None, but tests/test_training.py::test_training_and_cycling returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
SKIPPED [1] tests/test_specialist.py:271: could not import 'sklearn': No module named 'sklearn'
SKIPPED [1] tests/test_specialist.py:361: could not import 'sklearn': No module named 'sklearn'
SKIPPED [1] tests/test_specialist.py:409: could not import 'sklearn': No module named 'sklearn'
SKIPPED [1] tests/test_specialist.py:451: could not import 'sklearn': No module named 'sklearn'
SKIPPED [1] tests/test_specialist.py:469: could not import 'sklearn': No module named 'sklearn'
126 passed, 5 skipped, 8 warnings in 9.86s

```

`/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/bin/python -m pytest -q ../LLM-Architecture/tests ../LLM-Tokenizer/tests ../LLM-Data/tests ../LLM-Training/tests ../LLM-Inference/tests tests/test_chat_preparation.py`

Exit 0; 2.67s. Evidence: `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/regression-01.txt`.

```text
........................................................                 [100%]
=============================== warnings summary ===============================
test_checkpointing.py::test_exact_resume
  /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/lib64/python3.14/site-packages/_pytest/python.py:171: PytestReturnNotNoneWarning: Test functions should return None, but test_checkpointing.py::test_exact_resume returned <class 'tuple'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

test_config.py::test_valid_smoke_config
  /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/lib64/python3.14/site-packages/_pytest/python.py:171: PytestReturnNotNoneWarning: Test functions should return None, but test_config.py::test_valid_smoke_config returned <class 'llm_training.config.ProjectConfig'>.
  Did you mean to use `assert` instead of `return`?
  See https://docs.pytest.org/en/stable/how-to/assert.html#return-not-none for more information.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
56 passed, 2 warnings in 1.93s

```

`/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/.venv/bin/python -m pip check`

Exit 0; 0.24s. Evidence: `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/regression-02.txt`.

```text
No broken requirements found.

```

Preparation commands actually executed: preflight/audit/candidates/benchmark/recovery/pilots/resume-pilot/diagnostics
(and deterministic recovery if its report is present), all using the integration generative Python. Isolated
full-model checks and five checkpoint comparisons accompany CPU regression results. Five sklearn-dependent specialist
tests are skipped in this environment; existing pytest return-value warnings are not newly introduced failures.
No blanket claim that every optional test passed. Checkpoint bitwise continuity and behavioral gates are separate
failed evidence, even when automated regressions pass.

## J. Risks and Remaining Limitations

The model has poor exact instruction/reasoning/uncertainty behavior. Lower SFT validation loss cannot select a good
assistant by itself. Repair is provisional; repetition/termination may regress through continued SFT. Current data
is dominated by UltraChat supervision and repeated conversation prefixes; content correctness was not certified.
Historical split exposure and one exact-overlap development prompt prevent clean generalization claims. The synthetic
protected suite shares task families/facts despite different wording/values; it is not a large independent benchmark.
Neither semantic human coherence ratings nor broad unsupported-claim rates were fabricated.

GPU free memory fluctuates with the desktop. Checkpoint reserve is measured with margins, not a guaranteed storage
latency bound; the watchdog can interrupt finalization and leave only the last verified checkpoint. Resume counts
downtime and rejects backwards wall-clock reconciliation; same-boot enforcement also preserves the monotonic deadline.
Startup SIGKILL/power loss before first valid save cannot create new recovery state. Viewer attachment failure blocks
updates; viewer closure afterwards does not. No backup, thermal endurance ten-hour soak, compile speedup or full
physical OOM behavior is certified. Final acceptance/promotion requires independent review.

## K. Exact Production Commands

Run from the integration repository:

```bash
cd '/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch'
# Read-only preflight / dry run; prints BLOCKED, starts no training.
.venv/bin/python scripts/chat_training.py preflight
# Freeze current regression evidence, measured specification and report; no production launch.
.venv/bin/python scripts/chat_training.py prepare --stage finalize
```

**The following production command is prepared but currently refuses failed gates. Do not run it as approval
or edit the manifest to bypass gates. It requires a later READY evidence set and explicit user approval.**

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/chat_training.py start
# Attach/restore the visible 100×22 Kitty viewer (also auto-attached by the worker).
.venv/bin/python scripts/chat_training.py view
# Recent structured metrics, once a production job exists.
.venv/bin/python scripts/chat_training.py logs --tail 50
tail -n 50 outputs/chat-training/codexa-900m-chat-10h-v1/console.log
# Graceful stop; verifies worker PID command and process-start identity.
.venv/bin/python scripts/chat_training.py stop
# Full-state resume; original deadline includes all downtime, never resets to ten hours.
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/chat_training.py resume
# Development comparison after stopping. Protected acceptance is restricted to a finished phase.
.venv/bin/python scripts/chat_training.py evaluate --suite development
.venv/bin/python scripts/chat_training.py evaluate --suite acceptance
```

Start verifies exact Python, inputs, initialization/config identities, installed source trees, disk/GPU headroom,
conflicting training processes and an exclusive supervisor lock; creates a fresh isolated namespace and detached
worker. Budget begins at the operator command's captured monotonic start, before imports/preflight/loading, and covers
training, evaluation, saves and verification. No partial accumulation update, long validation near the reserve,
or deliberate consumption of unused hours. Structured logs/viewer are independent of the launch terminal.

After training compare base, selected initialization, best and final (if different) with common 64-case protected
greedy and sampled settings, plus development/retention diagnostics. The worker reserves finalization time and records
incomplete acceptance if it cannot finish. Preserve the first protected comparison; interrupted phases cannot tune
against it. Review all outputs, unchanged cases and regressions; semantic human review remains necessary.
No automatic inference-default replacement, export overwrite or promotion. Promotion is a separate decision.

## L. Final Readiness Decision

**BLOCKED — mandatory requirements have not all been satisfied.**

| Gate | Result |
|---|---|
| dataset_audit | PASS |
| initialization_quality | FAIL / UNPROVEN |
| worst_shape_memory | PASS |
| full_optimizer_recovery | PASS |
| realistic_accumulation_memory | PASS |
| behavioral_pilot | FAIL / UNPROVEN |
| deadline_and_signal_tests | PASS |
| regression_tests | PASS |
| installed_implementations | PASS |
| held_out_exposure_provenance | FAIL / UNPROVEN |
| strict_live_memory_headroom | FAIL / UNPROVEN |

Next justified phase: resolve trustworthy exposure provenance and inspect
existing conversational supervision quality and test a separately bounded corrective strategy. Require measurable
instruction/reasoning compliance and reduced degeneration on development data before considering a ten-hour launch.
Do not spend ten hours merely to lower loss. Historical datasets/checkpoints/exports, NumPy workspace, accepted model,
and Git remotes remain untouched. No cleanup, push, merge or release occurred.
