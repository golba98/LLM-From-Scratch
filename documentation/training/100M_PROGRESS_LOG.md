# Codexa 100M — Three-Month Progress Log

## 2026-08-21 — Native chat behavioral probe

The current native 900M conversational checkpoint was loaded through
`scripts/native_chat_bridge.py` and exercised with multi-turn and reset-isolated
prompts. The runtime had no CUDA device available, so this was a CPU behavioral
probe rather than a performance benchmark. The 900M model generated generic or
semantically incorrect answers, repeated a previous weather-app response across
later turns, stopped by the token limit instead of `<|end|>`, and failed basic
arithmetic and format-following prompts. The checkpoint is therefore retained
for diagnosis but is not accepted as conversationally ready. A retained 100M
pilot produced the same failure pattern in separate reset-isolated probes.

Next milestone: run a bounded chat-format overfit diagnostic covering role
serialization, assistant-only masks, EOS/stop supervision, and deterministic
behavioral probes before selecting the next SFT recipe.

## 2026-08-21 — Chat protocol overfit gate passed

The bounded diagnostic used the real tokenizer and chat-v1 serializer on a tiny
CUDA model for 300 steps. Loss decreased from `9.625757` to `0.0001216`. The
model reproduced fixed arithmetic, identity, explanation, and correction
answers exactly, and every response stopped on `<|end|>`. This proves the
role-token format, shifted assistant-only labels, and native stop handling are
mechanically capable of learning.

The 900M SFT failure should therefore be investigated in the training corpus,
checkpoint lineage, replay mixture, optimization schedule, and scale-specific
training behavior. No 900M checkpoint was deleted or promoted during this
diagnostic.

## 2026-08-21 — 900M SFT sanity recipe rejected

A 205-step CUDA sanity run from the verified 900M base completed successfully
at the numerical level, but its reloaded checkpoint was behaviorally worse than
the existing SFT-v2 checkpoint. It repeated an agricultural-field template,
failed arithmetic, explanation, correction, and requested list formatting, and
often exhausted the generation limit. This first-500-record-per-source recipe
is rejected as a narrow-subset overfit or catastrophic-forgetting experiment;
the checkpoint remains preserved for comparison.

## 2026-08-21 — Balanced 900M SFT v1 launched

The replacement run `codexa-900m-sft-balanced-v1` is using deterministic
corpus-wide samples of 30,000 UltraChat and 30,000 OASST1 records, 50% base
replay, a quarter-rate SFT learning-rate scale, and 1,000 warmup steps over a
6,000-step target. The run has its own checkpoint lineage and attached 100×22
Kitty viewer; previous checkpoints remain preserved until behavioral acceptance.

## 2026-08-22 — Balanced 900M SFT v1 failed behavioral acceptance

The balanced 6,000-step run completed numerically with validation loss
`2.028321`, but the reloaded model still failed basic arithmetic, explanation,
correction, and exact list-format probes. It is preserved but not accepted as
chat-ready. Further long SFT runs are gated on auditing sample alignment and
adding explicit short-answer and instruction-following coverage.

This document records the model's measurable progress across the next 100M
training phase and the later conversational SFT phase.

## Objective

Produce a versatile 100M-class model that can:

- continue ordinary text coherently;
- answer simple questions after conversational SFT;
- maintain a short multi-turn conversation;
- load and generate correctly in LM Studio;
- improve measurably over a fixed three-month evaluation schedule.

## Baseline — 2026-08-03

The first base checkpoint completed 10,000 optimizer steps and processed
655,360,000 tokens. Its native validation loss was 3.61696 (perplexity 37.22).

The current LM Studio GGUF is only a compatibility adapter. Its output is
recorded as a conversion failure, not as the model's true quality baseline:

> The model produced repetitive and malformed text in LM Studio. The adapter
> changes the trained architecture's positional and feed-forward behavior.

Baseline artifacts:

- Native checkpoint: `checkpoints/codexa-100m-base-v1/latest.pt`
- Native evaluation: `logs/codexa-100m-base-v1/evaluation.json`
- LM Studio adapter: `exports/codexa-100m-base-v1-lmstudio-llama-f16.gguf`

## Native checkpoint — 2026-08-03

- Run: `codexa-100m-native-v1`
- Architecture: native rotary/RMSNorm/SwiGLU
- Optimizer steps / tokens seen: 10,000 / 655,360,000
- Parameters: 125,848,320
- Training loss: 3.16557
- Validation loss / perplexity: 3.43051 / 30.89
- Speed: 52,850.6 tokens/second
- LM Studio load: pass
- LM Studio generation: pass at runtime level; quality gate pending
- Native generation: coherent fragments, but repetition remains high
- Full fixed-prompt outputs: `logs/codexa-100m-native-v1/evaluation.json`
- GGUF artifact: `exports/codexa-100m-native-v1-lmstudio-f16.gguf`
- LM Studio test transcript: `logs/codexa-100m-native-v1/lmstudio_test_2026-08-03.md`
- Multi-turn chat transcript: `logs/codexa-100m-native-v1/multi_turn_chat_2026-08-03.md`

This checkpoint is a meaningful improvement over the previous validation
perplexity (37.22 to 30.89), but it is not yet accepted as a conversational
model. Repetition must be addressed through additional base training and then
assistant-only conversational SFT.

The natural four-turn chat test failed memory and grounding: the model ignored
the greeting, did not recall the name Maya or Cape Town, ignored astronomy, and
repeated unrelated health text. This is the baseline that conversational SFT
must improve.

The LM Studio transcript confirms that the remaining problem is model quality:
runtime compatibility passes, but factual answers, arithmetic, instruction
following, and conversation all fail through repetition or topic drift.

## Next model contract

The replacement 100M model will use an LM Studio-compatible architecture from
the start: RMSNorm, SwiGLU, rotary position embeddings, and exact tokenizer
mapping. No post-training architecture adapter will be accepted.

## Fixed evaluation suite

Every checkpoint evaluation must record the same prompts and settings:

1. ordinary prose continuation;
2. factual explanation;
3. short instruction response;
4. multi-turn conversation;
5. repetition stress test;
6. LM Studio GGUF load and generation smoke test.

For each prompt, record the complete generated output, checkpoint step, token
count, tokens/second, temperature, top-k/top-p, and finish reason.

## Three-month schedule

| Period | Required record |
|---|---|
| Week 0 | Architecture/export compatibility gate and baseline outputs |
| Weeks 1–4 | Base pretraining checkpoints and weekly evaluation |
| Weeks 5–8 | Continued base training, regression checks, tokenizer/runtime checks |
| Weeks 9–10 | Conversational SFT checkpoints and multi-turn tests |
| Weeks 11–12 | Final SFT selection, LM Studio validation, comparison report |

## Checkpoint entry template

### YYYY-MM-DD — checkpoint name

- Training stage:
- Optimizer step / tokens seen:
- Validation loss / perplexity:
- LM Studio load: pass/fail
- Generation speed:
- Repetition score:
- Notes:

#### Fixed-prompt outputs

```text
Prompt:
Output:
```

## Acceptance gates

- GGUF tokenization matches native token IDs.
- LM Studio loads without an architecture warning.
- Native and LM Studio generations are materially equivalent.
- Repetition decreases across checkpoints.
- Conversational SFT must improve multi-turn behavior without unacceptable
  regression on the base-language prompts.

## 2026-08-03 — SFT pilot in progress

The first conversational fine-tuning run is active as `codexa-100m-sft-pilot`.
It is configured for 2,000 optimizer steps over 60,000 UltraChat training
conversations with 1,000 validation conversations and 10% base-corpus replay.
The visible viewer is `scripts/run_100m_sft_pilot.sh`; live metrics are written
to `logs/codexa-100m-sft-pilot/train_metrics.jsonl`.

## 2026-08-03 — Base corpus preparation milestone

The separate 1B base-corpus preparation completed before the interactive
session restarted. The run processed all 13,700,814 raw rows in 6,590 seconds
(1h 49m 50s) and produced approximately 54 GB of cleaned JSONL.

- Wikipedia accepted: 6,406,791; validation: 31,957.
- FineWeb-Edu accepted: 7,043,681; validation: 35,232.
- Exact within-source duplicates rejected: 1,023 Wikipedia and 249,319
  FineWeb-Edu.
- Manifest: `data/processed/base-v1/base_preparation_manifest.json`.
- Near-duplicate clustering: deferred and remains a production gate.

## 2026-08-03 — SFT pilot resumed

The interrupted conversational pilot resumed from its last saved milestone at
step 1,500 and is continuing toward the configured step 2,000 endpoint. The
native base checkpoint remains unchanged. Progress is visible in Kitty through
the shared dashboard and is recorded in
`logs/codexa-100m-sft-pilot/train_metrics.jsonl`.

## 2026-08-03 — SFT pilot complete

The conversational SFT pilot completed its planned 2,000 optimizer steps in
the restored Kitty viewer. Final metrics were 2.198 training loss, 2.003
validation loss, and 15,784,215 processed tokens. The checkpoint remains at
`checkpoints/codexa-100m-sft-pilot/latest.pt` pending evaluation and export
smoke gates.

## 2026-08-03 — 900M-class base training launched

The 934,356,480-parameter 900M-class base model is now training from random
weights as `codexa-900m-base-v1`. The run uses the passed production shape:
BF16, gradient checkpointing, 8-bit AdamW, context 2,048, microbatch 1, and
gradient accumulation 32. The configured budget is 10,000 optimizer steps
(655,360,000 tokens). The Kitty dashboard is attached and the first metrics are
finite at approximately 7,697 tokens/second.

## 2026-08-03 — 900M-class base run paused

The live base process is paused in place at optimizer step 418 (27,394,048
tokens) with `SIGSTOP`. A durable checkpoint was completed at step 250 and is
available for recovery if the process cannot be continued. Resume in place
with `kill -CONT 33993`.

## 2026-08-03 — Conversational quality gate failed

Deterministic probes of the native LM Studio export and completed SFT pilot
showed severe repetition. The SFT pilot can begin a relevant answer, but it
loops the same sentence and does not stop reliably; the native export also
drifts and fails factual and memory prompts. Neither checkpoint is accepted as
a chat model. Further work is required on SFT serialization/EOS supervision,
training duration, and controlled base replay before export.

## 2026-08-03 — Paused at checkpoint 500

Training resumed until the next durable checkpoint and is now paused at step
500, with `latest.pt` and its checksum/completion metadata verified. The user
can use the GPU; resume in place with `kill -CONT 33993`.

## 2026-08-04 — 900M-class base training complete

The 934,356,480-parameter base run completed at 10,000 optimizer steps and
655,360,000 processed tokens. Final training loss was 3.1042 and validation
loss was 3.3491. The final step-10,000 checkpoint is verified and preserved;
evaluation and export gates are next.

## 2026-08-04 — 900M base handoff exported

The verified step-10,000 checkpoint was exported to
`exports/codexa-900m-base-v1-lmstudio-f16.gguf` (SHA256
`506bbd45becc4b711cd512f204260d3496b3d5c76aa966d276f476f7dd8e789d`). Native
generation and GGUF structural validation passed. Redundant 900M milestones,
the duplicate best copy, and intermediate HF staging directories were moved
to Trash after validation; `latest.pt`, `previous.pt`, the tokenizer, datasets,
and the GGUF export remain preserved.
## 2026-08-05 — Conversational SFT v2 resumed

`codexa-900m-sft-v2` was interrupted after log-only progress reached optimizer step 1,836. The latest durable checkpoint was validated and contains optimizer step 1,750; its refreshed SHA-256 sidecar is `2364eaddc4668d077727ecf53b9dbb058e52852ef6151bad78537dcd6532e483`. Resume is being launched toward the existing 6,000-step target with BF16, activation checkpointing, microbatch 1, and gradient accumulation 32. Peak reserved VRAM observed before interruption was approximately 12.9 GiB; the run remains unvalidated until completion and smoke/quality gates pass.
The resume is confirmed at optimizer step 1,751 under a fresh run ID. The process is active as PID 6682 and the required Kitty viewer is attached separately as PID 6683. The resumed metric reports training loss 2.5412 and peak reserved VRAM 11,070,865,408 bytes; the run is not yet complete or acceptance-validated.

## 2026-08-06 — Conversational SFT v2 paused

The active `codexa-900m-sft-v2` process was suspended with `SIGSTOP` for a
power outage. PID `6682` is stopped, the 100x22 Kitty viewer remains attached,
and `latest.pt` is preserved at 04:17:52. The latest log-only metric reached
optimizer step 5,117; resume only after power returns, using the durable
checkpoint rather than assuming log-only progress is recoverable.

## 2026-08-06 — Conversational SFT v2 resumed

Power was restored and the paused process was continued with `SIGCONT`. PID
`6682` is active again, the required 100x22 viewer remains attached, and the
first post-resume metric reached optimizer step 5,118. The run remains
incomplete and acceptance validation is still pending.

## 2026-08-06 — 900M conversational SFT v2 completed and staged

The `codexa-900m-sft-v2` run completed at optimizer step 6,000 with
103,459,920 total tokens, training loss 1.5768, and validation loss 2.0316.
The final checkpoint passed its checksum check and was exported to
`exports/codexa-900m-sft-v2-lmstudio-f16.gguf` (SHA256
`5b5a2f7a17282946e6573c80027886ed4d0f41f705305b73fc678411f34d5698`). LM
Studio imported and loaded the 900M GGUF successfully, but its chat and raw
completion API smoke tests failed with the expected-format engine errors.
The checkpoint, datasets, logs, and prior lineages remain preserved while the
exporter/runtime compatibility gate is unresolved.

## 2026-08-06 — Native interactive chat path verified

Added `scripts/chat_native.py` for direct CUDA inference from the completed
900M SFT checkpoint. A native probe generated a coherent answer without LM
Studio, confirming that the checkpoint can be tested independently of the
failed GGUF adapter. The response still used “Open Assistant” identity text,
so native runtime success is not a conversational quality acceptance result.

## 2026-08-06 — Native model added to local Codexa CLI

The local-development Codexa launcher now supports `codexa-dev native`, which
loads the verified SFT v2 checkpoint and tokenizer directly on CUDA. The exact
CLI command completed an end-to-end smoke conversation without LM Studio. This
is a runtime-access success only; conversational quality remains under review.

The shell launcher was subsequently repaired so `codexa-dev native` keeps the
current terminal's stdin instead of disappearing into the generic external
terminal handoff. A visible Kitty session is running with the native checkpoint
loaded on CUDA and ready at the `You:` prompt.
## 2026-08-06 — Dedicated Codexa Native provider accepted structurally

The local Codexa CLI now exposes `Codexa Native` as a separate provider and
loads the 900M SFT v2 PyTorch checkpoint through a persistent JSONL bridge. A
two-turn CUDA smoke test passed bridge loading, request handling, and retained
history transport without LM Studio. Response quality remains below acceptance:
both smoke answers ignored the requested exact phrase and terminated at the
token limit. Provider transport is working; checkpoint behavior remains the
next quality gate.

## 2026-08-11 — Next base-training data prepared

The next base-training input is already present and was not rebuilt. The mixed
FineWeb-Edu/Wikipedia stream contains 12,819,847,234 training tokens across
13,383,283 documents, packed at context length 2,048 with zero repeated
documents. Recorded SHA-256 checksums for the mixed train file, index, and
manifest match the files on disk. No training process was launched.

The current 10,000-step configuration processes 655,360,000 tokens, or about
5.1% of this stream. A full single pass would require approximately 195,616
optimizer steps. Using observed 900M SFT throughput of roughly 2,800–3,100
tokens/second, that is approximately 48–53 days of continuous GPU time before
checkpoint, validation, and downtime overhead. The Python validator remains
pending because the local `.venv` is missing the `tokenizers` dependency.
## 2026-08-11 — 900M SFT-v2 behavior checkpoint

The native `codexa-900m-sft-v2` checkpoint at optimizer step 6,000 loads with
the repository's legacy checkpoint fallback, but the bounded CPU behavior probe
does not pass reasoning quality. It generated a relevant photosynthesis
sentence, then answered `17 * 24` incorrectly and repeated the same answer.
The probe was stopped before the full suite because the host PyTorch build has
no CUDA support and CPU generation is very slow. This is a quality failure,
not evidence that the checkpoint is corrupted. Keep the checkpoint, tokenizer,
and model configuration until the next training/evaluation decision is made.

## 2026-08-14 — Preparation speed and acceptance recheck

The existing full preparation output was verified without rebuilding it. The
manifest records 13,700,814 input rows and 6,590.5 seconds elapsed, or about
2,078 rows/second, producing approximately 54 GB of processed JSONL. All four
processed output checksums match the manifest. A 200-batch host memmap probe
read 409,600 tokens at approximately 14.9M tokens/second.

The repository test run completed with 56 passing tests and 3 failures. The
failures are stale model/config expectations: tests still assert 921,773,568
parameters and older model keys, while `configs/1b.yaml` currently resolves to
934,356,480 parameters. The production mixed token-data validator also fails
before scanning because the older validator requires `eos_token_id`, which is
absent from that mixed manifest. The previous verified 900M CUDA BF16 benchmark
remains 7,700.7 tokens/second, or 8.51 seconds per 65,536-token optimizer
step.

Next gate: align the validator and stale fixtures, rerun all validation, then
perform a fresh visible 12-step production-shape benchmark before launching
any extended base-training budget.

## 2026-08-14 — Native inference speed diagnosis

The current native CUDA path was benchmarked on the RTX 4080 with the 900M SFT
v2 checkpoint. Generation measured approximately 117 tokens/second for a
32-token prompt, 61 tokens/second for a 128-token prompt, and 21 tokens/second
for a 512-token prompt. A 128-token prompt with 64 generated tokens measured
59 tokens/second. Inspection confirms that `generate_sequences` reruns the
full growing sequence for each new token; attention has no reusable KV-cache
state. The speed degradation with prompt length is therefore an implementation
bottleneck, not a training-data problem.

Next step: add an opt-in KV-cache implementation, preserve uncached behavior as
a reference path, test cached/uncached logit and stopping equivalence, and
repeat the same CUDA benchmark. Quantized GGUF comparison follows after a
local runtime is available or explicitly selected.

## 2026-08-14 — KV-cache milestone

The native inference path now supports cached attention keys and values while
retaining the uncached reference path. Twelve focused model/generation tests
pass, including cached/uncached logit equivalence and identical greedy output.

On the RTX 4080 with the 900M SFT v2 checkpoint, cached versus uncached speed
was 137 versus 117 tokens/second for a 32-token prompt, 124 versus 61 for a
128-token prompt, 96 versus 21 for a 512-token prompt, and 125 versus 59 for a
128-token prompt generating 64 tokens. The largest measured improvement is
approximately 4.5x for long context.

Next gate: run the full regression suite and measure first-token latency and
steady-state multi-turn chat speed before adding another optimization.

## 2026-08-14 — Torch compile gate blocked by environment

The opt-in `torch.compile` path was wired into one-shot generation and native
chat. The first real 900M CUDA attempt did not reach benchmarking: PyTorch
Inductor/Triton failed while compiling its CUDA helper because the Python 3.14
environment lacks `Python.h`. The compile path remains disabled by default.
The already validated KV-cache path is unaffected.

Next gate: use a Python environment with compatible development headers and a
working Triton compiler, then compare compiled and cached generation for output
equivalence, stopping behavior, and steady-state tokens/second.

## 2026-08-14 — Torch compile benchmark complete

The missing Python headers were installed and a temporary no-space compiler
alias allowed Inductor to build around the repository path issue. The compiled
900M CUDA path produced identical output to the KV-cache path, but required
approximately 45 seconds to warm up and reached about 126.6 tokens/second,
compared with 125.5 tokens/second uncached by compile. The approximately 1%
steady-state gain is not sufficient to make compilation the default.

Next step: evaluate weight-only quantization or another optimized inference
runtime. Keep the stable KV-cache path as the production default.

## 2026-08-16 — Base corpus storage cleanup

The obsolete raw and derived base-training datasets were moved to the desktop
Trash to reclaim working-tree storage while retaining the current
conversational improvement path. The preserved base replay data is
`data/tokenized/base-v1/mixed/` at approximately 27 GB, alongside the chat/SFT
datasets. The old base raw/processed copies and unused tokenized variants are
recoverable from Trash until it is emptied.

Next step: prepare only the data required by the next training phase after the
model-improvement plan is selected.

## 2026-08-16 — 100M checkpoint storage cleanup

The completed 61 GB `checkpoints/codexa-100m-native-v1/` lineage was moved to
the desktop Trash. Its HF and LM Studio exports remain available for comparison
or inference, while the 900M base and SFT checkpoints remain the active
improvement path.

The user subsequently confirmed permanent deletion of the trashed 100M native
checkpoint directory. Approximately 61 GB of checkpoint artifacts were
reclaimed; the 100M exports remain available.

## 2026-08-22 — Diagnostic gate complete; training paused

No new training was launched. The chat serializer and shifted assistant-only
targets passed 12 focused tests, including exact END-token alignment; the tiny
CUDA overfit reproduced all four exact responses. A streaming audit found
26,082 accepted UltraChat records and 29,186 accepted OASST1 records in the
balanced sample, with complete over-context conversations rejected rather than
truncated. The balanced 900M checkpoint was re-probed with a reset before every
prompt: it stopped on some responses but failed arithmetic, explanation,
correction, and list-quality gates. The next run remains blocked pending a
reviewed short-answer/correction/instruction-following dataset and recipe.

## 2026-09-10 — Next run planned: bounded conversational repair pilot

The next proposed run is `codexa-900m-sft-repair-v1`, starting from the
retained 900M base checkpoint. It will use a deterministic filtered
short-response dataset, 512-token pilot context, no base replay, a `1e-5` peak
learning rate, 500 warmup steps, and 2,000 optimizer steps. The launch remains
paused until the dataset audit, 100-record dry run, and CUDA overfit gate pass;
the final checkpoint must also pass isolated arithmetic, recall, correction,
list, explanation, stopping, and contamination probes before export or
promotion.

## 2026-09-10 — Repair SFT completed; conversational gate failed

The 2,000-step repair run completed and its checkpoint checksum passed. Native
probes improved arithmetic, genuine two-turn memory, and correction behavior,
but explanation prompts repeated `photosynthetic` text until the length limit
and an exact three-item tea request returned only two items. The checkpoint is
retained but not exported or promoted.

## 2026-09-10 — Repair failure diagnosed as template memorization

Additional probes showed exact anchor prompts succeeding while unseen variants
failed: `18 + 24` was rewritten as `19 + 24 = 42`, an unseen correction was
wrong, and unrelated list requests collapsed to the same fruit list. The repair
corpus contained only 71 train anchors built from four repeated templates per
category, while validation loss flattened as training loss continued down. The
checkpoint therefore improved memorized forms without learning robust task
generalization.

## 2026-09-11 — Five-hour 900M base continuation launched

The approved continuation initializes a fresh AdamW8bit optimizer from the
verified 900M base weights and runs 2,000 updates, approximately 131.07M new
tokens. A non-overlapping 39.30M-token tail is reserved from the retained base
stream for validation. The pre-run 16-batch baseline loss is `3.2042928`; range
and checkpoint tests pass. The required Kitty 100x22 viewer is attached, and
post-run checksum, loss comparison, and generation checks remain required.

## 2026-09-11 — Base continuation paused at step 7

The user-requested pause was applied with `SIGSTOP` to PID `236816`. The process
is retained in stopped state with 458,752 tokens completed; logs, data, model
weights, and the Kitty viewer remain intact. Resume requires explicit approval.

## 2026-09-11 — Five-hour continuation ended

The user ended the paused run. The process was terminated at optimizer step 13
after 851,968 tokens, before the first scheduled checkpoint at step 500. No new
model checkpoint exists; the original 900M base, datasets, tokenizer, exports,
and all prior model artifacts remain unchanged. The run log is retained and
marked interrupted.

## 2026-09-10 — Repair corpus preflight passed; launch held

The repair corpus was generated with 12,000 deterministic natural records and
96 behavioral anchors, producing 9,627 train and 2,469 validation records at a
512-token serialized limit. Both splits passed serialization and END alignment
audits with no prompt echoes. The CUDA 300-step chat overfit reproduced all four
diagnostic responses exactly and stopped on the native END token. The prepared
2,000-step repair launcher exists, but the actual training run was not started.

## 2026-09-10 — 900M repair SFT launched

The approved `codexa-900m-sft-repair-v1` run was launched from the checksum-
verified 900M base checkpoint with the prepared repair corpus, no base replay,
`1e-5` peak learning rate, 500 warmup steps, and 2,000 optimizer steps. The
required Kitty 100x22 viewer was opened. Behavioral acceptance and export are
pending completion.

## 2026-10-08 — Independent specialist Stage 1 infrastructure

A frozen EmbeddingGemma 2 programming-request classifier is being added outside
all native/base/SFT model lineages. The baseline remains three existing failures
(934M versus stale 921M assertions and missing position-embedding fixture fields).
The first full check after adding the specialist reports 91 passed, 5 skipped,
and those same three failures. The skips concern scikit-learn-dependent new tests;
the separate dependency environment is still installing. No generative run was
started and no artifact was removed. No human-reviewed intent dataset exists,
so classifier training and accuracy are intentionally unavailable.

## 2026-10-08 — Specialist Stage 1 infrastructure and real encoder verification complete

The separate frozen-embedding classifier infrastructure passes all 39 specialist
tests in its own environment. The existing generative suite still has the same
three baseline failures (92 passed, 5 specialist dependency skips). Actual
EmbeddingGemma 2 text-only loading passed on CUDA BF16 and CPU FP32 with
271,002,624 frozen parameters and normalized deterministic 768-dimensional
vectors. The final five-call warmed encoder median was 15.208 ms on the RTX 4080
and 46.008 ms on CPU; encoder CUDA peak allocation was 553.4 MiB. These are
encoder-only measurements. No intent dataset, task-trained head, task accuracy,
or baseline-versus-MLP task comparison is available. No generative run was
started and no existing model/data/log/checkpoint artifact was removed.


## 2026-10-08 — Stage 2 implementation begins

Decision: implement opt-in semantic memory, preserve Stage 1 classification and all existing artifacts; conditional SFT limited to 200 steps and 60 minutes including preflight/evaluation. Evidence: 39 specialist and 26 focused generative tests pass; full baseline 92 passed / 5 skipped / 3 existing failures; verified base and repair checkpoint checksums and base tokenizer fingerprint. Affected paths: src/memory, native chat, SFT infrastructure, tests and documentation. Next action: implement isolated retrieval worker, scoped storage, transactional context construction, then measure retrieval before training. No cleanup, commit or push.


## 2026-10-08 — Stage 2 retrieval and SFT data acceptance

Decision: use opt-in, user/conversation-scoped SQLite exact cosine memory with CPU FP32 encoder worker; threshold 0.68 selected only on development fixtures. Use 480-byte exact-text chunks to bound byte-fallback token counts without crossing environment tokenizers; preserve full raw turns separately. Evidence: held-out synthetic retrieval Recall@3 0.90, MRR 1.00, false-positive rate 0/5; 30 held-out queries. Original repair chat output comparison retrieves names but fails correction queries and remains repetitive/inaccurate; this is not general intelligence improvement. Affected paths: src/memory, scripts/evaluate_memory.py, logs/memory/retrieval-stage2-cpu-v1, logs/memory/chat-original-v1. Next action: remeasure optimized single-load native inference and both GPU models.

Decision: prepare general-chat-sft-v2 using only canonical existing UltraChat/OpenAssistant derivatives, not repeated repair anchors. Evidence: 182,892 training / 10,135 validation / 9,874 test conversations; source/root and normalized-message duplicate groups do not cross new splits. 26,497 overlength/role rejects, 42 normalized duplicates, 56 severe repetitions, 5 control-marker records rejected. Historic SFT exposure remains recorded; semantic near-duplicate/factual audits remain limitations. Affected paths: src/data/general_chat.py, src/conversational_training.py, data/processed/general-chat-sft-v2. Next action: conditional 200-step BF16/8-bit AdamW pilot from checksum/fingerprint-verified base; exact initialization restored after preflight; strict 14 GiB reserved VRAM gate, visible 100x22 Kitty viewer, 60-minute inclusive budget. No old artifact overwritten/deleted.


## 2026-10-08 — Stage 2 pilot preflight and first validation

Decision: proceed with bounded full-parameter SFT; the automatically launched 100x22 Kitty viewer attached before backward. Evidence: preflight assistant loss 2.262700, finite gradient norm 2.834639, verified parameter update, peak reserved 10,118,758,400 bytes. Restored original base weights and restarted optimizer before training. Baseline validation loss 2.652032; step-50 validation 2.460254. Training is in progress and this loss change does not establish conversational improvement. Runtime peak reserved 13,103,005,696 bytes remains under the 14 GiB gate. Affected paths: logs/codexa-900m-chat-stage2-pilot-v1 and separately named checkpoints/codexa-900m-chat-stage2-pilot-v1. Next action: complete up to 200 steps within the deadline, verify checkpoint, evaluate real outputs, and export only to a fresh experimental directory.

Hardware evidence: optimized native chat plus CUDA BF16 encoder reached 7,647 MiB total GPU usage, including desktop/driver allocations; CPU encoder mode reached 6,526 MiB. Native CPU-encoder median full reply latency: 0.427s memory off, 0.496s on; CUDA-encoder 0.429s off, 0.443s on in the 24 synthetic scenarios. These medians mix response lengths and are not a pure encoder speed comparison. Affected paths: logs/memory/chat-original-cpu-v2 and logs/memory/chat-combined-cuda-v1. Next action: retain CPU default to isolate generation/training and keep CUDA optional.


## 2026-10-08 — Stage 2 final acceptance

Decision: deliver opt-in semantic-memory native chat; retain the original repair checkpoint as the default and do not promote the experimental pilot. Evidence: 200 bounded SFT steps / 4,029,761 assistant tokens; validation 2.652032 to 2.379369, test 2.270733; actual conversation quality did not improve, expected whole values 0/8 pilot versus 2/8 original-with-memory with incorrect self attribution. END rates 10/24 pilot versus 14/24 original-with-memory. Frozen retrieval Recall@3 0.90 / MRR 1.00 on 30 synthetic held-out queries. Source/export smoke token IDs identical; original base/repair hashes unchanged. Full tests 114 pass / 5 dependency skips / 0 fail; isolated tests 58 pass including all original 39. Dataset root/group violations zero; 21 exact common diagnostic-prompt occurrences disclosed. Affected paths: documentation/training/STAGE2_RESULTS.md, documentation/reference/CONVERSATIONAL_MEMORY.md, ENCODER_CONDITIONING.md, logs/memory, the separately named pilot checkpoint and experimental exports/codexa-900m-chat-stage2-pilot-v1. Next action: broader human-held-out conversation evaluation and provenance-preserving memory-format supervision, not automatic longer training. Inclusive acceptance 58.25 minutes from launch; no cleanup, commit or push.

Implementation hardening: stable source labels prevent random IDs from changing prompts; raw turns and chronological references are preserved; clearing/deleting active memory also resets live history. Scoped embedding cache returns copies and clears on deletion/shutdown. Foreign database files are rejected unchanged. Future bounded launches check competing CUDA processes and use two-update steady-state preflight. The completed pilot ran one-update preflight and all 200 steps verified steady-state VRAM below 14 GiB.


## 2026-10-08 — Independent repository extraction

Decision: implement the approved sibling workspace architecture, preserving the original project and assets. Eight local working trees now hold seven packaged components plus integration; no commits, remote repositories, pushes or PRs were made. Evidence: source_manifest/history.bundle, compatibility.json, package-build metadata and offline validation records. Affected paths: all sibling repositories under 37-LLM-From-Scratch-Workspace; the original 31 directory is unchanged. Neutral native checkpoint loading is Architecture-owned, SFT protocol Tokenizer-owned, context composition Inference-owned, and encoder subprocess configuration replaces hardcoded environment/script paths. Original model equations/state keys and tokenizer fingerprint are preserved. No model promotion or production training was performed. Independent asset backup and real-model/GPU/encoder validation remain release gates. Next action: review VALIDATION.md and authorize publication separately.


## 2026-10-08 — Authorized GitHub README PR

Published existing committed history (66eec75) as main to golba98/LLM-From-Scratch-PyTorch-Codexa-v1 under explicit user authorization. Created README-only commit 300a8eb (first commit) on docs/codexa-v1-readme and PR https://github.com/golba98/LLM-From-Scratch-PyTorch-Codexa-v1/pull/1. Existing local README edits plus the requested heading are included; all other working-tree changes remain unpublished. README references unpublished integration files and sibling packages, disclosed in the PR. GitHub reported a bypassed main rule for an existing historical merge commit. No training launched.


## 2026-10-08 — Canonical workspace consolidation

Lifted integration history to the renamed workspace root and registered seven existing repositories as pinned submodules. Preserved source/Git recovery records and relocated historical assets to a single protected temporary store with before/after SHA-256 verification. Canonical environments and asset resolution no longer depend on project 31. NumPy remains independently implemented and packaged. Actual-checkpoint parity and bounded CPU/CUDA validations are recorded in documentation/migration/. Originals remain pending review and independent backup; no model lineage was promoted.


## 2026-10-09 — Restore eight sibling repository boundaries

Moved integration and its Git history into LLM-From-Scratch; made existing component Git metadata independent without changing HEADs or indexes. Shared recovery/build/validation evidence moved once to workspace-infrastructure. Full pre-change plan, ownership inventory and journal are retained there. Source imports use pinned sibling paths; environments retain separate profiles. Original 31 and 32 are untouched; independent backup remains unavailable. No commits, pushes, PRs, training runs or repository deletion authorized. Validation results are recorded in the migration audit report.


## 2026-10-09 — Publish sibling-layout review changes

User explicitly authorized saving and publishing LLM-Data and LLM-From-Scratch changes to PRs. Existing Data PR #3 and integration PR #2 are updated without merging. Original Data integration test is preserved byte for byte in its tracked documentation archive; central active coverage remains. No original project, component repository, checkpoint or dataset is deleted. Other component documentation changes remain locally saved and unpublished. Private assets, environments and local recovery evidence remain outside Git. See documentation/migration/PUBLICATION_2026-10-09.md for validation and preservation details.


## 2026-10-09 — Save every remaining workspace source/documentation change

User authorized PRs for all remaining unsaved files. Six remaining component README/AGENTS changes are saved on their existing review branches. Integration compatibility pins advance to their saved commits. Workspace-level README, instructions and shared-infrastructure README now have single canonical tracked owners in documentation/workspace with local links. Every public source/documentation change is reviewed and published; private assets/configuration/environments/recovery records remain preserved and excluded from public Git. No merge, repository deletion, original-project mutation or production training is authorized.


## 2026-10-09 — Local storage consolidation completed

Remaining PyTorch datasets/logs and shared recovery files are physically in `31-LLM Source`; staging is only a required compatibility symlink. Verified all 882 relocated entries, 13 NumPy checks, and two PyTorch path tests. Removed only the two archived NumPy top-level shortcuts, leaving exactly two 32-prefixed root folders. No model data was deleted or training launched; artifact space reclaimed is 0 bytes.


## 2026-10-09 — Read-only ChatGPT environment audit

Decision: record the verified current environment in `CHATGPT_CURRENT_STATE.md`; assess ten-hour training as NOT READY AS CONFIGURED, without changing or launching any pipeline.
Evidence: current component sources/Git/configuration, meta-device model count (934,356,480), small checkpoint/NPZ metadata, saved training/test reports, hardware snapshot, and missing production validation binary/index. Current large-checkpoint hashes and functional recovery were NOT VERIFIED; historical results are explicitly separated from fresh observations.
Affected paths: `CHATGPT_CURRENT_STATE.md`, `documentation/training/SESSION_DECISIONS.md`, `documentation/training/100M_PROGRESS_LOG.md`. No model, tokenizer, dataset, checkpoint, exporter, Git history, or system settings changed.
Next action: user reviews the handoff; separately authorize checkpoint recovery verification, validation resolution and a chosen training phase before any ten-hour run.

## 2026-10-09 — 934M conversational successor preparation (separate from 100M lineage)

The active 934,356,480-parameter learned-position model is being prepared in the modular 37 workspace. Full General Chat v2 mask/hash auditing passed for 202,901 conversations. All five candidate checkpoints failed the fixed 64-case production-entry behavior floor, despite successful integrity/strict restore checks. Repair is a provisional bounded-test initialization only. Production remains blocked pending measured training/recovery evidence and conversational improvement; no 100M or historical model artifact was changed.

## 2026-10-10 — 934M handoff complete; production BLOCKED

The modular successor, audited full chat pool, deterministic recovery, bounded 128-step pilot, evaluation artifacts and exact commands are documented in CODEXA_900M_CHAT_10H_READINESS.md. No ten-hour run or promotion occurred. Integration regression: 126 passed / 5 optional sklearn skips; component/controller regression: 56 passed. Conversational quality, evaluation exposure provenance and strict current GPU headroom remain launch blockers. This update does not alter the separate 100M lineage.


## 2026-10-10 — Two-hour session requested

Decision: user replaced the ten-hour proposal with a two-hour session and intends continuation tomorrow. Added isolated configs/chat-2h-v1.yaml; configurable budget now flows through specification, supervisor deadline and throughput projection. Evidence: prior readiness remains BLOCKED for conversational quality, exposure provenance and strict memory headroom; fresh GPU snapshot 13,156/16,376 MiB free. No training launched, no gate bypassed and no historical state overwritten. Affected integration paths: chat_preparation.py, chat_operations.py, chat_validation.py, test_chat_preparation.py, new configuration and CODEXA_900M_CHAT_2H_STATUS.md. Next action: resolve corrective-phase quality and memory gates, then establish a sealed two-hour manifest; tomorrow needs explicitly authorized additional budget and full-state continuity, not silent deadline reset.


## 2026-10-10 — Explicit two-hour experimental training authorization

User explicitly instructed immediate two-hour training after disclosure of blockers. Authorize isolated codexa-900m-chat-2h-v1 experimental phase, repair weights initialization (not exact resume), conservative LR5e-6, micro1/accum32, max1100 steps, 7200-second total deadline with800-second finalization reserve. Preserve original failed readiness gates honestly; permit only documented quality/provenance/headroom-benchmark exceptions, retaining live preflight memory checks, runtime1.5GiB headroom, finite-gradient/OOM recovery, signal/deadline watchdog and periodic quality/validation early stopping. Seven controller tests pass; integration wheel rebuilt/reinstalled without dependencies. No protected final acceptance during this experiment; no promotion. All historical assets preserved. Next action: launch, verify optimizer updates and visible viewer, record PID and deadline.


2026-10-10: Initial2h launch refused before run creation because conservative mixedaccum32 free-memory threshold exceeded live availability. Fresh experimental configuration accumulation8/effective8, micro1 unchanged, max4000updates, checkpoint400/eval800. Requiredfree11.8GB derived long-micro1 peak+headroom, not an accumulation8 benchmark claim. Live1.5GiB check and OOM recovery remain enabled; no existing run batching changed. Retry authorized2h experimental launch.


2026-10-10: Two-hour experimental phase RUNNING verified at optimizerstep5; finite loss2.43947, initial recoverable slot-0 checkpoint published, viewer attached/viewer_closed=false. Detached supervisorPID549510, workerPID549511; deadline2026-10-10T06:32:51.500343+02:00. User requested leave independent job running and will notify completion. No further automatic sessions authorized. Output: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/chat-training/codexa-900m-chat-2h-v1. Next action on user return: inspect report/recovery and compare development behavior; do not promote automatically.


2026-10-10 — Two-hour experiment completed and assessed: step4000 STEP_CAP, session6326.50seconds, 25,912,988assistanttokens/32,000conversations. Common validation2.229543→1.964267; literal task macro14.29→10.71%, END68.75→62.50%, severe degeneration35.94→40.63%. No behavioral improvement demonstrated; no promotion. Finalslot2 SHA374f3f1746b0ff0b24febc5176f3d20f5bb4ccadc3fb6aa24a120d1f31481918 rehashed and actualCPUload succeeded. Source/checkpoint paths and results in CODEXA_900M_CHAT_2H_RESULTS.md; nextaction investigate degeneration before new authorized phase, preserve all recovery artifacts.


## 2026-10-10 — Four-hour regression investigation authorized

User authorized investigation now with a four-hour ceiling. Begin isolated outputs/investigation/chat-regression-20261010; persist original UTCdeadline including interruptions. Diagnostics pair original/final outputs, inspect source supervision/END fractions, measure source-stratified teacher-forced content/END losses, local-data generations and cached/uncached generation parity for repair, balanced and completed2h weights. Planned bounded2x2 initialization/source-mixture pilots only if diagnostics justify them, with viewer/recovery enabled; no protected acceptance, promotion, new datasets or historical overwrites. Owner: integration chat_investigation.py/script/tests. Two focused paired-diagnostic tests pass. Nextaction inspect results before training ablations.


2026-10-10 investigation evidence: source-stratified48record diagnostics show repair→2h OASST contentloss2.53779→2.44063 but ENDloss1.44426→1.89861; UltraChatcontent2.37014→2.05361 and END0.94558→0.77378. Local12promptdegeneration41.67→66.67%, END50→25%. Current heldoutpopulation mayhave historicalexposure. Paired64devcases1taskimproved/3regressed,12newdegenerate/9fixed; differences small and taskmetric counts some incoherent factualkeywordanswers aspassing. Four bounded2x2 repair/balanced×proportional/balanced pilots justified; seed73 LR5e-6 micro1acc8 max512updates1200sec reserve180 each, retainrecoveries; onlydevelopmentprobes. NoENDweightingimplementedbefore ablationevidence.


2026-10-10: Historical repair-anchor audit found71trainanchors/26uniquecontents and25validationanchors/19uniquecontents; all19uniquevalidationanchorcontentsalsointraining. Eleven arithmetictraininganchors sum42(seven),13(two),17(two), matching the restricted answer totals observed on new arithmeticdevprompts. This strengthens template-memorization and historical-validation-exposure concerns; currentGeneralChatgroupisolation isseparateandunchanged. Full48recordprotocolinspection passed, no causalshift/assistantENDmask defect found. Preserveoldrepairdata; no historicalresults rewritten. Nextaction finishmatchedablationandcachedprefixnumericalcheck.


2026-10-10: Trainingowner quality_guard.py extracts existingbehavioral comparison with backward-compatible defaults and validated optionalthresholds; chat_run.py reads optionalregression_policy/patience. Proposednextphase configchat-behavior-guard-v1.yaml uses taskdrop5pp ENDdrop10pp degenerationincrease10pp patience2. Offlineprevious2hcounterfactual firststop1600; not a real new-policy run result. Activefourablationtrials retain already-imported legacywheelpolicy for faircomparison. Seven guard/pairedtestcases pass including malformedthresholds. No activeconfigchanged. Nextaction finishablations, rebuildaffectedwheels and regressiontest beforehand off.

### 2026-10-10T09:15:34.140640+02:00 — Completed conversational regression investigation

Decision: Retain all models without promotion; do not continue the original two-hour schedule unchanged. Five isolated 512-update pilots completed. END3 vs matched repair-proportional control improved END 57.8125%→75% and degeneration 45.3125%→29.6875%, with task macro unchanged at 14.2857%; instruction/arithmetic remained zero. All 4,096 sample identities matched exactly. No protected acceptance suite opened.

Evidence: Protocol audit passed 48 real records; UltraChat supplied 92.98% of original supervised tokens. Historical repair split shared 1,258 roots and 24 normalized complete contents; all 19 unique validation anchors were exposed in training. FP32 cache/full-prefix argmax matched all 27 checks; BF16 differed once near a tie. Five final checkpoint hashes verified; END3 native full payload reloaded. Integration tests 130 passed/5 optional specialist skips; final focused tests 65 passed. Installed changed wheels match source; pip check and three repository diff checks passed. Free disk 106.2 GiB, below existing 180 GiB production storage gate. No assets deleted.

Affected paths: LLM-Data repair_split.py, prepare_repair_sft.py and tests; LLM-Training quality_guard.py, token_objective.py, training.py, chat_run.py and tests; integration chat_investigation.py, investigation CLI/tests, chat_preparation.py and proposed guard config. Full report: documentation/training/CODEXA_900M_CHAT_REGRESSION_INVESTIGATION.md; artifacts: outputs/investigation/chat-regression-20261010.

Next action: Review report; validate END3 on a second seed/independent training window and review instruction coverage/evaluation provenance before authorizing a new training phase. Budget and disk policy must be explicit; no automatic restart, export or promotion.

### 2026-10-10T13:48:36.828918+02:00 — Begin opt-in checkpoint component storage implementation

Decision: Implement immutable SHA256 model/optimizer/recovery objects shared only within new experiment groups; legacy stays default, no GC/migration. Evidence: separate legacy copy manager and rotating conversational native writer; repeated final tensor payloads with distinct recovery metadata. Affected repositories: Architecture, Training, Inference, integration. Validation budget: 512 MiB, no full-size checkpoint copies or production training. Next action: bounded fixtures, exact continuation/fault tests and physical-space benchmarks; report full-size qualification limits.

### 2026-10-10T14:09:51.414081+02:00 — Checkpoint component backend validation handoff

Decision: Keep legacy default; optimized immutable experiment-group backend requires additional validation. Implemented lossless model/optimizer/recovery components, atomic manifests/indexes, native compatibility, group references, no GC/migration and storage planning retaining the 180 GiB reserve.

Evidence: Installed-wheel suites 218 passed/5 sklearn skips/10 existing return-value warnings; CPU stochastic continuation bitwise equal. Synthetic unchanged-save growth reduced 99.8376%; refined changed save 72.491 vs 72.547 ms. Load regression 10.68%, longer sample 16.78%, so latency gate failed. Full-size CUDA/8-bit optimizer qualification not performed. Historical base 5,636,066,645 bytes loaded read-only in FakeTensorMode after checksum verification. 1,331 historical file metadata records unchanged; validation allocation 331,091,968 bytes; free disk 105.74 GiB below 180 GiB.

Affected paths: Architecture component_checkpoint.py/checkpointing.py; Training optimized_recovery.py/recovery.py/checkpointing.py/chat_run.py/preflight.py/CLI/tests; Inference native readers/export readers; integration preparation/operations/benchmark/config example/tests. Report: documentation/training/CODEXA_CHECKPOINT_STORAGE_OPTIMIZATION.md. No training or historical artifact cleanup.

Next action: Profile loader overhead and separately authorize bounded full-size qualification with adequate storage. No default activation, migration, pruning, production run or model promotion.

### 2026-10-10T14:31:43.204851+02:00 — Begin checkpoint storage v2

Decision: Extend component-checkpoint-v1 for fresh stores only. Selected six snapshot cap with safe stopping, final reservations, no GC; legacy remains default. Profile loader without weakening integrity and benchmark bounded nonzero fixtures. Evidence: v1 exact CPU recovery passed but append-only storage and +10.68–16.78% restore regression remain; current free space ~105.74 GiB fails unchanged 180 GiB floor. Affected owners: Architecture/Training/integration, compatibility readers as necessary. Next action: implement policy/planner/leases/quota and tests; full-size GPU work remains blocked and unauthorized.

### 2026-10-10T15:00:53.814832+02:00 — Checkpoint storage v2 handoff

Decision: Legacy remains default; new bounded component stores use six snapshots, hard byte accounting, per-run final reservations and safe stopping, without GC. Implemented dependency/refcount dry-run planning, mapped storage reader leases, explicit previous fallback and verified loader optimization.

Evidence: Final installed regression 231 passed/5 sklearn skips/10 existing warnings. Refined 50-repeat nonzero restores: 12 MiB legacy 6.369 ms vs direct 6.660/store 6.758 ms (+4.58/+6.12%); 96 MiB 48.227 vs47.982/48.072 ms (-0.51/-0.32%). Both benchmark sets pass <=10%. CPU stop/final-reservation and exact continuation tests passed. Historical base SHA/FakeTensorMode load passed; 1,472 historical file metadata records unchanged. V2 outputs ~454 MB under512MiB. Current free ~105.31GiB fails180GiB; conservative bounded934M quota needs194.006GiB. Full-size GPU work not started.

Affected owners: Architecture component_checkpoint.py; Training checkpoint_retention.py, optimized_recovery.py/recovery.py/checkpointing.py/preflight.py/chat_run.py/baseCLI and tests; integration preparation/benchmark/planCLI/new bounded config fragment. Report: documentation/training/CODEXA_CHECKPOINT_STORAGE_V2.md. V1 report and historical artifacts preserved.

Next action: Review stopped-at-cap semantics; provide sufficient storage and separate bounded full-model CUDA/bitsandbytes/throughput qualification authorization. No cleanup, GC, default activation, long training or automatic model promotion.

### 2026-10-10T15:04:08.872180+02:00 — Final v2 concurrency acceptance correction

Correction/evidence: The first final 231-test run had 230 passes and one concurrent-initializer race failure. Corrected the shared-store writer to wait read-only on an existing initializer lock before rejecting temporary marker publication. Thirty simultaneous fresh-store writer trials passed. Rebuilt Architecture wheel; final installed suites now 231 passed / 5 skipped / 10 warnings (7.81s). Installed source identity checks passed and 1,472 historical files remained unchanged. Report updated with this failed-run evidence; no historical artifacts modified or GC performed. Next action remains storage-gated, separately authorized full-size qualification.

## 2026-10-10 — V3 checkpoint engineering milestone

Continuous six-state synthetic rotation and CPU continuation are implemented.
Reclamation has executed only in task-created disposable fixture namespaces.
Matched small-fixture restore gate passes; GPU/storage promotion gates remain open.
See CODEXA_CHECKPOINT_STORAGE_V3.md for final measured acceptance results.

## 2026-10-10 — Checkpoint storage V3 acceptance evidence

Decision: REQUIRES ADDITIONAL VALIDATION. Legacy remains the production default; no real-store reclamation grant was issued, no historical cleanup or full-model GPU workload ran. V3 extends component-checkpoint-v1 with new-store-only continuous retention, protected dependency roots, reader admission/leases and independently authorized journaled reclamation.

Evidence: final installed restore medians passed the 10% CPU gate (+7.26% at 12 MiB, -3.09% at 96 MiB); 62-publication fixture retained six states. Save+rotation median was 204.356 ms versus legacy 66.675 ms, requiring representative profiling. Latest aggregate repository invocations: 304 passed, 10 optional skips, zero final failures; initial Specialist import failures and warnings remain documented. Installed package/source byte checks and pip check passed. Historical metadata checks covered 1,540 + 35 files with zero changes. Fixture-only reclamation occurred in task-created namespaces.

Affected paths: Architecture component_checkpoint/checkpoint_inspection; Training retention/reclamation/optimized recovery/chat controller/preflight; integration operational scripts, qualification harness, experimental configuration and CODEXA_CHECKPOINT_STORAGE_V3.md. Artifacts: outputs/checkpoint-storage-v3-validation.

Next action: obtain at least approximately 77.9 GiB additional actual free capacity to satisfy the unchanged 180 GiB reserve, sufficient host RAM and separate bounded GPU authorization; then qualify actual 934M AdamW8bit recovery, save/load latency, throughput and representative continuous rotation. No production reclamation issuer is supplied.

## 2026-10-10 — V3.1 measured checkpoint-boundary optimization

Decision: preserve V3 format and durability barriers; optimize fresh inode traversal and completed-status parsing only. Original 120-publication CPU profile attributed 15.206 s to unknown-file discovery, 5.545 s to accounting and 6.567 s to fsync (inclusive stage times overlap). New traversal rejects symlinks without repeatedly resolving every ancestor. Completed status parsing is reused only while fresh device/inode/size/mtime/ctime identity matches, bounded to 4096 records; graph/deletion integrity is never cached across transactions. GPU harness comparisons now compare bytes rather than numerical torch.equal, and record host RSS.

Affected paths: Architecture component_checkpoint.py; Training checkpoint_reclamation.py/tests; integration qualification harness, profiling wrapper and tests. Evidence: outputs/checkpoint-storage-v3.1-validation. Historical storage/device investigation remains read-only; no real-store grant or GPU execution. Next: complete 500-publication timing, final regressions and capacity/archival readiness report.

## 2026-10-10 — V3.1 performance and readiness acceptance

Decision: REQUIRES ADDITIONAL VALIDATION; legacy default and disabled real reclamation unchanged. Matched 12 MiB save+rotation improved 210.910 to 163.061 ms at publication 62 (22.7%); CPU median 177.571 to 129.145 ms. Small-state 500-publication retention passed with six states, exact latest/previous/best recovery and quota accounting. Restore +7.04%/−1.13% versus legacy at 12/96 MiB passes CPU gate. Fresh audit accounting remains O(history); near-legacy save performance and full-model throughput are unresolved.

Evidence: final required-owner rerun 242 passed/5 skipped; aggregate current repository invocations 307 passed/10 optional skips/zero failures. Original development reporting failures are retained. Wheels/source identities and pip check passed. Historical metadata: 1,575 protected files, zero changes. New persistent validation approximately 228.55 MiB; only disposable fixture reclamation occurred. Final hardware plan reports 99.469 GiB free, 80.531 GiB below unchanged 180 GiB reserve, 15.560 GiB available RAM below 16 GiB gate; no GPU execution authorization.

Affected paths: Architecture fresh store traversal/accounting, Training bounded status cache/unknown-file traversal, integration bytewise GPU state checks/RSS/profile script/tests/example fragment and CODEXA_CHECKPOINT_STORAGE_V3_1.md. Next action: independently verify archive destination/dependencies if separately approved, obtain actual capacity and RAM headroom, then explicitly approve and execute bounded 934M recovery/performance qualification. No historical archival/copy/removal or real-store execution was performed.

### 2026-10-10 — V3.1 final byte-edge verification

Initial added edge test incorrectly expected NaN checkpoint storage; the existing finite-state guard rejected it (101 passed/1 failed, log preserved). Corrected test verifies signed-zero/finite-subnormal bytes and nonfinite rejection without losing prior recovery. Final Training rerun: 102 passed/2 warnings; latest aggregate repository invocations: 308 passed/10 optional skips/zero final failures. No implementation safety guard was weakened. Decision remains REQUIRES ADDITIONAL VALIDATION. Next action remains resource restoration and separately approved GPU qualification, not production deployment.

## 2026-10-10 — Authorized selective cleanup started

Target 200 GiB actual free space; deletion restricted to five user-allowlisted experiment areas. Source assets, END3, original two-hour recovery, deterministic recovery and corrective-02 active roles remain protected. New lossless initializer exports and CPU smoke/bitwise equivalence are prerequisite gates. Evidence: outputs/storage-cleanup/20261010-controlled-01. No training or real-store garbage collection is authorized.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/repair-proportional/checkpoints; 13 explicitly inventoried files, 20645732253 apparent bytes. Actual observed free-space increase 15853170688 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/repair-balanced/checkpoints; 13 explicitly inventoried files, 20645732219 apparent bytes. Actual observed free-space increase 19481157632 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/balanced-proportional/checkpoints; 13 explicitly inventoried files, 20645732272 apparent bytes. Actual observed free-space increase 21679349760 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/balanced-balanced/checkpoints; 13 explicitly inventoried files, 20645732238 apparent bytes. Actual observed free-space increase 20025335808 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-00/checkpoints; 7 explicitly inventoried files, 9373588623 apparent bytes. Actual observed free-space increase 8742764544 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-01/checkpoints; 7 explicitly inventoried files, 9373588623 apparent bytes. Actual observed free-space increase 9305767936 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/recovery/recoverable; 7 explicitly inventoried files, 11272459637 apparent bytes. Actual observed free-space increase 11270492160 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled retired checkpoint cleanup

Retired /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-1/checkpoint.pt; 3 explicitly inventoried files, 5636390972 apparent bytes. Actual observed free-space increase 0 bytes. Selected initializer exports passed CPU native/export smoke and bitwise tensor checks; deterministic replacement recovery retained. Evidence: /home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/storage-cleanup/20261010-controlled-01. Next: verify protected assets and readiness; no training launched.

## 2026-10-10 — Controlled cleanup complete; qualification remains blocked

Removed 23 retired checkpoint payloads, 46 exclusively owned sidecars and seven inactive recovery indexes, all inside the five authorized areas; full paths/hashes in outputs/storage-cleanup/20261010-controlled-01/deleted-artifacts.json. Removed apparent bytes 118238956837; measured net free-space increase 109430419456 bytes, from 97.981609 to 199.896629 GiB after two verified selected native exports. Observed 200.147259 GiB before later background allocations; unchanged 180 GiB statfs gate passes. All 25 protected native files hash/load, 1059 protected metadata identities, data/tokenizer checks and ten live distinct recovery references passed; repository invocations 282 passed/5 skipped, no final failure. Legacy/default inference unchanged; no training, real GC, Git push or model promotion. Prepared seed-74 20-minute END3 replication config/operator, not authorized. Full-model GPU qualification remains blocked by 16 GiB RAM gate and separate approval; extended training also blocked by quality/provenance acceptance. Report: documentation/training/CODEXA_CONTROLLED_CLEANUP_TRAINING_READINESS.md. Next: obtain resources and explicit bounded qualification/experiment approval without changing reserve or historical assets.

## 2026-10-10 — Controlled cleanup complete; qualification remains blocked

Removed 23 retired checkpoint payloads, 46 exclusively owned sidecars and seven inactive recovery indexes, all inside the five authorized areas; full paths/hashes in outputs/storage-cleanup/20261010-controlled-01/deleted-artifacts.json. Removed apparent bytes 118238956837; measured net free-space increase 109295476736 bytes, from 97.981609 to 199.770954 GiB after two verified selected native exports. Observed 200.147259 GiB before later background allocations; unchanged 180 GiB statfs gate passes. All 25 protected native files hash/load, 1059 protected metadata identities, data/tokenizer checks and ten live distinct recovery references passed; repository invocations 282 passed/5 skipped, no final failure. Legacy/default inference unchanged; no training, real GC, Git push or model promotion. Prepared seed-74 20-minute END3 replication config/operator, not authorized. Full-model GPU qualification remains blocked by 16 GiB RAM gate and separate approval; extended training also blocked by quality/provenance acceptance. Report: documentation/training/CODEXA_CONTROLLED_CLEANUP_TRAINING_READINESS.md. Next: obtain resources and explicit bounded qualification/experiment approval without changing reserve or historical assets.

## 2026-10-10 — Conversational readiness preparation

Implemented streaming data audit, proposed 576-record curriculum, 240-case strict development scoring, finite token-aware mixture, source/END validation diagnostics and independent best-validation recovery role. Fresh 25 checkpoint SHA/native-load checks and 1,059 unchanged protected metadata identities passed. Final affected tests: Training107, Data25, integration159/5 optional skips; unchanged Architecture6/Tokenizer6/Inference1 passed. No GPU934 job, deletion, initializer promotion or training. Disk passes 180GiB; host RAM/approval and semantic review block qualification and pilots. Evidence/report: CODEXA_934M_PRETRAINING_READINESS.md and outputs/readiness/codexa-934m-20261010-v1. Next: independent human review and precisely approved baseline/GPU qualification after resource gates; no ten-hour launch.


## 2026-10-11 — Final qualification execution

Completed five protected candidates on the frozen240-case development suite within the original3600-second deadline (1240.992s); no candidate passes strict entry gates, so no initializer selected. Actual934M AdamW8bit CUDA recovery completed eight trajectory updates plus four replay calls in135.770s: bitwise continuation passed, throughput +0.169% passed, optimized restore +16.886% failed. Legacy remains default.156 curated reviews remain human-pending; no dataset admission or pilot launch. Final disk falls below180GiB reserve; no cleanup performed.1059 protected metadata identities and preserved proposal hashes unchanged. Evidence: outputs/qualification/codexa-934m-20261011-v1; report CODEXA_934M_FINAL_QUALIFICATION_MODEL_SELECTION.md. Integration evaluator/orchestrator/review modules and focused tests changed; final169 passed/5 optional skips. Next action: blinded human factual/retention review, then resolve performance/capacity before any separately approved GPU retry or pilot.


## 2026-10-11 — Foundational diagnosis and matched-format preparation

Reviewed the original base10000-step history and historical Git sampler:655.36M targets,0.7014 tokens/parameter; late train/validation losses still improve. Reconstructed320000 unique windows with saved cursors;956 overlap the later continuation validation range, which must not be called historically unseen. Bounded256-document read-only sample round-trips exactly and has EOS but no chat role/END IDs. Prepared24-concept/5-format/5-checkpoint inference diagnostic (600 generations,3600s,zero updates) at outputs/diagnosis/codexa-934m-foundation-20261011-v1/job-v2; no approved grant exists and fresh free169.104GiB fails180GiB gate. Original alias map appeared in inspection; fresh540-response anonymous review reuses existing outputs, without new generations or fabricated review. Legacy remains default; optional optimized restore failure unchanged. Integration diagnostic/CLI/review admission guard and focused tests implemented; final184 passed/5 optional skips.1059 protected metadata identities and previous proposal/comparison hashes unchanged. No cleanup, model changes or GPU training. Decision: INSUFFICIENT EVIDENCE — TARGETED DIAGNOSTIC REQUIRED. Next: human review and separately approved diagnostic after capacity passes. Report: CODEXA_934M_FOUNDATIONAL_CAPABILITY_DIAGNOSIS.md.

## 2026-10-11T06:52:59.240299+00:00 — Frozen foundational diagnostic approval and blocked execution

Decision: record the user's “Implement the plan.” instruction as approval for frozen inference-only job `3fbe4da47492667a63a8f5dcaba907058b126542a6f0318b48c34a918eb3348b`; retain all mandatory gates. Evidence: `outputs/diagnosis/codexa-934m-foundation-20261011-v1/execution-preflight-20261011T065259240299Z/preflight.json`, matching plan/suite/code/tokenizer, 1059 protected metadata identities unchanged. Existing validator refused before model loading: disk 169.100136 GiB, deficit 10.899864 GiB. Affected paths: `outputs/diagnosis/codexa-934m-foundation-20261011-v1/job-v2/approval.json` and new compact execution-preflight evidence. No diagnostic execution, cleanup, model changes or budget start. Next action: separately satisfy storage capacity, then recheck and run the existing approved 3,600-second inference job.

## 2026-10-11T07:02:07.037595+00:00 — Preserve-and-add-capacity strategy; diagnostic remains blocked

Decision: implement the user's selected preservation strategy without deletion or archival. Evidence: `outputs/diagnosis/codexa-934m-foundation-20261011-v1/capacity-status-20261011T070207037595Z/status.json`; free 169.101170 GiB, additional 10.898830 GiB to mandatory gate and 20.898830 GiB to selected target; RAM/VRAM pass; frozen plan/suite/code/tokenizer/approval match; 1059 protected metadata identities unchanged. Affected paths: only this new compact evidence/report and session/progress entries. Existing approval and frozen job unchanged; no budget started, cleanup, training or inference job. Next action: provide additional actual filesystem capacity, then recheck and execute approved inference only.

### 2026-10-11 — Offline foundational-pretraining preparation

Analyzed existing 1,200 chat generations without new inference. Evidence remains insufficient to choose foundational pretraining versus corrective SFT; the approved neutral-format diagnostic is still the next decision experiment. Created `outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared`: indexed remaining-position training view (12,162,572,288 targets), 256 held-out windows with explicitly limited historical/content isolation, exact finite pilot exposure, disabled 256-update fresh-optimizer proposal and prioritized genuine-human-review queue. No GPU, training, cleanup or acceptance-data use. LLM-Data owns the indexed view; integration owns offline preparation and tests. Data suite: 37 passed; targeted integration: 53 passed, three existing warnings. Initial wrong warmup test expectation failed and was corrected; evidence preserved. Rebuilt only the Data wheel without dependency upgrades. See `CODEXA_934M_FOUNDATIONAL_PRETRAINING_READINESS.md`. Next: restore capacity under separate scope, run the unchanged approved diagnostic when gates pass, obtain semantic review; do not launch the provisional pilot yet.

### 2026-10-11 — Codexa Native integrated into Ubume Dev

Decision: expose all five explicit 934M candidates through the existing development-only native provider; retain greedy evaluation default, legacy training/storage behavior, and no model tools. Evidence: all five full SHA256 identities match; actual Balanced/END3 provider streaming, cancellation and switching passed (~33–35 generated tokens/s, ~3.64 GiB peak reserved). Both failed the READY instruction, so integration is not model-quality promotion. Sequential Ubume suite 2,135 passed; CPU inference 16 passed; architecture/model/identity 6 passed; integration generation/memory 12 passed. Failed earlier suites and final RAM-blocked repeat are retained in outputs/integration/ubume-native-20261011-v1. A single-hash native identity check and incomplete-turn recovery are separately covered by focused CPU tests. No training/checkpoint writes/cleanup occurred.

Affected paths: Ubume Dev provider runtime/config, existing selectors/settings/commands/history/provenance, scripts and docs; LLM-Inference native bridge/callback/finite checks; LLM-Architecture checkpointing optional expected identity; integration ignored local registry and documentation/training/CODEXA_934M_UBUME_NATIVE_INTEGRATION.md. Complete file inventory: outputs/integration/ubume-native-20261011-v1/modified-files.json. Existing dirty work and NumPy assets preserved.

Next action: launch Ubume Dev with bun run dev:run, choose Codexa Native → Codexa PyTorch → an explicit 934M model, and perform exploratory chats when the 16 GiB RAM / 8 GiB VRAM gates pass. Formal diagnostic approvals and model promotion remain separate.

2026-10-11 completion addendum: final native/prompt/conversation-store recovery suite 18 passed; Biome and TypeScript checks pass. Task-created isolated PTY validation data (automatic local dev copies, no original assets) was disposed after identity checks; retained evidence is ~1.1 MiB. No historical storage cleanup occurred.

## 2026-10-11 — Codexa 934M five-hour foundational run started

Run `codexa-934m-foundation-5h-v1` launched under a systemd user unit with a Python supervisor:
- 1,792-update ceiling over 117.4M fresh-position targets.
- Pilot hyperparameters, fresh AdamW8bit and the original base initializer.

Baseline held-out weighted CE is 3.11296, and the first updates ran at ~8.0k tok/s.

On completion or failure, the supervisor will:
- Verify the final checkpoint.
- Notify on the desktop.
- Wake Codex once for analysis.

No further training is automatic. Status: `.venv/bin/python scripts/training_status.py`.

## 2026-10-12 — Foundation step-1792 checkpoint added to Ubume Dev

Added verified foundation step-1792 checkpoint to ubume-dev as codexa-934m-foundation-5h (SHA256 ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e). Preserved existing model choices; no training started.

## 2026-10-12 — Post-training generation investigation

Completed post-training generation investigation against verified baseline/final checkpoints; exact historical validation and 96 generation reproductions, complete selected-target EOS/source audit, 17 focused tests passed. Seven private reports and raw evidence: `outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256`. Confirmed absent conversational supervision and response-termination failure; isolated unused masked-cache offset defect. Checkpoints preserved; no new training, optimizer updates, production edits, commits or pushes.

## 2026-10-12 — Diagnostic navigation cleanup

One canonical report: `outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256/REPORT.md`. Evidence, figures and archived originals organized beneath it. Preserved all 109 original files and all 72 generation excerpts; hash/link/rebuild checks and four offline organization tests passed. No model/training changes.

## 2026-10-12 — Conversational SFT feasibility: NOT READY

Preparation-only conversational SFT feasibility is NOT READY. Single report: `outputs/preparation/codexa-934m-conversational-feasibility/20261012T051805Z-a5c8725c/REPORT.md`. Prepared 64 authored examples, reviewed 365 source conversations, preserved baseline/source artifacts and passed 10 offline tests plus optimizer/backward-guarded validation. Full source admission, reviewed validation population, sealed gold baseline and specialized runtime remain incomplete. Zero training updates; no launch or optimizer construction.

## 2026-10-12 — Masked KV-cache position defect repaired

Repaired masked KV-cache position defect in LLM-Architecture model.py (full [batch, past+sequence] mask contract; cumsum positions continue from cache). Final-checkpoint masked cached max logit error 5.794949 -> 1.907e-05; 48/48 historical greedy generations unchanged on final and baseline; 73 new tests. Evidence: `outputs/diagnostics/masked-kv-cache-repair/20261012T060900Z/REPORT.md`. Completed run code_identity no longer verifies (accepted). No training, optimizer, checkpoint changes or commits.

## 2026-10-12 — Masked KV-cache repair committed locally

Committed masked KV-cache repair locally (LLM-Architecture bdf3683, LLM-Inference 4758dee, LLM-From-Scratch 55812c6/a763027/79a75c6/fc022b3; no push). Historical env = .venv without bootstrap (code_identity still matches); current = source-first (tools/codexa-dev). Preserved 10 never-committed hash-bound sources read-only. Claude and Codex audit enforcement verified in the real workspace (9/9 each). Report: `outputs/diagnostics/post-repair-integration/20261012T095454Z/REPORT.md`. No training or checkpoint changes.

## 2026-10-12 — Phase 4 conversational SFT preparation complete

Phase 4 conversational SFT preparation completed with no training.
- Frozen data: 512 training and 32 validation records; manifest f5514dce.
- Sealed step-zero baseline: gold END NLL 18.16; 0/16 natural endings.
- Built and tested a non-runnable SFT job and runtime (prepare/evaluate/train-gated).
- Verdict and conditions: `outputs/preparation/codexa-934m-sft-phase4/20261012T145429Z-643f4379/SFT_PHASE4_READINESS_REPORT.md`.
- Zero optimizer constructions, zero updates, no launch.

## 2026-10-12 — Step 5B conversational SFT pilot executed (FAIL)

Ran the first conversational SFT pilot, `codexa-934m-sft-feasibility-pilot-v1`. The job is the Step 5B re-freeze f36f5478 (it supersedes Step 5A 924da9f3), approved by the user and launched through the OS supervisor.
- Result: 128/128 updates, 51,026 supervised tokens in 331 s; STEP_CAP; checkpoints at update 64 and the final update verified; foundation ca779a16 unchanged; nothing promoted.
- Gates: natural endings 0→11/16 and END NLL 18.16→1.14 PASS; reference CE +0.03 PASS; severe repetition 15→5/16 FAIL (limit 4); strict accuracy 0/7 FAIL; human review PENDING. Overall FAIL.
- Every metric plateaued from update 32.
- Reports: `outputs/diagnostics/codexa-934m-sft-pilot-v1-results/20261012T211243Z/CODEXA_SFT_PILOT_V1_RESULTS.md` and `TEN_HOUR_FOLLOWUP_RECOMMENDATION.md`. The recommendation: no 10-hour SFT; investigate. The foundation has seen about 773M tokens (about 0.8 tokens per parameter).

## 2026-10-12 — Foundation capability assessment and telemetry repair (no training)

Repaired per-update telemetry for SFT runs (append-only `metrics/telemetry.jsonl`, crash and resume safe; 9427fa0). Added job-type-aware supervisor reporting (`run_reporting`, `supervisor/run-summary.json`; fbcc5c0) and an inference-only capability probe (eccd16a).
- **Foundation, plain form:** gold answer ranked first on 26/56 items (chance 14.8, p=0.0007). Factual, copy and completion are well above chance; arithmetic, grammar and reasoning are at chance.
- **SFT:** added no plain-form knowledge (25/56). It installed a "The X is Y." answer template with P(END after a bare answer) ≈ 0, giving 0 exact answers.
- **SFT data:** 30% of targets restate the prompt; the constrained examples carry 1% of supervised tokens.
- **Regression:** 464 integration tests plus all component and audit suites pass; checkpoints unchanged.
- **Verdict:** READY TO PREPARE NEW FOUNDATION EXPERIMENT (10 h continuation of the 934M; not prepared or launched).
- **Reports:** `outputs/diagnostics/codexa-934m-foundation-capability/20261012T213231Z/{FOUNDATION_CAPABILITY_ASSESSMENT,TRAINING_TELEMETRY_REPAIR,NEXT_TRAINING_STRATEGY}.md`.

## 2026-10-13 — Step 6: 10-hour foundation continuation prepared and frozen (not authorized)

Prepared `codexa-934m-foundation-continuation-10h-v1`, job `6d9d8453fa504f15841b7e3b6a1b31cc0776462a40d19db717ee893a039e0ef6`. No training was run.
- **Initialization:** weights only from foundation ca779a16, with a fresh AdamW8bit optimizer and a fresh cosine schedule (warmup 20, 2e-5 to 2e-6).
- **Data:** the seed-42 permutation continued from position 57,344, giving 114,688 windows and 234.9M tokens. It is disjoint from the 5-hour run, from validation and from base-v1's reconstructed positions. Mix 61.10% FineWeb-Edu / 38.90% Wikipedia; full corpus checksum verified.
- **Budget:** 36,000 s with a 540 s reserve and a 3,584-update cap; the p95 plan is 31,687 s.
- **Sealed step-zero baseline** b9e0a920: CE 3.0132 (Wikipedia) / 3.1403 (FineWeb-Edu); plain top-1 126/280.
- **Engineering:** foundation telemetry (LLM-Training 933358c, 6d4acbc); expanded benchmark (v1 preserved, v2 with 32 per category).
- **Readiness:** 0 engineering failures. CONDITIONALLY READY: free disk is 43.7 MB below the 180 GiB launch gate.
- **Reports:** `outputs/preparation/codexa-934m-foundation-10h/20261012T223033Z/`.
