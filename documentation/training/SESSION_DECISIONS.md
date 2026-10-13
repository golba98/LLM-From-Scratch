# Codexa Session Decision Log

## 2026-08-21 — Native conversational quality probe failed

- Decision: Do not approve `checkpoints/codexa-900m-sft-v2/latest.pt` as a usable
  conversational model yet. Treat this as a behavioral-quality failure, not an
  inference-speed issue.
- Evidence: The native bridge loaded the checkpoint successfully on CPU, but
  CUDA was unavailable in the current PyTorch runtime (`torch.cuda.is_available()`
  returned `False`). A multi-turn probe against this 900M checkpoint produced a
  generic greeting that hit the 24-token length limit, then repeated the
  weather-app answer for an arithmetic prompt. Fresh isolated 900M probes also
  failed: `17 + 25` produced `1. Two numbers and a short calculation` and a
  three-bullet tea request produced unrelated exercise instructions; both hit
  the 16-token length limit instead of emitting `<|end|>`. A separate reset
  probe against the retained 100M SFT pilot showed the same class of failures:
  incorrect arithmetic/explanation, generic continuations, and unreliable
  stopping. The bridge also reports the hard-coded 900M model name for that
  older pilot, which should be corrected before using bridge metadata as proof.
- Affected paths: `checkpoints/codexa-900m-sft-v2/latest.pt`,
  `checkpoints/tokenizer-base-v1/tokenizer.json`, `scripts/native_chat_bridge.py`,
  and `src/native_chat.py`.
- Next action: Audit role-token serialization, assistant-only labels, checkpoint
  lineage, and EOS/`<|end|>` supervision with a deterministic 100–500-example
  overfit diagnostic. Require readable greeting, arithmetic, explanation,
  correction, multi-turn memory, and reliable stop behavior before any further
  checkpoint promotion or export.

## 2026-08-21 — Chat protocol overfit gate passed

- Decision: Accept the chat serialization and native stopping path as
  mechanically functional; move the investigation to the 900M SFT data,
  checkpoint lineage, and optimization recipe.
- Evidence: Added `scripts/run_chat_overfit.py` and its viewer launcher. The
  300-step CUDA diagnostic used the real 16,384-token tokenizer, four fixed
  user/assistant examples, assistant-only next-token labels, greedy decoding,
  and `<|end|>` stopping. Loss fell from `9.625757` to `0.0001216`; all four
  expected answers were reproduced exactly and all four generations terminated
  with `stop_token`. Focused regression coverage is `14 passed`.
- Affected paths: `scripts/run_chat_overfit.py`,
  `scripts/run_chat_overfit.sh`, `tests/test_sft_serialization.py`, and
  `scripts/native_chat_bridge.py`; diagnostic evidence is under
  `logs/chat-overfit-diagnostic-v1/`.
- Next action: inspect the 900M SFT records and training composition, then run
  a larger bounded SFT sanity experiment from the verified base checkpoint.
  Do not delete or replace the existing 900M checkpoints yet.

## 2026-08-21 — 900M SFT sanity recipe rejected

- Decision: Reject `codexa-900m-sft-sanity-v1` as a candidate model. Preserve
  its checkpoint for comparison and do not export or promote it.
- Evidence: The run loaded the checksum-verified
  `checkpoints/codexa-900m-base-v1/latest.pt`, completed 205 CUDA optimizer
  steps over approximately 5.06M tokens with finite loss, and wrote a valid
  checkpoint whose SHA-256 sidecar passes. After reload, multi-turn and
  reset-isolated probes collapsed into repeated agricultural-field text,
  failed arithmetic and explanation, failed the three-bullet tea instruction,
  and frequently stopped by the 48-token length limit. The final training loss
  was `1.376913`; numerical completion did not produce behavioral quality.
- Affected paths: `scripts/run_900m_sft_sanity.sh`,
  `checkpoints/codexa-900m-sft-sanity-v1/latest.pt`,
  `logs/codexa-900m-sft-sanity-v1/`, and the preserved base checkpoint.
- Next action: do not repeat the first-500-record recipe. Build a seeded,
  source-balanced and length-balanced diagnostic subset, reduce the SFT update
  aggressiveness or increase replay protection, and compare against the base
  checkpoint with fixed behavioral probes before another longer run.

This file records the decisions made during each Codexa development session.
Entries are append-only and include the reason, affected artifacts, and next
action.

## 2026-08-11 — Native SFT-v2 quality evaluation and retention gate

- Decision: Treat `checkpoints/codexa-900m-sft-v2/latest.pt` as loadable but not
  quality-approved for reasoning or conversational use. Do not delete any
  checkpoint, dataset, tokenizer, export, or log until the retention set is
  explicitly confirmed.
- Evidence: The checkpoint passed checksum/load through
  `src.native_chat.NativeChatEngine` on CPU. A bounded native probe produced a
  relevant photosynthesis explanation but truncated at the length limit, and
  answered `17 * 24` incorrectly with repeated text (`The multiplication by
  24 is 16`) before stopping. The generic evaluator also rejected this legacy
  checkpoint payload because it lacks `format_version`; the native loader's
  legacy fallback was used for the probe. The active Python installation is
  CPU-only despite an available RTX 4080, so full prompt-suite evaluation was
  not completed in this session.
- Affected paths: `checkpoints/codexa-900m-sft-v2/latest.pt`,
  `checkpoints/tokenizer-base-v1/tokenizer.json`,
  `scripts/evaluate_checkpoint.py`, `src/native_chat.py`, and
  `logs/codexa-900m-sft-v2/train_metrics.jsonl`.
- Next action: Preserve the model plus its tokenizer and architecture metadata;
  obtain confirmation of the exact disposable directories, then use reversible
  cleanup only after confirming no active process references them. Re-run the
  complete behavior suite with a CUDA-enabled PyTorch environment before any
  production-quality claim.

## 2026-08-03 — LM Studio compatibility and next training phase

### Decisions

1. The existing `codexa-100m-base-v1` checkpoint remains a native PyTorch
   training artifact and is not accepted as the LM Studio release model.
2. The GPT-2/Llama GGUF adapters are compatibility experiments only. They are
   not quality-approved because their runtime architecture does not exactly
   match the trained Codexa architecture.
3. The next 100M model will use an LM Studio-compatible architecture from the
   beginning: RMSNorm, SwiGLU, rotary position embeddings, and exact tokenizer
   mapping.
4. The next base phase will use the existing FineWeb-Edu/Wikipedia mixed corpus
   and its untouched validation splits.
5. Conversational SFT will use the prepared UltraChat and OASST1 datasets with
   assistant-only labels and a small percentage of base-language replay.
6. Every evaluation checkpoint will record complete fixed-prompt outputs and
   metrics in `documentation/training/100M_PROGRESS_LOG.md`.
7. This decision log will receive a dated entry after every future session.

### Evidence

- Native checkpoint completed 10,000 optimizer steps and processed
  655,360,000 tokens.
- Native validation loss was 3.61696 and perplexity was 37.22.
- LM Studio accepted the generated GGUF files, but generation was malformed
  and highly repetitive, confirming that loading alone is not a quality gate.
- The trained Codexa model uses learned absolute position embeddings while the
  Llama adapter uses rotary position embeddings.

### Next action

Implement and validate the native LM Studio-compatible model architecture
before starting the replacement 100M training run.

## 2026-08-03 — Native rotary architecture gate

### Decisions

1. Preserve the legacy learned-position mode for existing checkpoints and
   configs; add rotary mode as a separate selectable architecture.
2. Use `configs/100m-native.yaml` for the replacement training run.
3. Export native checkpoints through the Llama-compatible GGUF path rather than
   adapting an already-trained learned-position checkpoint.
4. Treat a clean llama.cpp/LM Studio model load and tokenizer metadata check as
   a pre-training gate.

### Evidence

- Rotary forward/backward smoke passed with 125,848,320 parameters.
- Legacy learned-position configs still load and run unchanged.
- A native rotary checkpoint exported to GGUF successfully with 111 tensors.
- Bundled llama.cpp loaded the GGUF and returned `{"status":"ok"}` without
  the previous special-token metadata warnings.

### Next action

Run the replacement 100M base training job from `configs/100m-native.yaml`,
then evaluate native and LM Studio generations before beginning conversational
SFT.

## 2026-08-03 — Native 100M training started

### Decisions

1. Start a separate run named `codexa-100m-native-v1`; do not overwrite the
   completed learned-position checkpoint.
2. Keep the same 10,000-step base-training schedule and mixed token corpus so
   the architecture comparison remains interpretable.
3. Expose the run through a dedicated 100x22 Kitty progress viewer with steps,
   speed, loss, tokens, elapsed time, and ETA.

### Evidence

- The native run reached optimizer step 13 successfully.
- Current speed is approximately 52,170 tokens/second on CUDA BF16.
- Current training loss is 9.7109 during early warmup.
- GPU utilization is active and no training error has occurred.

### Next action

Let the native run complete, then run the fixed evaluation suite and export the
accepted checkpoint to GGUF for an LM Studio generation comparison.

## 2026-08-03 — Viewer command corrected

### Decision

The documented viewer command now targets `codexa-100m-native-v1` through the
`CODEXA_TRAINING_RUN` environment variable. It attaches to the active native
run and cannot accidentally restart the legacy base run.

### Evidence

- `scripts/open_100m_training_viewer.sh` passes shell syntax validation.
- `commands/TRAINING_VIEWER.md` now opens the native run by default.

## 2026-08-03 — Native run completed and evaluated

### Decision

Accept `codexa-100m-native-v1` as the new compatible base checkpoint for the
next training stage, but do not call it conversationally ready yet.

### Evidence

- Completed 10,000 optimizer steps and 655,360,000 tokens.
- Validation perplexity improved from 37.22 to 30.89.
- Native GGUF loaded in LM Studio and generated without the prior protocol
  failure.
- Generation still shows severe repetition, so the quality gate remains open.

### Next action

Run the fixed output logger against this checkpoint, then continue base replay
or controlled training before assistant-only conversational SFT.

## 2026-08-03 — Stale LM Studio model removed from active session

### Decision

The old `codexa-100m-base-v1-lmstudio-llama` adapter is not to be used for
quality testing. LM Studio was unloaded from that model and the active model
was changed to `codexa-100m-native-v1-lmstudio`.

### Evidence

- The reported malformed output identified the old adapter model by name.
- LM Studio now reports `codexa-100m-native-v1-lmstudio` as the loaded Llama
  model.
- The old adapter remains on disk only as a historical artifact; it is not an
  accepted checkpoint.

## 2026-08-03 — Full LM Studio behavior test

### Decision

Use the native LM Studio model for future evaluation and treat the current
checkpoint as runtime-compatible but not behaviorally ready.

### Evidence

- Six LM Studio prompts completed without a runtime/protocol error.
- Prose continuation showed topic drift and repetition.
- Factual explanation, instruction following, arithmetic, and conversation
  failed.
- The complete transcript is recorded in
  `logs/codexa-100m-native-v1/lmstudio_test_2026-08-03.md`.

### Next action

Prioritize repetition/degeneration reduction through continued base training
and fixed evaluation before conversational SFT; preserve this transcript as the
first behavioral baseline.

## 2026-08-03 — Natural multi-turn chat test

### Decision

Do not expose the current native base checkpoint as a chat model. Require
conversational SFT and repeat the same multi-turn memory test after each SFT
checkpoint.

### Evidence

- Four-turn chat ignored the greeting and user-provided facts.
- The model failed to recall the name Maya and Cape Town.
- It ignored astronomy and emitted unrelated repeated health text.
- The complete conversation is recorded in
  `logs/codexa-100m-native-v1/multi_turn_chat_2026-08-03.md`.

### Next action

Implement the assistant-only SFT dataloader and train a first conversational
checkpoint from the prepared UltraChat/OASST1 data with base replay.

## 2026-08-03 — Conversational SFT pilot started

### Decision

Start a bounded 2,000-step conversational SFT pilot from the native 100M
checkpoint. Use 60,000 UltraChat conversations, 1,000 validation conversations,
and 10% mixed-base replay to reduce catastrophic forgetting.

### Runtime

- Viewer: `scripts/run_100m_sft_pilot.sh` in a 100×22 Kitty terminal
- Base checkpoint: `checkpoints/codexa-100m-native-v1/latest.pt`
- Metrics: `logs/codexa-100m-sft-pilot/train_metrics.jsonl`
- Pilot checkpoint: `checkpoints/codexa-100m-sft-pilot/latest.pt`

### Initial evidence

- Startup/import issue fixed before launch by making the repository importable
  from the script path.
- Optimizer steps are running; early loss is approximately 2.1–2.8 while the
  learning rate warms up.
- The viewer reports steps, tokens/second, loss, validation loss, elapsed time,
  and ETA.

## 2026-08-03 — Checkpoint retention and session logging policy

### Decision

Record every consequential update in this file and keep progress details in
`100M_PROGRESS_LOG.md`. After a successful training run, remove only that
run's disposable `.pt` checkpoints after export and native/exported smoke-test
validation. Preserve model exports, tokenizer files, datasets, logs, and any
active or recoverable run.

### Storage evidence

- Legacy base checkpoints: 62G.
- Native base checkpoints: 61G.
- Tiny overfit checkpoints: 1.7G.
- Active SFT checkpoint: 481M.
- Exported model files and tokenizer remain outside the cleanup target.

No deletion was performed in this update because the SFT pilot is still
running and the native checkpoint remains its recovery source.

## 2026-08-03 — Automatic training progress viewer

### Decision

Every future model-training launch must automatically open or attach the
100×22 Kitty progress viewer. Its progress-bar line must always include an
`ETA HH:MM:SS`; the viewer also reports steps, tokens/second, losses, total
tokens, and checkpoint path without requiring a follow-up request.

### Implementation

The SFT viewer is `scripts/view_100m_sft_progress.sh`, and the launch wrapper
is `scripts/run_100m_sft_pilot.sh`.

## 2026-08-03 — Complete two-source base corpus preparation after session restart

### Decision

Resume verification of the deterministic FineWeb-Edu plus Wikipedia preparation
stage after the interactive session restarted. The completed output is retained
and is not regenerated or overwritten.

### Evidence

- `data/processed/base-v1/base_preparation_manifest.json` exists and records
  `elapsed_seconds: 6590.489408666996` (1h 49m 50s).
- Wikipedia: 6,406,791 accepted documents from 6,407,814 rows; 1,023 exact
  within-source duplicates rejected; 31,957 validation documents.
- FineWeb-Edu: 7,043,681 accepted documents from 7,293,000 rows; 249,319 exact
  within-source duplicates rejected; 35,232 validation documents.
- The manifest records SHA-256 checksums for all 51 input Parquet files and all
  four output JSONL splits. Near-duplicate clustering remains explicitly
  deferred.
- Output size is approximately 54 GB under `data/processed/base-v1`.

### Affected paths

- `scripts/prepare_base_corpus.py`
- `data/processed/base-v1/`
- `documentation/planning/PHASE_PLAN.md`
- `documentation/planning/TRAINING_DATA_PLAN.md`

### Next action

Verify output checksums and split isolation, then run the tokenizer bake-off and
post-tokenization accounting. Do not launch base training until the tokenizer,
mixture, token budget, and production-shape benchmark gates pass.

## 2026-08-03 — Keep a completed preparation viewer visible

### Decision

The live preparation viewer ended with the process, which left no visible
100%-complete progress screen after the session restarted. Reopen a 100×22
Kitty terminal containing the final progress bar and output counts after a
completed preparation run.

### Evidence

- The preparation process and live dashboard are no longer running.
- The completed viewer reports 100%, 13,700,814 raw rows, 13,450,472 accepted
  documents, 01:49:50 elapsed, and the final manifest path.

### Affected paths

- `/tmp/codexa-base-prep-complete.sh`
- `data/processed/base-v1/base_preparation_manifest.json`

### Next action

Use the same persistent visible-viewer convention for tokenizer bake-off,
tokenization, benchmarking, and every future training launch.

## 2026-08-03 — Replace training viewer with compact completion dashboard

### Decision

Use one shared 100×22 Kitty dashboard for native base and conversational SFT
launches. It keeps the straight progress bar, percentage, `ETA HH:MM:SS`,
steps, token speed, losses, total tokens, and checkpoint path visible while
running and after completion.

### Evidence

- The native base log ends at 10,000/10,000 steps and has a completed latest
  checkpoint; no training process is running.
- The new viewer was syntax-checked and rendered the completed native run with
  `COMPLETE`, 100%, validation loss, token total, and checkpoint path.

### Affected paths

- `scripts/training_viewer.sh`
- `scripts/open_100m_training_viewer.sh`
- `scripts/run_100m_native_training.sh`
- `scripts/view_100m_sft_progress.sh`
- `scripts/run_100m_sft_pilot.sh`

### Next action

Use the completed native checkpoint only after the existing export and native
smoke-test acceptance checks are verified; do not launch another training run
from this viewer change.

## 2026-08-03 — Resume SFT pilot to its planned endpoint

### Decision

Resume `codexa-100m-sft-pilot` from its last saved checkpoint at optimizer step
1,500 and continue to the planned 2,000 steps. The immutable native base
checkpoint remains the source model and is not modified.

### Evidence

- `checkpoints/codexa-100m-sft-pilot/latest.pt` reports optimizer step 1,500.
- The prior metrics log reached step 1,617 before interruption.
- The resumed process is active and has appended new metrics; the shared Kitty
  dashboard is attached to it.

### Affected paths

- `scripts/train_conversational_sft.py`
- `scripts/run_100m_sft_pilot.sh`
- `checkpoints/codexa-100m-sft-pilot/latest.pt`
- `logs/codexa-100m-sft-pilot/`

### Next action

Wait for step 2,000, then verify the final checkpoint and run the SFT smoke and
evaluation gates before exposing any conversational checkpoint.

## 2026-08-03 — Restore the mandatory visible progress viewer

### Decision

Treat the Kitty dashboard as part of the training run, not an optional aid.
When the viewer disappeared while the SFT process remained active, reattach a
100×22 dashboard to the live PID immediately.

### Evidence

- SFT process PID 14205 remains active at optimizer step 1,587.
- The dashboard was restored with `scripts/training_viewer.sh` and points at
  the live metrics and checkpoint paths.

### Affected paths

- `scripts/training_viewer.sh`
- `logs/codexa-100m-sft-pilot/train_metrics.jsonl`

### Next action

Keep the viewer attached through step 2,000 and report its location with every
live training status update.

## 2026-08-03 — Restore the minimal training-monitor aesthetic

### Decision

Replace the dense box-drawing dashboard with the earlier minimal terminal
layout: strong title hierarchy, ASCII progress bar, compact metric rows, and
clear checkpoint/log sections. This avoids broken Unicode glyphs and preserves
the required 100×22 Kitty presentation.

### Evidence

- The updated viewer renders cleanly in a 100-column terminal and reports the
  live SFT run at step 1,721/2,000.
- Training process PID 14205 remains active; no model or dataset artifacts were
  changed.

### Affected paths

- `scripts/training_viewer.sh`

### Next action

Keep this shared viewer as the default for all future Codexa training launches.

## 2026-08-03 — Stack training telemetry in one aligned column

### Decision

Present steps, speed, training loss, validation loss, tokens, and elapsed time
as a straight vertical list below the progress bar. This matches the operator
workflow and avoids split-column scanning in the 100×22 terminal.

### Evidence

- The viewer syntax-checks successfully and renders all telemetry as aligned
  rows against the live SFT metrics.
- The dashboard was relaunched and remains attached to the active training PID.

### Affected paths

- `scripts/training_viewer.sh`

### Next action

Keep the vertical telemetry layout for completed and future training runs.

## 2026-08-03 — Save the approved viewer appearance

### Decision

Keep a checked-in visual reference for the current minimal dashboard so future
viewer changes preserve its hierarchy, spacing, and vertical metric ordering.

### Evidence

- The reference mirrors the dashboard currently attached to the live SFT run.
- It documents the ASCII bar, metric order, checkpoint/log placement, and
  100×22 terminal constraint.

### Affected paths

- `documentation/training/TRAINING_VIEWER_STYLE.md`
- `scripts/training_viewer.sh`

### Next action

Use the reference when modifying the viewer or adding another training
launcher.

## 2026-08-03 — Complete conversational SFT pilot

### Decision

Mark `codexa-100m-sft-pilot` complete at its planned 2,000 optimizer steps.
Retain the checkpoint as a recoverable artifact until export and native/exported
smoke tests pass.

### Evidence

- Final metrics report 2,000/2,000 steps and 15,784,215 processed tokens.
- Final training loss is 2.198 and validation loss is 2.003.
- No SFT training process remains active.
- `checkpoints/codexa-100m-sft-pilot/latest.pt` exists at 481 MB.

### Affected paths

- `logs/codexa-100m-sft-pilot/train_metrics.jsonl`
- `checkpoints/codexa-100m-sft-pilot/latest.pt`

### Next action

Run checkpoint integrity, conversational evaluation, and export smoke gates;
do not delete the checkpoint before those checks succeed.

## 2026-08-03 — Remove obsolete LM Studio entries and open native model

### Decision

Remove the obsolete GPT-2 and pre-native Llama model entries from the LM
Studio library, remove the unused `golba98/codexa-openai-adapter` plugin, and
leave the accepted native Llama export available. Keep repository checkpoints,
exports, and tokenizer artifacts until their required evaluation/export gates
are complete.

### Evidence

- Moved these exact obsolete library directories to the user Trash:
  `codexa-100m-base-v1-lmstudio-f16` and
  `codexa-100m-base-v1-lmstudio-llama-f16` (approximately 505 MB total).
- Moved the exact 17 MB adapter plugin directory to Trash.
- LM Studio now reports 17 models and no Codexa adapter; only
  `codexa-100m-native-v1-lmstudio` remains.
- Loaded `codexa-100m-native-v1-lmstudio` successfully through `lms`; it is
  available in the LM Studio UI/API.

### Affected paths

- `/home/k9-vortex/LMStudioLocalTest/lmstudio-community/k9-vortex/`
- `/home/k9-vortex/.cache/lm-studio/extensions/plugins/golba98/`
- `exports/codexa-100m-native-v1-lmstudio-f16.gguf` (preserved)

### Next action

Evaluate the conversational SFT checkpoint and export it separately before
replacing the native LM Studio model; do not delete recoverable checkpoints.

## 2026-08-03 — Remove disposable training artifacts after storage audit

### Decision

Preserve the cleaned/raw/tokenized corpora, the active native checkpoint, the
completed SFT checkpoint, tokenizer, and exports. Move only verified disposable
artifacts to the user Trash: the completed old base checkpoint, tiny-overfit
checkpoint, and three failed production-shape benchmark runs.

### Evidence

- Native checkpoint is actively loaded by LM Studio and remains untouched.
- Old base exports exist under `exports/`, and no process references the old
  checkpoint directory.
- Removed exact artifact sizes: 62 GB old base checkpoint, 1.7 GB tiny
  overfit checkpoint, and approximately 48 GB failed benchmark logs/checkpoint
  directories.
- Remaining live storage in this repository is approximately 146 GB data,
  61 GB native checkpoint, 481 MB SFT checkpoint, 403 MB logs, and 2.1 GB
  exports.

### Affected paths

- `checkpoints/codexa-100m-base-v1/` (moved to Trash)
- `checkpoints/tiny-overfit-base-v1/` (moved to Trash)
- `logs/production-shape-v1/` (moved to Trash)
- `logs/production-shape-v1-failed-20260802-evaluation/` (moved to Trash)
- `logs/production-shape-v1-failed-20260802-rerun/` (moved to Trash)

### Next action

Empty the user Trash when ready to reclaim the physical disk space; do not
remove the preserved corpora or active/recoverable model checkpoints.

## 2026-08-03 — Pass 900M-class production-shape gate

### Decision

Proceed with the 900M-class base run using the existing 16K-token data and
current 934,356,480-parameter configuration. The proven memory shape is BF16,
activation checkpointing, 8-bit AdamW, microbatch 1, and gradient accumulation
32. The production budget is 10,000 optimizer steps (655,360,000 tokens).

### Evidence

- The 12-step production gate passed at 7,700.7 tokens/second.
- Peak reserved VRAM was 12.92 GB with 3.83 GB headroom on the RTX 4080.
- Validation and fixed-prompt generation completed successfully.

### Affected paths

- `configs/1b.yaml`
- `logs/production-shape-900m-v1/production_shape_benchmark.json`
- `data/tokenized/base-v1/mixed/train.bin`

### Next action

Launch the 10,000-step run from random weights and monitor it through the
mandatory Kitty dashboard; preserve every resumable checkpoint.

## 2026-08-03 — Pause 900M-class base training

### Decision

Pause the live `codexa-900m-base-v1` process with `SIGSTOP` so it can resume
in-place without losing its current GPU state. Do not terminate the process.

### Evidence

- Process PID 33993 is stopped (`TNsl`) at optimizer step 418 and 27,394,048
  processed tokens.
- A durable atomic checkpoint exists at step 250 in
  `checkpoints/codexa-900m-base-v1/latest.pt` with checksum and completion
  metadata.
- The Kitty viewer remains attached to the stopped process.

### Affected paths

- `checkpoints/codexa-900m-base-v1/latest.pt`
- `logs/codexa-900m-base-v1/train_metrics.jsonl`

### Next action

Resume in place with `kill -CONT 33993`; if the process must be restarted,
resume from `checkpoints/codexa-900m-base-v1/latest.pt` at step 250.

## 2026-08-03 — Pause after next 900M checkpoint for interactive use

### Decision

Resume the stopped base process only through the next checkpoint, then pause
it again so GPU resources are available for the user. The process is now
paused after checkpoint step 500.

### Evidence

- Process PID 33993 is stopped (`TNsl`).
- Durable checkpoint `checkpoints/codexa-900m-base-v1/latest.pt` reports
  optimizer step 500 and has completion/checksum sidecars.
- Training remains recoverable without deleting any model artifacts.

### Affected paths

- `checkpoints/codexa-900m-base-v1/latest.pt`
- `logs/codexa-900m-base-v1/train_metrics.jsonl`

### Next action

The user can play games now. Resume with `kill -CONT 33993` when ready.

## 2026-08-04 — Resume 900M-class base training after logout

### Decision

The stopped process did not survive logout, so resume the base run from its
verified step-500 checkpoint. Start a fresh detached process with the standard
vertical Kitty dashboard attached.

### Evidence

- `checkpoints/codexa-900m-base-v1/latest.pt` reports optimizer step 500.
- The resumed process is active as PID 8719 with the expected config,
  optimizer, BF16 precision, and activation checkpointing.
- The viewer is attached as PID 8720.

### Affected paths

- `checkpoints/codexa-900m-base-v1/latest.pt`
- `logs/codexa-900m-base-v1/`

### Next action

Continue monitoring from step 500 toward the 10,000-step budget; preserve each
250-step checkpoint.

## 2026-08-04 — Complete and verify 900M-class base training

### Decision

Mark `codexa-900m-base-v1` training complete at 10,000 optimizer steps. Keep
the final checkpoint and all milestone checkpoints for evaluation and export.

### Evidence

- Final metrics: 10,000 steps, 655,360,000 tokens, training loss 3.1042,
  validation loss 3.3491.
- `latest.pt` reports optimizer step 10,000, format version 1, complete=true,
  and SHA-256 `808afec9923e3f00d91bcbef47643265ae5dbe6c5aeb4bd112c4e97be3d0a3a7`.
- No training process remains active.
- The interrupted temporary checksum artifact was moved to Trash; no temporary
  checkpoint files remain.

### Affected paths

- `checkpoints/codexa-900m-base-v1/latest.pt`
- `checkpoints/codexa-900m-base-v1/latest.pt.complete.json`
- `logs/codexa-900m-base-v1/train_metrics.jsonl`

### Next action

Run the fixed base-model evaluation suite, then export and smoke-test the
selected checkpoint before beginning conversational SFT.

## 2026-08-04 — Remove obsolete 900M benchmark checkpoints

### Decision

Keep the final 900M model, all resumable 900M milestones, native lineage,
training corpus, tokenizer, and SFT checkpoint. Remove only the disposable
production-shape benchmark checkpoint artifacts after the gate report was
recorded.

### Evidence

- The benchmark report, config, and evaluation JSON remain in
  `logs/production-shape-900m-v1/`.
- The exact nested benchmark checkpoint directory was 16 GB and is no longer
  needed for recovery or evaluation.
- The final 900M checkpoint and native model were not touched.

### Affected paths

- `logs/production-shape-900m-v1/checkpoints/` (moved to Trash)
- `logs/production-shape-900m-v1/production_shape_benchmark.json` (preserved)
- `checkpoints/codexa-900m-base-v1/` (preserved)

### Next action

Export and smoke-test the final 900M model. Only then evaluate deleting
redundant 900M milestone/optimizer checkpoints; retain at least one verified
recoverable checkpoint until the successor is accepted.

## 2026-08-03 — Conversational quality evaluation failed

### Decision

Do not expose either the native LM Studio export or the 100M SFT pilot as an
accepted chat model. The native export remains the loaded diagnostic model;
the SFT checkpoint remains preserved for further experiments.

### Evidence

- Native LM Studio probes repeated prompts, drifted into unrelated text, and
  failed simple factual, list, story, and memory prompts.
- Direct generation from the completed SFT checkpoint produced recognizable
  answers for photosynthesis and the Amina memory prompt, but repeated each
  answer many times and failed the requested conversational stopping behavior.
- The SFT checkpoint is a 100M pilot, not the approved 1B production model.

### Affected paths

- `logs/codexa-100m-sft-pilot/train_metrics.jsonl`
- `checkpoints/codexa-100m-sft-pilot/latest.pt`
- `logs/codexa-100m-native-v1/lmstudio_test_2026-08-03.md`

### Next action

Audit SFT serialization, assistant-only masks, EOS supervision, and stopping;
run a longer controlled SFT/replay comparison, then repeat deterministic chat
and base-regression gates before exporting any conversational model.

## 2026-08-04 — 900M base exported and redundant training artifacts retired

### Decision

Select the verified step-10,000 900M-class checkpoint as the immutable base
model handoff. Keep the final checkpoint and one recovery copy; remove only
intermediate milestone and staging artifacts after export and smoke validation.

### Evidence

- Native checkpoint generation completed with a deterministic factual
  continuation from `latest.pt`.
- GGUF export completed successfully and was parsed as 293 tensors with a
  GPT-2 architecture, 2,048 context length, and 16,384-token vocabulary.
- Export SHA256: `506bbd45becc4b711cd512f204260d3496b3d5c76aa966d276f476f7dd8e789d`.
- 84 redundant files/directories (approximately 215G) were moved to Trash,
  not permanently deleted; physical reclamation requires emptying Trash.

### Affected paths

- Preserved: `checkpoints/codexa-900m-base-v1/latest.pt`,
  `checkpoints/codexa-900m-base-v1/previous.pt` and integrity sidecars.
- Preserved: `exports/codexa-900m-base-v1-lmstudio-f16.gguf`.
- Moved to Trash: all 900M milestone files, `best.pt` and sidecar, and the
  intermediate HF export directories.
- Changed exporter metadata naming in `scripts/export_codexa_lmstudio.py` and
  `scripts/export_codexa_llama_lmstudio.py` to use the actual artifact name.

### Next action

Run the fixed base evaluation suite, then prepare tree-safe UltraChat/OASST1
SFT data. The exported base is not yet an accepted conversational model.

## 2026-08-04 — LM Studio catalog import verified; runtime load deferred

### Decision

Import the 900M GGUF into LM Studio for visibility, but do not replace the
working 100M native model or claim LM Studio inference acceptance yet.

### Evidence

- LM Studio indexed `codexa-900m-base-v1-lmstudio` as a 900M F16 GGUF with
  2,048-token context and the expected GPT-2 architecture.
- The llama-server runtime exited before becoming healthy when loading it.
- Native CUDA generation from the original checkpoint remains successful.

### Affected paths

- LM Studio catalog entry: `codexa-900m-base-v1-lmstudio`.
- Preserved export: `exports/codexa-900m-base-v1-lmstudio-f16.gguf`.

### Next action

Investigate a runtime-compatible exporter for Codexa's learned-position
architecture before exposing this checkpoint through LM Studio. Do not change
the architecture or silently substitute rotary embeddings.

## 2026-08-04 — 900M LM Studio load repaired

### Decision

Keep the trained architecture unchanged and correct the GGUF tokenizer metadata
only: llama.cpp expects `gpt-2`, not `gpt2`, for this ByteLevel BPE tokenizer.

### Evidence

- Direct llama-server diagnostics identified the original failure as
  `unknown pre-tokenizer type: 'gpt2'`.
- The regenerated GGUF reports `tokenizer.ggml.pre = gpt-2`.
- LM Studio successfully loaded `codexa-900m-base-v1-lmstudio` to 100% in
  1.26 seconds; it is idle at 1.47 GB with 2,048 context.
- The CLI chat wrapper rejects the base model's prompt as non-chat-native;
  this is expected before conversational SFT. Text-completion evaluation is
  the correct current interface.

### Affected paths

- `exports/codexa-900m-base-v1-lmstudio-f16.gguf`
- LM Studio catalog entry `codexa-900m-base-v1-lmstudio`

### Next action

Evaluate the loaded base model with completion prompts, then build and train
the approved conversational SFT datasets. Do not expose it as a chat model yet.

## 2026-08-04 — Base behavior and LM Studio fidelity check failed

### Decision

Do not treat the current LM Studio GGUF as an accepted model export. Preserve
the native checkpoint as the source of truth and defer public use of the GGUF.

### Evidence

- Native CUDA generation is executable but repeats the prompt heavily, which is
  a quality limitation of the unsupervised base checkpoint before SFT.
- LM Studio loads the GGUF, but direct completion produces corrupted output.
- The compatibility exporter maps Codexa RMSNorm and gated feed-forward blocks
  into GPT-2 LayerNorm and non-gated FFN tensors; this is not behaviorally
  equivalent even though the file is structurally loadable.

### Next action

Implement a faithful runtime exporter or use the native Codexa inference path.
Do not claim LM Studio output quality until an equivalence test against native
logits and fixed prompts passes.

## 2026-08-04 — 900M conversational SFT launched with memory-safe optimizer

### Decision

Begin the approved conversational SFT pilot from the immutable 900M base
checkpoint using the prepared UltraChat/OASST1-compatible serialization and a
controlled 10% base-language replay. Keep the base checkpoint unchanged.

### Evidence

- The first full-AdamW attempt reached optimizer step 1 with loss 2.5141 but
  exceeded the 16 GB GPU during step 2.
- The retry uses bitsandbytes AdamW8bit plus the existing activation
  checkpointing implementation; it is running successfully through step 9.
- Current measured peak allocated VRAM is approximately 11.98 GB, with no OOM.
- Current throughput is approximately 3,900–4,500 tokens/second and the live
  100x22 viewer is attached.

### Affected paths

- `scripts/train_conversational_sft.py`
- `scripts/run_900m_sft.sh`
- `logs/codexa-900m-sft-v1/`
- `checkpoints/codexa-900m-sft-v1/` (active run)
- Source checkpoint preserved at `checkpoints/codexa-900m-base-v1/latest.pt`.

### Next action

Let the 2,000-step pilot complete, validate its checkpoint and chat/base
regression metrics, then compare against a pure-SFT variant before selecting a
conversational successor.

## 2026-08-04 — 900M SFT paused safely for host switch

### Decision

Stop the active SFT process after a durable checkpoint so the host can switch
to Windows without losing recoverable training state.

### Evidence

- The active run reached optimizer step 285; the latest durable checkpoint is
  verified at optimizer step 250.
- `checkpoints/codexa-900m-sft-v1/latest.pt` is present at approximately 3.7 GB.
- The process was terminated after checkpoint verification; no training process
  remains active.
- Base checkpoint `checkpoints/codexa-900m-base-v1/latest.pt` is untouched.

### Affected paths

- `checkpoints/codexa-900m-sft-v1/latest.pt`
- `logs/codexa-900m-sft-v1/train_metrics.jsonl`

### Next action

Resume from `checkpoints/codexa-900m-sft-v1/latest.pt` after returning to
Linux. Do not resume from the immutable base checkpoint or delete this SFT
checkpoint.

## 2026-08-04 — 900M SFT resumed from step 250

### Decision

Resume the conversational SFT pilot from its verified step-250 checkpoint
after the host switch.

### Evidence

- Resume loaded `checkpoints/codexa-900m-sft-v1/latest.pt` at step 250.
- Training is active at step 257 with loss 2.5475 and approximately 3,300–4,100
  tokens/second.
- Activation checkpointing and AdamW8bit remain enabled; peak allocated VRAM
  remains approximately 11.98 GB.
- The live 100x22 training viewer is restored.

### Affected paths

- `scripts/run_900m_sft.sh`
- `checkpoints/codexa-900m-sft-v1/latest.pt`
- `logs/codexa-900m-sft-v1/`

### Next action

Continue to the 2,000-step pilot checkpoint, then run conversational and base
regression evaluation before selecting a successor.

## 2026-08-05 — 900M SFT pilot complete but conversational gate failed

### Decision

Preserve the completed 900M SFT checkpoint for analysis, but do not expose it
as the conversational model yet.

### Evidence

- The pilot completed at optimizer step 2,000 with training loss 2.0080 and
  validation loss 1.9259 (perplexity 6.86).
- Native greedy probe for a name question produced a relevant refusal-like
  opening but repeated `iphone` six times.
- Native greedy photosynthesis probe repeated the instruction instead of
  answering it and hit the generation limit.
- The checkpoint is checksum-protected at
  `checkpoints/codexa-900m-sft-v1/latest.pt`.

### Affected paths

- `checkpoints/codexa-900m-sft-v1/latest.pt`
- `checkpoints/codexa-900m-sft-v1/latest.pt.sha256`
- `logs/codexa-900m-sft-v1/train_metrics.jsonl`
- `scripts/generate.py` (supports inference from the pilot checkpoint format)

### Next action

Audit serialized prompt/label alignment, EOS supervision, truncation and
decoding before another SFT run. Compare pure SFT against controlled replay;
do not overwrite the immutable base or publish this pilot as chat-ready.

## 2026-08-05 — Chat stop-contract repair and SFT v2 preparation

### Decision

Treat `<|end|>` (the chat turn delimiter) as a required generation stop
sequence, and start the next conversational run from the immutable 900M base
with both approved conversational sources. Keep the failed v1 checkpoint for
comparison and do not overwrite it.

### Evidence

- The completed v1 checkpoint answered the correctly serialized Amina prompt
  cleanly when native generation stopped at token `<|end|>`.
- The existing CLI stopped only on `<eos>`, continued into subsequent turns,
  and produced the repeated output seen in LM Studio.
- The v1 LM Studio GGUF was structurally loadable but not behaviorally faithful:
  its compatibility exporter changes learned positions, RMSNorm, and the gated
  feed-forward path. The exporter now records the chat template for future
  exports, but LM Studio quality remains gated on an equivalence test.

### Affected paths

- `src/sft.py`
- `scripts/generate.py`
- `scripts/train_conversational_sft.py`
- `scripts/run_900m_sft.sh`
- `scripts/export_codexa_lmstudio.py`
- `tests/test_sft_serialization.py`

### Next action

Run `codexa-900m-sft-v2` for 6,000 steps with 30,000 examples per source,
validate multi-turn stopping natively, then export only if the chat and base
regression gates pass. Disable LM Studio web-search/tool mode during model
smoke tests because the model is not trained to emit tool-call protocol.

## 2026-08-05 — 900M SFT v2 paused safely for Windows switch

### Decision

Pause the active UltraChat-plus-OASST1 SFT v2 run after its latest durable
checkpoint so the host can switch operating systems safely.

### Evidence

- The v2 process was at optimizer step 521; its durable checkpoint is verified
  at step 500.
- `checkpoints/codexa-900m-sft-v2/latest.pt` is preserved and checksum-protected.
- The training process and its 100x22 viewer have been stopped.
- The immutable base and completed v1 SFT checkpoint remain unchanged.

### Affected paths

- `checkpoints/codexa-900m-sft-v2/latest.pt`
- `checkpoints/codexa-900m-sft-v2/latest.pt.sha256`
- `logs/codexa-900m-sft-v2/train_metrics.jsonl`

### Next action

Resume v2 from step 500 after returning to Linux. Do not delete this active
recovery checkpoint.
## 2026-08-05 — Resume conversational SFT v2 after host lag

- Decision: Resume `codexa-900m-sft-v2` from its highest durable checkpoint, `checkpoints/codexa-900m-sft-v2/latest.pt`, at optimizer step 1,750; do not resume from later log-only steps.
- Evidence: No training process is active; the checkpoint deserializes with a valid model state and `optimizer_step: 1750`; `sha256sum` produced `2364eaddc4668d077727ecf53b9dbb058e52852ef6151bad78537dcd6532e483`, while the prior sidecar was stale. The prior metrics reached step 1,836 and reported peak reserved VRAM of 12,920,553,472 bytes, consistent with the reported host lag.
- Affected paths: `checkpoints/codexa-900m-sft-v2/latest.pt`, `checkpoints/codexa-900m-sft-v2/latest.pt.sha256`, `logs/codexa-900m-sft-v2/train_metrics.jsonl`, `scripts/run_900m_sft.sh`.
- Next action: Relaunch the existing 6,000-step SFT command from the validated step-1,750 checkpoint and attach the visible 100×22 Kitty viewer; preserve all checkpoints and logs.
## 2026-08-05 — Resume confirmed at optimizer step 1,751

- Decision: Keep `codexa-900m-sft-v2` running from the validated step-1,750 checkpoint.
- Evidence: The resumed process is active as PID 6682; `train_metrics.jsonl` was truncated to 1,750 records and now contains a fresh optimizer-step 1,751 record with run ID `846dda94-8542-4eda-8b4b-8a92b5e08cb5`, training loss 2.5412, and peak reserved VRAM 11,070,865,408 bytes as reported by the metric. A separate Kitty viewer was launched with PID 6683 using the required 100×22 geometry.
- Affected paths: `logs/codexa-900m-sft-v2/train_metrics.jsonl`, `logs/codexa-900m-sft-v2/run_metadata.json`, `checkpoints/codexa-900m-sft-v2/latest.pt`, `scripts/train_conversational_sft.py`.
- Next action: Let the run continue toward step 6,000; validate the final checkpoint, export, and native/exported smoke test before any checkpoint cleanup.

## 2026-08-06 — Pause conversational SFT v2 for power outage

- Decision: Suspend the active `codexa-900m-sft-v2` process with `SIGSTOP` so the run can be resumed safely when power and connectivity return.
- Evidence: PID `6682` is stopped (`TNsl`); the required 100x22 Kitty viewer remains present and the latest durable checkpoint is `checkpoints/codexa-900m-sft-v2/latest.pt`, timestamped 2026-08-06 04:17:52. The latest observed metric reached optimizer step 5,117; no checkpoint or log files were deleted.
- Affected paths: `checkpoints/codexa-900m-sft-v2/latest.pt`, `checkpoints/codexa-900m-sft-v2/latest.pt.sha256`, `logs/codexa-900m-sft-v2/train_metrics.jsonl`.
- Next action: After power returns, resume PID `6682` with `kill -CONT 6682` only if its process and checkpoint are still present; otherwise relaunch from the validated `latest.pt` checkpoint and restore the 100x22 viewer.

## 2026-08-06 — Resume conversational SFT v2 after power restoration

- Decision: Continue `codexa-900m-sft-v2` from the paused process and preserved checkpoint.
- Evidence: `kill -CONT 6682` succeeded; PID `6682` is no longer stopped, and `train_metrics.jsonl` advanced to optimizer step `5118` at 2026-08-06 06:49:34 UTC. The 100x22 Kitty viewer remains attached as PID `8746`.
- Affected paths: `logs/codexa-900m-sft-v2/train_metrics.jsonl`, `checkpoints/codexa-900m-sft-v2/latest.pt`, `documentation/training/SESSION_DECISIONS.md`.
- Next action: Let the run continue toward step 6,000, then validate the final checkpoint, export, and native/exported smoke test before cleanup.

## 2026-08-06 — Conversational SFT v2 completed and staged in LM Studio

### Decision

Mark the 6,000-step conversational SFT v2 training phase complete and stage
its exported GGUF in LM Studio under the separate key
`codexa-900m-sft-v2-lmstudio`. Preserve the native checkpoint and all source
data while LM Studio behavioral compatibility remains unresolved.

### Evidence

- `logs/codexa-900m-sft-v2/train_metrics.jsonl` ends at optimizer step 6,000,
  103,459,920 total tokens, training loss 1.5768, and validation loss 2.0316.
- `checkpoints/codexa-900m-sft-v2/latest.pt` passes its SHA-256 sidecar check.
- `exports/codexa-900m-sft-v2-lmstudio-f16.gguf` was generated with 293
  tensors, verified structurally, and has SHA-256
  `5b5a2f7a17282946e6573c80027886ed4d0f41f705305b73fc678411f34d5698`.
- LM Studio imported the GGUF and loaded it to 100% in 1.95 seconds, but both
  chat and raw completion API smoke tests failed with the engine's
  `expected peg-native format` / `expected Content-only format` errors.

### Affected paths

- `checkpoints/codexa-900m-sft-v2/latest.pt`
- `checkpoints/codexa-900m-sft-v2/latest.pt.sha256`
- `logs/codexa-900m-sft-v2/train_metrics.jsonl`
- `exports/codexa-900m-sft-v2-hf-gpt2/`
- `exports/codexa-900m-sft-v2-lmstudio-f16.gguf`
- LM Studio model key `codexa-900m-sft-v2-lmstudio`

### Storage audit and next action

No checkpoint cleanup was performed because the exported model failed the
behavioral smoke gate. The largest candidates are the 61G native 100M
checkpoint lineage, 11G 900M base lineage, and 3.5G completed SFT v1 copy;
the native lineage is still protected by the repository retention policy, the
900M base is the SFT parent, and SFT v1 has no verified replacement export.
The 146G dataset tree and 417M logs are also retained. Fix or replace the
LM Studio-compatible exporter, then rerun native/exported equivalence and
quality gates before deleting any checkpoint artifacts.

## 2026-08-06 — Add native interactive SFT chat path

### Decision

Use native PyTorch inference as the immediate conversational test path and add
`scripts/chat_native.py` so the completed SFT checkpoint can be queried without
the incompatible LM Studio GGUF adapter.

### Evidence

- A direct native probe against `checkpoints/codexa-900m-sft-v2/latest.pt`
  generated a coherent response on CUDA.
- The new interactive command loads the checkpoint once, preserves prior turns,
  supports `/clear` and `/exit`, and stops generation at `<|end|>`.
- The native response identified itself as Open Assistant, so native execution
  works but conversational identity/data quality still needs evaluation.

### Affected paths

- `scripts/chat_native.py`
- `checkpoints/codexa-900m-sft-v2/latest.pt`
- `checkpoints/tokenizer-base-v1/tokenizer.json`

### Next action

Run the fixed native multi-turn, factual, arithmetic, and stopping probes before
any further training or exporter work.

## 2026-08-06 — Expose native SFT chat through local Codexa CLI

### Decision

Make the completed native checkpoint directly accessible through the existing
local-development Codexa CLI as `codexa-dev native`, while leaving the
published `codexa` command unchanged.

### Evidence

- The local launcher now resolves the Python environment, interactive chat
  script, SFT v2 checkpoint, tokenizer, and CUDA device explicitly.
- The stale `codexa-dev` and `cxd` shims were reinstalled to point at the
  current `13-Codexa CLI` checkout.
- An end-to-end `codexa-dev native` smoke conversation loaded the checkpoint on
  CUDA and returned assistant text without LM Studio or GGUF.

### Affected paths

- `/home/k9-vortex/Development/1-JavaScript(Type)/13-Codexa CLI/scripts/run-local-dev.mjs`
- `/home/k9-vortex/Development/1-JavaScript(Type)/13-Codexa CLI/scripts/run-local-dev.test.ts`
- `/home/k9-vortex/Development/1-JavaScript(Type)/13-Codexa CLI/scripts/README.md`
- `/home/k9-vortex/.local/share/npm/bin/codexa-dev`
- `/home/k9-vortex/.local/share/npm/bin/cxd`

### Next action

Use `codexa-dev native` for manual conversational evaluation and record native
quality separately from the rejected LM Studio adapter.

## 2026-08-06 — Repair interactive native CLI terminal ownership

- Decision: Keep `codexa-dev native` attached to the invoking terminal's stdin
  and open one verified visible Kitty native-chat session for immediate use.
- Evidence: The prior Zsh wrapper routed every `codexa-dev` invocation through
  an external-terminal handoff, making native chat appear to exit. The repaired
  function bypasses that handoff for `native`; Zsh syntax validation passes,
  and the visible native process is loaded on CUDA as PID 804619.
- Affected paths: `/home/k9-vortex/.zshrc`, backup
  `/home/k9-vortex/.zshrc.codexa-native-20260806-1125.bak`, and the running
  `codexa-dev native` Kitty session.
- Next action: Type directly at the visible `You:` prompt; use `/exit` to close.
## 2026-08-06 — Route Codexa CLI through a dedicated native provider

Decision: add `Codexa Native` as a first-class Codexa CLI provider backed by a
persistent PyTorch JSONL bridge. This provider bypasses LM Studio, llama.cpp,
and the GGUF compatibility export while preserving conversation history inside
one loaded `NativeChatEngine` process.

Evidence: `scripts/native_chat_bridge.py` loaded
`checkpoints/codexa-900m-sft-v2/latest.pt` on CUDA and completed two sequential
requests in one process. The bridge protocol and routing work, but the returned
answers were low quality and failed the requested exact-phrase instruction;
this is now isolated as checkpoint/SFT quality rather than LM Studio transport.
The Codexa CLI TypeScript typecheck and full Bun test suite passed. The model
repository suite reported 56 passes and three pre-existing stale fixture
failures involving parameter-count/config expectations.

Affected paths: `src/native_chat.py`, `scripts/chat_native.py`,
`scripts/native_chat_bridge.py`, and the Codexa CLI provider runtime/registry/UI
under `/home/k9-vortex/Development/1-JavaScript(Type)/13-Codexa CLI/src/`.

Next action: test through the visible `Codexa Native` route, then evaluate or
repair SFT data/checkpoint behavior independently of provider integration.

## 2026-08-11 — Reuse the prepared full base-training stream

- Decision: Use the existing mixed base-training stream for the next base phase;
  do not rebuild the corpus or delete raw, processed, tokenized, checkpoint, or
  export artifacts.
- Evidence: `data/tokenized/base-v1/mixed/token_data_manifest.json` records
  12,819,847,234 training tokens, 13,383,283 documents, the frozen
  FineWeb-Edu/Wikipedia mixture, and zero repeated documents. The train stream,
  index, and manifest match their recorded SHA-256 checksums. The full Python
  validator was not runnable because `.venv` lacks the `tokenizers` package.
- Affected paths: `data/tokenized/base-v1/mixed/`,
  `data/tokenized/base-v1/production/token_data_manifest.json`,
  `configs/1b.yaml`, and the existing training launch scripts. No generated
  data or model artifact was changed in this session.
- Next action: Restore repository dependencies, run the complete token-data
  validator, then choose the optimizer-step budget before launching. At 65,536
  tokens per optimizer step, the full stream requires approximately 195,616
  steps; the current 10,000-step config covers only about 5.1 percent.

## 2026-08-14 — Measure preparation speed and recheck production readiness

- Decision: Treat the existing prepared corpus as intact but not yet fully
  accepted for launch; repair the validator/test contract before starting a
  training run.
- Evidence: The completed preparation manifest records 13,700,814 input rows
  processed in 6,590.5 seconds, approximately 2,078 rows/second, and the four
  processed JSONL output SHA-256 values match the manifest. A bounded memmap
  probe read 409,600 tokens at approximately 14.9M tokens/second, so host token
  loading is not the observed bottleneck. The prior verified 900M production
  benchmark measured 7,700.7 tokens/second and 8.51 seconds per optimizer step.
  `inspect_token_data.py` currently rejects the mixed manifest with missing
  `eos_token_id`. The test suite reports 56 passed and 3 failures caused by
  stale 921,773,568-parameter/config fixtures versus the active
  934,356,480-parameter model and required rotary fields.
- Affected paths: `data/processed/base-v1/`,
  `data/tokenized/base-v1/mixed/token_data_manifest.json`,
  `scripts/inspect_token_data.py`, `src/token_data.py`, `tests/`,
  `configs/1b.yaml`, and `documentation/training/100M_PROGRESS_LOG.md`.
- Next action: align the validator and fixtures with the active manifest/model,
  rerun the full test suite and token-data validator, then run a fresh visible
  12-step CUDA BF16 production-shape benchmark before selecting the training
  budget. No training or cleanup was launched in this update.

## 2026-08-14 — Diagnose native inference speed

- Decision: Optimize native inference before spending additional training time;
  the measured bottleneck is missing attention KV caching, not insufficient
  base-training steps.
- Evidence: On the RTX 4080 using
  `checkpoints/codexa-900m-sft-v2/latest.pt`, native CUDA generation measured
  approximately 117 tokens/second with a 32-token prompt, 61 tokens/second
  with a 128-token prompt, and 21 tokens/second with a 512-token prompt. A
  128-token prompt with 64 generated tokens measured 59 tokens/second. The
  current `src/generate.py` calls the full model on the entire growing
  sequence for every generated token, and `src/model.py` exposes no KV-cache
  state in attention or transformer blocks. No local llama.cpp/LM Studio CLI
  was available for an immediate GGUF comparison.
- Affected paths: `src/generate.py`, `src/model.py`, `src/native_chat.py`,
  `scripts/generate.py`, `scripts/native_chat_bridge.py`, and the current 900M
  SFT checkpoint.
- Next action: implement an opt-in KV-cache path with equivalence tests for
  logits, EOS/`<|end|>` stopping, and multi-turn history; then rerun the same
  prompt-length benchmark before comparing quantized exports. No checkpoint
  or dataset was changed.

## 2026-08-14 — Start 900M base 10,000-step run

- Decision: Start a fresh 10,000-optimizer-step base-training run from random
  weights using the existing mixed FineWeb-Edu/Wikipedia token stream.
- Evidence: No active training process was found. The production model is
  934,356,480 parameters and the RTX 4080 supports CUDA BF16. The existing
  250-step checkpoint interval was unsafe with 116GB free because it could
  create approximately 40 full checkpoints. A dedicated run configuration
  uses a 1,000-step interval, retaining recoverable latest/previous state and
  ten milestone checkpoints while avoiding that storage overrun.
- Affected paths: `configs/900m-base-10k.yaml`,
  `data/tokenized/base-v1/mixed/train.bin`,
  `data/tokenized/base-v1/fineweb_edu/validation.bin`,
  `data/tokenized/base-v1/production/token_data_manifest.json`,
  `checkpoints/codexa-900m-base-10k-20260814/`, and
  `logs/codexa-900m-base-10k-20260814/`.
- Next action: Monitor the first metrics and checkpoint, confirm sustained
  CUDA BF16 throughput and finite losses, then retain the visible 100x22
  viewer through completion or resume it after interruption.

## 2026-08-14 — Stop mistaken base-training launch

- Decision: Stop the newly launched base-training run because the active user
  objective is inference speed, not additional pretraining.
- Evidence: The run was terminated cleanly at optimizer step 17 after seeing
  1,114,112 tokens. The final recorded training loss was 9.6977 and measured
  throughput was 7,859 tokens/second. No checkpoint was written because the
  configured safe milestone interval was 1,000 steps. The run log and metrics
  remain preserved; no dataset, existing checkpoint, or export was deleted.
- Affected paths: `logs/codexa-900m-base-10k-20260814/`,
  `/tmp/codexa-900m-base-10k-20260814.log`, and the stopped process/viewer.
- Next action: Do not resume this run. Continue with the already measured
  native inference diagnosis and implement/test KV caching in
  `src/model.py` and `src/generate.py`.

## 2026-08-14 — Add and benchmark native inference KV caching

- Decision: Keep the KV-cache path enabled by default for `LanguageModel`
  generation while preserving an explicit uncached reference path for tests and
  regression comparison.
- Evidence: The focused model/generation suite passes 12 tests, including
  cached versus uncached logit equivalence and identical greedy token output.
  On the RTX 4080 with the 900M SFT v2 checkpoint, cached generation measured
  137 versus 117 tokens/second for a 32-token prompt, 124 versus 61 for a
  128-token prompt, 96 versus 21 for a 512-token prompt, and 125 versus 59 for
  a 128-token prompt generating 64 tokens. The cache therefore improves long
  context generation by approximately 4.5x in this bounded test.
- Affected paths: `src/model.py`, `src/generate.py`, `tests/test_generation.py`,
  and the native inference path used by `src/native_chat.py`.
- Next action: run the full repository regression suite, then add explicit
  first-token/steady-state latency telemetry and benchmark multi-turn native
  chat before considering quantization or further model changes.

## 2026-08-14 — Torch compile environment gate

- Decision: Keep `torch.compile` opt-in and do not make it the default yet.
- Evidence: The compile wiring is present for one-shot generation and native
  chat, but the first real 900M CUDA attempt failed in PyTorch Inductor/Triton
  before producing a benchmark. The environment is missing `Python.h` for
  Python 3.14, so Triton cannot build its CUDA helper. The existing KV-cache
  path remains validated and unaffected.
- Affected paths: `src/generate.py`, `src/native_chat.py`,
  `scripts/generate.py`, and `scripts/chat_native.py`.
- Next action: provide a compatible Python development-header/toolchain
  environment, rerun the compile smoke test, then benchmark compiled versus
  cached generation. Do not enable compile by default until that benchmark and
  output/stopping checks pass.

## 2026-08-14 — Reject torch.compile as default speed path

- Decision: Keep `torch.compile` available only as an experimental opt-in; do
  not enable it for Codexa native inference.
- Evidence: After installing `python3.14-devel`, a temporary no-space toolchain
  alias bypassed the linker path issue caused by the repository name. The real
  900M CUDA benchmark completed with a 45.39-second compile warm-up and about
  126.6 tokens/second steady state versus about 125.5 tokens/second for the
  normal KV-cache path. Outputs matched, but the steady-state gain was only
  approximately 1% and does not justify the compile warm-up/runtime fragility.
- Affected paths: `src/generate.py`, `src/native_chat.py`,
  `scripts/generate.py`, and `scripts/chat_native.py`.
- Next action: leave compile opt-in and benchmark weight-only quantization or a
  compatible optimized inference runtime as the next speed candidate. The
  validated KV-cache path remains the default.

## 2026-08-16 — Remove obsolete base-training data

- Decision: Remove the old raw and derived base-training datasets to reclaim
  storage while preserving the data required for the next 900M conversational
  improvement.
- Evidence: `data/` occupied approximately 146 GB. No active training process
  referenced the dataset paths. The current 900M SFT launcher still references
  `data/tokenized/base-v1/mixed/train.bin`, so that 27 GB replay dataset was
  retained. Chat/SFT data was also retained.
- Affected paths removed: `data/processed/base-v1/`,
  `data/raw/fineweb-edu-10bt/`, `data/raw/wikipedia/`,
  `data/tokenized/base-v1/fineweb_edu/`,
  `data/tokenized/base-v1/wikipedia/`, and
  `data/tokenized/fineweb-edu-1b-v1/`. These exact directories were moved to
  the desktop Trash using `/usr/bin/gio trash`; the removal is recoverable
  until the Trash is emptied.
- Evidence after cleanup: `data/processed/` is approximately 4.7 GB,
  `data/raw/` approximately 820 MB, and `data/tokenized/` approximately 27 GB.
- Next action: rebuild only the dataset needed for the next approved training
  phase; do not empty the Trash until the retained SFT/replay path has been
  validated.

## 2026-08-16 — Remove obsolete 100M native checkpoints

- Decision: Remove the completed `codexa-100m-native-v1` checkpoint lineage
  because the active improvement path is the 900M model and the 100M native
  export is already present.
- Evidence: The checkpoint directory occupied approximately 61 GB, no matching
  100M training process was active, and
  `exports/codexa-100m-native-v1-lmstudio-f16.gguf` plus the HF export existed.
- Affected paths: `checkpoints/codexa-100m-native-v1/` was moved to the desktop
  Trash using `/usr/bin/gio trash`; the 100M exports, tokenizer, 900M
  checkpoints, and 100M SFT pilot checkpoint were preserved.
- Next action: use the retained 900M SFT-v2 checkpoint for model improvements;
  empty the Trash only after confirming the 100M export is no longer needed.

## 2026-08-16 — Permanently remove 100M native checkpoints from Trash

- Decision: Permanently delete the previously trashed
  `codexa-100m-native-v1` checkpoint directory after the user confirmed it was
  no longer needed.
- Evidence: The exact Trash item and matching `.trashinfo` metadata were found;
  the 100M HF and GGUF exports remain under `exports/`.
- Affected paths: `/home/k9-vortex/.local/share/Trash/files/codexa-100m-native-v1`
  and its matching Trash metadata were permanently removed. Approximately
  61 GB of checkpoint artifacts were reclaimed; unrelated Trash contents were
  left untouched.
- Next action: continue model improvements from the retained 900M lineage.

## 2026-08-21 — Launch balanced 900M SFT v1

- Decision: Start `codexa-900m-sft-balanced-v1` from the checksum-verified
  900M base checkpoint using a corpus-wide deterministic sample rather than
  the first-record subset that caused collapse.
- Evidence: New samples contain 30,000 UltraChat and 30,000 OASST1 records,
  selected by stable hash with seed `20260821`. The run uses 50% base replay,
  `learning_rate_scale=0.25`, 1,000 warmup steps, 6,000 optimizer steps,
  AdamW8bit, and the required Kitty viewer. Existing SFT-v1, SFT-v2, and sanity
  checkpoints are preserved.
- Affected paths: `scripts/sample_conversation_jsonl.py`,
  `scripts/train_conversational_sft.py`, `scripts/run_900m_sft_balanced.sh`,
  `data/processed/chat-sft-balanced-v1/`, and the new run paths under
  `logs/codexa-900m-sft-balanced-v1/` and
  `checkpoints/codexa-900m-sft-balanced-v1/`.
- Next action: monitor the attached viewer; after completion, verify checksum,
  reload the checkpoint, and run the fixed native behavioral probes before any
  export or promotion.

## 2026-08-22 — Balanced 900M SFT v1 behavioral gate failed

- Decision: Do not promote or export `codexa-900m-sft-balanced-v1` as a usable
  conversational model. Preserve its checkpoint and logs for comparison.
- Evidence: The run completed 6,000 CUDA optimizer steps and approximately
  210.18M tokens; the final validation loss was `2.028321` and the checkpoint
  sidecar passed. Reloaded native probes showed grammatical output and some
  `<|end|>` stops, but arithmetic was wrong (`17+16=42`), explanations drifted
  into invented Codexa plant facts, correction produced unrelated content, and
  the tea request failed the exact three-item requirement and often hit the
  length limit. Multi-turn responses also repeated the arithmetic template.
- Affected paths: `checkpoints/codexa-900m-sft-balanced-v1/latest.pt`,
  `logs/codexa-900m-sft-balanced-v1/`, and the balanced SFT launcher/config.
- Next action: stop treating validation loss as conversational proof. Audit
  training-example quality and prompt/answer alignment at the sampled-record
  level, compare logits and labels on fixed probes, and design a supervised
  chat dataset with explicit short-answer, correction, and instruction-following
  coverage before another long run.

## 2026-08-22 — Pre-training diagnostic complete; no new run authorized

- Decision: Do not start another training run yet. Keep all existing model
  checkpoints, exports, datasets, and logs intact while the next SFT recipe is
  reviewed.
- Evidence: The tiny four-example CUDA overfit reached exact target text and
  stopped on `<|end|>`, and the shifted-label audit reports zero END alignment
  failures. The balanced sampled corpus contains 26,082 accepted UltraChat
  records and 29,186 accepted OASST1 records at context length 2,048; 3,918
  and 814 records respectively are rejected only because complete conversations
  exceed context. Accepted UltraChat has 77.1% supervised tokens and accepted
  OASST1 has 84.7%. Native comparison shows base incoherence, while balanced
  SFT is grammatical but gives wrong arithmetic, prompt drift, and incomplete
  list answers even when each probe is reset.
- Affected paths: `scripts/audit_chat_training.py`, `src/native_chat.py`,
  `scripts/chat_native.py`, `scripts/native_chat_bridge.py`,
  `scripts/train_conversational_sft.py`, `tests/test_sft_serialization.py`,
  and the diagnostic output under `logs/diagnostics/`.
- Next action: review and approve a replacement data/recipe gate with explicit
  short-answer, arithmetic, correction, list, and multi-turn examples; only
  after that approval may a new training launch be started.

## 2026-09-10 — Plan bounded 900M conversational repair pilot

- Decision: The next run will be a bounded repair pilot from the retained
  `codexa-900m-base-v1` checkpoint, using a new filtered short-response corpus.
  Do not repeat the 6,000-step, 50%-base-replay balanced recipe before a
  behavioral gate passes.
- Recipe proposal: build `chat-sft-repair-v1` with explicit user-first
  conversations, a 512-token maximum serialized context for the pilot, exact
  deduplication, prompt-echo rejection, complete assistant turns only, and
  guaranteed coverage for arithmetic, factual short answers, explanations,
  corrections, lists, and multi-turn memory. Use a deterministic 80/20 train /
  validation split, AdamW8bit, batch size 1, gradient accumulation 32, peak
  learning rate `1e-5`, 500 warmup steps, 2,000 optimizer steps, and no base
  replay in the first repair pilot. Retain the base replay path for a later
  stability run only if this pilot passes.
- Preflight gate: run the serializer audit, the four-example CUDA overfit, and
  a 100-record dry-run accounting report before launching. Reject the launch
  if any shifted-label or END alignment check fails.
- Acceptance gate: checksum and reload the checkpoint, then run isolated native
  probes with a reset before every prompt. Require correct arithmetic, factual
  recall, correction, three-item list structure, bounded explanation, valid
  stop-token termination, and no prompt echo or cross-prompt contamination.
  Validation loss is secondary evidence and cannot promote a checkpoint by
  itself.
- Affected paths: planned `data/processed/chat-sft-repair-v1/`,
  `logs/codexa-900m-sft-repair-v1/`,
  `checkpoints/codexa-900m-sft-repair-v1/`, and a new launcher based on the
  required Kitty 100x22 viewer. Existing checkpoints, exports, data, and logs
  remain untouched.
- Next action: implement and inspect the repair dataset and dry-run report;
  wait for explicit approval before starting the 2,000-step pilot.

## 2026-09-10 — Repair corpus and preflight gates passed

- Decision: The repair pilot is technically ready but remains unlaunched until
  the user explicitly requests the training start.
- Evidence: `chat-sft-repair-v1` contains 12,096 total records: 12,000
  deterministic natural records and 96 behavioral anchors, split into 9,627
  train and 2,469 validation records. Both splits have 100% serializer
  acceptance, zero prompt echoes, zero END alignment failures, and a maximum
  serialized length of 512 tokens. The 300-step CUDA overfit finished at loss
  `0.0001215646` and reproduced all four exact targets with `stop_token`.
- Affected paths: `scripts/prepare_repair_sft.py`,
  `scripts/run_900m_sft_repair.sh`, `data/processed/chat-sft-repair-v1/`,
  `logs/diagnostics/chat-repair-audit-20260910.json`, and
  `logs/chat-overfit-repair-preflight-v1/`.
- Next action: on explicit start approval, launch the prepared 2,000-step
  repair run with the required Kitty 100x22 viewer; otherwise leave all model
  artifacts unchanged.

## 2026-09-10 — Launch approved repair SFT run

- Decision: Start `codexa-900m-sft-repair-v1` from the verified retained 900M
  base checkpoint after explicit user approval.
- Evidence: The base checkpoint sidecar passed, the repair train and validation
  files are present, the base replay token file is present, and no training
  process was active. The preflight serializer, END alignment, and CUDA
  overfit gates had already passed.
- Affected paths: `scripts/run_900m_sft_repair.sh`,
  `checkpoints/codexa-900m-sft-repair-v1/`,
  `logs/codexa-900m-sft-repair-v1/`, and the visible Kitty 100x22 viewer.
- Next action: monitor the run to completion; verify checkpoint checksum,
  reload it, and execute the isolated behavioral acceptance probes before any
  export or promotion.

## 2026-09-10 — Repair SFT completed; behavioral promotion gate failed

- Decision: Do not export or promote `codexa-900m-sft-repair-v1` as the usable
  conversational checkpoint. Preserve its checkpoint and logs for comparison.
- Evidence: The run completed all 2,000 optimizer steps, reached validation
  loss `2.4453267`, and passed the checkpoint checksum. Native CUDA probes with
  resets showed correct `17 + 25 = 42`, true two-turn color memory, and a
  correction response. However, explanation prompts produced repeated
  `photosynthetic` text and hit the length limit; the exact-three-item tea list
  returned only two items. The checkpoint therefore fails the explanation and
  exact-list quality gates despite targeted improvements.
- Affected paths: `checkpoints/codexa-900m-sft-repair-v1/latest.pt`,
  `logs/codexa-900m-sft-repair-v1/`, and the repair launcher. No checkpoint,
  export, tokenizer, dataset, or log was deleted.
- Next action: diagnose the repetition/list failure and revise the repair data
  or decoding/training recipe before any export or follow-up run.

## 2026-09-10 — Deeper repair checkpoint postmortem

- Decision: Treat the repair checkpoint as a narrow template-repair result,
  not a general conversational improvement.
- Evidence: Exact familiar prompts pass, but generalization probes fail: `18 +
  24` became `19 + 24 = 42`, an unseen correction claimed `4 is 8`, and both
  tea and fruit requests returned the same Apple/Banana/Orange answer. Simple
  anchor-matching explanations pass, while a photosynthesis request with a
  different wording loops on `photosynthetic` until the length limit. The
  training loss continued down to about `2.0` while validation loss flattened
  near `2.43-2.45`, indicating memorization/fit without robust behavioral
  generalization.
- Root cause assessment: The repair corpus contains 9,556 natural records but
  only 71 anchor records in train, with each anchor category generated from
  four repeated templates. Anchor answers also repeat across the split. This
  teaches fixed answer forms and does not provide enough varied arithmetic,
  correction, list, or explanation transformations. The low-learning-rate
  pilot improved exact templates but did not repair the underlying base-model
  generalization.
- Affected paths: `data/processed/chat-sft-repair-v1/`,
  `checkpoints/codexa-900m-sft-repair-v1/latest.pt`, and the native probe
  path. No artifacts were removed or exported.
- Next action: replace repeated anchors with a larger deduplicated, generated
  behavioral suite using many distinct operands, list subjects, corrections,
  and explanation prompts; hold out templates by task family so validation
  measures generalization before another training decision.

## 2026-09-11 — Launch five-hour 900M base continuation

- Decision: Start `codexa-900m-base-continuation-5h-v1` as a fresh optimizer
  continuation initialized from the verified `codexa-900m-base-v1` weights.
- Evidence: The source model completed 10,000 steps / 655.36M tokens but remains
  undertrained. The new run reserves the final 39,298,352 tokens from the
  retained 12.82B-token stream for validation and excludes that range from new
  training. The source checkpoint loads successfully, the baseline loss on 16
  held-out batches is `3.2042928`, and focused range/checkpoint tests pass
  (`9 passed`). The validation tail may have been sampled during the historic
  base run, but it is disjoint from this continuation and is suitable for a
  within-run comparison.
- Affected paths: `src/token_data.py`, `scripts/train.py`,
  `scripts/evaluate_checkpoint.py`, `tests/test_token_data.py`,
  `configs/900m-base-continuation-5h.yaml`,
  `data/tokenized/base-v1/continuation-5h/token_data_manifest.json`,
  `scripts/run_900m_base_continuation_5h.sh`, and the new run paths under
  `logs/` and `checkpoints/`.
- Next action: keep the Kitty 100x22 viewer attached through 2,000 optimizer
  steps, then verify checksum and compare final held-out loss and fixed outputs
  against the recorded baseline before any SFT or export decision.

## 2026-09-11 — Pause five-hour base continuation

- Decision: Pause `codexa-900m-base-continuation-5h-v1` immediately at the
  user's request using `SIGSTOP`; do not terminate or restart it.
- Evidence: PID `236816` is in stopped state `T`. The last completed optimizer
  step is 7 with 458,752 tokens processed and training loss `3.0028406`.
- Affected paths: the live training process, its existing logs, and the visible
  Kitty viewer. No checkpoint, log, dataset, or model was removed.
- Next action: wait for an explicit resume instruction before sending
  `SIGCONT` or performing further run actions.

## 2026-09-11 — End paused base continuation

- Decision: End `codexa-900m-base-continuation-5h-v1` at the user's request and
  close its viewer. Do not delete or alter the original 900M base checkpoint.
- Evidence: The detached process did not honor the first graceful interrupt
  after being continued from `SIGSTOP`; it was terminated explicitly after
  reaching optimizer step 13 and 851,968 tokens. The GPU process and Kitty
  viewer are gone. The run ended before its first scheduled checkpoint at step
  500, so it produced no recoverable continuation checkpoint.
- Affected paths: `logs/codexa-900m-base-continuation-5h-v1/` and
  `/tmp/codexa-900m-base-continuation-5h-v1.log`. Original checkpoints,
  datasets, exports, and tokenizer files remain intact.
- Next action: treat this run as interrupted and start any future continuation
  again from `checkpoints/codexa-900m-base-v1/latest.pt` after explicit approval.

## 2026-10-08 — Stage 1 specialist implementation authorized

- Decision: Implement a separate frozen EmbeddingGemma 2 classifier with an isolated dependency environment. Preserve the generative pipeline and record its three baseline failures separately. Use dominant-intent labels, hold uncertain cases for review, and return uncalibrated five-class scores only from trained heads.
- Evidence: The baseline suite reports 58 passed / 3 failed. Existing Transformers 4.57.6 lacks the encoder; Sentence Transformers is absent. Transformers 5.19.0 contains the model and dependency resolution with Sentence Transformers 6.1.0 succeeds. Inspected local data lacks the required intent labels; the user confirmed infrastructure-first delivery without a labeled dataset.
- Affected paths: `src/specialist/`, `scripts/specialist.py`, `configs/specialist.yaml`, specialist requirements/tests/documentation, `.venv-specialist/`, and the added environment ignore rule. Existing checkpoints, exports, datasets, logs, and unrelated working-tree changes are protected.
- Next action: Implement offline coverage and perform real text-only encoder initialization and embedding smoke verification. Do not launch generative training or fabricate classifier accuracy.

## 2026-10-08 — Specialist offline pipeline and baseline regression check

- Decision: Keep the initial specialist as infrastructure without a task-trained head. Enforce visible viewer attachment before public training launches, protected specialist output namespaces, immutable prepared bundles, and checksummed head checkpoints. Cache reuse now avoids loading the encoder when the execution/input identity matches.
- Evidence: The initial original-environment specialist pass has 33 passed and 5 dependency-related skips; the complete suite has 91 passed, 5 skipped, and the same 3 pre-existing failures. Tests exercise schema validation, frozen encoder contracts, gradient updates, losses, deterministic head/text inference, checkpoint corruption, and viewer configuration. The five scikit-learn-dependent tests await the isolated environment installation. Review corrected a possible prior-head CUDA allocation reference in checkpoint payloads.
- Affected paths: `src/specialist/`, `scripts/specialist.py`, `tests/test_specialist.py`, the specialist configuration and dependency lock, and `documentation/reference/SPECIALIST.md`. No existing checkpoint, dataset, export, or generative source file was changed.
- Next action: Complete the isolated dependency install, run every specialist test, and verify the real pinned text-only model. Report encoder measurements separately from task accuracy.

## 2026-10-08 — Specialist dependency setup and mechanical acceptance

- Decision: Add Pillow 12.3.0 and TorchVision 0.28.0 for the upstream processor loader, while keeping vision/audio neural towers disabled. Seed the new isolated environment with copies of exact-version existing PyTorch/CUDA distributions, then install the remaining lock; do not share package paths or modify the original environment.
- Evidence: Transformers' EmbeddingGemma2Processor requires vision utility backends even on text inputs; TorchVision 0.28.0 requires the selected PyTorch 2.13.0. The original native distributions match the lock versions. A slow redundant binary download was interrupted without deleting its cache. All 39 offline tests pass with compatible downloaded test dependencies. An earlier cache-only verification attempt selected a free-threaded wheel for a non-free-threaded Python; this setup-only mismatch was corrected and is not a pipeline acceptance failure. A real 100x22 Kitty viewer attached successfully and only that test window was closed.
- Affected paths: `.venv-specialist/`, specialist requirements/lock/documentation, `tests/test_specialist.py`, and the new viewer-smoke log under `logs/specialist/`. Original environment, model lineages, training checkpoints, datasets, and exports remain unchanged.
- Next action: Verify locked imports and all tests inside the completed isolated environment, then run the real pinned encoder smoke and record measured results separately from task metrics.

## 2026-10-08 — Stage 1 specialist accepted as verified infrastructure

- Decision: Accept Stage 1 infrastructure and real encoder verification. Do not claim a trained intent classifier, task accuracy, macro-F1, or baseline-versus-MLP task superiority until human-reviewed data is supplied. Retain all existing artifacts and baseline failures independently.
- Evidence: All 39 specialist tests pass in the isolated environment; all 65 locked packages pass compatibility checks with no lock mismatches. Representative copied PyTorch and cuBLAS binary hashes match the original and have separate inodes. Original package versions are unchanged. The original-environment full suite reports 92 passed, 5 dependency-related specialist skips, and the same 3 baseline failures. Actual pinned encoder tests pass on CUDA BF16 and CPU FP32: 271,002,624 text-only parameters, zero trainable encoder weights, normalized deterministic [2, 768] outputs. Final warmed batch-size-one encoder latency is 15.208 ms on the RTX 4080 and 46.008 ms on CPU; CUDA peaks are 580,295,680 allocated bytes and 599,785,472 reserved bytes. The actual Kitty viewer attachment check passed. No labeled task training or generative model training was launched.
- Affected paths: `src/specialist/` (9 modules), `scripts/specialist.py`, `configs/specialist.yaml`, specialist requirements/lock, `tests/test_specialist.py`, `documentation/reference/SPECIALIST.md`, the isolated environment ignore rule, and these append-only records. GPU evidence: `logs/specialist/encoder-smoke-1791451727274305893/report.json`. CPU evidence: `logs/specialist/encoder-smoke-1791451782146202907/report.json`. Review removed the deprecated embedding-dimension API call and corrected the active run metadata state; the specialist suite still passes after those changes.
- Next action: Obtain representative human-reviewed dominant-intent labels with independent group coverage, then run documented prepare/train/evaluate commands using fresh specialist artifact namespaces. Prioritize independent ambiguity/OOD probes and confidence calibration in Stage 2. No checkpoint, dataset, export, tokenizer, or existing log was deleted; nothing was committed or pushed.


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


## 2026-10-08 — Authorized training repository publication

Under explicit user authorization, published README-only first commit 8643336 to main of golba98/LLM-Training-PyTorch-Codexa-v1 and saved all remaining 29 project files in commit 33effac on feat/save-training-project. Opened PR https://github.com/golba98/LLM-Training-PyTorch-Codexa-v1/pull/1. Offline existing-environment validation: 10 passed, two existing return-value warnings. Weights, datasets, encoder caches, generated logs and environments remain excluded. No production training, downloads, architecture changes or checkpoint/token format changes. Sibling packages remain local version 0.1.0 dependencies.


## 2026-10-08 — Authorized data repository publication

Under explicit user authorization, saved all 42 remaining LLM-Data project files in commit 479f248 on feat/save-data-project and published PR https://github.com/golba98/LLM-Data-PyTorch-Codexa-v1/pull/2, including the existing README heading commit. Anchored the dataset ignore rule to /data/ so nine Python source modules under src/llm_data/data are tracked. Offline existing-environment validation: 14 passed; staged whitespace check passed. Validation uses sibling sources and an existing integration test helper; standalone installation remains unverified, disclosed in the PR. Weights, datasets, encoder caches, generated logs and build output remain excluded. No production training, downloads, checkpoint key, token ID or format changes.


## 2026-10-08 — Canonical workspace consolidation

Lifted integration history to the renamed workspace root and registered seven existing repositories as pinned submodules. Preserved source/Git recovery records and relocated historical assets to a single protected temporary store with before/after SHA-256 verification. Canonical environments and asset resolution no longer depend on project 31. NumPy remains independently implemented and packaged. Actual-checkpoint parity and bounded CPU/CUDA validations are recorded in documentation/migration/. Originals remain pending review and independent backup; no model lineage was promoted.


## 2026-10-09 — Restore eight sibling repository boundaries

Moved integration and its Git history into LLM-From-Scratch; made existing component Git metadata independent without changing HEADs or indexes. Shared recovery/build/validation evidence moved once to workspace-infrastructure. Full pre-change plan, ownership inventory and journal are retained there. Source imports use pinned sibling paths; environments retain separate profiles. Original 31 and 32 are untouched; independent backup remains unavailable. No commits, pushes, PRs, training runs or repository deletion authorized. Validation results are recorded in the migration audit report.


## 2026-10-09 — Publish sibling-layout review changes

User explicitly authorized saving and publishing LLM-Data and LLM-From-Scratch changes to PRs. Existing Data PR #3 and integration PR #2 are updated without merging. Original Data integration test is preserved byte for byte in its tracked documentation archive; central active coverage remains. No original project, component repository, checkpoint or dataset is deleted. Other component documentation changes remain locally saved and unpublished. Private assets, environments and local recovery evidence remain outside Git. See documentation/migration/PUBLICATION_2026-10-09.md for validation and preservation details.


## 2026-10-09 — Save every remaining workspace source/documentation change

User authorized PRs for all remaining unsaved files. Six remaining component README/AGENTS changes are saved on their existing review branches. Integration compatibility pins advance to their saved commits. Workspace-level README, instructions and shared-infrastructure README now have single canonical tracked owners in documentation/workspace with local links. Every public source/documentation change is reviewed and published; private assets/configuration/environments/recovery records remain preserved and excluded from public Git. No merge, repository deletion, original-project mutation or production training is authorized.


## 2026-10-09 — Finish Source storage consolidation and remove obsolete NumPy shortcuts

- Decision: At the user's explicit request, physically relocate the remaining PyTorch `data/` and `logs/` plus shared `recovery/` from `/home/k9-vortex/Development/LLM-Assets-Staging/` into `/home/k9-vortex/Development/3-Python/31-LLM Source/`. Move the remaining compatibility tree under `31-LLM Source/compatibility/LLM-Assets-Staging`; leave the old staging location as a symlink to retain required model-path compatibility. Configure active workspaces to use direct Source paths.
- Evidence: All 882 moved file/link entries match before/after relative paths, inode/device, byte size, and modification timestamps. Both original and direct model/data/recovery paths resolve correctly. All 13 NumPy packaging/launcher/preservation checks and both PyTorch workspace-path tests pass. Historical output protection was checked. The 31 Source folder now contains approximately 92 GiB; 32 Source contains approximately 38 GiB.
- Cleanup: Removed only `/home/k9-vortex/Development/3-Python/32-LLM (NumPy) [Deprecated]` and `/home/k9-vortex/Development/3-Python/32-LLM (NumPy)-local-artifacts`, after confirming they were symlinks and verifying all preserved archived entries under `32-LLM Source/legacy/`. No actual archived contents or model files were deleted; reclaimed model/artifact size is 0 bytes. Exactly two 32-prefixed entries remain at the Python folder root: `32-LLM (NumPy)` and `32-LLM Source`.
- Affected paths: The exact staging/data/logs/recovery/compatibility paths above; local settings in active 37/38 and original 32; agent notes in the Source folders and related workspaces; and NumPy recovery links, now pointing directly into 31 Source recovery. Historical records/manifests were preserved.
- Next action: Use the Source folders as canonical local storage and preserve the old staging compatibility symlink. No training, checkpoint deletion, commits, or pushes occurred.


## 2026-10-09 — Read-only ChatGPT environment audit

Decision: record the verified current environment in `CHATGPT_CURRENT_STATE.md`; assess ten-hour training as NOT READY AS CONFIGURED, without changing or launching any pipeline.
Evidence: current component sources/Git/configuration, meta-device model count (934,356,480), small checkpoint/NPZ metadata, saved training/test reports, hardware snapshot, and missing production validation binary/index. Current large-checkpoint hashes and functional recovery were NOT VERIFIED; historical results are explicitly separated from fresh observations.
Affected paths: `CHATGPT_CURRENT_STATE.md`, `documentation/training/SESSION_DECISIONS.md`, `documentation/training/100M_PROGRESS_LOG.md`. No model, tokenizer, dataset, checkpoint, exporter, Git history, or system settings changed.
Next action: user reviews the handoff; separately authorize checkpoint recovery verification, validation resolution and a chosen training phase before any ten-hour run.

## 2026-10-09 — Begin 900M ten-hour conversational preparation

- Decision: Implement the approved modular successor and bounded preparation, with a two-hour GPU preparation ceiling. Production remains unlaunched and requires explicit approval. Preserve the original ten-hour deadline, including downtime on resume.
- Evidence: Live architecture construction and five strict GPU checkpoint restores confirm 934,356,480 tied parameters and finite forward outputs; candidate hashes match. General Chat v2 hashes match, with 182,892 training conversations and 147,584,268 recorded assistant tokens. Current split roots/groups/content hashes do not overlap, but historical exposure is widespread. Eight common greedy development probes show serious behavioral failures; no initialization has been accepted. Baseline integration: 121 passed, 5 skipped; component suites: Architecture 5, Tokenizer 6, Data 12, Training 10, Inference 1 passed.
- Affected paths: LLM-Data dataset/sampling helpers; LLM-Training runtime, recovery, monitoring and tests; LLM-From-Scratch evaluation/controller/configuration/report. Existing integration dirty instruction/record files and CHATGPT_CURRENT_STATE.md are preserved.
- Next action: Implement and test supervision, budget and recovery contracts; rebuild local wheels; compare candidates with reproducible development diagnostics and run isolated bounded 934M checks. No historical checkpoint, export, data, NumPy code, or inference default is changed.

## 2026-10-09 — Full conversational audit and candidate behavior baseline

- Decision: Treat all five historical candidates as failing the production experiment-entry quality floor. Use repair only as the provisional technical starting point for isolated benchmarks/recovery and a bounded corrective phase; no conversational checkpoint is accepted for a ten-hour launch.
- Evidence: Recomputed serialization, token counts, shifted assistant masks and normalized content hashes for all 202,901 General Chat v2 conversations. Current split root/group/content overlap is zero. Training contains 147,584,268 assistant tokens; validation/test have extensive potential historical SFT exposure. Fixed 64-case greedy evaluation gives zero exact instruction, arithmetic and output-line compliance for every candidate. END rates: base 0%, Stage 2 40.625%, balanced/v2 75%, repair 68.75%. Speaker-aware scoring prevents an assistant's “My new number” from passing user-memory checks; raw generations and the original reference-only scores are retained separately. Repair has the strongest literal task combination but severe degeneration in 35.9375% of cases.
- Affected paths: outputs/preparation/codexa-900m-chat-10h-v1; LLM-Data chat_dataset.py; integration chat_evaluation.py and preparation/controller modules; training runtime/recovery tests. Historical checkpoints/data are untouched. The protected acceptance suite has not been executed.
- Next action: Measure worst-case 934M training memory/throughput, verify full 8-bit optimizer recovery and run one corrective pilot within the two-hour preparation ceiling. Block production if behavioral gates remain unmet. No production launch, promotion, cleanup, commit or push.

## 2026-10-09 — Memory reserve and full recovery evidence

- Decision: Preserve failed memory-reserve pilots, and test a separate preparation configuration that releases unused CUDA cache at optimizer boundaries. Microbatch 1, accumulation 32 and the learning objective remain fixed. Production remains blocked.
- Evidence: Seven fixed-shape 934M optimizer benchmarks passed. Full recovery restores all model/optimizer bytes, scheduler and counters, with finite continuation; bitwise continuation differs (second resumed loss 2.068998 versus reference 2.069218), so the strict continuity gate fails. Both default and expandable-segment-only corrective pilots stopped after two updates at the 1.5 GiB headroom gate and published verified final recovery checkpoints. The third isolated pilot explicitly enables boundary cache release; its viewer was closed after attachment at step 5, and training continued. Five exact development prompt matches occur in the training corpus; zero exact protected prompt matches does not certify semantic independence.
- Affected paths: LLM-Training recovery/chat runtime; integration chat validation/operations and outputs/preparation/codexa-900m-chat-10h-v1. No historical artifact or unrelated process was altered.
- Next action: Test controlled interruption and original-budget resume; complete sampled and retention diagnostics; issue a BLOCKED readiness report unless every mandatory gate is proven.

## 2026-10-10 — Final 934M conversational preparation handoff

- Decision: BLOCKED; do not launch the ten-hour operation. The successor implementation and A–L readiness report are complete, but initialization quality, corrective behavioral improvement, trustworthy held-out provenance and strict live memory headroom are unmet.
- Evidence: All 202,901 chat records audited; five compatible checkpoints compared with fixed greedy and sampled settings. The 128-step repair-initialized pilot processed 3,326,471 assistant tokens and 4,275,413 context tokens across 4,096 unique conversations. SIGTERM/full-state resume preserves the original deadline and sampler continuity; viewer closure did not stop training. Deterministic algorithms with CUBLAS workspace :4096:8 yielded bitwise-equal optimizer/model continuation. Corrective instruction/arithmetic compliance remains zero; END stops fall to 50% and severe degeneration rises to 53.125%. A separate strict allocator-limit test safely recorded CUDA OOM without promoting a checkpoint. Latest regression: integration 126 passed / 5 sklearn specialist skips; five component suites plus controller tests 56 passed; pip check passed. Relevant installed wheel sources match source bytes. Recorded preparation-stage runtime 2616.7 seconds; this ledger excludes earlier exploratory audit probes.
- Affected repositories: LLM-Data, LLM-Training and LLM-From-Scratch only; exact files, commits, dirty status, native recovery paths, hashes, benchmark evidence and operator commands are in documentation/training/CODEXA_900M_CHAT_10H_READINESS.md and outputs/preparation/codexa-900m-chat-10h-v1/launch-manifest.json.
- Preservation: No ten-hour production namespace/process, accepted-model replacement, export, old-run cleanup, dataset download/regeneration, NumPy modification, commit, push, merge or release. Historical asset digests rechecked. All new isolated smoke checkpoints retained; main storage has about 218 GiB free.
- Next action: Review the blocked report; investigate existing supervision/behavior, exposure provenance and memory availability before a separately approved corrective phase. A production launch still requires explicit user approval after mandatory gates pass.


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

## 2026-10-10 — Checkpoint storage V3 implementation

Decision: extend the existing component format with continuous new-store retention,
reader admission, dependency pins and separately granted journaled reclamation.
Legacy remains the default; no historical adoption, production grant issuance or
real-store executor invocation is permitted in this session.
Evidence: V2 six-publication admission stops; V3 isolated tests exercised 66
publications, exact CPU continuation and process termination/crash replay. Matched
one-pass restore measurements passed the 10% gate. END3 SHA256
4f4e5d1540eb6813ad6d6f91f1a1a503bf8952632d51f9dc6410343a8e0c456b
verified actual uint8 optimizer storage. Current storage reserve fails.
Affected paths: Architecture component_checkpoint/checkpoint_inspection; Training
checkpoint_retention/checkpoint_reclamation/optimized_recovery/recovery/checkpointing/
chat_run/preflight; integration configuration, CPU benchmarks and qualification tools.
Next action: installed-wheel regressions, preservation inventory, archival assessment
and CODEXA_CHECKPOINT_STORAGE_V3.md; full-model GPU execution remains blocked.

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

### 2026-10-11 — Human review dashboard and private-mapping exposure fix

Diagnosed the unusable fresh-blinded-baseline page: the shared `qualification_review.PAGE` template stringifies every non-string field and renders all 540 items plus the overall panels in the header. Package data itself was complete. Two `npx live-server` processes on 0.0.0.0 (:8080, :45891) and Zed's localhost live-server were serving the directory containing `private-mapping.json`; they were stopped with the user's approval. `qualification_review.py` is hash-bound by job-v2 `code_identity`, so it is unchanged (hashes re-verified). Added `review_dashboard.py`/`.html`, `scripts/review_dashboard.py` and `tests/test_review_dashboard.py`. Published `outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline-public/index.html` (only file; package_sha256 5ec93e67… unchanged; mapping/comparison hashes omitted). Original package.json, review.html and private-mapping.json hashes are unchanged. Export stays codexa-human-review-export-v1: Unsure → pending + needs_review, undecided overall → null; legacy unchecked `false` overall values import as Not decided with a notice. Tests: 7 new + 20 existing review/diagnosis passed. Chrome automation verified rendering, navigation, filters, decisions, refresh recovery, legacy/mismatch/invalid import, export, live-server on 127.0.0.1, file://, narrow layout and XSS-inert rendering; Firefox headless render only. No inference, training, cleanup or model changes. See `HUMAN_REVIEW_DASHBOARD.md`.

### 2026-10-11 — Codexa Native integrated into Ubume Dev

Decision: expose all five explicit 934M candidates through the existing development-only native provider; retain greedy evaluation default, legacy training/storage behavior, and no model tools. Evidence: all five full SHA256 identities match; actual Balanced/END3 provider streaming, cancellation and switching passed (~33–35 generated tokens/s, ~3.64 GiB peak reserved). Both failed the READY instruction, so integration is not model-quality promotion. Sequential Ubume suite 2,135 passed; CPU inference 16 passed; architecture/model/identity 6 passed; integration generation/memory 12 passed. Failed earlier suites and final RAM-blocked repeat are retained in outputs/integration/ubume-native-20261011-v1. A single-hash native identity check and incomplete-turn recovery are separately covered by focused CPU tests. No training/checkpoint writes/cleanup occurred.

Affected paths: Ubume Dev provider runtime/config, existing selectors/settings/commands/history/provenance, scripts and docs; LLM-Inference native bridge/callback/finite checks; LLM-Architecture checkpointing optional expected identity; integration ignored local registry and documentation/training/CODEXA_934M_UBUME_NATIVE_INTEGRATION.md. Complete file inventory: outputs/integration/ubume-native-20261011-v1/modified-files.json. Existing dirty work and NumPy assets preserved.

Next action: launch Ubume Dev with bun run dev:run, choose Codexa Native → Codexa PyTorch → an explicit 934M model, and perform exploratory chats when the 16 GiB RAM / 8 GiB VRAM gates pass. Formal diagnostic approvals and model promotion remain separate.

2026-10-11 completion addendum: final native/prompt/conversation-store recovery suite 18 passed; Biome and TypeScript checks pass. Task-created isolated PTY validation data (automatic local dev copies, no original assets) was disposed after identity checks; retained evidence is ~1.1 MiB. No historical storage cleanup occurred.

## 2026-10-11T16:20:35Z — Five-hour foundational run launched under OS supervision

Decision: the user approved the plan as the job-specific grant for frozen job `d7aa275e8904549ab799e283a0ef48d422bc5cc2393af544caf429a5c6eb01e6`. That job is the pilot proposal with only the ceilings changed: 18,000 s, 1,792 updates and generations at 0/896/final. The user explicitly overrode the "neutral-format evidence and genuine review" blocker. Wakeup uses Codex; there is no auto-resume.

Implementation:
- `LLM-Training`: `pretrain_run.py` plus a foundational viewer mode.
- Integration: `foundation_training`, `training_supervisor`, `training_wakeup` and `training_status` modules, plus scripts.

Evidence:
- All 17 launch gates passed. Disk 235.894 GiB; base, tokenizer and selection hashes match; tests 12+36 passed.
- Focused suites: Training 12 passed, supervisor 18 passed.
- Real Codex and Claude CLI checks passed.
- GPU smoke: 2 updates at 7.74k tok/s, 12.13 GiB peak reserved.

Launch:
- Unit `codexa-train-codexa-934m-foundation-5h-v1`; supervisor pid 32436, trainer pid 32460, run_id `d4080f1c…`.
- Baseline CE 3.11296. Updates are occurring at ~8.0k tok/s.

Incident: during testing, broken stub shebangs let about 16 read-only Codex runs and 1 Claude run ($0.34) execute on fake runs. Nothing in the repositories changed, and the tests now fail closed.

No deletion, no Git commit and no protected-asset changes. Report: `CODEXA_934M_TRAINING_SUPERVISOR.md`.

Next action: none until the supervisor's terminal event. Codex analysis and the human decision follow; no automatic follow-up training.

## 2026-10-11T17:20Z — Notification spam root-caused; single at-most-once owner; Rich dashboard

**Findings**
- The five-hour run had already early-stopped at 16:43 UTC on `MEMORY_HEADROOM_GATE`: 0.96 GiB free VRAM against the 1.5 GiB guard, caused by other GPU clients.
- The final step-129 checkpoint is verified, and Codex analysed the run once.
- The user chose to leave training stopped.

**Popup spam: root causes**
- **Primary:** test stubs fell through to the real `notify-send`/`codex`.
- **Structural:** production delivery had no execution-class guard.
- **Design:** the legacy notifier sent two popups per event and could retry the desktop popup.

**Decision**
- New `training_notifications.py` is the sole notification owner.
- Execution class comes from trusted configuration; tests use mocks only and a conftest downgrade.
- Separate desktop and AI records.
- At-most-once desktop delivery, with claim → send → result and `indeterminate` never resent.
- One popup per event, sent only after AI submission.
- Hash-bound modules left untouched; only the scripts were re-pointed.

**Dashboard:** a new read-only Rich dashboard. Rich was added to `.venv` as a 3-package delta (optional extra `viewer`); the lock file was not regenerated.

**Tests:** focused 69 passed; integration suite 278 passed with 5 skipped. The safety snapshot was unchanged.

**Report:** `CODEXA_934M_TRAINING_SUPERVISOR.md`.

**Next:** human decision on a new continuation job, ideally with GPU-heavy desktop apps closed.


## 2026-10-11 18:37 UTC — third VRAM stop; resume 3; interactive agent window

**Stop:** update 671 at 18:31 UTC, `MEMORY_HEADROOM_GATE`.
- Free VRAM was about 1.27 GiB, under the 1.5 GiB guard.
- Final checkpoint `slot-3` verified.
- Validation 3.11296 → 3.10401.
- One popup was sent and the Codex analysis was delivered.

**Resume 3:** same job and limits. The user asked to continue.
- Unit `codexa-train-codexa-934m-foundation-5h-v1-resume-1791743832`, trainer pid 65183.
- Original deadline 21:20:35 UTC.

**New:** an interactive codex/claude Kitty window opens at each terminal event.
- Read-only sandbox, on-request approvals.
- Runs in its own user unit; at most once per event.

**Late-binding notifier:** the running supervisor uses the notification code on disk when the run ends.

**Unchanged:** hash-bound files; `verify_job` passes.

**Tests:** 80 passed (notifications, supervisor and dashboard).


## 2026-10-11 ~18:50 UTC — every training start opens the Rich dashboard

**Cause:** the trainer's `start_chat_viewer` hard-coded the old 100x22 viewer.

**Fix:**
- New `LLM_TRAINING_VIEWER_COMMAND` hook in LLM-Training. The LLM-Training AGENTS viewer rule was generalised at the user's request.
- Integration attach mode opens or reuses one 120x30 dashboard in its own user unit.
- `write_launch` sets the hook.

**Tests:** 85 integration tests passed and 24 LLM-Training tests passed.

**Wheel:** rebuilt in the scratchpad. It differs from the installed copy only in `conversational_training.py`.
- Installing it during the live run was refused by the tool permission policy. It is pending the user's install, either now or after the run ends.
- Until it is installed, starts fall back to the old viewer.


## 2026-10-12 02:51 UTC — user-granted extension to finish 1,792 updates

**Previous stop:** the run completed at 21:12 UTC on 2026-10-11 with `TRAINING_DEADLINE` at update 1,697 of 1,792.
- Validation 3.11296 → 3.09144; checkpoint verified.

**Grant:** the user said "it can finish the 1792 runs dw".

**New `scripts/training_supervisor.py extend`:**
- Allowed only for a verified deadline stop below the step cap, with an explicit statement.
- Moves only the deadline: +2,400 s, of which 600 s is the reserve.
- Records the grant in `config/budget-extensions.jsonl`, including the previous budget.
- Re-runs every resume gate; if any fails, the budget is restored.
- Then performs the same exact resume (segment archived).

**Run:** unit `…-resume-1791773504`, trainer pid 117318, same trainer run_id `d4080f1c`. Update 1,699 was confirmed with 2,158 MiB of VRAM free.

**Packages:** the LLM-Training wheel with the viewer hook was installed while no trainer was running. `pip check` is clean and `verify_job` passes.
- The start opened the 120x30 Rich dashboard in its own unit (`codexa-dashboard-…`).
- The old viewer did not open.

**Tests:** 86 passed.

2026-10-12: Added verified foundation step-1792 checkpoint to ubume-dev as codexa-934m-foundation-5h (SHA256 ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e). Preserved existing model choices; no training started.


## 2026-10-12 — Post-training generation diagnostic completed

Investigated the verified foundation step-1792 checkpoint against its authentic base initializer, without training, optimizer steps, checkpoint writes or production changes. Evidence: `outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256` (private, ignored). Final/base/tokenizer SHA-256 identities are preserved. Both 524,288-target validations reproduce CE exactly; 96 historical greedy generations reproduce their respective recorded token arrays. Across baseline and final, 35/48 token sequences change while exact/repetition/length failure aggregates remain unchanged.

Complete selected-target scan finds 122,741 EOS targets and zero role/END targets; exact run source attribution is 71,696,026 FineWeb-Edu and 45,744,486 Wikipedia targets. Validation contains zero EOS/END targets. Sampling reduces repetition without demonstrating useful answers. A separate masked-cache position-offset defect is confirmed and corrected only through isolated call arguments; the historical generator does not use that mask path. BF16 branch ties and long-prefix numerical screening failures are preserved and localized.

Created isolated diagnostic runner, report renderer and focused offline tests. Tests: 11 new checks plus 6 existing generation regressions passed; optimizer-bearing checkpoint fixture deselected. Human semantic review remains pending. Seven reports, raw generations, JSON/CSV metrics, numerical traces, blinded review packet and checked training/memory charts are retained. Proposed only a separately authorized, bounded complete-turn SFT feasibility pilot with usefulness gates; no long run is justified or started. No commit, push, PR, dependency change or asset deletion.


### 2026-10-12 — Diagnostic report consolidation and ordered evidence

User requested a single navigable report and cleanup. Canonical report is now `outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256/REPORT.md`; supersedes the seven separate diagnostic reports and README, retained under `archive/reports/`. Top level contains only REPORT.md, evidence/, figures/, archive/. All 109 original diagnostic files preserved byte-for-byte with SHA-256 relocation accounting. Organized evidence has generation, tokenizer, inference, data, training, verification folders; retry logs, older measurements, reproduction snapshot and helper are archived. Renderer supports flat input and organized rebuilding without model loading. Verified 131 report links, 20 source links, all 72 original generation excerpts, structured evidence readability and byte-identical repeat rendering; four bounded offline organization tests passed. No training, model evaluation, optimizer updates, checkpoint changes, commits or dependency installs.


### 2026-10-12 — Conversational feasibility preparation (NOT READY)

Preparation only; no optimizer construction, backward pass, training update or launch. Independent private namespace: `outputs/preparation/codexa-934m-conversational-feasibility/20261012T051805Z-a5c8725c`, entry point REPORT.md. Reverified final checkpoint SHA ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e and protected original datasets/tokenizer/run records; component code preserved. Authored 64 constrained examples with explicit correctness oracles, 1,962 context tokens and 263 supervised tokens including END per pass. Reviewed 365 source conversations: 170 admission judgments, 195 rejects; three duplicate exclusions leave 167 provisional source candidates (153 UltraChat, 14 OASST1), 25,222 context tokens / 9,049 supervised tokens. No complete 512/32 manifest or invented baseline. Gold validation selection, baseline inference, full runtime/supervisor integration and final budget validation remain blocked. Ten offline tests and guarded preparation validation passed (audit EV-000010/EV-000011). Audit installed but automatic Codex hooks and both clients' full lifecycle/E2E remain unverified; use own manual session, preserve earlier records. No model/source-data/production edits, commits, upgrades or deletions.


### 2026-10-12 — Masked KV-cache position repair (LLM-Architecture)

Repaired the confirmed latent defect in `LanguageModel.forward` (LLM-Architecture `src/llm_architecture/model.py`, SHA-256 b7f857… → 3d4e9d1a…). There were two coupled causes. Default positions with an `attention_mask` were `cumsum(mask) - 1`, which ignored `past_length`. Attention also treated every cached key as valid (`past_mask = ones`), and the cache tuple stores no mask, so an offset alone cannot handle padded caches.

API contract, now documented in the `forward` docstring:
- `attention_mask` is either `[batch, past_length + sequence]` (full mask; required for correct padded caches; the caller keeps it across steps) or `[batch, sequence]` (current tokens; the cached prefix is defined as fully valid).
- Default positions are `cumsum(full_mask) - 1` over the current tokens.
- Unmasked paths are unchanged.
- Explicit negative `position_ids` are rejected, and any other mask shape raises `ValueError`.
- `CausalSelfAttention` accepts both mask forms.
- Weights, state-dict keys, cache tuple, tokenizer and checkpoint format are unchanged.

Read-only validation (`scripts/masked_cache_repair_validation.py`; private evidence `outputs/diagnostics/masked-kv-cache-repair/20261012T060900Z/REPORT.md`) ran on both checkpoints, with SHA unchanged and no optimizer:
- The FP32 masked cached max logit error falls from 5.794949 to 1.907e-05 on the final checkpoint, and from 6.122253 to 1.431e-05 on the baseline.
- The full-mask left-padded cached step is ≤2.34e-05.
- The original 48-prompt BF16 greedy benchmark gives 48/48 historical token matches before and after on both checkpoints, with 48/48 before/after identical generations and bit-identical unmasked logits digests.

New tests are `LLM-Architecture/tests/test_kv_cache_masking.py` (57) and `LLM-Inference/tests/test_cached_generation_regression.py` (16). The integration diagnostic test was updated to assert the repaired behavior. The architecture tests failed 41/57 before the fix (TDD). Focused suites after the fix: Architecture 64 passed, Inference 32 passed, integration 24 passed with 1 optimizer-constructing CLI test deselected, Training mask check 1 passed; 0 failed, 0 skipped.

Consequences:
- `llm_architecture.model` is hash-bound into the COMPLETED run `codexa-934m-foundation-5h-v1` `code_identity`. `verify_job`/resume and the old readiness identity checks will now refuse that run (accepted by the user; no resume is intended).
- The `.venv` site-packages copy of `llm_architecture` was not reinstalled. Consumers outside `workspace_bootstrap` keep the old behavior.
- No production caller passed masks with caches, so historical generation is unaffected. Conversational quality is not improved by this repair.

No training, optimizer, checkpoint write, dataset change, commit or dependency install.


### 2026-10-12 — Post-repair integration, historical environment and dual-agent audit enforcement

**Commits.** Local commits, all through the audit pre-commit hook; no push, no PR:

- LLM-Architecture `bdf3683`: masked KV-cache repair and tests;
- LLM-Inference `4758dee`: cached generation regressions;
- LLM-From-Scratch `55812c6`: diagnostic tooling;
- LLM-From-Scratch `a763027`: repair validator;
- LLM-From-Scratch `79a75c6`: environment docs and tooling, pins advanced to bdf3683/4758dee, with `historical_identity` keeping c334620/48a9f4b;
- LLM-From-Scratch `fc022b3`: migration byte-parity test pinned to historical c334620, after it failed against the intentional repair.

**Environments.** No `.venv` mutation and no new venv.

- `.venv` without the bootstrap is the historical environment. It still reproduces `codexa-934m-foundation-5h-v1` `code_identity` exactly, because `verify_job` resolves the installed wheels. This corrects the earlier note that the repair broke that run's verification; that holds only source-first.
- Current development is source-first: bootstrap scripts, the `codexa` CLI, pytest and the new `tools/codexa-dev`.
- `scripts/environment_identity.py --expect current|historical` reports the context.
- 10 of the 12 hash-bound sources were never committed. They are preserved, read-only, in `outputs/preservation/historical-source/codexa-934m-foundation-5h-v1` (231 files, 12/12 match).
- 48/48 historical generations hold in both environments on both checkpoints.
- Decision needed later: the supervisor and the trainer still resolve the historical wheels. Future training must choose its environment explicitly.

**Audit.**

- Doctor separates isolated from real-workspace E2E and reports per-agent ENFORCEMENT.
- The user approved the Codex hooks via `/hooks`. The first attempt failed because the running Codex had cwd `~`.
- Real-workspace E2E: Claude 9/9; Codex 9/9 (side-effect variant). Two explicit Codex runs scored 7/9 because Codex documented proactively.
- Doctor: claude ENFORCED, codex ENFORCED.
- `workspace-infrastructure` is not under git.

Report: `outputs/diagnostics/post-repair-integration/20261012T095454Z/REPORT.md`. No training, optimizer, checkpoint, tokenizer or dataset changes.

## 2026-10-12 — Phase 4 conversational SFT preparation (Claude Code; no training)

- Single report: `outputs/preparation/codexa-934m-sft-phase4/20261012T145429Z-643f4379/SFT_PHASE4_READINESS_REPORT.md`. Verdict is stated in section A.
- Frozen manifest `f5514dceabcfbb462a0a2173daf0947eed56de32ea174499cc3c47d8f6d54509`:
  - training: 224 OASST1 + 224 UltraChat + 64 authored;
  - validation: 16 + 16;
  - 1,024 presentations in seed-73 then seed-74 order;
  - 25,513 assistant-supervised training tokens per pass.
- Admission required both an agent first-pass admit and a lead second-pass admit. This is engineering review, not human review.
- The 30% category cap was not raised.
- The OASST1 validation pool was exhausted at 15 records, so a documented general-chat-v2 test-split extension supplied the rest.
- Step zero was sealed with the same evaluation code the pilot would use. Results:
  - 0/16 natural endings and 15/16 severe degeneration;
  - gold END NLL 18.16;
  - reference CE equal to the diagnostic's;
  - 64/64 greedy generations identical to the diagnostic.
- Budget: the full evaluation schedule needs about 974 s, which exceeds the proposed 900 s. Choosing a 1,200 s cap or a reduced intermediate schedule is a decision for authorization.
- The SFT runtime consumes Codex's `tools/training_environment_guard.py` read-only. It currently refuses a new development job because of uncommitted executed source.
- The commit is blocked: the Phase 4 modules depend on Codex-owned, uncommitted `sft_feasibility.py`.

## 2026-10-12 — Step 5A integration and SFT pilot readiness gate (Claude Code; no training)

- Single report: `outputs/preparation/codexa-934m-sft-step5a/20261012T170336Z-3d5556ba/CODEXA_STEP5A_FINAL_READINESS.md`. Verdict is stated in section A; readiness is not authorization.
- Uncommitted completed work in Architecture, Data, Training, Inference and the integration repository was reviewed, tested and committed in scoped commits (no push). Pins moved to the new HEADs; `historical_identity` is unchanged.
  - 11 of the 12 hash-bound historical files are now in Git byte-identical to the job hashes.
  - `model.py` differs only by the intended masked KV-cache repair.
- Codex's guard now accepts a new development job.
  - Historical verification and the source archive (231 files, 12/12 identities) still pass.
  - The guard itself was not modified.
- Test repairs:
  - The guard-broken trainer tests select `development` explicitly; the real guard validates the real workspace.
  - The supervisor "timeout" was a missing selection plus unbounded polling, now bounded and fail-fast.
  - Codex's dirty-tree test uses an isolated Git workspace.
- Frozen SFT job v2 `924da9f34272f3d2c80ae43f6eec0cd12b9dbdb6ec67026726961e08dff067a3`. It reuses the Phase 4 data manifest and step zero, and binds the guard report and component commits.
  - Proposed cap 1,200 s with a 180 s reserve; intermediate evaluations are reduced (greedy-16, gold, reference), the final one is full.
  - The data was independently re-derived: 25,513 supervised tokens per pass, isolation clean.
  - The validation exception record is isolated.
  - Earlier SFT models are siblings, not ancestors, of the initialization checkpoint.
  - Step zero reconfirmed 16/16 token-identical under the committed identity.
- GO/NO-GO: `tools/codexa-dev scripts/sft_readiness.py --experiment <job.json>`. Authorization-time commands are `scripts/sft_pilot.py approve|launch`; neither was run.

## 2026-10-12 — Step 5B SFT pilot executed; 10-hour SFT not recommended (Claude Code)

- **Job.** Step 5A job 924da9f3 had been superseded by the Step 5B re-freeze f36f5478. The re-freeze fixed the final-save and deadline-state trainer defects (a9adc65, 6d3e522) and kept the same experiment settings. The user chose f36f5478.
- **Preflight.** Readiness was READY FOR AUTHORIZATION, 24/24 checks (free VRAM 14,775 MiB).
- **Launch.** The user ran `approve` and `launch` themselves (wakeup provider claude). The recorded statement is the literal placeholder `I authorize ...`.
- **Outcome.** COMPLETED, STEP_CAP: 128 updates, 51,026 supervised tokens, 331 s.
  - Final checkpoint 43002c4d and update-64 checkpoint 2ddf6d80, both re-hashed independently.
  - Foundation ca779a16 unchanged.
- **Acceptance: FAIL.**
  - Severe repetition 5/16 (limit 4); strict accuracy 0/7.
  - END, natural endings, premature END and reference CE PASS.
  - Human review PENDING.
  - All behavioural metrics flat from update 32 to 128; historical-nucleus repetition 15 to 30.
- **Decision.**
  - No promotion, no resume, no extension, no second run.
  - A 10-hour General Chat v2 SFT is not recommended.
  - Next: a bounded investigation (decoding, data, a foundation completion probe, a real blinded review) and consider pretraining. The foundation has about 773M tokens, about 0.8 tokens per parameter.
- **Defects found.**
  - The supervisor completion schema has null validation and generations for SFT runs.
  - The handoff text uses the foundation-run template.
  - There is no per-update metrics file and peak VRAM is not logged.
- **Separate change.** The run-end agent window is now 100x22 and the dashboard stays 120x30 (bb5cd0f; PR #6).
- **Reports.** `outputs/diagnostics/codexa-934m-sft-pilot-v1-results/20261012T211243Z/`.

## 2026-10-12 — Post-SFT foundation assessment and training instrumentation repair (Claude Code; no training)

- **Telemetry.** New `training_telemetry` gives append-only, deduplicated, crash- and resume-safe per-update records:
  - loss, learning rate, pre-clip and derived post-clip gradient norm, tokens;
  - allocated, reserved, peak and driver-free VRAM, throughput;
  - evaluation and checkpoint timing, stop records.
  - It is wired into `sft_pilot_trainer`. Telemetry failures are never fatal.
- **Reporting.** New `run_reporting` distinguishes SFT and foundation jobs, reads the real SFT evaluations and the sealed baseline, separates execution from quality, names the best non-promoted candidate, and marks missing data as unavailable.
  - `training_notifications` writes a new `supervisor/run-summary.json` and a job-aware handoff and prompt, falling back to the legacy text on error.
  - `completion.json`, `training_supervisor` and `training_wakeup` are unchanged (hash-bound).
- **Capability probe** (inference only, 3 checkpoints, 360 rows):
  - foundation plain-form top-1 26/56 (p=0.0007); arithmetic, grammar and reasoning at chance;
  - SFT changed the chat surface, not knowledge; P(END after a bare answer) ≈ 0; 0 exact answers in every configuration;
  - a repetition penalty removes loops but halves correct statements.
- **Data** (frozen set read-only): 30% of targets restate the prompt (UltraChat 42%); the authored constrained examples are 1% of supervised tokens, in 4 rigid templates; 2 of 7 strict cases expect bare answers to open questions.
- **Strategy.** Continuing the 934M beats a fresh 150–350M model at 10 h and is level at 50–300 h. Throughput is measured (8,012 vs 52,851 tok/s); the loss fit is relative only.
  - Correction to the earlier 10-hour note: steady throughput is 8,012 tok/s, not 3,024.
  - Verdict: READY TO PREPARE NEW FOUNDATION EXPERIMENT. No job was prepared or launched.
- **Commits** (local, not pushed): 9427fa0, fbcc5c0, eccd16a. Audit CLD-000029 to CLD-000031.

## 2026-10-13 — Step 6: prepare the 10-hour foundation continuation (Claude Code; no training)

- **New job identity** `codexa-934m-foundation-continuation-10h-v1`, not an exact continuation of the predecessor:
  - weights only from ca779a16 with a fresh AdamW8bit optimizer and a fresh, independent cosine schedule;
  - the predecessor's schedule had already decayed to its floor, its order was exhausted and its code identity is superseded;
  - hyperparameters are identical to the predecessor apart from the cap and cadences;
  - the 2e-5 peak is kept as the proven, conservative value; no new hyperparameter was invented.
- **Data:** the predecessor's seed-42 permutation continued from position 57,344 (114,688 windows):
  - verified disjoint from the 5-hour run, validation and base-v1's reconstructed positions;
  - exact source mix 61.10/38.90 from index metadata;
  - full corpus SHA-256 matches, and every token is in range;
  - base-v1 exclusion is positional and reconstructed, not content-level.
- **Budget:** at most 36,000 s wall-clock including finalization (SessionBudget), with a supervisor backstop at +120 s. Cap 3,584 updates, the largest multiple of the cadence whose p95 plan leaves at least 10% slack. Reserve 540 s, which is 1.25 × the measured final validation, save/verify and full evaluation.
- **Evaluation:**
  - validation, then plain-form capability scoring every 224 updates;
  - full evaluation at 896, 1,792 and 2,688 and at the end: historical 48 + both-form scoring + greedy generations;
  - capability guard: McNemar p < 0.01 plus a lower margin, at 2 consecutive judgements.
- **Telemetry:** moved into LLM-Training (`llm_training.telemetry`, aliased by the integration). `pretrain_run` now records updates, gates, evaluations, checkpoints, stop and recovery state. Real tiny-run evidence exposed two defects, both fixed before freezing:
  - CPU VRAM was logged as 0; it is now null;
  - an all-writes-failed run reported 'unavailable'; it now reports 'degraded'.
- **Benchmark:** v1, the 56 + 8 items, is preserved byte-identical. v2 adds 224 + 8 new items (32 per category). Measured per item: likelihood rank, length-normalized rank, margin, both option orders, P(END), and greedy and repetition-penalty generations; Wilson and McNemar statistics. Independent human review of v2 is pending.
- **Baseline** b9e0a920 (inference only): grammar choices follow word frequency (0 order-sensitive items); arithmetic and reasoning at chance; the repetition penalty removes repetition but lowers correct statements.
- **First frozen attempt** a5373cf3 is superseded and preserved:
  - its paths were relative;
  - its trial readiness exposed the btrfs st_dev identity defect, fixed in 046af65.
- **Readiness for 6d9d8453:** 0 engineering failures (focused tests 24 + 58 pass). CONDITIONALLY READY, because free disk is 43.7 MB below the 180 GiB gate. The space was taken by the SFT pilot checkpoints and new user files. Nothing was deleted and no gate was lowered.
- **Regression:** integration 489 passed, all components pass, audit 65 passed.
- **Commits:** local only. No push, no PR, PR #6 untouched.


## 2026-10-13 — Dedicated encrypted SSD asset cutover

- Copied 75 Codexa checkpoint/export payloads and 141 small companion files to
  `/mnt/codexa`; original assets remain intact. Datasets, logs, repositories,
  environments, external model caches and Windows storage were not migrated.
- Saved PyTorch inputs use `/mnt/codexa/pytorch/saved`; selected workspace
  checkpoint stores retain relative layout under `pytorch/workspace-outputs`.
  NumPy inputs use `/mnt/codexa/numpy/current`, with deprecated and smoke assets
  in separate siblings. Dataset/log/recovery references remain on original storage.
- Fresh source/destination SHA-256 inventories matched. Native model schemas and
  three component-store snapshots loaded; representative PyTorch and NumPy CPU
  inference agreed with originals. NumPy optimizer/RNG restoration and bounded
  native smoke optimizer/scheduler loading passed without training updates.
- All five HF-format Codexa exports loaded offline and passed CPU inference; all
  five GGUF exports passed structure/tokenizer parsing. GGUF runtime generation
  and large CUDA optimizer/RNG restoration were not executed.
- Local asset roots, specific checkpoint aliases and six-model native registry
  paths changed without altering identities/hashes. Relative historical input
  references resolve to copied files. Optional mount-UUID checks prevent missing
  volume fallback; copied run trees and retained asset sources are read-only
  output destinations. Canonical source-first bridge readiness was verified.
- Historical installed wheels/frozen training code, production gates and output
  roots remain unchanged. Resolver changes remain uncommitted, so normal
  committed-source approval is still needed before a new production launch.
- SELinux remains enforcing; asset directories/files have user_home_t labels.
  A persistent fcontext rule requires the prepared local administrator step;
  its completion is recorded in private migration evidence, not assumed here.
- No sources were deleted or deduplicated. Fedora capacity is therefore unchanged.
  Independent protected backup and restore testing plus separate authorization
  are required before original retirement. Header/config backup remains protected
  on the encrypted Fedora system; an independent copy is still required.
- Operational commands, exact drive identity, rollback instructions and validation
  artifacts are documented in Documents/codexa-migration outside Git.
