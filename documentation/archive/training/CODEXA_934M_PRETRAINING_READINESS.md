# CODEXA 934M — PRE-TRAINING READINESS

Date: 10 October 2026. This report records actual CPU-safe preparation and verification. It does not authorize execution, promote a checkpoint, or claim conversational improvement.

## A. Executive assessment

Implemented streaming conversational audits, proposed provenance-tracked curriculum and finite token-aware sampling, strict behavioral evaluation, source/END-loss diagnostics, separate behavioral/validation checkpoint roles, quality guards and evidence-gated pilot preparation. Corrected a genuine training-controller NameError and strengthened the existing GPU qualification harness to compare midpoint-resumed versus uninterrupted continuation.

Infrastructure CPU regression gates pass. Model-quality readiness remains unproven. No full-model GPU qualification, five-model inference comparison, conversational pilot or ten-hour run was launched. No initializer has been selected. Historical low-quality results are not substituted for fresh measurements.

## B. Post-cleanup verification

Latest filesystem snapshot: **204.953 GiB free**, **24.953 GiB above** the unchanged 180 GiB floor. Free space fluctuates with unrelated workstation activity. Filesystem is encrypted Btrfs, data single/metadata DUP, zstd:1. `du` inode-accounted allocation: integration outputs 73.292 GiB; canonical checkpoints 44.119 GiB; canonical datasets 33.642 GiB. These are not exclusive Btrfs extent totals and must not be summed as guaranteed reclaimable capacity. Native Btrfs evidence is `btrfs-current.txt`; no filesystem changes were made.

Prior cleanup evidence reports 76 retired files, 23 checkpoints and 101.789 GiB net recovery, from 97.982 to 199.771 GiB free. That is a historical measured cleanup result, not space recovered by this assignment. No further deletion occurred.

All **25 protected native checkpoint/export files** passed fresh full SHA256 comparisons against the trusted cleanup records and actual CPU mapped native loads. The 1,059 protected device/inode/size/mtime/ctime identities remain unchanged. Metadata comparison alone is not content verification; the separate checkpoint SHA checks cover full payloads. Base, repair, balanced, Stage 2, original two-hour final/rollback and END3 final/best/previous survive. The canonical tokenizer matches SHA256 `a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e`; no tokenizer or model changes were made. New audit hashes verify all 11 conversational JSONLs. The large base token binary was not unnecessarily rehashed; existing trusted identity plus unchanged metadata is the present evidence.

Python 3.14.7; PyTorch 2.13.0+cu130; CUDA 13.0; driver 615.71.09; bitsandbytes 0.50.0. `pip check` passes. Installed Data/Training wheels were rebuilt and installed with `--no-deps --no-index`; seven repository source/import comparisons report no differences for installed generative components. Specialist is deliberately not installed in this generative profile. Full commits, dirty status, source fingerprints and module paths are in `prepared-v3/preparation.json`. Existing dirty work was preserved.

Latest MemAvailable **15.508 GiB**, below the required 16 GiB. Swap used **8.000 GiB**. GPU free **12,790 MiB** on RTX 4080 (16,376 MiB total); unrelated desktop GPU processes were left untouched. RAM and swap pressure remain operational blockers. These are snapshots, not reserved capacity.

## C. Full-model GPU qualification

Existing harness retained: `src/codexa_workspace/checkpoint_gpu_qualification.py` and its script wrapper. Native state identity is checked before execution. The corrected harness restores scheduled learning rate, counters and synthetic sampler cursor, saves a midpoint, and compares uninterrupted final state to resumed final state including model/optimizer tensor bytes and dtypes, CPU/CUDA RNG, scheduler/scaler and counters. Explicit previous-state fallback remains auditable. Native checkpoint loading is compared to the input payload. Eight controlled updates at sequence length 2048 are required for full qualification, not a reduced shape substituted as equivalent.

Actual native END3 payload is 5,636,070,993 bytes: FP32 model storage 3,737,425,920 bytes; uint8 AdamW8bit storage 1,868,562,432 bytes plus FP32 optimizer storage 29,800,448 bytes. Budget uses this representation, not hypothetical FP32 Adam moments. Qualification peak projection is **32.494 GiB** (two native states, three component states, one staging payload, 1 GiB logs/allowance). Disk admission remains max(180 GiB, 100 GiB + projected peak). Latest disk passes, RAM fails. Synthetic CPU exact continuation passes; real AdamW8bit/CUDA determinism, latency, throughput and memory remain unmeasured.

Approval must bind input SHA, fresh output, updates, sequence length, harness/wrapper identities and 900-second ceiling. Wrapper supervises only its owned process, counts startup in the original deadline, sends graceful SIGTERM, then bounded forced termination if needed. It does not reset the budget. Qualification requires bitwise continuation, restore regression ≤10%, throughput regression ≤5%; approximate output similarity does not qualify. A finished native scheduler may clamp to minimum LR; this is not evidence of a varying-LR recovery test. Deterministic kernels/optimizer support must actually pass before promotion.

Read-only plan command (executed, no `--execute`):

```bash
.venv/bin/python scripts/checkpoint_gpu_qualification.py   --checkpoint outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints/slot-0/checkpoint.pt   --output outputs/checkpoint-gpu-codexa-934m-readiness-v1 --updates 8 --sequence-length 2048
```

Future execution is the same command with `--execute --approval-file <precise-approved-artifact.json>`, only after fresh disk/RAM/VRAM gates pass. No approved artifact was created. Original checkpoints remain read-only; optimized output is a fresh namespace. The optimizer and CUDA math were not changed.

## D. Dataset audit

Streaming audit traversed all 11 existing conversational JSONLs in 424.890 seconds. Full file SHA, record schema, root/content identities, compatible tokenizer serialization, masks/counts and bounded deterministic review reservoirs are recorded in `conversation-audit.json` and its 113.55 MiB SQLite ledger. Legacy supervised counts below are usable records at current 2048 context; rejected overlength/schema records are excluded, not silently truncated. General Chat v2 uses existing compatible token metadata, sampled every 997 records for actual serializer comparison. Canonical data was not rewritten or retokenized.

| Dataset / split | Conversations | Usable supervised tokens | Problems | Recommended action |
|---|---:|---:|---|---|
| chat-sft-balanced-v1/oasst1/train.jsonl | 30,000 | 11,063,653 | 814 rejected at current context/schema; 42 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-balanced-v1/ultrachat/train.jsonl | 30,000 | 23,075,735 | 3,918 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-repair-v1/train.jsonl | 9,627 | 1,922,761 | 0 rejected at current context/schema; 53 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-repair-v1/validation.jsonl | 2,469 | 495,839 | 0 rejected at current context/schema; 7 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-v1/oasst1/train.jsonl | 30,404 | 11,207,999 | 826 rejected at current context/schema; 42 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-v1/oasst1/validation.jsonl | 313 | 112,086 | 5 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-v1/ultrachat/train.jsonl | 196,882 | 150,989,740 | 25,460 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| chat-sft-v1/ultrachat/validation.jsonl | 1,902 | 1,462,008 | 244 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| general-chat-sft-v2/test.jsonl | 9,874 | 7,974,145 | 0 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| general-chat-sft-v2/train.jsonl | 182,892 | 147,584,268 | 0 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |
| general-chat-sft-v2/validation.jsonl | 10,135 | 8,178,234 | 0 rejected at current context/schema; 0 normalized repeats | Retain original; isolate current splits and historical exposure |

General Chat v2 train has 182,892 records and 147,584,268 assistant tokens. Source allocation remains strongly UltraChat dominated (approximately 93% assistant tokens); equal conversation sampling would not imply equal token exposure. The current v2 train/validation/test root/content grouping has no exact cross-split overlap. This does not prove historical cleanliness or semantic near-duplicate isolation.

Fresh historical comparison: **304 distinct roots / 926 validation records** share roots with repair training; **450 normalized whole conversations** match repair training. Repair train/validation shares **1,258 roots and 24 normalized contents**. Historical flags do not prove every flagged record was sampled by an earlier training job, so they are labeled potentially exposed. Known repeated repair anchors are excluded from the new curriculum.

Instruction tags are contextual heuristic candidates, not semantic certificates. General train candidates include only four one-word and one number-only cases, versus 653 arithmetic candidates; programming-output discussion is explicitly rejected by contextual checks. Human review reservoirs and malformed-record examples are preserved. A complete human semantic inventory has not been completed. Near-duplicate support is conservative five-word Jaccard ≥0.85, minimum 25 words, length ratio ≥0.8; it is a pair screening tool, not a complete corpus-wide semantic deduplication claim.

Raw OASST tree/language metadata was streamed separately. OASST is multilingual, with substantial Spanish exposure; language is not safely assumed from source name. UltraChat English is declared-source provenance, not per-record language classification. Full raw sizes/language counts/source-card provenance are in `language-provenance.json`. Local source cards record OASST Apache-2.0 and UltraChat MIT. This records source licensing evidence without asserting legal advice or downloading new content.

## E. Data improvements

New isolated curriculum: **576 unique records, 4,945 supervised assistant tokens**, SHA `a35a4a6238f02279c8f199352a75e9523f8dd3ea9e296fffde7f4bb0c18ca035`. It contains instruction constraints (120), arithmetic (120), facts (36), multi-turn (60), uncertainty (60), length control (60), natural conversation (60) and bounded reasoning (60). Twenty-four duplicate factual phrasing variants were removed. Original examples have explicit provenance and a CC0 proposal; originals remain intact. **156 records require human review**; automatic construction validation of the other 420 does not certify model learning or semantic diversity.

Fresh exact checks found no curated whole-conversation matches in the 11 audited historical files and no exact user-prompt matches with either proposed suite. This does not establish semantic/template independence. The proposal is not admitted for training. Admission requires reviewer identity, manual method, exact manifest fingerprint, approval and coverage of every pending record; admitted loader also checks the immutable review-file fingerprint and source hashes. Setting a generic admitted flag is insufficient. Current proposals remain blocked.

Finite token-quota sampler chooses source deficits in supervised-token space and samples without replacement in each source window. Epoch/cursor restoration reconstructs identical ordering. A window ends when its selected source is exhausted; no silent repeated cycling or fallback changes the mixture. Source exposure counters are recorded only for committed optimizer batches, excluding padding and failed partial accumulation.

Original 60% UltraChat / 25% OASST / 15% curated token proposal can support only about **65,933 total assistant tokens** with two curated passes. It is unsuitable for long adaptation. New low-exposure pilot hypothesis is **60% / 39.5% / 0.5%**, supporting about **1,978,000 total assistant tokens** with at most two curated passes. This is a capacity-derived hypothesis, not a proven optimum. Source balance, multilingual effects and quality must be evaluated against a matched existing-mixture control.

Protected acceptance is kept separate from training and development. It is a newly drafted original-family suite, not claimed historically unseen: no exhaustive historical exposure search or completed independent human review establishes that yet. The complete corpus near-duplicate/template-family audit remains incomplete. No claim of proven generalization is justified.

## F. Evaluation framework

Version `codexa-behavior-strict-v1`: **240 development cases** (instruction 48, factual 40, arithmetic 40, reasoning 24, multi-turn 32, uncertainty 16, retention/manual 24, length 16), plus **120 separate acceptance cases** reserved for post-selection evaluation. Development SHA: `c914c2d88537243aad6064394050a8e26078c69fa6a2cfcabc002bd13c132d66`. Case IDs, prompts, histories, references, categories, template families and fingerprints are preserved. Templates remain correlated; reaching 240 cases is not 240 independent capability observations. Human prompt/reference and open-response review remains mandatory.

Whole-answer grading replaces keyword reward: arithmetic requires the correct complete number, exact formats enforce casing/length/structure, duplicate JSON keys are rejected, and a reference keyword embedded in a contradictory answer fails. Open coherence/retention remains unscored (`None`) pending human judgment, not fabricated as objective correctness. Empty/prefix termination differs from wrong completed answers. Correct model END, premature END, missing END and maximum-token stop are reported separately. N-gram/span degeneration, unexpected role tokens, empty answers and excessive lengths remain separate metrics.

Paired comparisons require identical IDs, prompts, histories and categories. Family-cluster bootstrap uses 2,000 resamples with seed 42; uncertainty remains important on small correlated suites. Fixed generation is greedy, 128 maximum new tokens, no aggressive repetition penalty, common serialization and tokenizer. All raw responses/failures are retained.

Common source-stratified teacher-forced evaluation selects 32 distinct validation roots per source (64 examples) with seed 42. CE/END CE and supervised counts are logged separately by source. These historically exposed data are development loss diagnostics, not final clean acceptance. Extra diagnostic FP32 CE work must have its runtime/memory measured in the future real GPU shape.

Factual reference checks use primary [NASA planetary data](https://science.nasa.gov/solar-system/planet-sizes-and-locations-in-our-solar-system/), [IUPAC periodic table](https://iupac.org/what-we-do/periodic-table-of-elements/), and [NIST SI definitions](https://www.nist.gov/pml/special-publication-330/sp-330-section-2). Other facts and semantic quality remain subject to manual validation; no paid judge API was used.

## G. Initializer comparison

No new full-model generations were executed. The consistent five-model inference-only comparison is prepared with full checkpoint fingerprints, suite identity, decoding, source-stratified validation and source/wrapper hashes, under a separately approved 3,600-second supervisor budget. Historical 64-case results are insufficient and not entered as new scores.

| Checkpoint | Instruction | Factual | Repetition | END | Retention / assessment | SHA256 |
|---|---:|---:|---:|---:|---|---|
| base | Not measured | Not measured | Not measured | Not measured | Pending common-suite comparison | `808afec9923e3f00d91bcbef47643265ae5dbe6c5aeb4bd112c4e97be3d0a3a7` |
| repair | Not measured | Not measured | Not measured | Not measured | Pending common-suite comparison | `c55f4edd1477a8ffefd03ce63d033a291eb261f6047b129baa47e10756899a4f` |
| balanced | Not measured | Not measured | Not measured | Not measured | Pending common-suite comparison | `a4c8c172152dd8e3ef23a7dde05bc4a4355ff548f84fc904d8d5534dd2183892` |
| original_2h | Not measured | Not measured | Not measured | Not measured | Pending common-suite comparison | `374f3f1746b0ff0b24febc5176f3d20f5bb4ccadc3fb6aa24a120d1f31481918` |
| END3 | Not measured | Not measured | Not measured | Not measured | Pending common-suite comparison | `4f4e5d1540eb6813ad6d6f91f1a1a503bf8952632d51f9dc6410343a8e0c456b` |

All candidates remain available and compatible by native CPU load. Historical END3 improved stopping but not instruction/factual correctness. A weights-only new phase is not exact optimizer resume. Full baseline job fingerprint and paths are `baseline-plan.json`; raw generation outputs are intentionally absent until approval.

## H. Selected initializer

**None selected.** Common-suite generation and human semantic review are mandatory. The selection API rejects absent/incomplete candidate evaluations and requires reviewed factual/multi-turn/retention cases. A minimal entry floor (instruction ≥25%, arithmetic ≥25%, factual ≥50%, correct END ≥50%, degeneration ≤25%, no empty/role collapse) is an experimental eligibility gate, not a final assistant benchmark. If no candidate passes, recommend a bounded corrective investigation rather than ten-hour launch.

The historical base received 655,360,000 packed training tokens for 934,356,480 parameters (approximately 0.70 tokens per parameter). Limited pretraining is a plausible capability bottleneck; it is not established as the cause without the common baseline comparison. SFT cannot be promised to replace missing foundational knowledge.

## I. Bounded pilot configuration

Prepared proposals (not runnable launch manifests): `prepared-v3/chat-control.proposed.yaml` and `chat-improved.proposed.yaml`. Improved training manifest is `mixture-low-exposure.proposed.json`, pending human admission. Initializer deliberately null. Architecture unchanged: 934,356,480 parameters, 24 layers, width 1536, 24 heads, FFN 6144, learned positions, RMSNorm/SwiGLU/SDPA, tied embeddings, vocabulary 16,384, context 2048.

AdamW8bit; LR 5e-6; betas (0.9,0.95); epsilon 1e-8; weight decay 0.1; clip 1; warmup 20; minimum LR ratio 0.5; BF16 autocast; FP32 master weights; activation checkpointing; TF32/compile disabled; deterministic settings required; microbatch 1 × accumulation 8; seed 42. Complete conversations dynamically right-padded; no cross-conversation packing; existing one causal shift and assistant-only labels preserved. END weight **1** is control, not automatically END3.

Ceilings: 512 updates, 3,600 seconds including startup/evaluation/save, 1,978,000 assistant tokens, at most two source windows. CE every 64 updates, behavioral evaluation every 128 with fresh pre-update baseline. Checkpoint every 128; legacy six-slot store, separate latest/previous/behavioral-best/validation-best/final aliases; no real reclamation. Finalization reserve 600 seconds is provisional until measured; no partial accumulation committed at stop. Token ceiling may overshoot by one complete optimizer batch. Storage projection: six full states + staging + 1 GiB = **37.743 GiB**; no append-only milestones.

Quality regression patience 2; task/instruction/factual drop 0.05; correct END/END drop 0.10; degeneration increase 0.10; empty/role increase 0.05; special-token fraction max 0.25; minimum response-length ratio 0.30. Existing nonfinite, loss/gradient, deadline and save-failure protections remain. Proposed pilot acceptance before results: instruction gain ≥0.10, factual gain ≥0.05, paired objective CI lower bound >0, degeneration increase ≤0.02, no human-assessed retention regression. Matched control and proposed data must use same initializer, seed, optimizer and decoding. Human paired acceptance is not yet an implemented automatic scientific decision; no loss-only acceptance is allowed.

Exact safe preflight (executed):

```bash
.venv/bin/python scripts/conversational_readiness.py proposal-preflight   --output outputs/readiness/codexa-934m-20261010-v1/prepared-v3   --config outputs/readiness/codexa-934m-20261010-v1/prepared-v3/chat-improved.proposed.yaml
```

Result BLOCKED: no selected initializer, data/suite review, measured GPU shape, execution approval; host RAM fails. Tokenizer/train/validation fingerprints and disk/VRAM snapshot gates pass.

Future evidence sequence (all paths below are explicit inputs, not manufactured approvals):

```bash
.venv/bin/python scripts/conversational_readiness.py baseline-run --output outputs/readiness/codexa-934m-20261010-v1/prepared-v3 --approval-file approvals/inference-baseline.json
.venv/bin/python scripts/conversational_readiness.py admit-mixture --output outputs/readiness/codexa-934m-20261010-v1/admitted/mixture.admitted.json --manifest outputs/readiness/codexa-934m-20261010-v1/prepared-v3/mixture-low-exposure.proposed.json --review approvals/data-review.json
.venv/bin/python scripts/conversational_readiness.py select-initializer --output outputs/readiness/codexa-934m-20261010-v1/prepared-v3 --manifest outputs/readiness/codexa-934m-20261010-v1/prepared-v3/baseline-generations/comparison.json --review approvals/initializer-review.json > outputs/readiness/codexa-934m-20261010-v1/initializer-selection.json
.venv/bin/python scripts/conversational_readiness.py materialize --output outputs/readiness/codexa-934m-20261010-v1/prepared-v3 --config outputs/readiness/codexa-934m-20261010-v1/prepared-v3/chat-improved.proposed.yaml --selection outputs/readiness/codexa-934m-20261010-v1/initializer-selection.json --suite-review approvals/development-review.json --qualification outputs/checkpoint-gpu-codexa-934m-readiness-v1/qualification.json --admitted-mixture outputs/readiness/codexa-934m-20261010-v1/admitted/mixture.admitted.json
```

Use returned materialized configuration/preparation paths. They still require the precise bounded-pilot approval through `authorize-launch`; that action records approval and does not start training. Once all gates genuinely pass, existing commands are:

```bash
.venv/bin/python scripts/conversational_readiness.py authorize-launch --output outputs/preparation/codexa-934m-chat-readiness-improved-v1 --approval-file approvals/bounded-pilot.json
.venv/bin/python scripts/chat_training.py start --config outputs/preparation/codexa-934m-chat-readiness-improved-v1/operator-config.yaml
.venv/bin/python scripts/chat_training.py view --run codexa-934m-chat-readiness-improved-v1
.venv/bin/python scripts/chat_training.py logs --run codexa-934m-chat-readiness-improved-v1 --tail 50
.venv/bin/python scripts/chat_training.py stop --run codexa-934m-chat-readiness-improved-v1
.venv/bin/python scripts/chat_training.py resume --config outputs/preparation/codexa-934m-chat-readiness-improved-v1/operator-config.yaml
.venv/bin/python scripts/chat_training.py evaluate --config outputs/preparation/codexa-934m-chat-readiness-improved-v1/operator-config.yaml --suite development
```

Start/resume retain existing automatic Kitty viewer and original cumulative deadline semantics. Viewer is not the training owner. Do not execute these future commands now; missing artifacts make them fail intentionally. Protected acceptance is only post-selection, under manual protocol. Verify actual generated artifact filenames before future execution; report commands do not create approval records.

## J. Ten-hour training proposal

`prepared-v3/chat-10h.proposed.yaml` holds all identities/hyperparameters, 36,000-second ceiling, 5,000-update safety cap, one finite source window, legacy checkpointing and 180 GiB reserve. Initializer remains null; schedule/cadence/final reserve and throughput are unmeasured. Proposed 60/40 UltraChat/OASST assistant-token mixture has a minority-source window of approximately 25.6 million total assistant tokens, not all 147.6 million unique tokens. At the historical 2,425 assistant-token/s rate that is roughly three hours before evaluation/startup overhead, a projection only. Ten hours is an upper budget; do not force repeated tiny data to consume it.

The 5,000-update cap and 20-step warmup are placeholders inherited into a blocked proposal, not validated long-run choices. Ten-hour materialization refuses until paired pilot acceptance, measured schedule and accepted final mixture exist. Legacy remains production default; V3.1 real-store reclamation cannot be activated by this preparation. Separate full-model GPU qualification must demonstrate exact AdamW8bit continuation and real save/load/throughput first. No ten-hour launch command is represented as currently valid.

## K. Regression results and implementation ownership

Final affected repository invocations: **Training 107 passed**, **Data 25 passed**, **integration 159 passed / 5 skipped**. Earlier unchanged Architecture 6, Tokenizer 6, Inference 1 passed. This is **304 passing tests across repository invocations**, with overlap; not 304 unique tests. Integration skips are five optional specialist tests requiring absent scikit-learn. Training has two existing warnings and integration eight (including tests returning non-None and existing Torch warnings). No final failures. An initial wheel build command from the wrong working directory failed before build; preserved logs/empty task-owned directories do not indicate a successful build. Subsequent correct builds/install and import comparisons passed. Earlier suites included tiny existing CUDA precision fixtures, never the 934M model; final integration was run with `CUDA_VISIBLE_DEVICES=''`.

Commands: integration `.venv/bin/python -m pytest -q -rs`; siblings use the same absolute generative interpreter with repository workdir. Logs: `integration-final-tests.log`, `training-final-tests.log`, `data-final-tests.log`, unchanged component `LLM-*-tests.log`. Training/inference `--help` passed. `pip check` passed. Exact CPU continuation includes optimizer tensors, scheduler, RNG and sampler, not merely output equality. Tiny orchestrator boundary test verifies quality baseline before update, committed source counters and distinct best roles.

Modified/new files owned by this assignment:

- LLM-Data: `src/llm_data/conversational_audit.py`, `curated_chat.py`, `chat_mixture.py`; `tests/test_conversational_readiness.py`.
- LLM-Training: `src/llm_training/recovery.py`, `chat_run.py`, `quality_guard.py`, `training.py`; `tests/test_readiness_recovery.py`, `test_validation_strata.py`, `codexa_fixture_equality.py`.
- Integration: `src/codexa_workspace/behavioral_readiness.py`, `conversational_readiness.py`, `chat_evaluation.py`, `chat_operations.py`, `checkpoint_gpu_qualification.py`; `scripts/conversational_readiness.py`, `scripts/checkpoint_gpu_qualification.py`; `tests/test_behavioral_readiness.py`, `test_conversational_readiness_operator.py`; this report and session/progress entries. Prototype and final isolated data/spec/log artifacts remain ignored under outputs.

Source ownership is unchanged; no architecture/tokenizer/inference model mathematics, canonical datasets, original checkpoints, historical specifications or NumPy files were modified. Data/Training wheel installation is the intended environment change, with no dependency upgrade. Task persistent validation allocation is approximately 117 MiB, below 512 MiB. Ledger/data generation is bounded, not another multi-GiB investigation.

## L. Final readiness decision

| Activity | Decision | Exact remaining requirements |
|---|---|---|
| FULL-MODEL GPU CHECKPOINT QUALIFICATION | **BLOCKED — REQUIREMENTS REMAIN** | ≥16 GiB available RAM, acceptable pressure, fresh resource gates, precise 900-second approval; then actual eight-update full-model results |
| BOUNDED CONVERSATIONAL TRAINING PILOT | **BLOCKED — REQUIREMENTS REMAIN** | Approved inference comparison, eligible reviewed initializer, semantic dataset/suite admission, full GPU qualification, measured shape/cadence, precise bounded execution approval |
| TEN-HOUR CONVERSATIONAL TRAINING | **BLOCKED — REQUIREMENTS REMAIN** | Accepted paired behavioral pilot, final data mixture, measured LR/cadence/deadline/checkpoint costs and realistic coverage; fresh resources and separate ten-hour approval |

CPU preparation is implemented and verified. Dataset contamination and narrow curated coverage are explicitly exposed, not solved by renaming splits. No new behavioral score is fabricated. No original checkpoint was changed or regenerated, no cleanup occurred, no long training started, and no production backend was switched. Next concrete step is human dataset/suite review plus explicitly approved inference baseline and full-model qualification after RAM gates pass.
