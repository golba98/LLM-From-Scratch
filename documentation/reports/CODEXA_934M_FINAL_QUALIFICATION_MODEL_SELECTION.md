# CODEXA 934M — Final Qualification & Model Selection

11 October 2026. Execution evidence, not production promotion.

## A. Resource readiness

Final encrypted Btrfs free space: **168.879 GiB**, versus unchanged 180 GiB minimum: **11.121 GiB shortfall**. RAM available 16.778 GiB passes 16 GiB; swap used 7.998 GiB is high. GPU free 12760 MiB passes the existing qualification minimum, but the actual allocator warning makes conversational-training headroom unqualified. Disk blocks another GPU job.

Job-entry gates passed; qualification artifacts remain preserved. Free space fell from 195.093 GiB to the value above; this net filesystem change includes concurrent workstation activity and is not exclusively attributable to this task. No cleanup was performed. Python 3.14.7, Torch 2.13.0+cu130, CUDA 13, driver 615.71.09, bitsandbytes 0.50.0; integration .venv used.

## B. Dataset review

**576 machine-valid, zero machine-rejected, zero human-approved, 156 pending human review.** Schema, independently calculated arithmetic, explicit output constraints, multi-turn reference checks, tokenizer compatibility, shifted assistant-only masks, END supervision and provenance checks passed. Exact historical whole-conversation and development prompt overlaps were not found. This does not establish semantic/template independence or historical nonexposure. Highly templated curated material remains a limitation.

Interactive packages under `outputs/qualification/codexa-934m-20261011-v1/review/`: `dataset/review.html` (156), `development/review.html` (240), `blinded-models/review.html` (480 anonymous responses, 96/model). Export genuine decisions before closing. No approvals were fabricated. The protected 120-case acceptance suite was not evaluated or used for tuning; its byte fingerprint is unchanged.

## C. Five-model comparison

All five completed 240 frozen development cases under one original 3,600-second deadline; cumulative elapsed 1,240.992 seconds. Greedy decoding, seed42, maximum128 tokens, repetition penalty1, identical tokenizer and serialization. Objective216 cases; retention24 requires human review.

| Model | Instruction | Factual exact-format | Arithmetic | Raw END | Correct END | Severe degeneration | Human coherence |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 0.0% | 0.0% | 0.0% | 0.00% | 0.0% | 98.75% | Pending |
| repair | 0.0% | 0.0% | 0.0% | 69.17% | 0.0% | 28.75% | Pending |
| balanced | 0.0% | 0.0% | 0.0% | 67.92% | 0.0% | 25.00% | Pending |
| original_2h | 0.0% | 0.0% | 0.0% | 64.17% | 0.0% | 34.58% | Pending |
| END3 | 0.0% | 0.0% | 0.0% | 72.50% | 0.0% | 25.00% | Pending |

These strict constrained-answer failures do not independently prove that all factual knowledge is absent. Factual content and coherence require blinded review. Correct END requires a correct completed answer; raw END is not an answer-quality score. All 64 format-compliance cases failed for each model.

| Model | Missing END | Forced cutoff | Premature END candidate | Empty | Role failure |
|---|---:|---:|---:|---:|---:|
| base | 100.00% | 100.00% | 0.00% | 0.00% | 0.00% |
| repair | 30.83% | 30.83% | 0.00% | 0.00% | 0.00% |
| balanced | 32.08% | 32.08% | 0.00% | 0.00% | 0.00% |
| original_2h | 35.83% | 35.83% | 0.00% | 0.00% | 0.00% |
| END3 | 27.50% | 27.50% | 0.00% | 0.00% | 0.00% |

Premature-END candidate is an automated heuristic, not verified human semantic completion. All ten paired comparisons have zero observed strict-score difference with family-cluster bootstrap interval [0,0]; this failure floor does not prove broad equivalence.

| Model | Common CE | OASST CE | UltraChat CE | END CE |
|---|---:|---:|---:|---:|
| base | 2.52670 | 3.17146 | 2.30606 | 19.59137 |
| repair | 2.26721 | 2.47921 | 2.19472 | 0.95389 |
| balanced | 1.73238 | 1.56981 | 1.78793 | 0.71273 |
| original_2h | 2.03447 | 2.39264 | 1.91193 | 1.02337 |
| END3 | 2.13137 | 2.44128 | 2.02525 | 0.65031 |

Common validation has64 distinct roots (32/source),40,923 assistant tokens and143 END targets. Historical exposure is not excluded; this is development loss, not sealed generalization evidence. All1,200 raw outputs and exact fingerprints are retained in `baseline-generations/`; paired evidence is `paired-comparisons.json`. Representative failures include repair responding to53+9 with “42 plus 9 = 42.” and balanced responding “The number of people who have 53 plus 9 is 33.” These illustrate errors, not a selected-success gallery.

## D. GPU checkpoint qualification

Executed actual934,356,480-parameter model, context2048, AdamW8bit, FP32 weights/BF16 autocast, gradient checkpointing. Eight logical trajectory updates plus replay of the lastfour means12 physical optimizer calls. Elapsed **135.770s**, under900s. Exact model/optimizer/scheduler/CPU-CUDA RNG/scaler/counters/synthetic-cursor restoration and subsequent CUDA updates passed bitwise. Previous fallback was explicitly injected without changing historical files.

| Measurement | Native | Optimized | Decision |
|---|---:|---:|---|
| GPU restore | 4.293s | 5.017s | +16.886%; FAIL ≤10% |
| Warm update median | 0.307128s | 0.307647s | +0.169%; PASS ≤5% |
| Initial save | 14.272s | 10.445s | Single observed timings |
| Final save | 19.790s | 12.498s | Single observed timings |

Peak GPU allocated 9.057 GiB; reserved 11.859 GiB; host RSS 12.507 GiB (includes mapped files). A100,663,296-byte allocator request emitted an OOM retry warning; execution recovered and completed all updates. This is a headroom concern, not an unreported failed optimizer step.

Actual checkpoint5.249 GiB: FP32 model3,737,425,920 bytes; optimizer uint8 1,868,562,432 +FP32 29,800,448 bytes. No FP32-Adam assumption, downcast or quantization was introduced. Source END3 scheduler is already at its completed minimum; equality here does not establish varying-LR scheduler behavior. Synthetic microbatch1 throughput (~3,334 assistant tokens/s warm) is not conversational accumulation8 throughput. Logical counters exclude replay; physical compute is documented separately.

**Full qualification failed its restore gate; production_ready=false.** Single-pair latency needs repeatable authorized profiling before promotion. Read-only CPU profiles show hashing dominates (~98.7% optimized time); native wrapper performed two integrity passes versus one optimized component traversal. Those CPU timings are not apples-to-apples and do not overturn the GPU failure. No integrity verification was weakened and no checkpoint architecture was rewritten.

## E. Initializer recommendation

**None selected.** Every candidate fails the frozen minimum instruction/arithmetic/factual/correct-END eligibility gates. Human reviews are also pending. Candidate paths and complete SHA256 values remain in `baseline-generations/comparison.json` and the approval-bound plans. Selecting balanced on lower CE or END3 on stopping alone would be unjustified.

The base received655.36 million tokens for934.36 million parameters (~0.70 tokens/parameter), making foundational undertraining plausible. A chat-format evaluation alone cannot separate missing knowledge from lack of instruction tuning; review the blinded factual/retention outputs before choosing a neutral-completion diagnostic or additional pretraining investigation.

## F. Final training-data plan

| Source | Records | Available assistant tokens | Proposed token share |
|---|---:|---:|---:|
| UltraChat |155,858|137,334,262|60%|
| OASST1 |27,034|10,250,006|39.5%|
| Curated |576|4,945|0.5%|

Manifest SHA256 `0fb10c887af281a44bc926111c06f30978007daac19ace2638dbddd1de1f7d71`; curated SHA256 `a35a4a6238f02279c8f199352a75e9523f8dd3ea9e296fffde7f4bb0c18ca035`. No admission. Finite two-window simulation:997,630 and992,652 tokens; no replacement within each window; sampler cursor reconstruction exact. Combined quotas: curated9,890, OASST786,225, UltraChat1,194,167. Pilot cap remains1,978,000 tokens; batch-boundary overshoot at most one eight-conversation optimizer batch.512-update cap may stop before all4,585 selected records. Curated maximumtwo passes is preserved.

Existing-mixture control remains unchanged. Improved versus control changes both source weighting and curation; any effect cannot be attributed solely to curated examples. Exact-hash deduplication is not proof of independent semantic training exposure.

## G. Bounded pilot readiness

**BLOCKED — not executable.** Existing `prepared-v3/chat-control.proposed.yaml` and `chat-improved.proposed.yaml` are unchanged, not materialized launch configurations. Proposed512 updates/3,600s, LR5e-6, micro1/acc8, BF16, clip1, warmup20, seed42, END weight1, loss evaluation64/generation128, quality guards, legacy six-state backend remain hypotheses. Missing: genuine dataset/development/model reviews, eligible initializer, successful full GPU qualification, disk reserve, sufficient demonstrated VRAM margin and a precise pilot approval. No training launch command is presented for nonexistent approved artifacts.

## H. Ten-hour feasibility

Available pool147,589,213 assistant tokens is not the usable duration under every finite mixture. A60/40 UltraChat/OASST window is OASST-limited to ~25.625M tokens: historical2,425 assistant tokens/s projects2.94h token processing before overhead, not ten measured useful hours. The reviewed-curated two-pass mixture is much smaller. Additional epochs need independent overfit evidence; do not cycle tiny curated data to spend time.

Legacy remains production default. Six5.249-GiB states plus temporarywrite/log allowance require ~37.745 GiB working allocation in addition to observing the unchanged180-GiB startup gate. No real-store reclamation authorized. No paired pilot acceptance exists, so the ten-hour proposal remains blocked and unchanged. Deadline/final-save reserve, quality deterioration stops and cumulative resume budget must remain active even when the user monitors training.

## I. Tests, implementation and preservation

Final command: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python -m pytest -q -rs` — **169 passed,5 skipped,8 warnings**,8.35s (`regression-final.log`). Five skips require optional sklearn specialist dependency; no dependency installed to conceal skips. Focused review13 passed; orchestration21 passed. Earlier repeated full runs162/164/166 passed are superseded, not added to inflate totals. Prior component suites remain recorded in the existing readiness folder; no sibling source changed this turn.

Preserved failed evidence: first baseline RAM refusal (15.693GiB) after base completed; repaired per-candidate process isolation continued within original deadline without regenerating base. One newly added Node review test initially failed due test-variable placement; fixed and subsequent runs passed. One test command named a nonexistent file; corrected. First CPU profiler called nonexistent RecoveryStore.load; corrected. A final inspection command lacked PYTHONPATH and failed import; corrected without changing installed wheels. No remaining test failure is hidden.

Integration-owned changes: `behavioral_readiness.py` (case-sensitive exact grading, correct-END eligibility, format scoring), `chat_evaluation.py` (family identity/format metrics), `conversational_readiness.py` and script (sequential owned candidate processes, completion evidence, original-deadline resume), new `qualification_review.py` and script (validated independent human-review packages/import), focused tests `test_behavioral_readiness.py`, `test_conversational_readiness_operator.py`, `test_qualification_review.py`. Special-token fraction expression was clarified and tested; it was not a discovered masking bug. Frozen source fingerprints distinguish this grading revision from older scores despite unchanged scoring-version label.

Final protected check: **1,059 device/inode/size/mtime/ctime identities unchanged**; this is metadata preservation evidence, not a fresh hash of every file. Five candidates had current fingerprint checks during authorized evaluation. All preserved proposals, curated data and sealed acceptance hashes unchanged. No historical checkpoint, canonical dataset, export, tokenizer, configuration or NumPy project was modified/deleted/migrated. New isolated GPU artifacts remain retained. Source changes do not automatically change installed packages: integration scripts use their established src import path; explicit PYTHONPATH is required for direct module development tests.

## J. Next required action

**Complete the blinded human model review first**, focusing on factual content and general-language retention. This determines whether constrained-format failure is masking recoverable knowledge or whether further SFT is poorly justified. Review approval alone does not override failed objective entry gates.

Final decisions: full-model optimized GPU qualification **BLOCKED — restore performance/headroom and current disk reserve remain**; matched conversational pilot **BLOCKED — requirements remain**; ten-hour conversational training **BLOCKED — no accepted pilot or justified useful-duration evidence**. No pilot or ten-hour run was started.
