# Codexa 934M — Foundational Pretraining Readiness

Date: 11 October 2026. Scope: offline analysis and preparation only.

## A. Existing model intelligence

The existing 1,200 generations were analyzed without generating new responses. All candidates failed the strict objective tasks; that does not establish absent knowledge in neutral completion. Semantic coherence and factual access remain human-review pending.

| Checkpoint | Common chat CE | Strict objective passes /216 | Median generated tokens | END /240 | Severe degeneration /240 | Forced cutoff /240 |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2.52670 | 0 | 128 | 0 | 237 | 240 |
| Repair | 2.26721 | 0 | 26 | 166 | 69 | 74 |
| Balanced | 1.73238 | 0 | 26.5 | 163 | 60 | 77 |
| Original 2h | 2.03447 | 0 | 32 | 154 | 83 | 86 |
| END3 | 2.13137 | 0 | 22 | 174 | 60 | 66 |

No empty answers or role-token failures were observed. Correct-END grading was zero across candidates because task correctness/compliance failed; raw END is a different metric. Premature-END heuristic flags were zero, which does not prove semantically complete answers. Common CE used a conversational population with uncertain historical exposure and cannot rank foundational language ability.

A conservative full-response recognizer screened 80 factual/arithmetic responses per model. It found no reference-consistent short answers, one recognized wrong answer for Balanced, and left the other 399 responses unknown. It does not classify long prose as incorrect or award keyword matches. Arithmetic references were independently recomputed. Factual reference consistency is not independent human factual review. See [behavior analysis](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/behavior-analysis.json) and [objective checks](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/objective-answer-analysis.json).

The [prioritized review queue](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/REVIEW_QUEUE.md) points to the existing anonymous interface: 40 matched factual/context/retention responses first, then 60 instruction responses. No model labels were disclosed or reviewer fields completed. Review complete answers for contradictions, relevance and coherence separately from format compliance, then export the interface's JSON before closing. Remaining packet decisions are still required. The 156 pending curated records remain unapproved; protected acceptance material was not used.

## B. Original pretraining assessment

Verified historical records show completion at the preset 10,000-update ceiling: microbatch 1, accumulation 32, context 2048, 65,536 targets/update, AdamW8bit, BF16 autocast, FP32 weights and activation checkpointing. Warmup was 2,000 updates, peak LR 2e-4 and final LR 2e-5. Exposure was 655,360,000 targets, or 0.7014 targets/parameter.

| Step | Historical validation CE |
|---|---:|
| 250 | 7.0169 |
| 2,000 | 4.6295 |
| 5,000 | 3.6689 |
| 9,000 | 3.3794 |
| 10,000 | 3.3491 |

The final 1,000 steps improved CE by approximately 0.0303. The run ended by schedule, rather than demonstrated convergence. Late training mean CE was approximately 3.0218 and mean gradient norm 0.4113; inspected logs show no catastrophic nonfinite event or obvious late divergence. These observations neither establish broad generalization nor exclude overfitting. Original validation contained only four FineWeb windows (8,192 targets), and its binary/index are missing.

Historical sampler exposure was reconstructed using the same Torch version, seed 42, DataLoader seed consumption and shuffled prefix; its existing prefix digest matched. This supports positional exclusion, not proof of every historic batch or semantic uniqueness. An earlier resume checkpoint is unavailable. Approximately 5.112% of packed positions were traversed. Continued learning and limited exposure make undertraining plausible, but chat-format failure, inference degeneration and deficient instruction tuning remain competing explanations. The bounded prior corpus sample contained no chat-role IDs; the base model cannot fairly be judged solely by chat-special-token behavior. No serializer mismatch was found previously.

Evidence hierarchy: completion, logged losses and configuration are directly recorded; exposure ordering is a validated reconstruction; inadequate foundational knowledge is a hypothesis requiring neutral-format evidence. See [history evidence](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/history-evidence.json).

## C. Corpus readiness and validation

The unchanged uint16 packed corpus has 12,819,847,234 tokens: FineWeb-Edu 7,832,893,798 (61.10%) and Wikipedia 4,986,953,436 (38.90%). The streamed index contains 13,383,283 documents and 6,259,691 complete context windows. Existing corpus payload digest was reused; the 25.64 GB binary was not fully rehashed or regenerated. The 3.303 GB document index was streamed three times: twice for selection and once for exact source budgeting. Bounded content checks inspected 5,194,808 token IDs across 1,024 candidates, rather than scanning the corpus payload.

New immutable selection arrays exclude 320,000 reconstructed historical windows and another 935 windows covering held-out document inputs and shifted targets. The training view contains 5,938,756 windows / 12,162,572,288 target positions. Held-out validation contains 256 windows / 524,288 targets: 128 document windows per source. Exact eligible source targets are FineWeb 7,430,961,405 and Wikipedia 4,731,610,883.

Validation label: **isolated under reconstructed base-lineage positional exposure; NOT guaranteed historically/content-unseen**. Entire candidate documents overlapping reconstructed exposure were rejected. Normalized exact hashes and bounded word-shingle family checks found no duplicate edges among 1,024 screened candidates. Full-corpus exact/near-duplicate contamination is not excluded. Candidate documents must fit a 2049-token window and be at most 16,384 tokens, introducing long-document selection bias. Validation windows are within documents, whereas original training windows can cross document boundaries. Report source losses separately and use predeclared corpus token weights for the aggregate. The historical continuation validation range has approximately 4.98% reconstructed positional overlap and is not substituted as clean validation.

FineWeb provenance points to revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9, sample-10BT. A compatible partial 1B manifest records ODC-BY; this is not a fresh license verification of the full corpus. Wikipedia provenance points to revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa, 20231101.en. Complete Wikipedia licensing evidence was not present in the inspected canonical manifests. Full raw sources are not all present; packing, indexes and recorded provenance remain available. Prior English heuristics and round-trip checks on 256 documents support compatibility, not corpus-wide language or quality guarantees.

[Selection manifest](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/selection-manifest.json) identity: `bd69d23b5749fd21a95826b0d93a2f02040ff3d8dfaea7f27ff6f71cf9c27444`. Arrays are checksummed and deterministically reconstructible; manifest identity includes namespace-specific paths.

## D. Proposed foundational configuration

[Proposed YAML](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/foundation-pilot.proposed.yaml) and [complete specification](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/foundation-pilot.proposed.json) are explicitly **PROPOSED_NOT_EXECUTABLE**. There is no training launch action.

Provisional initializer: canonical `checkpoints/codexa-900m-base-v1/latest.pt`, SHA256 `808afec9923e3f00d91bcbef47643265ae5dbe6c5aeb4bd112c4e97be3d0a3a7`. Base weights are the most defensible control for continued next-token training because they have not undergone an objective change. This is not a selected conversational initializer or evidence that adapted models are intrinsically worse.

Use a new phase with fresh AdamW8bit optimizer and scheduler, not exact continuation of the completed schedule. Proposed betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1, clipping 1; LR peak 2e-5, 20-update warmup, cosine decay to 2e-6. Context 2048, microbatch 1, accumulation 32, seed 42, BF16 autocast, FP32 weights and activation checkpointing remain unchanged. Train all valid shifted next-token targets with token-normalized loss; no assistant-only masking or END weighting.

Tokenizer SHA256 remains `a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e`. Architecture remains 934,356,480 parameters, 24 layers, width 1536, 24 heads, FFN 6144, vocabulary 16,384, learned positions, RMSNorm, SwiGLU, SDPA and tied embeddings. Configuration identity is recorded in the proposal.

Shuffle only eligible window positions for a finite remaining-position traversal. This avoids exact historical target-window reuse without claiming semantically unseen text. Do not introduce a source curriculum or LR sweep into the first experiment; any later sensitivity test changes LR alone under matched evaluation.

## E. Controlled pilot specification

Ceilings: 256 updates, 16,777,216 targets, one finite epoch and 3,600 startup-inclusive seconds. At historical 8,003 targets/s the pure training estimate is 2,096.37 seconds (34.94 minutes); evaluation, loading and saving consume the remaining allowance. Deadline final-save reserve is at least 180 seconds and must increase if measured save requirements demand it.

Exact proposed first 8,192 shuffled windows are unique. FineWeb exposure is 10,269,965 targets (61.2138%); Wikipedia 6,507,251 (38.7862%). [Pilot exposure](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/pilot-source-exposure.json) records their identity, including order SHA256 `999fa8a4b772b07851ddd39cff11a5df330b2b3650a452d3e3f672a83437f7ff`.

Evaluate source-stratified loss before updates and every 64 updates; preserve the no-update baseline. Neutral/direct generation uses the same 24 frozen concepts in two formats (48 cases), before training, at update 128 and at final update 256. Keep greedy decoding, 128-token maximum and repetition penalty 1. Pair formats within concepts; use 2,000 concept-cluster bootstrap samples, seed 42. These comparisons require genuine semantic review.

Predeclared progression requires at least three net improved concepts out of 24 with a positive 95% paired lower confidence bound, no reviewed semantic regression and no severe-degeneration increase above five percentage points. A CE decrease of 0.02 is an early learning signal only. Stop on nonfinite values or failed checkpoint publication; investigate gradient norm above 10 for three consecutive observations and source CE deterioration above 0.1 for two evaluations. A flat/ambiguous result requires the neutral diagnostic or a separately approved matched LR test, not automatic ten-hour training.

Legacy six-state recovery stays default. Save every 64 updates and at final, preserve latest/previous and distinct selected best states. Using measured native payload 5,636,391,032 bytes, six states plus one staging payload and 1 GiB logs require approximately 37.75 GiB peak. Preserve the existing 180 GiB startup floor and applicable protected-headroom/working-reserve checks. Resume only this new phase's verified optimizer-boundary checkpoint, restoring optimizer, scheduler, RNG, sampler cursor, counters and cumulative elapsed budget; base initialization is weights-only.

This is a validated proposal, not a runnable production pipeline. Existing legacy training CLI consumes contiguous token ranges; wiring the indexed view and its identities into launch/recovery still needs implementation and focused checks. The proposal's guards are predeclared requirements, not claims that an active new controller was deployed.

## F. Ten-hour estimate

Historical median late throughput was 8,002.73 context targets/s, rounded to 8,003 for planning. These are projections, not current performance measurements.

| Non-training overhead | Processed targets | Complete updates (65,536 targets) |
|---|---:|---:|
| 0% | 288,108,000 | 4,396 |
| 10% | 259,297,200 | 3,956 |
| 20% | 230,486,400 | 3,516 |

The zero-overhead projection covers approximately 2.368% of remaining packed positions and raises cumulative exposure to 943,468,000 targets, approximately 1.010 targets/parameter. Ten hours would be one bounded continuation phase, not full pretraining on the 12.82B corpus. Evaluation cost is unmeasured here; prior native checkpoint saves were roughly 14–20 seconds, and frequency must be set using fresh measurements. Legacy rotation bounds physical state count; a failed save must preserve the prior verified state. See [budget calculations](../../outputs/readiness/codexa-934m-foundational-pretraining-v1/prepared/budget-estimates.json).

Useful improvement is plausible given the still-improving historical curve; its magnitude and conversational transfer cannot be promised. A ten-hour commitment needs independent behavioral evidence and authorization after the pilot.

## G. Transition to conversational SFT

First establish accessible factual/comprehension and grammatical capability in neutral/direct framing. If neutral capability is useful but chat compliance weak, corrective SFT is preferable to assuming more general pretraining is needed. If weakness persists across formats and reviewed outputs corroborate it, test the bounded foundational pilot. Only progress after paired neutral-language gains and stable source validation.

Then use existing UltraChat/OASST1 and only genuinely approved curated records for a finite source-aware SFT comparison. Preserve short-answer formats, multi-turn context, assistant-only labels and natural END supervision. Compare against the matched existing-mixture control; do not select END weight 3 from stopping alone. The 576 curated records cannot substitute for foundational knowledge and their 156 pending reviews cannot be bypassed. Conversational acceptance requires instruction/factual gains, controlled repetition and reviewed coherence/retention, not loss or raw END alone.

## H. Blockers and approvals

1. Neutral-format evidence and human semantic review are still missing. The existing frozen, approved foundation diagnostic is the next decision experiment; this assignment explicitly excluded running it.
2. Capacity remains an unchanged operational blocker: last observed approximately 169.101 GiB versus 180 GiB minimum. This is historical, not a fresh measurement; no storage investigation or changes were made here.
3. Before any pilot, recheck at least 16 GiB available host RAM, disk reserve and VRAM. Proposed headroom is 13,928 MiB free VRAM: historical full-shape reserved peak 12,392 MiB plus provisional 1,536 MiB margin. This margin is not freshly qualified; the earlier allocator warning requires a bounded full-shape check. Do not reduce gates to run.
4. Complete indexed-view launch/recovery integration and obtain exact job-specific pilot authorization. No training approval was created.

Prior full-model AdamW8bit CUDA recovery passed bitwise checks, but optimized restore regressed 16.886%; legacy remains default. This task neither reran GPU qualification nor made optimized adoption a prerequisite.

## I. Implementation, tests and preservation

Ownership:
- LLM-Data: `src/llm_data/token_views.py`, `tests/test_token_views.py` — indexed immutable-window view, causal input/target interval exclusion and validation.
- Integration: `src/codexa_workspace/pretraining_readiness.py`, `scripts/pretraining_readiness.py`, `tests/test_pretraining_readiness.py` — offline evidence, bounded selection, budget/proposal validation, review queue and CPU continuation checks.
- Integration documentation: this report and dated session/progress entries. No Training, inference, architecture or checkpoint-format rewrite.

LLM-Data 0.1.0 wheel was rebuilt/reinstalled with no dependency upgrades; installed new implementation matches source. `pip check` passed. The wheel digest and import identities are recorded in the isolated namespace.

| Command / check | Result |
|---|---|
| `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -rs ../LLM-Data/tests` | 37 passed, no skips |
| `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m pytest -q -rs tests/test_pretraining_readiness.py tests/test_token_data.py tests/test_training.py tests/test_foundation_diagnosis.py tests/test_qualification_review.py` | 53 passed, no skips; 3 existing return-value warnings |
| Initial focused run | 1 failed / 26 passed: incorrect warmup test expectation; corrected test, not LR mathematics |
| CPU legacy optimizer-boundary continuation | Bitwise model/optimizer equality after subsequent updates; RNG/scheduler/cursor restored |
| Selection/proposal verification | Checksums, deterministic ordering, historical/held-out exclusion and disabled execution checked |

CPU continuation uses standard CPU AdamW and tiny fixtures; it is not a new AdamW8bit CUDA qualification. Unchanged full architecture/inference/integration suites were not rerun. Failed-run evidence remains in `focused-first-tests.log`.

Core preparation took 96.17 seconds; subsequent exact-source/review analysis took 64.37 seconds. Persistent artifacts are approximately 25 MiB, below the 64 MiB local ceiling; final exact bytes appear in the handoff verification. No full checkpoint copies, GPU inference or training occurred. Canonical inputs and frozen diagnostic fingerprints were checked unchanged. The existing 1,059-file protected metadata comparison found zero changes; metadata equality is not a new full-payload checksum. No checkpoints/datasets were deleted, moved, compressed or retokenized. NumPy and existing approval records remain untouched.

## J. Final recommendation

**EXISTING EVIDENCE IS INSUFFICIENT TO CHOOSE — NEUTRAL-FORMAT EVALUATION REQUIRED.**

Undertraining is a credible hypothesis, supported by limited exposure and continuing historical loss improvement, but all current free-running evidence is chat-framed. The concrete next experiment is the existing approved 600-generation neutral-versus-chat diagnostic, unchanged, after capacity and fresh resource/fingerprint gates pass in a separately scoped execution assignment. Complete genuine semantic review of its matched factual/comprehension outputs before deciding between corrective SFT and the proposed foundational pilot. No experiment was automatically launched.
