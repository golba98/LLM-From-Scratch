# CODEXA 934M CONVERSATIONAL REGRESSION INVESTIGATION

Investigation performed on 10 October 2026 under an explicitly authorized four-hour ceiling.

## Findings

No evidence of incorrect role masking, duplicate causal shifting, conversation cycling, or tokenizer replacement was found. Existing serializer supervises assistant content and END, the dataset shifts labels once, and the model compares logits to those shifted targets without shifting again. Microbatch 1 uses complete unpadded conversations.

The completed two-hour experiment contained 90,795 assistant turns with only 334 repeated full supervised prefixes. Unique conversation identities do not imply entirely unique context; this audit explicitly accounts for shared prefixes.

UltraChat supplied 24,094,389 of 25,912,988 assistant tokens (92.98%). OASST1 supplied 1,818,599. END targets were only 0.350% of supervised tokens. Low prevalence alone does not prove inadequate termination learning.

## Source-stratified teacher-forcing diagnostic

Exactly the same 24 validation conversations per source were used for all three checkpoints. Assistant loss includes END. These records are disjoint from current training but may have historical exposure; this is a controlled diagnostic, not proof of generalization.

| Checkpoint | Source | Assistant loss | END loss | Premature END probability |
|---|---|---:|---:|---:|
| repair | oasst1 | 2.53779 | 1.44426 | 0.00470 |
| repair | ultrachat | 2.37014 | 0.94558 | 0.00354 |
| balanced | oasst1 | 1.57441 | 0.75263 | 0.00304 |
| balanced | ultrachat | 1.89786 | 0.52328 | 0.00143 |
| final-2h | oasst1 | 2.44063 | 1.89861 | 0.00295 |
| final-2h | ultrachat | 2.05361 | 0.77378 | 0.00125 |

Repair-to-final adaptation improved assistant loss for both sources, but OASST END loss worsened 1.44426→1.89861 while UltraChat END loss improved 0.94558→0.77378. This supports investigating source weighting rather than assuming a global END-mask bug.

## Free-running generation and paired changes

On the same 12 local source-stratified conversation prompts, the repair initializer generated END in 50% and severe degeneration in 41.67%; the two-hour final generated END in 25% and degeneration in 66.67%. The balanced historical initializer generated END in 25% and degeneration in 33.33%. These are very small, multilingual diagnostic samples, not a conversational accuracy benchmark. Dummy local task checks are deliberately ignored.

Across 64 development cases: {'task_improved': 1, 'task_regressed': 3, 'degeneration_new': 12, 'degeneration_fixed': 9, 'lost_END': 11, 'gained_END': 7}. Literal task macro fell 14.29→10.71% through one newly passing case and three newly failing cases. This small change does not establish a statistically reliable general capability regression. Severe repetition remains common regardless of that uncertainty.

Reference-term scoring can count incoherent answers as correct: the initializer repeatedly mentioned Earth in a malformed answer, while the final said only “Yes, humans live on.” The literal factual failure is real, but neither output is an acceptable answer. No broad factual-reliability claim follows from keyword scores.

## Matched bounded ablations

Four trials crossed repair/balanced historical initialization with proportional/balanced source sampling. All used new weights initialization, seed 73, LR 5e-6, microbatch 1, accumulation 8, token-normalized loss, complete-conversation dynamic lengths, max 512 updates, warmup 20, and a 1,200-second session cap with 180 seconds reserved for recovery. The visible 100×22 Kitty viewer and native recovery store were used. The same development suite was scored at 256 and 512 updates when reached. No protected acceptance suite was opened.

Equal optimizer steps/conversation count is an operational comparison, not equal assistant-token exposure: sources have different lengths. One seed and short schedules cannot establish an optimal mixture.

| Trial | Updates | Assistant tokens | OASST examples | UltraChat examples | Final validation loss | Quality step | Task macro | END | Degeneration | Stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| repair-proportional | 512 | 3,348,416 | 578 | 3518 | 2.077632685812196 | 512 | 14.29% | 57.81% | 45.31% | STEP_CAP |
| repair-balanced | 512 | 2,563,840 | 2048 | 2048 | 2.096029065947892 | 512 | 12.50% | 54.69% | 43.75% | STEP_CAP |
| balanced-proportional | 512 | 3,348,416 | 578 | 3518 | 1.820172562955878 | 512 | 3.57% | 75.00% | 20.31% | STEP_CAP |
| balanced-balanced | 512 | 2,563,840 | 2048 | 2048 | 1.826198898990232 | 512 | 3.57% | 78.12% | 20.31% | STEP_CAP |

### Additional END-weight contrast

This fifth pilot held repair initialization, proportional sampling, seed 73, LR 5e-6, accumulation 8 and 512-step schedule fixed. END target 7 received weight 3; other targets remained weight 1. The objective divides the weighted CE sum by actual assistant-token count. Padding and masked roles remain excluded, and evaluation loss remains unweighted. It is a new experimental phase, not a silent objective change in any existing run.

Actual updates 512; final recorded development task macro 14.29%, END 75.00%, degeneration 29.69%.

Raw weighted training loss is not directly comparable with unweighted training loss. The actual token counters are unchanged by weight 3; a CPU engine update matched an independent reference gradient and token count.

### Matched checkpoint END and retention diagnostics

| Trial | OASST END loss | UltraChat END loss | Base-training replay loss |
|---|---:|---:|---:|
| repair-proportional | 2.19547 | 1.02366 | 3.20919 |
| repair-balanced | 2.14335 | 1.17624 | 3.21942 |
| balanced-proportional | 0.79998 | 0.57591 | 3.18631 |
| balanced-balanced | 0.85510 | 0.59275 | 3.18951 |
| repair-proportional-END3 | 1.44856 | 0.46527 | 3.20942 |

Replay uses the same eight 512-target windows as the earlier retention diagnostic; these are base-training samples, not replacement validation. Historical repair replay loss was 3.273845 and balanced 3.185360. Window SHA and offsets are retained. No claim of broad retained-capability generalization is made.

Entry-gate passing trials: none. The entry floor is a low diagnostic threshold, not promotion approval.

## Cache investigation

- bf16: 27 identical-prefix comparisons; 1 argmax differences; max absolute logit difference 0.156250. First differences: [{'position': 26, 'prefix_length': 41, 'maximum_absolute_logit_difference': 0.125, 'same_argmax': False, 'cached_top_ids': [600, 7], 'direct_top_ids': [600, 7], 'cached_margin': 0.0, 'direct_margin': 0.125, 'cached_END_probability': 0.18729113042354584, 'direct_END_probability': 0.16886024177074432}].
- fp32: 27 identical-prefix comparisons; 0 argmax differences; max absolute logit difference 0.000091. First differences: [].

The initial cached/uncached greedy test differed on one of four repair prompts after 26 shared tokens. Identical-prefix measurements distinguish numerical sensitivity from an offset/masking defect; no architecture or cache implementation was changed.

## Data and evaluation limitations

A full training-file literal phrase scan found only 4 “only the word”,5 “only the number”, and 8 “just the number” occurrences; direct inspection showed the “only the word” hits mostly describe program output rather than constrain the assistant reply. This is a narrow phrase audit, not a semantic inventory. General chat imitation supplies little explicit practice for this exact evaluation style. Existing repair data has 71 training anchors among 9,627 records, with repeated questions/answers and arithmetic targets containing explanations; this does not teach number-only responses reliably.

Historical repair-anchor audit:71 train anchors contain only 26 unique conversations;25 validation anchors contain 19 unique conversations, all 19 also appear in training. Arithmetic training anchors have totals 42 (seven), 13 (two), 17 (two). Seven final arithmetic responses with numerical answers use only those three totals. This is consistent with narrow template memorization; it is not proof that the exact historical training exposure alone caused it. Existing General Chat split isolation is a separate issue.

No new training examples, datasets, or tokenizer changes were introduced. No END-weighted objective was added without ablation evidence. All historical assets and the two-hour final checkpoint remain preserved.

## Reproducibility and verification

Exact protocol inspection passed on 48 real validation conversations: every inference prefix matches training, every assistant content/END target is shifted once, and role/user/system targets are ignored. Focused controller/paired-diagnostic tests passed. Full final test/environment results are recorded in investigation automated-checks.json when available.

Artifacts: `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010`. Current elapsed investigation time at report generation: 5306.1 seconds of the 14,400-second ceiling. Stage start/end records, dataset selections, paired raw outputs, source counts, checkpoint digests, full pilot specifications and native recovery indices are retained.

Commands from integration root:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/investigate_chat_regression.py --stage diagnostics
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/investigate_chat_regression.py --stage pilots
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/investigate_chat_regression.py --stage cache
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/investigate_chat_regression.py --stage report
```

Existing pilot result files are reused; completed pilots are never silently trained longer. Diagnostics create outputs and should be run only in an isolated investigation directory or with intentional review of existing artifacts.

## Decision

No model is automatically promoted. The completed two-hour final remains technically recoverable, but a lower loss is insufficient reason to continue its exact schedule. Use the ablation evidence to choose the next bounded phase, require improvement on both task behavior and degeneration, and retain the protected final suite for a genuinely final decision. If no trial passes, continue diagnosis/curation rather than spending the remaining ceiling on blind repetition.


## Historical exposure and role coverage

Repair train/validation share 1,258 natural roots and 24 normalized complete-conversation contents. Of the current General Chat validation, 926 records share roots with repair training. None of the 96 possibly-clean validation records match repair training by root or normalized complete content; other historical phases remain unproven.

The current training pool contains 517,729 assistant messages and zero system messages. Only 2,169 answers are 32 characters or shorter. This is descriptive coverage evidence, not a semantic classification of task quality.

## Implemented corrections and ownership

- LLM-Data: new data/repair_split.py and revised cli/prepare_repair_sft.py group roots, normalized whole-conversation duplicates and preexisting duplicate components. Stable group hashing makes assignment independent of input order. The existing nonempty-output refusal remains. Only synthetic unit fixtures were split; no historical corpus was regenerated.
- LLM-Training: optional validated quality_guard.py thresholds and patience; token_objective.py target weighting; training.py propagates the optional weighted objective and keeps actual token counts; chat_run.py binds these choices to the saved specification. Defaults preserve existing behavior. New guard policy is proposed for future phases, not retroactively applied to these trials.
- LLM-From-Scratch: investigation driver, script and paired tests; optional behavioral-policy configuration flows through make_spec; proposed configs/chat-behavior-guard-v 1.yaml. All relevant wheels rebuilt and installed locally without dependency upgrades. Model/tokenizer/inference architecture remained unchanged.

The stricter proposed guard would have flagged two consecutive deteriorating checks by step 1600 in the original two-hour run. This is an offline counterfactual, not a new-policy training result. No selected checkpoint or export was promoted.

## Test and environment record

- `integration-tests`: 130 passed, 5 skipped (scikit-learn unavailable), 8 warnings.
- `final-tests`: 65 passed, 2 warnings.
- `specialist-skips`: 34 passed, 5 skipped: scikit-learn unavailable.
- `Compare all 4096 control/END3 sampled identities`: PASS: byte-identical sampling.
- `Five final checkpoint SHA256 checks; END3 actual native CPU reload`: PASS; no fresh continuation update performed in this final verification.
- `git diff --check: LLM-From-Scratch`: PASS.
- `git diff --check: LLM-Training`: PASS.
- `git diff --check: LLM-Data`: PASS.
- `pip check`: No broken requirements found..

Five integration skips are optional scikit-learn specialist tests in the generative profile. Existing pytest return-value warnings are recorded. Component and integration test counts overlap and must not be summed as unique coverage.


## Matched END3 verification and representative failures

All 4,096 sampled conversation identities and their ordering were byte-identical to the unweighted control. Paired outcomes: one task improved and one regressed; 12 degeneration cases fixed and two newly degenerated; 12 END stops gained and one lost. This is a single-seed development result, not a protected acceptance result.

END teacher-forced losses fell from 2.19547 to 1.44856 on OASST and 1.02366 to 0.46527 on UltraChat. Combined assistant losses were almost unchanged (2.48586→2.48883 and 2.18806→2.18995). The fixed eight-window base-training replay loss was likewise similar: 3.209186→3.209423. These diagnostics support a targeted stopping effect, without demonstrating factual or instruction improvement.

Representative outputs below use the first case in each stated category; responses are excerpts, and complete outputs remain in paired-END3-control.json.

**Improved repetition — development-007**

Prompt: Why does ice float on liquid water? Keep the explanation brief.

Control: The ice is a liquid that is not liquid water. It is not liquid water. It is not liquid water. It is not liquid water. It is not liquid water.

END3: The ice is a liquid that is not liquid water. It is not liquid water. It is not liquid water.

**New repetition failure — development-026**

Prompt: How many sides does a triangle have?

Control: There are seven sides in a triangle, and each side has a unique number of sides.

END3: The triangle has seven sides:

1. A triangle is a triangle that is not triangular.
2. A triangle is a triangle that is not triangular.
3. A triangle is a triangle that is not triangular.
4. A triangle is a triangle that is not triangular.
5. A triangle is a triangle that is not triangular.
6. A triangle is a triangle t

**Unchanged instruction failure — development-000**

Prompt: Reply with only the word READY.

Control: READY is a word that is used to describe a type of language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is no

END3: READY is a word that is used to describe a type of language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is not spoken by humans. READY is a language that is no

**Task regression — development-042**

Prompt: Which ocean is the largest?

Control: The largest ocean in the world is the Pacific Ocean.

END3: The largest ocean on Earth is the Arctic Ocean.

## Final investigation disposition

Investigation complete; production continuation remains BLOCKED on conversational acceptance. No pilot passed the existing entry floor; instruction following and arithmetic remained at zero. END3 is a candidate for a second-seed, independent-window bounded validation, not a promoted model. Further useful work is instruction-data coverage review and provenance-safe evaluation, rather than repeating the original adaptation unchanged.

Five final checkpoint digests were recomputed; the END3 final was loaded successfully as a complete native payload with optimizer, scheduler, RNG, counters and resume context. No additional continuation update was performed during this final verification, so this does not claim a new exact-resume equivalence test for END3.

Installed source-byte comparisons matched Architecture, Tokenizer, Data, Training, Inference, Memory and integration. Specialist is intentionally absent from the generative profile. There were no dependency upgrades.

Free disk at handoff was 106.2 GiB. The existing production preflight requires 100 GiB safe headroom plus 80 GiB working reserve; a new production launch under that policy will block until its storage budget is reviewed. All historical assets and five pilot recovery stores remain retained; no cleanup was performed.

No training remains active. The original two-hour budget is expired and must not be reset implicitly. This investigation consumed substantially less than the authorized four-hour ceiling; the ceiling was not a requirement to continue training.
