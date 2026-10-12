# CODEXA 934M — Foundational Capability Diagnosis & Final Training Strategy

11 October 2026. Evidence namespace: `outputs/diagnosis/codexa-934m-foundation-20261011-v1/`.

## A. Foundational training diagnosis

**Undertraining is plausible; foundational viability remains unproven.** The selected base received 655,360,000 next-token targets for 934,356,480 parameters: 0.7014 tokens/parameter and 5.1121% of the declared corpus. Exposure alone does not establish the cause of chat failure. The logs show substantial learning and continued improvement, not a demonstrated absence of language representations.

| Original run window | Mean training CE | Validation CE at boundary | Learning rate |
|---|---:|---:|---:|
| 1–250 | 7.8415 | 7.0169 | 0.0000249 |
| 1751–2000 | 4.4601 | 4.6295 | 0.0001999 |
| 4751–5000 | 3.3487 | 3.6689 | 0.0001445 |
| 8751–9000 | 3.0446 | 3.3794 | 0.0000269 |
| 9751–10000 | 3.0218 | 3.3491 | 0.0000200 |

The original schedule warmed up for 2,000 of 10,000 updates, peaked at 2e-4 and decayed to 2e-5. Full 2048-token windows and accumulation 32 were used throughout. Final mean gradient norm 0.4113 is finite; logs do not show a late train/validation divergence. Late validation improvement was smaller (3.3794→3.3491 over the last 1,000 steps), but that does not prove convergence or justify blindly extending the old schedule.

Original validation was FineWeb-only, limited to four microbatch 1 windows—8,192 target tokens per evaluation. Its binary and index are missing today. The population differs from the mixed training stream; their loss gap is not a direct overfitting estimate. Final historical perplexity 28.477 measures this small teacher-forced population, not factual or instruction capability.

Corpus: 12,819,847,234 tokens; 7,832,893,798 FineWeb-Edu (61.10%) and 4,986,953,436 Wikipedia (38.90%); 13,383,283 documents. Manifest zero repeated documents means no mixing oversampling, not proof of semantic deduplication. Configured provenance is FineWeb-Edu `sample-10BT` and Wikipedia 20231101.en; near-duplicate clustering is explicitly incomplete. The complete original raw source stores are not present in the canonical data/raw inventory; packed training data remains available. The preserved FineWeb 1B processed subset is not the full 7M-document source.

Historical Git commit `66eec752688afad348206f22db4c90f8f2a47eda` confirms shuffled no-replacement loading, epoch reseeding and saved-cursor skipping. CPU-mapped latest/previous checkpoint metadata confirm cursors 320,000/312,000 and the same run identity. Same-version Torch reconstruction gives 320,000 distinct window locations, consistent with 655.36M distinct target positions during the partial epoch. This is reconstruction, not recorded per-batch proof: the earlier 500-step resume state is not available, and distinct locations do not exclude repeated content.

Do not conflate the separate 17-step restart or13-step continuation with the selected base. The later continuation validation range overlaps 956 reconstructed base windows, or1,957,888 target tokens (4.9821% of that range). It is not a demonstrably unseen substitute for missing original validation. Evidence: `training-history.json`, `historical-sampler-evidence.json`, `sampler-reconstruction.json`, `continuation-range-exposure.json`.

| Bounded corpus sample | Documents | Tokens inspected | Bytes/token | Tokens/word | English-like heuristic |
|---|---:|---:|---:|---:|---:|
| FineWeb-Edu |128|148,530|4.306|1.475|128/128|
| Wikipedia |128|118,592|4.040|1.550|126/128|

The sample reads 68,321 bytes of the 3.1 GiB index and 267,122 token IDs from read-only mappings. Two long documents are truncated at16,384 tokens. All 256 sampled ID sequences round-trip exactly; zero replacement characters and no duplicate sample texts were found. Two Wikipedia samples contain repeated nonempty lines above 10%. These are bounded screens, not corpus-wide quality/language measurements. English-like uses function words; two short/wiki fragments remain unknown. No tokenizer defect or replacement is justified.

Crucially, this sample contains 254 EOS tokens and **zero chat role/END IDs 4–7**. That supports a format-distribution explanation for the base’s chat failures; it does not prove those tokens never occur anywhere in the corpus. Base chat END failure and 98.75% degeneration cannot by themselves diagnose ordinary-completion ability.

## B. Neutral completion versus chat comparison

**No new full-model generation ran.** Resource policy and precise approval block the prepared experiment. Do not interpret CPU fixture outputs as934M results.

| Checkpoint | New neutral/direct diagnostic | Existing strict chat result | Reviewed semantic capability |
|---|---|---|---|
| base | Not executed |0/216 objective cases|Unknown|
| repair | Not executed |0/216 objective cases|Unknown|
| balanced | Not executed |0/216 objective cases|Unknown|
| original_2h | Not executed |0/216 objective cases|Unknown|
| END3 | Not executed |0/216 objective cases|Unknown|

Existing raw evidence shows heterogeneous failure: base repeats “the Sun is not the same as the Sun”; balanced names Jupiter correctly in one answer but continues repetitively; repair describes Jupiter as a “large-billion-dollar asteroid.” These examples demonstrate why neither exact-format failure nor keyword presence proves the extent of knowledge. They are existing outputs, not new matched-format results. All 1,200 remain unchanged.

The real tokenizer was checked against training serialization for 72 chat-form cases: every inference prefix matches the complete conversation’s prefix through the open assistant marker, and the reference assistant target ends with END. No BOS is injected. This finds no mismatch between the inspected inference and SFT serialization. Likelihood probes preserve joint tokenization of the conditioning prefix: plain completions add a separating space; chat references follow assistant ID directly.

A confirmed defensive gap exists in the shared greedy decoder: argmax itself does not reject NaN/Inf. The diagnostic installs a temporary finite-logit observation hook on every decode forward, without altering logits/cache mathematics. There is no evidence that historical failures were caused by NaNs. Cache versus noncache greedy IDs match in a small CPU fixture; this is not full-model CUDA qualification.

## C. Human review status

**Human approvals: 0. Curated records pending:156. Development definitions pending: 240. Semantic model review pending.** Machine validation of576 curated records remains distinct from human approval. No admission or reviewer identity was invented.

[Prioritized workflow](../../outputs/diagnosis/codexa-934m-foundation-20261011-v1/REVIEW_WORKFLOW.md) and [fresh anonymous baseline review](../../outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline/review.html). The fresh packet reuses the previous 480 semantic responses and 60 instruction examples across 12 families; it generates no new model responses. The original alias mapping appeared during inspection, so old blindness cannot be guaranteed if seen. New private mapping was not printed and is absent from the public page.

Start with 20 matched factual responses,10 multi-turn,10 retention and 60 instruction cases. Then complete the remaining packet before importing complete model-review inputs. Enter a real name, decide each response, rate coherence separately, comment, and export JSON before closing. No automatic save or approval. Representative screening is not model or dataset admission.

The original dataset and development review pages remain unchanged under `outputs/qualification/codexa-934m-20261011-v1/review/`. Protected acceptance examples remain untouched; only their existing byte fingerprint was checked.

## D. Training strategy comparison

| Path | Expected value | Data/compute evidence | Risks and prerequisites |
|---|---|---|---|
| A — Continued foundational pretraining | Test missing general representations if plain-format capability is weak | Existing12.82B packed-token pool; roughly12.16B positions beyond reconstructed original exposure. Historical~8,003 context tokens/s projects~28.8M/hour before overhead | Missing original validation; semantic duplicates and prior range exposure; new schedule is a new phase, not exact completed-schedule resume. Behavior gains unproven |
| B — Corrective SFT | Teach constraints, concise answers and chat behavior when knowledge is accessible | Existing OASST/UltraChat plus576 curated proposals. Improved mixture limited to1.978M supervised tokens by two-pass curated budget |156 human decisions pending; all previous entry gates failed; narrow templates cannot replace missing knowledge |
| C — Bounded pretraining then SFT | Separately measure foundations and instruction access | Requires two approved phases and matched controls, more compute than either alone | Confounds changes unless phases measured separately; not justified before the format diagnostic |

No training path is selected yet. A has greater foundational coverage, B targets behavior directly, and C may combine them—but current evidence cannot rank their expected behavioral gains reliably. Historical throughput is a projection input, not a new benchmark. Retain AdamW 8bit/BF16/FP32 master values and legacy recovery for any later approved phase; do not silently reset optimizer/scheduler or call a changed schedule exact resume.

## E. GPU and storage readiness

Fresh snapshot: **169.104 GiB free**, **10.896 GiB below the 180 GiB gate**; host available 16.879 GiB (16 GiB gate passes); swap used 7.710 GiB; GPU free 12,260 MiB. CUDA is visible (one device), Torch 2.13.0+cu130, Python 3.14.7, bitsandbytes 0.50.0. Preflight observes resources without terminating processes or clearing caches.

Encrypted Btrfs zstd:1 reports device allocated 903,042,039,808 bytes, used 836,657,774,592 bytes, statfs free 181,573,615,616 bytes at its separate snapshot. Metadata is DUP; detailed per-device chunks require privileges, which were not requested. These are filesystem totals, not new artifact allocation.

The new inference job writes no checkpoints and is capped at128 MiB; it physically fits, but the mandatory 180 GiB policy still blocks it. To cover the whole artifact allowance without crossing the gate between candidates, target at least 180.125 GiB plus metadata/fluctuation margin—about 11.021 GiB above this snapshot. No storage was reclaimed.

Unmounted 476.9 GiB SATA NTFS and 930.5 GiB NVMe NTFS partitions are on distinct devices from the active Linux NVMe. Their free capacity, ownership and suitability are unverified; neither is credited or mounted. A separately approved output volume must itself pass the 180 GiB gate. Additional dedicated capacity or independently verified archival are future options only; no deletion or movement is authorized here.

Previous qualification still provides exact CUDA/AdamW 8bit recovery, but optimized restore +16.886% fails its optional adoption gate. Legacy remains default and does not require this optimization to be fixed first. The 100,663,296-byte allocator OOM retry warning remains a concern: peak reserved minus allocated was~2.802 GiB, consistent with significant allocator reserve but not proof of fragmentation as the cause. The console does not identify the failing lifecycle phase.

For later training, the measured 11.859 GiB reserved footprint plus a practical 1.5 GiB margin is roughly 13,680 MiB—more than current 12,260 MiB free. This is a planning margin, not a qualified training configuration. The inference diagnostic excludes optimizer/gradient allocations; retain its existing stricter free-VRAM gate. Do not overcommit or silently change sequence length, precision or allocator policy.

## F. Recommended experiment — one only

**Run the neutral-versus-chat diagnostic when capacity and precise approval pass.** Configuration: `outputs/diagnosis/codexa-934m-foundation-20261011-v1/job-v2/plan.json`.

Job SHA 256 `3fbe4da47492667a63a8f5dcaba907058b126542a6f0318b48c34a918eb3348b`. Suite SHA 256 `ea56779e58705f4b0a865d87ae014d38c5b2ba5ba2b5bdcced1925580fb8c873`. Tokenizer SHA 256 `a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e`.

| Control / comparator | Exact checkpoint SHA256 |
|---|---|
| base | `808afec9923e3f00d91bcbef47643265ae5dbe6c5aeb4bd112c4e97be3d0a3a7` |
| repair | `c55f4edd1477a8ffefd03ce63d033a291eb261f6047b129baa47e10756899a4f` |
| balanced | `a4c8c172152dd8e3ef23a7dde05bc4a4355ff548f84fc904d8d5534dd2183892` |
| original_2h | `374f3f1746b0ff0b24febc5176f3d20f5bb4ccadc3fb6aa24a120d1f31481918` |
| END3 | `4f4e5d1540eb6813ad6d6f91f1a1a503bf8952632d51f9dc6410343a8e0c456b` |

Base is a control, not a promoted conversational initializer. Use all five sequentially, one per owned process.24 concepts (six each facts, arithmetic, grammar and comprehension) ×five forms ×five candidates =600 generations. Forms: neutral prose, plain Question/Answer, serialized chat, constrained short answer, and multi-turn framing. Framing does not independently benchmark retention; existing reference-update reviews cover that gap. Wording and context presentation differ, so a format effect is not automatically a serializer bug.

Greedy 128 new tokens, penalty 1, seed 42, BF16/FP32, KV cache unchanged, CUBLAS_WORKSPACE_CONFIG=:4096:8 and prior inference deterministic-algorithm settingfalse. No sampling, optimizer, LR, updates or checkpoints.600 generations/ 76,800 generated-token ceiling;3,600 seconds aggregate including startup, hashing, likelihood probes and writing. Original deadline persists to all children; no reset/retry path. Maximum prompt 37 tokens fits the 2048 context.

New cases have no exact existing development-prompt or whole-conversation ledger matches. Common facts/template skills intentionally overlap semantically; do not label this a sealed historically independent benchmark. Six arithmetic answers are independently verified and distinct. The first preparation was superseded before execution because it repeated some numerical answers; it remains preserved and cannot pass current code-identity checks.

Acceptance interpretation is predeclared: format gain ≥6/24 with positive paired lower bound; corrective-SFT signal additionally requires ≥12/24 neutral/direct semantic correctness, including≥4/6 factual and≥4/6 comprehension. Weak base signal is≤6/24 and≤2/6 in those two domains across plain forms, corroborated by reviewed degeneration/incoherence. These are small diagnostic thresholds, not production eligibility or a proof of cause.

The reviewed comparison uses 2,000 concept-cluster bootstrap samples, seed 42 and 99.5% intervals for ten model/form comparisons. Complete real 600-response decisions are required; unknown semantics are not converted to failures. Teacher-forced likelihood and first-token probabilities remain auxiliary. Objective format and arithmetic grades do not automatically become semantic scores.

Stop on deadline, OOM, nonfinite values, identity drift, resource failure or artifact quota. Preserve completed raw rows; incomplete candidates cannot enter a full comparison. No recovered hypothetical capacity is credited. Existing previous-job approvals are invalid for this new job; `approval-request.json` explicitly has approved=false and a real `approval.json` is absent.

Expected compute: previous 1,200-case comparison generated 85,153 tokens in732.664 generation seconds. Scaling its~116 tokens/s to the new maximum 76,800 tokens gives~660s generation alone; allow loading, likelihood passes and hardware variability. A10–20 minute projection is plausible, not measured;3,600s remains the hard ceiling. No GPU run started.

After resources pass and the exact approval artifact exists, the conditional entry point is:

```bash
cd "/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/conversational_readiness.py foundation-run \
  --output outputs/diagnosis/codexa-934m-foundation-20261011-v1/job-v2 \
  --approval-file outputs/diagnosis/codexa-934m-foundation-20261011-v1/job-v2/approval.json
```

This is intentionally not runnable now: the grant is absent and disk fails. It does not launch training. Future genuine exported diagnostic decisions are summarized with `foundation-report --output <job-directory> --review <real-export.json>`; no human input is generated by that action.

## G. Ten-hour training decision

**INSUFFICIENT EVIDENCE — TARGETED DIAGNOSTIC REQUIRED**

Do not invest ten hours in immediate SFT or claim additional pretraining is conclusively required. First determine whether accessible language knowledge changes across prompt formats. Strong plain-format knowledge with weak chat favors a bounded corrective-SFT investigation; weak reviewed capability across forms favors a bounded foundational investigation. Neither outcome automatically authorizes long training, overrides original entry gates or promotes a checkpoint.

## H. Implementation, tests and preservation

Integration-owned changes: new `src/codexa_workspace/foundation_diagnosis.py`; readiness CLI actions foundation-build/run/candidate/report; unknown/screening human-review packages now explicitly refused for admission/model selection in `qualification_review.py`; new `tests/test_foundation_diagnosis.py`. Serialization and architecture are unchanged. No sibling component source, wheel or environment was modified. Direct scripts insert the integration src path; tests use its established PYTHONPATH, avoiding stale installed integration imports.

Final command: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=src .venv/bin/python -m pytest -q -rs` — **184 passed, 5 skipped, 8 existing warnings**, 8.59 seconds. Five specialist skips require absent optional sklearn; no dependency installed. Earlier 183-pass run is preserved, not added to the final count. No regression test failed. The final document-link check caught two spacing errors in relative URLs; corrected before handoff and rechecked (handoff-first-check.json). Historical script lookup initially used a relocated path; the Git tree supplied the correct original scripts/train.py.

Coverage includes 120 matched prompts, no injected BOS, real serializer-prefix parity 72/72, finite-logit rejection on every decode, cache/uncached CPU greedy agreement, likelihood finiteness, case/format scoring, invalid arithmetic rejection, approval/code/suite drift, gate refusal, quota exhaustion, expired/original child deadlines, failed-child evidence, all 600 review-decision pairing, output tampering and screening-admission refusal. Fixtures are small and isolated; no real GPU workload or cleanup.

Artifacts were about 1.4 MiB before the extra reblinded review, still far below 128 MiB. Final protected metadata check:1,059 files unchanged (device/inode/size/mtime/ctime; not a new full-file hash audit). Previous comparison hash, historical proposals, curated dataset and protected acceptance fingerprint are unchanged. Historical checkpoints, datasets, exports, NumPy project and recovery stores were not deleted, migrated or altered. Model tensors were mapped read-only on CPU solely for cursor metadata; CUDA was not initialized by that inspection.

No reviewed initializer, new dataset admission, real-store reclamation, pilot or ten-hour launch was produced. The sole next experiment remains the approval-gated inference diagnostic above.
