# Codexa controlled cleanup and training readiness

10 October 2026. **CLEANUP COMPLETE — ADDITIONAL REQUIREMENTS REMAIN.**

## A. Cleanup summary

Removed **23 retired native checkpoint payloads** and their exclusively owned SHA/completeness records, plus seven retired recovery indexes: **76 files total**. All removals were inside the five user-authorized areas. No recursive project cleanup, real optimized-store reclamation, cache cleanup, dataset removal or training launch occurred.

The recorded filesystem free space increased from **97.981609 GiB** to **199.770954 GiB**, a net observed increase of **101.789345 GiB** (109,295,476,736 bytes). This includes the cost of two new lossless selected-model exports and compact validation/evidence files. A post-transaction `df` reading reached **200.147259 GiB**; later background allocations lowered the final recorded reading slightly. The 200 GiB stop target was observed, and no further checkpoint was removed to chase fluctuations around it. The unchanged **180 GiB** launch floor passes, with **19.770954 GiB** of surplus at the final recorded snapshot.

Repository policy required a verified selected-model export before retiring working checkpoints. Two unpromoted native exports were created under `outputs/preserved-models/20261010-controlled-01/exports/`, outside the compact evidence directory. Their selected weights are bitwise equal to the protected canonical repair/balanced initializers and the retired pilots' step-zero best states. Both native/exported 16-token CPU generation smoke tests produced identical token sequences. Exports consumed 6.963713 GiB logical storage; inode allocation was 7,477,256,192 bytes. Existing exports and inference defaults were unchanged.

## B. Deleted artifacts

Complete per-file manifest: [`deleted-artifacts.json`](../../../outputs/storage-cleanup/20261010-controlled-01/deleted-artifacts.json). It records every absolute former path, SHA256, device/inode, logical/allocation sizes, timestamps, checkpoint publication metadata and intentional nonrecoverability. The original manifest, inactive experiment/index evidence, durable intent/detachment/deletion journal and batch observations are in `outputs/storage-cleanup/20261010-controlled-01/`.

| Original store or checkpoint | Payloads | Removed logical bytes | Why safe to retire | Batch-end observed free-space increase |
|---|---:|---:|---|---:|
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/repair-proportional/checkpoints` | 4 | 19.227837 GiB | Retired unpromoted pilot; selected step-zero weights exported; no active dependency | 14.764416 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/repair-balanced/checkpoints` | 4 | 19.227836 GiB | Retired unpromoted pilot; selected step-zero weights exported; no active dependency | 18.143242 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/balanced-proportional/checkpoints` | 4 | 19.227837 GiB | Retired unpromoted pilot; selected step-zero weights exported; no active dependency | 20.190468 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/investigation/chat-regression-20261010/pilots/balanced-balanced/checkpoints` | 4 | 19.227837 GiB | Retired unpromoted pilot; selected step-zero weights exported; no active dependency | 18.650047 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-00/checkpoints` | 2 | 8.729835 GiB | Retired headroom pilot; verified selected repair export/initializer preserved | 8.142334 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-01/checkpoints` | 2 | 8.729835 GiB | Retired headroom pilot; verified selected repair export/initializer preserved | 8.666672 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/recovery/recoverable` | 2 | 10.498296 GiB | Failed nondeterministic fixture replaced by retained, verified deterministic recovery | 10.496464 GiB |
| `/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch/outputs/preparation/codexa-900m-chat-10h-v1/pilots/corrective-02/checkpoints/slot-1/checkpoint.pt` | 1 | 5.249298 GiB | Superseded step-96 slot; absent from current protected-role index | 0.000000 GiB |

The four non-END3 stores each contained steps 0, 256, 512 and 512. Corrective-00 and corrective-01 each contained steps 0 and 2. The failed nondeterministic recovery store contained steps 2 and 4. The final optional removal was only corrective-02 slot-1, step 96; its SHA was `47e8ab4779d66ac478eb8bb48a9797c0bda089dbb3ffd589f153b1aaf38fca3a`.

Batch measurements are temporal observations, **not exact per-file physical extent savings**. Btrfs delayed freeing explains the optional slot's immediate zero change and the later rise after all deletions. Total removed apparent bytes: **118,238,956,837** (110.118610 GiB). Inode-accounted allocation: **118,239,178,752** (110.118816 GiB). These must not be substituted for measured free-space gains. External activity and filesystem metadata also affect the net measurement.

For each retired whole store, the original recovery index was preserved as inactive historical evidence and removed; a checkpoint-directory retirement record and a run-level `CHECKPOINT_RETIREMENT-20261010.json` explicitly state the payloads are gone. Specifications, reports, evaluation outputs, logs, source/dataset provenance and negative findings remain. Historic report copies of indexes are evidence, not live recovery indexes.

## C. Preserved assets

- **Base, repair, balanced, SFT v1/v2/sanity and Stage 2:** canonical Source checkpoints retained, trusted SHA256 checks and CPU mapped loads passed. Source base latest/previous and Stage 2 best/previous/latest remain intact.
- **END3:** all three payloads and unchanged latest/best/previous/final index roles retained. Qualification input remains slot-0, SHA `4f4e5d1540eb6813ad6d6f91f1a1a503bf8952632d51f9dc6410343a8e0c456b`.
- **Original two-hour run:** all four payloads, final/latest, previous and best retained; no budget reset or promotion occurred.
- **Corrective-02:** best step zero, latest/final step 128 and previous step 128 retained. Old step-32 slot-2 is also retained. Its existing index/spec/report were unchanged; only unreferenced step-96 slot-1 was retired.
- **Deterministic 934M recovery fixture:** both full states retained and checksum/CPU-load verified. Its preserved report demonstrates bitwise subsequent model and optimizer updates. The failed nondeterministic fixture's report remains as negative evidence.
- **Current optimizer/scheduler/RNG/sampler state:** protected complete native payloads unchanged byte-for-byte against trusted hashes; no tensor precision or recovery field was changed.
- **Tokenizer:** SHA `a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e` verified; vocabulary 16,384 and reserved IDs 0–7 verified.
- **Datasets:** General Chat train/validation/test full hashes and base `train.bin` full hash passed against existing manifests. Dataset files were neither transformed nor removed. Base index is present. Missing original base validation assets remain a pre-existing limitation.
- **Exports, environments, current configurations and components:** existing assets/installed packages/default backend untouched. Two additional experimental exports are preserved, not promoted. New operator/test/config files belong solely to integration; existing checkpoint infrastructure was not changed.
- **NumPy and unrelated projects:** no cleanup operation targeted them. No filesystem mounts, snapshots or partitions were modified.

Metadata comparison found **zero changes among 1059 protected historical files**, excluding explicitly selected deletion artifacts. This is a device/inode/size/mtime/ctime comparison, not a claim of full hashing every historical file. All 25 protected native `.pt` files were independently checksum-verified and actually CPU-loaded. SHA256 covers complete protected model/recovery bytes.

## D. Storage before and after

| Measurement | Bytes | GiB |
|---|---:|---:|
| Filesystem total | 1,021,415,067,648 | 951.266911 |
| Recorded free before exports/cleanup | 105,206,951,936 | 97.981609 |
| Recorded free after settling/validation | 214,502,428,672 | 199.770954 |
| Net observed free-space increase | 109,295,476,736 | 101.789345 |
| Remaining filesystem used | 804,198,100,992 | 748.967846 |
| Surplus over unchanged 180 GiB floor | 21,228,900,352 | 19.770954 |

Measurements use the live encrypted Btrfs filesystem's `statfs`/`df` and native usage observations. Existing `zstd:1` compression remains unchanged. No hypothetical reclaimed bytes were credited before deletion. No project-related open-but-deleted descriptors/mappings were found in accessible same-user processes. Three unrelated service/sandbox processes had unreadable maps/fds; their identities were documented, no permissions were changed and no process was terminated.

Native Btrfs usage additionally reported an estimated free-space pessimistic minimum of **147.463430 GiB**, below the statfs figure, because future chunk/profile allocation assumptions differ. The existing launch policy uses statfs; it was not lowered or rewritten. Detailed per-device chunk information and subvolume listing were permission-denied. Thus the successful existing storage preflight is not an independent guarantee about every future Btrfs allocation scenario. These uncertainties remain documented; no forced transaction cleanup or snapshot removal was performed.

## E. Integrity validation

| Check | Actual outcome |
|---|---|
| Exact selective-file cleanup/fault fixtures | 7 passed: normal removal, changed file, symlink parent, hardlink, outside scope, detached hash mismatch restoration, interrupted unlink preservation |
| Bounded adapter/spec guards | 6 passed; valid TrainingConfig and refusal of expanded budget/backend/steps/reclamation/reduced reserve |
| Integration suite | **148 passed, 5 skipped, 8 warnings** |
| Architecture | **6 passed** |
| Training | **102 passed, 2 warnings** |
| Inference | **1 passed** |
| Data | **19 passed** |
| Tokenizer | **6 passed** |
| Protected native checksums and CPU mapped load | **25 passed; no checksum skips/failures** |
| Protected recovery indexes | All 16 logical role references checked; live RecoveryStore resolved all ten distinct indexed payloads |
| Selected initializer exports | Repair and balanced: native/export token IDs identical; complete state-dictionary byte/dtype equality passed |
| Tokenizer + datasets | Special IDs/vocabulary, conversational splits and base token stream passed |
| Architecture construction | Meta-device parameter count exactly **934,356,480**, no GPU tensors allocated |
| CLI/environment | Training, inference and new adapter help passed; `pip check` passed; imported owners remain in the intended generative `.venv` |
| Final removal/protection checks | 76 intended real artifacts absent, no leftover quarantine file, zero protected metadata changes |
| Next-run preflight | Disk 180 GiB, host 12 GiB and measured GPU-headroom gates passed; no run directory created |
| Full-model CUDA qualification | **Not run: separate approval absent; host 16 GiB gate fails** |

There were **282 passing tests across the six overlapping repository invocations**, not 282 distinct new tests. The final focused rerun passed 13 and is not added to that count. Integration's five skips remain skips; they were not changed to passes. Training's displayed 447-second test duration includes an intentional pause while reducing I/O contention and is not a performance benchmark. Relative interpreter invocation emitted prefix spelling warnings while resolving the same virtual environment; no environment was substituted.

Preserved failed attempts: initial inventory conservatively refused unreadable service-process inspection; their non-model identities were subsequently recorded. The first canonical-initializer export attempt correctly rejected missing historical tokenizer fingerprint metadata. No metadata was backfilled and the check was not weakened: selected step-zero native payloads supplied the correct tokenizer identity, followed by independent bitwise comparison against the unchanged canonical initializers. All final gates passed. Raw logs, including failed attempts, remain in the evidence directory.

Commands used included `.venv/bin/python -m pytest -q` for integration and the same generative interpreter against each sibling's tests, `scripts/chat_training.py --help`, `python -m llm_inference.cli.chat_native --help`, and `scripts/bounded_chat_confirmation.py --help/preflight`. Controlled fixtures were created under the new evidence namespace; no historical fixture store was pruned.

## F. Training readiness

### 1. Bounded conversational validation

**Prepared; separate approval required.** Resource/identity preflight passed at 200.085316 GiB free, 13.615452 GiB available host RAM and 11.711914 GiB free GPU memory. Fresh gates must be repeated at actual launch. The model has not passed conversational production quality acceptance.

New proposal: `configs/chat-end3-confirmation-v1.yaml`, run `codexa-900m-chat-end3-confirmation-v1`. This is a **20-minute independent-seed replication** of END3 termination weighting, not blind repetition of the unsuccessful two-hour adaptation or a proven intelligence improvement. It initializes protected repair weights with a new optimizer; it is **not exact optimizer resume**.

- Existing full General Chat training pool, proportional conversation sampling, no replacement within epoch, complete-conversation dynamic padding, assistant-only supervision; tokenizer/data identities frozen. Existing validation population/indices retained; exposure limitations remain.
- BF16, AdamW8bit, gradient checkpointing, microbatch 1 × accumulation 8, seed **74**, LR **5e-6**, warmup **20**, decay **0.1**, clip **1.0**, END token 7 weight **3**.
- Safety cap **512 updates**, **1,200-second** session including startup, **180-second** finalization reservation; save/evaluate each **128** updates, existing behavioral patience **2** and declared task/termination/repetition/empty/role/length guards.
- Legacy six-slot recovery, reclamation **disabled**. Conservative capacity: six 5.249-GiB payloads + one staging payload + 1 GiB logs ≈ **37.743 GiB**, inside the existing 80 GiB working allowance. This is a projection using actual AdamW8bit checkpoint size, not an FP32 Adam assumption or a measured future run.
- Development suite fingerprint matches the existing fixed 64-case suite. Historical selected-initializer instruction and arithmetic accuracy were zero; macro task accuracy approximately 0.142857. These are reused historical diagnostics, not newly measured quality scores. Current full weights match that initializer, but no new quality benchmark is claimed.

The thin new operator delegates to existing `run_chat`, SessionBudget, RecoveryStore, behavioral evaluator, hard-deadline supervisor and automatic visible 100×22 Kitty viewer. It does not change training mathematics or current defaults. Configuration/operator identities are frozen at launch; approval must explicitly accept the bounded experimental quality limitations. Existing run names are refused rather than overwritten or given a fresh budget.

Existing short-answer coverage, UltraChat token dominance, poor reasoning/factual behavior and repair-data exposure remain unresolved. General Chat has 182,892 training conversations and approximately 147.58M assistant tokens in the existing audit; UltraChat supplies approximately 93% of those targets. Source-proportional replication does not correct that imbalance. The possibly clean validation indices are limited, predominantly OASST; the protected acceptance suite must remain outside tuning. Source-stratified and short-instruction acceptance work is required before an extended adaptation. No training data was altered during cleanup.

### 2. Full-model optimized-checkpoint GPU qualification

**Not ready to execute.** Read-only qualification plan passes the existing **180 GiB** disk gate. Bounded peak artifact/staging estimate is **27.245001 GiB**. Actual full END3 representation remains FP32 model 3,737,425,920 bytes, uint8 optimizer 1,868,562,432 bytes plus FP32 optimizer 29,800,448 bytes, with complete native recovery metadata. It was neither downcast nor quantized by this task.

The harness's unchanged **16 GiB available host RAM** gate fails: planning observed 13.591156 GiB; final snapshot observed 14.537132 GiB. Final free GPU memory was 12.239258 GiB; swap pressure remains material. No desktop application was terminated. Separate GPU authorization also remains absent. Deterministic CUDA/AdamW8bit continuation, full-size save/restore latency, VRAM/host peaks and training throughput remain unqualified; CPU tests do not establish those results.

### 3. Extended ten-hour training

**Not ready.** Storage shortage has been removed under the existing statfs launch policy. Conversational-quality/data-exposure acceptance, representative GPU qualification, full-size throughput/recovery evidence and explicit launch approval remain outstanding. Legacy remains default; real-store V3 reclamation remains disabled and has not been executed. No accepted model/export/default inference checkpoint was replaced.

### Exact proposed commands

Run from the integration repository:

```bash
cd '/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch'
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python scripts/bounded_chat_confirmation.py preflight
```

After **separate explicit approval** is recorded in an experiment-specific approval file (none was created by this task):

```bash
.venv/bin/python scripts/bounded_chat_confirmation.py start --approval-file outputs/storage-cleanup/20261010-controlled-01/approved-end3-confirmation.json
.venv/bin/python scripts/bounded_chat_confirmation.py view
tail -n 50 outputs/chat-training/codexa-900m-chat-end3-confirmation-v1/console.log
.venv/bin/python scripts/bounded_chat_confirmation.py stop
.venv/bin/python scripts/bounded_chat_confirmation.py evaluate
```

Required approval content, **only after actual consent**:

```json
{
  "authorize_bounded_training": true,
  "accept_experimental_quality_limitations": true,
  "run_name": "codexa-900m-chat-end3-confirmation-v1",
  "configuration_sha256": "87a1904825157456ab1a47e3a25f2f8c1ce90cab73b4d360eeab36e2d52725a3"
}
```

The adapter has no implicit resume/budget-reset operation. A stopped phase retains its complete recovery payload; any continuation needs an explicit remaining-budget/recovery review or an independently authorized new phase. The viewer is independent after automatic attachment; graceful stop validates PID/start ticks before SIGTERM. The existing watchdog enforces the original hard deadline, with recovery from the last verified checkpoint after a hard kill.

GPU **plan only**, safe without executing the model:

```bash
.venv/bin/python scripts/checkpoint_gpu_qualification.py --checkpoint outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints/slot-0/checkpoint.pt --output outputs/checkpoint-gpu-end3-post-cleanup-v1 --updates 2 --sequence-length 2048
```

GPU execution must wait for all resource gates and separate approval. No `--execute` command was run. Future qualification must compare actual full model/optimizer/RNG/scheduler/sampler state and continuous/resumed CUDA updates, not approximate output similarity.

## G. Remaining risks and deliberate loss

The 23 removed checkpoint payloads have **no independently verified complete-state backup**. The four old pilot experiments, two retired memory pilots and failed nondeterministic recovery fixture can no longer be resumed from their former local states. Additional diagnostics requiring those trained weights, including historical pilot-teacher replay, are intentionally unavailable. Lightweight prior scores/generations/specifications remain; rerunning an experiment would create a new artifact, not recover the original exact state. The retained selected-model exports preserve weights only, not deleted optimizer/RNG/sampler histories.

No current executable configuration or registered run index referenced the removed payloads outside their own retired stores. Dependency evidence includes literal searches and dynamic inspection of investigation/preparation path construction: cached results and END3's control spec/report remain accessible. It is not a proof about arbitrary future commands that explicitly request retired weights. No active training/recovery consumer was found; standard operator launch lock was held during cleanup. File ancestors/mounts/symlinks/hardlinks, full hashes and fresh inode identities were checked; exclusive quarantine rename and detached content verification preceded unlink.

Storage values fluctuate with other workstation activity. Btrfs delayed accounting was observed, open-deleted inspection found no related hold, and snapshot/per-device visibility was limited by permissions. No elevated or forced cleanup was attempted. The final free reading is slightly below the observed 200 GiB target, but remains above the unchanged production floor. Additional historical states were retained rather than deleting another 5.249-GiB checkpoint to chase a small fluctuation.

## H. Final status

**CLEANUP COMPLETE — ADDITIONAL REQUIREMENTS REMAIN.**

Actual selective deletion and validation are complete. Protected model/recovery/data/tokenizer assets remain intact, current recovery controllers resolve their states, and a concrete separately gated bounded proposal is prepared. No training, production optimized-store reclamation, checkpoint promotion, dependency installation, Git publication or unrelated-project cleanup occurred.

Implementation ownership: new integration files `src/codexa_workspace/selective_cleanup.py`, `src/codexa_workspace/bounded_chat_confirmation.py`, `scripts/bounded_chat_confirmation.py`, `configs/chat-end3-confirmation-v1.yaml`, `tests/test_selective_cleanup.py`, `tests/test_bounded_chat_confirmation.py`; this report and dated session/progress journal appends. No component repository implementation was modified. Ignored evidence, retirement markers and selected exports are the only artifact writes.
