# Codexa checkpoint storage V3 engineering report

Date: 10 October 2026. Scope: continuation of V1/V2, not a new serialization architecture. Decision: **REQUIRES ADDITIONAL VALIDATION**. Legacy remains the production default. Real-store reclamation is disabled. No full-model CUDA qualification or long training was launched.

## A. Root-cause review

V2 counts every published immutable snapshot against its admission limit, including superseded manifests. Updating latest/previous/best references does not remove those old manifests or their object dependencies. Six distinct states therefore exhaust admission; reserving final publication can stop the illustrative ten-minute schedule after approximately 30–40 minutes. V1 avoids admission failure by retaining everything, which permits unbounded payload growth. Identical model weights can share objects, but distinct optimizer, RNG, sampler and recovery metadata remain distinct states.

V3 separates logical roles, retained roots, superseded snapshots and component reachability. It can rotate continuously only in a newly enrolled store with independent execution authorization. Without that authorization it retains V2's safe admission refusal; the experimental configuration alone cannot authorize deletion.

## B. V3 architecture

The serialization format remains `codexa-component-checkpoint-v1`. Newly created continuous stores additionally declare `codexa-continuous-retention-v1` in their immutable store marker and bind an immutable retention-policy digest. Existing stores cannot be adopted retrospectively.

Publication lifecycle:

1. At an optimizer boundary, check actual free space, current roots, quota, next-write reservation and outstanding final reservations. Never credit projected reclamation as free capacity.
2. Acquire the existing exclusive writer lock. Complete previously authorized, journaled reclamation before another publication.
3. Serialize and validate immutable model, optimizer and recovery objects. With a valid grant, admit one transient incoming snapshot beyond the retained-state limit.
4. Publish the complete immutable manifest, then atomically update the recovery index and registered-run initialization state. Preserve latest, previous, requested rollback depth, behavioral best, validation best, final, milestones and boundary roles. Several labels may share one manifest.
5. Identify roots across **all registered runs**, policy pins, explicit cross-run dependency registrations and active readers. Validate manifests, component SHA256, index ownership and reference counts.
6. Create a bound candidate plan. Under writer and exclusive reader-admission locks, revalidate the plan and acquire candidate snapshot lease locks. Publish and synchronize immutable reclamation intent before any unlink.
7. Remove only explicitly enumerated, eligible superseded manifests, synchronize, recalculate reachability, then remove only now-unreferenced objects with matching exclusive file identity. Synchronize and verify the resulting graph; publish an auditable completion status with approval identity and observed free-space measurements.

Readers take a shared admission gate before index selection, verification and mapping. Snapshot leases survive the returned dictionary through tensor-storage lifetime; detached tensors and NumPy views are covered. The executor takes the exclusive gate and candidate leases. Registered peer-run roots and shared objects remain protected. Process termination releases OS leases. Direct unmanaged file readers and applications bypassing this protocol are outside its guarantee.

A plan binds marker/policy, indexes, roots, manifests, object dependencies and file identities. A stale initial plan aborts. An existing immutable intent supports interruption replay, including already-absent candidates and safe final-role publication, after current roots are revalidated. Completed replay is idempotent. Unknown owners, corrupt indexes, unproven orphan payloads, missing reader gates and quota conflicts stop safely. More protected states than the quota never triggers forced eviction.

`ReclamationGrant` is a separate store-scoped capability, not a training flag. It binds the approved new namespace, store ID, immutable marker and policy digests, approval ID and purpose. Training configuration accepts only `disabled` or `simulate`; the operator API accepts a separately validated grant. This repository provides a grant factory only for newly created disposable fixtures and **no production grant issuer or cleanup CLI**. Trusted local operators are the authorization boundary; this is not a sandbox against malicious Python code with filesystem access.

## C. Implementation summary and ownership

These are V3-owned changes; working trees already contained earlier conversational/V1/V2 changes, which were preserved.

| Repository | Files | Change |
|---|---|---|
| LLM-Architecture | `src/llm_architecture/component_checkpoint.py` | New-store lifecycle marker, reader admission gate, manifest/policy guards, transient publication allowance and transaction-bound atomic metadata staging |
| LLM-Architecture | `src/llm_architecture/checkpoint_inspection.py`, `tests/test_checkpoint_inspection.py` | Read-only native hash/FakeTensor storage evidence, tied-storage accounting and optimizer completeness |
| LLM-Training | `src/llm_training/checkpoint_retention.py` | Continuous policy, rollback/pins, registered dependency roots, projected role admission, final reservations and transaction-local verification reuse |
| LLM-Training | `src/llm_training/checkpoint_reclamation.py` | Read-only planning, separately granted executor, durable intent/replay, reader-safe object graph and fixture-only grant factory |
| LLM-Training | `src/llm_training/optimized_recovery.py` | Publication/rotation integration, boundary admission cache, pin registration, explicit fallback and final alias after post-publication reclamation failure |
| LLM-Training | `src/llm_training/recovery.py`, `checkpointing.py`, `chat_run.py` | Backend configuration forwarding, separate authorization parameter and accurate published-versus-failed handling |
| LLM-Training | `src/llm_training/preflight.py`, `tests/test_checkpoint_reclamation.py` | Capacity projections, explicit verified optimizer evidence, fault/race/state-continuation tests |
| LLM-From-Scratch | `src/codexa_workspace/chat_preparation.py` | Validate configuration without minting deletion authority; optional matching optimizer evidence |
| LLM-From-Scratch | `src/codexa_workspace/checkpoint_benchmark_v3.py`, `checkpoint_gpu_qualification.py` | Bounded CPU benchmark and separately gated future GPU qualification harness |
| LLM-From-Scratch | `scripts/benchmark_checkpoint_storage_v3.py`, `checkpoint_gpu_qualification.py`, `checkpoint_storage_capacity.py`, `checkpoint_retention_plan.py` | Thin operator entry points; capacity/default GPU planning and retention planning are read-only |
| LLM-From-Scratch | `configs/checkpoint-storage-v3.example.yaml`, `tests/test_checkpoint_v3_operations.py` | Experimental fresh-store fragment and operational gates |
| LLM-From-Scratch | this report, `SESSION_DECISIONS.md`, `100M_PROGRESS_LOG.md` | Evidence, decisions and limitations |

New APIs include `plan_reclamation`, `execute_reclamation`, `load_reclamation_grant`, `create_disposable_fixture_store`, `authorize_reclamation`, `pin_snapshot`, `finalize_published_checkpoint`, `inspect_checkpoint_storage`, `project_continuous_storage` and `verified_optimizer_storage_evidence`.

Configuration exposes backend/root, continuous policy, maximum snapshots/bytes, minimum rollback depth, policy pins and disabled/simulate mode. Default V2 policy documents retain their prior byte identity. Legacy/native loading remains available. Architecture, Training and integration wheels were rebuilt using existing no-dependency workflows and force-reinstalled without dependency upgrades. Installed Python sources match their corresponding source trees byte-for-byte; `pip check` passed.

## D. Safety guarantees and limits

Proven on isolated fixtures: protected roles and registered dependencies are not evicted; shared model objects survive distinct optimizer states; active mapped-storage leases prevent reclamation; stale/corrupt/unknown graph states fail closed; configuration booleans cannot issue grants; a failed save does not masquerade as successful publication; interrupted unlink/status/synchronization can replay durable intent; explicit previous fallback is visible; model/optimizer precision and state are preserved.

A post-index reclamation failure raises `PublishedCheckpointReclamationError` with the published record. The trainer records that checkpoint and halts as `RECLAMATION_FAILED`, preserving a final alias rather than silently retrying a new payload. A failed write before publication retains the previous valid index.

Crash tests include actual child-process exit after unlink, reader process termination, injected I/O/fsync failures and partial status writes. Proven transaction-owned status staging files can be identified from a valid durable intent, are preserved and counted, and do not prevent replay. Unproven `.object-*` or generic publication stages remain unknown and block execution. Automatic orphan deletion was deliberately not implemented.

Audit journals grow modestly with publication count and are counted toward the hard byte quota. Tensor snapshots rotate within a fixed cap; unlimited operation is **not** promised when metadata consumes the quota, readers retain many states, milestones accumulate or protected peer-run roots exceed the quota. Such conflicts stop safely. Grant/quota must be sized for the actual number of registered experiments and required pins.

No production full-model power-loss, filesystem-durability, actual bitsandbytes continuation or GPU-throughput guarantee is claimed from CPU fault injection. Shared-object corruption can affect every snapshot referencing that object; independent verified backups remain advisable. Unmanaged writers, stale remote filesystems and unsupported reader protocols are not qualified.

## E. Storage comparison and planning

Read-only inspection of existing END3 slot-0 native checkpoint verified its full SHA256 and sidecar without CUDA/model allocation:

- File: 5,636,070,993 bytes = approximately 5.249 GiB.
- Unique model storage: 3,737,425,920 bytes, FP32.
- Optimizer: 1,868,562,432 uint8 bytes plus 29,800,448 FP32 bytes = 1,898,362,880 bytes. This is actual 8-bit optimizer representation, not a hypothetical FP32 Adam estimate.
- SHA256: `4f4e5d1540eb6813ad6d6f91f1a1a503bf8952632d51f9dc6410343a8e0c456b`.
- Explicit evidence is rehashed and matched against complete optimizer state, exact model configuration and `adamw8bit` before it affects startup sizing. Without evidence, the existing conservative FP32 estimate remains.

The following are **projections**, not reclaimed space. Ten-minute cadence plus initialization/final gives 14/32/62 publications. Assume all payloads are distinct and no compression/sharing credit. Six protected states fit; no indefinitely pinned readers/milestones. Values exclude metadata/staging; conservative quota covers them.

| Metric | Legacy six-slot | V1 append-only | V2 admission-only | V3 authorized continuous |
|---|---:|---:|---:|---:|
| Retained states | At most 6 slots | All publications | At most 6; then refusal | At most 6 after successful rotation |
| 2-hour payload projection | 31.494 GiB | 73.486 GiB | 31.494 GiB; cannot finish schedule | 31.494 GiB |
| 5-hour payload projection | 31.494 GiB | 167.968 GiB | 31.494 GiB; cannot finish schedule | 31.494 GiB |
| 10-hour payload projection | 31.494 GiB | 325.438 GiB | 31.494 GiB; cannot finish schedule | 31.494 GiB |
| Temporary write allowance | Full incoming slot/write | Incoming components | Reserved incoming/final | Incoming components plus transient manifest, final reservations |
| Recovery reserve | Existing production policy | Existing production policy | Existing production policy | Existing 180 GiB floor; never reduced |
| Growth | Fixed slots | Linear payload growth | Bounded; safe stop | Bounded payloads, accounted audit metadata |
| Continuous operation | Existing slot semantics | Yes if space lasts | No after quota | Only separately authorized new stores; synthetic proof only |
| Recovery | Existing native semantics | Complete components | Complete components | Complete components; exact CPU continuation verified |

V3 measured-size bound adds 10% plus 16 MiB and rounds up to MiB: 6,217,007,104 bytes (5.790 GiB). One active run/six retained states uses `(6 + 2×1 + 2) × bound + 8 MiB` = 62,178,459,648 bytes, approximately 57.91 GiB. The required free-space threshold remains `max(180 GiB, 100 GiB + quota)` = **180 GiB**. It does not use hypothetical future reclaimed bytes.

Sequential completed experiments retain their boundary/final/best roots. If every experiment keeps six states, two/five/ten experiments can retain approximately 62.99/157.47/314.94 GiB of distinct payloads. A global six-state quota cannot satisfy unlimited experiment preservation. Increase an explicitly planned quota or stop; do not evict protected experiment history.

For concurrent experiments sharing a store, global retained roots must fit max snapshots and each active run reserves incoming/final capacity. Two concurrent runs needing six independent roots each require at least twelve slots; worst-case quota `(12+2×2+2)×bound+8 MiB` ≈104.23 GiB and required free space ≈204.23 GiB. Shared objects may reduce actual allocation, but planning does not credit unverified sharing.

## F. Performance benchmarks

Representative **synthetic nonzero CPU** payloads, two PyTorch threads, five warmups, fifty rotated-order restore samples. Full SHA verification remains enabled. Fixtures are 12 MiB and 96 MiB, not 934M qualification. V2 modules were frozen before V3 wheel installation.

Final installed measurement (`restore-installed-final.json`):

| Payload | Legacy restore median | V2 median | V3 median | V3 versus legacy | V3 versus V2 |
|---|---:|---:|---:|---:|---:|
| 12 MiB | 7.205 ms | 7.528 ms | 7.728 ms | +7.26% | +2.66% |
| 96 MiB | 51.939 ms | 50.350 ms | 50.336 ms | −3.09% | −0.03% |

Both meet the ≤10% CPU restore regression gate. SHA is verified once per component per load; metadata work is not duplicated behind a second reader gate. Writer hash reuse is confined to one exclusive transaction; no long-lived tensor-verification cache is trusted for loading.

Additional sampled RSS above isolated worker baseline, every 1 ms: 12 MiB legacy/V2/V3 = 16,744/16,952/16,948 KiB; 96 MiB = 102,760/102,980/102,984 KiB. These are sampled resident increments, not guaranteed instantaneous peaks or GPU memory measurements.

A separate 62-publication rotating fixture changed model and optimizer tensors every update. V3 retained six snapshots, reclaiming 56 superseded states in its task-created disposable namespace. Legacy steady save median 66.675 ms, V3 save+rotation 204.356 ms; CPU medians 71.247/158.103 ms. V3 checkpoint-boundary latency is about **3.07×**, and increases across this short history as audit records accumulate. This does not pass a save-parity claim and must be evaluated at representative scale. No GPU/training-throughput claim is made.

Final allocated fixture storage: legacy 50,450,432 bytes versus V3 76,607,488 bytes. This particular legacy scenario kept four physical states (latest/previous/old-best/final alias), whereas V3 intentionally retained six; the larger V3 result is not a like-for-like six-state duplication saving. Its demonstrated benefit is bounded continuous history versus append-only growth with protected roots. Repeated logical labels share objects and do not duplicate payloads.

Btrfs extent inspection reported no shared extents for these fixtures: 12 MiB comparison set 37,818,368 exclusive bytes; 96 MiB comparison set 302,059,520 exclusive bytes; rotation set 126,746,624 exclusive bytes. No optimistic reflink/compression savings were assumed. The total new validation area occupied about 479 MiB after tests/builds, below the 512 MiB CPU fixture guard; no multi-gigabyte test checkpoint was created.

The first benchmark report had a legacy restore path performing duplicate hash verification and an unsuitable inherited RSS high-water metric. Its original restore/RSS numbers are superseded by corrected and final installed reports; original artifacts remain available. Rotation/save/allocation observations are independent of that restore correction. One final CLI invocation used a baseline-file argument where a directory was required, failed without publishing a report, and was corrected.

## G. Recovery verification and regression tests

Latest installed Architecture/Training/integration combined rerun: **239 passed, 5 skipped, 11 warnings**, 39.03 seconds, zero failures. Other unchanged suites: Inference 1 passed, Data 19 passed, Tokenizer 6 passed, Memory 5 passed, Specialist 34 passed/5 skipped. Latest aggregate across repository invocations: **304 passed, 10 skipped, zero failures**. Counts include overlapping integration/component coverage and are not unique behavioral test counts.

Five optional sklearn tests were skipped in both Specialist and integration because that dependency is absent. No dependency was installed to hide skips. Existing return-value warnings remain; the combined invocation also reported pytest's inability to create `/dev/.pytest_cache` with `-c /dev/null`, which does not affect results. Initial Specialist execution had two subprocess import failures because its package is intentionally absent from the generative environment; an unchanged-source retry with `PYTHONPATH=../LLM-Specialist/src` passed. Original failure logs are preserved.

Coverage includes six-state and 66-publication histories; protected roles/milestones/rollback-depth conflicts; model sharing with distinct optimizer tensors; multiple registered runs and cross-run dependencies; concurrent readers/writers; detached storage and NumPy leases; actual process interruption; missing/corrupt indexes, markers and objects; stale/mutated plans; interrupted journal/fsync/unlink; repeated execution; explicit fallback; legacy compatibility; bitwise model and optimizer recovery; scheduler/RNG/sampler/cursor/counter restoration; stochastic CPU subsequent optimizer updates. uint8 optimizer tensors are preserved without conversion. Actual bitsandbytes GPU continuation is not proven by that dtype fixture.

Commands used include:

```bash
.venv/bin/python -m pytest -c /dev/null ../LLM-Architecture/tests ../LLM-Training/tests tests --basetemp outputs/checkpoint-storage-v3-validation/tests-final -q
# Other repositories were invoked separately; Specialist retry used:
PYTHONPATH=../LLM-Specialist/src .venv/bin/python -m pytest -c /dev/null --rootdir ../LLM-Specialist ../LLM-Specialist/tests -q
.venv/bin/python -m pip check
```

`git diff --check` passed for all three affected owners. Installed-source byte comparison passed for Architecture, Training and integration. No publish, commit, PR or dependency upgrade was performed.

## H. Historical preservation and archival assessment

Baseline filesystem metadata checks covered 1,540 existing asset/investigation/preparation/configuration files plus 35 original chat-training files: **zero device/inode/size/mtime/ctime changes** and no missing files. Atime was excluded because reads may update access metadata. Historical checkpoints, datasets, exports and experiment specifications were not modified, migrated or deleted. Source code, new documentation, installed project wheels and task-created disposable fixtures changed as authorized. Fixture reclamation did unlink fixture objects; this report therefore does not claim that no files anywhere were deleted.

Read-only current inventory found 47 native checkpoint files in the selected canonical/preparation/investigation/original-chat scope; this is not an exhaustive count of every checkpoint format on the machine. Existing sidecar hashes and completeness metadata were used rather than hashing hundreds of GiB.

The three historical output areas occupy approximately 174.86 GiB: investigation 92.67 GiB, preparation 62.95 GiB, original two-hour training 19.24 GiB. Initializers, original two-hour final, END3 candidate and its recovery payload, behavioral comparisons and published-investigation dependencies remain protected. END3 has three full native slots (approximately 15.75 GiB); matching latest/final labels do not establish interchangeable recovery metadata. Base, repair, balanced, v2 and Stage2 lineages are referenced by current configurations and remain locally required unless those dependencies are explicitly reviewed.

Preparation plus one non-END3 pilot could represent roughly 82 GiB of future archival capacity, **conditional upper bound only**. No independent mounted destination is verified. Unmounted approximately 476.9 GiB SATA and 930.5 GiB NVMe NTFS partitions were observed; their available capacity, reliability and intended ownership remain unknown. Nothing was mounted, copied or removed.

Future archival requires complete dependency inventory; verified independent destination capacity; exact copies including sidecars/indexes/manifests/state; hashes and actual restore testing from destination; durable long-term accessibility; explicit authorization before any local original removal. An initializer needed offline or an active recovery state must remain accessible. Actual reclaimed historical space: **0 bytes**.

## I. Full-model GPU qualification

Read-only plan observed RTX 4080 total 16,376 MiB, approximately 11,967 MiB free; host available RAM approximately 14.85 GiB and swap used approximately 5.83 GiB. Desktop processes were left untouched. Available disk at final resource-plan snapshot: 109,653,270,528 bytes, approximately **102.12 GiB**, a **77.88 GiB deficit** against 180 GiB. A subsequent lightweight check observed about 102.05 GiB; availability fluctuates.

Bounded qualification estimates two native/two optimized states plus one full staging allowance and 1 GiB logs: approximately **27.25 GiB**. Required startup remains 180 GiB, not 27.25 GiB. The plan refuses execution even though the fixture bytes alone could fit. The harness additionally requires ≥16 GiB available host RAM and a live GPU-memory estimate of two model copies plus optimizer and 2.5 GiB workspace. Current RAM also fails its conservative gate.

Prepared harness: 2–8 controlled updates, context 64–2048, BF16/autograd checkpointing, existing AdamW8bit, deterministic CUDA settings, original model configuration, separate synthetic qualification cursor, ≤900-second session with boundary save reserve. It checks native and optimized bitwise model/optimizer/RNG/scheduler/scaler/counter restoration, continuous-versus-resumed updates, explicit previous fallback, GPU memory, save/load latency and update throughput. It requires a new output namespace, input SHA and a separately approved bounded job, opens the required visible Kitty viewer, and never enables reclamation.

CUDA numerical equivalence depends on supported deterministic kernels and actual optimizer implementation. The harness must fail/report unsupported determinism rather than treating approximate equality as exact. Synthetic full-model continuation would still not establish original dataset-session replay or long full-model rotation; representative SFT and continuous-retention qualification remain separate gates.

Safe read-only operator commands, from integration root:

```bash
.venv/bin/python scripts/checkpoint_storage_capacity.py --checkpoint /absolute/path/to/verified/checkpoint.pt
.venv/bin/python scripts/checkpoint_retention_plan.py --checkpoint-directory /absolute/path/to/new/optimized/run --store-root /absolute/path/to/new/store --reclamation
.venv/bin/python scripts/checkpoint_gpu_qualification.py --checkpoint /absolute/path/to/verified/checkpoint.pt --output outputs/codexa-934m-v3-gpu-qualification --updates 8 --sequence-length 2048
```

After reserve/resource gates pass and **separate authorization** is obtained, the prepared execution command is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/checkpoint_gpu_qualification.py --checkpoint /absolute/path/to/verified/checkpoint.pt --output outputs/codexa-934m-v3-gpu-qualification --updates 8 --sequence-length 2048 --execute --approval-file /absolute/path/to/approved-bounded-qualification.json
```

Approval JSON format `codexa-gpu-qualification-approval-v1` must set `approved: true`, exact absolute output, updates, sequence_length and verified checkpoint_sha256. This GPU approval does not authorize reclamation. No execution approval was issued or used here.

For fixture-only reruns, use new pytest temporary paths. Existing benchmark payloads can be remeasured without duplication:

```bash
.venv/bin/python scripts/benchmark_checkpoint_storage_v3.py --baseline outputs/checkpoint-storage-v3-validation --restore-existing outputs/checkpoint-storage-v3-validation/benchmark-run2 --report-name another-exclusive-measurement.json
```

## J. Remaining work

1. Restore actual free capacity to the unchanged 180 GiB threshold (at least approximately 77.9 GiB additional at the measured snapshot), without counting hypothetical archival/reclamation. No cleanup is authorized here.
2. Verify host RAM/GPU headroom and obtain separate bounded full-model approval. Execute the prepared harness and investigate actual AdamW8bit/CUDA determinism, restore ≤10% regression and throughput ≤5% regression.
3. Measure full-size serialization/restore, CPU contention and temporary peak allocation. Small-fixture checkpoint-boundary overhead is substantial; profile graph/journal traversal at representative history length before promotion.
4. Qualify sufficiently representative continuous full-model publication/recovery, failure injection and independent durability/backup behavior. CPU process faults cannot prove real power-loss safety.
5. Establish an explicitly reviewed new-store operator grant issuance procedure before enabling any real-store execution. No automatic cleanup configuration or production issuer is supplied.
6. Investigate unproven orphan/staging ownership manually if encountered; automatic removal remains disallowed. Account for long-lived reader/pin growth and audit-journal quota exhaustion.
7. Optional Specialist sklearn coverage remains skipped; core required suites passed. Any archive destination requires independent verification and separate local-removal authorization.

Evidence is under ignored `outputs/checkpoint-storage-v3-validation`: final restore report, original save/rotation report, capacity/optimizer evidence, GPU plan, tests, wheel logs, installed-source verification, archival inventory and historical metadata comparisons. No simulated savings have been credited as current free space.

## K. Final decision

**REQUIRES ADDITIONAL VALIDATION**

V3 now provides tested continuous six-state rotation, dependency-aware shared-object retention, reader protection, separately granted journaled reclamation and exact CPU recovery while meeting the small-fixture restore threshold. Historical artifacts remain unchanged. Production storage preflight, full-model CUDA recovery/throughput and representative long-run performance/durability are unresolved mandatory gates. Legacy stays the production default; no real-store reclamation or long training is enabled.
