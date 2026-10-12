# Codexa V3.1 checkpoint performance, storage and GPU readiness

Date: 10 October 2026. Decision: **REQUIRES ADDITIONAL VALIDATION**.

## A. Executive findings

V3.1 makes targeted changes to V3; both `codexa-component-checkpoint-v1` and `codexa-continuous-retention-v1` remain unchanged. No migration is required. Legacy remains the production default and real-store reclamation remains disabled.

Measured matched 12 MiB publication-boundary latency at publication 62 fell from 210.910 ms to 163.061 ms, **22.7%**. A smaller 192 KiB tensor fixture fell from 216.117 ms to 157.743 ms at publication 120, **27.0%**. Continuous retention passed through 500 publications with six retained states and intact latest/previous/best roots. Restore measurements pass the 10% CPU gate. Near-legacy publication performance has **not** been achieved, and full-model throughput is unqualified.

Final resource snapshot: 99.469 GiB free; unchanged reserve 180 GiB; deficit **80.531 GiB**. Available host RAM 15.560 GiB also fails the unchanged 16 GiB harness gate. No GPU execution authorization exists. Historical artifacts were neither archived nor cleaned; no space from historical data was reclaimed.

## B. Checkpoint performance root cause

The original V3 120-publication profile used nonzero tensors, two PyTorch threads and the same factory-created disposable lifecycle. It took 40.294 seconds under cProfile. Inclusive attribution:

| Stage | Calls | Profiled inclusive time |
|---|---:|---:|
| Entire save | 120 | 40.284 s |
| Rotation, including pre/post checks | 240 | 29.982 s |
| Reclamation plan construction | 354 | 21.358 s |
| Unknown-file checks | 354 | 15.206 s |
| Ancestor/path resolution `_plain` | 108,873 | 14.151 s |
| Dependency graph `plan` | 822 | 10.152 s |
| Storage accounting | 1,531 | 5.545 s |
| Manifest validation | 5,561 | 5.278 s |
| Component snapshot publication | 120 | 5.211 s |
| Object serialization/hash/publication `_object` | 360 | 3.808 s |
| Atomic JSON publication | 818 | 3.452 s |
| fsync system calls | 3,360 | 6.567 s |

Times overlap and must not be added. The profiler saw 3,702,167 `lstat` and 621,207 `stat` calls. Unknown-file discovery repeatedly checked every ancestor for every completed journal entry, and accounting repeatedly reconstructed Path objects and walked the growing audit area. Durable status discovery reparsed completed records at each boundary. These explain history-dependent CPU cost; it was not merely tensor serialization.

`stage-profile.json` contains complete measured calls/exclusive/inclusive timings for serialization, hashes, manifests, graphs, journal replay, unlink, synchronization and locking. The frozen original and first optimized profiler outputs are preserved. Profiling significantly increases Python-heavy latency; final acceptance timings below have profiling disabled. First optimized profiling and some development tests overlapped, so those wall times are diagnostic rather than isolated acceptance measurements.

No syscall tracer or GPU synchronization profiler was used. Per-stage read/write bytes and separate lock-wait durations were not instrumented; aggregate `/proc/self/io` deltas are retained for unprofiled runs. `rchar/wchar` include cached/logical operations; `read_bytes/write_bytes` describe process I/O accounting, not exact Btrfs compressed extent consumption. No unsupported attribution is inferred from those counters.

## C. Implemented optimizations

| Owner / path | Change and rationale |
|---|---|
| LLM-Architecture `src/llm_architecture/component_checkpoint.py` | `store_entries` uses fresh `os.scandir` inode metadata, rejects internal symlinks and visits each entry once. `store_usage` deduplicates actual inode identities without repeated per-file stat/lstat and Path reconstruction. Registry-referenced external indexes remain counted and checked. Optional string paths reduce metadata allocation overhead. |
| LLM-Training `src/llm_training/checkpoint_reclamation.py` | Unknown-file checks validate root ancestry once and use the symlink-rejecting traversal, with relative names derived from that scoped walk. Partial status-stage ownership validation remains intact. Completion-stage discovery examines only the reclamation directory rather than the entire object tree. |
| Same Training module | Completed status JSON parsing is cached only when fresh device/inode/size/mtime/ctime matches. Missing/replaced/in-place-modified records invalidate it. Cache is bounded to 4,096 entries; older records are freshly parsed. This caches status interpretation, **never reachability, object hashes or deletion authority**. |
| Integration `src/codexa_workspace/checkpoint_gpu_qualification.py` | Tensor equality now compares contiguous dtype-preserving byte views; NumPy equality checks dtype/shape/bytes. Signed zero differences are rejected and equal NaN payloads accepted. Complete optimizer recovery state is required. Linux process peak host RSS is recorded. |
| Training `tests/test_checkpoint_reclamation.py` | New in-place completed-status corruption invalidation and symlink/unknown-file traversal tests. Existing reader, writer, stale-plan, exact-continuation and fault tests remain. |
| Integration `tests/test_checkpoint_v3_operations.py` | Byte-level signed-zero and NaN comparison tests. |
| Integration `scripts/profile_checkpoint_storage_v31.py` | Bounded original-source/legacy/current profiling and unprofiled history runner, 6–500 publications, 128/1024-side nonzero tensors, exclusive output creation, next-fixture allocation guard and CPU/I/O reports. Destructive behavior remains inside factory-created fixture grants. |
| Integration configuration/documentation | Experimental V3 fragment comments link this migration-free report; session/progress journals record evidence and acceptance. |

All source changes use existing ownership and APIs. Project wheels were rebuilt/reinstalled with `--no-index --no-deps --no-build-isolation`; installed module bytes match Architecture, Training and integration source. No dependency upgrade occurred; `pip check` passed.

No durability barrier was removed. No new serializer, quantization, tensor cast, scheduler/optimizer/data/tokenizer change, persistent graph-trust cache, production grant issuer or automatic cleanup was introduced. Logical aliases still share the existing immutable payload. Historical completed journals are intact: no compaction, truncation or relocation.

## D. Performance comparison

CPU-only measurements. “Original V3” below is the task-captured implementation, not rewritten historical files. The 192 KiB fixture consists of three nonzero 128×128 FP32 tensors plus recovery metadata. V2 was measured for six fresh publications; it has no continuous rotation. Legacy and continuous stores protect different logical roles, so their retained-state storage is not an identical retention policy.

| Metric | Legacy | V2 | Original V3 | V3.1 |
|---|---:|---:|---:|---:|
| Small fixture, first-six save median | 20.887 ms | 31.629 ms | 53.475 ms | 55.215 ms |
| Small fixture, last-30 at publication 120 | 23.627 ms | Not continuous | 216.117 ms | 157.743 ms |
| Small fixture, last-30 at publication 500 | 24.225 ms | Not continuous | Not measured unprofiled | 162.364 ms |
| Matched 12 MiB, last-30 at publication 62 | Prior V3 report: 66.675 ms, different invocation | Not measured | 210.910 ms | 163.061 ms |
| Restore 12 MiB, median | 7.374 ms | 7.622 ms | Prior final: 7.728 ms | 7.893 ms |
| Restore 96 MiB, median | 51.734 ms | 51.528 ms | Prior final: 50.336 ms | 51.152 ms |
| Rotating snapshot cap | Up to six legacy slots | Six then admission stops | Six | Six |
| CPU time, 12 MiB last-30 at 62 | Prior V3 report: 71.247 ms | Not measured at 62 | 177.571 ms | 129.145 ms |
| Allocated 12 MiB rotating fixture, 62 | Prior report: 50,450,432 B (four physical slots) | Not continuous | 76,607,488 B (six states) | 76,607,488 B (six states) |
| New peak CPU-memory qualification | Not repeated | Not repeated | Prior isolated RSS evidence | Not repeated; bounded status cache |
| Exact CPU recovery | Regression tests pass | Regression tests pass | Regression tests pass | Regression tests pass |

V3.1 restore versus current legacy: **+7.04%** at 12 MiB and **−1.13%** at 96 MiB. SHA verification remains enabled once per independently loaded component. Existing read-only payload fixtures were reused; no new restore payloads were needed. The V3 prior values are context, not a same-session improvement claim.

Matched 12 MiB publication median/P95/worst at 62:

- Original: 210.910 / 226.270 / 298.727 ms.
- V3.1: 163.061 / 181.919 / 214.287 ms.

The remaining publication cost includes fresh graph validation, accounted-byte scans, durable immutable object writes, role/index replacement, reclamation intent/status commits and candidate removal synchronization. There are about thirteen accounting passes per publication in the profiled lifecycle. Completed audit history still incurs fresh metadata traversal. It is explained **O(history)** work, not constant-time discovery. This optimization materially reduces overhead but does not finish the near-legacy writer goal.

Actual CPU time per publication is recorded in every unprofiled row, and a machine-readable comparison accompanies the report. Separate pure-rotation time is an inclusive profiler stage, not an independently timed identical workload. No full-model training-throughput parity or inference regression claim is inferred from these fixtures. No GPU memory was allocated by publication tests.

The bounded completed-status cache has at most 4,096 inode-identity entries. Full payload memory remains controlled by existing serialization. V3's earlier isolated restore RSS measurements remain contextual evidence; V3.1 did not repeat an isolated peak-RSS benchmark. CPU memory overhead beyond the bounded cache is not newly quantified. The GPU harness now captures host high-water RSS when separately authorized.

Using the small 500-state boundary median, 0.162 s per checkpoint is 0.162% / 0.054% / 0.027% of a hypothetical 100/300/600-second interval. Additional latency versus 0.024 s legacy is about 0.138% / 0.046% / 0.023%. These are arithmetic projections for this fixture, **not 5.249 GiB save-time predictions**.

## E. Long-run retention results

Final V3.1 unprofiled results; last thirty publications at each point, except the initial six. All observations belong to one 500-publication run.

| Publications | Median | P95 | Worst observed in window |
|---:|---:|---:|---:|
| 6 | 55.215 ms | 62.393 ms | 64.450 ms |
| 30 | 102.227 ms | 113.945 ms | 123.283 ms |
| 62 | 102.698 ms | 119.901 ms | 127.129 ms |
| 120 | 157.743 ms | 171.516 ms | 178.207 ms |
| 250 | 152.169 ms | 182.498 ms | 202.349 ms |
| 500 | 162.364 ms | 171.778 ms | 193.095 ms |

System background activity and filesystem synchronization fluctuate; these are observed windows, not an assertion of monotonic latency or stable percentile guarantees. Every publication preserves protected roots; final load confirms exact tensors/optimizer fields/counters, previous=499, behavioral best=1 and validation best=2. Boundary state remains protected. Full graph is unblocked and accounting fits quota. Cross-run dependencies, pinned milestones and active readers are tested separately, including readers outliving their parent dictionary and process termination.

The 500-state store accounted 9,461,760 bytes (approximately 9.02 MiB) under a 64 MiB quota; six states remained and 494 immutable reclamation transactions survived. Each completed plan/status pair accounted at most 16,384 bytes in this fixture. With four 1 MiB reserve bounds and unchanged metadata size, approximately 3,262 additional transactions could fit, roughly 3,762 total publications. This is a **conditional capacity projection**, not a guarantee for arbitrary manifests/reader/pin counts. Unknown files, larger provenance, filesystem allocation and accumulated protected experiments can reduce it. Quota is checked before publication/intent and includes journals; exhaustion stops safely rather than silently overcommitting.

The 934M measured-size V3 quota remains approximately 57.91 GiB for six retained states, one active run and staging/final allowances. Worst-case distinct payloads after 2/5/10 hours remain approximately 31.49 GiB under successful authorized rotation versus append-only 73.49/167.97/325.44 GiB. No simulated space is credited as available capacity. Multi-experiment protected roots and concurrent final reservations require larger explicit quotas, as documented in V3.

## F. Read-only archival investigation

47 native checkpoint records were re-inventoried; 116 explicit references were found in current configurations and bounded preparation/investigation documents. Sizes use actual inode allocation. The search is evidence, not proof that every possible dynamic dependency is known. Sidecars/completeness metadata were reused; hundreds of GiB were not rehashed or loaded. Archive candidates must preserve their complete experiment directory and dependency inventory, not isolated `.pt` files.

| Area | Allocated size | Classification | Dependencies and risk |
|---|---:|---|---|
| `outputs/preparation/codexa-900m-chat-10h-v1` | 62.95 GiB | Conditional archival | Pilots, recovery and deterministic recovery underpin preparation comparisons/exact-recovery evidence; preserve reports/specs/logs/sidecars and restart availability |
| `outputs/investigation/chat-regression-20261010/pilots/repair-proportional` | 19.23 GiB | Conditional archival | Published regression diagnostics/quality comparisons; repair initializer and tokenizer dependencies |
| `.../pilots/repair-balanced` | 19.23 GiB | Conditional archival | Same scientific/recovery preservation requirements |
| `.../pilots/balanced-proportional` | 19.23 GiB | Conditional archival | Balanced initializer plus investigation comparisons |
| `.../pilots/balanced-balanced` | 19.23 GiB | Conditional archival | Balanced initializer and exact experiment state |
| `.../pilots/repair-proportional-END3` | 15.75 GiB | Must remain locally available for current qualification | Selected candidate native state and optimizer evidence; multiple recovery metadata states must not be collapsed |
| `outputs/chat-training/codexa-900m-chat-2h-v1` | 19.24 GiB | Must remain locally available | Original two-hour final/rollback and accepted comparison population |
| Canonical Source initializers, tokenizer, datasets, exports | Individual inventory | Must remain local / protected | Active base/repair/balanced/v2/Stage2 configuration and offline input dependencies |

No category is verified disposable. Preparation plus one non-END3 pilot is about **82.18 GiB gross conditional allocation**. Preparation plus all four is about **139.87 GiB gross**; this sacrifices local access to a broader regression history and is not recommended without complete external restore/availability qualification. Gross inode allocation is not guaranteed free-space gain if shared extents, snapshots or other references retain blocks. Archival inventory records checkpoint paths, references, existing hashes and allocation; uncertain/dynamically referenced files remain retained by default.

Device inventory:

- Active volume: encrypted Btrfs on `nvme1n1`, VT-SSD-JET_P_ 1024G, mounted `/` and `/home`.
- `sda1`: approximately 476.9 GiB NTFS on separate internal SATA HS-SSD-E100 512G, serial 30035028816, not mounted, removable flag false.
- `nvme0n1p3`: approximately 930.5 GiB NTFS on separate internal NVMe MSI M371 1TB, serial 511240126203000152, not mounted, removable flag false.
- Mounted removable USB is installation/BIOS media, not a verified archive destination.

These are separate physical drives from the source, but share the workstation's power, theft and host failure domain. Device capacity is not free capacity. No destination was mounted and no raw filesystem contents were written/read to infer availability; permissions, occupancy, health, intended use, backup independence and long-term accessibility are unverified. No essential archive should be entrusted solely to an unverified internal partition.

Future archival procedure — proposal only:

1. Select exact experiment directories and enumerate all checkpoints/sidecars/indexes/specs/logs, upstream initializer, dataset/tokenizer identities and downstream references. Confirm no process/restart depends on local paths.
2. Separately approve destination inspection and copy. Verify destination identity, available bytes plus margin, filesystem large-file support, persistence, health and an independent backup. Do not overwrite existing destination contents.
3. Copy complete directories into new destination namespaces with source originals preserved. Btrfs reflinks/snapshots do not provide cross-filesystem sharing on NTFS; estimate a full physical copy. Preserve relevant timestamp/provenance/permissions explicitly; document NTFS differences.
4. Compare complete inventories and full SHA256 for every checkpoint/sidecar-critical payload; validate manifests and completeness. A successful copy alone is insufficient.
5. Restore weights, complete AdamW8bit optimizer, scheduler, RNG, sampler and counters from the destination in an isolated qualified environment; test subsequent updates where exact recovery is required. Reverify tokenizer/dataset identities.
6. Validate any approved path-compatibility strategy and offline access. Do not edit old manifests automatically. Define restoration to canonical paths if destination becomes unavailable and verify independent durable backup.
7. Obtain **separate explicit local-removal authorization**, recheck dependencies/extents/active processes and report actual free-space gains. Until then all originals remain intact. No copy or removal step ran here.

## G. Current capacity and alternatives

Final GPU-plan snapshot: **99.469 GiB free**, **180 GiB required**, **80.531 GiB short**. Actual capacity fluctuated during the session; resource gates must run again at launch. Verified native footprint remains 5,636,070,993 bytes, FP32 model 3,737,425,920 bytes and actual AdamW8bit optimizer 1,898,362,880 bytes. Full hash and complete optimizer evidence were reverified without constructing the GPU model.

Qualification peak plan: two native plus two optimized states, one complete staging allowance and 1 GiB logs = approximately **27.25 GiB**. The threshold still remains 180 GiB, not that smaller fixture footprint. New validation artifacts occupied 239,648,768 bytes (about 228.55 MiB), below the task's 512 MiB allowance; no historical free-space gain is assumed.

| Strategy | Capacity implication | Integrity / complexity / performance |
|---|---|---|
| A: preserve all local assets | At least 80.531 GiB additional actual free capacity on the configured volume, plus fluctuation margin | Simplest dependency continuity; no archive retrieval delays; adding a separate drive does not increase `/home` free space without separately approved path/volume planning |
| B: verified external archival | 82.18 GiB gross conditional set would project about 181.65 GiB free **only if all allocated bytes become free** | Too little margin for a reliable launch claim; source-copy/restore/independent-backup and local-removal approvals required. Historical resume needs archive access; tensor/model quality unchanged when verified. No actual gain yet |
| C: dedicated storage | Plan roughly 200 GiB qualification/run operating headroom plus ≈58 GiB per protected six-state experiment; five such groups plus existing 175 GiB history is around 665 GiB before extra datasets/backups | A ≥1 TiB usable artifact budget, preferably more for growth and independent backup, reduces repeated shortages. This is workload sizing, not a hardware purchase recommendation or authorized relocation. Training I/O latency and path compatibility must be qualified |

Costs are not priced: no product selection/purchase or external research is part of this local engineering task. Strategy B incurs validation and access costs; strategy C requires independent storage/path administration authorization. No destructive strategy was selected or executed.

## H. GPU harness readiness

The prepared harness was inspected rather than replaced. It loads native state, publishes initial/final component state, checks byte-identical recovery including actual optimizer tensors, scheduler, RNG, sampler context/counters and controlled synthetic cursor, and compares continuous/resumed CUDA updates. It measures native/component save and GPU restore, update durations, GPU peak allocation/reservation and now host high-water RSS. Injected unavailable-latest fallback is explicit and auditable.

Actual kernels/AdamW8bit determinism remain untested. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms, disabled TF32 and unchanged optimizer/configuration are required. Unsupported deterministic operations or different bytes must fail rather than be accepted approximately. The controlled synthetic update harness is not an original dataset-session replay or full-model continuous-reclamation qualification.

Final observed resources: RTX 4080 free 12,558 MiB; host available RAM 15.560 GiB; swap used 5.859 GiB. GPU-memory estimates remain two model copies + actual optimizer + 2.5 GiB workspace, checked live. RAM fails ≥16 GiB, storage fails ≥180 GiB, and no separate approved job exists. Desktop/GPU processes were not terminated. The full-model output namespace does not exist.

Read-only plan command:

```bash
.venv/bin/python scripts/checkpoint_gpu_qualification.py --checkpoint /absolute/path/to/verified/native/checkpoint.pt --output outputs/codexa-934m-v31-gpu-qualification --updates 8 --sequence-length 2048
```

Only after all gates pass and separate approval is issued:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/checkpoint_gpu_qualification.py --checkpoint /absolute/path/to/verified/native/checkpoint.pt --output outputs/codexa-934m-v31-gpu-qualification --updates 8 --sequence-length 2048 --execute --approval-file /absolute/path/to/approved-bounded-job.json
```

Approval format/fields remain `codexa-gpu-qualification-approval-v1`, approved=true, exact output/updates/context and input SHA. Output must be fresh. Run is bounded to 2–8 updates and 900 seconds, reserves finalization at boundaries, automatically opens visible Kitty progress and respects SIGINT/SIGTERM. Interruptions preserve verified published states; unsaved work is not promised recoverable. GPU approval does not authorize real-store reclamation. Actual full-model failure recovery, latency ≤10%, update throughput ≤5% regression and realistic continuous operation remain mandatory before production promotion.

## I. Tests and evidence

Final changed-owner installed rerun: **242 passed, 5 skipped, 10 warnings**, zero failures, 40.30 s. Other current suites: Inference 1, Data 19, Tokenizer 6, Memory 5 passed; Specialist 34 passed/5 skipped using unchanged source via explicit PYTHONPATH. A subsequent Training byte-edge augmentation passed **102 tests, two warnings**, including signed-zero/subnormal byte preservation and nonfinite-save rejection with intact prior recovery. Latest aggregate **308 passed, 10 skipped, zero final failures** across repository invocations; overlapping integration/component tests are not unique coverage. Optional sklearn dependency skips remain; no dependency was installed. Existing pytest return-value warnings remain documented.

An initial edge-test attempt expected NaN checkpoint tensors to save and failed the existing nonfinite guard (101 passed, one failed). The test expectation was corrected, not the guard; original failure evidence remains.

New tests verify fresh-cache invalidation on in-place status corruption, traversal symlink refusal and byte-level GPU comparison. Existing tests cover exact stochastic CPU subsequent updates, optimizer/RNG/scheduler/sampler/counters, cross-run shared objects, active readers, stale plans, corrupt indexes/components, published-state errors, interrupted unlink/fsync/status intent replay, actual process exit, atomic publication and native compatibility. No durability barrier changed, and all relevant prior fault tests still pass.

Two development benchmark invocations reached a reporting bug: empty cProfile stats were passed to pstats in no-profile mode. One earlier legacy invocation also passed unsupported best-validation flags. Those logs/fixtures remain preserved; the wrapper was corrected and final exclusive runs passed. One restore prototype included legacy resolve hashing inside timed load (double verification); `restore.json` is superseded by `restore-corrected.json`. No result from failed/superseded runs is presented as an acceptance score.

Reproducible commands, from integration root:

```bash
.venv/bin/python -m pytest -c /dev/null -p no:cacheprovider ../LLM-Architecture/tests ../LLM-Training/tests tests --basetemp outputs/NEW-task-owned-test-directory -q
.venv/bin/python scripts/profile_checkpoint_storage_v31.py --output outputs/NEW-task-owned-validation-root/run --publications 500 --no-profile
.venv/bin/python scripts/checkpoint_storage_capacity.py --checkpoint /absolute/path/to/verified/native/checkpoint.pt
.venv/bin/python -m pip check
```

Create a fresh validation parent explicitly; the profile wrapper refuses an existing run directory and reserves ≤128 MiB next-fixture allowance against a 512 MiB parent cap. It never runs against an existing project output as disposable. `git diff --check`, CLI help and installed-source byte checks passed. No commit/PR/push occurred.

## J. Historical preservation

Final comparison of 1,575 protected historical files found **zero** device/inode/size/mtime/ctime changes and no missing files. Access timestamps are excluded because read-only inspection can update them. No historical checkpoint, dataset, export, recovery metadata or experiment configuration was copied, moved, migrated, overwritten, renamed, archived or deleted. Reclamation and failure injections occurred only in task-created disposable fixture namespaces. Source/wheels/new reports and authorized session logs changed as intended.

## K. Production readiness decision

**REQUIRES ADDITIONAL VALIDATION**

V3.1 improves measured checkpoint-boundary CPU performance, proves six-state retention through 500 publications, preserves exact CPU recovery and meets the restore gate. It still performs history-dependent fresh audit accounting and is substantially slower than legacy publication on small tensors. Full 934M CUDA continuation/save/load/throughput and representative long-run qualification remain unexecuted. GPU execution is **not authorized or ready**: storage and RAM gates fail. Legacy stays the default; real reclamation remains disabled; historical assets remain intact.
