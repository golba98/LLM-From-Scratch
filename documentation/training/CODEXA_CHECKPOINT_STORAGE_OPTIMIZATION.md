# Codexa checkpoint storage optimization — engineering report

Date: 10 October 2026. Decision: **REQUIRES ADDITIONAL VALIDATION**.
The versioned component backend is implemented and installed for optional testing.
**Legacy remains the default. No historical checkpoint was migrated, replaced or deleted.**

## A. Root-cause analysis

There are two active checkpoint controllers, not one:

- `LLM-Training/checkpointing.py::CheckpointManager` serializes a complete native
  payload, then copies it for previous, best and milestone roles. Its implementation
  uses `torch.save`, checksums, temporary files and `shutil.copyfile`.
- `LLM-Training/recovery.py::RecoveryStore` publishes complete native payloads into
  six rotating slots and updates a small recovery index. It already avoids physical
  copies per role, but repeats the full tensor serialization when only finalization,
  elapsed budget or recovery metadata changes. It can reuse unreferenced slots.

Model-only inference/SFT exports are a separate class of artifact. They are not a
substitute for complete recovery state. The older conversational-SFT CLI still writes
its existing weights-only representation; it does not gain an exact-resume claim.

The preceding audit attributed approximately 174.86 GiB to recent outputs:
92.67 GiB investigation, 62.95 GiB preparation and 19.23 GiB original two-hour run.
Those numbers are historical audit observations, not a newly repeated complete disk
inventory. A main-model full checkpoint is approximately 5.249 GiB; model tensors
approximately 3.481 GiB. Independent updated weights and optimizer moments are
necessary storage. Repeated unchanged tensors and copied logical role identities
are avoidable duplication.

On this Btrfs filesystem, apparent copies do not always occupy independent extents:
the synthetic copy-manager fixture had 75,620,352 bytes reported total extents but
25,206,784 exclusive plus 25,206,784 set-shared bytes. Its unique extent consumption
was approximately 50,413,568 bytes, not 75,620,352. No savings are claimed from
removing or converting existing files.

## B. Architecture comparison

| Approach | Storage benefit | Cost / recovery considerations | Decision |
|---|---|---|---|
| Component content addressing | Reuses exactly identical serialized model/optimizer objects; roles are small references | Hashing and serialization remain necessary; three-object validation | Implemented, opt-in |
| Metadata separation | Different budgets/RNG/cursors can share unchanged tensor objects without sharing recovery identity | Must atomically publish a complete manifest and role index | Implemented |
| Btrfs reflinks | Shares identical whole-file extents without copying bytes | Filesystem-specific; metadata changes still require a different file; immutable discipline needed | Verified on new fixture; not required by backend |
| Extra zstd compression | Modest nonzero FP32 sample savings | Adds CPU and restore work; filesystem already uses zstd:1 | Not enabled |
| Exact XOR deltas | May compress small updates | Baseline chains, corruption propagation and reconstruction latency; optimizer behavior unqualified | Not enabled |
| Per-role references | Avoids copies when best/latest/final describe the same snapshot | Referenced immutable payload must remain available | Implemented |
| Automatic retention/GC | Could bound unique-state accumulation | Deletion was not authorized | Absent; inventory only |

A new 1 MiB random FP32 fixture compressed with zstd:1 to 969,061 bytes
(1.0821×; 1.286 ms). Its artificial +0.0001 XOR delta compressed to 566,350
bytes (1.8515×; 3.636 ms); exact XOR reconstruction was asserted.
These are synthetic samples, not full-model compression predictions.
Earlier read-only 1 MiB real tensor samples yielded approximately 1.079×
model and 1.180× optimizer zstd ratios; they do not justify a new compression layer.

The new reflink fixture initially had 1,048,576 set-shared bytes and zero exclusive
bytes for two 1 MiB files. Writing 4 KiB to the new clone left the source bitwise
unchanged and added 4 KiB exclusive extents. No existing checkpoint was reflinked.

## C. Selected architecture

Format identifier: `codexa-component-checkpoint-v1`.

```text
experiment-group/
  store.json                         immutable store identity
  .writer.lock                       single-writer flock
  objects/<sha-prefix>/<sha>.pt       immutable serialized components
  snapshots/<run-id>/<sha>.ckpt.json  immutable complete snapshot manifest

run/checkpoints/recovery-index.json   atomically replaced logical role index
```

Each snapshot contains three components:

1. Model state, preserving dtype, values and intra-component storage aliases.
2. Complete optimizer state, preserving tensors, parameter groups and optimizer scalars.
3. Everything else from the native payload: scheduler, scaler, Python/NumPy/CPU/CUDA
   RNG, counters, configuration, tokenizer identity, sampler/cursor, time budget,
   initialization provenance and any additional recovery fields.

The manifest identifies every component by SHA256, size and constrained relative
path. Its own digest is bound to its filename. The run index references an immutable
manifest and the explicit experiment-group store. Entire groups remain portable
using relative references; moving one manifest alone does not make a checkpoint portable.

Components use stable file-like PyTorch serialization. Deduplication requires
identical serialized bytes, not merely matching shapes, file sizes or ZIP CRCs.
Different dictionaries/storage layouts may serialize differently despite equal tensor
values. SHA256 identity never substitutes for optimizer or recovery metadata identity.

Writes stage new files, fsync bytes, validate hashes/deserialization, publish immutable
names without replacement, synchronize parent directories, then atomically publish the
snapshot and run index. Failed writes leave the last published index valid. Orphaned
new objects may remain after failure; no cleanup is performed. Power-loss durability
still depends on filesystem/device guarantees. No software promises unsaved updates
will survive a hard kill.

One writer lock protects the entire group. Readers see either the old or new complete
index. Path escapes, symlinks, unsupported versions, incomplete manifests, corrupt
objects and identity drift cause controlled failures. Checkpoint files remain trusted
local PyTorch pickle inputs; this does not make untrusted pickle safe.

Cross-component tensor aliases are rejected because splitting them would change
alias semantics. Mid-accumulation checkpoints are rejected; gradients are not silently
discarded. Exact recovery is supported at optimizer boundaries only. The original
native format remains readable without conversion.

Roles include latest, previous, best, best_behavioral, best_validation, final,
milestone-N and boundary-N. The generic trainer's best alias means validation best.
The chat trainer records best-validation only when the saved current model was actually
evaluated; it does not label newer weights using an older historical minimum.

**Important trade-off:** all unique optimized objects/manifests are retained. Unlike
the legacy rotating slot store, this backend is not bounded to six full states.
For many distinct updates it can consume more cumulative storage than that controller.
The improvement applies where duplication genuinely exists, not to arbitrary changed
training states. Capacity planning counts the complete retained publication history.

## D. Implementation summary

| Owner | Modified / new files | Purpose |
|---|---|---|
| LLM-Architecture | `src/llm_architecture/component_checkpoint.py` (new), `checkpointing.py` | Immutable component store, manifest integrity, shared native/component reader |
| LLM-Training | `optimized_recovery.py` (new), `recovery.py`, `checkpointing.py` | Factory dispatch, role publication, backward-compatible restore |
| LLM-Training | `chat_run.py`, `cli/train.py`, `cli/train_conversational_sft.py` | Opt-in full-state writes, capacity gates, compatibility reads |
| LLM-Training | `preflight.py`, `tests/test_component_recovery.py` (new) | Capacity estimator, publication bound, exact recovery/fault tests |
| LLM-Inference | `native_chat.py`, `cli/generate.py`, `cli/export_chat_native.py`, `cli/export_codexa_lmstudio.py`, `cli/export_codexa_llama_lmstudio.py` | Shared component/native checkpoint loading; export writers unchanged |
| Integration | `chat_preparation.py`, `chat_operations.py` | New-spec backend/group selection, preflight and evaluation dispatch |
| Integration | `checkpoint_benchmark.py`, `scripts/benchmark_checkpoint_storage.py` (new) | Bounded CPU fixtures and extent measurements |
| Integration | `tests/test_chat_preparation.py`, `configs/checkpoint-storage-optimized.example.yaml` (new) | Early storage refusal and documented opt-in fragment |
| Integration | This report, `SESSION_DECISIONS.md`, `100M_PROGRESS_LOG.md` | Engineering evidence and handoff |

Other dirty/untracked files already existed from earlier work. They are not all changes
from this implementation. NumPy and unrelated component source were not changed.

New API examples, for new isolated namespaces only:

```python
from llm_training.recovery import RecoveryStore
store = RecoveryStore(new_run / "checkpoints", backend="optimized",
                      store_root=new_group / "objects")
record = store.save(full_native_payload, best=True, final=True)
manifest = store.resolve("latest")
# Existing load_checkpoint / load_model_checkpoint accept manifest or legacy .pt.
```

Generic training CLI opt-in arguments:

```text
--checkpoint-storage-backend optimized
--checkpoint-store-root /absolute/new/experiment-group
```

A new integration chat config may include:

```yaml
checkpoint_storage_backend: optimized
checkpoint_store_root: checkpoint-groups/new-reviewed-experiment-v1
```

The checked-in example is a fragment, not a runnable production config.
No historical specifications were edited. Default is legacy everywhere.
No training command was launched.

Changed Architecture, Training, Inference and integration wheels were rebuilt with
the existing setuptools backend into new isolated validation directories and installed
using `pip install --no-deps --force-reinstall`. Versions/dependencies were not upgraded.
All four installed Python source trees match their corresponding source files byte
for byte; paths are recorded in `installed-source-verification.json`.

## E. Storage benchmarks

Synthetic fixture: 12 MiB of nonzero FP32 tensors, including Adam-like optimizer
moments. Six publications contain three changed-tensor states interleaved with
three unchanged-tensor finalization states. Process-isolated measurements, two CPU
threads. Save timing includes publication and resolving/verifying the current snapshot.
Loads verify integrity; warm page-cache results are not cold-device benchmarks.

| Metric | Legacy rotating store | Optimized | Difference |
|---|---:|---:|---:|
| Tensor payload per snapshot | 12,582,912 B | 12,582,912 B | No precision reduction |
| Final inode-accounted allocated bytes | 50,450,432 | 37,904,384 | 24.87% lower for this schedule |
| Btrfs unique extents, fixture directory | 50,413,568 B | 37,871,616 B | 24.88% lower |
| First unchanged publication growth | 12,611,584 B | 20,480 B | **99.8376% lower** |
| Changed save median | 72.547 ms | 72.491 ms | 0.08% faster in this sample |
| Six-load median | 6.611 ms | 7.317 ms | **10.68% slower** |
| Median save CPU time, all six saves | 76.650 ms | 53.210 ms | Fixture-specific |
| Peak process RSS | 601,780 KiB | 589,632 KiB | 12,148 KiB lower |
| RSS growth during fixture | 78,504 KiB | 66,340 KiB | 12,164 KiB lower |
| Reload tensor/metadata checks | Exact | Exact | Passed |
| Further CPU AdamW/dropout updates | Exact | Exact | Passed |

A 30-repetition alternating read-only load benchmark on these new fixtures measured
6.593 ms legacy versus 7.699 ms optimized: **16.78% slower**. This confirms the
<=10% small-fixture load gate has not passed. No result is concealed.

The first implementation benchmark had a 22.32% changed-save regression. It was
improved by transaction-local verified-object caching and reading only prior recovery
metadata rather than deserializing prior large tensor components. Initial and refined
results are both preserved. The first Btrfs command used an unsupported flag; the
refined benchmark used the corrected command. No cross-operation integrity cache
weakens subsequent load verification.

All validation outputs currently occupy approximately 331,091,968 allocated bytes
(315.75 MiB), below the 512 MiB aggregate ceiling. No multi-gigabyte test copies exist.

Evidence directory:
`outputs/checkpoint-storage-validation-v1/`.
Key files: `benchmark/report.json`, `benchmark-refined/report.json`,
`load-repetitions.json`, `alternatives.json`, `historical-load.json`,
`acceptance-tests.log`, `historical-preservation.json`.

## F. Training and inference impact

No model architecture, tokenizer, dataset, tensor precision, optimizer mathematics,
learning rate or training objective changed. After loading, inference uses the same
model and ordinary tensor state; there is no runtime storage wrapper.

CPU continuation was verified bitwise through three further stochastic AdamW updates,
including model/optimizer state, losses and RNG state. Sampler/cursor, budget,
configuration and scaler-state dictionary round trips were checked. CUDA RNG states
are captured/restored by the existing mechanism; this is not a demonstrated
full-model CUDA/bitsandbytes continuation guarantee.

No full-size save/load latency, additional GPU serialization allocation, steady-state
training throughput or GPU-utilization equivalence has been measured for the new
backend. Serialization/finite checks may synchronize CUDA tensors; full-size overhead
requires measurement. Hashing does not disappear on unchanged saves: tensors are still
serialized into staging so identity can be proven. Savings describe retained storage,
not necessarily save speed.

A historical 934M base checkpoint (5,636,066,645 bytes) passed the common legacy
reader with SHA sidecar verification and FakeTensorMode metadata loading in 10.173 s.
It contained 172 model keys, optimizer, scheduler and RNG state; optimizer step 10,000.
This was read-only, without GPU model allocation. It does not replace a full numerical
934M optimized restore/continuation test.

## G. Future capacity planning

The production minimum remains **100 GiB safe headroom plus 80 GiB working reserve**.
Current measured free space was 113,533,603,840 bytes (**105.74 GiB**), approximately
74.26 GiB below the 180 GiB floor. No launch is authorized by these results.

Planning code computes retained model/optimizer objects, metadata allowance,
temporary write space and required reserve. It assumes FP32 model bytes and conservative
FP32 Adam moments even if an 8-bit optimizer is selected; this avoids underbudgeting an
unqualified optimizer representation. Publication bounds cover checkpoint intervals,
every evaluation improving enough to save, and four boundary publications. Coincident
intervals are counted separately conservatively. Resume rechecks available capacity.
No deletion or unverified cross-run reuse is credited.

The following separate projections use the historical ~5.249 GiB full-state size
and a proposed ten-minute checkpoint/evaluation schedule, with two boundary states.
Best roles reference existing publications. They are **planning examples, not the
current step-based run schedule or measured optimized full-size allocation**.
Compression/shared extents are not assumed.

| Future workload | Unique full states | Retained payload GiB | Peak with one full temporary + 8 MiB | Required free GiB |
|---|---:|---:|---:|---:|
| Two hours | 14 | 73.486 | 78.743 | 180.000 |
| Five hours | 32 | 167.968 | 173.225 | 273.225 |
| Ten hours | 62 | 325.438 | 330.695 | 430.695 |
| Five independent two-hour experiments | 70 | 367.430 | 372.687 | 472.687 |

These projections conservatively give no credit for unchanged finalization tensors
or shared initializers. Real immutable-component retention must be planned from the
chosen optimizer, actual save schedule and maximum steps. Logical aggregate sizes
are not guaranteed Btrfs exclusive allocations. Legacy six-slot retention can consume
less than these totals for long runs; the optimized backend should not replace it
blindly without a separately authorized retention design.

Per-save guards also require safe reserve plus new/staging tensor bytes before writing.
Failures propagate; a trainer cannot report a failed save as successful. No storage
reserve was lowered to pass a gate.

## H. Test results

Final installed-wheel regression command, run from integration root:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  ../LLM-Training/tests ../LLM-Architecture/tests ../LLM-Inference/tests \
  ../LLM-Data/tests ../LLM-Tokenizer/tests tests -q -rs
```

Result: **218 passed, 5 skipped, 10 warnings, 7.77 s**.
Five specialist tests skipped because sklearn is absent from this generative profile;
no dependency was installed to bypass that separation. Ten existing pytest warnings
concern test functions returning values. Earlier failed runs are preserved: one CLI
import-scoping bug was corrected; one rejection-message assertion was corrected
because the symlink was correctly rejected as a store escape.

Focused coverage includes:

- Native/component round trips; tied tensor aliases and exact optimizer/scaler/RNG state.
- Equal further CPU stochastic training updates; sampler/cursor/budget fields retained.
- Metadata-only tensor reuse, experiment-group sharing and distinct optimizer states.
- Latest/previous/best/final/milestone references and publication rotation.
- Object, manifest and index failure injection; partial tensor-stream serialization.
- Missing/corrupt components and manifests; malformed JSON object structure.
- Path/symlink rejection, nonfinite state, config drift and legacy migration refusal.
- Concurrent readers/publication and two concurrent group writers.
- Capacity refusal before heavy asset reads, accumulation and cross-component alias rejection.
- Historical compatibility without conversion.

`compileall`, four repository `git diff --check`, CLI help and
`pip check` passed; pip reported no broken requirements.
Data/tokenizer suites separately passed 25 tests and are included in the final total.

Preservation check compared device/inode/size/mtime/ctime for **1,331 historical files**:
**zero changes**. Atime was excluded because ordinary reads may update it. This is
a strong metadata preservation check, not a full content-hash inventory. Historical
datasets, checkpoints, exports and prior experiment outputs were not intentionally
written, migrated or deleted. New fixture modifications are confined to this validation
namespace. No NumPy source or assets were altered.

Not performed: full-size optimized write/restore, actual full-model CUDA/bitsandbytes
deterministic continuation, cold-device latency, production throughput, power-loss
simulation on the physical device, or historical checkpoint conversion. These are
qualification limits, not passing tests.

## I. Deployment recommendation

**REQUIRES ADDITIONAL VALIDATION.** The implementation is available for bounded,
explicit optional testing. Keep the original backend selected for production.

Acceptance thresholds established for this implementation:

| Gate | Threshold | Current evidence |
|---|---|---|
| Unchanged save growth | >=95% reduction | Pass: 99.8376% |
| State preservation / CPU continuation | Bitwise exact | Pass on bounded fixtures |
| Interrupted publication | Last verified checkpoint remains usable | Pass fault tests |
| Changed-state save latency | <=10% regression | Pass refined fixture; first version failed |
| Restore latency | <=10% regression | **Fail: +10.68%; longer sample +16.78%** |
| Additional peak memory | <=128 MiB | Pass synthetic RSS comparison; full-size unqualified |
| 934M GPU throughput/utilization/recovery | No meaningful degradation, exact state | Not measured |
| Production capacity | >=180 GiB plus projected retention where greater | **Fail current disk capacity** |

Next engineering work: profile small-manifest loader overhead, then obtain suitable
headroom and separate authorization for a bounded full-size checkpoint benchmark.
Verify actual optimizer representation, full-size save/load memory and latency,
CUDA continuation and throughput before recommending deployment. Further retention
or historical migration would require a separate authorization.

For an additional small benchmark, choose a fresh directory and account for the
existing aggregate validation budget; do not rerun into an existing directory:

```bash
.venv/bin/python scripts/benchmark_checkpoint_storage.py --help
# New writes require a newly reviewed bounded output path:
# .venv/bin/python scripts/benchmark_checkpoint_storage.py --output <fresh-small-output>
```

No garbage collection, cleanup, production training, export promotion or publication
was performed. Existing checkpoint storage remains intact.
