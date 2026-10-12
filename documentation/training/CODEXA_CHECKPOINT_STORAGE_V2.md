# Codexa checkpoint storage v2 — bounded retention and performance

Date: 10 October 2026. Status: **REQUIRES ADDITIONAL VALIDATION — NOT PRODUCTION READY**.

The existing `codexa-component-checkpoint-v1` backend now supports opt-in bounded
stores, dependency planning, mapped-reader leases and faster verified restores.
There is no independent checkpoint format, migration or garbage-collection executor.
**Legacy remains the production default.**

## 1. Implemented behavior

### Fixed admission limits, without deletion

A fresh optimized store can declare `RetentionPolicy` with:

- `max_snapshots`: default six distinct manifests, shared across its experiment group.
- `max_checkpoint_bytes`: declared complete native-state serialization bound.
- `max_bytes`: hard accounted-byte quota; default
  `(max_snapshots + 3) * max_checkpoint_bytes + 8 MiB`.

The byte bound includes model, optimizer and complete recovery metadata. Accounting
includes registered run indexes, immutable objects, orphaned writes, staging files
and directories. It deduplicates identical device/inode identities. Btrfs reflink
sharing is not discounted for admission, so this accounting is conservative rather
than a claim of exact exclusive extents.

Each admitted active run reserves one final manifest slot and two checkpoint byte
bounds for final publication/staging. Reservations are persistent and reused on
resume. They are released only after a successful final index publication.
Interrupted/uncertain reservations remain protected; they are not expired automatically.

Training checks capacity before the next optimizer update. If a peer consumes the
remaining ordinary-save capacity while an update is running, the completed update
is finalized through the reserved capacity. The stop reason is
`RETENTION_LIMIT_REACHED`. No incomplete accumulation is committed, checkpoint
cadence is not changed, and elapsed-budget semantics are not reset.

The generic and chat trainers support the policy. Chat startup can also finalize
safely when its baseline publication exhausts ordinary capacity.

**This is bounded admission, not continuous rotation.** Without deletion, a six-state
store eventually stops accepting new states. It cannot continue indefinitely like a
controller that overwrites unreferenced slots. An exhausted store remains loadable;
more training requires a separately reviewed new phase/store or future separately
authorized reclamation. No automatic continuation into a new namespace occurs.

### Retention planning and references

A new group run registry identifies all participating recovery indexes. Planning
validates manifests, component hashes, role references and ownership, then computes
snapshot-to-object reference counts.

Latest, previous, best behavioral, best validation, final, milestone and boundary
references are protected. Active reader leases add protected roots. Remaining capacity
is filled with the newest other states. Protected roots exceeding the selected limit
produce a blocked plan, never silent eviction. Missing/corrupt dependencies or unknown
owners block reclamation recommendations and conservatively protect known snapshots.

Logical aliases do not create another tensor payload or snapshot. Bounded milestone
and boundary aliases must identify the referenced optimizer step; labels cannot grow
without limit independently of the snapshot cap. Run IDs/names are length-bounded.

`plan_retention()` and the pure `simulate_retention()` return retained states,
hypothetical retirement candidates, shared-object reference counts, uncertainty,
reservations and byte accounting. Their output explicitly has:
`executable: false`, `garbage_collection_enabled: false` and no reclamation authority.
Unknown orphan objects are reported, not deleted.

### Compatibility and crash behavior

Snapshot/object format remains `codexa-component-checkpoint-v1`. Fresh bounded
stores add a separate `codexa-retention-policy-v1` document, cryptographically bound
by the immutable store marker. Existing append-only stores cannot be adopted.
Historical stores and manifests are not edited.

Object serialization is constrained by the byte quota and remaining component-size
bound. Nonfinite tensors, cross-component aliases, mid-accumulation state, identity/
scheduler drift and over-budget writes fail before a recovery index can advertise
an incomplete state.

The existing synchronized staging, immutable publication and atomic index replacement
remain in use. Failed saves preserve the previous published recovery index. Failed
objects/reservations may consume quota and cause an earlier safe stop; they are not
assumed free space. Physical power-loss testing has not been performed.

`OptimizedRecoveryStore.load(allow_previous=True)` explicitly warns and records
selection of previous when latest cannot be verified. It does not rewrite the index.
Expected configuration/dataset checks still apply during restoration. Incompatible
configuration is not disguised as a corruption fallback.

## 2. Restore performance and reader lifetime

The original report recorded 10.68–16.78% restore regression. Read-only profiling
showed SHA256 dominating, with smaller costs from repeated path resolution,
three deserializations and metadata traversal.

Changes:

- Reuse a hashing buffer through `hashlib.file_digest`.
- Validate shared path ancestry once within each operation, retaining symlink and
  store-escape rejection.
- Parse/verify the manifest once per load and deserialize each needed component once.
- Use `store.load()` for recovery instead of full `resolve()` verification followed
  by another complete read.
- Cache verified objects only within the current save/read transaction.
  Every independent load rehashes required component contents.
- Skip scalar-only metadata branches when attaching storage leases.

Fresh bounded snapshots have reader-lock sidecars. Returned dictionaries and
underlying mapped tensor storages hold shared locks through weak-reference finalizers.
A detached tensor retaining its storage keeps the lease after its parent payload is
discarded. `open_components()` provides a managed context; escaped mapped storage
continues to retain its lease. Planner probes are nonblocking and protect locked
snapshots. No lease files are created when reading historical append-only stores.

This storage-lifetime mechanism is tested on the current PyTorch runtime. It needs
qualification again after a runtime change. There is no deletion executor, so even
an uncertain or unmanaged historical reader cannot lose files through this system.
External programs that modify immutable files are outside the protocol; no persistent
tensor-verification cache is used to trust their contents.

### Measured nonzero fixtures

Fixtures contain FP32 model tensors and Adam-like optimizer moments: 12 MiB and
96 MiB total. Five warm-ups and 50 timed loads per engine, rotating/reversing order,
two CPU threads, complete integrity checks. The v1 baseline is a source copy captured
before modifications and reads the same component fixture. No zeros-only compression
benchmark or model quantization is involved.

| Fixture / engine | Median restore ms | P95 ms | Median CPU ms | Relative to legacy |
|---|---:|---:|---:|---:|
| 12 MiB legacy | 6.369 | 6.467 | 6.353 | baseline |
| 12 MiB v1 baseline | 6.798 | 6.938 | 6.783 | +6.73% |
| 12 MiB revised direct | 6.660 | 6.821 | 6.647 | +4.58% |
| 12 MiB revised recovery-store | 6.758 | 6.867 | 6.741 | **+6.12%** |
| 96 MiB legacy | 48.227 | 49.332 | 48.077 | baseline |
| 96 MiB v1 baseline | 48.987 | 50.067 | 48.822 | +1.58% |
| 96 MiB revised direct | 47.982 | 49.190 | 47.806 | -0.51% |
| 96 MiB revised recovery-store | 48.072 | 49.265 | 47.926 | **-0.32%** |

Both direct and store-load medians pass the <=10% regression gate at both sizes.
The first v2 benchmark also passed: 12 MiB +7.54% direct/+8.61% store;
96 MiB -0.14% direct/+0.14% store. Both reports are retained.

The contemporaneous v1 baseline did not reproduce the earlier worst regression.
Do not interpret comparisons across sessions as a controlled 16.78→6.12% speedup.
The matched fixture comparison and final threshold result are the relevant evidence.

These are warm-cache measurements. They do not prove cold-disk or full-model latency.
The benchmark did not establish a new process RSS high-water mark during measured
loads; that is not an isolated memory-overhead comparison.

Btrfs measurements for one refined fixture set:

| Fixture directory, both legacy and component payloads | Total/exclusive extent bytes | Shared |
|---|---:|---:|
| 12 MiB | 25,210,880 | 0 |
| 96 MiB | 201,371,648 | 0 |

These measure actual newly written fixtures. Retention simulations did not reclaim
any of those extents. Two benchmark sets, tests, build logs and small wheels occupied
approximately 454 MB allocated, below the 512 MiB v2 validation ceiling.

## 3. Capacity comparisons — projections, not recovered space

Common illustrative schedule: initial checkpoint, one every ten minutes, final
checkpoint; final is conservatively counted separately. Each state is assumed to
require the historical approximately 5.249 GiB full payload, without reuse credit.
Thus there are 14, 32 and 62 publications for two, five and ten hours.

| Planned duration | Legacy six-slot maximum retained GiB | Append-only upper payload GiB | Bounded six-state upper payload GiB | No-GC bounded outcome |
|---|---:|---:|---:|---|
| 2 hours | 31.494 | 73.486 | 31.494 | Stops at cap |
| 5 hours | 31.494 | 167.968 | 31.494 | Stops at cap |
| 10 hours | 31.494 | 325.438 | 31.494 | Stops at cap |

Legacy six slots can rotate unreferenced state; its current role controller does not
provide all the new independent behavioral/validation/milestone protections.
The figures compare capacity bounds, not equivalent preservation policies.

For one active bounded run with one startup state, four ordinary ten-minute saves
fill five slots; the reserved final uses the sixth, so it stops around 40 minutes.
The chat controller's extra baseline snapshot can make this around 30 minutes.
Shared active runs, explicit pins, failed writes or a tighter byte limit may stop sooner.
It is not claimed that six no-GC slots complete any of these longer schedules.

At the historical payload size, the default nine-payload-plus-8-MiB byte quota is
approximately 47.249 GiB. The six-state retained-payload bound is 31.494 GiB;
the larger hard quota covers staging, metadata and failed/orphan writes. Actual
component reuse can reduce use, but capacity admission assumes no unverified savings.

Hypothetical future authorized reclamation could keep six selected snapshots for
longer runs, provided all protected roots and readers fit. That would be approximately
31.494 GiB retained plus staging in the conservative example. This is a dry-run
projection, not an implemented continuous-retention result. Too many essential pins
block that projection. No GC implementation or authorization exists.

The trainer's actual planning is more conservative than the historical 8-bit payload:
934M FP32 model bytes plus FP32 Adam moments, even when an 8-bit optimizer is selected.

- Complete checkpoint bound: 11,214,374,912 bytes, approximately 10.444 GiB.
- Default six-state group quota: 100,937,762,816 bytes, **94.006 GiB**.
- Required free space: 208,311,945,216 bytes, **194.006 GiB**.
- Formula remains `max(180 GiB, 100 GiB + hard quota)`; no reserve was lowered.

Shared run groups use one total cap/quota, not six states per run. An investigation
requiring many independent initial/final/best states must explicitly budget a larger
new group. The engine does not increase limits automatically.

### Full-model GPU qualification

Current free space is approximately **105.31 GiB**:

- Minimum 180 GiB gate shortfall: approximately **74.69 GiB**.
- Conservative bounded 934M plan shortfall: approximately **88.70 GiB**.

A proposed bounded qualification sequence needs one new native fixture and two
optimized generations, loaded/tested sequentially. Using the historical size,
retained fixture data is about **15.747 GiB**. A full staging allowance plus 1 GiB
for logs/metadata gives approximately **21.996 GiB peak**. This could fit inside
the existing 80 GiB working reserve once the launch floor and fresh resource checks
pass; it is not permission to run below that floor.

Available GPU memory was approximately 11 GiB and host available RAM approximately
20 GiB at preflight snapshots; swap was substantially used. These values do not prove
the intended training shape fits with desktop headroom. Do not stop unrelated processes.

**No full-size GPU benchmark started.** Separate explicit authorization and sufficient
capacity are still required for full-model CUDA/bitsandbytes continuation, representative
save/load latency, peak allocation/reservation and steady-state training/GPU throughput.
No meaningful GPU throughput-regression result is claimed.

## 4. Correctness, regression and preservation evidence

Final installed-wheel suites: **231 passed, 5 skipped, 10 warnings**.
The five specialist tests require sklearn, absent from the generative profile.
The ten warnings are existing pytest return-value warnings, not hidden failures.

Coverage includes:

- Bitwise model/optimizer preservation and subsequent stochastic CPU optimizer updates.
- Tied storage aliases, scheduler, scaler dictionary, RNG, sampler/cursor and budget.
- Six-state admission, final slots, two registered writers and byte reservation.
- Shared-component reference counts; excess protected pins and unknown dependencies.
- Detached mapped tensors retaining reader leases; release after storage lifetime ends.
- Explicit previous fallback without index rewrite, corruption and missing metadata.
- Interrupted bounded serialization, unchanged prior recovery index and retained reservation.
- Store adoption/policy drift rejection, registered-index accounting and bounded aliases.
- A tiny CPU loop that completes exactly two further optimizer updates, stops before
  the next update, and publishes its exact counters/state in the reserved fourth slot.
- Existing architecture, tokenizer, data, inference and integration regression behavior.

Early failed test runs are retained: one test expected six manifests after identical
final payloads correctly reused existing snapshots; one early-validation ordering
regression was fixed. A final concurrent fresh-store initialization race was also
fixed by waiting read-only on the first initializer’s lock; 30 simultaneous
fresh-store writer trials passed afterward. Final results are not inferred from
those failed runs.

A read-only SHA-verified historical base checkpoint load in FakeTensorMode passed:
5,636,066,645 bytes, 172 model keys, optimizer/scheduler/RNG present, step 10,000,
6.666 seconds. This confirms actual historical compatibility without GPU allocation;
it is not full numerical 934M qualification.

Changed Architecture, Training and integration wheels were rebuilt with setuptools
and installed using `--no-deps --force-reinstall`. No dependency upgrade occurred.
Installed source trees match the changed source implementations. `pip check`,
compilation, CLI help and repository whitespace checks passed.

Device/inode/size/mtime/ctime comparisons covered **1,472 historical files**, including
v1 validation artifacts: **zero changes**. Read-induced atime was excluded; this is
metadata preservation evidence rather than a full content-hash inventory. The earlier
optimization report was not rewritten. Historical checkpoints, datasets, exports
and experiments were not migrated, pruned or overwritten. No NumPy changes occurred.

No cleanup or GC ran. Existing atomic-writer staging cleanup and temporary synthetic
test fixture lifecycle remain part of the existing test infrastructure; no historical
data was deleted.

## 5. Ownership, interfaces and commands

| Owner | v2 changes |
|---|---|
| Architecture | `component_checkpoint.py`: verified reads, storage leases, quota-aware serialization/accounting |
| Training | New `checkpoint_retention.py`; optimized recovery/controller, checkpoint APIs, preflight and CLI/chat hooks |
| Training tests | New `test_checkpoint_retention.py`; existing recovery/regression suites retained |
| Integration | New benchmark module/CLI, read-only plan CLI, bounded configuration fragment, preparation plumbing |
| Documentation | This new report and appended session/progress entries; v1 report preserved |

No new inference implementation or model mathematics were introduced.
Important interfaces:

- `RecoveryStore(..., backend='optimized', retention={...})` — fresh-store opt-in.
- `prepare_update(run_id)` — admission before another optimizer update.
- `load(..., allow_previous=False)` — verified load and explicit fallback option.
- `reference(role, target_role)` — alias without a physical snapshot.
- `plan_retention()`, `simulate_retention()` — non-executable dependency plans.
- `load_checkpoint(None, recovery_store=store, ...)` — avoid duplicate recovery verification.
- `open_components(path)` — managed mapped-reader lifetime.

Example fragment for a NEW reviewed chat configuration:

```yaml
checkpoint_storage_backend: optimized
checkpoint_store_root: checkpoint-groups/new-bounded-experiment-v1
checkpoint_retention:
  max_snapshots: 6
```

Conservative byte bounds are derived automatically by trainer preflight. Explicit
bounds may increase capacity but cannot understate its FP32 optimizer estimate.
See `configs/checkpoint-storage-bounded.example.yaml`; it is not a standalone
launch configuration. No historical config was changed.

Generic CLI additions, not a training launch command:

```text
--checkpoint-storage-backend optimized
--checkpoint-max-snapshots 6
--checkpoint-store-root /absolute/new/experiment-group
--checkpoint-max-bytes <optional-explicit-byte-quota>
```

Read-only dry-run example on the new controlled fixture:

```bash
.venv/bin/python scripts/checkpoint_retention_plan.py \
  --checkpoint-directory outputs/checkpoint-storage-v2-validation/benchmark-refined/12MiB/revised
```

Reproducible regression command:

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest \
  ../LLM-Training/tests ../LLM-Architecture/tests ../LLM-Inference/tests \
  ../LLM-Data/tests ../LLM-Tokenizer/tests tests -q -rs
```

Benchmark entry point:

```bash
.venv/bin/python scripts/benchmark_checkpoint_storage_v2.py --help
```

Another full fixture benchmark would exceed the selected aggregate validation budget.
Do not rerun blindly; the two existing reports and source baseline are preserved under
`outputs/checkpoint-storage-v2-validation/`.

## 6. Acceptance decision

| Gate | Result |
|---|---|
| Predictable count/byte bounds with safe stopping | Passed bounded tests |
| Essential role protection / shared reference planning | Passed bounded tests |
| Active mapped reader protection | Passed current-runtime tests |
| Exact CPU recovery and optimizer continuation | Passed |
| Restore median <=10% regression, 12/96 MiB | Passed both benchmark sets |
| Historical files unchanged | Passed metadata checks |
| Legacy remains default / no automatic GC | Confirmed |
| Full-model CUDA/bitsandbytes continuation | Not performed — blocked |
| Full-size save/load and throughput qualification | Not performed — blocked |
| Production storage preflight | Failed current capacity |

**REQUIRES ADDITIONAL VALIDATION.** Suitable for explicit bounded fixture testing;
not approved for production adoption. Additional storage and separate GPU-test
authorization are required. Continuous reclamation/rotation remains a separate
unimplemented and unauthorized operation.
