"""Readiness handoff generated from measured artifacts, never invented scores."""

import json
from datetime import datetime
from pathlib import Path
import statistics


def write_report(root: Path, directory: Path, manifest: dict) -> Path:
    """Write the requested A–L report with measurements and explicit failed gates."""
    def read(name):
        path = directory / name
        return json.loads(path.read_text()) if path.exists() else None
    audit = read('data-audit.json')
    candidates = read('deterministic-candidate-comparison.json') or read('candidate-comparison.json')
    benchmark = read('benchmark/report.json')
    recovery = read('recovery/report.json')
    deterministic = read('recovery-deterministic/report.json')
    selected_benchmark = read('benchmark-deterministic/report.json')
    pilots = read('pilot-comparison.json') or []
    retention = read('retention-diagnostic.json')
    overlap = read('development-overlap-review.json')
    checks = read('automated-checks.json')
    preflight = read('preflight.json')
    spec = manifest['spec']
    chosen = next(r for r in candidates if r['checkpoint'] == spec['initialization']['path'])
    def percent(value):
        return f'{value * 100:.2f}%'
    quality_table = '\n'.join('| ' + ' | '.join([
        Path(r['checkpoint']).parent.name, percent(r['metrics']['macro_task_accuracy']),
        percent(r['metrics']['category_pass_rates']['factual']), percent(r['metrics']['category_pass_rates']['multi_turn']),
        percent(r['metrics']['END_rate']), percent(r['metrics']['length_limit_rate']),
        percent(r['metrics']['severe_degeneration_rate'])]) + ' |' for r in candidates)
    data_table = '\n'.join(f"| {name} | {s['totals']['examples']:,} | {s['totals']['context_tokens']:,} | "
        f"{s['totals']['assistant_tokens']:,} | {s['totals']['historically_exposed']:,} | {len(s['possibly_clean_indices'])} |"
        for name, s in audit['splits'].items())
    performance_table = '\n'.join(f"| {r['name']} | {r['assistant_tokens_per_second']:.0f} | "
        f"{r['context_tokens_per_second']:.0f} | {r['steady_step_seconds']:.3f} | {r['peak_reserved_bytes']/1024**3:.2f} |"
        for r in benchmark['results'] if not r.get('skipped') and not r.get('failed'))
    deterministic_performance = '\n'.join(f"| {r['name']} | {r['assistant_tokens_per_second']:.0f} | "
        f"{r['context_tokens_per_second']:.0f} | {r['steady_step_seconds']:.3f} | {r['peak_reserved_bytes']/1024**3:.2f} |"
        for r in (selected_benchmark or {'results': []})['results'] if not r.get('skipped') and not r.get('failed'))
    pilot_table = '\n'.join(f"| {r['training_report']['run_name']} {'resumed' if r['training_report'].get('exact_resume') else ''} | "
        f"{r['training_report'].get('training_state', {}).get('optimizer_step', '?')} | "
        f"{r['training_report'].get('stop_reason')} | {percent(r['metrics']['macro_task_accuracy'])} | "
        f"{percent(r['metrics']['END_rate'])} | {percent(r['metrics']['severe_degeneration_rate'])} |" for r in pilots)
    git_table = '\n'.join(f"| {name} | `{r['commit']}` | {'dirty' if r['dirty'] else 'clean'} | "
        f"{'MATCH' if not r['source_import_differences'] else 'DIFFERENT/UNAVAILABLE'} |"
        for name, r in manifest['environment']['repositories'].items())
    gate_table = '\n'.join(f"| {name} | {'PASS' if value else 'FAIL / UNPROVEN'} |" for name, value in manifest['mandatory_gates'].items())
    metric_paths = {Path(r['checkpoint']).parents[2] / 'train_metrics.jsonl' for r in pilots}
    records = [json.loads(line) for path in metric_paths if path.exists() for line in path.read_text().splitlines()]
    records = [r for r in records if r['optimizer_step'] > 2]
    measured_step = statistics.median(r['step_time_seconds'] for r in records) if records else 0
    assistant_rate = sum(r['tokens_processed'] for r in records) / sum(r['step_time_seconds'] for r in records) if records else 0
    context_rate = sum(r['context_tokens_processed'] for r in records) / sum(r['step_time_seconds'] for r in records) if records else 0
    projection = min(audit['splits']['train']['totals']['assistant_tokens'],
                     assistant_rate * (36000 - spec['finalization_reserve_seconds']) * .90)
    test_results = '\n\n'.join(f"`{' '.join(r['command'])}`\n\nExit {r['returncode']}; {r['seconds']:.2f}s. "
        f"Evidence: `{r['log']}`.\n\n```text\n{Path(r['log']).read_text()[-2300:]}\n```" for r in checks['results'])
    examples = '\n\n'.join(f"**{row['id']} — {row['category']}**\n\nPrompt: {row['prompt']}\n\n"
        f"```text\n{row['response'][:650]}\n```\nStop: {row['termination']}; task pass: {row['metrics']['task_pass']}."
        for row in chosen['outputs'][:8])
    paired = []
    if pilots:
        original = {r['id']: r for r in chosen['outputs']}
        updated = pilots[-1]['outputs']
        predicates = [
            ('Literal check improved', lambda a, b: not a['metrics']['task_pass'] and b['metrics']['task_pass']),
            ('Literal check regressed', lambda a, b: a['metrics']['task_pass'] and not b['metrics']['task_pass']),
            ('Failure unchanged', lambda a, b: not a['metrics']['task_pass'] and not b['metrics']['task_pass']),
        ]
        for title, predicate in predicates:
            item = next((r for r in updated if r['category'] != 'coherence_proxy' and predicate(original[r['id']], r)), None)
            if item:
                baseline = original[item['id']]
                paired.append(f"**{title}: first qualifying case {item['id']}**\n\n{item['prompt']}\n\n"
                    f"Initialization ({baseline['termination']}):\n\n```text\n{baseline['response'][:450]}\n```\n\n"
                    f"Pilot ({item['termination']}):\n\n```text\n{item['response'][:450]}\n```")
            else:
                paired.append(f'{title}: no qualifying task-status transition observed.')
    paired_examples = '\n\n'.join(paired)
    losses = '\n'.join(f"| {Path(r['checkpoint']).parent.name} | {r['loss']:.5f} | "
        f"{r['common_conversational_validation_loss']:.5f} | {r['common_validation_tokens']:,} |"
        for r in retention['results']) if retention else 'Not measured; no retention claim.'
    failed = ', '.join(k for k, value in manifest['mandatory_gates'].items() if not value)
    path = root / 'documentation/training/CODEXA_900M_CHAT_10H_READINESS.md'
    report = f'''# CODEXA_900M_CHAT_10H_READINESS

Prepared {datetime.now().strftime('%d %B %Y')} from current local execution. Decision: **{manifest['readiness']}**.

## A. Executive Summary

Implemented a modular conversational successor, lazy audited chat loading, deterministic full-pool sampling,
assistant-token-weighted loss, transactional native recovery checkpoints, original-deadline accounting,
an independent production watchdog, signal handling, a detached visible Kitty viewer, structured logs,
reproducible development/protected acceptance evaluation, gated operator commands and this generated handoff.
No ten-hour production job or model promotion occurred. All five existing candidates fail minimum instruction
and arithmetic compliance. Lower loss during the corrective pilot did not establish a better assistant.

Failed/unproven mandatory gates: **{failed}**. Preparation is implemented and tested; approval is not being
requested for a blocked launch. Resolve these gates before a separate production approval.

## B. Architecture and Initialization

934,356,480 trainable parameters; decoder-only, 24 layers, hidden 1,536, 24 heads × 64, FFN 6,144,
context 2,048, vocabulary 16,384, learned positional embeddings, RMSNorm, SwiGLU, causal SDPA, tied token/output
embeddings. Model dimensions, positional scheme, state keys and tokenizer are unchanged. Strict restores,
tied Parameter identity and finite forward/loss checks were performed on the main model.

**No production initialization is accepted.** Provisional benchmark/corrective weights initialization:
`{spec['initialization']['path']}`.

SHA256: `{spec['initialization']['sha256']}`.
Repair ranked highest on the fixed literal task aggregate, despite worse repetition than balanced/v2.
Its weights-only file starts a new optimizer/scheduler phase. This is not an exact historical optimizer resume.
Base and Stage 2 have full native state; balanced/v2/repair do not. All five sidecar digests were verified and
actual strict loads performed. Missing embedded tokenizer hashes in older weights-only files are disclosed:
compatibility is supported by architecture, verified tokenizer identity and local lineage, not an embedded binding.

Tokenizer: `{spec['tokenizer']['path']}`; SHA256 `{spec['tokenizer']['sha256']}`.
IDs pad/BOS/EOS/UNK/system/user/assistant/end = 0/1/2/3/4/5/6/7. END (7) is supervised assistant termination;
EOS (2) remains an alternate inference stop. Existing chat serialization does not insert BOS/EOS. One causal
target shift is applied by the dataset; assistant content and END are supervised, role/user/system/padding ignored.

## C. Dataset Strategy

Physical asset root remains `/home/k9-vortex/Development/3-Python/31-LLM Source`.
Dataset: `{Path(spec['data']['train']).parent}`; manifest SHA256 `{spec['data']['manifest_sha256']}`.
All 202,901 JSONL conversations were read-only audited against manifest digests, normalized content hashes,
reserialized boundaries, token lengths and shifted response masks. No retokenized corpus or duplicate store was created.

| Split | Conversations | Context tokens | Assistant tokens | Potential historical exposure | Group-filtered possibly clean |
|---|---:|---:|---:|---:|---:|
{data_table}

Training sources: OASST1 27,034 conversations / 10,250,006 assistant tokens; UltraChat 155,858 /
137,334,262. Assistant totals count repeated shared turns/prefixes across distinct paths, not unique linguistic tokens.
There are 163,889 training duplicate/root groups. No train/validation/test root, group or exact-content overlap was found.
Train SHA256 `{spec['data']['train_sha256']}`; validation SHA256 `{spec['data']['validation_sha256']}`.
Test SHA256 `{audit['splits']['test']['sha256']}`.

Use seeded no-replacement ordering across the full 182,892 training pool, with local length bucketing; natural
source proportions are the provisional choice, not a proven optimum. Balanced interleaving is supported but was
not selected through an accepted mixture experiment. At most one epoch; partial final accumulation is discarded.
No base replay is enabled. Lazy serialization, 64 MiB cache, right dynamic padding, workers 0 and pinned transfers
avoid storing the entire padded pool in host RAM. Complete-conversation truncation is rejected, empty targets rejected,
padding labels -100, causal right padding validated. Conversations are not packed together.

Validation uses 96 records from 92 groups whose entire group lacks a positive historical exposure flag.
The pilot used a fixed first 32 of those; benchmark probes used first 8. These losses are different populations.
Common 96-record validation appears separately below. The 98 similarly flagged test records in 86 groups remain
unused in training and preparation evaluation. Flags cannot prove access history; earlier SFT validation exposure
and previous Stage 2 test use prevent a claim of clean historical holdout. Only 1,759 training records remain
possibly clean after group propagation (1,772 individual flags were false).

Five exact normalized training matches concern one development case, `development-002` (Tokyo).
The archived 63-case exact-match-excluded comparison preserves the strongest provisional candidate and all failures;
it is not semantic decontamination. No exact protected prompt match was found; paraphrase exposure is unproven.
The protected suite was not generated against during preparation. Missing original base validation binary/index
was not replaced by continuation validation or these replay windows. **Proven generalization is blocked.**

## D. Final Training Configuration

Prepared configuration: `configs/chat-10h-v1.yaml`; resolved proposed specification:
`{directory / 'launch-manifest.json'}`. These values are prepared, not accepted production hyperparameters.

```json
{json.dumps(spec, indent=2)}
```

Conservative proposed peak LR 1e-5 (20× below historical base peak), cosine to 1e-6, measured-derived warmup and
step/interval caps above. Weight decay 0.1 with bias/norm exclusions and tied-parameter deduplication;
gradient clip 1.0; BF16 autocast with FP32 parameters; no AMP scaler. Microbatch 1 × accumulation 32 gives
32 conversations per complete optimizer update. Loss normalization sums assistant losses then divides the
accumulated gradients by the actual supervised token count, rather than averaging uneven microbatch losses.
Seed 42. No architecture, tokenizer, special ID or context-limit change.

The failed behavioral entry gate led to one LR corrective experiment, not an accepted three-LR search. Its separate
memory configurations preserve batch semantics and all artifacts. LR/mixture optimality is therefore unproven.
One-epoch stopping prevents time-budget-driven repetition. No new complex objective, base replay or external evaluator.

## E. GPU Performance

Fresh preflight: GPU `{preflight['hardware']['gpu']}`; host memory/disk snapshots in preflight JSON.
Generative Python 3.14.7 / PyTorch 2.13.0+cu130 / CUDA 13.0, BF16 supported; unrelated desktop processes retained.
Existing host swap use was observed. Free memory is a launch-time gate, not a promised constant.

Full 934M controlled probes: six complete optimizer updates each, first two excluded from steady-state throughput.
All seven variants passed finite loss/update checks. Per-probe sequence populations differ; short microbatch 2 is
not evidence that long microbatch 2 fits. Validation differences across probes are reported, not assumed identical.

| Variant | Assistant tok/s | Context tok/s | Median update seconds | Peak reserved GiB |
|---|---:|---:|---:|---:|
{performance_table}

Final deterministic-setting confirmation (including complete accumulation-32 mixed-length updates):

| Variant | Assistant tok/s | Context tok/s | Median update seconds | Peak reserved GiB |
|---|---:|---:|---:|---:|
{deterministic_performance}

The long dynamic probe retained about 2.65 GiB free VRAM. Accumulation-32 variable-length pilots initially exceeded
the 1.5 GiB reserve after two updates; both stopped safely. Expandable segments alone was insufficient. The explicit
third configuration releases unused CUDA cache at optimizer boundaries; sampler/optimizer/batch semantics remain fixed.
Cold start includes indexing, model restore, viewer attachment and initial save. The pilot tests this realistic shape. A later strict allocator-headroom test failed its accumulation-32 probe under the then-current desktop load. That memory gate remains BLOCKED; the failed cap configuration is not adopted.

Pilot optimizer-only aggregate: **{assistant_rate:.0f} assistant tok/s**, **{context_rate:.0f} context tok/s**;
median complete update **{measured_step:.3f}s**. This excludes cache-release, journaling, validation, quality generation
and checkpoint overhead; session budget includes all of them. Initial full-state recovery save/verify
{recovery['save']['write_verify_seconds']:.2f}s; final {recovery['final_save']['write_verify_seconds']:.2f}s;
restore {recovery['restore_seconds']:.2f}s. Development 64-case generation for repair took {chosen['seconds']:.2f}s.
Pilot evaluation duration and every validation/checkpoint duration remain in the JSON/JSONL artifacts.

**Projection, not a training result:** optimizer throughput × remaining budget × 90% overhead factor gives
approximately {projection:,.0f} assistant tokens ({projection / audit['splits']['train']['totals']['assistant_tokens']:.2f}
of the recorded one-epoch pool), capped by available examples and the resolved step cap. Length mix, desktop load,
evaluation and serialization can change this. No promise of 2,425 historical tok/s or guaranteed full-pool coverage.
BF16/SDPA/checkpointing/length padding/microbatch/workers/pinning were measured. Full FP32 AdamW would require about
14.95 GB just for weights, gradients and two moments, before activations/desktop; it was not deliberately OOM-tested.
torch.compile/TF32 and more aggressive optimizers remain unbenchmarked and disabled; no speedup claim.

## F. Checkpoint and Recovery

Native version-1 payload retained; six slots under each isolated run, atomic verified index references latest,
previous, best and final. Native staged serialization, SHA sidecars, deserialize/counter verification and publication
are reused. New finite saved-tensor checks and directory fsync protect publication. Interrupted writes never change
the published index. Changed run/config/tokenizer identity is rejected. Historical runs/exports are untouched.

Full checkpoints preserve model, optimizer, scheduler, RNG, step/microstep/token/epoch cursor, configuration,
tokenizer/data identity, initialization provenance, source/coverage counters and original budget. No partial accumulation
is serialized: interruption finishes the current complete window; deadline discards partial gradients and restores counters/RNG.
Reconstruction uses seeded epoch ordering and committed batch position. Logs beyond recovered state are archived before trimming.
Hard kill/power loss recovers only published state; unsaved steps cannot be guaranteed. Host/CUDA kernel execution and
opaque optimizer runtime behavior are not claimed bitwise reproducible unless validated below.

Original full-model recovery evidence:

```json
{json.dumps(recovery, indent=2)}
```

Deterministic-algorithm follow-up:

```json
{json.dumps(deterministic, indent=2) if deterministic else 'Not executed.'}
```

Model/optimizer bytes, scheduler and counters restore exactly in the original test; finite continuation succeeds.
Bitwise continuation initially failed; the deterministic follow-up passed and its settings are required in the
prepared specification. This is tested on this stack, not a cross-driver bitwise guarantee. The real pilot SIGTERM
stopped after step 32, saved, then resumed its full state without repeated sample identities or reset deadline.
Viewer closure after step 5 did not stop training. Tiny tests also deliver actual SIGINT/SIGTERM, inject pre-update OOM,
interrupt save publication, discard partial deadline accumulation and preserve committed updates across validation timeout.
A separate allocator-limit test produced a controlled CUDA OOM during accumulation 32, safely recorded without promoting weights. Physical VRAM exhaustion and power loss were not deliberately induced.

Storage plan: {manifest['planned_checkpoint_slots']} slots × {manifest['checkpoint_bytes'] / 10**9:.3f} GB,
one temporary serialization generation, small logs/evaluation and optional separate final export; roughly 45 GB budget.
Preflight requires 100 GiB safe disk headroom plus an 80 GiB working reserve and 12 GiB available host RAM.
Smoke artifacts are retained and separately consume disk. No old-run cleanup occurred.
No independent historical backup was verified; originals remain irreplaceable. Large artifacts remain ignored by Git.

## G. Quality Baseline

64 fixed development cases, eight each in instruction/arithmetic/factual/multi-turn/uncertainty/logic/length/coherence-proxy.
Literal compliance checks, speaker attribution and mechanical repetition metrics; no external paid judge or subjective score
presented as objective truth. Greedy BF16 max-new-tokens 128, seed 42, KV cache, penalty 1.0, no blocked n-grams.
Sampled secondary comparisons use temperature 0.7/top-p 0.9 with the same prompts/seed; stored separately.
END/EOS model stops are distinguished from length limits; generation cannot continue past configured END/EOS stop.
Macro excludes the keyword coherence proxy. Original and speaker-aware scored outputs are preserved separately.
Primary suite SHA256 `{chosen['suite_sha256']}`; decoder/tokenizer/checkpoint fingerprints appear in every report.
The final comparison repeats every candidate with deterministic algorithms and CUBLAS workspace :4096:8;
primary aggregate results match the original comparison. The corrective pilot used the original numerical settings.
Unscored human-review templates define relevance/coherence/context/length ratings and require verification evidence
for checkable unsupported claims. No human rating is claimed.

| Candidate | Literal macro | Factual | Multi-turn | END stop | Length limit | Severe degeneration |
|---|---:|---:|---:|---:|---:|---:|
{quality_table}

Every candidate scores zero on exact instruction, arithmetic, logic, line-format length and uncertainty checks.
Entry floor: instruction ≥25%, arithmetic ≥25%, END ≥50%, severe degeneration ≤25%, no empty or unexpected role cases.
These deliberately low entry thresholds are not promotion criteria. Factual keyword rates do not measure unrestricted
hallucination frequency, contradiction or semantic coherence. No reliable broad hallucination percentage is claimed.
All raw outputs are available, including failures; the following consecutive cases were not selected for impressiveness:

{examples}

Corrective results:

| Pilot | Final step | Stop reason | Literal macro | END | Severe degeneration |
|---|---:|---|---:|---:|---:|
{pilot_table}

Paired cases are selected deterministically by first case ID satisfying the stated transition, not by impressive
appearance. Literal status changes do not establish broad semantic improvement:

{paired_examples}

Retention and common validation (same population/settings per checkpoint):

| Checkpoint folder | Exposed base replay loss | Common chat validation loss | Common chat target tokens |
|---|---:|---:|---:|
{losses}

Replay is eight fixed 512-target-token windows from base TRAINING, not missing held-out base validation.
Offsets/digest are in `retention-diagnostic.json`. Development coding/factual/arithmetic tasks supply bounded capability
checks; they cannot certify catastrophic forgetting absence. Protected final acceptance is still sealed.

## H. Implemented Changes

- **LLM-Data:** `src/llm_data/chat_dataset.py`, `tests/test_chat_dataset.py` — lazy indexing, masks, collate,
  deterministic source-aware sampler, split isolation and restoration coverage.
- **LLM-Training:** `src/llm_training/training.py`, `session.py`, `recovery.py`, `chat_run.py`,
  `conversational_training.py`, `viewer.py`, `tests/test_chat_runtime.py` — runtime, recovery, deadline, signals,
  supervised normalization and conversational viewer. Existing training defaults remain compatible.
- **LLM-From-Scratch:** `configs/chat-10h-v1.yaml`, `scripts/chat_training.py`,
  `src/codexa_workspace/chat_preparation.py`, `chat_evaluation.py`, `chat_operations.py`, `chat_validation.py`,
  `chat_report.py`, `tests/test_chat_preparation.py`, this report and appended central session/progress records.
  Pre-existing dirty AGENTS/CLAUDE/CHATGPT_CURRENT_STATE files were preserved, not authored by this preparation.

Offline build via `tools/build_packages.py`; force-reinstalled only llm-data, llm-training and codexa-workspace wheels
with `--no-deps`, no dependency upgrades. Imported locations, source byte comparison and tree hashes are frozen in
the launch manifest. All component revisions and dirty state:

| Repository | Commit | Working tree | Installed Python source |
|---|---|---|---|
{git_table}

## I. Testing Results

{test_results}

Preparation commands actually executed: preflight/audit/candidates/benchmark/recovery/pilots/resume-pilot/diagnostics
(and deterministic recovery if its report is present), all using the integration generative Python. Isolated
full-model checks and five checkpoint comparisons accompany CPU regression results. Five sklearn-dependent specialist
tests are skipped in this environment; existing pytest return-value warnings are not newly introduced failures.
No blanket claim that every optional test passed. Checkpoint bitwise continuity and behavioral gates are separate
failed evidence, even when automated regressions pass.

## J. Risks and Remaining Limitations

The model has poor exact instruction/reasoning/uncertainty behavior. Lower SFT validation loss cannot select a good
assistant by itself. Repair is provisional; repetition/termination may regress through continued SFT. Current data
is dominated by UltraChat supervision and repeated conversation prefixes; content correctness was not certified.
Historical split exposure and one exact-overlap development prompt prevent clean generalization claims. The synthetic
protected suite shares task families/facts despite different wording/values; it is not a large independent benchmark.
Neither semantic human coherence ratings nor broad unsupported-claim rates were fabricated.

GPU free memory fluctuates with the desktop. Checkpoint reserve is measured with margins, not a guaranteed storage
latency bound; the watchdog can interrupt finalization and leave only the last verified checkpoint. Resume counts
downtime and rejects backwards wall-clock reconciliation; same-boot enforcement also preserves the monotonic deadline.
Startup SIGKILL/power loss before first valid save cannot create new recovery state. Viewer attachment failure blocks
updates; viewer closure afterwards does not. No backup, thermal endurance ten-hour soak, compile speedup or full
physical OOM behavior is certified. Final acceptance/promotion requires independent review.

## K. Exact Production Commands

Run from the integration repository:

```bash
cd '/home/k9-vortex/Development/3-Python/37-LLM From Scratch (PyTorch Codexa v1)/LLM-From-Scratch'
# Read-only preflight / dry run; prints BLOCKED, starts no training.
.venv/bin/python scripts/chat_training.py preflight
# Freeze current regression evidence, measured specification and report; no production launch.
.venv/bin/python scripts/chat_training.py prepare --stage finalize
```

**The following production command is prepared but currently refuses failed gates. Do not run it as approval
or edit the manifest to bypass gates. It requires a later READY evidence set and explicit user approval.**

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/chat_training.py start
# Attach/restore the visible 100×22 Kitty viewer (also auto-attached by the worker).
.venv/bin/python scripts/chat_training.py view
# Recent structured metrics, once a production job exists.
.venv/bin/python scripts/chat_training.py logs --tail 50
tail -n 50 outputs/chat-training/codexa-900m-chat-10h-v1/console.log
# Graceful stop; verifies worker PID command and process-start identity.
.venv/bin/python scripts/chat_training.py stop
# Full-state resume; original deadline includes all downtime, never resets to ten hours.
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 .venv/bin/python scripts/chat_training.py resume
# Development comparison after stopping. Protected acceptance is restricted to a finished phase.
.venv/bin/python scripts/chat_training.py evaluate --suite development
.venv/bin/python scripts/chat_training.py evaluate --suite acceptance
```

Start verifies exact Python, inputs, initialization/config identities, installed source trees, disk/GPU headroom,
conflicting training processes and an exclusive supervisor lock; creates a fresh isolated namespace and detached
worker. Budget begins at the operator command's captured monotonic start, before imports/preflight/loading, and covers
training, evaluation, saves and verification. No partial accumulation update, long validation near the reserve,
or deliberate consumption of unused hours. Structured logs/viewer are independent of the launch terminal.

After training compare base, selected initialization, best and final (if different) with common 64-case protected
greedy and sampled settings, plus development/retention diagnostics. The worker reserves finalization time and records
incomplete acceptance if it cannot finish. Preserve the first protected comparison; interrupted phases cannot tune
against it. Review all outputs, unchanged cases and regressions; semantic human review remains necessary.
No automatic inference-default replacement, export overwrite or promotion. Promotion is a separate decision.

## L. Final Readiness Decision

**{manifest['readiness']} — mandatory requirements have not all been satisfied.**

| Gate | Result |
|---|---|
{gate_table}

Next justified phase: resolve trustworthy exposure provenance and inspect
existing conversational supervision quality and test a separately bounded corrective strategy. Require measurable
instruction/reasoning compliance and reduced degeneration on development data before considering a ten-hour launch.
Do not spend ten hours merely to lower loss. Historical datasets/checkpoints/exports, NumPy workspace, accepted model,
and Git remotes remain untouched. No cleanup, push, merge or release occurred.
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return path
