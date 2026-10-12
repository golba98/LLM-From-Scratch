# CODEXA 934M — UBUME DEV NATIVE INTEGRATION

Date: 11 October 2026. Implementation is available in the development checkout; no npm release, deployment, push, training, or model promotion occurred.

## A. Integration summary

Extended the existing Codexa Native adapter, existing provider/model selectors, conversation store, settings overlay, and streaming renderer. Five explicit variants are discoverable. Native prompts bypass coding/plan wrappers and action parsing; tools, vision, and reasoning capabilities are honestly unsupported. Raw generated answers are not corrected. The production Ubume package and unrelated provider implementations remain separate.

## B. Model availability

| Variant | Full SHA256 verified | Selector discovery | Real inference |
|---|---|---|---|
| Base | PASS | PASS | Not run |
| Repair | PASS | PASS | Not run |
| Balanced | PASS | PASS | Streamed response + cancellation |
| Original 2H | PASS | PASS | Not run |
| END3 | PASS | PASS | Streamed response after switching |

All five full-file hashes match the user's identities; metadata remained stable during hashing. See `outputs/integration/ubume-native-20261011-v1/checkpoint-integrity.json` for exact paths, hashes, inode/mtime/size, and verification timings. Actual paths come from the existing frozen inventory; an ignored `artifacts.codexa-native.local.json` references them without copying weights. Full identities and sidecars are checked on each load, never silently substituted. The tokenizer identity is verified on startup. Missing variants retain an unavailable explanation.

## C. Local inference architecture

TypeScript manages one owned Python process over versioned JSONL v2. Existing integration script bootstrap loads sibling architecture/inference/tokenizer sources. Native engine, serialization, generation, END/EOS, KV cache, and context-window policy are reused. Unicode-safe progressive decoding emits ordered deltas; final output must match the stream. Errors, malformed events, invalid identity/order, startup/generation timeouts, OOM, nonfinite logits and cancellation stop the owned process. A later request can restart it. No HTTP server or cloud transport was added.

Structured full conversation history is sent on each request; native tokenizer counts prompt tokens and reserves generation capacity within 2048. Old complete turns may be omitted with a visible notice. The latest request/system instruction cannot silently overflow. A switched model rebuilds context and cannot inherit KV cache. Raw output, checkpoint ID/hash, settings and termination metrics are retained in conversation metadata; completion events keep model provenance visible. Diagnostic logs exclude conversation text and are bounded to 1 MiB of appends.

A selected model is reused between completed turns and unloaded on cancellation, explicit unload, selection/provider change, or exit. No unrelated GPU processes are terminated. CPU fallback is explicit. Interactive preflight requires 16 GiB available host RAM and 8 GiB free VRAM for CUDA. The formal diagnostic/training 180 GiB reserve remains unchanged; it is not imposed on ordinary inference, which writes no checkpoints.

## D. User experience and exact launch

```sh
cd '/home/k9-vortex/Development/2-TypeScript/8-Ubume CLI'
bun run dev:run
```

`ubume-dev` is also verified to point to this checkout. Select `/providers` → Codexa Native → Codexa PyTorch → Select model (or `/model`), then choose an explicit 934M variant. Type a question. Use `/codexa settings`, `/codexa unload`, and `/codexa replay` to inspect controls, release memory, and compare the latest prompt after switching. Existing cancel shortcuts terminate inference. Evaluation defaults are greedy / 128 new tokens / penalty 1 / seed 42. Conversation defaults use native sampling at temperature .7/top-p .9. Individual controls require the custom preset. No candidate is designated production-quality.

`bun run dev:run --help` was validated. PTY startup with TERM=xterm-256color rendered Ubume v0.1.17-dev local; a bare non-VT terminal correctly refused startup. Full interactive Codexa-selector traversal was covered by picker/runtime tests, not a visual end-to-end manual user review. The separate old `ubume-dev native` direct-chat shortcut was intentionally preserved and is not the command for this integration.

## E. Measured performance

The real provider smoke used fresh `outputs/integration/ubume-native-20261011-v1`, max32 greedy tokens, and the actual managed provider. No optimizer updates or checkpoint saves.

| Measurement | Balanced | END3 |
|---|---:|---:|
| Loading seconds | 25.393 | 16.153 |
| First-token seconds after generation start | 0.608 | 0.560 |
| First visible token including loading, seconds | 27.015 | 17.872 |
| Generation tokens/second | 33.031 | 34.938 |
| Peak CUDA allocated bytes | 3,834,716,160 | 3,834,716,160 |
| Peak CUDA reserved bytes | 3,906,994,176 | 3,906,994,176 |
| Host high-water bytes | 4,538,363,904 | 4,538,183,680 |
| Completion | 32-token cutoff | 32-token cutoff |

A reused Balanced generation was cancelled after its first delta: 11.81 ms to first visible token, 90.80 ms total to stopped completion. END3 subsequently loaded alone; switch timing includes loading above. Native END termination was exercised with CPU fixtures; neither real smoke answer emitted natural END within 32 tokens.

Balanced output began “Sure, I can help you with that” and repeated READY; END3 described READY as a language. Both FAILED the requested one-word instruction. Raw text is preserved in smoke.json. Integration success does not imply conversational capability.

After this smoke, startup was improved to combine selected-identity checking and required sidecar verification into ONE full-file hash before deserialization. CPU regression verifies one hash and rejects wrong identities/corruption. The timing table predates that optimization; no faster final-loading measurement is claimed. A final real repeat in fresh v2 namespace correctly refused loading because RAM fell below 16 GiB. The blocked attempt is preserved. After unload, GPU free memory was 12,302 MiB and no native bridge remained running.

## F. Testing

- TypeScript typecheck PASS after final source changes.
- Native runtime/settings-config/display focused suite: 46 PASS.
- Picker/model-picker/native transport suite: 55 PASS.
- Native prompt preparation regression: 1 PASS (editing/cleanup-looking prompts cannot activate native tools).
- LLM-Inference full CPU suite: 16 PASS.
- LLM-Architecture model + selected-identity tests: 6 PASS, including one-hash/wrong-identity/corrupt-file checks.
- Integration generation/native-memory CPU regressions: 12 PASS.
- Real Balanced streaming/cancel/END3 switching smoke: PASS before final single-hash optimization.
- Repeat final smoke: BLOCKED at RAM preflight, no model loaded.
- Initial full Ubume suite: 2,129 PASS / 4 FAIL. Three App workbench state assertions and an obsolete native-label assertion failed. Label expectation was corrected; all 8 workbench cases passed on isolated rerun.
- Second full suite under concurrent filesystem verification: 2,063 PASS / 7 FAIL / 5 errors. Two unrelated Local Harness startup timeouts caused follow-on Bun node:test nested-test errors. Both browser-harness cases passed isolated (2 PASS); failed logs retained. No all-green full-suite claim is based on these attempts.

Final sequential full-suite: **2,135 PASS / 0 FAIL** (143.77 seconds). After the final incomplete/empty-turn recovery fix, the targeted native/prompt/conversation-store suite passed; 18 PASS / 0 FAIL. Final `bun run check` and `bun run typecheck` PASS. Initial Biome check found 13 import-order/format errors introduced in changed files; fixed with safe organization/formatting only on task-owned files, with failed evidence preserved. Tests are evidence, not sums of unique tests; repeated invocations overlap. No optional dependencies were installed and no tests suppressed.

## G. Remaining limitations

Three checkpoint variants have not received a real interactive GPU smoke. Final one-hash startup has CPU proof but its repeated GPU smoke is resource-blocked. RAM availability fluctuates near the gate; swap remained elevated (~6.6 GiB). Model response quality remains weak. CPU fallback is slower and has not been full-model benchmarked. Persistent raw conversation content follows ordinary local Ubume history storage; diagnostics omit it. Interactive conversations are exploratory, not blinded formal acceptance results. Frozen diagnostic approvals/plans were not modified; future formal jobs must revalidate any affected implementation fingerprint under their existing approval mechanism.

## H. Modified files and preservation

Complete repository-owned source file list: `outputs/integration/ubume-native-20261011-v1/modified-files.json`. Ubume owns provider config/adapter, launcher wiring, UI settings/commands/provenance, conversation metadata and scripts. LLM-Inference owns JSONL transport and native callback integration/finite checks. LLM-Architecture owns the optional expected-checkpoint-identity loader parameter. Integration owns ignored local registry configuration and records. Existing dirty export/CLI files, architecture checkpoint-storage work, and Ubume buildInfo changes were preserved; they are not attributed to this task.

No existing checkpoints, datasets, exports, model weights, tokenizer, historical experiment configurations, approvals, or NumPy assets were deleted, migrated, converted, overwritten, or copied. No model training, production install/release, global package installation, checkpoint reclamation or public server occurred. Small smoke/log records live only in new task namespaces.

## I. Decision

Native development integration is usable when interactive resource gates pass. Model-quality promotion, full five-variant interactive GPU coverage and final optimized-startup GPU timing remain unqualified. Keep using these candidates for exploratory evaluation rather than trusted factual answers or coding-agent execution.

## Completion record

Empty native completions now surface an explicit error and retain raw generation metadata even when there is no visible assistant text. Unanswered/empty stored turns are omitted from subsequent native prompts with a visible notice; valid complete history and whitespace are preserved. The final focused regression includes this failure-recovery case. No real GPU test was retried after the RAM refusal. Full checkpoint-integrity verification completed for all five candidates. Final production source, weights, data and approvals remain untouched.

Final evidence footprint: approximately 1.1 MiB in the v1 smoke/evidence namespace, plus 8 KiB for the RAM-blocked v2 attempt. The task-created PTY fixture had automatically imported local dev copies and worktrees; after validating its exact isolated root and inode identities, its files and symlinks were disposed without following symlink targets. No original content was removed. This fixture cleanup is recorded in tui-fixture-cleanup.json; no cleanup targeted real output/model stores.
