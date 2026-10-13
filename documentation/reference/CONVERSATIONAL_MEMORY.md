# Codexa conversational memory — Stage 2

Codexa's existing 934,356,480-parameter decoder generates every response. The frozen EmbeddingGemma encoder retrieves earlier conversation text; it does not generate responses. Programming classification remains available and was not trained during Stage 2.

## Running native chat

Run commands from the repository root. Existing environments are retained: `.venv` runs Codexa; `.venv-specialist` runs the local embedding worker. No new service or dependency environment is required.

```bash
# Existing conversational checkpoint; memory stays off.
python run.py chat_native \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json

# Explicit opt-in, RAM-only memory.
python run.py chat_native \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --memory ephemeral

# Explicitly persistent, user-scoped local storage.
python run.py chat_native \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --tokenizer checkpoints/tokenizer-base-v1/tokenizer.json \
  --memory persistent --user-id my-user \
  --memory-path data/memory/my-chat.sqlite3
```

`--memory-device cpu` is the default. Use `--memory-device cuda` for BF16 CUDA encoding when the GPU is available. The decoder continues using its existing FP32 native inference path. `--memory-threshold 0.68` is a small-benchmark development choice, not a calibrated probability or universal relevance guarantee.

Terminal commands:

| Command | Behavior |
| --- | --- |
| `/memory on` or `/memory ephemeral` | Enable memory in RAM; no persistent database |
| `/memory persistent` | Enable the explicitly selected database |
| `/memory off` | Stop the encoder and close storage; previously saved data remains |
| `/new` or `/clear` | Start a new conversation and clear live history; preserve saved conversations |
| `/memory list` | List only this user's saved conversation IDs |
| `/memory sources ID ...` | Explicitly select earlier conversations to search alongside the active one |
| `/memory sources` | Stop searching earlier conversations |
| `/memory refs` | Show excerpts actually supplied to the latest response, roles, scores and source references |
| `/memory stats` | Show readiness, latency, token overhead and selected scopes |
| `/memory clear` | Delete the active conversation's saved memory and clear live context |
| `/memory delete ID` | Delete one selected user's conversation and its vectors; clearing the active ID also clears live context |
| `/memory rebuild` | Finish indexing selected scopes under the current encoder identity |
| `/help` | Display commands |

Switching storage modes closes the previous store; it does not silently transfer conversations between ephemeral and persistent storage. Enabling memory during an existing chat stores subsequent completed turns. Earlier turns that occurred while memory was off are not retrospectively persisted. An encoder warming-up notice means that response used ordinary chat; completed turns are still stored for later indexing.

The JSON bridge accepts the same CLI flags. Existing `chat` and `reset` requests remain supported. Responses add a `memory` object. Memory operations use:

```json
{"type":"memory","id":"m1","operation":"persistent"}
{"type":"memory","id":"m2","operation":"list"}
{"type":"memory","id":"m3","operation":"sources","value":["selected-conversation-id"]}
{"type":"memory","id":"m4","operation":"refs"}
{"type":"memory","id":"m5","operation":"delete","value":"selected-conversation-id"}
```

User identity is fixed by `--user-id` for a bridge process. These are local trusted-user scopes, not an authentication service. New database files have mode 0600; SQLite secure deletion is enabled. Database deletion does not erase backups or filesystem snapshots.

## Encoder and memory architecture

The encoder is pinned to `google/embeddinggemma-2` revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, with vision/audio excluded and all weights frozen. It emits normalized 768-dimensional vectors. Queries use `SearchQuery` (`task: search result | query: `); records use `Document` (`title: none | text: `). The published task registry and BF16/FP32 requirement are verified. FP16 is rejected. See the [Google model card](https://huggingface.co/google/embeddinggemma-2).

The stdlib subprocess protocol isolates Transformers 5.19 / Sentence Transformers 6.1 from the generative environment's Transformers 4.57.6. Existing PyTorch 2.13 installations remain untouched. Worker stdout carries JSON only; loading diagnostics go to stderr. Initialization is asynchronous, requests are bounded, failed requests fall back to ordinary chat, and timeout closes the worker so a late response cannot be assigned to another query. Shutdown first requests a normal exit, then terminates only the owned worker if necessary.

SQLite stores original complete user/assistant turns, exact-text chunks, source relationships and versioned FP32 vector blobs. A 480-byte UTF-8 content cap bounds byte-fallback token counts while preserving exact original text; original raw turns remain available independently. Search filters user and explicitly selected conversations before cosine scoring. It returns up to three hits, excludes recent messages already in context, and removes repeated/overlapping message hits. Stable IDs break equal-score ties.

Record embeddings persist under an identity covering model revision, prompts, dimensions, normalization, chunk policy, device/precision and library versions. Missing selected records are indexed lazily. Changed settings create a parallel index, committed only after all new vectors validate; old vectors are retained and incompatible identities are never scored together. The worker client also keeps a bounded 128-entry embedding cache, cleared on shutdown/deletion.

Prompt construction preserves the optional initial system message and newest complete turns. It reserves the requested generation budget (128 tokens by default) inside the 2,048-token window. Memory overhead, including wrappers, is capped at 256 Codexa tokens. Newest history takes priority; insufficient space or confidence omits memory. Exact token IDs delimit roles; literal reserved markers in user/memory content become ordinary visible text. Memory is an untrusted reference block inside the current user message, with chronological turns and speaker labels. Stable local source labels avoid putting random storage IDs into model input; debug references retain original conversation IDs. Retrieved statements are not certified facts or instructions.

Prompt fitting never changes archived history. Failed generation leaves history unchanged; only successful complete turns are stored. Clearing native memory also clears live context, so deleted text cannot remain in the next prompt.

## Reproducible evaluation

The retrieval fixture contains 60 synthetic evaluation-only queries across direct references, paraphrases, long-distance references, related distractors, updated preferences, and no-context requests. Development and test use different conversation families (30 queries each). These fixtures are not training data.

The Stage 2 scripts (`evaluate_memory`, `evaluate_chat_memory`, `evaluate_chat_decoding`, `train_chat_stage2`) were retired on
2026-10-13; recover them from git tag `pre-rewrite-2026-10-13` to rerun these commands.

```bash
.venv-specialist/bin/python scripts/evaluation/evaluate_memory.py \
  --device cpu --output logs/memory/retrieval-new-run

.venv/bin/python scripts/evaluation/evaluate_chat_memory.py \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --output logs/memory/chat-new-run

.venv/bin/python scripts/evaluation/evaluate_chat_memory.py \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --memory-device cuda --output logs/memory/chat-cuda-new-run
```

Outputs require fresh directories. The chat suite saves 24 real scenarios with memory off/on, the same seed 42, temperature 0.7, top-p 0.9, KV caching and 64 generated-token limit. Stable benchmark conversation IDs and stable source labels make prompt construction reproducible. Benchmark histories are deliberately supplied fixtures, not claims about real users.

On `logs/memory/retrieval-stage2-cpu-v1/report.json`, threshold 0.68 was selected on development data only:

| Held-out retrieval measure | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.50 / 0.90 / 1.00 |
| Precision@1 / @3 / @5 | 1.00 / 0.60 / 0.40 |
| MRR | 1.00 |
| False-positive queries | 0 of 5 no-context test queries |
| CPU query embedding median / p95 | 29.56 / 31.26 ms |
| Exact search median | 0.264 ms |
| Index build | 2.389 s |
| Vector payload | 491,520 bytes |
| Raw retrieved excerpt tokens, median | 18.5 |

Recall includes both relevant speaker records; one correct first hit therefore yields Recall@1 0.5. Precision divides by fixed K, including unfilled result slots. Raw excerpt counts exclude prompt wrappers; actual generation logs report exact wrapper-inclusive overhead. Five negative queries are a small sample and do not establish a real-world false-positive guarantee.

Preliminary 24-scenario hardware comparisons, recorded before the final stable-label evaluation, demonstrated simultaneous loading:

| Encoder mode | Median reply off / on | Peak total GPU use | Worker peak RSS |
| --- | ---: | ---: | ---: |
| CPU FP32 | 0.427 / 0.496 s | 6,526 MiB | 2.44 GiB |
| CUDA BF16 | 0.429 / 0.443 s | 7,647 MiB | 1.95 GiB |

Total GPU values include desktop/driver usage. CUDA encoder allocator peaks were 737 MiB allocated / 796 MiB reserved. Response medians mix different response lengths and are not isolated encoder speed comparisons. CPU remains the default for isolation. The SFT pilot runs without an embedding worker on CUDA.

Generation results, representative outputs, and completed pilot metrics are recorded in `documentation/reports/STAGE2_RESULTS.md`. Retrieval ranking, literal-value checks, loop measurements and END statistics are automated diagnostics. Qualitative notes are agent assessments; no human ratings have been collected. A recalled name can still be misattributed to the assistant, and retrieved corrections can still be ignored.

## Conversational data and bounded training

Existing UltraChat and OpenAssistant prepared derivatives are reused. Balanced and repair data are audited as earlier experiments; repeated repair anchors are not included again. Full source/root provenance and historical SFT exposure are retained. The new deterministic 90/5/5 split groups source roots, normalized duplicate conversations and shared substantial prompts/answers together. It does not certify factual accuracy or eliminate semantic paraphrases.

`data/processed/general-chat-sft-v2/` contains 182,892 train, 10,135 validation and 9,874 test conversations. Preparation rejected 26,497 role/overlength records, 42 normalized duplicates, 56 severely repetitive records and 5 reserved-marker records. Long conversations are rejected rather than truncating assistant answers. Earlier SFT models may have seen records in these new splits; generation test prompts are separately authored and the new pilot starts from the base checkpoint.

```bash
# Existing output is protected; choose a fresh path for another preparation.
python run.py prepare_general_chat \
  --output data/processed/general-chat-sft-v2-new-run

# Hard limits remain <=200 steps and <=60 minutes.
.venv/bin/python scripts/legacy/train_chat_stage2.py \
  --run-name codexa-900m-chat-stage2-pilot-new-run \
  --max-steps 200 --minutes 60
```

The launcher requires a visible 100x22 Kitty viewer before any backward/update. Source checksum, exact architecture and tokenizer fingerprint, dataset checksums, shifted assistant targets and group isolation are checked. A real optimizer update and a reserved-VRAM gate precede training; original base weights are restored after preflight. Full-parameter SFT uses BF16 autocast, gradient checkpointing, 8-bit AdamW, microbatch 1, accumulation 32, peak LR 1e-5, 20-step warmup, weight decay 0.1, clipping 1.0 and zero base replay. It selects 20,000 seeded source-balanced conversations and uses assistant content plus END targets only.

Training uses a deadline and reserves five minutes for final checks, evaluation and export. Fresh namespaces are required. Atomic checkpoints include optimizer/scheduler/RNG/iterator recovery state and tokenizer/dataset identities. The Stage 2 launcher does not yet offer an automatic resume CLI: recovery artifacts are retained, but an interrupted pilot is not silently extended. If preflight is unsuitable, the experiment stops and preserves its evidence.

The pilot is experimental and is not promoted automatically. A native weights-only inference export is saved separately, with an identical-token source/export smoke test. No checkpoint, export, tokenizer, dataset or log cleanup occurs.

## Decoding controls and research

Native chat exposes the decoder's existing `--greedy`, `--seed`, `--repetition-penalty` and `--no-repeat-ngram-size` controls; defaults are unchanged. Repetition controls can inhibit useful repetitions and copying names from context. They do not supply missing knowledge.

```bash
.venv/bin/python scripts/evaluation/evaluate_chat_decoding.py \
  --checkpoint checkpoints/codexa-900m-sft-repair-v1/latest.pt \
  --output logs/memory/decoding-new-run.json

python run.py --profile specialist probe_embedding_states \
  --device cpu --output logs/memory/token-state-probe-new-run.json
```

See `ENCODER_CONDITIONING.md` for architecture A/B/C, accessible token states, tokenizer compatibility, training objectives and measured-versus-estimated memory costs. No encoder-decoder training was launched.

## Verification and limitations

```bash
tools/codexa-dev -m pytest -q tests/integration/test_memory.py tests/integration/test_native_memory.py \
  tests/integration/test_general_chat.py
python run.py --profile specialist --repo LLM-Specialist test -q
```

Existing classifier contracts and all 39 specialist tests are preserved. Tests cover role safety, context fitting with/without a system prompt, rollback, normalized shapes, scopes, deterministic ranking, persistence, deletion, safe rebuild failure, unsupported prompts, unavailable encoder fallback, shifted assistant targets, duplicate splits, deadlines and tiny native checkpoint inference.

Known limits: exact search is intended for small local indexes; old index revisions consume storage; no factual memory reconciliation or semantic training deduplication is performed; no external knowledge corpus is indexed; the generative model still has weak knowledge/instruction following; prompt-injection resistance is limited by that model's understanding even though role boundaries are protected. This stage provides working retrieval context, not a guarantee of capable conversation.
