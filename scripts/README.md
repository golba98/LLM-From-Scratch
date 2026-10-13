# scripts/

Top level = **operator commands**. Their paths are fixed: systemd units, `launch.json`, frozen job `code_identity` and the flat `run.py <name>` lookup refer to them. See [FROZEN_CODE.md](../documentation/operations/FROZEN_CODE.md). Everything else is grouped by area in subfolders.

Run any script with `tools/codexa-dev scripts/<path>.py ...` (source-first), or by name with `python run.py <name> ...`. `run.py` also finds scripts inside the subfolders.

## Operator commands (top level)

| Script | Purpose |
| --- | --- |
| `training_supervisor.py` | OS-level supervisor: prepare / approve / launch / resume / extend / reconcile |
| `training_supervisor_ssd.py` | Same supervisor, restricted to run directories on the `/mnt/codexa` SSD output tree |
| `training_dashboard.py` | Read-only 120x30 Rich dashboard (`--once`, `--compact`, `--kitty`, `--attach`) |
| `training_status.py` | Instant run status (`--json`) |
| `training_wakeup.py` | Run-end AI wakeup delivery; `open-agent` reopens the agent window |
| `foundation_continuation.py` | 10-hour 934M foundation continuation: prepare / approve / launch / resume |
| `pretraining_readiness.py` | CPU-only foundational pretraining analysis and proposals |
| `conversational_readiness.py` | CPU-safe conversational preparation and baseline comparison |
| `chat_training.py` | 900M chat successor operator (preflight … evaluate) |
| `sft_readiness.py` | GO/NO-GO verdict for the SFT pilot |
| `prepare_sft_feasibility.py` | SFT feasibility evidence preparation |
| `environment_identity.py` | Reports which implementation/environment is loaded (`--expect current`) |
| `foundation_capability_probe.py` | Inference-only foundation capability probe |
| `build_capability_suite_v2.py` | Builds the frozen capability suite v2 JSON (do not re-run casually) |
| `train.py`, `train_conversational_sft.py`, `run_chat_overfit.py` | Base/SFT/overfit training entry points used by `workflows/recipes.json` |

## Subfolders

| Folder | Contents |
| --- | --- |
| `data/` | Corpus download/preparation, tokenization, token-data inspection/benchmarks, tokenizer training and comparison |
| `evaluation/` | Checkpoint, chat-memory and decoding evaluation; comparisons; qualification review packages; review dashboard |
| `diagnostics/` | Post-training generation diagnostic (run/organize/render), masked KV-cache validation, tiny overfit, embedding probes |
| `inference/` | Generation, native chat and bridge, memory worker, specialist, native/LM Studio exports |
| `storage/` | Checkpoint storage benchmarks (V1–V3.1), GPU qualification, retention/capacity planning, production-shape benchmark |
| `sft/` | SFT phase 4 preparation, SFT pilot approve/launch, Step 5A verification, chat training audit |
| `monitoring/` | GPU monitoring and training/GPU summaries |
| `legacy/` | Historical one-offs: 100M/900M shell launchers and the 100x22 viewer, stage-2 chat pilot, chat-regression investigation, bounded chat confirmation, preflight, run-manifest validation |

Many `data/`, `inference/` and `monitoring/` files are thin compatibility wrappers that forward to the owning sibling package (`llm_data`, `llm_inference`, `llm_training`, …). New reusable code belongs in that owner, not here.
