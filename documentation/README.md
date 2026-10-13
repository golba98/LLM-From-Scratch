# Documentation

| Folder | What is in it |
| --- | --- |
| [guides/](#guides) | How to work in this repository and run training: start here |
| [reference/](#reference) | How the model and its components work |
| [reports/](#reports) | Dated results and investigations (read-only history) |
| [training/](#training-live-logs) | The two live logs that every session appends to |
| [workspace/](#workspace) | Workspace-wide files (the workspace root symlinks to some of them) |
| [migration/](#migration) | The eight-repository layout and how to check it out |

Older material (archived reports, the 900M chat and storage-benchmark reports, the consolidation records) was removed on 2026-10-13. It is preserved at git tag `pre-rewrite-2026-10-13`.

## guides

| Guide | Covers |
| --- | --- |
| [DEVELOPMENT.md](guides/DEVELOPMENT.md) | Setup, repository layout, package layers, where new code and tests go |
| [TRAINING_OPERATIONS.md](guides/TRAINING_OPERATIONS.md) | Preparing, approving, launching, resuming and monitoring a run; gates; GPU headroom |
| [NOTIFICATIONS.md](guides/NOTIFICATIONS.md) | Run-end popup, AI wakeup, agent window |
| [STORAGE_AND_ASSETS.md](guides/STORAGE_AND_ASSETS.md) | Where weights, data and outputs live (`/mnt/codexa` SSD) and the rules for them |
| [CODE_IDENTITY.md](guides/CODE_IDENTITY.md) | What a frozen job binds, stable external paths, the historical source archive |
| [HUMAN_REVIEW.md](guides/HUMAN_REVIEW.md) | Building human review packages and the public review page |

## reference

| Document | Covers |
| --- | --- |
| [MODEL_CARD.md](reference/MODEL_CARD.md) | The model, its current status, intended use and limits |
| [ARCHITECTURE.md](reference/ARCHITECTURE.md) | Base architecture (`configs/1b.yaml`) |
| [TOKENIZER.md](reference/TOKENIZER.md) | Byte-level BPE tokenizer |
| [DATASET.md](reference/DATASET.md) | Base dataset strategy and provenance |
| [CONVERSATIONAL_MEMORY.md](reference/CONVERSATIONAL_MEMORY.md) | Scoped memory and retrieval for chat |
| [ENCODER_CONDITIONING.md](reference/ENCODER_CONDITIONING.md) | EmbeddingGemma representations and conditioning options |
| [SPECIALIST.md](reference/SPECIALIST.md) | The programming-request specialist (frozen EmbeddingGemma 2 + heads) |

## reports

| Report | Date | About |
| --- | --- | --- |
| [CODEXA_934M_TRAINING_SUPERVISOR.md](reports/CODEXA_934M_TRAINING_SUPERVISOR.md) | 2026-10-11/12 | Supervisor design and the 934M foundation run |
| [CODEXA_934M_FOUNDATIONAL_PRETRAINING_READINESS.md](reports/CODEXA_934M_FOUNDATIONAL_PRETRAINING_READINESS.md) | 2026-10-11 | Data selection and readiness for foundation pretraining |
| [CODEXA_934M_FOUNDATIONAL_CAPABILITY_DIAGNOSIS.md](reports/CODEXA_934M_FOUNDATIONAL_CAPABILITY_DIAGNOSIS.md) | 2026-10-11 | Language vs. chat capability diagnosis |
| [CODEXA_934M_FINAL_QUALIFICATION_MODEL_SELECTION.md](reports/CODEXA_934M_FINAL_QUALIFICATION_MODEL_SELECTION.md) | 2026-10-11 | Final qualification and model selection |
| [CODEXA_934M_UBUME_NATIVE_INTEGRATION.md](reports/CODEXA_934M_UBUME_NATIVE_INTEGRATION.md) | 2026-10-11 | Native integration with the Ubume CLI |
| [MASKED_KV_CACHE_REPAIR.md](reports/MASKED_KV_CACHE_REPAIR.md) | 2026-10-12 | Masked KV-cache repair in LLM-Architecture |
| [STAGE2_RESULTS.md](reports/STAGE2_RESULTS.md) | 2026-10-08 | Stage 2 memory/chat pilot results |

## training (live logs)

Fixed paths; the sibling repositories point here.

- [SESSION_DECISIONS.md](training/SESSION_DECISIONS.md): append dated decisions and evidence.
- [100M_PROGRESS_LOG.md](training/100M_PROGRESS_LOG.md): append milestones.

## workspace

The workspace root's `README.md` and `AGENTS.md` are symlinks to `OVERVIEW.md` and `WORKSPACE_GUIDELINES.md`; see [workspace/README.md](workspace/README.md).

- [ENVIRONMENTS.md](workspace/ENVIRONMENTS.md): current vs. historical interpreters and the training guard
- [HISTORICAL_SOURCE_RECOVERY.md](workspace/HISTORICAL_SOURCE_RECOVERY.md): restoring the completed 5h run's exact source (inventory: `HISTORICAL_SOURCE_V1.json`)
- [SHARED_INFRASTRUCTURE.md](workspace/SHARED_INFRASTRUCTURE.md): the `workspace-infrastructure` README

## migration

- [SIBLING_LAYOUT.md](migration/SIBLING_LAYOUT.md): the eight repositories, their dependencies and checkout
- `submodules-before-2026-10-09.gitmodules`: clone URLs for the sibling repositories
