# Sibling repository layout

The workspace directory is not a Git root. It holds eight independent repositories side by side:

| Repository | Owns |
| --- | --- |
| **LLM-From-Scratch** (integration) | `src/codexa_workspace` (orchestration, evaluation, training jobs, operator tools), integration tests, docs, environments and runtime outputs |
| LLM-Architecture | Native Transformer, KV cache, checkpoint readers |
| LLM-Tokenizer | Byte-level BPE and conversation/SFT serialization |
| LLM-Data | Corpus provenance, preparation, packing and token data |
| LLM-Training | Base/SFT training, optimizer state, restoration, telemetry |
| LLM-Inference | Native completion, chat and exports |
| LLM-Memory | Scoped SQLite persistence, retrieval and worker client |
| LLM-Specialist | Frozen EmbeddingGemma 2 and classifier heads |

`compatibility.json` records each sibling's path and exact Git pin. Shared build, preservation and validation directories live in `../workspace-infrastructure` and are linked from the integration root.

## Dependency graph

- Architecture and Tokenizer are leaves.
- Data depends on Tokenizer.
- Training depends on Architecture, Tokenizer and Data.
- Inference depends on Architecture and Tokenizer, with Memory optional.
- Memory talks to a separately configured JSON-lines Specialist worker; there is no Python import dependency.
- Specialist is independent.
- Integration depends on the generative components and optionally Specialist.

Components never import the integration package (`tests/infra/test_repository_boundaries.py`).

## Checking out a fresh workspace

1. Clone the integration repository into `LLM-From-Scratch`.
2. Clone each component listed in `compatibility.json` into its sibling directory, using the URLs in `submodules-before-2026-10-09.gitmodules`, and check out the recorded commit.
3. From the integration root:
   - `python run.py test -q` runs this repository's tests.
   - `python run.py --repo LLM-Data test -q` runs a sibling's tests.
   - `--profile specialist` selects the specialist environment.

   Source-first imports follow the explicit sibling paths in `compatibility.json`.

Do not reset existing worktrees to recreate this layout. The pre-2026-10-09 submodule layout, its file map and move journal are in `../workspace-infrastructure/audit-2026-10-09`.
