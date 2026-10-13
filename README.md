> Current local layout: this repository is `LLM-From-Scratch` among eight sibling repositories. See [current layout](documentation/migration/SIBLING_LAYOUT.md); older consolidation records describe the previous submodule layout.

# Codexa v1 — PyTorch workspace

This is the canonical PyTorch development repository. Its seven component repositories are independent siblings with exact Git pins in compatibility.json. This directory retains the complete integration Git history.

| Sibling repository | Responsibility |
| --- | --- |
| LLM-Architecture | Native Transformer, RMSNorm, SwiGLU, learned/RoPE positions, KV cache and checkpoint readers |
| LLM-Tokenizer | Byte-level BPE and shared conversation/SFT serialization |
| LLM-Data | Corpus provenance, preparation, packing and token data |
| LLM-Training | Base/SFT training, optimizer state, restoration and monitoring |
| LLM-Inference | Native completion, chat and experimental exports |
| LLM-Memory | Scoped SQLite persistence, retrieval and worker client |
| LLM-Specialist | Independently frozen EmbeddingGemma 2 and classifier heads |

Reusable integration tools live in `src/codexa_workspace/`. Legacy `src` imports, script paths and `--repo LLM-From-Scratch` remain compatibility adapters. NumPy is a separate implementation and repository.

## Checkout and environments

```bash
# See documentation/migration/SIBLING_LAYOUT.md for sibling checkout and installation.
# Existing checkout:
cd LLM-From-Scratch
python run.py test -q
python run.py --repo LLM-Architecture test -q
python run.py --profile specialist --repo LLM-Specialist test -q
```

The local `.venv` and `.venv-specialist` were relocated from the original checkout on the same machine, their generated launch paths repaired, and local package wheels installed. They no longer require the old directory. Keep the generative and specialist environments separate. Exact external package inventories are recorded in requirements lock files; sibling package and Git pins are in `compatibility.json`. See [documentation/workspace/ENVIRONMENTS.md](documentation/workspace/ENVIRONMENTS.md) for interpreters (`tools/codexa-dev` for development), rebuilding and limitations.

## Protected assets and new outputs

Copy `artifacts.local.example.json` to ignored `artifacts.local.json` and set `asset_root`, or export `CODEXA_ASSET_ROOT`. The configured asset root contains the historical `checkpoints/`, `data/`, `exports/` and `logs/` trees; it is currently the guarded encrypted SSD at `/mnt/codexa/pytorch/saved` (see [storage and assets](documentation/operations/STORAGE_AND_ASSETS.md)). Model weights, datasets, private logs, source snapshots and recovery bundles are never committed.

```bash
python run.py module workflows.chat --describe
python run.py module workflows.chat --device cuda
python run.py generate --help
python run.py train --help
```

The chat catalog retains the repair comparison baseline and exact 16K tokenizer. The stage-2 pilot remains experimental. Learned-position/RMSNorm/SwiGLU native checkpoints are authoritative; historical GPT-2/Llama conversion paths remain experimental. New launcher outputs go to the configured `output_root` (default `outputs/`, currently `/mnt/codexa/pytorch/runs`); direct component commands must receive explicit output paths. Operator training attaches a visible viewer before training: the 120x30 dashboard for supervisor launches, the 100x22 Kitty viewer for direct/recipe launches ([training operations](documentation/operations/TRAINING_OPERATIONS.md)).

See [MIGRATION.md](documentation/migration/MIGRATION.md), [VALIDATION.md](VALIDATION.md) and the detailed migration report under documentation/migration/. The original source directories remain pending full retirement (only a selective, authorized retirement has happened), PRs are not auto-merged, and a single asset store is not an independent disk backup.

## Repository layout

```text
AGENTS.md / CLAUDE.md     agent rules (short index → documentation/operations/)
run.py                    CLI entry: run.py test | module | <script name>
src/codexa_workspace/     integration orchestration (many modules hash-bound — see FROZEN_CODE.md)
src/*.py, src/{data,memory,specialist}/   legacy import adapters → sibling packages
scripts/                  operator commands at top level; helpers grouped in subfolders (scripts/README.md)
tests/                    gate-pinned tests at top level; the rest grouped by area
tools/                    codexa-dev, training environment guard, build/check helpers (fixed path)
configs/  workflows/  artifact-catalog/  schemas/   run configs, chat recipes, asset catalog, JSON schemas
documentation/            operations, reference, training, migration, workspace, archive (documentation/README.md)
outputs/                  local run state (ignored)
```
