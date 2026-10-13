> Historical consolidation record. The current 2026-10-09 sibling layout and validation results are documented in documentation/migration/SIBLING_LAYOUT.md and ../workspace-infrastructure/audit-2026-10-09/REPORT.md.

# Consolidation and recovery

The canonical PyTorch checkout is the root of project 37. Its former nested integration Git directory, files, remotes, branches and index were moved intact to this root. Seven component checkouts retain independent histories and become pinned Git submodules; .git directories were absorbed by Git into the root's modules storage. No component history was copied into the root as ordinary files. The old nested directory contains a retirement notice only.

Original source is assigned by documentation/migration/module-mapping.json. Architecture owns the byte-preserved Transformer; Tokenizer owns tokenizer/SFT serialization; Data owns ingestion, preparation and token shards; Training owns configuration, optimizers, checkpoint state, SFT, monitoring/reporting; Inference owns generation/native chat/export; Memory owns retrieval/index semantics; Specialist owns the frozen encoder and worker. Central integration owns configuration, evaluation, command dispatch, workflows and compatibility adapters. Package dependencies form an acyclic graph, checked by tests; the specialist communicates across its separate environment through JSON lines.

Old src imports and script entry points delegate to owners, retaining checkpoint class names and APIs. Unique original conversational, stage-2, memory, data-governance, export and research entry points remain present. Existing experimental checkpoints remain experimental.

## Source-to-destination map

| Source | Canonical destination / treatment |
|---|---|
| 31/src/model.py | 37/LLM-Architecture/src/llm_architecture/model.py |
| 31/src/tokenizer.py, sft.py | 37/LLM-Tokenizer/src/llm_tokenizer/ |
| 31/src/data/, token_data.py and preparation scripts | 37/LLM-Data/src/llm_data/ |
| 31/src/training.py, train.py, config.py, checkpointing.py, conversational_training.py, monitoring/reporting | 37/LLM-Training/src/llm_training/ |
| 31/src/generate.py, native_chat.py and generation/export scripts | 37/LLM-Inference/src/llm_inference/ |
| 31/src/memory/ | 37/LLM-Memory/src/llm_memory/; inference-context boundary in Inference |
| 31/src/specialist/ and worker scripts | 37/LLM-Specialist/src/llm_specialist/ |
| 31/src/evaluation.py, workflows, configs, documentation, tests, remaining research scripts | 37 root; named codexa_workspace where appropriate; legacy adapters retained |
| 37/LLM-From-Scratch/* including .git | 37/* preserving integration history |
| 32/src/* (55 modules) | 32/src/llm_numpy/*; old src files become module-identity adapters |
| 32/scripts and training/evaluation examples | Named llm_numpy.cli owners; legacy launch adapters |
| 32/benchmarks, cpp, experiments, tests, reference fixtures | Retained in NumPy monorepo with canonical imports |
| Deprecated NumPy source | Retained old directory plus provenance-preserving recovery source archives |
| Historical local-artifact source/manifests | Retained; conflicting historical manifests not overwritten |

## Assets and paths

A single protected local asset store is configured by ignored artifacts.local.json; no weights or datasets are in public repositories. The local store is Development/LLM-Assets-Staging. Root moves were atomic on the same filesystem:

* 31/{data,checkpoints,exports,logs} → store/pytorch/{same}.
* 32/{data,runs,logs} → store/numpy/current/{same}.
* NumPy-local-artifacts/data → store/numpy/historical-local/data.
* Deprecated/checkpoints → store/numpy/deprecated/checkpoints.

Hashes distinguish all files; 212 historical files have no byte-identical current counterpart and are preserved separately. Deprecated source has 85 substantive files: 50 byte-identical, 35 differing. No file was discarded merely because a newer implementation exists. New experiments use canonical outputs/; input aliases resolve relocated historical paths. Historical manifests retain provenance, while runtime resolution prevents a permanent dependency on 31. Tokenizer identity/hashes and the differing NumPy/PyTorch special token IDs are documented; tokenizers are never swapped across models.

The existing PyTorch environments moved to 37 on the same machine; generated launcher paths were repaired. They require reconstruction on another host, not blind copying. NumPy is a modular monorepo because its custom Tensor/autograd/layers/trainer share tightly coupled APIs; no unnecessary new GitHub repos were created.

## Recovery

The protected store's recovery/2026-10-08-consolidation contains source inventories, hash manifests, relocation journals, ten verified history.bundle files, original Git metadata, branch/remotes/worktree records, binary uncommitted patches and source snapshots. The supplemental-source-data.tar.gz and its manifest are REQUIRED alongside initial source.tar.gz snapshots: a directory-name exclusion in the initial snapshot omitted source data packages, which were recovered and verified before changes. Complete reference trees are present. Small files and Git bundles are recoverable locally, but the asset store and recovery archives share a disk; this is NOT an independent disaster-recovery backup.

Before rollback stop writers, verify manifests, restore source snapshots AND supplemental archive into a separate location, restore bundles into independent repositories, reapply saved uncommitted patches, and use the relocation journal to reverse asset root renames only if destinations are absent and hashes match. Do not overwrite canonical outputs. Restore private path configuration/environment launchers to the intended layout. Restoring original workspace root metadata requires the preserved metadata archive and worktree records. PR commits can be reverted without moving assets; old input aliases can be reinstated separately. Validate tiny inference/training before resuming.

## Retirement gates

31, deprecated NumPy, historical local-artifacts and the old nested directory remain, with notices. No project directory or remote repository was deleted. Remove a retained directory only after unique source/history and all assets are recoverable, dependencies point to canonical locations, PRs are reviewed/merged, local changes preserved and an independent verified backup exists. Do not merge automatically.

GitHub Projects coordinates issues/PRs; it is not submodule storage. The current credential lacks read:project/project scopes, so automatic association with Build an LLM From Scratch is blocked. Add the PRs manually or authorize appropriate access separately; repository permissions/protections/visibility remain unchanged.

## Recorded caveat

A planning-stage NumPy baseline test rewrote the generated logs/train.csv through its old default. The implementation-start CSV was preserved; the earlier CSV contents were not recovered. Tests now run in temporary working directories to prevent repeating this. There is no claim that every historical generated byte predating that baseline is recoverable.
