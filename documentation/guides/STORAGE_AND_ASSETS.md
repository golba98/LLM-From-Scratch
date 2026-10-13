# Storage and assets

Current state as of 2026-10-13. The dated history at the end explains how it got here.

## Where things live now

| What | Location | Notes |
| --- | --- | --- |
| Model/data inputs (canonical) | `/mnt/codexa/pytorch/saved` (`asset_root`) | Manually unlocked encrypted SSD. `artifacts.local.json` pins its filesystem/LUKS UUIDs with `asset_mount`; resolvers refuse to run if the expected mount is missing. |
| New run outputs | `/mnt/codexa/pytorch/runs` and `/mnt/codexa/numpy/runs` (`output_root`) | Choose explicit new paths here for separately approved jobs. |
| Historical workspace outputs | `/mnt/codexa/pytorch/workspace-outputs` | Read-only (`read_only_asset_roots`); `input_aliases` map old `outputs/...` paths here. |
| Originals, datasets, logs, recovery | `3-Python/31-LLM Source/` (`checkpoints/`, `exports/`, `data/`, `logs/`, `training/`, `recovery/`) | Read-only; `recovery_root` and `reference_source` still point here. Dataset originals remain protected on Fedora. |
| NumPy legacy folders | `32-LLM Source/legacy/` | Active NumPy work is in workspace 38. |
| Compatibility links | `Development/LLM-Assets-Staging` → `31-LLM Source/compatibility/LLM-Assets-Staging` | Legacy asset-path links for existing tools. Keep them. |
| Migration evidence and rollback | `/home/k9-vortex/Documents/codexa-migration` (incl. `rollback/`), SSD `manifests/` | Private. Never commit. |

Configuration: copy `artifacts.local.example.json` to the ignored `artifacts.local.json`, or export `CODEXA_ASSET_ROOT`. Never fall back to retired project directories.

## Rules

- Never commit weights, datasets, private logs, environments, source snapshots, credentials, recovery bundles or migration evidence.
- Historical assets are read-only inputs. New experiments need distinct output paths under `output_root` or an explicitly configured output root.
- `scripts/training_supervisor_ssd.py` is the supervisor CLI restricted to run directories inside the SSD output tree. Note that the foundation continuation job still places its run under the repository's `outputs/training/`; choose an SSD run directory when a new job is prepared.
- Completed runs keep their recorded paths and prepared-data references; these are retained, not certified SSD-only. Their source is preserved in the historical archive (see [CODE_IDENTITY.md](CODE_IDENTITY.md)).
- Do not follow dataset/log/recovery navigation links during copy or label operations.
- Keep the original 31/32 Source trees, legacy compatibility links and original checkpoints until independent external backup, restore verification and separate user authorization permit retirement. Selective retirement so far is **not** permission to prune remaining history.
- Do not repair or restart the legacy inference service (`codexa-lmstudio.service`) as part of storage work.

## History

- **2026-10-09 — consolidation.** All saved PyTorch weights, datasets, logs, training records and recovery archives were gathered under `31-LLM Source/`. `LLM-Assets-Staging` became a compatibility symlink. The NumPy deprecated/local-artifact folders moved to `32-LLM Source/legacy/`.
- **2026-10-13 — SSD asset copy.** Checkpoint/export weights and their tokenizer/config/index/checksum companions were copied to `/mnt/codexa`. Local settings and the native registry were backed up to the rollback directory first. The originals stayed in place.
- **2026-10-13 — dataset cutover and selective retirement.** Canonical inputs now resolve on the SSD, and output roots moved to `/mnt/codexa/*/runs`.
  - Six explicitly authorized redundant original payloads were removed after fresh hash, recovery-source and dependency checks. Each retired path now links to its unchanged SSD copy.
  - Four legacy navigation links were retargeted to SSD trees.
  - Eight NumPy smoke aliases now resolve to SSD files and passed restore plus CPU inference.
  - Mappings and transaction records are in the migration directory.
  - Independent external backup and restore verification is **not** complete.
