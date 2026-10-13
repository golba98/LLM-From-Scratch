# LLM-From-Scratch — agent guide

The canonical PyTorch integration repository. It pins seven independent sibling component repositories through `compatibility.json` (see [README.md](README.md)). Keep this file short. Details live in the linked sub files.

## Hard rules

**Code ownership**
- Reusable implementations belong in their owner repositories. Integration and evaluation orchestration belongs in `src/codexa_workspace`.
- Generic `src/*` modules and script wrappers are temporary compatibility adapters.
- Never import the integration package from a component, and never introduce reverse or circular package dependencies.

**Environments**
- Use `.venv` for generative work and `.venv-specialist` for the frozen encoder/heads.
- For development and tests, run through `tools/codexa-dev` (source-first components).

**Assets**
- Assets resolve through `CODEXA_ASSET_ROOT` or the ignored `artifacts.local.json`. Never fall back to retired project directories.
- Historical assets are read-only inputs. New experiments need distinct output paths. See [storage](documentation/operations/STORAGE_AND_ASSETS.md).

**Never commit**
- Weights, datasets, private logs, environments, source snapshots, credentials, recovery bundles or migration evidence.

**Compatibility to preserve**
- State-dictionary keys, NPZ/native checkpoint dialects, tokenizer IDs and hashes, model lineage, RNG/optimizer restoration and memory scopes.

**Frozen code**
- Do not move or edit hash-bound files while any job may resume. Put changes in new modules instead. Full list: [FROZEN_CODE.md](documentation/operations/FROZEN_CODE.md).

**Refactors**
- Run bounded synthetic tests only. No expensive training runs.
- Base/SFT operator launches must have a visible viewer. Supervisor launches attach the 120x30 dashboard automatically; direct and recipe launches attach `llm_training`'s 100x22 Kitty viewer.

**Logging**
- Append dated decisions and evidence to `documentation/training/SESSION_DECISIONS.md`.
- Append milestones to `documentation/training/100M_PROGRESS_LOG.md`. Never create a repository-root copy.

**Git and PRs**
- Run the relevant regression suites, wheel and CLI checks before PRs.
- Stage source changes and intentional sibling compatibility pins only.
- No commit, push or PR without explicit approval. Never force-push, merge PRs, alter repository permissions/protection, or remove pending-retirement directories without meeting the documented gates.

## Where things are

| Need | Go to |
| --- | --- |
| Monitor, launch, resume or extend a training run | [documentation/operations/TRAINING_OPERATIONS.md](documentation/operations/TRAINING_OPERATIONS.md) |
| Run-end popups, AI wakeup, agent window | [documentation/operations/NOTIFICATIONS.md](documentation/operations/NOTIFICATIONS.md) |
| Where weights, data and outputs live (SSD `/mnt/codexa`) | [documentation/operations/STORAGE_AND_ASSETS.md](documentation/operations/STORAGE_AND_ASSETS.md) |
| Files that must not move or change | [documentation/operations/FROZEN_CODE.md](documentation/operations/FROZEN_CODE.md) |
| Supervisor design and 934M run history | [documentation/operations/CODEXA_934M_TRAINING_SUPERVISOR.md](documentation/operations/CODEXA_934M_TRAINING_SUPERVISOR.md) |
| Map of all documentation (current vs. archived) | [documentation/README.md](documentation/README.md) |
| Which script does what | [scripts/README.md](scripts/README.md) |
| Interpreters and environment rebuilds | [documentation/workspace/ENVIRONMENTS.md](documentation/workspace/ENVIRONMENTS.md) |

## Quick reference

- Status: `scripts/training_status.py [--json]`. Dashboard: `.venv/bin/python scripts/training_dashboard.py` (read-only; never poll runs with an AI).
- Supervisor: `scripts/training_supervisor.py prepare → approve → launch`, `resume --confirm-run`, `extend ...`. For new SSD output roots use `scripts/training_supervisor_ssd.py`.
- Tests: `tools/codexa-dev -m pytest -q`.
- GPU: the 934M trainer needs about 13 GiB of the 16 GiB card. Quit GPU apps before and during runs.
