# LLM-From-Scratch — agent guide

The PyTorch integration repository for Codexa. It pins seven sibling component repositories through `compatibility.json`. Start with [README.md](README.md) and [the development guide](documentation/guides/DEVELOPMENT.md). Keep this file short; details live in the linked guides.

## Hard rules

**Code**
- Reusable model, data, tokenizer, training or inference code belongs in its owner sibling repository. This repository holds orchestration, evaluation and operator workflows in `src/codexa_workspace`.
- Respect the package layers. `tests/infra/test_layering.py` enforces them: no upward imports and no relative imports.
- Components never import `codexa_workspace`. Never introduce circular package dependencies.
- Scripts in `scripts/` stay thin: each calls one package `main`.

**Environments**
- Use `.venv` for generative work and `.venv-specialist` for the frozen encoder/heads. Never merge or reinstall them; `.venv`'s installed wheels are the preserved historical implementation.
- Run development code and tests through `tools/codexa-dev`.

**Assets**
- Assets resolve through `CODEXA_ASSET_ROOT` or the ignored `artifacts.local.json`. Never fall back to retired project directories.
- Historical assets are read-only. New runs need new output paths. See [STORAGE_AND_ASSETS.md](documentation/guides/STORAGE_AND_ASSETS.md).

**Never commit**
- Weights, datasets, private logs, environments, source snapshots, credentials, recovery bundles or migration evidence.

**Compatibility to preserve**
- State-dictionary keys, NPZ/native checkpoint dialects, tokenizer IDs and hashes, model lineage, RNG/optimizer restoration and memory scopes.

**Training**
- Never poll a run with an AI, never stop or restart a live supervisor, never lower a gate, and never start training on your own.
- A prepared job binds its code ([CODE_IDENTITY.md](documentation/guides/CODE_IDENTITY.md)): don't edit bound files while it may launch or resume.
- During refactors, run bounded synthetic tests only.

**Logging**
- Append dated decisions and evidence to `documentation/training/SESSION_DECISIONS.md`.
- Append milestones to `documentation/training/100M_PROGRESS_LOG.md`.

**Git and PRs**
- No commit, push or PR without explicit approval. No `Co-Authored-By` lines or tool footers.
- Run the checks in [VALIDATION.md](VALIDATION.md) first. Stage source changes and intentional sibling pins only.
- Never force-push or merge PRs.

## Where things are

| Need | Go to |
| --- | --- |
| Layout, layers, where new code goes | [guides/DEVELOPMENT.md](documentation/guides/DEVELOPMENT.md) |
| Prepare, launch, resume or monitor a run | [guides/TRAINING_OPERATIONS.md](documentation/guides/TRAINING_OPERATIONS.md) |
| Run-end popups, AI wakeup, agent window | [guides/NOTIFICATIONS.md](documentation/guides/NOTIFICATIONS.md) |
| Where weights, data and outputs live | [guides/STORAGE_AND_ASSETS.md](documentation/guides/STORAGE_AND_ASSETS.md) |
| What a frozen job binds; stable external paths | [guides/CODE_IDENTITY.md](documentation/guides/CODE_IDENTITY.md) |
| Interpreters and the training guard | [workspace/ENVIRONMENTS.md](documentation/workspace/ENVIRONMENTS.md) |
| Every document | [documentation/README.md](documentation/README.md) |
| Every script | [scripts/README.md](scripts/README.md) |
