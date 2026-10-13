# Historical and current Python environments

Codexa has two implementation contexts. Both use the **same interpreter and third-party packages** (`.venv`). They differ only in where the Codexa components (`llm_architecture`, `llm_training`, …) are imported from.

| Context | Selected by | Codexa components come from | Use it for |
|---|---|---|---|
| **Current development** | `tools/codexa-dev …`, every operator script in `scripts/` (they import `workspace_bootstrap`), `python run.py …`, pytest via `conftest.py` | sibling `../LLM-*/src` (source first) | all development, tests, training, inference and evaluation |
| **Historical** | `.venv/bin/python` without the bootstrap (for example `scripts/environment_identity.py`, the `.venv/bin/llm-*` console scripts) | wheels installed in `.venv/lib64/python3.14/site-packages` (snapshot of 2026-10-10) | inspecting what the completed 5h run was built from |

There is no second virtual environment, and `.venv` is never reinstalled: its installed wheels are the historical implementation, preserved byte-for-byte. That includes an old `codexa_workspace` copy behind `.venv/bin/codexa`, so use `python run.py` for the current CLI.

## Checking a context

```bash
tools/codexa-dev scripts/environment_identity.py --expect current        # repaired, source-first
.venv/bin/python scripts/environment_identity.py --expect historical     # installed historical components
```

`scripts/environment_identity.py` is read-only. It reports:
- the interpreter and whether the bootstrap is active;
- the resolved `llm_architecture/model.py`: its SHA, and whether it is historical (`b7f857ad…`), repaired or unknown;
- for each component, whether it loads from source or the installed wheel, and which installed files differ from source (`STALE vs source`);
- tokenizer identity and meta-device model compatibility against the foundation run.

`--expect` exits non-zero on a mismatch.

## The training guard

`tools/training_environment_guard.py` is loaded at interpreter start by `.venv/.../codexa_training_guard.pth`. Install or verify it with `.venv/bin/python tools/install_training_guard.py install|verify`; it changes no installed package bytes.

- Before training starts, it requires an explicit `CODEXA_TRAINING_ENVIRONMENT` selection. It intercepts `-m llm_training.cli.train*`, `-m codexa_workspace.training.foundation.continuation` and `scripts/training_supervisor.py run`.
- **Development** selection requires that every sibling is at its `compatibility.json` pin, has no uncommitted source, and is imported from source, plus the repaired architecture and a clean integration `src scripts tools`.
- **Historical** selection reports that the completed run's integration source must be restored from the archive (below). It can never create or resume a job.

Construction-free check (reads paths, hashes, revisions and the tokenizer; builds no model):

```bash
tools/codexa-dev scripts/environment_identity.py --preflight-environment development --operation new
```

## Reproducing the completed 5h run

1. Its exact source (all 12 `code_identity` files, the installed component trees, the integration package, `pip-freeze.txt`) is in the private content-addressed archive. Restore and verify it with `tools/historical_source_archive.py`; see [HISTORICAL_SOURCE_RECOVERY.md](HISTORICAL_SOURCE_RECOVERY.md).
2. Historical architecture source in Git: `git -C ../LLM-Architecture show c334620:src/llm_architecture/model.py` (SHA-256 `b7f857ad…`). `compatibility.json` records the historical component commits under `historical_identity`.
3. If `.venv` is ever damaged, build a *separate* environment from `pip-freeze.txt` and copy the archive's `venv-components/` into its `site-packages`. Never rebuild `.venv` in place without preserving it first.

The masked KV-cache repair ([report](../reports/MASKED_KV_CACHE_REPAIR.md)) is why the two contexts differ for `llm_architecture.model`. Generation itself is unaffected: `llm_inference.generate` passes no model mask, and the 48-prompt historical benchmark is token-identical in both contexts.

## Rollback

- Removing `codexa_training_guard.pth` rolls back the startup guard. That reopens unsafe direct training paths and must fail the guard verification check.
- `compatibility.json` keeps the previous pins under `historical_identity.component_commits`.
