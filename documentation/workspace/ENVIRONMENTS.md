# Historical and current Python environments

Codexa has two implementation contexts. Both use the **same interpreter and the same third-party packages** (`.venv`). They differ only in where the Codexa components (`llm_architecture`, `llm_inference`, `llm_training`, …) are imported from.

| Context | How it is selected | Codexa components come from | Use it for |
|---|---|---|---|
| **Current development** | `tools/codexa-dev …`, any script that imports `workspace_bootstrap`, the `codexa` CLI (`run.py`, `.venv/bin/codexa`), pytest via `conftest.py` | sibling `../LLM-*/src` (source first) | all new development, tests, inference and evaluation |
| **Historical** | `.venv/bin/python` without the bootstrap, with `--environment historical` for training-related operations | wheels installed in `.venv/lib64/python3.14/site-packages` (snapshot of 2026-10-10) | verifying `codexa-934m-foundation-5h-v1` and eligible exact recovery |

There is no second virtual environment, and `.venv` is never reinstalled. Its installed wheels are the historical implementation, hash-bound into the completed run's `config/job.json` `code_identity`.

## Why the source identity changed

The masked KV-cache repair (LLM-Architecture `bdf3683`, see `documentation/training/MASKED_KV_CACHE_REPAIR.md`) changed `llm_architecture/model.py` from SHA-256 `b7f857ad…` to `3d4e9d1a…`.

The completed run `codexa-934m-foundation-5h-v1` (1,792 updates; final checkpoint `ca779a16…`) recorded `b7f857ad…` in `code_identity`. Its frozen metadata is **not** rewritten:

- **Current context:** `code_identity` reports exactly one difference, `llm_architecture.model`. `verify_job` and resume would refuse here, which is the intended protection.
- **Historical context:** `code_identity` matches the job exactly.

## Selecting and checking a context

```bash
cd LLM-From-Scratch
tools/codexa-dev scripts/environment_identity.py --expect current        # repaired, source-first
.venv/bin/python scripts/environment_identity.py --expect historical     # frozen run identity
tools/codexa-dev -m llm_inference.cli.chat_native --help                 # corrected console-script equivalent
tools/codexa-dev                                                         # interactive, source-first
```

`scripts/environment_identity.py` is read-only. It reports:

- the interpreter and whether the bootstrap is active;
- the resolved `llm_architecture/model.py` and its SHA, classified as historical, repaired or unknown;
- for each component, whether it loads from source or the installed wheel, and which installed files differ from source;
- tokenizer identity and meta-device architecture compatibility;
- whether `code_identity` under this interpreter matches the frozen job.

`--expect` exits non-zero on a mismatch.

For training operations, use the construction-free preflight. It reads source
paths, SHA-256s, Git revisions, the tokenizer and the frozen job without
initializing a model or optimizer:

```bash
.venv/bin/python scripts/environment_identity.py --preflight-environment historical --operation verify
tools/codexa-dev scripts/environment_identity.py --preflight-environment development --operation inspect
tools/codexa-dev scripts/environment_identity.py --preflight-environment development --operation new
```

The last command currently fails because several executed owner source trees
have uncommitted changes. A future new job requires those source changes to be
reviewed, committed locally and reflected in `compatibility.json`. Once clean,
pass `--environment development` to its training entry point. Historical
selection cannot create a new job or resume the completed foundation job.

The removable `.venv` startup guard is installed and checked with
`.venv/bin/python tools/install_training_guard.py install|verify`. It changes
no installed package bytes. Normal direct component CLIs, `python -m`,
supervisor commands, and source-first scripts call the same preflight before
training begins. A missing or mismatched selection fails instead of falling
back to installed wheels. The existing `code_identity` and approval gates
remain in force after this environment gate.

## Entry points that intentionally stay historical

Each of these resolves the installed `b7f857ad…` model.

Runtime-verified on 2026-10-12:

- `.venv/bin/python` without the bootstrap;
- `.venv/bin/llm-inference-chat-native` and `.venv/bin/llm-inference-generate`;
- `scripts/training_supervisor.py`, `post_training_generation_diagnostic.py` and `bounded_chat_confirmation.py`.

Checked statically; they use the identical `sys.path.insert(0, ROOT / "src")` setup without the bootstrap:

- `scripts/training_wakeup.py`, `training_status.py`, `training_dashboard.py`, `pretraining_readiness.py`, `conversational_readiness.py`, `investigate_chat_regression.py`, `prepare_sft_feasibility.py`, `chat_training.py`;
- the other `.venv/bin/llm-*` console scripts.

The supervisor launches the trainer with `PYTHONPATH=src` only, so a supervised trainer also uses the installed wheels.

These paths continue to load historical packages for read-only verification. Training-capable commands require explicit environment selection and identity validation. For corrected inference, use the bootstrap scripts (`scripts/inference/chat_native.py`, `scripts/inference/generate.py`, `scripts/evaluation/evaluate_checkpoint.py`), the `codexa` CLI, or `tools/codexa-dev -m <module>`.

The `.venv/bin/codexa` console script is current: it launches its subcommands with a source-first `PYTHONPATH`.

Generation itself is unaffected by the repair. `llm_inference.generate` compacts prompts and passes no model mask, and the 48-prompt historical benchmark is token-identical in both contexts. The difference only matters for callers that pass `attention_mask` together with a KV cache.

## Reproducing the completed run's environment

1. Use `.venv/bin/python` without the bootstrap, and check it with `scripts/environment_identity.py --expect historical`.
2. The exact source is also preserved, read-only, in `outputs/preservation/historical-source/codexa-934m-foundation-5h-v1/`, which is private and git-ignored. It contains:
   - all 12 `code_identity` files;
   - the installed component wheels' trees;
   - the integration `src/codexa_workspace`;
   - `pip-freeze.txt` and `compatibility.json` at that time;
   - per-repo `HEAD` and working-tree diffs.

   Verify it with:

   ```bash
   .venv/bin/python scripts/environment_identity.py --verify-preservation \
       outputs/preservation/historical-source/codexa-934m-foundation-5h-v1
   ```

   10 of the 12 hash-bound files were never committed to Git (the training supervisor modules, five `llm_training` modules and `llm_data.token_views`). A separate private content-addressed archive now restores all 12 without `.venv`; see [recovery instructions](HISTORICAL_SOURCE_RECOVERY.md). Off-device backup remains pending.
3. Historical architecture source in Git: `git -C ../LLM-Architecture show c334620:src/llm_architecture/model.py`, SHA-256 `b7f857ad…`. `compatibility.json` records the historical component commits under `historical_identity`.
4. If `.venv` is ever damaged, rebuild a separate environment from `pip-freeze.txt`. Then copy `venv-components/` into its `site-packages` and re-run `--expect historical`. Do not rebuild `.venv` in place without preserving it first.

## Detecting stale installed packages

`environment_identity.py` lists `STALE vs source: N file(s)` for every component that loads from an installed wheel which differs from its source tree. In the historical context this is expected for LLM-Architecture (the repair) and LLM-Inference.

`codexa_workspace.chat_preparation.experiment_environment` applies the same installed-versus-source rule when freezing chat experiments.

## Rollback

- Installed package bytes are unchanged. The added `codexa_training_guard.pth`
  can be removed to roll back the local startup guard, but doing so reopens
  unsafe direct training paths and must fail the guard verification check.
- To remove the current-development tooling: delete `tools/codexa-dev` and `scripts/environment_identity.py`, or `git revert` the integration commit that added them.
- To return the current context to the pre-repair architecture, run `git -C ../LLM-Architecture revert bdf3683`. This also restores `b7f857ad…` in source, after which `code_identity` matches in both contexts.
- `compatibility.json` pins: the previous values are recorded under `historical_identity.component_commits`.
