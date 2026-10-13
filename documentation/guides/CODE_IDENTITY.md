# Code identity of frozen jobs

A prepared training job is *frozen*: its `config/job.json` records a `code_identity`, a sha256 per file that decides what the job trains on and how it is evaluated and supervised. Approve, launch and resume recompute it and refuse on any difference. A moved file fails the check, and so does a shim left at the old path, because its bytes differ.

## What the foundation continuation binds

`CODE_FILES` in `src/codexa_workspace/training/foundation/continuation.py`:

| Area | Files |
| --- | --- |
| The job | `training/foundation/continuation.py`, `training/foundation/selection.py` |
| Gates and environment | `training/gates.py`, `training/environment.py`, `hashing.py` |
| Supervisor | `training/supervisor/{state,runner,cli}.py`, `scripts/training_supervisor.py`, `scripts/foundation_continuation.py` |
| Evaluation | `evaluation/capability/benchmark.py`, `evaluation/capability/suites/foundation_capability_v{1,2}.json`, `evaluation/language_suite.py`, `evaluation/generation.py` |
| Training guard | `tools/training_environment_guard.py` |
| Sibling components | the installed or source modules in `COMPONENT_MODULES` (`llm_training.pretrain_run`, `llm_architecture.model`, …) |

## Rules

- While a prepared job may still be launched or resumed, do not edit or move the files it binds. Put changes in new modules, or prepare a new job.
- Changing bound code, or moving to new sibling commits, invalidates prepared jobs. Prepare and freeze a fresh one (see [TRAINING_OPERATIONS.md](TRAINING_OPERATIONS.md)). Never edit a frozen job to make it match.
- **Committed-source gate:** the training guard refuses `new`/`resume` while `git status -- src scripts tools` is dirty, and while any sibling has uncommitted source or differs from its `compatibility.json` pin. Commit (with approval) before preparing, approving or launching.
- Paths used by things outside this repository stay stable:
  - `scripts/training_supervisor.py`: the systemd reconcile unit.
  - `scripts/training_dashboard.py`: the trainer viewer command.
  - `scripts/native_chat_bridge.py`: the Ubume CLI.
  - `tools/training_environment_guard.py`: the `.venv` startup hook and LLM-Training's tests.
  - `compatibility.json`, `run.py`, `VALIDATION.md`, `documentation/training/SESSION_DECISIONS.md`: the sibling READMEs and AGENTS files.

## Completed runs

The completed `codexa-934m-foundation-5h-v1` run bound integration files that no longer exist in this tree (removed in the 2026-10-13 restructure; git tag `pre-rewrite-2026-10-13`). Its exact source is preserved in the private, content-addressed archive:

```bash
python3 tools/historical_source_archive.py verify                       # check the archive
python3 tools/historical_source_archive.py verify --restore <new-dir>   # extract it
```

See [HISTORICAL_SOURCE_RECOVERY.md](../workspace/HISTORICAL_SOURCE_RECOVERY.md). The guard's `--environment historical` mode now reports that the source must be restored from this archive.
