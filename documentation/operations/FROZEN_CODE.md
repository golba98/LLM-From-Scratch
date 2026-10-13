# Frozen (hash-bound) code — do not move or edit while a job may resume

Prepared and approved jobs record a `code_identity`, which is a sha256 per file path. Approve, launch and resume recompute it and refuse on any mismatch. A moved file fails the check, and so does a shim left at the old path, because its bytes differ. Put new behaviour in **new modules** or in unhashed scripts instead.

## Hash-bound files (union of all current jobs)

| Bound by | Files |
| --- | --- |
| 5h foundation job (`foundation_training.CODE_FILES`) | `src/codexa_workspace/{foundation_training,training_supervisor,training_wakeup,foundation_diagnosis}.py` |
| 10h foundation continuation (`foundation_continuation.CODE_FILES`, pending job under `outputs/preparation/codexa-934m-foundation-10h/`) | `src/codexa_workspace/{foundation_continuation,capability_benchmark,foundation_training,foundation_diagnosis,sft_pilot_evaluation,training_supervisor}.py`, `src/codexa_workspace/benchmarks/foundation_capability_v{1,2}.json`, `scripts/training_supervisor.py`, `scripts/foundation_continuation.py`, `tools/training_environment_guard.py` |
| SFT pilot (`sft_pilot_runtime.CODE_FILES`) | `src/codexa_workspace/{sft_pilot_runtime,sft_pilot_trainer,sft_pilot_launch,sft_pilot_evaluation,sft_phase4_data,sft_feasibility,training_supervisor}.py`, `scripts/training_supervisor.py`, `tools/training_environment_guard.py` |
| Foundation diagnosis (`foundation_diagnosis.code_identity`, keyed by **absolute** path) | `foundation_diagnosis.py`, `chat_operations.py`, `qualification_review.py`, `conversational_readiness.py` (src), `scripts/conversational_readiness.py` |
| Installed components (`COMPONENT_MODULES`) | the `llm_training`, `llm_data`, `llm_architecture`, `llm_inference`, `llm_tokenizer` modules each job lists |

## Also fixed in place — do not move (paths referenced by frozen code, units or the venv)

- **Relative-import neighbours of frozen modules:** `assets`, `chat_evaluation`, `evaluation`, `pretraining_readiness`. Renaming any of them breaks a frozen import. Do not move any `src/codexa_workspace/` module; edit only modules outside the lists above, and re-check those lists first.
- **`tools/`:** `.venv/.../codexa_training_guard.pth` hard-codes the absolute `tools/` path.
- **Scripts that stay at `scripts/` top level:**
  - `training_supervisor.py`: systemd `codexa-train-reconcile.service` ExecStart.
  - `training_dashboard.py`: `LLM_TRAINING_VIEWER_COMMAND` in `launch.json`.
  - `environment_identity.py`, `prepare_sft_feasibility.py`, `chat_training.py`: loaded by path from hash-bound src.
  - `foundation_capability_probe.py`, `build_capability_suite_v2.py`: named inside the frozen benchmark JSON.
  - `foundation_continuation.py`, `conversational_readiness.py`: hash-bound (see table). `pretraining_readiness.py`: its hash is recorded in readiness evidence (`preparation_source_hashes`).
  - `training_status.py`, `training_wakeup.py`, `training_supervisor_ssd.py`: documented operator commands.
  - `train.py`, `train_conversational_sft.py`, `run_chat_overfit.py`, `sft_readiness.py`: flat lookup by `run.py`/`codexa` and the installed `.venv/bin/codexa` copy.
- **Tests that gates run by path** (stay in `tests/` top level): `test_training_supervisor`, `test_pretraining_readiness`, `test_foundation_continuation`, `test_capability_benchmark`, `test_run_reporting`, `test_chat_preparation`, `test_post_training_generation_diagnostic`.
- **Root files:** `run.py`, `workspace_bootstrap.py`, `conftest.py`, `compatibility.json`, `pyproject.toml`, `artifacts*.json`, `artifact-catalog/`, `configs/`, `outputs/`.

## Committed-source gate

`foundation_continuation` and `sft_readiness` refuse to proceed while `git status -- src scripts tools` is dirty. The training environment guard likewise rejects uncommitted integration or LLM-Architecture source under the plain `.venv` interpreter. Commit (with user approval) before preparing or approving a job.
