# scripts/

Operator entry points. Each script is about 15 lines: it puts the component sources first (`workspace_bootstrap`) and calls one `main` in the package, where all the logic lives. Run them directly (`scripts/<name>.py …`), through `tools/codexa-dev`, or as `python run.py <name> …`. Commands that act on a run default to the most recently active run; pass `--run-dir` for another.

| Script | What it does | Implementation |
| --- | --- | --- |
| `training_status.py` | Instant run status (`--json`, `--watch`) | `monitoring.status` |
| `training_dashboard.py` | Live read-only 120x30 dashboard (`--once`, `--compact`, `--kitty`) | `monitoring.dashboard` |
| `training_supervisor.py` | `run` one prepared run under OS supervision; `reconcile` after a restart | `training.supervisor.cli` |
| `training_supervisor_ssd.py` | The supervisor, restricted to run directories on the SSD output tree | `training.supervisor.ssd` |
| `training_wakeup.py` | Deliver a run's notification, reopen the agent window, test a provider | `notifications.cli` |
| `foundation_continuation.py` | Foundation continuation job: baseline, freeze, readiness, approve, launch, resume | `training.foundation.cli` |
| `environment_identity.py` | Which Codexa implementation this interpreter loads; training-guard preflight | `training.identity` |
| `qualification_review.py` | Build human review packages or import reviewer decisions | `review.qualification` |
| `review_dashboard.py` | Publish the public-safe human review page | `review.dashboard` |
| `native_chat_bridge.py` | Native chat bridge used by the Ubume CLI | `llm_inference.cli.native_chat_bridge` |

Sibling-package commands (data preparation, tokenizer training, generation, export, …) have no script here. Run them by name with `python run.py <name>` (the table is in `src/codexa_workspace/sibling_commands.py`), or with `python -m llm_<component>.cli.<name>`.

Paths that outside tools rely on (the systemd reconcile unit, the trainer viewer command, Ubume) must not move; see [CODE_IDENTITY.md](../documentation/guides/CODE_IDENTITY.md).
