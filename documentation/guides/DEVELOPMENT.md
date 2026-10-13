# Development guide

How the integration repository is laid out, how to run it, and where new code goes.

## Setup

The repository sits next to its seven sibling component repositories (see
[SIBLING_LAYOUT.md](../migration/SIBLING_LAYOUT.md)). It uses two existing environments, which you should never merge:

| Environment | Used for |
| --- | --- |
| `.venv` | Everything generative: training, inference, evaluation, tests |
| `.venv-specialist` | The frozen EmbeddingGemma specialist and its heads |

Run development code through **`tools/codexa-dev`**. It uses `.venv`'s third-party packages but loads every Codexa component from source (`../LLM-*/src`, then `src/`), so you always run what is checked out:

```bash
tools/codexa-dev -m pytest -q                       # the test suite
tools/codexa-dev scripts/training_status.py         # any script
tools/codexa-dev -m codexa_workspace.chat --describe
```

`python run.py …` does the same for the CLI workflow (`run.py test`, `run.py module <module>`, `run.py <sibling command>`). See [ENVIRONMENTS.md](../workspace/ENVIRONMENTS.md) for interpreter details and rebuilds.

## Layout

```text
src/codexa_workspace/
  paths.py, hashing.py          repository locations; content fingerprints
  sibling_commands.py           names that `run.py <name>` forwards to sibling CLIs
  assets/                       asset, input and output resolution (guarded SSD mount)
  evaluation/                   text metrics, chat probes, behavioural scoring, generation helpers,
                                the 24-concept language suite, capability/ benchmark + frozen suites
  training/
    gates.py                    production resource gates (single source of truth)
    environment.py              training-guard adapter, source-first search path
    runs.py                     where runs live; "latest run" default for operator commands
    identity.py                 which implementation an interpreter loads
    telemetry.py                per-update telemetry (alias of llm_training.telemetry)
    supervisor/                 state, runner, systemd units, CLI, SSD front end
    foundation/                 the foundation continuation job: data selection, job, CLI
  notifications/                run-end popup, AI wakeup, run summary, agent window
  monitoring/                   dashboard (reader → render → live app), window, status
  review/                       human review packages and the public review page
  chat.py, cli.py               catalog chat; the `run.py` / `codexa` runner
scripts/                        thin entry points (each calls one package `main`)
tests/                          mirrors src/codexa_workspace/; integration/ for cross-component tests
tools/                          codexa-dev, the training guard, packaging and archive tools
configs/                        model config, specialist config, asset catalog
```

### Layer rule

A module may import only from its own layer or lower ones. Imports inside functions count too. `tests/infra/test_layering.py` enforces this.

| Layer | Modules |
| --- | --- |
| 0 | `paths`, `hashing`, `sibling_commands` |
| 1 | `assets` |
| 2 | `evaluation` |
| 3 | `training` (gates, environment, runs, identity, telemetry, supervisor state/runner/systemd) |
| 4 | `notifications` |
| 5 | `monitoring`, `training.supervisor.cli`/`ssd` |
| 6 | `training.foundation`, `review`, `chat`, `cli` |

## Where new code goes

- **Reusable model, data, tokenizer, training or inference code** belongs in its owner sibling repository, never here.
- **Orchestration, evaluation and operator workflows** belong in the matching subpackage above. Use absolute `codexa_workspace…` imports, and take repository paths from `codexa_workspace.paths`.
- **A new operator command:** add `main(arguments=None)` to a package module, then a 14-line script in `scripts/` that calls it (copy any existing script).
- **A new training job:** follow `training/foundation/`. A frozen job document records a `code_identity` over the files that decide its behaviour (see [CODE_IDENTITY.md](CODE_IDENTITY.md)). The job must use `training.gates` and `training.environment`, and start `scripts/training_supervisor.py run` in a systemd unit.
- **Tests:** add them next to the module's mirror path under `tests/`. They must use fakes and mocks only (no real notifications, agents or training; `conftest.py` sets `CODEXA_EXECUTION_CLASS=unit_test`).

## Before a PR

```bash
tools/codexa-dev -m pytest -q          # full suite
tools/codexa-dev tools/check_cli.py    # every sibling CLI answers --help
.venv/bin/python tools/build_packages.py
```

Tests that start real training code pass only when the training guard accepts the workspace: all siblings at their `compatibility.json` pins, committed, with the repaired architecture. Elsewhere they fail with "Codexa training environment rejected". That is expected and is not a regression.
