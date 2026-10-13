# Codexa v1 — PyTorch integration repository

Codexa is a 934M-parameter language model built from scratch in PyTorch. Its reusable code lives in seven sibling repositories. **This repository ties them together**: it holds the training jobs and their OS-level supervision, the evaluation suites, run notifications and monitoring, and the operator scripts. It also keeps the decision log for the whole project.

| Sibling repository | Responsibility |
| --- | --- |
| LLM-Architecture | Native Transformer, RMSNorm, SwiGLU, learned/RoPE positions, KV cache, checkpoint readers |
| LLM-Tokenizer | Byte-level BPE and conversation/SFT serialization |
| LLM-Data | Corpus provenance, preparation, packing and token data |
| LLM-Training | Base/SFT training, optimizer state, restoration and telemetry |
| LLM-Inference | Native completion, chat and exports |
| LLM-Memory | Scoped SQLite persistence, retrieval and worker client |
| LLM-Specialist | Frozen EmbeddingGemma 2 and classifier heads |

Exact sibling Git pins are in `compatibility.json`. See [SIBLING_LAYOUT.md](documentation/migration/SIBLING_LAYOUT.md) to check out a workspace.

## Quick start

```bash
cd LLM-From-Scratch
tools/codexa-dev -m pytest -q                          # tests, with component sources first
python run.py --repo LLM-Architecture test -q          # a sibling's tests
scripts/training_status.py                             # status of the latest training run
scripts/training_dashboard.py                          # live read-only dashboard
python run.py generate --help                          # any sibling CLI by name
python run.py module codexa_workspace.chat --describe  # chat with the catalog model
```

Use the existing `.venv` (generative) and `.venv-specialist` (specialist) environments, and never merge them. Assets resolve through the ignored `artifacts.local.json` (copy `artifacts.local.example.json`) or `CODEXA_ASSET_ROOT`. Weights, datasets, private logs and recovery bundles are never committed.

## Repository layout

```text
src/codexa_workspace/      the integration package (layered; see the development guide)
  assets/                  asset, input and output resolution on the guarded SSD
  evaluation/              metrics, language suite, generation helpers, capability benchmark
  training/                gates, guard adapter, runs, supervisor/, foundation/ (the current job)
  notifications/           run-end popup, AI wakeup, run summary, agent window
  monitoring/              dashboard, window and status
  review/                  human review packages and the public review page
scripts/                   thin operator entry points (see scripts/README.md)
tests/                     mirrors the package; integration/ holds cross-component tests
tools/                     codexa-dev, the training guard, packaging and archive tools
configs/                   model config, specialist config, asset catalog
documentation/             guides, reference, reports, live logs (documentation/README.md)
outputs/                   local run state (ignored)
```

## Read next

- [documentation/guides/DEVELOPMENT.md](documentation/guides/DEVELOPMENT.md): setup, layout, layer rules, where new code goes
- [documentation/guides/TRAINING_OPERATIONS.md](documentation/guides/TRAINING_OPERATIONS.md): preparing, launching and watching a run
- [documentation/README.md](documentation/README.md): map of every document
- [AGENTS.md](AGENTS.md): rules for AI agents working in this repository
- [VALIDATION.md](VALIDATION.md): what to run before a PR

Removed history (old experiments, adapters, archived reports) is preserved at git tag `pre-rewrite-2026-10-13`.
