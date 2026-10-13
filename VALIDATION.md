# Validation

What to run before a PR, what results to expect, and what is not verified offline. All checks are bounded and synthetic: none of them trains a model, touches assets or contacts an AI service.

## Commands

From `LLM-From-Scratch`:

| Check | Command |
| --- | --- |
| Integration tests | `tools/codexa-dev -m pytest -q` |
| One sibling's tests | `python run.py --repo LLM-Data test -q` (any `LLM-*` name) |
| Specialist tests | `python run.py --profile specialist --repo LLM-Specialist test -q` |
| Package layering | `tools/codexa-dev -m pytest -q tests/infra` |
| Every sibling CLI answers `--help` | `tools/codexa-dev tools/check_cli.py` |
| Wheels build and contain no tests or data | `.venv/bin/python tools/build_packages.py` |
| Training-guard verdict (no model built) | `tools/codexa-dev scripts/environment_identity.py --preflight-environment development --operation new` |
| Historical 5h-run source archive | `python3 tools/historical_source_archive.py verify` |

## Expected results

- The integration suite passes, except tests that start real training code, which need the training guard to accept the workspace:
  - every sibling at its `compatibility.json` pin with no uncommitted source;
  - the repaired architecture;
  - a clean integration `src scripts tools`;
  - the encrypted SSD mounted, for the tokenizer.

  Otherwise they fail with `Codexa training environment rejected …` or `Explicit --environment … is required`. That reflects the workspace state, not the code under test.
- Mock backends and `CODEXA_EXECUTION_CLASS=unit_test` keep tests away from the real `notify-send`, `codex` and `claude`.

## Not verified offline

- Real GPU training, VRAM headroom and multi-hour supervision. These are exercised only by approved runs, whose evidence lives in their run directories and in `documentation/training/SESSION_DECISIONS.md`.
- Large-model generation quality, which needs genuine human review.
- An independent off-device backup of the asset store and the historical source archive.

The 2026-10-08 consolidation results previously recorded here are preserved at git tag `pre-rewrite-2026-10-13`.
