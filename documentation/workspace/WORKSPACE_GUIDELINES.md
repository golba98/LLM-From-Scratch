# Workspace guidelines

Eight independent sibling repositories; integration is LLM-From-Scratch. Read each owner's AGENTS.md before editing. Shared local recovery and validation infrastructure is workspace-infrastructure. Never commit private evidence or assets. Keep original 31 and 32 projects until completeness and independent backup verification. No commit, push, repository deletion or PR without explicit approval. Bounded synthetic/offline checks only during refactors.

Training runs: monitor with `LLM-From-Scratch/scripts/training_dashboard.py` (read-only Rich dashboard), never poll with an AI, and follow LLM-From-Scratch/AGENTS.md and its `documentation/operations/` sub files (TRAINING_OPERATIONS: supervisor, resume rules, GPU headroom; NOTIFICATIONS; FROZEN_CODE: hash-bound files; STORAGE_AND_ASSETS).

Development audit: Claude Code and Codex sessions started from the workspace root are audited by hooks (`.claude/settings.json`, `.codex/hooks.json`); a turn cannot end with undocumented changes. After each completed logical unit of work run `./audit record ...` (`./audit status` shows what is pending; `./audit history` is the shared Claude/Codex history in `development-log/`). Never edit `development-log/` directly; git pre-commit/pre-push hooks require records. Details: `workspace-infrastructure/development_audit/README.md`.
