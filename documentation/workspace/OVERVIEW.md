# Codexa PyTorch workspace

Exactly eight LLM Git working trees live here: LLM-Architecture, LLM-Data, LLM-Tokenizer, LLM-Training, LLM-Inference, LLM-Memory, LLM-Specialist and LLM-From-Scratch. Integration code (training jobs, supervision, evaluation, operator tools) and the central documentation live in LLM-From-Scratch. Reusable code lives in its component package. There is no enclosing Git repository.

Run `cd LLM-From-Scratch` then `tools/codexa-dev -m pytest -q`; start with LLM-From-Scratch/README.md and documentation/guides/DEVELOPMENT.md. Repository dependencies and exact Git pins are in LLM-From-Scratch/compatibility.json. Shared recovery/build/validation evidence lives in workspace-infrastructure; its README explains ownership and exclusions. External historical weights/data stay in the configured protected asset store.

The 2026-10-09 sibling restructure is saved on the repositories’ open review branches. Original 31-LLM and 32-LLM projects are retained; independent backup remains a retirement gate. See LLM-From-Scratch/documentation/migration/SIBLING_LAYOUT.md and workspace-infrastructure/audit-2026-10-09/PLAN.md.
