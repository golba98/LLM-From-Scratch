"""Per-update training telemetry: the single implementation lives in ``llm_training.telemetry``.

The foundation trainer (``llm_training.pretrain_run``) and the SFT trainer share one
subsystem. Components may not import the integration package, so the implementation moved
into LLM-Training; this module is an alias, so ``codexa_workspace.training_telemetry`` and
``llm_training.telemetry`` are the same module object (patching one patches the other).
"""
import sys

from llm_training import telemetry as _implementation

sys.modules[__name__] = _implementation
