"""Legacy entry point for llm_specialist.cli.specialist."""
import importlib
from pathlib import Path
import sys
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import workspace_bootstrap
_impl = importlib.import_module('llm_specialist.cli.specialist')
if __name__ == "__main__":
    raise SystemExit(_impl.main())
sys.modules[__name__] = _impl
