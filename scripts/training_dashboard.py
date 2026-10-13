#!/usr/bin/env python3
"""Live read-only training dashboard (120x30); closing it never affects training.

Implementation: ``codexa_workspace.monitoring.dashboard``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace_bootstrap  # noqa: E402,F401  (component sources first, like tools/codexa-dev)
from codexa_workspace.monitoring.dashboard import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
