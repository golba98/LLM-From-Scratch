#!/usr/bin/env python3
"""Deliver a run's notification/AI wakeup, reopen the agent window, or test a provider.

Implementation: ``codexa_workspace.notifications.cli``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace_bootstrap  # noqa: E402,F401  (component sources first, like tools/codexa-dev)
from codexa_workspace.notifications.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
