#!/usr/bin/env python3
"""Publish a public-safe human review dashboard.

Implementation: ``codexa_workspace.review.dashboard``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace_bootstrap  # noqa: E402,F401  (component sources first, like tools/codexa-dev)
from codexa_workspace.review.dashboard import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
