#!/usr/bin/env python3
"""Report which Codexa implementation this interpreter context loads.

Implementation: ``codexa_workspace.training.identity``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codexa_workspace.training.identity import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
