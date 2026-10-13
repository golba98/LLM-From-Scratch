#!/usr/bin/env python3
"""Native chat bridge for the Ubume CLI (which runs this exact path).

Implementation: ``llm_inference.cli.native_chat_bridge``, loaded source-first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace_bootstrap  # noqa: E402,F401  (sibling component sources first)
from llm_inference.cli.native_chat_bridge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
