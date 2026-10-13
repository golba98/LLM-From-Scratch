"""Fixed locations inside the integration checkout, computed once.

Every module that needs the repository root imports it from here instead of
walking ``Path(__file__).parents``.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = REPO_ROOT.parent
SCRIPTS = REPO_ROOT / 'scripts'
TOOLS = REPO_ROOT / 'tools'
OUTPUTS = REPO_ROOT / 'outputs'
PYTHON = REPO_ROOT / '.venv/bin/python'
