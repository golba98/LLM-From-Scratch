"""Where supervised training runs live, and how commands pick one when no ``--run-dir`` is given.

A run is any directory with ``supervisor/state.json`` under ``<root>/outputs/training`` or
under ``<output_root>/training`` (the encrypted SSD output tree when it is configured).
"""
from pathlib import Path

from codexa_workspace.assets import output_root
from codexa_workspace.paths import REPO_ROOT


def run_roots(root: Path = REPO_ROOT) -> list[Path]:
    """Directories that contain run directories, without duplicates."""
    roots = [Path(root) / 'outputs/training']
    try:
        roots.append(output_root(root) / 'training')
    except (ValueError, FileNotFoundError):
        pass  # no asset store configured (tests, fresh checkouts)
    unique = []
    for candidate in roots:
        if candidate.resolve() not in {r.resolve() for r in unique}:
            unique.append(candidate)
    return unique


def run_states(root: Path = REPO_ROOT) -> list[Path]:
    """Every ``supervisor/state.json`` in the run roots."""
    return sorted(path for base in run_roots(root) for path in base.glob('*/supervisor/state.json'))


def latest_run(root: Path = REPO_ROOT) -> Path | None:
    """The run whose supervisor state changed most recently."""
    states = run_states(root)
    return max(states, key=lambda path: path.stat().st_mtime).parents[1] if states else None


def resolve_run_dir(value: Path | None, root: Path = REPO_ROOT) -> Path:
    """An explicit ``--run-dir`` (absolute or relative to the repository), else the latest run."""
    if value is not None:
        return value if value.is_absolute() else Path(root) / value
    latest = latest_run(root)
    if latest is None:
        raise SystemExit('No training run found; pass --run-dir.')
    return latest
