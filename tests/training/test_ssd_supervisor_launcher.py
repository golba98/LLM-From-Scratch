import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/training_supervisor_ssd.py"
    spec = importlib.util.spec_from_file_location("ssd_launcher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "resolve_output", lambda root, value:
                        Path(value) if Path(value).is_absolute()
                        else Path("/mnt/codexa/pytorch/runs") / value)
    return module


def test_explicit_ssd_run_preserves_operator_arguments(launcher):
    command = launcher.build_command(["validate", "--run-dir", "training/future", "--provider", "claude"])
    assert command[2:] == ["validate", "--provider", "claude", "--run-dir",
                           "/mnt/codexa/pytorch/runs/training/future"]


def test_historical_and_asset_paths_refused(launcher):
    for path in ("/home/k9-vortex/old-run", "/mnt/codexa/pytorch/saved/run"):
        with pytest.raises(ValueError, match="SSD launcher"):
            launcher.build_command(["resume", "--run-dir=" + path])


def test_duplicate_run_paths_refused(launcher):
    with pytest.raises(ValueError, match="only once"):
        launcher.build_command(["launch", "--run-dir=a", "--run-dir=b"])


def test_mount_failure_prevents_delegation(launcher, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("SSD unavailable")
    monkeypatch.setattr(launcher, "resolve_output", unavailable)
    with pytest.raises(RuntimeError, match="SSD unavailable"):
        launcher.build_command(["prepare", "--run-dir", "training/future"])
