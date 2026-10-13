from pathlib import Path

import pytest

from codexa_workspace.training.supervisor import ssd


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setattr(ssd, "resolve_output", lambda root, value:
                        Path(value) if Path(value).is_absolute()
                        else Path("/mnt/codexa/pytorch/runs") / value)
    return ssd


def test_explicit_ssd_run_preserves_operator_arguments(launcher):
    command = launcher.build_command(["run", "--run-dir", "training/future"])
    assert command[1] == str(ssd.SUPERVISOR_SCRIPT)
    assert command[2:] == ["run", "--run-dir", "/mnt/codexa/pytorch/runs/training/future"]


def test_historical_and_asset_paths_refused(launcher):
    for path in ("/home/k9-vortex/old-run", "/mnt/codexa/pytorch/saved/run"):
        with pytest.raises(ValueError, match="SSD launcher"):
            launcher.build_command(["run", "--run-dir=" + path])


def test_duplicate_run_paths_refused(launcher):
    with pytest.raises(ValueError, match="only once"):
        launcher.build_command(["run", "--run-dir=a", "--run-dir=b"])


def test_reconcile_needs_no_run_directory(launcher):
    assert launcher.build_command(["reconcile"])[2:] == ["reconcile"]


def test_mount_failure_prevents_delegation(launcher, monkeypatch):
    def unavailable(*args):
        raise RuntimeError("SSD unavailable")
    monkeypatch.setattr(launcher, "resolve_output", unavailable)
    with pytest.raises(RuntimeError, match="SSD unavailable"):
        launcher.build_command(["run", "--run-dir", "training/future"])
