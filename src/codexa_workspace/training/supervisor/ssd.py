"""SSD-only front end to the supervisor CLI: refuses run directories outside the SSD output tree.

Historical runs keep their original paths and use ``scripts/training_supervisor.py`` directly.
"""
from pathlib import Path
import subprocess
import sys

from codexa_workspace.assets import resolve_output
from codexa_workspace.paths import REPO_ROOT as ROOT
from codexa_workspace.training.environment import SUPERVISOR_SCRIPT

SSD_RUNS = Path("/mnt/codexa/pytorch/runs")


def build_command(arguments):
    if not arguments or arguments == ["--help"]:
        return [sys.executable, str(SUPERVISOR_SCRIPT), "--help"]
    values = list(arguments)
    selected = None
    for index, value in enumerate(values):
        if value == "--run-dir":
            if index + 1 >= len(values):
                raise ValueError("--run-dir requires a path")
            selected = values[index + 1]
            del values[index:index + 2]
            break
        if value.startswith("--run-dir="):
            selected = value.split("=", 1)[1]
            del values[index]
            break
    if any(v == "--run-dir" or v.startswith("--run-dir=") for v in values):
        raise ValueError("Specify --run-dir only once")
    if selected is None:
        return [sys.executable, str(SUPERVISOR_SCRIPT), *values]   # reconcile: no run directory
    run_dir = resolve_output(ROOT, selected)
    expected = SSD_RUNS.resolve()
    if run_dir != expected and expected not in run_dir.parents:
        raise ValueError("SSD launcher requires a run under /mnt/codexa/pytorch/runs; use the original launcher for historical jobs")
    return [sys.executable, str(SUPERVISOR_SCRIPT), *values, "--run-dir", str(run_dir)]


def main(arguments: list[str] | None = None) -> int:
    command = build_command(sys.argv[1:] if arguments is None else arguments)
    return subprocess.run(command, cwd=ROOT).returncode

