"""SSD-only entry point for separately approved future supervisor operations.

Historical/frozen jobs retain their original launcher and run paths. This wrapper
does not prepare, approve, or launch anything unless the operator requests it.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codexa_workspace.assets import resolve_output


def build_command(arguments):
    if not arguments or arguments == ["--help"]:
        return [sys.executable, str(ROOT / "scripts/training_supervisor.py"), "--help"]
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
        from codexa_workspace.foundation_training import RUN_ID
        selected = "training/" + RUN_ID
    run_dir = resolve_output(ROOT, selected)
    expected = Path("/mnt/codexa/pytorch/runs").resolve()
    if run_dir != expected and expected not in run_dir.parents:
        raise ValueError("SSD launcher requires a run under /mnt/codexa/pytorch/runs; use the original launcher for historical jobs")
    return [sys.executable, str(ROOT / "scripts/training_supervisor.py"), *values,
            "--run-dir", str(run_dir)]


def main():
    command = build_command(sys.argv[1:])
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
