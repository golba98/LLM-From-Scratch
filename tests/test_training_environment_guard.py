"""Training identity gates use file evidence only; never build a model or optimizer."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import runpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import training_environment_guard as gate


def run(*args, env=None):
    if env is None:
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "CODEXA_WORKSPACE_ROOT", "CODEXA_TRAINING_ENVIRONMENT")}
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)


def test_historical_verification_and_development_inspection():
    historical = run(str(ROOT / ".venv/bin/python"), "scripts/environment_identity.py",
                     "--preflight-environment", "historical", "--operation", "verify")
    assert historical.returncode == 0, historical.stderr
    assert json.loads(historical.stdout)["valid"]
    development = run(str(ROOT / "tools/codexa-dev"), "scripts/environment_identity.py",
                      "--preflight-environment", "development", "--operation", "inspect")
    assert development.returncode == 0, development.stderr
    report = json.loads(development.stdout)
    assert report["valid"] and report["architecture"]["repaired"]


def test_missing_and_historical_new_job_fail_at_python_startup():
    base = [str(ROOT / ".venv/bin/python"), "-m", "llm_training.cli.train"]
    missing = run(*base, "--config", "nonexistent")
    assert missing.returncode == 2 and "Explicit --environment" in missing.stderr
    wrong = run(*base, "--environment", "historical", "--config", "nonexistent")
    assert wrong.returncode == 2 and "New training jobs require" in wrong.stderr
    assert "FileNotFoundError" not in (missing.stderr + wrong.stderr)


def test_supervisor_and_console_script_cannot_bypass_gate():
    supervisor = run(str(ROOT / ".venv/bin/python"), "scripts/training_supervisor.py",
                     "launch", "--run-dir", "/tmp/codexa-no-run")
    console = run(str(ROOT / ".venv/bin/llm-training-train"), "--config", "nonexistent")
    assert supervisor.returncode == 2 and "Explicit --environment" in supervisor.stderr
    assert console.returncode != 0 and "Explicit --environment" in console.stderr
    assert "FileNotFoundError" not in (supervisor.stderr + console.stderr)


def test_completed_historical_job_cannot_resume():
    report = gate.inspect("historical", "resume")
    # The pytest bootstrap loads source, so this context also fails code
    # identity. The completed-job blocker must be independently present.
    assert any("completed and cannot resume" in error for error in report["errors"])


def test_swapped_environment_selection_is_rejected():
    historical_as_development = run(str(ROOT / ".venv/bin/python"), "scripts/environment_identity.py",
                                    "--preflight-environment", "development", "--operation", "inspect")
    development_as_historical = run(str(ROOT / "tools/codexa-dev"), "scripts/environment_identity.py",
                                    "--preflight-environment", "historical", "--operation", "verify")
    assert historical_as_development.returncode == 2
    assert development_as_historical.returncode == 2
    assert "architecture" in historical_as_development.stdout
    assert "historical" in development_as_historical.stdout


def test_stale_installed_architecture_is_detected(tmp_path):
    package = tmp_path / "llm_architecture"
    package.mkdir()
    (package / "__init__.py").write_text("")
    original_model = ROOT / ".venv/lib/python3.14/site-packages/llm_architecture/model.py"
    (package / "model.py").write_bytes(original_model.read_bytes() + b"\n# stale wheel\n")
    code = ("import json,sys,training_environment_guard as g; "
            "sys.path.insert(0,sys.argv[1]); "
            "print(json.dumps(g.inspect('historical','verify')['errors']))")
    result = run(str(ROOT / ".venv/bin/python"), "-c", code, str(tmp_path))
    assert result.returncode == 0
    assert "historical llm_architecture.model" in result.stdout


def test_changed_architecture_and_import_path_fail_without_importing_model(tmp_path, monkeypatch):
    source = ROOT.parent / "LLM-Architecture/src/llm_architecture/model.py"
    fake = tmp_path / "llm_architecture"
    fake.mkdir()
    shutil.copyfile(source, fake / "model.py")
    (fake / "model.py").write_bytes(source.read_bytes() + b"\n# changed architecture\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    imported_before = sys.modules.get("llm_architecture.model")
    report = gate.inspect("development")
    assert not report["valid"]
    assert any("architecture" in error for error in report["errors"])
    assert sys.modules.get("llm_architecture.model") is imported_before


def _isolated_workspace(tmp_path, monkeypatch, *, dirty_component, dirty_integration):
    """A real, separate Git workspace, so the check never depends on this checkout's state."""
    def commit(repo, *paths):
        for path in paths:
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=fixture", "-c", "user.email=fixture@invalid",
                        "commit", "-q", "--no-verify", "-m", "fixture"], check=True)
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    root, component = tmp_path / "LLM-From-Scratch", tmp_path / "LLM-Architecture"
    head = commit(component, "src/llm_architecture/__init__.py")
    commit(root, "src/codexa_workspace/__init__.py")
    (root / "compatibility.json").write_text(json.dumps({"components": {"LLM-Architecture": {
        "path": "../LLM-Architecture", "commit": head}}}))
    if dirty_component:
        (component / "src/llm_architecture/uncommitted.py").write_text("")
    if dirty_integration:
        (root / "scripts").mkdir()
        (root / "scripts/uncommitted.py").write_text("")
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "WORKSPACE", tmp_path)


def _forbid_model_imports(monkeypatch):
    def fail_import(name, *args, **kwargs):
        if name.startswith(("torch", "llm_architecture")):
            raise AssertionError("preflight imported model or optimizer code")
        return original(name, *args, **kwargs)
    import builtins
    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", fail_import)


def test_dirty_source_blocks_new_development_job_without_initialization(tmp_path, monkeypatch):
    _isolated_workspace(tmp_path, monkeypatch, dirty_component=True, dirty_integration=True)
    _forbid_model_imports(monkeypatch)
    report = gate.inspect("development", "new")
    assert not report["valid"]
    assert "LLM-Architecture has uncommitted executed source" in report["errors"]
    assert "Integration training source has uncommitted changes" in report["errors"]


def test_clean_isolated_source_raises_no_uncommitted_error(tmp_path, monkeypatch):
    """Positive control: the same fixture, committed, produces no dirty-source error."""
    _isolated_workspace(tmp_path, monkeypatch, dirty_component=False, dirty_integration=False)
    _forbid_model_imports(monkeypatch)
    report = gate.inspect("development", "new")
    assert not any("uncommitted" in error for error in report["errors"])
    assert report["component_revisions"]["LLM-Architecture"]["expected"] == \
        report["component_revisions"]["LLM-Architecture"]["detected"]


def test_guard_installation_is_required_for_direct_cli():
    checked = run(str(ROOT / ".venv/bin/python"), "tools/install_training_guard.py", "verify")
    assert checked.returncode == 0
    assert "guard installed and verified" in checked.stdout


def test_training_help_explains_environment_selection():
    direct = run(str(ROOT / ".venv/bin/python"), "-m", "llm_training.cli.train", "--help")
    console = run(str(ROOT / ".venv/bin/llm-training-train"), "--help")
    assert direct.returncode == console.returncode == 0
    assert "--environment development" in direct.stderr
    assert "--environment development" in console.stderr


def test_supervisor_child_receives_verified_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEXA_TRAINING_ENVIRONMENT", "historical")
    script = runpy.run_path(str(ROOT / "scripts/training_supervisor.py"), run_name="codexa_test_adapter")
    launch = script["write_launch"](tmp_path / "example", provider="codex")
    assert launch["environment"]["CODEXA_TRAINING_ENVIRONMENT"] == "historical"
    assert "--resume" not in launch["command"]
