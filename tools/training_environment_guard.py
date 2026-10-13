"""Construction-free Codexa training identity check and startup gate.

The installed historical wheels are immutable. A small .pth startup hook
loads this module and guards their public trainer functions at import time.
The same hook also guards source-first imports made by workspace_bootstrap.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
JOB = ROOT / "outputs/training/codexa-934m-foundation-5h-v1/config/job.json"
REPAIR_MARKER = b"attention_mask must have shape [batch, past_length + sequence]"
MODULES = {
    "llm_architecture.model": "LLM-Architecture",
    "llm_data.token_data": "LLM-Data",
    "llm_data.token_views": "LLM-Data",
    "llm_training.checkpointing": "LLM-Training",
    "llm_training.pretrain_run": "LLM-Training",
    "llm_training.recovery": "LLM-Training",
    "llm_training.session": "LLM-Training",
    "llm_training.training": "LLM-Training",
}
GUARDED = {
    "llm_training.cli.train": ("main", "run"),
    "llm_training.cli.train_conversational_sft": ("main",),
    "llm_training.conversational_training": ("main",),
    "llm_training.pretrain_run": ("run_pretraining",),
    "llm_training.chat_run": ("run_chat",),
    "llm_training.training": ("create_adamw_optimizer", "train_model"),
}
_selected: str | None = None
_installed = False
HELP_TEXT = ("Codexa training: pass --environment development for a new committed-source job. "
             "Historical selection is for verification or eligible exact recovery. "
             "Inspect with scripts/environment_identity.py --preflight-environment ...")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()


def loaded_file(module: str) -> Path | None:
    relative = Path(*module.split(".")).with_suffix(".py")
    for entry in sys.path:
        base = Path(entry or os.getcwd())
        candidate = base / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def selected_environment() -> str:
    """Consume one explicit CLI selection or the supervisor's explicit env."""
    global _selected
    args = list(sys.argv)
    cli = []
    for index, value in enumerate(args):
        if value == "--environment":
            if index + 1 >= len(args):
                raise RuntimeError("--environment requires historical or development")
            cli.append((index, args[index + 1], 2))
        elif value.startswith("--environment="):
            cli.append((index, value.split("=", 1)[1], 1))
    if len(cli) > 1:
        raise RuntimeError("Environment selection is ambiguous")
    if cli:
        index, choice, width = cli[0]
        del sys.argv[index:index + width]
    else:
        choice = None
    inherited = os.environ.get("CODEXA_TRAINING_ENVIRONMENT")
    choices = {x for x in (choice, inherited, _selected) if x}
    if len(choices) != 1:
        raise RuntimeError("Explicit --environment historical|development is required")
    value = choices.pop()
    if value not in ("historical", "development"):
        raise RuntimeError(f"Invalid training environment: {value!r}")
    _selected = value
    os.environ["CODEXA_TRAINING_ENVIRONMENT"] = value
    return value


def inspect(environment: str, operation: str = "inspect") -> dict:
    """Read paths, hashes, revisions and tokenizer without importing a model."""
    errors = []
    manifest = json.loads((ROOT / "compatibility.json").read_text())
    frozen = json.loads(JOB.read_text())
    venv = ROOT / ".venv"
    if Path(sys.prefix).resolve() != venv.resolve():
        errors.append(f"Python prefix is {sys.prefix}; expected {venv}")
    origins = {}
    for name in MODULES:
        path = loaded_file(name)
        origins[name] = {"path": str(path) if path else None, "sha256": sha256(path) if path else None}
        if path is None:
            errors.append(f"{name} is not importable")
    arch = origins["llm_architecture.model"]
    arch_source = (WORKSPACE / "LLM-Architecture/src/llm_architecture/model.py").resolve()
    repaired = bool(arch["path"] and REPAIR_MARKER in Path(arch["path"]).read_bytes())
    revisions = {}
    if environment == "historical":
        missing = [name for name in frozen["code_identity"] if name.startswith("src/") and not (ROOT / name).is_file()]
        if missing:
            errors.append("The 5h foundation job's integration source is not in this tree; restore it with "
                          "tools/historical_source_archive.py verify --restore <new-dir>")
        for name, expected in frozen["code_identity"].items():
            if name in missing:
                continue
            path = ROOT / name if name.startswith("src/") else loaded_file(name)
            actual = sha256(path) if path and path.is_file() else None
            if actual != expected:
                errors.append(f"historical {name}: expected {expected}, detected {actual} at {path}")
        if repaired or arch["path"] == str(arch_source):
            errors.append("historical selection loaded repaired/source architecture")
    elif environment == "development":
        for name, record in manifest["components"].items():
            repo = (ROOT / record["path"]).resolve()
            head = git(repo, "rev-parse", "HEAD")
            revisions[name] = {"expected": record["commit"], "detected": head}
            if head != record["commit"]:
                errors.append(f"{name} revision differs: expected {record['commit']}, detected {head}")
            package = "llm_" + name.split("-", 1)[1].lower()
            source = (repo / "src" / package).resolve()
            visible = loaded_file(package + ".__init__")
            # __init__ is a package file rather than a module.py file.
            if visible is None:
                for entry in sys.path:
                    candidate = Path(entry or os.getcwd()) / package / "__init__.py"
                    if candidate.is_file():
                        visible = candidate.resolve()
                        break
            if visible is None or visible.parent != source:
                errors.append(f"{package} import path is {visible}; expected {source}")
            if operation in ("new", "resume") and git(repo, "status", "--porcelain", "--untracked-files=all", "--", "src"):
                errors.append(f"{name} has uncommitted executed source")
        if arch["path"] != str(arch_source) or not repaired:
            errors.append(f"development architecture is {arch['path']}; expected repaired {arch_source}")
        else:
            committed = subprocess.run(["git", "-C", str(WORKSPACE / "LLM-Architecture"), "show",
                                        "HEAD:src/llm_architecture/model.py"], check=True,
                                       capture_output=True).stdout
            expected = hashlib.sha256(committed).hexdigest()
            if arch["sha256"] != expected:
                errors.append("Development architecture differs from committed source")
        if operation in ("new", "resume") and git(ROOT, "status", "--porcelain", "--untracked-files=all",
                                                   "--", "src", "scripts", "tools"):
            errors.append("Integration training source has uncommitted changes")
    else:
        errors.append(f"unknown environment {environment!r}")
    token = Path(frozen["spec"]["tokenizer"]["path"])
    token_hash = sha256(token) if token.is_file() else None
    if token_hash != frozen["spec"]["tokenizer"]["sha256"]:
        errors.append(f"Tokenizer differs: expected {frozen['spec']['tokenizer']['sha256']}, detected {token_hash}")
    if operation == "new" and environment != "development":
        errors.append("New training jobs require the development environment")
    if operation == "resume" and environment == "historical":
        completion = ROOT / "outputs/training/codexa-934m-foundation-5h-v1/supervisor/completion.json"
        if completion.exists() and json.loads(completion.read_text()).get("terminal_state") == "COMPLETED":
            errors.append("The historical foundation job is completed and cannot resume")
    return {"selected": environment, "operation": operation, "python": sys.executable,
            "architecture": arch | {"repaired": repaired}, "tokenizer_sha256": token_hash,
            "component_revisions": revisions, "origins": origins, "errors": errors,
            "valid": not errors}


def require_environment(operation: str = "new") -> dict:
    try:
        selected = selected_environment()
        report = inspect(selected, operation)
    except Exception as exc:
        raise RuntimeError(f"Codexa training environment preflight failed: {exc}") from exc
    if not report["valid"]:
        raise RuntimeError("Codexa training environment rejected: " + "; ".join(report["errors"]) +
                           f". Selected={selected}, detected architecture={report['architecture']}. "
                           "Use tools/codexa-dev for development or the historical interpreter for verification.")
    return report


class GuardLoader(importlib.abc.Loader):
    def __init__(self, original, fullname):
        self.original, self.fullname = original, fullname

    def create_module(self, spec):
        return self.original.create_module(spec) if hasattr(self.original, "create_module") else None

    def exec_module(self, module):
        self.original.exec_module(module)
        for name in GUARDED[self.fullname]:
            original = getattr(module, name, None)
            if not callable(original):
                continue
            @functools.wraps(original)
            def guarded(*args, __original=original, __name=name, **kwargs):
                if "--help" in sys.argv:
                    print(HELP_TEXT, file=sys.stderr)
                    return __original(*args, **kwargs)
                operation = "resume" if (__name == "run_pretraining" and kwargs.get("resume")) else "new"
                require_environment(operation)
                return __original(*args, **kwargs)
            setattr(module, name, guarded)

    def __getattr__(self, name):
        return getattr(self.original, name)


class GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in GUARDED:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = GuardLoader(spec.loader, fullname)
        return spec


def install() -> None:
    global _installed
    if _installed:
        return
    sys.meta_path.insert(0, GuardFinder())
    _installed = True
    # runpy executes -m targets as __main__, so their module wrapper is not
    # used. Check the invocation before the module starts or builds a model.
    original = list(getattr(sys, "orig_argv", []))
    modules = ("llm_training.cli.train", "llm_training.cli.train_conversational_sft",
               "codexa_workspace.training.foundation.continuation")
    for index, arg in enumerate(original[:-1]):
        if arg == "-m" and original[index + 1] in modules:
            if "--help" in sys.argv:
                print(HELP_TEXT, file=sys.stderr)
                break
            operation = "resume" if "--resume" in sys.argv else "new"
            try:
                require_environment(operation)
            except Exception as error:
                # Python's site.py prints and ignores exceptions from .pth
                # files. Exit explicitly or an old CLI would keep running.
                sys.stderr.write(str(error) + "\n")
                sys.stderr.flush()
                os._exit(2)
            break
    script = Path(original[1]).name if len(original) > 1 and not original[1].startswith("-") else ""
    if script == "training_supervisor.py" and "--help" in sys.argv:
        print(HELP_TEXT, file=sys.stderr)
    if script == "training_supervisor.py" and "--help" not in sys.argv:
        command = next((arg for arg in sys.argv[1:] if not arg.startswith("-")), "")
        operation = {"run": "new"}.get(command)          # reconcile never trains
        if command == "run" and "--run-dir" in sys.argv:
            position = sys.argv.index("--run-dir")
            if position + 1 < len(sys.argv):
                launch = Path(sys.argv[position + 1]) / "supervisor/launch.json"
                if launch.is_file() and "--resume" in json.loads(launch.read_text()).get("command", []):
                    operation = "resume"
        if operation:
            try:
                require_environment(operation)
            except Exception as error:
                sys.stderr.write(str(error) + "\n")
                sys.stderr.flush()
                os._exit(2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=("historical", "development"))
    parser.add_argument("--operation", choices=("inspect", "verify", "new", "resume"), default="inspect")
    args = parser.parse_args()
    report = inspect(args.environment, args.operation)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 2)
