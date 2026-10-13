"""Report which Codexa implementation this interpreter context actually loads.

Read-only. It deliberately does not import ``workspace_bootstrap``: run it the same way as
the entry point you want to inspect.

    .venv/bin/python scripts/environment_identity.py            # installed (historical) components
    tools/codexa-dev scripts/environment_identity.py            # current (source-first)
    ... --expect current|historical                             # non-zero exit on mismatch
    ... --preflight-environment development --operation new     # the training guard's verdict

Reproducing the completed 5h foundation run needs its original source, which is no longer in
this tree: restore and verify it with ``tools/historical_source_archive.py verify``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from codexa_workspace.hashing import file_sha256 as sha256
from codexa_workspace.paths import REPO_ROOT as ROOT, TOOLS

# The tokenizer and model shape every current job inherits from the 5h foundation run.
FOUNDATION_RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1"
HISTORICAL_MODEL_SHA256 = "b7f857ad8a10ff022ccb3b8873fcf687f3acf990cae12c94aea97a874c867a74"
# Error text introduced by the masked KV-cache repair (full-mask contract).
REPAIR_MARKER = "attention_mask must have shape [batch, past_length + sequence]"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def components() -> dict[str, dict]:
    manifest = json.loads((ROOT / "compatibility.json").read_text())
    return {**manifest["components"], "LLM-From-Scratch": {"path": "."}}


def package_name(component: str) -> str:
    return "codexa_workspace" if component == "LLM-From-Scratch" else "llm_" + component.split("-", 1)[1].lower()


def component_report(name: str, details: dict) -> dict:
    """Where a component package is imported from, and which source files an installed copy lacks."""
    repo = (ROOT / details["path"]).resolve()
    package = package_name(name)
    source = repo / "src" / package
    record: dict = {"package": package, "pinned_commit": details.get("commit")}
    try:
        record["head"] = git(repo, "rev-parse", "HEAD").strip()
        record["dirty_entries"] = len(git(repo, "status", "--porcelain").splitlines())
    except (subprocess.CalledProcessError, FileNotFoundError):
        record["head"] = None
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        record.update(module=None, loaded_from="not importable")
        return record
    imported = Path(list(spec.submodule_search_locations)[0]).resolve()
    record["module"] = str(imported)
    record["loaded_from"] = "source" if imported == source.resolve() else (
        "installed" if "site-packages" in imported.parts else "other")
    if record["loaded_from"] != "source" and source.is_dir():
        record["differs_from_source"] = sorted(
            str(p.relative_to(source)) for p in source.rglob("*.py")
            if "__pycache__" not in p.parts
            and (not (imported / p.relative_to(source)).exists()
                 or sha256(p) != sha256(imported / p.relative_to(source))))
    return record


def architecture_report() -> dict:
    spec = importlib.util.find_spec("llm_architecture.model")
    path = Path(spec.origin).resolve()
    digest = sha256(path)
    text = path.read_text()
    if digest == HISTORICAL_MODEL_SHA256:
        implementation = "historical"
    elif REPAIR_MARKER in text:
        implementation = "repaired"
    else:
        implementation = "unknown"
    return {"path": str(path), "sha256": digest, "implementation": implementation,
            "masked_cache_repaired": REPAIR_MARKER in text,
            "source_first": path == (ROOT / "../LLM-Architecture/src/llm_architecture/model.py").resolve()}


def tokenizer_report() -> dict:
    spec = json.loads((FOUNDATION_RUN / "config/job.json").read_text())["spec"]
    path = Path(spec["tokenizer"]["path"])
    actual = sha256(path) if path.exists() else None
    return {"path": str(path), "sha256": actual, "matches_job": actual == spec["tokenizer"]["sha256"]}


def model_compatibility() -> dict:
    """Constructs the job's architecture on the meta device; no weights are touched."""
    try:
        import torch
        from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
        spec = json.loads((FOUNDATION_RUN / "config/job.json").read_text())["spec"]
        with torch.device("meta"):
            model = LanguageModel(ModelConfig(**spec["model"]))
        return {"constructs": True, "parameters": count_parameters(model),
                "matches_expected_parameters": count_parameters(model) == spec["expected_parameters"],
                "state_dict_keys": len(model.state_dict())}
    except Exception as error:
        return {"constructs": False, "error": f"{type(error).__name__}: {error}"}


def report() -> dict:
    return {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "bootstrap_active": "workspace_bootstrap" in sys.modules or bool(os.environ.get("CODEXA_WORKSPACE_ROOT")),
        "architecture": architecture_report(),
        "components": {name: component_report(name, details) for name, details in components().items()},
        "tokenizer": tokenizer_report(),
        "model_compatibility": model_compatibility(),
    }


def context_verdict(data: dict) -> str:
    architecture = data["architecture"]
    if architecture["implementation"] == "repaired" and architecture["source_first"]:
        return "current"
    if architecture["implementation"] == "historical":
        return "historical"
    return "mixed"


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", choices=("current", "historical"))
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--preflight-environment", choices=("historical", "development"),
                        help="construction-free training identity check (the training guard's verdict)")
    parser.add_argument("--operation", choices=("inspect", "verify", "new", "resume"), default="inspect",
                        help="operation to validate with --preflight-environment")
    args = parser.parse_args(arguments)
    if args.preflight_environment:
        sys.path.insert(0, str(TOOLS))
        from training_environment_guard import inspect
        checked = inspect(args.preflight_environment, args.operation)
        print(json.dumps(checked, indent=2, sort_keys=True))
        return 0 if checked["valid"] else 2
    data = report()
    data["context"] = context_verdict(data)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        architecture = data["architecture"]
        print(f"context:        {data['context']}")
        print(f"python:         {data['executable']} (bootstrap {'on' if data['bootstrap_active'] else 'off'})")
        print(f"architecture:   {architecture['implementation']} sha {architecture['sha256'][:12]} "
              f"masked-cache repaired={architecture['masked_cache_repaired']}")
        print(f"                {architecture['path']}")
        for name, record in data["components"].items():
            stale = record.get("differs_from_source")
            print(f"{name:17} {record.get('loaded_from', '?'):10} head {str(record.get('head'))[:9]}"
                  f" pin {str(record.get('pinned_commit'))[:9]}"
                  + (f" STALE vs source: {len(stale)} file(s)" if stale else ""))
        print(f"tokenizer:      matches foundation run={data['tokenizer']['matches_job']}")
        print(f"model config:   {data['model_compatibility']}")
    if args.expect and data["context"] != args.expect:
        print(f"EXPECTED {args.expect} CONTEXT, GOT {data['context']}", file=sys.stderr)
        return 1
    return 0
