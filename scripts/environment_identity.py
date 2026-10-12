#!/usr/bin/env python3
"""Report which Codexa implementation this interpreter context actually loads.

Read-only by default. It deliberately does not import ``workspace_bootstrap``:
run it the same way as the entry point you want to inspect.

    .venv/bin/python scripts/environment_identity.py            # historical context
    tools/codexa-dev scripts/environment_identity.py            # current (source-first)
    ... --expect current|historical                             # non-zero exit on mismatch

Like the training supervisor, it only puts the integration ``src`` first, so
the historical ``code_identity`` check matches how ``verify_job`` resolves
modules. It never rewrites run metadata or frozen checksums.

Write modes (each refuses to overwrite an existing directory):
    --snapshot DIR            starting-state record (git, audit, artifact hashes)
    --preserve-history DIR    immutable copy of the completed run's source identity
    --verify-preservation DIR re-hash a preserved artifact against its manifest
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1"
HISTORICAL_MODEL_SHA256 = "b7f857ad8a10ff022ccb3b8873fcf687f3acf990cae12c94aea97a874c867a74"
HISTORICAL_ARCHITECTURE_COMMIT = "c33462047ad75b67d39ffbaf4a95e064fbbe5be4"
# Error text introduced by the masked KV-cache repair (full-mask contract).
REPAIR_MARKER = "attention_mask must have shape [batch, past_length + sequence]"
FINAL_CHECKPOINT_SHA256 = "ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def components() -> dict[str, dict]:
    manifest = json.loads((ROOT / "compatibility.json").read_text())
    return {**manifest["components"], "LLM-From-Scratch": {"path": "."}}


def package_name(component: str) -> str:
    return "codexa_workspace" if component == "LLM-From-Scratch" else "llm_" + component.split("-", 1)[1].lower()


def component_report(name: str, details: dict) -> dict:
    """Same installed-vs-source rule as chat_preparation.experiment_environment, without torch."""
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


def historical_reproducibility() -> dict:
    """code_identity(root) under this interpreter context versus the frozen job."""
    job = json.loads((RUN / "config/job.json").read_text())
    try:
        from codexa_workspace.foundation_training import code_identity
        current = code_identity(ROOT)
    except Exception as error:  # report, never mask
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}
    frozen = job["code_identity"]
    mismatched = sorted(k for k in frozen if current.get(k) != frozen[k])
    return {"status": "matches_job" if not mismatched else "differs_from_job",
            "job_sha256_unchanged": True, "mismatched": {k: [frozen[k][:12], (current.get(k) or "")[:12]]
                                                          for k in mismatched}}


def tokenizer_report() -> dict:
    spec = json.loads((RUN / "config/job.json").read_text())["spec"]
    path = Path(spec["tokenizer"]["path"])
    actual = sha256(path) if path.exists() else None
    return {"path": str(path), "sha256": actual, "matches_job": actual == spec["tokenizer"]["sha256"]}


def model_compatibility() -> dict:
    """Constructs the job's architecture on the meta device; no weights are touched."""
    try:
        import torch
        from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
        spec = json.loads((RUN / "config/job.json").read_text())["spec"]
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
        "historical_reproducibility": historical_reproducibility(),
    }


def context_verdict(data: dict) -> str:
    architecture = data["architecture"]
    if architecture["implementation"] == "repaired" and architecture["source_first"]:
        return "current"
    if data["historical_reproducibility"]["status"] == "matches_job" and architecture["implementation"] == "historical":
        return "historical"
    return "mixed"


def fresh_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise SystemExit(f"Refusing to overwrite existing {path}")
    path.mkdir(parents=True)
    return path


def snapshot(directory: Path, data: dict) -> None:
    directory = fresh_directory(directory)
    repos = {}
    for name, details in components().items():
        repo = (ROOT / details["path"]).resolve()
        repos[name] = {"branch": git(repo, "branch", "--show-current").strip(),
                       "head": git(repo, "rev-parse", "HEAD").strip(),
                       "status": git(repo, "status", "--porcelain", "--untracked-files=all").splitlines()}
    audit = ROOT.parent / "audit"
    outputs = {cmd: subprocess.run([str(audit), cmd], capture_output=True, text=True).stdout
               for cmd in ("status", "doctor")}
    spec = json.loads((RUN / "config/job.json").read_text())["spec"]
    artifacts = {"final_checkpoint": RUN / "checkpoints/slot-0/checkpoint.pt",
                 "baseline_checkpoint": Path(spec["initialization"]["path"]),
                 "job": RUN / "config/job.json", "metrics_report": RUN / "metrics/report.json",
                 "completion": RUN / "supervisor/completion.json", "tokenizer": Path(spec["tokenizer"]["path"])}
    hashes = {key: sha256(path) for key, path in artifacts.items()}
    evaluations = {str(p.relative_to(RUN)): sha256(p) for p in sorted((RUN / "evaluations").rglob("*")) if p.is_file()}
    (directory / "state.json").write_text(json.dumps(
        {"identity": data, "repositories": repos, "artifact_sha256": hashes, "evaluation_sha256": evaluations,
         "expected": {"final_checkpoint": FINAL_CHECKPOINT_SHA256,
                      "baseline_checkpoint": spec["initialization"]["sha256"],
                      "tokenizer": spec["tokenizer"]["sha256"]}}, indent=2, sort_keys=True) + "\n")
    for cmd, text in outputs.items():
        (directory / f"audit-{cmd}.txt").write_text(text)
    print(f"snapshot written to {directory}")


def preserve_history(directory: Path) -> None:
    """Copy the exact files that produced the completed run's code_identity."""
    job = json.loads((RUN / "config/job.json").read_text())
    frozen = job["code_identity"]
    sources: dict[str, Path] = {}
    for key in frozen:
        if key.startswith("src/"):
            sources[key] = ROOT / key
        else:
            sources[key] = Path(importlib.util.find_spec(key).origin).resolve()
    wrong = {k: str(p) for k, p in sources.items() if sha256(p) != frozen[k]}
    if wrong:
        raise SystemExit(f"Run in the historical context (.venv without bootstrap); mismatched: {wrong}")
    directory = fresh_directory(directory)
    manifest: dict = {"run": RUN.name, "job_sha256": sha256(RUN / "config/job.json"),
                      "python": sys.version, "executable": sys.executable,
                      "code_identity": {}, "files": {}}
    for key, path in sources.items():
        target = directory / "code-identity" / (key if key.startswith("src/") else key.replace(".", "/") + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        manifest["code_identity"][key] = {"origin": str(path), "copy": str(target.relative_to(directory)),
                                          "sha256": sha256(target), "matches_job": sha256(target) == frozen[key]}
    purelib = Path(sysconfig.get_paths()["purelib"])
    for entry in sorted(purelib.iterdir()):
        if entry.name.startswith(("llm_", "codexa_workspace")):
            shutil.copytree(entry, directory / "venv-components" / entry.name,
                            ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "src/codexa_workspace", directory / "integration-src/codexa_workspace",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT / "compatibility.json", directory / "compatibility.json")
    (directory / "pip-freeze.txt").write_text(subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True).stdout)
    for name in ("LLM-Architecture", "LLM-Data", "LLM-Training", "LLM-Inference", "LLM-From-Scratch"):
        repo = (ROOT / components()[name]["path"]).resolve()
        (directory / "git" / name).mkdir(parents=True, exist_ok=True)
        (directory / "git" / name / "HEAD").write_text(git(repo, "rev-parse", "HEAD"))
        (directory / "git" / name / "tracked.diff").write_text(git(repo, "diff", "HEAD", "--binary"))
        (directory / "git" / name / "untracked.txt").write_text(
            git(repo, "ls-files", "--others", "--exclude-standard"))
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(directory))] = sha256(path)
    (directory / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in sorted(directory.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    directory.chmod(directory.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    matched = sum(v["matches_job"] for v in manifest["code_identity"].values())
    print(f"preserved {len(manifest['files'])} files; code_identity {matched}/{len(frozen)} match job.json")


def verify_preservation(directory: Path) -> int:
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    frozen = json.loads((RUN / "config/job.json").read_text())["code_identity"]
    bad = [p for p, digest in manifest["files"].items() if sha256(directory / p) != digest]
    identity = [k for k, v in manifest["code_identity"].items() if sha256(directory / v["copy"]) != frozen[k]]
    writable = [str(p) for p in directory.rglob("*") if p.stat().st_mode & stat.S_IWUSR]
    print(json.dumps({"files": len(manifest["files"]), "hash_mismatches": bad,
                      "code_identity_mismatches": identity, "writable_entries": len(writable)}, indent=2))
    return 1 if bad or identity else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expect", choices=("current", "historical"))
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--preserve-history", type=Path)
    group.add_argument("--verify-preservation", type=Path)
    group.add_argument("--preflight-environment", choices=("historical", "development"),
                       help="construction-free training identity check")
    parser.add_argument("--operation", choices=("inspect", "verify", "new", "resume"), default="inspect",
                        help="operation to validate with --preflight-environment")
    args = parser.parse_args()
    if args.verify_preservation:
        return verify_preservation(args.verify_preservation)
    if args.preflight_environment:
        sys.path.insert(0, str(ROOT / "tools"))
        from training_environment_guard import inspect
        checked = inspect(args.preflight_environment, args.operation)
        print(json.dumps(checked, indent=2, sort_keys=True))
        return 0 if checked["valid"] else 2
    if args.preserve_history:
        preserve_history(args.preserve_history)
        return 0
    data = report()
    data["context"] = context_verdict(data)
    if args.snapshot:
        snapshot(args.snapshot, data)
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
        print(f"tokenizer:      matches job={data['tokenizer']['matches_job']}")
        print(f"model config:   {data['model_compatibility']}")
        print(f"historical run: {data['historical_reproducibility']['status']} "
              f"{data['historical_reproducibility'].get('mismatched', '')}")
    if args.expect and data["context"] != args.expect:
        print(f"EXPECTED {args.expect} CONTEXT, GOT {data['context']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
