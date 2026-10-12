#!/usr/bin/env python3
"""Create and verify a private, byte-exact archive of the completed foundation run.

Only the inventory is suitable for Git. The archive includes historical working
tree diffs and installed package metadata, so it must remain in protected storage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1/config/job.json"
ORIGINAL = ROOT / "outputs/preservation/historical-source/codexa-934m-foundation-5h-v1"
STORE = ROOT / "outputs/preservation/historical-source/durable"
INVENTORY = ROOT / "documentation/workspace/HISTORICAL_SOURCE_V1.json"
VERSION = "historical-source-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute() or not raw or ".." in path.parts or str(path) != raw:
        raise ValueError(f"Unsafe archive path: {raw!r}")
    return path


def original_manifest() -> dict:
    manifest = json.loads((ORIGINAL / "MANIFEST.json").read_text())
    job = json.loads(RUN.read_text())
    if manifest["job_sha256"] != digest(RUN):
        raise ValueError("Completed job differs from preserved manifest")
    for rel, expected in manifest["files"].items():
        path = ORIGINAL / safe_relative(rel)
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError(f"Preserved file differs: {rel}")
    for name, record in manifest["code_identity"].items():
        if digest(ORIGINAL / safe_relative(record["copy"])) != job["code_identity"][name]:
            raise ValueError(f"Frozen source differs: {name}")
    return manifest


def scan_private_material(manifest: dict) -> dict:
    """Conservative scan; findings are counts because archive content stays private."""
    patterns = {
        "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "github_token": re.compile(rb"gh[pousr]_[A-Za-z0-9_]{25,}"),
        "cloud_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
        "password_assignment": re.compile(rb"(?im)^\s*(?:password|api_key|secret_key)\s*[:=]\s*['\"]?[^\s'\"]{12,}"),
    }
    counts = {name: 0 for name in patterns}
    for rel in manifest["files"]:
        data = (ORIGINAL / safe_relative(rel)).read_bytes()
        for name, pattern in patterns.items():
            counts[name] += len(pattern.findall(data))
    return counts


def source_record(name: str, row: dict) -> dict:
    """Map a frozen source identity to its owner and byte-matched Git history."""
    if name.startswith("src/codexa_workspace/"):
        repository = "LLM-From-Scratch"
        relative = name
    else:
        package, module = name.rsplit(".", 1)
        repository = {
            "llm_architecture": "LLM-Architecture",
            "llm_data": "LLM-Data",
            "llm_training": "LLM-Training",
        }[package]
        relative = f"src/{package}/{module}.py"
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H", "--", relative],
        cwd=WORKSPACE / repository, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    recovery_commit = None
    for revision in history:
        blob = subprocess.run(
            ["git", "show", f"{revision}:{relative}"],
            cwd=WORKSPACE / repository, capture_output=True, check=True,
        ).stdout
        if hashlib.sha256(blob).hexdigest() == row["sha256"]:
            recovery_commit = revision
            break
    return {
        "archive_path": row["copy"],
        "sha256": row["sha256"],
        "repository": repository,
        "original_relative_path": relative,
        "git_recovery_commit": recovery_commit,
    }


def create() -> dict:
    manifest = original_manifest()
    findings = scan_private_material(manifest)
    if any(findings.values()):
        raise ValueError(f"Archive requires private-material review: {findings}")
    if INVENTORY.exists():
        raise FileExistsError(f"Inventory already exists: {INVENTORY}")
    STORE.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="historical-source-", suffix=".tar.gz", dir=STORE, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for rel in sorted(manifest["files"]):
                source = ORIGINAL / safe_relative(rel)
                info = archive.gettarinfo(str(source), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o444
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        archive_sha = digest(temporary)
        target = STORE / f"{VERSION}-{archive_sha}.tar.gz"
        if target.exists():
            raise FileExistsError(target)
        temporary.rename(target)
        target.chmod(0o444)
        inventory = {
            "version": VERSION,
            "preserved_at_utc": datetime.now(timezone.utc).isoformat(),
            "run": manifest["run"],
            "job_sha256": manifest["job_sha256"],
            "archive_relative_to_integration": str(target.relative_to(ROOT)),
            "archive_sha256": archive_sha,
            "original_manifest_sha256": digest(ORIGINAL / "MANIFEST.json"),
            "file_count": len(manifest["files"]),
            "files": manifest["files"],
            "code_identity": {name: source_record(name, row)
                              for name, row in manifest["code_identity"].items()},
            "component_paths": {"llm_architecture": "LLM-Architecture/src/llm_architecture",
                                "llm_data": "LLM-Data/src/llm_data",
                                "llm_training": "LLM-Training/src/llm_training",
                                "codexa_workspace": "LLM-From-Scratch/src/codexa_workspace"},
            "package_inventory": "pip-freeze.txt",
            "private_material_scan": findings,
            "off_device_backup_verified": False,
        }
        INVENTORY.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
        return inventory
    finally:
        temporary.unlink(missing_ok=True)


def verify(restore: Path | None = None) -> dict:
    inventory = json.loads(INVENTORY.read_text())
    archive = ROOT / safe_relative(inventory["archive_relative_to_integration"])
    if digest(archive) != inventory["archive_sha256"]:
        raise ValueError("Archive SHA-256 mismatch")
    job = json.loads(RUN.read_text())
    if digest(RUN) != inventory["job_sha256"]:
        raise ValueError("Completed job metadata changed")
    with tempfile.TemporaryDirectory(prefix="codexa-historical-restore-") as temp:
        destination = Path(temp)
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if len(members) != inventory["file_count"]:
                raise ValueError("Archive file count mismatch")
            actual = [member.name for member in members]
            if set(actual) != set(inventory["files"]) or len(actual) != len(set(actual)):
                raise ValueError("Archive inventory mismatch")
            for member in members:
                if not member.isfile():
                    raise ValueError(f"Non-file archive entry: {member.name}")
                target = destination / safe_relative(member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
        mismatched = [rel for rel, expected in inventory["files"].items()
                      if digest(destination / safe_relative(rel)) != expected]
        identity_mismatched = [name for name, row in inventory["code_identity"].items()
                               if digest(destination / safe_relative(row["archive_path"]))
                               != job["code_identity"][name]]
        if mismatched or identity_mismatched:
            raise ValueError(f"Restoration mismatch: files={mismatched}, identity={identity_mismatched}")
        if restore is not None:
            if restore.exists():
                raise FileExistsError(restore)
            shutil.copytree(destination, restore)
    return {"archive": str(archive), "archive_sha256": inventory["archive_sha256"],
            "files_verified": len(inventory["files"]),
            "code_identity_verified": len(inventory["code_identity"]),
            "historical_model_recovered": "llm_architecture.model" in inventory["code_identity"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--restore", type=Path, help="verify and copy into a new directory")
    args = parser.parse_args()
    result = create() if args.command == "create" else verify(args.restore)
    print(json.dumps(result if args.command == "verify" else {
        "archive_sha256": result["archive_sha256"], "files": result["file_count"],
        "code_identity": len(result["code_identity"]), "inventory": str(INVENTORY)}, indent=2))


if __name__ == "__main__":
    main()
