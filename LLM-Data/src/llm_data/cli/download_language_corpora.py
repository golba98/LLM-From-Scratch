"""Download pinned Wikipedia and conversational training corpora."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


CORPORA = {
    "wikipedia": {
        "repository": "wikimedia/wikipedia",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "license": "CC-BY-SA-3.0 and GFDL",
        "patterns": ["20231101.en/*.parquet", "README.md"],
    },
    "ultrachat": {
        "repository": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "license": "MIT",
        "patterns": ["data/train_sft-*.parquet", "data/test_sft-*.parquet", "README.md"],
    },
    "oasst1": {
        "repository": "OpenAssistant/oasst1",
        "revision": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
        "license": "Apache-2.0",
        "patterns": ["data/*.parquet", "README.md"],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(root: Path, *, name: str, metadata: dict[str, object]) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus": name,
        **metadata,
        "files": files,
    }
    root.joinpath("download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=generated_path('data/raw'),
    )
    parser.add_argument(
        "--corpus",
        action="append",
        choices=tuple(CORPORA),
        help="Corpus to download; repeat as needed. Defaults to all.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")

    selected = arguments.corpus or list(CORPORA)
    for name in selected:
        metadata = CORPORA[name]
        output_dir = arguments.output_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=str(metadata["repository"]),
            repo_type="dataset",
            revision=str(metadata["revision"]),
            allow_patterns=list(metadata["patterns"]),
            local_dir=output_dir,
            max_workers=arguments.max_workers,
        )
        _manifest(output_dir, name=name, metadata=metadata)
        print(f"Downloaded {name} to {output_dir}")


if __name__ == "__main__":
    main()
