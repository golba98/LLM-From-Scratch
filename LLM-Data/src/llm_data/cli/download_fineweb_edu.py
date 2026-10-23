"""Download pinned FineWeb-Edu sample-10BT Parquet shards."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY_ID = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARD_COUNT = 14


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=generated_path('data/raw/fineweb-edu-10bt'),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=5,
        help="Leading sample-10BT shards to download (1-14).",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    arguments = parser.parse_args()
    if not 1 <= arguments.shard_count <= SHARD_COUNT:
        raise ValueError(f"--shard-count must be between 1 and {SHARD_COUNT}.")
    if arguments.max_workers <= 0:
        raise ValueError("--max-workers must be positive.")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        f"sample/10BT/{shard_index:03d}_00000.parquet"
        for shard_index in range(arguments.shard_count)
    ]
    snapshot_download(
        repo_id=REPOSITORY_ID,
        repo_type="dataset",
        revision=REVISION,
        allow_patterns=filenames,
        local_dir=arguments.output_dir,
        max_workers=arguments.max_workers,
    )

    files: dict[str, object] = {}
    for filename in filenames:
        downloaded = arguments.output_dir / filename
        files[filename] = {
            "path": str(downloaded),
            "bytes": downloaded.stat().st_size,
            "sha256": _sha256(downloaded),
        }
        print(f"Downloaded {filename}: {downloaded.stat().st_size:,} bytes")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY_ID,
        "revision": REVISION,
        "configuration": "sample-10BT",
        "license": "odc-by",
        "source_url": (
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu"
        ),
        "requested_shard_count": arguments.shard_count,
        "files": files,
    }
    manifest_path = arguments.output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
