"""Stream, clean, deduplicate, and split FineWeb-Edu Parquet shards."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Iterator

import pyarrow.parquet as parquet

from llm_data.data.cleaning import CLEANING_VERSION, clean_text, text_sha256


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FORMAT_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _iter_parquet_rows(paths: list[Path]) -> Iterator[tuple[Path, dict[str, object]]]:
    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        if "text" not in parquet_file.schema.names:
            raise ValueError(f"{path}: Parquet schema has no text column.")
        for batch in parquet_file.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: Parquet row must be an object.")
                yield path, row


def _is_validation(identity: str, *, ratio: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big")
    return value / 2**64 < ratio


def prepare_fineweb_edu(
    input_paths: list[Path],
    *,
    output_dir: Path,
    validation_ratio: float,
    seed: int,
    overwrite: bool,
) -> dict[str, object]:
    """Prepare deterministic clean JSONL splits from Parquet shards."""

    if not input_paths:
        raise ValueError("At least one Parquet input is required.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Input shard does not exist: {path}")

    final_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "statistics": output_dir / "dataset_stats.json",
        "manifest": output_dir / "dataset_manifest.json",
    }
    existing = [path for path in final_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite FineWeb-Edu outputs: "
            + ", ".join(str(path) for path in existing)
        )
    temporary = {
        name: _temporary_path(path) for name, path in final_paths.items()
    }
    database_path = _temporary_path(output_dir / "dedup.sqlite3")
    counts = {
        "raw_document_count": 0,
        "cleaned_document_count": 0,
        "empty_documents_removed": 0,
        "duplicate_documents_removed": 0,
        "train_document_count": 0,
        "validation_document_count": 0,
        "train_characters": 0,
        "validation_characters": 0,
        "total_utf8_bytes": 0,
        "minimum_document_length": None,
        "maximum_document_length": 0,
    }
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(database_path)
        database.execute(
            "CREATE TABLE digest (value TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        with (
            temporary["train"].open("w", encoding="utf-8") as train_file,
            temporary["validation"].open(
                "w", encoding="utf-8"
            ) as validation_file,
        ):
            for input_path, row in _iter_parquet_rows(input_paths):
                counts["raw_document_count"] += 1
                text = row.get("text")
                if not isinstance(text, str):
                    raise ValueError(
                        f"{input_path}: text column must contain strings."
                    )
                cleaned = clean_text(text)
                if cleaned is None:
                    counts["empty_documents_removed"] += 1
                    continue
                digest = text_sha256(cleaned)
                inserted = database.execute(
                    "INSERT OR IGNORE INTO digest(value) VALUES (?)",
                    (digest,),
                ).rowcount
                if not inserted:
                    counts["duplicate_documents_removed"] += 1
                    continue

                upstream_id = row.get("id")
                identity = (
                    upstream_id
                    if isinstance(upstream_id, str) and upstream_id
                    else digest
                )
                validation = _is_validation(
                    identity,
                    ratio=validation_ratio,
                    seed=seed,
                )
                split = "validation" if validation else "train"
                output_file = validation_file if validation else train_file
                metadata = {
                    "upstream_revision": REVISION,
                    "upstream_shard": input_path.name,
                }
                for key in ("url", "language_score", "score"):
                    value = row.get(key)
                    if isinstance(value, float) and not math.isfinite(value):
                        continue
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                record = {
                    "text": cleaned,
                    "source": DATASET_NAME,
                    "document_id": identity,
                    "metadata": metadata,
                }
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                length = len(cleaned)
                counts["cleaned_document_count"] += 1
                counts[f"{split}_document_count"] += 1
                counts[f"{split}_characters"] += length
                counts["total_utf8_bytes"] += len(cleaned.encode("utf-8"))
                minimum = counts["minimum_document_length"]
                counts["minimum_document_length"] = (
                    length if minimum is None else min(minimum, length)
                )
                counts["maximum_document_length"] = max(
                    counts["maximum_document_length"],
                    length,
                )
                if counts["cleaned_document_count"] % 100_000 == 0:
                    database.commit()
                    print(
                        "Prepared "
                        f"{counts['cleaned_document_count']:,} documents"
                    )
        database.commit()
        database.close()
        database = None
        total_characters = (
            counts["train_characters"] + counts["validation_characters"]
        )
        statistics = {
            **counts,
            "minimum_document_length": (
                counts["minimum_document_length"] or 0
            ),
            "total_characters": total_characters,
            "mean_document_length": (
                total_characters / counts["cleaned_document_count"]
                if counts["cleaned_document_count"]
                else 0.0
            ),
        }
        temporary["statistics"].write_text(
            json.dumps(statistics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": FORMAT_VERSION,
            "dataset_name": DATASET_NAME,
            "dataset_configuration": "sample-10BT",
            "revision": REVISION,
            "license": "odc-by",
            "cleaning_version": CLEANING_VERSION,
            "validation_ratio": validation_ratio,
            "seed": seed,
            "input_paths": [str(path) for path in input_paths],
            "input_checksums": {
                str(path): _sha256(path) for path in input_paths
            },
            "output_paths": {
                name: str(path) for name, path in final_paths.items()
                if name != "manifest"
            },
            "output_checksums": {
                "train": _sha256(temporary["train"]),
                "validation": _sha256(temporary["validation"]),
            },
            "statistics": statistics,
        }
        temporary["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("train", "validation", "statistics", "manifest"):
            os.replace(temporary[name], final_paths[name])
    finally:
        if database is not None:
            database.close()
        for path in temporary.values():
            path.unlink(missing_ok=True)
        database_path.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=generated_path('data/processed/fineweb-edu'),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    manifest = prepare_fineweb_edu(
        arguments.inputs,
        output_dir=arguments.output_dir,
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
