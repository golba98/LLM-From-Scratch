"""Prepare leakage-safe FineWeb-Edu and Wikipedia base splits at scale."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

import pyarrow.parquet as pq

from llm_data.data.cleaning import CLEANING_VERSION, clean_text, text_sha256


FORMAT_VERSION = "1.0"
SOURCE_ORDER = ("wikipedia", "fineweb_edu")
SOURCE_NAMES = {
    "wikipedia": "wikimedia/wikipedia:20231101.en",
    "fineweb_edu": "HuggingFaceFW/fineweb-edu:sample-10BT",
}
SOURCE_REVISIONS = {
    "wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "fineweb_edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _is_validation(cluster_id: str, ratio: float, seed: int) -> bool:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\0{cluster_id}".encode("utf-8")).digest(),
        "big",
    )
    return value < int(ratio * (1 << 256))


def _record(
    *,
    text: str,
    source: str,
    document_id: str,
    cluster_id: str,
) -> str:
    return json.dumps(
        {
            "document_id": document_id,
            "metadata": {
                "duplicate_cluster_id": cluster_id,
                "upstream_revision": SOURCE_REVISIONS[source],
            },
            "source": SOURCE_NAMES[source],
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _input_rows(paths: list[Path]):
    for path in sorted(paths, key=lambda item: item.as_posix()):
        parquet = pq.ParquetFile(path)
        if "text" not in parquet.schema.names:
            raise ValueError(f"{path}: missing text column.")
        columns = [name for name in ("text", "id") if name in parquet.schema.names]
        row_ordinal = 0
        for batch in parquet.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                yield path, row_ordinal, row
                row_ordinal += 1


def prepare_base_corpus(
    *,
    fineweb_paths: list[Path],
    wikipedia_paths: list[Path],
    output_dir: Path,
    validation_ratio: float = 0.005,
    seed: int = 42,
    commit_interval: int = 10_000,
    progress_interval: int = 100_000,
) -> dict[str, object]:
    """Normalize, exactly deduplicate, split, and write both base sources."""

    if not fineweb_paths or not wikipedia_paths:
        raise ValueError("FineWeb-Edu and Wikipedia inputs are both required.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if commit_interval <= 0 or progress_interval <= 0:
        raise ValueError("commit and progress intervals must be positive.")
    paths_by_source = {
        "wikipedia": wikipedia_paths,
        "fineweb_edu": fineweb_paths,
    }
    for paths in paths_by_source.values():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)

    final = {
        source: {
            split: output_dir / source / f"{split}.jsonl"
            for split in ("train", "validation")
        }
        for source in SOURCE_ORDER
    }
    manifest_path = output_dir / "base_preparation_manifest.json"
    all_final = [
        path for source in final.values() for path in source.values()
    ] + [manifest_path]
    existing = [path for path in all_final if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite base preparation outputs: "
            + ", ".join(str(path) for path in existing)
        )
    temporary = {
        source: {split: _temporary(path) for split, path in values.items()}
        for source, values in final.items()
    }
    temporary_manifest = _temporary(manifest_path)
    database_path = output_dir / ".exact_dedup.sqlite3.tmp"
    if database_path.exists():
        raise FileExistsError(
            f"Incomplete prior dedup database exists: {database_path}"
        )
    counts = {
        source: {
            "raw_rows": 0,
            "accepted_documents": 0,
            "train_documents": 0,
            "validation_documents": 0,
            "content_characters": 0,
            "utf8_bytes": 0,
            "rejected": Counter(),
        }
        for source in SOURCE_ORDER
    }
    started = time.monotonic()
    database: sqlite3.Connection | None = None
    handles = {}
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=NORMAL")
        database.execute(
            "CREATE TABLE exact_digest (digest TEXT PRIMARY KEY, source TEXT NOT NULL) WITHOUT ROWID"
        )
        handles = {
            source: {
                split: path.open("w", encoding="utf-8", newline="\n")
                for split, path in values.items()
            }
            for source, values in temporary.items()
        }
        total_rows = 0
        for source in SOURCE_ORDER:
            for path, row_ordinal, row in _input_rows(paths_by_source[source]):
                total_rows += 1
                source_counts = counts[source]
                source_counts["raw_rows"] += 1
                text = row.get("text")
                if not isinstance(text, str):
                    source_counts["rejected"]["malformed_text"] += 1
                    continue
                cleaned = clean_text(text)
                if cleaned is None:
                    source_counts["rejected"]["empty_or_invalid"] += 1
                    continue
                cluster_id = text_sha256(cleaned)
                inserted = database.execute(
                    "INSERT OR IGNORE INTO exact_digest(digest, source) VALUES (?, ?)",
                    (cluster_id, source),
                ).rowcount
                if not inserted:
                    owner = database.execute(
                        "SELECT source FROM exact_digest WHERE digest = ?",
                        (cluster_id,),
                    ).fetchone()[0]
                    reason = (
                        "exact_duplicate_within_source"
                        if owner == source
                        else "exact_duplicate_cross_source"
                    )
                    source_counts["rejected"][reason] += 1
                    continue
                upstream_id = row.get("id")
                identity = (
                    upstream_id
                    if isinstance(upstream_id, str) and upstream_id
                    else f"{path.name}:{row_ordinal}:{cluster_id}"
                )
                split = (
                    "validation"
                    if _is_validation(cluster_id, validation_ratio, seed)
                    else "train"
                )
                handles[source][split].write(
                    _record(
                        text=cleaned,
                        source=source,
                        document_id=identity,
                        cluster_id=cluster_id,
                    )
                )
                source_counts["accepted_documents"] += 1
                source_counts[f"{split}_documents"] += 1
                source_counts["content_characters"] += len(cleaned)
                source_counts["utf8_bytes"] += len(cleaned.encode("utf-8"))
                if total_rows % commit_interval == 0:
                    database.commit()
                if total_rows % progress_interval == 0:
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            {
                                "elapsed_seconds": round(elapsed, 1),
                                "rows_per_second": round(total_rows / elapsed, 1),
                                "source": source,
                                "total_rows": total_rows,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            database.commit()
        for source_handles in handles.values():
            for handle in source_handles.values():
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
        handles = {}
        database.close()
        database = None
        elapsed = time.monotonic() - started
        serializable_counts = {
            source: {
                **{
                    key: value
                    for key, value in values.items()
                    if key != "rejected"
                },
                "rejected": dict(sorted(values["rejected"].items())),
            }
            for source, values in counts.items()
        }
        output_checksums = {
            source: {
                split: _sha256(path) for split, path in values.items()
            }
            for source, values in temporary.items()
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": FORMAT_VERSION,
            "cleaning_version": CLEANING_VERSION,
            "seed": seed,
            "validation_ratio": validation_ratio,
            "source_processing_order": list(SOURCE_ORDER),
            "cross_source_duplicate_ownership": (
                "Wikipedia is processed first and deterministically owns exact "
                "cross-source duplicate text. Split assignment uses the exact "
                "text digest, not the retained source identifier."
            ),
            "near_duplicate_status": "deferred_not_completed",
            "input_paths": {
                source: [str(path) for path in sorted(paths)]
                for source, paths in paths_by_source.items()
            },
            "input_checksums": {
                source: {str(path): _sha256(path) for path in sorted(paths)}
                for source, paths in paths_by_source.items()
            },
            "output_paths": {
                source: {split: str(path) for split, path in values.items()}
                for source, values in final.items()
            },
            "output_checksums": output_checksums,
            "statistics": serializable_counts,
            "elapsed_seconds": elapsed,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for source in SOURCE_ORDER:
            for split in ("train", "validation"):
                os.replace(temporary[source][split], final[source][split])
        os.replace(temporary_manifest, manifest_path)
        database_path.unlink(missing_ok=True)
        database_path.with_suffix(database_path.suffix + "-wal").unlink(missing_ok=True)
        database_path.with_suffix(database_path.suffix + "-shm").unlink(missing_ok=True)
        return manifest
    except BaseException:
        for source_handles in handles.values():
            for handle in source_handles.values():
                handle.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineweb-root", type=Path, required=True)
    parser.add_argument("--wikipedia-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commit-interval", type=int, default=10_000)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    arguments = parser.parse_args()
    manifest = prepare_base_corpus(
        fineweb_paths=sorted(arguments.fineweb_root.glob("*.parquet")),
        wikipedia_paths=sorted(arguments.wikipedia_root.glob("*.parquet")),
        output_dir=arguments.output_dir,
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
        commit_interval=arguments.commit_interval,
        progress_interval=arguments.progress_interval,
    )
    print(json.dumps(manifest["statistics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
