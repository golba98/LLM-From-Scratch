"""Integration tests for two-source exact deduplication and stable splits."""

import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.data.prepare_base_corpus import prepare_base_corpus


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_cross_source_preparation_is_reproducible_and_leakage_safe() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        fineweb = root / "fineweb.parquet"
        wikipedia = root / "wikipedia.parquet"
        _write(
            fineweb,
            [
                {"id": "f-shared", "text": "Shared article"},
                {"id": "f-only", "text": "FineWeb only"},
                {"id": "f-duplicate", "text": "FineWeb   only"},
                {"id": "f-empty", "text": "\x00  "},
            ],
        )
        _write(
            wikipedia,
            [
                {"id": "w-shared", "text": "Shared article"},
                {"id": "w-only", "text": "Wikipedia only"},
            ],
        )
        first = prepare_base_corpus(
            fineweb_paths=[fineweb],
            wikipedia_paths=[wikipedia],
            output_dir=root / "first",
            validation_ratio=0.5,
            seed=42,
            commit_interval=1,
            progress_interval=100,
        )
        second = prepare_base_corpus(
            fineweb_paths=[fineweb],
            wikipedia_paths=[wikipedia],
            output_dir=root / "second",
            validation_ratio=0.5,
            seed=42,
            commit_interval=2,
            progress_interval=100,
        )
        assert first["statistics"] == second["statistics"]
        assert first["output_checksums"] == second["output_checksums"]
        statistics = first["statistics"]
        assert statistics["wikipedia"]["accepted_documents"] == 2
        assert statistics["fineweb_edu"]["accepted_documents"] == 1
        assert statistics["fineweb_edu"]["rejected"] == {
            "empty_or_invalid": 1,
            "exact_duplicate_cross_source": 1,
            "exact_duplicate_within_source": 1,
        }
        clusters: dict[str, str] = {}
        for source in ("wikipedia", "fineweb_edu"):
            for split in ("train", "validation"):
                path = root / "first" / source / f"{split}.jsonl"
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    cluster = record["metadata"]["duplicate_cluster_id"]
                    assert clusters.setdefault(cluster, split) == split
