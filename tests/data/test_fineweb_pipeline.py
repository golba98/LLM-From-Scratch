"""Tests for streaming FineWeb-Edu Parquet preparation."""

import json
from pathlib import Path
import tempfile

import pyarrow as arrow
import pyarrow.parquet as parquet

from scripts.data.estimate_parquet_tokens import estimate_shard
from scripts.data.prepare_fineweb_edu import prepare_fineweb_edu
from src.tokenizer import train_tokenizer


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        shard = root / "000_00000.parquet"
        parquet.write_table(
            arrow.table(
                {
                    "id": ["a", "b", "c", "d"],
                    "text": [
                        " First   document. ",
                        "\x00",
                        "First document.",
                        "A distinct document.",
                    ],
                    "url": [
                        "https://example.com/a",
                        "https://example.com/b",
                        "https://example.com/c",
                        "https://example.com/d",
                    ],
                    "score": [float("nan"), 3.0, 4.0, 5.0],
                }
            ),
            shard,
        )
        first = root / "first"
        second = root / "second"
        manifest = prepare_fineweb_edu(
            [shard],
            output_dir=first,
            validation_ratio=0.5,
            seed=42,
            overwrite=False,
        )
        prepare_fineweb_edu(
            [shard],
            output_dir=second,
            validation_ratio=0.5,
            seed=42,
            overwrite=False,
        )
        statistics = manifest["statistics"]
        assert statistics["raw_document_count"] == 4
        assert statistics["cleaned_document_count"] == 2
        assert statistics["empty_documents_removed"] == 1
        assert statistics["duplicate_documents_removed"] == 1
        assert (
            first.joinpath("train.jsonl").read_bytes()
            == second.joinpath("train.jsonl").read_bytes()
        )
        assert (
            first.joinpath("validation.jsonl").read_bytes()
            == second.joinpath("validation.jsonl").read_bytes()
        )
        records = []
        for filename in ("train.jsonl", "validation.jsonl"):
            records.extend(
                json.loads(line)
                for line in first.joinpath(filename)
                .read_text(encoding="utf-8")
                .splitlines()
            )
        assert {record["text"] for record in records} == {
            "First document.",
            "A distinct document.",
        }
        first_record = next(
            record for record in records if record["document_id"] == "a"
        )
        assert "score" not in first_record["metadata"]
        tokenizer = train_tokenizer(
            [first / "train.jsonl", first / "validation.jsonl"],
            output_dir=root / "tokenizer",
            vocab_size=300,
            min_frequency=1,
        )
        estimate = estimate_shard(
            shard,
            tokenizer_path=tokenizer.tokenizer_path,
        )
        assert estimate.raw_documents == 4
        assert estimate.cleaned_documents == 3
        assert estimate.removed_empty_documents == 1
        assert estimate.stored_tokens_with_eos == (
            estimate.content_tokens + 3
        )

    print("All FineWeb-Edu pipeline tests passed.")


if __name__ == "__main__":
    main()
