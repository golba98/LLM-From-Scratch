"""Executable integration tests for local dataset preparation."""

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.data.prepare_dataset import prepare_dataset
from src.data.io import read_documents
from src.data.statistics import DatasetStatistics


SAMPLE_PATH = Path("tests/fixtures/data/sample.txt")


def assert_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str,
) -> None:
    """Assert that an operation raises an informative exception."""

    try:
        operation()
    except exception_type as error:
        assert message_fragment in str(error), (
            f"Expected {message_fragment!r} in error message, got {error!r}."
        )
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def file_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_text_input_modes() -> None:
    """Treat text files as one document unless paragraph splitting is enabled."""

    sample_size = SAMPLE_PATH.stat().st_size
    assert 1024 <= sample_size <= 3072

    whole_file = read_documents(SAMPLE_PATH)
    paragraphs = read_documents(
        SAMPLE_PATH,
        split_text_on_blank_lines=True,
    )
    assert len(whole_file) == 1
    assert len(paragraphs) == 8
    assert all(document.source == SAMPLE_PATH.name for document in paragraphs)


def test_jsonl_validation() -> None:
    """Read the strict JSONL schema and reject every malformed row."""

    with TemporaryDirectory() as directory:
        temporary_dir = Path(directory)
        valid_path = temporary_dir / "valid.jsonl"
        valid_path.write_text(
            "\n".join(
                [
                    json.dumps({"text": "First"}),
                    json.dumps(
                        {
                            "text": "Second",
                            "source": "custom",
                            "document_id": "doc-2",
                            "metadata": {"language": "en"},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        documents = read_documents(valid_path)
        assert documents[0].source == valid_path.name
        assert documents[1].source == "custom"
        assert documents[1].document_id == "doc-2"
        assert documents[1].metadata == {"language": "en"}

        invalid_cases = (
            ('{"text": "valid"}\nnot-json\n', 2, "malformed JSON"),
            ("\n", 1, "malformed JSON"),
            (json.dumps({"source": "missing"}) + "\n", 1, "'text' must be a string"),
            (json.dumps({"text": 3}) + "\n", 1, "'text' must be a string"),
            (
                json.dumps({"text": "bad metadata", "metadata": []}) + "\n",
                1,
                "'metadata' must be an object or null",
            ),
        )
        for content, line_number, message in invalid_cases:
            invalid_path = temporary_dir / "invalid.jsonl"
            invalid_path.write_text(content, encoding="utf-8")
            assert_raises(
                ValueError,
                lambda invalid_path=invalid_path: read_documents(invalid_path),
                f"{invalid_path}:{line_number}:",
            )
            assert_raises(
                ValueError,
                lambda invalid_path=invalid_path: read_documents(invalid_path),
                message,
            )


def test_reproducible_pipeline() -> DatasetStatistics:
    """Verify exact counters, output checksums, manifest, and reproducibility."""

    records = [
        {"text": "Alpha", "source": "original"},
        {"text": "  Beta\r\n"},
        {"text": "Alpha", "source": "duplicate"},
        {"text": "\u0000 \t"},
        {"text": "Gamma", "metadata": {"kind": "example"}},
    ]

    with TemporaryDirectory() as directory:
        temporary_dir = Path(directory)
        input_path = temporary_dir / "input.jsonl"
        input_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        first_output = temporary_dir / "first"
        second_output = temporary_dir / "second"

        statistics = prepare_dataset(
            [input_path],
            output_dir=first_output,
            dataset_name="Pipeline Test",
            dataset_license="CC0-1.0",
            validation_ratio=0.25,
            seed=42,
        )
        repeated_statistics = prepare_dataset(
            [input_path],
            output_dir=second_output,
            dataset_name="Pipeline Test",
            dataset_license="CC0-1.0",
            validation_ratio=0.25,
            seed=42,
        )

        assert statistics == repeated_statistics
        assert statistics.input_file_count == 1
        assert statistics.raw_document_count == 5
        assert statistics.cleaned_document_count == 3
        assert statistics.empty_documents_removed == 1
        assert statistics.duplicate_documents_removed == 1
        assert statistics.training_document_count + (
            statistics.validation_document_count
        ) == 3
        assert statistics.validation_document_count >= 1
        assert statistics.total_characters == 14
        assert statistics.total_utf8_bytes == 14
        assert statistics.minimum_document_length == 4
        assert statistics.maximum_document_length == 5
        assert statistics.mean_document_length == 14 / 3
        assert statistics.source_counts == {
            "input.jsonl": 2,
            "original": 1,
        }

        train_path = first_output / "train.jsonl"
        validation_path = first_output / "validation.jsonl"
        statistics_path = first_output / "dataset_stats.json"
        manifest_path = first_output / "dataset_manifest.json"
        assert statistics.train_sha256 == file_sha256(train_path)
        assert statistics.validation_sha256 == file_sha256(validation_path)

        written_statistics = json.loads(
            statistics_path.read_text(encoding="utf-8")
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert written_statistics == statistics.to_dict()
        assert manifest["created_at_utc"].endswith("Z")
        assert manifest["input_paths"] == [str(input_path.resolve())]
        assert manifest["dataset_name"] == "Pipeline Test"
        assert manifest["license"] == "CC0-1.0"
        assert manifest["cleaning_version"] == "1.0"
        assert manifest["validation_ratio"] == 0.25
        assert manifest["seed"] == 42
        assert manifest["statistics"] == written_statistics
        assert manifest["checksums"] == {
            "train_sha256": statistics.train_sha256,
            "validation_sha256": statistics.validation_sha256,
        }
        assert set(manifest["output_paths"]) == {
            "train",
            "validation",
            "statistics",
            "manifest",
        }
        assert manifest["cli_arguments"]["split_text_on_blank_lines"] is False

        for filename in (
            "train.jsonl",
            "validation.jsonl",
            "dataset_stats.json",
        ):
            assert (first_output / filename).read_bytes() == (
                second_output / filename
            ).read_bytes()

        return statistics


def test_sample_cli() -> DatasetStatistics:
    """Run the documented CLI shape against the tracked original fixture."""

    with TemporaryDirectory() as directory:
        output_dir = Path(directory) / "prepared"
        command = [
            sys.executable,
            "scripts/data/prepare_dataset.py",
            str(SAMPLE_PATH),
            "--output-dir",
            str(output_dir),
            "--dataset-name",
            "Codexa Original Sample",
            "--license",
            "CC0-1.0",
            "--validation-ratio",
            "0.2",
            "--seed",
            "42",
            "--split-text-on-blank-lines",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Prepared 8 cleaned documents." in completed.stdout

        statistics_data = json.loads(
            (output_dir / "dataset_stats.json").read_text(encoding="utf-8")
        )
        assert statistics_data["input_file_count"] == 1
        assert statistics_data["raw_document_count"] == 8
        assert statistics_data["cleaned_document_count"] == 8
        assert statistics_data["empty_documents_removed"] == 0
        assert statistics_data["duplicate_documents_removed"] == 0
        assert statistics_data["validation_document_count"] >= 1
        return DatasetStatistics(**statistics_data)


def main() -> None:
    """Run all dataset pipeline integration tests."""

    test_text_input_modes()
    test_jsonl_validation()
    pipeline_statistics = test_reproducible_pipeline()
    sample_statistics = test_sample_cli()

    print(
        "Pipeline counts: "
        f"{pipeline_statistics.training_document_count} train, "
        f"{pipeline_statistics.validation_document_count} validation"
    )
    print(
        "Sample counts: "
        f"{sample_statistics.training_document_count} train, "
        f"{sample_statistics.validation_document_count} validation"
    )
    print(f"Sample train SHA-256: {sample_statistics.train_sha256}")
    print(
        "Sample validation SHA-256: "
        f"{sample_statistics.validation_sha256}"
    )
    print("All data pipeline tests passed.")


if __name__ == "__main__":
    main()
