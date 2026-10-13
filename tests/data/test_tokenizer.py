"""Executable tests for byte-level BPE tokenizer training and loading."""

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from scripts.data.prepare_dataset import prepare_dataset
from src.tokenizer import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    END_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    SYSTEM_TOKEN,
    UNK_TOKEN,
    USER_TOKEN,
    finalize_streaming_tokenizer,
    inspect_tokenizer,
    inspect_tokenizer_streaming,
    load_tokenizer,
    train_tokenizer,
    train_tokenizer_streaming,
)


SAMPLE_PATH = Path("tests/fixtures/data/sample.txt")
TEST_VOCAB_SIZE = 512


def assert_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str,
) -> None:
    """Assert that an operation raises an informative exception."""

    try:
        operation()
    except exception_type as error:
        assert message_fragment in str(error)
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 checksum."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_sample_dataset(output_dir: Path) -> tuple[Path, Path]:
    """Prepare deterministic train and validation JSONL sample files."""

    prepare_dataset(
        [SAMPLE_PATH],
        output_dir=output_dir,
        dataset_name="Codexa Original Sample",
        dataset_license="CC0-1.0",
        validation_ratio=0.05,
        seed=42,
        split_text_on_blank_lines=True,
    )
    return output_dir / "train.jsonl", output_dir / "validation.jsonl"


def test_training_and_round_trip() -> tuple[int, float, str]:
    """Train, reload, inspect, and reproduce one tokenizer."""

    with TemporaryDirectory() as directory:
        temporary_dir = Path(directory)
        train_path, validation_path = prepare_sample_dataset(
            temporary_dir / "dataset"
        )
        input_paths = [train_path, validation_path]

        first_result = train_tokenizer(
            input_paths,
            output_dir=temporary_dir / "tokenizer-first",
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
        )
        second_result = train_tokenizer(
            input_paths,
            output_dir=temporary_dir / "tokenizer-second",
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
        )
        streaming_result = train_tokenizer_streaming(
            input_paths,
            output_dir=temporary_dir / "tokenizer-streaming",
            vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            inspection_encoding_batch_size=2,
        )

        assert first_result.actual_vocab_size == TEST_VOCAB_SIZE
        assert second_result.actual_vocab_size == TEST_VOCAB_SIZE
        first_checksum = file_sha256(first_result.tokenizer_path)
        assert first_checksum == file_sha256(second_result.tokenizer_path)
        assert first_checksum == file_sha256(streaming_result.tokenizer_path)
        streaming_result.manifest_path.unlink()
        finalized_result = finalize_streaming_tokenizer(
            streaming_result.tokenizer_path,
            input_paths,
            requested_vocab_size=TEST_VOCAB_SIZE,
            min_frequency=1,
            inspection_encoding_batch_size=1,
        )
        assert finalized_result.inspection == streaming_result.inspection
        finalized_manifest = json.loads(
            finalized_result.manifest_path.read_text(encoding="utf-8")
        )
        assert finalized_manifest["inspection_encoding_batch_size"] == 1
        assert finalized_manifest["tokenizer_sha256"] == first_checksum
        assert_raises(
            ValueError,
            lambda: finalize_streaming_tokenizer(
                streaming_result.tokenizer_path,
                input_paths,
                requested_vocab_size=TEST_VOCAB_SIZE,
                min_frequency=1,
            ),
            "manifest already exists",
        )

        tokenizer = load_tokenizer(first_result.tokenizer_path)
        assert tokenizer.get_vocab_size(with_added_tokens=True) == TEST_VOCAB_SIZE
        for expected_id, special_token in enumerate(SPECIAL_TOKENS):
            assert tokenizer.token_to_id(special_token) == expected_id
        assert tokenizer.id_to_token(0) == PAD_TOKEN
        assert tokenizer.id_to_token(1) == BOS_TOKEN
        assert tokenizer.id_to_token(2) == EOS_TOKEN
        assert tokenizer.id_to_token(3) == UNK_TOKEN

        samples = [
            "Plain English with MIXED case.",
            "Unicode: café, κόσμος, 你好, 👋🏽.",
            "Punctuation: !?—()[]{};:'\"",
            "Numbers: 0 17 3.14159 -42 +1_000",
            "def greet(name: str) -> str:\n    return f\"Hello, {name}!\"",
        ]
        for sample in samples:
            encoding = tokenizer.encode(sample)
            assert encoding.ids
            assert encoding.ids[0] != tokenizer.token_to_id(BOS_TOKEN)
            assert encoding.ids[-1] != tokenizer.token_to_id(EOS_TOKEN)
            assert tokenizer.decode(encoding.ids) == sample
            assert tokenizer.token_to_id(UNK_TOKEN) not in encoding.ids

        inspection = inspect_tokenizer(tokenizer, samples)
        assert inspection.document_count == len(samples)
        assert inspection.total_tokens > 0
        assert inspection.unknown_token_count == 0
        assert inspection.unknown_token_rate == 0.0
        assert inspection.average_characters_per_token > 0.0
        streaming_inspection = inspect_tokenizer_streaming(
            tokenizer,
            input_paths,
        )
        single_inspection = inspect_tokenizer_streaming(
            tokenizer,
            input_paths,
            encoding_batch_size=1,
        )
        assert streaming_inspection == single_inspection
        assert streaming_inspection.document_count == 8
        assert streaming_inspection.total_tokens > 0
        assert_raises(
            ValueError,
            lambda: inspect_tokenizer_streaming(
                tokenizer,
                input_paths,
                encoding_batch_size=0,
            ),
            "encoding_batch_size must be a positive integer",
        )

        manifest = json.loads(
            first_result.manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["model_type"] == "BPE"
        assert manifest["pre_tokenizer"] == "ByteLevel"
        assert manifest["decoder"] == "ByteLevel"
        assert manifest["add_prefix_space"] is False
        assert manifest["automatic_bos_eos"] is False
        assert manifest["requested_vocab_size"] == TEST_VOCAB_SIZE
        assert manifest["actual_vocab_size"] == TEST_VOCAB_SIZE
        assert manifest["special_tokens"] == {
            PAD_TOKEN: 0,
            BOS_TOKEN: 1,
            EOS_TOKEN: 2,
            UNK_TOKEN: 3,
            SYSTEM_TOKEN: 4,
            USER_TOKEN: 5,
            ASSISTANT_TOKEN: 6,
            END_TOKEN: 7,
        }
        assert manifest["tokenizer_sha256"] == first_checksum

        return (
            first_result.actual_vocab_size,
            inspection.average_characters_per_token,
            first_checksum,
        )


def test_validation_errors() -> None:
    """Reject invalid settings and non-JSONL or empty corpora."""

    with TemporaryDirectory() as directory:
        temporary_dir = Path(directory)
        empty_path = temporary_dir / "empty.jsonl"
        empty_path.write_text("", encoding="utf-8")

        assert_raises(
            ValueError,
            lambda: train_tokenizer(
                [empty_path],
                output_dir=temporary_dir / "tokenizer",
                vocab_size=259,
            ),
            "vocab_size must be an integer of at least 264",
        )
        assert_raises(
            ValueError,
            lambda: train_tokenizer(
                [empty_path],
                output_dir=temporary_dir / "tokenizer",
            ),
            "contains no documents",
        )
        assert_raises(
            ValueError,
            lambda: train_tokenizer(
                [SAMPLE_PATH],
                output_dir=temporary_dir / "tokenizer",
            ),
            "must be JSONL/NDJSON",
        )


def test_cli_tools() -> None:
    """Exercise direct training and JSON inspection command-line tools."""

    with TemporaryDirectory() as directory:
        temporary_dir = Path(directory)
        train_path, validation_path = prepare_sample_dataset(
            temporary_dir / "dataset"
        )
        tokenizer_dir = temporary_dir / "tokenizer"

        train_command = [
            sys.executable,
            "scripts/data/train_tokenizer.py",
            str(train_path),
            str(validation_path),
            "--output-dir",
            str(tokenizer_dir),
            "--vocab-size",
            str(TEST_VOCAB_SIZE),
            "--min-frequency",
            "1",
        ]
        trained = subprocess.run(
            train_command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert trained.returncode == 0, trained.stderr
        assert "Actual vocabulary size: 512" in trained.stdout

        (tokenizer_dir / "tokenizer_manifest.json").unlink()
        finalized = subprocess.run(
            [
                *train_command,
                "--streaming",
                "--finalize-existing",
                "--inspection-batch-size",
                "1",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert finalized.returncode == 0, finalized.stderr
        finalized_manifest = json.loads(
            tokenizer_dir.joinpath("tokenizer_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert finalized_manifest["streaming"] is True
        assert finalized_manifest["inspection_encoding_batch_size"] == 1

        inspect_command = [
            sys.executable,
            "scripts/data/inspect_tokenizer.py",
            str(tokenizer_dir / "tokenizer.json"),
            str(train_path),
            str(validation_path),
            "--json",
        ]
        inspected = subprocess.run(
            inspect_command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert inspected.returncode == 0, inspected.stderr
        inspection = json.loads(inspected.stdout)
        assert inspection["document_count"] == 8
        assert inspection["total_tokens"] > 0
        assert inspection["unknown_token_count"] == 0


def main() -> None:
    """Run all tokenizer tests and print training metrics."""

    actual_vocab_size, characters_per_token, checksum = (
        test_training_and_round_trip()
    )
    test_validation_errors()
    test_cli_tools()

    print(f"Tokenizer vocabulary size: {actual_vocab_size}")
    print(f"Average characters per token: {characters_per_token:.6f}")
    print("Unknown-token rate: 0.000000")
    print(f"Tokenizer SHA-256: {checksum}")
    print("All tokenizer tests passed.")


if __name__ == "__main__":
    main()
