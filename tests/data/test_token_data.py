"""Executable tests for binary token datasets and memory-mapped loading."""

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

from scripts.data.prepare_dataset import prepare_dataset
from src.token_data import (
    MemmapTokenDataset,
    build_token_data,
    benchmark_token_dataloader,
    choose_token_dtype,
    count_packed_examples,
    create_token_dataloader,
    file_sha256,
    inspect_token_data,
    production_packing_policy,
)
from src.tokenizer import train_tokenizer


SAMPLE_PATH = Path("tests/fixtures/data/sample.txt")
TEST_VOCAB_SIZE = 512
CONTEXT_LENGTH = 16


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


def prepare_inputs(root: Path) -> tuple[Path, Path, Path]:
    """Prepare cleaned splits and train a small deterministic tokenizer."""

    dataset_dir = root / "dataset"
    prepare_dataset(
        [SAMPLE_PATH],
        output_dir=dataset_dir,
        dataset_name="Codexa Original Sample",
        dataset_license="CC0-1.0",
        validation_ratio=0.05,
        seed=42,
        split_text_on_blank_lines=True,
    )
    train_path = dataset_dir / "train.jsonl"
    validation_path = dataset_dir / "validation.jsonl"
    tokenizer_result = train_tokenizer(
        [train_path, validation_path],
        output_dir=root / "tokenizer",
        vocab_size=TEST_VOCAB_SIZE,
        min_frequency=1,
    )
    return train_path, validation_path, tokenizer_result.tokenizer_path


def test_packing_math() -> None:
    """Count complete windows and trailing tokens for supported strides."""

    assert count_packed_examples(
        33,
        context_length=8,
        stride=8,
    ) == (4, 0)
    assert count_packed_examples(
        32,
        context_length=8,
        stride=8,
    ) == (3, 7)
    assert count_packed_examples(
        17,
        context_length=8,
        stride=4,
    ) == (3, 0)
    assert count_packed_examples(
        8,
        context_length=8,
    ) == (0, 8)
    assert_raises(
        ValueError,
        lambda: count_packed_examples(20, context_length=0),
        "context_length must be a positive integer",
    )
    assert_raises(
        ValueError,
        lambda: count_packed_examples(20, context_length=8, stride=0),
        "stride must be a positive integer",
    )


def test_binary_pipeline() -> tuple[int, int, str, str]:
    """Build reproducible binaries and validate all offsets and metadata."""

    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_input, validation_input, tokenizer_path = prepare_inputs(root)
        first_output = root / "token-data-first"
        second_output = root / "token-data-second"

        result = build_token_data(
            train_jsonl=train_input,
            validation_jsonl=validation_input,
            tokenizer_path=tokenizer_path,
            output_dir=first_output,
            model_vocab_size=8192,
            context_length=CONTEXT_LENGTH,
            encoding_batch_size=1,
        )
        repeated_result = build_token_data(
            train_jsonl=train_input,
            validation_jsonl=validation_input,
            tokenizer_path=tokenizer_path,
            output_dir=second_output,
            model_vocab_size=8192,
            context_length=CONTEXT_LENGTH,
        )

        assert result.dtype == "uint16"
        assert result.train_document_count == 7
        assert result.validation_document_count == 1
        assert result.train_token_count > CONTEXT_LENGTH
        assert result.validation_token_count > CONTEXT_LENGTH
        assert result.train_token_count == repeated_result.train_token_count
        assert result.validation_token_count == (
            repeated_result.validation_token_count
        )

        manifest = json.loads(
            result.manifest_path.read_text(encoding="utf-8")
        )
        assert manifest["format_version"] == "1.0"
        assert manifest["dtype"] == "uint16"
        assert manifest["model_vocab_size"] == 8192
        assert manifest["tokenizer_actual_vocab_size"] == TEST_VOCAB_SIZE
        assert manifest["eos_token"] == "<eos>"
        assert manifest["eos_token_id"] == 2
        assert manifest["context_length"] == CONTEXT_LENGTH
        assert manifest["packing_policy"] == production_packing_policy()
        assert manifest["packing_policy"]["eos_tokens_between_documents"] == 1
        assert manifest["packing_policy"][
            "attention_crosses_document_boundaries"
        ] is True
        assert manifest["packing_policy"][
            "position_ids_reset_at_document_boundaries"
        ] is False
        assert manifest["cli_arguments"]["stride"] == CONTEXT_LENGTH
        assert manifest["cli_arguments"]["encoding_batch_size"] == 1
        assert not Path(manifest["tokenizer_path"]).is_absolute()
        assert all(
            not Path(path).is_absolute()
            for path in manifest["output_paths"].values()
        )

        for split_name in ("train", "validation"):
            token_path = first_output / f"{split_name}.bin"
            repeated_path = second_output / f"{split_name}.bin"
            index_path = first_output / f"{split_name}_index.json"
            tokens = np.memmap(token_path, mode="r", dtype=np.uint16)
            index_entries = json.loads(index_path.read_text(encoding="utf-8"))

            assert file_sha256(token_path) == (
                manifest["output_checksums"][split_name]
            )
            assert file_sha256(token_path) == file_sha256(repeated_path)
            assert token_path.stat().st_size == (
                manifest["file_byte_sizes"][split_name]
            )
            assert token_path.stat().st_size == len(tokens) * 2

            expected_start = 0
            for ordinal, entry in enumerate(index_entries):
                assert entry["document_ordinal"] == ordinal
                assert entry["token_start"] == expected_start
                assert entry["token_end"] == (
                    entry["token_start"]
                    + entry["total_stored_token_count"]
                )
                assert entry["eos_token_position"] == (
                    entry["token_start"] + entry["content_token_count"]
                )
                assert entry["token_end"] == entry["eos_token_position"] + 1
                assert int(tokens[entry["eos_token_position"]]) == 2
                assert "text" not in entry
                expected_start = entry["token_end"]
            assert expected_start == len(tokens)

            if split_name == "train":
                boundary = index_entries[0]["eos_token_position"]
                sequence_index = boundary // CONTEXT_LENGTH
                sequence_start = sequence_index * CONTEXT_LENGTH
                if sequence_start <= boundary < sequence_start + CONTEXT_LENGTH:
                    dataset = MemmapTokenDataset(
                        token_path,
                        dtype=np.uint16,
                        context_length=CONTEXT_LENGTH,
                    )
                    inputs, labels = dataset[sequence_index]
                    local = boundary - sequence_start
                    assert inputs[local].item() == 2
                    assert labels[local].item() == int(tokens[boundary + 1])
                    assert not torch.any(labels == -100)

        summary = inspect_token_data(result.manifest_path)
        assert summary["dtype"] == "uint16"
        assert summary["splits"]["train"]["token_count"] == (
            result.train_token_count
        )

        train_checksum = file_sha256(first_output / "train.bin")
        validation_checksum = file_sha256(
            first_output / "validation.bin"
        )
        assert_raises(
            FileExistsError,
            lambda: build_token_data(
                train_jsonl=train_input,
                validation_jsonl=validation_input,
                tokenizer_path=tokenizer_path,
                output_dir=first_output,
                model_vocab_size=8192,
                context_length=CONTEXT_LENGTH,
            ),
            "Refusing to overwrite",
        )
        assert file_sha256(first_output / "train.bin") == train_checksum

        overwritten = build_token_data(
            train_jsonl=train_input,
            validation_jsonl=validation_input,
            tokenizer_path=tokenizer_path,
            output_dir=first_output,
            model_vocab_size=8192,
            context_length=CONTEXT_LENGTH,
            overwrite=True,
        )
        assert overwritten.train_token_count == result.train_token_count

        uint32_output = root / "token-data-uint32"
        uint32_result = build_token_data(
            train_jsonl=train_input,
            validation_jsonl=validation_input,
            tokenizer_path=tokenizer_path,
            output_dir=uint32_output,
            model_vocab_size=70_000,
            context_length=CONTEXT_LENGTH,
        )
        assert uint32_result.dtype == "uint32"
        assert (uint32_output / "train.bin").stat().st_size == (
            uint32_result.train_token_count * 4
        )

        return (
            result.train_token_count,
            result.validation_token_count,
            train_checksum,
            validation_checksum,
        )


def test_memmap_dataset_and_loader() -> float:
    """Verify causal alignment, bounds, dtype, and deterministic batching."""

    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_input, validation_input, tokenizer_path = prepare_inputs(root)
        output_dir = root / "token-data"
        build_token_data(
            train_jsonl=train_input,
            validation_jsonl=validation_input,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            model_vocab_size=8192,
            context_length=CONTEXT_LENGTH,
            stride=8,
        )

        dataset = MemmapTokenDataset(
            output_dir / "train.bin",
            dtype="uint16",
            context_length=CONTEXT_LENGTH,
            stride=8,
            model_vocab_size=8192,
        )
        assert len(dataset) > 1
        input_ids, labels = dataset[0]
        assert input_ids.shape == (CONTEXT_LENGTH,)
        assert labels.shape == (CONTEXT_LENGTH,)
        assert input_ids.dtype == torch.int64
        assert labels.dtype == torch.int64
        assert torch.equal(input_ids[1:], labels[:-1])
        assert torch.equal(dataset[-1][0][1:], dataset[-1][1][:-1])

        ranged = MemmapTokenDataset(
            output_dir / "train.bin",
            dtype="uint16",
            context_length=CONTEXT_LENGTH,
            stride=8,
            model_vocab_size=8192,
            token_offset=8,
            token_count=25,
        )
        assert len(ranged) == 2
        assert torch.equal(ranged[0][0], dataset[1][0])
        assert torch.equal(ranged[0][1], dataset[1][1])

        assert_raises(
            IndexError,
            lambda: dataset[len(dataset)],
            "outside",
        )

        first_loader = create_token_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
        )
        second_loader = create_token_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            seed=42,
        )
        first_batch = next(iter(first_loader))
        second_batch = next(iter(second_loader))
        assert first_batch[0].shape == (2, CONTEXT_LENGTH)
        assert first_batch[1].shape == (2, CONTEXT_LENGTH)
        assert torch.equal(first_batch[0], second_batch[0])
        assert torch.equal(first_batch[1], second_batch[1])
        benchmark = benchmark_token_dataloader(
            first_loader,
            max_batches=2,
        )
        assert benchmark.batches == 2
        assert benchmark.tokens > 0
        assert benchmark.tokens_per_second > 0
        assert_raises(
            ValueError,
            lambda: benchmark_token_dataloader(
                first_loader,
                max_batches=0,
            ),
            "max_batches must be a positive integer",
        )

        partial_file = root / "partial.bin"
        partial_file.write_bytes(b"\x00\x01\x02")
        assert_raises(
            ValueError,
            lambda: MemmapTokenDataset(
                partial_file,
                dtype="uint16",
                context_length=2,
            ),
            "not divisible",
        )

        short_file = root / "short.bin"
        np.asarray([1, 2], dtype=np.uint16).tofile(short_file)
        assert_raises(
            ValueError,
            lambda: MemmapTokenDataset(
                short_file,
                dtype="uint16",
                context_length=4,
            ),
            "at least 5 are required",
        )
        return benchmark.tokens_per_second


def test_validation_and_atomic_failure() -> None:
    """Reject incompatible vocabulary and malformed input without final files."""

    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_input, validation_input, tokenizer_path = prepare_inputs(root)

        assert choose_token_dtype(8192) == np.dtype(np.uint16)
        assert choose_token_dtype(70_000) == np.dtype(np.uint32)
        assert_raises(
            ValueError,
            lambda: build_token_data(
                train_jsonl=train_input,
                validation_jsonl=validation_input,
                tokenizer_path=tokenizer_path,
                output_dir=root / "too-small-vocab",
                model_vocab_size=256,
                context_length=CONTEXT_LENGTH,
            ),
            "exceeds model vocabulary size",
        )
        assert_raises(
            ValueError,
            lambda: build_token_data(
                train_jsonl=train_input,
                validation_jsonl=validation_input,
                tokenizer_path=tokenizer_path,
                output_dir=root / "invalid-batch",
                model_vocab_size=8192,
                context_length=CONTEXT_LENGTH,
                encoding_batch_size=0,
            ),
            "encoding_batch_size must be a positive integer",
        )

        malformed = root / "malformed.jsonl"
        malformed.write_text('{"text":"valid"}\nnot-json\n', encoding="utf-8")
        failed_output = root / "failed-output"
        assert_raises(
            ValueError,
            lambda: build_token_data(
                train_jsonl=malformed,
                validation_jsonl=validation_input,
                tokenizer_path=tokenizer_path,
                output_dir=failed_output,
                model_vocab_size=8192,
                context_length=CONTEXT_LENGTH,
            ),
            ":2: malformed JSON",
        )
        assert not (failed_output / "train.bin").exists()
        assert not (failed_output / "token_data_manifest.json").exists()


def test_cli_tools() -> None:
    """Exercise direct tokenization and inspection command-line tools."""

    with TemporaryDirectory() as directory:
        root = Path(directory)
        train_input, validation_input, tokenizer_path = prepare_inputs(root)
        output_dir = root / "token-data"
        tokenize_command = [
            sys.executable,
            "scripts/data/tokenize_dataset.py",
            "--train-jsonl",
            str(train_input),
            "--validation-jsonl",
            str(validation_input),
            "--tokenizer",
            str(tokenizer_path),
            "--output-dir",
            str(output_dir),
            "--model-vocab-size",
            "8192",
            "--context-length",
            str(CONTEXT_LENGTH),
        ]
        tokenized = subprocess.run(
            tokenize_command,
            check=False,
            capture_output=True,
            text=True,
        )
        assert tokenized.returncode == 0, tokenized.stderr
        assert "Dtype: uint16" in tokenized.stdout

        inspected = subprocess.run(
            [
                sys.executable,
                "scripts/data/inspect_token_data.py",
                str(output_dir / "token_data_manifest.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert inspected.returncode == 0, inspected.stderr
        summary = json.loads(inspected.stdout)
        assert summary["dtype"] == "uint16"
        assert summary["splits"]["train"]["document_count"] == 7
        assert summary["splits"]["validation"]["document_count"] == 1


def main() -> None:
    """Run all token-data tests and print reproducible output metrics."""

    test_packing_math()
    train_count, validation_count, train_checksum, validation_checksum = (
        test_binary_pipeline()
    )
    loader_tokens_per_second = test_memmap_dataset_and_loader()
    test_validation_and_atomic_failure()
    test_cli_tools()

    print(f"Training tokens: {train_count}")
    print(f"Validation tokens: {validation_count}")
    print(f"Train SHA-256: {train_checksum}")
    print(f"Validation SHA-256: {validation_checksum}")
    print(
        "DataLoader throughput: "
        f"{loader_tokens_per_second:,.0f} tokens/second"
    )
    print("All token data tests passed.")


if __name__ == "__main__":
    main()
