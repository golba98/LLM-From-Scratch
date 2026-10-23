"""Binary token dataset creation and memory-mapped PyTorch loading."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from llm_tokenizer.tokenizer import EOS_TOKEN, load_tokenizer


TOKEN_DATA_FORMAT_VERSION = "1.0"
TRAIN_FILENAME = "train.bin"
VALIDATION_FILENAME = "validation.bin"
MANIFEST_FILENAME = "token_data_manifest.json"
TRAIN_INDEX_FILENAME = "train_index.json"
VALIDATION_INDEX_FILENAME = "validation_index.json"


def production_packing_policy() -> dict[str, object]:
    """Return the versioned initial document-packing contract."""

    return {
        "version": "1.0",
        "documents_may_share_sequence": True,
        "eos_tokens_between_documents": 1,
        "predict_first_token_after_eos": True,
        "attention_crosses_document_boundaries": True,
        "position_ids_reset_at_document_boundaries": False,
        "trailing_partial_sequence": "discard",
        "padding": "none",
        "padding_token_id": 0,
        "padding_label_id": -100,
        "loss_exclusions": "none_without_padding",
        "boundary_metadata": "split_index_json",
        "same_policy_for_train_and_validation": True,
    }


@dataclass(frozen=True)
class TokenDataBuildResult:
    """Files and counts produced by token dataset creation."""

    manifest_path: Path
    train_token_count: int
    validation_token_count: int
    train_document_count: int
    validation_document_count: int
    dtype: str


@dataclass(frozen=True)
class DataLoaderBenchmark:
    """Measured DataLoader throughput."""

    batches: int
    examples: int
    tokens: int
    elapsed_seconds: float
    tokens_per_second: float


def file_sha256(path: str | Path) -> str:
    """Calculate a file checksum without loading the full file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_token_dtype(model_vocab_size: int) -> np.dtype:
    """Choose the smallest supported unsigned dtype for model token IDs."""

    if (
        not isinstance(model_vocab_size, int)
        or isinstance(model_vocab_size, bool)
        or model_vocab_size <= 0
    ):
        raise ValueError(
            "model_vocab_size must be a positive integer; "
            f"got {model_vocab_size!r}."
        )

    maximum_token_id = model_vocab_size - 1
    if maximum_token_id <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_token_id <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    raise ValueError(
        "model_vocab_size exceeds the supported uint32 token range; "
        f"got {model_vocab_size}."
    )


def _iter_jsonl_records(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: JSONL row must be an object"
                )

            text = record.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"{path}:{line_number}: required field 'text' "
                    "must be a string"
                )
            source = record.get("source", path.name)
            if not isinstance(source, str):
                raise ValueError(
                    f"{path}:{line_number}: optional field 'source' "
                    "must be a string"
                )
            document_id = record.get("document_id")
            if document_id is not None and not isinstance(document_id, str):
                raise ValueError(
                    f"{path}:{line_number}: optional field 'document_id' "
                    "must be a string or null"
                )
            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(
                    f"{path}:{line_number}: optional field 'metadata' "
                    "must be an object or null"
                )
            yield record


def _temporary_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(temporary_name)


def _serialize_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _iter_index_entries(path: Path) -> Iterator[dict[str, object]]:
    """Read the line-oriented JSON array emitted by the token-data writer."""

    with path.open("r", encoding="utf-8") as index_file:
        if index_file.readline().strip() != "[":
            raise ValueError(f"{path}: token index must start with '['.")
        for line_number, line in enumerate(index_file, start=2):
            stripped = line.strip()
            if stripped == "]":
                return
            if stripped.endswith(","):
                stripped = stripped[:-1]
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed index JSON "
                    f"({error.msg})"
                ) from error
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{path}:{line_number}: index entry must be an object."
                )
            yield entry
    raise ValueError(f"{path}: token index is missing its closing ']'.")


def _tokenize_split(
    *,
    input_path: Path,
    output_path: Path,
    index_output_path: Path,
    tokenizer: Tokenizer,
    eos_token_id: int,
    model_vocab_size: int,
    dtype: np.dtype,
    encoding_batch_size: int,
) -> tuple[int, int]:
    token_count = 0
    document_count = 0

    with (
        output_path.open("wb") as output_file,
        index_output_path.open("w", encoding="utf-8") as index_file,
    ):
        index_file.write("[\n")
        records = _iter_jsonl_records(input_path)
        while True:
            batch: list[dict[str, object]] = []
            for _ in range(encoding_batch_size):
                try:
                    batch.append(next(records))
                except StopIteration:
                    break
            if not batch:
                break
            texts = [record["text"] for record in batch]
            assert all(isinstance(text, str) for text in texts)
            encodings = tokenizer.encode_batch(
                texts,
                add_special_tokens=False,
            )
            if len(encodings) != len(batch):
                raise RuntimeError("Tokenizer batch output length mismatch.")
            for record, encoding in zip(batch, encodings, strict=True):
                ordinal = document_count
                content_token_ids = encoding.ids
                for token_id in content_token_ids:
                    if (
                        not isinstance(token_id, int)
                        or isinstance(token_id, bool)
                        or token_id < 0
                        or token_id >= model_vocab_size
                    ):
                        raise ValueError(
                            f"{input_path}: document {ordinal} produced "
                            f"invalid token ID {token_id!r} for model "
                            f"vocabulary size {model_vocab_size}."
                        )

                token_start = token_count
                eos_position = token_start + len(content_token_ids)
                stored_token_ids = [*content_token_ids, eos_token_id]
                np.asarray(stored_token_ids, dtype=dtype).tofile(output_file)
                token_count += len(stored_token_ids)

                entry: dict[str, object] = {
                    "document_ordinal": ordinal,
                    "source": record.get("source", input_path.name),
                    "token_start": token_start,
                    "token_end": token_count,
                    "content_token_count": len(content_token_ids),
                    "eos_token_position": eos_position,
                    "total_stored_token_count": len(stored_token_ids),
                }
                document_id = record.get("document_id")
                if document_id is not None:
                    entry["document_id"] = document_id
                if document_count:
                    index_file.write(",\n")
                index_file.write(
                    json.dumps(entry, ensure_ascii=False, sort_keys=True)
                )
                document_count += 1
        index_file.write("\n]\n")

    expected_size = token_count * dtype.itemsize
    actual_size = output_path.stat().st_size
    if actual_size != expected_size or actual_size % dtype.itemsize != 0:
        raise ValueError(
            f"Incomplete token output {output_path}: expected "
            f"{expected_size} bytes, got {actual_size}."
        )
    return token_count, document_count


def count_packed_examples(
    token_count: int,
    *,
    context_length: int,
    stride: int | None = None,
) -> tuple[int, int]:
    """Return complete example count and trailing token count."""

    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length <= 0
    ):
        raise ValueError(
            "context_length must be a positive integer; "
            f"got {context_length!r}."
        )
    resolved_stride = context_length if stride is None else stride
    if (
        not isinstance(resolved_stride, int)
        or isinstance(resolved_stride, bool)
        or resolved_stride <= 0
    ):
        raise ValueError(
            f"stride must be a positive integer; got {resolved_stride!r}."
        )
    if token_count < 0:
        raise ValueError(f"token_count must be non-negative; got {token_count}.")

    required_tokens = context_length + 1
    if token_count < required_tokens:
        return 0, token_count

    example_count = ((token_count - required_tokens) // resolved_stride) + 1
    final_start = (example_count - 1) * resolved_stride
    trailing_tokens = token_count - (final_start + required_tokens)
    return example_count, trailing_tokens


def _manifest_path_value(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return os.path.relpath(path, start=output_dir)


def build_token_data(
    *,
    train_jsonl: str | Path,
    validation_jsonl: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    model_vocab_size: int,
    context_length: int,
    stride: int | None = None,
    encoding_batch_size: int = 16,
    overwrite: bool = False,
) -> TokenDataBuildResult:
    """Tokenize clean splits into atomic memory-mappable binary arrays."""

    train_input = Path(train_jsonl)
    validation_input = Path(validation_jsonl)
    tokenizer_file = Path(tokenizer_path)
    destination = Path(output_dir)
    resolved_stride = context_length if stride is None else stride
    count_packed_examples(
        0,
        context_length=context_length,
        stride=resolved_stride,
    )
    if (
        not isinstance(encoding_batch_size, int)
        or isinstance(encoding_batch_size, bool)
        or encoding_batch_size <= 0
    ):
        raise ValueError("encoding_batch_size must be a positive integer.")

    tokenizer = load_tokenizer(tokenizer_file)
    actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_vocab_size > model_vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary size {actual_vocab_size} exceeds model "
            f"vocabulary size {model_vocab_size}."
        )
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_token_id != 2:
        raise ValueError(f"<eos> must have token ID 2; got {eos_token_id!r}.")

    dtype = choose_token_dtype(model_vocab_size)
    final_paths = {
        "train": destination / TRAIN_FILENAME,
        "validation": destination / VALIDATION_FILENAME,
        "manifest": destination / MANIFEST_FILENAME,
        "train_index": destination / TRAIN_INDEX_FILENAME,
        "validation_index": destination / VALIDATION_INDEX_FILENAME,
    }
    existing_paths = [path for path in final_paths.values() if path.exists()]
    if existing_paths and not overwrite:
        rendered = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Refusing to overwrite existing token data files: {rendered}"
        )

    temporary_paths = {
        name: _temporary_path(path) for name, path in final_paths.items()
    }
    try:
        train_token_count, train_document_count = _tokenize_split(
            input_path=train_input,
            output_path=temporary_paths["train"],
            index_output_path=temporary_paths["train_index"],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            model_vocab_size=model_vocab_size,
            dtype=dtype,
            encoding_batch_size=encoding_batch_size,
        )
        validation_token_count, validation_document_count = _tokenize_split(
            input_path=validation_input,
            output_path=temporary_paths["validation"],
            index_output_path=temporary_paths["validation_index"],
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            model_vocab_size=model_vocab_size,
            dtype=dtype,
            encoding_batch_size=encoding_batch_size,
        )
        train_examples, train_trailing = count_packed_examples(
            train_token_count,
            context_length=context_length,
            stride=resolved_stride,
        )
        validation_examples, validation_trailing = count_packed_examples(
            validation_token_count,
            context_length=context_length,
            stride=resolved_stride,
        )
        output_checksums = {
            name: file_sha256(temporary_paths[name])
            for name in ("train", "validation", "train_index", "validation_index")
        }
        manifest = {
            "created_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "format_version": TOKEN_DATA_FORMAT_VERSION,
            "tokenizer_path": _manifest_path_value(
                tokenizer_file,
                destination,
            ),
            "tokenizer_sha256": file_sha256(tokenizer_file),
            "tokenizer_actual_vocab_size": actual_vocab_size,
            "model_vocab_size": model_vocab_size,
            "eos_token": EOS_TOKEN,
            "eos_token_id": eos_token_id,
            "dtype": dtype.name,
            "context_length": context_length,
            "packing_policy": production_packing_policy(),
            "input_paths": {
                "train": _manifest_path_value(train_input, destination),
                "validation": _manifest_path_value(
                    validation_input,
                    destination,
                ),
            },
            "input_checksums": {
                "train": file_sha256(train_input),
                "validation": file_sha256(validation_input),
            },
            "output_paths": {
                name: path.name for name, path in final_paths.items()
            },
            "output_checksums": output_checksums,
            "file_byte_sizes": {
                "train": temporary_paths["train"].stat().st_size,
                "validation": temporary_paths["validation"].stat().st_size,
            },
            "train_token_count": train_token_count,
            "validation_token_count": validation_token_count,
            "train_document_count": train_document_count,
            "validation_document_count": validation_document_count,
            "complete_sequences": {
                "train": train_examples,
                "validation": validation_examples,
            },
            "trailing_tokens": {
                "train": train_trailing,
                "validation": validation_trailing,
            },
            "cli_arguments": {
                "model_vocab_size": model_vocab_size,
                "context_length": context_length,
                "stride": resolved_stride,
                "encoding_batch_size": encoding_batch_size,
                "overwrite": overwrite,
            },
        }
        _serialize_json(temporary_paths["manifest"], manifest)

        replacement_order = (
            "train",
            "validation",
            "train_index",
            "validation_index",
            "manifest",
        )
        for name in replacement_order:
            os.replace(temporary_paths[name], final_paths[name])
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)

    return TokenDataBuildResult(
        manifest_path=final_paths["manifest"],
        train_token_count=train_token_count,
        validation_token_count=validation_token_count,
        train_document_count=train_document_count,
        validation_document_count=validation_document_count,
        dtype=dtype.name,
    )


class MemmapTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length next-token examples backed by a NumPy memory map."""

    def __init__(
        self,
        token_file: str | Path,
        *,
        dtype: str | np.dtype,
        context_length: int,
        stride: int | None = None,
        model_vocab_size: int | None = None,
        token_offset: int = 0,
        token_count: int | None = None,
    ) -> None:
        self.token_file = Path(token_file)
        self.dtype = np.dtype(dtype)
        if self.dtype not in {np.dtype(np.uint16), np.dtype(np.uint32)}:
            raise ValueError(
                f"dtype must be uint16 or uint32; got {self.dtype.name}."
            )
        self.context_length = context_length
        self.stride = context_length if stride is None else stride

        file_size = self.token_file.stat().st_size
        if file_size % self.dtype.itemsize != 0:
            raise ValueError(
                f"Token file size {file_size} is not divisible by "
                f"dtype item size {self.dtype.itemsize}."
            )
        file_token_count = file_size // self.dtype.itemsize
        if token_offset < 0:
            raise ValueError("token_offset must be non-negative.")
        if token_count is None:
            token_count = file_token_count - token_offset
        if token_count <= 0:
            raise ValueError("token_count must be positive.")
        if token_offset + token_count > file_token_count:
            raise ValueError("Requested token range exceeds the token file.")
        self.token_offset = token_offset
        self.token_count = token_count
        self.example_count, self.trailing_token_count = count_packed_examples(
            token_count,
            context_length=context_length,
            stride=self.stride,
        )
        if self.example_count == 0:
            raise ValueError(
                f"Token file contains {token_count} tokens, but at least "
                f"{context_length + 1} are required."
            )

        self._tokens = np.memmap(
            self.token_file,
            mode="r",
            dtype=self.dtype,
            offset=token_offset * self.dtype.itemsize,
            shape=(token_count,),
        )
        if model_vocab_size is not None:
            choose_token_dtype(model_vocab_size)
            maximum_token_id = int(self._tokens.max())
            if maximum_token_id >= model_vocab_size:
                raise ValueError(
                    f"Token file contains ID {maximum_token_id}, outside model "
                    f"vocabulary size {model_vocab_size}."
                )

    def __len__(self) -> int:
        return self.example_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise TypeError(f"Dataset index must be an integer; got {index!r}.")
        if index < 0:
            index += self.example_count
        if index < 0 or index >= self.example_count:
            raise IndexError(
                f"Dataset index {index} is outside [0, {self.example_count})."
            )

        start = index * self.stride
        stop = start + self.context_length + 1
        token_window = np.asarray(self._tokens[start:stop], dtype=np.int64)
        if token_window.shape[0] != self.context_length + 1:
            raise IndexError("Token window would read beyond the token file.")
        input_ids = torch.from_numpy(token_window[:-1].copy())
        labels = torch.from_numpy(token_window[1:].copy())
        return input_ids, labels


def create_token_dataloader(
    dataset: MemmapTokenDataset,
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a deterministic PyTorch DataLoader for token examples."""

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError(
            f"batch_size must be a positive integer; got {batch_size!r}."
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )


def benchmark_token_dataloader(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_batches: int = 100,
) -> DataLoaderBenchmark:
    """Measure host-side batch loading and tensor collation throughput."""

    if (
        not isinstance(max_batches, int)
        or isinstance(max_batches, bool)
        or max_batches <= 0
    ):
        raise ValueError("max_batches must be a positive integer.")
    if len(data_loader) == 0:
        raise ValueError("DataLoader must not be empty.")

    batch_count = 0
    example_count = 0
    token_count = 0
    start = time.perf_counter()
    for input_ids, labels in data_loader:
        if input_ids.shape != labels.shape or input_ids.ndim != 2:
            raise ValueError("DataLoader returned invalid causal-LM tensors.")
        batch_count += 1
        example_count += input_ids.shape[0]
        token_count += labels.numel()
        if batch_count >= max_batches:
            break
    elapsed = time.perf_counter() - start
    if elapsed <= 0 or token_count <= 0:
        raise RuntimeError("DataLoader benchmark produced no measurable work.")
    return DataLoaderBenchmark(
        batches=batch_count,
        examples=example_count,
        tokens=token_count,
        elapsed_seconds=elapsed,
        tokens_per_second=token_count / elapsed,
    )


def inspect_token_data(manifest_path: str | Path) -> dict[str, object]:
    """Validate token binaries, indexes, checksums, and EOS positions."""

    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Token data manifest must contain a JSON object.")

    base_dir = manifest_file.parent
    dtype = np.dtype(manifest["dtype"])
    eos_token_id = manifest["eos_token_id"]
    output_paths = manifest["output_paths"]
    checksums = manifest["output_checksums"]
    split_summaries: dict[str, object] = {}

    for split_name in ("train", "validation"):
        token_path = base_dir / output_paths[split_name]
        index_path = base_dir / output_paths[f"{split_name}_index"]
        if file_sha256(token_path) != checksums[split_name]:
            raise ValueError(f"{split_name} token checksum does not match.")
        if file_sha256(index_path) != checksums[f"{split_name}_index"]:
            raise ValueError(f"{split_name} index checksum does not match.")

        file_size = token_path.stat().st_size
        if file_size % dtype.itemsize != 0:
            raise ValueError(f"{split_name} token file has a partial element.")
        tokens = np.memmap(token_path, mode="r", dtype=dtype)
        expected_start = 0
        document_count = 0
        for expected_ordinal, entry in enumerate(
            _iter_index_entries(index_path)
        ):
            if entry["document_ordinal"] != expected_ordinal:
                raise ValueError(f"{split_name} document ordinal mismatch.")
            if entry["token_start"] != expected_start:
                raise ValueError(f"{split_name} token ranges are not contiguous.")
            if entry["eos_token_position"] != (
                entry["token_start"] + entry["content_token_count"]
            ):
                raise ValueError(f"{split_name} EOS offset is invalid.")
            if entry["token_end"] != entry["eos_token_position"] + 1:
                raise ValueError(f"{split_name} token end is invalid.")
            if entry["total_stored_token_count"] != (
                entry["token_end"] - entry["token_start"]
            ):
                raise ValueError(f"{split_name} stored-token count is invalid.")
            if int(tokens[entry["eos_token_position"]]) != eos_token_id:
                raise ValueError(f"{split_name} EOS token value is invalid.")
            expected_start = entry["token_end"]
            document_count += 1
        if expected_start != len(tokens):
            raise ValueError(f"{split_name} index does not cover all tokens.")

        split_summaries[split_name] = {
            "document_count": document_count,
            "token_count": len(tokens),
            "byte_size": file_size,
            "sha256": checksums[split_name],
        }

    return {
        "format_version": manifest["format_version"],
        "dtype": dtype.name,
        "context_length": manifest["context_length"],
        "splits": split_summaries,
    }
