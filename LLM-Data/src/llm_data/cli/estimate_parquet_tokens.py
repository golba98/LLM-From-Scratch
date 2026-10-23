"""Estimate exact cleaned Codexa token counts in Parquet text shards."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import pyarrow.parquet as parquet



from llm_data.data.cleaning import clean_text
from llm_data.token_data import file_sha256
from llm_tokenizer.tokenizer import EOS_TOKEN, load_tokenizer


@dataclass(frozen=True)
class ShardTokenEstimate:
    path: str
    sha256: str
    raw_documents: int
    cleaned_documents: int
    removed_empty_documents: int
    content_tokens: int
    stored_tokens_with_eos: int
    characters: int
    characters_per_content_token: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_shard(
    path: Path,
    *,
    tokenizer_path: Path,
    maximum_documents: int | None = None,
) -> ShardTokenEstimate:
    """Clean and tokenize a shard without retaining documents or token IDs."""

    if maximum_documents is not None and maximum_documents <= 0:
        raise ValueError("maximum_documents must be positive when supplied.")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.token_to_id(EOS_TOKEN) != 2:
        raise ValueError("Tokenizer must use <eos>=2.")
    parquet_file = parquet.ParquetFile(path)
    if "text" not in parquet_file.schema.names:
        raise ValueError(f"{path}: Parquet schema has no text column.")

    raw_documents = 0
    cleaned_documents = 0
    removed_empty_documents = 0
    content_tokens = 0
    characters = 0
    stop = False
    for batch in parquet_file.iter_batches(batch_size=1024, columns=["text"]):
        for value in batch.column(0).to_pylist():
            raw_documents += 1
            if not isinstance(value, str):
                raise ValueError(f"{path}: text column must contain strings.")
            cleaned = clean_text(value)
            if cleaned is None:
                removed_empty_documents += 1
            else:
                cleaned_documents += 1
                characters += len(cleaned)
                content_tokens += len(
                    tokenizer.encode(
                        cleaned,
                        add_special_tokens=False,
                    ).ids
                )
            if (
                maximum_documents is not None
                and raw_documents >= maximum_documents
            ):
                stop = True
                break
        if stop:
            break
    return ShardTokenEstimate(
        path=str(path),
        sha256=file_sha256(path),
        raw_documents=raw_documents,
        cleaned_documents=cleaned_documents,
        removed_empty_documents=removed_empty_documents,
        content_tokens=content_tokens,
        stored_tokens_with_eos=content_tokens + cleaned_documents,
        characters=characters,
        characters_per_content_token=(
            characters / content_tokens if content_tokens else 0.0
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    estimates = [
        estimate_shard(
            path,
            tokenizer_path=arguments.tokenizer,
            maximum_documents=arguments.maximum_documents,
        ).to_dict()
        for path in arguments.inputs
    ]
    output = {
        "shards": estimates,
        "total_stored_tokens_with_eos": sum(
            int(estimate["stored_tokens_with_eos"])
            for estimate in estimates
        ),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
