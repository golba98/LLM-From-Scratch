"""Tokenize cleaned JSONL splits into memory-mappable binary arrays."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
from pathlib import Path
import sys



from llm_data.token_data import build_token_data


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the token-dataset command-line parser."""

    parser = argparse.ArgumentParser(
        description="Create binary token datasets from cleaned JSONL splits."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=generated_path('data/tokenized'),
    )
    parser.add_argument("--model-vocab-size", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--encoding-batch-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Build token data from command-line arguments."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        result = build_token_data(
            train_jsonl=arguments.train_jsonl,
            validation_jsonl=arguments.validation_jsonl,
            tokenizer_path=arguments.tokenizer,
            output_dir=arguments.output_dir,
            model_vocab_size=arguments.model_vocab_size,
            context_length=arguments.context_length,
            stride=arguments.stride,
            encoding_batch_size=arguments.encoding_batch_size,
            overwrite=arguments.overwrite,
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    print(f"Manifest: {result.manifest_path}")
    print(f"Dtype: {result.dtype}")
    print(f"Training documents: {result.train_document_count}")
    print(f"Training tokens: {result.train_token_count}")
    print(f"Validation documents: {result.validation_document_count}")
    print(f"Validation tokens: {result.validation_token_count}")


if __name__ == "__main__":
    main()
