"""Inspect a saved tokenizer against cleaned JSONL documents."""

import argparse
import json
from pathlib import Path
import sys



from llm_tokenizer.tokenizer import (
    inspect_tokenizer,
    load_tokenizer,
    load_tokenizer_corpus,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the tokenizer-inspection command-line parser."""

    parser = argparse.ArgumentParser(
        description="Inspect a saved tokenizer on cleaned JSONL data."
    )
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    """Inspect a tokenizer from command-line arguments."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        tokenizer = load_tokenizer(arguments.tokenizer)
        documents = load_tokenizer_corpus(arguments.inputs)
        inspection = inspect_tokenizer(
            tokenizer,
            (document.text for document in documents),
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    if arguments.as_json:
        print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
        return

    print(f"Documents: {inspection.document_count}")
    print(f"Characters: {inspection.total_characters}")
    print(f"UTF-8 bytes: {inspection.total_utf8_bytes}")
    print(f"Tokens: {inspection.total_tokens}")
    print(f"Unknown tokens: {inspection.unknown_token_count}")
    print(f"Unknown-token rate: {inspection.unknown_token_rate:.6f}")
    print(
        "Average characters per token: "
        f"{inspection.average_characters_per_token:.6f}"
    )


if __name__ == "__main__":
    main()
