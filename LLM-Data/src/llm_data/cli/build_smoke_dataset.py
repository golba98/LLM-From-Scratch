"""Build a reproducible million-token smoke corpus from the original fixture."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
import json
from pathlib import Path
import sys



from llm_data.cli.prepare_dataset import prepare_dataset
from llm_data.token_data import build_token_data


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("tests/fixtures/data/sample.txt"),
    )
    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=generated_path('data/raw/smoke-million.jsonl'),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=generated_path('data/processed/smoke-million'),
    )
    parser.add_argument(
        "--tokenized-dir",
        type=Path,
        default=generated_path('data/tokenized/smoke-million'),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=generated_path('checkpoints/tokenizer-smoke/tokenizer.json'),
    )
    parser.add_argument("--documents", type=int, default=10_000)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.documents <= 1:
        raise ValueError("--documents must be greater than one.")
    paragraphs = [
        paragraph.strip()
        for paragraph in arguments.source.read_text(encoding="utf-8").split("\n\n")
        if paragraph.strip()
    ]
    if not paragraphs:
        raise ValueError("Smoke source contains no paragraphs.")
    if arguments.raw_jsonl.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Raw smoke corpus exists: {arguments.raw_jsonl}"
        )
    arguments.raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with arguments.raw_jsonl.open("w", encoding="utf-8") as output_file:
        for ordinal in range(arguments.documents):
            base_index = ordinal % len(paragraphs)
            record = {
                "text": (
                    f"Codexa smoke document {ordinal:05d}. "
                    f"{paragraphs[base_index]}"
                ),
                "source": arguments.source.name,
                "document_id": f"smoke-{ordinal:05d}",
                "metadata": {
                    "synthetic_repetition": True,
                    "base_paragraph_index": base_index,
                },
            }
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    statistics = prepare_dataset(
        [arguments.raw_jsonl],
        output_dir=arguments.processed_dir,
        dataset_name="Codexa reproducible smoke corpus",
        dataset_license="Original fixture; synthetic deterministic expansion",
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
        split_text_on_blank_lines=False,
    )
    tokenized = build_token_data(
        train_jsonl=arguments.processed_dir / "train.jsonl",
        validation_jsonl=arguments.processed_dir / "validation.jsonl",
        tokenizer_path=arguments.tokenizer,
        output_dir=arguments.tokenized_dir,
        model_vocab_size=8192,
        context_length=256,
        overwrite=arguments.overwrite,
    )
    result = {
        "raw_documents": arguments.documents,
        "cleaned_documents": statistics.cleaned_document_count,
        "train_documents": tokenized.train_document_count,
        "validation_documents": tokenized.validation_document_count,
        "train_tokens": tokenized.train_token_count,
        "validation_tokens": tokenized.validation_token_count,
        "token_manifest": str(tokenized.manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
