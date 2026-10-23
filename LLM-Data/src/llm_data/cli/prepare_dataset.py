"""Prepare deterministic cleaned JSONL splits from local text files."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
import sys



from llm_data.data.cleaning import CLEANING_VERSION, clean_and_deduplicate
from llm_data.data.io import load_documents, write_json, write_jsonl
from llm_data.data.split import split_documents
from llm_data.data.statistics import DatasetStatistics, build_statistics


def prepare_dataset(
    input_paths: Sequence[str | Path],
    *,
    output_dir: str | Path,
    dataset_name: str,
    dataset_license: str,
    validation_ratio: float = 0.05,
    seed: int = 42,
    split_text_on_blank_lines: bool = False,
) -> DatasetStatistics:
    """Run the complete local dataset preparation pipeline."""

    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")
    if not dataset_license.strip():
        raise ValueError("dataset_license must not be empty.")

    normalized_inputs = [Path(path) for path in input_paths]
    raw_documents = load_documents(
        normalized_inputs,
        split_text_on_blank_lines=split_text_on_blank_lines,
    )
    cleaning_result = clean_and_deduplicate(raw_documents)
    training_documents, validation_documents = split_documents(
        cleaning_result.documents,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    destination = Path(output_dir)
    train_path = destination / "train.jsonl"
    validation_path = destination / "validation.jsonl"
    statistics_path = destination / "dataset_stats.json"
    manifest_path = destination / "dataset_manifest.json"

    train_sha256 = write_jsonl(train_path, training_documents)
    validation_sha256 = write_jsonl(
        validation_path,
        validation_documents,
    )
    statistics = build_statistics(
        input_file_count=len(normalized_inputs),
        raw_document_count=len(raw_documents),
        cleaned_documents=cleaning_result.documents,
        empty_documents_removed=cleaning_result.empty_documents_removed,
        duplicate_documents_removed=(
            cleaning_result.duplicate_documents_removed
        ),
        training_documents=training_documents,
        validation_documents=validation_documents,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )
    statistics_data = statistics.to_dict()
    write_json(statistics_path, statistics_data)

    output_paths = {
        "train": str(train_path.resolve()),
        "validation": str(validation_path.resolve()),
        "statistics": str(statistics_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "input_paths": [str(path.resolve()) for path in normalized_inputs],
        "dataset_name": dataset_name,
        "license": dataset_license,
        "cleaning_version": CLEANING_VERSION,
        "validation_ratio": float(validation_ratio),
        "seed": seed,
        "output_paths": output_paths,
        "checksums": {
            "train_sha256": train_sha256,
            "validation_sha256": validation_sha256,
        },
        "statistics": statistics_data,
        "cli_arguments": {
            "dataset_name": dataset_name,
            "license": dataset_license,
            "output_dir": str(destination),
            "validation_ratio": float(validation_ratio),
            "seed": seed,
            "split_text_on_blank_lines": split_text_on_blank_lines,
        },
    }
    write_json(manifest_path, manifest)
    return statistics


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Clean and split local UTF-8 text or JSONL datasets."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .txt, .jsonl, or .ndjson files in stable order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=generated_path('data/processed'),
        help="Destination directory for prepared outputs.",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--license", dest="dataset_license", required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-text-on-blank-lines",
        action="store_true",
        help="Treat blank-line-separated .txt paragraphs as documents.",
    )
    return parser


def main() -> None:
    """Run dataset preparation from command-line arguments."""

    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        statistics = prepare_dataset(
            arguments.inputs,
            output_dir=arguments.output_dir,
            dataset_name=arguments.dataset_name,
            dataset_license=arguments.dataset_license,
            validation_ratio=arguments.validation_ratio,
            seed=arguments.seed,
            split_text_on_blank_lines=(
                arguments.split_text_on_blank_lines
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))

    print(f"Prepared {statistics.cleaned_document_count} cleaned documents.")
    print(f"Training documents: {statistics.training_document_count}")
    print(f"Validation documents: {statistics.validation_document_count}")
    print(f"Train SHA-256: {statistics.train_sha256}")
    print(f"Validation SHA-256: {statistics.validation_sha256}")


if __name__ == "__main__":
    main()
