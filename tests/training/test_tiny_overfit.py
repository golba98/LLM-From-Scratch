"""Lightweight structural tests for the tiny overfit experiment."""

import json
from pathlib import Path
import tempfile

import torch

from scripts.diagnostics.run_tiny_overfit import _single_example
from src.config import load_config
from src.model import LanguageModel, count_parameters
from src.tokenizer import train_tokenizer


def main() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    with torch.device("meta"):
        model = LanguageModel(config.model)
    parameter_count = count_parameters(model)
    assert 10_000_000 <= parameter_count <= 30_000_000
    assert config.model.context_length == 64

    text = Path("tests/fixtures/data/tiny_overfit.txt").read_text(
        encoding="utf-8"
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text(
            json.dumps({"text": text, "source": "tiny-overfit-test"}) + "\n",
            encoding="utf-8",
        )
        tokenizer_path = train_tokenizer(
            [corpus],
            output_dir=root / "tokenizer",
            vocab_size=300,
            min_frequency=1,
        ).tokenizer_path
        input_ids, labels, content_ids = _single_example(
            text,
            tokenizer_path=tokenizer_path,
            context_length=config.model.context_length,
        )
    assert input_ids.shape == labels.shape == (1, 64)
    assert torch.equal(input_ids[:, 1:], labels[:, :-1])
    assert content_ids
    print(f"Tiny model parameter count: {parameter_count:,}")
    print(f"Tiny context length: {config.model.context_length}")
    print("All tiny-overfit structural tests passed.")


if __name__ == "__main__":
    main()
