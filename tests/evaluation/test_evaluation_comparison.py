"""Executable tests for strict checkpoint-evaluation comparison."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from scripts.evaluation.compare_evaluations import run


def _report(path: Path, *, step: int, loss: float, repetition: float) -> None:
    path.write_text(
        json.dumps(
            {
                "checkpoint": f"checkpoint-{step}.pt",
                "checkpoint_optimizer_step": step,
                "checkpoint_run_id": "test-run",
                "tokenizer_sha256": "tokenizer-checksum",
                "prompt_suite_sha256": "prompt-checksum",
                "validation_token_sha256": "validation-checksum",
                "validation_loss": loss,
                "validation_perplexity": 2.0**loss,
                "samples": [
                    {
                        "category": "coherence",
                        "expected_term_matches": {"answer": step == 200},
                        "expected_pattern_match": step == 200,
                        "quality": {
                            "repeated_ngram_rate": repetition,
                            "malformed_character_count": 0,
                            "word_count": 12,
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _raises(exception_type: type[BaseException], operation) -> None:
    try:
        operation()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        first = root / "step-100.json"
        second = root / "step-200.json"
        output = root / "comparison.json"
        _report(first, step=100, loss=2.5, repetition=0.0)
        _report(second, step=200, loss=2.0, repetition=0.1)
        comparison = run(
            SimpleNamespace(
                reports=[first, second],
                output=output,
                selection_metric="validation",
                overwrite=False,
            )
        )
        assert comparison["candidate_count"] == 2
        assert comparison["best_checkpoint"] == "checkpoint-200.pt"
        saved = json.loads(output.read_text(encoding="utf-8"))
        assert saved["candidates"][0]["checkpoint_optimizer_step"] == 200
        assert saved["candidates"][1]["checkpoint_optimizer_step"] == 100
        assert (
            saved["candidates"][0]["category_metrics"]["coherence"][
                "expected_term_match_rate"
            ]
            == 1.0
        )
        assert (
            saved["candidates"][0]["category_metrics"]["coherence"][
                "expected_pattern_match"
            ]
            is True
        )
        second_value = json.loads(second.read_text(encoding="utf-8"))
        second_value["tokenizer_sha256"] = "different-tokenizer"
        second.write_text(
            json.dumps(second_value) + "\n",
            encoding="utf-8",
        )
        quality = run(
            SimpleNamespace(
                reports=[first, second],
                output=root / "quality.json",
                selection_metric="quality",
                overwrite=False,
            )
        )
        assert quality["selection_metric"] == "quality"
        assert quality["best_checkpoint"] == "checkpoint-200.pt"
        _raises(
            ValueError,
            lambda: run(
                SimpleNamespace(
                    reports=[first, second],
                    output=root / "invalid-validation.json",
                    selection_metric="validation",
                    overwrite=False,
                )
            ),
        )
        _raises(
            FileExistsError,
            lambda: run(
                SimpleNamespace(
                    reports=[first, second],
                    output=output,
                    selection_metric="validation",
                    overwrite=False,
                )
            ),
        )
        _raises(
            ValueError,
            lambda: run(
                SimpleNamespace(
                    reports=[first],
                    output=root / "one.json",
                    selection_metric="validation",
                    overwrite=False,
                )
            ),
        )

    print("All evaluation-comparison tests passed.")


if __name__ == "__main__":
    main()
