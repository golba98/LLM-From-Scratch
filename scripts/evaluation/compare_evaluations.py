"""Compare trusted checkpoint-evaluation reports and select a best candidate."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection-metric",
        choices=("validation", "quality"),
        default="validation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _finite_number(value: object, field: str, path: Path) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{path}: {field} must be a finite number.")
    return float(value)


def _load_summary(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: evaluation report must be an object.")
    step = value.get("checkpoint_optimizer_step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(
            f"{path}: checkpoint_optimizer_step must be non-negative."
        )
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path}: samples must be a non-empty array.")

    category_metrics: dict[str, dict[str, object]] = {}
    repetition_rates: list[float] = []
    malformed_count = 0
    expected_outcomes = 0
    matched_outcomes = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"{path}: sample {index} must be an object.")
        category = sample.get("category")
        quality = sample.get("quality")
        if not isinstance(category, str) or not category:
            raise ValueError(f"{path}: sample {index} has an invalid category.")
        if not isinstance(quality, dict):
            raise ValueError(f"{path}: sample {index} has invalid quality.")
        repetition = _finite_number(
            quality.get("repeated_ngram_rate"),
            f"samples[{index}].quality.repeated_ngram_rate",
            path,
        )
        malformed = quality.get("malformed_character_count")
        if (
            not isinstance(malformed, int)
            or isinstance(malformed, bool)
            or malformed < 0
        ):
            raise ValueError(
                f"{path}: sample {index} malformed count must be non-negative."
            )
        expected_matches = sample.get("expected_term_matches", {})
        if (
            not isinstance(expected_matches, dict)
            or any(
                not isinstance(term, str) or not isinstance(matched, bool)
                for term, matched in expected_matches.items()
            )
        ):
            raise ValueError(
                f"{path}: sample {index} has invalid expected-term matches."
            )
        expected_pattern_match = sample.get("expected_pattern_match")
        if expected_pattern_match is not None and not isinstance(
            expected_pattern_match, bool
        ):
            raise ValueError(
                f"{path}: sample {index} has invalid expected-pattern result."
            )
        repetition_rates.append(repetition)
        malformed_count += malformed
        expected_outcomes += len(expected_matches)
        matched_outcomes += sum(expected_matches.values())
        if expected_pattern_match is not None:
            expected_outcomes += 1
            matched_outcomes += int(expected_pattern_match)
        category_metrics[category] = {
            "repeated_ngram_rate": repetition,
            "malformed_character_count": malformed,
            "word_count": quality.get("word_count"),
            "expected_term_match_rate": (
                None
                if not expected_matches
                else sum(expected_matches.values()) / len(expected_matches)
            ),
            "expected_pattern_match": expected_pattern_match,
        }

    validation_loss = value.get("validation_loss")
    if validation_loss is not None:
        validation_loss = _finite_number(
            validation_loss,
            "validation_loss",
            path,
        )
    return {
        "report": str(path),
        "checkpoint": value.get("checkpoint"),
        "checkpoint_run_id": value.get("checkpoint_run_id"),
        "checkpoint_optimizer_step": step,
        "tokenizer_sha256": value.get("tokenizer_sha256"),
        "prompt_suite_sha256": value.get("prompt_suite_sha256"),
        "validation_token_sha256": value.get("validation_token_sha256"),
        "validation_loss": validation_loss,
        "validation_perplexity": value.get("validation_perplexity"),
        "mean_repeated_ngram_rate": (
            sum(repetition_rates) / len(repetition_rates)
        ),
        "malformed_character_count": malformed_count,
        "expected_outcome_count": expected_outcomes,
        "matched_expected_outcome_count": matched_outcomes,
        "expected_outcome_match_rate": (
            None
            if expected_outcomes == 0
            else matched_outcomes / expected_outcomes
        ),
        "category_metrics": category_metrics,
    }


def _atomic_json(path: Path, value: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if len(arguments.reports) < 2:
        raise ValueError("At least two evaluation reports are required.")
    if arguments.output.exists() and not arguments.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite comparison: {arguments.output}"
        )
    candidates = [_load_summary(path) for path in arguments.reports]
    prompt_checksums = {
        candidate["prompt_suite_sha256"] for candidate in candidates
    }
    if None in prompt_checksums or len(prompt_checksums) != 1:
        raise ValueError(
            "Evaluation reports must use the same checksummed prompt suite."
        )
    if arguments.selection_metric == "validation":
        tokenizer_checksums = {
            candidate["tokenizer_sha256"] for candidate in candidates
        }
        validation_checksums = {
            candidate["validation_token_sha256"] for candidate in candidates
        }
        if (
            None in tokenizer_checksums
            or len(tokenizer_checksums) != 1
            or None in validation_checksums
            or len(validation_checksums) != 1
            or any(
                candidate["validation_loss"] is None
                for candidate in candidates
            )
        ):
            raise ValueError(
                "Validation ranking requires the same tokenizer and "
                "validation token data with finite losses."
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                item["validation_loss"],
                item["mean_repeated_ngram_rate"],
                item["malformed_character_count"],
                -item["checkpoint_optimizer_step"],
            ),
        )
        selection_rule = (
            "lowest validation loss, then repetition, malformed characters, "
            "and later optimizer step; tokenizer and validation checksums "
            "must match"
        )
    else:
        if any(
            candidate["expected_outcome_match_rate"] is None
            for candidate in candidates
        ):
            raise ValueError(
                "Quality ranking requires at least one expected outcome in "
                "every report."
            )
        ranked = sorted(
            candidates,
            key=lambda item: (
                -item["expected_outcome_match_rate"],
                item["mean_repeated_ngram_rate"],
                item["malformed_character_count"],
                -item["checkpoint_optimizer_step"],
            ),
        )
        selection_rule = (
            "highest expected-outcome match rate, then lowest repetition, "
            "malformed characters, and later optimizer step"
        )
    comparison = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": arguments.selection_metric,
        "selection_rule": selection_rule,
        "candidate_count": len(ranked),
        "best_checkpoint": ranked[0]["checkpoint"],
        "best_report": ranked[0]["report"],
        "candidates": ranked,
    }
    _atomic_json(arguments.output, comparison, overwrite=arguments.overwrite)
    print(f"Candidates: {len(ranked)}")
    print(f"Best checkpoint: {comparison['best_checkpoint']}")
    print(f"Comparison: {arguments.output}")
    return comparison


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
