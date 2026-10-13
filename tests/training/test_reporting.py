"""Tests for strict training-metrics summaries."""

import json
from pathlib import Path
import tempfile

from src.reporting import load_metrics, summarize_training_metrics


def _record(step: int, validation_loss: float | None) -> dict[str, object]:
    return {
        "run_name": "test",
        "run_id": "run-id",
        "optimizer_step": step,
        "micro_step": step * 2,
        "total_tokens_seen": step * 16,
        "training_loss": 4.0 - step / 10,
        "validation_loss": validation_loss,
        "tokens_per_second": 1000.0 + step,
        "step_time_seconds": 0.1,
        "peak_allocated_vram_bytes": step * 100,
        "peak_reserved_vram_bytes": step * 200,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory) / "metrics.jsonl"
        records = [_record(1, None), _record(2, 3.0), _record(3, 2.5)]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        assert load_metrics(path) == records
        summary = summarize_training_metrics(path)
        assert summary.optimizer_steps == 3
        assert summary.micro_steps == 6
        assert summary.tokens_seen == 48
        assert summary.best_validation_loss == 2.5
        assert summary.best_validation_step == 3
        assert summary.validation_evaluations == 2
        assert summary.peak_allocated_vram_bytes == 300
        assert json.loads(json.dumps(summary.to_dict())) == summary.to_dict()

        path.write_text("", encoding="utf-8")
        try:
            load_metrics(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected empty metrics to fail.")

    print("All training-report tests passed.")


if __name__ == "__main__":
    main()
