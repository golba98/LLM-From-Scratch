"""Executable tests for flushed GPU telemetry logging."""

import json
from pathlib import Path
import tempfile

from scripts.monitoring.monitor_gpu import monitor_gpu
from src.hardware_monitor import parse_nvidia_smi_row, summarize_gpu_metrics


def _raises(exception_type: type[BaseException], operation) -> None:
    try:
        operation()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def main() -> None:
    parsed = parse_nvidia_smi_row(
        "NVIDIA Test GPU, 63, 250.5, 11840, 16376, 99, 2505, 11201"
    )
    assert parsed["gpu_name"] == "NVIDIA Test GPU"
    assert parsed["temperature_celsius"] == 63
    assert parsed["power_draw_watts"] == 250.5
    assert parsed["timestamp_utc"].endswith("+00:00")
    _raises(ValueError, lambda: parse_nvidia_smi_row("too,few,fields"))

    with tempfile.TemporaryDirectory() as temporary_directory:
        output = Path(temporary_directory) / "gpu.jsonl"
        sequence = iter(
            [
                {**parsed, "utilization_percent": 90},
                {**parsed, "utilization_percent": 95},
            ]
        )
        sleeps: list[float] = []
        count = monitor_gpu(
            output,
            interval_seconds=2.0,
            maximum_samples=2,
            overwrite=False,
            append=False,
            query=lambda: next(sequence),
            sleep=sleeps.append,
        )
        assert count == 2
        records = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["utilization_percent"] for record in records] == [
            90,
            95,
        ]
        assert sleeps == [2.0]
        _raises(
            FileExistsError,
            lambda: monitor_gpu(
                output,
                interval_seconds=1,
                maximum_samples=1,
                overwrite=False,
                append=False,
                query=lambda: parsed,
            ),
        )
        assert (
            monitor_gpu(
                output,
                interval_seconds=1,
                maximum_samples=1,
                overwrite=False,
                append=True,
                query=lambda: parsed,
            )
            == 1
        )
        assert len(output.read_text(encoding="utf-8").splitlines()) == 3
        summary = summarize_gpu_metrics(output)
        assert summary.sample_count == 3
        assert summary.minimum_temperature_celsius == 63
        assert summary.maximum_temperature_celsius == 63
        assert summary.maximum_memory_used_mib == 11840
        assert summary.mean_utilization_percent == (90 + 95 + 99) / 3
        _raises(
            ValueError,
            lambda: monitor_gpu(
                output,
                interval_seconds=0,
                maximum_samples=1,
                overwrite=True,
                append=False,
                query=lambda: parsed,
            ),
        )

    print("All GPU-monitor tests passed.")


if __name__ == "__main__":
    main()
