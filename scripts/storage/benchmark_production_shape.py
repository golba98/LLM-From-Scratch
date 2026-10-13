"""Benchmark sustained production-shape training, recovery, and evaluation."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from llm_training.config import load_config
from llm_architecture.model import LanguageModel, count_parameters


CANDIDATE_TOKEN_BUDGETS = (1_000_000_000, 3_000_000_000, 6_000_000_000, 10_000_000_000)


def _run(command: list[str]) -> float:
    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        if completed.returncode in {137, -9}:
            raise RuntimeError("Production-shape command was killed; probable out-of-memory condition.")
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")
    return time.perf_counter() - started


def _metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists() and path.name == "metrics.jsonl":
        fallback = path.with_name("train_metrics.jsonl")
        if fallback.exists():
            path = fallback
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("Benchmark produced no optimizer-step metrics.")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/1b.yaml"))
    parser.add_argument("--train-token-file", type=Path, required=True)
    parser.add_argument("--validation-token-file", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="codexa-1b-production-shape")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--minimum-free-vram-bytes", type=int, default=1_500_000_000)
    arguments = parser.parse_args()
    if arguments.steps <= arguments.warmup_steps or arguments.warmup_steps < 1:
        raise ValueError("--steps must exceed a positive --warmup-steps.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The production-shape gate requires a BF16-capable CUDA GPU.")
    config = load_config(arguments.config)
    if config.model.context_length != 2048:
        raise ValueError("Production-shape context length must be 2048.")
    parameter_count = count_parameters(LanguageModel(config.model))
    expected_parameter_count = 934_356_480
    if parameter_count != expected_parameter_count:
        raise ValueError(
            "Unexpected production parameter count for the selected 16K "
            f"tokenizer: expected {expected_parameter_count}, got {parameter_count}."
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=False)
    benchmark_config = arguments.output_dir / "benchmark.yaml"
    raw = arguments.config.read_text(encoding="utf-8")
    import yaml

    values = yaml.safe_load(raw)
    values["training"]["max_steps"] = arguments.steps
    values["training"]["warmup_steps"] = arguments.warmup_steps
    values["training"]["checkpoint_interval"] = arguments.steps
    values["training"]["evaluation_interval"] = arguments.steps
    benchmark_config.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    log_dir = arguments.output_dir / "logs"
    checkpoint_dir = arguments.output_dir / "checkpoints"
    train_command = [
        sys.executable,
        "-m",
        "scripts.train",
        "--config",
        str(benchmark_config),
        "--train-token-file",
        str(arguments.train_token_file),
        "--validation-token-file",
        str(arguments.validation_token_file),
        "--token-manifest",
        str(arguments.token_manifest),
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--gradient-checkpointing",
        "--optimizer",
        "adamw8bit",
        "--max-validation-batches",
        str(arguments.validation_batches),
        "--run-name",
        arguments.run_name,
        "--log-dir",
        str(log_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    total_training_seconds = _run(train_command)
    records = _metrics(log_dir / arguments.run_name / "metrics.jsonl")
    sustained = records[arguments.warmup_steps :]
    if not sustained:
        raise ValueError("No sustained steps remain after warmup exclusion.")
    if any(not math.isfinite(float(record["training_loss"])) for record in records):
        raise FloatingPointError("Benchmark produced non-finite loss.")
    checkpoint = checkpoint_dir / arguments.run_name / "latest.pt"
    completion = json.loads(
        checkpoint.with_suffix(".pt.complete.json").read_text(encoding="utf-8")
    )
    evaluation_path = arguments.output_dir / "evaluation.json"
    evaluation_command = [
        sys.executable,
        "-m",
        "scripts.evaluation.evaluate_checkpoint",
        "--checkpoint",
        str(checkpoint),
        "--tokenizer",
        str(arguments.tokenizer),
        "--validation-token-file",
        str(arguments.validation_token_file),
        "--token-manifest",
        str(arguments.token_manifest),
        "--max-validation-batches",
        str(arguments.validation_batches),
        "--device",
        "cuda",
        "--output",
        str(evaluation_path),
    ]
    evaluation_wall_seconds = _run(evaluation_command)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    tokens_per_second = statistics.median(float(item["tokens_per_second"]) for item in sustained)
    sequences_per_second = tokens_per_second / config.model.context_length
    peak_reserved = max(int(item["peak_reserved_vram_bytes"]) for item in records)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    free_headroom = total_vram - peak_reserved
    if free_headroom < arguments.minimum_free_vram_bytes:
        raise RuntimeError(
            f"Insufficient VRAM safety headroom: {free_headroom} bytes; "
            f"required {arguments.minimum_free_vram_bytes}."
        )
    checkpoint_bytes = int(completion["bytes"])
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "production_shape": {
            "parameter_count": parameter_count,
            "context_length": config.model.context_length,
            "micro_batch_size": config.training.micro_batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "tokens_per_optimizer_step": config.model.context_length * config.training.micro_batch_size * config.training.gradient_accumulation_steps,
            "precision": "bf16",
            "optimizer": "adamw8bit",
            "activation_checkpointing": True,
            "data_loader": "MemmapTokenDataset",
        },
        "measurement": {
            "warmup_steps_excluded": arguments.warmup_steps,
            "sustained_steps": len(sustained),
            "median_tokens_per_second": tokens_per_second,
            "median_sequences_per_second": sequences_per_second,
            "median_optimizer_step_seconds": statistics.median(float(item["step_time_seconds"]) for item in sustained),
            "median_forward_backward_seconds": statistics.median(float(item["forward_backward_seconds"]) for item in sustained),
            "median_optimizer_update_seconds": statistics.median(float(item["optimizer_update_seconds"]) for item in sustained),
            "peak_allocated_gpu_bytes": max(int(item["peak_allocated_vram_bytes"]) for item in records),
            "peak_reserved_gpu_bytes": peak_reserved,
            "free_vram_headroom_bytes": free_headroom,
            "checkpoint_write_seconds": completion["write_duration_seconds"],
            "validation_seconds": evaluation["validation_duration_seconds"],
            "generation_seconds": evaluation["generation_duration_seconds"],
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "training_wall_seconds": total_training_seconds,
        },
        "runtime_estimates": {
            str(budget): {
                "seconds": budget / tokens_per_second,
                "days": budget / tokens_per_second / 86400,
            }
            for budget in CANDIDATE_TOKEN_BUDGETS
        },
        "storage_estimates": {
            "bytes_per_checkpoint": checkpoint_bytes,
            "three_recovery_plus_four_milestones_bytes": checkpoint_bytes * 7,
        },
        "commands": {"train": train_command, "evaluate": evaluation_command},
    }
    output = arguments.output_dir / "production_shape_benchmark.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
