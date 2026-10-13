"""Executable tests for deterministic mixed-precision training."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from scripts.train import _record_resume, build_argument_parser, run
from src.model import LanguageModel, ModelConfig
from src.token_data import file_sha256
from src.training import (
    CyclingDataIterator,
    JsonlRunLogger,
    TrainingMetrics,
    TrainingState,
    cosine_learning_rate,
    create_adamw_optimizer,
    create_grad_scaler,
    evaluate,
    requires_grad_scaler,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
    utc_timestamp,
)


class EmptyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise IndexError(index)


class NonFiniteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.weight * torch.ones(
            (*input_ids.shape, 4),
            device=input_ids.device,
        )
        return logits, logits.sum() * torch.tensor(
            float("nan"),
            device=input_ids.device,
        )


def assert_raises(
    exception_type: type[BaseException],
    callable_object: object,
    *args: object,
    **kwargs: object,
) -> BaseException:
    try:
        callable_object(*args, **kwargs)  # type: ignore[operator]
    except exception_type as error:
        return error
    raise AssertionError(f"Expected {exception_type.__name__}.")


def tiny_model() -> LanguageModel:
    return LanguageModel(
        ModelConfig(
            vocab_size=32,
            context_length=4,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
            dropout=0.0,
        )
    )


def tiny_loader(
    *,
    examples: int = 3,
    batch_size: int = 2,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    values = torch.arange(examples * 5, dtype=torch.long).reshape(examples, 5)
    values %= 32
    dataset = TensorDataset(values[:, :-1], values[:, 1:])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def test_state_and_metrics() -> None:
    state = TrainingState(
        micro_step=2,
        optimizer_step=1,
        tokens_seen=16,
        completed_epochs=1,
        best_validation_loss=3.0,
    )
    assert state.to_dict() == {
        "micro_step": 2,
        "optimizer_step": 1,
        "tokens_seen": 16,
        "completed_epochs": 1,
        "batches_in_epoch": 0,
        "best_validation_loss": 3.0,
    }
    timestamp = utc_timestamp()
    assert timestamp.endswith("+00:00")
    metrics = TrainingMetrics(
        run_name="test",
        run_id="run",
        micro_step=2,
        optimizer_step=1,
        completed_epochs=0,
        training_loss=3.0,
        validation_loss=None,
        learning_rate=1e-3,
        tokens_processed=16,
        total_tokens_seen=16,
        tokens_per_second=100.0,
        step_time_seconds=0.16,
        forward_backward_seconds=0.14,
        optimizer_update_seconds=0.02,
        gradient_norm=1.0,
        allocated_vram_bytes=0,
        reserved_vram_bytes=0,
        peak_allocated_vram_bytes=0,
        peak_reserved_vram_bytes=0,
        device="cpu",
        precision="fp32",
        timestamp_utc=timestamp,
    )
    encoded = json.dumps(metrics.to_dict(), allow_nan=False)
    assert json.loads(encoded)["validation_loss"] is None
    invalid = TrainingMetrics(**{**asdict(metrics), "training_loss": float("nan")})
    assert_raises(ValueError, invalid.to_dict)


def test_seed_device_and_precision() -> None:
    set_deterministic_seed(7)
    first = (np.random.rand(), torch.rand(3))
    set_deterministic_seed(7)
    second = (np.random.rand(), torch.rand(3))
    assert first[0] == second[0]
    assert torch.equal(first[1], second[1])
    assert_raises(ValueError, set_deterministic_seed, -1)

    auto = resolve_device("auto")
    assert auto.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert_raises(RuntimeError, resolve_device, "cuda")
    assert_raises(ValueError, resolve_device, "mps")

    cpu_policy = resolve_precision("fp32", torch.device("cpu"))
    assert cpu_policy.name == "fp32"
    assert not requires_grad_scaler(cpu_policy)
    assert create_grad_scaler(torch.device("cpu"), cpu_policy) is None
    assert_raises(
        ValueError,
        resolve_precision,
        "bf16",
        torch.device("cpu"),
    )
    assert_raises(
        ValueError,
        resolve_precision,
        "fp16",
        torch.device("cpu"),
    )
    if torch.cuda.is_available():
        fp16 = resolve_precision("fp16", torch.device("cuda"))
        assert requires_grad_scaler(fp16)
        assert create_grad_scaler(torch.device("cuda"), fp16) is not None
        if torch.cuda.is_bf16_supported():
            bf16 = resolve_precision("bf16", torch.device("cuda"))
            assert not requires_grad_scaler(bf16)


@pytest.mark.usefixtures("development_training_environment")
def test_optimizer_groups() -> None:
    model = tiny_model()
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=1e-3,
        weight_decay=0.1,
    )
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(optimizer_parameters) == len({id(p) for p in optimizer_parameters})
    assert {id(p) for p in optimizer_parameters} == {id(p) for p in trainable}
    assert {group["group_name"] for group in optimizer.param_groups} == {
        "decay",
        "no_decay",
    }
    decay_group = next(
        group for group in optimizer.param_groups if group["group_name"] == "decay"
    )
    no_decay_group = next(
        group
        for group in optimizer.param_groups
        if group["group_name"] == "no_decay"
    )
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert all(parameter.ndim > 1 for parameter in decay_group["params"])


def test_scheduler() -> None:
    arguments = {
        "warmup_steps": 2,
        "max_steps": 10,
        "peak_learning_rate": 1.0,
        "minimum_learning_rate": 0.1,
    }
    assert cosine_learning_rate(0, **arguments) == 0.0
    assert cosine_learning_rate(1, **arguments) == 0.5
    assert cosine_learning_rate(2, **arguments) == 1.0
    interior = cosine_learning_rate(6, **arguments)
    assert 0.1 < interior < 1.0
    assert cosine_learning_rate(10, **arguments) == 0.1
    assert cosine_learning_rate(50, **arguments) == 0.1
    assert all(cosine_learning_rate(step, **arguments) >= 0 for step in range(20))
    no_warmup = {**arguments, "warmup_steps": 0}
    assert cosine_learning_rate(0, **no_warmup) == 1.0
    invalid = [
        (-1, arguments),
        (0, {**arguments, "warmup_steps": -1}),
        (0, {**arguments, "max_steps": 0}),
        (0, {**arguments, "warmup_steps": 10}),
        (0, {**arguments, "peak_learning_rate": 0.0}),
        (0, {**arguments, "minimum_learning_rate": -0.1}),
        (0, {**arguments, "minimum_learning_rate": 2.0}),
    ]
    for step, keyword_arguments in invalid:
        assert_raises(
            ValueError,
            cosine_learning_rate,
            step,
            **keyword_arguments,
        )


def run_cpu_training(
    *,
    max_steps: int = 2,
    accumulation: int = 2,
    max_micro_steps: int | None = None,
) -> tuple[TrainingState, list[TrainingMetrics], LanguageModel]:
    set_deterministic_seed(9)
    model = tiny_model()
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=1e-3,
        weight_decay=0.01,
    )
    state, metrics = train_model(
        model,
        tiny_loader(examples=3, batch_size=2, shuffle=True, seed=9),
        optimizer,
        device=torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        max_steps=max_steps,
        gradient_accumulation_steps=accumulation,
        gradient_clip=0.25,
        warmup_steps=1 if max_steps > 1 else 0,
        peak_learning_rate=1e-3,
        seed=9,
        max_micro_steps=max_micro_steps,
        run_name="cpu-test",
        run_id="deterministic",
    )
    return state, metrics, model


@pytest.mark.usefixtures("development_training_environment")
def test_training_and_cycling() -> tuple[TrainingState, list[TrainingMetrics]]:
    set_deterministic_seed(9)
    baseline = tiny_model()
    before = [parameter.detach().clone() for parameter in baseline.parameters()]
    optimizer = create_adamw_optimizer(
        baseline,
        learning_rate=1e-3,
        weight_decay=0.01,
    )
    state, metrics = train_model(
        baseline,
        tiny_loader(examples=3, batch_size=5, shuffle=True, seed=9),
        optimizer,
        device=torch.device("cpu"),
        precision=resolve_precision("fp32", torch.device("cpu")),
        max_steps=2,
        gradient_accumulation_steps=2,
        gradient_clip=0.25,
        warmup_steps=1,
        peak_learning_rate=1e-3,
        seed=9,
        run_name="cpu-test",
        run_id="deterministic",
    )
    assert state.optimizer_step == 2
    assert state.micro_step == 4
    assert state.tokens_seen == 3 * 4 * 4
    assert state.completed_epochs == 4
    assert len(metrics) == 2
    assert all(math.isfinite(item.training_loss) for item in metrics)
    assert all(math.isfinite(item.gradient_norm) for item in metrics)
    assert all(item.gradient_norm >= 0 for item in metrics)
    assert all(item.tokens_processed == 24 for item in metrics)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, baseline.parameters(), strict=True)
    )
    assert optimizer.param_groups[0]["lr"] == metrics[-1].learning_rate

    incomplete_state, incomplete_metrics, incomplete_model = run_cpu_training(
        max_steps=2,
        accumulation=2,
        max_micro_steps=1,
    )
    assert incomplete_state.optimizer_step == 0
    assert incomplete_state.micro_step == 0
    assert incomplete_metrics == []
    assert all(parameter.grad is None for parameter in incomplete_model.parameters())

    loader = tiny_loader(examples=3, batch_size=1, shuffle=True, seed=10)
    first_cycle = CyclingDataIterator(loader, seed=10)
    first_order = [
        int(first_cycle.next()[0][0, 0]) for _ in range(6)
    ]
    second_loader = tiny_loader(examples=3, batch_size=1, shuffle=True, seed=10)
    second_cycle = CyclingDataIterator(second_loader, seed=10)
    second_order = [
        int(second_cycle.next()[0][0, 0]) for _ in range(6)
    ]
    assert first_order == second_order
    assert first_cycle.completed_epochs == 2
    empty_loader = DataLoader(EmptyDataset(), batch_size=1)
    assert_raises(ValueError, CyclingDataIterator, empty_loader, seed=1)
    return state, metrics


@pytest.mark.usefixtures("development_training_environment")
def test_validation() -> None:
    model = tiny_model()
    loader = tiny_loader(examples=3, batch_size=2)
    policy = resolve_precision("fp32", torch.device("cpu"))
    model.train()
    loss, tokens = evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        precision=policy,
        max_batches=2,
    )
    assert math.isfinite(loss)
    assert tokens == 12
    assert model.training
    model.eval()
    evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        precision=policy,
        max_batches=1,
    )
    assert not model.training
    assert_raises(
        ValueError,
        evaluate,
        model,
        DataLoader(EmptyDataset(), batch_size=1),
        device=torch.device("cpu"),
        precision=policy,
    )
    assert_raises(
        ValueError,
        evaluate,
        model,
        loader,
        device=torch.device("cpu"),
        precision=policy,
        max_batches=0,
    )
    nonfinite = NonFiniteModel()
    assert_raises(
        FloatingPointError,
        evaluate,
        nonfinite,
        loader,
        device=torch.device("cpu"),
        precision=policy,
    )
    optimizer = create_adamw_optimizer(
        nonfinite,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    assert_raises(
        FloatingPointError,
        train_model,
        nonfinite,
        loader,
        optimizer,
        device=torch.device("cpu"),
        precision=policy,
        max_steps=1,
        gradient_accumulation_steps=1,
        gradient_clip=1.0,
        warmup_steps=0,
        peak_learning_rate=1e-3,
    )

    interval_model = tiny_model()
    interval_optimizer = create_adamw_optimizer(
        interval_model,
        learning_rate=1e-3,
        weight_decay=0.0,
    )
    interval_state, interval_metrics = train_model(
        interval_model,
        loader,
        interval_optimizer,
        device=torch.device("cpu"),
        precision=policy,
        max_steps=2,
        gradient_accumulation_steps=1,
        gradient_clip=1.0,
        warmup_steps=0,
        peak_learning_rate=1e-3,
        validation_loader=loader,
        evaluation_interval=2,
        max_validation_batches=1,
    )
    assert interval_state.optimizer_step == 2
    assert interval_metrics[0].validation_loss is None
    assert interval_metrics[1].validation_loss is not None


def test_logger() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        logger = JsonlRunLogger(root, "run")
        logger.write_metadata({"run_name": "run", "value": 1})
        logger.write_metrics({"optimizer_step": 1})
        assert logger.metrics_path.read_text(encoding="utf-8").strip()
        logger.close()
        assert json.loads(logger.metadata_path.read_text(encoding="utf-8"))[
            "run_name"
        ] == "run"
        assert json.loads(logger.metrics_path.read_text(encoding="utf-8"))[
            "optimizer_step"
        ] == 1
        assert_raises(FileExistsError, JsonlRunLogger, root, "run")
        replacement = JsonlRunLogger(root, "run", overwrite=True)
        replacement.write_metadata({"run_name": "replacement"})
        replacement.write_metrics({"optimizer_step": 2})
        replacement.close()
        lines = replacement.metrics_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["optimizer_step"] == 2


def _write_cli_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    config_path = root / "config.yaml"
    config_path.write_text(
        """
model:
  vocab_size: 32
  context_length: 4
  num_layers: 1
  hidden_size: 16
  num_heads: 4
  intermediate_size: 32
  dropout: 0.0
  tie_embeddings: true
  position_embedding_type: learned
  rope_theta: 10000.0
training:
  micro_batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 0.001
  weight_decay: 0.01
  warmup_steps: 0
  max_steps: 5
  gradient_clip: 1.0
  precision: fp32
  checkpoint_interval: 5
  evaluation_interval: 1
  seed: 42
""".lstrip(),
        encoding="utf-8",
    )
    train_path = root / "train.bin"
    validation_path = root / "validation.bin"
    np.arange(25, dtype=np.uint16).tofile(train_path)
    np.arange(10, dtype=np.uint16).tofile(validation_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "context_length": 4,
                "tokenizer_actual_vocab_size": 28,
                "model_vocab_size": 32,
                "output_checksums": {
                    "train": file_sha256(train_path),
                    "validation": file_sha256(validation_path),
                },
            }
        ),
        encoding="utf-8",
    )
    return config_path, train_path, validation_path, manifest_path


@pytest.mark.usefixtures("development_training_environment")
def test_cli_cpu() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        config, train, validation, manifest = _write_cli_fixture(root)
        original_config = config.read_bytes()
        arguments = build_argument_parser().parse_args(
            [
                "--config",
                str(config),
                "--train-token-file",
                str(train),
                "--validation-token-file",
                str(validation),
                "--token-manifest",
                str(manifest),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--max-steps",
                "1",
                "--num-workers",
                "0",
                "--log-dir",
                str(root / "logs"),
                "--checkpoint-dir",
                str(root / "checkpoints"),
                "--run-name",
                "cli",
                "--no-validation",
            ]
        )
        assert run(arguments) == 0
        assert config.read_bytes() == original_config
        metrics_path = root / "logs" / "cli" / "train_metrics.jsonl"
        record = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert record["optimizer_step"] == 1
        assert record["validation_loss"] is None
        checkpoint_path = root / "checkpoints" / "cli" / "latest.pt"
        assert checkpoint_path.is_file()
        assert checkpoint_path.with_suffix(".pt.sha256").is_file()
        metadata_path = root / "logs" / "cli" / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["run_status"] == "completed"
        assert metadata["completed_optimizer_steps"] == 1
        assert metadata["completed_micro_steps"] == 2
        assert metadata["tokens_seen"] == 16
        assert metadata["resume_events"] == []
        assert metadata["last_stop_timestamp_utc"]

        resume_arguments = build_argument_parser().parse_args(
            [
                "--config",
                str(config),
                "--train-token-file",
                str(train),
                "--validation-token-file",
                str(validation),
                "--token-manifest",
                str(manifest),
                "--device",
                "cpu",
                "--precision",
                "fp32",
                "--max-steps",
                "1",
                "--num-workers",
                "0",
                "--log-dir",
                str(root / "logs"),
                "--checkpoint-dir",
                str(root / "checkpoints"),
                "--run-name",
                "cli",
                "--no-validation",
                "--resume",
                str(checkpoint_path),
            ]
        )
        assert run(resume_arguments) == 0
        resumed_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
        assert resumed_metadata["run_status"] == "completed"
        assert len(resumed_metadata["resume_events"]) == 1
        resume_event = resumed_metadata["resume_events"][0]
        assert resume_event["optimizer_step"] == 1
        assert resume_event["micro_step"] == 2
        assert resume_event["tokens_seen"] == 16
        assert resume_event["downtime_seconds"] is not None
        assert resume_event["downtime_seconds"] >= 0
        assert resume_event["downtime_basis"] == "last_stop"


def test_resume_downtime_falls_back_to_last_metric() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        logger = JsonlRunLogger(temporary_directory, "power-loss")
        logger.write_metadata(
            {
                "run_name": "power-loss",
                "run_id": "run-id",
                "resume_events": [],
            }
        )
        logger.write_metrics(
            {
                "optimizer_step": 3,
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
            }
        )
        _record_resume(
            logger,
            timestamp="2026-01-01T00:01:30+00:00",
            checkpoint=Path("checkpoints/power-loss/latest.pt"),
            state=TrainingState(
                optimizer_step=3,
                micro_step=6,
                tokens_seen=48,
            ),
        )
        metadata = json.loads(
            logger.metadata_path.read_text(encoding="utf-8")
        )
        event = metadata["resume_events"][0]
        assert event["downtime_seconds"] == 90.0
        assert event["downtime_basis"] == "last_metric"
        logger.close()


def test_cuda_precision_paths() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    for precision_name in ("fp16", "bf16"):
        if precision_name == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        set_deterministic_seed(11)
        model = tiny_model().to(device)
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        policy = resolve_precision(precision_name, device)
        scaler = create_grad_scaler(device, policy)
        state, metrics = train_model(
            model,
            tiny_loader(examples=2, batch_size=2),
            optimizer,
            device=device,
            precision=policy,
            max_steps=1,
            gradient_accumulation_steps=1,
            gradient_clip=1.0,
            warmup_steps=0,
            peak_learning_rate=1e-3,
            scaler=scaler,
            run_name=f"cuda-{precision_name}",
        )
        assert state.optimizer_step == 1
        assert math.isfinite(metrics[0].training_loss)
        assert (scaler is not None) == (precision_name == "fp16")


def main() -> None:
    test_state_and_metrics()
    test_seed_device_and_precision()
    test_optimizer_groups()
    test_scheduler()
    state, metrics = test_training_and_cycling()
    test_validation()
    test_logger()
    test_cli_cpu()
    test_resume_downtime_falls_back_to_last_metric()
    test_cuda_precision_paths()

    final = metrics[-1]
    print("Device used: cpu")
    print("Precision used: fp32")
    print(f"Optimizer steps: {state.optimizer_step}")
    print(f"Micro-steps: {state.micro_step}")
    print(f"Tokens processed: {state.tokens_seen}")
    print(f"Final loss: {final.training_loss:.6f}")
    print("All training-loop tests passed.")


if __name__ == "__main__":
    main()
