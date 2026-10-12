"""Deliberately overfit a short sequence and verify checkpoint resume."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys
import uuid

import torch
from torch.utils.data import DataLoader, TensorDataset


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_training.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
    load_checkpoint,
)
from llm_training.config import load_config
from llm_inference.generate import GenerationConfig, generate_token_ids
from llm_architecture.model import LanguageModel, count_parameters
from llm_data.token_data import file_sha256
from llm_tokenizer.tokenizer import EOS_TOKEN, load_tokenizer
from llm_training.training import (
    JsonlRunLogger,
    TrainingState,
    create_adamw_optimizer,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("development", "historical"),
                        help="Required training identity, checked before model setup")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny_overfit.yaml"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/tokenizer-smoke/tokenizer.json"),
    )
    parser.add_argument(
        "--text",
        type=Path,
        default=Path("tests/fixtures/data/tiny_overfit.txt"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-name", default="tiny-overfit-phase10")
    parser.add_argument("--target-loss", type=float, default=0.15)
    parser.add_argument("--target-accuracy", type=float, default=0.99)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _single_example(
    text: str,
    *,
    tokenizer_path: Path,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    tokenizer = load_tokenizer(tokenizer_path)
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_token_id != 2:
        raise ValueError("Tokenizer must use EOS token ID 2.")
    content = tokenizer.encode(text, add_special_tokens=False).ids
    if not content:
        raise ValueError("Tiny overfit text produced no tokens.")
    stream: list[int] = []
    while len(stream) < context_length + 1:
        stream.extend(content)
        stream.append(eos_token_id)
    window = stream[: context_length + 1]
    input_ids = torch.tensor([window[:-1]], dtype=torch.long)
    labels = torch.tensor([window[1:]], dtype=torch.long)
    return input_ids, labels, content


def run(arguments: argparse.Namespace) -> dict[str, object]:
    config = load_config(arguments.config)
    device = resolve_device(arguments.device)
    precision = resolve_precision(arguments.precision, device)
    if arguments.target_loss <= 0:
        raise ValueError("--target-loss must be positive.")
    if not 0 < arguments.target_accuracy <= 1:
        raise ValueError("--target-accuracy must be in (0, 1].")
    run_checkpoint_dir = arguments.checkpoint_dir / arguments.run_name
    run_log_dir = arguments.log_dir / arguments.run_name
    if arguments.overwrite:
        for path in (run_checkpoint_dir, run_log_dir):
            if path.exists():
                shutil.rmtree(path)
    elif run_checkpoint_dir.exists() or run_log_dir.exists():
        raise FileExistsError(
            "Tiny overfit outputs already exist; use --overwrite to replace them."
        )

    text = arguments.text.read_text(encoding="utf-8").strip()
    input_ids, labels, content_ids = _single_example(
        text,
        tokenizer_path=arguments.tokenizer,
        context_length=config.model.context_length,
    )
    loader = DataLoader(
        TensorDataset(input_ids, labels),
        batch_size=1,
        shuffle=False,
    )
    set_deterministic_seed(config.training.seed)
    model = LanguageModel(config.model).to(device)
    parameter_count = count_parameters(model)
    if not 10_000_000 <= parameter_count <= 30_000_000:
        raise ValueError(
            f"Tiny overfit model must contain 10M–30M parameters; "
            f"got {parameter_count:,}."
        )
    optimizer = create_adamw_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = SchedulerState(
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.max_steps,
        peak_learning_rate=config.training.learning_rate,
        minimum_learning_rate=config.training.learning_rate * 0.1,
    )
    state = TrainingState()
    run_id = str(uuid.uuid4())
    manager = CheckpointManager(arguments.checkpoint_dir, arguments.run_name)
    logger = JsonlRunLogger(arguments.log_dir, arguments.run_name)

    def save_boundary(
        current_state: TrainingState,
        _metrics: object,
        *,
        force: bool = False,
    ) -> None:
        if not force and (
            current_state.optimizer_step % config.training.checkpoint_interval
            != 0
        ):
            return
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=None,
            state=current_state,
            scheduler=scheduler,
            config=config,
            run_name=arguments.run_name,
            run_id=run_id,
            tokenizer_reference=str(arguments.tokenizer),
            tokenizer_sha256=file_sha256(arguments.tokenizer),
        )
        manager.save(payload, milestone=not force)

    logger.write_metadata(
        {
            "run_name": arguments.run_name,
            "run_id": run_id,
            "purpose": "Deliberate tiny-sequence overfit validation",
            "config": {"model": asdict(config.model), "training": asdict(config.training)},
            "text_path": str(arguments.text),
            "tokenizer_path": str(arguments.tokenizer),
            "device": str(device),
            "precision": precision.name,
            "parameter_count": parameter_count,
        }
    )
    split_step = config.training.max_steps // 2
    try:
        state, first_metrics = train_model(
            model,
            loader,
            optimizer,
            device=device,
            precision=precision,
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=1,
            gradient_clip=config.training.gradient_clip,
            warmup_steps=scheduler.warmup_steps,
            peak_learning_rate=scheduler.peak_learning_rate,
            minimum_learning_rate=scheduler.minimum_learning_rate,
            seed=config.training.seed,
            state=state,
            run_name=arguments.run_name,
            run_id=run_id,
            logger=logger,
            on_optimizer_step=save_boundary,
            max_micro_steps=split_step,
            progress=False,
        )
        save_boundary(state, object(), force=True)
        print(
            f"Interrupted boundary saved at optimizer step "
            f"{state.optimizer_step}."
        )

        resumed_model = LanguageModel(config.model).to(device)
        resumed_optimizer = create_adamw_optimizer(
            resumed_model,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        loaded = load_checkpoint(
            manager.latest_path,
            model=resumed_model,
            optimizer=resumed_optimizer,
            scaler=None,
            expected_config=config,
            map_location=device,
        )
        model = resumed_model
        optimizer = resumed_optimizer
        state, resumed_metrics = train_model(
            model,
            loader,
            optimizer,
            device=device,
            precision=precision,
            max_steps=config.training.max_steps,
            gradient_accumulation_steps=1,
            gradient_clip=config.training.gradient_clip,
            warmup_steps=scheduler.warmup_steps,
            peak_learning_rate=scheduler.peak_learning_rate,
            minimum_learning_rate=scheduler.minimum_learning_rate,
            seed=config.training.seed,
            state=loaded.state,
            run_name=arguments.run_name,
            run_id=run_id,
            logger=logger,
            on_optimizer_step=save_boundary,
            progress=False,
        )
        save_boundary(state, object(), force=True)
    finally:
        logger.close()

    all_metrics = [*first_metrics, *resumed_metrics]
    final_loss = all_metrics[-1].training_loss
    if final_loss > arguments.target_loss:
        raise RuntimeError(
            f"Tiny overfit target was not reached: "
            f"{final_loss:.6f} > {arguments.target_loss:.6f}."
        )

    model.eval()
    with torch.no_grad():
        logits, _loss = model(input_ids.to(device), labels.to(device))
    predictions = logits.argmax(dim=-1)
    target_accuracy = float(
        (predictions == labels.to(device)).float().mean().item()
    )
    if target_accuracy < arguments.target_accuracy:
        raise RuntimeError(
            f"Tiny overfit accuracy target was not reached: "
            f"{target_accuracy:.6f} < {arguments.target_accuracy:.6f}."
        )

    prompt_ids = content_ids[: min(8, len(content_ids))]
    generated_ids = generate_token_ids(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        eos_token_id=2,
        config=GenerationConfig(
            max_new_tokens=24,
            do_sample=False,
            seed=config.training.seed,
        ),
    )[0].tolist()
    tokenizer = load_tokenizer(arguments.tokenizer)
    report = {
        "run_name": arguments.run_name,
        "run_id": run_id,
        "parameter_count": parameter_count,
        "context_length": config.model.context_length,
        "optimizer_steps": state.optimizer_step,
        "resume_step": split_step,
        "tokens_seen": state.tokens_seen,
        "fixture_token_count": labels.numel(),
        "initial_loss": all_metrics[0].training_loss,
        "final_loss": final_loss,
        "target_loss": arguments.target_loss,
        "target_token_accuracy": target_accuracy,
        "required_target_token_accuracy": arguments.target_accuracy,
        "generated_text": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ),
        "checkpoint": str(manager.latest_path),
    }
    report_path = run_log_dir / "overfit_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    if "--help" not in sys.argv:
        from training_environment_guard import require_environment
        require_environment("new")
    arguments = build_argument_parser().parse_args()
    from llm_training.viewer import attached_viewer
    config = load_config(arguments.config)
    with attached_viewer(metrics_path=arguments.log_dir / arguments.run_name / "train_metrics.jsonl",
                         checkpoint_path=arguments.checkpoint_dir / arguments.run_name / "latest.pt",
                         total_steps=config.training.max_steps, title="LLM tiny overfit diagnostic"):
        run(arguments)


if __name__ == "__main__":
    main()
