"""Run a bounded chat-format overfit diagnostic on a tiny model."""

from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[1]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_inference.generate import GenerationConfig, generate_sequences
from llm_architecture.model import LanguageModel, ModelConfig
from llm_tokenizer.sft import ChatMessage, IGNORE_LABEL, format_chat_prompt, serialize_conversation
from llm_tokenizer.tokenizer import END_TOKEN, load_tokenizer
from llm_training.training import (
    JsonlRunLogger,
    TrainingState,
    create_adamw_optimizer,
    resolve_device,
    resolve_precision,
    set_deterministic_seed,
    train_model,
)


DIAGNOSTIC_CONVERSATIONS = (
    ("What is 17 plus 25?", "17 plus 25 equals 42."),
    ("My name is Lindiwe. What is my name?", "Your name is Lindiwe."),
    ("Explain photosynthesis simply.", "Plants use sunlight, water, and air to make food."),
    ("Actually, my name is Amina. What is my name now?", "Your name is Amina."),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the diagnostic command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", choices=("development", "historical"),
                        help="Required training identity, checked before model setup")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("checkpoints/tokenizer-base-v1/tokenizer.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--run-name", default="chat-overfit-diagnostic-v1")
    return parser


def _records(tokenizer, context_length: int) -> tuple[list[list[int]], list[list[int]]]:
    """Serialize fixed conversations and apply the model's next-token shift."""

    input_records: list[list[int]] = []
    label_records: list[list[int]] = []
    for question, answer in DIAGNOSTIC_CONVERSATIONS:
        serialized = serialize_conversation(
            [ChatMessage("user", question), ChatMessage("assistant", answer)],
            tokenizer,
            maximum_tokens=context_length,
        )
        input_records.append(list(serialized.input_ids))
        label_records.append(list(serialized.labels[1:]) + [IGNORE_LABEL])
    width = max(len(record) for record in input_records)
    if width > context_length:
        raise ValueError("Diagnostic conversation exceeds context length.")
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer is missing <pad>.")
    padded_inputs = [record + [pad_id] * (width - len(record)) for record in input_records]
    padded_labels = [record + [IGNORE_LABEL] * (width - len(record)) for record in label_records]
    return padded_inputs, padded_labels


def run(arguments: argparse.Namespace) -> dict[str, object]:
    """Train the tiny diagnostic model and evaluate every fixed prompt."""

    if arguments.steps <= 0:
        raise ValueError("--steps must be positive.")
    tokenizer = load_tokenizer(arguments.tokenizer)
    end_id = tokenizer.token_to_id(END_TOKEN)
    if end_id is None:
        raise ValueError("Tokenizer is missing <|end|>.")
    device = resolve_device(arguments.device)
    precision = resolve_precision("fp32", device)
    set_deterministic_seed(42)
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        context_length=arguments.context_length,
        num_layers=2,
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        dropout=0.0,
        tie_embeddings=True,
        position_embedding_type="rotary",
        rope_theta=10000.0,
    )
    input_records, label_records = _records(tokenizer, config.context_length)
    inputs = torch.tensor(input_records, dtype=torch.long)
    labels = torch.tensor(label_records, dtype=torch.long)
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=1, shuffle=False)
    model = LanguageModel(config).to(device)
    optimizer = create_adamw_optimizer(model, learning_rate=0.003, weight_decay=0.0)
    run_dir = arguments.log_dir / arguments.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlRunLogger(arguments.log_dir, arguments.run_name, overwrite=True)
    state = TrainingState()
    logger.write_metadata({
        "run_name": arguments.run_name,
        "purpose": "Chat protocol, assistant-mask, and stop-token overfit diagnostic",
        "config": asdict(config),
        "device": str(device),
        "precision": precision.name,
        "conversation_count": len(DIAGNOSTIC_CONVERSATIONS),
        "tokenizer": str(arguments.tokenizer),
    })
    try:
        state, metrics = train_model(
            model,
            loader,
            optimizer,
            device=device,
            precision=precision,
            max_steps=arguments.steps,
            gradient_accumulation_steps=1,
            gradient_clip=1.0,
            warmup_steps=min(20, arguments.steps - 1),
            peak_learning_rate=0.003,
            minimum_learning_rate=0.0003,
            seed=42,
            state=state,
            logger=logger,
            run_name=arguments.run_name,
            evaluation_interval=arguments.steps,
            validation_loader=loader,
            max_validation_batches=len(DIAGNOSTIC_CONVERSATIONS),
            progress=True,
        )
    finally:
        logger.close()

    model.eval()
    outputs: list[dict[str, object]] = []
    for question, expected in DIAGNOSTIC_CONVERSATIONS:
        prompt = format_chat_prompt([ChatMessage("user", question)], tokenizer)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        with torch.no_grad():
            generated = generate_sequences(
                model,
                torch.tensor([prompt_ids], dtype=torch.long, device=device),
                eos_token_id=tokenizer.token_to_id("<eos>"),
                pad_token_id=tokenizer.token_to_id("<pad>"),
                stop_sequences=[[end_id]],
                config=GenerationConfig(max_new_tokens=32, do_sample=False),
            ).sequences[0]
        text = tokenizer.decode(list(generated.visible_token_ids), skip_special_tokens=True).strip()
        outputs.append({
            "question": question,
            "expected": expected,
            "actual": text,
            "finish_reason": generated.finish_reason,
            "termination_cause": generated.termination_cause,
            "generated_tokens": generated.generated_token_count,
        })
    report = {
        "run_name": arguments.run_name,
        "device": str(device),
        "steps": state.optimizer_step,
        "initial_loss": metrics[0].training_loss,
        "final_loss": metrics[-1].training_loss,
        "outputs": outputs,
        "report_path": str(run_dir / "chat_overfit_report.json"),
    }
    (run_dir / "chat_overfit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    if "--help" not in sys.argv:
        from training_environment_guard import require_environment
        require_environment("new")
    arguments = build_parser().parse_args()
    from llm_training.viewer import attached_viewer
    with attached_viewer(metrics_path=arguments.log_dir / arguments.run_name / "train_metrics.jsonl",
                         checkpoint_path=arguments.log_dir / arguments.run_name / "no-checkpoint",
                         total_steps=arguments.steps, title="LLM chat overfit diagnostic"):
        run(arguments)
