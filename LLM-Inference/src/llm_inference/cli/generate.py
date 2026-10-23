"""Generate a base-model text continuation from a trusted checkpoint."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import torch



from llm_architecture.checkpointing import load_model_checkpoint, verify_checkpoint_checksum, checkpoint_model_config, read_native_checkpoint
_checkpoint_model_config = checkpoint_model_config
from llm_inference.generate import (
    GenerationConfig,
    compile_generation_model,
    generate_sequences,
)
from llm_architecture.model import LanguageModel, ModelConfig
from llm_architecture.checkpointing import file_sha256
from llm_tokenizer.tokenizer import BOS_TOKEN, END_TOKEN, EOS_TOKEN, load_tokenizer
from llm_architecture.runtime import resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the base-generation command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model forward for cached CUDA decoding.",
    )
    parser.add_argument(
        "--chat-prompt",
        action="store_true",
        help="Treat the prompt as chat-v1 text and stop at <|end|>.",
    )
    parser.add_argument("--output", type=Path)
    return parser




def _atomic_json_write(path: Path, value: object) -> None:
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
    """Generate and optionally save one continuation."""

    device = resolve_device(arguments.device)
    model = LanguageModel(_checkpoint_model_config(arguments.checkpoint)).to(device)
    try:
        checkpoint = load_model_checkpoint(
            arguments.checkpoint,
            model=model,
            map_location=device,
        )
    except ValueError as error:
        # Early conversational-SFT pilot checkpoints predate the full
        # resumable-checkpoint envelope but still contain a complete model
        # state and config. Allow deterministic inference without weakening
        # checksum verification performed above.
        if "Unsupported checkpoint format" not in str(error):
            raise
        payload = read_native_checkpoint(arguments.checkpoint, map_location=device)
        if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
            raise
        model.load_state_dict(payload["model_state_dict"], strict=True)
        checkpoint = type("PilotCheckpoint", (), {
            "tokenizer_sha256": None,
            "run_id": None,
            "training_state": type("PilotState", (), {
                "optimizer_step": int(payload.get("optimizer_step", 0)),
            })(),
        })()
    tokenizer = load_tokenizer(arguments.tokenizer)
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    if (
        checkpoint.tokenizer_sha256 is not None
        and checkpoint.tokenizer_sha256 != tokenizer_checksum
    ):
        raise ValueError("Tokenizer checksum does not match the checkpoint.")
    if arguments.torch_compile:
        if device.type != "cuda":
            raise ValueError("--torch-compile requires CUDA.")
        model = compile_generation_model(model)

    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    if eos_token_id != 2 or bos_token_id != 1:
        raise ValueError("Tokenizer must use <bos>=1 and <eos>=2.")
    prompt_ids = tokenizer.encode(
        arguments.prompt,
        add_special_tokens=False,
    ).ids or [bos_token_id]

    generation_config = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        do_sample=not arguments.greedy,
        seed=arguments.seed,
    )
    generated = generate_sequences(
        model,
        torch.tensor([prompt_ids], dtype=torch.long, device=device),
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.token_to_id("<pad>"),
        stop_sequences=(
            [[tokenizer.token_to_id(END_TOKEN)]]
            if arguments.chat_prompt and tokenizer.token_to_id(END_TOKEN) is not None
            else ()
        ),
        config=generation_config,
    ).sequences[0]
    continuation_ids = list(generated.visible_token_ids)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_run_id": checkpoint.run_id,
        "checkpoint_optimizer_step": checkpoint.training_state.optimizer_step,
        "tokenizer": str(arguments.tokenizer),
        "tokenizer_sha256": tokenizer_checksum,
        "device": str(device),
        "prompt": arguments.prompt,
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": continuation_ids,
        "generated_token_count": generated.generated_token_count,
        "finish_reason": generated.finish_reason,
        "termination_cause": generated.termination_cause,
        "terminating_token_id": generated.terminating_token_id,
        "text": tokenizer.decode(continuation_ids, skip_special_tokens=True),
        "generation_config": asdict(generation_config),
    }
    if arguments.output is not None:
        _atomic_json_write(arguments.output, output)
    print(output["text"])
    return output


def main() -> None:
    run(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
