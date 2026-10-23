"""Interact with a Codexa conversational checkpoint through native inference."""

from __future__ import annotations
from llm_inference.cli.paths import asset_path, generated_path

import argparse
from pathlib import Path
import sys


from llm_inference.native_chat import NativeChatEngine
from llm_inference.memory_adapter import create_memory


def build_parser() -> argparse.ArgumentParser:
    """Build the native chat command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int)
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the model forward for cached CUDA decoding.",
    )
    parser.add_argument(
        "--system",
        help="Optional system message. Omit it to match user-first SFT records.",
    )
    parser.add_argument("--memory", choices=("off", "ephemeral", "persistent"), default="off")
    parser.add_argument("--memory-path", type=Path, default=generated_path('data/memory/codexa.sqlite3'))
    parser.add_argument("--memory-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--memory-threshold", type=float, default=0.68)
    parser.add_argument("--user-id", default="local")
    return parser


def main() -> None:
    """Run an interactive native Codexa chat session."""

    arguments = build_parser().parse_args()
    engine = NativeChatEngine(
        checkpoint=arguments.checkpoint,
        tokenizer_path=arguments.tokenizer,
        device=arguments.device,
        maximum_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        seed=arguments.seed, do_sample=not arguments.greedy,
        repetition_penalty=arguments.repetition_penalty,
        no_repeat_ngram_size=arguments.no_repeat_ngram_size,
        system_prompt=arguments.system,
        memory=create_memory(user_id=arguments.user_id, mode=arguments.memory,
                                  path=arguments.memory_path, device=arguments.memory_device,
                                  threshold=arguments.memory_threshold),
        torch_compile=arguments.torch_compile,
    )
    print("Native Codexa chat. Type /exit to quit or /clear to reset history.")
    print(f"Device: {engine.device}\n")
    print("Memory is opt-in. /memory on|persistent|off|list|refs|stats|clear; /memory sources ID ...")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/help":
            print("/new, /clear, /memory on|ephemeral|persistent|off|clear|list|refs|stats|rebuild, /memory sources ID ..., /memory delete ID")
            continue
        if user_text in ("/clear", "/new"):
            engine.reset()
            print("Conversation cleared.\n")
            continue

        try:
            if user_text == "/memory":
                user_text = "/memory stats"
            if user_text.startswith("/memory "):
                parts = user_text.split()
                value = parts[2:] if parts[1] == "sources" else (parts[2] if len(parts) > 2 else None)
                print(engine.memory_command(parts[1], value))
                continue
            response, _generated = engine.reply(user_text)
            print(f"Codexa: {response}\n")
            if engine.memory.status.get("error"):
                print(f"Memory: {engine.memory.status['error']}\n")
        except Exception as error:
            print(f"Error: {error}\n")
    engine.close()


if __name__ == "__main__":
    main()
