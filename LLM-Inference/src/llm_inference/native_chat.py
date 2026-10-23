"""Persistent native conversational inference for Codexa checkpoints."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

import torch

from llm_architecture.checkpointing import CHECKPOINT_FORMAT_VERSION, verify_checkpoint_checksum
from llm_inference.generate import (
    GenerationConfig,
    GeneratedSequence,
    compile_generation_model,
    generate_sequences,
)
from llm_architecture.model import LanguageModel, ModelConfig
from llm_tokenizer.sft import ChatMessage
from llm_inference.context import build_context
from typing import TYPE_CHECKING
from .memory_adapter import create_memory, DisabledMemory
if TYPE_CHECKING:
    from llm_memory.service import ConversationMemory
from llm_architecture.checkpointing import file_sha256, read_native_checkpoint
from llm_tokenizer.tokenizer import END_TOKEN, EOS_TOKEN, load_tokenizer
from llm_architecture.runtime import resolve_device


class NativeChatEngine:
    """Load one native checkpoint and retain validated multi-turn history."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        tokenizer_path: Path,
        device: str = "auto",
        maximum_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system_prompt: str | None = None,
        torch_compile: bool = False,
        memory: ConversationMemory | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
        do_sample: bool = True,
        seed: int = 42,
        expected_checkpoint_sha256: str | None = None,
    ) -> None:
        payload = (read_native_checkpoint(checkpoint, model_only=True)
                   if expected_checkpoint_sha256 is None else
                   read_native_checkpoint(checkpoint, model_only=True, expected_sha256=expected_checkpoint_sha256))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint payload must be an object.")
        config = payload.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            raise ValueError("Checkpoint model configuration is missing.")

        self.device = resolve_device(device)
        self.model = LanguageModel(ModelConfig(**config["model"])).to(self.device)
        if payload.get("format_version") not in (None, CHECKPOINT_FORMAT_VERSION):
            raise ValueError("Unsupported checkpoint format.")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint model state is missing.")
        # Reuse the already verified CPU mapping; do not load optimizer tensors onto CUDA.
        self.model.load_state_dict(state, strict=True)

        if torch_compile:
            if self.device.type != "cuda":
                raise ValueError("torch_compile requires CUDA.")
            self.model = compile_generation_model(self.model)

        self.tokenizer = load_tokenizer(tokenizer_path)
        if self.tokenizer.get_vocab_size() != self.model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match checkpoint.")
        fingerprint = payload.get("tokenizer_sha256")
        if fingerprint is not None and file_sha256(tokenizer_path) != fingerprint:
            raise ValueError("Tokenizer fingerprint does not match checkpoint.")
        self.memory = memory or create_memory()
        self.eos_token_id = self._required_token_id(EOS_TOKEN)
        self.end_token_id = self._required_token_id(END_TOKEN)
        self.pad_token_id = self._required_token_id("<pad>")
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty when supplied.")
        self.system_prompt = system_prompt
        self.messages: list[ChatMessage] = []
        self.generation_config = GenerationConfig(
            max_new_tokens=maximum_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            seed=seed,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        self.reset()

    def _required_token_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Tokenizer is missing required token {token!r}.")
        return token_id

    def reset(self) -> None:
        """Clear conversation history while retaining the loaded model."""

        if hasattr(self, "memory"):
            self.memory.new_conversation()
        self.messages = (
            []
            if self.system_prompt is None
            else [ChatMessage("system", self.system_prompt)]
        )

    def memory_command(self, operation: str, value=None) -> dict:
        """Apply explicit memory operations; clearing also clears live context."""
        if isinstance(self.memory, DisabledMemory) and operation in ("on", "ephemeral", "persistent"):
            previous = self.memory
            self.memory = create_memory(mode="ephemeral" if operation == "on" else operation, **previous.settings)
            self.memory.conversation_id = previous.conversation_id
            self.memory.turn = previous.turn
        current = self.memory.conversation_id
        result = self.memory.command(operation, value)
        if operation == "clear" or (operation == "delete" and value == current):
            self.reset()
            result = self.memory.command("stats")
        return result

    def close(self) -> None:
        """Release optional memory resources."""
        self.memory.close()

    def reply(self, user_text: str, *, token_callback: Callable[[int], None] | None = None,
              context_callback: Callable[[object], None] | None = None,
              preserve_whitespace: bool = False) -> tuple[str, GeneratedSequence]:
        """Generate one response and append the completed turn to history."""

        clean_text = user_text.strip()
        if not clean_text:
            raise ValueError("User message cannot be empty.")
        baseline = build_context(self.messages, clean_text, [], self.tokenizer,
                                 self.model.config.context_length,
                                 self.generation_config.max_new_tokens)
        completed = (len(self.messages) - int(bool(self.system_prompt))) // 2
        exclude = {(self.memory.conversation_id, turn, role)
                   for turn in range(max(0, completed - baseline.retained_turns), completed)
                   for role in ("user", "assistant")}
        hits = self.memory.retrieve(clean_text, exclude)
        context = build_context(self.messages, clean_text, hits, self.tokenizer,
                                self.model.config.context_length,
                                self.generation_config.max_new_tokens)
        self.memory.status.update(references=context.references, memory_tokens=context.memory_tokens)
        prompt_ids = context.ids
        if context_callback is not None:
            context_callback(context)
        try:
            generated = generate_sequences(
                self.model,
                torch.tensor([prompt_ids], dtype=torch.long, device=self.device),
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                stop_sequences=[[self.end_token_id]],
                config=self.generation_config,
                token_callback=token_callback,
            ).sequences[0]
            response = self.tokenizer.decode(
                list(generated.visible_token_ids),
                skip_special_tokens=True,
            )
            if not preserve_whitespace:
                response = response.strip()
            if not response and not preserve_whitespace:
                raise ValueError("Model generated an empty response.")
            self.messages.extend([ChatMessage("user", clean_text), ChatMessage("assistant", response)])
            self.memory.remember(clean_text, response)
            return response, generated
        except BaseException:
            raise
