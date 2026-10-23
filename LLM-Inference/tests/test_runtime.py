"""Offline native inference including plain inference without memory extras."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_architecture.checkpointing import file_sha256
from llm_architecture.model import LanguageModel, ModelConfig
import llm_inference.native_chat as native


class Tokenizer:
    def get_vocab_size(self):
        return 32
    def token_to_id(self, text):
        return {"<pad>": 0, "<eos>": 2, "<|system|>": 4, "<|user|>": 5, "<|assistant|>": 6, "<|end|>": 7}.get(text)
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[8 + ord(c) % 24 for c in text])
    def decode(self, ids, skip_special_tokens=True):
        return "response"


def test_native_export_dialect_and_transactional_history(tmp_path, monkeypatch):
    config = ModelConfig(vocab_size=32, context_length=64, num_layers=1, hidden_size=16, num_heads=4, intermediate_size=32)
    checkpoint = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("fixture")
    torch.save(dict(model_state_dict=LanguageModel(config).state_dict(), config={"model": asdict(config)}, tokenizer_sha256=file_sha256(tokenizer)), checkpoint)
    checkpoint.with_suffix(".pt.sha256").write_text(file_sha256(checkpoint) + "  model.pt\n")
    monkeypatch.setattr(native, "load_tokenizer", lambda _: Tokenizer())
    engine = native.NativeChatEngine(checkpoint=checkpoint, tokenizer_path=tokenizer, device="cpu", maximum_new_tokens=8)
    assert engine.memory.mode == "off" and engine.memory.store is None
    first = engine.reply("hello")[1].visible_token_ids
    engine.reset()
    assert engine.reply("hello")[1].visible_token_ids == first
    before = engine.messages.copy()
    def fail(*args, **kwargs):
        raise RuntimeError("fixture generation failure")
    monkeypatch.setattr(native, "generate_sequences", fail)
    with pytest.raises(RuntimeError):
        engine.reply("next")
    assert engine.messages == before
    engine.close()
