"""Native chat integration without local training artifacts or external encoders."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

import src.native_chat as native
from src.checkpointing import file_sha256
from src.memory.service import ConversationMemory
from src.model import LanguageModel, ModelConfig


class Tokenizer:
    def get_vocab_size(self):
        return 32
    def token_to_id(self, text):
        return {'<eos>': 2, '<pad>': 0, '<|system|>': 4, '<|user|>': 5, '<|assistant|>': 6, '<|end|>': 7}.get(text)
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[8 + ord(c) % 24 for c in text])
    def decode(self, ids, skip_special_tokens=True):
        return 'response'


def fixture(root, monkeypatch, *, fingerprint=True):
    config = ModelConfig(vocab_size=32, context_length=64, num_layers=1, hidden_size=16, num_heads=4, intermediate_size=32)
    checkpoint = root / 'tiny.pt'
    tokenizer_path = root / 'tokenizer.json'
    tokenizer_path.write_text('mock tokenizer')
    payload = {'model_state_dict': LanguageModel(config).state_dict(), 'config': {'model': asdict(config)}}
    if fingerprint:
        payload['tokenizer_sha256'] = file_sha256(tokenizer_path)
    torch.save(payload, checkpoint)
    checkpoint.with_suffix('.pt.sha256').write_text(file_sha256(checkpoint) + "  " + checkpoint.name + "\n")
    monkeypatch.setattr(native, 'load_tokenizer', lambda _: Tokenizer())
    return checkpoint, tokenizer_path


def test_native_history_no_system_and_rollback(monkeypatch):
    with TemporaryDirectory() as directory:
        checkpoint, tokenizer_path = fixture(Path(directory), monkeypatch)
        calls = []
        generated = SimpleNamespace(visible_token_ids=[8], generated_token_count=1)
        def generate(model, ids, **kwargs):
            calls.append(ids.clone())
            return SimpleNamespace(sequences=[generated])
        monkeypatch.setattr(native, 'generate_sequences', generate)
        engine = native.NativeChatEngine(checkpoint=checkpoint, tokenizer_path=tokenizer_path, device='cpu', maximum_new_tokens=8)
        for i in range(5):
            assert engine.reply(f'question {i}')[0] == 'response'
        assert len(engine.messages) == 10  # Prompt truncation never removes archived turns.
        assert calls[-1].shape[1] <= 56
        before = engine.messages.copy()
        def fail(*args, **kwargs):
            raise RuntimeError('generation failed')
        monkeypatch.setattr(native, 'generate_sequences', fail)
        with pytest.raises(RuntimeError):
            engine.reply('another question')
        assert engine.messages == before
        old = engine.memory.conversation_id
        engine.reset()
        assert not engine.messages and engine.memory.conversation_id != old
        engine.close()


def test_tokenizer_fingerprint_guard(monkeypatch):
    with TemporaryDirectory() as directory:
        checkpoint, tokenizer_path = fixture(Path(directory), monkeypatch)
        tokenizer_path.write_text('changed')
        with pytest.raises(ValueError, match='fingerprint'):
            native.NativeChatEngine(checkpoint=checkpoint, tokenizer_path=tokenizer_path, device='cpu')


def test_actual_tiny_generation_is_deterministic(monkeypatch):
    with TemporaryDirectory() as directory:
        checkpoint, tokenizer_path = fixture(Path(directory), monkeypatch)
        engine = native.NativeChatEngine(checkpoint=checkpoint, tokenizer_path=tokenizer_path, device='cpu', maximum_new_tokens=8)
        first = engine.reply('hello')[1]
        engine.reset()
        second = engine.reply('hello')[1]
        assert first.visible_token_ids == second.visible_token_ids
        engine.close()


def test_cli_and_bridge_memory_defaults():
    from scripts.inference.chat_native import build_parser
    from scripts.inference.native_chat_bridge import build_parser as bridge_parser
    for parser in (build_parser(), bridge_parser()):
        args = parser.parse_args(['--checkpoint', 'unused.pt', '--tokenizer', 'unused.json'])
        assert args.memory == 'off' and args.memory_device == 'cpu'


def test_clear_memory_also_clears_live_context(monkeypatch):
    from tests.inference.test_memory import Client
    with TemporaryDirectory() as directory:
        checkpoint, tokenizer_path = fixture(Path(directory), monkeypatch)
        engine = native.NativeChatEngine(checkpoint=checkpoint, tokenizer_path=tokenizer_path,
            device='cpu', maximum_new_tokens=8, memory=ConversationMemory(mode='ephemeral', client=Client()))
        monkeypatch.setattr(native, 'generate_sequences', lambda *args, **kwargs:
                            SimpleNamespace(sequences=[SimpleNamespace(visible_token_ids=[8])]))
        engine.reply('cat')
        old = engine.memory.conversation_id
        assert engine.memory.store.conversations('local') == [old]
        result = engine.memory_command('clear')
        assert not engine.messages and result['conversation_id'] != old
        assert not engine.memory.store.conversations('local')
        engine.close()
