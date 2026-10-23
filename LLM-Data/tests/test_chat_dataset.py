"""Lazy indexing preserves token masks and deterministic source-wide ordering."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from llm_data.chat_dataset import IndexedChatDataset, ConversationSampler, collate_chat, verify_split_isolation
from llm_data.data.general_chat import normalized
from llm_tokenizer.sft import ChatMessage, serialize_conversation


class Tokenizer:
    def token_to_id(self, value):
        return {'<pad>': 0, '<|user|>': 5, '<|assistant|>': 6, '<|end|>': 7}[value]
    def encode(self, text, add_special_tokens=False):
        return type('Encoded', (), {'ids': [8 + int(word) for word in text.split()]})()


def write_records(path: Path):
    rows = []
    for i in range(10):
        messages = [dict(role='user', content=str(i)), dict(role='assistant', content=f'{i} {i+1}')]
        record = serialize_conversation([ChatMessage(m['role'], m['content']) for m in messages], Tokenizer())
        canonical = [(m['role'], normalized(m['content'])) for m in messages]
        rows.append(dict(conversation_id=str(i), root_id=str(i), duplicate_group=str(i),
            content_sha256=hashlib.sha256(json.dumps(canonical).encode()).hexdigest(), source='a' if i % 2 else 'b',
            messages=messages, tokens=len(record.input_ids), assistant_tokens=record.supervised_token_count))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_lazy_shift_padding_and_no_replacement(tmp_path):
    path = tmp_path/'records.jsonl'; write_records(path)
    data = IndexedChatDataset(path, Tokenizer(), context=16, verify_tokens=True)
    first = data[0]
    assert first['labels'].tolist() == [-100, -100, -100, 8, 9, 7, -100]
    batch = collate_chat([first, data[1]], fixed_length=16)
    assert batch['attention_mask'].sum() == 14
    assert (batch['labels'][:, 7:] == -100).all()
    a = list(ConversationSampler(data, seed=42, bucket_size=2))
    b = list(ConversationSampler(data, seed=42, bucket_size=2))
    assert a == b and sorted(a) == list(range(10))
    c = ConversationSampler(data, seed=42, bucket_size=2); c.set_epoch(1)
    assert list(c) != a
    assert len(data._cache) <= 10


def test_index_count_drift_and_split_leakage(tmp_path):
    path = tmp_path/'records.jsonl'; write_records(path)
    data = IndexedChatDataset(path, Tokenizer(), context=16)
    with pytest.raises(ValueError, match='leakage'):
        verify_split_isolation({'train': data, 'test': data})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['assistant_tokens'] = 1
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='count mismatch'):
        IndexedChatDataset(path, Tokenizer(), context=16, verify_tokens=True)


def test_reconstructed_sampler_continues_without_repeating_prefix(tmp_path):
    path = tmp_path / 'records.jsonl'; write_records(path)
    data = IndexedChatDataset(path, Tokenizer(), context=16)
    sampler = ConversationSampler(data, seed=42, bucket_size=2)
    sampler.set_epoch(3)
    order = list(sampler)
    restored = ConversationSampler(data, seed=42, bucket_size=2)
    restored.set_epoch(3)
    remaining = list(restored)[4:]
    assert remaining == order[4:] and not set(order[:4]) & set(remaining)
    assert sorted(order[:4] + remaining) == list(range(len(data)))
