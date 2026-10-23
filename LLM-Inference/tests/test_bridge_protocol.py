"""Small, offline protocol fixtures: native serializer and byte-safe streaming."""
from io import StringIO
import json
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, trainers, decoders, pre_tokenizers

from llm_inference.bridge_protocol import PROTOCOL, parse_chat, serve
from llm_inference.context import build_context, prompt_ids
from llm_inference.generate import GeneratedSequence, GenerationConfig, generate_sequences
from llm_tokenizer.sft import ChatMessage


@pytest.fixture
def tokenizer():
    value = Tokenizer(models.BPE(unk_token='<unk>'))
    value.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    value.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=280, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=['<pad>', '<bos>', '<eos>', '<unk>', '<|system|>', '<|user|>', '<|assistant|>', '<|end|>'])
    value.train_from_iterator(['Hello é 🌍.', 'A short conversation.'], trainer=trainer)
    return value


def request(messages, **generation):
    return dict(protocol=PROTOCOL, type='chat', id='request-1', messages=messages, generation=generation)


def test_role_serialization_and_context_accounting(tokenizer):
    rows = [dict(role='system', content='Be concise.'), dict(role='user', content='Hello'),
            dict(role='assistant', content='Hi'), dict(role='user', content='Next')]
    messages, config = parse_chat(request(rows, max_new_tokens=8), tokenizer, 64)
    expected = prompt_ids(messages, tokenizer)
    result = build_context(messages[:-1], messages[-1].content, [], tokenizer, 64, 8)
    assert result.ids == expected and result.retained_turns == 1
    assert len(result.ids) + config.max_new_tokens <= 64


@pytest.mark.parametrize('messages', [[], [dict(role='assistant', content='wrong')],
    [dict(role='user', content='one'), dict(role='user', content='two')],
    [dict(role='system', content=''), dict(role='user', content='x')]])
def test_bad_messages_rejected(tokenizer, messages):
    with pytest.raises(ValueError):
        parse_chat(request(messages), tokenizer, 2048)


@pytest.mark.parametrize('settings', [dict(max_new_tokens=2048), dict(seed=-1),
    dict(top_k=50000), dict(temperature=float('nan')), dict(do_sample='false'), dict(tools=True)])
def test_invalid_controls_rejected(tokenizer, settings):
    with pytest.raises((TypeError, ValueError)):
        parse_chat(request([dict(role='user', content='x')], **settings), tokenizer, 2048)


def test_context_drops_complete_turns_but_not_current_message(tokenizer):
    history = [ChatMessage('system', 'Stay short.'), ChatMessage('user', 'old '*60), ChatMessage('assistant', 'answer '*60)]
    result = build_context(history, 'current', [], tokenizer, 64, 8)
    assert result.retained_turns == 0
    assert result.ids == prompt_ids([history[0], ChatMessage('user', 'current')], tokenizer)
    with pytest.raises(ValueError):
        build_context(history, 'oversized '*500, [], tokenizer, 64, 8)


class Engine:
    def __init__(self, tokenizer):
        self.device = torch.device('cpu')
        self.tokenizer = tokenizer
        self.model = SimpleNamespace(config=SimpleNamespace(context_length=2048))
        self.closed = False
    def reset(self):
        self.messages = []
    def reply(self, text, *, token_callback, context_callback, preserve_whitespace):
        ids = self.tokenizer.encode(' é 🌍.', add_special_tokens=False).ids
        context_callback(SimpleNamespace(ids=[1, 5, 8, 7, 6], retained_turns=0))
        for token_id in ids:
            token_callback(token_id)
        return self.tokenizer.decode(ids), GeneratedSequence(tuple(ids+[7]), tuple(ids), 'stop', 'stop_token', 7, (7,))
    def close(self):
        self.closed = True


def test_stream_events_reconstruct_exact_unicode_and_stopping(tokenizer, capsys):
    engine = Engine(tokenizer)
    serve(engine, model_id='fixture', checkpoint_sha256='a'*64, tokenizer_sha256='b'*64,
          startup_seconds=0.01, input_stream=StringIO(json.dumps(request([dict(role='user', content='hello')]))+'\n'))
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[0]['type'] == 'ready' and engine.closed
    assert all(row['protocol'] == PROTOCOL for row in rows)
    assert ''.join(row['text'] for row in rows if row['type'] == 'delta') == ' é 🌍.'
    assert rows[-1]['text'] == ' é 🌍.' and rows[-1]['termination_cause'] == 'stop_token'
    events = rows[1:]
    assert [row['sequence'] for row in events] == list(range(1, len(events)+1))


def test_malformed_request_is_visible_and_next_request_works(tokenizer, capsys):
    engine = Engine(tokenizer)
    serve(engine, model_id='fixture', checkpoint_sha256='a'*64, tokenizer_sha256='b'*64,
          startup_seconds=0, input_stream=StringIO('not json\n'+json.dumps(request([dict(role='user', content='hello')]))+'\n'))
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[1]['type'] == 'error' and rows[-1]['type'] == 'done'


def test_greedy_nonfinite_logits_rejected():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(context_length=16,vocab_size=8)
        def forward(self, ids):
            return torch.full((*ids.shape,8),float('nan')),None
    with pytest.raises(FloatingPointError):
        generate_sequences(Model(),torch.tensor([[1,5]]),eos_token_id=2,config=GenerationConfig(max_new_tokens=2))
