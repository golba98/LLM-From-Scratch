"""Offline diagnostic checks. No backward passes or optimizer construction."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from llm_architecture.model import LanguageModel, ModelConfig
from llm_inference.generate import GenerationConfig, generate_sequences

spec = importlib.util.spec_from_file_location("post_training_diagnostic", Path(__file__).parents[1] / "scripts/diagnostics/post_training_generation_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class ScriptedModel(torch.nn.Module):
    def __init__(self, transitions, context_length=16):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(context_length=context_length, vocab_size=16)
        self.transitions = transitions

    def forward(self, inputs):
        logits = torch.full((*inputs.shape, 16), -100., device=inputs.device)
        for i in range(inputs.shape[0]):
            for j in range(inputs.shape[1]):
                logits[i, j, self.transitions.get(int(inputs[i, j]), 2)] = 100.
        return logits, None


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("stop", [2, 7])
def test_immediate_stop_keeps_raw_token(device, stop):
    model = ScriptedModel({5: stop}).to(device).eval()
    callbacks = []
    result = generate_sequences(model, torch.tensor([[5]], device=device), eos_token_id=2,
        stop_sequences=[[7]], config=GenerationConfig(max_new_tokens=8), token_callback=callbacks.append).sequences[0]
    assert result.token_ids == (stop,)
    assert result.visible_token_ids == ()
    assert result.finish_reason == "stop"
    assert result.terminating_token_id == stop
    assert callbacks == []


def test_multitoken_stop_and_masked_batch():
    model = ScriptedModel({5: 8, 8: 9, 9: 10, 10: 2}).eval()
    result = generate_sequences(model, torch.tensor([[0, 5], [5, 0]]),
        attention_mask=torch.tensor([[0, 1], [1, 0]], dtype=torch.bool), eos_token_id=2,
        stop_sequences=[[9, 10]], config=GenerationConfig(max_new_tokens=8))
    for row in result.sequences:
        assert row.token_ids == (8, 9, 10)
        assert row.visible_token_ids == (8,)
        assert row.stop_sequence == (9, 10)


def test_context_exhaustion():
    model = ScriptedModel({5: 5}, context_length=4).eval()
    result = generate_sequences(model, torch.tensor([[5, 5, 5]]), eos_token_id=2,
        config=GenerationConfig(max_new_tokens=8)).sequences[0]
    assert result.token_ids == (5,)
    assert result.finish_reason == "length"
    with pytest.raises(ValueError):
        generate_sequences(model, torch.tensor([[5]*5]), eos_token_id=2, config=GenerationConfig())


def test_repetition_metrics_do_not_call_single_duplicate_severe():
    legitimate = "The first cup is red. The second cup is blue."
    assert not diagnostic.mechanical(list(range(20)), legitimate)["independent_severe"]
    loop = "The chemical symbol for carbon is carbon.\n" * 15
    assert diagnostic.mechanical([8, 9, 10, 11] * 30, loop)["independent_severe"]


def test_validation_sources_follow_window_ids_not_json_key_order():
    assert diagnostic.validation_source_list({"12": "wikipedia", "7": "fineweb_edu"}, [7, 12]) == ["fineweb_edu", "wikipedia"]


@pytest.mark.parametrize("positions", ["learned", "rotary"])
def test_cpu_cache_reference_and_eos_loss(positions):
    torch.manual_seed(42)
    model = LanguageModel(ModelConfig(vocab_size=32, context_length=32, num_layers=2,
        hidden_size=16, num_heads=4, intermediate_size=32, position_embedding_type=positions)).eval()
    ids = torch.tensor([[8, 9, 10, 2, 11, 12]])
    with torch.inference_mode():
        full = model(ids)[0]
        cache = model(ids[:, :3], use_cache=True)[2]
        cached = model(ids[:, 3:], past_key_values=cache, use_cache=True)[0]
        torch.testing.assert_close(full[:, 3:], cached, atol=1e-5, rtol=1e-4)
        targets = torch.tensor([[9, 10, 2, 11, 12, -100]])
        logits, loss = model(ids, labels=targets)
        expected = torch.nn.functional.cross_entropy(logits[0, :5], targets[0, :5])
        torch.testing.assert_close(loss, expected)
        assert targets[0, 2] == 2


def test_masked_cache_offset_repaired_in_architecture():
    # The diagnostic isolated a position reset in masked cached forwards
    # (formerly asserted here as `broken`); LLM-Architecture now continues
    # default positions from the cached length.
    torch.manual_seed(42)
    model = LanguageModel(ModelConfig(vocab_size=32, context_length=32, num_layers=2,
        hidden_size=16, num_heads=4, intermediate_size=32)).eval()
    ids = torch.tensor([[8, 9, 10, 11]])
    with torch.inference_mode():
        full = model(ids)[0][:, -1:]
        cache = model(ids[:, :3], use_cache=True)[2]
        mask = torch.ones((1, 1), dtype=torch.bool)
        default = model(ids[:, 3:], attention_mask=mask, past_key_values=cache, use_cache=True)[0]
        explicit = model(ids[:, 3:], attention_mask=mask, position_ids=torch.tensor([[3]]), past_key_values=cache, use_cache=True)[0]
        reset = model(ids[:, 3:], attention_mask=mask, position_ids=torch.tensor([[0]]), past_key_values=cache, use_cache=True)[0]
        torch.testing.assert_close(full, default, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(full, explicit, atol=1e-5, rtol=1e-4)
        assert not torch.allclose(full, reset, atol=1e-4, rtol=1e-4)
