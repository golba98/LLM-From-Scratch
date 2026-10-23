"""generate_sequences regressions on a real tiny LanguageModel (cache on/off,
padded batches, stop semantics). Offline; no optimizer or checkpoint I/O."""

import pytest
import torch

from llm_architecture.model import LanguageModel, ModelConfig
from llm_inference.generate import GenerationConfig, generate_sequences

POSITIONS = ("learned", "rotary")
DEVICES = (
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ),
)
UNUSED_EOS = 1


def build(positions: str, device: str = "cpu", context_length: int = 32) -> LanguageModel:
    torch.manual_seed(7)
    config = ModelConfig(
        vocab_size=48,
        context_length=context_length,
        num_layers=2,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        position_embedding_type=positions,
    )
    return LanguageModel(config).to(device).eval()


def generate(model, prompt, *, eos=UNUSED_EOS, stops=(), use_kv_cache=True, max_new_tokens=12, mask=None):
    return generate_sequences(
        model,
        prompt,
        attention_mask=mask,
        eos_token_id=eos,
        pad_token_id=0,
        stop_sequences=stops,
        config=GenerationConfig(max_new_tokens=max_new_tokens, seed=42, use_kv_cache=use_kv_cache),
    ).sequences


def free_tokens(model, prompt: list[int], device: str) -> tuple[int, ...]:
    return generate(model, torch.tensor([prompt], device=device))[0].token_ids


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("do_sample", (False, True))
def test_cached_and_uncached_generation_identical(device, positions, do_sample):
    model = build(positions, device)
    prompt = torch.tensor([[5, 9, 13, 21, 8]], device=device)
    config = dict(max_new_tokens=16, seed=42, do_sample=do_sample, temperature=0.9 if do_sample else None)
    results = [
        generate_sequences(model, prompt, eos_token_id=UNUSED_EOS,
                           config=GenerationConfig(**config, use_kv_cache=cache)).sequences[0]
        for cache in (True, False)
    ]
    assert results[0] == results[1]
    if not do_sample:  # sampling may legitimately draw the EOS id
        assert len(results[0].token_ids) == 16 and results[0].finish_reason == "length"


@pytest.mark.parametrize("positions", POSITIONS)
@pytest.mark.parametrize("side", ("left", "right"))
def test_padded_batch_matches_per_row_generation(positions, side):
    model = build(positions)
    prompts = [[5, 9, 13], [7, 11, 17, 19, 23], [29, 31, 37, 41]]
    width = max(map(len, prompts))
    ids, masks = [], []
    for prompt in prompts:
        padding = [0] * (width - len(prompt))
        ids.append(padding + prompt if side == "left" else prompt + padding)
        valid = [0] * len(padding)
        masks.append(valid + [1] * len(prompt) if side == "left" else [1] * len(prompt) + valid)
    batch = generate(model, torch.tensor(ids), mask=torch.tensor(masks, dtype=torch.bool))
    for row, prompt in enumerate(prompts):
        assert batch[row].token_ids == free_tokens(model, prompt, "cpu")


@pytest.mark.parametrize("positions", POSITIONS)
def test_eos_and_multitoken_stop_semantics(positions):
    model = build(positions)
    prompt = [5, 9, 13, 21, 8]
    reference = free_tokens(model, prompt, "cpu")

    eos = reference[2]
    first = reference.index(eos)
    result = generate(model, torch.tensor([prompt]), eos=eos)[0]
    assert result.token_ids == reference[: first + 1]
    assert result.visible_token_ids == reference[:first]
    assert (result.finish_reason, result.termination_cause, result.terminating_token_id) == ("stop", "eos", eos)

    stop = reference[3:5]
    end = next(i for i in range(2, len(reference) + 1) if reference[i - 2 : i] == stop)
    result = generate(model, torch.tensor([prompt]), stops=[list(stop)])[0]
    assert result.token_ids == reference[:end]
    assert result.visible_token_ids == reference[: end - 2]
    assert result.stop_sequence == stop and result.termination_cause == "stop_token"


@pytest.mark.parametrize("positions", POSITIONS)
def test_generation_near_maximum_context(positions):
    model = build(positions, context_length=16)
    prompt = torch.tensor([list(range(3, 17))])
    cached = generate(model, prompt, max_new_tokens=8)[0]
    uncached = generate(model, prompt, max_new_tokens=8, use_kv_cache=False)[0]
    assert cached == uncached
    assert len(cached.token_ids) == 2 and cached.finish_reason == "length"
