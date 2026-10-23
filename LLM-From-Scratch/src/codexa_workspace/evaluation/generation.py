"""Shared inference helpers for evaluation: model loading, decoding context and output metrics."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from pathlib import Path
import re
import time

EXPECTED_PARAMETERS = 934_356_480
PAD_ID, EOS_ID, END_ID = 0, 2, 7


def mechanical(ids, text: str) -> dict:
    """Token and word n-gram repetition metrics plus the severe-degeneration flags."""
    words = re.findall(r"\b[\w']+\b", text.casefold())
    result = {}
    for name, sequence in (('token', list(ids)), ('word', words)):
        for n in (1, 2, 3, 4):
            grams = [tuple(sequence[i:i + n]) for i in range(max(0, len(sequence) - n + 1))]
            distinct = len(set(grams)) / len(grams) if grams else 0.0
            result[f'{name}_distinct_{n}'] = distinct
            result[f'{name}_repeat_{n}'] = 1 - distinct if grams else 0.0
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    result['maximum_repeated_line_count'] = max(lines.values(), default=0)
    result['historical_severe'] = result['word_repeat_4'] > .5
    result['independent_severe'] = (result['historical_severe'] or result['token_repeat_4'] > .65
                                    or result['maximum_repeated_line_count'] >= 3)
    return result


def strict_success(response: str, expected: str | None) -> bool:
    """Exact surface only: no case folding, trimming or cleanup can rescue an answer."""
    return expected is not None and response == expected


def termination(terminator: int | None, finish_reason: str) -> dict:
    natural = finish_reason == 'stop' and terminator in (EOS_ID, END_ID)
    return dict(natural=natural, chat_end=natural and terminator == END_ID,
                document_eos=natural and terminator == EOS_ID, length_limit=finish_reason == 'length')


@contextmanager
def inference(model, device: str):
    """Eval mode + inference mode + BF16 autocast on CUDA; restore the prior mode."""
    import torch
    was_training = model.training
    model.eval()
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            yield
    finally:
        model.train(was_training)


def load_inference_model(path: Path, expected_sha256: str, *, device: str = 'cuda',
                         memory_fraction: float | None = .5):
    """Weights only, strict load, frozen parameters, eval mode. Returns (model, metadata)."""
    import gc
    import torch
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    started = time.monotonic()
    payload = read_native_checkpoint(path, expected_sha256=expected_sha256)
    if not all(bool(torch.isfinite(v).all()) for v in payload['model_state_dict'].values()):
        raise FloatingPointError('Nonfinite checkpoint weights.')
    config = payload['config']['model']
    model = LanguageModel(ModelConfig(**config))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError('Tied embedding identity drift.')
    if count_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError('Architecture parameter count drift.')
    model.requires_grad_(False)
    model.eval()
    del payload
    gc.collect()
    loaded = time.monotonic()
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        model.to(device)
        torch.cuda.synchronize()
    metadata = dict(path=str(path), sha256=expected_sha256, model_config=config, parameters=EXPECTED_PARAMETERS,
                    tied_embeddings=True, weights_dtype=str(next(model.parameters()).dtype),
                    autocast='bf16' if device == 'cuda' else None, tf32=False, device=device,
                    requires_grad=any(p.requires_grad for p in model.parameters()),
                    seconds=dict(read_and_build=loaded - started, to_device=time.monotonic() - loaded))
    return model, metadata
