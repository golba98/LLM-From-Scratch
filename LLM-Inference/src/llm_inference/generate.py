"""Autoregressive text-generation utilities for decoder-only models."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GenerationConfig:
    """Validated decoding settings."""

    max_new_tokens: int = 128
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int | None = None
    do_sample: bool = False
    seed: int = 42
    use_kv_cache: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer.")
        if self.temperature is not None and (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(float(self.temperature))
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be positive and finite when supplied.")
        if self.do_sample and self.temperature is None:
            raise ValueError("Sampling requires a positive temperature.")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer when supplied.")
        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(float(self.top_p))
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1].")
        if (
            not isinstance(self.repetition_penalty, (int, float))
            or isinstance(self.repetition_penalty, bool)
            or not math.isfinite(float(self.repetition_penalty))
            or self.repetition_penalty <= 0
        ):
            raise ValueError("repetition_penalty must be positive and finite.")
        if self.no_repeat_ngram_size is not None and (
            not isinstance(self.no_repeat_ngram_size, int)
            or isinstance(self.no_repeat_ngram_size, bool)
            or self.no_repeat_ngram_size <= 1
        ):
            raise ValueError("no_repeat_ngram_size must be greater than one when supplied.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(self.use_kv_cache, bool):
            raise ValueError("use_kv_cache must be a boolean.")


@dataclass(frozen=True)
class GeneratedSequence:
    """Generated IDs and exact termination metadata for one prompt."""

    token_ids: tuple[int, ...]
    visible_token_ids: tuple[int, ...]
    finish_reason: str
    termination_cause: str
    terminating_token_id: int | None
    stop_sequence: tuple[int, ...] | None

    @property
    def generated_token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class GenerationBatchResult:
    """Per-row generation results for a possibly padded input batch."""

    sequences: tuple[GeneratedSequence, ...]


def compile_generation_model(model: nn.Module) -> nn.Module:
    """Compile the model forward used by cached autoregressive decoding."""

    compiler = getattr(torch, "compile", None)
    if compiler is None:
        raise RuntimeError("This PyTorch build does not provide torch.compile.")
    try:
        return compiler(model, mode="default", dynamic=True)
    except Exception as error:
        raise RuntimeError(
            "torch.compile could not initialize its CUDA backend. "
            "The environment needs a working Triton toolchain and Python "
            "development headers (Python.h)."
        ) from error


def _mark_compiled_step(model: nn.Module) -> None:
    """Tell CUDA Graphs that a compiled decode invocation is independent."""

    if not hasattr(model, "_orig_mod"):
        return
    compiler = getattr(torch, "compiler", None)
    marker = None if compiler is None else getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is not None:
        marker()


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Apply the standard sign-aware repetition penalty once per token ID."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary].")
    if generated_token_ids.ndim != 2:
        raise ValueError("generated_token_ids must have shape [batch, sequence].")
    if logits.shape[0] != generated_token_ids.shape[0]:
        raise ValueError("logits and generated tokens must share a batch size.")
    if penalty <= 0 or not math.isfinite(penalty):
        raise ValueError("penalty must be positive and finite.")
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()
    for batch_index in range(adjusted.shape[0]):
        token_ids = torch.unique(generated_token_ids[batch_index])
        selected = adjusted[batch_index, token_ids]
        adjusted[batch_index, token_ids] = torch.where(
            selected < 0,
            selected * penalty,
            selected / penalty,
        )
    return adjusted


def ban_repeated_ngrams(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    ngram_size: int | None,
) -> torch.Tensor:
    """Prevent completing an n-gram that already occurred in the sequence."""

    if logits.ndim != 2 or token_ids.ndim != 2:
        raise ValueError("logits and token_ids must be two-dimensional.")
    if logits.shape[0] != token_ids.shape[0]:
        raise ValueError("logits and token IDs must share a batch size.")
    if ngram_size is None:
        return logits
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool):
        raise ValueError("ngram_size must be an integer when supplied.")
    if ngram_size <= 1:
        raise ValueError("ngram_size must be greater than one when supplied.")
    if token_ids.shape[1] < ngram_size - 1:
        return logits

    adjusted = logits.clone()
    for batch_index in range(token_ids.shape[0]):
        sequence = token_ids[batch_index].tolist()
        prefix = tuple(sequence[-(ngram_size - 1) :])
        banned: set[int] = set()
        for start in range(len(sequence) - ngram_size + 1):
            if tuple(sequence[start : start + ngram_size - 1]) == prefix:
                banned.add(sequence[start + ngram_size - 1])
        if banned:
            candidate_logits = adjusted[batch_index]
            candidate_logits[list(banned)] = float("-inf")
            # A tiny or degenerate vocabulary can make every candidate invalid.
            # Keep the original row in that case rather than producing NaNs.
            if torch.isneginf(candidate_logits).all():
                adjusted[batch_index] = logits[batch_index]
    return adjusted


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and nucleus filtering to one logit row per batch."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary].")
    filtered = logits.clone()
    vocabulary_size = filtered.shape[-1]
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        retained = min(top_k, vocabulary_size)
        threshold = torch.topk(filtered, retained, dim=-1).values[:, -1:]
        filtered.masked_fill_(filtered < threshold, float("-inf"))

    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            descending=True,
            dim=-1,
        )
        cumulative = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1,
        )
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits.masked_fill_(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


def _validate_stop_sequences(
    stop_sequences: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> tuple[tuple[int, ...], ...]:
    normalized: list[tuple[int, ...]] = []
    for sequence in stop_sequences:
        value = tuple(sequence)
        if not value:
            raise ValueError("Stop-token sequences must not be empty.")
        if any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or not 0 <= token_id < vocab_size
            for token_id in value
        ):
            raise ValueError("Stop-token sequence contains an invalid token ID.")
        if value not in normalized:
            normalized.append(value)
    return tuple(normalized)


def _pending_stop_prefix_length(
    token_ids: Sequence[int],
    stop_sequences: Sequence[tuple[int, ...]],
) -> int:
    maximum = 0
    for stop in stop_sequences:
        for length in range(1, len(stop)):
            if len(token_ids) >= length and tuple(token_ids[-length:]) == stop[:length]:
                maximum = max(maximum, length)
    return maximum


@torch.inference_mode()
def generate_sequences(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int = 0,
    attention_mask: torch.Tensor | None = None,
    stop_sequences: Sequence[Sequence[int]] = (),
    config: GenerationConfig,
    token_callback: Callable[[int], None] | None = None,
) -> GenerationBatchResult:
    """Generate per-row continuations with token-level stop semantics."""

    if input_ids.ndim != 2 or input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError("input_ids must have shape [batch, non-empty sequence].")
    if input_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("input_ids must use an integer dtype.")
    model_config = getattr(model, "config", None)
    context_length = getattr(model_config, "context_length", None)
    vocab_size = getattr(model_config, "vocab_size", None)
    if not isinstance(context_length, int) or not isinstance(vocab_size, int):
        raise ValueError("model must expose config.context_length and vocab_size.")
    if input_ids.shape[1] > context_length:
        raise ValueError("Prompt exceeds the model context length.")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id is outside the model vocabulary.")
    if config.top_k is not None and config.top_k > vocab_size:
        raise ValueError("top_k cannot exceed the model vocabulary size.")

    if not 0 <= pad_token_id < vocab_size:
        raise ValueError("pad_token_id is outside the model vocabulary.")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids shape.")
    attention_mask = attention_mask.to(dtype=torch.bool)
    if not attention_mask.any(dim=1).all():
        raise ValueError("Every prompt must contain at least one non-padding token.")
    if token_callback is not None and input_ids.shape[0] != 1:
        raise ValueError("token_callback is supported only for one prompt.")
    normalized_stops = _validate_stop_sequences(
        ((eos_token_id,), *stop_sequences),
        vocab_size=vocab_size,
    )
    device = next(model.parameters()).device
    was_training = model.training
    results: list[GeneratedSequence] = []
    try:
        model.eval()
        original_model = getattr(model, "_orig_mod", None)
        cache_enabled = bool(
            config.use_kv_cache
            and (
                getattr(model, "supports_kv_cache", False)
                or getattr(original_model, "supports_kv_cache", False)
            )
        )
        for row_index in range(input_ids.shape[0]):
            prompt = input_ids[row_index][attention_mask[row_index]].to(device)
            generated = prompt.unsqueeze(0)
            generated_ids: list[int] = []
            emitted_count = 0
            generator = torch.Generator(device=device)
            generator.manual_seed(config.seed + row_index)
            available = context_length - generated.shape[1]
            limit = min(config.max_new_tokens, available)
            matched_stop: tuple[int, ...] | None = None
            past_key_values = None
            next_input = generated
            for _ in range(limit):
                if cache_enabled:
                    _mark_compiled_step(model)
                    logits, _loss, past_key_values = model(
                        next_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    logits, _loss = model(generated)
                next_logits = logits[:, -1, :].float()
                if not torch.isfinite(next_logits).all():
                    raise FloatingPointError("Generation logits contain NaN or Infinity.")
                next_logits = apply_repetition_penalty(
                    next_logits,
                    generated,
                    float(config.repetition_penalty),
                )
                next_logits = ban_repeated_ngrams(
                    next_logits,
                    generated,
                    config.no_repeat_ngram_size,
                )
                if config.do_sample:
                    assert config.temperature is not None
                    next_logits = next_logits / float(config.temperature)
                    next_logits = filter_logits(
                        next_logits,
                        top_k=config.top_k,
                        top_p=config.top_p,
                    )
                    probabilities = torch.softmax(next_logits, dim=-1)
                    if not torch.isfinite(probabilities).all():
                        raise FloatingPointError(
                            "Sampling probabilities contain NaN or Infinity."
                        )
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                else:
                    next_token = torch.argmax(
                        next_logits,
                        dim=-1,
                        keepdim=True,
                    )
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                generated = torch.cat((generated, next_token), dim=1)
                next_input = next_token
                for stop in normalized_stops:
                    if len(generated_ids) >= len(stop) and tuple(
                        generated_ids[-len(stop) :]
                    ) == stop:
                        matched_stop = stop
                        break
                visible_end = len(generated_ids) - (
                    len(matched_stop)
                    if matched_stop is not None
                    else _pending_stop_prefix_length(
                        generated_ids,
                        normalized_stops,
                    )
                )
                if token_callback is not None:
                    for visible_id in generated_ids[emitted_count:visible_end]:
                        token_callback(visible_id)
                emitted_count = visible_end
                if matched_stop is not None:
                    break
            if matched_stop is None and token_callback is not None:
                for visible_id in generated_ids[emitted_count:]:
                    token_callback(visible_id)
            visible_count = (
                len(generated_ids) - len(matched_stop)
                if matched_stop is not None
                else len(generated_ids)
            )
            results.append(
                GeneratedSequence(
                    token_ids=tuple(generated_ids),
                    visible_token_ids=tuple(generated_ids[:visible_count]),
                    finish_reason="stop" if matched_stop is not None else "length",
                    termination_cause=(
                        "eos"
                        if matched_stop == (eos_token_id,)
                        else "stop_token"
                        if matched_stop is not None
                        else "length"
                    ),
                    terminating_token_id=(
                        matched_stop[-1] if matched_stop is not None else None
                    ),
                    stop_sequence=matched_stop,
                )
            )
        return GenerationBatchResult(tuple(results))
    finally:
        model.train(was_training)


@torch.inference_mode()
def generate_token_ids(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    eos_token_id: int,
    config: GenerationConfig,
) -> torch.Tensor:
    """Compatibility wrapper returning padded prompt-plus-generation IDs."""

    result = generate_sequences(
        model,
        input_ids,
        eos_token_id=eos_token_id,
        config=config,
    )
    rows = [
        torch.tensor(
            [*input_ids[index].tolist(), *sequence.token_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        for index, sequence in enumerate(result.sequences)
    ]
    maximum = max(len(row) for row in rows)
    padded = [
        torch.cat(
            (
                row,
                torch.full(
                    (maximum - len(row),),
                    eos_token_id,
                    dtype=row.dtype,
                    device=row.device,
                ),
            )
        )
        for row in rows
    ]
    return torch.stack(padded)
