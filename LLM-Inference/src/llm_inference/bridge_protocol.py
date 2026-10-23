"""Versioned, local-only streaming native chat transport; no tools or file requests."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time

import torch
from tokenizers.decoders import DecodeStream

from llm_tokenizer.sft import ChatMessage
from llm_inference.context import prompt_ids
from llm_inference.generate import GenerationConfig

PROTOCOL = "codexa-native-jsonl-v2"
MAX_REQUEST_BYTES = 2 * 1024 * 1024
GENERATION_FIELDS = {"max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "seed", "do_sample"}


def emit(payload: dict) -> None:
    """Flush one machine-readable event without logging conversation content."""
    print(json.dumps({"protocol": PROTOCOL, **payload}, ensure_ascii=False, allow_nan=False), flush=True)


def parse_chat(request: object, tokenizer, context_length: int):
    """Validate role-ordered conversation and effective decoding settings."""
    if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported native bridge protocol.")
    if request.get("type") != "chat" or not isinstance(request.get("id"), str) or not request["id"]:
        raise ValueError("Chat requires a non-empty request ID.")
    if set(request) - {"protocol", "type", "id", "messages", "generation"}:
        raise ValueError("Unknown request fields; file and tool requests are unsupported.")
    rows = request.get("messages")
    if not isinstance(rows, list) or not rows or len(rows) > 4096:
        raise ValueError("Chat requires a bounded message list.")
    messages = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"role", "content"}
                or row["role"] not in {"system", "user", "assistant"}
                or not isinstance(row["content"], str) or not row["content"].strip()):
            raise ValueError("Invalid conversation message.")
        messages.append(ChatMessage(row["role"], row["content"]))
    prompt_ids(messages, tokenizer)  # Reuse native alternation and boundary validation.
    values = request.get("generation", {})
    if not isinstance(values, dict) or set(values) - GENERATION_FIELDS:
        raise ValueError("Unknown generation option.")
    if "do_sample" in values and not isinstance(values["do_sample"], bool):
        raise ValueError("do_sample must be boolean.")
    config = GenerationConfig(**values)
    if config.max_new_tokens >= context_length:
        raise ValueError("Generation reservation must be less than context length.")
    if config.top_k is not None and config.top_k > tokenizer.get_vocab_size():
        raise ValueError("top_k exceeds vocabulary size.")
    return messages, config


def available_host_bytes() -> int:
    """Read Linux available memory without clearing caches or modifying swap."""
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot establish available host memory.")


def preflight(device: str) -> dict:
    """Conservative interactive inference gates, separate from training disk policy."""
    host = available_host_bytes()
    if host < 16 * 1024**3:
        raise RuntimeError("Codexa requires at least 16 GiB available host RAM before loading.")
    result = {"host_available_bytes": host}
    if device != 'cpu':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select CPU explicitly to use slow CPU inference.")
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_free_bytes=free, gpu_total_bytes=total)
        if free < 8 * 1024**3:
            raise RuntimeError("Codexa requires 8 GiB free VRAM for FP32 inference and workspace headroom.")
    return result


def memory_stats(device) -> dict:
    """Report current-process memory, without querying unrelated private content."""
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            result[line.split(':')[0]] = int(line.split()[1]) * 1024
    if device.type == 'cuda':
        result.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                      gpu_reserved_bytes=torch.cuda.memory_reserved(),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def serve(engine, *, model_id: str, checkpoint_sha256: str, tokenizer_sha256: str,
          startup_seconds: float, input_stream=None) -> None:
    """Serve stateless conversations on one owned, persistent model process."""
    stream = input_stream if input_stream is not None else sys.stdin
    emit(dict(type='ready', model_id=model_id, checkpoint_sha256=checkpoint_sha256,
              tokenizer_sha256=tokenizer_sha256, device=str(engine.device),
              context_length=engine.model.config.context_length, loading_seconds=startup_seconds,
              **memory_stats(engine.device)))
    try:
        while True:
            line = stream.readline(MAX_REQUEST_BYTES + 1)
            if not line:
                break
            request = None
            request_id = ''
            try:
                if len(line.encode('utf-8')) > MAX_REQUEST_BYTES or not line.endswith('\n'):
                    raise ValueError("Native bridge request is oversized or incomplete.")
                request = json.loads(line)
                if isinstance(request, dict) and isinstance(request.get('id'), str):
                    request_id = request['id']
                messages, config = parse_chat(request, engine.tokenizer, engine.model.config.context_length)
                engine.reset()
                engine.messages = messages[:-1]
                engine.system_prompt = messages[0].content if messages[0].role == 'system' else None
                engine.generation_config = config
                started = time.perf_counter()
                first_token = None
                visible_ids = []
                sequence = 0
                decoded = ''
                decoder = DecodeStream(skip_special_tokens=True)
                context_tokens = 0

                def event(kind, **fields):
                    nonlocal sequence
                    sequence += 1
                    emit(dict(type=kind, id=request_id, sequence=sequence, **fields))

                def context_ready(context):
                    nonlocal context_tokens
                    context_tokens = len(context.ids)
                    historic_turns = (len(messages) - 1 - int(bool(engine.system_prompt))) // 2
                    event('context', input_tokens=context_tokens, reserved_output_tokens=config.max_new_tokens,
                          context_length=engine.model.config.context_length,
                          dropped_turns=historic_turns-context.retained_turns)
                    event('status', stage='prefill')

                def token_visible(token_id):
                    nonlocal first_token, decoded
                    if first_token is None:
                        first_token = time.perf_counter()
                    visible_ids.append(token_id)
                    delta = decoder.step(engine.tokenizer, token_id)
                    if delta:
                        decoded += delta
                        event('delta', text=delta)

                autocast = (torch.autocast('cuda', dtype=torch.bfloat16)
                            if engine.device.type == 'cuda' else nullcontext())
                with autocast:
                    response, generated = engine.reply(messages[-1].content, token_callback=token_visible,
                        context_callback=context_ready, preserve_whitespace=True)
                # DecodeStream may retain an incomplete byte sequence until EOS.
                if not response.startswith(decoded):
                    raise RuntimeError("Streaming decoder did not preserve native output.")
                if response[len(decoded):]:
                    event('delta', text=response[len(decoded):])
                elapsed = time.perf_counter() - started
                event('done', text=response, model_id=model_id, checkpoint_sha256=checkpoint_sha256,
                      finish_reason=generated.finish_reason, termination_cause=generated.termination_cause,
                      generated_tokens=generated.generated_token_count,
                      visible_tokens=len(generated.visible_token_ids), input_tokens=context_tokens,
                      context_length=engine.model.config.context_length, generation=config.__dict__,
                      elapsed_seconds=elapsed, first_token_seconds=(first_token-started if first_token else None),
                      tokens_per_second=generated.generated_token_count/elapsed if elapsed else 0,
                      **memory_stats(engine.device))
            except Exception as error:
                emit(dict(type='error', id=request_id, message=str(error), error_type=type(error).__name__))
                if isinstance(error, (torch.OutOfMemoryError, FloatingPointError)):
                    break
    finally:
        engine.close()
