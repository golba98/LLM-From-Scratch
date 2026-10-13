"""Executable tests for greedy and sampled autoregressive generation."""

import json
from pathlib import Path
import tempfile
import torch
from torch import nn

from scripts.inference.generate import build_argument_parser, run
from src.checkpointing import (
    CheckpointManager,
    SchedulerState,
    build_checkpoint_payload,
)
from src.config import ProjectConfig, TrainingConfig
from src.generate import (
    GenerationConfig,
    apply_repetition_penalty,
    ban_repeated_ngrams,
    filter_logits,
    generate_sequences,
    generate_token_ids,
)
from src.model import LanguageModel, ModelConfig
from src.token_data import file_sha256
from src.tokenizer import train_tokenizer
from src.training import TrainingState, create_adamw_optimizer


class ScriptedModel(nn.Module):
    """Return a fixed next-token preference for generation tests."""

    def __init__(
        self,
        *,
        next_token_id: int,
        vocab_size: int = 8,
        context_length: int = 8,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = ModelConfig(
            vocab_size=vocab_size,
            context_length=context_length,
            num_layers=1,
            hidden_size=8,
            num_heads=2,
            intermediate_size=16,
        )
        self.next_token_id = next_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -10.0,
            device=input_ids.device,
        )
        logits[:, -1, self.next_token_id] = 10.0 + self.anchor
        return logits, None


def assert_raises(
    exception_type: type[BaseException],
    callable_object: object,
    *args: object,
    **kwargs: object,
) -> None:
    try:
        callable_object(*args, **kwargs)  # type: ignore[operator]
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def test_generation_config_validation() -> None:
    assert_raises(ValueError, GenerationConfig, max_new_tokens=0)
    assert_raises(ValueError, GenerationConfig, temperature=0)
    assert_raises(ValueError, GenerationConfig, top_k=0)
    assert_raises(ValueError, GenerationConfig, top_p=0)
    assert_raises(ValueError, GenerationConfig, repetition_penalty=0)
    assert_raises(ValueError, GenerationConfig, seed=-1)
    assert_raises(ValueError, GenerationConfig, use_kv_cache=1)


def test_kv_cache_matches_uncached_logits_and_generation() -> None:
    torch.manual_seed(19)
    config = ModelConfig(
        vocab_size=32,
        context_length=16,
        num_layers=2,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
    )
    model = LanguageModel(config).eval()
    prompt = torch.tensor([[3, 7, 11, 5, 2]], dtype=torch.long)

    with torch.inference_mode():
        uncached_logits, _ = model(prompt)
        cached_logits, _, cache = model(
            prompt[:, :3],
            use_cache=True,
        )
        torch.testing.assert_close(
            cached_logits,
            uncached_logits[:, :3],
            rtol=1e-5,
            atol=1e-6,
        )
        for index in range(3, prompt.shape[1]):
            cached_logits, _, cache = model(
                prompt[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            torch.testing.assert_close(
                cached_logits[:, -1],
                uncached_logits[:, index],
                rtol=1e-5,
                atol=1e-6,
            )

        uncached = generate_sequences(
            model,
            prompt[:, :3],
            eos_token_id=31,
            config=GenerationConfig(
                max_new_tokens=4,
                do_sample=False,
                use_kv_cache=False,
            ),
        ).sequences[0]
        cached = generate_sequences(
            model,
            prompt[:, :3],
            eos_token_id=31,
            config=GenerationConfig(
                max_new_tokens=4,
                do_sample=False,
                use_kv_cache=True,
            ),
        ).sequences[0]

    assert cached.token_ids == uncached.token_ids
    assert cached.finish_reason == uncached.finish_reason


def test_filters_and_penalty() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    top_k = filter_logits(logits, top_k=2, top_p=None)
    assert torch.isfinite(top_k[0, :2]).all()
    assert torch.isneginf(top_k[0, 2:]).all()
    top_p = filter_logits(logits, top_k=None, top_p=0.7)
    assert torch.isfinite(top_p[0, 0])
    assert torch.isneginf(top_p[0, -1])

    penalized = apply_repetition_penalty(
        torch.tensor([[4.0, -4.0, 2.0]]),
        torch.tensor([[0, 1, 1]]),
        2.0,
    )
    assert penalized.tolist() == [[2.0, -8.0, 2.0]]

    guarded = ban_repeated_ngrams(
        torch.tensor([[4.0, 3.0, 2.0]]),
        torch.tensor([[1, 2, 1, 2]]),
        3,
    )
    assert torch.isneginf(guarded[0, 1])
    assert torch.isfinite(guarded[0, 0])


def test_greedy_eos_and_context() -> None:
    model = ScriptedModel(next_token_id=2)
    model.train()
    generated = generate_token_ids(
        model,
        torch.tensor([[1, 4]], dtype=torch.long),
        eos_token_id=2,
        config=GenerationConfig(max_new_tokens=5, do_sample=False),
    )
    assert generated.tolist() == [[1, 4, 2]]
    assert model.training

    context_model = ScriptedModel(next_token_id=4, context_length=3)
    limited = generate_token_ids(
        context_model,
        torch.tensor([[1, 5]], dtype=torch.long),
        eos_token_id=2,
        config=GenerationConfig(max_new_tokens=5, do_sample=False),
    )
    assert limited.shape == (1, 3)


def _sampling_is_seeded(device: torch.device) -> None:
    model = ScriptedModel(next_token_id=4).to(device)
    prompt = torch.tensor([[1]], dtype=torch.long, device=device)
    config = GenerationConfig(
        max_new_tokens=4,
        temperature=1.0,
        top_k=4,
        top_p=0.95,
        do_sample=True,
        seed=17,
    )
    first = generate_token_ids(model, prompt, eos_token_id=2, config=config)
    second = generate_token_ids(model, prompt, eos_token_id=2, config=config)
    assert torch.equal(first, second)
    assert first.device.type == device.type


def test_sampling_is_seeded() -> None:
    _sampling_is_seeded(torch.device("cpu"))


def test_stop_tokens_and_padded_batch() -> None:
    stop_model = ScriptedModel(next_token_id=5, vocab_size=8)
    stopped = generate_sequences(
        stop_model,
        torch.tensor([[1, 4]], dtype=torch.long),
        eos_token_id=2,
        stop_sequences=((5,),),
        config=GenerationConfig(max_new_tokens=4),
    ).sequences[0]
    assert stopped.token_ids == (5,)
    assert stopped.visible_token_ids == ()
    assert stopped.finish_reason == "stop"
    assert stopped.terminating_token_id == 5

    length_model = ScriptedModel(next_token_id=4, context_length=8)
    padded = torch.tensor([[1, 6, 0], [1, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    batch = generate_sequences(
        length_model,
        padded,
        attention_mask=mask,
        eos_token_id=2,
        pad_token_id=0,
        config=GenerationConfig(max_new_tokens=2),
    )
    assert len(batch.sequences) == 2
    assert all(sequence.token_ids == (4, 4) for sequence in batch.sequences)
    assert all(sequence.finish_reason == "length" for sequence in batch.sequences)


def _write_tokenizer(root: Path) -> Path:
    corpus = root / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "text": "Codexa learns this small original sentence.",
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return train_tokenizer(
        [corpus],
        output_dir=root / "tokenizer",
        vocab_size=300,
        min_frequency=1,
    ).tokenizer_path


def _project_config() -> ProjectConfig:
    return ProjectConfig(
        model=ModelConfig(
            vocab_size=512,
            context_length=64,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            intermediate_size=32,
        ),
        training=TrainingConfig(
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0.0,
            warmup_steps=1,
            max_steps=4,
            gradient_clip=1.0,
            precision="fp32",
            checkpoint_interval=2,
            evaluation_interval=2,
            seed=42,
        ),
    )


def test_checkpoint_cli_generation() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        tokenizer_path = _write_tokenizer(root)
        config = _project_config()
        model = LanguageModel(config.model)
        optimizer = create_adamw_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=None,
            state=TrainingState(optimizer_step=1, micro_step=1),
            scheduler=SchedulerState(
                warmup_steps=1,
                max_steps=4,
                peak_learning_rate=1e-3,
                minimum_learning_rate=1e-4,
            ),
            config=config,
            run_name="generation",
            run_id="generation-run-id",
            tokenizer_reference=str(tokenizer_path),
            tokenizer_sha256=file_sha256(tokenizer_path),
        )
        checkpoint = CheckpointManager(root / "checkpoints", "generation").save(
            payload
        )
        output_path = root / "samples" / "sample.json"
        arguments = build_argument_parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--tokenizer",
                str(tokenizer_path),
                "--prompt",
                "Codexa",
                "--device",
                "cpu",
                "--max-new-tokens",
                "3",
                "--greedy",
                "--output",
                str(output_path),
            ]
        )
        output = run(arguments)
        assert output["checkpoint_run_id"] == "generation-run-id"
        assert output["device"] == "cpu"
        assert len(output["generated_token_ids"]) <= 3
        assert output_path.is_file()
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["generation_config"]["do_sample"] is False

def main() -> None:
    test_generation_config_validation()
    test_filters_and_penalty()
    test_greedy_eos_and_context()
    test_sampling_is_seeded()
    test_stop_tokens_and_padded_batch()
    test_checkpoint_cli_generation()
    cuda_result = "skipped"
    if torch.cuda.is_available():
        _sampling_is_seeded(torch.device("cuda"))
        cuda_result = "passed"
    print("CPU generation: passed")
    print(f"CUDA generation: {cuda_result}")
    print("Greedy decoding: passed")
    print("Top-k/top-p sampling: passed")
    print("EOS stopping: passed")
    print("All generation tests passed.")


if __name__ == "__main__":
    main()
