"""Evaluate one Codexa checkpoint with validation loss and fixed prompts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_architecture.checkpointing import checkpoint_model_config as _checkpoint_model_config
from llm_training.checkpointing import load_model_checkpoint, verify_checkpoint_checksum
from src.evaluation import (
    analyze_generated_text,
    ngram_overlap_rate,
    perplexity_from_loss,
)
from llm_inference.generate import GenerationConfig, generate_sequences
from llm_architecture.model import LanguageModel, count_parameters
from llm_data.token_data import (
    MemmapTokenDataset,
    create_token_dataloader,
    file_sha256,
)
from llm_tokenizer.tokenizer import BOS_TOKEN, EOS_TOKEN, load_tokenizer
from llm_training.training import evaluate, resolve_device, resolve_precision


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("configs/evaluation_prompts.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-token-file", type=Path)
    parser.add_argument("--token-manifest", type=Path)
    parser.add_argument("--validation-token-offset", type=int, default=0)
    parser.add_argument("--validation-token-count", type=int)
    parser.add_argument(
        "--source-validation",
        action="append",
        nargs=3,
        metavar=("NAME", "TOKEN_FILE", "MANIFEST"),
        help="Evaluate a named source independently; repeat for each source.",
    )
    parser.add_argument("--max-validation-batches", type=int, default=16)
    parser.add_argument("--reference-jsonl", type=Path)
    parser.add_argument("--max-reference-documents", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--instruction-template")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_prompts(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Prompt suite must be a non-empty JSON array.")
    prompts: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or not isinstance(
            item.get("category"), str
        ):
            raise ValueError(f"Invalid prompt suite entry at index {index}.")
        prompt = item.get("prompt")
        target_tokens = item.get("target_prompt_tokens")
        simple = isinstance(prompt, str) and bool(prompt)
        long_context = (
            isinstance(target_tokens, int)
            and not isinstance(target_tokens, bool)
            and target_tokens > 0
            and all(
                isinstance(item.get(key), str) and bool(item[key])
                for key in ("prompt_prefix", "prompt_filler", "prompt_suffix")
            )
        )
        if simple == long_context:
            raise ValueError(
                f"Prompt suite entry {index} must define exactly one prompt "
                "form."
            )
        expected_terms = item.get("expected_terms", [])
        if (
            not isinstance(expected_terms, list)
            or any(
                not isinstance(term, str) or not term
                for term in expected_terms
            )
        ):
            raise ValueError(
                f"Prompt suite entry {index} has invalid expected_terms."
            )
        expected_term_search_characters = item.get(
            "expected_term_search_characters"
        )
        if expected_term_search_characters is not None and (
            not isinstance(expected_term_search_characters, int)
            or isinstance(expected_term_search_characters, bool)
            or expected_term_search_characters <= 0
        ):
            raise ValueError(
                f"Prompt suite entry {index} has an invalid expected-term "
                "search window."
            )
        expected_pattern = item.get("expected_pattern")
        if expected_pattern is not None:
            if not isinstance(expected_pattern, str) or not expected_pattern:
                raise ValueError(
                    f"Prompt suite entry {index} has invalid expected_pattern."
                )
            try:
                re.compile(expected_pattern)
            except re.error as error:
                raise ValueError(
                    f"Prompt suite entry {index} has invalid expected_pattern."
                ) from error
        prompts.append(dict(item))
    return prompts


def _expected_term_matches(
    terms: list[str],
    continuation: str,
    *,
    search_characters: int | None,
) -> dict[str, bool]:
    searchable = (
        continuation
        if search_characters is None
        else continuation[:search_characters]
    )
    folded = searchable.casefold()
    return {term: term.casefold() in folded for term in terms}


def _build_prompt(
    entry: dict[str, object],
    *,
    tokenizer,
    context_length: int,
    bos_token_id: int | None = None,
) -> tuple[str, list[int]]:
    prompt = entry.get("prompt")
    if isinstance(prompt, str):
        token_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        ).ids
        return prompt, token_ids

    target_tokens = int(entry["target_prompt_tokens"])
    if target_tokens >= context_length:
        raise ValueError(
            f"Long-context target {target_tokens} must be below model "
            f"context length {context_length}."
        )
    prefix_ids = tokenizer.encode(
        str(entry["prompt_prefix"]),
        add_special_tokens=False,
    ).ids
    filler_ids = tokenizer.encode(
        str(entry["prompt_filler"]),
        add_special_tokens=False,
    ).ids
    suffix_ids = tokenizer.encode(
        str(entry["prompt_suffix"]),
        add_special_tokens=False,
    ).ids
    fixed_length = len(prefix_ids) + len(suffix_ids)
    if not filler_ids:
        raise ValueError("Long-context filler must produce at least one token.")
    if fixed_length > target_tokens:
        raise ValueError(
            "Long-context prefix and suffix exceed target_prompt_tokens."
        )
    filler_length = target_tokens - fixed_length
    repeated_filler = (
        filler_ids * (filler_length // len(filler_ids) + 1)
    )[:filler_length]
    prompt_ids = prefix_ids + repeated_filler + suffix_ids
    if len(prompt_ids) != target_tokens:
        raise RuntimeError("Long-context prompt token accounting mismatch.")
    return tokenizer.decode(prompt_ids), prompt_ids


def _reference_text(path: Path, maximum_documents: int) -> str:
    if maximum_documents <= 0:
        raise ValueError("--max-reference-documents must be positive.")
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON ({error.msg})"
                ) from error
            if not isinstance(record, dict) or not isinstance(
                record.get("text"), str
            ):
                raise ValueError(
                    f"{path}:{line_number}: required text must be a string."
                )
            texts.append(record["text"])
            if len(texts) >= maximum_documents:
                break
    return "\n".join(texts)


def _validate_tokenizer_compatibility(
    *,
    checkpoint_checksum: str | None,
    tokenizer_checksum: str,
    tokenizer_vocab_size: int,
    model_vocab_size: int,
) -> None:
    if (
        checkpoint_checksum is not None
        and checkpoint_checksum != tokenizer_checksum
    ):
        raise ValueError("Tokenizer checksum does not match the checkpoint.")
    if tokenizer_vocab_size > model_vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds the model vocabulary.")


def _atomic_json(path: Path, value: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(
                value,
                output_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.max_validation_batches <= 0:
        raise ValueError("--max-validation-batches must be positive.")
    verify_checkpoint_checksum(arguments.checkpoint)
    device = resolve_device(arguments.device)
    model_config = _checkpoint_model_config(arguments.checkpoint)
    model = LanguageModel(model_config).to(device)
    checkpoint = load_model_checkpoint(
        arguments.checkpoint,
        model=model,
        map_location=device,
    )
    tokenizer = load_tokenizer(arguments.tokenizer)
    tokenizer_checksum = file_sha256(arguments.tokenizer)
    _validate_tokenizer_compatibility(
        checkpoint_checksum=checkpoint.tokenizer_sha256,
        tokenizer_checksum=tokenizer_checksum,
        tokenizer_vocab_size=tokenizer.get_vocab_size(
            with_added_tokens=True
        ),
        model_vocab_size=model_config.vocab_size,
    )
    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_token_id = tokenizer.token_to_id(BOS_TOKEN)
    if eos_token_id != 2 or bos_token_id != 1:
        raise ValueError("Tokenizer must use <bos>=1 and <eos>=2.")
    validation_loss: float | None = None
    validation_tokens = 0
    validation_start = time.perf_counter()
    if (arguments.validation_token_file is None) != (
        arguments.token_manifest is None
    ):
        raise ValueError(
            "Validation token file and token manifest must be supplied together."
        )
    if arguments.validation_token_file is not None:
        manifest = json.loads(
            arguments.token_manifest.read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            raise ValueError("Token manifest must contain a JSON object.")
        if manifest.get("context_length") != model_config.context_length:
            raise ValueError(
                "Token-data context length does not match the checkpoint."
            )
        if manifest.get("model_vocab_size") != model_config.vocab_size:
            raise ValueError(
                "Token-data model vocabulary does not match the checkpoint."
            )
        if manifest.get("tokenizer_sha256") != tokenizer_checksum:
            raise ValueError(
                "Token-data tokenizer checksum does not match the tokenizer."
            )
        output_checksums = manifest.get("output_checksums")
        if not isinstance(output_checksums, dict) or not isinstance(
            output_checksums.get("validation"), str
        ):
            raise ValueError(
                "Token manifest validation checksum is missing."
            )
        if (
            file_sha256(arguments.validation_token_file)
            != output_checksums["validation"]
        ):
            raise ValueError("Validation token-file checksum mismatch.")
        dtype = np.dtype(manifest["dtype"])
        dataset = MemmapTokenDataset(
            arguments.validation_token_file,
            dtype=dtype,
            context_length=model_config.context_length,
            model_vocab_size=model_config.vocab_size,
            token_offset=arguments.validation_token_offset,
            token_count=arguments.validation_token_count,
        )
        loader = create_token_dataloader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        precision_name = "bf16" if device.type == "cuda" else "fp32"
        validation_loss, validation_tokens = evaluate(
            model,
            loader,
            device=device,
            precision=resolve_precision(precision_name, device),
            max_batches=arguments.max_validation_batches,
            non_blocking=device.type == "cuda",
        )
    validation_duration_seconds = time.perf_counter() - validation_start
    source_validation: dict[str, dict[str, float | int]] = {}
    if arguments.source_validation:
        for name, token_file_text, manifest_text in arguments.source_validation:
            if name in source_validation:
                raise ValueError(f"Duplicate source-validation name {name!r}.")
            token_file = Path(token_file_text)
            source_manifest = json.loads(Path(manifest_text).read_text(encoding="utf-8"))
            if source_manifest.get("tokenizer_sha256") != tokenizer_checksum:
                raise ValueError(f"{name} tokenizer checksum mismatch.")
            checksums = source_manifest.get("output_checksums")
            if not isinstance(checksums, dict) or file_sha256(token_file) != checksums.get("validation"):
                raise ValueError(f"{name} validation checksum mismatch.")
            source_dataset = MemmapTokenDataset(
                token_file,
                dtype=np.dtype(source_manifest["dtype"]),
                context_length=model_config.context_length,
                model_vocab_size=model_config.vocab_size,
            )
            source_loader = create_token_dataloader(
                source_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            source_loss, source_tokens = evaluate(
                model,
                source_loader,
                device=device,
                precision=resolve_precision(
                    "bf16" if device.type == "cuda" else "fp32",
                    device,
                ),
                max_batches=arguments.max_validation_batches,
                non_blocking=device.type == "cuda",
            )
            source_validation[name] = {
                "loss": source_loss,
                "perplexity": perplexity_from_loss(source_loss),
                "tokens": source_tokens,
            }

    reference_text = (
        None
        if arguments.reference_jsonl is None
        else _reference_text(
            arguments.reference_jsonl,
            arguments.max_reference_documents,
        )
    )
    temperature = arguments.temperature
    if temperature == 0:
        temperature = None
    if temperature is None and (
        arguments.top_k is not None
        or arguments.top_p not in {None, 1, 1.0}
    ):
        raise ValueError("top-k/top-p filtering requires positive-temperature sampling.")
    generation_config = GenerationConfig(
        max_new_tokens=arguments.max_new_tokens,
        temperature=temperature,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        repetition_penalty=arguments.repetition_penalty,
        do_sample=temperature is not None,
        seed=arguments.seed,
    )
    samples: list[dict[str, object]] = []
    generation_start = time.perf_counter()
    for prompt_entry in _read_prompts(arguments.prompts):
        prompt, prompt_ids = _build_prompt(
            prompt_entry,
            tokenizer=tokenizer,
            context_length=model_config.context_length,
            bos_token_id=bos_token_id,
        )
        prompt_ids = prompt_ids or [bos_token_id]
        sequence = generate_sequences(
            model,
            torch.tensor([prompt_ids], dtype=torch.long, device=device),
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.token_to_id("<pad>"),
            config=generation_config,
        ).sequences[0]
        continuation_ids = list(sequence.visible_token_ids)
        text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
        continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
        )
        expected_terms = [
            str(term) for term in prompt_entry.get("expected_terms", [])
        ]
        expected_pattern = prompt_entry.get("expected_pattern")
        expected_term_search_characters = prompt_entry.get(
            "expected_term_search_characters"
        )
        sample: dict[str, object] = {
            "category": prompt_entry["category"],
            "prompt": prompt,
            "prompt_token_count": len(prompt_ids),
            "text": text,
            "continuation": continuation,
            "generated_token_count": sequence.generated_token_count,
            "finish_reason": sequence.finish_reason,
            "termination_cause": sequence.termination_cause,
            "terminating_token_id": sequence.terminating_token_id,
            "expected_terms": expected_terms,
            "expected_term_search_characters": (
                expected_term_search_characters
            ),
            "expected_term_matches": _expected_term_matches(
                expected_terms,
                continuation,
                search_characters=expected_term_search_characters,
            ),
            "expected_pattern": expected_pattern,
            "expected_pattern_match": (
                None
                if expected_pattern is None
                else re.fullmatch(
                    str(expected_pattern),
                    continuation,
                    flags=re.DOTALL,
                )
                is not None
            ),
            "quality": analyze_generated_text(continuation).to_dict(),
            "prompt_four_gram_overlap": ngram_overlap_rate(
                continuation,
                prompt,
                ngram_size=4,
            ),
            "reference_eight_gram_overlap": (
                None
                if reference_text is None
                else ngram_overlap_rate(
                    continuation,
                    reference_text,
                    ngram_size=8,
                )
            ),
        }
        samples.append(sample)
    generation_duration_seconds = time.perf_counter() - generation_start

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(arguments.checkpoint),
        "checkpoint_optimizer_step": checkpoint.training_state.optimizer_step,
        "checkpoint_run_id": checkpoint.run_id,
        "tokenizer": str(arguments.tokenizer),
        "tokenizer_sha256": tokenizer_checksum,
        "prompt_suite": str(arguments.prompts),
        "prompt_suite_sha256": file_sha256(arguments.prompts),
        "instruction_template": arguments.instruction_template,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "validation_loss": validation_loss,
        "validation_perplexity": (
            None
            if validation_loss is None
            else perplexity_from_loss(validation_loss)
        ),
        "validation_token_count": validation_tokens,
        "validation_duration_seconds": validation_duration_seconds,
        "source_validation": source_validation,
        "weighted_source_validation_loss": (
            None
            if not source_validation
            else sum(
                float(value["loss"]) * int(value["tokens"])
                for value in source_validation.values()
            )
            / sum(int(value["tokens"]) for value in source_validation.values())
        ),
        "validation_token_sha256": (
            None
            if arguments.validation_token_file is None
            else file_sha256(arguments.validation_token_file)
        ),
        "generation_config": asdict(generation_config),
        "generation_duration_seconds": generation_duration_seconds,
        "reference_document_limit": (
            None
            if arguments.reference_jsonl is None
            else arguments.max_reference_documents
        ),
        "samples": samples,
        "aggregate_generation_metrics": {
            "average_completion_length": sum(
                int(sample["generated_token_count"]) for sample in samples
            ) / len(samples),
            "premature_eos_rate": sum(
                sample["finish_reason"] == "eos" and int(sample["generated_token_count"]) < 8
                for sample in samples
            ) / len(samples),
            "maximum_length_rate": sum(
                sample["finish_reason"] == "length" for sample in samples
            ) / len(samples),
            "mean_repeated_4gram_rate": sum(
                float(sample["quality"]["repeated_ngram_rate"]) for sample in samples
            ) / len(samples),
            "maximum_longest_repeated_span": max(
                int(sample["quality"]["longest_repeated_span"]) for sample in samples
            ),
            "mean_distinct_2": sum(
                float(sample["quality"]["distinct_2"]) for sample in samples
            ) / len(samples),
            "mean_distinct_3": sum(
                float(sample["quality"]["distinct_3"]) for sample in samples
            ) / len(samples),
            "mean_prompt_copy_rate": sum(
                float(sample["prompt_four_gram_overlap"]) for sample in samples
            ) / len(samples),
            "invalid_unicode_count": sum(
                int(sample["quality"]["malformed_character_count"]) for sample in samples
            ),
            "malformed_token_count": 0,
            "human_scores": {
                "grammar": None,
                "topic_continuity": None,
                "prompt_relevance": None
            }
        },
    }
    for value in (
        report["validation_loss"],
        report["validation_perplexity"],
    ):
        if value is not None and not math.isfinite(value):
            raise FloatingPointError("Evaluation produced a non-finite metric.")
    _atomic_json(arguments.output, report, overwrite=arguments.overwrite)
    print(f"Checkpoint step: {report['checkpoint_optimizer_step']}")
    print(f"Validation loss: {validation_loss}")
    print(f"Samples: {len(samples)}")
    print(f"Report: {arguments.output}")
    return report


def main() -> None:
    arguments = build_argument_parser().parse_args()
    run(arguments)


if __name__ == "__main__":
    main()
