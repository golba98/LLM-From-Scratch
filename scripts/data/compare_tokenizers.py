"""Run a deterministic 8K-versus-16K tokenizer bake-off."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time

import yaml

from llm_data.data.io import TextDocument, load_documents, write_jsonl
from llm_tokenizer.tokenizer import inspect_tokenizer, load_tokenizer, train_tokenizer


def _stratified_sample(path: Path, count: int, seed: int) -> list[TextDocument]:
    """Select the smallest stable hashes without depending on row order."""

    heap: list[tuple[int, int, TextDocument]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{path}:{line_number}: invalid cleaned JSONL row.")
            source = str(record.get("source", path.stem))
            document_id = str(record.get("document_id", ""))
            identity = document_id or hashlib.sha256(
                record["text"].encode("utf-8")
            ).hexdigest()
            score = int.from_bytes(
                hashlib.sha256(f"{seed}\0{source}\0{identity}".encode()).digest(),
                "big",
            )
            document = TextDocument(record["text"], source, identity)
            candidate = (-score, -line_number, document)
            if len(heap) < count:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    return [item[2] for item in sorted(heap, key=lambda value: -value[0])]


def _timing(
    tokenizer,
    texts: list[str],
    rounds: int,
    *,
    batch_size: int = 256,
) -> dict[str, float]:
    """Measure throughput without retaining every encoding in memory."""

    encode_seconds = 0.0
    decode_seconds = 0.0
    total_bytes = 0
    round_trip_failures = 0
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        encoded = []
        start = time.perf_counter()
        for _ in range(rounds):
            encoded = tokenizer.encode_batch(batch, add_special_tokens=False)
        encode_seconds += time.perf_counter() - start
        token_ids = [item.ids for item in encoded]
        decoded = []
        start = time.perf_counter()
        for _ in range(rounds):
            decoded = tokenizer.decode_batch(token_ids)
        decode_seconds += time.perf_counter() - start
        total_bytes += sum(len(text.encode("utf-8")) for text in batch) * rounds
        round_trip_failures += sum(
            original != restored
            for original, restored in zip(batch, decoded, strict=True)
        )
    return {
        "encoding_seconds": encode_seconds,
        "decoding_seconds": decode_seconds,
        "encoding_bytes_per_second": total_bytes / encode_seconds,
        "decoding_bytes_per_second": total_bytes / decode_seconds,
        "round_trip_failures": round_trip_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tokenizer_bakeoff.yaml"))
    parser.add_argument("--fineweb-jsonl", type=Path, required=True)
    parser.add_argument("--wikipedia-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--projected-clean-corpus-bytes", type=int)
    parser.add_argument("--measured-training-tokens-per-second", type=float)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing deterministic sample and candidate tokenizers.",
    )
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    sample_count = int(config["documents_per_source"])
    seed = int(config["seed"])
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = arguments.output_dir / "stratified_sample.jsonl"
    if arguments.reuse_existing and sample_path.is_file():
        sample_documents = load_documents([sample_path])
        source_samples = {
            "fineweb_edu": [
                item for item in sample_documents if "fineweb-edu" in item.source.casefold()
            ],
            "wikipedia": [
                item for item in sample_documents if "wikipedia" in item.source.casefold()
            ],
        }
        if any(len(items) != sample_count for items in source_samples.values()):
            raise ValueError("Existing stratified sample has unexpected source counts.")
    else:
        source_samples = {
            "fineweb_edu": _stratified_sample(arguments.fineweb_jsonl, sample_count, seed),
            "wikipedia": _stratified_sample(arguments.wikipedia_jsonl, sample_count, seed),
        }
        sample_documents = [
            document
            for source in sorted(source_samples)
            for document in source_samples[source]
        ]
        write_jsonl(sample_path, sample_documents)
    report: dict[str, object] = {
        "config": config,
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "sample_documents_by_source": {
            source: len(documents) for source, documents in source_samples.items()
        },
        "candidates": [],
        "production_selection": config.get("production_selection"),
    }
    baseline_tokens: int | None = None
    for vocab_size in config["candidate_vocab_sizes"]:
        candidate_dir = arguments.output_dir / f"vocab-{vocab_size}"
        tokenizer_path = candidate_dir / "tokenizer.json"
        if arguments.reuse_existing and tokenizer_path.is_file():
            tokenizer = load_tokenizer(tokenizer_path)
        else:
            result = train_tokenizer(
                [sample_path],
                output_dir=candidate_dir,
                vocab_size=int(vocab_size),
                min_frequency=int(config["min_frequency"]),
            )
            tokenizer_path = result.tokenizer_path
            tokenizer = load_tokenizer(tokenizer_path)
        texts = [document.text for document in sample_documents]
        overall = inspect_tokenizer(tokenizer, texts)
        by_source = {
            source: inspect_tokenizer(tokenizer, [item.text for item in documents]).to_dict()
            for source, documents in source_samples.items()
        }
        baseline_tokens = overall.total_tokens if baseline_tokens is None else baseline_tokens
        projected_tokens = None
        projected_runtime = None
        if arguments.projected_clean_corpus_bytes:
            bytes_per_token = overall.total_utf8_bytes / overall.total_tokens
            projected_tokens = round(arguments.projected_clean_corpus_bytes / bytes_per_token)
            if arguments.measured_training_tokens_per_second:
                projected_runtime = projected_tokens / arguments.measured_training_tokens_per_second
        embedding_parameters = int(vocab_size) * int(config["model_hidden_size"])
        candidate = {
            "vocab_size": int(vocab_size),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            "overall": {
                **overall.to_dict(),
                "bytes_per_token": overall.total_utf8_bytes / overall.total_tokens,
                "tokens_per_document": overall.total_tokens / overall.document_count,
                "sequence_length_inflation_vs_8192": overall.total_tokens / baseline_tokens,
            },
            "by_source": by_source,
            "timing": _timing(tokenizer, texts, int(config["timing_rounds"])),
            "projected_full_corpus_tokens": projected_tokens,
            "projected_training_seconds": projected_runtime,
            "embedding_parameters": embedding_parameters,
            "embedding_parameter_increase_vs_8192": (
                int(vocab_size) - 8192
            ) * int(config["model_hidden_size"]),
        }
        report["candidates"].append(candidate)  # type: ignore[union-attr]
    output = arguments.output_dir / "tokenizer_comparison.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
