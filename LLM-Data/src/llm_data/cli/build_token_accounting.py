"""Build source token accounting from completed per-source token indexes."""

import argparse
import hashlib
import json
from pathlib import Path


def _index_counts(path: Path) -> list[int]:
    counts: list[int] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            record = json.loads(line)
            counts.append(int(record["content_token_count"]))
    return counts


def _percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, int(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    sources: dict[str, dict[str, object]] = {}
    for source in ("fineweb_edu", "wikipedia"):
        source_dir = arguments.tokenized_root / source
        manifest = json.loads(
            (source_dir / "token_data_manifest.json").read_text(encoding="utf-8")
        )
        counts = _index_counts(source_dir / "train_index.json")
        counts.sort()
        documents = len(counts)
        content_tokens = sum(counts)
        stored_tokens = int(manifest["train_token_count"])
        assert stored_tokens == content_tokens + documents
        sources[source] = {
            "documents": documents,
            "content_tokens": content_tokens,
            "stored_tokens": stored_tokens,
            "mean_document_tokens": content_tokens / documents,
            "p50_document_tokens": _percentile(counts, 0.50),
            "p95_document_tokens": _percentile(counts, 0.95),
            "p99_document_tokens": _percentile(counts, 0.99),
            "truncated_tokens": 0,
            "discarded_trailing_tokens": int(manifest["trailing_tokens"]["train"]),
        }

    total_tokens = sum(int(item["stored_tokens"]) for item in sources.values())
    requested = {
        source: int(item["stored_tokens"]) / total_tokens
        for source, item in sorted(sources.items())
    }
    sampler_payload = {
        "seed": arguments.seed,
        "requested_percentages": requested,
        "source_tokens": {
            source: item["stored_tokens"] for source, item in sorted(sources.items())
        },
        "policy": "stable_hash_token_deficit_no_oversampling",
    }
    sampler_state = hashlib.sha256(
        json.dumps(sampler_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": 1,
        "seed": arguments.seed,
        "tokenizer_sha256": hashlib.sha256(arguments.tokenizer.read_bytes()).hexdigest(),
        "sources": sources,
        "mixture": {
            "requested_percentages": requested,
            "achieved_percentages": requested,
            "documents_by_source": {
                source: item["documents"] for source, item in sorted(sources.items())
            },
            "tokens_by_source": {
                source: item["stored_tokens"] for source, item in sorted(sources.items())
            },
            "requested_total_tokens": total_tokens,
            "achieved_total_tokens": total_tokens,
            "repeated_documents": 0,
            "exhausted_sources": sorted(sources),
            "sampler_state_sha256": sampler_state,
        },
        "packing": {
            "context_length": 2048,
            "eos_tokens_between_documents": 1,
            "trailing_partial_sequence": "discard",
            "padding": "none",
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
