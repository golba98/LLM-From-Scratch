"""Build a deterministic mixed training stream from source token artifacts."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from llm_data.token_data import count_packed_examples, file_sha256, production_packing_policy


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    return Path(name)


def _read_index(path: Path, source: str, seed: int) -> list[tuple[bytes, str, int, int, int]]:
    records: list[tuple[bytes, str, int, int, int]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip().rstrip(",")
            if not stripped or stripped in {"[", "]"}:
                continue
            item = json.loads(stripped)
            document_id = str(item["document_id"])
            identity = f"{seed}\0{source}\0{document_id}".encode()
            order_key = hashlib.sha256(identity).digest()
            records.append(
                (
                    order_key,
                    document_id,
                    int(item["token_start"]),
                    int(item["token_end"]),
                    int(item["document_ordinal"]),
                )
            )
    records.sort(key=lambda item: item[0])
    return records


def _sha256(path: Path) -> str:
    return file_sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument("--accounting-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--progress-interval", type=int, default=100_000)
    arguments = parser.parse_args()

    accounting = json.loads(arguments.accounting_report.read_text(encoding="utf-8"))
    requested = {
        source: float(value)
        for source, value in accounting["mixture"]["requested_percentages"].items()
    }
    total_budget = int(accounting["mixture"]["requested_total_tokens"])
    source_records = {
        source: _read_index(
            arguments.tokenized_root / source / "train_index.json",
            source,
            arguments.seed,
        )
        for source in sorted(requested)
    }
    source_data = {
        source: np.memmap(
            arguments.tokenized_root / source / "train.bin",
            dtype=np.uint16,
            mode="r",
        )
        for source in source_records
    }
    destination = arguments.output_dir
    final_paths = {
        "train": destination / "train.bin",
        "index": destination / "train_index.json",
        "manifest": destination / "token_data_manifest.json",
    }
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError("Mixed token output already exists; refusing overwrite.")
    temporary = {name: _temporary(path) for name, path in final_paths.items()}
    cursors = {source: 0 for source in source_records}
    contributed = {source: 0 for source in source_records}
    document_counts = {source: 0 for source in source_records}
    truncated = {source: 0 for source in source_records}
    started = time.monotonic()
    total_documents = 0
    try:
        with temporary["train"].open("wb") as output_file, temporary["index"].open(
            "w", encoding="utf-8"
        ) as index_file:
            index_file.write("[\n")
            while sum(contributed.values()) < total_budget:
                available = [
                    source
                    for source in source_records
                    if cursors[source] < len(source_records[source])
                ]
                if not available:
                    break
                processed = sum(contributed.values())
                source = max(
                    available,
                    key=lambda name: (
                        requested[name] * max(processed, 1) - contributed[name],
                        requested[name],
                        name,
                    ),
                )
                _, document_id, token_start, token_end, ordinal = source_records[source][
                    cursors[source]
                ]
                cursors[source] += 1
                available_tokens = token_end - token_start
                contribution = min(available_tokens, total_budget - processed)
                chunk = source_data[source][token_start : token_start + contribution]
                np.asarray(chunk, dtype=np.uint16).tofile(output_file)
                contributed[source] += contribution
                truncated[source] += available_tokens - contribution
                document_counts[source] += 1
                entry = {
                    "document_ordinal": total_documents,
                    "source": source,
                    "source_document_ordinal": ordinal,
                    "document_id": document_id,
                    "token_start": processed,
                    "token_end": processed + contribution,
                    "total_stored_token_count": contribution,
                    "truncated_tokens": available_tokens - contribution,
                }
                if total_documents:
                    index_file.write(",\n")
                index_file.write(json.dumps(entry, sort_keys=True))
                total_documents += 1
                if total_documents % arguments.progress_interval == 0:
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            {
                                "documents": total_documents,
                                "tokens": processed + contribution,
                                "target_tokens": total_budget,
                                "percent": round((processed + contribution) * 100 / total_budget, 2),
                                "documents_per_second": round(total_documents / elapsed, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            index_file.write("\n]\n")

        achieved = sum(contributed.values())
        if achieved != total_budget:
            raise RuntimeError(f"Mixed stream stopped at {achieved} of {total_budget} tokens.")
        expected_bytes = achieved * np.dtype(np.uint16).itemsize
        if temporary["train"].stat().st_size != expected_bytes:
            raise RuntimeError("Mixed stream byte size does not match token count.")
        complete_sequences, trailing = count_packed_examples(
            achieved,
            context_length=arguments.context_length,
        )
        achieved_percentages = {
            source: contributed[source] / achieved for source in source_records
        }
        sampler_payload = {
            "seed": arguments.seed,
            "requested_percentages": requested,
            "achieved_percentages": achieved_percentages,
            "total_tokens": achieved,
            "document_counts": document_counts,
        }
        sampler_state = hashlib.sha256(
            json.dumps(sampler_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0-mixed",
            "tokenizer_sha256": accounting["tokenizer_sha256"],
            "tokenizer_actual_vocab_size": 16384,
            "model_vocab_size": 16384,
            "dtype": "uint16",
            "context_length": arguments.context_length,
            "packing_policy": production_packing_policy(),
            "source_manifests": {
                source: file_sha256(
                    arguments.tokenized_root / source / "token_data_manifest.json"
                )
                for source in source_records
            },
            "source_percentages": requested,
            "seed": arguments.seed,
            "sampler_state_sha256": sampler_state,
            "train_document_count": total_documents,
            "train_token_count": achieved,
            "complete_sequences": complete_sequences,
            "trailing_tokens": trailing,
            "documents_by_source": document_counts,
            "tokens_by_source": contributed,
            "truncated_tokens_by_source": truncated,
            "repeated_documents": 0,
            "exhausted_sources": sorted(source_records),
            "output_paths": {name: path.name for name, path in final_paths.items()},
            "output_checksums": {
                "train": _sha256(temporary["train"]),
                "index": _sha256(temporary["index"]),
            },
        }
        temporary["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for name in ("train", "index", "manifest"):
            os.replace(temporary[name], final_paths[name])
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
