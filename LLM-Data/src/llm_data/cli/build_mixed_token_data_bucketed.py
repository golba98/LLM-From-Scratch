"""Build the mixed stream with sequential source scans and hash buckets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from llm_data.token_data import count_packed_examples, file_sha256, production_packing_policy


BUCKETS = 256


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    return Path(name)


def _read_index(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip().rstrip(",")
            if not line or line in {"[", "]"}:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenized-root", type=Path, required=True)
    parser.add_argument("--accounting-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, default=2048)
    arguments = parser.parse_args()
    accounting = json.loads(arguments.accounting_report.read_text())
    requested = {k: float(v) for k, v in accounting["mixture"]["requested_percentages"].items()}
    sources = tuple(sorted(requested))
    total_budget = int(accounting["mixture"]["requested_total_tokens"])
    bucket_root = arguments.output_dir.parent / ".base-v1-mix-buckets"
    if bucket_root.exists():
        raise FileExistsError(f"Temporary bucket directory already exists: {bucket_root}")
    bucket_root.mkdir(parents=True)
    output_paths = {
        "train": arguments.output_dir / "train.bin",
        "index": arguments.output_dir / "train_index.json",
        "manifest": arguments.output_dir / "token_data_manifest.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("Mixed output already exists; refusing overwrite.")
    temporary = {name: _tmp(path) for name, path in output_paths.items()}
    try:
        bucket_handles = {}
        meta_handles = {}
        for source in sources:
            bucket_handles[source] = [
                (bucket_root / f"{source}-{bucket:03d}.bin").open("wb")
                for bucket in range(BUCKETS)
            ]
            meta_handles[source] = [
                (bucket_root / f"{source}-{bucket:03d}.jsonl").open("w")
                for bucket in range(BUCKETS)
            ]
        for source in sources:
            binary = np.memmap(
                arguments.tokenized_root / source / "train.bin", dtype=np.uint16, mode="r"
            )
            for item in _read_index(arguments.tokenized_root / source / "train_index.json"):
                document_id = str(item["document_id"])
                key = hashlib.sha256(
                    f"{arguments.seed}\0{source}\0{document_id}".encode()
                ).digest()
                bucket = key[0]
                chunk = binary[int(item["token_start"]):int(item["token_end"])]
                local_start = bucket_handles[source][bucket].tell() // 2
                np.asarray(chunk, dtype=np.uint16).tofile(bucket_handles[source][bucket])
                local_end = local_start + len(chunk)
                meta_handles[source][bucket].write(
                    json.dumps(
                        {
                            "key": key.hex(),
                            "document_id": document_id,
                            "ordinal": int(item["document_ordinal"]),
                            "start": local_start,
                            "end": local_end,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            del binary
        for source in sources:
            for handle in (*bucket_handles[source], *meta_handles[source]):
                handle.close()

        contributed = {source: 0 for source in sources}
        document_counts = {source: 0 for source in sources}
        truncated = {source: 0 for source in sources}
        total_documents = 0
        with temporary["train"].open("wb") as output, temporary["index"].open("w") as index:
            index.write("[\n")
            for bucket in range(BUCKETS):
                records = {}
                data = {}
                for source in sources:
                    meta_path = bucket_root / f"{source}-{bucket:03d}.jsonl"
                    records[source] = sorted(
                        [json.loads(line) for line in meta_path.read_text().splitlines()],
                        key=lambda item: item["key"],
                    )
                    data[source] = np.memmap(
                        bucket_root / f"{source}-{bucket:03d}.bin",
                        dtype=np.uint16,
                        mode="r",
                    )
                cursors = {source: 0 for source in sources}
                while any(cursors[source] < len(records[source]) for source in sources):
                    available = [
                        source for source in sources if cursors[source] < len(records[source])
                    ]
                    processed = sum(contributed.values())
                    source = max(
                        available,
                        key=lambda name: (
                            requested[name] * max(processed, 1) - contributed[name],
                            requested[name],
                            name,
                        ),
                    )
                    item = records[source][cursors[source]]
                    cursors[source] += 1
                    available_tokens = int(item["end"]) - int(item["start"])
                    contribution = min(available_tokens, total_budget - processed)
                    chunk = data[source][int(item["start"]):int(item["start"]) + contribution]
                    np.asarray(chunk, dtype=np.uint16).tofile(output)
                    contributed[source] += contribution
                    truncated[source] += available_tokens - contribution
                    document_counts[source] += 1
                    entry = {
                        "document_ordinal": total_documents,
                        "source": source,
                        "source_document_ordinal": int(item["ordinal"]),
                        "document_id": item["document_id"],
                        "token_start": processed,
                        "token_end": processed + contribution,
                        "total_stored_token_count": contribution,
                        "truncated_tokens": available_tokens - contribution,
                    }
                    if total_documents:
                        index.write(",\n")
                    index.write(json.dumps(entry, sort_keys=True))
                    total_documents += 1
                for mmap in data.values():
                    del mmap
                print(json.dumps({"bucket": bucket + 1, "buckets": BUCKETS, "documents": total_documents}), flush=True)

            index.write("\n]\n")
        if sum(contributed.values()) != total_budget:
            raise RuntimeError("Mixed stream did not reach the accounting token budget.")
        complete_sequences, trailing = count_packed_examples(
            total_budget, context_length=arguments.context_length
        )
        achieved = {source: contributed[source] / total_budget for source in sources}
        sampler_payload = {
            "seed": arguments.seed,
            "requested_percentages": requested,
            "achieved_percentages": achieved,
            "documents_by_source": document_counts,
            "total_tokens": total_budget,
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
            "source_percentages": requested,
            "seed": arguments.seed,
            "sampler_state_sha256": sampler_state,
            "train_document_count": total_documents,
            "train_token_count": total_budget,
            "complete_sequences": complete_sequences,
            "trailing_tokens": trailing,
            "documents_by_source": document_counts,
            "tokens_by_source": contributed,
            "truncated_tokens_by_source": truncated,
            "repeated_documents": 0,
            "exhausted_sources": list(sources),
            "output_paths": {name: path.name for name, path in output_paths.items()},
            "output_checksums": {
                "train": file_sha256(temporary["train"]),
                "index": file_sha256(temporary["index"]),
            },
        }
        temporary["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for name in ("train", "index", "manifest"):
            os.replace(temporary[name], output_paths[name])
        print(json.dumps(manifest, indent=2, sort_keys=True))
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        shutil.rmtree(bucket_root, ignore_errors=True)


if __name__ == "__main__":
    main()
