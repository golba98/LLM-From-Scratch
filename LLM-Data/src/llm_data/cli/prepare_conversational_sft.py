"""Prepare role-aware UltraChat and OASST1 conversation paths for SFT."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from llm_data.data.conversation import (
    ConversationCandidate,
    filter_conversations,
    split_conversations_by_root,
)
from llm_data.data.cleaning import clean_text
from llm_tokenizer.sft import ChatMessage


def _write_split(path: Path, conversations: tuple[ConversationCandidate, ...]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for candidate in conversations:
            record = {
                "conversation_id": candidate.conversation_id,
                "root_id": candidate.root_id,
                "source": candidate.source,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in candidate.messages
                ],
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _ultrachat(path: Path, progress_interval: int) -> list[ConversationCandidate]:
    candidates: list[ConversationCandidate] = []
    rows = 0
    for parquet_path in sorted(path.glob("data/train_sft-*.parquet")):
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                rows += 1
                messages = []
                for item in row.get("messages") or []:
                    if not isinstance(item, dict):
                        continue
                    role = item.get("role")
                    content = item.get("content")
                    if role in {"system", "user", "assistant"} and isinstance(content, str):
                        cleaned = clean_text(content)
                        if cleaned is not None:
                            messages.append(ChatMessage(role, cleaned))
                candidates.append(
                    ConversationCandidate(
                        conversation_id=str(row.get("prompt_id", rows)),
                        root_id=str(row.get("prompt_id", rows)),
                        messages=tuple(messages),
                        source="ultrachat",
                    )
                )
                if rows % progress_interval == 0:
                    print(json.dumps({"source": "ultrachat", "rows": rows}), flush=True)
    return candidates


def _oasst(path: Path, progress_interval: int) -> list[ConversationCandidate]:
    nodes: dict[str, dict[str, object]] = {}
    children: dict[str, list[str]] = defaultdict(list)
    rows = 0
    parquet_paths = sorted(path.glob("data/train-*.parquet"))
    for parquet_path in parquet_paths:
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                rows += 1
                message_id = row.get("message_id")
                if not isinstance(message_id, str):
                    continue
                nodes[message_id] = row
                parent_id = row.get("parent_id")
                if isinstance(parent_id, str):
                    children[parent_id].append(message_id)
                if rows % progress_interval == 0:
                    print(json.dumps({"source": "oasst1", "rows": rows}), flush=True)

    candidates: list[ConversationCandidate] = []
    for root_id, root in sorted(nodes.items()):
        if root.get("parent_id") is not None:
            continue
        tree_id = str(root.get("message_tree_id") or root_id)
        stack: list[tuple[str, list[str]]] = [(root_id, [root_id])]
        while stack:
            current_id, path_ids = stack.pop()
            child_ids = sorted(children.get(current_id, []), reverse=True)
            if child_ids:
                for child_id in child_ids:
                    stack.append((child_id, [*path_ids, child_id]))
                continue
            messages: list[ChatMessage] = []
            for message_id in path_ids:
                row = nodes[message_id]
                role = "user" if row.get("role") == "prompter" else row.get("role")
                content = row.get("text")
                if role not in {"user", "assistant", "system"} or not isinstance(content, str):
                    messages = []
                    break
                cleaned = clean_text(content)
                if cleaned is None:
                    messages = []
                    break
                messages.append(ChatMessage(role, cleaned))
            leaf = nodes[current_id]
            candidates.append(
                ConversationCandidate(
                    conversation_id=current_id,
                    root_id=tree_id,
                    messages=tuple(messages),
                    source="oasst1",
                    review_count=int(leaf.get("review_count") or 0),
                    review_result=leaf.get("review_result")
                    if isinstance(leaf.get("review_result"), bool)
                    else None,
                    rank=leaf.get("rank") if isinstance(leaf.get("rank"), int) else None,
                    deleted=bool(leaf.get("deleted")),
                )
            )
    return candidates


def _prepare(
    *,
    source: str,
    candidates: list[ConversationCandidate],
    output_dir: Path,
    validation_ratio: float,
    seed: int,
) -> dict[str, object]:
    accepted, report = filter_conversations(candidates)
    splits = split_conversations_by_root(
        accepted,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    checksums = {
        split: _write_split(output_dir / source / f"{split}.jsonl", items)
        for split, items in splits.items()
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "format_version": "chat-v1-prepared",
        "source": source,
        "seed": seed,
        "validation_ratio": validation_ratio,
        "input_count": report.input_count,
        "accepted_count": report.accepted_count,
        "accepted_by_split": {split: len(items) for split, items in splits.items()},
        "rejected_by_reason": report.rejected_by_reason,
        "exact_duplicate_groups": report.exact_duplicate_groups,
        "output_checksums": checksums,
    }
    manifest_path = output_dir / source / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ultrachat-root", type=Path, required=True)
    parser.add_argument("--oasst-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "ultrachat": _prepare(
            source="ultrachat",
            candidates=_ultrachat(arguments.ultrachat_root, arguments.progress_interval),
            output_dir=arguments.output_dir,
            validation_ratio=arguments.validation_ratio,
            seed=arguments.seed,
        ),
        "oasst1": _prepare(
            source="oasst1",
            candidates=_oasst(arguments.oasst_root, arguments.progress_interval),
            output_dir=arguments.output_dir,
            validation_ratio=arguments.validation_ratio,
            seed=arguments.seed,
        ),
    }
    report_path = arguments.output_dir / "preparation_manifest.json"
    report_path.write_text(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
