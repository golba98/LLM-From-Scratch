"""Audit conversational SFT records without changing data or model weights."""

from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_tokenizer.sft import ChatMessage, END_TOKEN, IGNORE_LABEL, serialize_conversation
from llm_tokenizer.tokenizer import load_tokenizer


def build_parser() -> argparse.ArgumentParser:
    """Build the chat-data audit parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--validation-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _prefix(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()[:80]


def _audit_path(path: Path, tokenizer, context_length: int) -> dict[str, object]:
    rows = accepted = rejected = 0
    total_tokens = supervised_tokens = 0
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    role_patterns: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    answer_prefixes: Counter[str] = Counter()
    duplicate_answers: Counter[str] = Counter()
    prompt_echoes = 0
    end_alignment_failures = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            rows += 1
            try:
                row = json.loads(line)
                messages = [
                    ChatMessage(item["role"], item["content"])
                    for item in row["messages"]
                ]
                role_patterns["/".join(message.role for message in messages)] += 1
                assistants = [message.content.strip() for message in messages if message.role == "assistant"]
                users = [message.content.strip() for message in messages if message.role == "user"]
                if assistants:
                    answer_prefixes[_prefix(assistants[-1])] += 1
                    duplicate_answers[hashlib.sha256(assistants[-1].encode()).hexdigest()] += 1
                    if users and _prefix(assistants[-1]).startswith(_prefix(users[-1])):
                        prompt_echoes += 1
                serialized = serialize_conversation(
                    messages,
                    tokenizer,
                    maximum_tokens=context_length,
                )
            except Exception as error:
                rejected += 1
                rejection_reasons[type(error).__name__ + ":" + str(error).split(";")[0][:100]] += 1
                continue
            accepted += 1
            token_count = len(serialized.input_ids)
            lengths.append(token_count)
            assistant_lengths.extend(
                len(tokenizer.encode(text, add_special_tokens=False).ids)
                for text in assistants
            )
            total_tokens += token_count
            supervised_tokens += serialized.supervised_token_count
            end_id = tokenizer.token_to_id(END_TOKEN)
            shifted = list(serialized.labels[1:]) + [IGNORE_LABEL]
            for index, label in enumerate(shifted):
                if label == end_id:
                    if index == 0 or serialized.input_ids[index - 1] == end_id:
                        end_alignment_failures += 1
    repeated_answer_groups = sum(count > 1 for count in duplicate_answers.values())
    return {
        "path": str(path),
        "rows": rows,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / rows if rows else 0.0,
        "total_serialized_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "supervised_fraction": supervised_tokens / total_tokens if total_tokens else 0.0,
        "length": {
            "mean": statistics.mean(lengths) if lengths else 0.0,
            "median": statistics.median(lengths) if lengths else 0.0,
            "max": max(lengths) if lengths else 0,
        },
        "assistant_token_length": {
            "mean": statistics.mean(assistant_lengths) if assistant_lengths else 0.0,
            "median": statistics.median(assistant_lengths) if assistant_lengths else 0.0,
            "max": max(assistant_lengths) if assistant_lengths else 0,
        },
        "role_patterns_top10": role_patterns.most_common(10),
        "rejection_reasons_top10": rejection_reasons.most_common(10),
        "prompt_echoes": prompt_echoes,
        "repeated_answer_hash_groups": repeated_answer_groups,
        "answer_prefixes_top10": answer_prefixes.most_common(10),
        "end_alignment_failures": end_alignment_failures,
    }


def main() -> None:
    arguments = build_parser().parse_args()
    tokenizer = load_tokenizer(arguments.tokenizer)
    report = {
        "format": "chat-v1",
        "context_length": arguments.context_length,
        "train": [
            _audit_path(path, tokenizer, arguments.context_length)
            for path in arguments.train_jsonl
        ],
        "validation": [
            _audit_path(path, tokenizer, arguments.context_length)
            for path in arguments.validation_jsonl
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
