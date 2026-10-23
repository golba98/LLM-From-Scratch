"""Build a bounded, deterministic conversational repair dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


from llm_data.data.repair_split import split_repair_records
from llm_tokenizer.sft import ChatMessage, serialize_conversation
from llm_tokenizer.tokenizer import load_tokenizer


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _identity(messages: list[ChatMessage]) -> str:
    payload = "\n".join(f"{message.role}\0{_normalise(message.content)}" for message in messages)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _anchor_records() -> list[dict[str, object]]:
    """Return small deterministic probes that guarantee behavioral coverage."""

    records: list[dict[str, object]] = []
    arithmetic = [(17, 25), (12, 30), (9, 8), (41, 1), (7, 6), (23, 19)]
    for index in range(16):
        left, right = arithmetic[index % len(arithmetic)]
        if index % 2:
            left, right = right, left
        records.append({
            "conversation_id": f"anchor-arithmetic-{index:02d}",
            "root_id": f"anchor-arithmetic-{index:02d}",
            "source": "repair-anchor",
            "category": "arithmetic",
            "messages": [
                {"role": "user", "content": f"What is {left} + {right}?"},
                {"role": "assistant", "content": f"{left} + {right} = {left + right}."},
            ],
        })
    facts = [
        ("What color is a ripe banana?", "A ripe banana is usually yellow."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("What gas do humans breathe in?", "Humans breathe in oxygen."),
    ]
    for index in range(16):
        question, answer = facts[index % len(facts)]
        records.append({
            "conversation_id": f"anchor-fact-{index:02d}",
            "root_id": f"anchor-fact-{index:02d}",
            "source": "repair-anchor",
            "category": "factual_short_answer",
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        })
    explanations = [
        ("Explain evaporation simply.", "Evaporation is when liquid water warms up and changes into an invisible gas."),
        ("Explain plants simply.", "Plants use sunlight, water, and carbon dioxide to make food and release oxygen."),
        ("Explain a fraction simply.", "A fraction shows part of a whole, such as one piece out of four equal pieces."),
        ("Explain gravity simply.", "Gravity is the pull that attracts objects toward one another, including toward Earth."),
    ]
    for index in range(16):
        question, answer = explanations[index % len(explanations)]
        records.append({
            "conversation_id": f"anchor-explanation-{index:02d}",
            "root_id": f"anchor-explanation-{index:02d}",
            "source": "repair-anchor",
            "category": "explanation",
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        })
    corrections = [
        ("You said 17 + 25 is 40. Correct yourself.", "You are right to question it. 17 + 25 = 42, not 40."),
        ("You said a week has eight days. Correct yourself.", "That was incorrect. A week has seven days."),
        ("You said the opposite of hot is warm. Correct yourself.", "That was incorrect. The opposite of hot is cold."),
        ("You said plants breathe in oxygen only. Correct yourself.", "That was incomplete. Plants use carbon dioxide and release oxygen during photosynthesis."),
    ]
    for index in range(16):
        question, answer = corrections[index % len(corrections)]
        records.append({
            "conversation_id": f"anchor-correction-{index:02d}",
            "root_id": f"anchor-correction-{index:02d}",
            "source": "repair-anchor",
            "category": "correction",
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        })
    lists = [
        ("List three primary colors.", "1. Red\n2. Blue\n3. Yellow"),
        ("List three kinds of fruit.", "1. Apple\n2. Banana\n3. Orange"),
        ("List three days of the week.", "1. Monday\n2. Tuesday\n3. Wednesday"),
        ("List three renewable energy sources.", "1. Solar power\n2. Wind power\n3. Hydropower"),
    ]
    for index in range(16):
        question, answer = lists[index % len(lists)]
        records.append({
            "conversation_id": f"anchor-list-{index:02d}",
            "root_id": f"anchor-list-{index:02d}",
            "source": "repair-anchor",
            "category": "list",
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        })
    memories = [
        ("My favorite color is green.", "What is my favorite color?", "Your favorite color is green."),
        ("Remember that I have a red bicycle.", "What do I have?", "You have a red bicycle."),
        ("Please remember that my name is Lindiwe.", "What is my name?", "Your name is Lindiwe."),
        ("Remember that tea is my favorite drink.", "What is my favorite drink?", "Your favorite drink is tea."),
    ]
    for index in range(16):
        first, question, answer = memories[index % len(memories)]
        records.append({
            "conversation_id": f"anchor-memory-{index:02d}",
            "root_id": f"anchor-memory-{index:02d}",
            "source": "repair-anchor",
            "category": "multi_turn_memory",
            "messages": [
                {"role": "user", "content": first},
                {"role": "assistant", "content": "I will remember that."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        })
    return records


def _record_from_line(line: str) -> tuple[dict[str, object], list[ChatMessage]]:
    row = json.loads(line)
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be a list")
    messages = [ChatMessage(item["role"], item["content"]) for item in raw_messages]
    return row, messages


def _valid(messages: list[ChatMessage], tokenizer, context_length: int) -> bool:
    if not messages or messages[0].role != "user" or messages[-1].role != "assistant":
        return False
    previous: str | None = None
    for message in messages:
        if message.role not in {"user", "assistant"} or not message.content.strip():
            return False
        if message.role == previous:
            return False
        previous = message.role
    last_user = next(message.content for message in reversed(messages) if message.role == "user")
    last_answer = messages[-1].content
    if _normalise(last_answer).startswith(_normalise(last_user)[:32]):
        return False
    try:
        serialize_conversation(messages, tokenizer, maximum_tokens=context_length)
    except ValueError:
        return False
    return True


def _write(path: Path, rows: list[dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-natural-records", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260910)
    arguments = parser.parse_args()
    if arguments.output_dir.exists() and any(arguments.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {arguments.output_dir}")
    tokenizer = load_tokenizer(arguments.tokenizer)
    natural: list[tuple[int, str, dict[str, object]]] = []
    rejected = Counter()
    seen: set[str] = set()
    for input_path in arguments.input:
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row, messages = _record_from_line(line)
                    if not _valid(messages, tokenizer, arguments.context_length):
                        rejected["quality_or_context"] += 1
                        continue
                    identity = _identity(messages)
                    if identity in seen:
                        rejected["exact_duplicate"] += 1
                        continue
                    seen.add(identity)
                    digest = hashlib.sha256(f"{arguments.seed}:{identity}".encode()).digest()
                    score = int.from_bytes(digest[:8], "big")
                    row["category"] = "natural"
                    natural.append((score, identity, row))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    rejected["malformed"] += 1
    natural.sort(key=lambda item: (item[0], item[1]))
    natural_rows = [row for _score, _identity_value, row in natural[: arguments.max_natural_records]]
    rows = [_row for _row in _anchor_records()] + natural_rows
    rows.sort(key=lambda row: hashlib.sha256(f"{arguments.seed}:{row['conversation_id']}".encode()).hexdigest())
    train, validation = split_repair_records(rows, seed=arguments.seed)
    output_dir = arguments.output_dir
    checksums = {
        "train": _write(output_dir / "train.jsonl", train),
        "validation": _write(output_dir / "validation.jsonl", validation),
    }
    manifest = {
        "format_version": "chat-v1-repair",
        "split_policy": "root and normalized duplicate conversation components; grouped 80/20 v2",
        "seed": arguments.seed,
        "context_length": arguments.context_length,
        "natural_records": len(natural_rows),
        "anchor_records": len(_anchor_records()),
        "train_records": len(train),
        "validation_records": len(validation),
        "categories": dict(Counter(str(row["category"]) for row in rows)),
        "rejected": dict(rejected),
        "checksums": checksums,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
