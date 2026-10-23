"""Quality-controlled, provenance-preserving conversational corpus preparation."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from llm_tokenizer.sft import ChatMessage, serialize_conversation
from llm_tokenizer.tokenizer import SPECIAL_TOKENS


def normalized(text: str) -> str:
    """Normalize punctuation and whitespace for duplicate grouping only."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def prepare_general_chat(inputs: list[Path], output: Path, tokenizer,
                         *, context: int = 2048, seed: int = 42) -> dict:
    """Keep complete turns, root/duplicate groups and deterministic held-out splits."""
    output.mkdir(parents=True, exist_ok=False)
    rejected = Counter()
    records = []
    parents = []
    owners = {}
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def join(a, b):
        a, b = root(a), root(b)
        parents[max(a, b)] = min(a, b)
    for path in inputs:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if not isinstance(row, dict):
                    rejected['malformed'] += 1
                    continue
                if any(not isinstance(row.get(key), str) or not row[key] for key in ('source', 'root_id', 'conversation_id')):
                    rejected['missing_identity'] += 1
                    continue
                messages = row.get('messages', [])
                if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get('content'), str) or m.get('role') not in ('user', 'assistant', 'system') for m in messages):
                    rejected['malformed'] += 1
                    continue
                if any(any(token in m['content'] for token in SPECIAL_TOKENS) for m in messages):
                    rejected['reserved_control_token'] += 1
                    continue
                if any(m['role'] == 'assistant' and len(re.findall(r'\w+', m['content'])) > 20
                       and len(set(re.findall(r'\w+', m['content'].lower()))) / len(re.findall(r'\w+', m['content'])) < 0.15
                       for m in messages):
                    rejected['severe_repetition'] += 1
                    continue
                try:
                    serialized = serialize_conversation([ChatMessage(m['role'], m['content']) for m in messages],
                                                         tokenizer, maximum_tokens=context)
                except (ValueError, KeyError):
                    rejected['roles_or_context'] += 1
                    continue
                canonical = [(m['role'], normalized(m['content'])) for m in messages]
                digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
                full_key = ('conversation', digest)
                if full_key in owners:
                    rejected['normalized_duplicate_conversation'] += 1
                    continue
                i = len(records)
                parents.append(i)
                source = row.get('source')
                if source not in ('ultrachat', 'oasst1'):
                    rejected['unknown_source'] += 1
                    parents.pop()
                    continue
                row = {**row, 'source_path': str(path), 'historical_sft_exposure': path.name == 'train.jsonl',
                       'content_sha256': digest, 'tokens': len(serialized.input_ids),
                       'assistant_tokens': serialized.supervised_token_count}
                records.append(row)
                keys = [full_key, ('root', source, row['root_id'])]
                # Shared substantial prompts/answers join groups before any split.
                for role, content in canonical:
                    if len(content.split()) >= 6:
                        keys.append((role, hashlib.sha256(content.encode()).hexdigest()))
                for key in keys:
                    if key in owners:
                        join(i, owners[key])
                    else:
                        owners[key] = i
    clusters = {}
    for i, row in enumerate(records):
        r = root(i)
        clusters.setdefault(r, []).append(i)
    groups = {}
    for indices in clusters.values():
        key = min(records[i]['content_sha256'] for i in indices)
        value = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) / 2**64
        split = 'train' if value < 0.9 else ('validation' if value < 0.95 else 'test')
        for i in indices:
            groups[i] = (split, key)
    counts = Counter()
    checksums = {}
    for split in ('train', 'validation', 'test'):
        target = output / f'{split}.jsonl'
        with target.open('w') as handle:
            for i, row in enumerate(records):
                if groups[i][0] == split:
                    handle.write(json.dumps({**row, 'duplicate_group': groups[i][1]}, ensure_ascii=False) + '\n')
                    counts[f'{split}/{row["source"]}'] += 1
        with target.open('rb') as handle:
            checksums[split] = hashlib.file_digest(handle, 'sha256').hexdigest()
    if not all(any(key.startswith(split + '/') for key in counts) for split in ('train', 'validation', 'test')):
        raise ValueError('Corpus lacks a nonempty split; do not train.')
    report = dict(format='general-chat-v2', seed=seed, context_length=context, counts=dict(counts),
                  rejected=dict(rejected), groups=len(clusters), largest_group=max(map(len, clusters.values())),
                  output_sha256=checksums, input_paths=[str(p) for p in inputs],
                  split_policy='90/5/5 root and normalized substantial message duplicate components',
                  limitations=['No semantic paraphrase deduplication or factual correctness certification.',
                               'Earlier SFT models may have seen these records; exposure is retained per record.'])
    (output / 'dataset_manifest.json').write_text(json.dumps(report, indent=2))
    return report
