"""Read-only conversational indexing, causal batches and reproducible ordering."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from llm_tokenizer.sft import ChatMessage, IGNORE_LABEL, serialize_conversation


@dataclass(frozen=True)
class ConversationIndex:
    """Small immutable metadata; conversation contents stay on disk."""

    offset: int
    identity: str
    group: str
    root: tuple[str, str]
    source: str
    tokens: int
    assistant_tokens: int
    historical_exposure: bool
    content_sha256: str


class IndexedChatDataset(Dataset):
    """Read existing records lazily without materializing padded tensors."""

    def __init__(self, path: Path, tokenizer: Any, *, context: int = 2048,
                 cache_bytes: int = 64 * 1024**2, verify_tokens: bool = False) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context = context
        self.cache_bytes = cache_bytes
        self.index: list[ConversationIndex] = []
        self._cache: OrderedDict[int, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._cached_bytes = 0
        self.audit = Counter()
        digest = hashlib.sha256()
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                row = json.loads(line)
                required = ('conversation_id', 'duplicate_group', 'root_id', 'source',
                            'tokens', 'assistant_tokens', 'content_sha256')
                if any(key not in row for key in required):
                    raise ValueError(f'Missing record identity at byte {offset}: {self.path}')
                tokens, assistant = row['tokens'], row['assistant_tokens']
                if not isinstance(tokens, int) or not 1 < tokens <= context:
                    raise ValueError(f'Invalid complete sequence length at byte {offset}')
                if not isinstance(assistant, int) or not 0 < assistant < tokens:
                    raise ValueError(f'Invalid supervision count at byte {offset}')
                entry = ConversationIndex(offset, row['conversation_id'], row['duplicate_group'],
                    (row['source'], row['root_id']), row['source'], tokens, assistant,
                    bool(row.get('historical_sft_exposure', True)), row['content_sha256'])
                self.index.append(entry)
                self.audit['examples'] += 1
                self.audit['context_tokens'] += tokens
                self.audit['assistant_tokens'] += assistant
                self.audit[f'examples/{entry.source}'] += 1
                self.audit[f'assistant_tokens/{entry.source}'] += assistant
                self.audit['historically_exposed'] += entry.historical_exposure
                if verify_tokens:
                    self._serialize(row, entry)
        self.sha256 = digest.hexdigest()
        if not self.index:
            raise ValueError('Conversational dataset is empty.')

    def _serialize(self, row: dict, entry: ConversationIndex) -> tuple[torch.Tensor, torch.Tensor]:
        from llm_tokenizer.tokenizer import SPECIAL_TOKENS
        from llm_data.data.general_chat import normalized
        messages = row['messages']
        if any(any(token in message['content'] for token in SPECIAL_TOKENS) for message in messages):
            raise ValueError(f'Reserved marker injection in {entry.identity}')
        canonical = [(message['role'], normalized(message['content'])) for message in messages]
        digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
        if digest != entry.content_sha256:
            raise ValueError(f'Conversation fingerprint mismatch: {entry.identity}')
        record = serialize_conversation([ChatMessage(m['role'], m['content']) for m in messages],
                                        self.tokenizer, maximum_tokens=self.context)
        ids = torch.tensor(record.input_ids, dtype=torch.long)
        labels = torch.tensor((*record.labels[1:], IGNORE_LABEL), dtype=torch.long)
        if len(ids) != entry.tokens or int((labels != IGNORE_LABEL).sum()) != entry.assistant_tokens:
            raise ValueError(f'Tokenizer/count mismatch: {entry.identity}')
        return ids, labels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        entry = self.index[item]
        if item in self._cache:
            ids, labels = self._cache.pop(item)
            self._cache[item] = (ids, labels)
        else:
            with self.path.open('rb') as stream:
                stream.seek(entry.offset)
                ids, labels = self._serialize(json.loads(stream.readline()), entry)
            size = (ids.numel() + labels.numel()) * ids.element_size()
            while self._cache and self._cached_bytes + size > self.cache_bytes:
                _, old = self._cache.popitem(last=False)
                self._cached_bytes -= sum(t.numel() * t.element_size() for t in old)
            if size <= self.cache_bytes:
                self._cache[item] = (ids, labels)
                self._cached_bytes += size
        return {'input_ids': ids, 'labels': labels, 'identity': entry.identity,
                'source': entry.source, 'index': item}


def collate_chat(rows: list[dict[str, Any]], *, pad_id: int = 0,
                 fixed_length: int | None = None) -> dict[str, Any]:
    """Right-pad complete conversations; padding never bears loss."""
    maximum = fixed_length or max(len(row['input_ids']) for row in rows)
    if maximum < max(len(row['input_ids']) for row in rows):
        raise ValueError('Collation cannot truncate conversations.')
    ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, IGNORE_LABEL)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        length = len(row['input_ids'])
        ids[i, :length] = row['input_ids']
        labels[i, :length] = row['labels']
        mask[i, :length] = True
    return {'input_ids': ids, 'labels': labels, 'attention_mask': mask,
            'identities': [row['identity'] for row in rows],
            'sources': [row['source'] for row in rows],
            'indices': [row['index'] for row in rows]}


class ConversationSampler(Sampler[int]):
    """Seeded no-replacement ordering; checkpoint cursor counts consumed batches."""

    def __init__(self, dataset: IndexedChatDataset, *, seed: int,
                 balanced: bool = False, bucket_size: int = 128) -> None:
        self.dataset, self.seed = dataset, seed
        self.balanced, self.bucket_size = balanced, bucket_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.dataset)))
        rng.shuffle(order)
        if self.balanced:
            sources: dict[str, list[int]] = {}
            for i in order:
                sources.setdefault(self.dataset.index[i].source, []).append(i)
            order = []
            while any(sources.values()):
                for source in sorted(sources):
                    if sources[source]:
                        order.append(sources[source].pop())
        # Bucket locally, preserving corpus-wide shuffling and source coverage.
        for start in range(0, len(order), self.bucket_size):
            bucket = order[start:start + self.bucket_size]
            bucket.sort(key=lambda i: self.dataset.index[i].tokens)
            yield from bucket


def verify_split_isolation(datasets: dict[str, IndexedChatDataset]) -> dict[str, int]:
    """Reject current root, duplicate-component or content leakage."""
    names = list(datasets)
    report = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for field in ('group', 'root', 'content_sha256'):
                overlap = {getattr(r, field) for r in datasets[left].index} & {
                    getattr(r, field) for r in datasets[right].index}
                report[f'{left}/{right}/{field}'] = len(overlap)
                if overlap:
                    raise ValueError(f'{field} leakage between {left} and {right}')
    return report
