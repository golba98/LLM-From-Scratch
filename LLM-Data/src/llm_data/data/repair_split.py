"""Leakage-safe prospective splits for existing conversational repair records."""

import hashlib
import json

from llm_data.data.general_chat import normalized


def split_repair_records(rows: list[dict], *, seed: int) -> tuple[list[dict], list[dict]]:
    """Keep roots and normalized duplicate conversations in one deterministic split."""
    parents = list(range(len(rows)))
    owners: dict[tuple, int] = {}
    digests: list[str] = []

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left: int, right: int) -> None:
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    for index, row in enumerate(rows):
        if any(not isinstance(row.get(key), str) or not row[key]
               for key in ('source', 'root_id', 'conversation_id')):
            raise ValueError('Repair split records require source, root and conversation identities.')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or message.get('role') not in ('system', 'user', 'assistant')
            or not isinstance(message.get('content'), str) or not message['content'].strip() for message in messages
        ):
            raise ValueError('Repair split records require valid messages.')
        canonical = [(message['role'], normalized(message['content'])) for message in messages]
        digest = hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
        digests.append(digest)
        keys = [('root', row['source'], row['root_id']), ('content', digest),
                ('identity', row['source'], row['conversation_id'])]
        if isinstance(row.get('duplicate_group'), str) and row['duplicate_group']:
            keys.append(('existing_group', row['duplicate_group']))
        for key in keys:
            if key in owners:
                join(index, owners[key])
            else:
                owners[key] = index
    clusters: dict[int, list[int]] = {}
    for index in range(len(rows)):
        clusters.setdefault(root(index), []).append(index)
    groups = {}
    for indices in clusters.values():
        identity = min(digests[index] for index in indices)
        held_out = int(hashlib.sha256(f'{seed}:repair-validation:{identity}'.encode()).hexdigest()[:8], 16) % 5 == 0
        for index in indices:
            groups[index] = (identity, held_out)
    train, validation = [], []
    for index, row in enumerate(rows):
        identity, held_out = groups[index]
        target = validation if held_out else train
        target.append({**row, 'duplicate_group': identity})
    if not train or not validation:
        raise ValueError('Root/duplicate grouping leaves an empty repair split; do not train.')
    return train, validation
