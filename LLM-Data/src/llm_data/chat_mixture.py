"""Manifest views and finite token-quota sampling over existing chat records."""

from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from torch.utils.data import Dataset, Sampler
from llm_data.chat_dataset import IndexedChatDataset


class ManifestChatDataset(Dataset):
    """Compose existing compatible records without copying the corpus."""

    def __init__(self, manifest: Path, tokenizer, *, context: int = 2048):
        document = json.loads(Path(manifest).read_text())
        if document.get('format') != 'codexa-chat-mixture-v1':
            raise ValueError('Unknown conversational mixture format.')
        if not document.get('admitted_for_training'):
            raise ValueError('Mixture is awaiting data-quality admission.')
        if document.get('status')!='HUMAN_ADMITTED':
            raise ValueError('A configuration flag is not reviewed data admission.')
        review_path=Path(document['review_path'])
        if hashlib.sha256(review_path.read_bytes()).hexdigest()!=document['review_sha256']:
            raise ValueError('Mixture review identity drift.')
        review=json.loads(review_path.read_text())
        if (review.get('approved') is not True or review.get('method')!='human_manual_review'
                or not review.get('reviewer') or review.get('manifest_sha256')!=document['reviewed_manifest_sha256']):
            raise ValueError('Mixture human admission is incomplete.')
        self.datasets, self.ends, self.index = [], [], []
        self.audit = Counter()
        for source in document['sources']:
            dataset = IndexedChatDataset(Path(source['path']), tokenizer, context=context)
            if dataset.sha256 != source['sha256']:
                raise ValueError('Mixture source fingerprint drift.')
            self.datasets.append(dataset)
            self.index.extend(dataset.index)
            self.ends.append(len(self.index))
            self.audit.update(dataset.audit)
        self.sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        identities = [(row.source,row.identity) for row in self.index]
        if len(set(identities)) != len(identities):
            raise ValueError('Duplicate source identity in mixture.')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        part = bisect_right(self.ends,index)
        offset = self.ends[part-1] if part else 0
        row = dict(self.datasets[part][index-offset])
        row['index'] = index
        return row


class TokenQuotaSampler(Sampler[int]):
    """Deterministic finite quota window; exhausted sources never silently cycle.

    At most one pass per source per epoch. A second epoch must be explicitly
    enabled by the trainer. Epoch and consumed batch cursor reconstruct ordering.
    """

    def __init__(self, dataset, *, seed: int, probabilities: dict[str,float], bucket_size: int = 128):
        import math
        if (not probabilities or any(not math.isfinite(p) or p <= 0 for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-9)):
            raise ValueError('Token probabilities must be positive and sum to one.')
        actual = {row.source for row in dataset.index}
        if actual != set(probabilities):
            raise ValueError('Token quotas must specify every source exactly once.')
        if bucket_size < 1:
            raise ValueError('Bucket size must be positive.')
        self.dataset, self.seed, self.probabilities = dataset, seed, dict(probabilities)
        self.bucket_size, self.epoch = bucket_size, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed+self.epoch)
        pools = {source: [] for source in self.probabilities}
        for i,row in enumerate(self.dataset.index):
            pools[row.source].append(i)
        for pool in pools.values():
            rng.shuffle(pool)
        consumed = Counter()
        total, bucket = 0, []
        while True:
            source = max(sorted(pools), key=lambda s: self.probabilities[s]*total-consumed[s])
            if not pools[source]:
                break
            index = pools[source].pop()
            tokens = self.dataset.index[index].assistant_tokens
            consumed[source] += tokens
            total += tokens
            bucket.append(index)
            if len(bucket) == self.bucket_size:
                yield from sorted(bucket,key=lambda i:self.dataset.index[i].tokens)
                bucket = []
        yield from sorted(bucket,key=lambda i:self.dataset.index[i].tokens)

    def __len__(self):
        return sum(1 for _ in self)
