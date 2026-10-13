"""Reproducible synthetic retrieval evaluation, never generative training data."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_memory.store import MemoryStore
from llm_specialist.config import EncoderConfig
from llm_specialist.encoder import FrozenEncoder
from llm_tokenizer.tokenizer import load_tokenizer


def metrics(rows: list[dict]) -> dict:
    """Separate ranking metrics from abstention on negative queries."""
    positive = [r for r in rows if r['gold']]
    negative = [r for r in rows if not r['gold']]
    output = {}
    for k in (1, 3, 5):
        output[f'recall@{k}'] = statistics.mean(len(set(r['ranked'][:k]) & set(r['gold'])) / len(r['gold']) for r in positive)
        output[f'precision@{k}'] = statistics.mean(len(set(r['ranked'][:k]) & set(r['gold'])) / k for r in positive)
    output['mrr'] = statistics.mean(next((1 / (i + 1) for i, item in enumerate(r['ranked']) if item in r['gold']), 0) for r in positive)
    output['false_positive_rate'] = statistics.mean(bool(r['ranked']) for r in negative)
    output['query_count'] = len(rows)
    return output


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('tests/fixtures/memory_retrieval.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--tokenizer', type=Path, default=Path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(args.fixture.read_text())
    enc = FrozenEncoder(EncoderConfig(device=args.device))
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_tokenizer(args.tokenizer)
    store = MemoryStore(args.output / 'evaluation.sqlite3')
    for record in fixture['records']:
        store.add_turn('benchmark', record['conversation'], record['turn'],
                       [('user', record['user']), ('assistant', record['assistant'])])
    metadata = enc.metadata()
    metadata.update(prompts={k: enc.model.prompts[k] for k in ('SearchQuery', 'Document')},
                    prefix=None, chunking='480-utf8-bytes-v1')
    encode = lambda texts, task: enc.encode(texts, task=task)[0].numpy()
    started = time.perf_counter()
    identity = store.ensure_index(metadata, encode, 'benchmark')
    index_seconds = time.perf_counter() - started
    queries = fixture['queries']
    query_vectors = []
    embedding_ms = []
    for query in queries:
        started = time.perf_counter()
        query_vectors.append(encode([query['text']], 'SearchQuery'))
        embedding_ms.append((time.perf_counter() - started) * 1000)
    def score(split, threshold):
        rows = []
        for query, vector in zip(queries, query_vectors):
            if query['split'] != split:
                continue
            start = time.perf_counter()
            hits = store.search('benchmark', [query['conversation']], vector, identity,
                                limit=5, threshold=threshold)
            rows.append(dict(query=query['text'], category=query['category'],
                             gold=query['gold'], ranked=[f'{h.turn}:{h.role}' for h in hits],
                             search_ms=(time.perf_counter() - start) * 1000,
                             excerpts=[asdict(h) for h in hits],
                             retrieved_tokens=sum(len(tokenizer.encode(h.content, add_special_tokens=False).ids) for h in hits[:3])))
        return rows
    candidates = []
    for threshold in np.arange(0.3, 0.951, 0.01):
        result = metrics(score('development', float(threshold)))
        # Constrain negatives first, then optimize positive recall and ranking.
        candidates.append((result['false_positive_rate'] <= 0.05,
                           -result['false_positive_rate'] if result['false_positive_rate'] > 0.05 else result['recall@3'],
                           result['mrr'], -float(threshold), float(threshold)))
    threshold = max(candidates)[-1]
    development = score('development', threshold)
    test = score('test', threshold)
    report = dict(description='Synthetic reproducible retrieval fixtures; not evidence of language generation quality.',
                  encoder=metadata, threshold=threshold, development=metrics(development), test=metrics(test),
                  index_seconds=index_seconds, embedding_ms_median=statistics.median(embedding_ms),
                  embedding_ms_p95=float(np.percentile(embedding_ms, 95)),
                  search_ms_median=statistics.median(r['search_ms'] for r in test),
                  retrieved_tokens_median=statistics.median(r['retrieved_tokens'] for r in test),
                  vector_bytes=store.connection.execute('SELECT SUM(length(value)) FROM vectors').fetchone()[0],
                  gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None,
                  test_outputs=test, development_outputs=development)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('threshold', 'test', 'embedding_ms_median', 'search_ms_median')}))
    store.close()


if __name__ == '__main__':
    main()
