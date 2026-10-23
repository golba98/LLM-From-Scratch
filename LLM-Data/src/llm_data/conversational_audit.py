"""Streaming conversational inventory and conservative duplicate screening."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import random
import sqlite3
from typing import Iterable

from llm_data.data.general_chat import normalized
from llm_tokenizer.sft import ChatMessage, serialize_conversation


def instruction_tags(messages: list[dict]) -> list[str]:
    """Return contextual candidates, not verified semantic coverage labels."""
    text = ' '.join(m['content'] for m in messages if m['role'] in ('user', 'system')).lower()
    patterns = {
        'one_word': r'\b(?:reply|respond|answer|return|write|output)\b.{0,65}\b(?:one|single|only the) word\b',
        'number_only': r'\b(?:reply|respond|answer|return)\b.{0,65}\b(?:only|just) (?:a |the )?(?:number|result)\b',
        'yes_no': r'\b(?:answer|reply|respond)\b.{0,55}\byes or no\b',
        'short_answer': r'\b(?:answer|reply|explain|respond)\b.{0,55}\b(?:briefly|short|concise|one sentence)\b',
        'item_count': r'\b(?:give|list|provide)\b.{0,35}\b(?:exactly )?(?:three|3)\b.{0,25}\b(?:items|examples|reasons)\b',
        'no_commentary': r'\b(?:no|without|do not (?:add|include))\b.{0,25}\b(?:commentary|explanation|extra text)\b',
        'structured_format': r'\b(?:respond|reply|return|answer)\b.{0,50}\b(?:json|csv|bullet|format)\b',
        'arithmetic_candidate': r'\b(?:calculate|sum|subtract|multiply|divide|plus|minus)\b.{0,45}\d',
    }
    return [key for key, pattern in patterns.items() if re.search(pattern, text)]


def near_duplicate(left: str, right: str, *, threshold: float = .85) -> bool:
    """Conservative five-word Jaccard screen; matches require review."""
    a, b = normalized(left).split(), normalized(right).split()
    if min(len(a), len(b)) < 25 or min(len(a), len(b)) / max(len(a), len(b)) < .8:
        return False
    x = {tuple(a[i:i+5]) for i in range(len(a)-4)}
    y = {tuple(b[i:i+5]) for i in range(len(b)-4)}
    return len(x & y) / len(x | y) >= threshold


def content_identity(messages: list[dict]) -> str:
    """Match existing General Chat normalization and fingerprint conventions."""
    canonical = [(m['role'], normalized(m['content'])) for m in messages]
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def audit_files(paths: Iterable[Path], tokenizer, ledger: Path, *, context: int = 2048,
                samples_per_file: int = 24) -> dict:
    """Stream records; retain compact cross-split identities in a fresh SQLite ledger.

    Existing token counts are checked on deterministic samples. Other records are
    serialized for exact token counts. Language without provenance stays unknown.
    """
    if ledger.exists():
        raise FileExistsError('Audit ledger requires a new namespace.')
    reports, review = [], []
    db = sqlite3.connect(ledger)
    db.execute('CREATE TABLE records(dataset TEXT, split TEXT, root TEXT, content TEXT)')
    try:
        for path in paths:
            path = Path(path)
            counts = Counter()
            digest = hashlib.sha256()
            length_bins = Counter()
            supervision = context_tokens = verified = 0
            dataset, split = str(path.parent), path.stem
            rng = random.Random(42)
            samples = []
            with path.open('rb') as stream:
                for position, line in enumerate(stream):
                    digest.update(line)
                    counts['records'] += 1
                    try:
                        row = json.loads(line)
                        messages = row['messages']
                        if not messages or any(m['role'] not in ('user', 'assistant', 'system')
                            or not isinstance(m['content'], str) or not m['content'].strip() for m in messages):
                            raise ValueError('Malformed messages')
                        identity = content_identity(messages)
                        if row.get('content_sha256', identity) != identity:
                            raise ValueError('Content digest mismatch')
                        # Metadata is useful evidence; it is never labeled freshly token-verified.
                        if 'assistant_tokens' not in row or position % 997 == 0:
                            record = serialize_conversation([ChatMessage(**m) for m in messages],
                                                            tokenizer, maximum_tokens=context)
                            tokens, assistant = len(record.input_ids), record.supervised_token_count
                            if 'tokens' in row and (tokens != row['tokens'] or assistant != row['assistant_tokens']):
                                raise ValueError('Tokenizer/count drift')
                            verified += 1
                        else:
                            tokens, assistant = row['tokens'], row['assistant_tokens']
                        if not 0 < assistant < tokens <= context:
                            raise ValueError('Invalid target count')
                    except (ValueError, KeyError, TypeError) as error:
                        counts['rejected_or_invalid'] += 1
                        if len(review) < 200:
                            review.append(dict(path=str(path), line=position+1, issue=str(error)))
                        continue
                    counts['usable'] += 1
                    turns = [m for m in messages if m['role'] == 'assistant']
                    counts['assistant_turns'] += len(turns)
                    counts['multi_turn'] += len(turns) > 1
                    counts['system_instruction'] += messages[0]['role'] == 'system'
                    counts['language/' + row.get('language', row.get('lang', 'unknown'))] += 1
                    counts['source/' + row.get('source', 'unknown')] += 1
                    counts['manifest_prior_split_exposure_flag'] += row.get('historical_sft_exposure') is True
                    counts['exposure_not_established'] += row.get('historical_sft_exposure') is not True
                    supervision += assistant
                    context_tokens += tokens
                    counts['supervised_tokens/' + row.get('source', 'unknown')] += assistant
                    length_bins[str(256 * ((tokens-1)//256 + 1))] += 1
                    tags = instruction_tags(messages)
                    for tag in tags:
                        counts['heuristic/' + tag] += 1
                    for m in turns:
                        words = normalized(m['content']).split()
                        counts['assistant_under_12_words'] += len(words) <= 12
                        counts['assistant_over_200_words'] += len(words) > 200
                        counts['repetition_candidate'] += len(words) > 20 and len(set(words))/len(words) < .15
                    sample = dict(path=str(path), line=position+1, tags=tags,
                                  messages=messages, semantic_review=None, language_review=None)
                    if len(samples) < samples_per_file:
                        samples.append(sample)
                    else:
                        slot = rng.randrange(counts['usable'])
                        if slot < samples_per_file:
                            samples[slot] = sample
                    root = json.dumps([row.get('source', 'unknown'), row.get('root_id', row.get('conversation_id'))])
                    db.execute('INSERT INTO records VALUES(?,?,?,?)', (dataset, split, root, identity))
                    if position % 10000 == 0:
                        db.commit()
                        if ledger.stat().st_size > 400*1024**2:
                            raise RuntimeError('Bounded audit ledger allowance exceeded.')
            db.commit()
            review.extend(samples)
            duplicate_count = db.execute('SELECT coalesce(sum(n-1),0) FROM '
                '(SELECT count(*) n FROM records WHERE dataset=? AND split=? GROUP BY content)',
                (dataset, split)).fetchone()[0]
            reports.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest(),
                counts=dict(counts), normalized_duplicate_records=duplicate_count,
                supervised_tokens=supervision, context_tokens=context_tokens,
                freshly_serialized_records=verified, sequence_length_bins=dict(length_bins)))
        overlaps = []
        for field in ('root', 'content'):
            rows = db.execute(f'SELECT dataset,{field},count(DISTINCT split) FROM records '
                              f'GROUP BY dataset,{field} HAVING count(DISTINCT split)>1').fetchall()
            totals = Counter(row[0] for row in rows)
            overlaps.extend(dict(dataset=d, kind=field, cross_split_groups=n) for d, n in totals.items())
        return dict(format='codexa-conversation-audit-v1', datasets=reports,
            cross_split_overlap=overlaps, manual_review=review,
            limitations=['Instruction categories are candidates pending semantic review.',
                'Language is unknown unless recorded by source; ASCII is not proof of English.',
                'Metadata supervision counts are sampled, not exhaustively re-tokenized.',
                'No exhaustive corpus-wide semantic near-duplicate certification.',
                'Absence from this corpus does not establish absence from historical pretraining.'])
    finally:
        db.close()
