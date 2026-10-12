"""Phase 4 SFT data: review batches, admission, contamination screening, frozen manifest.

Preparation only. Nothing here imports a trainer, constructs an optimizer or writes
model state. Codex's feasibility directory is a read-only input; its artifacts are
bound by SHA-256 and never edited. Agent reviews are engineering judgments, never
independent human evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
PRIOR = ROOT / 'outputs/preparation/codexa-934m-conversational-feasibility/20261012T051805Z-a5c8725c'
DIAGNOSTIC = ROOT / 'outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256'
FOUNDATION = ROOT / 'outputs/training/codexa-934m-foundation-5h-v1'
BASE = ROOT / 'outputs/preparation/codexa-934m-sft-phase4'
EXPECTED_CHECKPOINT_SHA256 = 'ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e'

TRAIN_QUOTA = {'oasst1': 224, 'ultrachat': 224}
VALIDATION_QUOTA = {'oasst1': 16, 'ultrachat': 16}
AUTHORED_COUNT = 64
MAXIMUM_TOKENS = 512
EPOCH_SEEDS = (73, 74)
CATEGORIES = ('greeting', 'factual', 'explanation', 'multi-sentence', 'instruction',
              'reasoning', 'arithmetic', 'context-grounded', 'clarifying', 'concise',
              'structured', 'creative', 'ack-refusal-branding')
TRAINING_EXAMPLES = 512
CATEGORY_CAP = int(.30 * TRAINING_EXAMPLES)       # 153: no category above 30% of the training set
ACK_CAP = int(.05 * TRAINING_EXAMPLES)            # 25: acknowledgment/refusal/branding at most 5%
AUTHORED_CATEGORY = {'word-only': 'instruction', 'formatting': 'structured', 'arithmetic': 'arithmetic',
                     'supplied-context': 'context-grounded'}
NEAR_DUPLICATE_JACCARD = .5
NEAR_DUPLICATE_N = 5
SUBSTANTIAL_WORDS = 8
FIRST_PASS_REVIEWERS = ('Codex engineer review', 'Claude review subagent')
SECOND_PASS_REVIEWER = 'Claude lead second-pass review (agent engineering review, not human)'


# ---------------------------------------------------------------- basic I/O

def digest(path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, values) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def normalize(text: str) -> str:
    return ' '.join(re.findall(r"[\w']+", text.casefold()))


def shingles(text: str, n: int = 8) -> set:
    words = normalize(text).split()
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def conversation_text(row: dict) -> str:
    return '\n'.join(message['content'] for message in row['messages'])


# ---------------------------------------------------------------- coordination

def git_snapshot() -> dict:
    """Git branch/status/worktrees for every component repository (read-only)."""
    result = {}
    for path in sorted(WORKSPACE.iterdir()):
        if not (path / '.git').exists():
            continue
        entry = {}
        for name, argv in [('branch', ['status', '-sb']), ('porcelain', ['status', '--porcelain']),
                           ('head', ['rev-parse', 'HEAD']), ('worktrees', ['worktree', 'list'])]:
            done = subprocess.run(['git', '-C', str(path), *argv], capture_output=True, text=True, timeout=60)
            entry[name] = done.stdout
        result[path.name] = entry
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(), repositories=result)


# ---------------------------------------------------------------- initialization

def prior_inputs() -> dict:
    """Codex's feasibility artifacts, bound by hash; never modified."""
    names = ['data/candidates.jsonl', 'data/review-order.json', 'data/engineer-review.jsonl',
             'data/authored.jsonl', 'data/authored-admission.jsonl', 'evidence/environment.json',
             'evidence/candidate-counts.json', 'config/experiment-draft.json', 'REPORT.md']
    return {name: dict(path=str(PRIOR / name), sha256=digest(PRIOR / name)) for name in names}


def verify_authored(rows: list[dict], tokenizer) -> list[dict]:
    """The 64 authored records must equal a regeneration from the frozen specifications."""
    from codexa_workspace.sft_feasibility import authored_specs, correct
    from llm_data.data.general_chat import normalized
    from llm_tokenizer.sft import ChatMessage, serialize_conversation
    specs = authored_specs()
    if len(rows) != AUTHORED_COUNT or len(specs) != AUTHORED_COUNT:
        raise ValueError('Authored set must contain exactly 64 records.')
    for index, (row, spec) in enumerate(zip(rows, specs)):
        if not correct(spec):
            raise ValueError(f'Authored oracle failure {index}')
        messages = [dict(role='user', content=spec['prompt']), dict(role='assistant', content=spec['answer'])]
        if row['messages'] != messages or row['conversation_id'] != f'codexa-feasibility-authored-{index:02d}':
            raise ValueError(f'Authored record {index} differs from its specification.')
        canonical = [(m['role'], normalized(m['content'])) for m in messages]
        if row['content_sha256'] != hashlib.sha256(json.dumps(canonical).encode()).hexdigest():
            raise ValueError(f'Authored fingerprint drift {index}')
        serialized = serialize_conversation([ChatMessage(**m) for m in messages], tokenizer,
                                            maximum_tokens=MAXIMUM_TOKENS)
        if (row['tokens'], row['assistant_tokens']) != (len(serialized.input_ids), serialized.supervised_token_count):
            raise ValueError(f'Authored token count drift {index}')
    return [dict(row, category=AUTHORED_CATEGORY[spec['category']], authored_kind=spec['category'])
            for row, spec in zip(rows, specs)]


def initialize() -> Path:
    directory = BASE / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('review/batches', 'review/decisions', 'review/second-pass', 'data', 'manifest',
                 'evaluation', 'config', 'evidence'):
        (directory / name).mkdir(parents=True)
    write(directory / 'evidence/coordination-start.json', git_snapshot())
    write(directory / 'evidence/prior-inputs.json', prior_inputs())
    shutil.copyfile(PRIOR / 'data/authored.jsonl', directory / 'data/authored-source.jsonl')
    write(directory / 'review/queue-state.json', dict(next_position={}, batches=[]))
    return directory


# ---------------------------------------------------------------- review batches

EXTENSION_CANDIDATES = 'data/candidates-extension.jsonl'


def candidate_index(directory: Path | None = None) -> dict:
    """Codex's frozen pool plus any documented extension pool in the Phase 4 directory."""
    rows = {row['record']['conversation_id']: row for row in jsonlines(PRIOR / 'data/candidates.jsonl')}
    if directory is not None and (Path(directory) / EXTENSION_CANDIDATES).exists():
        for row in jsonlines(Path(directory) / EXTENSION_CANDIDATES):
            if row['record']['conversation_id'] in rows:
                raise ValueError('Extension candidate duplicates the frozen pool.')
            rows[row['record']['conversation_id']] = row
    return rows


def extend_validation_pool(directory: Path, source: str) -> dict:
    """Validation-only extension from the general-chat-v2 test split, used only after the
    validation split's eligible pool is exhausted. Filters copy Codex's candidate_pool()
    exactly (scripts/prepare_sft_feasibility.py:116-148); ranking uses the same
    sha256('73:<source>:<id>') rule. Extension rows keep split='validation' as their role and
    record source_split='test'."""
    directory = Path(directory)
    if (directory / EXTENSION_CANDIDATES).exists():
        raise FileExistsError('Extension pool already exists; never regenerate it.')
    prior = read(PRIOR / 'evidence/environment.json')
    source_file = Path(prior['source_directory']) / 'test.jsonl'
    cases = benchmark_cases()
    questions = {normalize(re.sub(r'<[^>]+>', ' ', x['prompt'])) for x in cases}
    forbidden = set().union(*(shingles(re.sub(r'<[^>]+>', ' ', x['prompt'])) for x in cases))
    markers = ('as an ai language', 'as a language model', 'i cannot browse', "i don't have access to",
               'consult a doctor', 'consult a lawyer')
    pool, seen = [], 0
    for ordinal, row in enumerate(jsonlines(source_file)):
        if row['source'] != source:
            continue
        seen += 1
        if row['tokens'] > 512 or not 8 <= row['assistant_tokens'] <= 96:
            continue
        texts = [m['content'] for m in row['messages']]
        letters = [c for c in '\n'.join(texts) if c.isalpha()]
        if not letters or sum(not c.isascii() for c in letters) / len(letters) > .02:
            continue
        if any(normalize(m['content']) in questions for m in row['messages'] if m['role'] == 'user'):
            continue
        if any(shingles(text) & forbidden for text in texts):
            continue
        if any(marker in '\n'.join(texts).casefold() for marker in markers):
            continue
        pool.append(dict(record=row, split='validation', source_split='test', ordinal=ordinal,
                         source_file=str(source_file),
                         ranking_sha256=hashlib.sha256(f'73:{source}:{row["conversation_id"]}'.encode()).hexdigest()))
    pool.sort(key=lambda x: x['ranking_sha256'])
    write_rows(directory / EXTENSION_CANDIDATES, pool)
    evidence = dict(source=source, source_file=str(source_file), source_sha256=digest(source_file), seen=seen,
                    eligible=len(pool), reason='validation-split eligible pool exhausted before 16 admissions',
                    filters='identical to Codex candidate_pool(); role split=validation, source_split=test')
    write(directory / 'evidence/validation-pool-extension.json', evidence)
    return evidence


def pool_order(split: str, source: str, candidates: dict) -> list[str]:
    """Train pools continue Codex's review order; validation uses the hash-ranked candidate order."""
    if split == 'train':
        return list(read(PRIOR / 'data/review-order.json')[source])
    rows = [row for row in candidates.values() if row['split'] == split and row['record']['source'] == source]
    # Frozen validation-split rows first, then any test-split extension rows (each hash-ranked).
    return [row['record']['conversation_id'] for row in
            sorted(rows, key=lambda r: (r.get('source_split', split) != split, r['ranking_sha256']))]


def prior_review_ids() -> set:
    return {row['conversation_id'] for row in jsonlines(PRIOR / 'data/engineer-review.jsonl')}


ENGLISH_FUNCTION_WORDS = frozenset((
    'the a an and or but of to in on at for with is are was were be been it this that these those you your '
    'i we he she they my our not no do does did have has had can will would should could what which who how '
    'why when where if as by from about than then so there here me him her them its his their just also more '
    'most very all any some one two into out up over').split())
PREFILTER_RATIO, PREFILTER_MINIMUM_WORDS = .05, 12


def english_function_word_ratio(row: dict) -> tuple[float, int]:
    words = re.findall(r"[a-z\u00c0-\u00ff']+", conversation_text(row).lower())
    return (sum(w in ENGLISH_FUNCTION_WORDS for w in words) / len(words) if words else 1.0), len(words)


def prefilter_reject(row: dict) -> str | None:
    """Conservative non-English skip (rubric rule 12). It can only exclude, never admit.

    Calibrated on wave-1 agent decisions: at ratio < 0.05 with >= 8 words, 214/214 flagged
    records were non-English rejects and 0 were admits; production uses >= 12 words.
    """
    ratio, count = english_function_word_ratio(row)
    if count >= PREFILTER_MINIMUM_WORDS and ratio < PREFILTER_RATIO:
        return f'prefilter: non-English heuristic (function-word ratio {ratio:.3f} over {count} words)'
    return None


def make_batches(directory: Path, plan: list[tuple[str, str, int, int]], *, prefilter: bool = False) -> list[str]:
    """Write fixed review batches (split, source, records, batch size), continuing the pool order."""
    directory = Path(directory)
    candidates = candidate_index(directory)
    reviewed = prior_review_ids()
    skipped = []
    state = read(directory / 'review/queue-state.json')
    created = []
    for split, source, count, size in plan:
        key = f'{split}/{source}'
        order = pool_order(split, source, candidates)
        position = state['next_position'].get(key, 0)
        selected = []
        while len(selected) < count and position < len(order):
            identity = order[position]
            if identity not in reviewed:
                reason = prefilter_reject(candidates[identity]['record']) if prefilter else None
                if reason:
                    skipped.append(dict(conversation_id=identity, split=split, source=source, review_position=position,
                                        content_sha256=candidates[identity]['record']['content_sha256'],
                                        disposition='reject', reason=reason, reviewer='automatic prefilter',
                                        independent_human_review=False))
                else:
                    selected.append((position, identity))
            position += 1
        state['next_position'][key] = position
        for start in range(0, len(selected), size):
            name = f"B{len(state['batches']) + 1:03d}"
            rows = []
            for review_position, identity in selected[start:start + size]:
                candidate = candidates[identity]
                record = candidate['record']
                rows.append(dict(batch=name, conversation_id=identity, split=split, source=source,
                                 review_position=review_position, content_sha256=record['content_sha256'],
                                 tokens=record['tokens'], assistant_tokens=record['assistant_tokens'],
                                 messages=record['messages']))
            write_rows(directory / f'review/batches/{name}.jsonl', rows)
            state['batches'].append(dict(name=name, pool=key, records=len(rows),
                                         first_position=rows[0]['review_position'],
                                         last_position=rows[-1]['review_position']))
            created.append(name)
    if skipped:
        with (directory / 'review/prefilter.jsonl').open('a') as stream:
            for row in skipped:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    write(directory / 'review/queue-state.json', state)
    return created


# ---------------------------------------------------------------- decisions

def _validate_decision(row: dict, candidates: dict, *, second_pass: bool) -> None:
    identity = row.get('conversation_id')
    if identity not in candidates:
        raise ValueError(f'Unknown conversation {identity}')
    if row.get('content_sha256') != candidates[identity]['record']['content_sha256']:
        raise ValueError(f'Stale review content hash {identity}')
    if row.get('disposition') not in ('admit', 'reject'):
        raise ValueError(f'Invalid disposition {identity}')
    if not isinstance(row.get('reason'), str) or not row['reason'].strip():
        raise ValueError(f'Missing review rationale {identity}')
    if row.get('independent_human_review') is not False:
        raise ValueError(f'Agent review must not claim human review {identity}')
    reviewer = row.get('reviewer', '')
    if second_pass:
        if reviewer != SECOND_PASS_REVIEWER:
            raise ValueError(f'Second-pass reviewer label mismatch {identity}')
    elif not reviewer.startswith(FIRST_PASS_REVIEWERS):
        raise ValueError(f'Unknown first-pass reviewer {identity}')
    if row['disposition'] == 'admit' and (second_pass or reviewer.startswith('Claude')):
        if row.get('category') not in CATEGORIES:
            raise ValueError(f'Admitted record needs a known category {identity}')


def ledger(directory: Path) -> dict:
    """Combine Codex's first-pass ledger, subagent batches and lead second-pass decisions."""
    directory = Path(directory)
    candidates = candidate_index(directory)
    first, second = {}, {}
    for row in jsonlines(PRIOR / 'data/engineer-review.jsonl'):
        _validate_decision(row, candidates, second_pass=False)
        first[row['conversation_id']] = dict(row, origin='codex-feasibility-ledger')
    batches = {path.stem: [r['conversation_id'] for r in jsonlines(path)]
               for path in sorted((directory / 'review/batches').glob('*.jsonl'))}
    for path in sorted((directory / 'review/decisions').glob('*.jsonl')):
        rows = list(jsonlines(path))
        expected = batches.get(path.stem)
        if expected is None or [r['conversation_id'] for r in rows] != expected:
            raise ValueError(f'Decision file {path.name} does not cover its batch exactly, in order.')
        for row in rows:
            _validate_decision(row, candidates, second_pass=False)
            if row['conversation_id'] in first:
                raise ValueError(f"Duplicate first-pass decision {row['conversation_id']}")
            first[row['conversation_id']] = dict(row, origin=f'batch {path.stem}')
    for path in sorted((directory / 'review/second-pass').glob('*.jsonl')):
        for row in jsonlines(path):
            _validate_decision(row, candidates, second_pass=True)
            if row['conversation_id'] in second:
                raise ValueError(f"Duplicate second-pass decision {row['conversation_id']}")
            if row['conversation_id'] not in first:
                raise ValueError(f"Second pass without first pass {row['conversation_id']}")
            second[row['conversation_id']] = dict(row, origin=path.name)
    return dict(first=first, second=second, candidates=candidates)


def review_statistics(review: dict) -> dict:
    first, second = review['first'], review['second']
    by_pool = defaultdict(Counter)
    for identity, row in first.items():
        candidate = review['candidates'][identity]
        by_pool[f"{candidate['split']}/{candidate['record']['source']}"][row['disposition']] += 1
    agreement = Counter()
    for identity, row in second.items():
        agreement[f"first {first[identity]['disposition']} / second {row['disposition']}"] += 1
    rejects = [i for i, r in first.items() if r['disposition'] == 'reject']
    checked = [i for i in rejects if i in second]
    overturned = [i for i in checked if second[i]['disposition'] == 'admit']
    return dict(first_pass=len(first), second_pass=len(second),
                first_pass_by_pool={k: dict(v) for k, v in sorted(by_pool.items())},
                agreement=dict(agreement),
                reject_spot_check=dict(rejects=len(rejects), checked=len(checked),
                                       second_pass_would_admit=len(overturned),
                                       note='Disagreements are excluded; a reject is never promoted to admission.'),
                reason_counts=Counter(r['reason'] for r in first.values() if r['disposition'] == 'reject').most_common(40))


def pending_second_pass(directory: Path) -> list[str]:
    """First-pass admits still lacking a lead second-pass decision, in pool order."""
    review = ledger(directory)
    return [i for i, row in review['first'].items() if row['disposition'] == 'admit' and i not in review['second']]


# ---------------------------------------------------------------- contamination

def benchmark_cases() -> list[dict]:
    """The 48 historical prompts and 16 conversational cases (greedy rows, one per case)."""
    rows = [row for row in jsonlines(DIAGNOSTIC / 'evidence/generation/generations-final.jsonl')
            if row['decode'] == 'greedy']
    if sorted(Counter(row['suite'] for row in rows).items()) != [('conversation', 16), ('historical', 48)]:
        raise ValueError('Benchmark identity drift: expected 48 historical and 16 conversational prompts.')
    return rows


def benchmark_texts() -> list[str]:
    texts = []
    for row in benchmark_cases():
        texts.append(re.sub(r'<[^>]+>', ' ', row['prompt']))
        if row.get('expected'):
            texts.append(row['expected'])
    for case in read(DIAGNOSTIC / 'evidence/generation/conversational-cases.json'):
        texts.append(case['question'])
    return texts


class ContaminationScreen:
    """Exact normalized questions and 8-word shingles are hard rejects; weaker overlap is flagged."""

    def __init__(self, texts: list[str]) -> None:
        self.questions = {normalize(t) for t in texts if normalize(t)}
        self.forbidden = set().union(*(shingles(t) for t in texts)) if texts else set()
        self.five = [shingles(t, 5) for t in texts]

    def screen(self, row: dict) -> dict:
        texts = [m['content'] for m in row['messages']]
        exact = any(normalize(m['content']) in self.questions for m in row['messages'] if m['role'] == 'user')
        overlap = sorted({' '.join(s) for t in texts for s in shingles(t) & self.forbidden})
        five = shingles(conversation_text(row), 5)
        weak = max((len(five & other) / len(other) for other in self.five if other), default=0.0)
        return dict(exact_question=exact, eight_word_overlap=overlap[:5],
                    hard_reject=exact or bool(overlap),
                    possible_semantic_overlap=weak >= .3,
                    max_benchmark_five_gram_coverage=round(weak, 4))


TEMPLATE_SHINGLE_MINIMUM_CONVERSATIONS = 10


def template_shingles(candidates: dict) -> set:
    """8-word shingles shared by >= 10 pool conversations: prompt templates, not content leakage.

    UltraChat requests reuse boilerplate such as 'Based on the text material above, generate
    the response to the following question'; such shingles are ignored for train/validation
    shingle isolation only (exact-message, root, group and hash isolation are unaffected).
    """
    counts = Counter()
    for candidate in candidates.values():
        counts.update(shingles(conversation_text(candidate['record'])))
    return {gram for gram, count in counts.items() if count >= TEMPLATE_SHINGLE_MINIMUM_CONVERSATIONS}


def substantial_messages(row: dict) -> set:
    return {normalize(m['content']) for m in row['messages'] if len(normalize(m['content']).split()) >= SUBSTANTIAL_WORDS}


# ---------------------------------------------------------------- selection

def _admitted(review: dict) -> list[str]:
    """Both passes admit; any disagreement or missing second pass excludes."""
    return [i for i, row in review['first'].items()
            if row['disposition'] == 'admit' and review['second'].get(i, {}).get('disposition') == 'admit']


def _pool_rank(review: dict) -> dict:
    ranks = {}
    candidates = review['candidates']
    for split in ('train', 'validation'):
        for source in ('oasst1', 'ultrachat'):
            for position, identity in enumerate(pool_order(split, source, candidates)):
                ranks[identity] = position
    return ranks


def select(directory: Path) -> dict:
    """Deterministic selection: validation first, then training, without relaxing any standard."""
    directory = Path(directory)
    review = ledger(directory)
    candidates = review['candidates']
    ranks = _pool_rank(review)
    screen = ContaminationScreen(benchmark_texts())
    templates = template_shingles(candidates)
    admitted = sorted(_admitted(review), key=lambda i: (candidates[i]['split'], candidates[i]['record']['source'], ranks[i]))
    taken_ids, taken_roots, taken_groups, taken_hashes = set(), set(), set(), set()
    taken_five: list[tuple[str, set]] = []
    exclusions, screening = [], {}
    selected = {'validation': [], 'train': []}
    counts = Counter()
    category_counts = defaultdict(Counter)
    # Caps apply to the whole 512-record training set; the 64 authored records count first.
    training_categories = Counter(AUTHORED_CATEGORY[spec['category']] for spec in _authored_specs())

    def conflict(row: dict, split: str) -> str | None:
        if row['conversation_id'] in taken_ids:
            return 'duplicate conversation identity'
        if (row['source'], row['root_id']) in taken_roots:
            return 'shared conversation root'
        if row['duplicate_group'] in taken_groups:
            return 'shared duplicate group'
        if row['content_sha256'] in taken_hashes:
            return 'identical content hash'
        five = shingles(conversation_text(row), NEAR_DUPLICATE_N)
        for other, other_five in taken_five:
            if jaccard(five, other_five) >= NEAR_DUPLICATE_JACCARD:
                return f'near duplicate of {other} (5-gram Jaccard >= {NEAR_DUPLICATE_JACCARD})'
        if split == 'train':
            validation_text = set().union(*(substantial_messages(v) for v in selected['validation'])) if selected['validation'] else set()
            if substantial_messages(row) & validation_text:
                return 'substantial normalized message shared with validation'
            validation_shingles = set().union(*(shingles(conversation_text(v)) for v in selected['validation'])) if selected['validation'] else set()
            if (shingles(conversation_text(row)) - templates) & validation_shingles:
                return 'eight-word overlap with validation'
        return None

    for split, quota in (('validation', VALIDATION_QUOTA), ('train', TRAIN_QUOTA)):
        for identity in admitted:
            candidate = candidates[identity]
            if candidate['split'] != split:
                continue
            row = candidate['record']
            source = row['source']
            if counts[(split, source)] >= quota[source]:
                continue
            check = screen.screen(row)
            screening[identity] = check
            if check['hard_reject']:
                exclusions.append(dict(conversation_id=identity, split=split, reason='benchmark contamination', detail=check))
                continue
            if row['tokens'] > MAXIMUM_TOKENS:
                exclusions.append(dict(conversation_id=identity, split=split, reason='exceeds 512 serialized tokens'))
                continue
            reason = conflict(row, split)
            if reason:
                exclusions.append(dict(conversation_id=identity, split=split, reason=reason))
                continue
            category = review['second'][identity]['category']
            if split == 'train':
                if training_categories[category] >= CATEGORY_CAP:
                    exclusions.append(dict(conversation_id=identity, split=split, reason=f'category cap {category} reached'))
                    continue
                if category == 'ack-refusal-branding' and training_categories[category] >= ACK_CAP:
                    exclusions.append(dict(conversation_id=identity, split=split, reason='ack/refusal/branding cap reached'))
                    continue
                training_categories[category] += 1
            taken_ids.add(identity)
            taken_roots.add((source, row['root_id']))
            taken_groups.add(row['duplicate_group'])
            taken_hashes.add(row['content_sha256'])
            taken_five.append((identity, shingles(conversation_text(row), NEAR_DUPLICATE_N)))
            category_counts[(split, source)][category] += 1
            counts[(split, source)] += 1
            selected[split].append(dict(row, category=category, split=split,
                provenance=dict(source_file=candidate['source_file'], source_ordinal=candidate['ordinal'],
                                source_path=row.get('source_path'), ranking_sha256=candidate['ranking_sha256'],
                                pool_position=ranks[identity]),
                review=dict(first=_public(review['first'][identity]), second=_public(review['second'][identity])),
                contamination=check))
    shortfall = {f'{split}/{source}': quota[source] - counts[(split, source)]
                 for split, quota in (('validation', VALIDATION_QUOTA), ('train', TRAIN_QUOTA))
                 for source in quota if counts[(split, source)] < quota[source]}
    result = dict(complete=not shortfall, shortfall=shortfall,
                  counts={f'{k[0]}/{k[1]}': v for k, v in sorted(counts.items())},
                  categories={f'{k[0]}/{k[1]}': dict(v) for k, v in sorted(category_counts.items())},
                  training_categories=dict(training_categories), template_shingles=len(templates),
                  exclusions=exclusions, admitted_both_passes=len(admitted),
                  review=review_statistics(review))
    write(directory / 'evidence/selection-status.json', result)
    write_rows(directory / 'data/selection-exclusions.jsonl', exclusions)
    if not shortfall:
        _write_selection(directory, selected, review, screen)
    return result


def _authored_specs() -> list[dict]:
    from codexa_workspace.sft_feasibility import authored_specs
    return authored_specs()


def _public(decision: dict) -> dict:
    return {k: decision.get(k) for k in ('reviewer', 'disposition', 'reason', 'category', 'flags', 'origin',
                                         'independent_human_review')}


def _write_selection(directory: Path, selected: dict, review: dict, screen: ContaminationScreen) -> None:
    from llm_tokenizer.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(tokenizer_identity()['path'])
    authored = verify_authored(list(jsonlines(directory / 'data/authored-source.jsonl')), tokenizer)
    validation_text = set().union(*(substantial_messages(v) for v in selected['validation']))
    authored_rows = []
    for row in authored:
        check = screen.screen(row)
        if check['hard_reject'] or substantial_messages(row) & validation_text:
            raise ValueError(f"Authored record overlaps a benchmark or validation: {row['conversation_id']}")
        authored_rows.append(dict(row, split='train', provenance=dict(source_file=str(PRIOR / 'data/authored.jsonl')),
            review=dict(first=dict(reviewer='Codex engineer authored-template review', disposition='admit',
                                   reason='Oracle-verified constrained answer (Codex feasibility preparation).',
                                   independent_human_review=False),
                        second=dict(reviewer='Claude lead regeneration check', disposition='admit',
                                    reason='Byte-identical regeneration from sft_feasibility.authored_specs(); oracle re-verified.',
                                    independent_human_review=False)),
            contamination=check))
    train = selected['train'] + authored_rows
    write_rows(directory / 'data/train.jsonl', [_dataset_row(r) for r in train])
    write_rows(directory / 'data/validation.jsonl', [_dataset_row(r) for r in selected['validation']])
    write_rows(directory / 'data/train-provenance.jsonl', train)
    write_rows(directory / 'data/validation-provenance.jsonl', selected['validation'])
    chosen = {r['conversation_id'] for r in train + selected['validation']}
    rejected = []
    for identity, first in review['first'].items():
        if identity in chosen:
            continue
        second = review['second'].get(identity)
        rejected.append(dict(conversation_id=identity, split=review['candidates'][identity]['split'],
                             source=review['candidates'][identity]['record']['source'],
                             content_sha256=first['content_sha256'], first=_public(first),
                             second=_public(second) if second else None,
                             outcome='rejected' if first['disposition'] == 'reject' or (second and second['disposition'] == 'reject')
                             else ('awaiting second pass' if not second else 'admitted but not selected (quota, cap, duplicate or isolation)')))
    write_rows(directory / 'data/rejected.jsonl', rejected)


DATASET_FIELDS = ('conversation_id', 'messages', 'root_id', 'source', 'source_path', 'historical_sft_exposure',
                  'content_sha256', 'tokens', 'assistant_tokens', 'duplicate_group', 'category')


def _dataset_row(row: dict) -> dict:
    return {k: row.get(k) for k in DATASET_FIELDS}


# ---------------------------------------------------------------- tokenizer and masks

def tokenizer_identity() -> dict:
    job = read(FOUNDATION / 'config/job.json')
    return dict(job['spec']['tokenizer'])


def verify_records(path: Path, tokenizer) -> tuple[dict, list[dict]]:
    """Real IndexedChatDataset verification plus explicit role/END/EOS/PAD/shift checks."""
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_tokenizer.sft import ChatMessage, IGNORE_LABEL, serialize_conversation
    from llm_tokenizer.tokenizer import (ASSISTANT_TOKEN, END_TOKEN, SYSTEM_TOKEN, USER_TOKEN)
    ids = dict(pad=tokenizer.token_to_id('<pad>'), bos=tokenizer.token_to_id('<bos>'),
               eos=tokenizer.token_to_id('<eos>'), system=tokenizer.token_to_id(SYSTEM_TOKEN),
               user=tokenizer.token_to_id(USER_TOKEN), assistant=tokenizer.token_to_id(ASSISTANT_TOKEN),
               end=tokenizer.token_to_id(END_TOKEN))
    vocabulary = tokenizer.get_vocab_size()
    dataset = IndexedChatDataset(path, tokenizer, context=MAXIMUM_TOKENS, verify_tokens=True)
    rows = list(jsonlines(path))
    records = []
    for index, row in enumerate(rows):
        sample = dataset[index]
        messages = [ChatMessage(m['role'], m['content']) for m in row['messages']]
        serialized = serialize_conversation(messages, tokenizer, maximum_tokens=MAXIMUM_TOKENS)
        inputs = list(serialized.input_ids)
        shifted = [*serialized.labels[1:], IGNORE_LABEL]
        if sample['input_ids'].tolist() != inputs or sample['labels'].tolist() != shifted:
            raise ValueError(f"Dataset/serializer mismatch {row['conversation_id']}")
        if len(inputs) > MAXIMUM_TOKENS or max(inputs) >= vocabulary or min(inputs) < 0:
            raise ValueError(f"Length or vocabulary violation {row['conversation_id']}")
        # Every label equals the next input (exactly one shift) wherever supervised.
        for position, label in enumerate(shifted):
            if label != IGNORE_LABEL and inputs[position + 1] != label:
                raise ValueError(f"Shift violation {row['conversation_id']} at {position}")
        supervised = [p for p, label in enumerate(shifted) if label != IGNORE_LABEL]
        end_targets = [p for p in supervised if shifted[p] == ids['end']]
        content_targets = [p for p in supervised if shifted[p] != ids['end']]
        forbidden = {ids['pad'], ids['bos'], ids['eos'], ids['system'], ids['user'], ids['assistant']}
        if any(shifted[p] in forbidden for p in supervised):
            raise ValueError(f"Special/role/EOS token supervised {row['conversation_id']}")
        assistant_turns = sum(m['role'] == 'assistant' for m in row['messages'])
        if len(end_targets) != assistant_turns or shifted[-2] != ids['end'] or inputs[-1] != ids['end']:
            raise ValueError(f"END supervision mismatch {row['conversation_id']}")
        # Each assistant turn's content is supervised completely and decodes back intact.
        cursor = 0
        for message in row['messages']:
            content = list(tokenizer.encode(message['content'], add_special_tokens=False).ids)
            span = inputs[cursor + 1: cursor + 1 + len(content)]
            if span != content:
                raise ValueError(f"Role segment misalignment {row['conversation_id']}")
            labels_span = shifted[cursor: cursor + len(content) + 1]
            if message['role'] == 'assistant':
                if labels_span != [*content, ids['end']]:
                    raise ValueError(f"Assistant supervision incomplete {row['conversation_id']}")
                if tokenizer.decode(content, skip_special_tokens=False).strip() != message['content'].strip() and \
                        normalize(tokenizer.decode(content, skip_special_tokens=False)) != normalize(message['content']):
                    raise ValueError(f"Assistant decode round-trip failure {row['conversation_id']}")
            elif any(label != IGNORE_LABEL for label in labels_span):
                raise ValueError(f"Non-assistant content supervised {row['conversation_id']}")
            cursor += len(content) + 2
        batch = collate_chat([sample], fixed_length=MAXIMUM_TOKENS)
        if not (batch['labels'][0, len(inputs):] == IGNORE_LABEL).all() or batch['attention_mask'][0, len(inputs):].any():
            raise ValueError(f"Padding supervised {row['conversation_id']}")
        records.append(dict(conversation_id=row['conversation_id'], source=row['source'],
            category=row.get('category'), tokens=len(inputs), assistant_supervised_tokens=len(supervised),
            end_targets=len(end_targets), end_target_positions=end_targets,
            premature_end_probe_positions=content_targets,
            input_ids_sha256=hashlib.sha256(json.dumps(inputs).encode()).hexdigest(),
            shifted_labels_sha256=hashlib.sha256(json.dumps(shifted).encode()).hexdigest()))
    return dict(dataset_sha256=dataset.sha256, audit=dict(dataset.audit), token_ids=ids,
                vocabulary_size=vocabulary), records


def token_accounting(records: list[dict]) -> dict:
    def total(items):
        return dict(examples=len(items), serialized_tokens=sum(r['tokens'] for r in items),
                    supervised_tokens=sum(r['assistant_supervised_tokens'] for r in items),
                    end_targets=sum(r['end_targets'] for r in items))
    by_source = defaultdict(list)
    by_category = defaultdict(list)
    for record in records:
        by_source[record['source']].append(record)
        by_category[record['category']].append(record)
    return dict(total=total(records), by_source={k: total(v) for k, v in sorted(by_source.items())},
                by_category={k: total(v) for k, v in sorted(by_category.items())})


# ---------------------------------------------------------------- manifest

def presentation_orders(dataset) -> list[dict]:
    """Seed 73 then 74, via the real sampler (bucket size 1); cross-checked independently."""
    from llm_data.chat_dataset import ConversationSampler
    sampler = ConversationSampler(dataset, seed=EPOCH_SEEDS[0], balanced=False, bucket_size=1)
    orders = []
    for epoch, seed in enumerate(EPOCH_SEEDS):
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))
        reference = list(range(len(dataset)))
        random.Random(seed).shuffle(reference)
        if indices != reference:
            raise ValueError('ConversationSampler order differs from the seeded reference shuffle.')
        orders.append(dict(epoch=epoch, seed=seed, indices=indices,
                           conversation_ids=[dataset.index[i].identity for i in indices]))
    return orders


def lineage_exposure() -> dict:
    """Did this checkpoint's own lineage ever train on chat-sft-v1? Read lineage records only."""
    job = read(FOUNDATION / 'config/job.json')
    initialization = job['spec']['initialization']
    data = job['spec']['data']
    base_dir = Path(initialization['path']).parent
    evidence = {}
    for name in ('lineage.json', 'manifest.json', 'metadata.json', 'README.md'):
        if (base_dir / name).exists():
            evidence[name] = (base_dir / name).read_text()[:4000]
    corpus = str(data.get('corpus_path', ''))
    return dict(initialization=initialization, foundation_corpus=corpus,
                foundation_run_targets_include_chat=False,
                foundation_basis='Diagnostic source-index accounting: all 117,440,512 foundation targets scanned, zero role/END targets.',
                base_lineage_records=evidence,
                classification=('historical_sft_exposure flags mark records seen by earlier Codexa SFT models; '
                                'this 934M lineage record shows no chat-sft-v1 training, but base lineage records are '
                                'not exhaustive, so records are still reported as historically exposed, never unseen.'))


def freeze(directory: Path) -> dict:
    directory = Path(directory)
    status = read(directory / 'evidence/selection-status.json')
    if not status['complete']:
        raise RuntimeError(f"Selection incomplete; refusing to freeze: {status['shortfall']}")
    from llm_data.chat_dataset import IndexedChatDataset, verify_split_isolation
    from llm_tokenizer.tokenizer import load_tokenizer
    identity = tokenizer_identity()
    if digest(identity['path']) != identity['sha256']:
        raise ValueError('Tokenizer identity drift.')
    tokenizer = load_tokenizer(identity['path'])
    train_path, validation_path = directory / 'data/train.jsonl', directory / 'data/validation.jsonl'
    train_meta, train_records = verify_records(train_path, tokenizer)
    validation_meta, validation_records = verify_records(validation_path, tokenizer)
    train = IndexedChatDataset(train_path, tokenizer, context=MAXIMUM_TOKENS)
    validation = IndexedChatDataset(validation_path, tokenizer, context=MAXIMUM_TOKENS)
    isolation = verify_split_isolation({'train': train, 'validation': validation})
    train_rows, validation_rows = list(jsonlines(train_path)), list(jsonlines(validation_path))
    composition = Counter(r['source'] for r in train_rows)
    if len(train_rows) != 512 or len({r['conversation_id'] for r in train_rows}) != 512 or \
            composition != Counter(oasst1=224, ultrachat=224, authored=64):
        raise ValueError(f'Training composition violation: {dict(composition)}')
    if len(validation_rows) != 32 or Counter(r['source'] for r in validation_rows) != Counter(oasst1=16, ultrachat=16):
        raise ValueError('Validation composition violation.')
    validation_text = set().union(*(substantial_messages(r) for r in validation_rows))
    if any(substantial_messages(r) & validation_text for r in train_rows):
        raise ValueError('Substantial normalized text crosses train/validation.')
    orders = presentation_orders(train)
    write_rows(directory / 'evidence/train-masks.jsonl', train_records)
    write_rows(directory / 'evidence/validation-masks.jsonl', validation_records)
    provenance = {r['conversation_id']: r for r in jsonlines(directory / 'data/train-provenance.jsonl')}
    provenance.update({r['conversation_id']: r for r in jsonlines(directory / 'data/validation-provenance.jsonl')})
    masks = {r['conversation_id']: r for r in train_records + validation_records}

    def entry(row, split):
        record, mask = provenance[row['conversation_id']], masks[row['conversation_id']]
        return dict(conversation_id=row['conversation_id'], split=split, source=row['source'],
                    category=row['category'], root_id=row['root_id'], duplicate_group=row['duplicate_group'],
                    content_sha256=row['content_sha256'], historical_sft_exposure=row['historical_sft_exposure'],
                    provenance=record['provenance'], admission=record['review'],
                    contamination=record['contamination'], serialized_tokens=mask['tokens'],
                    assistant_supervised_tokens=mask['assistant_supervised_tokens'],
                    end_target_positions=mask['end_target_positions'],
                    input_ids_sha256=mask['input_ids_sha256'], shifted_labels_sha256=mask['shifted_labels_sha256'])

    accounting = dict(train=token_accounting(train_records), validation=token_accounting(validation_records))
    accounting['train_two_passes'] = {k: 2 * v for k, v in accounting['train']['total'].items()}
    manifest = dict(format='codexa-sft-phase4-manifest-v1', frozen_utc=datetime.now(timezone.utc).isoformat(),
        immutable_policy='Never edit; any change requires a new manifest and a new authorization.',
        tokenizer=dict(identity, special_token_ids=train_meta['token_ids'],
                       vocabulary_size=train_meta['vocabulary_size']),
        serialization=dict(format='chat-v1', maximum_serialized_tokens=MAXIMUM_TOKENS, truncation=False,
            supervised='assistant content + END (7); roles, user, system, padding ignored (-100)',
            shift='labels[t] = input_ids[t+1]; exactly once', end_vs_eos='END 7 terminates turns; EOS 2 never supervised'),
        datasets=dict(train=dict(path=str(train_path), sha256=train.sha256, examples=len(train_rows)),
                      validation=dict(path=str(validation_path), sha256=validation.sha256, examples=len(validation_rows))),
        composition=dict(train=dict(composition), validation=dict(Counter(r['source'] for r in validation_rows))),
        categories=dict(train=dict(Counter(r['category'] for r in train_rows)),
                        validation=dict(Counter(r['category'] for r in validation_rows))),
        token_accounting=accounting,
        orders=orders, presentations=sum(len(o['indices']) for o in orders),
        split_isolation=isolation,
        contamination_scope=dict(benchmarks='48 historical + 16 conversational diagnostic prompts and expected answers',
            hard=['exact normalized user question', 'any shared 8-word shingle'],
            flagged='>=30% coverage of a benchmark text 5-gram set (possible semantic overlap; not proof of independence)',
            train_validation=['conversation id', 'root', 'duplicate group', 'content hash',
                              f'substantial normalized message (>= {SUBSTANTIAL_WORDS} words)', '8-word shingle',
                              f'near duplicate 5-gram Jaccard >= {NEAR_DUPLICATE_JACCARD}']),
        historical_exposure=dict(train=sum(bool(r['historical_sft_exposure']) for r in train_rows),
                                 validation=sum(bool(r['historical_sft_exposure']) for r in validation_rows),
                                 lineage=lineage_exposure()),
        records=[entry(r, 'train') for r in train_rows] + [entry(r, 'validation') for r in validation_rows])
    if manifest['presentations'] != 1024:
        raise ValueError('Two passes must yield exactly 1024 presentations.')
    manifest['manifest_sha256'] = canonical_sha256(manifest)
    write(directory / 'manifest/phase4-manifest.json', manifest)
    write(directory / 'evidence/token-mask-report.json', dict(train=train_meta, validation=validation_meta,
        accounting=accounting, checks=['IndexedChatDataset(verify_tokens=True) fingerprint/count verification',
        'serializer equals dataset tensors', 'single shift', 'roles/user/system/PAD/BOS/EOS never supervised',
        'one END target per assistant turn, final target END', 'assistant content fully supervised and decodes intact',
        'all IDs inside vocabulary', '<= 512 tokens', 'fixed-length padding unsupervised and unmasked']))
    return manifest


def verify_manifest(path: Path) -> dict:
    """Recompute content, masks, order and tokenizer identity; any drift raises."""
    manifest = read(path)
    claimed = manifest.pop('manifest_sha256')
    if canonical_sha256(manifest) != claimed:
        raise ValueError('Manifest content hash mismatch.')
    from llm_data.chat_dataset import IndexedChatDataset
    from llm_tokenizer.tokenizer import load_tokenizer
    tokenizer_path = manifest['tokenizer']['path']
    if digest(tokenizer_path) != manifest['tokenizer']['sha256']:
        raise ValueError('Tokenizer identity drift.')
    tokenizer = load_tokenizer(tokenizer_path)
    datasets = {}
    for split in ('train', 'validation'):
        info = manifest['datasets'][split]
        dataset = IndexedChatDataset(Path(info['path']), tokenizer, context=MAXIMUM_TOKENS)
        if dataset.sha256 != info['sha256']:
            raise ValueError(f'{split} dataset content drift.')
        _, records = verify_records(Path(info['path']), tokenizer)
        expected = [r for r in manifest['records'] if r['split'] == split]
        for record, entry in zip(records, expected):
            if (record['conversation_id'], record['input_ids_sha256'], record['shifted_labels_sha256']) != (
                    entry['conversation_id'], entry['input_ids_sha256'], entry['shifted_labels_sha256']):
                raise ValueError(f"Token/mask drift {entry['conversation_id']}")
        if len(records) != len(expected):
            raise ValueError(f'{split} record count drift.')
        datasets[split] = dataset
    if [dict(o, indices=o['indices']) for o in presentation_orders(datasets['train'])] != manifest['orders']:
        raise ValueError('Presentation order drift.')
    return dict(verified=True, manifest_sha256=claimed, train_sha256=datasets['train'].sha256,
                validation_sha256=datasets['validation'].sha256, tokenizer_sha256=manifest['tokenizer']['sha256'],
                presentations=manifest['presentations'])
