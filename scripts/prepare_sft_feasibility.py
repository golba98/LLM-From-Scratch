#!/usr/bin/env python3
"""Prepare and validate conversational feasibility evidence. No training actions."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
DIAGNOSTIC = ROOT / 'outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256'
RUN = ROOT / 'outputs/training/codexa-934m-foundation-5h-v1'
EXPECTED = 'ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e'
BASE = ROOT / 'outputs/preparation/codexa-934m-conversational-feasibility'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def jsonlines(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    return dict(argv=argv, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def normalize(text):
    return ' '.join(re.findall(r"[\w']+", text.casefold()))


def shingles(text, n=8):
    words = normalize(text).split()
    return {tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1))}


def benchmark(directory):
    records = [x for x in jsonlines(DIAGNOSTIC/'evidence/generation/generations-final.jsonl')
               if x['decode'] in ('greedy','nucleus')]
    assert len(records) == 128
    return records


def bootstrap():
    directory = BASE / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('data','config','evaluation','evidence'):
        (directory/name).mkdir()
    assets = read(ROOT/'artifacts.local.json')
    source = Path(assets['asset_root'])/'data/processed/general-chat-sft-v2'
    job = read(RUN/'config/job.json')
    checkpoint = RUN/'checkpoints/slot-0/checkpoint.pt'
    assert digest(checkpoint) == EXPECTED
    protected = [checkpoint, Path(job['spec']['tokenizer']['path']), RUN/'config/job.json',
                 ROOT/'configs/1b.yaml', DIAGNOSTIC/'REPORT.md', source/'dataset_manifest.json',
                 source/'train.jsonl', source/'validation.jsonl', source/'test.jsonl']
    # Bind all training records, original evaluations, and the frozen validation selectors.
    protected.extend(p for tree in ('config','metrics','evaluations','supervisor')
                     for p in (RUN/tree).rglob('*') if p.is_file())
    protected = sorted(set(protected))
    identities = {str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in protected}
    original = read(source/'dataset_manifest.json')
    for split in ('train','validation','test'):
        assert identities[str(source/(split+'.jsonl'))]['sha256'] == original['output_sha256'][split]
    git = {p.name:command(['git','-C',str(p),'status','--porcelain']) for p in WORKSPACE.iterdir() if (p/'.git').exists()}
    modules = {}
    for name in ('llm_training.training','llm_training.chat_run','llm_training.config',
                 'llm_data.chat_dataset','llm_tokenizer.sft','llm_tokenizer.tokenizer',
                 'llm_architecture.model','llm_inference.generate'):
        path = Path(importlib.util.find_spec(name).origin)
        source_path = WORKSPACE / {'llm_training':'LLM-Training','llm_data':'LLM-Data','llm_tokenizer':'LLM-Tokenizer','llm_architecture':'LLM-Architecture','llm_inference':'LLM-Inference'}[name.split('.')[0]] / 'src' / Path(*name.split('.')).with_suffix('.py')
        modules[name] = dict(path=str(path),sha256=digest(path),source=str(source_path),source_sha256=digest(source_path))
    audit = command([str(WORKSPACE/'audit'),'doctor','--json'])
    write(directory/'evidence/environment.json',dict(protected=identities,git=git,modules=modules,
        hardware=command(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv']),
        ram=command(['free','-b']),disk=command(['df','-B1',str(ROOT)]),audit=audit,
        architecture=job['spec']['model'],tokenizer=job['spec']['tokenizer'],source_directory=str(source),
        preparation_only=True,optimizer_constructed=False,training_updates=0))
    write_rows(directory/'evaluation/step-zero-generations.jsonl',benchmark(directory))
    return directory


def candidate_pool(directory):
    env = read(directory/'evidence/environment.json')
    source = Path(env['source_directory'])
    cases = benchmark(directory)
    questions = {normalize(re.sub(r'<[^>]+>',' ',x['prompt'])) for x in cases}
    forbidden = set().union(*(shingles(re.sub(r'<[^>]+>',' ',x['prompt'])) for x in cases))
    result=[]; counts=Counter()
    for split in ('train','validation'):
        pool = {'oasst1':[], 'ultrachat':[]}
        for ordinal, row in enumerate(jsonlines(source/(split+'.jsonl'))):
            counts[f'{split}/{row["source"]}/seen'] += 1
            # Short, complete candidates keep manual correctness review tractable.
            if row['tokens'] > 512 or not 8 <= row['assistant_tokens'] <= 96:
                continue
            texts = [m['content'] for m in row['messages']]
            letters = [c for c in '\n'.join(texts) if c.isalpha()]
            if not letters or sum(not c.isascii() for c in letters)/len(letters) > .02:
                continue
            if any(normalize(m['content']) in questions for m in row['messages'] if m['role']=='user'):
                continue
            if any(shingles(text)&forbidden for text in texts):
                continue
            if any(marker in '\n'.join(texts).casefold() for marker in ('as an ai language','as a language model','i cannot browse','i don\'t have access to','consult a doctor','consult a lawyer')):
                continue
            item=dict(record=row,split=split,ordinal=ordinal,source_file=str(source/(split+'.jsonl')),
                      ranking_sha256=hashlib.sha256(f'73:{row["source"]}:{row["conversation_id"]}'.encode()).hexdigest())
            pool[row['source']].append(item)
        for name, entries in pool.items():
            entries.sort(key=lambda x:x['ranking_sha256'])
            result.extend(entries)
            counts[f'{split}/{name}/eligible'] = len(entries)
    write_rows(directory/'data/candidates.jsonl',result)
    write(directory/'evidence/candidate-counts.json',dict(counts))
    print(json.dumps(dict(counts),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('initialize','candidates','finalize','baseline','validate','report'))
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    if args.action=='initialize':
        directory=bootstrap();candidate_pool(directory);print(directory);return
    directory=args.directory.resolve() if args.directory else None
    if directory is None or not directory.is_relative_to(BASE.resolve()) or directory==BASE.resolve():
        parser.error('Use a dedicated feasibility preparation directory')
    if args.action=='candidates':candidate_pool(directory)
    else:
        from codexa_workspace import sft_feasibility
        getattr(sft_feasibility,args.action)(directory)


if __name__=='__main__':
    main()
