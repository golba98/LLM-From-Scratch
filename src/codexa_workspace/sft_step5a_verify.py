"""Step 5A: independent verification of the frozen SFT pilot inputs. Read-only and construction-free.

Nothing here imports a trainer or an optimizer, or writes to a checkpoint or dataset. The
Phase 4 output directory is a read-only input: any correction is written as a new,
explicitly versioned artifact in the Step 5A directory.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess

from codexa_workspace import sft_phase4_data as data

ROOT = data.ROOT
WORKSPACE = data.WORKSPACE
PHASE4 = data.BASE / '20261012T145429Z-643f4379'
STEP5A_BASE = ROOT / 'outputs/preparation/codexa-934m-sft-step5a'
SOURCE = Path('/home/k9-vortex/Development/3-Python/31-LLM Source')
REPOSITORIES = ('LLM-From-Scratch', 'LLM-Architecture', 'LLM-Tokenizer', 'LLM-Data', 'LLM-Training',
                'LLM-Inference', 'LLM-Memory', 'LLM-Specialist', 'workspace-infrastructure')
PHASE4_PROTECTED = ('manifest/phase4-manifest.json', 'data/train.jsonl', 'data/validation.jsonl',
                    'evaluation/step-zero/summary.json', 'evaluation/step-zero/generations.jsonl',
                    'evaluation/step-zero/determinism-repeat.jsonl', 'evaluation/step-zero/gold-per-record.jsonl',
                    'config/job.json', 'config/budget-recommendation.json', 'SFT_PHASE4_READINESS_REPORT.md')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def repository_snapshot() -> dict:
    """HEAD, branch and porcelain status of every workspace repository."""
    result = {}
    for name in REPOSITORIES:
        repo = WORKSPACE / name
        porcelain = _git(repo, 'status', '--porcelain', '--untracked-files=all').splitlines()
        result[name] = dict(head=_git(repo, 'rev-parse', 'HEAD'), branch=_git(repo, 'branch', '--show-current'),
                            dirty=len(porcelain), status=porcelain,
                            index_locked=(repo / '.git/index.lock').exists())
    return dict(captured_utc=now(), repositories=result)


# ---------------------------------------------------------------- preservation

def protected_artifacts() -> dict[str, str]:
    """Codex's 88 protected artifacts plus the frozen Phase 4 inputs: path -> expected SHA-256."""
    prior = data.read(data.PRIOR / 'evidence/environment.json')['protected']
    expected = {path: identity['sha256'] for path, identity in prior.items()}
    phase4 = data.read(PHASE4 / 'evidence/preservation-check.json')
    expected.setdefault(str(data.FOUNDATION / 'checkpoints/slot-0/checkpoint.pt'), phase4['checkpoint_sha256'])
    return expected


def phase4_artifacts() -> dict[str, str]:
    return {str(PHASE4 / relative): data.digest(PHASE4 / relative) for relative in PHASE4_PROTECTED}


def preservation(directory: Path, label: str, *, phase4_expected: dict | None = None) -> dict:
    """Rehash every protected artifact. 'before' also fixes the Phase 4 artifact hashes for 'after'."""
    checks = [dict(path=path, expected=sha, actual=data.digest(path)) for path, sha in sorted(protected_artifacts().items())]
    current = phase4_artifacts()
    expected4 = phase4_expected or current
    checks += [dict(path=path, expected=expected4[path], actual=current[path]) for path in sorted(current)]
    for check in checks:
        check['unchanged'] = check['expected'] == check['actual']
    result = dict(label=label, checked_utc=now(), artifacts=len(checks),
                  unchanged=sum(c['unchanged'] for c in checks), all_unchanged=all(c['unchanged'] for c in checks),
                  checkpoint_sha256=data.digest(data.FOUNDATION / 'checkpoints/slot-0/checkpoint.pt'),
                  checks=checks)
    data.write(Path(directory) / f'evidence/preservation-{label}.json', result)
    return {k: v for k, v in result.items() if k != 'checks'}


# ---------------------------------------------------------------- dataset (independent of Phase 4 accounting)

def _canonical_content_sha256(messages: list[dict]) -> str:
    """The chat_dataset fingerprint, recomputed here from the raw source rows."""
    import hashlib
    from llm_data.data.general_chat import normalized
    canonical = [(message['role'], normalized(message['content'])) for message in messages]
    return hashlib.sha256(json.dumps(canonical).encode()).hexdigest()


def _source_rows(records: list[dict]) -> dict[str, dict]:
    """Re-read each manifest record from its recorded source file: ordinal lines, or id for authored."""
    wanted: dict[str, dict[int, str]] = {}
    by_id: dict[str, set] = {}
    for record in records:
        provenance = record['provenance']
        if 'source_ordinal' in provenance:
            wanted.setdefault(provenance['source_file'], {})[provenance['source_ordinal']] = record['conversation_id']
        else:
            by_id.setdefault(provenance['source_file'], set()).add(record['conversation_id'])
    found = {}
    for path, ordinals in wanted.items():
        remaining = dict(ordinals)
        with open(path, encoding='utf-8') as stream:
            for ordinal, line in enumerate(stream):
                if ordinal in remaining:
                    row = json.loads(line)
                    if row.get('conversation_id') != remaining[ordinal]:
                        raise ValueError(f'{path}:{ordinal} holds {row.get("conversation_id")}, '
                                         f'not {remaining[ordinal]}')
                    found[row['conversation_id']] = row
                    del remaining[ordinal]
                    if not remaining:
                        break
        if remaining:
            raise ValueError(f'{len(remaining)} provenance ordinals missing from {path}')
    for path, identities in by_id.items():
        for row in data.jsonlines(path):
            if row['conversation_id'] in identities:
                found[row['conversation_id']] = row
    return found


def _masks(path: Path, tokenizer, token_ids: dict) -> dict[str, dict]:
    """Per-record supervision recomputed through llm_data's dataset and collation."""
    from llm_data.chat_dataset import IndexedChatDataset, collate_chat
    from llm_tokenizer.sft import IGNORE_LABEL
    dataset = IndexedChatDataset(path, tokenizer, context=data.MAXIMUM_TOKENS, verify_tokens=True)
    never = {token_ids[k] for k in ('pad', 'bos', 'eos', 'system', 'user', 'assistant')}
    result = {}
    for item in range(len(dataset)):
        batch = collate_chat([dataset[item]])
        ids, labels = batch['input_ids'][0], batch['labels'][0]
        supervised = labels != IGNORE_LABEL
        positions = supervised.nonzero().flatten().tolist()
        result[batch['identities'][0]] = dict(
            serialized=int(ids.numel()), supervised=int(supervised.sum()),
            end_targets=int((labels == token_ids['end']).sum()),
            forbidden_targets=int(sum(int(labels[p]) in never for p in positions)),
            shifted_once=all(int(labels[p]) == int(ids[p + 1]) for p in positions),
            last_unsupervised=bool(labels[-1] == IGNORE_LABEL),
            final_target_is_end=bool(positions and int(labels[positions[-1]]) == token_ids['end']))
    return dict(sha256=dataset.sha256, records=result)


def verify_dataset(manifest_path: Path) -> dict:
    """Independently re-derive identities, content, masks, counts, orders and isolation."""
    from llm_data.chat_dataset import ConversationSampler, IndexedChatDataset
    from llm_tokenizer.tokenizer import load_tokenizer
    manifest = data.read(manifest_path)
    tokenizer_path = Path(manifest['tokenizer']['path'])
    if data.digest(tokenizer_path) != manifest['tokenizer']['sha256']:
        raise ValueError('Tokenizer differs from the frozen manifest.')
    tokenizer = load_tokenizer(tokenizer_path)
    from llm_tokenizer import tokenizer as names
    token_ids = {key: tokenizer.token_to_id(getattr(names, f'{key.upper()}_TOKEN'))
                 for key in ('pad', 'bos', 'eos', 'system', 'user', 'assistant', 'end')}
    if None in token_ids.values():
        raise ValueError(f'Tokenizer lacks a chat-v1 special token: {token_ids}')
    records = manifest['records']
    sources = _source_rows(records)
    content = []
    for record in records:
        row = sources.get(record['conversation_id'])
        actual = _canonical_content_sha256(row['messages']) if row else None
        content.append(dict(conversation_id=record['conversation_id'], expected=record['content_sha256'], actual=actual,
                            matches=actual == record['content_sha256'],
                            complete_assistant_turns=bool(row) and row['messages'][-1]['role'] == 'assistant'))
    splits, frozen_rows = {}, {}
    for split in ('train', 'validation'):
        path = Path(manifest['datasets'][split]['path'])
        if data.digest(path) != manifest['datasets'][split]['sha256']:
            raise ValueError(f'{split} file differs from the frozen manifest')
        frozen_rows[split] = {row['conversation_id']: row for row in data.jsonlines(path)}
        splits[split] = _masks(path, tokenizer, token_ids)
    per_record = {r['conversation_id']: r for r in records}
    mismatched_counts = [identity for split in splits for identity, m in splits[split]['records'].items()
                         if (m['serialized'], m['supervised'], m['end_targets'])
                         != (per_record[identity]['serialized_tokens'], per_record[identity]['assistant_supervised_tokens'],
                             len(per_record[identity]['end_target_positions']))]
    frozen_vs_source = [identity for split in frozen_rows for identity, row in frozen_rows[split].items()
                        if _canonical_content_sha256(row['messages']) != per_record[identity]['content_sha256']]

    def totals(split):
        values = splits[split]['records'].values()
        return dict(examples=len(values), serialized_tokens=sum(v['serialized'] for v in values),
                    supervised_tokens=sum(v['supervised'] for v in values),
                    end_targets=sum(v['end_targets'] for v in values))
    train_total, validation_total = totals('train'), totals('validation')
    dataset = IndexedChatDataset(Path(manifest['datasets']['train']['path']), tokenizer, context=data.MAXIMUM_TOKENS)
    sampler = ConversationSampler(dataset, seed=data.EPOCH_SEEDS[0], bucket_size=1)
    orders = []
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        orders.append(list(sampler))
    train = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    isolation = {field: len({r[field] for r in train} & {r[field] for r in validation})
                 for field in ('conversation_id', 'root_id', 'duplicate_group', 'content_sha256')}
    isolation['source_root'] = len({(r['source'], r['root_id']) for r in train}
                                   & {(r['source'], r['root_id']) for r in validation})
    flags = dict(
        content_matches_source=all(c['matches'] for c in content),
        complete_assistant_turns=all(c['complete_assistant_turns'] for c in content),
        frozen_rows_match_manifest=not frozen_vs_source,
        counts_match_manifest=not mismatched_counts,
        composition=(Counter(r['source'] for r in train) == Counter(oasst1=224, ultrachat=224, authored=64)
                     and Counter(r['source'] for r in validation) == Counter(oasst1=16, ultrachat=16)),
        unique_identities=len({r['conversation_id'] for r in records}) == len(records) == 544,
        end_targets_present=all(m['end_targets'] >= 1 and m['final_target_is_end']
                                for s in splits.values() for m in s['records'].values()),
        no_forbidden_targets=all(m['forbidden_targets'] == 0 for s in splits.values() for m in s['records'].values()),
        shifted_exactly_once=all(m['shifted_once'] and m['last_unsupervised']
                                 for s in splits.values() for m in s['records'].values()),
        orders_match_manifest=[o['indices'] for o in manifest['orders']] == orders,
        orders_are_permutations=all(sorted(o) == list(range(512)) for o in orders),
        presentations=sum(len(o) for o in orders) == manifest['presentations'] == 1024,
        accounting_matches=(train_total == manifest['token_accounting']['train']['total']
                            and validation_total == manifest['token_accounting']['validation']['total']),
        split_isolation=all(v == 0 for v in isolation.values()))
    return dict(manifest=str(manifest_path), manifest_sha256=manifest['manifest_sha256'], token_ids=token_ids,
                train=train_total, validation=validation_total,
                train_two_passes=dict(presentations=sum(len(o) for o in orders),
                                      supervised_tokens=2 * train_total['supervised_tokens'],
                                      end_targets=2 * train_total['end_targets']),
                isolation=isolation, mismatched_counts=mismatched_counts, frozen_vs_source=frozen_vs_source,
                content_mismatches=[c for c in content if not c['matches']], checks=flags,
                verified=all(flags.values()), dataset_sha256={k: v['sha256'] for k, v in splits.items()})


def validation_exception(manifest_path: Path, conversation_id: str) -> dict:
    """Isolation and exposure evidence for the one validation record taken from the test split."""
    manifest = data.read(manifest_path)
    records = {r['conversation_id']: r for r in manifest['records']}
    record = records[conversation_id]
    rows = {split: {row['conversation_id']: row for row in data.jsonlines(manifest['datasets'][split]['path'])}
            for split in ('train', 'validation')}
    target = rows['validation'][conversation_id]
    target_shingles = data.shingles(data.conversation_text(target), data.NEAR_DUPLICATE_N)
    candidates = {identity: dict(record=row) for split in rows for identity, row in rows[split].items()}
    template = data.template_shingles(candidates)
    similarity = sorted(((data.jaccard(target_shingles, data.shingles(data.conversation_text(row), data.NEAR_DUPLICATE_N)),
                          data.jaccard(target_shingles - template,
                                       data.shingles(data.conversation_text(row), data.NEAR_DUPLICATE_N) - template),
                          identity) for identity, row in rows['train'].items()), reverse=True)[:5]
    train = [r for r in manifest['records'] if r['split'] == 'train']
    # Sibling SFT exposure: does this thread appear in the general-chat-v2 training split?
    v2_train = Path(record['provenance']['source_file']).with_name('train.jsonl')
    in_v2_train = dict(root=0, duplicate_group=0, content_sha256=0)
    with open(v2_train, encoding='utf-8') as stream:
        for line in stream:
            if record['root_id'] in line or record['duplicate_group'] in line or record['content_sha256'] in line:
                row = json.loads(line)
                in_v2_train['root'] += row.get('root_id') == record['root_id']
                in_v2_train['duplicate_group'] += row.get('duplicate_group') == record['duplicate_group']
                in_v2_train['content_sha256'] += row.get('content_sha256') == record['content_sha256']
    shared = dict(conversation_id=conversation_id in {r['conversation_id'] for r in train},
                  root=record['root_id'] in {r['root_id'] for r in train},
                  duplicate_group=record['duplicate_group'] in {r['duplicate_group'] for r in train},
                  content_sha256=record['content_sha256'] in {r['content_sha256'] for r in train})
    near_duplicate = similarity[0][0] >= data.NEAR_DUPLICATE_JACCARD
    return dict(conversation_id=conversation_id, source_file=record['provenance']['source_file'],
                upstream_source_path=record['provenance']['source_path'], first_pass_flags=record['admission']['first']['flags'],
                shared_with_training=shared, contamination=record['contamination'],
                target_five_gram_shingles=len(target_shingles),
                nearest_training=[dict(jaccard=round(a, 4), jaccard_without_templates=round(b, 4), conversation_id=i)
                                  for a, b, i in similarity],
                near_duplicate_of_training=near_duplicate, in_general_chat_v2_train=in_v2_train,
                historical_sft_exposure_flag=record['historical_sft_exposure'],
                exposure_flag_basis=('row flag from the source corpus (chat-sft-v1 OASST1 train was used by earlier '
                                     'sibling SFT models); the v2 train split is checked separately above'),
                isolation_policy_satisfied=not any(shared.values()) and not near_duplicate and
                not record['contamination']['hard_reject'],
                replacement_required=any(shared.values()) or near_duplicate or record['contamination']['hard_reject'])


def lineage_exposure(manifest_path: Path) -> dict:
    """Is any earlier SFT model an ancestor of the initialization checkpoint? Evidence, not assumption."""
    manifest = data.read(manifest_path)
    foundation = data.read(data.FOUNDATION / 'config/job.json')['spec']
    base_metadata = data.read(SOURCE / 'logs/codexa-900m-base-v1/run_metadata.json')
    mixed = data.read(SOURCE / 'data/tokenized/base-v1/mixed/token_data_manifest.json')
    continuation = data.read(SOURCE / 'data/tokenized/base-v1/continuation-5h/token_data_manifest.json')
    base_arguments = base_metadata['command_line_arguments']
    lineage = [
        dict(model='codexa-900m-base-v1', initialization=base_metadata.get('resume_checkpoint') or 'random (resume=None)',
             resumes=[e['resume_checkpoint'] for e in base_metadata.get('resume_events', [])],
             train_token_file=base_arguments['train_token_file'], optimizer_steps=base_metadata['completed_optimizer_steps']),
        dict(model='codexa-934m-foundation-5h-v1', initialization=foundation['initialization']['path'],
             corpus=foundation['data']['corpus_path']),
    ]
    corpus_sources = sorted(mixed['documents_by_source'])
    same_corpus = continuation['output_checksums']['train'] == mixed['output_checksums']['train']
    only_base = all(Path(r).name == 'latest.pt' and 'codexa-900m-base-v1' in r for r in lineage[0]['resumes'])
    chat_free = (corpus_sources == ['fineweb_edu', 'wikipedia'] and same_corpus and only_base
                 and base_arguments['train_token_file'].endswith('base-v1/mixed/train.bin')
                 and foundation['data']['corpus_path'].endswith('base-v1/mixed/train.bin'))
    train = [r for r in manifest['records'] if r['split'] == 'train']
    validation = [r for r in manifest['records'] if r['split'] == 'validation']
    return dict(lineage=lineage, corpus_sources=corpus_sources, continuation_is_range_of_mixed=same_corpus,
                base_resumed_only_from_itself=only_base, sft_model_in_initialization_lineage=not chat_free,
                flags=dict(train_historical_sft_exposure=sum(r['historical_sft_exposure'] for r in train),
                           validation_historical_sft_exposure=sum(r['historical_sft_exposure'] for r in validation)),
                classification=('historical_sft_exposure marks records seen by earlier sibling Codexa SFT models. '
                                'Those models are not ancestors of the initialization checkpoint: its lineage was '
                                'trained only on base-v1/mixed (FineWeb-Edu + Wikipedia). Incidental web-crawl '
                                'overlap with OASST1/UltraChat text cannot be excluded, so no record is called unseen.')
                if chat_free else 'Lineage evidence incomplete: exposure of the initialization lineage is UNKNOWN.')


# ---------------------------------------------------------------- step-zero binding

def step_zero_binding(summary_path: Path, manifest_path: Path, job_spec: dict | None = None) -> dict:
    """Verify the sealed step-zero summary is bound to the frozen inputs (static; no GPU)."""
    from codexa_workspace import sft_pilot_evaluation as evaluation
    summary = data.read(summary_path)
    # Phase 4 sealed the hash over every field except the seal time and the hash itself.
    body = {k: v for k, v in summary.items() if k not in ('summary_sha256', 'sealed_utc')}
    manifest = data.read(manifest_path)
    sealed = datetime.fromisoformat(summary['sealed_utc']).timestamp()
    sources = {name: Path(path) for name, path in (
        ('sft_pilot_evaluation', ROOT / 'src/codexa_workspace/sft_pilot_evaluation.py'),
        ('llm_inference.generate', WORKSPACE / 'LLM-Inference/src/llm_inference/generate.py'),
        ('llm_architecture.model', WORKSPACE / 'LLM-Architecture/src/llm_architecture/model.py'),
        ('llm_architecture.checkpointing', WORKSPACE / 'LLM-Architecture/src/llm_architecture/checkpointing.py'),
        ('llm_data.chat_dataset', WORKSPACE / 'LLM-Data/src/llm_data/chat_dataset.py'),
        ('llm_tokenizer.sft', WORKSPACE / 'LLM-Tokenizer/src/llm_tokenizer/sft.py'))}
    code = {name: dict(sha256=data.digest(path), modified_before_seal=path.stat().st_mtime < sealed)
            for name, path in sources.items()}
    suites = evaluation.suite_identity()
    checks = dict(
        summary_hash=data.canonical_sha256(body) == summary['summary_sha256'],
        checkpoint=summary['checkpoint']['sha256_before'] == summary['checkpoint']['sha256_after']
        == data.EXPECTED_CHECKPOINT_SHA256,
        manifest=summary['manifest_sha256'] == manifest['manifest_sha256'],
        validation_file=summary['validation_sha256'] == manifest['datasets']['validation']['sha256'],
        suites={k: summary['suites'][k]['sha256'] == suites[k]['sha256'] for k in ('conversational', 'historical')},
        architecture=summary['environment']['architecture']['sha256'] == code['llm_architecture.model']['sha256'],
        no_training=(summary['optimizer_constructed'] is False and summary['backward_calls'] == 0
                     and summary['training_updates'] == 0),
        source_unmodified_since_seal=all(c['modified_before_seal'] for c in code.values()))
    if job_spec is not None:
        checks['job_binds_summary'] = job_spec['evaluation']['step_zero_sha256'] == summary['summary_sha256']
    flat = [v for v in checks.values() if isinstance(v, bool)] + list(checks['suites'].values())
    return dict(summary=str(summary_path), summary_sha256=summary['summary_sha256'], sealed_utc=summary['sealed_utc'],
                tokenizer_sha256=manifest['tokenizer']['sha256'], decoding=summary.get('suites') and
                {k: dict(v.__dict__) for k, v in evaluation.decoding_configs().items()},
                source_code=code, checks=checks, bound=all(flat),
                note='Source binding: file mtimes predate the seal (not cryptographic); a GPU reconfirmation '
                     'under the committed identity provides the token-level binding.')


# ---------------------------------------------------------------- budget (measured, historical, estimated)

THROUGHPUT_EVIDENCE = ROOT / 'documentation/training/CODEXA_900M_CHAT_10H_READINESS.md'


def _documented_probe(name: str) -> dict:
    """The documented full-934M controlled throughput probe (section E); the slowest, largest row wins."""
    rows = []
    for line in THROUGHPUT_EVIDENCE.read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if cells and cells[0] == name:
            rows.append(dict(variant=name, assistant_tokens_per_second=float(cells[1]),
                             context_tokens_per_second=float(cells[2]), median_microbatch_seconds=float(cells[3]),
                             peak_reserved_gib=float(cells[4])))
    if not rows:
        raise ValueError(f'Probe {name!r} not found in {THROUGHPUT_EVIDENCE}')
    return dict(rows[0], median_microbatch_seconds=max(r['median_microbatch_seconds'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows), rows=len(rows))


def budget_analysis(manifest_path: Path, summary_path: Path) -> dict:
    """Compare 900 s and 1,200 s caps. Every input is labelled measured, historical or estimated."""
    import statistics
    manifest, step = data.read(manifest_path), data.read(summary_path)
    timings = step['timings_seconds']
    foundation_rows = [json.loads(line) for line in (data.FOUNDATION / 'metrics/train_metrics.jsonl').read_text().splitlines()]
    optimizer_update = statistics.median(r['optimizer_update_seconds'] for r in foundation_rows)
    writes = [c['write_verify_seconds'] for c in data.read(data.FOUNDATION / 'metrics/report.json')['checkpoints']]
    probes = {name: _documented_probe(name) for name in ('dynamic-short-m1', 'dynamic-long-m1')}
    presentations = manifest['presentations']
    longest = max(r['serialized_tokens'] for r in manifest['records'] if r['split'] == 'train')
    microbatch = probes['dynamic-short-m1']['median_microbatch_seconds']
    forward_backward = presentations * microbatch
    optimizer = 128 * optimizer_update
    conversation = timings['conversation_generation']
    full = conversation + timings['historical_generation'] + timings['gold_validation'] + timings['reference_validation']
    # Greedy is the primary half of the measured greedy+nucleus conversation timing (derived, not separately timed).
    reduced = conversation / 2 + timings['gold_validation'] + timings['reference_validation']
    items = dict(
        startup=dict(seconds=round(timings['load'] + 25, 1), kind='estimate',
                     basis=f"measured weights load {timings['load']:.1f} s + ~25 s process/data/CUDA start (historical)"),
        forward_backward=dict(seconds=round(forward_backward, 1), kind='historical-measured analog',
                              basis=f'{presentations} microbatches x {microbatch} s (documented full-934M probe '
                                    f'dynamic-short-m1; pilot sequences <= {longest} tokens)'),
        optimizer_updates=dict(seconds=round(optimizer, 1), kind='historical-measured',
                               basis=f'128 x median AdamW8bit update {optimizer_update:.3f} s (foundation run, 1,792 updates)'),
        checkpoint_writes=dict(seconds=round(2 * max(writes) + 10, 1), kind='historical-measured + margin',
                               basis=f'2 x max measured write+verify {max(writes):.1f} s (foundation) + 10 s'),
        evaluation_full_round=dict(seconds=round(full, 1), kind='measured', basis='Phase 4 step zero, all suites'),
        evaluation_reduced_round=dict(seconds=round(reduced, 1), kind='derived from measured',
                                      basis='greedy-16 (half of measured greedy+nucleus) + gold + reference'),
        finalization_reserve=dict(seconds=180, kind='fixed', basis='proposal; never reduced'))
    fixed = items['startup']['seconds'] + forward_backward + optimizer + items['checkpoint_writes']['seconds']
    scenarios = {}
    for name, rounds in (('full_at_32_64_128', [full, full, full]), ('reduced_at_32_64_full_at_128', [reduced, reduced, full])):
        work = fixed + sum(rounds)
        scenarios[name] = {f'cap_{cap}': dict(total_with_reserve=round(work + 180, 1), fits=work + 180 <= cap,
                                               slack_seconds=round(cap - work - 180, 1)) for cap in (900, 1200)}
        scenarios[name]['work_seconds'] = round(work, 1)
    recommended = 1200 if scenarios['reduced_at_32_64_full_at_128']['cap_1200']['fits'] else None
    return dict(
        format='codexa-sft-pilot-budget-analysis-v1', items=items, scenarios=scenarios,
        memory=dict(inference_peak_reserved_bytes=dict(value=step['peak_cuda_memory_bytes']['reserved'], kind='measured'),
                    training_peak_reserved_gib=dict(value=probes['dynamic-long-m1']['peak_reserved_gib'],
                                                    kind='historical-measured analog (upper: 2048-token sequences)',
                                                    basis='documented full-934M AdamW8bit probes; short sequences '
                                                          f"{probes['dynamic-short-m1']['peak_reserved_gib']} GiB"),
                    this_pilot='UNVERIFIED until an authorized run: 512-context SFT plus in-run evaluation was never executed'),
        recommendation=dict(total_seconds=recommended, finalization_reserve_seconds=180,
                            schedule='reduced at 32 and 64 (greedy-16, gold, reference), full at 128',
                            rationale=('Every stop gate (greedy-16 termination and repetition, gold END/premature END, '
                                       'reference-CE retention) still runs at 32/64/128 and the full suite runs at 128. '
                                       'The slack absorbs unmeasured 512-context throughput, memory behaviour and '
                                       'checkpoint variance. 900 s cannot hold the full schedule; with the reduced '
                                       'schedule it fits with less margin.'),
                            status='recommendation only; a cap is granted solely by the user approving the job hash'),
        evidence=dict(throughput=str(THROUGHPUT_EVIDENCE), foundation_metrics=str(data.FOUNDATION / 'metrics/train_metrics.jsonl'),
                      step_zero=str(summary_path)))


# ---------------------------------------------------------------- resources (read-only; nothing is terminated)

def resources() -> dict:
    from codexa_workspace import sft_pilot_runtime as runtime
    from codexa_workspace.sft_pilot_launch import supervisor_available
    measured = runtime.measure_resources(ROOT)
    checks = runtime.resource_checks(runtime.canonical_gates(), measured, required_disk_bytes=2 * 5636391032)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    graphics = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=30).stdout
    processes = []
    for line in filter(None, apps):
        pid, rest = line.split(',', 1)
        name, used = rest.rsplit(',', 1)
        processes.append(dict(pid=int(pid), name=Path(name.strip().split()[0]).name,
                              used_mib=int(used) if used.strip().isdigit() else None))
    units = subprocess.run(['systemctl', '--user', 'list-units', '--all', '--no-legend', 'codexa*'],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    return dict(measured_utc=now(), gates=runtime.canonical_gates(), measured=measured, checks=checks,
                startup_headroom_met=all(c['passed'] for c in checks if c['name'] == 'startup_free_vram'),
                host_checks_met=all(c['passed'] for c in checks if c['name'] != 'startup_free_vram'),
                gpu_processes=processes, nvidia_smi_process_table=graphics.split('Processes:')[-1].strip()[:4000],
                codexa_units=units, supervisor=supervisor_available(),
                in_run=dict(free_vram_floor_bytes=runtime.canonical_gates()['in_run_free_vram_bytes'],
                            abort=('CUDA OOM -> FAILED immediately, no save; free VRAM below 1.5 GiB after an update -> '
                                   'MEMORY_HEADROOM_GATE stop; supervisor heartbeat/stall watchdog SIGTERM then SIGKILL')),
                policy='Competing GPU processes are reported, never terminated.')


# ---------------------------------------------------------------- step-zero reconfirmation (inference only)

def reconfirm_step_zero(manifest_path: Path, summary_path: Path) -> dict:
    """Greedy-16 and gold validation under the committed identity; must equal the sealed baseline."""
    import time
    import torch
    from llm_data.chat_dataset import IndexedChatDataset
    from llm_tokenizer.tokenizer import load_tokenizer
    from codexa_workspace import sft_pilot_evaluation as evaluation
    from codexa_workspace import sft_pilot_runtime as runtime
    summary = data.read(summary_path)
    manifest = data.read(manifest_path)
    measured = runtime.measure_resources(ROOT)
    if (measured['vram'] or {}).get('free_mib', 0) < 8192:
        raise RuntimeError(f'Insufficient free VRAM for bounded inference: {measured["vram"]}')
    before = data.digest(runtime.INITIALIZATION)
    if before != runtime.INITIALIZATION_SHA256:
        raise ValueError('Initialization checkpoint hash mismatch.')
    tokenizer = load_tokenizer(manifest['tokenizer']['path'])
    validation = IndexedChatDataset(Path(manifest['datasets']['validation']['path']), tokenizer, context=512)
    started = time.monotonic()
    torch.manual_seed(42)
    with runtime.no_training_guard():
        model, _ = evaluation.load_inference_model(runtime.INITIALIZATION, runtime.INITIALIZATION_SHA256)
        with runtime.no_training_guard(model):
            rows = evaluation.generate_suite(model, tokenizer, evaluation.conversational_cases(), suite='conversation',
                                             decodes={'greedy': evaluation.decoding_configs()['greedy']}, device='cuda')
            gold, _ = evaluation.gold_validation(model, validation, device='cuda')
    del model
    torch.cuda.empty_cache()
    after = data.digest(runtime.INITIALIZATION)
    sealed = {r['id']: r for r in data.jsonlines(Path(summary_path).parent / 'generations.jsonl')
              if r['suite'] == 'conversation' and r['decode'] == 'greedy'}
    identical = sum(r['raw_token_ids'] == sealed[r['id']]['raw_token_ids'] for r in rows)
    gold_equal = {k: (gold[k], summary['gold'][k]) for k in summary['gold'] if isinstance(summary['gold'][k], (int, float))}
    gold_match = all(abs(a - b) <= 1e-6 * max(1, abs(b)) for a, b in gold_equal.values())
    return dict(checked_utc=now(), seconds=round(time.monotonic() - started, 1), greedy_cases=len(rows),
                greedy_identical_tokens=identical, gold=gold_equal, gold_match=gold_match,
                checkpoint=dict(before=before, after=after, unchanged=before == after),
                optimizer_constructed=False, backward_calls=0, training_updates=0,
                reconfirmed=identical == len(rows) == 16 and gold_match and before == after)


# ---------------------------------------------------------------- frozen future experiment (new versioned identity)

def freeze_job(directory: Path, manifest_path: Path, summary_path: Path) -> dict:
    """Freeze job v2 beside the Step 5A evidence. Never overwrites; Phase 4's job is untouched."""
    from codexa_workspace import sft_pilot_runtime as runtime
    target = Path(directory) / 'config/job.json'
    if target.exists():
        raise FileExistsError('The Step 5A job is frozen; never overwrite it.')
    analysis = data.read(Path(directory) / 'config/budget-analysis.json')
    if analysis['recommendation']['total_seconds'] != runtime.PROPOSED_BUDGET['total_seconds']:
        raise ValueError('The proposed cap differs from the evidence-backed recommendation.')
    report = runtime.TrainingGuardProvider('new').resolve()
    with runtime.no_training_guard():
        job = runtime.build_job(PHASE4, data.read(manifest_path), step_zero=data.read(summary_path),
                                step_zero_path=summary_path, manifest_path=manifest_path, environment_report=report,
                                budget_analysis=str(Path(directory) / 'config/budget-analysis.json'))
    data.write(target, job)
    return dict(path=str(target), job_sha256=job['job_sha256'], spec_sha256=job['spec_sha256'],
                guard_valid_at_freeze=report['context'] == 'current', guard_errors=report.get('errors'))
