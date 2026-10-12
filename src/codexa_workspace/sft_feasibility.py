"""Preparation-only SFT admission and readiness checks; deliberately no trainer.

No callable here constructs an optimizer, loads training state, or launches jobs.
The draft cannot become a runnable job until all admissions and evidence exist.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re


def helpers():
    path = Path(__file__).resolve().parents[2] / 'scripts/prepare_sft_feasibility.py'
    spec = importlib.util.spec_from_file_location('_sft_preparation_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authored_specs():
    """64 explicit answers, with independent mechanical correctness checks."""
    words = ('amber cedar velvet pebble lantern orchard copper meadow '
             'violet quartz ribbon willow marble comet fern coral').split()
    result = []
    for word in words:
        result.append(dict(category='word-only', prompt=f'Output exactly this word: {word}. '
                           'Include no punctuation or other text.', answer=word,
                           oracle=dict(kind='literal', value=word)))
    for a, b in [('red','blue'),('tea','milk'),('oak','elm'),('sun','moon')]:
        result.append(dict(category='formatting', prompt=f'Write {a} then {b}, one per line. '
                           'Do not add bullets or a heading.', answer=f'{a}\n{b}',
                           oracle=dict(kind='lines', values=[a,b])))
    for a, b in [('mint','sage'),('east','west'),('kiwi','pear'),('silk','wool')]:
        result.append(dict(category='formatting', prompt=f'Return {a} and {b} as two '
                           'comma-separated values with no spaces or extra text.', answer=f'{a},{b}',
                           oracle=dict(kind='csv', values=[a,b])))
    for a, b in [('lamp','desk'),('rain','snow'),('bell','drum'),('rice','corn')]:
        result.append(dict(category='formatting', prompt=f'Return a JSON array containing '
                           f'"{a}" then "{b}". Use no spaces, code fences, or explanation.',
                           answer=json.dumps([a,b],separators=(',',':')),
                           oracle=dict(kind='json-array', values=[a,b])))
    for key, value in [('color','navy'),('animal','otter'),('shape','oval'),('season','autumn')]:
        result.append(dict(category='formatting', prompt=f'Give one JSON object with key '
                           f'"{key}" and string value "{value}". No whitespace or other text.',
                           answer=json.dumps({key:value},separators=(',',':')),
                           oracle=dict(kind='json-object', value={key:value})))
    arithmetic = [(9,'+',4),(15,'-',3),(6,'*',3),(24,'/',4),
                  (8,'+',7),(20,'-',9),(4,'*',7),(32,'/',4),
                  (16,'+',7),(30,'-',13),(3,'*',7),(36,'/',4),
                  (19,'+',7),(40,'-',9),(8,'*',4),(56,'/',4)]
    operations = {'+':lambda a,b:a+b,'-':lambda a,b:a-b,
                  '*':lambda a,b:a*b,'/':lambda a,b:a//b}
    for a, op, b in arithmetic:
        result.append(dict(category='arithmetic', prompt=f'Calculate {a} {op} {b}. '
                           'Respond with the integer alone.', answer=str(operations[op](a,b)),
                           oracle=dict(kind='arithmetic', a=a, op=op, b=b)))
    contexts = [('Mira','locker','H8'),('Tariq','badge','C2'),('Nora','parcel','P6'),
                ('Evan','seat','D9'),('Zuri','folder','F4'),('Iris','ticket','T7'),
                ('Theo','shelf','S3'),('Anya','drawer','J6'),('Ravi','box','B8'),
                ('Cleo','key','K4'),('Hugo','room','R6'),('Esme','basket','V2'),
                ('Dara','bin','N7'),('Otis','label','L4'),('Wren','gate','G9'),
                ('Sana','desk','W6')]
    for name, field, value in contexts:
        context=f'{name} has {field} {value}. The spare {field} is X0.'
        result.append(dict(category='supplied-context', prompt=f'Context: {context}\n'
                           f'Which {field} belongs to {name}? Give only its code.', answer=value,
                           oracle=dict(kind='context', context=context, name=name,
                                       field=field, value=value)))
    assert len(result)==64
    assert Counter(x['category'] for x in result)==dict.fromkeys(
        ('word-only','formatting','arithmetic','supplied-context'),16)
    return result


def correct(spec):
    oracle=spec['oracle']; answer=spec['answer']; kind=oracle['kind']
    if kind=='literal': return answer==oracle['value']
    if kind=='lines': return answer=='\n'.join(oracle['values'])
    if kind=='csv': return answer==','.join(oracle['values'])
    if kind.startswith('json-'):
        value=oracle.get('values',oracle.get('value'))
        return answer==json.dumps(value,separators=(',',':'))
    if kind=='arithmetic':
        a,b,op=oracle['a'],oracle['b'],oracle['op']
        if op=='/' and a%b: return False
        value={'+':a+b,'-':a-b,'*':a*b,'/':a//b}[op]
        return answer==str(value)
    if kind=='context':
        return (answer==oracle['value'] and
                f"{oracle['name']} has {oracle['field']} {answer}." in oracle['context'])
    return False


def strict_reference_success(response, expected):
    """Only exact surfaces pass; a correct keyword cannot hide a contradiction."""
    return expected is not None and response == expected


def usefulness_gate(human_reviews):
    """No automatic metric is permitted to stand in for blinded human evidence."""
    if len(human_reviews)!=16 or any(
        not row.get('reviewer') or row.get('blinded') is not True or
        any(row.get(key) is None for key in
            ('relevance','coherence','factual_context_consistency','paired_preference'))
        for row in human_reviews):
        return 'PENDING'
    accepted=sum(all(row[key] is True for key in
        ('relevance','coherence','factual_context_consistency')) for row in human_reviews)
    improvements=sum(row['paired_preference']=='pilot' for row in human_reviews)
    regressions=sum(row['paired_preference']=='baseline' for row in human_reviews)
    return 'PASS' if accepted>=8 and improvements>=4 and regressions<=1 else 'FAIL'


def authored(directory):
    h=helpers()
    from llm_tokenizer.tokenizer import load_tokenizer
    from llm_tokenizer.sft import ChatMessage, serialize_conversation
    from llm_data.data.general_chat import normalized
    env=h.read(directory/'evidence/environment.json')
    tokenizer=load_tokenizer(env['tokenizer']['path'])
    cases=h.benchmark(directory)
    questions={h.normalize(re.sub(r'<[^>]+>',' ',r['prompt'])) for r in cases}
    forbidden=set().union(*(h.shingles(re.sub(r'<[^>]+>',' ',r['prompt'])) for r in cases))
    rows=[]; admissions=[]
    for index,spec in enumerate(authored_specs()):
        if not correct(spec): raise ValueError(f'Incorrect authored answer {index}')
        if h.normalize(spec['prompt']) in questions or any(
                h.shingles(text)&forbidden for text in (spec['prompt'],spec['answer'])):
            raise ValueError(f'Benchmark overlap: authored {index}')
        messages=[dict(role='user',content=spec['prompt']),dict(role='assistant',content=spec['answer'])]
        serialized=serialize_conversation([ChatMessage(**m) for m in messages],tokenizer,maximum_tokens=512)
        canonical=[(m['role'],normalized(m['content'])) for m in messages]
        fingerprint=hashlib.sha256(json.dumps(canonical).encode()).hexdigest()
        identity=f'codexa-feasibility-authored-{index:02d}'
        row=dict(conversation_id=identity,root_id=identity,duplicate_group=identity,
                 source='authored',tokens=len(serialized.input_ids),
                 assistant_tokens=serialized.supervised_token_count,
                 content_sha256=fingerprint,messages=messages,historical_sft_exposure=False)
        rows.append(row)
        admissions.append(dict(conversation_id=identity,source='authored',split='train',
             content_sha256=fingerprint,reviewer='Codex engineer authored-template review',
             independent_human_review=False,disposition='admit',reason='Explicit constrained answer; oracle verified; complete serializer; no frozen-prompt overlap.',
             category=spec['category'],oracle=spec['oracle']))
    h.write_rows(directory/'data/authored.jsonl',rows)
    h.write_rows(directory/'data/authored-admission.jsonl',admissions)
    return rows


def draft_config(directory):
    h=helpers();env=h.read(directory/'evidence/environment.json')
    from llm_training.config import TrainingConfig
    training=TrainingConfig(micro_batch_size=1,gradient_accumulation_steps=8,
        learning_rate=5e-6,weight_decay=.1,warmup_steps=8,max_steps=128,
        gradient_clip=1.,precision='bf16',checkpoint_interval=64,evaluation_interval=32,seed=73)
    config=dict(schema='codexa-sft-feasibility-preparation-v1',runnable=False,
        authorization_required=True,baseline_checkpoint=str(h.RUN/'checkpoints/slot-0/checkpoint.pt'),
        baseline_sha256=h.EXPECTED,tokenizer=env['tokenizer'],model=env['architecture'],
        training=asdict(training),optimization=dict(optimizer='AdamW8bit',beta1=.9,beta2=.95,
        epsilon=1e-8,final_learning_rate=5e-7,schedule='warmup-cosine',
        weight_dtype='fp32',autocast='bf16',gradient_checkpointing=True,compile=False,
        tf32=False,loss_normalization='supervised_tokens',loss_weights=None),
        data=dict(train=str(directory/'data/train.jsonl'),validation=str(directory/'data/validation.jsonl'),
        maximum_serialized_tokens=512,required_train_counts=dict(oasst1=224,ultrachat=224,authored=64),
        required_validation_counts=dict(oasst1=16,ultrachat=16),maximum_epochs=2,
        epoch_seeds=[73,74],balanced=False,bucket_size=1,truncation=False),
        evaluation=dict(updates=[0,32,64,128],historical_cases=48,conversational_cases=16,
        primary_decoding='greedy',secondary_decoding='nucleus',maximum_new_tokens=128,
        seed=42,eos_token_id=2,end_token_id=7,pad_token_id=0,stop_sequences=[[7]],
        repetition_penalty=1.,top_k=None,greedy_temperature=None,nucleus_temperature=.8,nucleus_top_p=.9,
        prompt_template='Frozen diagnostic prompts verbatim; chat-v1 for conversational cases',
        attention_mask='Unpadded single-row inference; right-pad masked training',
        context_length=2048,baseline_status='historical generations copied; gold validation not sealed'),
        gates=dict(startup_free_vram_mib=13928,available_ram_bytes=16*1024**3,
        free_disk_bytes=180*1024**3,recovery_reserve_bytes=100*1024**3,in_run_free_vram_bytes=int(1.5*1024**3)),
        budget=dict(total_seconds=900,finalization_reserve_seconds=180,automatic_extension=False),
        outputs=dict(root=str(directory/'future-run'),checkpoint_updates=[64,'final'],
        overwrite=False,launch_preconditions='READY plus separate explicit authorization; OS supervisor and dashboard'),
        acceptance=dict(natural_end_min=8,chat_end_min=6,severe_repetition_max=4,
        exact_success_min=3,exact_cases=7,human_accepted_min=8,paired_improvements_min=4,
        paired_regressions_max=1,end_nll_improvement_min=.5,premature_end_probability_increase_max=.02,
        reference_source_ce_increase_max=.1,all_finite=True),
        failure=dict(immediate=['nonfinite','OOM','resource gate breach'],
        consecutive_deteriorations=2,deadline_partial='inconclusive',
        termination_only_improvement='fail usefulness hypothesis',rollback='Retain baseline; no pilot promotion; keep all evidence'))
    h.write(directory/'config/experiment-draft.json',config)
    return config


def configuration_consistent(config,directory,env):
    """Fail closed on changed bounds or paths, without materializing runtime state."""
    assert config['runnable'] is False and config['authorization_required'] is True
    assert config['baseline_sha256']==helpers().EXPECTED
    assert config['model']==env['architecture'] and config['tokenizer']==env['tokenizer']
    assert config['training']['max_steps']==128
    assert config['training']['micro_batch_size']==1
    assert config['training']['gradient_accumulation_steps']==8
    assert config['training']['learning_rate']==5e-6 and config['training']['warmup_steps']==8
    assert config['data']['maximum_serialized_tokens']==512
    assert config['data']['maximum_epochs']==2 and config['data']['epoch_seeds']==[73,74]
    assert config['budget']==dict(total_seconds=900,finalization_reserve_seconds=180,automatic_extension=False)
    output=Path(config['outputs']['root']).resolve()
    assert output.is_relative_to(directory.resolve()) and output!=directory.resolve()
    assert not output.exists(), 'A future-run namespace already exists; never overwrite it'
    assert config['outputs']['overwrite'] is False
    return True


def check_records(path,tokenizer):
    h=helpers()
    from llm_data.chat_dataset import IndexedChatDataset,collate_chat
    from llm_tokenizer.sft import ChatMessage,serialize_conversation,IGNORE_LABEL
    dataset=IndexedChatDataset(path,tokenizer,context=512,verify_tokens=True)
    masks=[]
    for index,row in enumerate(h.jsonlines(path)):
        sample=dataset[index]
        serialized=serialize_conversation([ChatMessage(**m) for m in row['messages']],tokenizer,maximum_tokens=512)
        expected=(*serialized.labels[1:],IGNORE_LABEL)
        assert sample['labels'].tolist()==list(expected)
        batch=collate_chat([sample],fixed_length=512)
        assert (batch['labels'][0,len(expected):]==IGNORE_LABEL).all()
        assert serialized.input_ids[-1]==7 and serialized.labels[-1]==7
        assert sample['labels'][-2].item()==7
        end_positions=[i for i,x in enumerate(expected) if x==7]
        content_positions=[i for i,x in enumerate(expected) if x not in (IGNORE_LABEL,7)]
        masks.append(dict(conversation_id=row['conversation_id'],tokens=len(expected),
            assistant_supervised_tokens=sum(x!=IGNORE_LABEL for x in expected),
            end_target_positions=end_positions,premature_end_probe_positions=content_positions,
            input_ids=sample['input_ids'].tolist(),shifted_labels=list(expected)))
    return dict(dataset.audit),masks


def finalize(directory):
    """Write authored data and draft; refuse to invent missing source admissions."""
    h=helpers();authored(directory);draft_config(directory)
    candidates={r['record']['conversation_id']:r for r in h.jsonlines(directory/'data/candidates.jsonl')}
    review_path=directory/'data/engineer-review.jsonl'
    decisions=list(h.jsonlines(review_path)) if review_path.exists() else []
    selected=[];seen=set();groups=set();roots=set();fingerprints=set()
    counts=Counter();conflicts=[]
    for decision in decisions:
        identity=decision['conversation_id']
        if identity in seen: raise ValueError(f'Duplicate admission decision {identity}')
        seen.add(identity)
        candidate=candidates[identity];row=candidate['record']
        if decision['content_sha256']!=row['content_sha256']: raise ValueError('Stale review content hash')
        if decision['disposition']!='admit': continue
        key=(candidate['split'],row['source']);limit=224 if key[0]=='train' else 16
        if counts[key]>=limit: continue
        root=(row['source'],row['root_id'])
        if root in roots or row['duplicate_group'] in groups or row['content_sha256'] in fingerprints:
            conflicts.append(dict(conversation_id=identity,reason='root/group/content duplicate'))
            continue
        roots.add(root);groups.add(row['duplicate_group']);fingerprints.add(row['content_sha256'])
        counts[key]+=1;selected.append(dict(record=row,decision=decision,provenance=candidate))
    h.write_rows(directory/'data/source-admitted-candidates.jsonl',selected)
    for split in ('train','validation'):
        admitted=[r['record'] for r in selected if r['provenance']['split']==split]
        if admitted:
            h.write_rows(directory/f'data/provisional-{split}.jsonl',admitted)
    complete=all(counts[(split,source)]==n for split,n in [('train',224),('validation',16)] for source in ['oasst1','ultrachat'])
    if complete:
        train=[r['record'] for r in selected if r['provenance']['split']=='train']+list(h.jsonlines(directory/'data/authored.jsonl'))
        validation=[r['record'] for r in selected if r['provenance']['split']=='validation']
        # Exact root/group checks are insufficient alone: also compare substantial text.
        train_text={h.normalize(m['content']) for row in train for m in row['messages'] if len(h.normalize(m['content']).split())>=8}
        if any(h.normalize(m['content']) in train_text for row in validation for m in row['messages'] if len(h.normalize(m['content']).split())>=8):
            raise ValueError('Substantial normalized text crosses train/validation')
        h.write_rows(directory/'data/train.jsonl',train);h.write_rows(directory/'data/validation.jsonl',validation)
        orders=[]
        for seed in (73,74):
            order=list(range(512));random.Random(seed).shuffle(order)
            orders.append(dict(seed=seed,indices=order,conversation_ids=[train[i]['conversation_id'] for i in order]))
        h.write(directory/'data/presentation-orders.json',orders)
    h.write(directory/'evidence/admission-status.json',dict(complete=complete,
        reviewed=len(decisions),dispositions=dict(Counter(x['disposition'] for x in decisions)),
        source_admitted_counts={f'{k[0]}/{k[1]}':v for k,v in counts.items()},
        duplicate_exclusions=conflicts,authored=64,final_train_manifest_created=complete))
    return validate(directory)


def baseline(directory):
    raise RuntimeError('Gold validation population is not frozen and reviewed. No substitute baseline may be manufactured.')


def review_queue(directory):
    h=helpers();queue=[];key=[]
    for row in h.jsonlines(directory/'evaluation/step-zero-generations.jsonl'):
        if row['decode']!='greedy' or row['suite']!='conversation': continue
        identity=hashlib.sha256(('review-73:'+row['id']).encode()).hexdigest()[:16]
        queue.append(dict(review_id=identity,prompt=row['prompt'],response=row.get('raw_text',row.get('response')),
            comparison_response=None,relevance=None,coherence=None,factual_context_consistency=None,
            instruction_following=None,useful=None,paired_preference=None,reviewer=None,
            status='Awaiting pilot output, then independently randomized pair assignment and real human review'))
        key.append(dict(review_id=identity,case_id=row['id'],arm='verified-final-step-zero'))
    h.write_rows(directory/'evaluation/human-review-queue.jsonl',queue)
    h.write_rows(directory/'evidence/private-review-key.jsonl',key)


def validate(directory):
    h=helpers();env=h.read(directory/'evidence/environment.json')
    from llm_tokenizer.tokenizer import load_tokenizer,SPECIAL_TOKENS
    tokenizer=load_tokenizer(env['tokenizer']['path'])
    checks=[]
    def result(name,passed,detail):checks.append(dict(name=name,status='PASS' if passed else 'BLOCKED',detail=detail))
    authored_path=directory/'data/authored.jsonl'
    if authored_path.exists():
        counts,masks=check_records(authored_path,tokenizer)
        h.write(directory/'evidence/authored-token-accounting.json',counts)
        h.write_rows(directory/'evidence/authored-masks.jsonl',masks)
        result('64 authored correctness and serialization',len(masks)==64,counts)
    for split in ('train','validation'):
        path=directory/f'data/provisional-{split}.jsonl'
        if path.exists():
            counts,masks=check_records(path,tokenizer)
            h.write(directory/f'evidence/provisional-{split}-token-accounting.json',counts)
            h.write_rows(directory/f'evidence/provisional-{split}-masks.jsonl',masks)
            result(f'Provisional {split} serialization',True,counts)
    status=h.read(directory/'evidence/admission-status.json') if (directory/'evidence/admission-status.json').exists() else {}
    result('512 training + 32 reviewed validation examples',status.get('complete',False),status)
    config=h.read(directory/'config/experiment-draft.json')
    result('Native config and independent output bounds',configuration_consistent(config,directory,env),
        'Typed configuration constructed without optimizer; runtime adapter not validated')
    modules=[]
    for name,identity in env['modules'].items():
        modules.append(dict(module=name,installed_unchanged=h.digest(identity['path'])==identity['sha256'],
            owner_source_unchanged=h.digest(identity['source'])==identity['source_sha256']))
    h.write(directory/'evidence/component-preservation.json',modules)
    result('Component code preserved',all(x['installed_unchanged'] and x['owner_source_unchanged'] for x in modules),modules)
    protected=[]
    for path,identity in env['protected'].items():
        protected.append(dict(path=path,unchanged=h.digest(path)==identity['sha256']))
    result('Protected checkpoint, tokenizer, datasets, records',all(x['unchanged'] for x in protected),f'{len(protected)} SHA-256 identities rechecked')
    h.write(directory/'evidence/preservation-check.json',protected)
    ids={token:tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    result('Frozen tokenizer identity',h.digest(env['tokenizer']['path'])==env['tokenizer']['sha256'],ids)
    records=list(h.jsonlines(directory/'evaluation/step-zero-generations.jsonl'))
    result('Historical step-zero generation copy',records==h.benchmark(directory),f'{len(records)} records; not a new sealed gold-validation baseline')
    for name,detail in [
        ('Gold validation END/reference-loss baseline','No reviewed 32-case gold population yet; END NLL and premature END baseline unavailable'),
        ('Frozen manifest and split isolation','Cannot freeze a complete 512/32 selection while admissions are incomplete'),
        ('Supervised runtime adapter','Existing chat_run constructs optimizer before baseline; specialized supervisor/evaluation integration not validated'),
        ('15-minute feasibility budget','Historical generation round 156.555 s × 3 + 180 s reserve leaves 250.335 s for startup, updates, validation and saves; actual pilot peak/time unavailable')]:
        checks.append(dict(name=name,status='BLOCKED',detail=detail))
    h.write(directory/'evidence/readiness-checks.json',checks)
    review_queue(directory)
    return report(directory)


def report(directory):
    h=helpers();env=h.read(directory/'evidence/environment.json')
    status=h.read(directory/'evidence/admission-status.json')
    counts=h.read(directory/'evidence/authored-token-accounting.json')
    provisional=h.read(directory/'evidence/provisional-train-token-accounting.json') if (directory/'evidence/provisional-train-token-accounting.json').exists() else {}
    checks=h.read(directory/'evidence/readiness-checks.json')
    audit=json.loads(env['audit']['stdout'])
    blocks=[x for x in checks if x['status']=='BLOCKED']
    # This module has no training adapter, so it must never claim readiness.
    decision='NOT READY'
    h.write(directory/'evidence/decision.json',dict(decision=decision,blockers=blocks,
        optimizer_constructed=False,training_updates=0,training_authorized=False))
    sources=status['source_admitted_counts']
    text=f'''# Conversational SFT feasibility preparation

**Decision: {decision}.** The experiment is a draft, not a launchable or authorized job. No optimizer was constructed and no training updates were executed.

## Verified baseline and preservation

Initialization is the verified 934,356,480-parameter checkpoint, SHA-256 `{h.EXPECTED}`. All {len(env['protected'])} recorded protected artifacts were rehashed after preparation; see [preservation check](evidence/preservation-check.json). Tokenizer SHA is `{env['tokenizer']['sha256']}`. Architecture remains 24 layers, width 1536, 24 heads, vocabulary 16384, context 2048, learned positions and tied embeddings. No source dataset, tokenizer, model weight or production configuration was edited.

The [canonical diagnostic](<{h.DIAGNOSTIC/'REPORT.md'}>) is the specification. [Environment evidence](evidence/environment.json) captures initial component Git state, installed/source module identities, hardware and audit status. Prior unrelated dirty files were preserved.

## Data admission and actual token counts

{status['reviewed']} source conversations received full engineer review; {status['dispositions']}. These are engineering judgments, not independent human review. After identity/group/root exclusions, provisionally admitted counts are `{json.dumps(sources,sort_keys=True)}`. Exact IDs, source offsets, original content hashes and decisions are in [source admissions](data/source-admitted-candidates.jsonl) and [review ledger](data/engineer-review.jsonl). These are provisional candidates, not the frozen 512-example experiment dataset.

All 64 new constrained examples are prepared, 16 each of word-only commands, formatting, arithmetic and supplied-context answers. They contain **{counts['context_tokens']} serialized tokens and {counts['assistant_tokens']} assistant-supervised tokens including 64 END targets** per pass; two passes would supervise {2*counts['assistant_tokens']} authored tokens. [Exact authored records](data/authored.jsonl), [correctness oracles](data/authored-admission.jsonl), and [complete IDs/shifted masks](evidence/authored-masks.jsonl) provide reproducible evidence. Whole conversations are limited to 512 tokens; roles/user/system/padding are ignored; assistant content plus END is supervised, shifted once. END is 7, EOS 2 and PAD 0. No assistant truncation is allowed.

The total 512-example supervised-token count and 32-example gold validation count are **unavailable** until admission completes. No final train/validation manifest or invented token estimate is supplied. Existing dataset provenance and historical-exposure flags are preserved. Candidate filtering does not certify correctness. Short-source filtering biases toward supplied-context tasks, creative branding and missing-context acknowledgments; the final set needs a category balance review, not simply enough records.

The {provisional.get('examples',0)} provisional source candidates contain {provisional.get('context_tokens',0)} serialized and {provisional.get('assistant_tokens',0)} assistant-supervised tokens per pass. These counts are validated against the real indexed dataset implementation; {provisional.get('historically_exposed',0)} carry historical SFT exposure flags. They do not constitute unseen validation data. [Source token accounting](evidence/provisional-train-token-accounting.json) preserves the per-source breakdown.

Review found false context-loss refusals, unsupported factual assertions, incorrect time attribution, violated word/haiku counts, and near duplicates. Passing serialization did not admit them. Current automatic contamination screening covers exact normalized frozen questions and eight-word overlap against the diagnostic's 64 prompt identities. Complete selected-set substantial-text isolation and any additional evaluation suite screening remain pending.

## Draft experiment and compatibility

[Experiment draft](config/experiment-draft.json) preserves maximum 128 updates, microbatch 1, accumulation 8, context 512, assistant-only unweighted supervised-token CE, LR 5e-6, eight warmup steps, cosine to 5e-7, AdamW8bit, betas .9/.95, epsilon 1e-8, weight decay .1, clipping 1, FP32 weights/BF16 autocast and activation checkpointing. Two seeded complete passes (73/74) must yield exactly 1024 presentations. No optimizer state is resumed; a future separately authorized pilot would initialize only model weights.

Native `TrainingConfig` validates these base settings without constructing an optimizer. Existing `IndexedChatDataset` verifies content fingerprints, counts and shifted masks. However, the full runnable runtime adapter is not yet validated. Existing `run_chat` constructs an optimizer before its optional baseline evaluation; it was never invoked. Its quality callback, termination/retention gates and supervisor summary must be explicitly wired and dry-run tested before authorization. The draft is intentionally `runnable: false` and has no launch command.

Installed and owner module hashes match for the training, data, tokenizer and architecture modules inspected. The inference owner has an additional nonfinite-logit check absent from the installed copy; both versions were preserved. Future evaluation must pin the actual executed version and explicitly validate finite logits. This pre-existing safety-check difference is not evidence that finite baseline token selection changed.

Evaluation remains at 0/32/64/128, checkpoint publication at 64/final under a new `future-run` namespace. The existing generation outputs were copied verbatim, not regenerated or cleaned: [128 historical step-zero records](evaluation/step-zero-generations.jsonl). They are traceable diagnostic evidence, not a complete sealed SFT step-zero evaluation: reviewed gold-END NLL, premature-END probes and contemporaneous source-loss retention baseline remain pending.

## Exact acceptance and useful-answer evaluation

Primary gates use greedy decoding on the fixed 16 conversational cases: at least 8 natural endings, at least 6 chat-END endings, no more than 4 severe-degeneration flags, at least 3/7 strict constrained successes, real blinded human acceptance on at least 8/16, at least four paired improvements and no more than one regression. Gold assistant-END NLL must improve by at least .5, premature-END probability must rise by no more than .02, and each historical source CE by no more than .1. All tensors/logits and resource gates must pass. No case folding, whitespace trimming or rewritten model answers may rescue strict formatting checks. Nucleus is secondary only.

[Human-review queue](evaluation/human-review-queue.jsonl) contains actual baseline outputs with all ratings null. The second arm is pending, not fabricated. Final pairs must be randomized after actual pilot outputs exist; the private key must stay out of reviewer material. Real reviewers judge relevance, coherence, instruction following, factual/context consistency and usefulness. END/loss/repetition alone cannot pass. Scoring regression fixtures verify that fluent incorrect arithmetic, an expected keyword followed by contradiction, and an irrelevant naturally terminated response cannot pass usefulness automatically.

Nonfinite/OOM or normal headroom breach stops immediately. Two consecutive evaluation deteriorations in degeneration or source retention stop the job. Deadline-limited partial results are inconclusive; no automatic extension. Failure at 128 updates is evidence to reassess the hypothesis, not permission for more hours. Rollback preserves the baseline selection and retains pilot evidence without deletion.

## Resources and reproducibility

Target hardware is RTX 4080 16 GiB. Preserve startup 13,928 MiB free VRAM, 16 GiB available RAM, 180 GiB free disk, 100 GiB recovery reserve and 1.5 GiB in-run free VRAM. Pilot activation/optimizer peak has not been measured, because no optimizer construction is authorized. The historical full-context footprint is a conservative bound, not a validated 512-context measurement.

Budget is 900 seconds including 180-second finalization reserve, supervised by the existing OS mechanism with its read-only dashboard. A historical 128-generation round took 156.555 seconds. Three future rounds plus reserve leave only 250.335 seconds for loading, optimizer work, gold/reference evaluation and checkpoint saves. Sealing step zero beforehand avoids a fourth round; actual finalization/pilot timing remains a readiness risk. Gates must not be lowered to fit this budget.

## Development audit enforcement

Audit hooks and Git shims are installed. The recorded doctor result shows Claude completion blocking observed, but lifecycle/E2E incomplete. Codex hook approvals, lifecycle, completion blocking and E2E are unverified. Therefore mandatory automatic enforcement cannot be claimed working for both clients. Manual records use the independent session `codex:sft-feasibility-20261012`; no existing audit records or unrelated Claude obligations were replaced. [Raw audit evidence](evidence/environment.json) preserves the actual statuses.

## Remaining blockers and smallest corrections

'''
    for item in blocks:text+=f"- **{item['name']}**: {item['detail']}\n"
    text+='''
Complete engineer review of 224 source-unique OASST1 and 224 UltraChat training conversations plus 16/16 disjoint validation conversations; check the resulting category balance and split isolation. Freeze exact selected IDs, hashes, masks, two shuffle orders and actual supervised counts. Then run bounded inference to seal gold-END/premature-END and historical source-loss baselines. Prepare and validate a dedicated authorized-runtime adapter with mocked optimizer/launcher paths and unchanged normal safety gates. Verify the full evaluation/finalization cost against the 15-minute cap. Re-run all readiness checks before asking for training authorization.

## Navigation and reproduction

One report is the entry point. `config/` contains the non-runnable proposal, `data/` admission/provenance and authored records, `evaluation/` raw baseline generations and pending human queue, and `evidence/` machine-readable checks and protection hashes. Private assets remain ignored and must not be committed.

From the integration repository, use `.venv/bin/python scripts/prepare_sft_feasibility.py finalize --directory DIRECTORY` to validate current admissions, then `validate` or `report`. `baseline` fails closed while validation admission is incomplete. No command launches training.

## Test evidence and change scope

Ten bounded offline tests passed, including wrong-answer oracles, strict formatting/contradiction rejection, required blinded usefulness evidence, refusal of an unfrozen baseline, and the admitted SQL example on an unsorted fixture. Full preparation validation also passed with optimizer construction and backward calls patched to raise. SHA preservation checks passed. Gold inference, complete split isolation and runtime/supervisor checks are blocked, not passed. [Test results](evidence/test-results.json) identify captured audit evidence.

New source files: `scripts/prepare_sft_feasibility.py`, `src/codexa_workspace/sft_feasibility.py`, `tests/test_sft_feasibility_preparation.py`. Only dated append-only development log entries were added to existing files. No commit, push, PR, dependency upgrade, training process, optimizer construction, production inference edit or artifact deletion was performed.
'''
    (directory/'REPORT.md').write_text(text)
    return decision
