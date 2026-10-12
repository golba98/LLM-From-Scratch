"""Inference-only evaluation for the conversational SFT pilot.

The same functions measure the sealed step-zero baseline and, in a separately
authorized future run, updates 32/64/128, so measurements stay comparable.
Nothing here constructs an optimizer or calls backward; every model pass runs
under ``torch.inference_mode`` with the model in eval mode, restoring the
caller's train/eval state afterwards. Automatic metrics never stand in for the
blinded human usefulness review.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / 'outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256'
CONVERSATIONAL_CASES = DIAGNOSTIC / 'evidence/generation/conversational-cases.json'
HISTORICAL_GENERATIONS = DIAGNOSTIC / 'evidence/generation/generations-final.jsonl'
EXPECTED_PARAMETERS = 934_356_480
PAD_ID, EOS_ID, END_ID = 0, 2, 7
IGNORE_LABEL = -100
MAX_NEW_TOKENS = 128
GENERATION_SEED = 42

ACCEPTANCE = dict(natural_end_min=8, chat_end_min=6, severe_degeneration_max=4, strict_success_min=3,
                  strict_cases=7, conversational_cases=16, human_accepted_min=8, paired_improvements_min=4,
                  paired_regressions_max=1, end_nll_improvement_min=.5,
                  premature_end_probability_increase_max=.02, reference_ce_increase_max=.1)


def decoding_configs() -> dict:
    from llm_inference.generate import GenerationConfig
    return {'greedy': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=GENERATION_SEED),
            'nucleus': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=GENERATION_SEED, do_sample=True,
                                        temperature=.8, top_p=.9)}


# ---------------------------------------------------------------- frozen suites

def conversational_cases() -> list[dict]:
    cases = json.loads(CONVERSATIONAL_CASES.read_text())
    if len(cases) != 16 or sum(c['expected'] is not None for c in cases) != ACCEPTANCE['strict_cases']:
        raise ValueError('Conversational suite identity drift (expected 16 cases, 7 strict).')
    return cases


def historical_cases() -> list[dict]:
    rows = []
    with HISTORICAL_GENERATIONS.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['suite'] == 'historical' and row['decode'] == 'greedy':
                rows.append({k: row[k] for k in ('id', 'category', 'form', 'prompt', 'input_ids', 'expected')})
    if len(rows) != 48:
        raise ValueError('Historical 48-prompt suite identity drift.')
    return rows


def suite_identity() -> dict:
    def sha(rows):
        return hashlib.sha256(json.dumps([(r['id'], r['input_ids']) for r in rows]).encode()).hexdigest()
    return dict(conversational=dict(path=str(CONVERSATIONAL_CASES), cases=16, sha256=sha(conversational_cases())),
                historical=dict(path=str(HISTORICAL_GENERATIONS), cases=48, sha256=sha(historical_cases())))


# ---------------------------------------------------------------- mechanical scoring

def mechanical(ids, text: str) -> dict:
    """Port of the diagnostic's repetition metrics (post_training_generation_diagnostic.mechanical)."""
    words = re.findall(r"\b[\w']+\b", text.casefold())
    result = {}
    for name, sequence in (('token', list(ids)), ('word', words)):
        for n in (1, 2, 3, 4):
            grams = [tuple(sequence[i:i + n]) for i in range(max(0, len(sequence) - n + 1))]
            distinct = len(set(grams)) / len(grams) if grams else 0.0
            result[f'{name}_distinct_{n}'] = distinct
            result[f'{name}_repeat_{n}'] = 1 - distinct if grams else 0.0
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    result['maximum_repeated_line_count'] = max(lines.values(), default=0)
    result['historical_severe'] = result['word_repeat_4'] > .5
    result['independent_severe'] = (result['historical_severe'] or result['token_repeat_4'] > .65
                                    or result['maximum_repeated_line_count'] >= 3)
    return result


def strict_success(response: str, expected: str | None) -> bool:
    """Exact surface only: no case folding, trimming or cleanup can rescue an answer."""
    from codexa_workspace.sft_feasibility import strict_reference_success
    return strict_reference_success(response, expected)


def termination(terminator: int | None, finish_reason: str) -> dict:
    natural = finish_reason == 'stop' and terminator in (EOS_ID, END_ID)
    return dict(natural=natural, chat_end=natural and terminator == END_ID,
                document_eos=natural and terminator == EOS_ID, length_limit=finish_reason == 'length')


def summarize_generations(rows: list[dict]) -> dict:
    """Aggregate per suite/decode; strict successes use the raw decoded visible text."""
    summary = {}
    for (suite, decode) in sorted({(r['suite'], r['decode']) for r in rows}):
        group = [r for r in rows if r['suite'] == suite and r['decode'] == decode]
        strict = [r for r in group if r.get('expected') is not None]
        summary[f'{suite}/{decode}'] = dict(
            cases=len(group), natural_end=sum(r['termination']['natural'] for r in group),
            chat_end=sum(r['termination']['chat_end'] for r in group),
            document_eos=sum(r['termination']['document_eos'] for r in group),
            length_limit=sum(r['termination']['length_limit'] for r in group),
            severe_degeneration=sum(r['metrics']['independent_severe'] for r in group),
            historical_severe=sum(r['metrics']['historical_severe'] for r in group),
            strict_cases=len(strict), strict_success=sum(r['strict_success'] for r in strict),
            mean_generated_tokens=sum(r['generated_tokens'] for r in group) / len(group),
            empty_responses=sum(not r['response'].strip() for r in group),
            seconds=sum(r['seconds'] for r in group))
    return summary


# ---------------------------------------------------------------- teacher-forced metrics

def gold_metrics_from_logits(logits, labels, *, end_id: int = END_ID) -> dict:
    """Per-conversation sums from shifted labels (labels[t] = input[t+1]); pure tensor math."""
    import torch
    logits = logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Nonfinite teacher-forced logits.')
    supervised = labels != IGNORE_LABEL
    log_probabilities = logits.log_softmax(-1)
    targets = labels.clamp_min(0)
    nll = -log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    end_positions = supervised & (labels == end_id)
    content_positions = supervised & (labels != end_id)
    end_probability = log_probabilities[..., end_id].exp()
    return dict(supervised_tokens=int(supervised.sum()), supervised_nll_sum=float(nll[supervised].sum()),
                end_targets=int(end_positions.sum()), end_nll_sum=float(nll[end_positions].sum()),
                content_positions=int(content_positions.sum()),
                premature_end_probability_sum=float(end_probability[content_positions].sum()))


def combine_gold(parts: list[dict]) -> dict:
    total = Counter()
    for part in parts:
        total.update(part)
    return dict(conversations=len(parts), supervised_tokens=total['supervised_tokens'],
                end_targets=total['end_targets'], content_positions=total['content_positions'],
                assistant_supervised_ce=total['supervised_nll_sum'] / total['supervised_tokens'],
                mean_gold_end_nll=total['end_nll_sum'] / total['end_targets'],
                mean_premature_end_probability=total['premature_end_probability_sum'] / total['content_positions'])


def source_cross_entropy(totals: dict, counts: dict, weights: dict) -> dict:
    sources = {name: totals[name] / counts[name] for name in sorted(totals)}
    return dict(sources=sources, tokens=sum(counts.values()),
                weighted=sum(sources[name] * weight for name, weight in weights.items()))


# ---------------------------------------------------------------- model passes

@contextmanager
def inference(model, device: str):
    """Eval mode + inference mode + BF16 autocast on CUDA; restore the prior mode."""
    import torch
    was_training = model.training
    model.eval()
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    try:
        with torch.inference_mode(), autocast:
            yield
    finally:
        model.train(was_training)


def load_inference_model(path: Path, expected_sha256: str, *, device: str = 'cuda',
                         memory_fraction: float | None = .5):
    """Weights only, strict load, frozen parameters, eval mode. Returns (model, metadata)."""
    import gc
    import torch
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    started = time.monotonic()
    payload = read_native_checkpoint(path, expected_sha256=expected_sha256)
    if not all(bool(torch.isfinite(v).all()) for v in payload['model_state_dict'].values()):
        raise FloatingPointError('Nonfinite checkpoint weights.')
    config = payload['config']['model']
    model = LanguageModel(ModelConfig(**config))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError('Tied embedding identity drift.')
    if count_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError('Architecture parameter count drift.')
    model.requires_grad_(False)
    model.eval()
    del payload
    gc.collect()
    loaded = time.monotonic()
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        if memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        model.to(device)
        torch.cuda.synchronize()
    metadata = dict(path=str(path), sha256=expected_sha256, model_config=config, parameters=EXPECTED_PARAMETERS,
                    tied_embeddings=True, weights_dtype=str(next(model.parameters()).dtype),
                    autocast='bf16' if device == 'cuda' else None, tf32=False, device=device,
                    requires_grad=any(p.requires_grad for p in model.parameters()),
                    seconds=dict(read_and_build=loaded - started, to_device=time.monotonic() - loaded))
    return model, metadata


def generate_suite(model, tokenizer, cases: list[dict], *, suite: str, decodes: dict, device: str,
                   guard=None) -> list[dict]:
    import torch
    from llm_inference.generate import generate_sequences
    rows = []
    with inference(model, device):
        for decode, config in decodes.items():
            for case in cases:
                if guard is not None:
                    guard()
                inputs = torch.tensor([case['input_ids']], device=device)
                started = time.monotonic()
                result = generate_sequences(model, inputs, eos_token_id=EOS_ID, pad_token_id=PAD_ID,
                                            stop_sequences=[[END_ID]], config=config).sequences[0]
                seconds = time.monotonic() - started
                visible = list(result.visible_token_ids)
                response = tokenizer.decode(visible, skip_special_tokens=True).strip()
                rows.append(dict(suite=suite, id=case['id'], category=case.get('category'), decode=decode,
                    configuration=asdict(config), prompt=case['prompt'], input_ids=case['input_ids'],
                    expected=case.get('expected'), raw_token_ids=list(result.token_ids), visible_token_ids=visible,
                    response=response, generated_tokens=result.generated_token_count,
                    finish_reason=result.finish_reason, termination_cause=result.termination_cause,
                    terminator=result.terminating_token_id,
                    termination=termination(result.terminating_token_id, result.finish_reason),
                    metrics=mechanical(visible, response),
                    strict_success=strict_success(response, case.get('expected')) if case.get('expected') is not None else None,
                    seconds=seconds))
    return rows


def gold_validation(model, dataset, *, device: str, guard=None) -> tuple[dict, list[dict]]:
    parts, per_record = [], []
    with inference(model, device):
        for index in range(len(dataset)):
            if guard is not None:
                guard()
            sample = dataset[index]
            ids = sample['input_ids'].unsqueeze(0).to(device)
            labels = sample['labels'].unsqueeze(0).to(device)
            logits, _ = model(ids)
            part = gold_metrics_from_logits(logits, labels)
            parts.append(part)
            per_record.append(dict(conversation_id=sample['identity'], source=sample['source'], **part))
    return combine_gold(parts), per_record


def reference_validation(model, data_spec: dict, *, device: str, guard=None) -> dict:
    """Historical Wikipedia / FineWeb-Edu CE on the frozen 256 x 2048 windows (unchanged data)."""
    import numpy as np
    import torch
    windows = np.load(data_spec['validation_window_ids'])
    mapping = json.loads(Path(data_spec['validation_sources']).read_text())
    sources = [mapping[str(int(w))] for w in windows]
    corpus = np.memmap(data_spec['corpus_path'], dtype=data_spec.get('dtype', 'uint16'), mode='r')
    totals, counts = Counter(), Counter()
    with inference(model, device):
        for index, window in enumerate(windows):
            if guard is not None:
                guard()
            start = int(window) * 2048
            ids = torch.tensor(corpus[start:start + 2049].astype(np.int64), device=device)
            logits, _ = model(ids[:-1].unsqueeze(0))
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite reference logits.')
            losses = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), ids[1:].unsqueeze(0),
                                                       reduction='none')[0]
            totals[sources[index]] += losses.sum().item()
            counts[sources[index]] += len(losses)
    return source_cross_entropy(totals, counts, data_spec['source_loss_weights'])


# ---------------------------------------------------------------- acceptance

def _gate(passed: bool | None, measured, threshold) -> dict:
    return dict(status='PENDING' if passed is None else ('PASS' if passed else 'FAIL'),
                measured=measured, threshold=threshold)


def acceptance_gates(step_zero: dict, candidate: dict | None, human_reviews: list[dict] | None,
                     *, resources_passed: bool | None = None) -> dict:
    """Every gate must PASS; missing evidence is PENDING, and PENDING never counts as success."""
    from codexa_workspace.sft_feasibility import usefulness_gate
    a = ACCEPTANCE
    gates = {}
    conversation = (candidate or {}).get('generation', {}).get('conversation/greedy')
    if conversation is None:
        for name in ('natural_end', 'chat_end', 'severe_degeneration', 'strict_success'):
            gates[name] = _gate(None, None, None)
    else:
        if conversation['cases'] != a['conversational_cases'] or conversation['strict_cases'] != a['strict_cases']:
            raise ValueError('Conversational suite size drift.')
        gates['natural_end'] = _gate(conversation['natural_end'] >= a['natural_end_min'], conversation['natural_end'], f">= {a['natural_end_min']}/16")
        gates['chat_end'] = _gate(conversation['chat_end'] >= a['chat_end_min'], conversation['chat_end'], f">= {a['chat_end_min']}/16")
        gates['severe_degeneration'] = _gate(conversation['severe_degeneration'] <= a['severe_degeneration_max'],
                                             conversation['severe_degeneration'], f"<= {a['severe_degeneration_max']}/16")
        gates['strict_success'] = _gate(conversation['strict_success'] >= a['strict_success_min'],
                                        conversation['strict_success'], f">= {a['strict_success_min']}/7")
    human = usefulness_gate(human_reviews or [])
    gates['human_usefulness'] = dict(status=human, measured=None if human == 'PENDING' else len(human_reviews),
        threshold='blinded real reviewer: >= 8/16 accepted, >= 4 paired improvements, <= 1 regression')
    gold0, gold = step_zero.get('gold'), (candidate or {}).get('gold')
    if gold0 is None or gold is None:
        gates['end_nll_improvement'] = _gate(None, None, None)
        gates['premature_end_probability'] = _gate(None, None, None)
    else:
        improvement = gold0['mean_gold_end_nll'] - gold['mean_gold_end_nll']
        increase = gold['mean_premature_end_probability'] - gold0['mean_premature_end_probability']
        gates['end_nll_improvement'] = _gate(improvement >= a['end_nll_improvement_min'], improvement, f">= {a['end_nll_improvement_min']}")
        gates['premature_end_probability'] = _gate(increase <= a['premature_end_probability_increase_max'], increase,
                                                   f"<= +{a['premature_end_probability_increase_max']} absolute")
    reference0, reference = step_zero.get('reference'), (candidate or {}).get('reference')
    for source in ('wikipedia', 'fineweb_edu'):
        if reference0 is None or reference is None:
            gates[f'reference_ce_{source}'] = _gate(None, None, None)
        else:
            delta = reference['sources'][source] - reference0['sources'][source]
            gates[f'reference_ce_{source}'] = _gate(delta <= a['reference_ce_increase_max'], delta, f"<= +{a['reference_ce_increase_max']}")
    finite = (candidate or {}).get('finite')
    gates['finite'] = _gate(finite, finite, 'weights and logits finite')
    gates['resources'] = _gate(resources_passed, resources_passed, 'normal resource gates held')
    statuses = [g['status'] for g in gates.values()]
    overall = 'FAIL' if 'FAIL' in statuses else ('PENDING' if 'PENDING' in statuses else 'PASS')
    return dict(overall=overall, gates=gates,
                note='Termination, loss or repetition improvements alone cannot pass; human usefulness is mandatory.')


# ---------------------------------------------------------------- blinded human review

def blinded_review_package(baseline_rows: list[dict], pilot_rows: list[dict], *, seed: int) -> tuple[list[dict], list[dict]]:
    """Randomized A/B pairs for real reviewers (all ratings null) and a separate private key."""
    baseline = {r['id']: r for r in baseline_rows if r['suite'] == 'conversation' and r['decode'] == 'greedy'}
    pilot = {r['id']: r for r in pilot_rows if r['suite'] == 'conversation' and r['decode'] == 'greedy'}
    if set(baseline) != set(pilot) or len(baseline) != ACCEPTANCE['conversational_cases']:
        raise ValueError('Both arms must contain the same 16 greedy conversational cases.')
    rng = random.Random(seed)
    sheet, key = [], []
    for case_id in sorted(baseline):
        flip = rng.random() < .5
        first, second = (pilot, baseline) if flip else (baseline, pilot)
        review_id = hashlib.sha256(f'{seed}:{case_id}'.encode()).hexdigest()[:16]
        sheet.append(dict(review_id=review_id, prompt=baseline[case_id]['prompt'],
            response_a=first[case_id]['response'], response_b=second[case_id]['response'],
            ratings={arm: dict(relevance=None, coherence=None, factual_context_consistency=None,
                               instruction_following=None, useful=None) for arm in ('a', 'b')},
            paired_preference=None, reviewer=None, blinded=True))
        key.append(dict(review_id=review_id, case_id=case_id, a='pilot' if flip else 'baseline',
                        b='baseline' if flip else 'pilot'))
    return sheet, key


def unblind_reviews(sheet: list[dict], key: list[dict]) -> list[dict]:
    """Map completed reviewer sheets to the gate schema; incomplete rows stay incomplete (PENDING)."""
    arms = {k['review_id']: k for k in key}
    reviews = []
    for row in sheet:
        mapping = arms[row['review_id']]
        pilot_arm = 'a' if mapping['a'] == 'pilot' else 'b'
        ratings = row['ratings'][pilot_arm]
        preference = row.get('paired_preference')
        reviews.append(dict(reviewer=row.get('reviewer'), blinded=row.get('blinded') is True,
            relevance=ratings.get('relevance'), coherence=ratings.get('coherence'),
            factual_context_consistency=ratings.get('factual_context_consistency'),
            paired_preference=None if preference is None else (
                'tie' if preference == 'tie' else mapping[preference])))
    return reviews


def is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
