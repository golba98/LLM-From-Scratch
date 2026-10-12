"""Offline conversational probes with explicit, limited deterministic scoring."""

from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch

from llm_architecture.checkpointing import read_native_checkpoint
from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
from llm_inference.generate import GenerationConfig, generate_sequences
from llm_tokenizer.sft import ChatMessage, format_chat_prompt
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.session import atomic_document
from codexa_workspace.evaluation import analyze_generated_text, repeated_ngram_rate


def fingerprint(value: object) -> str:
    """Hash canonical JSON experiment identities."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_suite(*, acceptance: bool = False) -> list[dict]:
    """Create evaluation-only tasks; acceptance uses distinct wording/values."""
    rows = []
    words = ['READY', 'DONE', 'YES', 'OK', 'GREEN', 'STOP', 'BLUE', 'TRUE']
    facts = [('What is the capital of Japan?', ['tokyo']),
             ('How many days are in a non-leap year?', ['365']),
             ('Which planet do humans live on?', ['earth']),
             ('How many sides does a triangle have?', ['three', '3']),
             ('What is the chemical symbol for oxygen?', ['o']),
             ('Which ocean is the largest?', ['pacific']),
             ('What is the freezing point of pure water in degrees Celsius?', ['0', 'zero']),
             ('How many hours are in one day?', ['24', 'twenty four'])]
    explanations = [('Why does ice float on liquid water?', ['less dense', 'lower density']),
                    ('Why should you check the divisor before division in code?', ['zero']),
                    ('Why is water important to plants?', ['photosynthesis', 'nutrients']),
                    ('Explain how a Python for-loop works.', ['iterate', 'iteration', 'each']),
                    ('Why does rain fall from clouds?', ['droplet', 'gravity', 'heavy']),
                    ('Explain what a function does in programming.', ['reusable', 'reuse', 'input']),
                    ('Why is the sky usually blue?', ['scatter']),
                    ('Explain what a variable is in a program.', ['value', 'store'])]
    for i in range(8):
        prefix = 'Please answer this question: ' if acceptance else ''
        target = words[(i + 3) % 8] if acceptance else words[i]
        instruction = (f'Write {target} alone, without explanation or punctuation.' if acceptance
                       else f'Reply with only the word {target}.')
        rows.append(dict(category='instruction', prompt=instruction, check='exact', expected=target))
        a, b = (27 + i * 3, 13 + i * 2) if acceptance else (11 + i * 3, 7 + i * 2)
        question = (f'Calculate {a} + {b}. Give only the result.' if acceptance
                    else f'What is {a} plus {b}? Answer with just the number.')
        rows.append(dict(category='arithmetic', prompt=question, check='exact', expected=str(a + b)))
        question, answers = facts[(i + 3) % 8] if acceptance else facts[i]
        rows.append(dict(category='factual', prompt=prefix + question, check='contains', expected=answers))
        old, new = str(20 + i), str(70 + i if acceptance else 40 + i)
        rows.append(dict(category='multi_turn', prompt=('Which number did I most recently give you?'
            if acceptance else 'What is my updated number?'), check='contains_not', expected=[new], forbidden=[old],
            history=[['user', f'My number is {old}.'], ['assistant', f'Your number is {old}.'],
                     ['user', f'Correction: my number is {new}.'], ['assistant', f'Your new number is {new}.']]))
        unknown = ['neighbour', 'teacher', 'cousin', 'dentist', 'manager', 'friend', 'landlord', 'classmate'][i]
        rows.append(dict(category='uncertainty', prompt=(f'Without any information about me, can you identify my {unknown}?'
            if acceptance else f'What is the name of my {unknown}? If you do not know, say so.'), check='contains',
            expected=["do not know", "don't know", 'not know', 'not enough information', 'cannot determine',
                      "can't determine", 'no information', 'need more information', 'not have information']))
        rows.append(dict(category='logic', prompt=(f'There are {i + 3} red balls and {i + 2} blue balls in a box. '
            'How many balls are there altogether? Reply with only the number.' if acceptance else
            f'A box holds {i + 2} pencils. I add {i + 1} pencils. How many pencils are in the box? Reply with only the number.'),
            check='exact', expected=str(2 * i + (5 if acceptance else 3))))
        numbers = [str(i + 4 + int(acceptance)), str(i + 5 + int(acceptance)), str(i + 6 + int(acceptance))]
        rows.append(dict(category='length', prompt=(f'Put these numbers on separate lines with no other text: {", ".join(numbers)}.'
            if acceptance else f'Write {", ".join(numbers)}, one number per line. Do not add other text.'),
            check='lines', expected=numbers))
        question, answers = explanations[(i + 2) % 8] if acceptance else explanations[i]
        rows.append(dict(category='coherence_proxy', prompt=prefix + question + ' Keep the explanation brief.',
                         check='contains', expected=answers))
    for i, row in enumerate(rows):
        row.update(id=f'{"acceptance" if acceptance else "development"}-{i:03}', history=row.get('history', []))
    return rows


def check_answer(text: str, case: dict) -> bool:
    """Score literal constraints/reference terms, never unrestricted truthfulness."""
    normalized = ' '.join(re.findall(r'\w+', text.casefold()))
    def present(term: str) -> bool:
        term = ' '.join(re.findall(r'\w+', term.casefold()))
        return bool(re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', normalized))
    if case['check'] == 'exact':
        return text.strip().casefold() == case['expected'].casefold()
    if case['check'] == 'lines':
        return [line.strip() for line in text.strip().splitlines()] == case['expected']
    result = any(present(term) for term in case['expected'])
    if case['check'] == 'contains_not':
        result = result and not any(present(term) for term in case['forbidden'])
        # A user's personal fact must not become the assistant's own identity.
        attribution = text.strip() == case['expected'][0] or bool(re.search(r'\b(?:you|your)\b', text, re.I))
        result = result and attribution and not bool(re.search(r'^my (?:new |updated )?number\b', text.strip(), re.I))
    return result


def score_output(text: str, token_ids: list[int], terminator: int | None, case: dict) -> dict:
    """Keep mechanical degeneration and stop correctness separate from accuracy."""
    quality = analyze_generated_text(text)
    longest_character_run = max((len(m.group()) for m in re.finditer(r'(.)\1*', text)), default=0)
    token_repetition = repeated_ngram_rate(token_ids, 4)
    lines = [' '.join(re.findall(r'[a-z]+', line.casefold())) for line in text.splitlines()]
    repeated_lines = max(Counter(line for line in lines if line).values(), default=0)
    severe = (quality.repeated_ngram_rate > 0.5 or token_repetition > 0.65
              or longest_character_run >= 20 or repeated_lines >= 3)
    return dict(task_pass=check_answer(text, case), ended_with_END=terminator == 7,
                ended_with_EOS=terminator == 2, empty=not bool(text.strip()),
                unexpected_role_tokens=sum(t in (4, 5, 6) for t in token_ids),
                severe_degeneration=severe, token_repeated_4gram_rate=token_repetition,
                longest_character_run=longest_character_run, repeated_lines=repeated_lines,
                **quality.to_dict())


def summarize(rows: list[dict]) -> dict:
    """Aggregate observable checks with explicit population denominators."""
    categories = sorted({row['category'] for row in rows})
    rates = {name: sum(r['metrics']['task_pass'] for r in rows if r['category'] == name)
                   / sum(r['category'] == name for r in rows) for name in categories}
    objective = [v for k, v in rates.items() if k != 'coherence_proxy']
    count = len(rows)
    result = dict(cases=count, category_pass_rates=rates, macro_task_accuracy=sum(objective) / len(objective),
        END_rate=sum(r['metrics']['ended_with_END'] for r in rows) / count,
        EOS_rate=sum(r['metrics']['ended_with_EOS'] for r in rows) / count,
        length_limit_rate=sum(r['termination'] == 'length' for r in rows) / count,
        severe_degeneration_rate=sum(r['metrics']['severe_degeneration'] for r in rows) / count,
        empty_rate=sum(r['metrics']['empty'] for r in rows) / count,
        role_switch_rate=sum(bool(r['metrics']['unexpected_role_tokens']) for r in rows) / count,
        mean_generated_tokens=sum(r['generated_tokens'] for r in rows) / count,
        special_token_dominated_rate=sum(sum(t < 8 for t in r['raw_token_ids'])
            > max(1, len(r['raw_token_ids'])) * .5 for r in rows) / count,
        mean_repeated_4gram_rate=sum(r['metrics']['repeated_ngram_rate'] for r in rows) / count)
    result['entry_gate_passed'] = (rates.get('instruction', 0) >= 0.25
        and rates.get('arithmetic', 0) >= 0.25 and result['END_rate'] >= 0.5
        and result['severe_degeneration_rate'] <= 0.25 and result['empty_rate'] == 0
        and result['role_switch_rate'] == 0)
    return result


def write_human_review(report: dict, path: Path) -> None:
    """Preserve unscored semantic review fields; automatic checks are not judge ratings."""
    rubric = {
        'relevance': '0 off-topic; 1 partly answers; 2 directly addresses every requested part.',
        'coherence': '0 incoherent; 1 understandable with broken connections; 2 consistent connected answer.',
        'context_consistency': '0 contradicts supplied history; 1 ambiguous; 2 preserves role and updated user facts.',
        'length': '0 substantially violates requested length; 1 modest excess/omission; 2 proportionate complete answer.',
        'unsupported_claims': 'List each checkable claim and verification evidence; unknown is not false. No automatic truth rating.',
    }
    path.write_text(''.join(json.dumps(dict(checkpoint_sha256=report.get('checkpoint_sha256'),
        case_id=row['id'], prompt=row['prompt'], history=row['history'], response=row['response'],
        automated_metrics=row['metrics'], rubric=rubric,
        human_scores={key: None for key in rubric}, reviewer=None, verification_evidence=[]), ensure_ascii=False) + '\n'
        for row in report['outputs']))


def evaluate_loaded(model: LanguageModel, tokenizer, cases: list[dict], *, sampled: bool = False,
                    deadline: float | None = None, output: Path | None = None) -> dict:
    """Evaluate an already loaded model without altering training/RNG state."""
    from llm_training.checkpointing import capture_random_states, restore_random_states
    rng = capture_random_states()
    was_training = model.training
    device = next(model.parameters()).device
    decode = GenerationConfig(max_new_tokens=128, do_sample=sampled, temperature=0.7 if sampled else None,
        top_p=0.9 if sampled else None, repetition_penalty=1.0, no_repeat_ngram_size=None, seed=42)
    rows = []
    strict = bool(cases and cases[0].get('id', '').startswith(('dev-v2-', 'accept-v2-')))
    if strict:
        from .behavioral_readiness import strict_score, strict_summary, SCORING_VERSION
    else:
        strict_score, strict_summary, SCORING_VERSION = score_output, summarize, 'speaker-aware-v1'
    started = time.monotonic()
    model.eval()
    try:
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for case in cases:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Incomplete conversational evaluation is not scored.')
                messages = [ChatMessage(*item) for item in case['history']] + [ChatMessage('user', case['prompt'])]
                ids = tokenizer.encode(format_chat_prompt(messages, tokenizer), add_special_tokens=False).ids
                inputs = torch.tensor([ids], device=device, dtype=torch.long)
                generation = generate_sequences(model, inputs, eos_token_id=2, pad_token_id=0,
                    stop_sequences=[[7]], config=decode).sequences[0]
                text = tokenizer.decode(list(generation.visible_token_ids), skip_special_tokens=True).strip()
                row = dict(id=case['id'], category=case['category'], family=case.get('family',case['id']),
                    prompt=case['prompt'], history=case['history'],
                    response=text, raw_token_ids=list(generation.token_ids), termination=generation.termination_cause,
                    terminator=generation.terminating_token_id, generated_tokens=generation.generated_token_count,
                    metrics=strict_score(text, list(generation.token_ids), generation.terminating_token_id, case))
                if strict:
                    from .behavioral_readiness import format_compliance
                    row['metrics']['format_compliance']=format_compliance(text,case)
                rows.append(row)
                if output is not None:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
        return dict(suite_sha256=fingerprint(cases), decoding=asdict(decode),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    seconds=time.monotonic() - started, metrics=strict_summary(rows), outputs=rows, scoring_version=SCORING_VERSION,
                    limitations=(['Open-ended coherence, relevance and unsupported assertions require human review.',
                                  'Strict objective checks are not a general intelligence benchmark.'] if strict else
                                 ['Literal checks do not establish semantic coherence or general truthfulness.',
                                  'Coherence_proxy measures reference-term presence only.']))
    finally:
        model.train(was_training)
        restore_random_states(rng)


def evaluate_candidate(checkpoint: Path, tokenizer_path: Path, directory: Path,
                       *, cases: list[dict] | None = None, sampled: bool = False, deadline: float | None = None,
                       validation_loader=None, validation_identity: dict | None = None) -> dict:
    """Strict restore, finite-weight check and fingerprinted fixed-suite generation."""
    from llm_architecture.checkpointing import file_sha256
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    payload = read_native_checkpoint(checkpoint)
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_hash = file_sha256(tokenizer_path)
    if payload.get('tokenizer_sha256') not in (None, tokenizer_hash):
        raise ValueError('Tokenizer identity mismatch.')
    model = LanguageModel(ModelConfig(**payload['config']['model']))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    model.to('cuda').eval()
    if count_parameters(model) != 934356480 or model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError('Main architecture or tied embedding identity mismatch.')
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise FloatingPointError('Nonfinite checkpoint parameters.')
    from llm_tokenizer.sft import serialize_conversation
    sanity = serialize_conversation([ChatMessage('user', 'Say hello.'), ChatMessage('assistant', 'Hello.')], tokenizer)
    sanity_ids = torch.tensor([sanity.input_ids], device='cuda', dtype=torch.long)
    sanity_labels = torch.tensor([[*sanity.labels[1:], -100]], device='cuda', dtype=torch.long)
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(sanity_ids, sanity_labels)
    if not torch.isfinite(logits).all().item() or loss is None or not torch.isfinite(loss).item():
        raise FloatingPointError('Checkpoint failed finite chat forward/loss smoke.')
    del logits, sanity_ids, sanity_labels
    torch.cuda.reset_peak_memory_stats()
    try:
        report = evaluate_loaded(model, tokenizer, cases or make_suite(), sampled=sampled, deadline=deadline,
                                 output=directory / 'outputs.jsonl')
        report.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
            tokenizer_sha256=tokenizer_hash, restore_and_evaluation_seconds=time.monotonic() - started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), parameters=count_parameters(model))
        report['finite_forward_and_loss'] = True
        if validation_loader is not None:
            from llm_training.training import evaluate, resolve_precision
            details={}
            loss,tokens=evaluate(model,validation_loader,device=torch.device('cuda'),
                precision=resolve_precision('bf16',torch.device('cuda')),details=details,
                deadline_monotonic=deadline)
            report.update(common_validation=dict(loss=loss,tokens=tokens,details=details,
                                                identity=validation_identity))
        atomic_document(directory / 'report.json', report)
        write_human_review(report, directory / 'human-review-unscored.jsonl')
        return report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
