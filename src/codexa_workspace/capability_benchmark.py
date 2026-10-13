"""Frozen foundation capability benchmark: likelihood ranking, margins and generation, inference only.

Suites (``benchmarks/``): v1 is the original 56 gold + 8 conversational items, preserved byte for
byte; v2 adds 32 new gold items in each of the same seven categories plus 8 conversational
prompts. Every gold item is scored in plain-completion form (leading space: the pretraining
format) and chat form (no leading space: the SFT serialization).

Per item and form:
- teacher-forced total and per-token (length-normalized) log-likelihood of the gold answer and
  every distractor; rank among the candidates and margin over the best distractor;
- grammar items are scored with their two options presented in both orders, and count as
  correct only if both orders rank the gold answer first (choice-order control);
- full-vocabulary rank and top-5 at the answer position; chat form: P(END) right after gold;
- generations (greedy, optionally repetition penalty 1.3) with termination and repetition.

Uncertainty: Wilson 95% intervals per category; checkpoint comparisons are paired over the same
items with an exact McNemar test, so small category differences are reported with their p-values
instead of being read as real changes. ``degraded`` (the in-run capability guard input) requires
a significant net loss of plain-form top-1 items (one-sided exact McNemar p < 0.01) together with
a lower mean margin; trainers additionally require repeated confirmation before stopping.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

BENCHMARKS = Path(__file__).resolve().parent / 'benchmarks'
SUITES = {'v1': ('foundation_capability_v1.json', 'a2df29d1867d00ad55e0918844958c116ec5c3f50cc7ebf6d793fcbf22243bfa'),
          'v2': ('foundation_capability_v2.json', 'd3e664fc7501123d5979cd349009ef8cb94c479cdbfffa26e7103373eaccb105')}
CATEGORIES = ('factual', 'arithmetic', 'copy_extract', 'sentence_completion', 'grammar', 'reasoning', 'exact_format')
USER, END, ASSISTANT = '<|user|>', '<|end|>', '<|assistant|>'
END_ID, EOS_ID, PAD_ID = 7, 2, 0
MAX_NEW_TOKENS = 48
DEGRADED_P = 0.01


# ---------------------------------------------------------------- suite identity

def content_sha256(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def load_suites(names=('v1', 'v2')) -> dict:
    """Items of the named suites, each verified against its frozen content hash."""
    items, identity = [], {}
    for name in names:
        filename, expected = SUITES[name]
        path = BENCHMARKS / filename
        document = json.loads(path.read_text())
        actual = content_sha256(document['items'])
        if actual != expected or document['content_sha256'] != expected:
            raise ValueError(f'Benchmark suite {name} drifted from its frozen identity.')
        identity[name] = dict(path=str(path), content_sha256=expected, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              items=len(document['items']))
        items += [dict(item, suite=name) for item in document['items']]
    ids = [f"{i['suite']}:{i['id']}" for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate benchmark item ids.')
    return dict(items=items, identity=identity, sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest())


def chat_prompt(question: str) -> str:
    return f'{USER}{question}{END}{ASSISTANT}'


def variants(item: dict, form: str) -> list[str]:
    """Prompt texts for one item and form; grammar options are rendered in both orders."""
    text = item['plain'] if form == 'plain' else item['chat']
    if text is None:
        return []
    options = item.get('options')
    rendered = ([text.format(options='/'.join(order), options_or=' or '.join(order)) for order in (options, options[::-1])]
                if options else [text])
    return [chat_prompt(t) if form == 'chat' else t for t in rendered]


def first_candidate(response: str, candidates: list[str]) -> str | None:
    """The candidate the response states first (case-sensitive word match, then case-insensitive)."""
    for flags in (0, re.IGNORECASE):
        best = None
        for candidate in candidates:
            match = re.search(rf'(?<![\w]){re.escape(candidate)}(?![\w])', response, flags)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), candidate)
        if best:
            return best[1]
    return None


def classify(*, top1: bool, stated_gold: bool, exact: bool) -> str:
    if exact:
        return 'correct_exact'
    if top1:
        return 'knows_wrong_format' if stated_gold else 'knows_generates_other'
    return 'does_not_know'


# ---------------------------------------------------------------- statistics

def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def binomial_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def mcnemar(gained: int, lost: int) -> dict:
    """Exact McNemar on discordant pairs: two-sided p and one-sided p for a net loss."""
    n = gained + lost
    two_sided = min(1.0, 2 * binomial_tail(max(gained, lost), n)) if n else 1.0
    return dict(discordant=n, gained=gained, lost=lost, p_two_sided=two_sided,
                p_loss_one_sided=binomial_tail(lost, n) if n else 1.0)


# ---------------------------------------------------------------- scoring and generation

def _encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def score_prompt(model, tokenizer, prompt_ids: list[int], candidates: list[str], *, leading_space: bool,
                 device: str) -> list[dict]:
    import torch
    results = []
    for index, candidate in enumerate(candidates):
        ids = _encode(tokenizer, (' ' if leading_space else '') + candidate)
        inputs = torch.tensor([prompt_ids + ids + [END_ID]], device=device)
        logits, _ = model(inputs)
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        if not torch.isfinite(logprobs).all():
            raise FloatingPointError('Nonfinite logits during capability scoring.')
        positions = range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(ids))
        token_logprobs = [float(logprobs[p, t]) for p, t in zip(positions, ids)]
        record = dict(candidate=candidate, token_ids=ids, tokens=len(ids), logprob=sum(token_logprobs),
                      mean_logprob=sum(token_logprobs) / len(ids),
                      end_after_probability=float(logprobs[len(prompt_ids) + len(ids) - 1, END_ID].exp()))
        if index == 0:
            first = logprobs[len(prompt_ids) - 1]
            record['first_token_vocab_rank'] = int((first > first[ids[0]]).sum()) + 1
            record['first_token_probability'] = float(first[ids[0]].exp())
            top = torch.topk(first, 5)
            record['top5_next'] = [dict(token=tokenizer.decode([int(t)]), id=int(t), probability=float(v.exp()))
                                   for v, t in zip(top.values, top.indices)]
        results.append(record)
    return results


def _ranking(scores: list[dict], key: str) -> dict:
    order = sorted(range(len(scores)), key=lambda i: -scores[i][key])
    best_other = max(s[key] for s in scores[1:]) if len(scores) > 1 else None
    return dict(rank=order.index(0) + 1, top1=order[0] == 0,
                margin=None if best_other is None else scores[0][key] - best_other)


def decoding_configs(names=('greedy',)) -> dict:
    from llm_inference.generate import GenerationConfig
    available = {'greedy': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=42),
                 'greedy_repetition_penalty_1.3': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=42,
                                                                    repetition_penalty=1.3)}
    return {name: available[name] for name in names}


def generate(model, tokenizer, prompt_ids: list[int], config, *, device: str) -> dict:
    import torch
    from llm_inference.generate import generate_sequences
    from codexa_workspace.sft_pilot_evaluation import mechanical, termination
    started = time.monotonic()
    result = generate_sequences(model, torch.tensor([prompt_ids], device=device), eos_token_id=EOS_ID,
                                pad_token_id=PAD_ID, stop_sequences=[[END_ID]], config=config).sequences[0]
    visible = list(result.visible_token_ids)
    response = tokenizer.decode(visible, skip_special_tokens=True).strip()
    return dict(configuration=asdict(config), response=response, visible_token_ids=visible,
                generated_tokens=result.generated_token_count, finish_reason=result.finish_reason,
                terminator=result.terminating_token_id,
                termination=termination(result.terminating_token_id, result.finish_reason),
                severe_repetition=mechanical(visible, response)['independent_severe'],
                seconds=time.monotonic() - started)


def evaluate_items(model, tokenizer, items: list[dict], *, forms=('plain', 'chat'), decodes: dict | None = None,
                   generate_forms=('plain', 'chat'), device: str = 'cuda', deadline: float | None = None) -> list[dict]:
    """One row per item and form. Caller supplies an eval/inference context."""
    from codexa_workspace.sft_pilot_evaluation import strict_success
    rows = []
    for item in items:
        for form in forms:
            prompts = variants(item, form)
            if not prompts:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('Capability evaluation deadline reached.')
            candidates = ([item['gold']] + item['distractors']) if item['gold'] else []
            scored = []
            for text in prompts if candidates else []:
                ids = _encode(tokenizer, text)
                scores = score_prompt(model, tokenizer, ids, candidates, leading_space=form == 'plain', device=device)
                total, mean = _ranking(scores, 'logprob'), _ranking(scores, 'mean_logprob')
                weights = [math.exp(s['logprob'] - max(x['logprob'] for x in scores)) for s in scores]
                scored.append(dict(prompt=text, prompt_ids=ids, candidates=scores, rank=total['rank'],
                                   top1=total['top1'], margin=total['margin'], rank_length_normalized=mean['rank'],
                                   top1_length_normalized=mean['top1'], margin_length_normalized=mean['margin'],
                                   gold_candidate_probability=weights[0] / sum(weights)))
            row = dict(id=f"{item['suite']}:{item['id']}", suite=item['suite'], category=item['category'], form=form,
                       gold=item['gold'], candidate_count=len(candidates), variants=scored)
            if scored:
                primary = scored[0]
                row.update(top1=all(v['top1'] for v in scored), top1_primary=primary['top1'],
                           order_sensitive=len({v['top1'] for v in scored}) > 1,
                           top1_length_normalized=all(v['top1_length_normalized'] for v in scored),
                           margin=min(v['margin'] for v in scored), rank=max(v['rank'] for v in scored),
                           gold_candidate_probability=sum(v['gold_candidate_probability'] for v in scored) / len(scored),
                           first_token_vocab_rank=primary['candidates'][0]['first_token_vocab_rank'],
                           end_after_gold=primary['candidates'][0]['end_after_probability'])
            if decodes and form in generate_forms:
                ids = _encode(tokenizer, prompts[0])
                row['generations'] = {}
                for name, config in decodes.items():
                    generation = generate(model, tokenizer, ids, config, device=device)
                    if candidates:
                        stated = first_candidate(generation['response'], candidates)
                        exact = form == 'chat' and strict_success(generation['response'], item['gold'])
                        generation.update(stated_candidate=stated, stated_gold=stated == item['gold'], exact=exact,
                                          classification=classify(top1=row['top1'], stated_gold=stated == item['gold'],
                                                                  exact=exact))
                    row['generations'][name] = generation
            rows.append(row)
    return rows


# ---------------------------------------------------------------- summaries and comparison

def _category_summary(rows: list[dict]) -> dict:
    gold = [r for r in rows if r.get('variants')]
    n = len(gold)
    top1 = sum(r['top1'] for r in gold)
    low, high = wilson(top1, n)
    result = dict(n=n, top1=top1, top1_rate=top1 / n if n else None, top1_ci95=[low, high],
                  chance=sum(1 / r['candidate_count'] for r in gold),
                  top1_length_normalized=sum(r['top1_length_normalized'] for r in gold),
                  order_sensitive=sum(r['order_sensitive'] for r in gold),
                  mean_margin=sum(r['margin'] for r in gold) / n if n else None,
                  mean_gold_candidate_probability=sum(r['gold_candidate_probability'] for r in gold) / n if n else None)
    decodes = sorted({d for r in rows for d in r.get('generations', {})})
    for decode in decodes:
        generated = [r for r in rows if decode in r.get('generations', {})]
        with_gold = [r for r in generated if r.get('variants')]
        result[f'generation:{decode}'] = dict(
            n=len(generated), stated_gold=sum(r['generations'][decode].get('stated_gold', False) for r in with_gold),
            exact=sum(r['generations'][decode].get('exact', False) for r in with_gold),
            natural_end=sum(r['generations'][decode]['termination']['natural'] for r in generated),
            severe_repetition=sum(r['generations'][decode]['severe_repetition'] for r in generated),
            classes={c: sum(r['generations'][decode].get('classification') == c for r in with_gold)
                     for c in ('correct_exact', 'knows_wrong_format', 'knows_generates_other', 'does_not_know')})
    return result


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for form in sorted({r['form'] for r in rows}):
        selected = [r for r in rows if r['form'] == form]
        summary[form] = dict(overall=_category_summary([r for r in selected if r['category'] in CATEGORIES]),
                             categories={c: _category_summary([r for r in selected if r['category'] == c])
                                         for c in CATEGORIES if any(r['category'] == c for r in selected)},
                             conversational=_category_summary([r for r in selected if r['category'] == 'conversational']))
    return summary


def compare(baseline_rows: list[dict], rows: list[dict], *, form: str = 'plain') -> dict:
    """Paired comparison over identical items (exact McNemar per category and overall)."""
    base = {r['id']: r for r in baseline_rows if r['form'] == form and r.get('variants')}
    new = {r['id']: r for r in rows if r['form'] == form and r.get('variants')}
    shared = sorted(set(base) & set(new))
    if len(shared) != len(base) or len(shared) != len(new):
        raise ValueError('Comparison requires the identical frozen item set.')

    def block(ids):
        gained = sum(not base[i]['top1'] and new[i]['top1'] for i in ids)
        lost = sum(base[i]['top1'] and not new[i]['top1'] for i in ids)
        margins = [new[i]['margin'] - base[i]['margin'] for i in ids]
        return dict(n=len(ids), baseline_top1=sum(base[i]['top1'] for i in ids), top1=sum(new[i]['top1'] for i in ids),
                    mean_margin_change=sum(margins) / len(margins) if margins else None, **mcnemar(gained, lost))
    overall = block([i for i in shared if base[i]['category'] in CATEGORIES])
    categories = {c: block([i for i in shared if base[i]['category'] == c]) for c in CATEGORIES
                  if any(base[i]['category'] == c for i in shared)}
    degraded = (overall['lost'] > overall['gained'] and overall['p_loss_one_sided'] < DEGRADED_P
                and (overall['mean_margin_change'] or 0) < 0)
    return dict(form=form, overall=overall, categories=categories, degraded=degraded,
                rule=f'degraded = net top-1 loss with one-sided exact McNemar p < {DEGRADED_P} and a lower mean margin')


# ---------------------------------------------------------------- in-run evaluators (bound to the sealed baseline)

def inference_context(model, device: str):
    import torch
    from codexa_workspace.sft_pilot_evaluation import inference
    return inference(model, device)


def load_baseline(path: Path, expected_sha256: str) -> dict:
    document = json.loads(Path(path).read_text())
    unsigned = {k: v for k, v in document.items() if k != 'summary_sha256'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest() != document['summary_sha256'] \
            or document['summary_sha256'] != expected_sha256:
        raise ValueError('Sealed step-zero baseline identity mismatch.')
    rows_path = Path(document['capability']['rows_path'])
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != document['capability']['rows_sha256']:
        raise ValueError('Sealed baseline rows were modified.')
    document['rows'] = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    return document


def periodic_evaluator(spec: dict, baseline: dict, output: Path):
    """Teacher-forced plain-form scoring of every frozen item (no generation) after cadence validations."""
    from llm_tokenizer.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(Path(spec['tokenizer']['path']))
    suites = load_suites(tuple(spec['evaluation']['capability']['suites']))
    if suites['sha256'] != spec['evaluation']['capability']['suites_sha256']:
        raise ValueError('Capability suite identity differs from the frozen job.')
    output = Path(output)

    def evaluate(model, step: int, deadline: float) -> dict:
        device = spec['device']
        with inference_context(model, device):
            rows = evaluate_items(model, tokenizer, suites['items'], forms=('plain',), decodes=None,
                                  device=device, deadline=deadline)
        output.mkdir(parents=True, exist_ok=True)
        with (output / f'periodic-step-{step:05d}.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        comparison = compare(baseline['rows'], rows, form='plain')
        summary = summarize(rows)['plain']
        return dict(schedule='periodic', step=step, plain=summary['overall'],
                    categories={c: dict(top1=v['top1'], n=v['n']) for c, v in summary['categories'].items()},
                    comparison=comparison, degraded=comparison['degraded'])
    return evaluate


def full_evaluator(spec: dict, baseline: dict):
    """Mid-run and final: historical 48-prompt suite + both-form scoring + greedy generations."""
    from llm_tokenizer.tokenizer import load_tokenizer
    from codexa_workspace.foundation_training import generation_evaluator as historical_evaluator
    tokenizer = load_tokenizer(Path(spec['tokenizer']['path']))
    suites = load_suites(tuple(spec['evaluation']['capability']['suites']))
    historical = historical_evaluator(spec)
    decodes = decoding_configs(tuple(spec['evaluation']['capability']['full_decodes']))

    def evaluate(model, target: Path, deadline: float) -> dict:
        target = Path(target)
        if target.name.startswith('generation-baseline'):
            return dict(skipped='sealed step-zero baseline is the reference', baseline_sha256=baseline['summary_sha256'])
        device = spec['device']
        started = time.monotonic()
        historical_summary = historical(model, target, deadline)
        with inference_context(model, device):
            rows = evaluate_items(model, tokenizer, suites['items'], forms=('plain', 'chat'), decodes=decodes,
                                  generate_forms=tuple(spec['evaluation']['capability']['full_generate_forms']),
                                  device=device, deadline=deadline)
        with (target / 'capability-rows.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        summary = summarize(rows)
        comparison = dict(plain=compare(baseline['rows'], rows, form='plain'),
                          chat=compare(baseline['rows'], rows, form='chat'))
        return dict(schedule='full', historical=historical_summary, capability=summary, comparison=comparison,
                    degraded=comparison['plain']['degraded'], seconds=time.monotonic() - started)
    return evaluate
