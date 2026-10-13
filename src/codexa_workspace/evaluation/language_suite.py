"""The frozen 24-concept language suite: cases in five prompt forms, exact-surface scoring,
likelihood probes, and the greedy generation evaluator used by foundation jobs."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
import time

from llm_tokenizer.sft import ChatMessage, format_chat_prompt


FORMS = ('neutral', 'direct', 'chat', 'short', 'multi_turn')


def concepts() -> list[dict]:
    """Fixed development-only concepts; references are not training examples."""
    rows = [
        ('factual', 'The chemical symbol for carbon is', 'What is the chemical symbol for carbon?', 'C', 'N'),
        ('factual', 'There are this many minutes in one hour:', 'How many minutes are in one hour?', '60', '100'),
        ('factual', 'A hexagon has this many sides:', 'How many sides does a hexagon have?', '6', '8'),
        ('factual', 'A standard chessboard has this many squares:', 'How many squares are on a standard chessboard?', '64', '32'),
        ('factual', 'The capital city of Italy is', 'What is the capital city of Italy?', 'Rome', 'Milan'),
        ('factual', 'The SI unit of electric current is the', 'What is the SI unit of electric current?', 'ampere', 'volt'),
        ('arithmetic', '37 + 26 =', 'What is 37 plus 26?', '63', '53'),
        ('arithmetic', '81 - 34 =', 'What is 81 minus 34?', '47', '57'),
        ('arithmetic', '9 * 8 =', 'What is 9 multiplied by 8?', '72', '56'),
        ('arithmetic', '96 / 8 =', 'What is 96 divided by 8?', '12', '14'),
        ('arithmetic', '18 + 49 =', 'What is 18 plus 49?', '67', '73'),
        ('arithmetic', '72 - 29 =', 'What is 72 minus 29?', '43', '53'),
        ('grammar', 'Yesterday, she [walk/walked] home. The correct choice is', 'Choose walk or walked: Yesterday, she ___ home.', 'walked', 'walk'),
        ('grammar', 'They [is/are] ready. The correct choice is', 'Choose is or are: They ___ ready.', 'are', 'is'),
        ('grammar', 'He owns [a/an] umbrella. The correct choice is', 'Choose a or an: He owns ___ umbrella.', 'an', 'a'),
        ('grammar', 'Two [child/children] arrived. The correct choice is', 'Choose child or children: Two ___ arrived.', 'children', 'child'),
        ('grammar', 'She [has/have] a ticket. The correct choice is', 'Choose has or have: She ___ a ticket.', 'has', 'have'),
        ('grammar', 'This box is [heavy/heavier] than that one. The correct choice is', 'Choose heavy or heavier: This box is ___ than that one.', 'heavier', 'heavy'),
        ('comprehension', 'Lina owns a violet scarf. The colour of her scarf is', 'Lina owns a violet scarf. What colour is her scarf?', 'violet', 'orange'),
        ('comprehension', 'The key was on the desk, then moved into the drawer. It is now in the', 'The key was on the desk, then moved into the drawer. Where is it now?', 'drawer', 'desk'),
        ('comprehension', 'Omar handed his notebook to Priya. The recipient was', 'Omar handed his notebook to Priya. Who received it?', 'Priya', 'Omar'),
        ('comprehension', 'A brass coin is inside a green box. The coin is made of', 'A brass coin is inside a green box. What is the coin made of?', 'brass', 'green'),
        ('comprehension', 'Nora visits before Eli. The earlier visitor is', 'Nora visits before Eli. Who visits earlier?', 'Nora', 'Eli'),
        ('comprehension', 'The meeting moved from Monday to Thursday. Its current day is', 'The meeting moved from Monday to Thursday. On which day is it now?', 'Thursday', 'Monday'),
    ]
    return [dict(id=f'foundation-{i:02}', category=category, prefix=prefix,
                 question=question, expected=answer, distractor=distractor)
            for i, (category, prefix, question, answer, distractor) in enumerate(rows)]


def make_cases(tokenizer) -> list[dict]:
    """Encode five explicitly different formats without injecting BOS."""
    cases = []
    arithmetic_answers = set()
    for concept in concepts():
        if concept['category'] == 'arithmetic':
            match = re.fullmatch(r'(\d+) ([+*/-]) (\d+) =', concept['prefix'])
            if match is None:
                raise ValueError('Unverifiable arithmetic concept.')
            left, operation, right = match.groups()
            left, right = Fraction(left), Fraction(right)
            result = {'+':lambda:left+right, '-':lambda:left-right,
                      '*':lambda:left*right, '/':lambda:left/right}[operation]()
            if str(result) != concept['expected'] or concept['expected'] in arithmetic_answers:
                raise ValueError('Incorrect or repeated arithmetic reference.')
            arithmetic_answers.add(concept['expected'])
        for form in FORMS:
            question = concept['question']
            history = []
            if form == 'neutral':
                prompt = concept['prefix']
            elif form == 'direct':
                prompt = f'Question: {question}\nAnswer:'
            else:
                if form == 'short':
                    question += ' Reply with only the answer, without explanation or punctuation.'
                if form == 'multi_turn':
                    history = [['user', 'We will discuss one question.'],
                               ['assistant', 'Please ask your question.']]
                prompt = format_chat_prompt([ChatMessage(*x) for x in history]
                                            + [ChatMessage('user', question)], tokenizer)
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if not ids or len(ids) + 128 > 2048:
                raise ValueError('Diagnostic prompt is empty or exceeds context.')
            if form in ('neutral', 'direct') and any(value < 8 for value in ids):
                raise ValueError('Plain completion unexpectedly contains special tokens.')
            answer_tokens = {}
            for name in ('expected', 'distractor'):
                # Plain prose needs a separating space. Chat training supervises
                # answer text directly after assistant ID, with no added space.
                text = (' ' if form in ('neutral','direct') else '') + concept[name]
                joint_ids = tokenizer.encode(prompt+text, add_special_tokens=False).ids
                if joint_ids[:len(ids)] != ids:
                    raise ValueError('Answer retokenization changed the conditioning prefix.')
                answer_tokens[name] = joint_ids[len(ids):]
            cases.append(dict(id=f'{concept["id"]}-{form}', concept=concept['id'],
                category=concept['category'], form=form, prompt=prompt, input_ids=ids,
                question=question, history=history, expected=concept['expected'],
                distractor=concept['distractor'], semantic_correct=None,
                answer_token_ids=answer_tokens,
                answer_token_policy='Separating space for plain text; direct assistant text for serialized chat.',
                provenance='locally authored development diagnostic; not historically unseen knowledge',
                format_caveat='multi_turn adds conversational framing, not factual information'))
    return cases


def scoring(text: str, case: dict, terminator: int | None) -> dict:
    """Exact surface answers are distinct from unreviewed semantic judgments."""
    from codexa_workspace.evaluation.text_metrics import analyze_generated_text
    exact = text.strip() == case['expected']
    analysis = analyze_generated_text(text).to_dict()
    repeated = analysis['repeated_ngram_rate'] > .5
    return dict(exact_surface_correct=exact, semantic_correct=None,
        arithmetic_exact_correct=exact if case['category'] == 'arithmetic' else None,
        factual_semantic_correct=None, empty=not text.strip(),
        ended_with_END=terminator == 7, ended_with_EOS=terminator == 2,
        chat_turn_stop_expected=case['form'] in ('chat', 'short', 'multi_turn'),
        missing_chat_END=terminator != 7 if case['form'] in ('chat', 'short', 'multi_turn') else None,
        repeated_ngram_severe=repeated, **analysis)


def likelihood_probe(model, tokenizer, case: dict, *, device: str) -> dict:
    """Full answer-token conditional likelihood, not evidence of generated knowledge."""
    import torch
    result = {}
    # Separately appended answer IDs make this conditional token test explicit;
    # it is not silently treated as a retokenized prose continuation.
    for name in ('expected', 'distractor'):
        answer_ids = case['answer_token_ids'][name]
        if not answer_ids:
            raise ValueError('Empty likelihood reference.')
        ids = [*case['input_ids'], *answer_ids]
        logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError('Nonfinite diagnostic likelihood logits.')
        start = len(case['input_ids']) - 1
        probabilities = logits[0, start:start + len(answer_ids)].float().log_softmax(-1)
        selected = probabilities.gather(-1, torch.tensor(answer_ids, device=device).unsqueeze(-1)).squeeze(-1)
        result[name] = dict(token_ids=answer_ids, mean_token_nll=-selected.mean().item(),
                            first_token_probability=selected[0].exp().item())
    result['limitation'] = 'Token lengths differ; mean NLL is auxiliary, not correctness. Joint encoding preserves each frozen prompt prefix.'
    return result


def generation_evaluator(spec: dict):
    """Frozen 24-concept neutral/direct cases (48), greedy 128 tokens, saved for human review.

    Mirrors the 2026-10-11 diagnostic job's evaluator,
    whose aggregate assumes all five formats and cannot summarize a neutral/direct subset.
    """
    from llm_tokenizer.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(Path(spec['tokenizer']['path']))
    forms = tuple(spec['evaluation']['generation_forms'])
    cases = [case for case in make_cases(tokenizer) if case['form'] in forms]
    if len(cases) != spec['evaluation']['generation_cases']:
        raise ValueError('Frozen generation case count drift.')

    def evaluate(model, directory: Path, deadline: float) -> dict:
        import torch
        from contextlib import nullcontext
        from llm_inference.generate import GenerationConfig, generate_sequences
        device = spec['device']
        decode = GenerationConfig(max_new_tokens=128, repetition_penalty=1, seed=42, do_sample=False)
        output = Path(directory) / 'outputs.jsonl'
        rows = []
        model.eval()
        context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()

        def finite_decode_logits(_model, _inputs, result):
            if not torch.isfinite(result[0][:, -1, :]).all().item():
                raise FloatingPointError('Nonfinite free-generation logits; no answer is scored.')
        with torch.inference_mode(), context, model.register_forward_hook(finite_decode_logits):
            for case in cases:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Generation deadline reached.')
                inputs = torch.tensor([case['input_ids']], device=device, dtype=torch.long)
                generated = generate_sequences(model, inputs, eos_token_id=2, pad_token_id=0,
                                               stop_sequences=[[7]], config=decode).sequences[0]
                text = tokenizer.decode(list(generated.visible_token_ids), skip_special_tokens=True).strip()
                row = dict(**case, response=text, raw_token_ids=list(generated.token_ids),
                           terminator=generated.terminating_token_id, termination=generated.termination_cause,
                           generated_tokens=generated.generated_token_count,
                           metrics=scoring(text, case, generated.terminating_token_id),
                           likelihood=likelihood_probe(model, tokenizer, case, device=device))
                with output.open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
        by_form = {form: [row for row in rows if row['form'] == form] for form in forms}
        return dict(cases=len(rows), outputs_path=str(output), semantic_review_pending=len(rows),
                    exact_surface_by_form={form: sum(r['metrics']['exact_surface_correct'] for r in values) / len(values)
                                           for form, values in by_form.items() if values},
                    expected_preferred_by_likelihood=sum(r['likelihood']['expected']['mean_token_nll']
                                                         < r['likelihood']['distractor']['mean_token_nll'] for r in rows))
    return evaluate
