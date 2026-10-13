#!/usr/bin/env python3
"""Inference-only capability probe: foundation vs SFT pilot checkpoints on controlled tasks.

  tools/codexa-dev scripts/foundation_capability_probe.py --output <new dir>

For each checkpoint (hash-verified, weights only, eval mode, no gradients, no optimizer) and each
probe item in plain-completion form and chat form:

- teacher-forced log-probability of the gold answer and of fixed distractors (multiple choice);
- full-vocabulary rank of the gold answer's first token and the top-5 next tokens (logit inspection);
- chat form: probability that the model ends its turn (END) immediately after the gold answer;
- greedy generation and greedy generation with repetition penalty 1.3 (controlled decoding),
  raw text and token ids preserved, with natural-termination and repetition metrics.

Each (checkpoint, form, item) is classified:
  correct_exact          the generation is exactly the required answer
  knows_wrong_format     gold is the model's top-ranked candidate and the generation states it,
                         but not in the required exact form                      (brief category 1)
  knows_generates_other  gold is top-ranked by likelihood, the generation says something else (category 2)
  does_not_know          gold is not the top-ranked candidate                     (category 3)
"Top-ranked" means the highest total log-probability among the item's candidate set; chance is
1/len(candidates). Weak greedy output alone is therefore never taken as absence of knowledge.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

USER, END, ASSISTANT = '<|user|>', '<|end|>', '<|assistant|>'
CHECKPOINTS = {
    'foundation': (ROOT / 'outputs/training/codexa-934m-foundation-5h-v1/checkpoints/slot-0/checkpoint.pt',
                   'ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e'),
    'sft-update-64': (ROOT / 'outputs/training/codexa-934m-sft-feasibility-pilot-v1/checkpoints/slot-0/checkpoint.pt',
                      '2ddf6d80182464025decabc77942f42629a717838b9867ac925fb1fa595fda99'),
    'sft-update-128': (ROOT / 'outputs/training/codexa-934m-sft-feasibility-pilot-v1/checkpoints/slot-1/checkpoint.pt',
                       '43002c4dd60d4cc32e2798510240f5d4e3e5f1d31cbfde37954d9966f7aa12dc'),
}
MAX_NEW_TOKENS = 48


def _items() -> list[dict]:
    """(category, id, plain completion, chat question, gold, distractors). Chat answers carry no leading space
    (as the SFT serializer encodes assistant text); plain answers are scored with a leading space."""
    one_word = ' Reply with only one word.'
    number = ' Reply with only the number.'
    rows = [
        # factual knowledge
        ('factual', 'The capital of France is', 'What is the capital of France?' + one_word, 'Paris', ['Rome', 'London', 'Berlin', 'Madrid']),
        ('factual', 'The capital of Japan is', 'What is the capital of Japan?' + one_word, 'Tokyo', ['Beijing', 'Seoul', 'Osaka', 'Bangkok']),
        ('factual', 'The capital of Italy is', 'What is the capital of Italy?' + one_word, 'Rome', ['Paris', 'Milan', 'Madrid', 'Athens']),
        ('factual', 'The largest planet in our solar system is', 'What is the largest planet in our solar system?' + one_word, 'Jupiter', ['Saturn', 'Mars', 'Earth', 'Venus']),
        ('factual', 'Water is made of hydrogen and', 'Water is made of hydrogen and which other element?' + one_word, 'oxygen', ['carbon', 'nitrogen', 'helium', 'iron']),
        ('factual', 'The opposite of hot is', 'What is the opposite of hot?' + one_word, 'cold', ['warm', 'wet', 'dry', 'big']),
        ('factual', 'Seven days make one', 'Seven days make one what?' + one_word, 'week', ['month', 'year', 'day', 'decade']),
        ('factual', 'On a clear day the colour of the sky is', 'What colour is the sky on a clear day?' + one_word, 'blue', ['green', 'red', 'yellow', 'purple']),
        # arithmetic
        ('arithmetic', '2 + 2 =', 'What is 2 + 2?' + number, '4', ['3', '5', '6', '22']),
        ('arithmetic', '3 + 2 =', 'What is 3 + 2?' + number, '5', ['4', '6', '7', '32']),
        ('arithmetic', '5 + 3 =', 'What is 5 + 3?' + number, '8', ['7', '9', '6', '53']),
        ('arithmetic', '10 - 4 =', 'What is 10 - 4?' + number, '6', ['5', '7', '4', '14']),
        ('arithmetic', '6 + 1 =', 'What is 6 + 1?' + number, '7', ['6', '8', '5', '61']),
        ('arithmetic', '9 - 5 =', 'What is 9 - 5?' + number, '4', ['3', '5', '6', '14']),
        ('arithmetic', '2 times 3 is', 'What is 2 times 3?' + number, '6', ['5', '8', '9', '23']),
        ('arithmetic', '4 + 4 =', 'What is 4 + 4?' + number, '8', ['6', '7', '9', '44']),
        # copying and extracting supplied information
        ('copy_extract', 'Word: banana. The word above is', 'The word is banana. What is the word?' + one_word, 'banana', ['apple', 'orange', 'grape', 'mango']),
        ('copy_extract', "Name: Maria. Age: 34. The person's name is", "Name: Maria. Age: 34. What is the person's name? Reply with only the name.", 'Maria', ['Anna', 'Sofia', 'Elena', 'Laura']),
        ('copy_extract', "Name: Maria. Age: 34. The person's age is", "Name: Maria. Age: 34. How old is the person?" + number, '34', ['43', '30', '24', '35']),
        ('copy_extract', "Sam has a red car and a blue bike. The colour of Sam's car is", "Sam has a red car and a blue bike. What colour is Sam's car?" + one_word, 'red', ['blue', 'green', 'black', 'white']),
        ('copy_extract', 'The meeting is on Friday. The meeting day is', 'The meeting is on Friday. On which day is the meeting?' + one_word, 'Friday', ['Monday', 'Tuesday', 'Sunday', 'Thursday']),
        ('copy_extract', 'The password is tiger. The password is', 'The password is tiger. What is the password?' + one_word, 'tiger', ['lion', 'bear', 'cat', 'wolf']),
        ('copy_extract', 'City: Lisbon. Country: Portugal. The city mentioned is', 'City: Lisbon. Country: Portugal. Which city is mentioned?' + one_word, 'Lisbon', ['Porto', 'Madrid', 'Portugal', 'Paris']),
        ('copy_extract', 'Copy this number: 58. The number is', 'Copy this number exactly: 58.' + number, '58', ['85', '50', '8', '59']),
        # elementary sentence completion
        ('sentence_completion', 'The sun rises in the', 'Complete the sentence: The sun rises in the ...' + one_word, 'east', ['west', 'north', 'south', 'morning']),
        ('sentence_completion', 'Fish live in', 'Complete the sentence: Fish live in ...' + one_word, 'water', ['trees', 'sand', 'air', 'houses']),
        ('sentence_completion', 'At night the sky is', 'Complete the sentence: At night the sky is ...' + one_word, 'dark', ['bright', 'green', 'loud', 'hot']),
        ('sentence_completion', 'We see with our', 'Complete the sentence: We see with our ...' + one_word, 'eyes', ['ears', 'hands', 'feet', 'nose']),
        ('sentence_completion', 'Ice is very', 'Complete the sentence: Ice is very ...' + one_word, 'cold', ['hot', 'soft', 'loud', 'sweet']),
        ('sentence_completion', 'A week has seven', 'Complete the sentence: A week has seven ...' + one_word, 'days', ['weeks', 'hours', 'months', 'years']),
        ('sentence_completion', 'Cows give us', 'Complete the sentence: Cows give us ...' + one_word, 'milk', ['eggs', 'wool', 'honey', 'water']),
        ('sentence_completion', 'We use a pen to', 'Complete the sentence: We use a pen to ...' + one_word, 'write', ['eat', 'sing', 'swim', 'sleep']),
        # basic grammar (binary choice)
        ('grammar', 'She ___ a ticket. (has/have) The correct word is', "Fill the gap: 'She ___ a ticket.' Is it has or have? Reply with only the word.", 'has', ['have']),
        ('grammar', 'They ___ happy. (is/are) The correct word is', "Fill the gap: 'They ___ happy.' Is it is or are? Reply with only the word.", 'are', ['is']),
        ('grammar', 'Yesterday he ___ home. (walk/walked) The correct word is', "Fill the gap: 'Yesterday he ___ home.' Is it walk or walked? Reply with only the word.", 'walked', ['walk']),
        ('grammar', 'I ___ a student. (am/is) The correct word is', "Fill the gap: 'I ___ a student.' Is it am or is? Reply with only the word.", 'am', ['is']),
        ('grammar', 'The children ___ outside every day. (play/plays) The correct word is', "Fill the gap: 'The children ___ outside every day.' Is it play or plays? Reply with only the word.", 'play', ['plays']),
        ('grammar', 'He ___ to school every day. (go/goes) The correct word is', "Fill the gap: 'He ___ to school every day.' Is it go or goes? Reply with only the word.", 'goes', ['go']),
        ('grammar', 'We ___ dinner last night. (eat/ate) The correct word is', "Fill the gap: 'We ___ dinner last night.' Is it eat or ate? Reply with only the word.", 'ate', ['eat']),
        ('grammar', 'There ___ two cats. (is/are) The correct word is', "Fill the gap: 'There ___ two cats.' Is it is or are? Reply with only the word.", 'are', ['is']),
        # short contextual reasoning
        ('reasoning', 'Lina arrived before Omar. The person who arrived first was', 'Lina arrived before Omar. Who arrived first? Reply with only the name.', 'Lina', ['Omar']),
        ('reasoning', 'Tom is taller than Ben. The shorter person is', 'Tom is taller than Ben. Who is shorter? Reply with only the name.', 'Ben', ['Tom']),
        ('reasoning', 'Anna gave her book to Paul. Now the book belongs to', 'Anna gave her book to Paul. Who has the book now? Reply with only the name.', 'Paul', ['Anna']),
        ('reasoning', 'Today is Monday. Tomorrow will be', 'Today is Monday. What day is tomorrow?' + one_word, 'Tuesday', ['Sunday', 'Wednesday', 'Friday', 'Monday']),
        ('reasoning', 'I have 3 apples and get 2 more. Now I have', 'I have 3 apples and get 2 more. How many apples do I have?' + number, '5', ['2', '3', '6', '4']),
        ('reasoning', 'The box was empty. The number of toys in the box was', 'The box was empty. How many toys were in the box?' + number, '0', ['1', '2', '3', '5']),
        ('reasoning', 'Mia is older than Leo. The younger one is', 'Mia is older than Leo. Who is younger? Reply with only the name.', 'Leo', ['Mia']),
        ('reasoning', 'It is raining, so I will take my', 'It is raining. What should I take with me when I go out?' + one_word, 'umbrella', ['sunglasses', 'bicycle', 'sandals', 'kite']),
        # exact-format instructions (case and surface variants are distractors)
        ('exact_format', 'Instruction: Reply with only the word READY.\nAnswer:', 'Reply with only the word READY.', 'READY', ['Ready', 'OK', 'YES', 'DONE']),
        ('exact_format', 'Instruction: Reply with only the number 7.\nAnswer:', 'Reply with only the number 7.', '7', ['8', 'seven', '6', '1']),
        ('exact_format', 'Instruction: Answer yes or no. Is fire hot?\nAnswer:', 'Answer yes or no: Is fire hot? No explanation.', 'yes', ['no', 'Yes', 'maybe', 'No']),
        ('exact_format', 'Instruction: Reply with only the word blue.\nAnswer:', 'Reply with only the word blue.', 'blue', ['Blue', 'red', 'green', 'BLUE']),
        ('exact_format', 'Instruction: Write only the letter A.\nAnswer:', 'Write only the letter A.', 'A', ['B', 'a', 'C', 'The']),
        ('exact_format', 'Instruction: Reply with only the word hello.\nAnswer:', 'Reply with only the word hello.', 'hello', ['Hello', 'hi', 'goodbye', 'HELLO']),
        ('exact_format', 'Instruction: How many legs does a dog have? Reply with only the number.\nAnswer:', 'How many legs does a dog have?' + number, '4', ['2', '3', '6', 'four']),
        ('exact_format', 'Instruction: Reply with only the word STOP.\nAnswer:', 'Reply with only the word STOP.', 'STOP', ['Stop', 'GO', 'stop', 'OK']),
    ]
    items = []
    counters = {}
    for category, plain, chat, gold, distractors in rows:
        counters[category] = counters.get(category, 0) + 1
        items.append(dict(id=f'{category}-{counters[category]:02d}', category=category, plain=plain, chat=chat,
                          gold=gold, distractors=distractors))
    for index, question in enumerate(['Hello!', 'How are you today?', 'What can you help me with?', 'Thank you!',
                                      'Tell me one fun fact about cats.', 'What is your name?',
                                      'Can you give me a tip for sleeping better?', 'Goodbye!'], start=1):
        items.append(dict(id=f'conversational-{index:02d}', category='conversational', plain=None, chat=question,
                          gold=None, distractors=[]))
    return items


PROBES = _items()


def suite_sha256() -> str:
    return hashlib.sha256(json.dumps(PROBES, sort_keys=True).encode()).hexdigest()


def chat_prompt(question: str) -> str:
    return f'{USER}{question}{END}{ASSISTANT}'


def first_candidate(response: str, candidates: list[str]) -> str | None:
    """The candidate the response states first (case-sensitive word match, then case-insensitive)."""
    best = None
    for flags in (0, re.IGNORECASE):
        for candidate in candidates:
            match = re.search(rf'(?<![\w]){re.escape(candidate)}(?![\w])', response, flags)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), candidate)
        if best:
            return best[1]
    return None


def classify(*, gold_rank: int, stated_gold: bool, exact: bool) -> str:
    if exact:
        return 'correct_exact'
    if gold_rank == 1:
        return 'knows_wrong_format' if stated_gold else 'knows_generates_other'
    return 'does_not_know'


def exact_answer(response: str, gold: str) -> bool:
    from codexa_workspace.sft_pilot_evaluation import strict_success
    return strict_success(response, gold)


def score_candidates(model, tokenizer, prompt_ids: list[int], candidates: list[str], *, leading_space: bool,
                     end_id: int, device: str) -> list[dict]:
    import torch
    results = []
    for index, candidate in enumerate(candidates):
        ids = tokenizer.encode((' ' if leading_space else '') + candidate, add_special_tokens=False).ids
        sequence = prompt_ids + ids + [end_id]
        inputs = torch.tensor([sequence], device=device)
        logits, _ = model(inputs)
        logprobs = torch.log_softmax(logits[0].float(), dim=-1)
        positions = range(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(ids))
        token_logprobs = [float(logprobs[p, t]) for p, t in zip(positions, ids)]
        record = dict(candidate=candidate, token_ids=ids, logprob=sum(token_logprobs),
                      mean_logprob=sum(token_logprobs) / len(ids),
                      end_after_probability=float(logprobs[len(prompt_ids) + len(ids) - 1, end_id].exp()))
        if index == 0:                                   # gold: logit inspection at the answer position
            first = logprobs[len(prompt_ids) - 1]
            record['first_token_vocab_rank'] = int((first > first[ids[0]]).sum()) + 1
            record['first_token_probability'] = float(first[ids[0]].exp())
            top = torch.topk(first, 5)
            record['top5_next'] = [dict(token=tokenizer.decode([int(t)]), id=int(t), probability=float(v.exp()))
                                   for v, t in zip(top.values, top.indices)]
        results.append(record)
    return results


def probe_checkpoint(name: str, path: Path, sha256: str, tokenizer, *, device: str) -> list[dict]:
    import torch
    from llm_inference.generate import GenerationConfig, generate_sequences
    from codexa_workspace import sft_pilot_evaluation as ev
    model, metadata = ev.load_inference_model(path, sha256, device=device)
    assert not metadata['requires_grad']
    end_id = ev.END_ID
    decodes = {'greedy': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=42),
               'greedy_repetition_penalty_1.3': GenerationConfig(max_new_tokens=MAX_NEW_TOKENS, seed=42,
                                                                 repetition_penalty=1.3)}
    rows = []
    with ev.inference(model, device):
        for item in PROBES:
            for form in ('plain', 'chat'):
                text = item['plain'] if form == 'plain' else (chat_prompt(item['chat']) if item['chat'] else None)
                if text is None:
                    continue
                prompt_ids = tokenizer.encode(text, add_special_tokens=False).ids
                candidates = ([item['gold']] + item['distractors']) if item['gold'] else []
                scores = score_candidates(model, tokenizer, prompt_ids, candidates, leading_space=form == 'plain',
                                          end_id=end_id, device=device) if candidates else []
                ranking = sorted(range(len(scores)), key=lambda i: -scores[i]['logprob'])
                gold_rank = ranking.index(0) + 1 if scores else None
                generations = {}
                for decode, config in decodes.items():
                    started = time.monotonic()
                    result = generate_sequences(model, torch.tensor([prompt_ids], device=device), eos_token_id=ev.EOS_ID,
                                                pad_token_id=ev.PAD_ID, stop_sequences=[[end_id]], config=config).sequences[0]
                    visible = list(result.visible_token_ids)
                    response = tokenizer.decode(visible, skip_special_tokens=True).strip()
                    stated = first_candidate(response, candidates) if candidates else None
                    exact = bool(item['gold']) and form == 'chat' and exact_answer(response, item['gold'])
                    generations[decode] = dict(
                        configuration=asdict(config), response=response, visible_token_ids=visible,
                        generated_tokens=result.generated_token_count, finish_reason=result.finish_reason,
                        terminator=result.terminating_token_id,
                        termination=ev.termination(result.terminating_token_id, result.finish_reason),
                        severe_repetition=ev.mechanical(visible, response)['independent_severe'],
                        stated_candidate=stated, exact=exact, seconds=time.monotonic() - started,
                        classification=classify(gold_rank=gold_rank, stated_gold=stated == item['gold'], exact=exact)
                        if item['gold'] else None)
                probabilities = None
                if scores:
                    weights = torch.softmax(torch.tensor([s['logprob'] for s in scores]), dim=0)
                    probabilities = [float(w) for w in weights]
                rows.append(dict(checkpoint=name, checkpoint_sha256=sha256, id=item['id'], category=item['category'],
                                 form=form, prompt=text, prompt_ids=prompt_ids, gold=item['gold'],
                                 candidates=scores, gold_rank=gold_rank, candidate_count=len(candidates),
                                 gold_candidate_probability=probabilities[0] if probabilities else None,
                                 generations=generations))
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='new directory (must not exist)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoints', nargs='*', default=list(CHECKPOINTS))
    args = parser.parse_args()
    import torch
    from llm_tokenizer.tokenizer import load_tokenizer
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads((ROOT / 'outputs/training/codexa-934m-sft-feasibility-pilot-v1/config/job.json').read_text())
    tokenizer = load_tokenizer(job['spec']['tokenizer']['path'])
    (args.output / 'probe-suite.json').write_text(json.dumps(dict(sha256=suite_sha256(), items=PROBES), indent=1))
    manifest = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), suite_sha256=suite_sha256(),
                    tokenizer_sha256=job['spec']['tokenizer']['sha256'], device=args.device, torch=torch.__version__,
                    optimizer_constructed=False, backward_calls=0, checkpoints={})
    for name in args.checkpoints:
        path, sha = CHECKPOINTS[name]
        started = time.monotonic()
        rows = probe_checkpoint(name, path, sha, tokenizer, device=args.device)
        with (args.output / f'results-{name}.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        manifest['checkpoints'][name] = dict(path=str(path), sha256=sha, rows=len(rows),
                                             seconds=round(time.monotonic() - started, 1),
                                             peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == 'cuda' else None)
        print(f'{name}: {len(rows)} rows in {manifest["checkpoints"][name]["seconds"]} s', flush=True)
    manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
