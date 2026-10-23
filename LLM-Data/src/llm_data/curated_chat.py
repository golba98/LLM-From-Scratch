"""Small original conversational curriculum with explicit review and provenance."""

import hashlib
import json
from pathlib import Path

from llm_data.conversational_audit import content_identity
from llm_tokenizer.sft import ChatMessage, serialize_conversation


def curated_records() -> list[dict]:
    """Produce 600 bounded examples; factual/open-ended admission needs review."""
    rows = []
    def add(category, family, messages, verified=True):
        identity = f'curated-v1-{len(rows):04d}'
        rows.append(dict(conversation_id=identity, root_id=identity, source='curated',
            category=category, family=family, messages=messages,
            provenance={'kind': 'original_programmatic_curriculum', 'license': 'CC0-1.0',
                        'verification': 'deterministic construction' if verified else 'human review required'},
            review_status='automatic_verified' if verified else 'pending_human',
            historical_sft_exposure=False, language='en'))
    def pair(prompt, answer):
        return [dict(role='user', content=prompt), dict(role='assistant', content=answer)]
    verbs = ('Respond', 'Answer', 'Return', 'Write', 'Reply')
    for i in range(120):
        word = ('Vireo', 'Maple', 'Cedar', 'Amber', 'Linen', 'Petal')[i % 6] + str(i)
        forms = (f'{verbs[i%5]} with the single word {word}. Omit punctuation.',
                 f'The required label is {word}. Print that label and nothing else.',
                 f'Use exactly one word in your response: {word}.',
                 f'Copy {word} as your entire reply, with no explanation.')
        add('instruction', 'train-copy-label-'+str(i%4), pair(forms[i%4], word))
    for i in range(120):
        a, b = 21 + i, 2 + (i*7) % 31
        operator = i % 4
        if operator == 0:
            question, result = f'{a} + {b}', a+b
        elif operator == 1:
            question, result = f'{a} - {b}', a-b
        elif operator == 2:
            question, result = f'{2+i%13} × {3+i%17}', (2+i%13)*(3+i%17)
        else:
            question, result = f'{a*b} ÷ {b}', a
        add('arithmetic', 'train-symbol-'+str(operator),
            pair(f'Calculate {question}. Give only its numerical value.', str(result)))
    # Stable unit definitions; these are review candidates, not certified factual data.
    facts = [('metres in a kilometre', '1000'), ('minutes in an hour', '60'),
             ('grams in a kilogram', '1000'), ('centimetres in a metre', '100'),
             ('millilitres in a litre', '1000'), ('degrees in a full turn', '360')]
    for i in range(60):
        topic, answer = facts[i%6]
        forms = (f'How many {topic} are there? Answer with only a number.',
                 f'State the number of {topic}, without extra text.',
                 f'I need the count of {topic}. Provide the number alone.',
                 f'What is the count of {topic}? Keep your answer numerical.',
                 f'Please give only the numerical count: {topic}.',
                 f'For a unit conversion, tell me the number of {topic}.')
        add('factual', 'train-unit-'+str(i%6), pair(forms[(i//6)%6], answer), False)
    for i in range(60):
        old, new = 'Locker-'+str(100+i), 'Cabinet-'+str(400+i)
        add('multi_turn', 'train-location-correction', [
            dict(role='user', content=f'I put my notes in {old}.'),
            dict(role='assistant', content=f'Your notes are in {old}.'),
            dict(role='user', content=f'I moved them to {new}. Where are my notes now? Use only the location.'),
            dict(role='assistant', content=new)])
    for i in range(60):
        person = ('nephew', 'aunt', 'colleague', 'coach', 'sibling')[i%5]
        add('uncertainty', 'train-private-unknown', pair(
            f'Without any details about my life, what did my {person} eat on day {i+1}?',
            'I do not have that information.'), False)
    for i in range(60):
        items = [f'tag-{i*3+j}' for j in range(3)]
        add('length', 'train-three-lines', pair(
            'Put these labels on exactly three separate lines, without bullets: '+', '.join(items)+'.',
            '\n'.join(items)))
    for i in range(60):
        subject = ('walk', 'read', 'draw', 'cook', 'garden', 'rest')[i%6]
        add('natural', 'train-small-plan', pair(
            f'I have {10+i} minutes free and want to {subject}. Suggest a simple starting point.',
            f'Start with a small {subject} activity that fits the time you have. Choose one manageable step.'), False)
    for i in range(60):
        left, right = 41+i, 17+i%13
        add('reasoning', 'train-integer-order', pair(
            f'Which integer is greater: {left} or {right}? Give the greater integer only.', str(max(left,right))))
    # Remove normalized whole-record duplicates rather than counting them twice.
    seen, unique = set(), []
    for row in rows:
        digest = content_identity(row['messages'])
        if digest not in seen:
            unique.append(row)
            seen.add(digest)
    return unique


def write_curated(path: Path, tokenizer) -> dict:
    """Write a new compatible corpus and its review queue, never replace originals."""
    rows = curated_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        for row in rows:
            record = serialize_conversation([ChatMessage(m['role'],m['content']) for m in row['messages']], tokenizer,
                                            maximum_tokens=2048)
            row.update(content_sha256=content_identity(row['messages']),
                       duplicate_group=row['family'], tokens=len(record.input_ids),
                       assistant_tokens=record.supervised_token_count)
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, records=len(rows),
        supervised_tokens=sum(row['assistant_tokens'] for row in rows),
        pending_human=sum(row['review_status']=='pending_human' for row in rows),
        admitted_for_training=False,
        limitations=['Original curated examples are proposed, not human-approved.',
                     'Two-pass exposure ceiling; this small pool cannot sustain 15% of a ten-hour run.'])
