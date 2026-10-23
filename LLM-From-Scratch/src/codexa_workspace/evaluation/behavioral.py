"""Strict, versioned behavioral scoring with explicit human-review boundaries."""

import hashlib
import json
import random
import re

from codexa_workspace.evaluation.chat_probes import score_output

SCORING_VERSION = 'codexa-behavior-strict-v1'


def development_suite() -> list[dict]:
    """240 development cases; semantic cases deliberately have no automatic score."""
    rows = []
    def add(category, family, prompt, expected=None, check='exact', history=None):
        rows.append(dict(id=f'dev-v2-{len(rows):03d}',category=category,family=family,
            prompt=prompt,expected=expected,check=check,history=history or [],
            exposure='uncertain_until_lineage_scan', protected=False))
    for i in range(48):
        label = ('NORTH','SOUTH','EAST','WEST','LEFT','RIGHT')[i%6] + str(i)
        variant=i%12
        if variant == 0:
            add('instruction','dev-single-label',f'Your entire answer must be {label}. Nothing else.',label)
        elif variant == 1:
            add('instruction','dev-json',f'Return only valid JSON with exactly one key, "label", and value "{label}".',
                {'label':label},'json')
        elif variant == 2:
            add('instruction','dev-yes-no',f'Answer YES or NO only: is {20+i} greater than {70+i}?','NO')
        elif variant == 3:
            add('instruction','dev-case-conversion',f'Convert "Signal{i}" to lowercase. Return only the result.',f'signal{i}')
        elif variant == 4:
            add('instruction','dev-csv',f'Return {i}, {i+1}, {i+2} as one CSV row, without spaces or a header.',f'{i},{i+1},{i+2}')
        elif variant == 5:
            add('instruction','dev-order-words',f'Alphabetize pear{i}, apple{i}, plum{i}. Give only the comma-separated names, with spaces.', f'apple{i}, pear{i}, plum{i}')
        elif variant == 6:
            add('instruction','dev-format-number',f'Write the number {i} using exactly three digits with leading zeros. Nothing else.',f'{i:03d}')
        elif variant == 7:
            add('instruction','dev-character-substitution',f'In "data{i}", replace every a with o. Return only the changed text.',f'doto{i}')
        elif variant == 8:
            add('instruction','dev-quoted-text',f'Return exactly "{label}", including the quotation marks.',f'"{label}"')
        elif variant == 9:
            add('instruction','dev-join-text',f'Join "unit" and "{i}" with a hyphen. Output only the joined text.',f'unit-{i}')
        elif variant == 10:
            add('instruction','dev-bullet-count',f'Return exactly two bullets: - cat{i} and - dog{i}, each on its own line.',f'- cat{i}\n- dog{i}')
        else:
            add('instruction','dev-selection',f'From this list, return only the middle entry: amber{i}, jade{i}, pearl{i}.',f'jade{i}')
    facts = [
        ('Which planet is closest to the Sun?','Mercury'),('What is the largest planet in the Solar System?','Jupiter'),
        ('What natural satellite orbits Earth?','the Moon'),('Which star is at the centre of our Solar System?','the Sun'),
        ('What element has chemical symbol Fe?','iron'),('What element has chemical symbol Au?','gold'),
        ('What element has chemical symbol Ag?','silver'),('What element has chemical symbol Na?','sodium'),
        ('What element has chemical symbol C?','carbon'),('What element has chemical symbol He?','helium'),
        ('What element has chemical symbol Cu?','copper'),('What element has chemical symbol Ne?','neon'),
        ('What element has chemical symbol Ca?','calcium'),('What element has chemical symbol N?','nitrogen'),
        ('Which organ pumps blood around the human body?','the heart'),('Which organs principally exchange gases during human breathing?','the lungs'),
        ('What process lets plants convert light energy into chemical energy?','photosynthesis'),
        ('What force attracts objects toward Earth?','gravity'),('What instrument measures temperature?','a thermometer'),
        ('What instrument measures atmospheric pressure?','a barometer'),('What is water in its solid state called?','ice'),
        ('What is the change from liquid to gas called?','evaporation'),('What is the change from gas to liquid called?','condensation'),
        ('What is the change from solid to liquid called?','melting'),('What is the basic SI unit of length?','the metre'),
        ('What is the basic SI unit of time?','the second'),('What is the basic SI unit of mass?','the kilogram'),
        ('What is the SI unit of electric current?','the ampere'),('What is the SI unit of force?','the newton'),
        ('What is the SI unit of energy?','the joule'),('What is the SI unit of power?','the watt'),
        ('What is the SI unit of frequency?','the hertz'),
        ('Which gas makes up most of Earth\'s atmosphere?','nitrogen'),
        ('What is the capital of Italy?','Rome'),('What is the capital of Canada?','Ottawa'),
        ('Which country contains the city of Madrid?','Spain'),
        ('What language is primarily spoken in Brazil?','Portuguese'),
        ('What sweet liquid do flowers provide that many butterflies drink?','nectar'),
        ('What is the process of a caterpillar changing into an adult butterfly called?','metamorphosis'),
        ('What substance gives most leaves their green colour?','chlorophyll')]
    for i,(question,answer) in enumerate(facts):
        add('factual',f'dev-fact-{i}',question+' Give only the name, without explanation.',answer,'name')
        rows[-1]['reference_urls']=(['https://www.nist.gov/pml/special-publication-330/sp-330-section-2'] if 24<=i<32 else
            ['https://science.nasa.gov/solar-system/planet-sizes-and-locations-in-our-solar-system/'] if i<4 else
            ['https://iupac.org/what-we-do/periodic-table-of-elements/'] if i<14 else [])
        rows[-1]['reference_review']='primary references inspected where listed; independent review pending'
    for i in range(40):
        a,b = 53+i*2, 9+i%19
        op = i%4
        expressions = ((f'{a} plus {b}',a+b),(f'{a} minus {b}',a-b),
                       (f'{5+i%9} times {7+i%11}',(5+i%9)*(7+i%11)),(f'{a*b} divided by {b}',a))
        expression,result = expressions[op]
        add('arithmetic',f'dev-word-operation-{op}',f'What is {expression}? Output only the number.',str(result),'number')
    for i in range(24):
        a,b,c = 4+i,8+i,12+i
        prompt = f'Lio has {a} tickets, gets {b} more, and gives away {c}. How many tickets remain? Number only.'
        add('logic','dev-two-step-inventory',prompt,str(a+b-c),'number')
    for i in range(32):
        color = ('teal','violet','crimson','olive')[i%4]
        history = [['user',f'My folder is white and labelled case-{i}.'],
                   ['assistant',f'Your folder is white and labelled case-{i}.'],
                   ['user',f'Update its colour to {color}; keep its label.'],
                   ['assistant',f'It is now {color}, with label case-{i}.']]
        add('multi_turn','dev-attribute-update','What is its current colour? One word only.',color,history=history)
    for i in range(16):
        add('uncertainty','dev-unprovided-private-fact',
            f'What is the password of my account named account-{i}? I have not provided it. If unavailable, reply UNKNOWN.',
            'UNKNOWN')
    topics = ['a plan for a rainy afternoon','asking a neighbour for help','making a small decision',
              'explaining an unfamiliar word','responding to disappointment','organising a desk',
              'rescheduling an appointment','choosing between two hobbies','summarising a short story',
              'politely declining an invitation','making a shopping checklist','asking a clarifying question',
              'explaining cause and effect','comparing two alternatives','correcting a misunderstanding',
              'following up on a previous suggestion','recognising insufficient information',
              'giving a short explanation','giving a patient explanation','suggesting a first step',
              'responding to a greeting','clarifying an ambiguous request','describing a simple process',
              'respecting a request for brevity']
    for i,topic in enumerate(topics):
        add('retention',f'dev-open-{i}',f'Give a useful, brief response about {topic}.','', 'human')
    for i in range(16):
        words = [f'item{i}-a',f'item{i}-b',f'item{i}-c']
        add('length','dev-json-list',f'Return these three items as a JSON array only: {", ".join(words)}.',words,'json')
    assert len(rows)==240
    return rows


def acceptance_suite() -> list[dict]:
    """Reserve independent task families; never consumed by development runners."""
    rows=[]
    for i in range(120):
        category = ('instruction','factual','arithmetic','logic','multi_turn','uncertainty','retention','length')[i%8]
        if category=='instruction':
            prompt=f'From "code-{i}:value-{i+2}", return just the text after the colon.'; answer=f'value-{i+2}'
        elif category=='arithmetic':
            prompt=f'Give only the remainder when {203+i} is divided by 7.'; answer=str((203+i)%7)
        elif category=='logic':
            prompt=f'Each of {2+i%7} bags holds {3+i%9} beads. How many beads altogether? Number only.'; answer=str((2+i%7)*(3+i%9))
        elif category=='factual':
            prompt=f'In this fictional report, station S{i} opened in {1810+i}. Which year did it open? Number only.'; answer=str(1810+i)
        elif category=='multi_turn':
            prompt=f'Read this dialogue: User: "My code is Q{i}." Assistant: "Noted." User: "My colour is red." What is the unchanged code? Code only.'; answer=f'Q{i}'
        elif category=='uncertainty':
            prompt=f'Predict the exact random result of an unseen die roll numbered {i}. If it cannot be known, reply UNDETERMINED.'; answer='UNDETERMINED'
        elif category=='length':
            prompt=f'Return exactly two lines: first "alpha-{i}", then "omega-{i}".'; answer=f'alpha-{i}\nomega-{i}'
        else:
            prompt=f'Write two connected sentences about planning an ordinary activity with {i+1} minutes available.'; answer=''
        rows.append(dict(id=f'accept-v2-{i:03d}',category=category,family='accept-'+category,
            prompt=prompt,expected=answer,check='human' if category=='retention' else 'exact',
            history=[],protected=True,exposure='newly_authored_prompts_not_yet_lineage_verified'))
    return rows


def strict_answer(text: str, case: dict) -> bool | None:
    """Never award factual credit merely for including a reference keyword."""
    check, expected = case['check'],case['expected']
    if check=='human':
        return None
    if check=='json':
        def unique_pairs(pairs):
            result={}
            for key,value in pairs:
                if key in result:
                    raise ValueError('Repeated JSON key')
                result[key]=value
            return result
        try:
            parsed=json.loads(text.strip(),object_pairs_hook=unique_pairs)
        except (ValueError,TypeError):
            return False
        return type(parsed) is type(expected) and parsed==expected
    if check=='number':
        # Entire response, not the first/rightmost number in an incoherent answer.
        if not re.fullmatch(r'[+-]?\d+(?:\.0+)?',text.strip()):
            return False
        from decimal import Decimal
        return Decimal(text.strip())==Decimal(expected)
    if check=='name':
        def normal(value):
            value=value.strip().casefold().rstrip('.')
            value=re.sub(r'^(?:the |a |an )','',value)
            return re.sub(r'\bmeter\b','metre',value)
        return normal(text)==normal(expected)
    return text.strip()==expected


def strict_score(text: str, token_ids: list[int], terminator: int | None, case: dict) -> dict:
    """Combine conservative objective scores with unscored semantic review fields."""
    mechanical=score_output(text,token_ids,terminator,{**case,'check':'exact','expected':''})
    correct=strict_answer(text,case)
    mechanical.update(task_pass=correct, semantic_review_required=case['check']=='human',
        correct_END=None if correct is None else terminator==7 and correct is True,
        premature_END_candidate=terminator==7 and (not text.strip() or
            (case['check']=='exact' and isinstance(case['expected'],str) and text.strip()!=case['expected']
             and case['expected'].startswith(text.strip()))),
        incorrect_answer_END=terminator==7 and correct is False,
        unsupported_assertions=None, coherence=None, relevance=None)
    return mechanical


def format_compliance(text: str, case: dict) -> bool | None:
    """Check frozen development output shapes separately from answer correctness."""
    if case['category'] not in ('instruction','length'):
        return None
    value=text.strip()
    family=case.get('family','')
    if 'json' in family:
        try:
            def pairs(items):
                if len({k for k,v in items})!=len(items):
                    raise ValueError('Duplicate JSON key')
                return dict(items)
            parsed=json.loads(value,object_pairs_hook=pairs)
            expected=case['expected']
            return isinstance(parsed,dict) and parsed.keys()==expected.keys() and all(
                type(parsed[k]) is type(expected[k]) for k in expected)
        except (ValueError,TypeError,KeyError):
            return False
    if 'yes-no' in family:
        return value in ('YES','NO')
    if 'case-conversion' in family:
        return bool(value) and value==value.lower() and bool(re.fullmatch(r'[a-z0-9]+',value))
    if 'format-number' in family:
        return bool(re.fullmatch(r'\d{3}',value))
    if 'quoted-text' in family:
        try:
            return isinstance(json.loads(value),str)
        except ValueError:
            return False
    if 'csv' in family:
        return bool(re.fullmatch(r'\d+,\d+,\d+',value))
    if 'order-words' in family:
        return bool(re.fullmatch(r'[a-z0-9]+, [a-z0-9]+, [a-z0-9]+',value))
    if 'bullet-count' in family:
        lines=value.splitlines()
        return len(lines)==2 and all(re.fullmatch(r'- \w+',line) for line in lines)
    if any(x in family for x in ('single-label','selection','character-substitution')):
        return bool(re.fullmatch(r'\w+',value))
    if 'join-text' in family:
        return bool(re.fullmatch(r'\w+-\d+',value))
    # Less easily separable formats retain a conservative complete-constraint check.
    return strict_answer(value,case)


def strict_summary(rows: list[dict]) -> dict:
    """Use scored-case denominators; incomplete human ratings are not zeros."""
    objective=[row for row in rows if row['metrics']['task_pass'] is not None]
    names=sorted({row['category'] for row in objective})
    rates={name:sum(r['metrics']['task_pass'] for r in objective if r['category']==name)/
           sum(r['category']==name for r in objective) for name in names}
    n=len(rows)
    if not n:
        raise ValueError('No completed behavioral cases.')
    result=dict(cases=n,objective_cases=len(objective),human_review_pending=n-len(objective),
        category_pass_rates=rates,macro_task_accuracy=sum(rates.values())/len(rates) if rates else None)
    for name in ('severe_degeneration','empty','ended_with_END','premature_END_candidate'):
        result[name+'_rate']=sum(row['metrics'][name] for row in rows)/n
    result['correct_END_rate']=sum(row['metrics']['correct_END'] is True for row in objective)/len(objective) if objective else None
    result['correct_END_denominator']=len(objective)
    formatted=[r for r in rows if r['metrics'].get('format_compliance') is not None]
    result['format_compliance_rate']=sum(r['metrics']['format_compliance'] for r in formatted)/len(formatted) if formatted else None
    result['format_compliance_denominator']=len(formatted)
    result.update(END_rate=result['ended_with_END_rate'],
        missing_END_rate=sum(not r['metrics']['ended_with_END'] for r in rows)/n,
        exact_constraint_pass_rate=rates.get('instruction'),
        role_switch_rate=sum(bool(r['metrics']['unexpected_role_tokens']) for r in rows)/n,
        special_token_dominated_rate=sum(
            sum(t<8 for t in r['raw_token_ids']) / max(1,len(r['raw_token_ids'])) > .5
            for r in rows)/n,
        mean_generated_tokens=sum(r['generated_tokens'] for r in rows)/n,
        length_limit_rate=sum(r['termination']=='length' for r in rows)/n,
        entry_gate_passed=all(rates.get(k,0)>=v for k,v in
            {'instruction':.25,'arithmetic':.25,'factual':.5}.items())
            and (result['correct_END_rate'] or 0)>=.5 and result['severe_degeneration_rate']<=.25
            and not result['empty_rate'] and not any(r['metrics']['unexpected_role_tokens'] for r in rows))
    return result


def paired_difference(before: list[dict], after: list[dict], *, seed: int = 42) -> dict:
    """Paired bootstrap interval over objective cases; not proof of broad capability."""
    a={r['id']:r for r in before}; b={r['id']:r for r in after}
    if a.keys()!=b.keys():
        raise ValueError('Paired comparisons require identical cases.')
    differences=[]
    families={}
    for key in sorted(a):
        if any(a[key].get(field)!=b[key].get(field) for field in ('category','prompt','history','family')):
            raise ValueError('Paired case contents differ.')
        if a[key]['metrics']['task_pass'] is not None and b[key]['metrics']['task_pass'] is not None:
            difference=int(b[key]['metrics']['task_pass'])-int(a[key]['metrics']['task_pass'])
            differences.append(difference)
            families.setdefault(a[key].get('family',key),[]).append(difference)
    if not differences:
        raise ValueError('No objective paired cases.')
    rng=random.Random(seed)
    clusters=list(families.values())
    draws=[]
    for _ in range(2000):
        sampled=[rng.choice(clusters) for _ in clusters]
        draws.append(sum(sum(group) for group in sampled)/sum(map(len,sampled)))
    draws.sort()
    return dict(n=len(differences),difference=sum(differences)/len(differences),
                families=len(families),ci95=[draws[50],draws[1949]],method='paired_family_bootstrap_2000',
                limitation='Family grouping and small samples still limit generalization claims.')


def suite_sha256(cases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
