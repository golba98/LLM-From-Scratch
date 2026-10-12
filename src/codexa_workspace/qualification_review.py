"""Bounded validation and honest, fingerprint-bound human review packages."""

from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3

from llm_data.conversational_audit import content_identity
from llm_tokenizer.sft import ChatMessage, serialize_conversation
from llm_training.session import atomic_document


def digest(path: Path) -> str:
    """Fingerprint a complete file without loading it in memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_record(row: dict, tokenizer) -> dict:
    """Independently check mechanically decidable examples; never certify semantics."""
    failures, warnings = [], []
    try:
        messages = row['messages']
        record = serialize_conversation([ChatMessage(m['role'], m['content']) for m in messages],
                                        tokenizer, maximum_tokens=2048)
        if record.supervised_token_count != row['assistant_tokens'] or len(record.input_ids) != row['tokens']:
            failures.append('Serialized token counts differ.')
        if content_identity(messages) != row['content_sha256']:
            failures.append('Normalized content fingerprint differs.')
        if record.labels[-1] != 7 or record.supervised_token_count <= 0:
            failures.append('Missing supervised END or empty target.')
        if row.get('provenance', {}).get('kind') != 'original_programmatic_curriculum':
            failures.append('Unexpected provenance.')
        if (row.get('source')!='curated' or row.get('provenance',{}).get('license')!='CC0-1.0'
                or not row.get('conversation_id') or row.get('root_id')!=row.get('conversation_id')
                or row.get('duplicate_group')!=row.get('family')):
            failures.append('Source, license or grouping provenance is inconsistent.')
        if record.input_ids[0] in (4,5):
            assistant=False
            for token,label in zip(record.input_ids,record.labels,strict=True):
                if token in (4,5,6):
                    assistant=token==6
                    expected=-100
                else:
                    expected=token if assistant else -100
                if label!=expected:
                    failures.append('Assistant-only label mask is inconsistent.')
                    break
                if token==7:
                    assistant=False
        prompt, answer = messages[-2]['content'], messages[-1]['content']
        category = row['category']
        if category == 'arithmetic':
            match = re.fullmatch(r'Calculate (\d+) ([+−\-×÷]) (\d+)\. Give only its numerical value\.', prompt)
            if match is None:
                failures.append('Arithmetic expression is not independently parseable.')
            else:
                a, op, b = Decimal(match[1]), match[2], Decimal(match[3])
                expected = {'+': lambda: a+b, '-': lambda: a-b, '−': lambda: a-b,
                            '×': lambda: a*b, '÷': lambda: a/b}[op]()
                if not re.fullmatch(r'-?\d+(?:\.\d+)?', answer) or Decimal(answer) != expected:
                    failures.append('Incorrect arithmetic result or extra answer text.')
        elif category == 'instruction':
            if not re.fullmatch(r'[A-Za-z]+\d+', answer) or answer not in prompt:
                failures.append('Single-label instruction is not satisfied.')
        elif category == 'length':
            expected = prompt.split(': ', 1)[1].removesuffix('.').split(', ')
            if answer.splitlines() != expected or len(expected) != 3:
                failures.append('Three-line output constraint violated.')
        elif category == 'reasoning':
            match = re.search(r'greater: (\d+) or (\d+)\?', prompt)
            if match is None or answer != str(max(int(match[1]), int(match[2]))):
                failures.append('Integer comparison answer is incorrect.')
        elif category == 'multi_turn':
            match = re.search(r'I moved them to ([\w-]+)\.', prompt)
            old = re.search(r'in ([\w-]+)\.', messages[0]['content'])
            if (match is None or old is None or answer != match[1]
                    or messages[1]['content'] != f'Your notes are in {old[1]}.'):
                failures.append('Conversation location update is inconsistent.')
        else:
            warnings.append('Semantic/factual approval requires a human decision.')
        if category == 'natural':
            warnings.append('Generic template may be awkward or insufficiently helpful; inspect individually.')
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        failures.append(f'{type(error).__name__}: {error}')
    return dict(valid=not failures, failures=failures, warnings=warnings)


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Codexa independent review</title><style>
body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:15px;background:#fafafa;color:#202020}
header{position:sticky;top:0;background:#fafafa;padding:10px;border-bottom:1px solid #ccc}
article{background:white;border:1px solid #ddd;border-radius:8px;margin:14px 0;padding:18px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:10px}
textarea{width:95%;height:55px}select,input,button{font:inherit;margin:5px;padding:6px}
.warning{color:#783d00}.bad{color:#a00000}small{color:#555}label{display:block}
</style><header><h1>Codexa review</h1><p>Decisions are yours. Automated findings are not human approval.
No acceptance answers are included. Nothing is uploaded.</p>
<label>Reviewer name <input id="reviewer"></label><button id="export">Export decisions JSON</button>
<select id="filter"><option value="all">All groups</option></select><span id="progress"></span></header>
<main id="records"></main><script id="package" type="application/json">PACKAGE</script><script>
const data=JSON.parse(document.getElementById('package').textContent), decisions={}, candidates={};
const make=(tag,text,parent)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;parent?.append(e);return e};
for(const group of [...new Set(data.items.map(x=>x.group))]){const e=make('option',group,document.getElementById('filter'));e.value=group}
const update=()=>document.getElementById('progress').textContent=` ${Object.values(decisions).filter(x=>x.decision!=='pending').length}/${data.items.length} decisions`;
function render(){const root=document.getElementById('records');root.replaceChildren();const filter=document.getElementById('filter').value;
 for(const x of data.items){if(filter!=='all'&&x.group!==filter)continue;
 const a=make('article',undefined,root);make('h2',x.group+' — '+x.id,a);
 for(const key of ['instruction','answer','provenance','findings','warnings']){if(x[key]!==undefined){make('strong',key,a);make('pre',typeof x[key]==='string'?x[key]:JSON.stringify(x[key],null,2),a)}}
 const d=decisions[x.id]??={decision:'pending',comments:'',coherence:null};
 const sel=make('select',undefined,a);for(const value of ['pending','approve','reject']){const o=make('option',value,sel);o.value=value}sel.value=d.decision;sel.onchange=()=>{d.decision=sel.value;update()};
 if(x.kind==='model'){const rating=make('select',undefined,a);for(const v of ['Unrated','0 — incoherent','1 — partly coherent','2 — coherent'])make('option',v,rating);rating.selectedIndex=d.coherence===null?0:d.coherence+1;rating.onchange=()=>d.coherence=rating.selectedIndex===0?null:rating.selectedIndex-1}
 const comment=make('textarea',undefined,a);comment.placeholder='Reviewer comments';comment.value=d.comments;comment.oninput=()=>d.comments=comment.value;
 }update()}
for(const alias of data.candidate_aliases??[]){const a=make('article',undefined,document.querySelector('header'));make('strong',alias+' overall judgment',a);const state=candidates[alias]={accepted:false,retention_acceptable:false};for(const key of Object.keys(state)){const l=make('label',key,a),c=make('input',undefined,l);c.type='checkbox';c.onchange=()=>state[key]=c.checked}}
document.getElementById('filter').onchange=render;render();
document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('Enter the real reviewer name.');return}
 const result={format:'codexa-human-review-export-v1',package_sha256:data.package_sha256,reviewer,method:'human_manual_review',decisions,candidate_decisions:candidates};
 const a=make('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='codexa-review-decisions.json';a.click();URL.revokeObjectURL(a.href)};
</script></html>'''


def write_review_page(directory: Path, package: dict) -> dict:
    """Publish a fresh local page and its immutable decision identity."""
    directory.mkdir(parents=True, exist_ok=False)
    body = json.dumps(package, sort_keys=True, ensure_ascii=False)
    package = {**package, 'package_sha256': hashlib.sha256(body.encode()).hexdigest()}
    atomic_document(directory/'package.json', package)
    encoded = json.dumps(package, ensure_ascii=False).replace('<', '\\u003c')
    (directory/'review.html').write_text(PAGE.replace('PACKAGE', encoded))
    return package


def build_dataset_review(prepared: Path, ledger: Path, tokenizer, directory: Path) -> dict:
    """Reuse frozen proposed records and the existing read-only audit ledger."""
    source = prepared/'curated/train.proposed.jsonl'
    rows = [json.loads(line) for line in source.open()]
    families = Counter(row['family'] for row in rows)
    identities = {content_identity(row['messages']) for row in rows}
    connection = sqlite3.connect(f'file:{ledger.absolute()}?mode=ro', uri=True)
    matches = []
    try:
        for dataset, split, value in connection.execute('SELECT dataset,split,content FROM records'):
            if value in identities:
                matches.append(dict(dataset=dataset, split=split, content=value))
    finally:
        connection.close()
    dev = json.loads((prepared/'development-suite.json').read_text())
    prompts = {m['content'].strip().casefold() for row in rows for m in row['messages'] if m['role']=='user'}
    overlap = [case['id'] for case in dev if case['prompt'].strip().casefold() in prompts]
    items, results = [], []
    for row in rows:
        checks = validate_record(row, tokenizer)
        results.append(dict(id=row['conversation_id'], category=row['category'], **checks))
        if row['review_status']=='pending_human' or not checks['valid']:
            items.append(dict(id=row['conversation_id'], kind='dataset', group=row['category'],
                instruction=row['messages'][:-1], answer=row['messages'][-1]['content'],
                provenance=row['provenance'], findings=checks,
                warnings=[f"Template family {row['family']} has {families[row['family']]} records.",
                          'Exact uniqueness does not prove semantic independence.']))
    dataset_package = dict(format='codexa-review-package-v1', kind='dataset', source_sha256=digest(source),
        mixture_sha256=digest(prepared/'mixture-low-exposure.proposed.json'), items=items)
    write_review_page(directory/'dataset', dataset_package)
    dev_items = [dict(id=c['id'], kind='development', group=c['category'],
        instruction={'history':c['history'], 'prompt':c['prompt']}, answer=c['expected'],
        warnings=['Check correctness, ambiguity and independence before admitting this development case.']) for c in dev]
    from .behavioral_readiness import suite_sha256
    write_review_page(directory/'development', dict(format='codexa-review-package-v1', kind='development',
        suite_sha256=suite_sha256(dev), items=dev_items))
    report = dict(records=len(rows), automatically_valid=sum(r['valid'] for r in results),
        rejected=sum(not r['valid'] for r in results), human_approved=0, pending=len(items),
        results=results, exact_historical_matches=matches, exact_development_prompt_matches=overlap,
        acceptance='Identity preserved; no acceptance answers consumed.',
        limitations=['Semantic/template independence unresolved; no human decisions invented.'])
    atomic_document(directory/'validation.json', report)
    return report


def build_blinded_review(comparison: Path, prepared: Path, directory: Path) -> dict:
    """Hide checkpoint identities from a consistent per-candidate review package."""
    document = json.loads(comparison.read_text())
    dev = {c['id']:c for c in json.loads((prepared/'development-suite.json').read_text())}
    names = sorted(document['results'])
    random.Random(42).shuffle(names)
    mapping = {f'Candidate {chr(65+i)}':name for i,name in enumerate(names)}
    items = []
    for alias, name in mapping.items():
        result = document['results'][name]
        for row in result['outputs']:
            if row['category'] in ('factual','multi_turn','retention'):
                items.append(dict(id=alias+'::'+row['id'], kind='model', group=row['category'],
                    instruction={'candidate':alias,'history':row['history'],'prompt':row['prompt']},
                    answer=row['response'], findings={'reference':dev[row['id']]['expected']},
                    warnings=['Rate meaning and relevance, not reference-word presence.']))
    package = write_review_page(directory, dict(format='codexa-review-package-v1',kind='model',
        comparison_sha256=digest(comparison),candidate_aliases=list(mapping),items=items,
        mapping_sha256=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()))
    atomic_document(directory/'private-mapping.json', mapping)
    return package


def import_decisions(package: Path, decisions: Path, output: Path,
                     *, comparison: Path | None = None, mapping: Path | None = None) -> dict:
    """Translate real completed decisions to existing APIs; missing/rejected admission blocks."""
    data = json.loads(package.read_text()); review = json.loads(decisions.read_text())
    if data.get('kind') not in ('dataset','development','model'):
        raise ValueError('Screening or unknown review packages cannot authorize admission or model selection.')
    if (review.get('package_sha256') != data['package_sha256'] or not review.get('reviewer')
            or review.get('method') != 'human_manual_review'):
        raise ValueError('Human review identity is missing or mismatched.')
    unsigned = {k:v for k,v in data.items() if k!='package_sha256'}
    if hashlib.sha256(json.dumps(unsigned,sort_keys=True,ensure_ascii=False).encode()).hexdigest()!=data['package_sha256']:
        raise ValueError('Review package was modified.')
    for item in data['items']:
        value = review.get('decisions', {}).get(item['id'], {}).get('decision')
        if value not in ('approve','reject'):
            raise ValueError('An explicit human decision is missing: '+item['id'])
    result = dict(reviewer=review['reviewer'], method='human_manual_review',
                  review_export_sha256=digest(decisions))
    if data['kind'] in ('dataset','development'):
        rejected = [x['id'] for x in data['items'] if review['decisions'][x['id']]['decision']=='reject']
        if rejected:
            raise ValueError('Rejected examples require a new isolated proposal, never blanket admission: '+str(rejected))
        result.update(approved=True, approved_record_ids=[x['id'] for x in data['items']])
        if data['kind']=='dataset':
            result['manifest_sha256'] = data['mixture_sha256']
        else:
            result['suite_sha256'] = data['suite_sha256']
    else:
        if comparison is None or mapping is None or digest(comparison)!=data['comparison_sha256']:
            raise ValueError('Exact comparison and private mapping are required.')
        document = json.loads(comparison.read_text()); aliases=json.loads(mapping.read_text())
        if hashlib.sha256(json.dumps(aliases,sort_keys=True).encode()).hexdigest()!=data['mapping_sha256']:
            raise ValueError('Blinded candidate mapping changed.')
        result.update(comparison_sha256=digest(comparison), checkpoints={})
        for alias,name in aliases.items():
            judgments=review.get('candidate_decisions',{}).get(alias,{})
            if any(type(judgments.get(k)) is not bool for k in ('accepted','retention_acceptable')):
                raise ValueError('Missing overall candidate judgment.')
            result['checkpoints'][document['results'][name]['checkpoint_sha256']]={**judgments,
                'reviewed_ids':[x['id'].split('::',1)[1] for x in data['items'] if x['id'].startswith(alias+'::')]}
    if output.exists():
        raise FileExistsError('Review imports require a fresh output file.')
    atomic_document(output,result)
    return result
