"""Human review cannot be manufactured by automated validation or missing decisions."""

import json
import shutil
import subprocess
from types import SimpleNamespace
import pytest

from codexa_workspace.review.qualification import write_review_page, import_decisions,validate_record,build_blinded_review
from llm_data.conversational_audit import content_identity


def test_independent_arithmetic_and_context_checks(monkeypatch):
    monkeypatch.setattr('codexa_workspace.review.qualification.serialize_conversation',
        lambda *args,**kwargs:SimpleNamespace(input_ids=[8,7],labels=[8,7],supervised_token_count=2))
    row=dict(category='arithmetic',messages=[dict(role='user',content='Calculate 12 × 7. Give only its numerical value.'),
        dict(role='assistant',content='84')],tokens=2,assistant_tokens=2,
        source='curated',conversation_id='id',root_id='id',family='math',duplicate_group='math',
        provenance={'kind':'original_programmatic_curriculum','license':'CC0-1.0'})
    row['content_sha256']=content_identity(row['messages'])
    assert validate_record(row,None)['valid']
    row['messages'][-1]['content']='85';row['content_sha256']=content_identity(row['messages'])
    assert not validate_record(row,None)['valid']
    row['category']='multi_turn';row['messages']=[dict(role='user',content='I put my notes in Locker-1.'),
        dict(role='assistant',content='Your notes are in Locker-1.'),
        dict(role='user',content='I moved them to Cabinet-2. Where are they?'),
        dict(role='assistant',content='Locker-1')]
    row['content_sha256']=content_identity(row['messages'])
    assert not validate_record(row,None)['valid']


def test_blinded_names_and_mapping_integrity(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    (prepared/'development-suite.json').write_text(json.dumps([dict(id='x',expected='reference')]))
    comparison=tmp_path/'comparison.json';comparison.write_text(json.dumps(dict(results={
        'SECRET_MODEL':dict(checkpoint_sha256='sha',outputs=[dict(id='x',category='retention',
            history=[],prompt='Explain',response='answer')])})))
    package=build_blinded_review(comparison,prepared,tmp_path/'blind')
    assert 'SECRET_MODEL' not in (tmp_path/'blind/review.html').read_text()
    export=tmp_path/'decisions.json';export.write_text(json.dumps(dict(package_sha256=package['package_sha256'],
        reviewer='Real Human',method='human_manual_review',decisions={'Candidate A::x':{'decision':'approve'}},
        candidate_decisions={'Candidate A':{'accepted':True,'retention_acceptable':True}})))
    result=import_decisions(tmp_path/'blind/package.json',export,tmp_path/'import.json',
                           comparison=comparison,mapping=tmp_path/'blind/private-mapping.json')
    assert result['checkpoints']['sha']['reviewed_ids']==['x']
    (tmp_path/'blind/private-mapping.json').write_text(json.dumps({'Candidate A':'OTHER_MODEL'}))
    with pytest.raises(ValueError,match='mapping changed'):
        import_decisions(tmp_path/'blind/package.json',export,tmp_path/'other.json',
                         comparison=comparison,mapping=tmp_path/'blind/private-mapping.json')


def test_safe_html_package_and_missing_human_decision(tmp_path):
    package=write_review_page(tmp_path/'review',dict(kind='dataset',mixture_sha256='abc',
        items=[dict(id='one',group='factual',answer='</script><script>alert(1)</script>')]))
    page=(tmp_path/'review/review.html').read_text()
    assert '</script><script>alert(1)' not in page
    decision=tmp_path/'decision.json'
    base=dict(package_sha256=package['package_sha256'],reviewer='Real Reviewer',
              method='human_manual_review',decisions={})
    decision.write_text(json.dumps(base))
    with pytest.raises(ValueError,match='decision is missing'):
        import_decisions(tmp_path/'review/package.json',decision,tmp_path/'admitted.json')
    base['decisions']={'one':{'decision':'approve','comments':'Checked independently.'}}
    decision.write_text(json.dumps(base))
    result=import_decisions(tmp_path/'review/package.json',decision,tmp_path/'admitted.json')
    assert result['approved_record_ids']==['one'] and result['manifest_sha256']=='abc'


def test_rejections_and_package_drift_do_not_admit(tmp_path):
    data=write_review_page(tmp_path/'review',dict(kind='development',suite_sha256='suite',
                                               items=[dict(id='x',group='logic')]))
    decision=tmp_path/'decision.json'
    decision.write_text(json.dumps(dict(package_sha256=data['package_sha256'],reviewer='Human',
        method='human_manual_review',decisions={'x':{'decision':'reject'}})))
    with pytest.raises(ValueError,match='Rejected examples'):
        import_decisions(tmp_path/'review/package.json',decision,tmp_path/'approved.json')
    data['suite_sha256']='changed'
    (tmp_path/'review/package.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='modified'):
        import_decisions(tmp_path/'review/package.json',decision,tmp_path/'approved.json')


def test_review_page_exports_pending_without_implicit_approval(tmp_path):
    if shutil.which('node') is None:
        pytest.skip('Optional Node executable absent; Python package tests remain available.')
    package=write_review_page(tmp_path/'review',dict(kind='dataset',mixture_sha256='x',
        items=[dict(id='one',group='factual',answer='A')]))
    script=(tmp_path/'review/review.html').read_text().split('<script>')[-1].split('</script>')[0]
    harness=r'''
class Element{constructor(){this.children=[];this.value='all';this.selectedIndex=0;this.textContent='';}
append(x){this.children.push(x)}replaceChildren(){this.children=[]}click(){}}
const elements={};const document={getElementById:(id)=>elements[id]??=(new Element()),
createElement:()=>new Element(),querySelector:()=>new Element()};
document.getElementById('package').textContent=PACKAGE;
let exported;const Blob=class{constructor(parts){exported=parts.join('')}};
const URL={createObjectURL:()=>'',revokeObjectURL:()=>{}};const alert=()=>{};
'''.replace('PACKAGE',json.dumps(json.dumps(package)))
    footer="document.getElementById('reviewer').value='Fixture Reviewer';document.getElementById('export').onclick();console.log(exported);"
    run=subprocess.run(['node'],input=harness+script+footer,text=True,capture_output=True,check=True)
    exported=json.loads(run.stdout)
    assert exported['decisions']['one']['decision']=='pending'
    assert exported['reviewer']=='Fixture Reviewer' and 'approved' not in exported
