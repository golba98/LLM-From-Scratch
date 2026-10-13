"""Public review dashboard: blinding, safe rendering and export compatibility with existing importers."""

import json
import re
import shutil
import subprocess

import pytest

from codexa_workspace.review.qualification import build_blinded_review, import_decisions, write_review_page
from codexa_workspace.review.dashboard import publish_public_review, render_dashboard

SECRET = 'SECRET_MODEL_alpha'
OTHER = 'SECRET_MODEL_beta'


def blinded_fixture(tmp_path):
    prepared = tmp_path/'prepared'; prepared.mkdir()
    (prepared/'development-suite.json').write_text(json.dumps(
        [dict(id='q1', expected='Mercury'), dict(id='q2', expected='teal'), dict(id='q3', expected='')]))
    outputs = [dict(id='q1', category='factual', history=[], prompt='Which planet is closest to the Sun?',
                    response='Giant planet.\n\nUranus, Uranus, Uranus'),
               dict(id='q2', category='multi_turn', history=[['user', 'My folder is white.'], ['assistant', 'Noted.']],
                    prompt='What colour is it now?', response='</script><script>alert(1)</script> <b>teal</b>'),
               dict(id='q3', category='retention', history=[], prompt='Plan a rainy afternoon.', response='')]
    comparison = tmp_path/'comparison.json'
    comparison.write_text(json.dumps(dict(results={
        SECRET: dict(checkpoint_sha256='sha-alpha', outputs=outputs),
        OTHER: dict(checkpoint_sha256='sha-beta', outputs=outputs)})))
    package = build_blinded_review(comparison, prepared, tmp_path/'private')
    return package, comparison


def test_public_page_is_blinded_and_contains_only_index(tmp_path):
    package, _ = blinded_fixture(tmp_path)
    result = publish_public_review(tmp_path/'private', tmp_path/'public', mapping=tmp_path/'private/private-mapping.json')
    assert sorted(p.name for p in (tmp_path/'public').iterdir()) == ['index.html']
    assert result['package_sha256'] == package['package_sha256']
    page = (tmp_path/'public/index.html').read_text()
    for secret in (SECRET, OTHER, 'sha-alpha', package['mapping_sha256'], package['comparison_sha256']):
        assert secret not in page
    assert package['package_sha256'] in page
    # Original private evidence is untouched and still present.
    assert (tmp_path/'private/private-mapping.json').exists()


def test_page_is_inert_and_fully_local(tmp_path):
    blinded_fixture(tmp_path)
    publish_public_review(tmp_path/'private', tmp_path/'public')
    page = (tmp_path/'public/index.html').read_text()
    assert '</script><script>alert(1)' not in page and '<b>teal</b>' not in page
    assert re.search(r'https?://', page) is None
    assert re.search(r'<(link|img|iframe)\b|\bsrc=', page) is None
    assert 'innerHTML' not in page and 'eval(' not in page
    assert page.count('<script') == 2


def test_publish_refuses_tampering_nesting_and_reuse(tmp_path):
    blinded_fixture(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        publish_public_review(tmp_path/'private', tmp_path/'private/public')
    publish_public_review(tmp_path/'private', tmp_path/'public')
    with pytest.raises(FileExistsError):
        publish_public_review(tmp_path/'private', tmp_path/'public')
    data = json.loads((tmp_path/'private/package.json').read_text())
    data['items'][0]['answer'] = 'edited'
    (tmp_path/'private/package.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='modified'):
        publish_public_review(tmp_path/'private', tmp_path/'other')


def test_publish_refuses_identity_in_structural_fields(tmp_path):
    data = write_review_page(tmp_path/'private', dict(kind='model', items=[dict(id=f'{SECRET}::x', kind='model',
        group='factual', instruction=dict(candidate='Candidate A', history=[], prompt='p'), answer='a')]))
    (tmp_path/'map.json').write_text(json.dumps({'Candidate A': SECRET}))
    with pytest.raises(ValueError, match='private identity'):
        publish_public_review(tmp_path/'private', tmp_path/'public', mapping=tmp_path/'map.json')
    assert not (tmp_path/'public').exists() and data['package_sha256']


HARNESS = r'''
const CodexaCore = CORE_PLACEHOLDER;
const pkg = PKG_PLACEHOLDER;
const C = CodexaCore, items = pkg.items.map(C.normalizeItem), out = {};
const ids = items.map(i => i.id);
let state = C.emptyState(pkg, items);
const f = {status: 'all', group: 'all', alias: 'all', query: '', order: 'package'};
out.first = {prompt: items[0].prompt, answer: items[0].answer, alias: items[0].alias, group: C.groupLabel(items[0].group),
  reference: items[0].reference, history: items[1].history, emptyAnswer: items[2].answer};
out.all = C.filterItems(items, state, f, ['Candidate A', 'Candidate B']).length;
out.byAlias = C.filterItems(items, state, {...f, alias: 'Candidate B'}).map(i => items[i].alias);
out.byGroup = C.filterItems(items, state, {...f, group: 'multi_turn'}).length;
out.search = C.filterItems(items, state, {...f, query: 'CLOSEST sun'}).length;
out.searchId = C.filterItems(items, state, {...f, query: 'candidate b::q3'}).map(i => items[i].id);
out.byQuestion = C.filterItems(items, state, {...f, order: 'question'}, ['Candidate A', 'Candidate B']).map(i => items[i].id);
state.decisions[ids[0]].decision = 'approve';
state.decisions[ids[1]].decision = 'reject';
state.decisions[ids[2]] = {decision: 'unsure', comments: 'odd <b>', coherence: 1};
state.candidates['Candidate A'].accepted = true;
out.tally = C.tally(items, state);
out.pendingList = C.filterItems(items, state, {...f, status: 'pending'}).map(i => items[i].id);
out.reviewedCount = C.facetCounts(items, state, f, 'status').reviewed;
const list = C.filterItems(items, state, f);
out.nextUnreviewed = C.nextUnreviewed(list, 0, items, state);
out.step = [C.step(list, 0, -1), C.step(list, list.length - 1, 1)];
let missingName; try { C.buildExport(pkg, items, state, '  '); } catch (e) { missingName = e.message; }
out.missingName = missingName;
const exp = C.buildExport(pkg, items, state, 'Fixture Reviewer');
out.exported = exp;
const back = C.parseImport(JSON.stringify(exp), pkg, items);
out.roundTrip = JSON.stringify(back.state.decisions) === JSON.stringify(state.decisions)
  && JSON.stringify(back.state.candidates) === JSON.stringify(state.candidates);
out.roundTripNotices = back.notices;
const legacy = {format: 'codexa-human-review-export-v1', package_sha256: pkg.package_sha256, reviewer: 'Jordan',
  method: 'human_manual_review', decisions: Object.fromEntries(ids.map(id => [id, {decision: 'pending', comments: '', coherence: null}])),
  candidate_decisions: {'Candidate A': {accepted: false, retention_acceptable: true}, 'Candidate B': {accepted: false, retention_acceptable: false}}};
const leg = C.parseImport(JSON.stringify(legacy), pkg, items);
out.legacy = {candidates: leg.state.candidates, notices: leg.notices, reviewer: leg.state.reviewer, pending: C.tally(items, leg.state).pending};
const partial = {...legacy, decisions: {[ids[0]]: {decision: 'approve', comments: 'ok', coherence: 2}}};
out.partialNotices = C.parseImport(JSON.stringify(partial), pkg, items).notices;
const errors = {};
const attempt = (name, text) => { try { C.parseImport(text, pkg, items); errors[name] = null; } catch (e) { errors[name] = e.message; } };
attempt('invalid', '{not json');
attempt('mismatch', JSON.stringify({...legacy, package_sha256: 'f'.repeat(64)}));
attempt('unknownId', JSON.stringify({...legacy, decisions: {'Candidate Z::nope': {decision: 'approve'}}}));
attempt('badDecision', JSON.stringify({...legacy, decisions: {[ids[0]]: {decision: 'yes'}}}));
attempt('badCoherence', JSON.stringify({...legacy, decisions: {[ids[0]]: {decision: 'approve', coherence: 5}}}));
attempt('badMethod', JSON.stringify({...legacy, method: 'automatic'}));
attempt('badAlias', JSON.stringify({...legacy, candidate_decisions: {'Candidate Q': {accepted: true}}}));
out.errors = errors;
const draft = C.serializeDraft(pkg, state, ids[2]);
const restored = C.restoreDraft(draft, pkg, items);
out.draft = {same: JSON.stringify(restored.state.decisions) === JSON.stringify(state.decisions), currentId: restored.currentId,
  status: JSON.parse(draft).status, otherPackage: C.restoreDraft(draft.replace(pkg.package_sha256, 'x'), pkg, items)};
out.keys = [C.storageKey(pkg.package_sha256, ' Jordan '), C.storageKey(pkg.package_sha256, '')];
out.stop = [C.stopReason({ended_with_END: true}), C.stopReason({ended_with_END: false, ended_with_EOS: false}), C.stopReason(null)];
console.log(JSON.stringify(out));
'''


def run_core(tmp_path):
    if shutil.which('node') is None:
        pytest.skip('Optional Node executable absent; Python package tests remain available.')
    package, comparison = blinded_fixture(tmp_path)
    page = render_dashboard(package)
    core = page.split('/*CORE*/', 1)[1].split('/*END CORE*/', 1)[0]
    core = core.strip().removeprefix('const CodexaCore = ').rstrip().removesuffix(';')
    script = HARNESS.replace('CORE_PLACEHOLDER', core).replace('PKG_PLACEHOLDER', json.dumps(package))
    run = subprocess.run(['node'], input=script, text=True, capture_output=True, check=True)
    return package, comparison, json.loads(run.stdout)


def test_core_rendering_filtering_and_navigation(tmp_path):
    _, _, out = run_core(tmp_path)
    first = out['first']
    assert first['prompt'] == 'Which planet is closest to the Sun?'
    assert first['answer'] == 'Giant planet.\n\nUranus, Uranus, Uranus'
    assert first['alias'] == 'Candidate A' and first['group'] == 'Factual knowledge' and first['reference'] == 'Mercury'
    assert first['history'] == [dict(role='user', text='My folder is white.'), dict(role='assistant', text='Noted.')]
    assert first['emptyAnswer'] == ''
    assert out['all'] == 6 and out['byAlias'] == ['Candidate B'] * 3 and out['byGroup'] == 2
    assert out['search'] == 2 and out['searchId'] == ['Candidate B::q3']
    assert out['byQuestion'] == ['Candidate A::q1', 'Candidate B::q1', 'Candidate A::q2',
                                 'Candidate B::q2', 'Candidate A::q3', 'Candidate B::q3']
    assert out['tally']['approve'] == 1 and out['tally']['reject'] == 1 and out['tally']['unsure'] == 1
    assert out['tally']['pending'] == 3 and out['tally']['reviewed'] == 2 and out['reviewedCount'] == 2
    assert out['pendingList'] == ['Candidate B::q1', 'Candidate B::q2', 'Candidate B::q3']
    assert out['nextUnreviewed'] == 2 and out['step'] == [0, 5]
    assert out['stop'][0]['cut'] is False and out['stop'][1]['cut'] is True and out['stop'][2] is None


def test_export_semantics_and_importer_compatibility(tmp_path):
    package, comparison, out = run_core(tmp_path)
    exp = out['exported']
    assert out['missingName'].startswith('Enter the real reviewer name')
    assert exp['format'] == 'codexa-human-review-export-v1' and exp['method'] == 'human_manual_review'
    assert exp['package_sha256'] == package['package_sha256'] and exp['reviewer'] == 'Fixture Reviewer'
    assert set(exp['decisions']) == {x['id'] for x in package['items']}
    unsure = exp['decisions']['Candidate A::q3']
    assert unsure == dict(decision='pending', comments='odd <b>', coherence=1, needs_review=True)
    assert exp['candidate_decisions']['Candidate A'] == dict(accepted=True, retention_acceptable=None)
    assert out['roundTrip'] and out['roundTripNotices'] == []
    # Existing importer refuses pending/unsure work rather than treating it as a decision.
    path = tmp_path/'partial.json'; path.write_text(json.dumps(exp))
    with pytest.raises(ValueError, match='decision is missing'):
        import_decisions(tmp_path/'private/package.json', path, tmp_path/'r1.json',
                         comparison=comparison, mapping=tmp_path/'private/private-mapping.json')
    # Fully decided but an undecided (null) overall judgment is still refused.
    for d in exp['decisions'].values():
        d['decision'] = 'approve'; d.pop('needs_review', None)
    path.write_text(json.dumps(exp))
    with pytest.raises(ValueError, match='Missing overall candidate judgment'):
        import_decisions(tmp_path/'private/package.json', path, tmp_path/'r2.json',
                         comparison=comparison, mapping=tmp_path/'private/private-mapping.json')
    # A complete export from the dashboard is accepted by the unchanged importer.
    exp['candidate_decisions'] = {a: dict(accepted=False, retention_acceptable=True) for a in exp['candidate_decisions']}
    path.write_text(json.dumps(exp))
    result = import_decisions(tmp_path/'private/package.json', path, tmp_path/'r3.json',
                              comparison=comparison, mapping=tmp_path/'private/private-mapping.json')
    assert set(result['checkpoints']) == {'sha-alpha', 'sha-beta'}
    assert all(v['reviewed_ids'] == ['q1', 'q2', 'q3'] for v in result['checkpoints'].values())
    # Structural checks used by foundation_diagnosis.reviewed_comparison.
    assert exp['format'] == 'codexa-human-review-export-v1' and str(exp['reviewer']).strip()
    assert set(exp['decisions']) == {x['id'] for x in package['items']}


def test_legacy_import_validation_and_drafts(tmp_path):
    _, _, out = run_core(tmp_path)
    legacy = out['legacy']
    assert legacy['candidates']['Candidate A'] == dict(accepted=None, retention_acceptable=True)
    assert legacy['candidates']['Candidate B'] == dict(accepted=None, retention_acceptable=None)
    assert legacy['pending'] == 6 and legacy['reviewer'] == 'Jordan'
    assert len(legacy['notices']) == 1 and 'Not decided' in legacy['notices'][0]
    assert any('remain pending' in n for n in out['partialNotices'])
    errors = out['errors']
    assert 'not valid JSON' in errors['invalid']
    assert 'different review package' in errors['mismatch']
    assert 'not in this package' in errors['unknownId']
    assert 'unknown decision' in errors['badDecision']
    assert 'coherence' in errors['badCoherence']
    assert 'human_manual_review' in errors['badMethod']
    assert 'Unknown model alias' in errors['badAlias']
    assert all('Nothing was imported' in message for message in errors.values())
    draft = out['draft']
    assert draft['same'] and draft['currentId'] == 'Candidate A::q3'
    assert draft['status'] == 'local-draft-not-submitted' and draft['otherPackage'] is None
    assert out['keys'][0].endswith(':Jordan') and out['keys'][1].endswith(':(unnamed)')
