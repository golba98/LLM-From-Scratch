"""The SFT pilot's real components, end to end, on a tiny CPU toy model.

Step 5A exercised the trainer's control flow only with stand-ins; this drives the actual
wiring (dataset, fixed-order sampler, collation, train_model, the optimizer factory, the
recovery store, checkpoint payload and independent final verification, heartbeat, budget,
generation and teacher-forced evaluation) so integration defects surface before a real run.

No Codexa weights and no GPU are used: the model has a few hundred thousand parameters and
the optimizer is plain AdamW on CPU (the 8-bit optimizer is CUDA-only). The guarded trainer
functions require the explicit development environment, validated by the real guard.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from codexa_workspace import sft_phase4_data as data
from codexa_workspace import sft_pilot_evaluation as evaluation
from codexa_workspace import sft_pilot_runtime as runtime
from codexa_workspace import sft_pilot_trainer as trainer
from codexa_workspace import sft_step5a_verify as verify
from codexa_workspace.training_supervisor import TERMINAL

FROZEN_JOB = verify.STEP5A_BASE / '20261012T170336Z-3d5556ba/config/job.json'
pytestmark = [pytest.mark.usefixtures('development_training_environment'),
              pytest.mark.skipif(not FROZEN_JOB.exists(), reason='frozen Step 5A job not present (ignored asset)')]
TINY = dict(vocab_size=16384, context_length=2048, num_layers=2, hidden_size=32, num_heads=2,
            intermediate_size=64, dropout=0.0, tie_embeddings=True, position_embedding_type='learned',
            rope_theta=10000.0)
CONVERSATIONS = [[('user', f'What is {a} plus {b}?'), ('assistant', f'{a} plus {b} is {a + b}.')]
                 for a, b in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9))]


def _records(tokenizer, conversations, prefix):
    from llm_tokenizer.sft import ChatMessage, IGNORE_LABEL, serialize_conversation
    rows = []
    for index, turns in enumerate(conversations):
        messages = [dict(role=role, content=content) for role, content in turns]
        record = serialize_conversation([ChatMessage(r, c) for r, c in turns], tokenizer, maximum_tokens=512)
        labels = (*record.labels[1:], IGNORE_LABEL)
        rows.append(dict(conversation_id=f'{prefix}-{index}', messages=messages, root_id=f'{prefix}-root-{index}',
                         duplicate_group=f'{prefix}-group-{index}', source='toy',
                         content_sha256=verify._canonical_content_sha256(messages), tokens=len(record.input_ids),
                         assistant_tokens=sum(label != IGNORE_LABEL for label in labels),
                         historical_sft_exposure=False))
    return rows


@pytest.fixture
def toy_job(tmp_path, monkeypatch):
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    from llm_tokenizer.tokenizer import load_tokenizer
    frozen = data.read(FROZEN_JOB)
    spec = json.loads(json.dumps(frozen['spec']))
    tokenizer = load_tokenizer(spec['tokenizer']['path'])
    torch.manual_seed(0)
    model = LanguageModel(ModelConfig(**TINY))
    checkpoint = tmp_path / 'toy.pt'
    torch.save(dict(model_state_dict=model.state_dict(), config=dict(model=TINY)), checkpoint)
    (tmp_path / 'toy.pt.sha256').write_text(f'{data.digest(checkpoint)}  toy.pt\n')
    train, validation = tmp_path / 'train.jsonl', tmp_path / 'validation.jsonl'
    data.write_rows(train, _records(tokenizer, CONVERSATIONS, 'train'))
    data.write_rows(validation, _records(tokenizer, CONVERSATIONS[:2], 'validation'))
    manifest = tmp_path / 'manifest.json'
    orders = [list(range(8)), list(reversed(range(8)))]
    manifest.write_text(json.dumps(dict(orders=[dict(seed=s, indices=o) for s, o in zip((73, 74), orders)])))
    # Synthetic reference corpus: four 2,048-token windows over two sources.
    corpus = tmp_path / 'corpus.bin'
    np.random.default_rng(0).integers(8, 16384, size=4 * 2048 + 1, dtype=np.uint16).tofile(corpus)
    np.save(tmp_path / 'windows.npy', np.arange(4))
    (tmp_path / 'sources.json').write_text(json.dumps({'0': 'wikipedia', '1': 'wikipedia', '2': 'fineweb_edu',
                                                       '3': 'fineweb_edu'}))
    step_zero = tmp_path / 'step0.json'
    step_zero.write_text(json.dumps(dict(generation={'conversation/greedy': dict(severe_degeneration=16)},
                                         reference=dict(sources=dict(wikipedia=99., fineweb_edu=99.)))))
    spec.update(model=TINY, expected_parameters=count_parameters(model), run_name='toy-sft-wiring')
    spec['initialization'].update(path=str(checkpoint), sha256=data.digest(checkpoint))
    spec['data'].update(train=str(train), train_sha256=data.digest(train), validation=str(validation),
                        validation_sha256=data.digest(validation), manifest=str(manifest),
                        reference=dict(corpus_path=str(corpus), dtype='uint16',
                                       validation_window_ids=str(tmp_path / 'windows.npy'),
                                       validation_sources=str(tmp_path / 'sources.json'),
                                       source_loss_weights=dict(wikipedia=.5, fineweb_edu=.5)))
    spec['training'].update(max_steps=8, warmup_steps=2, gradient_accumulation_steps=2, precision='fp32', checkpoint_interval=4,
                            evaluation_interval=2)
    spec['optimization'].update(optimizer='adamw')
    spec['evaluation'].update(updates=[0, 2, 4, 8], reduced_updates=[2, 4], step_zero_path=str(step_zero))
    spec['checkpoint'].update(publications=[4, 'final'], reserve_bytes=1024**3, minimum_save_seconds=1)
    spec['budget'].update(total_seconds=900, finalization_reserve_seconds=60)
    # Short generations over a few frozen prompts keep the CPU run quick; the code path is unchanged.
    conversation, historical = evaluation.conversational_cases()[:2], evaluation.historical_cases()[:2]
    monkeypatch.setattr(evaluation, 'MAX_NEW_TOKENS', 4)
    monkeypatch.setattr(evaluation, 'conversational_cases', lambda: conversation)
    monkeypatch.setattr(evaluation, 'historical_cases', lambda: historical)
    return dict(spec=spec, job_sha256='toy')


def test_real_components_train_publish_verify_and_evaluate(toy_job, tmp_path):
    run_dir = tmp_path / 'run'
    report = trainer.run_sft_pilot(toy_job, run_dir, approval={}, environment={},
                                   components=trainer.default_components(device='cpu'))
    assert report['state'] == 'COMPLETED' and report['state'] in TERMINAL, report.get('error')
    assert report['stop_reason'] == 'STEP_CAP' and report['outcome'] == 'STEP_CAP'
    assert report['training_state']['optimizer_step'] == 8
    assert [c['optimizer_step'] for c in report['checkpoints']] == [4, 8]
    assert report['final_checkpoint_verified'] and report['final_checkpoint']['verified']
    index = json.loads((run_dir / 'checkpoints/recovery-index.json').read_text())
    assert index['final']['optimizer_step'] == 8 and index['final']['sha256'] == report['final_checkpoint']['sha256']
    payload = torch.load(run_dir / 'checkpoints' / index['final']['path'], map_location='cpu', weights_only=False)
    assert {'model_state_dict', 'optimizer_state_dict', 'scheduler_state', 'random_states', 'training_state',
            'resume_context', 'config'} <= set(payload)
    assert payload['resume_context']['configuration_sha256'] == runtime.fingerprint(toy_job['spec'])
    assert all(torch.isfinite(v).all() for v in payload['model_state_dict'].values())
    assert [e['step'] for e in report['evaluations']] == [2, 4, 8]
    schedules = [data.read(e['summary_path'])['schedule'] for e in report['evaluations']]
    assert schedules == ['reduced', 'reduced', 'full']
    final = data.read(report['evaluations'][-1]['summary_path'])
    assert {'conversation/greedy', 'conversation/nucleus', 'historical/greedy'} <= set(final['generation'])
    assert final['finite'] and final['gold']['conversations'] == 2
    for name in ('heartbeat', 'progress', 'budget', 'report'):
        assert (run_dir / f'metrics/{name}.json').exists(), name
    assert data.read(run_dir / 'metrics/heartbeat.json')['terminal_state'] in TERMINAL
    assert {'deadline_utc', 'remaining_seconds'} <= set(data.read(run_dir / 'metrics/budget.json'))
    # The read-only dashboard renders this run's files (it is the trainer's required viewer).
    from codexa_workspace.training_dashboard import render_text
    frame = render_text(run_dir, width=120, gpu=False, color=False)
    assert 'Closing this viewer never stops training' in frame and 'Traceback' not in frame
    # The supervisor's own classification of this run directory agrees.
    from codexa_workspace.training_supervisor import classify
    verdict = classify(run_dir, exit_status=dict(code=0), supervisor_reason=None, stop_signal=False)
    assert verdict['terminal_state'] == 'COMPLETED' and verdict['final_checkpoint_verification'] == 'verified'


def test_deadline_stop_completes_with_a_verified_final_checkpoint(toy_job, tmp_path):
    from dataclasses import replace
    import time
    from llm_training.session import SessionBudget
    parts = trainer.default_components(device='cpu')
    # A budget whose training deadline is already reached: train_model returns before any update.
    parts.budget = lambda seconds, reserve: replace(SessionBudget.start(seconds, reserve),
                                                    deadline_monotonic=time.monotonic() + reserve)
    run_dir = tmp_path / 'deadline'
    report = trainer.run_sft_pilot(toy_job, run_dir, approval={}, environment={}, components=parts)
    assert report['state'] == 'COMPLETED' and report['stop_reason'] == 'TRAINING_DEADLINE'
    assert report['outcome'] == 'INCONCLUSIVE_DEADLINE' and report['final_checkpoint_verified']
    assert report['training_state']['optimizer_step'] < 8
    from codexa_workspace.training_supervisor import classify
    verdict = classify(run_dir, exit_status=dict(code=0), supervisor_reason=None, stop_signal=False)
    assert (verdict['terminal_state'], verdict['reason']) == ('COMPLETED', 'TRAINING_DEADLINE')


def test_production_components_require_the_visible_dashboard():
    parts = trainer.default_components(device='cuda')
    assert parts.viewer is not None and trainer.default_components(device='cpu').viewer is None
