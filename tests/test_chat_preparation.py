"""Readiness gates, process isolation and observable conversational scoring."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from codexa_workspace.chat_evaluation import make_suite, check_answer, score_output, fingerprint
from codexa_workspace.chat_operations import watch_owned_process, launch
from llm_training.session import SessionBudget


def test_development_and_acceptance_are_distinct_and_reproducible():
    development, acceptance = make_suite(), make_suite(acceptance=True)
    assert len(development) == len(acceptance) == 64
    assert fingerprint(development) == fingerprint(make_suite())
    assert not {r['prompt'] for r in development} & {r['prompt'] for r in acceptance}


def test_false_arithmetic_and_punctuation_loops_are_not_hidden():
    case = {'check': 'exact', 'expected': '42'}
    assert check_answer('42', case)
    assert not check_answer('19 + 24 = 42.', case)
    result = score_output('†' * 96, [300] * 96, None, case)
    assert result['severe_degeneration'] and not result['task_pass']
    assert not result['ended_with_END']


def test_user_facts_are_not_scored_as_assistant_identity():
    case = dict(check='contains_not', expected=['40'], forbidden=['39'])
    assert check_answer('Your new number is 40.', case)
    assert not check_answer('My new number is 40.', case)
    assert not check_answer('Your number is 39, or 40.', case)


def test_watchdog_bounds_only_its_owned_sleeping_worker(tmp_path):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    try:
        budget = SessionBudget.start(.25, .05)
        code = watch_owned_process(child, budget, tmp_path)
        assert code == -9
        assert json.loads((tmp_path / 'hard-deadline.json').read_text())['state'] == 'DEADLINE_KILLED'
    finally:
        if child.poll() is None:
            child.kill(); child.wait()


def test_failed_readiness_cannot_launch_a_production_process(tmp_path):
    preparation = tmp_path / 'preparation'; preparation.mkdir()
    (preparation / 'launch-manifest.json').write_text(json.dumps(
        {'readiness': 'BLOCKED', 'mandatory_gates': {'initialization_quality': False}}))
    with pytest.raises(RuntimeError, match='Production blocked'):
        launch(tmp_path, {}, preparation, tmp_path/'production', resume=False, started=None)
    assert not (tmp_path/'production').exists()


def test_two_hour_configuration_is_supported():
    from codexa_workspace.chat_preparation import read_configuration
    _, config, directory = read_configuration(Path('configs/chat-2h-v1.yaml'))
    assert config['wall_clock_seconds'] == 7200
    assert config['run_name'] == 'codexa-900m-chat-2h-v1'
    assert directory.name == config['run_name']


def test_experimental_authorization_preserves_technical_gates():
    from codexa_workspace.chat_operations import permitted_experimental_launch
    manifest = dict(experimental_authorization={'explicit_user_request': True},
                    spec={'purpose': 'experimental', 'wall_clock_seconds': 7200},
                    mandatory_gates={'initialization_quality': False, 'full_optimizer_recovery': True})
    assert permitted_experimental_launch(manifest)
    manifest['mandatory_gates']['full_optimizer_recovery'] = False
    assert not permitted_experimental_launch(manifest)


def test_optimized_storage_preflight_refuses_before_asset_reads(tmp_path, monkeypatch):
    import codexa_workspace.chat_preparation as prep
    # Identity check expects the actual generative environment.
    root = Path(sys.prefix).parent
    monkeypatch.setattr(prep.shutil, 'disk_usage', lambda p: type('Disk', (), {'free': 105 * 1024**3})())
    with pytest.raises(RuntimeError, match='Insufficient storage'):
        prep.preflight(root, {'minimum_disk_headroom_bytes': 100 * 1024**3,
            'checkpoint_storage_backend': 'optimized', 'maximum_optimizer_steps': 10}, tmp_path)
    assert not list(tmp_path.iterdir())
