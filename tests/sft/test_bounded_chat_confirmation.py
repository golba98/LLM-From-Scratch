import copy
from pathlib import Path
import pytest
import yaml
from codexa_workspace.bounded_chat_confirmation import validate_spec
from llm_training.config import TrainingConfig


def specification():
    return yaml.safe_load((Path(__file__).parents[2] / 'configs/chat-end3-confirmation-v1.yaml').read_text())['spec']


def test_proposed_spec_uses_existing_training_configuration():
    spec = specification()
    validate_spec(spec)
    training = TrainingConfig(**spec['training'])
    assert training.micro_batch_size == 1
    assert training.gradient_accumulation_steps == 8
    assert spec['optimization']['target_token_weights'] == {'7': 3.0}
    assert spec['initialization']['semantics'] == 'weights_initialization'
    assert spec['training']['seed'] == 74


@pytest.mark.parametrize('change', ['budget', 'backend', 'reserve', 'steps', 'reclamation'])
def test_adapter_rejects_expanded_scope(change):
    spec = copy.deepcopy(specification())
    if change == 'budget': spec['wall_clock_seconds'] = 36000
    if change == 'backend': spec['checkpoint']['storage_backend'] = 'optimized'
    if change == 'reserve': spec['checkpoint']['reserve_bytes'] = 1
    if change == 'steps': spec['training']['max_steps'] = 10000
    if change == 'reclamation': spec['checkpoint']['reclamation_mode'] = 'execute'
    with pytest.raises(ValueError): validate_spec(spec)
