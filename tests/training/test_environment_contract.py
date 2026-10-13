"""The explicit environment contract fails closed before any guarded trainer body runs.

No model, optimizer or real trainer module is imported: a sentinel module is wrapped by the
guard's own loader, exactly as the startup hook wraps llm_training.training.
"""
import types

import pytest

import training_environment_guard as guard


@pytest.fixture
def clean_selection(monkeypatch):
    monkeypatch.delenv("CODEXA_TRAINING_ENVIRONMENT", raising=False)
    monkeypatch.setattr(guard, "_selected", None)
    monkeypatch.setattr(guard.sys, "argv", ["pytest"])
    return monkeypatch


def guarded_sentinel():
    """A module loaded through GuardLoader whose guarded function records any call to its body."""
    calls = []

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.create_adamw_optimizer = lambda *args, **kwargs: calls.append((args, kwargs))
            module.train_model = lambda *args, **kwargs: calls.append((args, kwargs))

    module = types.ModuleType("sentinel")
    guard.GuardLoader(Loader(), "llm_training.training").exec_module(module)
    return module, calls


@pytest.mark.parametrize("function", ["create_adamw_optimizer", "train_model"])
def test_missing_selection_fails_before_trainer_body(clean_selection, function):
    module, calls = guarded_sentinel()
    with pytest.raises(RuntimeError, match="Explicit --environment"):
        getattr(module, function)()
    assert calls == []


def test_invalid_selection_is_rejected(clean_selection):
    clean_selection.setenv("CODEXA_TRAINING_ENVIRONMENT", "production")
    module, calls = guarded_sentinel()
    with pytest.raises(RuntimeError, match="Invalid training environment"):
        module.create_adamw_optimizer()
    assert calls == []


def test_conflicting_cli_and_inherited_selection_is_rejected(clean_selection):
    clean_selection.setenv("CODEXA_TRAINING_ENVIRONMENT", "development")
    clean_selection.setattr(guard.sys, "argv", ["trainer", "--environment", "historical"])
    module, calls = guarded_sentinel()
    with pytest.raises(RuntimeError, match="Explicit --environment"):
        module.train_model()
    assert calls == []


def test_repeated_cli_selection_is_ambiguous(clean_selection):
    clean_selection.setattr(guard.sys, "argv", ["trainer", "--environment", "development",
                                                "--environment=development"])
    with pytest.raises(RuntimeError, match="ambiguous"):
        guard.selected_environment()


def test_historical_selection_cannot_start_a_new_job(clean_selection):
    clean_selection.setenv("CODEXA_TRAINING_ENVIRONMENT", "historical")
    module, calls = guarded_sentinel()
    with pytest.raises(RuntimeError, match="New training jobs require the development environment"):
        module.create_adamw_optimizer()
    assert calls == []


def test_help_is_the_only_unchecked_path(clean_selection):
    """--help prints the contract and calls through (argparse exits); nothing else bypasses."""
    clean_selection.setattr(guard.sys, "argv", ["trainer", "--help"])
    module, calls = guarded_sentinel()
    module.create_adamw_optimizer()
    assert len(calls) == 1
