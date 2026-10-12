"""Make sibling packages available to preserved regression tests."""
import workspace_bootstrap
import os

import pytest

# Every pytest run is a unit test: notification code may only use mock backends.
# This can only downgrade a trusted production classification, never upgrade one.
os.environ["CODEXA_EXECUTION_CLASS"] = "unit_test"


@pytest.fixture
def development_training_environment():
    """Explicitly select the development environment for tests that call guarded trainer
    functions on tiny CPU models. Nothing is faked: the real guard then validates the real
    workspace (pinned, committed, source-first) and fails closed when it is not. A private
    MonkeyPatch keeps the selection when a test undoes its own monkeypatch."""
    import training_environment_guard as guard
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CODEXA_TRAINING_ENVIRONMENT", "development")
        patch.setattr(guard, "_selected", None)
        yield guard
