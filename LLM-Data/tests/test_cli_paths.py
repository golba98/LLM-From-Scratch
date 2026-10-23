from pathlib import Path
import pytest
from llm_data.cli.paths import asset_path, generated_path


def test_historical_default_inputs_are_separate_from_generated_defaults(tmp_path, monkeypatch):
    historical = tmp_path / "historical"
    output = tmp_path / "new-output"
    monkeypatch.setenv("CODEXA_ASSET_ROOT", str(historical))
    monkeypatch.setenv("CODEXA_OUTPUT_ROOT", str(output))
    assert asset_path("checkpoints/tokenizer.json") == historical / "checkpoints/tokenizer.json"
    assert generated_path("data/processed") == output / "data/processed"
    monkeypatch.setenv("CODEXA_OUTPUT_ROOT", str(historical))
    with pytest.raises(ValueError):
        generated_path("data/processed")
