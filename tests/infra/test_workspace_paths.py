"""Sibling repository and protected-output contracts."""
import json
from pathlib import Path
import pytest
from codexa_workspace.assets import asset_root, resolve_input, output_root


def test_relocated_inputs_and_output_guard(tmp_path, monkeypatch):
    assets = tmp_path / "protected"
    assets.mkdir()
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "artifacts.local.json").write_text(json.dumps({"asset_root": str(assets), "input_aliases": {"/retired/source": str(assets)}}))
    monkeypatch.delenv("CODEXA_ASSET_ROOT", raising=False)
    monkeypatch.delenv("CODEXA_OUTPUT_ROOT", raising=False)
    assert asset_root(root) == assets
    assert resolve_input(root, "/retired/source/checkpoints/model.pt") == assets / "checkpoints/model.pt"
    assert output_root(root) == root / "outputs"
    monkeypatch.setenv("CODEXA_OUTPUT_ROOT", str(assets / "overwritten"))
    with pytest.raises(ValueError):
        output_root(root)


def test_component_manifest_uses_sibling_paths_and_independent_git_roots():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "compatibility.json").read_text())
    assert len(manifest["components"]) == 7
    for name, details in manifest["components"].items():
        assert details["path"] == "../" + name
        assert (root / details["path"] / ".git").is_dir()
        assert len(details["commit"]) == 40
