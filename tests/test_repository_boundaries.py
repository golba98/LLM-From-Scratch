"""Extraction contracts that cannot be proven by legacy compatibility adapters."""

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import pytest

from llm_architecture.model import LanguageModel, ModelConfig
from llm_tokenizer.tokenizer import load_tokenizer
from llm_inference.memory_adapter import DisabledMemory
from llm_memory.service import ConversationMemory

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CATALOG = json.loads((ROOT / "artifact-catalog/catalog.json").read_text())
LOCAL_PATH = ROOT / "artifacts.local.json"
LOCAL = json.loads(LOCAL_PATH.read_text()) if LOCAL_PATH.is_file() else {}
ORIGINAL = Path(LOCAL.get("reference_source", ROOT / "missing-reference"))
ASSETS = Path(os.environ.get("CODEXA_ASSET_ROOT", LOCAL.get("asset_root", ROOT / "missing-assets")))
PACKAGES = {
    "llm_architecture": "LLM-Architecture",
    "llm_tokenizer": "LLM-Tokenizer",
    "llm_data": "LLM-Data",
    "llm_training": "LLM-Training",
    "llm_inference": "LLM-Inference",
    "llm_memory": "LLM-Memory",
    "llm_specialist": "LLM-Specialist",
}
ALLOWED = {
    "llm_architecture": set(),
    "llm_tokenizer": set(),
    "llm_data": {"llm_tokenizer"},
    "llm_training": {"llm_architecture", "llm_tokenizer", "llm_data"},
    "llm_inference": {"llm_architecture", "llm_tokenizer", "llm_memory"},
    "llm_memory": set(),
    "llm_specialist": set(),
}


def test_component_import_graph_has_no_reverse_dependencies():
    for package, repo in PACKAGES.items():
        for path in (WORKSPACE / repo / "src").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [n.name for n in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                for name in names:
                    target = name.split(".")[0]
                    assert target not in {"src", "scripts", "workspace_bootstrap"}, (path, name)
                    if target in PACKAGES and target != package:
                        assert target in ALLOWED[package], (path, name)


def test_plain_inference_imports_without_optional_or_training_packages(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(WORKSPACE / PACKAGES[p] / "src") for p in ("llm_architecture", "llm_tokenizer", "llm_inference"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    code = "import sys; import llm_inference.native_chat; assert not any(p in sys.modules for p in ('llm_training', 'llm_data', 'llm_memory', 'llm_specialist', 'src', 'scripts'))"
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_code_and_tiny_outputs_match_preserved_original():
    path = ORIGINAL / "src/model.py"
    if not path.is_file():
        pytest.skip("Preserved original source fixture is local-only")
    # Migration parity is byte-exact against the historical architecture revision (the one hash-bound
    # into codexa-934m-foundation-5h-v1). The current source deliberately adds the masked KV-cache
    # repair, so it must instead keep state-dict keys, initialization and unmasked outputs identical.
    manifest = json.loads((ROOT / "compatibility.json").read_text())
    historical = manifest["historical_identity"]["component_commits"]["LLM-Architecture"]
    migrated = subprocess.run(["git", "-C", str(WORKSPACE / "LLM-Architecture"), "show",
                               f"{historical}:src/llm_architecture/model.py"], capture_output=True, check=True).stdout
    assert path.read_bytes() == migrated
    spec = importlib.util.spec_from_file_location("_original_model_parity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for position in ("learned", "rotary"):
        config = dict(vocab_size=32, context_length=16, num_layers=2, hidden_size=16, num_heads=4, intermediate_size=32, position_embedding_type=position)
        torch.manual_seed(42)
        original = module.LanguageModel(module.ModelConfig(**config))
        torch.manual_seed(42)
        extracted = LanguageModel(ModelConfig(**config))
        assert list(original.state_dict()) == list(extracted.state_dict())
        for key, value in original.state_dict().items():
            assert torch.equal(value, extracted.state_dict()[key])
        inputs = torch.tensor([[1, 8, 9, 10]])
        labels = torch.tensor([[8, 9, 10, 2]])
        a, loss_a = original(inputs, labels)
        b, loss_b = extracted(inputs, labels)
        assert torch.equal(a, b) and torch.equal(loss_a, loss_b)


def test_production_tokenizer_identity_is_unchanged():
    path = ASSETS / CATALOG["tokenizer"]["path"]
    if not path.is_file():
        pytest.skip("Production tokenizer is local-only")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CATALOG["tokenizer"]["sha256"]
    tokenizer = load_tokenizer(path)
    assert tokenizer.get_vocab_size() == 16384
    for text in ("Hello café 👋", "return value + 1", "Conversation memory"):
        assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_disabled_memory_retains_off_mode_turn_and_command_semantics():
    legacy = ConversationMemory(mode="off")
    extracted = DisabledMemory()
    for memory in (legacy, extracted):
        memory.remember("one", "two")
        memory.remember("three", "four")
    assert legacy.turn == extracted.turn == 2
    assert legacy.store is extracted.store is None
    assert legacy.command("sources", [])["sources"] == extracted.command("sources", [])["sources"]
    legacy.new_conversation()
    extracted.new_conversation()
    assert legacy.turn == extracted.turn == 0
