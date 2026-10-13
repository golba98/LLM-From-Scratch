"""Reject incomplete or inconsistent Codexa production run manifests."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from llm_training.config import load_config
from llm_architecture.model import LanguageModel, count_parameters
from llm_tokenizer.tokenizer import SPECIAL_TOKENS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _null_paths(value: object, prefix: str = "") -> list[str]:
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, dict):
        return [
            item
            for key, nested in value.items()
            for item in _null_paths(nested, f"{prefix}.{key}" if prefix else str(key))
        ]
    if isinstance(value, list):
        return [
            item
            for index, nested in enumerate(value)
            for item in _null_paths(nested, f"{prefix}[{index}]")
        ]
    return []


def validate_manifest(path: Path) -> dict[str, object]:
    """Validate launch-blocking fields and architecture invariants."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Run manifest must contain an object.")
    nulls = _null_paths(value)
    if nulls:
        raise ValueError("Run manifest has unresolved decisions: " + ", ".join(nulls))
    if value.get("schema_version") != 1:
        raise ValueError("Run manifest schema_version must be 1.")
    architecture = value.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("Run manifest architecture must be an object.")
    config_path = Path(str(architecture.get("config_path")))
    config = load_config(config_path)
    import torch
    with torch.device("meta"):
        expected_parameters = count_parameters(LanguageModel(config.model))
    if architecture.get("parameter_count") != expected_parameters:
        raise ValueError("Manifest parameter count does not match the model.")
    if architecture.get("config_sha256") != _sha256(config_path):
        raise ValueError("Manifest configuration checksum mismatch.")
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise ValueError("Run manifest tokenizer must be an object.")
    expected_ids = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    if tokenizer.get("special_token_ids") != expected_ids:
        raise ValueError("Manifest special-token IDs are not the frozen IDs.")
    optimization = value.get("optimization")
    if not isinstance(optimization, dict):
        raise ValueError("Run manifest optimization must be an object.")
    expected_tokens = (
        config.model.context_length
        * int(optimization["micro_batch_size"])
        * int(optimization["gradient_accumulation"])
    )
    if optimization.get("tokens_per_optimizer_step") != expected_tokens:
        raise ValueError("Manifest tokens_per_optimizer_step is inconsistent.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    manifest = validate_manifest(arguments.manifest)
    if arguments.output_json:
        arguments.output_json.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("Production run manifest is complete and internally consistent.")


if __name__ == "__main__":
    main()
