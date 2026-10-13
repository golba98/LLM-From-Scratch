"""Explicit workspace, asset and output paths; no retired-project fallback."""
import json
import os
from pathlib import Path
from .asset_mount import require_asset_mount


def workspace_root(explicit=None):
    if explicit or os.environ.get("CODEXA_WORKSPACE_ROOT"):
        root = Path(explicit or os.environ["CODEXA_WORKSPACE_ROOT"]).expanduser().resolve()
        if (root / "LLM-From-Scratch/compatibility.json").is_file():
            root = root / "LLM-From-Scratch"
        if not (root / "compatibility.json").is_file():
            raise ValueError("Workspace is missing compatibility.json")
        return root
    candidates = [Path(__file__).resolve().parents[2], *[Path.cwd(), *Path.cwd().parents]]
    for root in candidates:
        if (root / "LLM-From-Scratch/compatibility.json").is_file():
            return root / "LLM-From-Scratch"
        if (root / "compatibility.json").is_file():
            return root
    raise ValueError("Specify --workspace or CODEXA_WORKSPACE_ROOT outside the checkout")


def local_settings(root):
    path = Path(root) / "artifacts.local.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def asset_root(root):
    settings = local_settings(root)
    require_asset_mount(settings)
    value = os.environ.get("CODEXA_ASSET_ROOT") or settings.get("asset_root")
    if not value:
        raise ValueError("Set CODEXA_ASSET_ROOT or asset_root in artifacts.local.json")
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Asset root does not exist: {path}")
    return path


def resolve_input(root, value):
    path = Path(value)
    settings = local_settings(root)
    require_asset_mount(settings)
    aliases = settings.get("input_aliases", {})
    if path.is_absolute():
        for old, new in aliases.items():
            try:
                return Path(new) / path.relative_to(old)
            except ValueError:
                continue
        return path
    canonical = Path(root) / path
    # Preserve relative run references while routing explicitly migrated inputs.
    for old, new in aliases.items():
        try:
            return Path(new) / canonical.relative_to(old)
        except ValueError:
            continue
    return canonical if canonical.exists() else asset_root(root) / path


def output_root(root):
    value = os.environ.get("CODEXA_OUTPUT_ROOT") or local_settings(root).get("output_root", "outputs")
    path = Path(value)
    path = (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()
    assets = asset_root(root)
    if path == assets or assets in path.parents:
        raise ValueError("New outputs must not overwrite the historical asset store")
    return path


def resolve_output(root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (output_root(root) / path).resolve()
    settings = local_settings(root)
    protected = [asset_root(root), *[Path(p).resolve() for p in settings.get("input_aliases", {})],
                 *[Path(p).resolve() for p in settings.get("read_only_asset_roots", [])]]
    if any(path == p or p in path.parents for p in protected):
        raise ValueError("Outputs must not overwrite historical or retired inputs")
    return path


def normalize_paths(root, arguments, *, training=False):
    values = list(arguments)
    inputs = {"--checkpoint", "--init-checkpoint", "--resume", "--resume-checkpoint", "--tokenizer", "--train-jsonl", "--validation-jsonl", "--train-token-file", "--validation-token-file", "--base-token-file", "--token-manifest"}
    outputs = {"--log-dir", "--checkpoint-dir", "--output-dir", "--output"}
    for i, argument in enumerate(values):
        flag, separator, value = argument.partition("=")
        if flag not in inputs | outputs:
            continue
        if not separator:
            if i + 1 >= len(values):
                continue
            value = values[i + 1]
        resolved = resolve_input(root, value) if flag in inputs else resolve_output(root, value)
        if separator:
            values[i] = flag + "=" + str(resolved)
        else:
            values[i + 1] = str(resolved)
    if training:
        for flag, name in [("--log-dir", "logs"), ("--checkpoint-dir", "checkpoints")]:
            if not any(v == flag or v.startswith(flag + "=") for v in values):
                values.extend([flag, str(resolve_output(root, name))])
    return values
