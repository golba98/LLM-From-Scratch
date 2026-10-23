"""Environment-based CLI defaults without depending on the integration package."""
import os
from pathlib import Path


def asset_path(relative):
    return Path(os.environ.get("CODEXA_ASSET_ROOT", ".")) / relative


def generated_path(relative):
    path = Path(os.environ.get("CODEXA_OUTPUT_ROOT", ".")) / relative
    protected = os.environ.get("CODEXA_ASSET_ROOT")
    if protected:
        root = Path(protected).resolve()
        resolved = path.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError("Generated outputs must not overwrite historical assets")
    return path
