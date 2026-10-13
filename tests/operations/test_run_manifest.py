"""Tests for launch-blocking production manifest validation."""

from pathlib import Path
import tempfile

import yaml

from scripts.legacy.validate_run_manifest import validate_manifest


def test_template_is_intentionally_blocked_by_unresolved_decisions() -> None:
    try:
        validate_manifest(Path("configs/production_run_manifest.template.yaml"))
    except ValueError as error:
        assert "unresolved decisions" in str(error)
        assert "total_token_budget" in str(error)
    else:
        raise AssertionError("Incomplete production template must be rejected.")


def test_inconsistent_complete_manifest_is_rejected() -> None:
    value = yaml.safe_load(
        Path("configs/production_run_manifest.template.yaml").read_text(
            encoding="utf-8"
        )
    )

    def fill(item):
        if isinstance(item, dict):
            return {key: fill(nested) for key, nested in item.items()}
        if isinstance(item, list):
            return [fill(nested) for nested in item]
        return "resolved" if item is None else item

    complete = fill(value)
    complete["architecture"]["parameter_count"] = 1
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "manifest.yaml"
        path.write_text(yaml.safe_dump(complete), encoding="utf-8")
        try:
            validate_manifest(path)
        except ValueError as error:
            assert "parameter count" in str(error)
        else:
            raise AssertionError("Mismatched model count must be rejected.")
