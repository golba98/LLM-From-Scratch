"""Tests for long-run preflight checks and storage estimates."""

from pathlib import Path
import tempfile

from src.preflight import (
    PreflightCheck,
    disk_capacity_check,
    estimate_checkpoint_storage,
    independent_filesystems,
)
from scripts.legacy.preflight_full_run import _backup_checks


def main() -> None:
    estimate = estimate_checkpoint_storage(
        parameter_count=250_000_000,
        max_steps=45_777,
        checkpoint_interval=1_000,
    )
    assert estimate.checkpoint_bytes == 3_001_048_576
    assert estimate.milestone_count == 46
    assert estimate.retained_checkpoint_count == 49
    assert estimate.projected_bytes == 147_051_380_224
    assert PreflightCheck("test", "pass", "ok").to_dict()["status"] == "pass"
    try:
        PreflightCheck("test", "unknown", "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid status should fail.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        path = Path(temporary_directory)
        assert disk_capacity_check(path, required_bytes=1).status == "pass"
        backup_check = disk_capacity_check(
            path,
            required_bytes=1,
            name="backup_capacity",
        )
        assert backup_check.name == "backup_capacity"
        assert backup_check.status == "pass"
        assert not independent_filesystems(path, path)
        waived = _backup_checks(
            path,
            None,
            required_bytes=1,
            accept_no_independent_backup=True,
        )
        assert len(waived) == 1
        assert waived[0].name == "independent_backup"
        assert waived[0].status == "pass"
        assert "explicitly accepted" in waived[0].detail
        missing = _backup_checks(
            path,
            None,
            required_bytes=1,
            accept_no_independent_backup=False,
        )
        assert missing[0].status == "fail"
        try:
            _backup_checks(
                path,
                path,
                required_bytes=1,
                accept_no_independent_backup=True,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Conflicting backup choices should fail.")
    try:
        estimate_checkpoint_storage(
            parameter_count=0,
            max_steps=1,
            checkpoint_interval=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid parameter count should fail.")

    print("All full-run preflight helper tests passed.")


if __name__ == "__main__":
    main()
