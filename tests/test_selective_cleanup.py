from pathlib import Path
import os
import pytest
from codexa_workspace.selective_cleanup import identity, remove_inventoried, ALLOWLIST


def fixture(tmp_path):
    root = tmp_path / 'outputs'
    file = root / ALLOWLIST[0] / 'slot-0/checkpoint.pt'
    file.parent.mkdir(parents=True)
    file.write_bytes(b'fixture only')
    return root, file, tmp_path / 'journal.jsonl'


def test_exact_file_removal(tmp_path):
    root, file, journal = fixture(tmp_path)
    remove_inventoried(root, file, identity(file), journal)
    assert not file.exists()
    assert '"event": "deleted"' in journal.read_text()


def test_changed_file_retained(tmp_path):
    root, file, journal = fixture(tmp_path)
    record = identity(file)
    file.write_bytes(b'changed')
    with pytest.raises(ValueError):
        remove_inventoried(root, file, record, journal)
    assert file.exists()


def test_symlink_parent_rejected(tmp_path):
    root, file, journal = fixture(tmp_path)
    record = identity(file)
    moved = file.parent.with_name('moved')
    file.parent.rename(moved)
    file.parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(OSError):
        remove_inventoried(root, file, record, journal)
    assert file.exists()


def test_hardlinks_rejected(tmp_path):
    _, file, _ = fixture(tmp_path)
    os.link(file, file.with_name('alias'))
    with pytest.raises(ValueError):
        identity(file)


def test_outside_scope_rejected(tmp_path):
    root, file, journal = fixture(tmp_path)
    other = root / 'protected/checkpoint.pt'
    other.parent.mkdir()
    other.write_bytes(b'protected')
    with pytest.raises(ValueError):
        remove_inventoried(root, other, identity(other), journal)
    assert other.exists()


def test_detached_hash_failure_restores_original(tmp_path):
    root, file, journal = fixture(tmp_path)
    with pytest.raises(ValueError):
        remove_inventoried(root, file, identity(file), journal, expected_digest='0' * 64)
    assert file.read_bytes() == b'fixture only'
    assert 'restored_changed_content' in journal.read_text()


def test_interrupted_unlink_preserves_detached_payload(tmp_path, monkeypatch):
    root, file, journal = fixture(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('injected unlink failure')
    monkeypatch.setattr(os, 'unlink', fail)
    with pytest.raises(OSError):
        remove_inventoried(root, file, identity(file), journal)
    assert not file.exists()
    assert len(list(file.parent.glob('.retired-*'))) == 1
    assert '"event": "deleted"' not in journal.read_text()
