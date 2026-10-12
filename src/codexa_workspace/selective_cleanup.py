"""Identity-checked removal of explicitly inventoried retired checkpoint files.

No directory recursion, retention policy or automatic discovery is provided.
Caller must establish dependency, export and retirement gates before use.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

ALLOWLIST = (
    'preparation/codexa-900m-chat-10h-v1',
    'investigation/chat-regression-20261010/pilots/repair-proportional',
    'investigation/chat-regression-20261010/pilots/repair-balanced',
    'investigation/chat-regression-20261010/pilots/balanced-proportional',
    'investigation/chat-regression-20261010/pilots/balanced-balanced',
)


def identity(path: Path) -> dict:
    """Record inode identity without following a final symlink."""
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError('Expected an exclusively owned regular file: ' + str(path))
    return dict(device=s.st_dev, inode=s.st_ino, bytes=s.st_size,
                allocated=s.st_blocks * 512, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, mode=s.st_mode, links=s.st_nlink)


def digest(path: Path) -> str:
    """Hash one explicit file."""
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def scoped_parent(root: Path, path: Path) -> int:
    """Open a no-follow directory chain, rejecting mount and scope escapes."""
    root, path = root.absolute(), path.absolute()
    relative = path.relative_to(root)
    if '..' in relative.parts or not any(relative.is_relative_to(Path(p)) for p in ALLOWLIST):
        raise ValueError('Outside approved cleanup scope.')
    # Restrict operations to native payloads, their paired records and retired indexes.
    if path.name not in ('checkpoint.pt', 'checkpoint.pt.sha256', 'checkpoint.pt.complete.json', 'recovery-index.json'):
        raise ValueError('Not a checkpoint artifact.')
    current = Path('/')
    fd = os.open(current, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parent.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if os.fstat(fd).st_dev != root.stat().st_dev:
            raise ValueError('Unexpected filesystem.')
        # Also reject intermediate mounts beneath the approved root.
        for parent in (path.parent, *path.parent.parents):
            if parent == root:
                break
            if parent.stat().st_dev != root.stat().st_dev or os.path.ismount(parent):
                raise ValueError('Unexpected mount point.')
        return fd
    except BaseException:
        os.close(fd)
        raise


def _rename_exclusive(fd: int, source: str, target: str) -> None:
    function = ctypes.CDLL(None, use_errno=True).renameat2
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if function(fd, os.fsencode(source), fd, os.fsencode(target), 1):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def remove_inventoried(root: Path, path: Path, record: dict, journal: Path, *, expected_digest: str | None = None) -> None:
    """Detach one checked inode, recheck it and unlink; journal every boundary.

    Execution is deliberately not exposed through training configuration.
    A crash after detachment leaves a recorded quarantine file, never a blind
    recursive cleanup. Operators must investigate it before continuing.
    """
    fd = scoped_parent(root, path)
    detached = '.retired-' + uuid.uuid4().hex
    def write(event: str) -> None:
        with journal.open('a') as handle:
            handle.write(json.dumps(dict(event=event, path=str(path), detached=detached, record=record)) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
    try:
        if identity(path) != record:
            raise ValueError('File changed after inventory.')
        write('intent')
        _rename_exclusive(fd, path.name, detached)
        os.fsync(fd)
        write('detached')
        observed = os.stat(detached, dir_fd=fd, follow_symlinks=False)
        # Rename changes ctime. All other identity fields must remain equal.
        expected = (record['device'], record['inode'], record['bytes'], record['mtime_ns'], record['mode'], 1)
        actual = (observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_mode, observed.st_nlink)
        if actual != expected:
            _rename_exclusive(fd, detached, path.name)
            os.fsync(fd)
            write('restored_changed_file')
            raise ValueError('Detached identity mismatch; not deleted.')
        if expected_digest is not None:
            file_fd = os.open(detached, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                h = hashlib.sha256()
                while block := os.read(file_fd, 4 * 1024**2):
                    h.update(block)
                checked = os.fstat(file_fd)
            finally:
                os.close(file_fd)
            if (h.hexdigest() != expected_digest or checked.st_mtime_ns != observed.st_mtime_ns
                    or checked.st_ctime_ns != observed.st_ctime_ns):
                _rename_exclusive(fd, detached, path.name)
                os.fsync(fd)
                write('restored_changed_content')
                raise ValueError('Detached content changed; not deleted.')
        os.unlink(detached, dir_fd=fd)
        os.fsync(fd)
        write('deleted')
    finally:
        os.close(fd)
