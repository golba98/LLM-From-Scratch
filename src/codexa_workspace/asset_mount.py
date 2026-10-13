"""Require the configured asset filesystem before resolving inputs or outputs."""
import json
import os
from pathlib import Path
import subprocess


def require_asset_mount(settings):
    expected = settings.get('asset_mount')
    if not expected:
        return
    root = Path(expected['path'])
    if root.is_symlink() or not os.path.ismount(root):
        raise FileNotFoundError(f'Codexa asset SSD is not mounted at {root}; unlock and mount it first.')
    actual = subprocess.run(
        ['findmnt', '-rn', '-M', str(root), '-o', 'UUID,FSTYPE'],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    if actual != [expected['filesystem_uuid'], 'ext4']:
        raise RuntimeError(f'Wrong asset filesystem mounted at {root}; refusing asset access.')
    marker = root / '.codexa-volume.json'
    if marker.is_symlink() or marker.stat().st_uid != 0:
        raise RuntimeError('Codexa volume identity marker must be a root-owned regular file.')
    identity = json.loads(marker.read_text())
    if any(identity.get(key) != expected[key] for key in ('filesystem_uuid', 'luks_uuid')):
        raise RuntimeError('Codexa volume identity does not match local asset configuration.')
