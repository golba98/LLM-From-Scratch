"""Content identities: canonical JSON fingerprints and streaming file digests (no torch import)."""
import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    """Canonical JSON identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path) -> str:
    """Streaming digest without importing torch."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
