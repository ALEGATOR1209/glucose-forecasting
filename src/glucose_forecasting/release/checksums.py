"""SHA256 checksum helpers for release artifacts."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA256 digest for in-memory content."""
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA256 digest for a file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        while chunk := artifact.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected_checksum: str) -> bool:
    """Return whether a file's SHA256 matches the expected hexadecimal digest."""
    return hmac.compare_digest(sha256_file(path), expected_checksum.lower())
