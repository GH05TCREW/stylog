"""Content-addressed scientific cache (spec 17).

The cache key encodes Stylog scientific compatibility — content identity,
scientific config, schema version, analyzer identities, resource signatures,
and runtime facts. Files are canonical
portable objects written atomically.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from stylog.serialization.jsonio import atomic_temp_path


class NullCacheStore:
    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, canonical_bytes: bytes) -> None:
        return None


class MemoryCacheStore:
    """In-memory fake used by tests."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self.objects.get(key)

    def put(self, key: str, canonical_bytes: bytes) -> None:
        self.objects[key] = canonical_bytes


def fingerprint_cache_key(
    *,
    content_sha256: str,
    kind: str,
    language: str,
    analysis_config_sha256: str,
    schema_version: str,
    analyzer_ids_versions: tuple[tuple[str, str], ...],
    resource_signatures: tuple[tuple[str, str, str], ...],
    runtime_fields: tuple[tuple[str, str], ...],
) -> str:
    """Cache key per spec 17.2.

    Kind and language are part of the scientific identity: both change the
    measurement (language-gated analyzers such as English function words
    produce ok / not_applicable / unavailable observations by language), so
    identical content analyzed under different kinds or languages MUST NOT
    share a cache entry (v2 format).
    """
    material = bytearray(b"stylog-cache-v2\x00")
    material += bytes.fromhex(content_sha256)
    material += kind.encode() + b"\x00" + language.encode() + b"\x00"
    material += bytes.fromhex(analysis_config_sha256)
    material += schema_version.encode("utf-8") + b"\x00"
    for analyzer_id, version in sorted(analyzer_ids_versions):
        material += analyzer_id.encode() + b"\x00" + version.encode() + b"\x00"
    for resource_id, version, sha in sorted(resource_signatures):
        material += resource_id.encode() + b"\x00" + version.encode() + b"\x00"
        material += bytes.fromhex(sha)
    for field_name, field_value in sorted(runtime_fields):
        material += field_name.encode() + b"\x00" + field_value.encode() + b"\x00"
    return hashlib.sha256(bytes(material)).hexdigest()


class FilesystemCacheStore:
    """<root>/objects/ab/cdef....json with atomic writes (spec 17.3-17.7)."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _path_for(self, key: str) -> Path:
        return self.root / "objects" / key[:2] / (key[2:] + ".json")

    def get(self, key: str) -> bytes | None:
        path = self._path_for(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None

    def remove(self, key: str) -> None:
        try:
            self._path_for(key).unlink()
        except OSError:
            pass

    def put(self, key: str, canonical_bytes: bytes) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "posix":
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass
        with atomic_temp_path(path) as (fd, _), os.fdopen(fd, "wb") as handle:
            if os.name == "posix":
                try:
                    os.fchmod(handle.fileno(), 0o600)
                except OSError:
                    pass
            handle.write(canonical_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            try:
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
