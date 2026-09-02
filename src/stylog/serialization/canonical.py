"""RFC 8785 (JCS) canonical serialization and scientific hashing (spec section 14).

Canonical bytes are produced from the validated portable tree only. Standalone
``.json`` files append exactly one LF, which is never part of a scientific hash.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import rfc8785
from pydantic import BaseModel

from stylog.domain._base import MAX_SAFE_INT
from stylog.exceptions import PortableArtifactError


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _walk_portable_tree(value: Any, path: str = "$") -> Any:
    """Validate and normalize a dumped portable tree.

    Enforces the Stylog restrictions beyond JCS: no null, no lone surrogates,
    no NaN/infinity, no negative zero, no unsafe integers.
    """
    if value is None:
        raise PortableArtifactError(f"portable JSON contains null at {path}")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INT:
            raise PortableArtifactError(f"unsafe integer at {path}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PortableArtifactError(f"non-finite float at {path}")
        if value == 0.0:
            return 0.0  # normalize -0.0
        return value
    if isinstance(value, str):
        if any(0xD800 <= ord(ch) <= 0xDFFF for ch in value):
            raise PortableArtifactError(f"lone surrogate at {path}")
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if any(0xD800 <= ord(ch) <= 0xDFFF for ch in key):
                raise PortableArtifactError(f"lone surrogate in object key at {path}")
            out[key] = _walk_portable_tree(item, f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_walk_portable_tree(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise PortableArtifactError(f"unsupported portable value of type {type(value)!r} at {path}")


def portable_tree(model: BaseModel) -> dict[str, Any]:
    """Dump a portable model to a JSON-compatible tree with Stylog restrictions."""
    tree = model.model_dump(mode="json", exclude_none=True)
    walked = _walk_portable_tree(tree)
    assert isinstance(walked, dict)
    return walked


def canonical_bytes(model: BaseModel) -> bytes:
    """RFC 8785 canonical bytes of a portable model (no trailing LF)."""
    return rfc8785.dumps(portable_tree(model))


def file_bytes(model: BaseModel) -> bytes:
    """Canonical bytes plus exactly one trailing LF for standalone .json files."""
    return canonical_bytes(model) + b"\n"


def scientific_sha256(model: BaseModel) -> str:
    """SHA-256 over the canonical JCS bytes (trailing file LF excluded)."""
    return sha256_hex(canonical_bytes(model))


def canonical_bytes_of_tree(tree: Any) -> bytes:
    """Canonicalize an already-dumped tree (used for config/compatibility hashes)."""
    return rfc8785.dumps(_walk_portable_tree(tree))


def sha256_of_tree(tree: Any) -> str:
    return sha256_hex(canonical_bytes_of_tree(tree))
