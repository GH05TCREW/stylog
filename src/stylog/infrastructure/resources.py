"""Package/local resource resolution (spec 4.6, 15.5).

Package resources (function words, tree-sitter mappings, grammar manifest)
are checked-in and hash-verified. Local resources (e.g. spaCy model names or
explicit local paths) resolve without any network access.
"""

from __future__ import annotations

import hashlib
from importlib import resources as importlib_resources
from pathlib import Path

from stylog.domain.provenance import ResourceSignature
from stylog.exceptions import ResourceError
from stylog.ports import ResolvedResource, ResourceRequest
from stylog.serialization.canonical import sha256_hex

PACKAGE_RESOURCE_IDS = {
    "stylog.function_words.en": "resources/function_words_en_v1.txt",
    "stylog.tree_sitter.grammar_manifest": "resources/grammar_manifest.json",
    "stylog.tree_sitter.mapping.javascript": "resources/tree_sitter_mappings/javascript.json",
    "stylog.tree_sitter.mapping.typescript": "resources/tree_sitter_mappings/typescript.json",
    "stylog.tree_sitter.mapping.c": "resources/tree_sitter_mappings/c.json",
    "stylog.tree_sitter.mapping.rust": "resources/tree_sitter_mappings/rust.json",
}

_PACKAGE_RESOURCE_VERSIONS = {
    "stylog.function_words.en": "1.0.0",
    "stylog.tree_sitter.grammar_manifest": "1.0.0",
    "stylog.tree_sitter.mapping.javascript": "1.0.0",
    "stylog.tree_sitter.mapping.typescript": "1.0.0",
    "stylog.tree_sitter.mapping.c": "1.0.0",
    "stylog.tree_sitter.mapping.rust": "1.0.0",
}


class PackageResourceResolver:
    """Resolves checked-in package resources and explicit local files."""

    def resolve(self, request: ResourceRequest) -> ResolvedResource:
        if request.resource_id in PACKAGE_RESOURCE_IDS:
            relative = PACKAGE_RESOURCE_IDS[request.resource_id]
            try:
                data = (
                    importlib_resources.files("stylog").joinpath(relative).read_bytes()
                )
            except FileNotFoundError as exc:
                raise ResourceError(
                    f"RESOURCE_MISMATCH: package resource missing: {request.resource_id}"
                ) from exc
            return ResolvedResource(
                signature=ResourceSignature(
                    id=request.resource_id,
                    version=_PACKAGE_RESOURCE_VERSIONS[request.resource_id],
                    sha256=sha256_hex(data),
                ),
                data=data,
            )
        # Explicit local resource path (e.g. a provisioned spaCy model directory).
        path = Path(request.resource_id)
        if path.is_dir() or path.is_file():
            return ResolvedResource(
                signature=ResourceSignature(
                    id=request.resource_id,
                    version=request.version or "0",
                    sha256=resource_tree_sha256(path),
                ),
                local_path=str(path),
            )
        raise ResourceError(f"RESOURCE_MISMATCH: cannot resolve resource {request.resource_id!r}")


def resource_tree_sha256(path: Path) -> str:
    """Sorted relative paths + per-file SHA-256; no absolute paths (spec 15.5)."""
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode("utf-8") + b"\x00")
        digest.update(bytes.fromhex(sha256_hex(path.read_bytes())))
        return digest.hexdigest()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        relative = file.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8") + b"\x00")
        digest.update(bytes.fromhex(sha256_hex(file.read_bytes())))
    return digest.hexdigest()
