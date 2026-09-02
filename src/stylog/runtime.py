"""Runtime-only (non-portable) types shared across the scientific core.

These are frozen dataclasses, not Pydantic models. Raw content lives here and
must never leak into portable serialization trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from stylog.domain.artifact import ArtifactKind
from stylog.domain.provenance import ResourceSignature, RuntimeSignature

if TYPE_CHECKING:
    from stylog.config import StylogConfig


@dataclass(frozen=True)
class RuntimeArtifact:
    """Decoded input plus exact raw identity information (runtime only)."""

    artifact_id: str
    kind: ArtifactKind
    language: str
    encoding: str
    raw_bytes: bytes
    text: str
    content_sha256: str

    @property
    def byte_count(self) -> int:
        return len(self.raw_bytes)

    @property
    def character_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class TreeSitterLanguageMapping:
    """Checked per-language mapping resource (runtime view)."""

    language: str
    version: str
    sha256: str
    identifier_node_types: frozenset[str]
    comment_node_types: frozenset[str]
    line_comment_delimiters: tuple[str, ...]
    block_comment_delimiters: tuple[str, ...]


@dataclass(frozen=True)
class GrammarManifestEntry:
    """Checked grammar identity for one supported tree-sitter language."""

    language: str
    grammar_id: str
    package: str
    module: str
    supported_versions: str
    installed_version: str
    upstream_repo: str
    upstream_revision: str
    node_types_sha256: str
    abi_version: int
    parser_compat_id: str


@dataclass(frozen=True)
class ResourceHandles:
    """Resolved resource handles handed to analyzers by the application layer."""

    function_words_en: frozenset[str] | None = None
    function_words_en_signature: ResourceSignature | None = None
    tree_sitter_mappings: dict[str, TreeSitterLanguageMapping] = field(default_factory=dict)
    grammar_manifest: dict[str, GrammarManifestEntry] = field(default_factory=dict)
    grammar_manifest_sha256: str | None = None
    nlp_model: Any | None = None  # e.g. a loaded spaCy Language (runtime only)
    nlp_model_backend: Any | None = None  # BackendSignature for the loaded model


@dataclass(frozen=True)
class AnalysisContext:
    """Everything a deterministic analyzer may see besides the artifact."""

    config: StylogConfig
    runtime: RuntimeSignature
    resources: ResourceHandles
