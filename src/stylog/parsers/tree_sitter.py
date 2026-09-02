"""Tree-sitter parser adapter (spec sections 4 and 6.13).

Owns the mechanics of the tree-sitter runtime: lazy grammar loading, checked
mapping/manifest resource loading, and deterministic parse facts. Reads only
checked package resources via ``importlib.resources``; no other IO, no network,
no canonical serialization.

Error contract of :func:`parse_tree_sitter` (explicit, deterministic):

- grammar package missing/incompatible, or language unsupported -> ``"UNAVAILABLE"``
- parsed root node ``has_error`` or ``is_error`` -> ``"PARSER_ERROR"``
- ``root`` is ``None`` exactly when ``error_code`` is not ``None``
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Any

from tree_sitter import Language, Parser

from stylog.domain.provenance import (
    BackendSignature,
    PackageProvenance,
    ParserGrammarSignature,
)
from stylog.infrastructure.resources import PackageResourceResolver
from stylog.parsers import TREE_SITTER_GRAMMARS
from stylog.ports import ResourceRequest, ResourceResolver
from stylog.runtime import (
    AnalysisContext,
    GrammarManifestEntry,
    RuntimeArtifact,
    TreeSitterLanguageMapping,
)

TREE_SITTER_RUNTIME_VERSION = importlib.metadata.version("tree-sitter")

SUPPORTED_LANGUAGES = tuple(TREE_SITTER_GRAMMARS)

# language -> (grammar module, factory callable returning the language capsule)
_LANGUAGE_FACTORIES = {
    language: (spec["module"], spec["factory"])
    for language, spec in TREE_SITTER_GRAMMARS.items()
}

_language_cache: dict[str, Language] = {}


def _get_language(language: str) -> Language:
    """Load and cache a grammar Language; imports the grammar module lazily."""
    cached = _language_cache.get(language)
    if cached is not None:
        return cached
    module_name, factory_name = _LANGUAGE_FACTORIES[language]
    module = importlib.import_module(module_name)
    lang = Language(getattr(module, factory_name)())
    _language_cache[language] = lang
    return lang


def _resolve(resources: ResourceResolver | None, resource_id: str) -> bytes:
    """Read one checked package resource through the resource port."""
    resolver = resources if resources is not None else PackageResourceResolver()
    resolved = resolver.resolve(ResourceRequest(resource_id))
    assert resolved.data is not None
    return resolved.data


def load_mappings(
    resources: ResourceResolver | None = None,
) -> dict[str, TreeSitterLanguageMapping]:
    """Load every checked per-language mapping resource.

    Each mapping's sha256 is the SHA-256 of its exact file bytes.
    """
    mappings: dict[str, TreeSitterLanguageMapping] = {}
    for language in SUPPORTED_LANGUAGES:
        resource_id = f"stylog.tree_sitter.mapping.{language}"
        resolver = resources if resources is not None else PackageResourceResolver()
        resolved = resolver.resolve(ResourceRequest(resource_id))
        assert resolved.data is not None
        data = json.loads(resolved.data.decode("utf-8"))
        mappings[language] = TreeSitterLanguageMapping(
            language=data["language"],
            version=data["mapping_version"],
            sha256=resolved.signature.sha256,
            identifier_node_types=frozenset(data["identifier_node_types"]),
            comment_node_types=frozenset(data["comment_node_types"]),
            line_comment_delimiters=tuple(data["line_comment_delimiters"]),
            block_comment_delimiters=tuple(data["block_comment_delimiters"]),
        )
    return mappings


def load_manifest(
    resources: ResourceResolver | None = None,
) -> dict[str, GrammarManifestEntry]:
    """Load the checked grammar manifest entries."""
    data = json.loads(
        _resolve(resources, "stylog.tree_sitter.grammar_manifest").decode("utf-8")
    )
    return {
        language: GrammarManifestEntry(**entry)
        for language, entry in data["languages"].items()
    }


def load_manifest_sha256(resources: ResourceResolver | None = None) -> str:
    """SHA-256 of the exact grammar manifest file bytes."""
    resolver = resources if resources is not None else PackageResourceResolver()
    resolved = resolver.resolve(ResourceRequest("stylog.tree_sitter.grammar_manifest"))
    return resolved.signature.sha256


@dataclass(frozen=True)
class TreeSitterParseFacts:
    """Runtime parse facts for one artifact (never serialized)."""

    language: str
    root: Any | None  # tree_sitter.Node; None exactly when error_code is set
    error_code: str | None  # None | "UNAVAILABLE" | "PARSER_ERROR"


def parse_tree_sitter(artifact: RuntimeArtifact, ctx: AnalysisContext) -> TreeSitterParseFacts:
    """Parse ``artifact.raw_bytes`` with the checked grammar for its language.

    Decoding is already done (``artifact.text`` exists); tree-sitter consumes
    the exact raw bytes.
    """
    del ctx  # parsing is context-independent; resources are checked package data
    language = artifact.language
    if language not in _LANGUAGE_FACTORIES:
        return TreeSitterParseFacts(language=language, root=None, error_code="UNAVAILABLE")
    try:
        lang = _get_language(language)
    except Exception:
        # Grammar package missing or ABI-incompatible with the runtime.
        return TreeSitterParseFacts(language=language, root=None, error_code="UNAVAILABLE")
    parser = Parser(lang)
    root = parser.parse(artifact.raw_bytes).root_node
    if root.has_error or root.is_error:
        return TreeSitterParseFacts(language=language, root=None, error_code="PARSER_ERROR")
    return TreeSitterParseFacts(language=language, root=root, error_code=None)


def tree_sitter_backend_signature(
    language: str, entry: GrammarManifestEntry
) -> BackendSignature:
    """Dependency-aware backend signature for one grammar (spec 5.17)."""
    return BackendSignature(
        backend_id="tree-sitter",
        implementation_version=TREE_SITTER_RUNTIME_VERSION,
        scientific_compatibility_id=entry.parser_compat_id,
        packages=(
            PackageProvenance(package="tree-sitter", version=TREE_SITTER_RUNTIME_VERSION),
            PackageProvenance(package=entry.package, version=entry.installed_version),
        ),
        parser_grammar=ParserGrammarSignature(
            language=language,
            grammar_id=entry.grammar_id,
            grammar_version=entry.installed_version,
            grammar_revision=entry.upstream_revision,
            node_types_sha256=entry.node_types_sha256,
            grammar_manifest_sha256=load_manifest_sha256(),
            language_abi_version=entry.abi_version,
        ),
    )
