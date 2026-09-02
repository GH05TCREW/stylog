"""Checked grammar manifest and mapping resource tests (spec 15, 6.13).

The checked-in grammar manifest must match the installed grammar packages
exactly; mapping resources must parse, hash consistently, and stay canonical.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json

from stylog.parsers import TREE_SITTER_GRAMMARS as LANGUAGE_SPECS
from stylog.parsers.tree_sitter import (
    SUPPORTED_LANGUAGES,
    TREE_SITTER_RUNTIME_VERSION,
    load_manifest,
    load_manifest_sha256,
    load_mappings,
)
from stylog.serialization.canonical import canonical_bytes_of_tree
from tools.generate_grammar_manifest import (
    MANIFEST_VERSION,
    NODE_TYPES_SOURCE,
    build_manifest,
)

EXPECTED_LANGUAGES = {"javascript", "typescript", "c", "rust"}

MAPPING_SCHEMA_KEYS = {
    "language",
    "mapping_version",
    "identifier_node_types",
    "comment_node_types",
    "line_comment_delimiters",
    "block_comment_delimiters",
}

EXPECTED_COMMENT_NODE_TYPES = {
    "javascript": {"comment"},
    "typescript": {"comment"},
    "c": {"comment"},
    "rust": {"line_comment", "block_comment"},
}


def _mapping_bytes(language: str) -> bytes:
    return (
        importlib.resources.files("stylog")
        .joinpath("resources")
        .joinpath("tree_sitter_mappings")
        .joinpath(f"{language}.json")
        .read_bytes()
    )


def _manifest_bytes() -> bytes:
    return (
        importlib.resources.files("stylog")
        .joinpath("resources")
        .joinpath("grammar_manifest.json")
        .read_bytes()
    )


def test_manifest_covers_exactly_the_supported_languages() -> None:
    data = json.loads(_manifest_bytes().decode("utf-8"))
    assert set(data["languages"]) == EXPECTED_LANGUAGES
    assert set(load_manifest()) == EXPECTED_LANGUAGES
    assert set(LANGUAGE_SPECS) == EXPECTED_LANGUAGES
    assert set(SUPPORTED_LANGUAGES) == EXPECTED_LANGUAGES


def test_manifest_file_is_canonical_json_plus_lf() -> None:
    raw = _manifest_bytes()
    tree = json.loads(raw.decode("utf-8"))
    assert raw == canonical_bytes_of_tree(tree) + b"\n"


def test_manifest_matches_installed_packages() -> None:
    """Recompute grammar identity like the dev script and compare exactly."""
    checked = json.loads(_manifest_bytes().decode("utf-8"))
    assert checked == build_manifest()
    assert checked["manifest_version"] == MANIFEST_VERSION
    assert checked["node_types_source"] == NODE_TYPES_SOURCE
    for language, entry in checked["languages"].items():
        spec = LANGUAGE_SPECS[language]
        assert entry["language"] == language
        assert entry["grammar_id"] == spec["grammar_id"]
        assert entry["package"] == spec["package"]
        assert entry["module"] == spec["module"]
        assert entry["installed_version"] == importlib.metadata.version(spec["package"])
        assert entry["upstream_revision"] == entry["installed_version"]
        assert entry["parser_compat_id"] == f"stylog.tree-sitter.{language}/1"
        assert len(entry["node_types_sha256"]) == 64
        assert entry["abi_version"] >= 13  # tree-sitter runtime ABI floor


def test_manifest_hash_and_load_round_trip() -> None:
    raw = _manifest_bytes()
    assert load_manifest_sha256() == hashlib.sha256(raw).hexdigest()
    data = json.loads(raw.decode("utf-8"))
    entries = load_manifest()
    for language, entry in entries.items():
        raw_entry = data["languages"][language]
        assert entry.grammar_id == raw_entry["grammar_id"]
        assert entry.supported_versions == raw_entry["supported_versions"]
        assert entry.installed_version == raw_entry["installed_version"]
        assert entry.upstream_repo == raw_entry["upstream_repo"]
        assert entry.upstream_revision == raw_entry["upstream_revision"]
        assert entry.node_types_sha256 == raw_entry["node_types_sha256"]
        assert entry.abi_version == raw_entry["abi_version"]
        assert entry.parser_compat_id == raw_entry["parser_compat_id"]
    # Runtime version is a base dependency pinned by pyproject.
    assert TREE_SITTER_RUNTIME_VERSION == importlib.metadata.version("tree-sitter")


def test_mappings_parse_canonical_and_hash_matches_load() -> None:
    loaded = load_mappings()
    assert set(loaded) == EXPECTED_LANGUAGES
    for language in EXPECTED_LANGUAGES:
        raw = _mapping_bytes(language)
        tree = json.loads(raw.decode("utf-8"))
        # Schema keys are exact; file stays canonical (RFC 8785 + one LF).
        assert set(tree) == MAPPING_SCHEMA_KEYS
        assert raw == canonical_bytes_of_tree(tree) + b"\n"
        mapping = loaded[language]
        assert mapping.sha256 == hashlib.sha256(raw).hexdigest()
        assert mapping.language == language
        assert mapping.version == tree["mapping_version"] == "1.0.0"


def test_mapping_content_matches_grammar_node_types() -> None:
    """Empirically verified node types; runtime must never guess unmapped types."""
    loaded = load_mappings()
    for language, mapping in loaded.items():
        assert mapping.identifier_node_types == {"identifier"}
        assert mapping.comment_node_types == EXPECTED_COMMENT_NODE_TYPES[language]
        assert mapping.line_comment_delimiters == ("//",)
        assert mapping.block_comment_delimiters == ("/*",)
