"""Regenerate the checked grammar manifest (dev-only tooling; performs IO).

Computes tree-sitter grammar identity from the *installed* grammar packages
and writes ``src/stylog/resources/grammar_manifest.json`` as canonical JSON
(RFC 8785) plus exactly one trailing LF. The manifest intentionally does NOT
store its own hash; ``grammar_manifest_sha256`` is recomputed from file bytes
by the runtime and printed here for reference.

Run from the repository root:

    python tools/generate_grammar_manifest.py
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tree_sitter import Language

from stylog.parsers import TREE_SITTER_GRAMMARS
from stylog.runtime import GrammarManifestEntry
from stylog.serialization.canonical import canonical_bytes_of_tree, sha256_of_tree

MANIFEST_VERSION = "1.0.0"

# How node_types_sha256 is derived (kept in the manifest for auditability).
NODE_TYPES_SOURCE = (
    "tree_sitter.Language.node_kind_count/node_kind_for_id; "
    "sha256 of JCS of the sorted unique node kind names"
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
MANIFEST_PATH = _REPO_ROOT / "src" / "stylog" / "resources" / "grammar_manifest.json"


def _load_language(language: str) -> Language:
    spec = TREE_SITTER_GRAMMARS[language]
    module = importlib.import_module(spec["module"])
    return Language(getattr(module, spec["factory"])())


def _node_types_sha256(lang: Language) -> str:
    kind_names = {
        lang.node_kind_for_id(kind_id) for kind_id in range(lang.node_kind_count)
    }
    kind_names.discard(None)
    return sha256_of_tree(sorted(kind_names))


def _supported_versions(package: str, pyproject_path: Path = PYPROJECT_PATH) -> str:
    """Extract the reviewed version range for ``package`` from pyproject."""
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    for requirement in data["project"]["dependencies"]:
        if requirement == package or (
            requirement.startswith(package)
            and requirement[len(package)] in "<>=!~("
        ):
            return requirement[len(package):]
    raise RuntimeError(f"no dependency pin found for {package} in {pyproject_path}")


def build_entry(language: str) -> GrammarManifestEntry:
    """Compute the checked grammar identity for one supported language."""
    spec = TREE_SITTER_GRAMMARS[language]
    lang = _load_language(language)
    installed_version = importlib.metadata.version(spec["package"])
    return GrammarManifestEntry(
        language=language,
        grammar_id=spec["grammar_id"],
        package=spec["package"],
        module=spec["module"],
        supported_versions=_supported_versions(spec["package"]),
        installed_version=installed_version,
        upstream_repo=spec["upstream_repo"],
        # No upstream commit is knowable from the installed wheel; the
        # package version is the reviewed revision proxy.
        upstream_revision=installed_version,
        node_types_sha256=_node_types_sha256(lang),
        abi_version=lang.abi_version,
        parser_compat_id=f"stylog.tree-sitter.{language}/1",
    )


def build_manifest() -> dict[str, Any]:
    """Return the full manifest tree (canonicalizable, no self-hash field)."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "node_types_source": NODE_TYPES_SOURCE,
        "languages": {
            language: asdict(build_entry(language)) for language in TREE_SITTER_GRAMMARS
        },
    }


def main() -> None:
    tree = build_manifest()
    payload = canonical_bytes_of_tree(tree) + b"\n"
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_bytes(payload)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    print(f"wrote {MANIFEST_PATH}")
    print(f"grammar_manifest_sha256={file_sha256}")


if __name__ == "__main__":
    main()
