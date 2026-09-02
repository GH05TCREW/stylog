"""Parser adapters (Python-native and tree-sitter).

Checked per-language tree-sitter grammar knowledge (spec 4.6). ``factory`` is
the module callable returning the language capsule consumed by
``tree_sitter.Language``; ``package``/``upstream_repo`` feed grammar-manifest
regeneration (tools/generate_grammar_manifest.py).
"""

from __future__ import annotations

TREE_SITTER_GRAMMARS: dict[str, dict[str, str]] = {
    "javascript": {
        "grammar_id": "tree-sitter-javascript",
        "package": "tree-sitter-javascript",
        "module": "tree_sitter_javascript",
        "factory": "language",
        "upstream_repo": "https://github.com/tree-sitter/tree-sitter-javascript",
    },
    "typescript": {
        "grammar_id": "tree-sitter-typescript",
        "package": "tree-sitter-typescript",
        "module": "tree_sitter_typescript",
        "factory": "language_typescript",
        "upstream_repo": "https://github.com/tree-sitter/tree-sitter-typescript",
    },
    "c": {
        "grammar_id": "tree-sitter-c",
        "package": "tree-sitter-c",
        "module": "tree_sitter_c",
        "factory": "language",
        "upstream_repo": "https://github.com/tree-sitter/tree-sitter-c",
    },
    "rust": {
        "grammar_id": "tree-sitter-rust",
        "package": "tree-sitter-rust",
        "module": "tree_sitter_rust",
        "factory": "language",
        "upstream_repo": "https://github.com/tree-sitter/tree-sitter-rust",
    },
}
