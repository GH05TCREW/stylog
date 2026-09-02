"""Embedded comment/docstring extraction tests (spec 9, fixtures 25.11-25.12)."""

from __future__ import annotations

import hashlib

from stylog.analysis.python import extract_embedded
from stylog.config import (
    AnalysisConfig,
    CodeAnalysisConfig,
    PythonAnalysisConfig,
    StylogConfig,
)
from stylog.domain.artifact import ArtifactKind
from stylog.domain.provenance import current_runtime_signature
from stylog.parsers.python_native import parse_python
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact

# Spec 25.11 fixture.
EMBEDDED_SRC = (
    "#!/usr/bin/env python\n"
    "# coding: utf-8\n"
    "# First\n"
    "# second\n"
    "x = 1  # inline\n"
    "# fmt: off\n"
    "# Third\n"
)

# Spec 25.12 fixture.
DOCSTRING_SRC = (
    '"""Module doc.\n'
    "\n"
    "More.\n"
    '"""\n'
    "\n"
    "\n"
    "class K:\n"
    '    """Class doc."""\n'
    "    def sync(self):\n"
    '        """Sync doc."""\n'
    "        async def inner():\n"
    '            """Async doc."""\n'
    "            return 1\n"
    "        return inner\n"
    "\n"
    "\n"
    '"standalone"\n'
    'x = "not a docstring"\n'
)


def make_artifact(src: str) -> RuntimeArtifact:
    raw = src.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="t",
        kind=ArtifactKind.CODE,
        language="python",
        encoding="utf-8",
        raw_bytes=raw,
        text=src,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def make_ctx(config: StylogConfig | None = None) -> AnalysisContext:
    return AnalysisContext(
        config=config or StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(),
    )


def run_extract(src: str, config: StylogConfig | None = None):
    ctx = make_ctx(config)
    artifact = make_artifact(src)
    facts = parse_python(artifact, ctx.config)
    return extract_embedded(artifact, facts, ctx.config)


def span_tuple(candidate):
    span = candidate.span
    return (
        span.start.line,
        span.start.column,
        span.end.line,
        span.end.column,
    )


# --- spec 25.11 ---


def test_25_11_comment_candidates():
    candidates = run_extract(EMBEDDED_SRC)
    assert [(c.kind, c.text) for c in candidates] == [
        ("comment_block", "First\nsecond"),
        ("inline_comment", "inline"),
        ("comment_block", "Third"),
    ]


def test_25_11_excludes_shebang_cookie_and_directives():
    candidates = run_extract(EMBEDDED_SRC)
    texts = [c.text for c in candidates]
    assert not any("usr/bin/env" in text for text in texts)
    assert not any("coding" in text for text in texts)
    assert not any("fmt" in text for text in texts)


def test_25_11_spans_and_ordinals():
    candidates = run_extract(EMBEDDED_SRC)
    first, inline, third = candidates
    assert span_tuple(first) == (3, 0, 4, 8)  # "# First" .. "# second"
    assert first.ordinal == 1
    assert span_tuple(inline) == (5, 7, 5, 15)
    assert inline.ordinal == 1  # ordinals are per kind
    assert span_tuple(third) == (7, 0, 7, 7)
    assert third.ordinal == 2
    assert all(c.docstring_owner is None for c in candidates)


def test_block_grouping_requires_same_column():
    candidates = run_extract("# a\n  # b\n")
    assert [(c.kind, c.text) for c in candidates] == [
        ("comment_block", "a"),
        ("comment_block", "b"),
    ]


def test_inline_comment_breaks_block_run():
    candidates = run_extract("# a\nx = 1  # i\n# b\n")
    assert [(c.kind, c.text) for c in candidates] == [
        ("comment_block", "a"),
        ("inline_comment", "i"),
        ("comment_block", "b"),
    ]


def test_whitespace_only_comment_excluded():
    assert run_extract("#   \n#\nx = 1\n") == []


def test_tab_indented_full_line_comment():
    candidates = run_extract("\t# note\n")
    assert [(c.kind, c.text) for c in candidates] == [("comment_block", "note")]
    assert span_tuple(candidates[0]) == (1, 1, 1, 7)


def test_inline_comment_after_non_ascii_source():
    candidates = run_extract("α = 1  # note\n")
    assert [(c.kind, c.text) for c in candidates] == [("inline_comment", "note")]
    # tokenize columns are code-point based: "#" is at code-point column 7
    assert span_tuple(candidates[0]) == (1, 7, 1, 13)


# --- spec 25.12 ---


def test_25_12_docstrings():
    candidates = run_extract(DOCSTRING_SRC)
    assert [(c.kind, c.docstring_owner) for c in candidates] == [
        ("docstring", "module"),
        ("docstring", "class:K"),
        ("docstring", "function:sync"),
        ("docstring", "async_function:inner"),
    ]
    assert [c.ordinal for c in candidates] == [1, 2, 3, 4]


def test_25_12_docstring_texts_cleaned():
    candidates = run_extract(DOCSTRING_SRC)
    texts = {c.docstring_owner: c.text for c in candidates}
    assert texts["module"] == "Module doc.\n\nMore."
    assert texts["class:K"] == "Class doc."
    assert texts["function:sync"] == "Sync doc."
    assert texts["async_function:inner"] == "Async doc."


def test_25_12_docstring_spans():
    candidates = run_extract(DOCSTRING_SRC)
    spans = {c.docstring_owner: span_tuple(c) for c in candidates}
    assert spans["module"] == (1, 0, 4, 3)
    assert spans["class:K"] == (8, 4, 8, 20)
    assert spans["function:sync"] == (10, 8, 10, 23)
    assert spans["async_function:inner"] == (12, 12, 12, 28)


def test_25_12_standalone_strings_are_not_docstrings():
    candidates = run_extract(DOCSTRING_SRC)
    assert len(candidates) == 4
    assert not any("standalone" in c.text for c in candidates)


def test_docstring_span_columns_are_codepoint_based():
    candidates = run_extract('"""café"""\n')
    assert len(candidates) == 1
    # AST col_offset is UTF-8 bytes (é = 2 bytes); the span must use code points
    assert span_tuple(candidates[0]) == (1, 0, 1, 10)
    assert candidates[0].text == "café"


def test_empty_docstring_counts():
    candidates = run_extract('def f():\n    ""\n    pass\n')
    assert [(c.kind, c.text, c.docstring_owner) for c in candidates] == [
        ("docstring", "", "function:f")
    ]


# --- config and failure interplay ---


def test_embedded_cap_in_source_order():
    config = StylogConfig(
        analysis=AnalysisConfig(
            code=CodeAnalysisConfig(python=PythonAnalysisConfig(max_embedded_artifacts=2))
        )
    )
    candidates = run_extract(EMBEDDED_SRC, config)
    assert [(c.kind, c.text) for c in candidates] == [
        ("comment_block", "First\nsecond"),
        ("inline_comment", "inline"),
    ]


def test_embedded_text_disabled():
    config = StylogConfig(
        analysis=AnalysisConfig(
            code=CodeAnalysisConfig(python=PythonAnalysisConfig(embedded_text=False))
        )
    )
    assert run_extract(EMBEDDED_SRC, config) == []


def test_tokenize_failure_no_comment_candidates():
    assert run_extract("def f(  # dangling\n") == []


def test_ast_failure_keeps_comments_but_drops_docstrings():
    src = "x = 1  # note\nif:\n    pass\n"
    candidates = run_extract(src)
    assert [(c.kind, c.text) for c in candidates] == [("inline_comment", "note")]


def test_ast_failure_would_have_docstring():
    # tokenizes fine, ast.parse fails -> no docstring candidate
    src = '"""doc"""\nif:\n    pass\n'
    assert run_extract(src) == []
