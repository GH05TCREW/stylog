"""Python AST analyzer tests (spec 6.8-6.11, 8.8-8.18, 25.7, 25.9, 25.10)."""

from __future__ import annotations

import dataclasses
import hashlib

from stylog.analysis.python import PythonAstAnalyzer
from stylog.analysis.registry import ANALYZER_CODE_PYTHON_AST, features_owned_by
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

FIXTURE_25_7 = (
    "import os as operating_system\n"
    "\n"
    "async def fetch(myURL2: str, *, retries=3):\n"
    "    if retries > 0:\n"
    '        return f"{myURL2=}"\n'
    "    return None  # noqa\n"
)

STRUCT_SRC = '''"""Mod."""

import sys
from os import path as p


@deco
def f(a, b=1, *c, d, **e):
    try:
        x = [i for i in range(3) if i]
        yield x
    except ValueError:
        raise
    finally:
        pass


async def g():
    async with lock:
        async for row in rows:
            if row:
                continue
    return {k: v for k, v in rows}


def h():
    while True:
        match (1, 2):
            case (1, y):
                break
            case _:
                pass
    return (z for z in []) if ok else {q for q in []}
'''


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


def by_id(output):
    return {obs.feature_id: obs for obs in output.observations}


def cat(obs) -> dict[str, int]:
    return {entry.key: entry.count for entry in obs.value.counts}


def hist(obs) -> dict[int, int]:
    return {point.point: point.count for point in obs.value.points}


def run_ast(src: str, config: StylogConfig | None = None):
    ctx = make_ctx(config)
    artifact = make_artifact(src)
    facts = parse_python(artifact, ctx.config)
    return PythonAstAnalyzer().analyze(artifact, ctx, facts)


# --- spec 25.7 token fixture, AST side ---


def test_25_7_ast_structure():
    out = by_id(run_ast(FIXTURE_25_7))
    assert out["code.python.structure.function_count"].value.value == 1
    assert out["code.python.structure.class_count"].value.value == 0
    assert cat(out["code.python.structure.function_kind"]) == {"async": 1}
    assert hist(out["code.python.structure.parameter_count"]) == {2: 1}
    assert hist(out["code.python.structure.return_count"]) == {2: 1}
    share = out["code.python.structure.nonterminal_return_function_share"].value
    assert (share.numerator, share.denominator) == (1, 1)
    assert cat(out["code.python.structure.control_construct"]) == {"if": 1}
    assert hist(out["code.python.structure.max_control_nesting"]) == {1: 1}
    assert hist(out["code.python.structure.branch_construct_count"]) == {1: 1}
    assert hist(out["code.python.structure.decorator_count"]) == {0: 1}
    assert cat(out["code.python.structure.import_kind"]) == {"import": 1}


def test_25_7_import_alias_share():
    out = by_id(run_ast(FIXTURE_25_7))
    value = out["code.python.structure.import_alias_share"].value
    # one ast.alias (os as operating_system), with asname -> 1/1
    assert (value.numerator, value.denominator) == (1, 1)
    assert value.value == 1.0


def test_25_7_bindings():
    out = by_id(run_ast(FIXTURE_25_7))
    assert cat(out["code.python.naming.binding_role"]) == {
        "async_function": 1,
        "import_binding": 1,
        "parameter_kwonly": 1,
        "parameter_positional": 1,
    }


def test_25_7_syntax_features():
    out = by_id(run_ast(FIXTURE_25_7))
    nodes = cat(out["code.python.syntax.node_distribution"])
    assert "Module" not in nodes
    assert "Load" not in nodes and "Store" not in nodes
    assert nodes["AsyncFunctionDef"] == 1
    assert nodes["If"] == 1
    assert nodes["Return"] == 2
    depths = hist(out["code.python.syntax.node_depth"])
    assert min(depths) == 1  # Module at depth 0 is excluded from observations
    edges = cat(out["code.python.syntax.parent_child_distribution"])
    assert edges["Import>alias"] == 1
    assert edges["AsyncFunctionDef>If"] == 1


def test_25_7_empty_populations_insufficient():
    out = by_id(run_ast(FIXTURE_25_7))
    assert out["code.python.structure.match_case_count"].status == "insufficient_support"
    assert out["code.python.comments.docstring_kind"].status == "insufficient_support"
    assert out["code.python.comments.docstring_length"].status == "insufficient_support"
    assert out["code.python.structure.assignment_kind"].status == "insufficient_support"


def test_registry_coverage_sorted():
    out = run_ast(FIXTURE_25_7)
    owned = [f.feature_id for f in features_owned_by(ANALYZER_CODE_PYTHON_AST)]
    assert [o.feature_id for o in out.observations] == sorted(owned)


# --- structural features over a richer module ---


def test_struct_counts_and_kinds():
    out = by_id(run_ast(STRUCT_SRC))
    assert out["code.python.structure.function_count"].value.value == 3
    assert out["code.python.structure.class_count"].value.value == 0
    assert cat(out["code.python.structure.function_kind"]) == {"async": 1, "sync": 2}
    assert hist(out["code.python.structure.function_length_lines"]) == {6: 1, 8: 2}
    assert hist(out["code.python.structure.parameter_count"]) == {0: 2, 5: 1}
    assert hist(out["code.python.structure.return_count"]) == {0: 1, 1: 2}
    assert hist(out["code.python.structure.decorator_count"]) == {0: 2, 1: 1}


def test_struct_control_and_branches():
    out = by_id(run_ast(STRUCT_SRC))
    assert cat(out["code.python.structure.control_construct"]) == {
        "async_for": 1,
        "async_with": 1,
        "dict_comprehension": 1,
        "generator_expression": 1,
        "if": 1,
        "if_expression": 1,
        "list_comprehension": 1,
        "match": 1,
        "set_comprehension": 1,
        "try": 1,
        "while": 1,
    }
    assert hist(out["code.python.structure.max_control_nesting"]) == {1: 1, 2: 1, 3: 1}
    assert hist(out["code.python.structure.branch_construct_count"]) == {3: 2, 6: 1}


def test_struct_exceptions_comprehensions_match():
    out = by_id(run_ast(STRUCT_SRC))
    assert cat(out["code.python.structure.exception_construct"]) == {
        "except_handler": 1,
        "finally_block": 1,
        "raise": 1,
        "try": 1,
    }
    assert cat(out["code.python.structure.comprehension_kind"]) == {
        "dict": 1,
        "generator": 1,
        "list": 1,
        "set": 1,
    }
    assert hist(out["code.python.structure.match_case_count"]) == {2: 1}
    assert cat(out["code.python.structure.assignment_kind"]) == {"assign": 1}
    value = out["code.python.structure.import_alias_share"].value
    assert (value.numerator, value.denominator) == (1, 2)


def test_struct_nonterminal_return_zero():
    out = by_id(run_ast(STRUCT_SRC))
    value = out["code.python.structure.nonterminal_return_function_share"].value
    assert (value.numerator, value.denominator) == (0, 3)
    assert value.value == 0.0


def test_struct_module_docstring():
    out = by_id(run_ast(STRUCT_SRC))
    assert cat(out["code.python.comments.docstring_kind"]) == {"module": 1}
    assert hist(out["code.python.comments.docstring_length"]) == {4: 1}  # "Mod."


def test_bare_except_and_try_star():
    src = (
        "try:\n"
        "    pass\n"
        "except:\n"
        "    pass\n"
    )
    out = by_id(run_ast(src))
    assert cat(out["code.python.structure.exception_construct"]) == {
        "bare_except": 1,
        "except_handler": 1,
        "try": 1,
    }


def test_try_star_when_runtime_exposes_it():
    import ast as ast_module

    if not hasattr(ast_module, "TryStar"):
        return  # spec 8.13: absent runtime class creates no category
    src = (
        "async def f():\n"
        "    try:\n"
        "        pass\n"
        "    except* ValueError:\n"
        "        pass\n"
    )
    out = by_id(run_ast(src))
    assert cat(out["code.python.structure.exception_construct"]) == {
        "except_handler": 1,
        "try_star": 1,
    }
    assert cat(out["code.python.structure.control_construct"]) == {"try_star": 1}


def test_no_match_insufficient():
    out = by_id(run_ast("x = 1\n"))
    assert out["code.python.structure.match_case_count"].status == "insufficient_support"


def test_empty_module():
    out = by_id(run_ast(""))
    assert out["code.python.structure.function_count"].value.value == 0
    assert out["code.python.structure.class_count"].value.value == 0
    assert out["code.python.structure.function_count"].status == "ok"
    assert out["code.python.syntax.node_distribution"].status == "insufficient_support"
    assert out["code.python.structure.parameter_count"].status == "insufficient_support"
    assert out["code.python.structure.import_alias_share"].status == "insufficient_support"


# --- failure semantics (spec 8.4, 8.8, 25.9, 25.10) ---


def test_tokenize_failure_ast_parser_error():
    ctx = make_ctx()
    artifact = make_artifact("def f(")
    facts = parse_python(artifact, ctx.config)
    assert facts.token_error_code == "PYTHON_TOKENIZE_ERROR"
    out = PythonAstAnalyzer().analyze(artifact, ctx, facts)
    assert all(o.status == "parser_error" for o in out.observations)
    assert len(out.observations) == len(features_owned_by(ANALYZER_CODE_PYTHON_AST))
    assert [d.code for d in out.diagnostics] == ["PYTHON_TOKENIZE_ERROR"]


def test_ast_failure_after_tokenize_success():
    ctx = make_ctx()
    artifact = make_artifact("x = 1\nif:\n    pass\n")
    facts = parse_python(artifact, ctx.config)
    assert facts.token_error_code is None
    assert facts.tokens is not None
    assert facts.ast_error_code == "PYTHON_AST_PARSE_ERROR"
    out = PythonAstAnalyzer().analyze(artifact, ctx, facts)
    assert all(o.status == "parser_error" for o in out.observations)
    assert [d.code for d in out.diagnostics] == ["PYTHON_AST_PARSE_ERROR"]


def test_max_ast_bytes_resource_limit():
    config = StylogConfig(
        analysis=AnalysisConfig(
            code=CodeAnalysisConfig(python=PythonAnalysisConfig(max_ast_bytes=5))
        )
    )
    ctx = make_ctx(config)
    artifact = make_artifact("x = 12345\n")
    facts = parse_python(artifact, ctx.config)
    assert facts.tokens is not None  # token features stay ok
    assert facts.ast_resource_limited is True
    assert facts.tree is None
    out = PythonAstAnalyzer().analyze(artifact, ctx, facts)
    assert all(o.status == "unavailable" for o in out.observations)
    assert [d.code for d in out.diagnostics] == ["PYTHON_AST_RESOURCE_LIMIT"]


def test_max_ast_nesting_resource_limit():
    config = StylogConfig(
        analysis=AnalysisConfig(
            code=CodeAnalysisConfig(python=PythonAnalysisConfig(max_ast_nesting=2))
        )
    )
    ctx = make_ctx(config)
    artifact = make_artifact("x = [1]\n")  # raw depth 3 (Module/Assign/List/Constant)
    facts = parse_python(artifact, ctx.config)
    assert facts.ast_resource_limited is True
    assert facts.max_depth == 3
    out = PythonAstAnalyzer().analyze(artifact, ctx, facts)
    assert all(o.status == "unavailable" for o in out.observations)
    assert [d.code for d in out.diagnostics] == ["PYTHON_AST_RESOURCE_LIMIT"]


def test_decode_error_ast_parser_error():
    raw = b"\xef\xbb\xbf# coding: latin-1\nx = 1\n"
    ctx = make_ctx()
    artifact = RuntimeArtifact(
        artifact_id="t",
        kind=ArtifactKind.CODE,
        language="python",
        encoding="utf-8",
        raw_bytes=raw,
        text="",
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )
    facts = parse_python(artifact, ctx.config)
    out = PythonAstAnalyzer().analyze(artifact, ctx, facts)
    assert all(o.status == "parser_error" for o in out.observations)
    assert [d.code for d in out.diagnostics] == ["PYTHON_ENCODING_ERROR"]


def test_python_disabled():
    config = StylogConfig(
        analysis=AnalysisConfig(
            code=CodeAnalysisConfig(python=PythonAnalysisConfig(enabled=False))
        )
    )
    out = run_ast("x = 1\n", config)
    assert all(o.status == "disabled" for o in out.observations)


def test_location_unavailable_excludes_length_only():
    ctx = make_ctx()
    artifact = make_artifact("def f():\n    pass\n")
    facts = parse_python(artifact, ctx.config)
    func = facts.tree.body[0]
    func.end_lineno = None  # simulate missing location metadata (spec 8.12)
    broken = dataclasses.replace(facts, tree=facts.tree)
    out = PythonAstAnalyzer().analyze(artifact, ctx, broken)
    by = by_id(out)
    assert by["code.python.structure.function_length_lines"].status == "insufficient_support"
    assert by["code.python.structure.function_count"].value.value == 1
    assert by["code.python.structure.parameter_count"].status == "ok"
    assert [d.code for d in out.diagnostics] == ["PYTHON_LOCATION_UNAVAILABLE"]
