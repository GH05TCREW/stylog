"""Binding-event extraction tests (spec 8.10, fixture 25.8)."""

from __future__ import annotations

import hashlib

from stylog.analysis.identifiers import split_components
from stylog.analysis.python import PythonAstAnalyzer
from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.provenance import current_runtime_signature
from stylog.parsers.python_native import parse_python
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact

# Spec 25.8: one module exercising every binding role.
BINDING_SRC = '''import os
import collections as col
from sys import path
from os import getcwd as cwd
from math import *


class Widget:
    pass


async def afunc():
    pass


def __main__(pos_only, /, positional, *args, kw_only, **kwargs):
    a, (b, c) = (1, (2, 3))
    d: int = 4
    d += 5
    _ = 10
    if (e := 6):
        pass
    for f, g in []:
        pass
    h = [i for i in []]
    with open("x") as w:
        pass
    try:
        pass
    except ValueError as exc:
        pass
    match a:
        case [m, *rest]:
            pass
        case {"k": v, **others}:
            pass
        case _:
            pass
    obj.attr = 1
'''

# Expected binding events as (name, role); "obj.attr" is NOT a binding.
EXPECTED_BINDINGS = [
    ("os", "import_binding"),
    ("col", "import_binding"),
    ("path", "import_binding"),
    ("cwd", "import_binding"),
    ("Widget", "class"),
    ("afunc", "async_function"),
    ("__main__", "function"),
    ("pos_only", "parameter_posonly"),
    ("positional", "parameter_positional"),
    ("args", "parameter_vararg"),
    ("kw_only", "parameter_kwonly"),
    ("kwargs", "parameter_kwarg"),
    ("a", "assignment"),
    ("b", "assignment"),
    ("c", "assignment"),
    ("d", "assignment"),  # AnnAssign
    ("d", "assignment"),  # AugAssign
    ("_", "assignment"),
    ("e", "walrus"),
    ("f", "loop_target"),
    ("g", "loop_target"),
    ("h", "assignment"),
    ("i", "comprehension_target"),
    ("w", "with_target"),
    ("exc", "exception_target"),
    ("m", "pattern_binding"),
    ("rest", "pattern_binding"),
    ("v", "pattern_binding"),
    ("others", "pattern_binding"),
]


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


def make_ctx() -> AnalysisContext:
    return AnalysisContext(
        config=StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(),
    )


def by_id(output):
    return {obs.feature_id: obs for obs in output.observations}


def cat(obs) -> dict[str, int]:
    return {entry.key: entry.count for entry in obs.value.counts}


def hist(obs) -> dict[int, int]:
    return {point.point: point.count for point in obs.value.points}


def run_ast(src: str):
    ctx = make_ctx()
    artifact = make_artifact(src)
    facts = parse_python(artifact, ctx.config)
    assert facts.tree is not None
    return PythonAstAnalyzer().analyze(artifact, ctx, facts)


def test_binding_role_multiset():
    out = by_id(run_ast(BINDING_SRC))
    expected_roles: dict[str, int] = {}
    for _name, role in EXPECTED_BINDINGS:
        expected_roles[role] = expected_roles.get(role, 0) + 1
    assert expected_roles == {
        "assignment": 7,
        "async_function": 1,
        "class": 1,
        "comprehension_target": 1,
        "exception_target": 1,
        "function": 1,
        "import_binding": 4,
        "loop_target": 2,
        "parameter_kwarg": 1,
        "parameter_kwonly": 1,
        "parameter_positional": 1,
        "parameter_posonly": 1,
        "parameter_vararg": 1,
        "pattern_binding": 4,
        "walrus": 1,
        "with_target": 1,
    }
    assert cat(out["code.python.naming.binding_role"]) == expected_roles
    assert out["code.python.naming.binding_role"].value.total == 29


def test_binding_length_multiset():
    out = by_id(run_ast(BINDING_SRC))
    expected: dict[int, int] = {}
    for name, _role in EXPECTED_BINDINGS:
        expected[len(name)] = expected.get(len(name), 0) + 1
    assert hist(out["code.python.naming.binding_length"]) == expected


def test_binding_case_style_multiset():
    out = by_id(run_ast(BINDING_SRC))
    # every name is lower except: Widget (pascal), __main__ (dunder),
    # pos_only/kw_only (snake_lower); "_" is classified discard (excluded)
    assert cat(out["code.python.naming.binding_case_style"]) == {
        "dunder": 1,
        "lower": 24,
        "pascal": 1,
        "snake_lower": 2,
    }


def test_binding_component_lengths():
    out = by_id(run_ast(BINDING_SRC))
    expected: dict[int, int] = {}
    for name, _role in EXPECTED_BINDINGS:
        for component in split_components(name):
            expected[len(component)] = expected.get(len(component), 0) + 1
    assert hist(out["code.python.naming.binding_component_length"]) == expected
    # "_" contributes no components
    assert out["code.python.naming.binding_component_length"].value.total == sum(
        len(split_components(name)) for name, _role in EXPECTED_BINDINGS
    )


def test_attribute_assignment_is_not_binding_but_attribute_occurrence():
    out = by_id(run_ast(BINDING_SRC))
    assert hist(out["code.python.naming.attribute_name_length"]) == {4: 1}  # attr
    assert cat(out["code.python.naming.attribute_case_style"]) == {"lower": 1}
    # 29 bindings, none named "obj" or "attr"
    assert out["code.python.naming.binding_role"].value.total == 29


def test_import_star_binds_nothing():
    out = by_id(run_ast("from math import *\n"))
    assert out["code.python.naming.binding_role"].status == "insufficient_support"
    # but the alias still counts toward the alias-share denominator
    share = out["code.python.structure.import_alias_share"].value
    assert (share.numerator, share.denominator) == (0, 1)


def test_nested_tuple_and_starred_targets():
    out = by_id(run_ast("a, *b = [1, 2, 3]\n[c, (d, e)] = [(1, (2, 3))]\n"))
    assert cat(out["code.python.naming.binding_role"]) == {"assignment": 5}


def test_loop_target_not_also_assignment():
    out = by_id(run_ast("for x in []:\n    pass\n"))
    assert cat(out["code.python.naming.binding_role"]) == {"loop_target": 1}


def test_lambda_parameters_bind_with_structural_roles():
    out = by_id(run_ast("f = lambda a, /, b, *c, d, **e: a\n"))
    assert cat(out["code.python.naming.binding_role"]) == {
        "assignment": 1,
        "parameter_kwarg": 1,
        "parameter_kwonly": 1,
        "parameter_positional": 1,
        "parameter_posonly": 1,
        "parameter_vararg": 1,
    }
