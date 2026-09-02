"""JCS / portable serialization conformance (spec 14, 25.20)."""

from __future__ import annotations

import math

import pytest
import rfc8785

from stylog.domain import (
    Diagnostic,
    DiagnosticSeverity,
    IntegerValue,
)
from stylog.exceptions import PortableArtifactError
from stylog.serialization.canonical import (
    canonical_bytes_of_tree,
    portable_tree,
    sha256_of_tree,
)
from stylog.serialization.jsonio import model_from_bytes


def test_rfc8785_official_vector_key_order() -> None:
    # RFC 8785 sorts object properties by UTF-16 code units.
    assert rfc8785.dumps({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert rfc8785.dumps({"€": 1, "a": 2}) == "{\"a\":2,\"€\":1}".encode()


def test_rfc8785_number_and_string_canonicalization() -> None:
    assert rfc8785.dumps({"x": 1.5, "y": True, "z": None}) == b'{"x":1.5,"y":true,"z":null}'
    assert rfc8785.dumps({"s": "a\nb"}) == b'{"s":"a\\nb"}'


def test_null_rejected_in_portable_tree() -> None:
    with pytest.raises(PortableArtifactError):
        canonical_bytes_of_tree({"a": None})


def test_nan_and_inf_rejected() -> None:
    with pytest.raises(PortableArtifactError):
        canonical_bytes_of_tree({"a": math.nan})
    with pytest.raises(PortableArtifactError):
        canonical_bytes_of_tree({"a": math.inf})


def test_negative_zero_normalized() -> None:
    value = IntegerValue(value=0)
    assert value.value == 0
    tree = portable_tree(value)
    assert tree["value"] == 0
    assert canonical_bytes_of_tree({"v": -0.0}) == b'{"v":0}'


def test_lone_surrogate_rejected() -> None:
    with pytest.raises(PortableArtifactError):
        canonical_bytes_of_tree({"a": "lone\ud800surrogate"})


def test_unsafe_integer_rejected() -> None:
    with pytest.raises(PortableArtifactError):
        canonical_bytes_of_tree({"a": 2**53})


def test_model_null_rejected_on_parse() -> None:
    with pytest.raises(PortableArtifactError):
        model_from_bytes(b'{"code":"X","severity":"warning","analyzer_id":null}', Diagnostic)


def test_diagnostic_roundtrip_strict() -> None:
    diagnostic = Diagnostic(code="A", severity=DiagnosticSeverity.ERROR)
    raw = b'{"code":"A","severity":"error"}'
    parsed = model_from_bytes(raw, Diagnostic)
    assert parsed == diagnostic


def test_tree_hash_deterministic() -> None:
    tree1 = {"b": [1, 2], "a": {"z": 1}}
    tree2 = {"a": {"z": 1}, "b": [1, 2]}
    assert sha256_of_tree(tree1) == sha256_of_tree(tree2)


def test_portable_tree_omits_none_fields() -> None:
    diagnostic = Diagnostic(code="A", severity=DiagnosticSeverity.WARNING)
    tree = portable_tree(diagnostic)
    assert set(tree) == {"code", "severity", "context"}
    assert "analyzer_id" not in tree
