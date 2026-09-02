"""Generic code sample/surface tests (spec 6.6, 8.2-8.3, 25)."""

from __future__ import annotations

import hashlib

from stylog.analysis.code import CodeSampleAnalyzer, CodeSurfaceAnalyzer
from stylog.analysis.registry import (
    ANALYZER_CODE_SAMPLE,
    ANALYZER_CODE_SURFACE,
    features_owned_by,
)
from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.provenance import current_runtime_signature
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact


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


def test_sample_counts_ascii():
    out = by_id(CodeSampleAnalyzer().analyze(make_artifact("a = 1\n"), make_ctx()))
    assert out["code.sample.byte_count"].value.value == 6
    assert out["code.sample.character_count"].value.value == 6
    assert out["code.sample.physical_line_count"].value.value == 2
    for obs in out.values():
        assert obs.status == "ok"
        assert obs.support.kind == "artifact"
        assert obs.support.count == 1


def test_sample_counts_non_ascii():
    out = by_id(CodeSampleAnalyzer().analyze(make_artifact("α = 1\n"), make_ctx()))
    assert out["code.sample.byte_count"].value.value == 7  # alpha is two UTF-8 bytes
    assert out["code.sample.character_count"].value.value == 6


def test_physical_line_count_edges():
    sample = CodeSampleAnalyzer()
    for src, expected in (("", 0), ("a", 1), ("a\n", 2), ("\n", 2), ("a\nb", 2)):
        out = by_id(sample.analyze(make_artifact(src), make_ctx()))
        assert out["code.sample.physical_line_count"].value.value == expected, src


def test_sample_always_ok_and_covers_registry():
    out = CodeSampleAnalyzer().analyze(make_artifact(""), make_ctx())
    owned = [f.feature_id for f in features_owned_by(ANALYZER_CODE_SAMPLE)]
    assert [o.feature_id for o in out.observations] == sorted(owned)
    assert all(o.status == "ok" for o in out.observations)


def test_surface_covers_registry_sorted():
    out = CodeSurfaceAnalyzer().analyze(make_artifact("x = 1\n"), make_ctx())
    owned = [f.feature_id for f in features_owned_by(ANALYZER_CODE_SURFACE)]
    assert [o.feature_id for o in out.observations] == sorted(owned)


def test_empty_source_insufficient_support():
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(""), make_ctx()))
    for feature_id, obs in out.items():
        assert obs.status == "insufficient_support", feature_id


def test_line_ending_categories():
    src = "a\r\nb\nc\rd e f"
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    assert cat(out["code.surface.line_ending"]) == {
        "crlf": 1,
        "lf": 1,
        "cr": 1,
        "line_separator": 1,
        "paragraph_separator": 1,
    }


def test_line_ending_crlf_single_sequence():
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact("a\r\nb\r\n"), make_ctx()))
    assert cat(out["code.surface.line_ending"]) == {"crlf": 2}
    ws = cat(out["code.surface.whitespace_class"])
    assert ws["carriage_return"] == 2
    assert ws["line_feed"] == 2


def test_indent_kind_and_char_count():
    src = "x\n  y\n\tz\n \tw\n"
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    assert cat(out["code.surface.indent_kind"]) == {
        "none": 1,
        "spaces": 1,
        "tabs": 1,
        "mixed": 1,
    }
    assert hist(out["code.surface.indent_char_count"]) == {0: 1, 2: 2, 1: 1}
    # nonblank line lengths count code points excluding the line ending
    assert hist(out["code.surface.nonblank_line_length"]) == {1: 1, 3: 2, 2: 1}


def test_indent_counts_characters_not_visual_columns():
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact("\tx\n"), make_ctx()))
    assert hist(out["code.surface.indent_char_count"]) == {1: 1}


def test_blank_line_share_and_runs():
    src = "a\n\n\nb\n\nc"
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    ratio = out["code.surface.blank_line_share"].value
    assert (ratio.numerator, ratio.denominator) == (3, 6)
    assert hist(out["code.surface.blank_run_length"]) == {2: 1, 1: 1}


def test_blank_run_top_code():
    src = "\n" * 15  # 16 blank lines, one run of 16 -> top-coded to 11
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    assert hist(out["code.surface.blank_run_length"]) == {11: 1}
    assert out["code.surface.blank_run_length"].value.top_code == 11


def test_no_blank_lines_blank_run_insufficient():
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact("a\nb"), make_ctx()))
    assert out["code.surface.blank_run_length"].status == "insufficient_support"
    ratio = out["code.surface.blank_line_share"].value
    assert (ratio.numerator, ratio.denominator) == (0, 2)


def test_trailing_space_line_share():
    src = "a \nb\t\nc\n"
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    ratio = out["code.surface.trailing_space_line_share"].value
    assert (ratio.numerator, ratio.denominator) == (2, 4)


def test_whitespace_classes():
    src = "a b\tc d\n"
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact(src), make_ctx()))
    assert cat(out["code.surface.whitespace_class"]) == {
        "space_ascii": 1,
        "tab": 1,
        "other_white_space": 1,  # U+00A0
        "line_feed": 1,
    }


def test_surface_ok_on_broken_python():
    # spec 25.9: generic surface stays computable when the language parser
    # rejects the input (no parser_error from the surface analyzer)
    out = by_id(CodeSurfaceAnalyzer().analyze(make_artifact("def f(\n"), make_ctx()))
    assert all(obs.status == "ok" for obs in out.values())
