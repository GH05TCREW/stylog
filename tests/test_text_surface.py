"""Text sample/surface analyzer conformance tests (spec 6.2, 7.2, 7.7, 25)."""

from __future__ import annotations

import hashlib

from stylog.analysis.registry import features_owned_by
from stylog.analysis.text import TextSampleAnalyzer, TextSurfaceAnalyzer
from stylog.analysis.whitespace import WHITE_SPACE_CODEPOINTS
from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.feature import FeatureObservation, FeatureStatus, OkFeatureObservation
from stylog.domain.provenance import current_runtime_signature
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact


def _make_artifact(text: str, language: str = "en") -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/surface",
        kind=ArtifactKind.TEXT,
        language=language,
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _make_context() -> AnalysisContext:
    return AnalysisContext(
        config=StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(),
    )


def _run(analyzer, text: str) -> dict[str, FeatureObservation]:
    output = analyzer.analyze(_make_artifact(text), _make_context())
    assert output.diagnostics == ()
    return {observation.feature_id: observation for observation in output.observations}


def _counts(observation: FeatureObservation) -> dict[str, int]:
    assert isinstance(observation, OkFeatureObservation)
    return {entry.key: entry.count for entry in observation.value.counts}  # type: ignore[union-attr]


def test_sample_counts() -> None:
    observations = _run(TextSampleAnalyzer(), "café")
    byte_count = observations["text.sample.byte_count"]
    char_count = observations["text.sample.character_count"]
    assert isinstance(byte_count, OkFeatureObservation)
    assert byte_count.value.value == 5  # "é" is two UTF-8 bytes
    assert byte_count.support.kind == "artifact"
    assert byte_count.support.count == 1
    assert isinstance(char_count, OkFeatureObservation)
    assert char_count.value.value == 4


def test_sample_counts_empty_text_are_valid_zero() -> None:
    observations = _run(TextSampleAnalyzer(), "")
    for feature_id in ("text.sample.byte_count", "text.sample.character_count"):
        observation = observations[feature_id]
        assert isinstance(observation, OkFeatureObservation)
        assert observation.value.value == 0


def test_owned_feature_coverage_is_complete_and_sorted() -> None:
    for analyzer in (TextSampleAnalyzer(), TextSurfaceAnalyzer()):
        output = analyzer.analyze(_make_artifact("anything"), _make_context())
        expected = [fdef.feature_id for fdef in features_owned_by(analyzer.analyzer_id)]
        assert [obs.feature_id for obs in output.observations] == expected
        assert expected == sorted(expected)


def test_whitespace_fixture_all_25_codepoints() -> None:
    text = "".join(chr(code_point) for code_point in sorted(WHITE_SPACE_CODEPOINTS))
    observation = _run(TextSurfaceAnalyzer(), text)["text.surface.whitespace_class"]
    assert _counts(observation) == {
        "space_ascii": 1,
        "tab": 1,
        "line_feed": 1,
        "carriage_return": 1,
        "line_separator": 1,
        "paragraph_separator": 1,
        "other_white_space": 19,
    }
    assert observation.value.total == 25  # type: ignore[union-attr]
    assert observation.support.kind == "whitespace code point"
    assert observation.support.count == 25


def test_line_ending_categories() -> None:
    text = "a\r\nb\nc\rd\u2028e\u2029f"
    observation = _run(TextSurfaceAnalyzer(), text)["text.surface.line_ending"]
    assert _counts(observation) == {
        "crlf": 1,
        "lf": 1,
        "cr": 1,
        "line_separator": 1,
        "paragraph_separator": 1,
    }
    assert observation.support.count == 5


def test_crlf_counts_once_for_line_endings_twice_for_whitespace() -> None:
    observations = _run(TextSurfaceAnalyzer(), "a\r\nb")
    assert _counts(observations["text.surface.line_ending"]) == {"crlf": 1}
    whitespace = _counts(observations["text.surface.whitespace_class"])
    assert whitespace["carriage_return"] == 1
    assert whitespace["line_feed"] == 1


def test_unicode_general_category_per_codepoint() -> None:
    observation = _run(TextSurfaceAnalyzer(), "Ab3 !")[
        "text.surface.unicode_general_category"
    ]
    assert _counts(observation) == {"Lu": 1, "Ll": 1, "Nd": 1, "Po": 1, "Zs": 1}
    assert observation.support.count == 5


def test_letter_case_classes() -> None:
    # Lu -> upper, Ll -> lower, Lt -> title (U+01C5), Lo -> uncased (U+65E5).
    observation = _run(TextSurfaceAnalyzer(), "Aaǅ日b9")["text.surface.letter_case"]
    assert _counts(observation) == {"upper": 1, "lower": 2, "title": 1, "uncased": 1}
    assert observation.support.count == 5


def test_punctuation_codepoint_keys() -> None:
    observation = _run(TextSurfaceAnalyzer(), "!…。")["text.surface.punctuation_codepoint"]
    assert _counts(observation) == {"U+0021": 1, "U+2026": 1, "U+3002": 1}


def test_marker_style_six_dots() -> None:
    observations = _run(TextSurfaceAnalyzer(), "......")
    marker = observations["text.surface.marker_style"]
    assert _counts(marker) == {"ascii_three_dots": 2}
    punctuation = observations["text.surface.punctuation_codepoint"]
    assert _counts(punctuation) == {"U+002E": 6}


def test_marker_style_all_thirteen_categories() -> None:
    text = "'’\"‘“”«»-–—…..."
    counts = _counts(_run(TextSurfaceAnalyzer(), text)["text.surface.marker_style"])
    assert counts == {
        "apostrophe_ascii": 1,
        "apostrophe_right": 1,
        "quote_ascii": 1,
        "quote_left_single": 1,
        "quote_left_double": 1,
        "quote_right_double": 1,
        "guillemet_left": 1,
        "guillemet_right": 1,
        "hyphen_minus": 1,
        "en_dash": 1,
        "em_dash": 1,
        "horizontal_ellipsis": 1,
        "ascii_three_dots": 1,
    }


def test_empty_text_all_surface_features_insufficient() -> None:
    observations = _run(TextSurfaceAnalyzer(), "")
    assert len(observations) == 6
    for observation in observations.values():
        assert observation.status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_no_whitespace_means_insufficient_whitespace_class() -> None:
    observation = _run(TextSurfaceAnalyzer(), "abc")["text.surface.whitespace_class"]
    assert observation.status == FeatureStatus.INSUFFICIENT_SUPPORT
