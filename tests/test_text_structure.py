"""Text structure analyzer conformance tests (spec 6.4, 7.8, 7.9, 25)."""

from __future__ import annotations

import hashlib

from stylog.analysis.text import TextStructureAnalyzer
from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.feature import (
    FeatureObservation,
    FeatureStatus,
    IntegerValue,
    OkFeatureObservation,
    OrderedHistogramValue,
)
from stylog.domain.provenance import current_runtime_signature
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact


def _make_artifact(text: str) -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/structure",
        kind=ArtifactKind.TEXT,
        language="en",
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _run(text: str) -> dict[str, FeatureObservation]:
    context = AnalysisContext(
        config=StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(),
    )
    output = TextStructureAnalyzer().analyze(_make_artifact(text), context)
    assert output.diagnostics == ()
    return {observation.feature_id: observation for observation in output.observations}


def _ok(observation: FeatureObservation) -> OkFeatureObservation:
    assert isinstance(observation, OkFeatureObservation)
    return observation


def _histogram_points(observation: FeatureObservation) -> list[tuple[int, int]]:
    ok = _ok(observation)
    assert isinstance(ok.value, OrderedHistogramValue)
    return [(point.point, point.count) for point in ok.value.points]


def test_sentence_fixture() -> None:
    observations = _run("Dr. Smith left. Value 3.14! Really?")
    sentence_count = _ok(observations["text.structure.sentence_count"])
    assert isinstance(sentence_count.value, IntegerValue)
    assert sentence_count.value.value == 4
    assert (sentence_count.support.kind, sentence_count.support.count) == ("artifact", 1)

    # "Dr." (1 token, 3 chars), "Smith left." (2, 11), "Value 3.14!" (2, 11),
    # "Really?" (1, 7).
    assert _histogram_points(observations["text.structure.sentence_length_tokens"]) == [
        (1, 2),
        (2, 2),
    ]
    assert _histogram_points(observations["text.structure.sentence_length_characters"]) == [
        (3, 1),
        (7, 1),
        (11, 2),
    ]
    assert _ok(observations["text.structure.paragraph_count"]).value.value == 1  # type: ignore[union-attr]
    assert _histogram_points(observations["text.structure.paragraph_sentence_count"]) == [(4, 1)]
    # Tokens: Dr, Smith, left, Value, 3.14, Really.
    assert _histogram_points(observations["text.structure.paragraph_token_count"]) == [(6, 1)]


def test_paragraph_fixture() -> None:
    # "a\r\n\r\nb\u2028c": blank line separates; U+2028 ends a line, not a paragraph.
    observations = _run("a\r\n\r\nb\u2028c")
    assert _ok(observations["text.structure.paragraph_count"]).value.value == 2  # type: ignore[union-attr]
    assert _ok(observations["text.structure.sentence_count"]).value.value == 2  # type: ignore[union-attr]
    assert _histogram_points(observations["text.structure.paragraph_sentence_count"]) == [(1, 2)]
    # Paragraph 1: "a" (1 token); paragraph 2: "b\u2028c" (2 tokens).
    assert _histogram_points(observations["text.structure.paragraph_token_count"]) == [
        (1, 1),
        (2, 1),
    ]
    # The second sentence span keeps its internal U+2028 (3 code points).
    assert _histogram_points(observations["text.structure.sentence_length_characters"]) == [
        (1, 1),
        (3, 1),
    ]


def test_u2029_always_terminates_paragraph() -> None:
    observations = _run("a\u2029b")
    assert _ok(observations["text.structure.paragraph_count"]).value.value == 2  # type: ignore[union-attr]


def test_empty_text_counts_ok_histograms_insufficient() -> None:
    observations = _run("")
    assert _ok(observations["text.structure.sentence_count"]).value.value == 0  # type: ignore[union-attr]
    assert _ok(observations["text.structure.paragraph_count"]).value.value == 0  # type: ignore[union-attr]
    for feature_id in (
        "text.structure.sentence_length_tokens",
        "text.structure.sentence_length_characters",
        "text.structure.paragraph_sentence_count",
        "text.structure.paragraph_token_count",
    ):
        assert observations[feature_id].status == FeatureStatus.INSUFFICIENT_SUPPORT, feature_id


def test_multi_paragraph_sentence_histogram() -> None:
    observations = _run("One. Two.\n\nThree.")
    assert _ok(observations["text.structure.sentence_count"]).value.value == 3  # type: ignore[union-attr]
    assert _ok(observations["text.structure.paragraph_count"]).value.value == 2  # type: ignore[union-attr]
    assert _histogram_points(observations["text.structure.paragraph_sentence_count"]) == [
        (1, 1),
        (2, 1),
    ]


def test_sentence_spanning_physical_line() -> None:
    # No boundary without whitespace after the cluster; LF inside a paragraph
    # is whitespace and ends the sentence, and remains inside a span when the
    # sentence continues across the line.
    observations = _run("one two\nthree four.")
    assert _ok(observations["text.structure.sentence_count"]).value.value == 1  # type: ignore[union-attr]
    # Span "one two\nthree four." is 19 code points including the LF.
    assert _histogram_points(observations["text.structure.sentence_length_characters"]) == [(19, 1)]
    assert _histogram_points(observations["text.structure.sentence_length_tokens"]) == [(4, 1)]
