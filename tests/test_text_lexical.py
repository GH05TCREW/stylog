"""Text lexical analyzer conformance tests (spec 6.3, 7.11-7.13, 25)."""

from __future__ import annotations

import hashlib

import pytest

from stylog.analysis.text import TextLexicalAnalyzer
from stylog.config import AnalysisConfig, StylogConfig, TextAnalysisConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.feature import (
    FeatureObservation,
    FeatureStatus,
    FloatValue,
    IntegerValue,
    OkFeatureObservation,
    OrderedHistogramValue,
    RatioValue,
    SummaryStatisticsValue,
)
from stylog.domain.provenance import current_runtime_signature
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact


def _make_artifact(text: str) -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/lexical",
        kind=ArtifactKind.TEXT,
        language="en",
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _make_context(config: StylogConfig | None = None) -> AnalysisContext:
    return AnalysisContext(
        config=config if config is not None else StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(),
    )


def _run(text: str, config: StylogConfig | None = None) -> dict[str, FeatureObservation]:
    output = TextLexicalAnalyzer().analyze(_make_artifact(text), _make_context(config))
    assert output.diagnostics == ()
    return {observation.feature_id: observation for observation in output.observations}


def _ok(observation: FeatureObservation) -> OkFeatureObservation:
    assert isinstance(observation, OkFeatureObservation)
    return observation


def test_lexical_stats_fixture_a_a_b_c() -> None:
    observations = _run("a a b c")

    word_count = _ok(observations["text.lexical.word_count"])
    assert isinstance(word_count.value, IntegerValue)
    assert word_count.value.value == 4
    assert (word_count.support.kind, word_count.support.count) == ("artifact", 1)

    number_count = _ok(observations["text.lexical.number_count"])
    assert number_count.value.value == 0  # type: ignore[union-attr]

    token_kind = _ok(observations["text.lexical.token_kind"])
    assert [(entry.key, entry.count) for entry in token_kind.value.counts] == [("word", 4)]  # type: ignore[union-attr]

    word_length = _ok(observations["text.lexical.word_length"])
    assert isinstance(word_length.value, OrderedHistogramValue)
    assert [(p.point, p.count) for p in word_length.value.points] == [(1, 4)]
    assert word_length.value.top_code == 31

    type_count = _ok(observations["text.lexical.type_count_casefold"])
    assert type_count.value.value == 3  # type: ignore[union-attr]
    assert (type_count.support.kind, type_count.support.count) == ("word", 4)

    ttr = _ok(observations["text.lexical.ttr_casefold"])
    assert isinstance(ttr.value, RatioValue)
    assert (ttr.value.numerator, ttr.value.denominator, ttr.value.multiplier) == (3, 4, 1.0)
    assert ttr.value.value == 0.75

    hapax_types = _ok(observations["text.lexical.hapax_type_count_casefold"])
    assert hapax_types.value.value == 2  # type: ignore[union-attr]

    hapax_share = _ok(observations["text.lexical.hapax_token_share_casefold"])
    assert isinstance(hapax_share.value, RatioValue)
    assert (hapax_share.value.numerator, hapax_share.value.denominator) == (2, 4)
    assert hapax_share.value.value == 0.5

    entropy = _ok(observations["text.lexical.word_entropy_bits_casefold"])
    assert isinstance(entropy.value, FloatValue)
    assert entropy.value.value == 1.5

    simpson = _ok(observations["text.lexical.word_simpson_concentration_casefold"])
    assert isinstance(simpson.value, FloatValue)
    assert simpson.value.value == 0.375

    window = observations["text.lexical.window_ttr_100"]
    assert window.status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_zero_words_statuses() -> None:
    observations = _run("!!!")
    assert _ok(observations["text.lexical.word_count"]).value.value == 0  # type: ignore[union-attr]
    assert _ok(observations["text.lexical.type_count_casefold"]).value.value == 0  # type: ignore[union-attr]
    assert _ok(observations["text.lexical.hapax_type_count_casefold"]).value.value == 0  # type: ignore[union-attr]
    for feature_id in (
        "text.lexical.ttr_casefold",
        "text.lexical.hapax_token_share_casefold",
        "text.lexical.word_entropy_bits_casefold",
        "text.lexical.word_simpson_concentration_casefold",
        "text.lexical.word_length",
        "text.lexical.token_kind",
        "text.lexical.window_ttr_100",
    ):
        assert observations[feature_id].status == FeatureStatus.INSUFFICIENT_SUPPORT, feature_id


def test_casefold_type_equality() -> None:
    observations = _run("Aa AA aa")
    assert _ok(observations["text.lexical.type_count_casefold"]).value.value == 1  # type: ignore[union-attr]
    ttr = _ok(observations["text.lexical.ttr_casefold"])
    assert isinstance(ttr.value, RatioValue)
    assert (ttr.value.numerator, ttr.value.denominator) == (1, 3)


def test_token_kind_with_numbers() -> None:
    observations = _run("x 1,000.50 y")
    token_kind = _ok(observations["text.lexical.token_kind"])
    assert [(entry.key, entry.count) for entry in token_kind.value.counts] == [  # type: ignore[union-attr]
        ("number", 1),
        ("word", 2),
    ]
    assert _ok(observations["text.lexical.number_count"]).value.value == 1  # type: ignore[union-attr]


def test_word_length_top_code_31() -> None:
    long_word = "a" * 40
    observations = _run(f"short {long_word}")
    word_length = _ok(observations["text.lexical.word_length"])
    assert isinstance(word_length.value, OrderedHistogramValue)
    assert [(p.point, p.count) for p in word_length.value.points] == [(5, 1), (31, 1)]


def _letter_word(index: int) -> str:
    """Unique letter-only token (digits would split into WORD+NUMBER)."""
    return chr(97 + index // 26) + chr(97 + index % 26)


def test_window_ttr_exactly_100_words_one_window() -> None:
    text = " ".join(_letter_word(i) for i in range(100))
    window = _ok(_run(text)["text.lexical.window_ttr_100"])
    assert isinstance(window.value, SummaryStatisticsValue)
    assert window.value.n == 1
    assert window.value.minimum == window.value.maximum == window.value.mean == 1.0
    assert window.value.q25 == window.value.median == window.value.q75 == 1.0
    assert window.value.sample_sd is None
    assert (window.support.kind, window.support.count) == ("100-word window", 1)


def test_window_ttr_99_words_insufficient() -> None:
    text = " ".join(_letter_word(i) for i in range(99))
    observation = _run(text)["text.lexical.window_ttr_100"]
    assert observation.status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_window_ttr_discards_incomplete_tail() -> None:
    first = [_letter_word(i) for i in range(100)]  # 100 distinct types
    second = [_letter_word(i) for i in range(50)] * 2  # 50 distinct types
    tail = [_letter_word(i) for i in range(50, 100)]  # discarded
    text = " ".join(first + second + tail)
    window = _ok(_run(text)["text.lexical.window_ttr_100"])
    assert isinstance(window.value, SummaryStatisticsValue)
    assert window.value.n == 2
    assert window.value.minimum == 0.5
    assert window.value.maximum == 1.0
    assert window.value.mean == 0.75
    assert window.value.median == 0.75
    assert window.value.sample_sd == pytest.approx((0.125) ** 0.5)
    assert window.support.count == 2


def test_window_ttr_disabled_by_config() -> None:
    config = StylogConfig(
        analysis=AnalysisConfig(text=TextAnalysisConfig(window_ttr_100=False))
    )
    observations = _run(" ".join(_letter_word(i) for i in range(200)), config)
    window = observations["text.lexical.window_ttr_100"]
    assert window.status == FeatureStatus.DISABLED
    # Other features are unaffected.
    assert _ok(observations["text.lexical.word_count"]).value.value == 200  # type: ignore[union-attr]
