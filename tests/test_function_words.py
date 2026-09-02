"""English function-word analyzer tests (spec 6.5, 10; fixture 25).

The pinned resource ``src/stylog/resources/function_words_en_v1.txt`` gates the
active-path tests: when it is not checked in yet, those tests skip. The
language/config gating tests run unconditionally (they never read lexemes).
"""

from __future__ import annotations

import hashlib
import importlib.resources

import pytest

from stylog.analysis.registry import (
    ANALYZER_TEXT_FUNCTION_WORDS_EN,
    FUNCTION_WORDS_EN_RESOURCE_ID,
    FUNCTION_WORDS_EN_RESOURCE_VERSION,
)
from stylog.analysis.text import TextFunctionWordsEnAnalyzer, tokenize_text
from stylog.config import AnalysisConfig, StylogConfig, TextAnalysisConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.feature import (
    FeatureStatus,
    OkFeatureObservation,
    RatioValue,
)
from stylog.domain.provenance import ResourceSignature, current_runtime_signature
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact

_RESOURCE_REF = importlib.resources.files("stylog").joinpath("resources").joinpath(
    "function_words_en_v1.txt"
)
try:
    _RESOURCE_BYTES = _RESOURCE_REF.read_bytes() if _RESOURCE_REF.is_file() else None
except (FileNotFoundError, NotADirectoryError, ModuleNotFoundError):
    _RESOURCE_BYTES = None

requires_resource = pytest.mark.skipif(
    _RESOURCE_BYTES is None, reason="function-words resource not yet checked in"
)

EXPECTED_SHA256 = "2177da3067b27ab7f1c1c228474bd5f3c6d59d3c71ffac6ab52c787c4ea881f5"


def _load_lexemes() -> frozenset[str]:
    assert _RESOURCE_BYTES is not None
    return frozenset(
        line for line in _RESOURCE_BYTES.decode("utf-8").split("\n") if line
    )


def _make_artifact(text: str, language: str) -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/function-words",
        kind=ArtifactKind.TEXT,
        language=language,
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _make_context(
    config: StylogConfig | None = None,
    resources: ResourceHandles | None = None,
) -> AnalysisContext:
    return AnalysisContext(
        config=config if config is not None else StylogConfig(),
        runtime=current_runtime_signature(),
        resources=resources if resources is not None else ResourceHandles(),
    )


def _loaded_resources() -> ResourceHandles:
    assert _RESOURCE_BYTES is not None
    return ResourceHandles(
        function_words_en=_load_lexemes(),
        function_words_en_signature=ResourceSignature(
            id=FUNCTION_WORDS_EN_RESOURCE_ID,
            version=FUNCTION_WORDS_EN_RESOURCE_VERSION,
            sha256=hashlib.sha256(_RESOURCE_BYTES).hexdigest(),
        ),
    )


@requires_resource
def test_resource_sha256_pinned() -> None:
    assert _RESOURCE_BYTES is not None
    assert hashlib.sha256(_RESOURCE_BYTES).hexdigest() == EXPECTED_SHA256


@requires_resource
def test_resource_shape() -> None:
    assert _RESOURCE_BYTES is not None
    assert _RESOURCE_BYTES.endswith(b"\n")
    lexemes = _RESOURCE_BYTES.decode("utf-8").split("\n")
    assert lexemes[-1] == ""
    entries = lexemes[:-1]
    assert len(entries) == 222
    assert entries == sorted(entries)
    assert all(entry == entry.casefold() for entry in entries)


@requires_resource
def test_en_active_exact_matches() -> None:
    text = "I would go to the house, but she wouldn't."
    context = _make_context(resources=_loaded_resources())
    output = TextFunctionWordsEnAnalyzer().analyze(_make_artifact(text, "en"), context)
    assert output.diagnostics == ()
    observations = {obs.feature_id: obs for obs in output.observations}

    words = [token for token in tokenize_text(text) if token.kind == "word"]
    assert len(words) == 9  # I would go to the house but she wouldn't
    lexemes = _load_lexemes()
    expected_matched = [token.text.casefold() for token in words if token.text.casefold() in lexemes]

    share = observations["text.function_words.en.token_share"]
    assert isinstance(share, OkFeatureObservation)
    assert isinstance(share.value, RatioValue)
    assert share.value.denominator == 9
    assert share.value.numerator == len(expected_matched)
    assert share.value.multiplier == 1.0
    assert share.value.value == len(expected_matched) / 9
    assert (share.support.kind, share.support.count) == ("word", 9)

    distribution = observations["text.function_words.en.lexeme_distribution"]
    assert isinstance(distribution, OkFeatureObservation)
    keys = {entry.key for entry in distribution.value.counts}  # type: ignore[union-attr]
    assert keys == set(expected_matched)
    assert keys <= lexemes
    assert "house" not in keys
    assert distribution.value.total == len(expected_matched)  # type: ignore[union-attr]
    assert distribution.support.kind == "matched function word"
    assert distribution.support.count == len(expected_matched)


@requires_resource
def test_en_no_words_insufficient() -> None:
    context = _make_context(resources=_loaded_resources())
    output = TextFunctionWordsEnAnalyzer().analyze(_make_artifact("?!", "en"), context)
    for observation in output.observations:
        assert observation.status == FeatureStatus.INSUFFICIENT_SUPPORT


@requires_resource
def test_en_no_matches_share_ok_distribution_insufficient() -> None:
    context = _make_context(resources=_loaded_resources())
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("xyzzy plugh", "en"), context
    )
    observations = {obs.feature_id: obs for obs in output.observations}
    share = observations["text.function_words.en.token_share"]
    assert isinstance(share, OkFeatureObservation)
    assert isinstance(share.value, RatioValue)
    assert (share.value.numerator, share.value.denominator, share.value.value) == (0, 2, 0.0)
    distribution = observations["text.function_words.en.lexeme_distribution"]
    assert distribution.status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_und_language_unavailable_with_diagnostic() -> None:
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("hello world", "und"), _make_context()
    )
    assert len(output.observations) == 2
    for observation in output.observations:
        assert observation.status == FeatureStatus.UNAVAILABLE
    assert len(output.diagnostics) == 1
    diagnostic = output.diagnostics[0]
    assert diagnostic.code == "LANGUAGE_UNSPECIFIED"
    assert diagnostic.severity == DiagnosticSeverity.WARNING
    assert diagnostic.analyzer_id == ANALYZER_TEXT_FUNCTION_WORDS_EN


def test_other_language_not_applicable() -> None:
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("bonjour le monde", "fr"), _make_context()
    )
    assert len(output.observations) == 2
    for observation in output.observations:
        assert observation.status == FeatureStatus.NOT_APPLICABLE
    assert output.diagnostics == ()


def test_config_disabled() -> None:
    config = StylogConfig(
        analysis=AnalysisConfig(text=TextAnalysisConfig(function_words_en=False))
    )
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("hello world", "en"), _make_context(config=config)
    )
    assert len(output.observations) == 2
    for observation in output.observations:
        assert observation.status == FeatureStatus.DISABLED
    assert output.diagnostics == ()


def test_en_missing_resource_unavailable_without_diagnostic() -> None:
    # ResourceHandles() has function_words_en=None: applies but unavailable.
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("hello world", "en"), _make_context()
    )
    assert len(output.observations) == 2
    for observation in output.observations:
        assert observation.status == FeatureStatus.UNAVAILABLE
    assert output.diagnostics == ()


def test_disabled_takes_precedence_over_language_gate() -> None:
    config = StylogConfig(
        analysis=AnalysisConfig(text=TextAnalysisConfig(function_words_en=False))
    )
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("hello world", "und"), _make_context(config=config)
    )
    for observation in output.observations:
        assert observation.status == FeatureStatus.DISABLED
    assert output.diagnostics == ()


def test_resources_signature_only_when_loaded() -> None:
    analyzer = TextFunctionWordsEnAnalyzer()
    signature = ResourceSignature(
        id=FUNCTION_WORDS_EN_RESOURCE_ID,
        version=FUNCTION_WORDS_EN_RESOURCE_VERSION,
        sha256="0" * 64,
    )
    loaded = _make_context(
        resources=ResourceHandles(function_words_en=frozenset({"the"}),
                                  function_words_en_signature=signature)
    )
    assert analyzer.resources(loaded) == (signature,)
    assert analyzer.signature(loaded).resources == (signature,)
    empty = _make_context()
    assert analyzer.resources(empty) == ()
    assert analyzer.signature(empty).resources == ()


def test_observation_coverage_and_order() -> None:
    output = TextFunctionWordsEnAnalyzer().analyze(
        _make_artifact("hello", "fr"), _make_context()
    )
    assert [obs.feature_id for obs in output.observations] == [
        "text.function_words.en.lexeme_distribution",
        "text.function_words.en.token_share",
    ]
