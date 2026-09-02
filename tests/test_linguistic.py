"""Linguistic analyzer tests (spec 6.14, 15.5, 22.8).

``en_core_web_sm`` is expected to be provisioned locally; if it is not
importable the whole module skips. The spaCy pipeline itself is used as the
test oracle: expected counts are recomputed from a fresh ``nlp(text)`` doc,
independently of the analyzer implementation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

try:
    import en_core_web_sm  # noqa: F401

    _MODEL_AVAILABLE = True
except ImportError:
    _MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _MODEL_AVAILABLE, reason="en_core_web_sm not provisioned locally"
)

from stylog.analysis.linguistic import LinguisticAnalyzer, load_spacy_model
from stylog.analysis.registry import (
    ANALYZER_TEXT_LINGUISTIC,
    features_owned_by,
)
from stylog.bootstrap import build_default_services
from stylog.config import StylogConfig
from stylog.domain.artifact import ArtifactKind
from stylog.domain.feature import (
    CategoricalDistributionValue,
    FeatureStatus,
    OkFeatureObservation,
    OrderedHistogramValue,
    RatioValue,
)
from stylog.domain.provenance import current_runtime_signature
from stylog.infrastructure.resources import resource_tree_sha256
from stylog.runtime import AnalysisContext, ResourceHandles, RuntimeArtifact

TEXT = "The quick brown fox jumps over the lazy dog."

EXPECTED_OWNED = (
    "text.linguistic.dependency_distance",
    "text.linguistic.dependency_relation",
    "text.linguistic.morph_attribute",
    "text.linguistic.morph_coverage",
    "text.linguistic.upos",
)


@pytest.fixture(scope="module")
def services():
    return build_default_services(StylogConfig())


@pytest.fixture(scope="module")
def loaded(services):
    return load_spacy_model("en_core_web_sm", services)


def _make_artifact(text: str) -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id="test/linguistic",
        kind=ArtifactKind.TEXT,
        language="en",
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _make_context(nlp=None, backend=None) -> AnalysisContext:
    return AnalysisContext(
        config=StylogConfig(),
        runtime=current_runtime_signature(),
        resources=ResourceHandles(nlp_model=nlp, nlp_model_backend=backend),
    )


def _linguistic_tokens(doc):
    return [token for token in doc if not token.is_space]


def test_load_backend_signature(loaded) -> None:
    import spacy

    nlp, backend = loaded
    assert backend.backend_id == "spacy"
    assert backend.implementation_version == spacy.__version__
    assert backend.scientific_compatibility_id == "stylog.spacy/1"
    assert [(p.package, p.version) for p in backend.packages] == [
        ("spacy", spacy.__version__)
    ]
    assert backend.resources == ()

    model = backend.model
    assert model is not None
    # spaCy meta carries the language-agnostic model name ("core_web_sm").
    assert model.model_id == nlp.meta["name"] == "core_web_sm"
    assert model.model_revision == nlp.meta["version"]
    # Tree hash is exactly the provisioned model directory tree hash.
    assert model.model_tree_sha256 == resource_tree_sha256(Path(nlp.path))
    assert model.tokenizer_id == model.model_id + "/tokenizer"
    assert model.tokenizer_version == spacy.__version__
    assert model.tokenizer_tree_sha256 == model.model_tree_sha256
    assert model.preprocessing_id == "stylog.spacy.preprocessing"
    assert model.preprocessing_version == "1.0.0"


def test_load_is_local_only(services) -> None:
    from stylog.exceptions import ResourceError

    # Unknown model package: spacy.load performs no download, so this fails
    # locally and is surfaced as a typed ResourceError.
    with pytest.raises(ResourceError, match="not provisioned locally"):
        load_spacy_model("en_core_web_nonexistent_stylog_test", services)
    # Existing directory path also fails cleanly when it is not a model.
    with pytest.raises(ResourceError, match="not provisioned locally"):
        load_spacy_model(str(Path.cwd()), services)


def test_load_from_directory_path(loaded, services, tmp_path) -> None:
    nlp, _ = loaded
    nlp.to_disk(tmp_path / "model")
    nlp2, backend2 = load_spacy_model(str(tmp_path / "model"), services)
    assert backend2.model is not None
    assert backend2.model.model_id == "core_web_sm"
    assert backend2.model.model_tree_sha256 == resource_tree_sha256(tmp_path / "model")
    assert nlp2.pipe_names == nlp.pipe_names


def test_ownership_exactly_five_registry_features() -> None:
    analyzer = LinguisticAnalyzer()
    assert analyzer.analyzer_id == ANALYZER_TEXT_LINGUISTIC
    assert analyzer.needs == "none"
    assert analyzer.owned_feature_ids() == EXPECTED_OWNED
    # The core text tokenization features stay with the text analyzers: the
    # linguistic analyzer owns nothing outside text.linguistic.* (spec 6.14:
    # "Core tokenization not reused for these labels").
    assert all(fid.startswith("text.linguistic.") for fid in EXPECTED_OWNED)
    assert {f.feature_id for f in features_owned_by(ANALYZER_TEXT_LINGUISTIC)} == set(
        EXPECTED_OWNED
    )


def test_signature_uses_context_backend(loaded) -> None:
    nlp, backend = loaded
    analyzer = LinguisticAnalyzer()
    assert analyzer.signature(_make_context(nlp, backend)).backend == backend
    fallback = analyzer.signature(_make_context()).backend
    assert fallback.backend_id == "spacy"
    assert fallback.implementation_version == "unknown"
    assert fallback.scientific_compatibility_id == "stylog.spacy/1"
    assert fallback.model is None


def test_analyze_sentence_all_ok(loaded) -> None:
    nlp, backend = loaded
    ctx = _make_context(nlp, backend)
    output = LinguisticAnalyzer().analyze(_make_artifact(TEXT), ctx)
    assert output.diagnostics == ()
    observations = {obs.feature_id: obs for obs in output.observations}
    assert list(observations) == sorted(EXPECTED_OWNED)
    for observation in observations.values():
        assert observation.status == FeatureStatus.OK

    # Oracle: independent recomputation from a fresh spaCy doc.
    doc = nlp(TEXT)
    tokens = _linguistic_tokens(doc)
    assert len(tokens) == 10  # 9 words plus the final punctuation token

    # --- text.linguistic.upos ---
    expected_pos: dict[str, int] = {}
    for token in tokens:
        if token.pos_:
            expected_pos[token.pos_] = expected_pos.get(token.pos_, 0) + 1
    upos = observations["text.linguistic.upos"]
    assert isinstance(upos, OkFeatureObservation)
    assert isinstance(upos.value, CategoricalDistributionValue)
    assert {e.key: e.count for e in upos.value.counts} == expected_pos
    # The sentence exercises the core nominal/verbal categories.
    assert {"NOUN", "ADJ", "VERB", "DET"} <= set(expected_pos)
    assert expected_pos["NOUN"] == 2  # fox, dog
    assert expected_pos["ADJ"] == 3  # quick, brown, lazy
    assert upos.value.total == len(tokens)
    assert (upos.support.kind, upos.support.count) == ("linguistic token", len(tokens))

    # --- text.linguistic.dependency_relation ---
    expected_dep: dict[str, int] = {}
    for token in tokens:
        if token.dep_:
            expected_dep[token.dep_] = expected_dep.get(token.dep_, 0) + 1
    dep_rel = observations["text.linguistic.dependency_relation"]
    assert isinstance(dep_rel, OkFeatureObservation)
    assert isinstance(dep_rel.value, CategoricalDistributionValue)
    assert {e.key: e.count for e in dep_rel.value.counts} == expected_dep
    assert expected_dep.get("ROOT") == 1
    assert dep_rel.value.total == len(tokens)

    # --- text.linguistic.dependency_distance ---
    expected_distances = sorted(
        min(abs(token.i - token.head.i), 32)
        for token in tokens
        if token.head.i != token.i
    )
    distance = observations["text.linguistic.dependency_distance"]
    assert isinstance(distance, OkFeatureObservation)
    assert isinstance(distance.value, OrderedHistogramValue)
    assert distance.value.top_code == 32
    assert distance.value.total == len(expected_distances)
    # Total = number of non-root tokens: 10 tokens, exactly one ROOT.
    assert distance.value.total == len(tokens) - 1
    actual_points = sorted(
        point for p in distance.value.points for point in [p.point] * p.count
    )
    assert actual_points == expected_distances
    assert (distance.support.kind, distance.support.count) == (
        "non-root dependency arc",
        len(expected_distances),
    )

    # --- text.linguistic.morph_attribute ---
    expected_morph: dict[str, int] = {}
    expected_with_morph = 0
    for token in tokens:
        pairs = [
            f"{feature}={value}"
            for feature, value in token.morph.to_dict().items()
            if feature and value
        ]
        if pairs:
            expected_with_morph += 1
        for pair in pairs:
            expected_morph[pair] = expected_morph.get(pair, 0) + 1
    morph_attr = observations["text.linguistic.morph_attribute"]
    assert isinstance(morph_attr, OkFeatureObservation)
    assert isinstance(morph_attr.value, CategoricalDistributionValue)
    assert {e.key: e.count for e in morph_attr.value.counts} == expected_morph
    assert "Number=Sing" in expected_morph  # e.g. fox, dog
    assert all(re.fullmatch(r"[A-Za-z]+=[A-Za-z0-9]+(,[A-Za-z0-9]+)*", key)
               for key in expected_morph)
    # No lemma/token/dependency text leaks into category keys.
    for word in ("the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"):
        assert word not in expected_morph
    assert morph_attr.value.total == sum(expected_morph.values())
    assert (morph_attr.support.kind, morph_attr.support.count) == (
        "morphology attribute event",
        sum(expected_morph.values()),
    )

    # --- text.linguistic.morph_coverage ---
    coverage = observations["text.linguistic.morph_coverage"]
    assert isinstance(coverage, OkFeatureObservation)
    assert isinstance(coverage.value, RatioValue)
    assert coverage.value.denominator == len(tokens)
    assert coverage.value.numerator == expected_with_morph
    assert coverage.value.multiplier == 1.0
    assert coverage.value.value == expected_with_morph / len(tokens)
    assert 0.0 <= coverage.value.value <= 1.0
    assert (coverage.support.kind, coverage.support.count) == (
        "linguistic token",
        len(tokens),
    )


def test_missing_model_all_unavailable() -> None:
    output = LinguisticAnalyzer().analyze(_make_artifact(TEXT), _make_context())
    assert output.diagnostics == ()
    assert len(output.observations) == len(EXPECTED_OWNED)
    assert [obs.feature_id for obs in output.observations] == sorted(EXPECTED_OWNED)
    for observation in output.observations:
        assert observation.status == FeatureStatus.UNAVAILABLE


def test_empty_text_insufficient_support(loaded) -> None:
    nlp, backend = loaded
    output = LinguisticAnalyzer().analyze(_make_artifact(""), _make_context(nlp, backend))
    assert len(output.observations) == len(EXPECTED_OWNED)
    for observation in output.observations:
        assert observation.status == FeatureStatus.INSUFFICIENT_SUPPORT


def test_missing_parser_dependency_features_unavailable(loaded) -> None:
    nlp, backend = loaded
    assert "parser" in nlp.pipe_names  # precondition of this gating test
    with nlp.select_pipes(disable=["parser"]):
        output = LinguisticAnalyzer().analyze(
            _make_artifact(TEXT), _make_context(nlp, backend)
        )
    observations = {obs.feature_id: obs for obs in output.observations}
    assert observations["text.linguistic.dependency_relation"].status == (
        FeatureStatus.UNAVAILABLE
    )
    assert observations["text.linguistic.dependency_distance"].status == (
        FeatureStatus.UNAVAILABLE
    )
    # POS and morph come from the tagger/attribute ruler and stay available.
    assert observations["text.linguistic.upos"].status == FeatureStatus.OK
    assert observations["text.linguistic.morph_attribute"].status == FeatureStatus.OK
    assert observations["text.linguistic.morph_coverage"].status == FeatureStatus.OK
