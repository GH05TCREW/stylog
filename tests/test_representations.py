"""Tests for the 'ml' capability: scikit-learn Representations (spec 20.7).

Guarded by pytest.importorskip so base installs without sklearn skip
cleanly. The CapabilityUnavailableError path (sklearn missing) is covered
by the CLI tests instead.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

pytest.importorskip("sklearn")

from stylog.domain.representation import Representation, RepresentationFit
from stylog.exceptions import ConfigurationError, PortableArtifactError, ResourceError
from stylog.infrastructure.ingest import artifact_from_text
from stylog.representations.fit import (
    fit_representation,
    fit_representation_cli,
    transform_representation,
    transform_representation_cli,
)
from stylog.representations.spec import (
    PREPROCESSING_VERSION,
    SCIENTIFIC_COMPATIBILITY_ID,
    SEMANTIC_VERSION,
    SPECS,
    representation_spec,
)
from stylog.serialization.canonical import canonical_bytes, file_bytes
from stylog.serialization.jsonio import model_from_bytes

ALL_KINDS = tuple(SPECS)


@pytest.fixture(autouse=True)
def fits_dir(tmp_path, monkeypatch):
    """Hermetic local fit-state store per test."""
    root = tmp_path / "fits"
    monkeypatch.setenv("STYLOG_FITS_DIR", str(root))
    return root


def _artifact(artifact_id: str, text: str):
    return artifact_from_text(text, artifact_id=artifact_id)


def _corpus():
    return [
        _artifact("subj-b", "The cat sat on the mat."),
        _artifact("subj-a", "Don't stop chasing the zebra."),
        _artifact("subj-c", "A zebra never changes its stripes."),
    ]


def _state(root, fit: RepresentationFit) -> dict:
    path = root / f"{fit.state_resource.sha256}.json"
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and not raw[:-1].endswith(b"\n")
    return json.loads(raw[:-1].decode("utf-8"))


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def test_spec_resolution_accepts_tokens_and_ids():
    spec = SPECS["char_tfidf"]
    assert representation_spec("char-tfidf") is spec
    assert representation_spec("char_tfidf") is spec
    assert representation_spec("stylog.representation.char_tfidf") is spec
    assert representation_spec("stylog.representation.char_tfidf/1") is spec
    assert representation_spec(spec) is spec


def test_spec_identities_and_params():
    for kind, spec in SPECS.items():
        assert spec.representation_id == f"stylog.representation.{kind}/1"
        assert spec.semantic_version == "1.0.0"
        assert spec.short_id == f"stylog.representation.{kind}"
        assert spec.cli_token == kind.replace("_", "-")
        assert spec.state_resource_id == f"stylog.representation.state/{kind}"
    char = SPECS["char_ngram_count"].params
    assert char["analyzer"] == "char"
    assert char["ngram_range"] == [3, 5]
    assert char["lowercase"] is False
    assert char["strip_accents"] == "none"
    assert char["preprocessor"] == "none"
    word = SPECS["word_ngram_count"].params
    assert word["analyzer"] == "word"
    assert word["ngram_range"] == [1, 3]
    assert word["tokenizer"] == "identity"
    assert word["token_pattern"] == "none"
    tfidf = SPECS["word_tfidf"].params
    assert (tfidf["use_idf"], tfidf["smooth_idf"], tfidf["sublinear_tf"]) == (True, True, False)
    assert tfidf["norm"] == "l2"
    for spec in SPECS.values():
        assert spec.params["stop_words"] == "none"
        assert spec.params["min_df"] == 1
        assert spec.params["max_df"] == 1.0
        assert spec.params["max_features"] == "none"
        assert spec.params["binary"] is False


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def test_fit_determinism(fits_dir):
    spec = representation_spec("stylog.representation.char_tfidf/1")
    first = fit_representation(spec, _corpus())
    second = fit_representation(spec, _corpus())
    assert first == second
    assert first.fit_id == second.fit_id == first.state_resource.sha256


def test_fit_corpus_order_invariance(fits_dir):
    spec = representation_spec("word-ngram-count")
    reference = fit_representation(spec, _corpus())
    shuffled = fit_representation(spec, [_corpus()[2], _corpus()[0], _corpus()[1]])
    reversed_order = fit_representation(spec, list(reversed(_corpus())))
    assert reference.fit_id == shuffled.fit_id == reversed_order.fit_id
    assert reference.source_manifest_sha256 == shuffled.source_manifest_sha256


def test_fit_provenance_and_identity(fits_dir):
    fit = fit_representation("char-ngram-count", _corpus())
    assert fit.representation_id == "stylog.representation.char_ngram_count/1"
    assert fit.representation_semantic_version == SEMANTIC_VERSION
    assert fit.state_resource.resource_id == "stylog.representation.state/char_ngram_count"
    assert fit.state_resource.resource_version == "1.0.0"
    backend = fit.backend
    assert backend.backend_id == "scikit-learn"
    assert backend.scientific_compatibility_id == SCIENTIFIC_COMPATIBILITY_ID
    assert [p.package for p in backend.packages] == ["numpy", "scikit-learn", "scipy"]
    assert backend.implementation_version


def test_fit_roundtrip_and_validation(fits_dir):
    fit = fit_representation("word-tfidf", _corpus())
    parsed = model_from_bytes(file_bytes(fit), RepresentationFit)
    assert parsed == fit
    with pytest.raises(PortableArtifactError):
        model_from_bytes(b'{"schema": "stylog.representation-fit"}', RepresentationFit)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_transform_determinism(fits_dir):
    fit = fit_representation("char-tfidf", _corpus())
    subject = _corpus()[0]
    first = transform_representation(fit, subject)
    second = transform_representation(fit, subject)
    assert first == second
    assert canonical_bytes(first) == canonical_bytes(second)


def test_sparse_vector_invariants(fits_dir):
    for kind in ALL_KINDS:
        fit = fit_representation(SPECS[kind], _corpus())
        state = _state(fits_dir, fit)
        for artifact in _corpus():
            rep = transform_representation(fit, artifact)
            value = rep.value
            assert value.dimension == len(state["vocabulary"])
            indices = [entry.index for entry in value.entries]
            assert indices == sorted(indices)
            assert len(set(indices)) == len(indices)
            assert all(entry.value != 0.0 for entry in value.entries)
            assert all(0 <= entry.index < value.dimension for entry in value.entries)


def test_transform_identity_fields(fits_dir):
    fit = fit_representation("word-ngram-count", _corpus())
    subject = _corpus()[1]
    rep = transform_representation(fit, subject)
    assert rep.subject_ref == subject.artifact_id
    assert rep.representation_id == "stylog.representation.word_ngram_count/1"
    assert rep.semantic_version == "1.0.0"
    assert rep.preprocessing_version == PREPROCESSING_VERSION
    assert rep.fit_id == fit.fit_id
    assert rep.resources == (fit.state_resource,)
    override = transform_representation(fit, subject, subject_ref="custom")
    assert override.subject_ref == "custom"
    as_text = transform_representation(fit, subject.text)
    assert as_text.subject_ref == "text"
    assert as_text.value == rep.value


def test_transform_with_spec_requires_fit(fits_dir):
    spec = representation_spec("char-tfidf")
    with pytest.raises(ResourceError):
        transform_representation(spec, _corpus()[0])


def test_transform_missing_state_raises_resource_error(fits_dir):
    fit = fit_representation("char-tfidf", _corpus())
    (fits_dir / f"{fit.state_resource.sha256}.json").unlink()
    with pytest.raises(ResourceError):
        transform_representation(fit, _corpus()[0])


def test_transform_rejects_tampered_state(fits_dir):
    fit = fit_representation("word-tfidf", _corpus())
    state_path = fits_dir / f"{fit.state_resource.sha256}.json"
    state = _state(fits_dir, fit)
    state["vocabulary"] = list(reversed(state["vocabulary"]))
    state_path.write_bytes((json.dumps(state) + "\n").encode("utf-8"))
    with pytest.raises(ResourceError):
        transform_representation(fit, _corpus()[0])


# ---------------------------------------------------------------------------
# Semantics
# ---------------------------------------------------------------------------


def test_canonical_vocabulary_order(fits_dir):
    fit = fit_representation("word-ngram-count", _corpus())
    vocabulary = _state(fits_dir, fit)["vocabulary"]
    assert vocabulary == sorted(vocabulary)


def test_char_ngram_semantics(fits_dir):
    # "aaaa" with ngram_range (3,5): 3-grams "aaa" x2, 4-gram "aaaa" x1,
    # no 5-grams. Canonical order: "aaa" < "aaaa".
    fit = fit_representation("char-ngram-count", [_artifact("train", "aaaa")])
    assert _state(fits_dir, fit)["vocabulary"] == ["aaa", "aaaa"]
    rep = transform_representation(fit, _artifact("probe", "aaaa"))
    assert [(e.index, e.value) for e in rep.value.entries] == [(0, 2.0), (1, 1.0)]


def test_char_representation_preserves_case(fits_dir):
    fit = fit_representation("char-ngram-count", [_artifact("train", "Aaa")])
    assert _state(fits_dir, fit)["vocabulary"] == ["Aaa"]


def test_word_ngrams_use_stylog_word_tokens_casefolded(fits_dir):
    fit = fit_representation("word-ngram-count", [_artifact("train", "Don't stop")])
    vocabulary = _state(fits_dir, fit)["vocabulary"]
    assert vocabulary == ["don't", "don't stop", "stop"]  # apostrophe kept, casefolded
    rep = transform_representation(fit, _artifact("probe", "DON'T STOP"))
    assert [(e.index, e.value) for e in rep.value.entries] == [(0, 1.0), (1, 1.0), (2, 1.0)]


def test_word_representation_ignores_numbers(fits_dir):
    fit = fit_representation("word-ngram-count", [_artifact("train", "value 3.14 words")])
    assert _state(fits_dir, fit)["vocabulary"] == ["value", "value words", "words"]


def test_tfidf_l2_norm_and_smooth_idf(fits_dir):
    corpus = [_artifact("d1", "apple banana"), _artifact("d2", "banana cherry")]
    fit = fit_representation("word-tfidf", corpus)
    vocabulary = _state(fits_dir, fit)["vocabulary"]
    assert vocabulary == ["apple", "apple banana", "banana", "banana cherry", "cherry"]
    rep = transform_representation(fit, corpus[0])

    # Hand computation: smooth_idf idf(t) = ln((n + 1) / (df + 1)) + 1, n = 2.
    idf_rare = math.log(3 / 2) + 1.0  # apple, "apple banana": df 1
    idf_banana = math.log(3 / 3) + 1.0  # banana: df 2
    norm = math.sqrt(2 * idf_rare**2 + idf_banana**2)
    expected = {
        "apple": idf_rare / norm,
        "apple banana": idf_rare / norm,
        "banana": idf_banana / norm,
    }
    actual = {vocabulary[e.index]: e.value for e in rep.value.entries}
    assert set(actual) == set(expected)
    for term, value in expected.items():
        assert actual[term] == pytest.approx(value, rel=1e-12)
    l2 = math.sqrt(math.fsum(e.value**2 for e in rep.value.entries))
    assert l2 == pytest.approx(1.0, rel=1e-12)


def test_count_representation_state_has_no_idf(fits_dir):
    count_fit = fit_representation("char-ngram-count", _corpus())
    assert "idf" not in _state(fits_dir, count_fit)
    tfidf_fit = fit_representation("char-tfidf", _corpus())
    tfidf_state = _state(fits_dir, tfidf_fit)
    assert len(tfidf_state["idf"]) == len(tfidf_state["vocabulary"])


def test_portable_representation_contains_no_vocabulary_terms(fits_dir):
    corpus = _corpus()  # contains the distinctive term "zebra"
    for kind in ALL_KINDS:
        fit = fit_representation(SPECS[kind], corpus)
        rep = transform_representation(fit, corpus[1])
        payload = canonical_bytes(rep)
        assert b"zebra" not in payload
        assert b"don't" not in payload
        assert b"stripes" not in payload


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_cli_helpers(fits_dir, tmp_path):
    artifacts = _corpus()
    fit_path = tmp_path / "fit.json"
    fit = fit_representation_cli("char-tfidf", artifacts, str(fit_path))
    assert fit_path.is_file()
    assert model_from_bytes(fit_path.read_bytes(), RepresentationFit) == fit
    with pytest.raises(PortableArtifactError):
        fit_representation_cli("char-tfidf", artifacts, str(fit_path))  # no overwrite

    reps = transform_representation_cli(str(fit_path), artifacts)
    assert len(reps) == len(artifacts)
    assert all(isinstance(rep, Representation) for rep in reps)
    assert [rep.subject_ref for rep in reps] == [a.artifact_id for a in artifacts]
    assert all(rep.fit_id == fit.fit_id for rep in reps)


def test_cli_helper_accepts_full_and_unsuffixed_ids(fits_dir, tmp_path):
    artifacts = [_artifact("one", "aaaa bbbb")]
    full = fit_representation_cli(
        "stylog.representation.word_ngram_count/1", artifacts, tmp_path / "a.json"
    )
    unsuffixed = fit_representation_cli(
        "stylog.representation.word_ngram_count", artifacts, tmp_path / "b.json"
    )
    assert full.fit_id == unsuffixed.fit_id


def test_transform_bare_string_id_is_typed_error():
    """A bare representation id string is fit-free usage: typed, never AttributeError."""
    with pytest.raises(ResourceError):
        transform_representation("char_tfidf", "some text")
    with pytest.raises(ConfigurationError):
        transform_representation("no_such_representation", "some text")


def test_fit_representation_cli_force_overwrites(fits_dir, tmp_path):
    artifacts = _corpus()
    fit_path = tmp_path / "fit.json"
    first = fit_representation_cli("char-tfidf", artifacts, str(fit_path))
    with pytest.raises(PortableArtifactError):
        fit_representation_cli("char-tfidf", artifacts, str(fit_path))  # no overwrite
    second = fit_representation_cli("char-tfidf", artifacts, str(fit_path), force=True)
    assert second == first
    assert model_from_bytes(fit_path.read_bytes(), RepresentationFit) == second


# ---------------------------------------------------------------------------
# Release gate: lazy optional imports
# ---------------------------------------------------------------------------


def test_no_optional_imports_at_module_level():
    code = (
        "import sys; "
        "import stylog.representations, stylog.representations.spec, "
        "stylog.representations.fit; "
        "bad = [m for m in ('sklearn', 'scipy', 'numpy') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_transform_many_matches_single_transforms():
    """Bulk transform is semantically identical to per-item transforms."""
    from stylog.representations.fit import transform_many

    artifacts = [
        artifact_from_text(f"bulk transform sample number {i}", artifact_id=f"m{i}")
        for i in range(30)
    ]
    fit = fit_representation(SPECS["word_ngram_count"], artifacts[:20])
    singles = [transform_representation(fit, artifact) for artifact in artifacts[20:]]
    bulk = transform_many(fit, artifacts[20:])
    assert singles == bulk
