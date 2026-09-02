"""Profiling conformance tests (spec section 13; fixtures 25.17-25.19)."""

from __future__ import annotations

import hashlib

import pytest

from stylog.analysis import compat
from stylog.analysis.profile import profile_fingerprint
from stylog.analysis.registry import FEATURES
from stylog.domain.artifact import ArtifactDescriptor, ArtifactKind, ContentIdentitySha256
from stylog.domain.baseline import (
    Baseline,
    BaselineCompatibility,
    BaselineDescriptor,
    BaselineFeature,
)
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.feature import IntegerValue, OkFeatureObservation, Support
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import (
    AnalyzerSignature,
    BackendSignature,
    RuntimeSignature,
)

MAD_FACTOR = 1.482602218505602  # normal-consistency factor (spec 13.5)

RUNTIME = RuntimeSignature(
    python_implementation="CPython",
    python_version="3.14.0",
    python_cache_tag="cpython-314",
    unicode_database_version="17.0.0",
)

WORD_COUNT = "text.lexical.word_count"

ANALYZER = AnalyzerSignature(
    analyzer_id=FEATURES[WORD_COUNT].analyzer_id,
    implementation_version="1.0.0",
    feature_registry_version="1.0.0",
    backend=BackendSignature(
        backend_id="stylog.native.text",
        implementation_version="1.0.0",
        scientific_compatibility_id="stylog.text-core/1",
    ),
)

COMPATIBILITY_SHA256 = compat.feature_compatibility_sha256(
    FEATURES[WORD_COUNT], ANALYZER, RUNTIME
)
assert COMPATIBILITY_SHA256 is not None


def make_subject(observed: int, *, semantic_version: str = "1.0.0") -> Fingerprint:
    observation = OkFeatureObservation(
        feature_id=WORD_COUNT,
        semantic_version=semantic_version,
        analyzer_id=FEATURES[WORD_COUNT].analyzer_id,
        analyzer_implementation_version="1.0.0",
        value=IntegerValue(value=observed),
        support=Support(kind="artifact", count=1),
    )
    return Fingerprint(
        artifact=ArtifactDescriptor(
            artifact_id="subject",
            kind=ArtifactKind.TEXT,
            language="en",
            encoding="utf-8",
            byte_count=10,
            character_count=10,
            content_identity=ContentIdentitySha256(
                sha256=hashlib.sha256(b"subject").hexdigest()
            ),
        ),
        runtime=RUNTIME,
        analysis_config_sha256="f" * 64,
        analyzers=(ANALYZER,),
        features=(observation,),
    )


def make_baseline(
    values: tuple[float, ...],
    *,
    feature_id: str = WORD_COUNT,
    semantic_version: str = "1.0.0",
    compatibility_sha256: str = COMPATIBILITY_SHA256,
    total_units: int | None = None,
) -> Baseline:
    return Baseline(
        baseline_id="baseline.test",
        baseline_version="1.0.0",
        descriptor=BaselineDescriptor(
            kind="text", language="en", domain="test", unit="artifact", source="test"
        ),
        source_manifest_sha256="b" * 64,
        compatibility=BaselineCompatibility(feature_registry_version="1.0.0"),
        features=(
            BaselineFeature(
                feature_id=feature_id,
                semantic_version=semantic_version,
                compatibility_sha256=compatibility_sha256,
                total_units=len(values) if total_units is None else total_units,
                values=values,
            ),
        ),
    )


def test_profile_exact_statistics():
    # Baseline 1..20 ascending, observed 10.
    baseline = make_baseline(tuple(float(v) for v in range(1, 21)))
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.diagnostics == ()
    assert len(profile.observations) == 1
    observation = profile.observations[0]
    assert observation.feature_id == WORD_COUNT
    assert observation.feature_semantic_version == "1.0.0"
    assert observation.baseline_n == 20
    assert observation.observed_value == 10.0
    # Midrank percentile: L=9 values below 10, E=1 equal, N=20 -> 100*9.5/20.
    assert observation.percentile_midrank == 47.5
    # Type-7 quantiles over 1..20.
    assert observation.median == 10.5
    assert observation.q25 == 5.75
    assert observation.q75 == 15.25
    assert observation.iqr == 9.5
    # Deviations from median 10.5 are 0.5..9.5 each twice; median deviation 5.0.
    assert observation.mad_raw == 5.0
    assert observation.mad_normal_scaled == 5.0 * MAD_FACTOR
    assert observation.robust_z == pytest.approx(-0.5 / (5.0 * MAD_FACTOR))


def test_profile_uses_all_nineteen_baseline_values():
    baseline = make_baseline(tuple(float(v) for v in range(1, 20)))  # n = 19
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.diagnostics == ()
    assert len(profile.observations) == 1
    observation = profile.observations[0]
    assert observation.baseline_n == 19
    assert observation.percentile_midrank == 50.0
    assert observation.median == 10.0
    assert observation.q25 == 5.5
    assert observation.q75 == 14.5
    assert observation.iqr == 9.0
    assert observation.mad_raw == 5.0
    assert observation.robust_z == 0.0


def test_empty_baseline_distribution_is_not_profiled():
    baseline = make_baseline((), total_units=3)
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.observations == ()
    assert len(profile.diagnostics) == 1
    diagnostic = profile.diagnostics[0]
    assert diagnostic.code == "BASELINE_INSUFFICIENT_SUPPORT"
    assert diagnostic.severity == DiagnosticSeverity.WARNING
    assert diagnostic.feature_id == WORD_COUNT


def test_zero_mad_observed_equals_median():
    baseline = make_baseline((7.0,) * 20)
    profile = profile_fingerprint(make_subject(7), baseline, "subject")
    assert len(profile.observations) == 1
    observation = profile.observations[0]
    assert observation.median == 7.0
    assert observation.iqr == 0.0
    assert observation.mad_raw == 0.0
    assert observation.mad_normal_scaled == 0.0
    # L=0, E=20, N=20 -> 50.
    assert observation.percentile_midrank == 50.0
    # robust_z omitted under the zero-MAD rule; never infinity.
    assert observation.robust_z is None
    assert [diagnostic.code for diagnostic in profile.diagnostics] == ["PROFILE_ZERO_MAD"]
    assert profile.diagnostics[0].severity == DiagnosticSeverity.INFO
    assert profile.diagnostics[0].feature_id == WORD_COUNT


def test_zero_mad_observed_differs_from_median():
    baseline = make_baseline((7.0,))
    profile = profile_fingerprint(make_subject(9), baseline, "subject")
    assert len(profile.observations) == 1
    observation = profile.observations[0]
    assert observation.baseline_n == 1
    # L=1, E=0, N=1 -> 100.
    assert observation.percentile_midrank == 100.0
    assert observation.robust_z is None
    assert [diagnostic.code for diagnostic in profile.diagnostics] == ["PROFILE_ZERO_MAD"]


def test_incompatible_compatibility_sha256():
    baseline = make_baseline(
        tuple(float(v) for v in range(1, 21)), compatibility_sha256="0" * 64
    )
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.observations == ()
    assert [diagnostic.code for diagnostic in profile.diagnostics] == [
        "BASELINE_INCOMPATIBLE"
    ]
    assert profile.diagnostics[0].severity == DiagnosticSeverity.WARNING
    assert profile.diagnostics[0].feature_id == WORD_COUNT


def test_semantic_version_mismatch_is_incompatible():
    baseline = make_baseline(
        tuple(float(v) for v in range(1, 21)), semantic_version="9.9.9"
    )
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.observations == ()
    assert [diagnostic.code for diagnostic in profile.diagnostics] == [
        "BASELINE_INCOMPATIBLE"
    ]


def test_unknown_registry_feature_is_incompatible():
    baseline = make_baseline(
        tuple(float(v) for v in range(1, 21)), feature_id="text.bogus.feature"
    )
    profile = profile_fingerprint(make_subject(10), baseline, "subject")
    assert profile.observations == ()
    assert [diagnostic.code for diagnostic in profile.diagnostics] == [
        "BASELINE_INCOMPATIBLE"
    ]
    assert profile.diagnostics[0].feature_id == "text.bogus.feature"


def test_feature_absent_from_subject_is_omitted_silently():
    baseline = make_baseline(tuple(float(v) for v in range(1, 21)))
    subject = make_subject(10).model_copy(update={"features": ()})
    profile = profile_fingerprint(subject, baseline, "subject")
    assert profile.observations == ()
    assert profile.diagnostics == ()
