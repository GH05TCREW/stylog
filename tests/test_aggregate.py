"""Aggregation conformance tests (spec section 11; fixtures 25.13-25.14)."""

from __future__ import annotations

import hashlib
import math

import pytest

from stylog.analysis import build
from stylog.analysis.aggregate import aggregate_fingerprints
from stylog.analysis.registry import (
    FEATURES,
    FUNCTION_WORDS_EN_RESOURCE_ID,
    FUNCTION_WORDS_EN_RESOURCE_VERSION,
    FUNCTION_WORDS_EN_SHA256,
)
from stylog.domain.artifact import ArtifactDescriptor, ArtifactKind, ContentIdentitySha256
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.evidence import (
    EvidenceMember,
    EvidenceSet,
    LinkageDescriptor,
    MissingStatusCount,
)
from stylog.domain.feature import (
    DisabledObservation,
    FeatureStatus,
    IntegerValue,
    OkFeatureObservation,
    ParserErrorObservation,
    RatioValue,
    Support,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import (
    AnalyzerSignature,
    BackendSignature,
    ResourceSignature,
    RuntimeSignature,
)

RUNTIME = RuntimeSignature(
    python_implementation="CPython",
    python_version="3.14.0",
    python_cache_tag="cpython-314",
    unicode_database_version="17.0.0",
)
OTHER_RUNTIME = RuntimeSignature(
    python_implementation="CPython",
    python_version="3.13.0",
    python_cache_tag="cpython-313",
    unicode_database_version="16.0.0",
)

FW_SHARE = "text.function_words.en.token_share"
TOKEN_KIND = "text.lexical.token_kind"
WORD_LENGTH = "text.lexical.word_length"
TTR = "text.lexical.ttr_casefold"
WORD_COUNT = "text.lexical.word_count"
WINDOW_TTR = "text.lexical.window_ttr_100"

FW_RESOURCE = ResourceSignature(
    id=FUNCTION_WORDS_EN_RESOURCE_ID,
    version=FUNCTION_WORDS_EN_RESOURCE_VERSION,
    sha256=FUNCTION_WORDS_EN_SHA256,
)


def make_analyzer(
    analyzer_id: str,
    *,
    resources: tuple[ResourceSignature, ...] = (),
    backend_compat: str = "stylog.text-core/1",
) -> AnalyzerSignature:
    return AnalyzerSignature(
        analyzer_id=analyzer_id,
        implementation_version="1.0.0",
        feature_registry_version="1.0.0",
        backend=BackendSignature(
            backend_id="stylog.native.text",
            implementation_version="1.0.0",
            scientific_compatibility_id=backend_compat,
        ),
        resources=resources,
    )


def ok_obs(
    feature_id: str,
    value,
    *,
    support_count: int = 10,
    semantic_version: str = "1.0.0",
) -> OkFeatureObservation:
    fdef = FEATURES[feature_id]
    return OkFeatureObservation(
        feature_id=feature_id,
        semantic_version=semantic_version,
        analyzer_id=fdef.analyzer_id,
        analyzer_implementation_version="1.0.0",
        value=value,
        support=Support(kind=fdef.support_kind, count=support_count),
    )


def status_obs(feature_id: str, status: str):
    fdef = FEATURES[feature_id]
    cls = {"parser_error": ParserErrorObservation, "disabled": DisabledObservation}[status]
    return cls(
        feature_id=feature_id,
        semantic_version="1.0.0",
        analyzer_id=fdef.analyzer_id,
        analyzer_implementation_version="1.0.0",
    )


def make_fingerprint(
    artifact_id: str,
    *,
    features=(),
    analyzers=(),
    runtime: RuntimeSignature = RUNTIME,
    content_sha256: str | None = None,
) -> Fingerprint:
    return Fingerprint(
        artifact=ArtifactDescriptor(
            artifact_id=artifact_id,
            kind=ArtifactKind.TEXT,
            language="en",
            encoding="utf-8",
            byte_count=10,
            character_count=10,
            content_identity=ContentIdentitySha256(
                sha256=content_sha256 or hashlib.sha256(artifact_id.encode()).hexdigest()
            ),
        ),
        runtime=runtime,
        analysis_config_sha256="f" * 64,
        analyzers=tuple(sorted(analyzers, key=lambda analyzer: analyzer.analyzer_id)),
        features=tuple(sorted(features, key=lambda observation: observation.feature_id)),
    )


def make_evidence_set(fps, evidence_set_id: str = "es1") -> EvidenceSet:
    return EvidenceSet(
        evidence_set_id=evidence_set_id,
        members=tuple(
            EvidenceMember(member_id=f"m{index + 1}", artifact_id=fp.artifact.artifact_id)
            for index, fp in enumerate(fps)
        ),
        linkage=LinkageDescriptor(kind="manual", source="test"),
    )


def fw_fingerprint(artifact_id: str, numerator: int, denominator: int) -> Fingerprint:
    return make_fingerprint(
        artifact_id,
        features=[ok_obs(FW_SHARE, build.ratio_value(numerator, denominator))],
        analyzers=[make_analyzer(FEATURES[FW_SHARE].analyzer_id, resources=(FW_RESOURCE,))],
    )


def aggregate_of(feature_id: str, result):
    matches = [aggregate for aggregate in result.aggregates if aggregate.feature_id == feature_id]
    assert len(matches) == 1
    return matches[0]


def test_ratio_pooling_fixture():
    # Spec 25.13: 1/10 + 9/90 -> pooled 10/100 = 0.1.
    fp1 = fw_fingerprint("a1", 1, 10)
    fp2 = fw_fingerprint("a2", 9, 90)
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.diagnostics == ()
    aggregate = aggregate_of(FW_SHARE, result)
    assert isinstance(aggregate.pooled, RatioValue)
    assert aggregate.pooled.numerator == 10
    assert aggregate.pooled.denominator == 100
    assert aggregate.pooled.value == 0.1
    assert aggregate.total_samples == 2
    assert aggregate.contributing_samples == 2
    assert aggregate.missing == ()
    assert aggregate.semantic_version == "1.0.0"


def test_ratio_pooling_differs_from_unweighted_mean():
    # Counterexample: 1/2 + 0/100 pools to 1/102, not the unweighted mean 0.25.
    fp1 = fw_fingerprint("a1", 1, 2)
    fp2 = fw_fingerprint("a2", 0, 100)
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(FW_SHARE, result)
    assert isinstance(aggregate.pooled, RatioValue)
    assert (aggregate.pooled.numerator, aggregate.pooled.denominator) == (1, 102)
    assert aggregate.pooled.value == pytest.approx(1 / 102)
    assert aggregate.pooled.value != pytest.approx(0.25)


def test_missing_statuses_counted_individually():
    # Spec 25.14: ok + parser_error + disabled -> total=3, contributing=1.
    fp_ok = fw_fingerprint("a1", 1, 10)
    analyzer = make_analyzer(FEATURES[FW_SHARE].analyzer_id, resources=(FW_RESOURCE,))
    fp_error = make_fingerprint(
        "a2", features=[status_obs(FW_SHARE, "parser_error")], analyzers=[analyzer]
    )
    fp_disabled = make_fingerprint(
        "a3", features=[status_obs(FW_SHARE, "disabled")], analyzers=[analyzer]
    )
    fps = [fp_ok, fp_error, fp_disabled]
    result = aggregate_fingerprints(make_evidence_set(fps), fps)
    aggregate = aggregate_of(FW_SHARE, result)
    assert aggregate.total_samples == 3
    assert aggregate.contributing_samples == 1
    assert aggregate.missing == (
        MissingStatusCount(status=FeatureStatus.DISABLED, count=1),
        MissingStatusCount(status=FeatureStatus.PARSER_ERROR, count=1),
    )
    assert isinstance(aggregate.pooled, RatioValue)
    assert (aggregate.pooled.numerator, aggregate.pooled.denominator) == (1, 10)


def test_absent_feature_counts_as_unavailable():
    fp_ok = fw_fingerprint("a1", 1, 10)
    fp_empty = make_fingerprint("a2")
    fps = [fp_ok, fp_empty]
    result = aggregate_fingerprints(make_evidence_set(fps), fps)
    aggregate = aggregate_of(FW_SHARE, result)
    assert aggregate.total_samples == 2
    assert aggregate.contributing_samples == 1
    assert aggregate.missing == (
        MissingStatusCount(status=FeatureStatus.UNAVAILABLE, count=1),
    )


def test_zero_contributing_samples_has_no_pooled_value():
    analyzer = make_analyzer(FEATURES[FW_SHARE].analyzer_id, resources=(FW_RESOURCE,))
    fp1 = make_fingerprint(
        "a1", features=[status_obs(FW_SHARE, "parser_error")], analyzers=[analyzer]
    )
    fp2 = make_fingerprint(
        "a2", features=[status_obs(FW_SHARE, "parser_error")], analyzers=[analyzer]
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(FW_SHARE, result)
    assert aggregate.contributing_samples == 0
    assert aggregate.pooled is None
    assert aggregate.sample_summary is None
    assert aggregate.sample_values is None
    assert aggregate.missing == (
        MissingStatusCount(status=FeatureStatus.PARSER_ERROR, count=2),
    )


def test_semantic_version_mismatch_omits_feature():
    analyzer = make_analyzer(FEATURES[FW_SHARE].analyzer_id, resources=(FW_RESOURCE,))
    fp1 = make_fingerprint(
        "a1",
        features=[ok_obs(FW_SHARE, build.ratio_value(1, 10), semantic_version="1.0.0")],
        analyzers=[analyzer],
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(FW_SHARE, build.ratio_value(9, 90), semantic_version="9.9.9")],
        analyzers=[analyzer],
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.aggregates == ()
    assert len(result.diagnostics) == 1
    diagnostic = result.diagnostics[0]
    assert diagnostic.code == "FEATURE_SEMANTIC_MISMATCH"
    assert diagnostic.severity == DiagnosticSeverity.ERROR
    assert diagnostic.feature_id == FW_SHARE


def test_resource_signature_mismatch_omits_feature():
    fp1 = fw_fingerprint("a1", 1, 10)
    other_resource = ResourceSignature(
        id=FUNCTION_WORDS_EN_RESOURCE_ID,
        version=FUNCTION_WORDS_EN_RESOURCE_VERSION,
        sha256="0" * 64,
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(FW_SHARE, build.ratio_value(9, 90))],
        analyzers=[
            make_analyzer(FEATURES[FW_SHARE].analyzer_id, resources=(other_resource,))
        ],
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.aggregates == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RESOURCE_SIGNATURE_MISMATCH"
    ]
    assert result.diagnostics[0].severity == DiagnosticSeverity.ERROR


def test_runtime_signature_mismatch_omits_feature():
    analyzer = make_analyzer(FEATURES[TTR].analyzer_id)
    fp1 = make_fingerprint(
        "a1", features=[ok_obs(TTR, build.ratio_value(1, 2))], analyzers=[analyzer]
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(TTR, build.ratio_value(3, 4))],
        analyzers=[analyzer],
        runtime=OTHER_RUNTIME,
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.aggregates == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "RUNTIME_SIGNATURE_MISMATCH"
    ]


def test_backend_compatibility_mismatch_omits_feature():
    analyzer1 = make_analyzer(FEATURES[TTR].analyzer_id, backend_compat="stylog.text-core/1")
    analyzer2 = make_analyzer(FEATURES[TTR].analyzer_id, backend_compat="stylog.text-core/2")
    fp1 = make_fingerprint(
        "a1", features=[ok_obs(TTR, build.ratio_value(1, 2))], analyzers=[analyzer1]
    )
    fp2 = make_fingerprint(
        "a2", features=[ok_obs(TTR, build.ratio_value(3, 4))], analyzers=[analyzer2]
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.aggregates == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "FEATURE_SEMANTIC_MISMATCH"
    ]


def test_categorical_pooling_sums_per_key():
    analyzer = make_analyzer(FEATURES[TOKEN_KIND].analyzer_id)
    fp1 = make_fingerprint(
        "a1",
        features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3, "number": 1}))],
        analyzers=[analyzer],
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 1, "punctuation": 2}))],
        analyzers=[analyzer],
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(TOKEN_KIND, result)
    assert {entry.key: entry.count for entry in aggregate.pooled.counts} == {
        "word": 4,
        "number": 1,
        "punctuation": 2,
    }
    assert aggregate.pooled.total == 7


def test_histogram_pooling_sums_per_point():
    analyzer = make_analyzer(FEATURES[WORD_LENGTH].analyzer_id)
    fp1 = make_fingerprint(
        "a1",
        features=[ok_obs(WORD_LENGTH, build.histogram_value([1, 2, 2], 31))],
        analyzers=[analyzer],
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(WORD_LENGTH, build.histogram_value([2, 31], 31))],
        analyzers=[analyzer],
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(WORD_LENGTH, result)
    assert {entry.point: entry.count for entry in aggregate.pooled.points} == {
        1: 1,
        2: 3,
        31: 1,
    }
    assert aggregate.pooled.total == 5
    assert aggregate.pooled.top_code == 31


def test_histogram_top_code_mismatch_omits_feature():
    analyzer = make_analyzer(FEATURES[WORD_LENGTH].analyzer_id)
    fp1 = make_fingerprint(
        "a1",
        features=[ok_obs(WORD_LENGTH, build.histogram_value([1, 2], 31))],
        analyzers=[analyzer],
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(WORD_LENGTH, build.histogram_value([1, 2], 30))],
        analyzers=[analyzer],
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    assert result.aggregates == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "FEATURE_SEMANTIC_MISMATCH"
    ]
    assert result.diagnostics[0].feature_id == WORD_LENGTH


def test_sample_summary_reducer():
    analyzer = make_analyzer(FEATURES[TTR].analyzer_id)
    fp1 = make_fingerprint(
        "a1", features=[ok_obs(TTR, build.ratio_value(1, 2))], analyzers=[analyzer]
    )
    fp2 = make_fingerprint(
        "a2", features=[ok_obs(TTR, build.ratio_value(3, 4))], analyzers=[analyzer]
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(TTR, result)
    assert aggregate.pooled is None
    assert aggregate.sample_values == (0.5, 0.75)
    summary = aggregate.sample_summary
    assert summary is not None
    assert summary.n == 2
    assert summary.minimum == 0.5
    assert summary.maximum == 0.75
    assert summary.mean == 0.625
    assert summary.median == 0.625
    assert summary.q25 == 0.5625
    assert summary.q75 == 0.6875
    assert summary.sample_sd == pytest.approx(math.sqrt(0.03125))


def test_sample_summary_single_sample_omits_sd():
    analyzer = make_analyzer(FEATURES[TTR].analyzer_id)
    fp = make_fingerprint(
        "a1", features=[ok_obs(TTR, build.ratio_value(1, 2))], analyzers=[analyzer]
    )
    result = aggregate_fingerprints(make_evidence_set([fp]), [fp])
    aggregate = aggregate_of(TTR, result)
    assert aggregate.sample_values == (0.5,)
    summary = aggregate.sample_summary
    assert summary is not None
    assert summary.n == 1
    assert summary.sample_sd is None
    assert summary.median == 0.5


def test_not_aggregatable_reducer_has_no_values():
    analyzer = make_analyzer(FEATURES[WINDOW_TTR].analyzer_id)
    fp = make_fingerprint(
        "a1",
        features=[ok_obs(WINDOW_TTR, build.summary_value([0.7, 0.8]))],
        analyzers=[analyzer],
    )
    result = aggregate_fingerprints(make_evidence_set([fp]), [fp])
    aggregate = aggregate_of(WINDOW_TTR, result)
    assert aggregate.contributing_samples == 1
    assert aggregate.pooled is None
    assert aggregate.sample_summary is None
    assert aggregate.sample_values is None


def test_duplicate_content_members_remain_two_samples():
    # Spec 11.12: identical content hashes remain two samples; no dedup.
    shared_sha = "d" * 64
    analyzer = make_analyzer(FEATURES[WORD_COUNT].analyzer_id)
    fp1 = make_fingerprint(
        "a1",
        features=[ok_obs(WORD_COUNT, IntegerValue(value=5))],
        analyzers=[analyzer],
        content_sha256=shared_sha,
    )
    fp2 = make_fingerprint(
        "a2",
        features=[ok_obs(WORD_COUNT, IntegerValue(value=5))],
        analyzers=[analyzer],
        content_sha256=shared_sha,
    )
    result = aggregate_fingerprints(make_evidence_set([fp1, fp2]), [fp1, fp2])
    aggregate = aggregate_of(WORD_COUNT, result)
    assert aggregate.total_samples == 2
    assert aggregate.contributing_samples == 2
    assert isinstance(aggregate.pooled, IntegerValue)
    assert aggregate.pooled.value == 10


def test_features_outside_registry_are_not_aggregated():
    bogus = OkFeatureObservation(
        feature_id="text.bogus.feature",
        semantic_version="1.0.0",
        analyzer_id="stylog.text.lexical",
        analyzer_implementation_version="1.0.0",
        value=IntegerValue(value=1),
        support=Support(kind="artifact", count=1),
    )
    fp = make_fingerprint(
        "a1",
        features=[ok_obs(WORD_COUNT, IntegerValue(value=3)), bogus],
        analyzers=[make_analyzer(FEATURES[WORD_COUNT].analyzer_id)],
    )
    result = aggregate_fingerprints(make_evidence_set([fp]), [fp])
    assert [aggregate.feature_id for aggregate in result.aggregates] == [WORD_COUNT]
