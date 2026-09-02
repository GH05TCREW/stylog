"""Comparison conformance tests (spec section 12; fixtures 25.15-25.16)."""

from __future__ import annotations

import hashlib
import math

import pytest

from stylog.analysis import build
from stylog.analysis.aggregate import aggregate_fingerprints
from stylog.analysis.compare import compare_aggregates, compare_fingerprints
from stylog.analysis.registry import FEATURES
from stylog.domain.artifact import ArtifactDescriptor, ArtifactKind, ContentIdentitySha256
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.evidence import EvidenceMember, EvidenceSet, LinkageDescriptor
from stylog.domain.feature import (
    FeatureStatus,
    IntegerValue,
    OkFeatureObservation,
    Support,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Comparison
from stylog.domain.provenance import (
    AnalyzerSignature,
    BackendSignature,
    RuntimeSignature,
)
from stylog.exceptions import PortableArtifactError

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

MORPH = "text.linguistic.morph_coverage"  # ratio, RATIO_POOL, ABS
TOKEN_KIND = "text.lexical.token_kind"  # categorical, JSD2
WORD_LENGTH = "text.lexical.word_length"  # histogram, W1, top_code 31
WORD_COUNT = "text.lexical.word_count"  # integer, metric NONE
TTR = "text.lexical.ttr_casefold"  # ratio, SAMPLE_SUMMARY, ABS, runtime-sensitive


def make_analyzer(analyzer_id: str) -> AnalyzerSignature:
    return AnalyzerSignature(
        analyzer_id=analyzer_id,
        implementation_version="1.0.0",
        feature_registry_version="1.0.0",
        backend=BackendSignature(
            backend_id="stylog.native.text",
            implementation_version="1.0.0",
            scientific_compatibility_id="stylog.text-core/1",
        ),
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


def status_obs(feature_id: str, status: FeatureStatus):
    fdef = FEATURES[feature_id]
    return build.status(fdef, fdef.analyzer_id, "1.0.0", status)


def make_fingerprint(
    artifact_id: str,
    *,
    kind: ArtifactKind = ArtifactKind.TEXT,
    language: str = "en",
    features=(),
    runtime: RuntimeSignature = RUNTIME,
) -> Fingerprint:
    analyzer_ids = {FEATURES[obs.feature_id].analyzer_id for obs in features}
    return Fingerprint(
        artifact=ArtifactDescriptor(
            artifact_id=artifact_id,
            kind=kind,
            language=language,
            encoding="utf-8",
            byte_count=10,
            character_count=10,
            content_identity=ContentIdentitySha256(
                sha256=hashlib.sha256(artifact_id.encode()).hexdigest()
            ),
        ),
        runtime=runtime,
        analysis_config_sha256="f" * 64,
        analyzers=tuple(make_analyzer(analyzer_id) for analyzer_id in sorted(analyzer_ids)),
        features=tuple(sorted(features, key=lambda observation: observation.feature_id)),
    )


def make_evidence_set(fps, evidence_set_id: str) -> EvidenceSet:
    return EvidenceSet(
        evidence_set_id=evidence_set_id,
        members=tuple(
            EvidenceMember(member_id=f"m{index + 1}", artifact_id=fp.artifact.artifact_id)
            for index, fp in enumerate(fps)
        ),
        linkage=LinkageDescriptor(kind="manual", source="test"),
    )


def components_of(comparison: Comparison):
    return [component for family in comparison.families for component in family.components]


def component_for(comparison: Comparison, feature_id: str):
    matches = [c for c in components_of(comparison) if c.feature_id == feature_id]
    assert len(matches) == 1
    return matches[0]


def test_abs_ratio_component():
    left = make_fingerprint(
        "a1", features=[ok_obs(MORPH, build.ratio_value(1, 5), support_count=40)]
    )
    right = make_fingerprint(
        "a2", features=[ok_obs(MORPH, build.ratio_value(1, 2), support_count=50)]
    )
    comparison = compare_fingerprints(left, right, "left", "right")
    assert comparison.diagnostics == ()
    component = component_for(comparison, MORPH)
    assert component.metric == "ABS"
    assert component.value == pytest.approx(0.3)
    assert component.unit == "proportion points on [0,1]"
    assert component.semantic_version == "1.0.0"
    assert component.left_support == Support(kind="linguistic token", count=40)
    assert component.right_support == Support(kind="linguistic token", count=50)


def test_families_sorted_and_components_grouped():
    features = [
        ok_obs(MORPH, build.ratio_value(1, 5)),
        ok_obs(TOKEN_KIND, build.categorical_value({"word": 3, "number": 1})),
        ok_obs(WORD_LENGTH, build.histogram_value([1, 2, 2], 31)),
    ]
    left = make_fingerprint("a1", features=features)
    right = make_fingerprint("a2", features=features)
    comparison = compare_fingerprints(left, right, "left", "right")
    assert [family.family for family in comparison.families] == [
        "text.lexical",
        "text.linguistic",
    ]
    lexical = comparison.families[0]
    assert [component.feature_id for component in lexical.components] == [
        TOKEN_KIND,
        WORD_LENGTH,
    ]
    for component in components_of(comparison):
        assert component.value == 0.0


def test_jsd2_identical_and_disjoint():
    identical_left = make_fingerprint(
        "a1", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3, "number": 1}))]
    )
    identical_right = make_fingerprint(
        "a2", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3, "number": 1}))]
    )
    comparison = compare_fingerprints(identical_left, identical_right, "l", "r")
    component = component_for(comparison, TOKEN_KIND)
    assert component.metric == "JSD2"
    assert component.value == 0.0
    assert component.unit == "jensen-shannon distance [0,1]"

    disjoint_left = make_fingerprint(
        "a3", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 1}))]
    )
    disjoint_right = make_fingerprint(
        "a4", features=[ok_obs(TOKEN_KIND, build.categorical_value({"number": 1}))]
    )
    comparison = compare_fingerprints(disjoint_left, disjoint_right, "l", "r")
    assert component_for(comparison, TOKEN_KIND).value == 1.0


def test_w1_histogram_component():
    left = make_fingerprint(
        "a1", features=[ok_obs(WORD_LENGTH, build.histogram_value([0], 31))]
    )
    right = make_fingerprint(
        "a2", features=[ok_obs(WORD_LENGTH, build.histogram_value([3], 31))]
    )
    comparison = compare_fingerprints(left, right, "l", "r")
    component = component_for(comparison, WORD_LENGTH)
    assert component.metric == "W1"
    assert component.value == 3.0
    assert component.unit == "word code points"


def test_missing_on_one_side_is_omitted():
    left = make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 5))])
    right = make_fingerprint("a2")
    comparison = compare_fingerprints(left, right, "l", "r")
    assert comparison.families == ()
    assert [diagnostic.code for diagnostic in comparison.diagnostics] == [
        "FEATURE_NOT_COMPARABLE",
        "NO_COMPARABLE_FEATURES"
    ]
    diagnostic = comparison.diagnostics[0]
    assert diagnostic.severity == DiagnosticSeverity.WARNING
    assert diagnostic.feature_id == MORPH
    assert {entry.key: entry.value for entry in diagnostic.context} == {
        "left_status": "ok",
        "right_status": "missing",
    }


def test_non_ok_on_both_sides_is_omitted_with_status_context():
    left = make_fingerprint(
        "a1", features=[status_obs(MORPH, FeatureStatus.INSUFFICIENT_SUPPORT)]
    )
    right = make_fingerprint(
        "a2", features=[status_obs(MORPH, FeatureStatus.UNAVAILABLE)]
    )
    comparison = compare_fingerprints(left, right, "l", "r")
    assert comparison.families == ()
    diagnostic = next(
        diagnostic
        for diagnostic in comparison.diagnostics
        if diagnostic.code == "FEATURE_NOT_COMPARABLE"
    )
    assert diagnostic.feature_id == MORPH
    assert [(entry.key, entry.value) for entry in diagnostic.context] == [
        ("left_status", "insufficient_support"),
        ("right_status", "unavailable"),
    ]


def test_metric_none_features_are_never_compared():
    left = make_fingerprint("a1", features=[ok_obs(WORD_COUNT, IntegerValue(value=5))])
    right = make_fingerprint("a2", features=[ok_obs(WORD_COUNT, IntegerValue(value=9))])
    comparison = compare_fingerprints(left, right, "l", "r")
    assert comparison.families == ()
    assert [diagnostic.code for diagnostic in comparison.diagnostics] == [
        "NO_COMPARABLE_FEATURES"
    ]


def test_text_vs_code_cross_comparison_raises():
    left = make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 5))])
    right = make_fingerprint(
        "a2",
        kind=ArtifactKind.CODE,
        language="python",
        features=[ok_obs(MORPH, build.ratio_value(1, 2))],
    )
    with pytest.raises(PortableArtifactError):
        compare_fingerprints(left, right, "l", "r")


def test_semantic_mismatch_skips_feature_with_warning():
    left = make_fingerprint(
        "a1", features=[ok_obs(MORPH, build.ratio_value(1, 5), semantic_version="1.0.0")]
    )
    right = make_fingerprint(
        "a2", features=[ok_obs(MORPH, build.ratio_value(1, 2), semantic_version="9.9.9")]
    )
    comparison = compare_fingerprints(left, right, "l", "r")
    assert comparison.families == ()
    by_code = {diagnostic.code: diagnostic for diagnostic in comparison.diagnostics}
    assert by_code["FEATURE_SEMANTIC_MISMATCH"].severity == DiagnosticSeverity.WARNING
    assert by_code["FEATURE_SEMANTIC_MISMATCH"].feature_id == MORPH
    assert "NO_COMPARABLE_FEATURES" in by_code


def test_runtime_signature_mismatch_for_runtime_sensitive_feature():
    left = make_fingerprint("a1", features=[ok_obs(TTR, build.ratio_value(1, 2))])
    right = make_fingerprint(
        "a2", features=[ok_obs(TTR, build.ratio_value(3, 4))], runtime=OTHER_RUNTIME
    )
    comparison = compare_fingerprints(left, right, "l", "r")
    assert comparison.families == ()
    by_code = {diagnostic.code: diagnostic for diagnostic in comparison.diagnostics}
    assert by_code["RUNTIME_SIGNATURE_MISMATCH"].severity == DiagnosticSeverity.WARNING
    assert by_code["RUNTIME_SIGNATURE_MISMATCH"].feature_id == TTR


def test_no_global_score_exists():
    left = make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 5))])
    right = make_fingerprint("a2", features=[ok_obs(MORPH, build.ratio_value(1, 2))])
    comparison = compare_fingerprints(left, right, "l", "r")
    assert "score" not in Comparison.model_fields
    assert "similarity" not in Comparison.model_fields
    assert not hasattr(comparison, "score")
    assert not hasattr(comparison, "similarity")


def _aggregate(evidence_set_id: str, fps):
    return aggregate_fingerprints(make_evidence_set(fps, evidence_set_id), fps)


def test_compare_aggregates_pooled_ratio():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 10))]),
        make_fingerprint("a2", features=[ok_obs(MORPH, build.ratio_value(9, 90))]),
    ]
    right_fps = [
        make_fingerprint("a3", features=[ok_obs(MORPH, build.ratio_value(1, 2))]),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    assert comparison.diagnostics == ()
    component = component_for(comparison, MORPH)
    assert component.metric == "ABS"
    assert component.value == pytest.approx(0.4)
    assert component.unit == "proportion points on [0,1]"
    assert component.left_support == Support(kind="sample", count=2)
    assert component.right_support == Support(kind="sample", count=1)


def test_compare_aggregates_pooled_categorical():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3}))]),
        make_fingerprint("a2", features=[ok_obs(TOKEN_KIND, build.categorical_value({"number": 1}))]),
    ]
    right_fps = [
        make_fingerprint("a3", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 4}))]),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    component = component_for(comparison, TOKEN_KIND)
    assert component.metric == "JSD2"
    # Pooled left {word: 3, number: 1} vs right {word: 4}, computed by hand:
    # M = {word: 0.875, number: 0.125}; JSD2 = sqrt(0.5*KL(P||M) + 0.5*KL(Q||M)).
    kl_pm = 0.75 * math.log2(0.75 / 0.875) + 0.25 * math.log2(0.25 / 0.125)
    kl_qm = math.log2(1.0 / 0.875)
    assert component.value == pytest.approx(math.sqrt(0.5 * kl_pm + 0.5 * kl_qm))


def test_compare_aggregates_sample_summary_uses_sample_wasserstein():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(TTR, build.ratio_value(1, 2))]),
        make_fingerprint("a2", features=[ok_obs(TTR, build.ratio_value(3, 4))]),
    ]
    right_fps = [
        make_fingerprint("a3", features=[ok_obs(TTR, build.ratio_value(9, 10))]),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    component = component_for(comparison, TTR)
    assert component.metric == "sample_wasserstein_1"
    assert component.unit == "sample units"
    # support [0.5, 0.75, 0.9]: |cdf diff| terms 0.5*0.25 + 1.0*0.15.
    assert component.value == pytest.approx(0.275)
    assert component.left_support == Support(kind="sample", count=2)
    assert component.right_support == Support(kind="sample", count=1)


def test_compare_aggregates_semantic_mismatch():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3}))]),
    ]
    right_fps = [
        make_fingerprint(
            "a2",
            features=[
                ok_obs(
                    TOKEN_KIND,
                    build.categorical_value({"word": 3}),
                    semantic_version="9.9.9",
                )
            ],
        ),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    assert comparison.families == ()
    by_code = {diagnostic.code: diagnostic for diagnostic in comparison.diagnostics}
    assert by_code["FEATURE_SEMANTIC_MISMATCH"].severity == DiagnosticSeverity.WARNING
    assert by_code["FEATURE_SEMANTIC_MISMATCH"].feature_id == TOKEN_KIND


def test_compare_aggregates_no_comparable_features():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 10))]),
    ]
    right_fps = [
        make_fingerprint("a2", features=[ok_obs(TOKEN_KIND, build.categorical_value({"word": 3}))]),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    assert comparison.families == ()
    assert [diagnostic.code for diagnostic in comparison.diagnostics] == [
        "FEATURE_NOT_COMPARABLE",
        "FEATURE_NOT_COMPARABLE",
        "NO_COMPARABLE_FEATURES"
    ]
    by_feature = {
        diagnostic.feature_id: {
            entry.key: entry.value for entry in diagnostic.context
        }
        for diagnostic in comparison.diagnostics
        if diagnostic.code == "FEATURE_NOT_COMPARABLE"
    }
    assert by_feature == {
        MORPH: {"left_status": "ok", "right_status": "missing"},
        TOKEN_KIND: {"left_status": "missing", "right_status": "ok"},
    }


def test_compare_aggregates_zero_contributors_has_status_diagnostic():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(MORPH, build.ratio_value(1, 10))]),
    ]
    right_fps = [
        make_fingerprint(
            "a2", features=[status_obs(MORPH, FeatureStatus.INSUFFICIENT_SUPPORT)]
        ),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    assert comparison.families == ()
    diagnostic = next(
        diagnostic
        for diagnostic in comparison.diagnostics
        if diagnostic.code == "FEATURE_NOT_COMPARABLE"
    )
    assert diagnostic.feature_id == MORPH
    assert [(entry.key, entry.value) for entry in diagnostic.context] == [
        ("left_status", "ok"),
        ("right_status", "insufficient_support"),
    ]


def test_metric_none_aggregate_features_are_not_missing_diagnostics():
    left_fps = [
        make_fingerprint("a1", features=[ok_obs(WORD_COUNT, IntegerValue(value=5))]),
    ]
    right_fps = [
        make_fingerprint("a2", features=[ok_obs(WORD_COUNT, IntegerValue(value=9))]),
    ]
    comparison = compare_aggregates(
        _aggregate("esL", left_fps), _aggregate("esR", right_fps), "left", "right"
    )
    assert comparison.families == ()
    assert [diagnostic.code for diagnostic in comparison.diagnostics] == [
        "NO_COMPARABLE_FEATURES"
    ]
