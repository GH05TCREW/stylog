"""Verification scoring core tests (spec section 23)."""

from __future__ import annotations

import hashlib
import math

import pytest

from stylog.analysis.registry import (
    FEATURES,
    FUNCTION_WORDS_EN_RESOURCE_VERSION,
    FUNCTION_WORDS_EN_SHA256,
    features_owned_by,
)
from stylog.analysis.verify import (
    VERIFIER_MODEL_ID,
    align_evidence,
    check_model_compatibility,
    decide,
    decision_score,
    sigmoid_open,
    verify_fingerprints,
)
from stylog.domain.artifact import ArtifactDescriptor, ArtifactKind, ContentIdentitySha256
from stylog.domain.diagnostic import DiagnosticSeverity
from stylog.domain.feature import (
    InsufficientSupportObservation,
    OkFeatureObservation,
    RatioValue,
    Support,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import (
    AnalyzerSignature,
    BackendSignature,
    PackageProvenance,
    ResourceSignature,
    RuntimeSignature,
)
from stylog.domain.verification import (
    VerifierCalibration,
    VerifierEligibility,
    VerifierFeature,
    VerifierFit,
    VerifierFitConfig,
    VerifierPairPolicy,
    VerifierThresholds,
)
from stylog.exceptions import CapabilityUnavailableError, ModelIncompatibilityError
from stylog.serialization.canonical import portable_tree, scientific_sha256

RUNTIME = RuntimeSignature(
    python_implementation="CPython",
    python_version="3.14.0",
    python_cache_tag="cpython-314",
    unicode_database_version="17.0.0",
)

TTR = "text.lexical.ttr_casefold"  # ratio, ABS
HAPAX = "text.lexical.hapax_token_share_casefold"  # ratio, ABS
MORPH = "text.linguistic.morph_coverage"  # ratio, ABS, [nlp]-only
FW = "text.function_words.en.token_share"  # ratio, ABS

MODEL_BACKEND = BackendSignature(
    backend_id="stylog",
    implementation_version="0.1.0",
    scientific_compatibility_id=VERIFIER_MODEL_ID,
    packages=(PackageProvenance(package="stylog", version="0.1.0"),),
)


def make_analyzer(analyzer_id: str, *, compat_id: str = "stylog.text-core/1") -> AnalyzerSignature:
    # Resource-backed features (English function words) need matching resource
    # signatures on both sides or the pairwise compat gate rejects them.
    resource_ids = sorted(
        {resource_id for fdef in features_owned_by(analyzer_id) for resource_id in fdef.resource_ids}
    )
    return AnalyzerSignature(
        analyzer_id=analyzer_id,
        implementation_version="1.0.0",
        feature_registry_version="1.0.0",
        backend=BackendSignature(
            backend_id="stylog.native.text",
            implementation_version="1.0.0",
            scientific_compatibility_id=compat_id,
        ),
        resources=tuple(
            ResourceSignature(
                id=resource_id,
                version=FUNCTION_WORDS_EN_RESOURCE_VERSION,
                sha256=FUNCTION_WORDS_EN_SHA256,
            )
            for resource_id in resource_ids
        ),
    )


def ok_obs(
    feature_id: str, value, *, semantic_version: str = "1.0.0"
) -> OkFeatureObservation:
    fdef = FEATURES[feature_id]
    return OkFeatureObservation(
        feature_id=feature_id,
        semantic_version=semantic_version,
        analyzer_id=fdef.analyzer_id,
        analyzer_implementation_version="1.0.0",
        value=value,
        support=Support(kind=fdef.support_kind, count=10),
    )


def ratio(value: float, numerator: int = 0, denominator: int = 1) -> RatioValue:
    if numerator == 0:
        numerator = int(value * 1000)
        denominator = 1000
    return RatioValue(numerator=numerator, denominator=denominator, multiplier=1.0, value=value)


def make_fingerprint(
    artifact_id: str,
    *,
    kind: ArtifactKind = ArtifactKind.TEXT,
    language: str = "en",
    features=(),
    compat_id: str = "stylog.text-core/1",
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
        runtime=RUNTIME,
        analysis_config_sha256="f" * 64,
        analyzers=tuple(
            make_analyzer(analyzer_id, compat_id=compat_id)
            for analyzer_id in sorted(analyzer_ids)
        ),
        features=tuple(sorted(features, key=lambda observation: observation.feature_id)),
    )


def make_model(
    feature_ids=(TTR,),
    *,
    coefficients=None,
    intercept=-1.0,
    t_same=0.7,
    t_diff=0.3,
    languages=("en",),
    kind=ArtifactKind.TEXT,
    calibration=None,
    means=None,
    scales=None,
) -> VerifierFit:
    features = tuple(
        VerifierFeature(
            feature_id=feature_id,
            semantic_version="1.0.0",
            metric=FEATURES[feature_id].metric,
            mean=(means or {}).get(feature_id, 0.5),
            scale=(scales or {}).get(feature_id, 0.25),
        )
        for feature_id in sorted(feature_ids)
    )
    n = len(features)
    threshold_rule = "fixed"
    config_kwargs = {
        "l2_lambda": 1.0,
        "max_iterations": 100,
        "tolerance": 1e-12,
        "min_support_fraction": 0.9,
        "min_class_support_fraction": 0.8,
        "min_pairs": 2,
        "threshold_rule": threshold_rule,
        "threshold_fixed": 0.5,
        "include_linguistic": True,
        "allow_unconstrained_language": not languages,
        "pair_policy": VerifierPairPolicy(selection_version="1"),
    }
    if calibration is not None:
        config_kwargs["calibration_method"] = "platt"
    fit_config = VerifierFitConfig(**config_kwargs)
    fit_kwargs = {
        "model_id": VERIFIER_MODEL_ID,
        "model_semantic_version": "1.0.0",
        "task": "pairwise_authorship_verification",
        "task_version": "1",
        "kind": kind,
        "languages": languages,
        "feature_registry_version": "1.0.0",
        "features": features,
        "coefficients": tuple(coefficients) if coefficients is not None else (2.0,) * n,
        "intercept": intercept,
        "thresholds": VerifierThresholds(t_same=t_same, t_diff=t_diff),
        "threshold_rule": threshold_rule,
        "fit_config": fit_config,
        "eligibility": VerifierEligibility(
            training_pair_count=10,
            eligible_pair_count=10,
            candidate_feature_count=n,
            selected_feature_count=n,
        ),
        "source_manifest_sha256": "a" * 64,
        "runtime": RUNTIME,
        "backend": MODEL_BACKEND,
    }
    if calibration is not None:
        fit_kwargs["calibration"] = calibration
        fit_kwargs["calibration_manifest_sha256"] = "b" * 64
    return VerifierFit(**fit_kwargs)


def test_sigmoid_open_extremes() -> None:
    high = sigmoid_open(1000.0)
    low = sigmoid_open(-1000.0)
    assert 0.999999999999999 < high < 1.0
    assert 0.0 < low < 1e-300
    assert sigmoid_open(0.0) == 0.5


def test_decision_score_golden() -> None:
    model = make_model((TTR,), coefficients=(2.0,), intercept=-1.0)
    # ABS distance between TTR 0.8 and 0.6 is 0.2; z = (0.2-0.5)/0.25 = -1.2
    score, logit = decision_score(model, (0.2,))
    expected_logit = math.fsum([-1.0, 2.0 * ((0.2 - 0.5) / 0.25)])
    assert logit == expected_logit == -3.4
    z = math.exp(-3.4)
    assert score == z / (1.0 + z)


def test_decision_score_standardization_uses_model_state() -> None:
    model = make_model(
        (TTR,), coefficients=(1.0,), intercept=0.0, means={TTR: 0.1}, scales={TTR: 2.0}
    )
    score, logit = decision_score(model, (0.5,))
    assert logit == (0.5 - 0.1) / 2.0
    assert score == sigmoid_open(logit)


def test_decide_boundaries() -> None:
    model = make_model(t_same=0.7, t_diff=0.3)
    assert decide(model, 0.7) == "same_author"
    assert decide(model, 0.7000000000000001) == "same_author"
    assert decide(model, 0.3) == "different_author"
    assert decide(model, 0.2999999999999999) == "different_author"
    assert decide(model, 0.5) == "abstain"


def test_decide_collapsed_band() -> None:
    model = make_model(t_same=0.5, t_diff=0.5)
    # same-check first: the boundary itself is same_author, deterministically
    assert decide(model, 0.5) == "same_author"
    assert decide(model, 0.49999999999999994) == "different_author"


def test_verify_end_to_end_different_author() -> None:
    model = make_model((TTR,), coefficients=(2.0,), intercept=-1.0)
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)),))
    verification = verify_fingerprints(model, left, right, left_ref="a", right_ref="b")
    # identical values -> distance 0.0 -> z = (0-0.5)/0.25 = -2 -> logit -5 -> low score
    score, _ = decision_score(model, (0.0,))
    assert verification.score == score
    assert verification.verdict == "different_author"
    assert verification.score < 0.3
    assert verification.abstain_reason is None
    assert verification.probability is None
    assert verification.calibration_method is None
    assert verification.features_used == 1
    assert verification.features_missing == ()
    assert verification.left_fingerprint_sha256 == scientific_sha256(left)
    assert verification.right_fingerprint_sha256 == scientific_sha256(right)
    assert verification.verifier_id == scientific_sha256(model)


def test_verify_emits_probability_only_when_calibrated() -> None:
    calibration = VerifierCalibration(a=1.5, b=0.25)
    model = make_model((TTR,), calibration=calibration)
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.9)),))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)),))
    verification = verify_fingerprints(model, left, right)
    assert verification.score is not None
    assert verification.calibration_method == "platt"
    _, logit = decision_score(model, (0.09999999999999998,))
    expected = sigmoid_open(calibration.a * logit + calibration.b)
    assert verification.probability == expected


def test_insufficient_evidence_carries_no_score() -> None:
    calibration = VerifierCalibration(a=1.5, b=0.25)
    model = make_model((TTR, HAPAX), calibration=calibration)
    fdef = FEATURES[HAPAX]
    bad_obs = InsufficientSupportObservation(
        feature_id=HAPAX,
        semantic_version="1.0.0",
        analyzer_id=fdef.analyzer_id,
        analyzer_implementation_version="1.0.0",
    )
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)), ok_obs(HAPAX, ratio(0.3))))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)), bad_obs))
    verification = verify_fingerprints(model, left, right)
    assert verification.verdict == "abstain"
    assert verification.abstain_reason == "insufficient_evidence"
    assert verification.score is None
    assert verification.probability is None
    assert verification.calibration_method is None
    assert verification.features_missing == (HAPAX,)
    assert verification.features_used == 1
    # evidence binding still present
    assert verification.left_fingerprint_sha256 == scientific_sha256(left)


def test_compat_gated_feature_is_missing_with_diagnostic() -> None:
    model = make_model((TTR,))
    left = make_fingerprint(
        "a", features=(ok_obs(TTR, ratio(0.8), semantic_version="0.9.0"),)
    )
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8), semantic_version="1.0.0"),))
    verification = verify_fingerprints(model, left, right)
    assert verification.verdict == "abstain"
    assert verification.abstain_reason == "insufficient_evidence"
    assert verification.features_missing == (TTR,)
    codes = [diagnostic.code for diagnostic in verification.diagnostics]
    assert codes == ["FEATURE_SEMANTIC_MISMATCH"]
    assert verification.diagnostics[0].severity == DiagnosticSeverity.WARNING


def test_absent_feature_is_capability_error() -> None:
    model = make_model((MORPH,))
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)),))
    with pytest.raises(CapabilityUnavailableError):
        verify_fingerprints(model, left, right)


def test_model_compatibility_gates() -> None:
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)),))

    bad_id_model = make_model((TTR,))
    object.__setattr__(bad_id_model, "model_id", "other/1")
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(bad_id_model, left, right)

    registry_model = make_model((TTR,))
    object.__setattr__(registry_model, "feature_registry_version", "0.9.0")
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(registry_model, left, right)

    code_model = make_model((TTR,), kind=ArtifactKind.CODE)
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(code_model, left, right)

    fr_right = make_fingerprint("b", language="fr", features=(ok_obs(TTR, ratio(0.8)),))
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(make_model((TTR,)), left, fr_right)

    bad_feature = VerifierFit.model_validate(
        {
            **portable_tree(make_model((TTR,))),
            "features": [
                {
                    "feature_id": TTR,
                    "semantic_version": "0.9.0",
                    "metric": "ABS",
                    "mean": 0.5,
                    "scale": 0.25,
                }
            ],
        }
    )
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(bad_feature, left, right)

    unknown_feature_model = make_model((TTR,))
    object.__setattr__(
        unknown_feature_model,
        "features",
        (
            VerifierFeature(
                feature_id="text.unknown.feature",
                semantic_version="1.0.0",
                metric="ABS",
                mean=0.0,
                scale=1.0,
            ),
        ),
    )
    with pytest.raises(ModelIncompatibilityError):
        check_model_compatibility(unknown_feature_model, left, right)


def test_semantic_symmetry() -> None:
    calibration = VerifierCalibration(a=1.2, b=-0.1)
    model = make_model((TTR, HAPAX, FW), calibration=calibration)
    left = make_fingerprint(
        "a",
        features=(
            ok_obs(TTR, ratio(0.83)),
            ok_obs(HAPAX, ratio(0.41)),
            ok_obs(FW, ratio(0.52)),
        ),
    )
    right = make_fingerprint(
        "b",
        features=(
            ok_obs(TTR, ratio(0.71)),
            ok_obs(HAPAX, ratio(0.44)),
            ok_obs(FW, ratio(0.49)),
        ),
    )
    forward = verify_fingerprints(model, left, right, left_ref="doc-a", right_ref="doc-b")
    backward = verify_fingerprints(model, right, left, left_ref="doc-b", right_ref="doc-a")
    # every scientific field agrees
    for field_name in (
        "verdict",
        "abstain_reason",
        "score",
        "probability",
        "calibration_method",
        "features_used",
        "features_missing",
        "diagnostics",
        "verifier_id",
    ):
        assert getattr(forward, field_name) == getattr(backward, field_name), field_name
    # refs and evidence hashes swap positions
    assert forward.left_ref == backward.right_ref == "doc-a"
    assert forward.right_ref == backward.left_ref == "doc-b"
    assert forward.left_fingerprint_sha256 == backward.right_fingerprint_sha256
    assert forward.right_fingerprint_sha256 == backward.left_fingerprint_sha256


def test_evidence_binding_distinguishes_pairs_with_identical_refs() -> None:
    model = make_model((TTR,))
    left_a = make_fingerprint("a1", features=(ok_obs(TTR, ratio(0.8)),))
    right_a = make_fingerprint("a2", features=(ok_obs(TTR, ratio(0.8)),))
    left_b = make_fingerprint("b1", features=(ok_obs(TTR, ratio(0.3)),))
    right_b = make_fingerprint("b2", features=(ok_obs(TTR, ratio(0.3)),))
    first = verify_fingerprints(model, left_a, right_a)  # default refs left/right
    second = verify_fingerprints(model, left_b, right_b)
    assert first.left_ref == second.left_ref == "left"
    assert first.left_fingerprint_sha256 != second.left_fingerprint_sha256
    assert scientific_sha256(first) != scientific_sha256(second)


def test_align_evidence_orders_missing_sorted() -> None:
    model = make_model((TTR, HAPAX, FW))
    fdef_h = FEATURES[HAPAX]
    fdef_f = FEATURES[FW]
    left = make_fingerprint(
        "a",
        features=(
            InsufficientSupportObservation(
                feature_id=HAPAX,
                semantic_version="1.0.0",
                analyzer_id=fdef_h.analyzer_id,
                analyzer_implementation_version="1.0.0",
            ),
            ok_obs(TTR, ratio(0.8)),
            InsufficientSupportObservation(
                feature_id=FW,
                semantic_version="1.0.0",
                analyzer_id=fdef_f.analyzer_id,
                analyzer_implementation_version="1.0.0",
            ),
        ),
    )
    right = make_fingerprint(
        "b",
        features=(ok_obs(TTR, ratio(0.8)), ok_obs(HAPAX, ratio(0.4)), ok_obs(FW, ratio(0.5))),
    )
    from stylog.analysis.compare import compare_fingerprints

    comparison = compare_fingerprints(left, right, "a", "b")
    aligned = align_evidence(model, left, right, comparison)
    assert not aligned.complete
    assert aligned.missing == tuple(sorted((HAPAX, FW)))
    assert aligned.values == ()


def test_repeated_verification_byte_identical() -> None:
    model = make_model((TTR, HAPAX))
    left = make_fingerprint(
        "a", features=(ok_obs(TTR, ratio(0.83)), ok_obs(HAPAX, ratio(0.41)))
    )
    right = make_fingerprint(
        "b", features=(ok_obs(TTR, ratio(0.71)), ok_obs(HAPAX, ratio(0.44)))
    )
    first = verify_fingerprints(model, left, right)
    second = verify_fingerprints(model, left, right)
    assert scientific_sha256(first) == scientific_sha256(second)


def test_comparison_stays_descriptive() -> None:
    # Acceptance guard: Comparison never gains score/verdict/similarity fields.
    from stylog.domain.interpretation import Comparison, ComparisonComponent, ComparisonFamily

    for model_type in (Comparison, ComparisonFamily, ComparisonComponent):
        field_names = set(model_type.model_fields)
        assert not (field_names & {"score", "verdict", "similarity", "probability"})
