"""VerifierFit / Verification portable contracts (spec 5.20-5.21, 23)."""

from __future__ import annotations

import pytest

from stylog.domain import (
    BackendSignature,
    PackageProvenance,
    RuntimeSignature,
    Verification,
    VerifierCalibration,
    VerifierEligibility,
    VerifierFeature,
    VerifierFit,
    VerifierFitConfig,
    VerifierPairPolicy,
    VerifierThresholds,
)
from stylog.exceptions import PortableArtifactError
from stylog.serialization.canonical import canonical_bytes, scientific_sha256
from stylog.serialization.jsonio import model_from_bytes

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64

RUNTIME = RuntimeSignature(
    python_implementation="CPython",
    python_version="3.12.0",
    python_cache_tag="cpython-312",
    unicode_database_version="15.0.0",
)

BACKEND = BackendSignature(
    backend_id="stylog",
    implementation_version="0.1.0",
    scientific_compatibility_id="stylog.verifier.logreg/1",
    packages=(PackageProvenance(package="stylog", version="0.1.0"),),
)

FEATURE_A = VerifierFeature(
    feature_id="text.function_words.en.token_share",
    semantic_version="1.0.0",
    metric="ABS",
    mean=0.4,
    scale=0.1,
)
FEATURE_B = VerifierFeature(
    feature_id="text.lexical.ttr_casefold",
    semantic_version="1.0.0",
    metric="ABS",
    mean=0.5,
    scale=0.2,
)


def make_fit(**overrides) -> VerifierFit:
    kwargs = {
        "model_id": "stylog.verifier.logreg/1",
        "model_semantic_version": "1.0.0",
        "task": "pairwise_authorship_verification",
        "task_version": "1",
        "kind": "text",
        "languages": ("en",),
        "feature_registry_version": "1.0.0",
        "features": (FEATURE_A, FEATURE_B),
        "coefficients": (1.5, -0.5),
        "intercept": -0.2,
        "thresholds": VerifierThresholds(t_same=0.7, t_diff=0.3),
        "threshold_rule": "fixed",
        "fit_config": VerifierFitConfig(
            l2_lambda=1.0,
            max_iterations=100,
            tolerance=1e-12,
            min_support_fraction=0.9,
            min_class_support_fraction=0.8,
            min_pairs=50,
            threshold_rule="fixed",
            threshold_fixed=0.5,
            include_linguistic=False,
            allow_unconstrained_language=False,
            pair_policy=VerifierPairPolicy(selection_version="1"),
        ),
        "eligibility": VerifierEligibility(
            training_pair_count=100,
            eligible_pair_count=90,
            candidate_feature_count=10,
            selected_feature_count=2,
        ),
        "source_manifest_sha256": H1,
        "runtime": RUNTIME,
        "backend": BACKEND,
    }
    kwargs.update(overrides)
    return VerifierFit(**kwargs)


def make_verification(drop: tuple[str, ...] = (), **overrides) -> Verification:
    kwargs = {
        "left_ref": "a.txt",
        "right_ref": "b.txt",
        "left_fingerprint_sha256": H2,
        "right_fingerprint_sha256": H3,
        "verifier_id": H4,
        "model_id": "stylog.verifier.logreg/1",
        "model_semantic_version": "1.0.0",
        "verdict": "same_author",
        "score": 0.9,
        "features_used": 2,
    }
    kwargs.update(overrides)
    for key in drop:
        kwargs.pop(key, None)
    return Verification(**kwargs)


def test_verifier_fit_roundtrip_canonical() -> None:
    fit = make_fit()
    parsed = model_from_bytes(canonical_bytes(fit), VerifierFit)
    assert scientific_sha256(parsed) == scientific_sha256(fit)


def test_verification_roundtrip_canonical() -> None:
    verification = make_verification()
    parsed = model_from_bytes(canonical_bytes(verification), Verification)
    assert scientific_sha256(parsed) == scientific_sha256(verification)


def test_complete_model_identity_binds_provenance() -> None:
    # Same numerical state but different training manifest -> different identity.
    fit_a = make_fit()
    fit_b = make_fit(source_manifest_sha256=H2)
    assert scientific_sha256(fit_a) != scientific_sha256(fit_b)


def test_coefficients_must_align_with_features() -> None:
    with pytest.raises(ValueError, match="align"):
        make_fit(coefficients=(1.5,))


def test_features_sorted_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_fit(features=(FEATURE_B, FEATURE_A), coefficients=(1.0, 2.0))


def test_feature_scale_positive() -> None:
    with pytest.raises(ValueError, match="scale"):
        VerifierFeature(
            feature_id="x", semantic_version="1.0.0", metric="ABS", mean=0.0, scale=0.0
        )


def test_threshold_order_enforced() -> None:
    with pytest.raises(ValueError, match="t_diff"):
        VerifierThresholds(t_same=0.3, t_diff=0.7)
    with pytest.raises(ValueError):
        VerifierThresholds(t_same=1.0, t_diff=0.5)
    with pytest.raises(ValueError):
        VerifierThresholds(t_same=0.5, t_diff=0.0)
    # collapsed band is legal
    assert VerifierThresholds(t_same=0.5, t_diff=0.5).t_same == 0.5


def test_languages_sorted_and_opt_in() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_fit(languages=("fr", "en"))
    # empty languages requires explicit opt-in
    with pytest.raises(ValueError, match="allow_unconstrained_language"):
        make_fit(languages=())


def test_unconstrained_language_with_opt_in_ok() -> None:
    config = VerifierFitConfig(
        l2_lambda=1.0,
        max_iterations=100,
        tolerance=1e-12,
        min_support_fraction=0.9,
        min_class_support_fraction=0.8,
        min_pairs=50,
        threshold_rule="fixed",
        threshold_fixed=0.5,
        include_linguistic=False,
        allow_unconstrained_language=True,
        pair_policy=VerifierPairPolicy(selection_version="1"),
    )
    fit = make_fit(languages=(), fit_config=config)
    assert fit.languages == ()


def test_threshold_rule_params() -> None:
    base = {
        "l2_lambda": 1.0,
        "max_iterations": 100,
        "tolerance": 1e-12,
        "min_support_fraction": 0.9,
        "min_class_support_fraction": 0.8,
        "min_pairs": 50,
        "include_linguistic": False,
        "allow_unconstrained_language": False,
        "pair_policy": VerifierPairPolicy(selection_version="1"),
    }
    with pytest.raises(ValueError, match="threshold_alpha"):
        VerifierFitConfig(threshold_rule="calibration_quantile_band", **base)
    with pytest.raises(ValueError, match="threshold_fixed"):
        VerifierFitConfig(threshold_rule="fixed", **base)
    with pytest.raises(ValueError, match="threshold_rule"):
        VerifierFitConfig(threshold_rule="bogus", threshold_fixed=0.5, **base)
    cfg = VerifierFitConfig(
        threshold_rule="calibration_quantile_band", threshold_alpha=0.05, **base
    )
    assert cfg.threshold_alpha == 0.05


def test_threshold_rule_consistency_and_calibration_manifest() -> None:
    # calibration_quantile_band requires a calibration manifest hash
    config = VerifierFitConfig(
        l2_lambda=1.0,
        max_iterations=100,
        tolerance=1e-12,
        min_support_fraction=0.9,
        min_class_support_fraction=0.8,
        min_pairs=50,
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.05,
        include_linguistic=False,
        allow_unconstrained_language=False,
        pair_policy=VerifierPairPolicy(selection_version="1"),
    )
    with pytest.raises(ValueError, match="calibration_manifest_sha256"):
        make_fit(
            threshold_rule="calibration_quantile_band",
            fit_config=config,
        )
    fit = make_fit(
        threshold_rule="calibration_quantile_band",
        fit_config=config,
        calibration_manifest_sha256=H3,
    )
    assert fit.calibration_manifest_sha256 == H3
    # fixed rule with no calibration must not carry a calibration manifest
    with pytest.raises(ValueError, match="calibration_manifest_sha256"):
        make_fit(calibration_manifest_sha256=H3)


def test_calibration_consistency() -> None:
    calibration = VerifierCalibration(a=1.0, b=0.0)
    # calibration present but config says uncalibrated
    with pytest.raises(ValueError, match="calibration"):
        make_fit(calibration=calibration, calibration_manifest_sha256=H3)
    config = VerifierFitConfig(
        l2_lambda=1.0,
        max_iterations=100,
        tolerance=1e-12,
        min_support_fraction=0.9,
        min_class_support_fraction=0.8,
        min_pairs=50,
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.05,
        calibration_method="platt",
        include_linguistic=False,
        allow_unconstrained_language=False,
        pair_policy=VerifierPairPolicy(selection_version="1"),
    )
    fit = make_fit(
        threshold_rule="calibration_quantile_band",
        fit_config=config,
        calibration=calibration,
        calibration_manifest_sha256=H3,
    )
    assert fit.calibration is not None and fit.calibration.method == "platt"


def test_score_presence_matrix() -> None:
    # same/different/uncertain require a score
    for verdict, reason in (
        ("same_author", None),
        ("different_author", None),
        ("abstain", "uncertain"),
    ):
        kwargs = {"verdict": verdict, "score": 0.5}
        if reason is not None:
            kwargs["abstain_reason"] = reason
        make_verification(**kwargs)
        with pytest.raises(ValueError, match="score"):
            bad = {"verdict": verdict}
            if reason is not None:
                bad["abstain_reason"] = reason
            make_verification(drop=("score",), **bad)
    # insufficient_evidence forbids a score
    v = make_verification(
        drop=("score",),
        verdict="abstain",
        abstain_reason="insufficient_evidence",
        features_missing=("text.lexical.ttr_casefold",),
        features_used=1,
    )
    assert v.score is None
    assert "score" not in canonical_bytes(v).decode()
    with pytest.raises(ValueError, match="score"):
        make_verification(
            verdict="abstain",
            abstain_reason="insufficient_evidence",
            score=0.5,
            features_missing=("text.lexical.ttr_casefold",),
        )


def test_probability_requires_calibration_method_and_score() -> None:
    with pytest.raises(ValueError, match="calibration_method"):
        make_verification(probability=0.8)
    v = make_verification(probability=0.8, calibration_method="platt")
    assert v.calibration_method == "platt"
    # probability without score (insufficient evidence) is impossible
    with pytest.raises(ValueError):
        make_verification(
            verdict="abstain",
            abstain_reason="insufficient_evidence",
            probability=0.8,
            calibration_method="platt",
            features_missing=("x",),
        )


def test_abstain_reason_iff_abstain() -> None:
    with pytest.raises(ValueError, match="abstain_reason"):
        make_verification(verdict="same_author", abstain_reason="uncertain", score=0.9)


def test_features_missing_implies_insufficient_evidence() -> None:
    with pytest.raises(ValueError, match="features_missing"):
        make_verification(features_missing=("x",), score=0.9)


def test_features_missing_sorted_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        make_verification(
            drop=("score",),
            verdict="abstain",
            abstain_reason="insufficient_evidence",
            features_missing=("b", "a"),
        )


def test_score_range_enforced() -> None:
    with pytest.raises(ValueError, match="\\(0, 1\\)"):
        make_verification(score=1.0)
    with pytest.raises(ValueError, match="\\(0, 1\\)"):
        make_verification(score=0.0)


def test_null_rejected() -> None:
    with pytest.raises(PortableArtifactError):
        model_from_bytes(
            b'{"schema":"stylog.verification","score":null}', Verification
        )


def test_evidence_hashes_required_and_well_formed() -> None:
    with pytest.raises(ValueError):
        make_verification(left_fingerprint_sha256="not-a-hash")
    tree = canonical_bytes(make_verification())
    assert b'"left_fingerprint_sha256":"' in tree
    assert b'"right_fingerprint_sha256":"' in tree
    assert b'"verifier_id":"' in tree


def test_schemas_cover_new_models() -> None:
    from tools.generate_schemas import SCHEMA_MODELS

    assert SCHEMA_MODELS["stylog.verifier-fit"] is VerifierFit
    assert SCHEMA_MODELS["stylog.verification"] is Verification
