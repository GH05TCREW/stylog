"""VerifierFit and Verification portable models (spec sections 5.20-5.21).

A ``VerifierFit`` is a fully self-contained fitted verifier: every fitted
parameter (features with normalization state, coefficients, intercept,
thresholds, optional calibration) plus complete fit provenance (fit config,
eligibility counts, train/tuning/calibration manifest identities, runtime,
backend) lives in one canonical artifact. Its scientific identity is
``scientific_sha256(VerifierFit)`` computed on demand.

A ``Verification`` is a decision-layer artifact: a model-relative verdict
bound to the scientific hashes of the two measured inputs and of the complete
verifier model. A saved decision therefore identifies exact measured A +
exact measured B + exact fitted verifier + resulting decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from stylog.domain._base import (
    HexDigest64,
    PortableFloat,
    PortableModel,
    is_sorted_unique,
    tuple_of,
)
from stylog.domain.artifact import ArtifactKind
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key
from stylog.domain.provenance import BackendSignature, RuntimeSignature

VERIFIER_TASK = "pairwise_authorship_verification"
VERIFIER_TASK_VERSION = "1"

THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND = "calibration_quantile_band"
THRESHOLD_RULE_FIXED = "fixed"
THRESHOLD_RULES = (THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND, THRESHOLD_RULE_FIXED)

CALIBRATION_METHOD_PLATT = "platt"

VERDICT_SAME = "same_author"
VERDICT_DIFFERENT = "different_author"
VERDICT_ABSTAIN = "abstain"

ABSTAIN_UNCERTAIN = "uncertain"
ABSTAIN_INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class VerifierFeature(PortableModel):
    """One model feature with its training-fitted normalization state."""

    feature_id: str
    semantic_version: str
    metric: str
    mean: PortableFloat
    scale: PortableFloat

    @model_validator(mode="after")
    def _check(self) -> VerifierFeature:
        if self.scale <= 0.0:
            raise ValueError("verifier feature scale must be > 0")
        return self


class VerifierThresholds(PortableModel):
    """Two-threshold abstention band on the decision score."""

    t_same: PortableFloat
    t_diff: PortableFloat

    @model_validator(mode="after")
    def _check(self) -> VerifierThresholds:
        if not (0.0 < self.t_diff <= self.t_same < 1.0):
            raise ValueError("thresholds must satisfy 0 < t_diff <= t_same < 1")
        return self


class VerifierCalibration(PortableModel):
    """Platt scaling parameters fitted on a disjoint calibration split.

    A calibrated probability is conditional on the calibration population
    (its prevalence and domain), never a universal real-world prior
    (spec 23.3).
    """

    method: Literal["platt"] = CALIBRATION_METHOD_PLATT
    a: PortableFloat
    b: PortableFloat


class VerifierPairPolicy(PortableModel):
    """The deterministic pair balance policy the training pairs were built under."""

    max_pairs_per_author: int | None = None  # omitted = uncapped
    max_pairs_per_problem: int | None = None  # omitted = uncapped
    negative_positive_ratio: PortableFloat | None = None  # omitted = no class ratio target
    selection_version: str

    @model_validator(mode="after")
    def _check(self) -> VerifierPairPolicy:
        if self.max_pairs_per_author is not None and self.max_pairs_per_author < 1:
            raise ValueError("max_pairs_per_author must be >= 1")
        if self.max_pairs_per_problem is not None and self.max_pairs_per_problem < 1:
            raise ValueError("max_pairs_per_problem must be >= 1")
        if self.negative_positive_ratio is not None and self.negative_positive_ratio <= 0.0:
            raise ValueError("negative_positive_ratio must be > 0")
        return self


class VerifierFitConfig(PortableModel):
    """The explicit fit configuration embedded in a VerifierFit."""

    l2_lambda: PortableFloat
    max_iterations: int
    tolerance: PortableFloat
    min_support_fraction: PortableFloat
    min_class_support_fraction: PortableFloat
    min_pairs: int
    threshold_rule: str
    threshold_alpha: PortableFloat | None = None  # required iff calibration_quantile_band
    threshold_fixed: PortableFloat | None = None  # required iff fixed
    calibration_method: str | None = None  # omitted = uncalibrated
    include_linguistic: bool
    allow_unconstrained_language: bool
    feature_ids: tuple_of(str) | None = None  # explicit ablation subset
    pair_policy: VerifierPairPolicy

    @model_validator(mode="after")
    def _check(self) -> VerifierFitConfig:
        if self.l2_lambda <= 0.0:
            raise ValueError("l2_lambda must be > 0")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be > 0")
        if not (0.0 < self.min_support_fraction <= 1.0):
            raise ValueError("min_support_fraction must be in (0, 1]")
        if not (0.0 < self.min_class_support_fraction <= 1.0):
            raise ValueError("min_class_support_fraction must be in (0, 1]")
        if self.min_pairs < 1:
            raise ValueError("min_pairs must be >= 1")
        if self.threshold_rule not in THRESHOLD_RULES:
            raise ValueError(f"threshold_rule must be one of {THRESHOLD_RULES}")
        if self.threshold_rule == THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND:
            if self.threshold_alpha is None:
                raise ValueError("threshold_alpha is required for calibration_quantile_band")
            if not (0.0 < self.threshold_alpha <= 0.5):
                raise ValueError("threshold_alpha must be in (0, 0.5]")
            if self.threshold_fixed is not None:
                raise ValueError("threshold_fixed must be omitted for calibration_quantile_band")
        else:
            if self.threshold_fixed is None:
                raise ValueError("threshold_fixed is required for the fixed rule")
            if not (0.0 < self.threshold_fixed < 1.0):
                raise ValueError("threshold_fixed must be in (0, 1)")
            if self.threshold_alpha is not None:
                raise ValueError("threshold_alpha must be omitted for the fixed rule")
        if self.calibration_method is not None and self.calibration_method != (
            CALIBRATION_METHOD_PLATT
        ):
            raise ValueError("calibration_method must be 'platt' when present")
        if self.feature_ids is not None:
            ids = list(self.feature_ids)
            if not is_sorted_unique(ids):
                raise ValueError("feature_ids must be sorted by unique feature_id")
        return self


class VerifierEligibility(PortableModel):
    """Audit record of the training-only feature/pair eligibility policy."""

    training_pair_count: int
    eligible_pair_count: int
    candidate_feature_count: int
    selected_feature_count: int

    @model_validator(mode="after")
    def _check(self) -> VerifierEligibility:
        if self.training_pair_count < 0:
            raise ValueError("training_pair_count must be >= 0")
        if not (0 <= self.eligible_pair_count <= self.training_pair_count):
            raise ValueError("eligible_pair_count must be in [0, training_pair_count]")
        if not (0 <= self.selected_feature_count <= self.candidate_feature_count):
            raise ValueError("selected_feature_count must be in [0, candidate_feature_count]")
        return self


class VerifierFit(PortableModel):
    """A fully self-contained fitted pairwise authorship verifier."""

    schema: Literal["stylog.verifier-fit"] = "stylog.verifier-fit"
    schema_version: Literal["0.1.0"] = "0.1.0"

    model_id: str
    model_semantic_version: str
    task: str
    task_version: str
    kind: ArtifactKind
    languages: tuple_of(str)  # sorted; empty = unconstrained (explicit opt-in)
    feature_registry_version: str
    features: tuple_of(VerifierFeature)  # sorted by feature_id
    coefficients: tuple_of(PortableFloat)  # aligned with features
    intercept: PortableFloat
    thresholds: VerifierThresholds
    threshold_rule: str
    calibration: VerifierCalibration | None = None  # omitted = uncalibrated
    fit_config: VerifierFitConfig
    eligibility: VerifierEligibility
    source_manifest_sha256: HexDigest64
    tuning_manifest_sha256: HexDigest64 | None = None  # omitted = no tuning data used
    calibration_manifest_sha256: HexDigest64 | None = None
    runtime: RuntimeSignature
    backend: BackendSignature

    @model_validator(mode="after")
    def _check(self) -> VerifierFit:
        feature_ids = [feature.feature_id for feature in self.features]
        if not is_sorted_unique(feature_ids):
            raise ValueError("features must be sorted by unique feature_id")
        if len(self.coefficients) != len(self.features):
            raise ValueError("coefficients must align with features")
        languages = list(self.languages)
        if not is_sorted_unique(languages):
            raise ValueError("languages must be sorted and unique")
        if not languages and not self.fit_config.allow_unconstrained_language:
            raise ValueError(
                "empty languages requires allow_unconstrained_language in fit_config"
            )
        if self.threshold_rule != self.fit_config.threshold_rule:
            raise ValueError("threshold_rule must match fit_config.threshold_rule")
        if (self.calibration is None) != (self.fit_config.calibration_method is None):
            raise ValueError("calibration must be present iff fit_config.calibration_method is")
        if self.calibration is not None and self.calibration.method != (
            self.fit_config.calibration_method
        ):
            raise ValueError("calibration.method must match fit_config.calibration_method")
        used_calibration_split = (
            self.threshold_rule == THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND
            or self.calibration is not None
        )
        if used_calibration_split != (self.calibration_manifest_sha256 is not None):
            raise ValueError(
                "calibration_manifest_sha256 must be present iff thresholds or "
                "calibration were fitted on a calibration split"
            )
        return self


class Verification(PortableModel):
    """A model-relative authorship decision bound to its evidence and model."""

    schema: Literal["stylog.verification"] = "stylog.verification"
    schema_version: Literal["0.1.0"] = "0.1.0"

    left_ref: str  # readability label; identity is carried by the hashes below
    right_ref: str
    left_fingerprint_sha256: HexDigest64
    right_fingerprint_sha256: HexDigest64
    verifier_id: HexDigest64
    model_id: str
    model_semantic_version: str
    verdict: Literal["same_author", "different_author", "abstain"]
    abstain_reason: Literal["uncertain", "insufficient_evidence"] | None = None
    score: PortableFloat | None = None  # absent on insufficient_evidence
    probability: PortableFloat | None = None  # present iff score and calibrated model
    calibration_method: Literal["platt"] | None = None  # present iff probability
    features_used: int
    features_missing: tuple_of(str) = ()
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _check(self) -> Verification:
        is_abstain = self.verdict == VERDICT_ABSTAIN
        if (self.abstain_reason is not None) != is_abstain:
            raise ValueError("abstain_reason must be present iff verdict is abstain")
        insufficient = is_abstain and self.abstain_reason == ABSTAIN_INSUFFICIENT_EVIDENCE
        if (self.score is None) != insufficient:
            raise ValueError(
                "score must be absent exactly for abstain/insufficient_evidence"
            )
        if self.score is not None and not (0.0 < self.score < 1.0):
            raise ValueError("score must be in (0, 1)")
        if (self.probability is None) != (self.calibration_method is None):
            raise ValueError("calibration_method must be present iff probability is present")
        if self.probability is not None:
            if self.score is None:
                raise ValueError("probability requires a score")
            if not (0.0 < self.probability < 1.0):
                raise ValueError("probability must be in (0, 1)")
        missing = list(self.features_missing)
        if not is_sorted_unique(missing):
            raise ValueError("features_missing must be sorted and unique")
        if missing and not insufficient:
            raise ValueError(
                "features_missing must be empty unless abstain/insufficient_evidence"
            )
        if self.features_used < 0:
            raise ValueError("features_used must be >= 0")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self
