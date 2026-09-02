"""Verification scoring core (spec section 23).

Pure functions over a ``VerifierFit`` model and two fingerprints: model
compatibility gates, evidence alignment to the model's ordered features,
deterministic decision scoring, and the threshold-band decision. No I/O, no
NumPy/BLAS — only ``math``. The single non-analysis import is the canonical
hashing helper (the accepted exception shared with ``analysis/compat.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stylog.analysis.compare import compare_fingerprints
from stylog.analysis.compat import observation_for
from stylog.analysis.registry import (
    FEATURE_REGISTRY_VERSION,
    FEATURE_SEMANTIC_VERSION,
    FEATURES,
)
from stylog.domain.diagnostic import Diagnostic, sort_diagnostics
from stylog.domain.feature import OkFeatureObservation
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Comparison
from stylog.domain.verification import (
    ABSTAIN_INSUFFICIENT_EVIDENCE,
    ABSTAIN_UNCERTAIN,
    VERDICT_ABSTAIN,
    VERDICT_DIFFERENT,
    VERDICT_SAME,
    Verification,
    VerifierFit,
)
from stylog.exceptions import (
    CapabilityUnavailableError,
    InternalStylogError,
    ModelIncompatibilityError,
)
from stylog.serialization.canonical import scientific_sha256

VERIFIER_MODEL_ID = "stylog.verifier.logreg/1"
VERIFIER_MODEL_SEMANTIC_VERSION = "1.0.0"

# Exponent arguments are clamped to this magnitude before math.exp. Realistic
# distances are bounded (JSD2/ABS ratios in [0,1], W1 bounded by top codes),
# so the clamp is only reachable for pathological models; it keeps the sigmoid
# defined for every finite input without changing any realistic score.
_LOGIT_CLAMP = 700.0


@dataclass(frozen=True)
class AlignedEvidence:
    """Model features decomposed into a value vector vs missing slots."""

    values: tuple[float, ...]  # one per model feature, model order; empty unless complete
    missing: tuple[str, ...]  # sorted feature ids without comparable evidence
    diagnostics: tuple[Diagnostic, ...]  # comparison diagnostics for missing features

    @property
    def complete(self) -> bool:
        return not self.missing


def clamp_logit(logit: float) -> float:
    """Clamp a logit to the documented +/-700 range (spec 23.10)."""
    return max(-_LOGIT_CLAMP, min(_LOGIT_CLAMP, logit))


def sigmoid_open(logit: float) -> float:
    """Logistic sigmoid, overflow-safe, result kept inside the open interval (0, 1).

    Extreme finite logits round to exactly 0.0/1.0 in IEEE arithmetic; the
    result is then moved one representable step inward so the portable
    contract's strict ``0 < score < 1`` invariant always holds.
    """
    if logit >= 0.0:
        result = 1.0 / (1.0 + math.exp(-min(logit, _LOGIT_CLAMP)))
        if result == 1.0:
            return math.nextafter(1.0, 0.0)
        return result
    result_denom = math.exp(max(logit, -_LOGIT_CLAMP))
    result = result_denom / (1.0 + result_denom)
    if result == 0.0:
        return math.nextafter(0.0, 1.0)
    return result


def check_model_compatibility(
    model: VerifierFit, left: Fingerprint, right: Fingerprint
) -> None:
    """Hard model gates (spec 23.6/23.17); failures are typed, never verdicts."""
    if model.model_id != VERIFIER_MODEL_ID:
        raise ModelIncompatibilityError(
            f"unsupported verifier model id {model.model_id!r} "
            f"(this runtime implements {VERIFIER_MODEL_ID!r})"
        )
    if model.backend.scientific_compatibility_id != VERIFIER_MODEL_ID:
        raise ModelIncompatibilityError(
            f"verifier scientific compatibility id "
            f"{model.backend.scientific_compatibility_id!r} does not match "
            f"{VERIFIER_MODEL_ID!r}"
        )
    if model.feature_registry_version != FEATURE_REGISTRY_VERSION:
        raise ModelIncompatibilityError(
            f"verifier feature registry version {model.feature_registry_version!r} "
            f"does not match runtime {FEATURE_REGISTRY_VERSION!r}"
        )
    for feature in model.features:
        fdef = FEATURES.get(feature.feature_id)
        if fdef is None:
            raise ModelIncompatibilityError(
                f"verifier feature {feature.feature_id!r} is not in the runtime registry"
            )
        if feature.semantic_version != FEATURE_SEMANTIC_VERSION:
            raise ModelIncompatibilityError(
                f"verifier feature {feature.feature_id!r} semantic version "
                f"{feature.semantic_version!r} does not match runtime "
                f"{FEATURE_SEMANTIC_VERSION!r}"
            )
        if feature.metric != fdef.metric:
            raise ModelIncompatibilityError(
                f"verifier feature {feature.feature_id!r} metric {feature.metric!r} "
                f"does not match registry metric {fdef.metric!r}"
            )
    if left.artifact.kind != model.kind or right.artifact.kind != model.kind:
        raise ModelIncompatibilityError(
            f"verifier kind {model.kind!r} does not match artifact kinds "
            f"{left.artifact.kind!r}/{right.artifact.kind!r}"
        )
    if model.languages:
        for side, fp in (("left", left), ("right", right)):
            if fp.artifact.language not in model.languages:
                raise ModelIncompatibilityError(
                    f"{side} artifact language {fp.artifact.language!r} is outside "
                    f"the verifier language scope {sorted(model.languages)!r}"
                )


def align_evidence(
    model: VerifierFit,
    left: Fingerprint,
    right: Fingerprint,
    comparison: Comparison,
) -> AlignedEvidence:
    """Align comparison components to the model's ordered features (spec 23.5).

    A model feature absent from a fingerprint entirely is a capability gap
    (typed error, never a reduced model). A feature present but non-ok on a
    side, dropped by the pairwise compatibility gate, or measured under a
    different semantic version is a missing slot.
    """
    components = {
        component.feature_id: component
        for family in comparison.families
        for component in family.components
    }
    values: list[float] = []
    missing: list[str] = []
    diagnostics: list[Diagnostic] = []
    for feature in model.features:
        feature_id = feature.feature_id
        left_observation = observation_for(left, feature_id)
        right_observation = observation_for(right, feature_id)
        if left_observation is None or right_observation is None:
            sides = [
                side
                for side, observation in (("left", left_observation), ("right", right_observation))
                if observation is None
            ]
            raise CapabilityUnavailableError(
                f"verifier feature {feature_id!r} is absent from the "
                f"{' and '.join(sides)} fingerprint(s); the fingerprints do not "
                f"contain the capabilities this model requires"
            )
        if not isinstance(left_observation, OkFeatureObservation) or not isinstance(
            right_observation, OkFeatureObservation
        ):
            missing.append(feature_id)
            continue
        component = components.get(feature_id)
        if component is None or component.semantic_version != feature.semantic_version:
            missing.append(feature_id)
            diagnostics.extend(
                diagnostic
                for diagnostic in comparison.diagnostics
                if diagnostic.feature_id == feature_id
            )
            continue
        values.append(component.value)
    return AlignedEvidence(
        values=tuple(values) if not missing else (),
        missing=tuple(missing),
        diagnostics=sort_diagnostics(diagnostics),
    )


def decision_score(model: VerifierFit, values: tuple[float, ...]) -> tuple[float, float]:
    """(score, logit): sigmoid(intercept + w·z) over standardized values.

    ``z_i = (x_i - mean_i) / scale_i`` per model feature in model order; the
    logit is ``math.fsum`` over ``[intercept] + [w_i * z_i ...]`` in that fixed
    order, clamped to +/-700 before the sigmoid. The returned logit is the
    clamped one (the clamp is unreachable for realistic bounded distances).
    """
    if len(values) != len(model.features):
        raise InternalStylogError(
            f"evidence vector length {len(values)} does not match model feature "
            f"count {len(model.features)}"
        )
    terms = [model.intercept]
    for coefficient, feature, value in zip(
        model.coefficients, model.features, values, strict=True
    ):
        z = (value - feature.mean) / feature.scale
        terms.append(coefficient * z)
    logit = math.fsum(terms)
    if math.isnan(logit):
        raise InternalStylogError("verifier logit is NaN")
    logit = clamp_logit(logit)
    return sigmoid_open(logit), logit


def decide(model: VerifierFit, score: float) -> str:
    """Threshold-band decision (spec 23.4); same-check first for collapse determinism."""
    if score >= model.thresholds.t_same:
        return VERDICT_SAME
    if score <= model.thresholds.t_diff:
        return VERDICT_DIFFERENT
    return VERDICT_ABSTAIN


def verify_fingerprints(
    model: VerifierFit,
    left: Fingerprint,
    right: Fingerprint,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Verification:
    """Verify two fingerprints under an explicit fitted model (spec 23).

    Gates -> descriptive comparison -> alignment -> score -> decide. Every
    emitted Verification binds the scientific hashes of both input
    fingerprints and of the complete model.
    """
    check_model_compatibility(model, left, right)
    comparison = compare_fingerprints(left, right, left_ref, right_ref)
    aligned = align_evidence(model, left, right, comparison)
    base = {
        "left_ref": left_ref,
        "right_ref": right_ref,
        "left_fingerprint_sha256": scientific_sha256(left),
        "right_fingerprint_sha256": scientific_sha256(right),
        "verifier_id": scientific_sha256(model),
        "model_id": model.model_id,
        "model_semantic_version": model.model_semantic_version,
        "features_used": len(model.features) - len(aligned.missing),
        "diagnostics": aligned.diagnostics,
    }
    if not aligned.complete:
        return Verification(
            **base,
            verdict=VERDICT_ABSTAIN,
            abstain_reason=ABSTAIN_INSUFFICIENT_EVIDENCE,
            features_missing=aligned.missing,
        )
    score, logit = decision_score(model, aligned.values)
    verdict = decide(model, score)
    kwargs: dict[str, object] = {**base, "verdict": verdict, "score": score}
    if verdict == VERDICT_ABSTAIN:
        kwargs["abstain_reason"] = ABSTAIN_UNCERTAIN
    if model.calibration is not None:
        kwargs["probability"] = sigmoid_open(model.calibration.a * logit + model.calibration.b)
        kwargs["calibration_method"] = model.calibration.method
    return Verification(**kwargs)
