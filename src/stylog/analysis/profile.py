"""Baseline-relative profiling (spec section 13).

Profiling is population-relative interpretation of profileable scalar
geometries: midrank percentile, type-7 quartiles, MAD, and a robust z-score
against a baseline distribution. No confidence intervals, no infinity under
degenerate baselines.
"""

from __future__ import annotations

from typing import Any

from stylog.analysis import compat, stats
from stylog.analysis.registry import FEATURES
from stylog.domain.baseline import Baseline
from stylog.domain.diagnostic import (
    Diagnostic,
    DiagnosticSeverity,
    make_diagnostic,
    sort_diagnostics,
)
from stylog.domain.feature import OkFeatureObservation
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Profile, ProfileObservation

BASELINE_INCOMPATIBLE = "BASELINE_INCOMPATIBLE"
BASELINE_INSUFFICIENT_SUPPORT = "BASELINE_INSUFFICIENT_SUPPORT"
PROFILE_ZERO_MAD = "PROFILE_ZERO_MAD"


def _observed_scalar(observation: OkFeatureObservation) -> float | None:
    """Profileable geometries only (13.1): integer/float value, ratio value."""
    return compat.primary_scalar(observation.value)


def profile_fingerprint(
    subject: Fingerprint,
    baseline: Baseline,
    subject_ref: str,
) -> Profile:
    """Profile one subject fingerprint against a baseline distribution."""
    observations: list[ProfileObservation] = []
    diagnostics: list[Diagnostic] = []
    for entry in baseline.features:
        feature_id = entry.feature_id
        fdef = FEATURES.get(feature_id)
        if fdef is None:
            diagnostics.append(
                make_diagnostic(
                    BASELINE_INCOMPATIBLE, DiagnosticSeverity.WARNING, feature_id=feature_id
                )
            )
            continue
        observation = compat.observation_for(subject, feature_id)
        if not isinstance(observation, OkFeatureObservation):
            # Feature missingness is data, not an incompatibility.
            continue
        if observation.semantic_version != entry.semantic_version:
            diagnostics.append(
                make_diagnostic(
                    BASELINE_INCOMPATIBLE, DiagnosticSeverity.WARNING, feature_id=feature_id
                )
            )
            continue
        analyzer_sig = compat.analyzer_signature_for(subject, fdef.analyzer_id)
        subject_compatibility = compat.feature_compatibility_sha256(
            fdef, analyzer_sig, subject.runtime
        )
        if (
            subject_compatibility is None
            or subject_compatibility != entry.compatibility_sha256
        ):
            diagnostics.append(
                make_diagnostic(
                    BASELINE_INCOMPATIBLE, DiagnosticSeverity.WARNING, feature_id=feature_id
                )
            )
            continue
        observed = _observed_scalar(observation)
        if observed is None:
            continue
        baseline_n = len(entry.values)
        # Percentiles, type-7 quantiles, and MAD are defined for every
        # non-empty distribution. Only the empty case is rejected; callers
        # receive the exact sample size in ProfileObservation.baseline_n.
        if baseline_n == 0:
            diagnostics.append(
                make_diagnostic(
                    BASELINE_INSUFFICIENT_SUPPORT, DiagnosticSeverity.WARNING, feature_id=feature_id
                )
            )
            continue
        values = list(entry.values)  # validated ascending
        percentile = stats.midrank_percentile(values, observed)
        q25 = stats.quantile_type7(values, 0.25)
        median = stats.quantile_type7(values, 0.5)
        q75 = stats.quantile_type7(values, 0.75)
        mad_raw, mad_normal_scaled = stats.median_absolute_deviation(values)
        robust_z: float | None = None
        if mad_raw > 0.0:
            robust_z = (observed - median) / mad_normal_scaled
        else:
            # Zero-MAD rule (13.6): omit robust_z whether or not observed ==
            # median; never emit infinity.
            diagnostics.append(
                make_diagnostic(PROFILE_ZERO_MAD, DiagnosticSeverity.INFO, feature_id=feature_id)
            )
        fields: dict[str, Any] = {
            "feature_id": feature_id,
            "feature_semantic_version": entry.semantic_version,
            "baseline_n": baseline_n,
            "observed_value": observed,
            "percentile_midrank": percentile,
            "median": median,
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "mad_raw": mad_raw,
            "mad_normal_scaled": mad_normal_scaled,
        }
        # robust_z is omitted entirely under the zero-MAD rule.
        if robust_z is not None:
            fields["robust_z"] = robust_z
        observations.append(ProfileObservation(**fields))
    return Profile(
        subject_ref=subject_ref,
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.baseline_version,
        observations=tuple(observations),
        diagnostics=sort_diagnostics(diagnostics),
    )
