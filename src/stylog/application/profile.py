"""Profile use case and local baseline construction."""

from __future__ import annotations

from collections.abc import Sequence

from stylog.analysis import compat
from stylog.analysis.profile import profile_fingerprint as _profile
from stylog.analysis.registry import FEATURE_REGISTRY_VERSION, FEATURE_SEMANTIC_VERSION, FEATURES
from stylog.domain.baseline import (
    Baseline,
    BaselineCompatibility,
    BaselineDescriptor,
    BaselineFeature,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Profile
from stylog.exceptions import BaselineError
from stylog.ports import RuntimeServices


def profile_subject(
    subject: Fingerprint,
    baseline_ref: str,
    *,
    services: RuntimeServices,
    subject_ref: str = "subject",
) -> Profile:
    baseline = services.baselines.resolve(baseline_ref)
    return _profile(subject, baseline, subject_ref)


def scalar_of_observation(fingerprint: Fingerprint, feature_id: str) -> float | None:
    """Scalar value of an ok profileable observation, else None."""
    observation = compat.observation_for(fingerprint, feature_id)
    if observation is None or observation.status != "ok":
        return None
    return compat.primary_scalar(observation.value)


def build_baseline(
    fingerprints: Sequence[Fingerprint],
    *,
    baseline_id: str,
    baseline_version: str,
    descriptor: BaselineDescriptor,
) -> Baseline:
    """Build a local baseline from analyzed units (spec 13.8, 13.10).

    Each fingerprint is one baseline unit. Only ok profileable observations
    contribute values; every feature records total source units.
    """
    if not fingerprints:
        raise BaselineError("BASELINE_INVALID: cannot build a baseline from zero units")
    from stylog.serialization.canonical import scientific_sha256, sha256_hex

    source_material = "".join(scientific_sha256(fp) for fp in fingerprints)
    source_manifest_sha256 = sha256_hex(source_material.encode("ascii"))

    feature_ids = sorted(
        {
            observation.feature_id
            for fp in fingerprints
            for observation in fp.features
            if observation.feature_id in FEATURES
        }
    )
    features: list[BaselineFeature] = []
    for feature_id in feature_ids:
        fdef = FEATURES[feature_id]
        if fdef.geometry not in ("integer", "float", "ratio"):
            continue
        values: list[float] = []
        for fp in fingerprints:
            scalar = scalar_of_observation(fp, feature_id)
            if scalar is not None:
                values.append(scalar)
        if not values:
            continue
        analyzer_sig = compat.analyzer_signature_for(fingerprints[0], fdef.analyzer_id)
        compatibility = compat.feature_compatibility_sha256(
            fdef, analyzer_sig, fingerprints[0].runtime
        )
        if compatibility is None:
            continue
        features.append(
            BaselineFeature(
                feature_id=feature_id,
                semantic_version=FEATURE_SEMANTIC_VERSION,
                compatibility_sha256=compatibility,
                total_units=len(fingerprints),
                values=tuple(sorted(values)),
            )
        )
    return Baseline(
        baseline_id=baseline_id,
        baseline_version=baseline_version,
        descriptor=descriptor,
        source_manifest_sha256=source_manifest_sha256,
        compatibility=BaselineCompatibility(feature_registry_version=FEATURE_REGISTRY_VERSION),
        features=tuple(features),
    )
