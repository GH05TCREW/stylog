"""Feature-level compatibility signatures and pairwise comparability gates.

A feature compatibility tree captures exactly the provenance that makes two
observations of the same feature scientifically comparable (spec 11.2, 12.2,
13.12): feature identity and semantic version, the backend's scientific
compatibility id, the signatures of the resources the feature depends on, and
the runtime signature for runtime-sensitive features. Hashing the canonical
tree yields the ``compatibility_sha256`` stored in baseline feature entries.
"""

from __future__ import annotations

from typing import Any

from stylog.analysis.registry import FEATURE_SEMANTIC_VERSION, FeatureDef
from stylog.domain.feature import (
    FeatureObservation,
    FeatureValue,
    FloatValue,
    IntegerValue,
    RatioValue,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import AnalyzerSignature, RuntimeSignature
from stylog.serialization.canonical import sha256_of_tree

FEATURE_SEMANTIC_MISMATCH = "FEATURE_SEMANTIC_MISMATCH"
RESOURCE_SIGNATURE_MISMATCH = "RESOURCE_SIGNATURE_MISMATCH"
RUNTIME_SIGNATURE_MISMATCH = "RUNTIME_SIGNATURE_MISMATCH"


def observation_for(fp: Fingerprint, feature_id: str) -> FeatureObservation | None:
    """Return the observation of ``feature_id`` in ``fp``, or None when absent."""
    for observation in fp.features:
        if observation.feature_id == feature_id:
            return observation
    return None


def primary_scalar(value: FeatureValue) -> float | None:
    """Primary scalar of an integer/float/ratio value; None for other geometries."""
    if isinstance(value, (IntegerValue, FloatValue, RatioValue)):
        return float(value.value)
    return None


def analyzer_signature_for(fp: Fingerprint, analyzer_id: str) -> AnalyzerSignature | None:
    """Return the analyzer signature for ``analyzer_id`` in ``fp``, or None."""
    for analyzer in fp.analyzers:
        if analyzer.analyzer_id == analyzer_id:
            return analyzer
    return None


def _resource_map(analyzer_sig: AnalyzerSignature) -> dict[str, Any]:
    return {resource.id: resource for resource in analyzer_sig.resources}


def feature_compatibility_dict(
    fdef: FeatureDef,
    analyzer_sig: AnalyzerSignature | None,
    runtime: RuntimeSignature,
) -> dict[str, Any] | None:
    """Build the compatibility tree for one feature observation context.

    Returns None when compatibility is unsatisfiable: no analyzer signature,
    or a resource the feature requires is absent from that signature.
    """
    if analyzer_sig is None:
        return None
    resources_by_id = _resource_map(analyzer_sig)
    resources: list[dict[str, str]] = []
    for resource_id in fdef.resource_ids:
        signature = resources_by_id.get(resource_id)
        if signature is None:
            return None
        resources.append(
            {"id": signature.id, "version": signature.version, "sha256": signature.sha256}
        )
    resources.sort(key=lambda entry: entry["id"])
    backend = getattr(analyzer_sig, "backend", None)
    backend_compatibility_id = (
        backend.scientific_compatibility_id
        if backend is not None
        else f"{analyzer_sig.analyzer_id}/{analyzer_sig.implementation_version}"
    )
    tree: dict[str, Any] = {
        "feature_id": fdef.feature_id,
        "semantic_version": FEATURE_SEMANTIC_VERSION,
        "backend_compatibility_id": backend_compatibility_id,
        "resources": resources,
    }
    if fdef.runtime_sensitive:
        tree["runtime"] = runtime.model_dump(mode="json")
    return tree


def feature_compatibility_sha256(
    fdef: FeatureDef,
    analyzer_sig: AnalyzerSignature | None,
    runtime: RuntimeSignature,
) -> str | None:
    """SHA-256 over the canonical compatibility tree; None when unsatisfiable."""
    tree = feature_compatibility_dict(fdef, analyzer_sig, runtime)
    if tree is None:
        return None
    return sha256_of_tree(tree)


def observation_pair_mismatch(
    fdef: FeatureDef,
    left_fp: Fingerprint,
    right_fp: Fingerprint,
) -> str | None:
    """Gate two fingerprints' observations of ``fdef`` (spec 11.2/12.2).

    Returns None when the observations are comparable; otherwise the stable
    diagnostic code for the first violated condition.
    """
    left_observation = observation_for(left_fp, fdef.feature_id)
    right_observation = observation_for(right_fp, fdef.feature_id)
    if left_observation is None or right_observation is None:
        return FEATURE_SEMANTIC_MISMATCH
    if left_observation.semantic_version != right_observation.semantic_version:
        return FEATURE_SEMANTIC_MISMATCH
    left_analyzer = analyzer_signature_for(left_fp, fdef.analyzer_id)
    right_analyzer = analyzer_signature_for(right_fp, fdef.analyzer_id)
    if fdef.resource_ids:
        if left_analyzer is None or right_analyzer is None:
            return RESOURCE_SIGNATURE_MISMATCH
        left_resources = _resource_map(left_analyzer)
        right_resources = _resource_map(right_analyzer)
        for resource_id in fdef.resource_ids:
            left_resource = left_resources.get(resource_id)
            right_resource = right_resources.get(resource_id)
            if left_resource is None or right_resource is None:
                return RESOURCE_SIGNATURE_MISMATCH
            if (left_resource.version, left_resource.sha256) != (
                right_resource.version,
                right_resource.sha256,
            ):
                return RESOURCE_SIGNATURE_MISMATCH
    if fdef.runtime_sensitive and left_fp.runtime != right_fp.runtime:
        return RUNTIME_SIGNATURE_MISMATCH
    if left_analyzer is None or right_analyzer is None:
        return FEATURE_SEMANTIC_MISMATCH
    if (
        left_analyzer.backend.scientific_compatibility_id
        != right_analyzer.backend.scientific_compatibility_id
    ):
        return FEATURE_SEMANTIC_MISMATCH
    return None
