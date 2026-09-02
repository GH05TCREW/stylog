"""EvidenceSet aggregation (spec section 11).

Pure reducer mathematics over fingerprint observations: pool compatible ok
observations per the registry reducer, count every non-ok status individually,
and omit features whose members fail the compatibility gate. No zero
imputation, no dedup of identical content.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from stylog.analysis import build, compat
from stylog.analysis.registry import FEATURES
from stylog.domain.diagnostic import (
    Diagnostic,
    DiagnosticSeverity,
    make_diagnostic,
    sort_diagnostics,
)
from stylog.domain.evidence import (
    AggregateObservation,
    AggregationKind,
    EvidenceAggregate,
    EvidenceSet,
    MissingStatusCount,
)
from stylog.domain.feature import (
    CategoricalDistributionValue,
    FeatureStatus,
    IntegerValue,
    OkFeatureObservation,
    OrderedHistogramValue,
    PointCount,
    RatioValue,
)
from stylog.domain.fingerprint import Fingerprint


def _typed_value(observation: OkFeatureObservation, expected: type) -> object:
    if not isinstance(observation.value, expected):
        raise ValueError(
            f"observation of {observation.feature_id} has geometry "
            f"{observation.value.kind!r}, expected {expected.__name__}"
        )
    return observation.value


def _sample_scalar(observation: OkFeatureObservation) -> float:
    """Reduce one ok sample's primary numeric value to a scalar (spec 11.8)."""
    scalar = compat.primary_scalar(observation.value)
    if scalar is None:
        raise ValueError(
            f"observation of {observation.feature_id} has no primary numeric value"
        )
    return scalar


def aggregate_fingerprints(
    evidence_set: EvidenceSet,
    fingerprints: Sequence[Fingerprint],
) -> EvidenceAggregate:
    """Aggregate the given fingerprints over the feature universe they define."""
    fps = list(fingerprints)
    universe = sorted(
        {
            observation.feature_id
            for fp in fps
            for observation in fp.features
            if observation.feature_id in FEATURES
        }
    )
    aggregates: list[AggregateObservation] = []
    diagnostics: list[Diagnostic] = []
    for feature_id in universe:
        aggregate = _aggregate_feature(feature_id, fps, diagnostics)
        if aggregate is not None:
            aggregates.append(aggregate)
    return EvidenceAggregate(
        evidence_set=evidence_set,
        aggregates=tuple(aggregates),
        diagnostics=sort_diagnostics(diagnostics),
    )


def _aggregate_feature(
    feature_id: str,
    fps: list[Fingerprint],
    diagnostics: list[Diagnostic],
) -> AggregateObservation | None:
    fdef = FEATURES[feature_id]
    present = [
        (fp, observation)
        for fp in fps
        if (observation := compat.observation_for(fp, feature_id)) is not None
    ]

    # Compatibility gate (spec 11.2): every member carrying the observation
    # must agree on semantic version, resource signatures, runtime signature
    # where runtime-sensitive, and backend scientific compatibility id.
    if present:
        reference_fp = present[0][0]
        for other_fp, _ in present[1:]:
            mismatch = compat.observation_pair_mismatch(fdef, reference_fp, other_fp)
            if mismatch is not None:
                diagnostics.append(
                    make_diagnostic(mismatch, DiagnosticSeverity.ERROR, feature_id=feature_id)
                )
                return None

    ok_observations = [
        observation
        for _, observation in present
        if observation.status == FeatureStatus.OK
    ]
    missing_counts: dict[FeatureStatus, int] = {}
    for fp in fps:
        observation = compat.observation_for(fp, feature_id)
        status = (
            FeatureStatus.UNAVAILABLE
            if observation is None
            else FeatureStatus(observation.status)
        )
        if status != FeatureStatus.OK:
            missing_counts[status] = missing_counts.get(status, 0) + 1
    missing = tuple(
        MissingStatusCount(status=status, count=missing_counts[status])
        for status in sorted(missing_counts)
    )

    common_semantic_version = present[0][1].semantic_version
    total_samples = len(fps)
    contributing = len(ok_observations)

    # 11.11: zero contributing samples -> no pooled value, no sample summary.
    if contributing == 0:
        return AggregateObservation(
            feature_id=feature_id,
            semantic_version=common_semantic_version,
            reducer=fdef.reducer,
            total_samples=total_samples,
            contributing_samples=0,
            missing=missing,
        )

    reducer = fdef.reducer
    pooled = None
    sample_summary = None
    sample_values = None

    if reducer is AggregationKind.EXACT_SUM:
        total = sum(
            _typed_value(observation, IntegerValue).value for observation in ok_observations
        )
        pooled = build.int_value(total)

    elif reducer is AggregationKind.RATIO_POOL:
        ratios = [
            _typed_value(observation, RatioValue) for observation in ok_observations
        ]
        multipliers = {ratio.multiplier for ratio in ratios}
        if len(multipliers) != 1:
            # Multipliers must match (11.5); treat as semantic mismatch.
            diagnostics.append(
                make_diagnostic(
                    compat.FEATURE_SEMANTIC_MISMATCH, DiagnosticSeverity.ERROR, feature_id=feature_id
                )
            )
            return None
        pooled = build.ratio_value(
            sum(ratio.numerator for ratio in ratios),
            sum(ratio.denominator for ratio in ratios),
            next(iter(multipliers)),
        )

    elif reducer is AggregationKind.CATEGORICAL_POOL:
        totals: dict[str, int] = {}
        for observation in ok_observations:
            distribution = _typed_value(observation, CategoricalDistributionValue)
            for entry in distribution.counts:
                totals[entry.key] = totals.get(entry.key, 0) + entry.count
        pooled = build.categorical_value(totals)

    elif reducer is AggregationKind.HISTOGRAM_POOL:
        histograms = [
            _typed_value(observation, OrderedHistogramValue)
            for observation in ok_observations
        ]
        top_codes = {histogram.top_code for histogram in histograms}
        if len(top_codes) != 1:
            # top_code MUST match (11.7); treat as semantic mismatch.
            diagnostics.append(
                make_diagnostic(
                    compat.FEATURE_SEMANTIC_MISMATCH, DiagnosticSeverity.ERROR, feature_id=feature_id
                )
            )
            return None
        counts: dict[int, int] = {}
        for histogram in histograms:
            for entry in histogram.points:
                counts[entry.point] = counts.get(entry.point, 0) + entry.count
        pooled = OrderedHistogramValue(
            points=tuple(
                PointCount(point=point, count=counts[point]) for point in sorted(counts)
            ),
            total=sum(counts.values()),
            top_code=next(iter(top_codes)),
        )

    elif reducer is AggregationKind.SAMPLE_SUMMARY:
        sample_values = tuple(
            sorted(_sample_scalar(observation) for observation in ok_observations)
        )
        sample_summary = build.summary_value(sample_values)

    # NOT_AGGREGATABLE: no pooled value, no summary, no sample values (11.9).

    fields: dict[str, Any] = {
        "feature_id": feature_id,
        "semantic_version": common_semantic_version,
        "reducer": reducer,
        "total_samples": total_samples,
        "contributing_samples": contributing,
        "missing": missing,
    }
    # Optional fields are omitted entirely (never passed as explicit null).
    if pooled is not None:
        fields["pooled"] = pooled
    if sample_summary is not None:
        fields["sample_summary"] = sample_summary
    if sample_values is not None:
        fields["sample_values"] = sample_values
    return AggregateObservation(**fields)
