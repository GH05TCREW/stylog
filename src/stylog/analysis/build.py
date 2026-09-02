"""Helpers to construct typed feature values and observations.

All builders enforce canonical ordering and the portable-null policy. Builders
return ``None`` for distributions/histograms over empty event populations;
callers then emit ``insufficient_support``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from stylog.analysis import stats
from stylog.analysis.registry import FeatureDef
from stylog.domain.feature import (
    CategoricalDistributionValue,
    CategoryCount,
    DisabledObservation,
    FeatureObservation,
    FeatureStatus,
    FloatValue,
    InsufficientSupportObservation,
    IntegerValue,
    NotApplicableObservation,
    OkFeatureObservation,
    OrderedHistogramValue,
    ParserErrorObservation,
    PointCount,
    RatioValue,
    SummaryStatisticsValue,
    Support,
    UnavailableObservation,
)

_STATUS_CLASSES = {
    FeatureStatus.INSUFFICIENT_SUPPORT: InsufficientSupportObservation,
    FeatureStatus.NOT_APPLICABLE: NotApplicableObservation,
    FeatureStatus.UNAVAILABLE: UnavailableObservation,
    FeatureStatus.PARSER_ERROR: ParserErrorObservation,
    FeatureStatus.DISABLED: DisabledObservation,
}


def int_value(n: int) -> IntegerValue:
    return IntegerValue(value=int(n))


def float_value(x: float) -> FloatValue:
    return FloatValue(value=float(x))


def ratio_value(numerator: int, denominator: int, multiplier: float = 1.0) -> RatioValue:
    return RatioValue(
        numerator=int(numerator),
        denominator=int(denominator),
        multiplier=float(multiplier),
        value=(numerator / denominator) * multiplier,
    )


def categorical_value(counts: Mapping[str, int]) -> CategoricalDistributionValue | None:
    entries = tuple(
        CategoryCount(key=key, count=count)
        for key, count in sorted(counts.items())
        if count > 0
    )
    total = sum(entry.count for entry in entries)
    if total == 0:
        return None
    return CategoricalDistributionValue(counts=entries, total=total)


def histogram_value(values: Iterable[int], top_code: int) -> OrderedHistogramValue | None:
    counts: dict[int, int] = {}
    for value in values:
        point = min(int(value), top_code)
        counts[point] = counts.get(point, 0) + 1
    if not counts:
        return None
    points = tuple(PointCount(point=point, count=counts[point]) for point in sorted(counts))
    return OrderedHistogramValue(points=points, total=sum(counts.values()), top_code=top_code)


def summary_value(values: Iterable[float]) -> SummaryStatisticsValue:
    n, minimum, q25, median, q75, maximum, mean, sample_sd = stats.summary_statistics(values)
    kwargs: dict[str, Any] = {
        "n": n,
        "minimum": minimum,
        "q25": q25,
        "median": median,
        "q75": q75,
        "maximum": maximum,
        "mean": mean,
    }
    if sample_sd is not None:
        kwargs["sample_sd"] = sample_sd
    return SummaryStatisticsValue(**kwargs)


def ok(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    value: IntegerValue
    | FloatValue
    | RatioValue
    | CategoricalDistributionValue
    | OrderedHistogramValue
    | SummaryStatisticsValue,
    support_count: int,
) -> OkFeatureObservation:
    return OkFeatureObservation(
        feature_id=fdef.feature_id,
        semantic_version=fdef_semantic_version(fdef),
        analyzer_id=analyzer_id,
        analyzer_implementation_version=implementation_version,
        value=value,
        support=Support(kind=fdef.support_kind, count=support_count),
    )


def fdef_semantic_version(fdef: FeatureDef) -> str:
    from stylog.analysis.registry import FEATURE_SEMANTIC_VERSION

    return FEATURE_SEMANTIC_VERSION


def status(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    observation_status: FeatureStatus,
) -> FeatureObservation:
    cls = _STATUS_CLASSES[observation_status]
    return cls(
        feature_id=fdef.feature_id,
        semantic_version=fdef_semantic_version(fdef),
        analyzer_id=analyzer_id,
        analyzer_implementation_version=implementation_version,
    )


def value_observation(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    value: IntegerValue
    | FloatValue
    | RatioValue
    | CategoricalDistributionValue
    | OrderedHistogramValue
    | SummaryStatisticsValue
    | None,
    support_count: int,
) -> FeatureObservation:
    """ok observation over a pre-built value; None -> insufficient_support."""
    if value is None:
        return status(
            fdef, analyzer_id, implementation_version, FeatureStatus.INSUFFICIENT_SUPPORT
        )
    return ok(fdef, analyzer_id, implementation_version, value, support_count)


def categorical_observation(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    counts: Mapping[str, int],
) -> FeatureObservation:
    value = categorical_value(counts)
    if value is None:
        return status(
            fdef, analyzer_id, implementation_version, FeatureStatus.INSUFFICIENT_SUPPORT
        )
    return ok(fdef, analyzer_id, implementation_version, value, value.total)


def histogram_observation(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    values: Iterable[int],
) -> FeatureObservation:
    assert fdef.top_code is not None
    value = histogram_value(values, fdef.top_code)
    if value is None:
        return status(
            fdef, analyzer_id, implementation_version, FeatureStatus.INSUFFICIENT_SUPPORT
        )
    return ok(fdef, analyzer_id, implementation_version, value, value.total)


def ratio_observation(
    fdef: FeatureDef,
    analyzer_id: str,
    implementation_version: str,
    numerator: int,
    denominator: int,
) -> FeatureObservation:
    if denominator == 0:
        return status(
            fdef, analyzer_id, implementation_version, FeatureStatus.INSUFFICIENT_SUPPORT
        )
    value = ratio_value(numerator, denominator)
    return ok(fdef, analyzer_id, implementation_version, value, denominator)
