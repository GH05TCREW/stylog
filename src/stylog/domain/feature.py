"""Typed feature values, support, and observation variants (spec 5.7-5.9)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from stylog.domain._base import PortableFloat, PortableModel, is_sorted_unique, tuple_of

RATIO_VALIDATION_TOLERANCE = 1e-12


class IntegerValue(PortableModel):
    kind: Literal["integer"] = "integer"
    value: int


class FloatValue(PortableModel):
    kind: Literal["float"] = "float"
    value: PortableFloat


class RatioValue(PortableModel):
    kind: Literal["ratio"] = "ratio"
    numerator: int
    denominator: int
    multiplier: PortableFloat
    value: PortableFloat

    @model_validator(mode="after")
    def _check_ratio(self) -> RatioValue:
        if self.denominator <= 0:
            raise ValueError("ratio denominator must be > 0")
        expected = (self.numerator / self.denominator) * self.multiplier
        if abs(self.value - expected) > RATIO_VALIDATION_TOLERANCE:
            raise ValueError("ratio value inconsistent with numerator/denominator/multiplier")
        return self


class CategoryCount(PortableModel):
    key: str
    count: int

    @model_validator(mode="after")
    def _check_count(self) -> CategoryCount:
        if self.count < 0:
            raise ValueError("category count must be >= 0")
        return self


class CategoricalDistributionValue(PortableModel):
    kind: Literal["categorical_distribution"] = "categorical_distribution"
    counts: tuple_of(CategoryCount)
    total: int

    @model_validator(mode="after")
    def _check_distribution(self) -> CategoricalDistributionValue:
        keys = [entry.key for entry in self.counts]
        if not is_sorted_unique(keys):
            raise ValueError("category counts must be sorted by unique key")
        if any(entry.count == 0 for entry in self.counts):
            raise ValueError("zero-count categories must be omitted")
        if self.total <= 0:
            raise ValueError("categorical total must be > 0")
        if sum(entry.count for entry in self.counts) != self.total:
            raise ValueError("category counts must sum to total")
        return self


class PointCount(PortableModel):
    point: int
    count: int

    @model_validator(mode="after")
    def _check(self) -> PointCount:
        if self.point < 0:
            raise ValueError("histogram point must be >= 0")
        if self.count <= 0:
            raise ValueError("histogram counts must be positive")
        return self


class OrderedHistogramValue(PortableModel):
    kind: Literal["ordered_histogram"] = "ordered_histogram"
    points: tuple_of(PointCount)
    total: int
    top_code: int

    @model_validator(mode="after")
    def _check_histogram(self) -> OrderedHistogramValue:
        pts = [entry.point for entry in self.points]
        if not is_sorted_unique(pts):
            raise ValueError("histogram points must be sorted ascending and unique")
        if self.top_code < 0:
            raise ValueError("top_code must be >= 0")
        if any(point > self.top_code for point in pts):
            raise ValueError("histogram point exceeds top_code")
        if self.total <= 0:
            raise ValueError("histogram total must be > 0")
        if sum(entry.count for entry in self.points) != self.total:
            raise ValueError("histogram counts must sum to total")
        return self


class SummaryStatisticsValue(PortableModel):
    kind: Literal["summary"] = "summary"
    n: int
    minimum: PortableFloat
    q25: PortableFloat
    median: PortableFloat
    q75: PortableFloat
    maximum: PortableFloat
    mean: PortableFloat
    sample_sd: PortableFloat | None = None  # omitted when n == 1

    @model_validator(mode="after")
    def _check_summary(self) -> SummaryStatisticsValue:
        if self.n < 1:
            raise ValueError("summary n must be >= 1")
        if self.n == 1 and self.sample_sd is not None:
            raise ValueError("sample_sd must be omitted when n == 1")
        if self.n >= 2 and self.sample_sd is None:
            raise ValueError("sample_sd is required when n >= 2")
        return self


FeatureValue = Annotated[
    IntegerValue
    | FloatValue
    | RatioValue
    | CategoricalDistributionValue
    | OrderedHistogramValue
    | SummaryStatisticsValue,
    Field(discriminator="kind"),
]


class Support(PortableModel):
    kind: str
    count: int

    @model_validator(mode="after")
    def _check_support(self) -> Support:
        if self.count < 0:
            raise ValueError("support count must be >= 0")
        return self


class FeatureStatus(StrEnum):
    OK = "ok"
    INSUFFICIENT_SUPPORT = "insufficient_support"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    PARSER_ERROR = "parser_error"
    DISABLED = "disabled"


class _ObservationBase(PortableModel):
    feature_id: str
    semantic_version: str
    analyzer_id: str
    analyzer_implementation_version: str


class OkFeatureObservation(_ObservationBase):
    status: Literal["ok"] = "ok"
    value: FeatureValue
    support: Support


class InsufficientSupportObservation(_ObservationBase):
    status: Literal["insufficient_support"] = "insufficient_support"


class NotApplicableObservation(_ObservationBase):
    status: Literal["not_applicable"] = "not_applicable"


class UnavailableObservation(_ObservationBase):
    status: Literal["unavailable"] = "unavailable"


class ParserErrorObservation(_ObservationBase):
    status: Literal["parser_error"] = "parser_error"


class DisabledObservation(_ObservationBase):
    status: Literal["disabled"] = "disabled"


FeatureObservation = Annotated[
    OkFeatureObservation
    | InsufficientSupportObservation
    | NotApplicableObservation
    | UnavailableObservation
    | ParserErrorObservation
    | DisabledObservation,
    Field(discriminator="status"),
]

OBSERVATION_STATUS_TO_CLASS = {
    "ok": OkFeatureObservation,
    "insufficient_support": InsufficientSupportObservation,
    "not_applicable": NotApplicableObservation,
    "unavailable": UnavailableObservation,
    "parser_error": ParserErrorObservation,
    "disabled": DisabledObservation,
}
