"""EvidenceSet and aggregation outputs (spec 5.13-5.14, section 11)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import model_validator

from stylog.domain._base import PortableModel, is_sorted_unique, tuple_of
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key
from stylog.domain.feature import FeatureStatus, FeatureValue, SummaryStatisticsValue


class AggregationKind(StrEnum):
    EXACT_SUM = "exact_sum"
    RATIO_POOL = "ratio_pool"
    CATEGORICAL_POOL = "categorical_pool"
    HISTOGRAM_POOL = "histogram_pool"
    SAMPLE_SUMMARY = "sample_summary"
    NOT_AGGREGATABLE = "not_aggregatable"


class EvidenceMember(PortableModel):
    member_id: str
    artifact_id: str


class LinkageDescriptor(PortableModel):
    kind: str
    source: str


class EvidenceSet(PortableModel):
    schema: Literal["stylog.evidence-set"] = "stylog.evidence-set"
    schema_version: Literal["0.1.0"] = "0.1.0"
    evidence_set_id: str
    members: tuple_of(EvidenceMember)
    linkage: LinkageDescriptor

    @model_validator(mode="after")
    def _members_sorted(self) -> EvidenceSet:
        member_ids = [member.member_id for member in self.members]
        if not is_sorted_unique(member_ids):
            raise ValueError("evidence members must be sorted by unique member_id")
        return self


class MissingStatusCount(PortableModel):
    status: FeatureStatus
    count: int

    @model_validator(mode="after")
    def _check_count(self) -> MissingStatusCount:
        if self.count < 0:
            raise ValueError("missing status count must be >= 0")
        return self


class AggregateObservation(PortableModel):
    feature_id: str
    semantic_version: str
    reducer: AggregationKind
    total_samples: int
    contributing_samples: int
    missing: tuple_of(MissingStatusCount) = ()
    pooled: FeatureValue | None = None  # omitted when reducer has no pooled value
    sample_summary: SummaryStatisticsValue | None = None  # omitted when not meaningful
    sample_values: tuple_of(float) | None = None  # only sample-summary reducers

    @model_validator(mode="after")
    def _check(self) -> AggregateObservation:
        if self.total_samples < 0 or self.contributing_samples < 0:
            raise ValueError("sample counts must be >= 0")
        if self.contributing_samples > self.total_samples:
            raise ValueError("contributing_samples cannot exceed total_samples")
        statuses = [entry.status for entry in self.missing]
        if not is_sorted_unique(statuses):
            raise ValueError("missing status counts must be sorted by unique status")
        if sum(entry.count for entry in self.missing) + self.contributing_samples != (
            self.total_samples
        ):
            raise ValueError("missing counts plus contributing must equal total_samples")
        if self.contributing_samples == 0 and (
            self.pooled is not None or self.sample_summary is not None
        ):
            raise ValueError("no contributing samples: pooled/summary must be omitted")
        return self


class EvidenceAggregate(PortableModel):
    """Reducer output over compatible observations from an EvidenceSet."""

    schema: Literal["stylog.evidence-aggregate"] = "stylog.evidence-aggregate"
    schema_version: Literal["0.1.0"] = "0.1.0"
    evidence_set: EvidenceSet
    aggregates: tuple_of(AggregateObservation)
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _aggregates_sorted(self) -> EvidenceAggregate:
        feature_ids = [aggregate.feature_id for aggregate in self.aggregates]
        if not is_sorted_unique(feature_ids):
            raise ValueError("aggregates must be sorted by unique feature_id")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self
