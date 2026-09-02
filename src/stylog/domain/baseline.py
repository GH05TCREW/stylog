"""Baseline portable model (spec 13.10)."""

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


class BaselineDescriptor(PortableModel):
    kind: str
    language: str
    domain: str
    unit: str  # "artifact" | "evidence_set"
    source: str


class BaselineCompatibility(PortableModel):
    feature_registry_version: str


class BaselineFeature(PortableModel):
    feature_id: str
    semantic_version: str
    compatibility_sha256: HexDigest64
    total_units: int
    values: tuple_of(PortableFloat)  # ascending

    @model_validator(mode="after")
    def _check(self) -> BaselineFeature:
        if self.total_units < 1:
            raise ValueError("total_units must be >= 1")
        values = list(self.values)
        if values != sorted(values):
            raise ValueError("baseline values must be ascending")
        if len(values) > self.total_units:
            raise ValueError("valid values cannot exceed total_units")
        return self


class Baseline(PortableModel):
    schema: Literal["stylog.baseline"] = "stylog.baseline"
    schema_version: Literal["0.1.0"] = "0.1.0"
    baseline_id: str
    baseline_version: str
    descriptor: BaselineDescriptor
    source_manifest_sha256: HexDigest64
    compatibility: BaselineCompatibility
    features: tuple_of(BaselineFeature)

    @model_validator(mode="after")
    def _sorted(self) -> Baseline:
        ids = [feature.feature_id for feature in self.features]
        if not is_sorted_unique(ids):
            raise ValueError("baseline features must be sorted by unique feature_id")
        return self
