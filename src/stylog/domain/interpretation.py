"""Comparison and Profile portable models (spec 5.15-5.16).

A Comparison is an ordered set of independently interpretable components.
There is deliberately no aggregate similarity field.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from stylog.domain._base import PortableFloat, PortableModel, is_sorted_unique, tuple_of
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key
from stylog.domain.feature import Support


class ComparisonComponent(PortableModel):
    feature_id: str
    semantic_version: str
    metric: str
    value: PortableFloat
    unit: str
    left_support: Support
    right_support: Support


class ComparisonFamily(PortableModel):
    family: str
    components: tuple_of(ComparisonComponent)

    @model_validator(mode="after")
    def _sorted(self) -> ComparisonFamily:
        ids = [component.feature_id for component in self.components]
        if not is_sorted_unique(ids):
            raise ValueError("comparison components must be sorted by unique feature_id")
        return self


class Comparison(PortableModel):
    schema: Literal["stylog.comparison"] = "stylog.comparison"
    schema_version: Literal["0.1.0"] = "0.1.0"
    left_ref: str
    right_ref: str
    families: tuple_of(ComparisonFamily)
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _sorted(self) -> Comparison:
        families = [family.family for family in self.families]
        if not is_sorted_unique(families):
            raise ValueError("comparison families must be sorted by unique family name")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self


class ProfileObservation(PortableModel):
    feature_id: str
    feature_semantic_version: str
    baseline_n: int
    observed_value: PortableFloat
    percentile_midrank: PortableFloat
    median: PortableFloat
    q25: PortableFloat
    q75: PortableFloat
    iqr: PortableFloat
    mad_raw: PortableFloat
    mad_normal_scaled: PortableFloat
    robust_z: PortableFloat | None = None  # omitted under the zero-MAD rule

    @model_validator(mode="after")
    def _check(self) -> ProfileObservation:
        if self.baseline_n < 1:
            raise ValueError("baseline_n must be >= 1")
        return self


class Profile(PortableModel):
    schema: Literal["stylog.profile"] = "stylog.profile"
    schema_version: Literal["0.1.0"] = "0.1.0"
    subject_ref: str
    baseline_id: str
    baseline_version: str
    observations: tuple_of(ProfileObservation)
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _sorted(self) -> Profile:
        ids = [observation.feature_id for observation in self.observations]
        if not is_sorted_unique(ids):
            raise ValueError("profile observations must be sorted by unique feature_id")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self
