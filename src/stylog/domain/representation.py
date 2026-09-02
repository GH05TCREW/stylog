"""Representation and RepresentationFit portable models (spec 5.18-5.19).

A Representation is a sparse/dense model-space vector. It is never a
Fingerprint feature and never contains raw source text.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from stylog.domain._base import (
    HexDigest64,
    PortableFloat,
    PortableModel,
    is_sorted_unique,
    tuple_of,
)
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key
from stylog.domain.provenance import BackendSignature


class SparseCoordinate(PortableModel):
    index: int
    value: PortableFloat

    @model_validator(mode="after")
    def _check(self) -> SparseCoordinate:
        if self.index < 0:
            raise ValueError("sparse index must be nonnegative")
        return self


class SparseVectorValue(PortableModel):
    kind: Literal["sparse"] = "sparse"
    dimension: int
    entries: tuple_of(SparseCoordinate)

    @model_validator(mode="after")
    def _check(self) -> SparseVectorValue:
        if self.dimension < 0:
            raise ValueError("dimension must be nonnegative")
        indices = [entry.index for entry in self.entries]
        if not is_sorted_unique(indices):
            raise ValueError("sparse entries must be sorted by unique index")
        if any(index >= self.dimension for index in indices):
            raise ValueError("sparse index out of dimension range")
        if any(entry.value == 0.0 for entry in self.entries):
            raise ValueError("sparse zero values must be omitted")
        return self


class DenseVectorValue(PortableModel):
    kind: Literal["dense"] = "dense"
    values: tuple_of(PortableFloat)


RepresentationValue = Annotated[
    SparseVectorValue | DenseVectorValue,
    Field(discriminator="kind"),
]


class RepresentationResourceSignature(PortableModel):
    resource_id: str
    resource_version: str
    sha256: HexDigest64


class Representation(PortableModel):
    schema: Literal["stylog.representation"] = "stylog.representation"
    schema_version: Literal["0.1.0"] = "0.1.0"
    subject_ref: str
    representation_id: str
    semantic_version: str
    preprocessing_version: str
    fit_id: str | None = None  # omitted for fit-free representations
    backend: BackendSignature
    resources: tuple_of(RepresentationResourceSignature) = ()
    value: RepresentationValue
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _sorted(self) -> Representation:
        ids = [resource.resource_id for resource in self.resources]
        if not is_sorted_unique(ids):
            raise ValueError("representation resources must be sorted by unique resource_id")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self


class RepresentationFit(PortableModel):
    schema: Literal["stylog.representation-fit"] = "stylog.representation-fit"
    schema_version: Literal["0.1.0"] = "0.1.0"
    fit_id: str
    representation_id: str
    representation_semantic_version: str
    source_manifest_sha256: HexDigest64
    fit_config_sha256: HexDigest64
    state_resource: RepresentationResourceSignature
    backend: BackendSignature
