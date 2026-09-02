"""Fingerprint, embedded analysis, and analysis bundle (spec 5.10-5.12)."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from stylog.domain._base import HexDigest64, PortableModel, is_sorted_unique, tuple_of
from stylog.domain.artifact import ArtifactDescriptor, EmbeddedArtifactDescriptor
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key
from stylog.domain.feature import FeatureObservation
from stylog.domain.provenance import AnalyzerSignature, RuntimeSignature


class Fingerprint(PortableModel):
    schema: Literal["stylog.fingerprint"] = "stylog.fingerprint"
    schema_version: Literal["0.1.0"] = "0.1.0"

    artifact: ArtifactDescriptor
    runtime: RuntimeSignature
    analysis_config_sha256: HexDigest64

    analyzers: tuple_of(AnalyzerSignature)
    features: tuple_of(FeatureObservation)
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _canonical_order(self) -> Fingerprint:
        analyzer_ids = [analyzer.analyzer_id for analyzer in self.analyzers]
        if not is_sorted_unique(analyzer_ids):
            raise ValueError("analyzers must be sorted by unique analyzer_id")
        feature_ids = [observation.feature_id for observation in self.features]
        if not is_sorted_unique(feature_ids):
            raise ValueError("features must be sorted by unique feature_id")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self


class EmbeddedAnalysis(PortableModel):
    descriptor: EmbeddedArtifactDescriptor
    fingerprint: Fingerprint


class AnalysisBundle(PortableModel):
    schema: Literal["stylog.analysis"] = "stylog.analysis"
    schema_version: Literal["0.1.0"] = "0.1.0"
    primary: Fingerprint
    embedded: tuple_of(EmbeddedAnalysis) = ()
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _embedded_order(self) -> AnalysisBundle:
        keys = [
            (
                item.descriptor.source_span.start.line,
                item.descriptor.source_span.start.column,
                item.descriptor.embedded_kind,
                item.descriptor.ordinal,
            )
            for item in self.embedded
        ]
        if keys != sorted(keys):
            raise ValueError("embedded analyses must be sorted by span, kind, ordinal")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self
