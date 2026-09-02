"""Analyzer protocol and shared backend signatures for the scientific core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol

from stylog.analysis.registry import (
    ANALYZER_IMPLEMENTATION_VERSION,
    FEATURE_REGISTRY_VERSION,
    features_owned_by,
)
from stylog.domain.diagnostic import Diagnostic
from stylog.domain.feature import FeatureObservation, FeatureStatus
from stylog.domain.provenance import AnalyzerSignature, BackendSignature, ResourceSignature
from stylog.runtime import AnalysisContext, RuntimeArtifact

# Backend identities for pure-stdlib mechanics. Compatibility IDs change only
# when conformance shows stored values changed.
TEXT_BACKEND = BackendSignature(
    backend_id="stylog.native.text",
    implementation_version="1.0.0",
    scientific_compatibility_id="stylog.text-core/1",
)
CODE_SURFACE_BACKEND = BackendSignature(
    backend_id="stylog.native.code",
    implementation_version="1.0.0",
    scientific_compatibility_id="stylog.code-surface/1",
)
PYTHON_TOKENS_BACKEND = BackendSignature(
    backend_id="cpython.tokenize",
    implementation_version="1.0.0",
    scientific_compatibility_id="stylog.python-native-tokenize/1",
)
PYTHON_AST_BACKEND = BackendSignature(
    backend_id="cpython.ast",
    implementation_version="1.0.0",
    scientific_compatibility_id="stylog.python-native-ast/1",
)


@dataclass(frozen=True)
class AnalyzerOutput:
    observations: tuple[FeatureObservation, ...]
    diagnostics: tuple[Diagnostic, ...] = ()


class Analyzer(Protocol):
    """Deterministic feature analyzer. Owns a fixed set of feature IDs."""

    analyzer_id: str
    implementation_version: str
    needs: str  # "none" | "python_parse" | "tree_sitter_parse"

    def owned_feature_ids(self) -> tuple[str, ...]: ...
    def signature(self, ctx: AnalysisContext) -> AnalyzerSignature: ...
    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput: ...


class BaseAnalyzer:
    """Boilerplate base for function-implemented analyzers."""

    analyzer_id: ClassVar[str]
    implementation_version: ClassVar[str] = ANALYZER_IMPLEMENTATION_VERSION
    needs: ClassVar[str] = "none"
    backend: ClassVar[BackendSignature] = TEXT_BACKEND

    def owned_feature_ids(self) -> tuple[str, ...]:
        return tuple(fdef.feature_id for fdef in features_owned_by(self.analyzer_id))

    def resources(self, ctx: AnalysisContext) -> tuple[ResourceSignature, ...]:
        return ()

    def signature(self, ctx: AnalysisContext) -> AnalyzerSignature:
        return AnalyzerSignature(
            analyzer_id=self.analyzer_id,
            implementation_version=self.implementation_version,
            feature_registry_version=FEATURE_REGISTRY_VERSION,
            backend=self.backend,
            resources=tuple(sorted(self.resources(ctx), key=lambda sig: sig.id)),
        )


def all_status_observations(
    analyzer_id: str,
    implementation_version: str,
    observation_status: FeatureStatus,
) -> tuple[FeatureObservation, ...]:
    """The same status observation for every feature owned by ``analyzer_id``."""
    from stylog.analysis import build

    return tuple(
        build.status(fdef, analyzer_id, implementation_version, observation_status)
        for fdef in features_owned_by(analyzer_id)
    )
