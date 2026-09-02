"""Analysis engine: deterministic analyzer orchestration (functional core).

The engine selects analyzers by artifact kind/language, computes shared parser
facts once, enforces analyzer atomicity (spec 10.3), and assembles the portable
Fingerprint. It performs no I/O, no caching, and no serialization.
"""

from __future__ import annotations

from dataclasses import dataclass

from stylog.analysis.base import Analyzer, AnalyzerOutput, all_status_observations
from stylog.analysis.registry import (
    ANALYZER_CODE_SAMPLE,
    ANALYZER_CODE_SURFACE,
)
from stylog.domain.artifact import (
    ArtifactDescriptor,
    ContentIdentitySha256,
    ContentIdentitySuppressed,
)
from stylog.domain.diagnostic import Diagnostic, DiagnosticSeverity, sort_diagnostics
from stylog.domain.feature import (
    FeatureObservation,
    FeatureStatus,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import AnalyzerSignature
from stylog.parsers import TREE_SITTER_GRAMMARS
from stylog.runtime import AnalysisContext, RuntimeArtifact

TREE_SITTER_LANGUAGES = tuple(TREE_SITTER_GRAMMARS)


@dataclass(frozen=True)
class EngineResult:
    fingerprint: Fingerprint
    internal_error: bool
    facts: object | None  # parser facts for reuse (e.g. embedded extraction)


def to_portable_descriptor(
    artifact: RuntimeArtifact, *, export_content_hashes: bool = True
) -> ArtifactDescriptor:
    if export_content_hashes:
        identity = ContentIdentitySha256(sha256=artifact.content_sha256)
    else:
        identity = ContentIdentitySuppressed()
    return ArtifactDescriptor(
        artifact_id=artifact.artifact_id,
        kind=artifact.kind,
        language=artifact.language,
        encoding=artifact.encoding,
        byte_count=artifact.byte_count,
        character_count=artifact.character_count,
        content_identity=identity,
    )


def _text_analyzers(ctx: AnalysisContext) -> list[Analyzer]:
    from stylog.analysis.text import (
        TextFunctionWordsEnAnalyzer,
        TextLexicalAnalyzer,
        TextSampleAnalyzer,
        TextStructureAnalyzer,
        TextSurfaceAnalyzer,
    )

    analyzers: list[Analyzer] = [
        TextSampleAnalyzer(),
        TextSurfaceAnalyzer(),
        TextLexicalAnalyzer(),
        TextStructureAnalyzer(),
        TextFunctionWordsEnAnalyzer(),
    ]
    nlp = ctx.config.nlp
    if nlp is not None and nlp.enabled and ctx.resources.nlp_model is not None:
        from stylog.analysis.linguistic import LinguisticAnalyzer

        analyzers.append(LinguisticAnalyzer())
    return analyzers


def _code_analyzers(artifact: RuntimeArtifact, ctx: AnalysisContext) -> list[Analyzer]:
    from stylog.analysis.code import CodeSampleAnalyzer, CodeSurfaceAnalyzer

    analyzers: list[Analyzer] = [CodeSampleAnalyzer(), CodeSurfaceAnalyzer()]
    language = artifact.language
    if language == "python":
        from stylog.analysis.python import PythonAstAnalyzer, PythonTokensAnalyzer

        analyzers.extend([PythonTokensAnalyzer(), PythonAstAnalyzer()])
    elif language in TREE_SITTER_LANGUAGES:
        from stylog.analysis.treesitter import TreeSitterAnalyzer

        tree_sitter_analyzer = TreeSitterAnalyzer()
        tree_sitter_analyzer._language = language
        analyzers.append(tree_sitter_analyzer)
    return analyzers


def _analyzer_enabled(analyzer: Analyzer, artifact: RuntimeArtifact, ctx: AnalysisContext) -> bool:
    analysis = ctx.config.analysis
    if artifact.kind.value == "text":
        if not analysis.text.enabled:
            return False
        if analyzer.analyzer_id == "stylog.text.linguistic":
            nlp = ctx.config.nlp
            return bool(nlp and nlp.enabled and ctx.resources.nlp_model is not None)
        return True
    # code artifact
    if not analysis.code.enabled:
        return False
    if analyzer.analyzer_id in (ANALYZER_CODE_SAMPLE, ANALYZER_CODE_SURFACE):
        return True
    if analyzer.analyzer_id.startswith("stylog.code.python"):
        return analysis.code.python.enabled
    if analyzer.analyzer_id == "stylog.code.tree_sitter":
        return analysis.code.tree_sitter.enabled
    return True


def _disabled_output(analyzer: Analyzer) -> AnalyzerOutput:
    observations = all_status_observations(
        analyzer.analyzer_id, analyzer.implementation_version, FeatureStatus.DISABLED
    )
    return AnalyzerOutput(observations=observations)


def _internal_error_output(analyzer: Analyzer, artifact: RuntimeArtifact) -> AnalyzerOutput:
    observations = all_status_observations(
        analyzer.analyzer_id, analyzer.implementation_version, FeatureStatus.UNAVAILABLE
    )
    diagnostic = Diagnostic(
        code="ANALYZER_INTERNAL_ERROR",
        severity=DiagnosticSeverity.ERROR,
        analyzer_id=analyzer.analyzer_id,
        artifact_id=artifact.artifact_id,
    )
    return AnalyzerOutput(observations=observations, diagnostics=(diagnostic,))


def _compute_facts(analyzer_needs: set[str], artifact: RuntimeArtifact, ctx: AnalysisContext):
    if "python_parse" in analyzer_needs:
        from stylog.parsers.python_native import parse_python

        return parse_python(artifact, ctx.config)
    if "tree_sitter_parse" in analyzer_needs:
        from stylog.parsers.tree_sitter import parse_tree_sitter

        return parse_tree_sitter(artifact, ctx)
    return None


def run_analysis(artifact: RuntimeArtifact, ctx: AnalysisContext) -> EngineResult:
    """Run the deterministic analysis for one artifact and build its Fingerprint."""
    if artifact.kind.value == "text":
        analyzers = _text_analyzers(ctx)
    else:
        analyzers = _code_analyzers(artifact, ctx)

    needs = {analyzer.needs for analyzer in analyzers if _analyzer_enabled(analyzer, artifact, ctx)}
    facts = _compute_facts(needs, artifact, ctx)

    observations: list[FeatureObservation] = []
    diagnostics: list[Diagnostic] = []
    signatures: list[AnalyzerSignature] = []
    internal_error = False

    for analyzer in analyzers:
        signatures.append(analyzer.signature(ctx))
        if not _analyzer_enabled(analyzer, artifact, ctx):
            output = _disabled_output(analyzer)
        else:
            try:
                output = analyzer.analyze(artifact, ctx, facts)
                owned = set(analyzer.owned_feature_ids())
                produced = {observation.feature_id for observation in output.observations}
                if produced != owned:
                    raise RuntimeError(
                        f"analyzer {analyzer.analyzer_id} produced {sorted(produced - owned)} "
                        f"extra / missing {sorted(owned - produced)}"
                    )
            except Exception:
                internal_error = True
                output = _internal_error_output(analyzer, artifact)
        observations.extend(output.observations)
        diagnostics.extend(output.diagnostics)

    fingerprint = Fingerprint(
        artifact=to_portable_descriptor(
            artifact,
            export_content_hashes=ctx.config.analysis.export_content_hashes,
        ),
        runtime=ctx.runtime,
        analysis_config_sha256=ctx.config.analysis_config_sha256(),
        analyzers=tuple(sorted(signatures, key=lambda sig: sig.analyzer_id)),
        features=tuple(sorted(observations, key=lambda observation: observation.feature_id)),
        diagnostics=sort_diagnostics(diagnostics),
    )
    return EngineResult(fingerprint=fingerprint, internal_error=internal_error, facts=facts)
