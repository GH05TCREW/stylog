"""Fingerprint use case, with the deterministic content-addressed cache."""

from __future__ import annotations

from dataclasses import dataclass

from stylog.analysis import engine
from stylog.domain.artifact import ContentIdentitySha256, ContentIdentitySuppressed
from stylog.domain.diagnostic import Diagnostic, DiagnosticSeverity
from stylog.domain.fingerprint import Fingerprint
from stylog.exceptions import PortableArtifactError
from stylog.infrastructure.cache import fingerprint_cache_key
from stylog.ports import RuntimeServices
from stylog.runtime import AnalysisContext, RuntimeArtifact
from stylog.serialization.jsonio import model_from_bytes


@dataclass(frozen=True)
class FingerprintResult:
    fingerprint: Fingerprint
    cache_hit: bool
    internal_error: bool
    warnings: tuple[Diagnostic, ...] = ()  # operational; never inside the portable object


def _cache_key(artifact: RuntimeArtifact, ctx: AnalysisContext) -> str:
    if artifact.kind.value == "text":
        analyzers = engine._text_analyzers(ctx)
    else:
        analyzers = engine._code_analyzers(artifact, ctx)
    signatures = [analyzer.signature(ctx) for analyzer in analyzers]
    analyzer_ids_versions = tuple(
        (sig.analyzer_id, sig.implementation_version) for sig in signatures
    )
    resource_sigs: list[tuple[str, str, str]] = []
    for sig in signatures:
        for resource in sig.resources:
            resource_sigs.append((resource.id, resource.version, resource.sha256))
        for resource in sig.backend.resources:
            resource_sigs.append((resource.id, resource.version, resource.sha256))
    runtime = ctx.runtime
    runtime_fields = (
        ("python_cache_tag", runtime.python_cache_tag),
        ("python_implementation", runtime.python_implementation),
        ("python_version", runtime.python_version),
        ("unicode_database_version", runtime.unicode_database_version),
    )
    return fingerprint_cache_key(
        content_sha256=artifact.content_sha256,
        kind=artifact.kind.value,
        language=artifact.language,
        analysis_config_sha256=ctx.config.analysis_config_sha256(),
        schema_version="0.1.0",
        analyzer_ids_versions=analyzer_ids_versions,
        resource_signatures=tuple(resource_sigs),
        runtime_fields=runtime_fields,
    )


def suppress_content_identity(fingerprint: Fingerprint) -> Fingerprint:
    """Export variant with suppressed content identity (spec 22.3)."""
    descriptor = fingerprint.artifact.model_copy(
        update={"content_identity": ContentIdentitySuppressed()}
    )
    return fingerprint.model_copy(update={"artifact": descriptor})


def fingerprint_artifact(
    artifact: RuntimeArtifact,
    *,
    config,
    services: RuntimeServices,
    ctx: AnalysisContext | None,
    no_cache: bool = False,
    refresh: bool = False,
) -> FingerprintResult:
    """The single fingerprint orchestration path used by API and CLI."""
    if ctx is None:
        from stylog.bootstrap import build_context

        ctx = build_context(config, services)

    export_hashes = config.analysis.export_content_hashes
    # The cache always stores the full content-hash object; suppression is an
    # export-only concern and never changes the internal key (spec 17.3, 22.3).
    key = _cache_key(artifact, ctx)
    cache_diagnostics: list[Diagnostic] = []

    if not no_cache and not refresh:
        try:
            cached = services.cache.get(key)
        except Exception:
            cached = None
            cache_diagnostics.append(
                Diagnostic(code="CACHE_READ_FAILED", severity=DiagnosticSeverity.WARNING)
            )
        if cached is not None:
            try:
                fingerprint = model_from_bytes(cached, Fingerprint)
            except PortableArtifactError:
                cache_diagnostics.append(
                    Diagnostic(code="CACHE_CORRUPT", severity=DiagnosticSeverity.WARNING)
                )
                remove = getattr(services.cache, "remove", None)
                if callable(remove):
                    remove(key)
            else:
                # The cache key is content+scientific-config identity, not the
                # artifact instance. artifact_id is instance metadata (spec
                # 18.1): rewrite it to the requesting artifact so duplicate-
                # content artifacts never receive a foreign instance id.
                if fingerprint.artifact.artifact_id != artifact.artifact_id:
                    fingerprint = fingerprint.model_copy(
                        update={
                            "artifact": fingerprint.artifact.model_copy(
                                update={"artifact_id": artifact.artifact_id}
                            )
                        }
                    )
                if not export_hashes:
                    fingerprint = suppress_content_identity(fingerprint)
                return FingerprintResult(
                    fingerprint=fingerprint, cache_hit=True, internal_error=False
                )

    result = engine.run_analysis(artifact, ctx)
    fingerprint = result.fingerprint

    if not no_cache:
        from stylog.serialization.canonical import file_bytes

        # Store the unsuppressed canonical object regardless of export mode.
        store_fp = fingerprint
        if not export_hashes:
            store_fp = fingerprint.model_copy(
                update={
                    "artifact": fingerprint.artifact.model_copy(
                        update={
                            "content_identity": ContentIdentitySha256(
                                sha256=artifact.content_sha256
                            )
                        }
                    )
                }
            )
        try:
            services.cache.put(key, file_bytes(store_fp))
        except Exception:
            cache_diagnostics.append(
                Diagnostic(code="CACHE_WRITE_FAILED", severity=DiagnosticSeverity.WARNING)
            )

    # Cache diagnostics are operational metadata (spec 14.9): they travel in
    # the result's warning channel, never inside the portable fingerprint.
    return FingerprintResult(
        fingerprint=fingerprint,
        cache_hit=False,
        internal_error=result.internal_error,
        warnings=tuple(cache_diagnostics),
    )
