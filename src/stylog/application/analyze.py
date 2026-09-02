"""Analyze use case: fingerprint + embedded artifacts -> AnalysisBundle."""

from __future__ import annotations

from stylog.application.fingerprint import fingerprint_artifact
from stylog.domain.artifact import ArtifactKind, EmbeddedArtifactDescriptor
from stylog.domain.fingerprint import AnalysisBundle, EmbeddedAnalysis
from stylog.ports import RuntimeServices
from stylog.runtime import AnalysisContext, RuntimeArtifact
from stylog.serialization.canonical import sha256_hex


def _embedded_runtime_artifact(
    parent: RuntimeArtifact, kind: str, ordinal: int, text: str
) -> RuntimeArtifact:
    raw = text.encode("utf-8")
    return RuntimeArtifact(
        artifact_id=f"{parent.artifact_id}/{kind}/{ordinal:06d}",
        kind=ArtifactKind.TEXT,
        language="und",  # replaced by caller with the configured embedded language
        encoding="utf-8",
        raw_bytes=raw,
        text=text,
        content_sha256=sha256_hex(raw),
    )


def analyze_artifact(
    artifact: RuntimeArtifact,
    *,
    config,
    services: RuntimeServices,
    ctx: AnalysisContext | None,
    no_cache: bool = False,
    refresh: bool = False,
) -> tuple[AnalysisBundle, bool]:
    """Primary fingerprint plus embedded-artifact analyses (spec 9, 5.12)."""
    from dataclasses import replace

    if ctx is None:
        from stylog.bootstrap import build_context

        ctx = build_context(config, services)

    primary_result = fingerprint_artifact(
        artifact,
        config=config,
        services=services,
        ctx=ctx,
        no_cache=no_cache,
        refresh=refresh,
    )
    primary = primary_result.fingerprint
    embedded: list[EmbeddedAnalysis] = []

    if (
        artifact.kind == ArtifactKind.CODE
        and artifact.language == "python"
        and config.analysis.code.python.enabled
        and config.analysis.code.python.embedded_text
    ):
        from stylog.analysis.python import extract_embedded
        from stylog.parsers.python_native import parse_python

        facts = parse_python(artifact, config)
        candidates = extract_embedded(artifact, facts, config)
        embedded_language = config.analysis.code.python.embedded_text_language.language
        for candidate in candidates:
            embedded_artifact = _embedded_runtime_artifact(
                artifact, candidate.kind, candidate.ordinal, candidate.text
            )
            embedded_artifact = replace(embedded_artifact, language=embedded_language)
            sub = fingerprint_artifact(
                embedded_artifact,
                config=config,
                services=services,
                ctx=ctx,
                no_cache=no_cache,
                refresh=refresh,
            )
            descriptor_kwargs = {
                "artifact": sub.fingerprint.artifact,
                "parent_artifact_id": artifact.artifact_id,
                "embedded_kind": candidate.kind,
                "ordinal": candidate.ordinal,
                "source_span": candidate.span,
            }
            if candidate.docstring_owner is not None:
                descriptor_kwargs["docstring_owner"] = candidate.docstring_owner
            descriptor = EmbeddedArtifactDescriptor(**descriptor_kwargs)
            embedded.append(EmbeddedAnalysis(descriptor=descriptor, fingerprint=sub.fingerprint))

    bundle = AnalysisBundle(
        primary=primary,
        embedded=tuple(embedded),
        diagnostics=(),
    )
    return bundle, primary_result.internal_error
