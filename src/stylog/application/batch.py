"""Streaming batch analysis (spec 18.15-18.16).

Ordinals are assigned before dispatch. Serial and process paths run the same
scientific application functions and emit results in ordinal order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from stylog.config import StylogConfig
from stylog.domain.fingerprint import AnalysisBundle
from stylog.infrastructure.execution import run_ordered
from stylog.runtime import RuntimeArtifact


def _nlp_model_name(config: StylogConfig) -> str | None:
    nlp = config.nlp
    if nlp is not None and nlp.enabled and nlp.model:
        return nlp.model
    return None


def _analyze_payload(payload: tuple[int, RuntimeArtifact, StylogConfig]):
    """Top-level worker (picklable): rebuild services/context per process."""
    from stylog.application.analyze import analyze_artifact
    from stylog.bootstrap import build_context, build_default_services

    ordinal, artifact, config = payload
    services = build_default_services(config)
    ctx = build_context(config, services, nlp_model_name=_nlp_model_name(config))
    bundle, internal_error = analyze_artifact(
        artifact, config=config, services=services, ctx=ctx
    )
    return ordinal, bundle, internal_error


def analyze_iter(
    artifacts: Sequence[RuntimeArtifact],
    *,
    config: StylogConfig,
    execution: str = "serial",
    workers: int = 0,
    max_in_flight: int = 0,
) -> Iterator[tuple[AnalysisBundle, bool]]:
    """Yield (AnalysisBundle, internal_error) in ordinal order."""
    payloads = [
        (ordinal, artifact, config) for ordinal, artifact in enumerate(artifacts)
    ]
    results = run_ordered(
        payloads,
        _analyze_payload,
        mode=execution,
        workers=workers,
        max_in_flight=max_in_flight,
    )
    for _, bundle, internal_error in results:
        yield bundle, internal_error


def fingerprint_iter(
    artifacts: Sequence[RuntimeArtifact],
    *,
    config: StylogConfig,
    execution: str = "serial",
    workers: int = 0,
    max_in_flight: int = 0,
) -> Iterator[AnalysisBundle]:
    """Yield AnalysisBundles in ordinal order, memory bounded by in-flight work."""
    for bundle, _ in analyze_iter(
        artifacts,
        config=config,
        execution=execution,
        workers=workers,
        max_in_flight=max_in_flight,
    ):
        yield bundle
