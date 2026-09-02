"""Public Python API (spec 20). Thin convenience layer over application use cases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from stylog.application.analyze import analyze_artifact
from stylog.application.compare import compare_subjects
from stylog.application.fingerprint import fingerprint_artifact
from stylog.application.profile import build_baseline as _build_baseline
from stylog.application.profile import profile_subject
from stylog.application.verify import verify_subjects
from stylog.bootstrap import build_context, build_default_services
from stylog.config import StylogConfig, load_config
from stylog.domain.baseline import Baseline, BaselineDescriptor
from stylog.domain.fingerprint import AnalysisBundle, Fingerprint
from stylog.domain.interpretation import Comparison, Profile
from stylog.domain.verification import Verification, VerifierFit
from stylog.infrastructure.ingest import (
    artifact_from_bytes,
    artifact_from_file,
    artifact_from_text,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stylog.verification.spec import TrainingPair, VerifierSpec


def _config(config: StylogConfig | None) -> StylogConfig:
    return config if config is not None else load_config()


def _ctx(cfg: StylogConfig, services) -> object:
    nlp = cfg.nlp
    model = nlp.model if (nlp is not None and nlp.enabled and nlp.model) else None
    return build_context(cfg, services, nlp_model_name=model)


def fingerprint_file(
    path: str | Path,
    *,
    kind: str = "auto",
    language: str = "auto",
    config: StylogConfig | None = None,
) -> Fingerprint:
    cfg = _config(config)
    services = build_default_services(cfg)
    artifact = artifact_from_file(Path(path), artifact_id=Path(path).name, kind=kind,
                                  language=language, config=cfg)
    return fingerprint_artifact(
        artifact, config=cfg, services=services, ctx=_ctx(cfg, services)
    ).fingerprint


def fingerprint_text(
    text: str,
    *,
    language: str = "und",
    config: StylogConfig | None = None,
) -> Fingerprint:
    cfg = _config(config)
    services = build_default_services(cfg)
    artifact = artifact_from_text(text, artifact_id="text", language=language)
    return fingerprint_artifact(
        artifact, config=cfg, services=services, ctx=_ctx(cfg, services)
    ).fingerprint


def fingerprint_bytes(
    data: bytes,
    *,
    kind: str,
    language: str,
    encoding: str = "utf-8",
    config: StylogConfig | None = None,
) -> Fingerprint:
    cfg = _config(config)
    services = build_default_services(cfg)
    artifact = artifact_from_bytes(
        data, artifact_id="bytes", kind=kind, language=language, encoding=encoding, config=cfg
    )
    return fingerprint_artifact(
        artifact, config=cfg, services=services, ctx=_ctx(cfg, services)
    ).fingerprint


def analyze_file(
    path: str | Path,
    *,
    kind: str = "auto",
    language: str = "auto",
    config: StylogConfig | None = None,
) -> AnalysisBundle:
    cfg = _config(config)
    services = build_default_services(cfg)
    artifact = artifact_from_file(Path(path), artifact_id=Path(path).name, kind=kind,
                                  language=language, config=cfg)
    bundle, _ = analyze_artifact(
        artifact, config=cfg, services=services, ctx=_ctx(cfg, services)
    )
    return bundle


def analyze_text(
    text: str,
    *,
    language: str = "und",
    config: StylogConfig | None = None,
) -> AnalysisBundle:
    cfg = _config(config)
    services = build_default_services(cfg)
    artifact = artifact_from_text(text, artifact_id="text", language=language)
    bundle, _ = analyze_artifact(
        artifact, config=cfg, services=services, ctx=_ctx(cfg, services)
    )
    return bundle


def compare_files(
    left: str | Path,
    right: str | Path,
    *,
    config: StylogConfig | None = None,
) -> Comparison:
    cfg = _config(config)
    left_fp = fingerprint_file(left, config=cfg)
    right_fp = fingerprint_file(right, config=cfg)
    return compare_subjects(left_fp, right_fp, left_ref=str(left), right_ref=str(right))


def compare_fingerprints(
    left: Fingerprint,
    right: Fingerprint,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Comparison:
    return compare_subjects(left, right, left_ref=left_ref, right_ref=right_ref)


def profile_fingerprint(
    fingerprint: Fingerprint,
    baseline_ref: str,
    *,
    config: StylogConfig | None = None,
    subject_ref: str = "subject",
) -> Profile:
    cfg = _config(config)
    services = build_default_services(cfg)
    return profile_subject(fingerprint, baseline_ref, services=services, subject_ref=subject_ref)


def build_baseline(
    fingerprints: Sequence[Fingerprint],
    *,
    baseline_id: str,
    baseline_version: str = "1.0.0",
    kind: str = "text",
    language: str = "und",
    domain: str = "general",
    source: str = "local",
) -> Baseline:
    """Build a local baseline (spec 13.8, 13.10) from analyzed units.

    Each fingerprint is one baseline unit. Profiling uses every non-empty
    per-feature distribution and records its exact ``baseline_n`` (spec 13.7).
    Serialize the result with
    ``stylog.serialization.jsonio.write_json_atomic`` and pass the path as
    the ``--baseline`` ref.
    """
    return _build_baseline(
        fingerprints,
        baseline_id=baseline_id,
        baseline_version=baseline_version,
        descriptor=BaselineDescriptor(
            kind=kind,
            language=language,
            domain=domain,
            unit="artifact",
            source=source,
        ),
    )


def fit_representation(spec, corpus, *, config: StylogConfig | None = None):
    """`stylog[ml]`: fit a representation (see representations/)."""
    cfg = _config(config)
    from stylog.representations.fit import fit_representation as _fit

    return _fit(spec, corpus, config=cfg)


def transform_representation(fit_or_spec, subject, *, config: StylogConfig | None = None):
    """`stylog[ml]`: transform a subject into a Representation."""
    cfg = _config(config)
    from stylog.representations.fit import transform_representation as _transform

    return _transform(fit_or_spec, subject, config=cfg)


def verify_fingerprints(
    left: Fingerprint,
    right: Fingerprint,
    model: VerifierFit,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Verification:
    """Verify two fingerprints under an explicit fitted model (spec 23)."""
    return verify_subjects(left, right, model, left_ref=left_ref, right_ref=right_ref)


def verify_files(
    left: str | Path,
    right: str | Path,
    model: VerifierFit,
    *,
    config: StylogConfig | None = None,
) -> Verification:
    """Fingerprint two files, then verify them under an explicit fitted model."""
    cfg = _config(config)
    left_fp = fingerprint_file(left, config=cfg)
    right_fp = fingerprint_file(right, config=cfg)
    return verify_subjects(left_fp, right_fp, model, left_ref=str(left), right_ref=str(right))


def fit_verifier(
    spec: VerifierSpec,
    pairs: Sequence[TrainingPair],
    *,
    calibration_pairs: Sequence[TrainingPair] | None = None,
    tuning_manifest_sha256: str | None = None,
    config: StylogConfig | None = None,
) -> VerifierFit:
    """Fit a self-contained VerifierFit (spec 23.10-23.15); stdlib-only solver."""
    _config(config)  # config reserved for future fit-time options; fit is pure
    from stylog.verification.fit import fit_verifier_model

    model, _diagnostics = fit_verifier_model(
        spec,
        pairs,
        calibration_pairs=calibration_pairs,
        tuning_manifest_sha256=tuning_manifest_sha256,
    )
    return model


def load_verifier(path: str | Path) -> VerifierFit:
    """Load and fully validate a portable VerifierFit JSON artifact."""
    from stylog.serialization.jsonio import read_json

    return read_json(path, VerifierFit)
