"""Benchmark task evaluators (spec section 21).

All evaluation is local and deterministic: artifacts are read from the
dataset root, fingerprinted through the application use case, and compared
with the ordinary comparison mathematics. The descriptive tasks produce
distance summaries only -- no attribution, no thresholds, no EER. The
verification task additionally produces decisions and PAN decision metrics
under an explicit fitted verifier model (benchmark outputs only).
"""

from __future__ import annotations

import math
from pathlib import Path

from stylog.analysis import stats
from stylog.analysis.compare import compare_fingerprints
from stylog.application.fingerprint import fingerprint_artifact
from stylog.application.verify import verify_subjects
from stylog.benchmark import metrics
from stylog.benchmark.manifest import (
    BenchmarkSpec,
    DatasetManifest,
    _fail,
)
from stylog.benchmark.split import SPLIT_ALGORITHM_VERSION, SplitResult, compute_split
from stylog.bootstrap import build_context, build_default_services
from stylog.config import StylogConfig
from stylog.domain.benchmark import (
    BenchmarkResult,
    BenchmarkSplitRealization,
    PairwiseFeatureMetrics,
    RiskEntry,
    TransformationFeatureDistance,
    VerificationMetrics,
)
from stylog.domain.diagnostic import (
    Diagnostic,
    DiagnosticSeverity,
    make_diagnostic,
    sort_diagnostics,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.interpretation import Comparison, ComparisonComponent
from stylog.domain.verification import VerifierFit
from stylog.exceptions import PortableArtifactError
from stylog.infrastructure.ingest import artifact_from_bytes
from stylog.serialization.canonical import scientific_sha256
from stylog.verification.fit import pairs_manifest_sha256
from stylog.verification.spec import TrainingPair

PAIRWISE_AUC_OMITTED = "PAIRWISE_AUC_OMITTED"
VERIFICATION_AUC_OMITTED = "VERIFICATION_AUC_OMITTED"
VERIFICATION_F1_OMITTED = "VERIFICATION_F1_OMITTED"


class _FingerprintSource:
    """Lazily fingerprints dataset artifacts (once each) for one benchmark run."""

    def __init__(self, manifest: DatasetManifest, root: Path, config: StylogConfig) -> None:
        self._manifest = manifest
        self._root = root
        self._config = config
        self._cache: dict[str, Fingerprint] = {}
        services = build_default_services(config)
        self._services = services
        self._ctx = build_context(config, services)

    def fingerprint(self, artifact_id: str) -> Fingerprint:
        if artifact_id not in self._cache:
            artifact = self._manifest.artifact_by_id(artifact_id)
            assert artifact is not None  # existence validated by the callers
            data = (self._root / artifact.path).read_bytes()
            runtime_artifact = artifact_from_bytes(
                data,
                artifact_id=artifact.id,
                kind=artifact.kind,
                language=artifact.language,
                config=self._config,
            )
            result = fingerprint_artifact(
                runtime_artifact,
                config=self._config,
                services=self._services,
                ctx=self._ctx,
                no_cache=True,
            )
            self._cache[artifact_id] = result.fingerprint
        return self._cache[artifact_id]


def _components(comparison: Comparison) -> list[ComparisonComponent]:
    return [component for family in comparison.families for component in family.components]


def _realization(split_result: SplitResult) -> BenchmarkSplitRealization:
    return BenchmarkSplitRealization(
        train=split_result.train, dev=split_result.dev, test=split_result.test
    )


def _base_kwargs(
    spec: BenchmarkSpec, dataset_manifest_sha256: str
) -> dict[str, object]:
    return {
        "benchmark_id": spec.id,
        "task": spec.task,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_algorithm_version": SPLIT_ALGORITHM_VERSION,
    }


def _validate_pairs(
    spec: BenchmarkSpec, manifest: DatasetManifest, split_result: SplitResult | None
) -> None:
    """Fail on identical/unknown pair members or split-crossing pairs."""
    for pair in spec.pairs:
        if pair.left == pair.right:
            _fail("PAIR_INVALID", f"pair members must differ: {pair.left!r}")
        for member in (pair.left, pair.right):
            if manifest.artifact_by_id(member) is None:
                _fail("PAIR_INVALID", f"unknown artifact id {member!r}")
        if split_result is not None:
            left_part = split_result.assignment[pair.left]
            right_part = split_result.assignment[pair.right]
            if left_part != right_part:
                _fail(
                    "PAIR_INVALID",
                    f"pair ({pair.left!r}, {pair.right!r}) crosses split parts "
                    f"({left_part} vs {right_part})",
                )


def _apply_split_kwargs(kwargs: dict[str, object], split_result: SplitResult | None) -> None:
    if split_result is not None:
        kwargs["split_config_sha256"] = split_result.split_config_sha256
        kwargs["splits"] = _realization(split_result)


def evaluate_split_audit(
    spec: BenchmarkSpec,
    manifest: DatasetManifest,
    dataset_manifest_sha256: str,
) -> BenchmarkResult:
    """Manifest/checksum validation plus split realization and risk echo.

    No content analysis happens here; artifact bytes were read only for the
    checksum verification requested by the spec.
    """
    if spec.split is None:
        _fail("BENCHMARK_INVALID", "split_audit requires a [split] section")
    split_result = compute_split(manifest, spec.split)
    risk_declarations = tuple(
        RiskEntry(key=key, value=manifest.risks[key]) for key in sorted(manifest.risks)
    )
    return BenchmarkResult(
        **_base_kwargs(spec, dataset_manifest_sha256),
        split_config_sha256=split_result.split_config_sha256,
        splits=_realization(split_result),
        risk_declarations=risk_declarations,
    )


def _pairwise_metrics(
    same: dict[tuple[str, str], list[float]],
    different: dict[tuple[str, str], list[float]],
    diagnostics: list[Diagnostic],
) -> tuple[PairwiseFeatureMetrics, ...]:
    metrics: list[PairwiseFeatureMetrics] = []
    for feature_id, metric in sorted(set(same) | set(different)):
        same_values = same.get((feature_id, metric), [])
        different_values = different.get((feature_id, metric), [])
        if not same_values and not different_values:
            continue  # no valid pairs: no entry
        fields: dict[str, object] = {}
        if same_values:
            fields["same_mean_distance"] = math.fsum(same_values) / len(same_values)
            fields["same_median_distance"] = stats.quantile_type7(sorted(same_values), 0.5)
        if different_values:
            fields["different_mean_distance"] = math.fsum(different_values) / len(different_values)
            fields["different_median_distance"] = stats.quantile_type7(
                sorted(different_values), 0.5
            )
        if same_values and different_values:
            # score = -distance, positive class = same (spec 21.11).
            fields["roc_auc"] = stats.roc_auc_mann_whitney(
                positive=[-value for value in same_values],
                negative=[-value for value in different_values],
            )
        else:
            diagnostics.append(
                make_diagnostic(
                    PAIRWISE_AUC_OMITTED,
                    DiagnosticSeverity.INFO,
                    feature_id=feature_id,
                    context=(("metric", metric),),
                )
            )
        metrics.append(
            PairwiseFeatureMetrics(
                feature_id=feature_id,
                metric=metric,
                same_count=len(same_values),
                different_count=len(different_values),
                **fields,
            )
        )
    return tuple(metrics)


def evaluate_pairwise(
    spec: BenchmarkSpec,
    manifest: DatasetManifest,
    root: Path,
    dataset_manifest_sha256: str,
    config: StylogConfig,
) -> BenchmarkResult:
    """Per-feature distance summaries over the explicitly supplied pairs."""
    if not spec.pairs:
        _fail("BENCHMARK_INVALID", "pairwise_comparison requires at least one [[pair]]")
    split_result = compute_split(manifest, spec.split) if spec.split is not None else None
    _validate_pairs(spec, manifest, split_result)
    source = _FingerprintSource(manifest, root, config)
    same: dict[tuple[str, str], list[float]] = {}
    different: dict[tuple[str, str], list[float]] = {}
    diagnostics: list[Diagnostic] = []
    for pair in spec.pairs:
        left_fp = source.fingerprint(pair.left)
        right_fp = source.fingerprint(pair.right)
        try:
            comparison = compare_fingerprints(left_fp, right_fp, pair.left, pair.right)
        except PortableArtifactError as exc:
            _fail("PAIR_INVALID", f"pair ({pair.left!r}, {pair.right!r}): {exc}")
        diagnostics.extend(comparison.diagnostics)
        target = same if pair.label == "same" else different
        for component in _components(comparison):
            target.setdefault((component.feature_id, component.metric), []).append(component.value)
    kwargs = _base_kwargs(spec, dataset_manifest_sha256)
    _apply_split_kwargs(kwargs, split_result)
    kwargs["pairwise_metrics"] = _pairwise_metrics(same, different, diagnostics)
    kwargs["diagnostics"] = sort_diagnostics(diagnostics)
    return BenchmarkResult(**kwargs)


def evaluate_transformation_stability(
    spec: BenchmarkSpec,
    manifest: DatasetManifest,
    root: Path,
    dataset_manifest_sha256: str,
    config: StylogConfig,
) -> BenchmarkResult:
    """Per-feature distances between transformation originals and variants."""
    if not manifest.transformations:
        _fail(
            "BENCHMARK_INVALID",
            "transformation_stability requires at least one [[transformation]] "
            "in the dataset manifest",
        )
    for transformation in manifest.transformations:
        for member in (transformation.original, transformation.variant):
            if manifest.artifact_by_id(member) is None:
                _fail("BENCHMARK_INVALID", f"unknown artifact id {member!r}")
    source = _FingerprintSource(manifest, root, config)
    distances: list[TransformationFeatureDistance] = []
    diagnostics: list[Diagnostic] = []
    for transformation in manifest.transformations:
        original_fp = source.fingerprint(transformation.original)
        variant_fp = source.fingerprint(transformation.variant)
        try:
            comparison = compare_fingerprints(
                original_fp, variant_fp, transformation.original, transformation.variant
            )
        except PortableArtifactError as exc:
            _fail(
                "BENCHMARK_INVALID",
                f"transformation {transformation.transformation_id!r}: {exc}",
            )
        diagnostics.extend(comparison.diagnostics)
        for component in _components(comparison):
            distances.append(
                TransformationFeatureDistance(
                    transformation_id=transformation.transformation_id,
                    original=transformation.original,
                    variant=transformation.variant,
                    feature_id=component.feature_id,
                    metric=component.metric,
                    value=component.value,
                )
            )
    distances.sort(key=lambda entry: (entry.transformation_id, entry.feature_id))
    return BenchmarkResult(
        **_base_kwargs(spec, dataset_manifest_sha256),
        transformation_distances=tuple(distances),
        diagnostics=sort_diagnostics(diagnostics),
    )


def evaluate_verification(
    spec: BenchmarkSpec,
    manifest: DatasetManifest,
    root: Path,
    dataset_manifest_sha256: str,
    model: VerifierFit,
    config: StylogConfig,
) -> BenchmarkResult:
    """Decisions + PAN decision metrics over pairs under an explicit verifier.

    The evaluation population is validated against the model's recorded
    train/tuning/calibration manifest identities: evaluating on a recorded
    population is a hard failure, never a reported number.
    """
    if not spec.pairs:
        _fail("BENCHMARK_INVALID", "verification requires at least one [[pair]]")
    split_result = compute_split(manifest, spec.split) if spec.split is not None else None
    _validate_pairs(spec, manifest, split_result)
    source = _FingerprintSource(manifest, root, config)
    rows: list[metrics.DecisionRow] = []
    evaluated_pairs: list[TrainingPair] = []
    answered = abstain_uncertain = abstain_insufficient = 0
    for pair in spec.pairs:
        left_fp = source.fingerprint(pair.left)
        right_fp = source.fingerprint(pair.right)
        evaluated_pairs.append(
            TrainingPair(left=left_fp, right=right_fp, label=pair.label)
        )
        verification = verify_subjects(
            left_fp, right_fp, model, left_ref=pair.left, right_ref=pair.right
        )
        if verification.verdict == "abstain":
            if verification.abstain_reason == "insufficient_evidence":
                abstain_insufficient += 1
            else:
                abstain_uncertain += 1
        else:
            answered += 1
        rows.append(
            metrics.DecisionRow(
                verdict=verification.verdict,
                label=pair.label,
                score=verification.score,
                probability=verification.probability,
            )
        )
    evaluation_manifest_sha256 = pairs_manifest_sha256(evaluated_pairs)
    recorded_populations = {
        "source": model.source_manifest_sha256,
        "tuning": model.tuning_manifest_sha256,
        "calibration": model.calibration_manifest_sha256,
    }
    for population, recorded_sha256 in recorded_populations.items():
        if recorded_sha256 is not None and recorded_sha256 == evaluation_manifest_sha256:
            _fail(
                "BENCHMARK_INVALID",
                "the evaluation pairs are identical to the model's "
                f"{population} population; held-out evaluation requires a "
                "disjoint manifest",
            )
    diagnostics: list[Diagnostic] = []
    metrics_kwargs: dict[str, object] = {
        "verifier_id": scientific_sha256(model),
        "pair_count": len(rows),
        "answered_count": answered,
        "abstain_uncertain_count": abstain_uncertain,
        "abstain_insufficient_evidence_count": abstain_insufficient,
        "c_at_1": metrics.c_at_1(rows),
        "f_05u": metrics.f_05u(rows),
    }
    auc = metrics.roc_auc(rows)
    if auc is None:
        diagnostics.append(make_diagnostic(VERIFICATION_AUC_OMITTED, DiagnosticSeverity.INFO))
    else:
        metrics_kwargs["roc_auc"] = auc
    f1_value = metrics.f1(rows)
    if f1_value is None:
        diagnostics.append(make_diagnostic(VERIFICATION_F1_OMITTED, DiagnosticSeverity.INFO))
    else:
        metrics_kwargs["f1"] = f1_value
    brier_value = metrics.brier(rows)
    if brier_value is not None:
        metrics_kwargs["brier"] = brier_value
    kwargs = _base_kwargs(spec, dataset_manifest_sha256)
    _apply_split_kwargs(kwargs, split_result)
    kwargs["verification_metrics"] = VerificationMetrics(**metrics_kwargs)
    kwargs["diagnostics"] = sort_diagnostics(diagnostics)
    return BenchmarkResult(**kwargs)
