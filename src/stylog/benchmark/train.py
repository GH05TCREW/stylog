"""Verifier training entry point: stylog.verifier-training manifests (spec 21/23).

Local and deterministic: the dataset manifest resolves relative to the
training file, pair populations come from the manifest (train / tuning /
calibration), artifacts are fingerprinted through the application use case,
and fitting is the pure-Python solver in ``verification/fit.py``.
"""

from __future__ import annotations

from pathlib import Path

from stylog.benchmark.evaluate import _FingerprintSource
from stylog.benchmark.manifest import (
    TrainingManifest,
    _fail,
    _load_dataset,
    load_training_manifest,
    validate_dataset,
)
from stylog.config import StylogConfig, load_config
from stylog.domain.diagnostic import Diagnostic
from stylog.domain.verification import VerifierFit, VerifierPairPolicy
from stylog.exceptions import VerifierFitError
from stylog.verification.fit import fit_verifier_model, pairs_manifest_sha256
from stylog.verification.spec import TrainingPair, VerifierSpec


def _verifier_spec(training: TrainingManifest) -> VerifierSpec:
    block = training.verifier
    policy_kwargs: dict[str, object] = {"selection_version": block.selection_version}
    if block.max_pairs_per_author is not None:
        policy_kwargs["max_pairs_per_author"] = block.max_pairs_per_author
    if block.max_pairs_per_problem is not None:
        policy_kwargs["max_pairs_per_problem"] = block.max_pairs_per_problem
    if block.negative_positive_ratio is not None:
        policy_kwargs["negative_positive_ratio"] = block.negative_positive_ratio
    kwargs: dict[str, object] = {
        "kind": block.kind,
        "l2_lambda": block.l2_lambda,
        "min_support_fraction": block.min_support_fraction,
        "min_class_support_fraction": block.min_class_support_fraction,
        "min_pairs": block.min_pairs,
        "threshold_rule": block.threshold_rule,
        "max_iterations": block.max_iterations,
        "tolerance": block.tolerance,
        "include_linguistic": block.include_linguistic,
        "allow_unconstrained_language": block.allow_unconstrained_language,
        "pair_policy": VerifierPairPolicy(**policy_kwargs),
    }
    if block.threshold_alpha is not None:
        kwargs["threshold_alpha"] = block.threshold_alpha
    if block.threshold_fixed is not None:
        kwargs["threshold_fixed"] = block.threshold_fixed
    if block.calibration_method is not None:
        kwargs["calibration_method"] = block.calibration_method
    if block.languages is not None:
        kwargs["languages"] = block.languages
    if block.feature_ids is not None:
        kwargs["feature_ids"] = block.feature_ids
    try:
        return VerifierSpec(**kwargs)
    except VerifierFitError as exc:
        _fail("BENCHMARK_INVALID", f"[verifier]: {exc}")


def fit_verifier_from_manifest(
    training_path: str | Path, *, config: StylogConfig | None = None
) -> tuple[VerifierFit, tuple[Diagnostic, ...]]:
    """Fit a VerifierFit from a ``stylog.verifier-training`` TOML manifest."""
    cfg = config if config is not None else load_config()
    training_path = Path(training_path)
    training = load_training_manifest(training_path)
    manifest, _, root = _load_dataset(training_path, training.dataset)
    validate_dataset(manifest, root, verify_checksums=True)
    spec = _verifier_spec(training)

    for pair in training.pairs:
        for endpoint in (pair.left, pair.right):
            if manifest.artifact_by_id(endpoint) is None:
                _fail("PAIR_INVALID", f"pair references unknown artifact {endpoint!r}")

    source = _FingerprintSource(manifest, root, cfg)

    def build(population: str) -> list[TrainingPair]:
        return [
            TrainingPair(
                left=source.fingerprint(pair.left),
                right=source.fingerprint(pair.right),
                label=pair.label,
            )
            for pair in training.pairs
            if pair.population == population
        ]

    train_pairs = build("train")
    if not train_pairs:
        _fail("BENCHMARK_INVALID", f"{training_path}: the train population is empty")
    tuning_pairs = build("tuning")
    calibration_pairs = build("calibration")
    tuning_manifest_sha256 = (
        pairs_manifest_sha256(tuning_pairs) if tuning_pairs else None
    )
    return fit_verifier_model(
        spec,
        train_pairs,
        calibration_pairs=calibration_pairs or None,
        tuning_manifest_sha256=tuning_manifest_sha256,
    )
