"""Deterministic pure-Python verifier fitting (spec 23.10-23.15).

Fully specified IRLS on the Python standard library alone — no NumPy, no
BLAS, no sklearn. Determinism scope (spec 23.11): fixed input order (pairs
sorted by content sha, features in sorted model order) + correctly-rounded
``math.fsum`` accumulation + a fully specified linear solve ⇒ byte-identical
repeated fits within the same recorded runtime environment. Cross-platform
byte-identity is NOT claimed (C libm ``exp``/``log`` may differ in the last
ulp — the same stance as spec 8.19). Correctness is anchored by sklearn/SciPy
differential oracle tests in the dev lane.
"""

from __future__ import annotations

import math
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING

from stylog.analysis.compare import compare_fingerprints
from stylog.analysis.registry import (
    FEATURE_REGISTRY_VERSION,
    FEATURE_SEMANTIC_VERSION,
    FEATURES,
)
from stylog.analysis.stats import quantile_type7
from stylog.analysis.verify import (
    VERIFIER_MODEL_ID,
    VERIFIER_MODEL_SEMANTIC_VERSION,
    clamp_logit,
    sigmoid_open,
)
from stylog.domain.artifact import ContentIdentitySha256
from stylog.domain.diagnostic import (
    Diagnostic,
    DiagnosticSeverity,
    make_diagnostic,
    sort_diagnostics,
)
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.provenance import (
    BackendSignature,
    PackageProvenance,
    current_runtime_signature,
)
from stylog.domain.verification import (
    THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND,
    THRESHOLD_RULE_FIXED,
    VERIFIER_TASK,
    VERIFIER_TASK_VERSION,
    VerifierCalibration,
    VerifierEligibility,
    VerifierFeature,
    VerifierFit,
    VerifierThresholds,
)
from stylog.exceptions import VerifierFitError
from stylog.serialization.canonical import scientific_sha256, sha256_of_tree
from stylog.verification.spec import LABEL_SAME, TrainingPair, VerifierSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

# Single documented source for language-scoped feature candidacy (spec 23.12):
# a scoped feature is a candidate only when the model's pinned languages are
# nonempty and fully inside the feature's scope.
FEATURE_LANGUAGE_SCOPE: dict[str, frozenset[str]] = {
    "text.function_words.en.token_share": frozenset({"en"}),
    "text.function_words.en.lexeme_distribution": frozenset({"en"}),
}

_LINGUISTIC_PREFIX = "text.linguistic."
_KIND_PREFIXES = {"text": "text.", "code": "code."}

# Pivot magnitude floor for the specified Gaussian elimination. Legitimate
# pivots are bounded below by roughly l2_lambda (realistically >= 1e-6), so a
# pivot below this floor indicates a genuinely degenerate design (e.g. an
# unpenalized intercept collinear with all-zero standardized columns).
PIVOT_FLOOR = 1e-12

# Fit diagnostic codes (ephemeral; reported on stderr, never portable).
THRESHOLD_BAND_COLLAPSED = "THRESHOLD_BAND_COLLAPSED"
VERIFIER_ZERO_VARIANCE_FEATURE = "VERIFIER_ZERO_VARIANCE_FEATURE"
VERIFIER_UNCONSTRAINED_LANGUAGE = "VERIFIER_UNCONSTRAINED_LANGUAGE"
VERIFIER_ELIGIBILITY = "VERIFIER_ELIGIBILITY"
VERIFIER_CALIBRATION_PAIRS_EXCLUDED = "VERIFIER_CALIBRATION_PAIRS_EXCLUDED"


def _stylog_version() -> str:
    try:
        return package_version("stylog")
    except Exception:  # pragma: no cover - source tree without installed metadata
        from stylog import __version__

        return __version__


def _content_sha(fingerprint: Fingerprint) -> str:
    identity = fingerprint.artifact.content_identity
    if isinstance(identity, ContentIdentitySha256):
        return identity.sha256
    return scientific_sha256(fingerprint)


def _pair_sort_key(pair: TrainingPair) -> tuple[str, str]:
    return (_content_sha(pair.left), _content_sha(pair.right))


def pairs_manifest_sha256(pairs: Sequence[TrainingPair]) -> str:
    """Manifest identity: hash of sorted (left_sha, right_sha, label) triples.

    The shas are the scientific hashes of the two fingerprints (measurement
    identity), so the same pair population always yields the same manifest id
    regardless of pair ordering or artifact refs.
    """
    triples = sorted(
        (scientific_sha256(pair.left), scientific_sha256(pair.right), pair.label)
        for pair in pairs
    )
    return sha256_of_tree({"pairs": [list(triple) for triple in triples]})


def candidate_feature_ids(spec: VerifierSpec, languages: tuple[str, ...]) -> tuple[str, ...]:
    """Deterministic, language-aware candidate universe (spec 23.12)."""
    prefix = _KIND_PREFIXES[spec.kind]
    if spec.feature_ids is not None:
        universe = list(spec.feature_ids)
        for feature_id in universe:
            fdef = FEATURES.get(feature_id)
            if fdef is None:
                raise VerifierFitError(f"unknown feature_id in spec: {feature_id!r}")
            if fdef.metric == "NONE":
                raise VerifierFitError(
                    f"feature {feature_id!r} carries no comparison metric"
                )
            if not feature_id.startswith(prefix):
                raise VerifierFitError(
                    f"feature {feature_id!r} does not apply to kind {spec.kind!r}"
                )
    else:
        universe = sorted(
            feature_id
            for feature_id, fdef in FEATURES.items()
            if fdef.metric != "NONE" and feature_id.startswith(prefix)
        )
    selected: list[str] = []
    for feature_id in universe:
        if feature_id.startswith(_LINGUISTIC_PREFIX) and not spec.include_linguistic:
            continue
        scope = FEATURE_LANGUAGE_SCOPE.get(feature_id)
        if scope is not None and (not languages or not set(languages) <= scope):
            continue
        selected.append(feature_id)
    return tuple(selected)


def _check_pair_kinds(kind: str, pairs: Sequence[TrainingPair], role: str) -> None:
    for index, pair in enumerate(pairs):
        left_kind = pair.left.artifact.kind
        right_kind = pair.right.artifact.kind
        if left_kind != right_kind:
            raise VerifierFitError(
                f"{role} pair {index} mixes kinds {left_kind!r}/{right_kind!r}"
            )
        if left_kind != kind:
            raise VerifierFitError(
                f"{role} pair {index} kind {left_kind!r} does not match spec kind {kind!r}"
            )


def _comparison_values(pair: TrainingPair, feature_ids: tuple[str, ...]) -> dict[str, float]:
    """Per-feature distances for one pair, restricted to requested features.

    A feature has a value exactly when both observations are ok, the pairwise
    compatibility gate passes, and the observation semantic version matches
    the current registry semantics (mirrors the verify-time alignment rule).
    """
    comparison = compare_fingerprints(pair.left, pair.right, "left", "right")
    components = {
        component.feature_id: component
        for family in comparison.families
        for component in family.components
    }
    values: dict[str, float] = {}
    for feature_id in feature_ids:
        component = components.get(feature_id)
        if component is None or component.semantic_version != FEATURE_SEMANTIC_VERSION:
            continue
        values[feature_id] = component.value
    return values


def _select_features(
    spec: VerifierSpec,
    candidates: tuple[str, ...],
    pair_values: list[dict[str, float]],
    labels: list[int],
) -> tuple[str, ...]:
    """Training-only support policy (spec 23.12), pure in its inputs."""
    n_pairs = len(pair_values)
    n_same = sum(labels)
    n_different = n_pairs - n_same
    selected: list[str] = []
    for feature_id in candidates:
        support_same = sum(
            1 for values, y in zip(pair_values, labels, strict=True) if feature_id in values and y
        )
        support_different = sum(
            1
            for values, y in zip(pair_values, labels, strict=True)
            if feature_id in values and not y
        )
        support = support_same + support_different
        if support / n_pairs < spec.min_support_fraction:
            continue
        if support_same / n_same < spec.min_class_support_fraction:
            continue
        if support_different / n_different < spec.min_class_support_fraction:
            continue
        selected.append(feature_id)
    return tuple(selected)


def _fit_normalization(
    selected: tuple[str, ...],
    eligible_values: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float], list[Diagnostic]]:
    """Per-feature mean / population std over eligible training pairs only."""
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    diagnostics: list[Diagnostic] = []
    n = len(eligible_values)
    for feature_id in selected:
        xs = [values[feature_id] for values in eligible_values]
        mean = math.fsum(xs) / n
        variance = math.fsum((x - mean) ** 2 for x in xs) / n
        scale = math.sqrt(variance)
        if scale == 0.0:
            scale = 1.0
            diagnostics.append(
                make_diagnostic(
                    VERIFIER_ZERO_VARIANCE_FEATURE,
                    DiagnosticSeverity.INFO,
                    feature_id=feature_id,
                )
            )
        means[feature_id] = mean
        scales[feature_id] = scale
    return means, scales, diagnostics


def _solve_linear(
    matrix: list[list[float]], rhs: list[float], pivot_floor: float
) -> list[float]:
    """Solve A x = b by Gaussian elimination with partial pivoting.

    Fully specified operation order: columns left to right; pivot = first row
    attaining the max absolute column value at or below the diagonal; rows
    eliminated top to bottom; back-substitution bottom to top with
    ``math.fsum`` over products in increasing column order.
    """
    n = len(matrix)
    augmented = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = col
        pivot_abs = abs(augmented[col][col])
        for row in range(col + 1, n):
            candidate = abs(augmented[row][col])
            if candidate > pivot_abs:
                pivot = row
                pivot_abs = candidate
        if pivot_abs < pivot_floor:
            raise VerifierFitError(
                "numerically singular design in IRLS: pivot magnitude below "
                f"{pivot_floor} at column {col}"
            )
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        for row in range(col + 1, n):
            factor = augmented[row][col] / pivot_value
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                augmented[row][c] -= factor * augmented[col][c]
    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        residual = math.fsum(augmented[row][c] * x[c] for c in range(row + 1, n))
        x[row] = (augmented[row][n] - residual) / augmented[row][row]
    return x


def _irls(
    design: list[tuple[float, ...]],
    labels: list[int],
    l2_lambda: float,
    max_iterations: int,
    tolerance: float,
) -> list[float]:
    """Newton/IRLS for ``sum logloss + (lambda/2)||w||^2``, intercept unpenalized.

    Parameters start at zero. Each iteration recomputes scores in fixed pair
    order with the overflow-safe sigmoid; the gradient Z^T(y-s) and Hessian
    Z^T W Z accumulate over pairs in that same fixed order via ``math.fsum``.
    The step solves (Z^T W Z + lambda*I~) delta = gradient; convergence is
    ``max|delta| <= tolerance``; exhausting ``max_iterations`` is a typed fit
    error, never a silent half-fit. There is no separation failure rule: with
    ``l2_lambda > 0`` the penalized optimum is finite on separable data.
    """
    n = len(design)
    k = len(design[0])
    p = k + 1
    w = [0.0] * p
    for _iteration in range(max_iterations):
        residuals: list[float] = []
        weights: list[float] = []
        for i in range(n):
            row = design[i]
            logit = math.fsum([w[0]] + [w[j + 1] * row[j] for j in range(k)])
            score = sigmoid_open(logit)
            residuals.append(labels[i] - score)
            weights.append(score * (1.0 - score))
        # wz[i][j] = W_i * z_ij with z_i0 = 1 (intercept column).
        wz = [
            [weights[i]] + [weights[i] * design[i][j] for j in range(k)]
            for i in range(n)
        ]
        gradient = [0.0] * p
        for j in range(p):
            if j == 0:
                gradient[j] = math.fsum(residuals[i] for i in range(n))
            else:
                gradient[j] = math.fsum(residuals[i] * design[i][j - 1] for i in range(n))
        for j in range(1, p):
            gradient[j] -= l2_lambda * w[j]
        hessian = [[0.0] * p for _ in range(p)]
        for a in range(p):
            for b in range(a, p):
                if b == 0:
                    value = math.fsum(wz[i][0] for i in range(n))
                else:
                    value = math.fsum(wz[i][a] * design[i][b - 1] for i in range(n))
                hessian[a][b] = value
                hessian[b][a] = value
        for j in range(1, p):
            hessian[j][j] += l2_lambda
        delta = _solve_linear(hessian, gradient, PIVOT_FLOOR)
        w = [wi + di for wi, di in zip(w, delta, strict=True)]
        if max(abs(d) for d in delta) <= tolerance:
            return w
    raise VerifierFitError(
        f"IRLS did not converge within {max_iterations} iterations"
    )


def _logit_for(
    w: list[float],
    selected: tuple[str, ...],
    means: dict[str, float],
    scales: dict[str, float],
    values: dict[str, float],
) -> float:
    """Model logit for one complete evidence vector (fixed fsum order, clamped)."""
    terms = [w[0]] + [
        w[j + 1] * ((values[feature_id] - means[feature_id]) / scales[feature_id])
        for j, feature_id in enumerate(selected)
    ]
    return clamp_logit(math.fsum(terms))


def _fit_thresholds(
    spec: VerifierSpec,
    scores_same: list[float],
    scores_different: list[float],
) -> tuple[VerifierThresholds, list[Diagnostic]]:
    """Calibration-split threshold band (spec 23.9) with deterministic collapse."""
    if spec.threshold_rule == THRESHOLD_RULE_FIXED:
        fixed = spec.threshold_fixed
        assert fixed is not None  # spec validation guarantees
        return VerifierThresholds(t_same=fixed, t_diff=fixed), []
    alpha = spec.threshold_alpha
    assert alpha is not None
    t_diff = quantile_type7(sorted(scores_different), 1.0 - alpha)
    t_same = quantile_type7(sorted(scores_same), alpha)
    if t_diff > t_same:
        collapsed = (t_diff + t_same) / 2.0
        return VerifierThresholds(t_same=collapsed, t_diff=collapsed), [
            make_diagnostic(THRESHOLD_BAND_COLLAPSED, DiagnosticSeverity.WARNING)
        ]
    return VerifierThresholds(t_same=t_same, t_diff=t_diff), []


def _fit_platt(
    spec: VerifierSpec, logits: list[float], labels: list[int]
) -> VerifierCalibration:
    """Platt (a, b) on calibration (logit, label) with the model frozen.

    The same specified Newton rule with two parameters and no penalty;
    perfectly separable calibration scores fail as typed non-convergence.
    """
    design = [(logit,) for logit in logits]
    w = _irls(design, labels, 0.0, spec.max_iterations, spec.tolerance)
    return VerifierCalibration(a=w[1], b=w[0])


def _population_scores(
    selected: tuple[str, ...],
    means: dict[str, float],
    scales: dict[str, float],
    w: list[float],
    pairs: Sequence[TrainingPair],
) -> tuple[list[float], list[int], int]:
    """(logits, labels, excluded_count) over a held-out population.

    Pairs lacking complete evidence over the selected features are excluded
    (and counted); scoring uses the frozen coefficients and normalization.
    """
    logits: list[float] = []
    labels: list[int] = []
    excluded = 0
    for pair in pairs:
        values = _comparison_values(pair, selected)
        if len(values) != len(selected):
            excluded += 1
            continue
        logits.append(_logit_for(w, selected, means, scales, values))
        labels.append(1 if pair.label == LABEL_SAME else 0)
    return logits, labels, excluded


def fit_verifier_model(
    spec: VerifierSpec,
    pairs: Sequence[TrainingPair],
    *,
    calibration_pairs: Sequence[TrainingPair] | None = None,
    tuning_manifest_sha256: str | None = None,
) -> tuple[VerifierFit, tuple[Diagnostic, ...]]:
    """Fit a self-contained VerifierFit (spec 23.10-23.15).

    TRAIN pairs drive eligibility, normalization, and coefficients. The
    CALIBRATION population (when supplied) drives thresholds and Platt with
    hyperparameters frozen. Tuning is a caller-side concern; when a tuning
    split informed the spec, its manifest identity is recorded via
    ``tuning_manifest_sha256``.
    """
    if not pairs:
        raise VerifierFitError("no training pairs supplied")
    needs_calibration = (
        spec.threshold_rule == THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND
        or spec.calibration_method is not None
    )
    if needs_calibration and not calibration_pairs:
        raise VerifierFitError(
            "a calibration split is required for threshold_rule "
            f"{spec.threshold_rule!r} / calibration {spec.calibration_method!r}"
        )
    sorted_pairs = sorted(pairs, key=_pair_sort_key)
    _check_pair_kinds(spec.kind, sorted_pairs, "training")
    labels = [1 if pair.label == LABEL_SAME else 0 for pair in sorted_pairs]
    if not any(labels) or all(labels):
        raise VerifierFitError("training requires at least one pair of each class")

    candidacy_languages = (
        spec.languages
        if spec.languages is not None
        else tuple(
            sorted(
                {
                    language
                    for pair in sorted_pairs
                    for language in (
                        pair.left.artifact.language,
                        pair.right.artifact.language,
                    )
                }
            )
        )
    )
    candidates = candidate_feature_ids(spec, candidacy_languages)
    pair_values = [_comparison_values(pair, candidates) for pair in sorted_pairs]
    selected = _select_features(spec, candidates, pair_values, labels)
    if not selected:
        raise VerifierFitError(
            "no candidate feature survived the eligibility policy; relax the "
            "support fractions or check evidence quality"
        )
    eligible_mask = [
        all(feature_id in values for feature_id in selected) for values in pair_values
    ]
    eligible_pairs = [pair for pair, ok in zip(sorted_pairs, eligible_mask) if ok]
    eligible_values = [values for values, ok in zip(pair_values, eligible_mask) if ok]
    eligible_labels = [y for y, ok in zip(labels, eligible_mask) if ok]
    eligible_pair_count = len(eligible_pairs)
    if eligible_pair_count < spec.min_pairs:
        raise VerifierFitError(
            f"only {eligible_pair_count} eligible training pairs over the selected "
            f"feature set; min_pairs is {spec.min_pairs}"
        )
    if spec.languages is not None:
        languages = spec.languages
    else:
        languages = tuple(
            sorted(
                {
                    language
                    for pair in eligible_pairs
                    for language in (
                        pair.left.artifact.language,
                        pair.right.artifact.language,
                    )
                }
            )
        )
    diagnostics: list[Diagnostic] = []
    if not languages:
        diagnostics.append(
            make_diagnostic(VERIFIER_UNCONSTRAINED_LANGUAGE, DiagnosticSeverity.WARNING)
        )
    means, scales, scale_diagnostics = _fit_normalization(selected, eligible_values)
    diagnostics.extend(scale_diagnostics)

    design = [
        tuple(
            (values[feature_id] - means[feature_id]) / scales[feature_id]
            for feature_id in selected
        )
        for values in eligible_values
    ]
    w = _irls(design, eligible_labels, spec.l2_lambda, spec.max_iterations, spec.tolerance)

    calibration: VerifierCalibration | None = None
    calibration_manifest: str | None = None
    if needs_calibration:
        assert calibration_pairs is not None
        sorted_calibration = sorted(calibration_pairs, key=_pair_sort_key)
        _check_pair_kinds(spec.kind, sorted_calibration, "calibration")
        logits, cal_labels, excluded = _population_scores(
            selected, means, scales, w, sorted_calibration
        )
        if excluded:
            diagnostics.append(
                make_diagnostic(
                    VERIFIER_CALIBRATION_PAIRS_EXCLUDED,
                    DiagnosticSeverity.INFO,
                    context=(("excluded_pair_count", str(excluded)),),
                )
            )
        n_cal_same = sum(cal_labels)
        if n_cal_same == 0 or n_cal_same == len(cal_labels):
            raise VerifierFitError(
                "calibration requires at least one usable pair of each class"
            )
        scores_same = [
            sigmoid_open(logit) for logit, y in zip(logits, cal_labels, strict=True) if y
        ]
        scores_different = [
            sigmoid_open(logit)
            for logit, y in zip(logits, cal_labels, strict=True)
            if not y
        ]
        thresholds, threshold_diagnostics = _fit_thresholds(
            spec, scores_same, scores_different
        )
        diagnostics.extend(threshold_diagnostics)
        if spec.calibration_method is not None:
            calibration = _fit_platt(spec, logits, cal_labels)
        calibration_manifest = pairs_manifest_sha256(sorted_calibration)
    else:
        thresholds, threshold_diagnostics = _fit_thresholds(spec, [], [])
        diagnostics.extend(threshold_diagnostics)

    diagnostics.append(
        make_diagnostic(
            VERIFIER_ELIGIBILITY,
            DiagnosticSeverity.INFO,
            context=(
                ("candidate_feature_count", str(len(candidates))),
                ("eligible_pair_count", str(eligible_pair_count)),
                ("selected_feature_count", str(len(selected))),
                ("training_pair_count", str(len(sorted_pairs))),
            ),
        )
    )

    features = tuple(
        VerifierFeature(
            feature_id=feature_id,
            semantic_version=FEATURE_SEMANTIC_VERSION,
            metric=FEATURES[feature_id].metric,
            mean=means[feature_id],
            scale=scales[feature_id],
        )
        for feature_id in selected
    )
    version = _stylog_version()
    fit_kwargs: dict[str, object] = {
        "model_id": VERIFIER_MODEL_ID,
        "model_semantic_version": VERIFIER_MODEL_SEMANTIC_VERSION,
        "task": VERIFIER_TASK,
        "task_version": VERIFIER_TASK_VERSION,
        "kind": spec.kind,
        "languages": languages,
        "feature_registry_version": FEATURE_REGISTRY_VERSION,
        "features": features,
        "coefficients": tuple(w[1:]),
        "intercept": w[0],
        "thresholds": thresholds,
        "threshold_rule": spec.threshold_rule,
        "fit_config": spec.fit_config(),
        "eligibility": VerifierEligibility(
            training_pair_count=len(sorted_pairs),
            eligible_pair_count=eligible_pair_count,
            candidate_feature_count=len(candidates),
            selected_feature_count=len(selected),
        ),
        "source_manifest_sha256": pairs_manifest_sha256(sorted_pairs),
        "runtime": current_runtime_signature(),
        "backend": BackendSignature(
            backend_id="stylog",
            implementation_version=version,
            scientific_compatibility_id=VERIFIER_MODEL_ID,
            packages=(PackageProvenance(package="stylog", version=version),),
        ),
    }
    if calibration is not None:
        fit_kwargs["calibration"] = calibration
    if calibration_manifest is not None:
        fit_kwargs["calibration_manifest_sha256"] = calibration_manifest
    if tuning_manifest_sha256 is not None:
        fit_kwargs["tuning_manifest_sha256"] = tuning_manifest_sha256
    model = VerifierFit(**fit_kwargs)
    return model, tuple(sort_diagnostics(diagnostics))
