"""Verifier fitting tests: deterministic IRLS, eligibility, thresholds, Platt (spec 23)."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest
from test_verify import FW, HAPAX, MORPH, TTR, make_fingerprint, ok_obs

from stylog.analysis.stats import quantile_type7
from stylog.analysis.verify import sigmoid_open, verify_fingerprints
from stylog.domain.feature import RatioValue
from stylog.exceptions import VerifierFitError
from stylog.serialization.canonical import scientific_sha256
from stylog.verification.fit import (
    THRESHOLD_BAND_COLLAPSED,
    VERIFIER_ZERO_VARIANCE_FEATURE,
    _fit_platt,
    _fit_thresholds,
    _irls,
    _select_features,
    candidate_feature_ids,
    fit_verifier_model,
    pairs_manifest_sha256,
)
from stylog.verification.spec import TrainingPair, VerifierSpec

TESTS_DIR = Path(__file__).resolve().parent


def rv(num: int, den: int = 1000) -> RatioValue:
    return RatioValue(numerator=num, denominator=den, multiplier=1.0, value=num / den)


def make_spec(**overrides) -> VerifierSpec:
    kwargs = {
        "kind": "text",
        "l2_lambda": 1.0,
        "min_support_fraction": 0.9,
        "min_class_support_fraction": 0.8,
        "min_pairs": 4,
        "threshold_rule": "fixed",
        "threshold_fixed": 0.5,
        "feature_ids": (HAPAX, TTR),
    }
    kwargs.update(overrides)
    return VerifierSpec(**kwargs)


def separable_pairs(n=8, *, lang="en") -> list[TrainingPair]:
    """Same pairs at distance ~0.001, different pairs at distance ~0.4."""
    pairs = []
    for i in range(n):
        left = make_fingerprint(
            f"s{i}a", language=lang, features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400)))
        )
        right = make_fingerprint(
            f"s{i}b", language=lang, features=(ok_obs(TTR, rv(801)), ok_obs(HAPAX, rv(400)))
        )
        pairs.append(TrainingPair(left, right, "same"))
    for i in range(n):
        left = make_fingerprint(
            f"d{i}a", language=lang, features=(ok_obs(TTR, rv(900)), ok_obs(HAPAX, rv(500)))
        )
        right = make_fingerprint(
            f"d{i}b", language=lang, features=(ok_obs(TTR, rv(500)), ok_obs(HAPAX, rv(200)))
        )
        pairs.append(TrainingPair(left, right, "different"))
    return pairs


def clustered_pairs(n=8, *, lang="en") -> list[TrainingPair]:
    """Same pairs at distance ~0.10, different pairs at distance ~0.12 (overlapping)."""
    pairs = []
    for i in range(n):
        left = make_fingerprint(
            f"cs{i}a", language=lang, features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400)))
        )
        right = make_fingerprint(
            f"cs{i}b", language=lang, features=(ok_obs(TTR, rv(700)), ok_obs(HAPAX, rv(400)))
        )
        pairs.append(TrainingPair(left, right, "same"))
    for i in range(n):
        left = make_fingerprint(
            f"cd{i}a", language=lang, features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400)))
        )
        right = make_fingerprint(
            f"cd{i}b", language=lang, features=(ok_obs(TTR, rv(680)), ok_obs(HAPAX, rv(400)))
        )
        pairs.append(TrainingPair(left, right, "different"))
    return pairs


def overlapping_pairs(*, lang="en") -> list[TrainingPair]:
    """Calibration population with class-overlapping scores (Platt converges).

    Same pairs at distances 0.10/0.20, different pairs at 0.15/0.25: some
    same pairs score below some different pairs, so the unpenalized Platt fit
    has a finite optimum.
    """
    pairs = []
    for i, delta in enumerate((100, 200)):
        for j in range(4):
            left = make_fingerprint(
                f"os{i}{j}a", language=lang,
                features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400))),
            )
            right = make_fingerprint(
                f"os{i}{j}b", language=lang,
                features=(ok_obs(TTR, rv(800 - delta)), ok_obs(HAPAX, rv(400))),
            )
            pairs.append(TrainingPair(left, right, "same"))
    for i, delta in enumerate((150, 250)):
        for j in range(4):
            left = make_fingerprint(
                f"od{i}{j}a", language=lang,
                features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400))),
            )
            right = make_fingerprint(
                f"od{i}{j}b", language=lang,
                features=(ok_obs(TTR, rv(800 - delta)), ok_obs(HAPAX, rv(400))),
            )
            pairs.append(TrainingPair(left, right, "different"))
    return pairs


# --- solver-level tests -------------------------------------------------


def test_irls_independent_reference() -> None:
    # Independent naive Newton reference (plain sums, different code path).
    design = [
        (-1.5, 0.5),
        (-0.5, -1.0),
        (0.2, 0.7),
        (1.0, -0.3),
        (1.8, 1.2),
        (-0.9, 1.4),
        (0.6, -1.6),
        (-1.1, -0.2),
    ]
    labels = [0, 0, 1, 1, 1, 0, 1, 0]
    l2 = 0.5
    w = _irls(design, labels, l2, 100, 1e-12)

    ref = [0.0, 0.0, 0.0]
    for _ in range(200):
        grad = [0.0, 0.0, 0.0]
        hess = [[0.0] * 3 for _ in range(3)]
        for row, y in zip(design, labels, strict=True):
            eta = ref[0] + ref[1] * row[0] + ref[2] * row[1]
            s = 1.0 / (1.0 + math.exp(-eta))
            r = y - s
            ww = s * (1.0 - s)
            z = (1.0, row[0], row[1])
            for a in range(3):
                grad[a] += r * z[a]
                for b in range(3):
                    hess[a][b] += ww * z[a] * z[b]
        grad[1] -= l2 * ref[1]
        grad[2] -= l2 * ref[2]
        hess[1][1] += l2
        hess[2][2] += l2
        # solve 3x3 by Cramer's rule (independent of the production solver)
        det = (
            hess[0][0] * (hess[1][1] * hess[2][2] - hess[1][2] * hess[2][1])
            - hess[0][1] * (hess[1][0] * hess[2][2] - hess[1][2] * hess[2][0])
            + hess[0][2] * (hess[1][0] * hess[2][1] - hess[1][1] * hess[2][0])
        )

        def det_replace(col, vec, matrix):
            m = [row[:] for row in matrix]
            for i in range(3):
                m[i][col] = vec[i]
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )

        delta = [det_replace(c, grad, hess) / det for c in range(3)]
        ref = [ri + di for ri, di in zip(ref, delta, strict=True)]
        if max(abs(d) for d in delta) < 1e-13:
            break
    for got, want in zip(w, ref, strict=True):
        assert got == pytest.approx(want, abs=1e-9)


def test_irls_separable_data_converges() -> None:
    # Perfectly separable with lambda > 0: finite optimum, no failure rule.
    design = [(-2.0,), (-1.0,), (1.0,), (2.0,)]
    labels = [0, 0, 1, 1]
    w = _irls(design, labels, 1.0, 100, 1e-12)
    assert all(math.isfinite(value) for value in w)
    assert w[1] > 0.0
    for row, y in zip(design, labels, strict=True):
        score = sigmoid_open(w[0] + w[1] * row[0])
        assert (score > 0.5) == bool(y)


def test_irls_nonconvergence_is_typed() -> None:
    design = [(-2.0,), (-1.0,), (1.0,), (2.0,)]
    labels = [0, 0, 1, 1]
    with pytest.raises(VerifierFitError, match="converge"):
        _irls(design, labels, 1.0, 1, 1e-12)


def test_irls_pivot_guard_degenerate_design() -> None:
    # Constant feature column with lambda=0: rank-deficient system.
    design = [(1.0,), (1.0,), (1.0,), (1.0,)]
    labels = [0, 0, 1, 1]
    with pytest.raises(VerifierFitError):
        _irls(design, labels, 0.0, 100, 1e-12)


def test_sklearn_differential_oracle() -> None:
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    # Deterministic non-separable fixture.
    design = []
    labels = []
    for i in range(60):
        row = tuple(
            (((i * (j + 7) + 13) % 41) / 20.0 - 1.0) * (1 if (i + j) % 2 else -1)
            for j in range(4)
        )
        design.append(row)
        labels.append(1 if row[0] + 0.5 * row[1] + 0.25 * ((i % 3) - 1) > 0 else 0)
    assert 0 < sum(labels) < len(labels)
    l2 = 1.0
    w = _irls(design, labels, l2, 500, 1e-14)
    model = sklearn_linear.LogisticRegression(
        C=1.0 / l2,
        fit_intercept=True,
        solver="newton-cg",
        tol=1e-16,
        max_iter=1000000,
    )
    model.fit(design, labels)
    sk_coef = list(model.coef_[0]) + [model.intercept_[0]]
    our_coef = list(w[1:]) + [w[0]]
    for ours, theirs in zip(our_coef, sk_coef, strict=True):
        assert ours == pytest.approx(theirs, abs=1e-6)
    for row in design:
        logit_ours = math.fsum([w[0]] + [w[j + 1] * row[j] for j in range(4)])
        logit_theirs = model.decision_function([row])[0]
        assert sigmoid_open(logit_ours) == pytest.approx(
            1.0 / (1.0 + math.exp(-logit_theirs)), abs=1e-9
        )


def test_platt_scipy_oracle() -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    # class-overlapping logits (unpenalized Platt has a finite optimum)
    logits = [-2.0, -1.0, -0.5, 0.4, 0.05, 0.8, 1.5, 2.5, -0.2, 0.1]
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    spec = make_spec()
    calibration = _fit_platt(spec, logits, labels)

    def nll(params):
        a, b = params
        total = 0.0
        for x, y in zip(logits, labels, strict=True):
            s = 1.0 / (1.0 + math.exp(-(a * x + b)))
            s = min(max(s, 1e-300), 1.0 - 1e-16)
            total += -(y * math.log(s) + (1 - y) * math.log(1.0 - s))
        return total

    result = scipy_optimize.minimize(
        nll, [0.0, 0.0], method="BFGS", options={"gtol": 1e-12}
    )
    assert calibration.a == pytest.approx(result.x[0], abs=1e-6)
    assert calibration.b == pytest.approx(result.x[1], abs=1e-6)


# --- candidacy / eligibility --------------------------------------------


def test_candidacy_base_default_core_only() -> None:
    spec = make_spec(feature_ids=None)
    candidates = candidate_feature_ids(spec, ("en",))
    assert candidates
    assert all(not cid.startswith("text.linguistic.") for cid in candidates)
    linguistic_spec = make_spec(feature_ids=None, include_linguistic=True)
    with_linguistic = candidate_feature_ids(linguistic_spec, ("en",))
    assert any(cid.startswith("text.linguistic.") for cid in with_linguistic)
    assert MORPH in with_linguistic


def test_candidacy_language_scope() -> None:
    spec = make_spec(feature_ids=None)
    english = candidate_feature_ids(spec, ("en",))
    assert FW in english
    french = candidate_feature_ids(spec, ("fr",))
    assert FW not in french
    unconstrained = candidate_feature_ids(spec, ())
    assert FW not in unconstrained
    multilingual = candidate_feature_ids(spec, ("en", "fr"))
    assert FW not in multilingual  # scope must fully contain the model languages


def test_candidacy_explicit_feature_ids_filtered() -> None:
    # feature_ids must be sorted: lexical < linguistic
    ordered = (FW, HAPAX, TTR, MORPH)
    spec = make_spec(feature_ids=ordered)
    # fr model: English function words drop out; linguistic needs the flag
    french = candidate_feature_ids(spec, ("fr",))
    assert french == (HAPAX, TTR)
    english = candidate_feature_ids(
        make_spec(feature_ids=ordered, include_linguistic=True), ("en",)
    )
    assert english == ordered


def test_candidacy_kind_and_metric_validation() -> None:
    with pytest.raises(VerifierFitError, match="unknown feature_id"):
        candidate_feature_ids(make_spec(feature_ids=("text.nope.feature",)), ("en",))
    with pytest.raises(VerifierFitError, match="no comparison metric"):
        candidate_feature_ids(make_spec(feature_ids=("text.lexical.word_count",)), ("en",))
    with pytest.raises(VerifierFitError, match="kind"):
        candidate_feature_ids(
            make_spec(feature_ids=("code.surface.blank_line_share",)), ("en",)
        )
    code_spec = make_spec(kind="code", feature_ids=None)
    code_candidates = candidate_feature_ids(code_spec, ("python",))
    assert code_candidates
    assert all(cid.startswith("code.") for cid in code_candidates)


def test_select_features_class_aware_rule() -> None:
    spec = make_spec(min_support_fraction=0.5, min_class_support_fraction=0.5)
    candidates = (HAPAX, TTR)
    # HAPAX supported everywhere; TTR supported only in the different class.
    pair_values = [{HAPAX: 0.1, TTR: 0.2}, {HAPAX: 0.1, TTR: 0.3}, {HAPAX: 0.2}, {HAPAX: 0.2}]
    labels = [0, 0, 1, 1]
    selected = _select_features(spec, candidates, pair_values, labels)
    assert selected == (HAPAX,)
    relaxed = make_spec(min_support_fraction=0.5, min_class_support_fraction=0.4)
    # TTR: overall 2/4 = 0.5 ok; same-class 0/2 = 0.0 -> still excluded
    assert _select_features(relaxed, candidates, pair_values, labels) == (HAPAX,)
    all_supported = [{HAPAX: 0.1, TTR: 0.2}] * 4
    assert _select_features(spec, candidates, all_supported, labels) == (HAPAX, TTR)


# --- thresholds ----------------------------------------------------------


def test_threshold_direction_golden() -> None:
    spec = make_spec(threshold_rule="calibration_quantile_band", threshold_alpha=0.2,
                     threshold_fixed=None)
    scores_same = [0.55, 0.6, 0.65, 0.7, 0.9]
    scores_different = [0.1, 0.2, 0.3, 0.4, 0.45]
    thresholds, diagnostics = _fit_thresholds(spec, scores_same, scores_different)
    assert diagnostics == []
    # t_same is the LOW quantile of same-class scores; t_diff the HIGH quantile
    # of different-class scores (swapping them would flip the band).
    assert thresholds.t_same == quantile_type7(sorted(scores_same), 0.2)
    assert thresholds.t_diff == quantile_type7(sorted(scores_different), 0.8)
    assert thresholds.t_diff < thresholds.t_same


def test_threshold_band_collapse_golden() -> None:
    spec = make_spec(threshold_rule="calibration_quantile_band", threshold_alpha=0.25,
                     threshold_fixed=None)
    scores_same = [0.2, 0.3]
    scores_different = [0.7, 0.8]
    thresholds, diagnostics = _fit_thresholds(spec, scores_same, scores_different)
    t_same_q = quantile_type7(sorted(scores_same), 0.25)
    t_diff_q = quantile_type7(sorted(scores_different), 0.75)
    assert t_diff_q > t_same_q  # crossed quantiles force the collapse
    midpoint = (t_same_q + t_diff_q) / 2.0
    assert thresholds.t_same == thresholds.t_diff == midpoint
    assert [d.code for d in diagnostics] == [THRESHOLD_BAND_COLLAPSED]


def test_thresholds_follow_calibration_not_training() -> None:
    training = separable_pairs()
    calibration = clustered_pairs()
    spec = make_spec(
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.25,
        threshold_fixed=None,
        feature_ids=(TTR,),
    )
    model, _ = fit_verifier_model(spec, training, calibration_pairs=calibration)
    # thresholds recomputed from the model's own calibration scores
    same_scores = []
    different_scores = []
    for pair in calibration:
        verification = verify_fingerprints(model, pair.left, pair.right)
        assert verification.score is not None
        (same_scores if pair.label == "same" else different_scores).append(verification.score)
    expected_t_same = quantile_type7(sorted(same_scores), 0.25)
    expected_t_diff = quantile_type7(sorted(different_scores), 0.75)
    if expected_t_diff > expected_t_same:
        midpoint = (expected_t_same + expected_t_diff) / 2.0
        assert model.thresholds.t_same == model.thresholds.t_diff == midpoint
    else:
        assert model.thresholds.t_same == expected_t_same
        assert model.thresholds.t_diff == expected_t_diff


def test_thresholds_differ_with_calibration_population() -> None:
    training = separable_pairs()
    spec = make_spec(
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.25,
        threshold_fixed=None,
        feature_ids=(TTR,),
    )
    model_a, _ = fit_verifier_model(
        spec, training, calibration_pairs=separable_pairs()
    )
    model_b, _ = fit_verifier_model(
        spec, training, calibration_pairs=clustered_pairs()
    )
    # coefficients are identical (training-only); thresholds follow calibration
    assert model_a.coefficients == model_b.coefficients
    assert model_a.intercept == model_b.intercept
    assert (model_a.thresholds.t_same, model_a.thresholds.t_diff) != (
        model_b.thresholds.t_same,
        model_b.thresholds.t_diff,
    )
    assert model_a.source_manifest_sha256 == model_b.source_manifest_sha256
    assert model_a.calibration_manifest_sha256 != model_b.calibration_manifest_sha256


# --- end-to-end fits -----------------------------------------------------


def test_fit_determinism_repeated_and_shuffled() -> None:
    pairs = separable_pairs()
    spec = make_spec()
    model_a, _ = fit_verifier_model(spec, pairs)
    model_b, _ = fit_verifier_model(spec, pairs)
    assert scientific_sha256(model_a) == scientific_sha256(model_b)
    shuffled = list(pairs)
    import random

    rng = random.Random(1234)
    rng.shuffle(shuffled)
    model_c, _ = fit_verifier_model(spec, shuffled)
    assert scientific_sha256(model_a) == scientific_sha256(model_c)


def test_fit_determinism_fresh_process() -> None:
    pairs = separable_pairs()
    spec = make_spec()
    model, _ = fit_verifier_model(spec, pairs)
    script = (
        f"import sys; sys.path.insert(0, {str(TESTS_DIR)!r});"
        "from test_verify_fit import separable_pairs, make_spec;"
        "from stylog.verification.fit import fit_verifier_model;"
        "from stylog.serialization.canonical import scientific_sha256;"
        "m, _ = fit_verifier_model(make_spec(), separable_pairs());"
        "print(scientific_sha256(m))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    assert result.stdout.strip() == scientific_sha256(model)


def test_fit_coefficient_goldens() -> None:
    # Same-runtime regression pin (spec 23.11 scopes byte determinism to the
    # recorded runtime; libm last-ulp drift across platforms is tolerated by
    # the 1e-12 window).
    model, _ = fit_verifier_model(make_spec(feature_ids=(TTR,)), separable_pairs())
    assert model.coefficients == pytest.approx((-1.9656720992272874,), abs=1e-12)
    assert model.intercept == pytest.approx(2.8939940353785986e-16, abs=1e-12)
    assert model.features[0].mean == pytest.approx(0.2005, abs=1e-12)


def test_fit_languages_pinned_from_training() -> None:
    model, _ = fit_verifier_model(make_spec(), separable_pairs(lang="fr"))
    assert model.languages == ("fr",)
    # a fr-pinned model can never contain English function-word features
    assert all(not f.feature_id.startswith("text.function_words.en.") for f in model.features)
    fr_right = make_fingerprint(
        "x", language="en", features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400)))
    )
    pair = separable_pairs(lang="fr")[0]
    from stylog.exceptions import ModelIncompatibilityError

    with pytest.raises(ModelIncompatibilityError):
        verify_fingerprints(model, pair.left, fr_right)


def test_fit_language_scope_override() -> None:
    spec = make_spec(languages=("fr",))
    model, _ = fit_verifier_model(spec, separable_pairs(lang="en"))
    assert model.languages == ("fr",)


def test_fit_zero_variance_feature_diagnostic() -> None:
    pairs = []
    for i in range(8):
        left = make_fingerprint(
            f"a{i}", features=(ok_obs(TTR, rv(800)), ok_obs(HAPAX, rv(400)))
        )
        right = make_fingerprint(
            f"b{i}", features=(ok_obs(TTR, rv(800 + (i % 2))), ok_obs(HAPAX, rv(400)))
        )
        pairs.append(TrainingPair(left, right, "same" if i % 2 == 0 else "different"))
    # HAPAX distances are all exactly 0.0 -> zero variance
    model, diagnostics = fit_verifier_model(make_spec(), pairs)
    scales = {f.feature_id: f.scale for f in model.features}
    assert scales[HAPAX] == 1.0
    assert VERIFIER_ZERO_VARIANCE_FEATURE in [d.code for d in diagnostics]
    zero_variance = [d for d in diagnostics if d.code == VERIFIER_ZERO_VARIANCE_FEATURE]
    assert zero_variance[0].feature_id == HAPAX


def test_fit_platt_end_to_end_and_probability() -> None:
    training = separable_pairs()
    calibration = overlapping_pairs()
    spec = make_spec(
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.25,
        threshold_fixed=None,
        calibration_method="platt",
        feature_ids=(TTR,),
    )
    model, _ = fit_verifier_model(spec, training, calibration_pairs=calibration)
    assert model.calibration is not None
    assert model.calibration.method == "platt"
    assert model.calibration_manifest_sha256 is not None
    verification = verify_fingerprints(model, training[0].left, training[0].right)
    assert verification.probability is not None
    assert verification.calibration_method == "platt"
    assert 0.0 < verification.probability < 1.0


def test_platt_invariant_to_tuning_manifest() -> None:
    training = separable_pairs()
    calibration = overlapping_pairs()
    spec = make_spec(
        threshold_rule="calibration_quantile_band",
        threshold_alpha=0.25,
        threshold_fixed=None,
        calibration_method="platt",
        feature_ids=(TTR,),
    )
    model_a, _ = fit_verifier_model(spec, training, calibration_pairs=calibration)
    model_b, _ = fit_verifier_model(
        spec,
        training,
        calibration_pairs=calibration,
        tuning_manifest_sha256="9" * 64,
    )
    # hyperparameters frozen: tuning identity does not perturb fitted numbers
    assert model_a.coefficients == model_b.coefficients
    assert model_a.calibration == model_b.calibration
    assert model_a.thresholds == model_b.thresholds
    # but it is part of the complete-model identity
    assert scientific_sha256(model_a) != scientific_sha256(model_b)
    assert model_a.tuning_manifest_sha256 is None
    assert model_b.tuning_manifest_sha256 == "9" * 64


def test_pairs_manifest_hash_order_invariant_and_sensitive() -> None:
    pairs = separable_pairs()
    assert pairs_manifest_sha256(pairs) == pairs_manifest_sha256(list(reversed(pairs)))
    other = separable_pairs()[:-1]
    assert pairs_manifest_sha256(pairs) != pairs_manifest_sha256(other)


def test_fit_error_matrix() -> None:
    spec = make_spec()
    with pytest.raises(VerifierFitError, match="no training pairs"):
        fit_verifier_model(spec, [])
    same_only = [pair for pair in separable_pairs() if pair.label == "same"]
    with pytest.raises(VerifierFitError, match="each class"):
        fit_verifier_model(spec, same_only)
    with pytest.raises(VerifierFitError, match="min_pairs"):
        fit_verifier_model(make_spec(min_pairs=1000), separable_pairs())
    # no surviving features: TTR everywhere insufficient
    pairs = []
    from stylog.analysis.registry import FEATURES as _F
    from stylog.domain.feature import InsufficientSupportObservation

    for i in range(8):
        bad = InsufficientSupportObservation(
            feature_id=TTR,
            semantic_version="1.0.0",
            analyzer_id=_F[TTR].analyzer_id,
            analyzer_implementation_version="1.0.0",
        )
        left = make_fingerprint(f"a{i}", features=(ok_obs(HAPAX, rv(400)), bad))
        right = make_fingerprint(f"b{i}", features=(ok_obs(HAPAX, rv(400)), bad))
        pairs.append(TrainingPair(left, right, "same" if i % 2 else "different"))
    with pytest.raises(VerifierFitError, match="no candidate feature"):
        fit_verifier_model(make_spec(feature_ids=(TTR,)), pairs)
    # kind mismatch
    from stylog.domain.artifact import ArtifactKind

    code_fp = make_fingerprint(
        "c", kind=ArtifactKind.CODE, language="python", features=()
    )
    mixed = [TrainingPair(separable_pairs()[0].left, code_fp, "same")]
    mixed += separable_pairs()[1:]
    with pytest.raises(VerifierFitError, match="kinds"):
        fit_verifier_model(spec, mixed)
    # calibration split required
    with pytest.raises(VerifierFitError, match="calibration split"):
        fit_verifier_model(
            make_spec(
                threshold_rule="calibration_quantile_band",
                threshold_alpha=0.25,
                threshold_fixed=None,
            ),
            separable_pairs(),
        )
    # calibration single class
    with pytest.raises(VerifierFitError, match="calibration requires"):
        fit_verifier_model(
            make_spec(
                threshold_rule="calibration_quantile_band",
                threshold_alpha=0.25,
                threshold_fixed=None,
            ),
            separable_pairs(),
            calibration_pairs=[p for p in separable_pairs() if p.label == "same"],
        )


def test_spec_validation() -> None:
    with pytest.raises(VerifierFitError):
        make_spec(l2_lambda=0.0)
    with pytest.raises(VerifierFitError):
        make_spec(threshold_rule="bogus")
    with pytest.raises(VerifierFitError):
        make_spec(threshold_rule="calibration_quantile_band", threshold_alpha=None,
                  threshold_fixed=None)
    with pytest.raises(VerifierFitError):
        make_spec(threshold_rule="calibration_quantile_band", threshold_alpha=0.7,
                  threshold_fixed=None)
    with pytest.raises(VerifierFitError):
        make_spec(kind="hybrid")
    with pytest.raises(VerifierFitError):
        make_spec(languages=("fr", "en"))
    spec = make_spec()
    config = spec.fit_config()
    assert config.threshold_rule == "fixed"
    assert config.pair_policy.selection_version == "1"
