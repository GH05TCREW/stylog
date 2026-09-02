"""Verifier fit specification (spec 23.10-23.15).

``VerifierSpec`` is a plain frozen dataclass of fit parameters — not a
portable artifact. The portable twin embedded in the fitted model is
``VerifierFitConfig``; ``VerifierSpec.fit_config()`` converts. There is no
algorithm registry: exactly one algorithm exists (``VERIFIER_MODEL_ID``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stylog.analysis.verify import VERIFIER_MODEL_ID, VERIFIER_MODEL_SEMANTIC_VERSION
from stylog.domain._base import is_sorted_unique
from stylog.domain.fingerprint import Fingerprint
from stylog.domain.verification import (
    CALIBRATION_METHOD_PLATT,
    THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND,
    THRESHOLD_RULE_FIXED,
    THRESHOLD_RULES,
    VerifierFitConfig,
    VerifierPairPolicy,
)
from stylog.exceptions import VerifierFitError

__all__ = [
    "VERIFIER_MODEL_ID",
    "VERIFIER_MODEL_SEMANTIC_VERSION",
    "TrainingPair",
    "VerifierSpec",
]

LABEL_SAME = "same"
LABEL_DIFFERENT = "different"
LABELS = (LABEL_SAME, LABEL_DIFFERENT)


@dataclass(frozen=True)
class TrainingPair:
    """One labeled pair of fingerprints for fitting/calibration."""

    left: Fingerprint
    right: Fingerprint
    label: str

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise VerifierFitError(f"training pair label must be one of {LABELS}")


@dataclass(frozen=True)
class VerifierSpec:
    """Explicit fit parameters for a pairwise authorship verifier."""

    kind: str  # "text" | "code"
    l2_lambda: float
    min_support_fraction: float
    min_class_support_fraction: float
    min_pairs: int
    threshold_rule: str
    threshold_alpha: float | None = None  # required iff calibration_quantile_band
    threshold_fixed: float | None = None  # required iff fixed
    calibration_method: str | None = None  # "platt" or None
    max_iterations: int = 100
    tolerance: float = 1e-12
    include_linguistic: bool = False
    allow_unconstrained_language: bool = False
    languages: tuple[str, ...] | None = None  # explicit scope override
    feature_ids: tuple[str, ...] | None = None  # explicit ablation subset
    pair_policy: VerifierPairPolicy = field(
        default_factory=lambda: VerifierPairPolicy(selection_version="1")
    )

    def __post_init__(self) -> None:
        if self.kind not in ("text", "code"):
            raise VerifierFitError(f"verifier kind must be 'text' or 'code', got {self.kind!r}")
        if self.l2_lambda <= 0.0:
            raise VerifierFitError("l2_lambda must be > 0")
        if not (0.0 < self.min_support_fraction <= 1.0):
            raise VerifierFitError("min_support_fraction must be in (0, 1]")
        if not (0.0 < self.min_class_support_fraction <= 1.0):
            raise VerifierFitError("min_class_support_fraction must be in (0, 1]")
        if self.min_pairs < 1:
            raise VerifierFitError("min_pairs must be >= 1")
        if self.max_iterations < 1:
            raise VerifierFitError("max_iterations must be >= 1")
        if self.tolerance <= 0.0:
            raise VerifierFitError("tolerance must be > 0")
        if self.threshold_rule not in THRESHOLD_RULES:
            raise VerifierFitError(f"threshold_rule must be one of {THRESHOLD_RULES}")
        if self.threshold_rule == THRESHOLD_RULE_CALIBRATION_QUANTILE_BAND:
            if self.threshold_alpha is None or not (0.0 < self.threshold_alpha <= 0.5):
                raise VerifierFitError(
                    "threshold_alpha in (0, 0.5] is required for calibration_quantile_band"
                )
            if self.threshold_fixed is not None:
                raise VerifierFitError(
                    "threshold_fixed must be omitted for calibration_quantile_band"
                )
        if self.threshold_rule == THRESHOLD_RULE_FIXED:
            if self.threshold_fixed is None or not (0.0 < self.threshold_fixed < 1.0):
                raise VerifierFitError("threshold_fixed in (0, 1) is required for the fixed rule")
            if self.threshold_alpha is not None:
                raise VerifierFitError("threshold_alpha must be omitted for the fixed rule")
        if self.calibration_method is not None and self.calibration_method != (
            CALIBRATION_METHOD_PLATT
        ):
            raise VerifierFitError("calibration_method must be 'platt' when present")
        if self.languages is not None:
            languages = list(self.languages)
            if not is_sorted_unique(languages):
                raise VerifierFitError("languages must be sorted and unique")
            if not languages and not self.allow_unconstrained_language:
                raise VerifierFitError(
                    "empty languages requires allow_unconstrained_language"
                )
        if self.feature_ids is not None:
            ids = list(self.feature_ids)
            if not is_sorted_unique(ids):
                raise VerifierFitError("feature_ids must be sorted by unique feature_id")

    def fit_config(self) -> VerifierFitConfig:
        """The portable fit configuration embedded in the fitted model."""
        kwargs = {
            "l2_lambda": self.l2_lambda,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "min_support_fraction": self.min_support_fraction,
            "min_class_support_fraction": self.min_class_support_fraction,
            "min_pairs": self.min_pairs,
            "threshold_rule": self.threshold_rule,
            "include_linguistic": self.include_linguistic,
            "allow_unconstrained_language": self.allow_unconstrained_language,
            "pair_policy": self.pair_policy,
        }
        if self.threshold_alpha is not None:
            kwargs["threshold_alpha"] = self.threshold_alpha
        if self.threshold_fixed is not None:
            kwargs["threshold_fixed"] = self.threshold_fixed
        if self.calibration_method is not None:
            kwargs["calibration_method"] = self.calibration_method
        if self.feature_ids is not None:
            kwargs["feature_ids"] = self.feature_ids
        return VerifierFitConfig(**kwargs)
