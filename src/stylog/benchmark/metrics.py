"""PAN-style decision metrics for the verification benchmark task (spec 21).

Pure functions over decision rows -- benchmark-only outputs, never domain
types or Comparison semantics. Formulas are pinned to the official PAN
authorship-verification evaluator (pan-webis-de/pan-code; identical across
the PAN20-23 evaluators):

- c@1 (Penas & Rodrigo 2011): ``(1/N) * (nc + nu * nc / N)`` where nc counts
  answered-and-correct and nu counts abstentions.
- F1: abstentions are excluded entirely; standard binary F1 with the
  same-author class as positive (zero-division yields 0.0, sklearn's
  default).
- F0.5u (Bevendorff et al. 2019, non-answers treated as false negatives):
  ``1.25*TP / (1.25*TP + 0.25*(FN + u) + FP)`` where u counts ALL
  abstentions regardless of true class.
- ROC AUC: Mann-Whitney midrank AUC over raw scores, positive class = same.
- Brier: plain mean squared error of probability vs 0/1 label (the official
  evaluator reports the complement 1 - BS; this module reports the loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stylog.analysis.stats import roc_auc_mann_whitney


@dataclass(frozen=True)
class DecisionRow:
    """One verification outcome against its truth label."""

    verdict: str  # "same_author" | "different_author" | "abstain"
    label: str  # "same" | "different"
    score: float | None  # absent on insufficient evidence
    probability: float | None  # absent unless calibrated and scored

    @property
    def answered(self) -> bool:
        return self.verdict in ("same_author", "different_author")


def _confusion(rows: list[DecisionRow]) -> tuple[int, int, int, int, int]:
    """(tp, fp, fn, tn, unanswered) over answered rows; abstentions = unanswered."""
    tp = fp = fn = tn = unanswered = 0
    for row in rows:
        if not row.answered:
            unanswered += 1
        elif row.verdict == "same_author" and row.label == "same":
            tp += 1
        elif row.verdict == "same_author":
            fp += 1
        elif row.label == "same":
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn, unanswered


def f1(rows: list[DecisionRow]) -> float | None:
    """Binary F1 over answered rows only; None when no row was answered."""
    tp, fp, fn, tn, _unanswered = _confusion(rows)
    answered = tp + fp + fn + tn
    if answered == 0:
        return None
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return (2.0 * tp) / denominator


def c_at_1(rows: list[DecisionRow]) -> float:
    """Penas & Rodrigo c@1; abstentions rewarded proportionally to accuracy."""
    n = len(rows)
    if n == 0:
        raise ValueError("c@1 requires at least one row")
    nc = sum(
        1
        for row in rows
        if row.answered
        and (row.verdict == "same_author") == (row.label == "same")
    )
    nu = n - sum(1 for row in rows if row.answered)
    return (1.0 / n) * (nc + (nu * nc / n))


def f_05u(rows: list[DecisionRow]) -> float:
    """F0.5u: F0.5 with every abstention counted as a false negative."""
    if not rows:
        raise ValueError("F0.5u requires at least one row")
    tp, fp, fn, _tn, unanswered = _confusion(rows)
    denominator = 1.25 * tp + 0.25 * (fn + unanswered) + fp
    if denominator == 0.0:
        return 0.0
    return (1.25 * tp) / denominator


def roc_auc(rows: list[DecisionRow]) -> float | None:
    """AUC over rows with a score; None unless both classes are present."""
    scored = [row for row in rows if row.score is not None]
    positive = [row.score for row in scored if row.label == "same"]
    negative = [row.score for row in scored if row.label == "different"]
    if not positive or not negative:
        return None
    return roc_auc_mann_whitney(positive=positive, negative=negative)


def brier(rows: list[DecisionRow]) -> float | None:
    """Mean squared error of probability vs label over rows with probabilities."""
    probs = [row for row in rows if row.probability is not None]
    if not probs:
        return None
    return math.fsum(
        (row.probability - (1.0 if row.label == "same" else 0.0)) ** 2 for row in probs
    ) / len(probs)
