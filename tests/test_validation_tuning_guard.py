"""Regression test: single-class populations must fail loudly, not score AUC 0.5.

The first FEVEC audit run silently scored a single-class tuning subset as
AUC 0.5 because the PAN20-large training file is class-ordered (all
same-author pairs first). The shipped AUC kernel raises instead of letting
AUC collapse to a meaningless constant; the benchmark metric omits AUC on
single-class rows (pinned in test_verify_benchmark.py). This file pins the
fail-loud invariant in shipped code.
"""

import pytest

from stylog.analysis import stats


def test_roc_auc_rejects_single_class_positive_only():
    with pytest.raises(ValueError, match="both classes nonempty"):
        stats.roc_auc_mann_whitney(positive=[0.5, 0.9], negative=[])


def test_roc_auc_rejects_single_class_negative_only():
    with pytest.raises(ValueError, match="both classes nonempty"):
        stats.roc_auc_mann_whitney(positive=[], negative=[0.1, 0.2])


def test_roc_auc_mixed_population_scores():
    assert stats.roc_auc_mann_whitney(
        positive=[0.5, 0.9], negative=[0.5, 0.1]
    ) == pytest.approx(0.875)
