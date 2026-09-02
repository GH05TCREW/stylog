"""Conformance tests for the reference statistical kernels (spec 25.15-25.19, 25.24)."""

from __future__ import annotations

import pytest

from stylog.analysis import build, stats


def test_quantile_type7_knots():
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert stats.quantile_type7(values, 0.25) == 10.0
    assert stats.quantile_type7(values, 0.5) == 20.0
    assert stats.quantile_type7(values, 0.75) == 30.0


def test_quantile_type7_non_knot():
    assert stats.quantile_type7([0.0, 10.0, 20.0, 30.0], 0.5) == 15.0


def test_midrank_percentile_ties():
    # [1, 2, 2, 4], observed 2: L=1, E=2, N=4 -> 100 * (1 + 1) / 4.
    assert stats.midrank_percentile([1.0, 2.0, 2.0, 4.0], 2.0) == 50.0


def test_mad_all_identical_is_zero():
    mad_raw, mad_normal_scaled = stats.median_absolute_deviation([5.0] * 6)
    assert mad_raw == 0.0
    assert mad_normal_scaled == 0.0


def test_jensen_shannon_distance2_identical_and_disjoint():
    assert stats.jensen_shannon_distance2({"a": 1}, 1, {"a": 1}, 1) == 0.0
    assert stats.jensen_shannon_distance2({"a": 1}, 1, {"b": 1}, 1) == 1.0


def test_wasserstein_1_point_mass_distance():
    assert stats.wasserstein_1({0: 1}, 1, {3: 1}, 1) == 3.0


def test_wasserstein_1_top_code_collapse():
    # Raw 500 vs 999 under top_code 201 both transform to point 201.
    left = build.histogram_value([500], 201)
    right = build.histogram_value([999], 201)
    assert left is not None and right is not None
    assert left.points == right.points
    distance = stats.wasserstein_1(
        {entry.point: entry.count for entry in left.points},
        left.total,
        {entry.point: entry.count for entry in right.points},
        right.total,
    )
    assert distance == 0.0


def test_symmetric_proportional_distance_both_zero():
    assert stats.symmetric_proportional_distance(0.0, 0.0) == 0.0


def test_symmetric_proportional_distance_normal():
    # 2*|1-3| / (|1|+|3|) = 1.0
    assert stats.symmetric_proportional_distance(1.0, 3.0) == 1.0


def test_roc_auc_midrank_ties():
    # Scores sorted: 0.1(neg, rank 1), 0.5(pos+neg tie, midrank 2.5), 0.9(pos, rank 4).
    # R_pos = 2.5 + 4 = 6.5; AUC = (6.5 - 2*3/2) / (2*2) = 0.875.
    assert stats.roc_auc_mann_whitney([0.5, 0.9], [0.5, 0.1]) == pytest.approx(0.875)
