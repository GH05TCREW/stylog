"""Reference mathematical kernels (pure Python, deterministic).

These implementations are the semantic reference for quantiles, midrank
percentiles, MAD, JSD2, Wasserstein-1, SPD, and ROC-AUC. All summation uses
``math.fsum`` over sorted keys so results are independent of hash seed and
worker scheduling.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping, Sequence

MAD_NORMAL_CONSISTENCY_FACTOR = 1.482602218505602


def _nz(value: float) -> float:
    """Normalize -0.0 to 0.0."""
    return 0.0 if value == 0.0 else value


def quantile_type7(sorted_values: Sequence[float], p: float) -> float:
    """Hyndman-Fan type-7 quantile over ascending values (spec 13.3)."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("quantile of empty sequence")
    h = (n - 1) * p
    j = math.floor(h)
    g = h - j
    xj = sorted_values[j]
    xj1 = sorted_values[min(j + 1, n - 1)]
    return _nz((1.0 - g) * xj + g * xj1)


def summary_statistics(values: Iterable[float]) -> tuple[int, float, float, float, float, float, float, float | None]:
    """(n, min, q25, median, q75, max, mean, sample_sd) with sample sd n-1."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        raise ValueError("summary of empty sequence")
    mean = _nz(math.fsum(xs) / n)
    q25 = quantile_type7(xs, 0.25)
    med = quantile_type7(xs, 0.50)
    q75 = quantile_type7(xs, 0.75)
    if n == 1:
        return (1, xs[0], q25, med, q75, xs[0], mean, None)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in xs) / (n - 1))
    return (n, xs[0], q25, med, q75, xs[-1], mean, _nz(sd))


def midrank_percentile(sorted_baseline: Sequence[float], observed: float) -> float:
    """100 * (L + 0.5*E) / N with midrank tie handling (spec 13.4)."""
    n = len(sorted_baseline)
    if n == 0:
        raise ValueError("percentile of empty baseline")
    below = bisect_left(sorted_baseline, observed)
    equal = bisect_right(sorted_baseline, observed) - below
    return _nz(100.0 * (below + 0.5 * equal) / n)


def median_absolute_deviation(sorted_values: Sequence[float]) -> tuple[float, float]:
    """(mad_raw, mad_normal_scaled) per spec 13.5."""
    if not sorted_values:
        raise ValueError("MAD of empty sequence")
    med = quantile_type7(sorted_values, 0.5)
    deviations = sorted(abs(x - med) for x in sorted_values)
    mad_raw = quantile_type7(deviations, 0.5)
    return mad_raw, _nz(mad_raw * MAD_NORMAL_CONSISTENCY_FACTOR)


def abs_distance(a: float, b: float) -> float:
    return _nz(abs(a - b))


def symmetric_proportional_distance(a: float, b: float) -> float:
    """SPD: 0 when a == b == 0 else 2|a-b| / (|a|+|b|), range [0, 2]."""
    if a == 0.0 and b == 0.0:
        return 0.0
    return _nz(2.0 * abs(a - b) / (abs(a) + abs(b)))


def jensen_shannon_distance2(
    p_counts: Mapping[str, int],
    p_total: int,
    q_counts: Mapping[str, int],
    q_total: int,
) -> float:
    """Jensen-Shannon distance (base 2). 0 = identical, 1 = disjoint (spec 12.5)."""
    keys = sorted(set(p_counts) | set(q_counts))
    kl_pm: list[float] = []
    kl_qm: list[float] = []
    for key in keys:
        pc = p_counts.get(key, 0)
        qc = q_counts.get(key, 0)
        pk = pc / p_total if pc else 0.0
        qk = qc / q_total if qc else 0.0
        mk = (pk + qk) / 2.0
        if pk > 0.0:
            kl_pm.append(pk * math.log2(pk / mk))
        if qk > 0.0:
            kl_qm.append(qk * math.log2(qk / mk))
    return _nz(math.sqrt(0.5 * math.fsum(kl_pm) + 0.5 * math.fsum(kl_qm)))


def wasserstein_1(
    p_counts: Mapping[int, int],
    p_total: int,
    q_counts: Mapping[int, int],
    q_total: int,
) -> float:
    """1D Wasserstein-1 over explicit (top-coded) integer support (spec 12.6)."""
    support = sorted(set(p_counts) | set(q_counts))
    if len(support) < 2:
        return 0.0
    cdf_diff = 0.0
    terms: list[float] = []
    for index in range(len(support) - 1):
        point = support[index]
        cdf_diff += p_counts.get(point, 0) / p_total - q_counts.get(point, 0) / q_total
        terms.append(abs(cdf_diff) * (support[index + 1] - point))
    return _nz(math.fsum(terms))


def wasserstein_1_samples(left: Sequence[float], right: Sequence[float]) -> float:
    """Wasserstein-1 between two empirical scalar samples (for sample summaries)."""
    xs = sorted(float(v) for v in left)
    ys = sorted(float(v) for v in right)
    if not xs or not ys:
        raise ValueError("sample Wasserstein requires nonempty samples")
    support = sorted(set(xs) | set(ys))
    if len(support) < 2:
        return 0.0

    def cdf_at(values: Sequence[float], point: float) -> float:
        return bisect_right(values, point) / len(values)

    terms = []
    for index in range(len(support) - 1):
        point = support[index]
        diff = cdf_at(xs, point) - cdf_at(ys, point)
        terms.append(abs(diff) * (support[index + 1] - point))
    return _nz(math.fsum(terms))


def roc_auc_mann_whitney(positive: Sequence[float], negative: Sequence[float]) -> float:
    """ROC-AUC via Mann-Whitney ranks with midranks for ties (spec 21.11)."""
    n_pos = len(positive)
    n_neg = len(negative)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both classes nonempty")
    combined = sorted([(float(s), 1) for s in positive] + [(float(s), 0) for s in negative])
    ranks = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        end = index
        while end + 1 < len(combined) and combined[end + 1][0] == combined[index][0]:
            end += 1
        midrank = (index + 1 + end + 1) / 2.0  # ranks are 1-based
        for j in range(index, end + 1):
            ranks[j] = midrank
        index = end + 1
    r_pos = math.fsum(rank for rank, (_, label) in zip(ranks, combined, strict=True) if label == 1)
    return _nz((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def shannon_entropy_bits(counts: Iterable[int], total: int) -> float:
    """-sum p_i log2(p_i), no smoothing (spec 7.13)."""
    terms = []
    for count in counts:
        if count > 0:
            p_i = count / total
            terms.append(p_i * math.log2(p_i))
    return _nz(-math.fsum(terms))


def simpson_concentration(counts: Iterable[int], total: int) -> float:
    """sum p_i^2 (spec 7.13)."""
    return _nz(math.fsum((count / total) ** 2 for count in counts if count > 0))
