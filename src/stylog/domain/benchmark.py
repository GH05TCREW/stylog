"""Portable benchmark models (spec section 21).

Benchmarking in v0.1 is descriptive and audit-oriented: split realizations,
per-feature pairwise distance summaries, and transformation distances. The
descriptive tasks carry no attribution, no probabilities, no thresholds, no
EER. Decision-level metrics exist only as benchmark outputs of the
verification task (``VerificationMetrics``), computed under explicit fitted
verifier models -- they never feed back into domain semantics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from stylog.domain._base import (
    HexDigest64,
    PortableFloat,
    PortableModel,
    is_sorted_unique,
    tuple_of,
)
from stylog.domain.diagnostic import Diagnostic, diagnostic_sort_key


class BenchmarkSplitRealization(PortableModel):
    """Sorted artifact ids realized per split part."""

    train: tuple_of(str)
    dev: tuple_of(str)
    test: tuple_of(str)

    @model_validator(mode="after")
    def _sorted(self) -> BenchmarkSplitRealization:
        for part in ("train", "dev", "test"):
            ids = list(getattr(self, part))
            if not is_sorted_unique(ids):
                raise ValueError(f"{part} artifact ids must be sorted by unique id")
        return self


class PairwiseFeatureMetrics(PortableModel):
    """Descriptive distance summary of one feature over labelled pairs.

    Mean/median of a label class are omitted when that class has no valid
    pair; ``roc_auc`` is omitted unless both classes are present.
    """

    feature_id: str
    metric: str
    same_count: int
    different_count: int
    same_mean_distance: PortableFloat | None = None
    different_mean_distance: PortableFloat | None = None
    same_median_distance: PortableFloat | None = None
    different_median_distance: PortableFloat | None = None
    roc_auc: PortableFloat | None = None  # omitted when a class is empty

    @model_validator(mode="after")
    def _check(self) -> PairwiseFeatureMetrics:
        if self.same_count < 0 or self.different_count < 0:
            raise ValueError("pair counts must be >= 0")
        return self


class TransformationFeatureDistance(PortableModel):
    """One per-feature distance between a transformation original and variant."""

    transformation_id: str
    original: str
    variant: str
    feature_id: str
    metric: str
    value: PortableFloat


class RiskEntry(PortableModel):
    """A declared contamination-risk entry echoed from the dataset manifest."""

    key: str
    value: str


class VerificationMetrics(PortableModel):
    """Decision-level benchmark metrics under one explicit verifier model.

    Benchmark-only output: these values never feed back into Comparison or
    domain semantics. ``roc_auc`` covers scored rows only, ``f1``/``c_at_1``/
    ``f_05u`` follow the official PAN abstention rules, ``brier`` covers
    rows with calibrated probabilities.
    """

    verifier_id: HexDigest64
    pair_count: int
    answered_count: int
    abstain_uncertain_count: int
    abstain_insufficient_evidence_count: int
    roc_auc: PortableFloat | None = None  # omitted unless both classes have scores
    f1: PortableFloat | None = None  # omitted when no pair was answered
    c_at_1: PortableFloat
    f_05u: PortableFloat
    brier: PortableFloat | None = None  # omitted without calibrated probabilities

    @model_validator(mode="after")
    def _check(self) -> VerificationMetrics:
        if self.pair_count < 0:
            raise ValueError("pair_count must be >= 0")
        if min(
            self.answered_count,
            self.abstain_uncertain_count,
            self.abstain_insufficient_evidence_count,
        ) < 0:
            raise ValueError("decision counts must be >= 0")
        if (
            self.answered_count
            + self.abstain_uncertain_count
            + self.abstain_insufficient_evidence_count
            != self.pair_count
        ):
            raise ValueError("decision counts must sum to pair_count")
        return self


class BenchmarkResult(PortableModel):
    schema: Literal["stylog.benchmark-result"] = "stylog.benchmark-result"
    schema_version: Literal["0.1.0"] = "0.1.0"
    benchmark_id: str
    task: str
    dataset_manifest_sha256: HexDigest64
    split_config_sha256: HexDigest64 | None = None  # present when the spec has [split]
    split_algorithm_version: str
    splits: BenchmarkSplitRealization | None = None  # present when a split was realized
    pairwise_metrics: tuple_of(PairwiseFeatureMetrics) = ()
    transformation_distances: tuple_of(TransformationFeatureDistance) = ()
    risk_declarations: tuple_of(RiskEntry) = ()
    verification_metrics: VerificationMetrics | None = None  # verification task only
    diagnostics: tuple_of(Diagnostic) = ()

    @model_validator(mode="after")
    def _sorted(self) -> BenchmarkResult:
        pairwise = [(entry.feature_id, entry.metric) for entry in self.pairwise_metrics]
        if not is_sorted_unique(pairwise):
            raise ValueError("pairwise metrics must be sorted by unique (feature_id, metric)")
        distances = [
            (entry.transformation_id, entry.feature_id) for entry in self.transformation_distances
        ]
        if not is_sorted_unique(distances):
            raise ValueError(
                "transformation distances must be sorted by unique (transformation_id, feature_id)"
            )
        risks = [entry.key for entry in self.risk_declarations]
        if not is_sorted_unique(risks):
            raise ValueError("risk declarations must be sorted by unique key")
        if tuple(self.diagnostics) != tuple(sorted(self.diagnostics, key=diagnostic_sort_key)):
            raise ValueError("diagnostics must be in canonical order")
        return self
