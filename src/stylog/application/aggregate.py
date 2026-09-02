"""EvidenceSet aggregation use case."""

from __future__ import annotations

from collections.abc import Sequence

from stylog.analysis.aggregate import aggregate_fingerprints
from stylog.domain.evidence import EvidenceAggregate, EvidenceSet
from stylog.domain.fingerprint import Fingerprint


def aggregate_evidence(
    evidence_set: EvidenceSet, fingerprints: Sequence[Fingerprint]
) -> EvidenceAggregate:
    """Aggregate compatible observations across an EvidenceSet (spec 11)."""
    return aggregate_fingerprints(evidence_set, fingerprints)
