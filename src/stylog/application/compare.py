"""Compare use case: fingerprints, bundles (with embedded sections), aggregates."""

from __future__ import annotations

from stylog.analysis.aggregate import aggregate_fingerprints
from stylog.analysis.compare import compare_aggregates, compare_fingerprints
from stylog.domain.diagnostic import sort_diagnostics
from stylog.domain.evidence import (
    EvidenceAggregate,
    EvidenceMember,
    EvidenceSet,
    LinkageDescriptor,
)
from stylog.domain.fingerprint import AnalysisBundle, Fingerprint
from stylog.domain.interpretation import Comparison, ComparisonFamily
from stylog.exceptions import StylogError


def _embedded_aggregates(bundle: AnalysisBundle, kinds: tuple[str, ...], set_id: str):
    members = []
    fps = []
    for index, item in enumerate(bundle.embedded):
        if item.descriptor.embedded_kind in kinds:
            member_id = f"m{index:06d}"
            members.append(
                EvidenceMember(member_id=member_id, artifact_id=item.descriptor.artifact.artifact_id)
            )
            fps.append(item.fingerprint)
    if not fps:
        return None
    evidence_set = EvidenceSet(
        evidence_set_id=set_id,
        members=tuple(members),
        linkage=LinkageDescriptor(kind="embedded", source="stylog.embedded"),
    )
    return aggregate_fingerprints(evidence_set, fps)


def _prefixed_families(aggregate_comparison: Comparison, prefix: str) -> list[ComparisonFamily]:
    families = []
    for family in aggregate_comparison.families:
        families.append(
            ComparisonFamily(family=f"{prefix}.{family.family}", components=family.components)
        )
    return families


def compare_subjects(
    left: Fingerprint | AnalysisBundle | EvidenceAggregate,
    right: Fingerprint | AnalysisBundle | EvidenceAggregate,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Comparison:
    """One comparison path regardless of input provenance (spec 12, 9.10)."""
    if isinstance(left, Fingerprint) and isinstance(right, Fingerprint):
        return compare_fingerprints(left, right, left_ref, right_ref)
    if isinstance(left, EvidenceAggregate) and isinstance(right, EvidenceAggregate):
        return compare_aggregates(left, right, left_ref, right_ref)
    if isinstance(left, AnalysisBundle) and isinstance(right, AnalysisBundle):
        comparison = compare_fingerprints(left.primary, right.primary, left_ref, right_ref)
        families: list[ComparisonFamily] = list(comparison.families)
        diagnostics = list(comparison.diagnostics)
        for section, kinds in (
            ("embedded.comments", ("comment_block", "inline_comment")),
            ("embedded.docstrings", ("docstring",)),
        ):
            left_agg = _embedded_aggregates(left, kinds, f"{section}.left")
            right_agg = _embedded_aggregates(right, kinds, f"{section}.right")
            if left_agg is None or right_agg is None:
                continue  # section unavailable, never zero-distance
            sub = compare_aggregates(left_agg, right_agg, left_ref, right_ref)
            families.extend(_prefixed_families(sub, section))
            diagnostics.extend(sub.diagnostics)
        return Comparison(
            left_ref=left_ref,
            right_ref=right_ref,
            families=tuple(sorted(families, key=lambda family: family.family)),
            diagnostics=sort_diagnostics(diagnostics),
        )
    raise StylogError(
        "unsupported comparison subjects: fingerprints compare with fingerprints, "
        "aggregates with aggregates, bundles with bundles (spec 12.11)"
    )
