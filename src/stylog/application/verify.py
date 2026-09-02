"""Verify use case: fingerprints or bundles under an explicit VerifierFit model."""

from __future__ import annotations

from stylog.analysis.verify import verify_fingerprints as _verify_fingerprints
from stylog.domain.fingerprint import AnalysisBundle, Fingerprint
from stylog.domain.verification import Verification, VerifierFit
from stylog.exceptions import StylogError


def verify_subjects(
    left: Fingerprint | AnalysisBundle,
    right: Fingerprint | AnalysisBundle,
    model: VerifierFit,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Verification:
    """Verify two subjects under an explicit fitted model (spec 23).

    AnalysisBundles reduce to their primary fingerprints (the bound evidence
    hashes are those of the primaries). Mixed subject kinds are an error,
    mirroring the comparison path.
    """
    if isinstance(left, Fingerprint) and isinstance(right, Fingerprint):
        return _verify_fingerprints(model, left, right, left_ref=left_ref, right_ref=right_ref)
    if isinstance(left, AnalysisBundle) and isinstance(right, AnalysisBundle):
        return _verify_fingerprints(
            model, left.primary, right.primary, left_ref=left_ref, right_ref=right_ref
        )
    raise StylogError(
        "unsupported verification subjects: fingerprints verify with fingerprints, "
        "bundles with bundles"
    )
