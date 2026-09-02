"""Portable diagnostics (spec section 5.6).

Diagnostics carry stable machine-readable codes, never human prose. Empty
optional fields are omitted from portable output (modelled as ``None`` and
excluded at serialization time).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import model_validator

from stylog.domain._base import PortableModel, is_sorted_unique, tuple_of


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


_SEVERITY_ORDER = {
    DiagnosticSeverity.ERROR: 0,
    DiagnosticSeverity.WARNING: 1,
    DiagnosticSeverity.INFO: 2,
}


class DiagnosticContextEntry(PortableModel):
    key: str
    value: str


class Diagnostic(PortableModel):
    code: str
    severity: DiagnosticSeverity
    analyzer_id: str | None = None  # omitted from portable output when absent
    feature_id: str | None = None
    artifact_id: str | None = None
    context: tuple_of(DiagnosticContextEntry) = ()

    @model_validator(mode="after")
    def _context_sorted(self) -> Diagnostic:
        keys = [entry.key for entry in self.context]
        if not is_sorted_unique(keys):
            raise ValueError("diagnostic context entries must be sorted by unique key")
        return self


def make_diagnostic(
    code: str,
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING,
    *,
    analyzer_id: str | None = None,
    feature_id: str | None = None,
    artifact_id: str | None = None,
    context: tuple[tuple[str, str], ...] = (),
) -> Diagnostic:
    entries = tuple(
        DiagnosticContextEntry(key=key, value=value)
        for key, value in sorted(context, key=lambda pair: pair[0])
    )
    kwargs: dict[str, Any] = {"code": code, "severity": severity, "context": entries}
    if analyzer_id is not None:
        kwargs["analyzer_id"] = analyzer_id
    if feature_id is not None:
        kwargs["feature_id"] = feature_id
    if artifact_id is not None:
        kwargs["artifact_id"] = artifact_id
    return Diagnostic(**kwargs)


def diagnostic_sort_key(diagnostic: Diagnostic) -> tuple[Any, ...]:
    """Canonical diagnostic ordering (spec section 14.7)."""
    return (
        _SEVERITY_ORDER[diagnostic.severity],
        diagnostic.code,
        diagnostic.artifact_id or "",
        diagnostic.analyzer_id or "",
        diagnostic.feature_id or "",
        tuple((entry.key, entry.value) for entry in diagnostic.context),
    )


def sort_diagnostics(diagnostics: list[Diagnostic] | tuple[Diagnostic, ...]) -> tuple[Diagnostic, ...]:
    return tuple(sorted(diagnostics, key=diagnostic_sort_key))
