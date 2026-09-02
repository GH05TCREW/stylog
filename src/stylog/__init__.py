"""Stylog: local-first scientific stylometry for text and source code.

Base imports stay light: no spaCy/scikit-learn/Arrow/Polars/DuckDB/pandas and
no tree-sitter grammars are loaded by importing this package.
"""

from __future__ import annotations

from typing import Any

from stylog.domain import *
from stylog.domain import __all__ as _DOMAIN_ALL
from stylog.exceptions import (
    BaselineError,
    BenchmarkError,
    CapabilityUnavailableError,
    ConfigurationError,
    DecodeError,
    InputError,
    InternalStylogError,
    ModelIncompatibilityError,
    PortableArtifactError,
    ResourceError,
    ResourceLimitError,
    StylogError,
    UnsupportedInputError,
    VerifierFitError,
)

__version__ = "0.1.0"

_API_NAMES = {
    "fingerprint_file",
    "fingerprint_text",
    "fingerprint_bytes",
    "analyze_file",
    "analyze_text",
    "compare_files",
    "compare_fingerprints",
    "profile_fingerprint",
    "build_baseline",
    "fit_representation",
    "transform_representation",
    "verify_fingerprints",
    "verify_files",
    "fit_verifier",
    "load_verifier",
}

_EXCEPTION_NAMES = {
    "BaselineError",
    "BenchmarkError",
    "CapabilityUnavailableError",
    "ConfigurationError",
    "DecodeError",
    "InputError",
    "InternalStylogError",
    "ModelIncompatibilityError",
    "PortableArtifactError",
    "ResourceError",
    "ResourceLimitError",
    "StylogError",
    "UnsupportedInputError",
    "VerifierFitError",
}

__all__ = tuple(sorted({*_DOMAIN_ALL, *_API_NAMES, *_EXCEPTION_NAMES, "__version__"}))


def __getattr__(name: str) -> Any:
    if name in _API_NAMES:
        from stylog import api

        return getattr(api, name)
    raise AttributeError(f"module 'stylog' has no attribute {name!r}")
