"""Operational exception hierarchy (spec section 20.10).

Feature missingness/status is data, not an exception; these exceptions cover
operational failures only. ``exit_code`` maps to the CLI contract (section 19.12).
"""

from __future__ import annotations


class StylogError(Exception):
    """Base class for all Stylog operational errors."""

    exit_code = 1
    diagnostic_code = "INTERNAL"


class ConfigurationError(StylogError):
    exit_code = 2
    diagnostic_code = "CONFIGURATION_ERROR"


class CapabilityUnavailableError(StylogError):
    exit_code = 2
    diagnostic_code = "CAPABILITY_UNAVAILABLE"


class InputError(StylogError):
    exit_code = 3
    diagnostic_code = "INPUT_ERROR"


class DecodeError(InputError):
    diagnostic_code = "INPUT_DECODE_ERROR"


class UnsupportedInputError(InputError):
    diagnostic_code = "INPUT_UNSUPPORTED"


class ResourceLimitError(InputError):
    exit_code = 7
    diagnostic_code = "INPUT_TOO_LARGE"


class PortableArtifactError(StylogError):
    exit_code = 4
    diagnostic_code = "PORTABLE_ARTIFACT_INVALID"


class BaselineError(StylogError):
    exit_code = 4
    diagnostic_code = "BASELINE_INVALID"


class ResourceError(StylogError):
    exit_code = 4
    diagnostic_code = "RESOURCE_MISMATCH"


class ModelIncompatibilityError(StylogError):
    exit_code = 4
    diagnostic_code = "MODEL_INCOMPATIBLE"


class VerifierFitError(StylogError):
    exit_code = 2
    diagnostic_code = "VERIFIER_FIT_FAILED"


class BenchmarkError(StylogError):
    exit_code = 6
    diagnostic_code = "BENCHMARK_INVALID"


class InternalStylogError(StylogError):
    exit_code = 5
    diagnostic_code = "ANALYZER_INTERNAL_ERROR"
