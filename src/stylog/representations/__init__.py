"""Stylog representations — the 'ml' capability (scikit-learn vectorizers).

Lazy exports only: importing this package never imports sklearn, numpy, or
scipy. The heavy optional stack is imported inside functions (spec 4.14).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY = {
    "BACKEND_ID": "stylog.representations.spec",
    "CLI_TOKENS": "stylog.representations.spec",
    "PREPROCESSING_VERSION": "stylog.representations.spec",
    "REPRESENTATION_IDS": "stylog.representations.spec",
    "REPRESENTATION_KINDS": "stylog.representations.spec",
    "SCIENTIFIC_COMPATIBILITY_ID": "stylog.representations.spec",
    "SEMANTIC_VERSION": "stylog.representations.spec",
    "SPECS": "stylog.representations.spec",
    "STATE_RESOURCE_PREFIX": "stylog.representations.spec",
    "RepresentationSpec": "stylog.representations.spec",
    "representation_spec": "stylog.representations.spec",
    "fit_representation": "stylog.representations.fit",
    "fit_representation_cli": "stylog.representations.fit",
    "transform_many": "stylog.representations.fit",
    "transform_representation": "stylog.representations.fit",
    "transform_representation_cli": "stylog.representations.fit",
}

__all__ = [
    "BACKEND_ID",
    "CLI_TOKENS",
    "PREPROCESSING_VERSION",
    "REPRESENTATION_IDS",
    "REPRESENTATION_KINDS",
    "SCIENTIFIC_COMPATIBILITY_ID",
    "SEMANTIC_VERSION",
    "SPECS",
    "STATE_RESOURCE_PREFIX",
    "RepresentationSpec",
    "fit_representation",
    "fit_representation_cli",
    "representation_spec",
    "transform_many",
    "transform_representation",
    "transform_representation_cli",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module 'stylog.representations' has no attribute {name!r}")
    return getattr(import_module(module_name), name)
