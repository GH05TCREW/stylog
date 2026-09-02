"""Optional-extra capability import guard: one typed error contract."""

from __future__ import annotations

import importlib
from typing import Any

from stylog.exceptions import CapabilityUnavailableError


def require_capability(module: str, extra: str) -> Any:
    """Import ``module``; absence raises the typed capability error for ``extra``."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise CapabilityUnavailableError(
            f"the '{extra}' capability requires: pip install stylog[{extra}]"
        ) from exc
