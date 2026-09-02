"""Local store-root resolution: env override, then platformdirs default."""

from __future__ import annotations

import os
from pathlib import Path


def store_root(env_var: str, default: Path) -> Path:
    """Return the ``env_var`` override path when set, else ``default``."""
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    return default
