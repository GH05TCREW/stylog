"""Polars/pandas frame adapters (spec 20: runtime conveniences).

Frames keep the same column contract as the Arrow table: ``canonical_json``
and ``scientific_sha256`` are always present, so scientific identity survives
the handoff to external dataframe objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from stylog.capability import require_capability
from stylog.data.arrow import to_arrow

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import polars as pl

    from stylog.domain import PortableModel


def _require_polars() -> Any:
    return require_capability("polars", "data")


def _require_pandas() -> Any:
    return require_capability("pandas", "data")


def to_polars(objects: Sequence[PortableModel]) -> pl.DataFrame:
    """Materialize portable objects as a Polars DataFrame (spec-5.20 columns)."""
    pl = _require_polars()
    frame = pl.from_arrow(to_arrow(objects))
    assert isinstance(frame, pl.DataFrame)
    return frame


def scan_parquet_polars(path: str | Path) -> pl.LazyFrame:
    """Lazy Polars scan over a corpus Parquet file written by ``write_parquet``."""
    pl = _require_polars()
    return pl.scan_parquet(str(Path(path)))


def to_pandas(objects: Sequence[PortableModel]) -> pd.DataFrame:
    """Materialize portable objects as a pandas DataFrame (spec-5.20 columns)."""
    _require_pandas()
    return to_arrow(objects).to_pandas()
