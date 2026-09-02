"""Data capability (``stylog[data]``): Arrow/Parquet/Polars/DuckDB/pandas.

Portable identity is never surrendered: every corpus row keeps ``schema``,
``schema_version``, ``scientific_sha256`` and ``canonical_json`` (spec 5.22).
All heavy dependencies (pyarrow/polars/duckdb/pandas) are imported lazily
inside functions; a missing dependency raises CapabilityUnavailableError.
"""

from stylog.data.arrow import (
    CORPUS_COLUMNS,
    FINGERPRINT_COLUMNS,
    IDENTITY_COLUMNS,
    from_arrow,
    to_arrow,
)
from stylog.data.duckdb import catalog, open_corpus, query_corpus
from stylog.data.frames import scan_parquet_polars, to_pandas, to_polars
from stylog.data.parquet import (
    fingerprint_index,
    read_baseline_values,
    read_parquet_objects,
    scan_parquet,
    write_baseline_values,
    write_parquet,
)

__all__ = [
    "CORPUS_COLUMNS",
    "FINGERPRINT_COLUMNS",
    "IDENTITY_COLUMNS",
    "catalog",
    "fingerprint_index",
    "from_arrow",
    "open_corpus",
    "query_corpus",
    "read_baseline_values",
    "read_parquet_objects",
    "scan_parquet",
    "scan_parquet_polars",
    "to_arrow",
    "to_pandas",
    "to_polars",
    "write_baseline_values",
    "write_parquet",
]
