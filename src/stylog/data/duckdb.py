"""Local-first DuckDB corpus access (spec 20, 22).

In-memory connections only, a ``corpus`` view over local Parquet files, no
extension installation, and no remote sources — URLs are rejected outright.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stylog.capability import require_capability
from stylog.exceptions import InputError, UnsupportedInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import duckdb

_URL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


def _require_duckdb() -> Any:
    return require_capability("duckdb", "data")


def _local_files(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    files: list[Path] = []
    for entry in paths:
        text = os.fspath(entry)
        if _URL_PATTERN.match(text):
            raise UnsupportedInputError(f"remote corpus sources are not supported: {text}")
        path = Path(text)
        if not path.is_file():
            raise InputError(f"parquet file not found: {path}")
        files.append(path)
    if not files:
        raise InputError("at least one parquet file is required")
    return files


def _source_literal(path: Path) -> str:
    # Forward slashes keep the SQL literal free of backslash ambiguity on
    # Windows; single quotes are escaped per SQL rules.
    return "'" + path.as_posix().replace("'", "''") + "'"


def open_corpus(paths: str | Path | Sequence[str | Path]) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with a ``corpus`` view over local Parquet."""
    duckdb = _require_duckdb()
    files = _local_files(paths)
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions=false")
    sources = ", ".join(_source_literal(path) for path in files)
    connection.execute(f"CREATE VIEW corpus AS SELECT * FROM read_parquet([{sources}])")
    return connection


def query_corpus(
    connection_or_paths: duckdb.DuckDBPyConnection | str | Path | Sequence[str | Path],
    sql: str,
) -> duckdb.DuckDBPyRelation:
    """Run SQL against a corpus connection, or open one over the given files.

    When paths are passed, the returned relation keeps its connection alive;
    materialize results (e.g. ``fetchall``) before discarding it.
    """
    duckdb = _require_duckdb()
    if isinstance(connection_or_paths, duckdb.DuckDBPyConnection):
        connection = connection_or_paths
    else:
        connection = open_corpus(connection_or_paths)
    return connection.sql(sql)


def catalog(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Lightweight per-file catalog: row counts and distinct schema versions."""
    duckdb = _require_duckdb()
    files = _local_files(paths)
    connection = duckdb.connect(":memory:")
    connection.execute("SET autoinstall_known_extensions=false")
    try:
        entries: list[dict[str, Any]] = []
        for path in files:
            source = _source_literal(path)
            rows = connection.execute(
                f"SELECT count(*) FROM read_parquet({source})"
            ).fetchone()[0]
            versions = [
                record[0]
                for record in connection.execute(
                    "SELECT DISTINCT schema_version FROM "
                    f"read_parquet({source}) ORDER BY 1"
                ).fetchall()
            ]
            entries.append({"path": str(path), "rows": rows, "schema_versions": versions})
        return entries
    finally:
        connection.close()
