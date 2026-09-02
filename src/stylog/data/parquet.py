"""Parquet persistence for Stylog corpora and baseline values (spec 5.22, 13.13).

Writes are atomic (temp file + os.replace) and refuse to overwrite without
``force``. Incremental reuse (spec 18.17) is keyed by Stylog scientific
identity via ``fingerprint_index``, never by file mtime.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stylog.capability import require_capability
from stylog.data.arrow import from_arrow, to_arrow
from stylog.domain import ContentIdentitySha256, Fingerprint, PortableModel
from stylog.exceptions import PortableArtifactError
from stylog.serialization.canonical import scientific_sha256, sha256_hex
from stylog.serialization.jsonio import atomic_temp_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from stylog.config import StylogConfig
    from stylog.domain import BaselineFeature

DEFAULT_ROW_GROUP_SIZE = 65_536
DEFAULT_COMPRESSION = "zstd"


def _require_pyarrow_parquet() -> tuple[Any, Any]:
    pyarrow = require_capability("pyarrow", "data")
    parquet = require_capability("pyarrow.parquet", "data")
    return pyarrow, parquet


def _parquet_settings(config: StylogConfig | None) -> tuple[int, str]:
    if config is not None and config.data is not None:
        return config.data.row_group_size, config.data.parquet_compression
    return DEFAULT_ROW_GROUP_SIZE, DEFAULT_COMPRESSION


def _normalize_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        return [Path(paths)]
    return [Path(entry) for entry in paths]


def _write_table_atomic(
    pq: Any,
    table: Any,
    path: str | Path,
    *,
    row_group_size: int,
    compression: str,
    force: bool,
) -> Path:
    target = Path(path)
    if target.exists() and not force:
        raise PortableArtifactError(f"output exists (use force to overwrite): {target}")
    with atomic_temp_path(target, suffix=".parquet") as (fd, temp_name):
        os.close(fd)  # pyarrow owns writing the temp file by path
        with pq.ParquetWriter(temp_name, table.schema, compression=compression) as writer:
            writer.write_table(table, row_group_size=row_group_size)
    return target


def write_parquet(
    objects: Sequence[PortableModel],
    path: str | Path,
    *,
    config: StylogConfig | None = None,
    force: bool = False,
) -> None:
    """Write portable objects as a spec-5.20 corpus Parquet file, atomically."""
    _, pq = _require_pyarrow_parquet()
    row_group_size, compression = _parquet_settings(config)
    _write_table_atomic(
        pq,
        to_arrow(objects),
        path,
        row_group_size=row_group_size,
        compression=compression,
        force=force,
    )


def read_parquet_objects(path: str | Path) -> list[PortableModel]:
    """Read a corpus Parquet file, validating every row like ``from_arrow``."""
    _, pq = _require_pyarrow_parquet()
    return from_arrow(pq.read_table(Path(path)))


def scan_parquet(path: str | Path) -> Iterator[PortableModel]:
    """Stream a corpus Parquet file record-batch by record-batch, validating rows."""
    _, pq = _require_pyarrow_parquet()
    parquet_file = pq.ParquetFile(Path(path))
    for batch in parquet_file.iter_batches():
        yield from from_arrow(batch)


def write_baseline_values(
    feature: BaselineFeature,
    path: str | Path,
    *,
    config: StylogConfig | None = None,
    force: bool = False,
) -> str:
    """Write one baseline feature's ascending values as a Parquet resource.

    The file carries a single non-null float64 column ``value`` (spec 13.13
    storage artifact). Returns the SHA-256 of the written file so callers can
    record the resource hash in the canonical baseline manifest.
    """
    pa, pq = _require_pyarrow_parquet()
    row_group_size, compression = _parquet_settings(config)
    schema = pa.schema([pa.field("value", pa.float64(), nullable=False)])
    table = pa.table(
        {"value": pa.array([float(value) for value in feature.values], type=pa.float64())},
        schema=schema,
    )
    target = _write_table_atomic(
        pq, table, path, row_group_size=row_group_size, compression=compression, force=force
    )
    return sha256_hex(target.read_bytes())


def read_baseline_values(path: str | Path) -> list[float]:
    """Read ascending baseline values from a spec-13.13 Parquet value resource."""
    pa, pq = _require_pyarrow_parquet()
    table = pq.read_table(Path(path))
    if table.column_names != ["value"]:
        raise PortableArtifactError(
            "baseline value resource must contain exactly one column named 'value'"
        )
    column = table.column("value")
    if not pa.types.is_float64(column.type):
        raise PortableArtifactError("baseline value column must be float64")
    values = column.to_pylist()
    if any(value is None for value in values):
        raise PortableArtifactError("baseline value column must not contain null")
    return [float(value) for value in values]


def fingerprint_index(paths: str | Path | Sequence[str | Path]) -> dict[str, str]:
    """Map artifact content SHA-256 -> scientific_sha256 over persisted fingerprints.

    Enables incremental corpus reuse (spec 18.17) keyed by Stylog scientific
    identity, never by file mtime. Fingerprints with suppressed content
    identity cannot be indexed and are skipped; on duplicate content hashes the
    later file wins.
    """
    index: dict[str, str] = {}
    for path in _normalize_paths(paths):
        for model in scan_parquet(path):
            if not isinstance(model, Fingerprint):
                continue
            identity = model.artifact.content_identity
            if isinstance(identity, ContentIdentitySha256):
                index[identity.sha256] = scientific_sha256(model)
    return index
