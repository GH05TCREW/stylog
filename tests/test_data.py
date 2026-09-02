"""Tests for the stylog[data] capability (spec 5.22, 13.13, 18.17, 20, 22)."""

from __future__ import annotations

import hashlib
import json

import pytest

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")
duckdb = pytest.importorskip("duckdb")
pd = pytest.importorskip("pandas")

from stylog.config import DEFAULT_CONFIG, DataConfig, StylogConfig
from stylog.data import (
    catalog,
    fingerprint_index,
    from_arrow,
    open_corpus,
    query_corpus,
    read_baseline_values,
    read_parquet_objects,
    scan_parquet,
    scan_parquet_polars,
    to_arrow,
    to_pandas,
    to_polars,
    write_baseline_values,
    write_parquet,
)
from stylog.data.arrow import CORPUS_COLUMNS, IDENTITY_COLUMNS
from stylog.domain.baseline import BaselineFeature
from stylog.domain.diagnostic import Diagnostic, DiagnosticSeverity
from stylog.exceptions import PortableArtifactError, UnsupportedInputError
from stylog.serialization.canonical import canonical_bytes, scientific_sha256

TEXTS = [
    ("The quick brown fox jumps over the lazy dog. It was a bright cold day.", "en"),
    ("She sells seashells by the seashore, and the shells she sells are seashells.", "en"),
    ("It was the best of times, it was the worst of times, it was the age of wisdom.", "en"),
    ("Call me Ishmael. Some years ago, having little money in my purse, I went to sea.", "en"),
    ("x_y 1,000.50 dogs rock", "und"),
]


@pytest.fixture(scope="module")
def fingerprints():
    from stylog import fingerprint_text

    return [
        fingerprint_text(text, language=language, config=DEFAULT_CONFIG)
        for text, language in TEXTS
    ]


def test_to_arrow_columns(fingerprints):
    table = to_arrow(fingerprints)
    assert table.num_rows == len(fingerprints)
    for name in IDENTITY_COLUMNS:
        assert name in table.column_names
        assert table.column(name).null_count == 0
    for name in CORPUS_COLUMNS:
        assert name in table.column_names
    row = {name: table.column(name)[0].as_py() for name in table.column_names}
    first = fingerprints[0]
    assert row["schema"] == "stylog.fingerprint"
    assert row["schema_version"] == "0.1.0"
    assert row["scientific_sha256"] == scientific_sha256(first)
    assert row["artifact_id"] == first.artifact.artifact_id
    assert row["kind"] == "text"
    assert row["language"] == "en"
    assert row["byte_count"] == first.artifact.byte_count
    assert row["character_count"] == first.artifact.character_count
    expected_ok = sum(1 for obs in first.features if obs.status == "ok")
    assert row["ok_feature_count"] == expected_ok


def test_from_arrow_roundtrip_byte_identical(fingerprints):
    table = to_arrow(fingerprints)
    models = from_arrow(table)
    assert models == fingerprints
    stored = table.column("canonical_json").to_pylist()
    for model, payload in zip(models, stored, strict=True):
        assert canonical_bytes(model) == payload.encode("utf-8")


def test_from_arrow_non_fingerprint_rows_have_null_flattened(fingerprints):
    # Non-fingerprint portable objects keep identity columns; flattened columns
    # are Arrow null (never empty-string/-1 sentinels).
    from stylog.application.compare import compare_subjects

    model = compare_subjects(
        fingerprints[0], fingerprints[1], left_ref="left", right_ref="right"
    )
    table = to_arrow([model])
    assert table.column("schema")[0].as_py() == "stylog.comparison"
    assert table.column("artifact_id").null_count == 1
    assert from_arrow(table) == [model]


def test_from_arrow_rejects_unsupported_object():
    diagnostic = Diagnostic(severity=DiagnosticSeverity.ERROR, code="X")
    with pytest.raises(PortableArtifactError):
        to_arrow([diagnostic])


def test_from_arrow_rejects_missing_identity_column(fingerprints):
    table = to_arrow(fingerprints).drop_columns("canonical_json")
    with pytest.raises(PortableArtifactError):
        from_arrow(table)


def test_from_arrow_rejects_tampered_canonical_json(fingerprints):
    table = to_arrow(fingerprints)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    payload = json.loads(columns["canonical_json"][0])
    payload["artifact"]["byte_count"] += 1
    columns["canonical_json"][0] = json.dumps(payload)
    tampered = pa.table(columns, schema=table.schema)
    with pytest.raises(PortableArtifactError):
        from_arrow(tampered)


def test_from_arrow_rejects_tampered_sha256(fingerprints):
    table = to_arrow(fingerprints)
    columns = {name: table.column(name).to_pylist() for name in table.column_names}
    columns["scientific_sha256"][0] = "0" * 64
    tampered = pa.table(columns, schema=table.schema)
    with pytest.raises(PortableArtifactError):
        from_arrow(tampered)


def test_parquet_roundtrip(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    assert read_parquet_objects(path) == fingerprints


def test_parquet_refuses_overwrite_without_force(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    with pytest.raises(PortableArtifactError):
        write_parquet(fingerprints, path)
    write_parquet(fingerprints[:2], path, force=True)
    assert read_parquet_objects(path) == fingerprints[:2]


def test_scan_parquet_streams_row_groups(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    config = StylogConfig(data=DataConfig(row_group_size=2))
    write_parquet(fingerprints, path, config=config)
    import pyarrow.parquet as pq

    assert pq.ParquetFile(path).metadata.num_row_groups == 3
    streamed = list(scan_parquet(path))
    assert streamed == fingerprints
    assert len(streamed) == len(fingerprints)


def test_to_polars_and_scan(fingerprints, tmp_path):
    frame = to_polars(fingerprints)
    assert isinstance(frame, pl.DataFrame)
    assert frame.height == len(fingerprints)
    for name in IDENTITY_COLUMNS:
        assert name in frame.columns
    assert frame["scientific_sha256"][0] == scientific_sha256(fingerprints[0])
    assert frame["language"].to_list() == [language for _, language in TEXTS]

    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    lazy = scan_parquet_polars(path)
    assert isinstance(lazy, pl.LazyFrame)
    collected = lazy.collect()
    assert collected.height == len(fingerprints)
    assert collected["canonical_json"][0] == canonical_bytes(fingerprints[0]).decode("utf-8")


def test_to_pandas(fingerprints):
    frame = to_pandas(fingerprints)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == len(fingerprints)
    for name in IDENTITY_COLUMNS:
        assert name in frame.columns
    assert frame.loc[0, "scientific_sha256"] == scientific_sha256(fingerprints[0])


def test_duckdb_open_and_query(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    connection = open_corpus(path)
    try:
        total = query_corpus(connection, "SELECT count(*) FROM corpus").fetchone()[0]
        assert total == len(fingerprints)
        english = query_corpus(
            connection, "SELECT count(*) FROM corpus WHERE language = 'en'"
        ).fetchone()[0]
        assert english == sum(1 for _, language in TEXTS if language == "en")
        rows = query_corpus(
            connection,
            "SELECT scientific_sha256 FROM corpus ORDER BY byte_count DESC LIMIT 1",
        ).fetchall()
        biggest = max(fingerprints, key=lambda fp: fp.artifact.byte_count)
        assert rows[0][0] == scientific_sha256(biggest)
    finally:
        connection.close()


def test_query_corpus_accepts_paths(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    relation = query_corpus([path], "SELECT count(*) FROM corpus")
    assert relation.fetchone()[0] == len(fingerprints)


def test_open_corpus_rejects_urls(tmp_path):
    with pytest.raises(UnsupportedInputError):
        open_corpus("https://example.com/corpus.parquet")


def test_catalog(fingerprints, tmp_path):
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_parquet(fingerprints[:3], first)
    write_parquet(fingerprints[3:], second)
    entries = catalog([first, second])
    assert entries == [
        {"path": str(first), "rows": 3, "schema_versions": ["0.1.0"]},
        {"path": str(second), "rows": 2, "schema_versions": ["0.1.0"]},
    ]


def test_baseline_values_roundtrip_with_hash(tmp_path):
    values = tuple(sorted(0.1 * index + 0.05 for index in range(25)))
    feature = BaselineFeature(
        feature_id="text.lexical.ttr_casefold",
        semantic_version="1.0.0",
        compatibility_sha256="ab" * 32,
        total_units=25,
        values=values,
    )
    path = tmp_path / "values.parquet"
    digest = write_baseline_values(feature, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert read_baseline_values(path) == list(values)


def test_fingerprint_index_maps_content_to_scientific_identity(fingerprints, tmp_path):
    path = tmp_path / "corpus.parquet"
    write_parquet(fingerprints, path)
    index = fingerprint_index(path)
    assert len(index) == len(fingerprints)
    for (text, _), fingerprint in zip(TEXTS, fingerprints, strict=True):
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert index[content_sha256] == scientific_sha256(fingerprint)
