"""Arrow corpus adapter for portable Stylog objects (spec 5.22, section 20).

Every row retains the normative identity columns ``schema``, ``schema_version``,
``scientific_sha256`` and ``canonical_json`` (the exact canonical JCS string, so
the row is lossless). Fingerprint rows additionally carry flattened query
columns; absent optional flattened values are Arrow nulls, never sentinels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stylog.capability import require_capability
from stylog.domain import PORTABLE_MODELS_BY_SCHEMA, Fingerprint, PortableModel
from stylog.exceptions import PortableArtifactError
from stylog.serialization.canonical import canonical_bytes, scientific_sha256
from stylog.serialization.jsonio import model_from_bytes

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa

# Normative spec-5.20 identity columns; never null.
IDENTITY_COLUMNS = ("schema", "schema_version", "scientific_sha256", "canonical_json")

# Optional flattened columns materialized for query ergonomics on fingerprint
# rows; Arrow null everywhere else.
FINGERPRINT_COLUMNS = (
    "artifact_id",
    "kind",
    "language",
    "byte_count",
    "character_count",
    "ok_feature_count",
)

CORPUS_COLUMNS = IDENTITY_COLUMNS + FINGERPRINT_COLUMNS

# Dispatch from the portable ``schema`` literal to the strict model type.
# Arrow tabular support intentionally covers only the 10 non-verifier schemas.
_MODEL_TYPES: dict[str, type[PortableModel]] = {
    name: model
    for name, model in PORTABLE_MODELS_BY_SCHEMA.items()
    if name not in {"stylog.verifier-fit", "stylog.verification"}
}


def _require_pyarrow() -> Any:
    return require_capability("pyarrow", "data")


def _corpus_schema(pa: Any) -> Any:
    return pa.schema(
        [
            pa.field("schema", pa.string(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("scientific_sha256", pa.string(), nullable=False),
            pa.field("canonical_json", pa.string(), nullable=False),
            pa.field("artifact_id", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("language", pa.string()),
            pa.field("byte_count", pa.int64()),
            pa.field("character_count", pa.int64()),
            pa.field("ok_feature_count", pa.int64()),
        ]
    )


def _row_for(obj: PortableModel) -> dict[str, Any]:
    schema = getattr(obj, "schema", None)
    if not isinstance(schema, str) or schema not in _MODEL_TYPES:
        raise PortableArtifactError(
            f"object of type {type(obj).__name__} has no portable schema literal"
        )
    row: dict[str, Any] = dict.fromkeys(CORPUS_COLUMNS)
    row["schema"] = schema
    row["schema_version"] = obj.schema_version  # type: ignore[attr-defined]
    row["scientific_sha256"] = scientific_sha256(obj)
    row["canonical_json"] = canonical_bytes(obj).decode("utf-8")
    if isinstance(obj, Fingerprint):
        artifact = obj.artifact
        row["artifact_id"] = artifact.artifact_id
        row["kind"] = artifact.kind.value
        row["language"] = artifact.language
        row["byte_count"] = artifact.byte_count
        row["character_count"] = artifact.character_count
        row["ok_feature_count"] = sum(1 for obs in obj.features if obs.status == "ok")
    return row


def to_arrow(objects: Sequence[PortableModel]) -> pa.Table:
    """Convert portable objects to a corpus table honoring the spec-5.20 contract."""
    pa = _require_pyarrow()
    columns: dict[str, list[Any]] = {name: [] for name in CORPUS_COLUMNS}
    for obj in objects:
        row = _row_for(obj)
        for name in CORPUS_COLUMNS:
            columns[name].append(row[name])
    return pa.table(columns, schema=_corpus_schema(pa))


def from_arrow(table: pa.Table | pa.RecordBatch) -> list[PortableModel]:
    """Validate each row's canonical payload and rebuild the portable models.

    Raises PortableArtifactError if identity columns are missing/null, the
    schema literal is unknown, the canonical_json is not the exact canonical
    serialization, or scientific_sha256 does not match the canonical bytes.
    """
    names = set(table.schema.names)
    missing = [name for name in IDENTITY_COLUMNS if name not in names]
    if missing:
        raise PortableArtifactError(
            f"corpus table is missing identity columns: {', '.join(missing)}"
        )
    columns = {name: table.column(name).to_pylist() for name in IDENTITY_COLUMNS}
    return _rows_to_models(columns)


def _rows_to_models(columns: dict[str, list[Any]]) -> list[PortableModel]:
    out: list[PortableModel] = []
    rows = zip(
        columns["schema"],
        columns["schema_version"],
        columns["scientific_sha256"],
        columns["canonical_json"],
        strict=True,
    )
    for index, (schema, version, sha256, payload) in enumerate(rows):
        if schema is None or version is None or sha256 is None or payload is None:
            raise PortableArtifactError(f"corpus row {index} has a null identity column")
        model_type = _MODEL_TYPES.get(schema)
        if model_type is None:
            raise PortableArtifactError(f"corpus row {index} has unknown schema: {schema!r}")
        stored = payload.encode("utf-8")
        model = model_from_bytes(stored, model_type)
        if model.schema_version != version:  # type: ignore[attr-defined]
            raise PortableArtifactError(
                f"corpus row {index} schema_version does not match the canonical payload"
            )
        if canonical_bytes(model) != stored:
            raise PortableArtifactError(
                f"corpus row {index} canonical_json is not the canonical serialization"
            )
        if scientific_sha256(model) != sha256:
            raise PortableArtifactError(
                f"corpus row {index} scientific_sha256 does not match the canonical bytes"
            )
        out.append(model)
    return out
