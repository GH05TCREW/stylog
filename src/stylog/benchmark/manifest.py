"""TOML dataset-manifest and benchmark-spec parsing (spec section 21).

Parsing is strict: unknown keys, wrong types, missing required fields, bad
checksums, and inconsistent split quotas all raise ``BenchmarkError``. The
parsed structures are plain frozen dataclasses -- runtime-only, never part of
portable serialization trees.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, NoReturn

from stylog.exceptions import BenchmarkError
from stylog.serialization.canonical import sha256_hex

DATASET_SCHEMA = "stylog.dataset"
BENCHMARK_SCHEMA = "stylog.benchmark"
VERIFIER_TRAINING_SCHEMA = "stylog.verifier-training"
SCHEMA_VERSION = "0.1.0"

TASKS = ("split_audit", "pairwise_comparison", "transformation_stability", "verification")
REDISTRIBUTION_VALUES = ("allowed", "restricted", "unknown")
PAIR_LABELS = ("same", "different")
TRAINING_POPULATIONS = ("train", "tuning", "calibration")

# Optional per-artifact context fields (spec section 21).
CONTEXT_FIELDS = (
    "author_id",
    "domain",
    "genre",
    "platform",
    "repository_id",
    "file_id",
    "problem_id",
    "framework_id",
    "commit_time",
    "formatter",
    "transformation_id",
    "label_source",
    "label_reliability",
)

_DATASET_KEYS = {
    "schema",
    "schema_version",
    "id",
    "version",
    "license",
    "redistribution",
    "source",
    "artifact",
    "risks",
    "transformation",
}
_ARTIFACT_KEYS = {"id", "path", "sha256", "kind", "language", *CONTEXT_FIELDS}
_TRANSFORMATION_KEYS = {"original", "variant", "transformation_id"}
_SPEC_KEYS = {
    "schema",
    "schema_version",
    "id",
    "task",
    "dataset",
    "split",
    "pair",
    "checksums",
    "verifier",
}
_SPLIT_KEYS = {
    "seed",
    "train_ppm",
    "dev_ppm",
    "test_ppm",
    "disjoint_by",
    "require_nonempty",
    "disjoint_content",
}
_PAIR_KEYS = {"left", "right", "label"}
_TRAINING_KEYS = {"schema", "schema_version", "id", "dataset", "verifier", "pair"}
_TRAINING_PAIR_KEYS = {"left", "right", "label", "population"}
_VERIFIER_KEYS = {
    "kind",
    "l2_lambda",
    "max_iterations",
    "tolerance",
    "min_support_fraction",
    "min_class_support_fraction",
    "min_pairs",
    "threshold_rule",
    "threshold_alpha",
    "threshold_fixed",
    "calibration_method",
    "include_linguistic",
    "allow_unconstrained_language",
    "languages",
    "feature_ids",
    "pair_policy",
}
_PAIR_POLICY_KEYS = {
    "max_pairs_per_author",
    "max_pairs_per_problem",
    "negative_positive_ratio",
    "selection_version",
}

_HEX64 = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class DatasetArtifact:
    id: str
    path: str  # relative to the dataset root
    sha256: str
    kind: str
    language: str
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetTransformation:
    original: str
    variant: str
    transformation_id: str


@dataclass(frozen=True)
class DatasetManifest:
    id: str
    version: str
    license: str
    redistribution: str
    source: str
    artifacts: tuple[DatasetArtifact, ...]
    risks: dict[str, str]
    transformations: tuple[DatasetTransformation, ...]

    def artifact_by_id(self, artifact_id: str) -> DatasetArtifact | None:
        for artifact in self.artifacts:
            if artifact.id == artifact_id:
                return artifact
        return None


@dataclass(frozen=True)
class SplitSpec:
    seed: str
    train_ppm: int
    dev_ppm: int
    test_ppm: int
    disjoint_by: tuple[str, ...]
    require_nonempty: bool
    disjoint_content: bool = False  # union identical content hashes (default off)

    def as_tree(self) -> dict[str, Any]:
        """Plain dict used for the split-config scientific hash."""
        tree: dict[str, Any] = {
            "seed": self.seed,
            "train_ppm": self.train_ppm,
            "dev_ppm": self.dev_ppm,
            "test_ppm": self.test_ppm,
            "disjoint_by": list(self.disjoint_by),
            "require_nonempty": self.require_nonempty,
        }
        # default-off flag: only materialized when set so existing split-config
        # hashes (and their goldens) stay byte-identical
        if self.disjoint_content:
            tree["disjoint_content"] = True
        return tree


@dataclass(frozen=True)
class PairSpec:
    left: str
    right: str
    label: str  # "same" | "different"


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    task: str
    dataset: str  # dataset manifest path, relative to the spec file
    split: SplitSpec | None
    pairs: tuple[PairSpec, ...]
    checksums: bool
    verifier_model: str | None = None  # [verifier] model path (verification task)


@dataclass(frozen=True)
class TrainingPairSpec:
    left: str
    right: str
    label: str  # "same" | "different"
    population: str  # train | tuning | calibration


@dataclass(frozen=True)
class VerifierBlock:
    """The raw [verifier] block of a training manifest (pre-VerifierSpec)."""

    kind: str
    l2_lambda: float
    min_support_fraction: float
    min_class_support_fraction: float
    min_pairs: int
    threshold_rule: str
    threshold_alpha: float | None
    threshold_fixed: float | None
    calibration_method: str | None
    max_iterations: int
    tolerance: float
    include_linguistic: bool
    allow_unconstrained_language: bool
    languages: tuple[str, ...] | None
    feature_ids: tuple[str, ...] | None
    max_pairs_per_author: int | None
    max_pairs_per_problem: int | None
    negative_positive_ratio: float | None
    selection_version: str


@dataclass(frozen=True)
class TrainingManifest:
    id: str
    dataset: str  # dataset manifest path, relative to the training file
    verifier: VerifierBlock
    pairs: tuple[TrainingPairSpec, ...]


def _fail(code: str, message: str) -> NoReturn:
    raise BenchmarkError(f"{code}: {message}")


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail("BENCHMARK_INVALID", f"file not found: {path}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        _fail("BENCHMARK_INVALID", f"invalid TOML in {path}: {exc}")
    if not isinstance(data, dict):
        _fail("BENCHMARK_INVALID", f"{path}: top-level table expected")
    return data


def _check_keys(table: dict[str, Any], allowed: set[str], where: str) -> None:
    for key in table:
        if key not in allowed:
            _fail("BENCHMARK_INVALID", f"{where}: unknown key {key!r}")


def _require_str(table: dict[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        _fail("BENCHMARK_INVALID", f"{where}: {key!r} must be a nonempty string")
    return value


def _require_int(table: dict[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("BENCHMARK_INVALID", f"{where}: {key!r} must be an integer")
    return value


def _check_header(data: dict[str, Any], schema: str, where: str) -> None:
    if data.get("schema") != schema:
        _fail("BENCHMARK_INVALID", f"{where}: schema must be {schema!r}")
    if data.get("schema_version") != SCHEMA_VERSION:
        _fail("BENCHMARK_INVALID", f"{where}: schema_version must be {SCHEMA_VERSION!r}")


def _parse_artifact(entry: Any, index: int) -> DatasetArtifact:
    where = f"[[artifact]] #{index}"
    if not isinstance(entry, dict):
        _fail("BENCHMARK_INVALID", f"{where}: table expected")
    _check_keys(entry, _ARTIFACT_KEYS, where)
    artifact_id = _require_str(entry, "id", where)
    path = _require_str(entry, "path", where)
    sha256 = _require_str(entry, "sha256", where)
    if len(sha256) != 64 or any(ch not in _HEX64 for ch in sha256):
        _fail("BENCHMARK_INVALID", f"{where}: 'sha256' must be 64 lowercase hex characters")
    kind = _require_str(entry, "kind", where)
    if kind not in ("text", "code"):
        _fail("BENCHMARK_INVALID", f"{where}: 'kind' must be 'text' or 'code'")
    language = _require_str(entry, "language", where)
    context: dict[str, str] = {}
    for field_name in CONTEXT_FIELDS:
        value = entry.get(field_name)
        if value is None:
            continue
        if not isinstance(value, str):
            _fail("BENCHMARK_INVALID", f"{where}: {field_name!r} must be a string")
        context[field_name] = value
    return DatasetArtifact(
        id=artifact_id, path=path, sha256=sha256, kind=kind, language=language, context=context
    )


def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Parse and strictly validate a ``stylog.dataset`` TOML manifest."""
    data = _load_toml(path)
    _check_keys(data, _DATASET_KEYS, str(path))
    _check_header(data, DATASET_SCHEMA, str(path))
    redistribution = _require_str(data, "redistribution", str(path))
    if redistribution not in REDISTRIBUTION_VALUES:
        _fail(
            "BENCHMARK_INVALID",
            f"{path}: 'redistribution' must be one of {REDISTRIBUTION_VALUES}",
        )
    raw_artifacts = data.get("artifact", [])
    if not isinstance(raw_artifacts, list):
        _fail("BENCHMARK_INVALID", f"{path}: 'artifact' must be an array of tables")
    artifacts = tuple(
        _parse_artifact(entry, index) for index, entry in enumerate(raw_artifacts)
    )
    ids = [artifact.id for artifact in artifacts]
    if len(set(ids)) != len(ids):
        _fail("BENCHMARK_INVALID", f"{path}: duplicate artifact ids")
    risks_raw = data.get("risks", {})
    if not isinstance(risks_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in risks_raw.items()
    ):
        _fail("BENCHMARK_INVALID", f"{path}: [risks] must be a string-to-string map")
    risks = dict(risks_raw)
    transformations_raw = data.get("transformation", [])
    if not isinstance(transformations_raw, list):
        _fail("BENCHMARK_INVALID", f"{path}: 'transformation' must be an array of tables")
    transformations: list[DatasetTransformation] = []
    for index, entry in enumerate(transformations_raw):
        where = f"[[transformation]] #{index}"
        if not isinstance(entry, dict):
            _fail("BENCHMARK_INVALID", f"{where}: table expected")
        _check_keys(entry, _TRANSFORMATION_KEYS, where)
        original = _require_str(entry, "original", where)
        variant = _require_str(entry, "variant", where)
        if original == variant:
            _fail("BENCHMARK_INVALID", f"{where}: original and variant must differ")
        transformations.append(
            DatasetTransformation(
                original=original,
                variant=variant,
                transformation_id=_require_str(entry, "transformation_id", where),
            )
        )
    return DatasetManifest(
        id=_require_str(data, "id", str(path)),
        version=_require_str(data, "version", str(path)),
        license=_require_str(data, "license", str(path)),
        redistribution=redistribution,
        source=_require_str(data, "source", str(path)),
        artifacts=artifacts,
        risks=risks,
        transformations=tuple(transformations),
    )


def _parse_split(table: Any) -> SplitSpec:
    where = "[split]"
    if not isinstance(table, dict):
        _fail("BENCHMARK_INVALID", f"{where}: table expected")
    _check_keys(table, _SPLIT_KEYS, where)
    train_ppm = _require_int(table, "train_ppm", where)
    dev_ppm = _require_int(table, "dev_ppm", where)
    test_ppm = _require_int(table, "test_ppm", where)
    for name, ppm in (("train_ppm", train_ppm), ("dev_ppm", dev_ppm), ("test_ppm", test_ppm)):
        if not 0 <= ppm <= 1_000_000:
            _fail("BENCHMARK_INVALID", f"{where}: {name!r} must be in [0, 1000000]")
    if train_ppm + dev_ppm + test_ppm != 1_000_000:
        _fail(
            "BENCHMARK_INVALID",
            f"{where}: train_ppm + dev_ppm + test_ppm must equal 1000000 exactly",
        )
    disjoint_raw = table.get("disjoint_by", [])
    if not isinstance(disjoint_raw, list) or not all(
        isinstance(item, str) and item for item in disjoint_raw
    ):
        _fail("BENCHMARK_INVALID", f"{where}: 'disjoint_by' must be a list of nonempty strings")
    disjoint_by = tuple(disjoint_raw)
    if len(set(disjoint_by)) != len(disjoint_by):
        _fail("BENCHMARK_INVALID", f"{where}: 'disjoint_by' contains duplicates")
    for field_name in disjoint_by:
        if field_name not in CONTEXT_FIELDS:
            _fail("BENCHMARK_INVALID", f"{where}: unknown context field {field_name!r}")
    require_nonempty = table.get("require_nonempty", True)
    if not isinstance(require_nonempty, bool):
        _fail("BENCHMARK_INVALID", f"{where}: 'require_nonempty' must be a boolean")
    disjoint_content = table.get("disjoint_content", False)
    if not isinstance(disjoint_content, bool):
        _fail("BENCHMARK_INVALID", f"{where}: 'disjoint_content' must be a boolean")
    return SplitSpec(
        seed=_require_str(table, "seed", where),
        train_ppm=train_ppm,
        dev_ppm=dev_ppm,
        test_ppm=test_ppm,
        disjoint_by=disjoint_by,
        require_nonempty=require_nonempty,
        disjoint_content=disjoint_content,
    )


def load_benchmark_spec(path: Path) -> BenchmarkSpec:
    """Parse and strictly validate a ``stylog.benchmark`` TOML spec."""
    data = _load_toml(path)
    _check_keys(data, _SPEC_KEYS, str(path))
    _check_header(data, BENCHMARK_SCHEMA, str(path))
    task = _require_str(data, "task", str(path))
    if task not in TASKS:
        _fail("BENCHMARK_INVALID", f"{path}: 'task' must be one of {TASKS}")
    split = _parse_split(data["split"]) if "split" in data else None
    pairs_raw = data.get("pair", [])
    if not isinstance(pairs_raw, list):
        _fail("BENCHMARK_INVALID", f"{path}: 'pair' must be an array of tables")
    pairs: list[PairSpec] = []
    for index, entry in enumerate(pairs_raw):
        where = f"[[pair]] #{index}"
        if not isinstance(entry, dict):
            _fail("BENCHMARK_INVALID", f"{where}: table expected")
        _check_keys(entry, _PAIR_KEYS, where)
        left = _require_str(entry, "left", where)
        right = _require_str(entry, "right", where)
        label = _require_str(entry, "label", where)
        if label not in PAIR_LABELS:
            _fail("PAIR_INVALID", f"{where}: 'label' must be one of {PAIR_LABELS}")
        pairs.append(PairSpec(left=left, right=right, label=label))
    checksums = data.get("checksums", True)
    if not isinstance(checksums, bool):
        _fail("BENCHMARK_INVALID", f"{path}: 'checksums' must be a boolean")
    verifier_model: str | None = None
    verifier_raw = data.get("verifier")
    if verifier_raw is not None:
        if not isinstance(verifier_raw, dict):
            _fail("BENCHMARK_INVALID", f"{path}: [verifier] must be a table")
        _check_keys(verifier_raw, {"model"}, "[verifier]")
        verifier_model = _require_str(verifier_raw, "model", "[verifier]")
    if task == "verification" and verifier_model is None:
        _fail(
            "BENCHMARK_INVALID",
            f"{path}: task 'verification' requires a [verifier] block with 'model'",
        )
    if task != "verification" and verifier_model is not None:
        _fail(
            "BENCHMARK_INVALID",
            f"{path}: a [verifier] block requires task 'verification'",
        )
    return BenchmarkSpec(
        id=_require_str(data, "id", str(path)),
        task=task,
        dataset=_require_str(data, "dataset", str(path)),
        split=split,
        pairs=tuple(pairs),
        checksums=checksums,
        verifier_model=verifier_model,
    )


def _check_relative(path: str, where: str) -> None:
    if (
        PurePosixPath(path).is_absolute()
        or PureWindowsPath(path).is_absolute()
        or PureWindowsPath(path).drive
    ):
        _fail("BENCHMARK_INVALID", f"{where}: path must be relative: {path!r}")


def _load_dataset(owner_path: Path, dataset: str) -> tuple[DatasetManifest, Path, Path]:
    """Resolve the declared dataset manifest; return (manifest, dataset_path, root)."""
    _check_relative(dataset, f"{owner_path}: 'dataset'")
    dataset_path = owner_path.parent / dataset
    return load_dataset_manifest(dataset_path), dataset_path, dataset_path.parent


def _require_number(table: dict[str, Any], key: str, where: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("BENCHMARK_INVALID", f"{where}: {key!r} must be a number")
    return float(value)


def _optional_number(table: dict[str, Any], key: str, where: str) -> float | None:
    if key not in table:
        return None
    return _require_number(table, key, where)


def _optional_int(table: dict[str, Any], key: str, where: str) -> int | None:
    if key not in table:
        return None
    return _require_int(table, key, where)


def _require_bool(table: dict[str, Any], key: str, where: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        _fail("BENCHMARK_INVALID", f"{where}: {key!r} must be a boolean")
    return value


def _optional_str_list(table: dict[str, Any], key: str, where: str) -> tuple[str, ...] | None:
    if key not in table:
        return None
    value = table[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _fail("BENCHMARK_INVALID", f"{where}: {key!r} must be a list of strings")
    return tuple(value)


def _int_with_default(table: dict[str, Any], key: str, where: str, default: int) -> int:
    if key not in table:
        return default
    return _require_int(table, key, where)


def _number_with_default(table: dict[str, Any], key: str, where: str, default: float) -> float:
    if key not in table:
        return default
    return _require_number(table, key, where)


def _bool_with_default(table: dict[str, Any], key: str, where: str, default: bool) -> bool:
    if key not in table:
        return default
    return _require_bool(table, key, where)


def _parse_verifier_block(table: Any) -> VerifierBlock:
    where = "[verifier]"
    if not isinstance(table, dict):
        _fail("BENCHMARK_INVALID", f"{where}: table expected")
    _check_keys(table, _VERIFIER_KEYS, where)
    policy_raw = table.get("pair_policy", {})
    if not isinstance(policy_raw, dict):
        _fail("BENCHMARK_INVALID", f"{where}: 'pair_policy' must be a table")
    _check_keys(policy_raw, _PAIR_POLICY_KEYS, "[verifier.pair_policy]")
    return VerifierBlock(
        kind=_require_str(table, "kind", where),
        l2_lambda=_require_number(table, "l2_lambda", where),
        min_support_fraction=_require_number(table, "min_support_fraction", where),
        min_class_support_fraction=_require_number(table, "min_class_support_fraction", where),
        min_pairs=_require_int(table, "min_pairs", where),
        threshold_rule=_require_str(table, "threshold_rule", where),
        threshold_alpha=_optional_number(table, "threshold_alpha", where),
        threshold_fixed=_optional_number(table, "threshold_fixed", where),
        calibration_method=(
            _require_str(table, "calibration_method", where)
            if "calibration_method" in table
            else None
        ),
        max_iterations=_int_with_default(table, "max_iterations", where, 100),
        tolerance=_number_with_default(table, "tolerance", where, 1e-12),
        include_linguistic=_bool_with_default(table, "include_linguistic", where, False),
        allow_unconstrained_language=_bool_with_default(
            table, "allow_unconstrained_language", where, False
        ),
        languages=_optional_str_list(table, "languages", where),
        feature_ids=_optional_str_list(table, "feature_ids", where),
        max_pairs_per_author=_optional_int(policy_raw, "max_pairs_per_author", where),
        max_pairs_per_problem=_optional_int(policy_raw, "max_pairs_per_problem", where),
        negative_positive_ratio=_optional_number(policy_raw, "negative_positive_ratio", where),
        selection_version=(
            _require_str(policy_raw, "selection_version", "[verifier.pair_policy]")
            if "selection_version" in policy_raw
            else "1"
        ),
    )


def load_training_manifest(path: Path) -> TrainingManifest:
    """Parse and strictly validate a ``stylog.verifier-training`` TOML manifest."""
    data = _load_toml(path)
    _check_keys(data, _TRAINING_KEYS, str(path))
    _check_header(data, VERIFIER_TRAINING_SCHEMA, str(path))
    if "verifier" not in data:
        _fail("BENCHMARK_INVALID", f"{path}: [verifier] block is required")
    verifier = _parse_verifier_block(data["verifier"])
    pairs_raw = data.get("pair", [])
    if not isinstance(pairs_raw, list):
        _fail("BENCHMARK_INVALID", f"{path}: 'pair' must be an array of tables")
    pairs: list[TrainingPairSpec] = []
    for index, entry in enumerate(pairs_raw):
        where = f"[[pair]] #{index}"
        if not isinstance(entry, dict):
            _fail("BENCHMARK_INVALID", f"{where}: table expected")
        _check_keys(entry, _TRAINING_PAIR_KEYS, where)
        left = _require_str(entry, "left", where)
        right = _require_str(entry, "right", where)
        if left == right:
            _fail("PAIR_INVALID", f"{where}: left and right must differ")
        label = _require_str(entry, "label", where)
        if label not in PAIR_LABELS:
            _fail("PAIR_INVALID", f"{where}: 'label' must be one of {PAIR_LABELS}")
        population = entry.get("population", "train")
        if not isinstance(population, str) or population not in TRAINING_POPULATIONS:
            _fail(
                "PAIR_INVALID",
                f"{where}: 'population' must be one of {TRAINING_POPULATIONS} "
                "(evaluation pairs belong in benchmark specs, never in training)",
            )
        pairs.append(
            TrainingPairSpec(left=left, right=right, label=label, population=population)
        )
    if not pairs:
        _fail("BENCHMARK_INVALID", f"{path}: at least one [[pair]] is required")
    return TrainingManifest(
        id=_require_str(data, "id", str(path)),
        dataset=_require_str(data, "dataset", str(path)),
        verifier=verifier,
        pairs=tuple(pairs),
    )


def validate_dataset(manifest: DatasetManifest, root: Path, verify_checksums: bool) -> None:
    """Check that every declared artifact file exists and matches its sha256."""
    for artifact in manifest.artifacts:
        _check_relative(artifact.path, f"artifact {artifact.id!r}")
        file_path = root / artifact.path
        if not file_path.is_file():
            _fail("DATASET_MISSING", f"artifact {artifact.id!r}: file not found: {artifact.path}")
        if verify_checksums:
            digest = sha256_hex(file_path.read_bytes())
            if digest != artifact.sha256:
                _fail(
                    "DATASET_CHECKSUM_MISMATCH",
                    f"artifact {artifact.id!r}: sha256 mismatch for {artifact.path}",
                )
