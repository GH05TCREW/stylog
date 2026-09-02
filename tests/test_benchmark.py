"""Benchmark subsystem tests (spec section 21; fixtures 25.23-25.24)."""

from __future__ import annotations

import hashlib

import pytest

from stylog.analysis import stats
from stylog.benchmark.api import run_benchmark
from stylog.config import StylogConfig
from stylog.domain.benchmark import BenchmarkResult
from stylog.exceptions import BenchmarkError
from stylog.serialization.canonical import canonical_bytes
from stylog.serialization.jsonio import model_from_bytes

CONFIG = StylogConfig()

SPLIT_AUDIT_FILES = {
    "a1": "alpha one\nThe quick brown fox jumps over the lazy dog.\n",
    "a2": "alpha two\nThe quick brown fox jumps over the lazy cat.\n",
    "b1": "beta one\nA completely different sentence about ships and harbors.\n",
    "b2": "beta two\nAnother unrelated paragraph regarding mountains and rivers.\n",
    "c1": "gamma one\nShort text.\n",
    "c2": "gamma two\nSlightly longer text with a few more words in it.\n",
    "d1": "delta one\nNumbers 1 2 3 and words mixed together here.\n",
    "d2": "delta two\nFinal document for the synthetic benchmark dataset.\n",
}
SPLIT_AUDIT_REPOS = {"a1": "repo-a", "a2": "repo-a", "b1": "repo-b", "b2": "repo-b"}
SPLIT_AUDIT_RISKS = {
    "repository_sensitive": "documents sharing repository_id may share templates",
    "topic_sensitive": "toy texts share vocabulary by construction",
}

PAIR_FILES = {
    "p1": "the cat sat on the mat and purred softly all afternoon\n",
    "p2": "the cat sat on the mat and purred loudly all morning\n",
    "q1": "quantum chromodynamics explains the strong interaction between quarks\n",
    "q2": "gauge bosons mediate fundamental forces in particle physics theories\n",
}

TRANSFORMATION_FILES = {
    "t1": "original wording stays mostly stable across the transformation\n",
    "t2": "original wording stays mostly stable across the transformation!\n",
}

DEMO_SPLIT = {
    "seed": "stylog-demo-1",
    "train_ppm": 500000,
    "dev_ppm": 250000,
    "test_ppm": 250000,
    "disjoint_by": ("repository_id",),
    "require_nonempty": False,
}
# Golden for DEMO_SPLIT over SPLIT_AUDIT_FILES (fixture 25.23), computed once
# with stylog.benchmark.split.compute_split and pinned here.
GOLDEN_TRAIN = ("c2", "d1")
GOLDEN_DEV = ("c1",)
GOLDEN_TEST = ("a1", "a2", "b1", "b2", "d2")
GOLDEN_SPLIT_CONFIG_SHA256 = "5c5d5228d4f5be68f3b1abb604a7027cfa8e3c2a6afc35339b7c9503d55d8358"


def _write_dataset(
    root,
    files,
    *,
    repos=None,
    risks=None,
    transformations=(),
) -> None:
    repos = repos or {}
    digests = {}
    for artifact_id, text in files.items():
        file_path = root / f"{artifact_id}.txt"
        file_path.write_text(text, encoding="utf-8")
        digests[artifact_id] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    lines = [
        'schema = "stylog.dataset"',
        'schema_version = "0.1.0"',
        'id = "demo-dataset"',
        'version = "1.0.0"',
        'license = "CC0-1.0"',
        'redistribution = "allowed"',
        'source = "synthetic test fixture"',
    ]
    if risks:
        lines.append("")
        lines.append("[risks]")
        for key in sorted(risks):
            lines.append(f'{key} = "{risks[key]}"')
    for artifact_id in sorted(files):
        lines.append("")
        lines.append("[[artifact]]")
        lines.append(f'id = "{artifact_id}"')
        lines.append(f'path = "{artifact_id}.txt"')
        lines.append(f'sha256 = "{digests[artifact_id]}"')
        lines.append('kind = "text"')
        lines.append('language = "und"')
        if artifact_id in repos:
            lines.append(f'repository_id = "{repos[artifact_id]}"')
    for original, variant, transformation_id in transformations:
        lines.append("")
        lines.append("[[transformation]]")
        lines.append(f'original = "{original}"')
        lines.append(f'variant = "{variant}"')
        lines.append(f'transformation_id = "{transformation_id}"')
    (root / "dataset.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_spec(root, task, *, split=None, pairs=(), extra_lines=()):
    lines = [
        'schema = "stylog.benchmark"',
        'schema_version = "0.1.0"',
        'id = "demo-benchmark"',
        f'task = "{task}"',
        'dataset = "dataset.toml"',
        *extra_lines,
    ]
    if split is not None:
        lines.append("")
        lines.append("[split]")
        lines.append(f'seed = "{split["seed"]}"')
        lines.append(f'train_ppm = {split["train_ppm"]}')
        lines.append(f'dev_ppm = {split["dev_ppm"]}')
        lines.append(f'test_ppm = {split["test_ppm"]}')
        disjoint = ", ".join(f'"{field_name}"' for field_name in split["disjoint_by"])
        lines.append(f"disjoint_by = [{disjoint}]")
        lines.append(f"require_nonempty = {'true' if split['require_nonempty'] else 'false'}")
    for left, right, label in pairs:
        lines.append("")
        lines.append("[[pair]]")
        lines.append(f'left = "{left}"')
        lines.append(f'right = "{right}"')
        lines.append(f'label = "{label}"')
    path = root / "spec.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_split_audit_project(root) -> None:
    _write_dataset(
        root, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS, risks=SPLIT_AUDIT_RISKS
    )
    _write_spec(root, "split_audit", split=DEMO_SPLIT)


def _assert_no_nulls(tree) -> None:
    assert tree is not None, "portable JSON must not contain null"
    if isinstance(tree, dict):
        for value in tree.values():
            _assert_no_nulls(value)
    elif isinstance(tree, list):
        for item in tree:
            _assert_no_nulls(item)


def _assert_roundtrip(result: BenchmarkResult) -> None:
    data = canonical_bytes(result)
    assert b"null" not in data
    _assert_no_nulls(result.model_dump(mode="json", exclude_none=True))
    if result.diagnostics:
        # NOTE: the repo-wide strict config rejects plain strings for the
        # StrEnum Diagnostic.severity field, so model_from_bytes cannot parse
        # ANY diagnostic-bearing portable model (pre-existing upstream quirk,
        # also affecting Fingerprint cache reads). Parse lax here instead.
        import json

        parsed = BenchmarkResult.model_validate(json.loads(data), strict=False)
    else:
        parsed = model_from_bytes(data, BenchmarkResult)
    assert parsed == result


def _parts(result: BenchmarkResult) -> dict[str, str]:
    parts = {}
    for part in ("train", "dev", "test"):
        for artifact_id in getattr(result.splits, part):
            parts[artifact_id] = part
    return parts


def test_split_audit_golden(tmp_path):
    _write_split_audit_project(tmp_path)
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert result.task == "split_audit"
    assert result.split_algorithm_version == "stylog-split-v1"
    assert result.split_config_sha256 == GOLDEN_SPLIT_CONFIG_SHA256
    assert result.splits is not None
    assert result.splits.train == GOLDEN_TRAIN
    assert result.splits.dev == GOLDEN_DEV
    assert result.splits.test == GOLDEN_TEST
    manifest_bytes = (tmp_path / "dataset.toml").read_bytes()
    assert result.dataset_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    _assert_roundtrip(result)


def test_split_audit_is_deterministic(tmp_path):
    _write_split_audit_project(tmp_path)
    first = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    second = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert canonical_bytes(first) == canonical_bytes(second)


def test_split_component_integrity(tmp_path):
    _write_split_audit_project(tmp_path)
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    parts = _parts(result)
    for members in (("a1", "a2"), ("b1", "b2")):
        assert len({parts[member] for member in members}) == 1


def test_split_audit_echoes_risks_without_analysis(tmp_path):
    _write_split_audit_project(tmp_path)
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert [(entry.key, entry.value) for entry in result.risk_declarations] == sorted(
        SPLIT_AUDIT_RISKS.items()
    )
    # split_audit performs no content analysis.
    assert result.pairwise_metrics == ()
    assert result.transformation_distances == ()


def test_split_audit_without_split_section_is_invalid(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    spec = _write_spec(tmp_path, "split_audit")
    with pytest.raises(BenchmarkError, match="BENCHMARK_INVALID"):
        run_benchmark(spec, config=CONFIG)


def test_split_impossible_when_disjoint_field_missing(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    split = {**DEMO_SPLIT, "require_nonempty": True}
    spec = _write_spec(tmp_path, "split_audit", split=split)
    with pytest.raises(BenchmarkError, match="SPLIT_IMPOSSIBLE"):
        run_benchmark(spec, config=CONFIG)


def test_split_ppm_zero_parts_may_be_empty(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    split = {
        "seed": "stylog-demo-1",
        "train_ppm": 1_000_000,
        "dev_ppm": 0,
        "test_ppm": 0,
        "disjoint_by": (),
        "require_nonempty": True,
    }
    spec = _write_spec(tmp_path, "split_audit", split=split)
    result = run_benchmark(spec, config=CONFIG)
    assert result.splits.train == tuple(sorted(SPLIT_AUDIT_FILES))
    assert result.splits.dev == ()
    assert result.splits.test == ()


def test_split_ppm_sum_must_be_exact(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    split = {**DEMO_SPLIT, "test_ppm": 250_001}
    spec = _write_spec(tmp_path, "split_audit", split=split)
    with pytest.raises(BenchmarkError, match="BENCHMARK_INVALID"):
        run_benchmark(spec, config=CONFIG)


def test_checksum_mismatch_rejected(tmp_path):
    _write_split_audit_project(tmp_path)
    (tmp_path / "a1.txt").write_text("tampered content\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="DATASET_CHECKSUM_MISMATCH"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def test_checksum_mismatch_ignored_when_checksums_disabled(tmp_path):
    _write_split_audit_project(tmp_path)
    _write_spec(tmp_path, "split_audit", split=DEMO_SPLIT, extra_lines=("checksums = false",))
    (tmp_path / "a1.txt").write_text("tampered content\n", encoding="utf-8")
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert result.splits.train == GOLDEN_TRAIN


def test_missing_artifact_file_rejected(tmp_path):
    _write_split_audit_project(tmp_path)
    (tmp_path / "d2.txt").unlink()
    with pytest.raises(BenchmarkError, match="DATASET_MISSING"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def test_unknown_spec_key_rejected(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    spec = _write_spec(tmp_path, "split_audit", split=DEMO_SPLIT, extra_lines=("bogus = 1",))
    with pytest.raises(BenchmarkError, match="BENCHMARK_INVALID"):
        run_benchmark(spec, config=CONFIG)


def test_unknown_dataset_key_rejected(tmp_path):
    _write_split_audit_project(tmp_path)
    dataset = tmp_path / "dataset.toml"
    dataset.write_text(
        dataset.read_text(encoding="utf-8").replace(
            'source = "synthetic test fixture"',
            'source = "synthetic test fixture"\nbogus = true',
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkError, match="BENCHMARK_INVALID"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def _write_pairwise_project(tmp_path, pairs) -> None:
    _write_dataset(tmp_path, PAIR_FILES)
    _write_spec(tmp_path, "pairwise_comparison", pairs=pairs)


def test_pairwise_comparison_metrics(tmp_path):
    _write_pairwise_project(
        tmp_path,
        pairs=(
            ("p1", "p2", "same"),
            ("q1", "q2", "same"),
            ("p1", "q1", "different"),
            ("p2", "q2", "different"),
        ),
    )
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert result.task == "pairwise_comparison"
    assert result.splits is None
    assert result.split_config_sha256 is None
    assert result.pairwise_metrics != ()
    feature_ids = [entry.feature_id for entry in result.pairwise_metrics]
    assert feature_ids == sorted(feature_ids)
    for entry in result.pairwise_metrics:
        assert entry.same_count == 2
        assert entry.different_count == 2
        assert entry.same_mean_distance is not None
        assert entry.different_mean_distance is not None
        assert entry.same_median_distance is not None
        assert entry.different_median_distance is not None
        assert entry.roc_auc is not None
        assert 0.0 <= entry.roc_auc <= 1.0
    _assert_roundtrip(result)


def test_pairwise_single_class_omits_roc_auc(tmp_path):
    _write_pairwise_project(tmp_path, pairs=(("p1", "p2", "same"), ("q1", "q2", "same")))
    result = run_benchmark(tmp_path / "spec.toml", config=CONFIG)
    assert result.pairwise_metrics != ()
    for entry in result.pairwise_metrics:
        assert entry.same_count == 2
        assert entry.different_count == 0
        assert entry.same_mean_distance is not None
        assert entry.different_mean_distance is None
        assert entry.roc_auc is None
    auc_diagnostics = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "PAIRWISE_AUC_OMITTED"
    ]
    assert len(auc_diagnostics) == len(result.pairwise_metrics)
    _assert_roundtrip(result)


def test_pairwise_rejects_unknown_artifact(tmp_path):
    _write_pairwise_project(tmp_path, pairs=(("p1", "nope", "same"),))
    with pytest.raises(BenchmarkError, match="PAIR_INVALID"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def test_pairwise_rejects_self_pair(tmp_path):
    _write_pairwise_project(tmp_path, pairs=(("p1", "p1", "same"),))
    with pytest.raises(BenchmarkError, match="PAIR_INVALID"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def test_pairwise_rejects_unknown_label(tmp_path):
    _write_pairwise_project(tmp_path, pairs=(("p1", "p2", "similar"),))
    with pytest.raises(BenchmarkError, match="PAIR_INVALID"):
        run_benchmark(tmp_path / "spec.toml", config=CONFIG)


def test_pairwise_rejects_cross_split_pair(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    # Golden: c2 lands in train, a1 in test, so the pair violates the protocol.
    spec = _write_spec(
        tmp_path,
        "pairwise_comparison",
        split=DEMO_SPLIT,
        pairs=(("c2", "a1", "different"),),
    )
    with pytest.raises(BenchmarkError, match="PAIR_INVALID"):
        run_benchmark(spec, config=CONFIG)


def test_pairwise_same_split_pair_accepted(tmp_path):
    _write_dataset(tmp_path, SPLIT_AUDIT_FILES, repos=SPLIT_AUDIT_REPOS)
    # Golden: a1 and a2 are both in test.
    spec = _write_spec(
        tmp_path,
        "pairwise_comparison",
        split=DEMO_SPLIT,
        pairs=(("a1", "a2", "same"),),
    )
    result = run_benchmark(spec, config=CONFIG)
    assert result.splits is not None
    assert result.splits.train == GOLDEN_TRAIN
    assert result.pairwise_metrics != ()
    for entry in result.pairwise_metrics:
        assert entry.same_count == 1
        assert entry.different_count == 0
        assert entry.roc_auc is None
    _assert_roundtrip(result)


def test_auc_mann_whitney_midrank_ties():
    # Fixture 25.24: ties share midranks.
    assert stats.roc_auc_mann_whitney(positive=[0.5, 0.9], negative=[0.5, 0.1]) == 0.875


def test_transformation_stability(tmp_path):
    _write_dataset(
        tmp_path,
        TRANSFORMATION_FILES,
        transformations=(("t1", "t2", "identity-demo"),),
    )
    spec = _write_spec(tmp_path, "transformation_stability")
    result = run_benchmark(spec, config=CONFIG)
    assert result.task == "transformation_stability"
    assert result.transformation_distances != ()
    feature_ids = [entry.feature_id for entry in result.transformation_distances]
    assert feature_ids == sorted(feature_ids)
    for entry in result.transformation_distances:
        assert entry.transformation_id == "identity-demo"
        assert entry.original == "t1"
        assert entry.variant == "t2"
        assert entry.value >= 0.0
    _assert_roundtrip(result)


def test_transformation_stability_requires_transformations(tmp_path):
    _write_dataset(tmp_path, TRANSFORMATION_FILES)
    spec = _write_spec(tmp_path, "transformation_stability")
    with pytest.raises(BenchmarkError, match="BENCHMARK_INVALID"):
        run_benchmark(spec, config=CONFIG)
