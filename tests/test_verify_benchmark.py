"""Verification benchmark task and PAN decision metrics (spec 21, 23)."""

from __future__ import annotations

import hashlib

import pytest
from test_verify import TTR
from test_verify_fit import make_spec, separable_pairs

from stylog.benchmark.evaluate import evaluate_verification
from stylog.benchmark.manifest import (
    BenchmarkSpec,
    PairSpec,
    SplitSpec,
    load_benchmark_spec,
)
from stylog.benchmark.metrics import DecisionRow, brier, c_at_1, f1, f_05u, roc_auc
from stylog.benchmark.split import compute_split
from stylog.config import StylogConfig
from stylog.domain.benchmark import BenchmarkResult
from stylog.domain.verification import VerifierFit
from stylog.exceptions import BenchmarkError
from stylog.serialization.canonical import canonical_bytes, scientific_sha256
from stylog.serialization.jsonio import write_json_atomic
from stylog.verification.fit import fit_verifier_model

# --- metric goldens (worked example) --------------------------------------
#
# rows: 1 same/same(0.9) 2 same/diff(0.2) 3 diff/diff(0.1) 4 diff/same(0.8)
#       5 same/abstain-uncertain(0.5)  6 diff/abstain-insufficient(no score)
ROWS = [
    DecisionRow(verdict="same_author", label="same", score=0.9, probability=0.9),
    DecisionRow(verdict="different_author", label="same", score=0.2, probability=0.2),
    DecisionRow(verdict="different_author", label="different", score=0.1, probability=0.1),
    DecisionRow(verdict="same_author", label="different", score=0.8, probability=0.8),
    DecisionRow(verdict="abstain", label="same", score=0.5, probability=0.5),
    DecisionRow(verdict="abstain", label="different", score=None, probability=None),
]


def test_c_at_1_golden() -> None:
    # nc=2 (rows 1,3), nu=2, N=6: (1/6)(2 + 2*(2/6)) = 4/9
    assert c_at_1(ROWS) == pytest.approx(4.0 / 9.0, abs=1e-15)


def test_f1_golden() -> None:
    # answered rows 1-4: TP=1 FP=1 FN=1 -> 2/(2+1+1) = 0.5
    assert f1(ROWS) == 0.5


def test_f_05u_golden() -> None:
    # 1.25*1 / (1.25 + 0.25*(1+2) + 1) = 1.25/3 = 5/12
    assert f_05u(ROWS) == pytest.approx(5.0 / 12.0, abs=1e-15)


def test_roc_auc_golden() -> None:
    # same scores [0.9, 0.2, 0.5], different scores [0.1, 0.8]: 4/6 wins
    assert roc_auc(ROWS) == pytest.approx(2.0 / 3.0, abs=1e-15)
    # insufficient-evidence row (no score) is excluded from AUC
    assert roc_auc([ROWS[5]]) is None


def test_brier_golden() -> None:
    # probabilities over rows 1-5: (0.01+0.64+0.01+0.64+0.25)/5 = 1.55/5
    assert brier(ROWS) == pytest.approx(0.31, abs=1e-15)
    assert brier([ROWS[5]]) is None  # no probability rows


def test_metric_edges() -> None:
    all_abstain = [
        DecisionRow(verdict="abstain", label="same", score=0.5, probability=0.5),
        DecisionRow(verdict="abstain", label="different", score=0.5, probability=0.5),
    ]
    assert c_at_1(all_abstain) == 0.0
    assert f1(all_abstain) is None
    assert f_05u(all_abstain) == 0.0
    none_abstain = [row for row in ROWS if row.answered]
    assert c_at_1(none_abstain) == 0.5  # 2 correct of 4
    assert f_05u(none_abstain) == pytest.approx(1.25 / (1.25 + 0.25 + 1.0), abs=1e-15)


# --- split disjoint_content flag ------------------------------------------


def _manifest_with_duplicates():
    from stylog.benchmark.manifest import DatasetArtifact, DatasetManifest

    artifacts = (
        DatasetArtifact(
            id="a1", path="a1.txt", sha256="1" * 64, kind="text", language="en",
            context={"author_id": "auth1"},
        ),
        DatasetArtifact(
            id="a2", path="a2.txt", sha256="1" * 64, kind="text", language="en",
            context={"author_id": "auth2"},  # duplicate content, different author
        ),
        DatasetArtifact(
            id="b1", path="b1.txt", sha256="2" * 64, kind="text", language="en",
            context={"author_id": "auth3"},
        ),
        DatasetArtifact(
            id="c1", path="c1.txt", sha256="3" * 64, kind="text", language="en",
            context={"author_id": "auth4"},
        ),
    )
    return DatasetManifest(
        id="d", version="1", license="CC0", redistribution="allowed",
        source="synthetic", artifacts=artifacts, risks={}, transformations=(),
    )


def test_disjoint_content_unions_duplicate_hashes() -> None:
    manifest = _manifest_with_duplicates()
    split = SplitSpec(
        seed="s",
        train_ppm=500_000,
        dev_ppm=250_000,
        test_ppm=250_000,
        disjoint_by=("author_id",),
        require_nonempty=False,
        disjoint_content=True,
    )
    result = compute_split(manifest, split)
    # a1 and a2 share content: same part despite different authors
    assert result.assignment["a1"] == result.assignment["a2"]
    plain = SplitSpec(
        seed="s",
        train_ppm=500_000,
        dev_ppm=250_000,
        test_ppm=250_000,
        disjoint_by=("author_id",),
        require_nonempty=False,
    )
    # default-off: config tree and hash are unchanged by the flag's existence
    assert "disjoint_content" not in plain.as_tree()
    assert plain.as_tree() != split.as_tree()
    assert result.split_config_sha256 != compute_split(manifest, plain).split_config_sha256


# --- evaluate_verification end-to-end --------------------------------------


def _write_dataset(tmp_path):
    texts = {}
    for i in range(6):
        texts[f"a{i}.txt"] = ("the quick brown fox jumps over the lazy dog. " * (10 + i)).strip() + "\n"
        texts[f"b{i}.txt"] = ("How vexingly quick daft zebras jump! Bright vixens jump. " * (10 + i)).strip() + "\n"
    lines = [
        'schema = "stylog.dataset"',
        'schema_version = "0.1.0"',
        'id = "bench"',
        'version = "1"',
        'license = "CC0"',
        'redistribution = "allowed"',
        'source = "synthetic"',
    ]
    for name, text in sorted(texts.items()):
        data = text.encode("utf-8")
        (tmp_path / name).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        lines += [
            "[[artifact]]",
            f'id = "{name}"',
            f'path = "{name}"',
            f'sha256 = "{sha}"',
            'kind = "text"',
            'language = "en"',
        ]
    (tmp_path / "dataset.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    from stylog.benchmark.manifest import load_dataset_manifest

    return load_dataset_manifest(tmp_path / "dataset.toml")


def _spec(tmp_path, pairs) -> BenchmarkSpec:
    return BenchmarkSpec(
        id="vb",
        task="verification",
        dataset="dataset.toml",
        split=None,
        pairs=tuple(pairs),
        checksums=True,
        verifier_model="model.json",
    )


def _fit_and_write_model(tmp_path) -> VerifierFit:
    model, _ = fit_verifier_model(make_spec(), separable_pairs())
    write_json_atomic(tmp_path / "model.json", model)
    return model


def test_evaluate_verification_end_to_end_deterministic(tmp_path) -> None:
    manifest = _write_dataset(tmp_path)
    model = _fit_and_write_model(tmp_path)
    pairs = [
        PairSpec(left="a0.txt", right="a1.txt", label="same"),
        PairSpec(left="a2.txt", right="b2.txt", label="different"),
        PairSpec(left="a3.txt", right="b3.txt", label="different"),
        PairSpec(left="a4.txt", right="a5.txt", label="same"),
    ]
    spec = _spec(tmp_path, pairs)
    config = StylogConfig()
    result_a = evaluate_verification(spec, manifest, tmp_path, "d" * 64, model, config)
    result_b = evaluate_verification(spec, manifest, tmp_path, "d" * 64, model, config)
    assert canonical_bytes(result_a) == canonical_bytes(result_b)
    metrics = result_a.verification_metrics
    assert metrics is not None
    assert metrics.pair_count == 4
    assert metrics.verifier_id == scientific_sha256(model)
    assert metrics.c_at_1 is not None
    # result stays a valid portable object
    BenchmarkResult.model_validate(
        __import__("json").loads(canonical_bytes(result_a).decode())
    )


def test_evaluate_verification_rejects_training_population(tmp_path) -> None:
    manifest = _write_dataset(tmp_path)
    _fit_and_write_model(tmp_path)
    config = StylogConfig()
    # fit a model on exactly the pairs that will later be "evaluated"
    from stylog.application.fingerprint import fingerprint_artifact
    from stylog.bootstrap import build_context, build_default_services
    from stylog.infrastructure.ingest import artifact_from_bytes
    from stylog.verification.spec import TrainingPair

    services = build_default_services(config)
    ctx = build_context(config, services)

    def fp(artifact_id):
        artifact = manifest.artifact_by_id(artifact_id)
        data = (tmp_path / artifact.path).read_bytes()
        runtime_artifact = artifact_from_bytes(
            data, artifact_id=artifact.id, kind="text", language="en", config=config
        )
        return fingerprint_artifact(
            runtime_artifact, config=config, services=services, ctx=ctx, no_cache=True
        ).fingerprint

    training_pairs = [
        TrainingPair(left=fp("a0.txt"), right=fp("a1.txt"), label="same"),
        TrainingPair(left=fp("a2.txt"), right=fp("b2.txt"), label="different"),
    ] + [
        TrainingPair(left=fp("a3.txt"), right=fp("b3.txt"), label="different"),
        TrainingPair(left=fp("a4.txt"), right=fp("a5.txt"), label="same"),
    ]
    trained, _ = fit_verifier_model(
        make_spec(min_pairs=2, feature_ids=(TTR,)),
        training_pairs,
    )
    evaluation_pairs = [
        PairSpec(left="a0.txt", right="a1.txt", label="same"),
        PairSpec(left="a2.txt", right="b2.txt", label="different"),
        PairSpec(left="a3.txt", right="b3.txt", label="different"),
        PairSpec(left="a4.txt", right="a5.txt", label="same"),
    ]
    spec_overlap = _spec(tmp_path, evaluation_pairs)
    with pytest.raises(BenchmarkError, match="disjoint"):
        evaluate_verification(spec_overlap, manifest, tmp_path, "d" * 64, trained, config)


def test_benchmark_result_byte_stability_without_verification() -> None:
    result = BenchmarkResult(
        benchmark_id="b",
        task="pairwise_comparison",
        dataset_manifest_sha256="d" * 64,
        split_algorithm_version="stylog-split-v1",
    )
    tree = canonical_bytes(result)
    assert b"verification_metrics" not in tree


def test_benchmark_spec_verifier_block_parsing(tmp_path) -> None:
    spec_text = '''schema = "stylog.benchmark"
schema_version = "0.1.0"
id = "vb"
task = "verification"
dataset = "dataset.toml"

[verifier]
model = "model.json"

[[pair]]
left = "a"
right = "b"
label = "same"
'''
    path = tmp_path / "spec.toml"
    path.write_text(spec_text, encoding="utf-8", newline="\n")
    spec = load_benchmark_spec(path)
    assert spec.task == "verification"
    assert spec.verifier_model == "model.json"
    # [verifier] requires the verification task
    bad = tmp_path / "bad.toml"
    bad.write_text(
        spec_text.replace('task = "verification"', 'task = "pairwise_comparison"'),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(BenchmarkError):
        load_benchmark_spec(bad)
    # verification task requires [verifier]
    bad2 = tmp_path / "bad2.toml"
    bad2.write_text(
        spec_text.replace('\n[verifier]\nmodel = "model.json"\n', "\n"),
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(BenchmarkError):
        load_benchmark_spec(bad2)
