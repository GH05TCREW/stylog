"""Benchmark entry points shared by the Python API and the CLI (spec 21).

Everything is local: the dataset manifest is resolved relative to the spec
file, the dataset root is the manifest's own directory, and nothing is ever
downloaded.
"""

from __future__ import annotations

from pathlib import Path

from stylog.benchmark import evaluate
from stylog.benchmark.manifest import (
    _check_relative,
    _fail,
    _load_dataset,
    load_benchmark_spec,
    validate_dataset,
)
from stylog.config import StylogConfig, load_config
from stylog.domain.benchmark import BenchmarkResult
from stylog.serialization.canonical import sha256_hex


def run_benchmark(spec_path: str | Path, *, config: StylogConfig | None = None) -> BenchmarkResult:
    """Run the benchmark described by a ``stylog.benchmark`` TOML spec."""
    cfg = config if config is not None else load_config()
    spec_path = Path(spec_path)
    spec = load_benchmark_spec(spec_path)
    manifest, dataset_path, root = _load_dataset(spec_path, spec.dataset)
    dataset_manifest_sha256 = sha256_hex(dataset_path.read_bytes())
    validate_dataset(manifest, root, spec.checksums)

    if spec.task == "split_audit":
        return evaluate.evaluate_split_audit(spec, manifest, dataset_manifest_sha256)
    if spec.task == "pairwise_comparison":
        return evaluate.evaluate_pairwise(spec, manifest, root, dataset_manifest_sha256, cfg)
    if spec.task == "transformation_stability":
        return evaluate.evaluate_transformation_stability(
            spec, manifest, root, dataset_manifest_sha256, cfg
        )
    if spec.task == "verification":
        from stylog.domain.verification import VerifierFit
        from stylog.serialization.jsonio import read_json

        assert spec.verifier_model is not None  # validated at parse time
        _check_relative(spec.verifier_model, f"{spec_path}: [verifier] 'model'")
        model_path = spec_path.parent / spec.verifier_model
        if not model_path.is_file():
            _fail("BENCHMARK_INVALID", f"verifier model not found: {spec.verifier_model}")
        model = read_json(model_path, VerifierFit)
        return evaluate.evaluate_verification(
            spec, manifest, root, dataset_manifest_sha256, model, cfg
        )
    _fail("BENCHMARK_INVALID", f"unknown benchmark task {spec.task!r}")
    raise AssertionError("unreachable")


def run_benchmark_file(path: str | Path) -> BenchmarkResult:
    """Thin wrapper for the CLI: run a benchmark spec with default config."""
    return run_benchmark(Path(path))
