# Stylog — agent/maintainer notes

## Commands

- Install (editable, all capabilities): `pip install -e ".[all,dev]"`
- Tests: `python -m pytest` (full suite). Architecture/offline/parallel gates
  live in `tests/test_architecture.py`, `tests/test_offline.py`,
  `tests/test_parallel.py`.
- Schema drift gate: `python tools/generate_schemas.py` / `--check`
  (committed schemas in `schemas/`).
- Grammar manifest regeneration (after grammar upgrades):
  `python tools/generate_grammar_manifest.py`.
- Lint: `python -m ruff check src tests`.

## Conventions

- Dev generators live in `tools/` (not shipped); tests import them via the
  repo-root `conftest.py` putting the repo root on sys.path.
- `docs/spec-v0.1.md` is the normative contract. Feature semantics live in
  `src/stylog/analysis/registry.py` (single source of truth for feature IDs,
  reducers, metrics, top-codes, owners).
- `domain/` is strict Pydantic v2 portable models only: no nulls, no heavy
  imports, tuples not lists. Domain never imports analysis/infrastructure/CLI.
- Analysis modules are pure functions over `RuntimeArtifact`; no I/O, no JCS,
  no network. Serialization lives only in `serialization/`; cache/baseline/
  resource adapters live in `infrastructure/` behind the three ports in
  `ports.py`.
- Optional extras (spacy/sklearn/pyarrow/polars/duckdb/pandas) are imported
  lazily inside functions only; `import stylog` must stay light (enforced by
  `tests/test_architecture.py`).
- Portable models reject explicit JSON null. Optional fields use `None`
  defaults and are omitted on serialization — never pass explicit `None`
  kwargs into portable model constructors (the null-rejection validator will
  raise).
- Windows: avoid non-ASCII prints to the console (cp1252); library code and
  CLI terminal renderers are ASCII-only.

## Verification (Decision layer, spec section 23)

- `domain/verification.py`: `VerifierFit` (self-contained fitted model; its
  identity is the complete-model `scientific_sha256`) and `Verification`
  (decision bound to both fingerprint hashes + `verifier_id`).
- `analysis/verify.py`: pure scoring core (gates, alignment, sigmoid, band
  decision). `verification/spec.py` + `verification/fit.py`: deterministic
  pure-Python IRLS fitting — no NumPy/BLAS/sklearn in the fit or verify path.
- `application/verify.py` (`verify_subjects`), API (`verify_fingerprints`,
  `verify_files`, `fit_verifier`, `load_verifier`), CLI (`verify`, `fit`;
  `verify-fit` and `capabilities` are hidden aliases for `fit` and `info`).
  Fingerprint cache key includes kind+language (v2 format).
- Training manifests are `stylog.verifier-training` TOML (dataset +
  `[verifier]` block + `[[pair]]` with train/tuning/calibration populations);
  `benchmark/train.py` orchestrates fits. The `verification` benchmark task
  computes PAN decision metrics (`benchmark/metrics.py`) under an explicit
  model; the deterministic validation builder is vendored at
  `tests/verifbuild.py` (SHA-256-ranked pair selection, author-disjoint
  population buckets).
- Fit/verify determinism scope: fixed input order + `math.fsum` ⇒
  byte-identical repeated fits within the same recorded runtime; no
  cross-platform byte claim (libm last-ulp).
