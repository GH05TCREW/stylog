# Python API

The public Python API is available after `pip install stylog`. Representations
additionally require `pip install "stylog[ml]"`. Import the package once and
call functions on it:

```python
import stylog
```

Importing `stylog` is light: no optional dependency (spaCy, scikit-learn,
Arrow, and similar) loads until a function that needs it runs. The API covers
six areas: fingerprints, analysis, comparison, baselines and profiles,
representations, and verification. `stylog.__version__` reports the package
version (`"0.1.0"`).

Most functions accept an optional `config` parameter of type
`stylog.config.StylogConfig` (the exceptions are `compare_fingerprints`,
`build_baseline`, `verify_fingerprints`, and `load_verifier`, which take
their inputs fully formed). The default `None` discovers configuration from
the `STYLOG_CONFIG` environment variable, a `stylog.toml` file, or the
`[tool.stylog]` table of `pyproject.toml` in the current directory.

## Reading results

All results are frozen Pydantic v2 models: attribute assignment raises
`ValidationError`, nested collections are tuples rather than lists, and
optional fields are omitted from serialized output instead of carrying null.

Portable, byte-stable output uses canonical JSON (RFC 8785) from the
serialization helpers, not the stock Pydantic methods:

```python
from stylog.serialization.canonical import canonical_bytes, scientific_sha256

digest = scientific_sha256(fp)   # SHA-256 over the canonical bytes
data = canonical_bytes(fp)       # canonical JSON bytes, no trailing newline
```

`model.model_dump()` and `model.model_dump_json()` work but are not canonical
and include `None` fields; do not use them for portable artifacts. Write
result files with `write_json_atomic` and read them back with `read_json`
(see [Serialization utilities](#serialization-utilities)).

## `fingerprint_file()`

Measure one file and return its fingerprint - the portable, deterministic
measurement record for one artifact.

### Signature

```python
stylog.fingerprint_file(
    path: str | Path,
    *,
    kind: str = "auto",
    language: str = "auto",
    config: StylogConfig | None = None,
) -> Fingerprint
```

### Parameters

- `path` - file to measure. Symlinks are rejected.
- `kind` - `"text"`, `"code"`, or `"auto"`. `"auto"` infers from the
  extension: `.py` is Python code; `.js`, `.mjs`, `.cjs`, `.jsx` are
  JavaScript; `.ts`, `.tsx` are TypeScript; `.c` is C; `.rs` is Rust;
  `.txt`, `.md`, `.rst` are text.
- `language` - language tag (for example `"en"`) or `"auto"`. `"auto"` uses
  the extension mapping for code and the configured default (`"und"`) for
  text. An explicit value wins over the mapping.
- `config` - optional `StylogConfig` override.

### Returns

`Fingerprint` - read `artifact` (kind, language, encoding, byte and Unicode
code-point counts, content identity), `features` (tuple of `FeatureObservation`, sorted
by `feature_id`), and `diagnostics`.

### Raises

- `InputError` - the file does not exist, or decoded input contains U+0000.
- `UnsupportedInputError` - the extension cannot be mapped, the input is a
  symlink, or an explicit `kind`/`language` combination is unsupported.
- `DecodeError` - the file does not decode under the resolved encoding.
- `ResourceLimitError` - the file exceeds `input.max_file_bytes`
  (8 MiB by default).

### Example

```python
import stylog

fp = stylog.fingerprint_file("alice_1.txt", language="en")
print(fp.artifact.kind, fp.artifact.language)  # text en
print(len(fp.features))                        # 27

obs = next(o for o in fp.features if o.feature_id == "text.lexical.ttr_casefold")
print(obs.status, round(obs.value.value, 4))   # ok 0.8197
```

### Notes

Fingerprints are cached in a local content-addressed store keyed by content,
kind, language, analysis configuration, analyzer versions, and runtime.
Repeated calls with unchanged inputs return byte-identical results without
re-analysis. `STYLOG_CACHE_DIR` overrides the cache location;
`STYLOG_NO_CACHE=1` disables the cache. Caching never changes results.

## `fingerprint_text()`

Measure an in-memory string. The artifact kind is always `"text"`, the
artifact id is `"text"`, and the encoding is recorded as UTF-8.

### Signature

```python
stylog.fingerprint_text(
    text: str,
    *,
    language: str = "und",
    config: StylogConfig | None = None,
) -> Fingerprint
```

### Raises

- `InputError` - the string contains lone surrogates or U+0000.

### Example

```python
fp = stylog.fingerprint_text("Don't re-enter now.", language="en")
```

## `fingerprint_bytes()`

Measure a byte string with an explicit kind and language. Python code is
decoded per PEP 263; all other input decodes strictly under `encoding` (a
UTF-8 BOM selects `utf-8-sig`).

### Signature

```python
stylog.fingerprint_bytes(
    data: bytes,
    *,
    kind: str,
    language: str,
    encoding: str = "utf-8",
    config: StylogConfig | None = None,
) -> Fingerprint
```

### Raises

- `UnsupportedInputError` - `kind` is not `"text"` or `"code"`, or
  `encoding` names an unknown codec.
- `DecodeError` - the bytes do not decode under the resolved encoding.
- `ResourceLimitError` - `data` exceeds `input.max_file_bytes`.

### Example

```python
fp = stylog.fingerprint_bytes(
    b"def f(x):\n    return x + 1\n", kind="code", language="python"
)
print(fp.artifact.kind, fp.artifact.language)  # code python
```

## `analyze_file()`

Inspect one file and its embedded artifacts. Returns the same fingerprint as
`fingerprint_file`, plus one nested analysis per extracted embedded artifact
(Python docstrings and comment blocks).

### Signature

```python
stylog.analyze_file(
    path: str | Path,
    *,
    kind: str = "auto",
    language: str = "auto",
    config: StylogConfig | None = None,
) -> AnalysisBundle
```

### Returns

`AnalysisBundle` - `primary` is the artifact's `Fingerprint`; `embedded` is a
tuple of `EmbeddedAnalysis` items, each pairing an embedded-artifact
descriptor (kind, source span) with its own fingerprint; `diagnostics`
covers bundle-level facts.

### Raises

Same input errors as `fingerprint_file()`.

### Example

```python
bundle = stylog.analyze_file("app.py")
print(type(bundle.primary).__name__)              # Fingerprint
print([e.descriptor.embedded_kind for e in bundle.embedded])
# ['docstring', 'docstring', 'comment_block']
```

## `analyze_text()`

Inspect an in-memory string and its embedded artifacts.

### Signature

```python
stylog.analyze_text(
    text: str,
    *,
    language: str = "und",
    config: StylogConfig | None = None,
) -> AnalysisBundle
```

### Raises

- `InputError` - the string contains lone surrogates or U+0000.

## `compare_files()`

Fingerprint two files and compare them feature by feature. A comparison
reports one distance per comparable feature; there is no aggregate
similarity score (see
[Comparing fingerprints](methodology.md#comparing-fingerprints)).

### Signature

```python
stylog.compare_files(
    left: str | Path,
    right: str | Path,
    *,
    config: StylogConfig | None = None,
) -> Comparison
```

### Returns

`Comparison` - `left_ref`/`right_ref` are the stringified paths; `families`
groups `ComparisonComponent` records by feature family, each carrying
`feature_id`, `metric`, `value`, `unit`, and the per-side `Support`.

### Raises

Same input errors as `fingerprint_file()`.

### Example

```python
comparison = stylog.compare_files("alice_1.txt", "alice_2.txt")
for family in comparison.families:
    for component in family.components:
        print(component.feature_id, component.metric, round(component.value, 4))
# text.lexical.hapax_token_share_casefold ABS 0.0081
# ...
```

## `compare_fingerprints()`

Compare two fingerprints you already hold.

### Signature

```python
stylog.compare_fingerprints(
    left: Fingerprint,
    right: Fingerprint,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Comparison
```

### Parameters

- `left_ref`, `right_ref` - readability labels copied onto the result; they
  carry no identity semantics.

### Example

```python
fp_a = stylog.fingerprint_file("alice_1.txt")
fp_b = stylog.fingerprint_file("bob_1.txt")
comparison = stylog.compare_fingerprints(fp_a, fp_b, left_ref="alice", right_ref="bob")
```

## `build_baseline()`

Build a local, versioned baseline - a reference distribution over a corpus -
from analyzed fingerprints. Each fingerprint contributes one baseline unit;
only `ok` scalar features (integer, float, ratio) enter the distributions.

### Signature

```python
stylog.build_baseline(
    fingerprints: Sequence[Fingerprint],
    *,
    baseline_id: str,
    baseline_version: str = "1.0.0",
    kind: str = "text",
    language: str = "und",
    domain: str = "general",
    source: str = "local",
) -> Baseline
```

### Raises

- `BaselineError` - `fingerprints` is empty.

### Example

```python
from pathlib import Path

import stylog

from stylog.serialization.jsonio import write_json_atomic

fps = [
    stylog.fingerprint_file(path, language="en")
    for path in sorted(Path("corpus").glob("*.txt"))
]
baseline = stylog.build_baseline(
    fps, baseline_id="my-base", kind="text", language="en", domain="news"
)
write_json_atomic("my-base.stylog-baseline.json", baseline)
```

## `profile_fingerprint()`

Place one fingerprint against one explicit baseline and return the
population-relative interpretation.

### Signature

```python
stylog.profile_fingerprint(
    fingerprint: Fingerprint,
    baseline_ref: str,
    *,
    config: StylogConfig | None = None,
    subject_ref: str = "subject",
) -> Profile
```

### Parameters

- `baseline_ref` - a path (containing a path separator or ending in `.json`)
  or a baseline id; resolution rules are covered in
  [Population baselines](methodology.md#population-baselines).

### Returns

`Profile` - `observations` holds one `ProfileObservation` per profiled
feature with `observed_value`, `baseline_n`, `percentile_midrank`, quartiles,
`iqr`, both MAD variants, and `robust_z` (`None` under the zero-MAD rule;
see [Profiles](methodology.md#profiles)).

### Raises

- `BaselineError` - no baseline matches the ref, the baseline file is
  invalid, or two distinct baselines share the id.

### Example

```python
profile = stylog.profile_fingerprint(fp, "my-base.stylog-baseline.json")
obs = next(o for o in profile.observations
           if o.feature_id == "text.lexical.ttr_casefold")
print(obs.observed_value, obs.baseline_n, obs.percentile_midrank, obs.robust_z)
```

## `fit_representation()`

Fit a sparse representation vocabulary on a corpus and store the fitted
state in the local content-addressed fit store. Requires the `ml` extra.

### Signature

```python
stylog.fit_representation(spec, corpus, *, config: StylogConfig | None = None)
```

### Parameters

- `spec` - one of the four representation identities: the full id
  (`"stylog.representation.char_tfidf/1"`), the id without version suffix,
  the bare kind (`"char_tfidf"`), or the CLI token (`"char-tfidf"`). Kinds:
  `char_ngram_count`, `word_ngram_count`, `char_tfidf`, `word_tfidf`.
- `corpus` - a sequence of runtime artifacts, processed in deterministic
  `artifact_id` order. Build them with the ingest helpers, as below.

### Returns

`RepresentationFit` - the portable fit record. `fit_id` is the content hash
of the fitted state; `state_resource` names the local state file.

### Raises

- `CapabilityUnavailableError` - the `ml` extra is not installed
  (`pip install "stylog[ml]"`).
- `ConfigurationError` - `spec` names an unknown representation.

### Example

```python
from pathlib import Path

import stylog

from stylog.infrastructure.ingest import artifact_from_text

corpus = [
    artifact_from_text(path.read_text(encoding="utf-8"), artifact_id=path.stem)
    for path in sorted(Path("corpus").glob("*.txt"))
]
fit = stylog.fit_representation("char_tfidf", corpus)
print(fit.representation_id)  # stylog.representation.char_tfidf/1
```

### Notes

Fitted state lives under the Stylog user data directory's `fits/` folder;
`STYLOG_FITS_DIR` overrides that location. Fit states are local
content-addressed resources: a `RepresentationFit` is portable, but the
state it references must be provisioned on each machine that transforms with
it (re-run the fit there).

## `transform_representation()`

Transform one subject into a sparse `Representation` under a fitted model.
Requires the `ml` extra.

### Signature

```python
stylog.transform_representation(fit_or_spec, subject, *, config: StylogConfig | None = None)
```

### Parameters

- `fit_or_spec` - a `RepresentationFit` whose state is provisioned in the
  local fit store. Pass the object returned by `fit_representation()` (or one
  read back from disk); a bare specification or representation id string is
  not accepted because Stylog 0.1.0 defines no fit-free representations.
- `subject` - a string of text, or a runtime artifact as produced by the
  ingest helpers.

### Returns

`Representation` - `value` is a `SparseVectorValue` with `dimension` and
`entries` (index/value coordinates, sorted, zeros omitted); `fit_id` and
`backend` record provenance. The portable vector carries no vocabulary terms
or source text.

### Raises

- `CapabilityUnavailableError` - the `ml` extra is not installed.
- `ConfigurationError` - `fit_or_spec` is an unrecognized representation id
  string.
- `ResourceError` - `fit_or_spec` is a bare specification or id string (Stylog
  0.1.0 defines no fit-free representations), or the referenced fit state is
  missing or fails its content-addressed hash check.

### Example

```python
rep = stylog.transform_representation(fit, "Alice walked to the market early.")
print(rep.value.kind, rep.value.dimension, len(rep.value.entries))
# sparse 529 90
```

## `fit_verifier()`

Fit a self-contained pairwise authorship verifier (`VerifierFit`) from
labeled fingerprint pairs. The solver is deterministic pure Python (IRLS
with fixed input order); it uses no NumPy, BLAS, or scikit-learn.

### Signature

```python
stylog.fit_verifier(
    spec: VerifierSpec,
    pairs: Sequence[TrainingPair],
    *,
    calibration_pairs: Sequence[TrainingPair] | None = None,
    tuning_manifest_sha256: str | None = None,
    config: StylogConfig | None = None,
) -> VerifierFit
```

### Parameters

- `spec` - a `stylog.verification.spec.VerifierSpec` naming the fit
  configuration: `kind` (`"text"` or `"code"`), `l2_lambda`,
  `min_support_fraction`, `min_class_support_fraction`, `min_pairs`,
  `threshold_rule` (`"fixed"` with `threshold_fixed`, or
  `"calibration_quantile_band"` with `threshold_alpha`), optional
  `calibration_method="platt"`, `max_iterations`, `tolerance`,
  `include_linguistic`, `allow_unconstrained_language`, `languages`,
  `feature_ids`, and `pair_policy`.
- `pairs` - `stylog.verification.spec.TrainingPair(left, right, label)`
  records with `label` `"same"` or `"different"`.
- `calibration_pairs` - disjoint calibration population; required for
  `"calibration_quantile_band"` thresholds and Platt calibration.
- `tuning_manifest_sha256` - identity of a tuning population used by an
  external, pre-fit selection procedure. The v0.1 fitter records this value
  but does not use tuning observations or labels.

### Returns

`VerifierFit` - the complete fitted model: features with normalization
state, coefficients, intercept, thresholds, optional calibration, fit
configuration, eligibility counts, and provenance. Its scientific identity
is `scientific_sha256(model)`.

### Raises

- `VerifierFitError` - the spec is invalid, the eligible population is below
  `min_pairs`, or the fit fails numerically.

### Example

```python
import stylog

from stylog.verification.spec import TrainingPair, VerifierSpec

labeled_text_pairs = [
    ("the quick brown fox jumps over the lazy dog. " * 12,
     "the quick brown fox jumps over the lazy dog. " * 12 + "again. ", "same"),
    ("pack my box with five dozen liquor jugs tonight. " * 12,
     "pack my box with five dozen liquor jugs tonight. " * 12 + "yes. ", "same"),
    ("the quick brown fox jumps over the lazy dog. " * 12,
     "How vexingly quick daft zebras jump! Bright vixens leap. " * 12, "different"),
    ("pack my box with five dozen liquor jugs tonight. " * 12,
     "Sphinx of black quartz, judge my vow! Hear me now. " * 12, "different"),
]
pairs = [
    TrainingPair(
        left=stylog.fingerprint_text(left_text),
        right=stylog.fingerprint_text(right_text),
        label=label,
    )
    for left_text, right_text, label in labeled_text_pairs
]
spec = VerifierSpec(
    kind="text", l2_lambda=1.0, min_support_fraction=0.9,
    min_class_support_fraction=0.8, min_pairs=2,
    threshold_rule="fixed", threshold_fixed=0.5,
    feature_ids=("text.lexical.ttr_casefold",),
)
model = stylog.fit_verifier(spec, pairs)
```

Real fits need far more than the four toy pairs above, and should come
from a `stylog.verifier-training` manifest with author-disjoint train,
tuning, and calibration populations; see
[Authorship verification](methodology.md#authorship-verification) and the
[specification](spec-v0.1.md).

### Notes

`config` is accepted for signature symmetry and reserved for future
fit-time options; the fit is pure and does not consult it.

## `load_verifier()`

Load a portable `VerifierFit` JSON file and validate it fully.

### Signature

```python
stylog.load_verifier(path: str | Path) -> VerifierFit
```

### Raises

- `PortableArtifactError` - the file is not a single valid portable
  `VerifierFit` JSON object.

## `verify_fingerprints()`

Evaluate two fingerprints under an explicit fitted verifier and return the
model-relative decision.

### Signature

```python
stylog.verify_fingerprints(
    left: Fingerprint,
    right: Fingerprint,
    model: VerifierFit,
    *,
    left_ref: str = "left",
    right_ref: str = "right",
) -> Verification
```

### Returns

`Verification` - `verdict` is `same_author`, `different_author`, or
`abstain`; `abstain_reason` is `uncertain` or `insufficient_evidence`;
`score` is the unitless decision value (absent on `insufficient_evidence`);
`probability` is present only for calibrated models; `features_used` and
`features_missing` record evidence coverage. The decision binds
`left_fingerprint_sha256`, `right_fingerprint_sha256`, and `verifier_id`.

A `same_author` verdict indicates model-relative support for the
same-author hypothesis under the fitted verifier and its training
distribution. It does not establish identity.

### Raises

- `ModelIncompatibilityError` - the model's kind, language scope, feature
  registry version, feature semantic versions, or scientific compatibility
  id do not match the fingerprints.
- `CapabilityUnavailableError` - a model feature is entirely absent from a
  fingerprint (for example `text.linguistic.*` without the `nlp` extra).

### Example

```python
verification = stylog.verify_fingerprints(pairs[0].left, pairs[0].right, model)
print(verification.verdict, round(verification.score, 4))
# same_author 0.5631
```

## `verify_files()`

Fingerprint two files, then verify them under an explicit fitted verifier.
The files' kinds and languages must fall inside the model's scope.

### Signature

```python
stylog.verify_files(
    left: str | Path,
    right: str | Path,
    model: VerifierFit,
    *,
    config: StylogConfig | None = None,
) -> Verification
```

### Raises

Same input errors as `fingerprint_file()`, plus the verification errors of
`verify_fingerprints()`.

### Example

```python
model = stylog.load_verifier("model.json")
verification = stylog.verify_files("alice_1.txt", "alice_2.txt", model)
print(verification.verdict)
```

## Result models

Every result model below is re-exported from the `stylog` top level. The
machine-readable structure of each is defined by the matching file in the
[schemas directory](../schemas/); the [specification](spec-v0.1.md)
is normative.

- `Fingerprint` (`stylog.fingerprint`) - the measurement record for one
  artifact: `artifact`, `runtime`, `analyzers`, `features`, `diagnostics`.
- `AnalysisBundle` (`stylog.analysis`) - `primary` fingerprint plus
  `embedded` analyses of extracted docstrings and comment blocks.
- `FeatureObservation` - one measured feature. `status` discriminates `ok`
  (carries `value` and `support`) from `insufficient_support`,
  `not_applicable`, `unavailable`, `parser_error`, and `disabled` (see
  [Feature status](methodology.md#feature-status)). Feature
  missingness is data, never an exception.
- `Comparison` (`stylog.comparison`) - `families` of per-feature distance
  components for two refs; deliberately no aggregate score.
- `Baseline` (`stylog.baseline`) - versioned reference distributions:
  `baseline_id`, `baseline_version`, `descriptor`, and per-feature sorted
  `values` with `total_units`.
- `Profile` (`stylog.profile`) - population-relative `observations` of one
  subject against one baseline.
- `Representation` (`stylog.representation`) - a sparse model-space vector
  (`value.dimension`, `value.entries`) with backend and resource provenance;
  never a fingerprint (see
  [Sparse representations](methodology.md#sparse-representations)).
- `RepresentationFit` (`stylog.representation-fit`) - the fitted vectorizer
  state reference: `fit_id`, `representation_id`, `state_resource`,
  `backend`.
- `VerifierFit` (`stylog.verifier-fit`) - the self-contained fitted
  verifier: `features`, `coefficients`, `intercept`, `thresholds`, optional
  `calibration`, `fit_config`, `eligibility`, and manifest hashes.
- `Verification` (`stylog.verification`) - the decision record:
  `verdict`, `abstain_reason`, `score`, `probability`, evidence coverage,
  and the three bound hashes.
- `Diagnostic` - a stable machine-readable fact: `code`, `severity`
  (`info`/`warning`/`error`), optional `analyzer_id`, `feature_id`,
  `artifact_id`, and sorted `context` entries.

## Exceptions

All operational errors derive from `StylogError`; import them from the
`stylog` top level. Each carries a `diagnostic_code` and the `exit_code`
used by the CLI.

```text
StylogError
|-- ConfigurationError
|-- CapabilityUnavailableError
|-- InputError
|   |-- DecodeError
|   |-- UnsupportedInputError
|   |-- ResourceLimitError
|-- PortableArtifactError
|-- BaselineError
|-- ResourceError
|-- ModelIncompatibilityError
|-- VerifierFitError
|-- BenchmarkError
|-- InternalStylogError
```

Which operations raise what:

- Input handling (`fingerprint_file`, `fingerprint_bytes`, `analyze_file`,
  `compare_files`, `verify_files`) - `InputError`, `DecodeError`,
  `UnsupportedInputError`, `ResourceLimitError`.
- `fingerprint_text`, `analyze_text` - `InputError` for lone surrogates or
  U+0000.
- `build_baseline`, `profile_fingerprint` - `BaselineError`.
- `fit_representation`, `transform_representation` -
  `CapabilityUnavailableError` (missing `ml` extra), `ConfigurationError`
  (unknown representation id), `ResourceError` (missing or mismatched fit
  state).
- `fit_verifier` - `VerifierFitError`.
- `load_verifier` - `PortableArtifactError`.
- `verify_fingerprints`, `verify_files` - `ModelIncompatibilityError`,
  `CapabilityUnavailableError`.
- Configuration discovery - `ConfigurationError` for an unreadable or
  invalid configuration file.

`BenchmarkError` and `InternalStylogError` do not surface from the functions
above under normal use; they belong to the benchmark subsystem and to
analyzer failure reporting.

## Serialization utilities

`stylog.serialization.jsonio` provides the file helpers used throughout the
documentation and tests. They are stable enough to build on, though they are
not part of the top-level namespace:

```python
from stylog.serialization.jsonio import read_json, write_json_atomic

write_json_atomic("model.json", model)            # canonical bytes + one LF
write_json_atomic("model.json", model, force=True)  # allow overwrite
loaded = read_json("model.json", stylog.VerifierFit)  # parse + full validation
```

`write_json_atomic` writes atomically (temp file plus rename), appends
exactly one trailing newline, and raises `PortableArtifactError` if the
target exists and `force` is not set. `read_json` rejects trailing garbage,
multi-line JSONL input, and explicit JSON null, raising
`PortableArtifactError` on any invalid content. For in-memory canonical
bytes and hashes, use `canonical_bytes` and `scientific_sha256` from
`stylog.serialization.canonical` (see [Reading results](#reading-results)).
