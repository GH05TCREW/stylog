# Stylog specification v0.1

This document is the normative contract for Stylog 0.1.0: the deterministic
stylometry library for text and source code. The key words **MUST**, **MUST
NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** in this document are to be
interpreted as described in RFC 2119. Every other document in this repository
(README.md, docs/methodology.md, docs/cli.md, docs/python-api.md,
docs/limitations.md, docs/getting-started.md) is explanatory only: explanatory
documents describe behavior, they do not legislate it. Where an explanatory
document and this specification disagree, this specification wins. Committed
JSON Schemas in `schemas/` are machine-readable renderings of the portable
models defined in section 5; they are regenerated and drift-checked by
`tools/generate_schemas.py` (section 26.5).

Section references elsewhere in the repository (code docstrings, tests) use
the section numbers of this document.

## 1. Scope and governing rule

Stylog is a local-first scientific measurement library. It turns observable
input samples into portable, deterministic, fully provenanced measurement
artifacts, and it defines exactly one decision-layer task on top of those
measurements: pairwise authorship verification under an explicit fitted model
(section 23).

Governing rule: **Stylog owns scientific semantics and portable scientific
contracts; mature dependencies own commodity mechanics when they can be
wrapped without surrendering scientific meaning.**

Key architectural decisions:

- One repository, one PyPI distribution `stylog`. No companion distributions.
- Text and source-code stylometry are both core. Plain `pip install stylog`
  MUST analyze natural-language text and the source languages python,
  javascript, typescript, c, and rust.
- Python parsing uses CPython `tokenize` + `ast`. The other four languages use
  Tree-sitter (a base dependency).
- Public extras: `stylog[nlp]` (spaCy POS/dependency/morphology),
  `stylog[ml]` (scikit-learn n-gram/TF-IDF representations), `stylog[data]`
  (Arrow/Parquet/Polars/DuckDB/pandas), `stylog[all]` = union. There is no
  `languages` extra.
- NumPy is a declared base dependency; SciPy is a dev/conformance oracle only
  (section 4.8).
- Functional core / imperative shell, with the application (use-case) layer
  shared by the CLI and the Python API.
- Exactly three environmental ports: cache storage, baseline resolution,
  resource resolution (section 4.6). No Repository/UoW/DI framework, no
  plugin ABI.
- Portable scientific artifacts are strict Pydantic v2 models (section 5);
  serialization is a separate module responsibility; domain and analysis code
  never call JCS, file writes, or schema generation.
- No anticipatory abstractions (`NumericBackend`, `CorpusStore`,
  `AnalyzerFactory`, and similar) exist or may be introduced
  (section 4.15).
- Architecture boundaries are mechanically tested (section 4.15).

## 2. Terminology

- `Artifact` - one observable input sample (a file, stdin, or an in-memory
  value via the API).
- `ContentIdentity` - identity of exact input bytes (never a person).
- `FeatureDefinition` - versioned scientific definition of one measurement.
- `FeatureObservation` - the result of applying one feature to one artifact.
- `Fingerprint` - portable deterministic observations for one primary
  artifact.
- `AnalysisBundle` - primary fingerprint plus embedded-artifact analyses and
  diagnostics.
- `EvidenceSet` - explicit collection of distinct artifacts with declared
  linkage.
- `EvidenceAggregate` - reducer output over an evidence set.
- `Baseline` - local, versioned reference distribution.
- `Profile` - population-relative interpretation of one subject against one
  explicit baseline.
- `Comparison` - descriptive per-feature distances; no global score.
- `Representation` - portable sparse/dense model-space vector with
  provenance. A Representation is NOT a Fingerprint and MUST never be
  serialized inside `Fingerprint.features`.
- `VerifierFit` - fully self-contained fitted pairwise authorship verifier
  (features with normalization state, coefficients, thresholds, optional
  Platt calibration, fit configuration, train/tuning/calibration manifest
  identities, runtime, backend). Its identity is the `scientific_sha256` of
  the complete model.
- `Verification` - model-relative decision (`same_author` |
  `different_author` | `abstain`) bound to both input fingerprint hashes and
  the complete model hash; never an identity claim.
- `Decision score` - sigmoid of the fitted linear combination of standardized
  feature distances; unitless model output, NOT a probability; exists only
  when a complete evidence vector exists (score != probability; no score
  without complete evidence).
- `Calibrated probability` - present only when the model carries explicit
  Platt calibration state; the only field interpretable as an estimated
  same-author probability, and only under the model's stated kind/language
  assumptions.
- `abstain` - typed non-decision: `uncertain` (score inside the threshold
  band) or `insufficient_evidence` (incomplete evidence; no score emitted).
- `Diagnostic` - stable machine-readable warning/error fact (stable codes, no
  human prose).
- `ResourceSignature` - version and hash of a non-code resource.
- `RuntimeSignature` - runtime facts relevant to deterministic extraction.
- `ParserGrammarSignature` / `ModelSignature` - analogous signatures for
  parser grammars and provisioned models.
- `ScientificCompatibilityId` - a Stylog-issued identity certifying that two
  runs are scientifically comparable.

## 3. Scientific separation and non-goals

### 3.1 Scientific separation (inviolable)

```text
Fingerprint      deterministic measurement
Profile          population-relative interpretation
Comparison       descriptive relationship
Representation   sparse or learned model space
Decision         task-specific inference (v0.1: pairwise authorship
                 verification via explicit fitted VerifierFit models only)
Identity claim   outside Stylog v0.1 (permanently out of scope)
```

Each layer consumes only the layers above it in this list. Measurement never
contains interpretation; interpretation never contains decision; the decision
layer never manufactures measurement fields.

### 3.2 Non-goals

Stylog v0.1 MUST NOT implement: author identity claims; attribution;
retrieval/ranking; machine-vs-human probability; a context-free overall
similarity score; uncalibrated probability claims; automatic source
formatting; LLM features; hidden remote access; automatic model or dataset
downloads; rewriting/anonymization; a public plugin ABI; automatic language
detection; universal baselines; a web UI; distributed execution; transformer
representations; any default or bundled verifier model. Verification verdicts
and calibrated probabilities are permitted ONLY via explicit fitted
`VerifierFit` models (sections 5.20-5.21, 12.11, 23); there is no implicit or
default model. No pre-fitted verifier model ships with the package.

## 4. Architecture

### 4.1 Layers

```text
portable domain
  -> pure scientific analysis (text/code/aggregate/metrics/profile/verify)
  -> parser mechanics (python native, tree-sitter)
  -> application use cases (fingerprint, analyze, aggregate, compare,
     profile, represent, verify, benchmark)
  -> API + CLI
  -> bootstrap
  -> infrastructure adapters (cache/baselines/resources/files/execution)
```

### 4.2 Layering rules

- `domain` MUST NOT import Typer, Tree-sitter, spaCy, scikit-learn, Arrow,
  Polars, DuckDB, pandas, NumPy, filesystem-cache implementations, or CLI
  code.
- Portable domain field types MUST NOT be NumPy arrays, dataframes, Arrow
  tables, sparse matrices, Tree-sitter nodes, spaCy objects, or torch
  tensors.
- Scientific analysis MUST NOT depend on CLI or report rendering, and MUST
  NOT perform implicit network access. (`analysis/verify.py` and
  `analysis/compat.py` share one accepted exception: they import the
  canonical hashing helper from `serialization`.)
- Infrastructure implements the ports; the ports do not import concrete
  infrastructure.
- Representations MAY consume artifacts/fingerprints but MUST emit
  `Representation` objects and MUST never mutate fingerprints.
- Benchmark protocol semantics are independent of dataframe, ML, and database
  mechanics.
- The `verification/` fit stack is pure Python over the standard library: no
  NumPy, no BLAS, no sklearn at module level (section 23.10).

### 4.3 Functional core and imperative shell

The functional core is pure transformations: text -> tokens -> measurements,
parser facts -> measurements, observations -> aggregate, observations ->
comparison, observation + baseline -> profile, model + evidence pair ->
verification. The imperative shell owns: file/stdin reading, decoding and
content identity, config discovery, cache lookup/write, baseline and resource
resolution, model and grammar validation, process scheduling, serialization
and output, and CLI rendering. Scientific functions MUST NOT know cache
directories, the current working directory, output filenames, terminal
colors, or Typer contexts.

### 4.4 Source layout

Responsibilities, not a section mirror:

```text
src/stylog/
  __init__.py  api.py  cli.py  bootstrap.py  ports.py  exceptions.py
  config.py  capability.py  runtime.py
  domain/         artifact.py baseline.py benchmark.py diagnostic.py
                  evidence.py feature.py fingerprint.py interpretation.py
                  provenance.py representation.py verification.py _base.py
  analysis/       engine.py base.py build.py text.py code.py python.py
                  linguistic.py treesitter.py aggregate.py compare.py
                  profile.py stats.py compat.py verify.py registry.py
                  identifiers.py lines.py whitespace.py
  parsers/        python_native.py tree_sitter.py
  application/    fingerprint.py analyze.py aggregate.py batch.py
                  compare.py profile.py verify.py
  representations/ spec.py fit.py
  data/           arrow.py parquet.py frames.py duckdb.py
  benchmark/      manifest.py split.py evaluate.py metrics.py train.py api.py
  verification/   spec.py fit.py
  infrastructure/ ingest.py files.py cache.py baselines.py resources.py
                  execution.py paths.py
  serialization/  canonical.py jsonio.py
  resources/      function_words_en_v1.txt  grammar_manifest.json
                  tree_sitter_mappings/*.json
```

### 4.5 Parser dispatch

Parser dispatch is a simple dict: `python` -> the CPython native parser;
`javascript`, `typescript`, `c`, `rust` -> the Tree-sitter parser.

### 4.6 Ports

Exactly these three ports exist. Nothing else is injected.

```python
class CacheStore(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, canonical_bytes: bytes) -> None: ...

class BaselineResolver(Protocol):
    def resolve(self, baseline_ref: str) -> Baseline: ...

class ResourceResolver(Protocol):
    def resolve(self, request: ResourceRequest) -> ResolvedResource: ...
```

`ResourceRequest` and `ResolvedResource` are small frozen dataclasses;
`ResolvedResource.local_path` is never portable. There is no
`ArtifactReader` port.

### 4.7 Service grouping

A small internal `RuntimeServices` frozen dataclass MAY group the three
ports. It is not a DI framework, and no global singleton exists (section
20.4).

### 4.8 Numeric policy

NumPy is a declared base dependency (pinned in section 4.10) and MAY be used
for optimized numeric kernels; in v0.1 its only production use is inside the
`ml` representation path. The semantic reference for every statistic
(quantiles, midranks, MAD, JSD2, W1, AUC) is the pure-Python implementation
in `analysis/stats.py` using `math.fsum`; any optimized kernel MUST match the
reference exactly and MUST NOT change stored integer widths. SciPy is a
dev-only differential oracle and MUST NOT be imported by shipped code.

### 4.9 Packaging

One wheel and one sdist named `stylog`, built from `src/stylog`. The sdist
also carries `docs/`, `schemas/`, `README.md`, `pyproject.toml`, and
`LICENSE`. Dev generators live in `tools/` and are not shipped.

### 4.10 Base dependencies

```toml
requires-python = ">=3.12,<3.15"
dependencies = [
  "pydantic>=2.13,<3",
  "typer>=0.27,<0.28",
  "rfc8785==0.1.4",
  "platformdirs>=4.11,<5",
  "numpy>=2.5,<3",
  "tree-sitter>=0.26,<0.27",
  "tree-sitter-javascript>=0.25,<0.26",
  "tree-sitter-typescript>=0.23.2,<0.24",
  "tree-sitter-c>=0.24,<0.25",
  "tree-sitter-rust>=0.24,<0.25",
]
```

### 4.11 Optional dependencies

```toml
nlp  = ["spacy>=3.8,<4", "click>=8,<9"]
ml   = ["scikit-learn>=1.9,<2"]
data = ["pyarrow>=25,<26", "polars>=1,<2", "duckdb>=1,<2", "pandas>=3,<4"]
all  = the union of nlp, ml, and data
dev  = ["pytest>=9", "scipy", "ruff", "build"]
```

`click` is pinned explicitly in `nlp` (and therefore `all`) because spaCy
3.8.x imports it without declaring it while Typer >= 0.27 no longer depends
on it; without the pin a clean `stylog[nlp]` install cannot `import spacy`.

### 4.12 Capability model

The optional extras are capabilities: `nlp`, `ml`, `data`. Using a capability
that is not installed MUST raise `CapabilityUnavailableError` (exit 2) or
produce typed feature unavailability - never an accidental `ImportError`.
Configuration blocks for unavailable capabilities are rejected the same way
(section 16.4).

### 4.13 Tree-sitter grammar loading

Tree-sitter is a base dependency, but grammar modules load lazily: the first
code analysis of a given language imports its grammar package, and grammars
are cached per process. The base install MUST prove that javascript,
typescript, c, and rust analysis works with no extras installed (section
26.4).

### 4.14 Lazy optional imports

`import stylog` MUST NOT import spaCy, scikit-learn, SciPy, PyArrow, Polars,
DuckDB, pandas, torch, transformers, or any tree-sitter grammar module.
Optional heavy dependencies are imported inside functions only.

### 4.15 Architecture enforcement

The architecture test suite (`tests/test_architecture.py`) MUST fail if:

- `domain` imports CLI, infrastructure, application, or any heavy/optional
  library (Typer, Tree-sitter, spaCy, sklearn, SciPy, Arrow, Polars, DuckDB,
  pandas, NumPy, torch, transformers);
- the core deterministic text science imports spaCy/Stanza/HF machinery;
- `analysis` or `verification` imports CLI/report rendering or any
  heavy/optional library at module level (spaCy is function-level lazy in
  `analysis/linguistic.py`);
- `import stylog` or `import stylog.api` transitively loads optional heavy
  libraries or tree-sitter;
- importing the verification stack loads any optional library;
- a forbidden abstraction (section 1: `NumericBackend`, `CorpusStore`,
  `AnalyzerFactory`, `BackendManager`, `PluginManager`, `ProviderRegistry`,
  `ServiceLocator`, `PipelineBuilder`, `UnitOfWork`, `MessageBus`) is defined
  anywhere in the package.

## 5. Typed domain contracts (all Pydantic v2)

### 5.1 Common portable-model rules

Every portable Stylog object is a strict Pydantic v2 model derived from the
shared `PortableModel` base with:

```python
model_config = ConfigDict(extra="forbid", frozen=True, strict=True,
                          allow_inf_nan=False)
```

Normative rules, all mechanically enforced at validation and at
serialization:

- Unknown fields are rejected (`extra="forbid"`); models are immutable
  (`frozen=True`); strict types are required (`strict=True`); NaN and
  infinity are rejected (`allow_inf_nan=False`).
- **Portable JSON MUST NOT contain `null`.** Absence is expressed as a
  discriminated status variant, an omitted optional non-null property, or a
  separate identity variant. Parsing an explicit `null` anywhere in a
  portable artifact MUST fail validation.
- Optional fields are declared `T | None = None` and are omitted from
  serialized output when `None` (`model_dump(mode="json", exclude_none=True)`).
  Callers MUST NOT pass explicit `None` into portable model constructors: the
  null-rejection validator raises. Invariants that reference an optional
  field ("present iff ...") are stated over serialized presence.
- Nested portable containers MUST be immutable tuples or frozen models,
  never lists. A JSON array in parsed input is accepted at the validation
  boundary and stored as a tuple.
- String-valued enums (for example `ArtifactKind`, `FeatureStatus`) are
  serialized as their string values and accepted in that form when parsing.

### 5.2 Primitive restrictions

Portable values are JSON-compatible. Additionally: no lone UTF-16 surrogates
(in strings or object keys); no NaN or infinite floats; `-0.0` is normalized
to `0.0`; integers MUST NOT exceed the JCS safe range (magnitude at most
2^53 - 1); tuples internally, sorted where order has scientific meaning.

### 5.3 Artifact

```python
class ArtifactKind(StrEnum): TEXT = "text"; CODE = "code"
class ContentIdentitySha256(PortableModel):
    mode: Literal["sha256"]; sha256: HexDigest64
class ContentIdentitySuppressed(PortableModel):
    mode: Literal["suppressed"]
ContentIdentity = Annotated[ContentIdentitySha256 | ContentIdentitySuppressed,
                            Field(discriminator="mode")]
class ArtifactDescriptor(PortableModel):
    artifact_id: str; kind: ArtifactKind; language: str; encoding: str
    byte_count: int; character_count: int; content_identity: ContentIdentity
```

`language` is always a string; an unknown natural language is `und`; Python
source is `"python"`. `HexDigest64` is a lowercase 64-character hex string.

### 5.4 Source span

```python
class SourcePosition(PortableModel):
    line: int     # 1-based
    column: int   # 0-based Unicode code-point column
class SourceSpan(PortableModel):
    start: SourcePosition; end: SourcePosition  # end exclusive
```

Python AST columns are UTF-8 byte offsets - they MUST be converted to
code-point columns before entering a `SourceSpan`.

### 5.5 Analyzer provenance

```python
class ResourceSignature(PortableModel):
    id: str; version: str; sha256: HexDigest64
class RuntimeSignature(PortableModel):
    python_implementation: str; python_version: str
    python_cache_tag: str; unicode_database_version: str
class AnalyzerSignature(PortableModel):
    analyzer_id: str; implementation_version: str
    feature_registry_version: str
    backend: BackendSignature
    resources: tuple[ResourceSignature, ...]  # sorted by id
```

The runtime signature is `platform.python_implementation()`,
`platform.python_version()`, `sys.implementation.cache_tag` (empty string
when unavailable), and `unicodedata.unidata_version`.

### 5.6 Diagnostics

```python
class DiagnosticSeverity(StrEnum):
    INFO = "info"; WARNING = "warning"; ERROR = "error"
class DiagnosticContextEntry(PortableModel):
    key: str; value: str
class Diagnostic(PortableModel):
    code: str; severity: DiagnosticSeverity
    analyzer_id: str | None = None   # omitted when absent
    feature_id: str | None = None    # omitted when absent
    artifact_id: str | None = None   # omitted when absent
    context: tuple[DiagnosticContextEntry, ...] = ()  # sorted by unique key
```

Diagnostics carry stable machine-readable codes (section 10.4), never human
prose. The canonical diagnostic order is defined in section 14.7.

### 5.7 Typed feature values

```python
class IntegerValue(PortableModel): kind: Literal["integer"]; value: int
class FloatValue(PortableModel):   kind: Literal["float"]; value: float
class RatioValue(PortableModel):
    kind: Literal["ratio"]
    numerator: int; denominator: int; multiplier: float; value: float
    # invariant: denominator > 0;
    # value == (numerator / denominator) * multiplier within 1e-12
    # multiplier 1.0 = proportion; 1000.0 only for per_1000_* registry features
class CategoryCount(PortableModel): key: str; count: int  # count >= 0
class CategoricalDistributionValue(PortableModel):
    kind: Literal["categorical_distribution"]
    counts: tuple[CategoryCount, ...]; total: int
    # counts sorted by unique key; zero-count categories omitted;
    # total > 0; sum(counts) == total
class PointCount(PortableModel): point: int; count: int  # point >= 0, count > 0
class OrderedHistogramValue(PortableModel):
    kind: Literal["ordered_histogram"]
    points: tuple[PointCount, ...]; total: int; top_code: int
    # observed' = min(observed, top_code); points sorted and unique;
    # 0 <= point <= top_code; total > 0; sum(counts) == total
class SummaryStatisticsValue(PortableModel):
    kind: Literal["summary"]
    n: int; minimum: float; q25: float; median: float; q75: float
    maximum: float; mean: float
    sample_sd: float | None = None  # present iff n >= 2
    # n >= 1; quantiles are Hyndman-Fan type 7; sample sd denominator n - 1
```

`FeatureValue` is the discriminated union of these six kinds (discriminator
`kind`).

### 5.8 Support

```python
class Support(PortableModel): kind: str; count: int  # count >= 0
```

Support kinds are registry-pinned strings describing the event population
behind one observation (for example `artifact`, `character`, `word`,
`lexical token`, `line break`, `sentence`, `paragraph`, `nonblank line`,
`physical line`, `blank run`, `token`, `string token`,
`identifier occurrence`, `binding event`, `attribute occurrence`,
`ast node`, `ast node edge`, `function`, `comment`, `docstring`,
`named node`, `named-node edge`, `linguistic token`). The support kind of
every feature is fixed by the registry tables in section 6.

### 5.9 Feature observation variants

`FeatureObservation` is a discriminated union on `status`:
`OkFeatureObservation` (`status="ok"`, carrying `value` and `support`),
`InsufficientSupportObservation`, `NotApplicableObservation`,
`UnavailableObservation`, `ParserErrorObservation`, `DisabledObservation`.
Every variant carries `feature_id`, `semantic_version`, `analyzer_id`, and
`analyzer_implementation_version`. Non-`ok` variants MUST NOT contain a
`value` field.

### 5.10 Fingerprint

```python
class Fingerprint(PortableModel):
    schema: Literal["stylog.fingerprint"]; schema_version: Literal["0.1.0"]
    artifact: ArtifactDescriptor
    runtime: RuntimeSignature
    analysis_config_sha256: HexDigest64
    analyzers: tuple[AnalyzerSignature, ...]   # sorted by analyzer_id
    features: tuple[FeatureObservation, ...]   # sorted by feature_id
    diagnostics: tuple[Diagnostic, ...]        # canonical order (section 14.7)
```

### 5.11 Embedded artifact analyses

```python
class EmbeddedArtifactDescriptor(PortableModel):
    artifact: ArtifactDescriptor; parent_artifact_id: str
    embedded_kind: Literal["comment_block", "inline_comment", "docstring"]
    ordinal: int; source_span: SourceSpan
    docstring_owner: str | None = None  # present only for docstrings
class EmbeddedAnalysis(PortableModel):
    descriptor: EmbeddedArtifactDescriptor; fingerprint: Fingerprint
```

### 5.12 Analysis bundle

```python
class AnalysisBundle(PortableModel):
    schema: Literal["stylog.analysis"]; schema_version: Literal["0.1.0"]
    primary: Fingerprint
    embedded: tuple[EmbeddedAnalysis, ...]
    diagnostics: tuple[Diagnostic, ...]
    # embedded sorted by (source_span.start.line, source_span.start.column,
    # embedded_kind, ordinal); diagnostics in canonical order
```

### 5.13 Evidence set

```python
class EvidenceMember(PortableModel): member_id: str; artifact_id: str
class LinkageDescriptor(PortableModel): kind: str; source: str
class EvidenceSet(PortableModel):
    schema: Literal["stylog.evidence-set"]; schema_version: Literal["0.1.0"]
    evidence_set_id: str
    members: tuple[EvidenceMember, ...]   # sorted by unique member_id
    linkage: LinkageDescriptor
```

### 5.14 Evidence aggregate

```python
class MissingStatusCount(PortableModel):
    status: FeatureStatus; count: int     # count >= 0
class AggregateObservation(PortableModel):
    feature_id: str; semantic_version: str; reducer: AggregationKind
    total_samples: int; contributing_samples: int
    missing: tuple[MissingStatusCount, ...]  # sorted by status
    pooled: FeatureValue | None = None        # omitted when the reducer has none
    sample_summary: SummaryStatisticsValue | None = None  # omitted when not meaningful
    sample_values: tuple[float, ...] | None = None        # sample-summary reducers only
class EvidenceAggregate(PortableModel):
    schema: Literal["stylog.evidence-aggregate"]; schema_version: Literal["0.1.0"]
    evidence_set: EvidenceSet
    aggregates: tuple[AggregateObservation, ...]  # sorted by feature_id
    diagnostics: tuple[Diagnostic, ...]
```

Invariants: `contributing_samples + sum(missing counts) == total_samples`;
when `contributing_samples == 0`, `pooled` and `sample_summary` are omitted.
`AggregationKind` is `exact_sum | ratio_pool | categorical_pool |
histogram_pool | sample_summary | not_aggregatable` (section 11.1).

### 5.15 Comparison

```python
class ComparisonComponent(PortableModel):
    feature_id: str; semantic_version: str; metric: str; value: float
    unit: str; left_support: Support; right_support: Support
class ComparisonFamily(PortableModel):
    family: str
    components: tuple[ComparisonComponent, ...]  # sorted by feature_id
class Comparison(PortableModel):
    schema: Literal["stylog.comparison"]; schema_version: Literal["0.1.0"]
    left_ref: str; right_ref: str
    families: tuple[ComparisonFamily, ...]       # sorted by family name
    diagnostics: tuple[Diagnostic, ...]
```

A Comparison is an ordered set of independently interpretable components.
There is deliberately NO aggregate similarity field, and none may be added
(section 12.1).

### 5.16 Profile

```python
class ProfileObservation(PortableModel):
    feature_id: str; feature_semantic_version: str; baseline_n: int
    observed_value: float; percentile_midrank: float
    median: float; q25: float; q75: float; iqr: float
    mad_raw: float; mad_normal_scaled: float
    robust_z: float | None = None  # omitted under the zero-MAD rule (13.6)
class Profile(PortableModel):
    schema: Literal["stylog.profile"]; schema_version: Literal["0.1.0"]
    subject_ref: str; baseline_id: str; baseline_version: str
    observations: tuple[ProfileObservation, ...]  # sorted by feature_id
    diagnostics: tuple[Diagnostic, ...]
```

`baseline_n >= 1` for every emitted observation (section 13.7).

### 5.17 Dependency-aware backend provenance

```python
class PackageProvenance(PortableModel): package: str; version: str
class ParserGrammarSignature(PortableModel):
    language: str; grammar_id: str; grammar_version: str
    grammar_revision: str; node_types_sha256: HexDigest64
    grammar_manifest_sha256: HexDigest64; language_abi_version: int
class ModelSignature(PortableModel):
    model_id: str; model_revision: str; model_tree_sha256: HexDigest64
    tokenizer_id: str; tokenizer_version: str; tokenizer_tree_sha256: HexDigest64
    preprocessing_id: str; preprocessing_version: str
class BackendSignature(PortableModel):
    backend_id: str; implementation_version: str
    scientific_compatibility_id: str
    packages: tuple[PackageProvenance, ...]      # sorted by package name
    resources: tuple[ResourceSignature, ...]     # sorted by id
    parser_grammar: ParserGrammarSignature | None = None  # present iff parser-backed
    model: ModelSignature | None = None                   # present iff model-backed
```

Package provenance records what executed; `scientific_compatibility_id`
records whether differences matter. The compatibility gate MUST NOT compare
every package version blindly: packages that cannot affect a measurement
(pydantic, typer, platformdirs) MUST NOT be copied into every fingerprint.
Backend signatures in v0.1: `stylog.native.text` (compat
`stylog.text-core/1`), `stylog.native.code` (`stylog.code-surface/1`),
`cpython.tokenize` (`stylog.python-native-tokenize/1`), `cpython.ast`
(`stylog.python-native-ast/1`), `tree-sitter` (per-language
`stylog.tree-sitter.<language>/1` with a `parser_grammar` signature), the
spaCy backend (`model` signature, section 6.14), `scikit-learn`
(`stylog.sklearn-vectorizers/1`, section 20.7), and the verifier backend
(`stylog.verifier.logreg/1`, section 23.10).

### 5.18 Representation

```python
class SparseCoordinate(PortableModel): index: int; value: float
class SparseVectorValue(PortableModel):
    kind: Literal["sparse"]; dimension: int
    entries: tuple[SparseCoordinate, ...]  # sorted; zeros omitted
class DenseVectorValue(PortableModel):
    kind: Literal["dense"]; values: tuple[float, ...]
RepresentationValue = Annotated[SparseVectorValue | DenseVectorValue,
                                Field(discriminator="kind")]
class RepresentationResourceSignature(PortableModel):
    resource_id: str; resource_version: str; sha256: HexDigest64
class Representation(PortableModel):
    schema: Literal["stylog.representation"]; schema_version: Literal["0.1.0"]
    subject_ref: str; representation_id: str; semantic_version: str
    preprocessing_version: str
    fit_id: str | None = None  # omitted for fit-free representations
    backend: BackendSignature
    resources: tuple[RepresentationResourceSignature, ...]  # sorted by resource_id
    value: RepresentationValue
    diagnostics: tuple[Diagnostic, ...]
```

Sparse indices are nonnegative, unique, sorted, and below `dimension`. A
portable representation MUST NOT contain raw source strings.

### 5.19 Representation fit

```python
class RepresentationFit(PortableModel):
    schema: Literal["stylog.representation-fit"]
    schema_version: Literal["0.1.0"]
    fit_id: str; representation_id: str; representation_semantic_version: str
    source_manifest_sha256: HexDigest64; fit_config_sha256: HexDigest64
    state_resource: RepresentationResourceSignature
    backend: BackendSignature
```

### 5.20 VerifierFit portable contract

Fully self-contained; no sidecar.

```python
class VerifierFeature(PortableModel):
    feature_id: str; semantic_version: str; metric: str
    mean: float; scale: float            # scale > 0; training normalization state
class VerifierThresholds(PortableModel):
    t_same: float; t_diff: float         # 0 < t_diff <= t_same < 1
class VerifierCalibration(PortableModel):
    method: Literal["platt"]; a: float; b: float
class VerifierPairPolicy(PortableModel):
    max_pairs_per_author: int | None      # omitted = uncapped
    max_pairs_per_problem: int | None     # omitted = uncapped
    negative_positive_ratio: float | None # omitted = no class-ratio cap
    selection_version: str
class VerifierFitConfig(PortableModel):
    l2_lambda: float; max_iterations: int; tolerance: float
    min_support_fraction: float; min_class_support_fraction: float
    min_pairs: int
    threshold_rule: str                  # calibration_quantile_band | fixed
    threshold_alpha: float | None        # required iff calibration_quantile_band, in (0, 0.5]
    threshold_fixed: float | None        # required iff fixed, in (0, 1)
    calibration_method: str | None       # "platt"; omitted = uncalibrated
    include_linguistic: bool; allow_unconstrained_language: bool
    feature_ids: tuple[str, ...] | None  # explicit ablation subset, sorted
    pair_policy: VerifierPairPolicy
class VerifierEligibility(PortableModel):
    training_pair_count: int; eligible_pair_count: int
    candidate_feature_count: int; selected_feature_count: int
class VerifierFit(PortableModel):
    schema: Literal["stylog.verifier-fit"]; schema_version: Literal["0.1.0"]
    model_id: str; model_semantic_version: str; task: str; task_version: str
    kind: ArtifactKind; languages: tuple[str, ...]  # sorted; empty = unconstrained (opt-in)
    feature_registry_version: str
    features: tuple[VerifierFeature, ...]  # sorted by feature_id
    coefficients: tuple[float, ...]; intercept: float  # len(coefficients) == len(features)
    thresholds: VerifierThresholds; threshold_rule: str
    calibration: VerifierCalibration | None  # omitted = uncalibrated
    fit_config: VerifierFitConfig; eligibility: VerifierEligibility
    source_manifest_sha256: HexDigest64
    tuning_manifest_sha256: HexDigest64 | None      # optional external-tuning provenance
    calibration_manifest_sha256: HexDigest64 | None
    runtime: RuntimeSignature; backend: BackendSignature
```

All fitted state is embedded in this ONE canonical artifact - a logistic
verifier's state is a few dozen floats, so no sidecar or state store exists
for verifiers. `verifier_id = scientific_sha256(VerifierFit)` is the
complete-model identity: coefficients, thresholds, calibration, eligibility,
pair policy, train/tuning/calibration manifest hashes, runtime, and semantics
all live under one hash. Validator invariants: `threshold_rule` matches
`fit_config.threshold_rule`; `calibration` is present iff
`fit_config.calibration_method` is present (and the methods match);
`calibration_manifest_sha256` is present iff thresholds or calibration were
fitted on a calibration split; empty `languages` requires
`fit_config.allow_unconstrained_language = true`. In v0.1 `task` is
`pairwise_authorship_verification` and `task_version` is `"1"`.

### 5.21 Verification portable contract

```python
class Verification(PortableModel):
    schema: Literal["stylog.verification"]; schema_version: Literal["0.1.0"]
    left_ref: str; right_ref: str        # readability labels only, NOT identity
    left_fingerprint_sha256: HexDigest64   # scientific hash of the left measurement
    right_fingerprint_sha256: HexDigest64
    verifier_id: HexDigest64               # scientific hash of the complete VerifierFit
    model_id: str; model_semantic_version: str
    verdict: Literal["same_author", "different_author", "abstain"]
    abstain_reason: Literal["uncertain", "insufficient_evidence"] | None
    score: float | None                  # absent exactly on abstain/insufficient_evidence
    probability: float | None            # present iff score present AND model calibrated
    calibration_method: Literal["platt"] | None  # present iff probability present
    features_used: int
    features_missing: tuple[str, ...]      # sorted
    diagnostics: tuple[Diagnostic, ...]
```

Validator invariants: `abstain_reason` present iff `verdict == "abstain"`;
`score` absent exactly for `abstain`/`insufficient_evidence`, else in the
open interval (0, 1); `probability` present iff `calibration_method` present;
`probability` requires `score` and is in (0, 1); `features_missing` nonempty
implies `abstain`/`insufficient_evidence`; both fingerprint hashes are always
present. Decision identity = `scientific_sha256(Verification)`: exact
measured A + exact measured B + exact fitted verifier + resulting decision.

### 5.22 Corpus record adapter contract

Every columnar row representing a portable object retains at least `schema`,
`schema_version`, `scientific_sha256`, and `canonical_json` (the canonical
bytes, or a lossless binary encoding of them). Extra flattened columns MAY be
materialized. Portable identity is never surrendered by an adapter (section
20.8).

### 5.23 Schema inventory

Every portable model serializes `schema` (its identifier) and
`schema_version` (`"0.1.0"` for all v0.1 models). Committed JSON Schemas are
generated from these models by `tools/generate_schemas.py` and gated by
`--check` (section 26.5):

| Schema id | Version | Purpose | Generating model |
|---|---|---|---|
| `stylog.fingerprint` | 0.1.0 | Deterministic measurements for one primary artifact | `Fingerprint` (domain/fingerprint.py) |
| `stylog.analysis` | 0.1.0 | Primary fingerprint + embedded analyses + diagnostics | `AnalysisBundle` (domain/fingerprint.py) |
| `stylog.evidence-set` | 0.1.0 | Declared collection of distinct artifacts with linkage | `EvidenceSet` (domain/evidence.py) |
| `stylog.evidence-aggregate` | 0.1.0 | Reducer output over an evidence set | `EvidenceAggregate` (domain/evidence.py) |
| `stylog.comparison` | 0.1.0 | Descriptive per-feature distances | `Comparison` (domain/interpretation.py) |
| `stylog.profile` | 0.1.0 | Population-relative interpretation vs one baseline | `Profile` (domain/interpretation.py) |
| `stylog.baseline` | 0.1.0 | Local versioned reference distribution | `Baseline` (domain/baseline.py) |
| `stylog.representation` | 0.1.0 | Sparse/dense model-space vector with provenance | `Representation` (domain/representation.py) |
| `stylog.representation-fit` | 0.1.0 | Fitted representation state identity and provenance | `RepresentationFit` (domain/representation.py) |
| `stylog.benchmark-result` | 0.1.0 | Benchmark task output (descriptive + decision metrics) | `BenchmarkResult` (domain/benchmark.py) |
| `stylog.verifier-fit` | 0.1.0 | Self-contained fitted pairwise verifier | `VerifierFit` (domain/verification.py) |
| `stylog.verification` | 0.1.0 | Model-relative pairwise decision | `Verification` (domain/verification.py) |

Two further versioned TOML manifests exist but are configuration/benchmark
formats, not portable JSON models: `stylog.dataset` and
`stylog.verifier-training` (section 21), plus benchmark spec files
`stylog.benchmark` (section 21.1); all carry `schema_version = "0.1.0"`.

## 6. Feature registry

### 6.1 Registry identity and semantics

Registry version `stylog.features/1.0.0`; every feature has
`semantic_version = "1.0.0"`. The machine-readable single source of truth for
feature IDs, owners, geometry, reducers, metrics, support kinds, top-codes,
confound tags, and resource bindings is `src/stylog/analysis/registry.py`;
the tables below are the normative rendering of that registry.

Reducers: `exact_sum` (SUM) exact integer sum; `ratio_pool` (RATIO) pool
numerator + denominator, then recompute; `categorical_pool` (CAT) union
categories, summing counts; `histogram_pool` (HIST) sum counts over identical
transformed points; `sample_summary` (SAMPLE) no pooled value, sample-value
summary only; `not_aggregatable` (NONE) not aggregatable.

Metrics: `ABS` |a - b|; `SPD` 0 when a == b == 0 else 2|a - b|/(|a| + |b|),
range [0, 2]; `JSD2` Jensen-Shannon distance, base 2; `W1` Wasserstein-1 on
top-coded integer support; `NONE` no direct comparison.

`insufficient_support` occurs ONLY when the mathematical denominator or
sample set is empty (section 10.1).

Analyzer identifiers: `stylog.text.sample`, `stylog.text.surface`,
`stylog.text.lexical`, `stylog.text.structure`,
`stylog.text.function_words.en`, `stylog.text.linguistic`,
`stylog.code.sample`, `stylog.code.surface`, `stylog.code.python.tokens`,
`stylog.code.python.ast`, `stylog.code.tree_sitter`. Every analyzer's
implementation version in v0.1 is `"1.0.0"`.

### 6.2 Text sample and surface

Owner: `stylog.text.sample` (sample), `stylog.text.surface` (surface).

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| text.sample.byte_count | integer | raw input-byte count; for str input the exact UTF-8 byte count | artifact | SUM | NONE |
| text.sample.character_count | integer | Unicode code-point count after decoding/BOM removal | artifact | SUM | NONE |
| text.surface.line_ending | categorical | counts of lf, crlf, cr, line_separator, paragraph_separator sequences | line break | CAT | JSD2 |
| text.surface.whitespace_class | categorical | Unicode White_Space code points classified per 7.2 | whitespace code point | CAT | JSD2 |
| text.surface.unicode_general_category | categorical | unicodedata.category(c) per code point | character | CAT | JSD2 |
| text.surface.letter_case | categorical | letter categories upper/lower/title/uncased per 7.10 | Unicode letter | CAT | JSD2 |
| text.surface.punctuation_codepoint | categorical | every Unicode P* code point, keyed U+XXXX / U+XXXXXX | punctuation code point | CAT | JSD2 |
| text.surface.marker_style | categorical | fixed marker events per 7.7 | marker event | CAT | JSD2 |

The marker_style vocabulary: apostrophe_ascii U+0027; apostrophe_right
U+2019; quote_ascii U+0022; quote_left_single U+2018; quote_left_double
U+201C; quote_right_double U+201D; guillemet_left U+00AB; guillemet_right
U+00BB; hyphen_minus U+002D; en_dash U+2013; em_dash U+2014;
horizontal_ellipsis U+2026; ascii_three_dots (literal non-overlapping `...`
counted left-to-right). U+2019 is classified by code point, not by role.

### 6.3 Text lexical

Owner: `stylog.text.lexical`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| text.lexical.word_count | integer | count of WORD tokens | artifact | SUM | NONE |
| text.lexical.number_count | integer | count of NUMBER tokens | artifact | SUM | NONE |
| text.lexical.token_kind | categorical | word vs number token counts | lexical token | CAT | JSD2 |
| text.lexical.word_length | histogram, top-code 31 | code-point length of each WORD; min(len, 31) | word | HIST | W1 |
| text.lexical.type_count_casefold | integer | distinct token.casefold() WORD types | word | SAMPLE | NONE |
| text.lexical.ttr_casefold | ratio x1 | distinct casefold types / WORD tokens | word | SAMPLE | ABS |
| text.lexical.hapax_type_count_casefold | integer | casefold types occurring exactly once | word | SAMPLE | NONE |
| text.lexical.hapax_token_share_casefold | ratio x1 | hapax token occurrences / WORD count | word | SAMPLE | ABS |
| text.lexical.window_ttr_100 | summary | TTR in consecutive complete 100-WORD windows; tail discarded | 100-word window | NONE | NONE |
| text.lexical.word_entropy_bits_casefold | float | -sum p_i log2 p_i over casefold WORD frequencies | word | SAMPLE | NONE |
| text.lexical.word_simpson_concentration_casefold | float | sum p_i^2 | word | SAMPLE | ABS |

`window_ttr_100` is `insufficient_support` below 100 WORD tokens. TTR, type,
and hapax measures MUST NOT be pooled across documents (they reduce as
SAMPLE only).

### 6.4 Text structural

Owner: `stylog.text.structure`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| text.structure.sentence_count | integer | sentence count per 7.8 | artifact | SUM | NONE |
| text.structure.sentence_length_tokens | histogram, top-code 101 | lexical tokens per sentence, min(n, 101) | sentence | HIST | W1 |
| text.structure.sentence_length_characters | histogram, top-code 501 | code points per sentence after sentence-edge whitespace trimming | sentence | HIST | W1 |
| text.structure.paragraph_count | integer | paragraph count per 7.9 | artifact | SUM | NONE |
| text.structure.paragraph_sentence_count | histogram, top-code 51 | sentences per paragraph | paragraph | HIST | W1 |
| text.structure.paragraph_token_count | histogram, top-code 501 | lexical tokens per paragraph | paragraph | HIST | W1 |

### 6.5 English function words

Owner: `stylog.text.function_words.en`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| text.function_words.en.token_share | ratio x1 | matched v1 function-word WORD tokens / all WORD tokens | word | RATIO | ABS |
| text.function_words.en.lexeme_distribution | categorical | distribution among matched resource lexemes | matched function word | CAT | JSD2 |

Resource `stylog.function_words.en` version 1.0.0: UTF-8, LF line endings,
one casefolded lexeme per line, sorted by UTF-8 bytes, trailing LF, 222
lexemes, sha256
`2177da3067b27ab7f1c1c228474bd5f3c6d59d3c71ffac6ab52c787c4ea881f5`. The
checked-in file `src/stylog/resources/function_words_en_v1.txt` is
authoritative; a test recomputes and asserts the SHA-256, and bootstrap
refuses a hash-mismatched resource (`RESOURCE_MISMATCH`). Matching is WORD
token casefold equality against the resource lexemes.

> Errata (decided 2026-08-16): an earlier draft of this specification pinned
> "exactly 217 lexemes" with a hash that no 217-subset of the rendered block
> reproduces (verified by exhaustive search). The rendered 222-lexeme block
> is the v1 resource and the pinned identity is the recomputed hash above.

Language gate: `en` -> active; `und` -> `unavailable` with a
`LANGUAGE_UNSPECIFIED` diagnostic; any other language -> `not_applicable`.
When the resource itself is absent, the features degrade to typed
unavailability.

### 6.6 Generic code (every kind="code" artifact, before language parsing)

Owners: `stylog.code.sample`, `stylog.code.surface`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.sample.byte_count | integer | raw bytes | artifact | SUM | NONE |
| code.sample.character_count | integer | decoded code points | artifact | SUM | NONE |
| code.sample.physical_line_count | integer | per 8.2 | artifact | SUM | NONE |
| code.surface.line_ending | categorical | same five categories as text | line break | CAT | JSD2 |
| code.surface.whitespace_class | categorical | fixed categories per 8.3/7.2 | whitespace code point | CAT | JSD2 |
| code.surface.indent_kind | categorical | none/spaces/tabs/mixed per nonblank physical line | nonblank line | CAT | JSD2 |
| code.surface.indent_char_count | histogram, top-code 33 | leading ASCII space/tab characters; 33 means >= 33 | nonblank line | HIST | W1 |
| code.surface.nonblank_line_length | histogram, top-code 201 | code points excluding line ending, nonblank lines | nonblank line | HIST | W1 |
| code.surface.blank_line_share | ratio x1 | blank physical lines / physical lines | physical line | RATIO | ABS |
| code.surface.blank_run_length | histogram, top-code 11 | lengths of maximal blank-line runs; 11 means >= 11 | blank run | HIST | W1 |
| code.surface.trailing_space_line_share | ratio x1 | lines ending in ASCII space/tab before the line ending / physical lines | physical line | RATIO | ABS |

### 6.7 Python lexical

Owner: `stylog.code.python.tokens`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.python.lexical.token_class | categorical | fixed token classes per 8.5 | token | CAT | JSD2 |
| code.python.lexical.keyword_distribution | categorical | hard/soft keyword lexemes | token | CAT | JSD2 |
| code.python.lexical.operator_distribution | categorical | exact OP lexemes | token | CAT | JSD2 |
| code.python.lexical.number_form | categorical | number classes per 8.6 | token | CAT | JSD2 |
| code.python.lexical.string_quote | categorical | ' " ''' \"\"\" for STRING tokens | string token | CAT | JSD2 |
| code.python.lexical.string_prefix | categorical | normalized prefix classes per 8.7 | string token | CAT | JSD2 |

Each categorical is `insufficient_support` when its event population is
empty.

### 6.8 Python naming

Owners: `stylog.code.python.tokens` (occurrence features),
`stylog.code.python.ast` (binding/attribute features).

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.python.naming.identifier_occurrence_length | histogram, top-code 65 | NAME-token identifier lengths excluding hard/soft keywords | identifier occurrence | HIST | W1 |
| code.python.naming.identifier_occurrence_case_style | categorical | style classes per 8.9 | identifier occurrence | CAT | JSD2 |
| code.python.naming.binding_role | categorical | AST binding events per 8.10 | binding event | CAT | JSD2 |
| code.python.naming.binding_length | histogram, top-code 65 | binding-name code-point length | binding event | HIST | W1 |
| code.python.naming.binding_case_style | categorical | 8.9 over bindings | binding event | CAT | JSD2 |
| code.python.naming.binding_component_length | histogram, top-code 33 | lengths of split identifier components per 8.9 | binding event | HIST | W1 |
| code.python.naming.attribute_name_length | histogram, top-code 65 | ast.Attribute.attr occurrence lengths | attribute occurrence | HIST | W1 |
| code.python.naming.attribute_case_style | categorical | case style of ast.Attribute.attr occurrences | attribute occurrence | CAT | JSD2 |

### 6.9 Python syntax

Owner: `stylog.code.python.ast`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.python.syntax.node_distribution | categorical | included AST class names per 8.11 | ast node | CAT | JSD2 |
| code.python.syntax.parent_child_distribution | categorical | direct included parent->child pairs, key `ParentClass>ChildClass` | ast node edge | CAT | JSD2 |
| code.python.syntax.node_depth | histogram, top-code 26 | raw-AST-edge depth of included nodes; Module depth 0 but excluded from observations | ast node | HIST | W1 |

### 6.10 Python structural

Owner: `stylog.code.python.ast`.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.python.structure.function_count | integer | FunctionDef + AsyncFunctionDef count | artifact | SUM | NONE |
| code.python.structure.class_count | integer | ClassDef count | artifact | SUM | NONE |
| code.python.structure.function_kind | categorical | sync, async | function | CAT | JSD2 |
| code.python.structure.function_length_lines | histogram, top-code 201 | end_lineno - lineno + 1 | function | HIST | W1 |
| code.python.structure.parameter_count | histogram, top-code 21 | parameters per function per 8.12 | function | HIST | W1 |
| code.python.structure.return_count | histogram, top-code 11 | returns per function excluding nested scopes | function | HIST | W1 |
| code.python.structure.nonterminal_return_function_share | ratio x1 | functions containing a return before a later statement position / functions | function | RATIO | ABS |
| code.python.structure.control_construct | categorical | fixed constructs per 8.13 | control construct | CAT | JSD2 |
| code.python.structure.max_control_nesting | histogram, top-code 11 | per-function maximum per 8.14 | function | HIST | W1 |
| code.python.structure.branch_construct_count | histogram, top-code 21 | per function per 8.15 (NEVER called cyclomatic complexity) | function | HIST | W1 |
| code.python.structure.decorator_count | histogram, top-code 11 | decorator list length for functions/classes | decorated definition | HIST | W1 |
| code.python.structure.assignment_kind | categorical | assign, ann_assign, aug_assign, named_expr | assignment | CAT | JSD2 |
| code.python.structure.import_kind | categorical | import, import_from | import | CAT | JSD2 |
| code.python.structure.import_alias_share | ratio x1 | import aliases with asname / all ast.alias items | import alias | RATIO | ABS |
| code.python.structure.exception_construct | categorical | per 8.16 | exception construct | CAT | JSD2 |
| code.python.structure.comprehension_kind | categorical | list, set, dict, generator | comprehension | CAT | JSD2 |
| code.python.structure.match_case_count | histogram, top-code 11 | cases per Match | match statement | HIST | W1 |

### 6.11 Python comments and docstrings

Owners: `stylog.code.python.tokens` (comments),
`stylog.code.python.ast` (docstrings).

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.python.comments.comment_count | integer | successful tokenize COMMENT tokens | comment | SUM | NONE |
| code.python.comments.comment_line_share | ratio x1 | unique physical rows containing a COMMENT / physical lines | line | RATIO | ABS |
| code.python.comments.comment_length | histogram, top-code 121 | cleaned comment content length per 9.1 | comment | HIST | W1 |
| code.python.comments.docstring_kind | categorical | module, class, function, async_function | docstring | CAT | JSD2 |
| code.python.comments.docstring_length | histogram, top-code 501 | cleaned docstring text length | docstring | HIST | W1 |

### 6.12 Toolchain marker feature

Owner: `stylog.code.python.tokens`.

`code.toolchain.python_comment_marker` is categorical with vocabulary:
shebang, encoding_cookie, type_ignore, ruff_noqa, noqa, fmt_off, fmt_on,
fmt_skip, isort_skip, isort_off, isort_on, isort_split, pylint, mypy,
pyright, generated_notice (support: marker event; CAT; JSD2). A comment MAY
contribute multiple marker events. Markers are evidence, never source-history
inference.

### 6.13 Tree-sitter parser-backed registry (javascript/typescript/c/rust only)

Owner: `stylog.code.tree_sitter`. Feature IDs are language-agnostic but
emitted per artifact language; compatibility includes language plus the
grammar signature.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| code.parser.named_node_type | categorical | every node with is_named=true, counted by grammar node type; root included when named; anonymous nodes excluded | named node | CAT | JSD2 |
| code.parser.named_parent_child | categorical | for every named node except a named root: nearest named ancestor, key parent_type + U+2192 + child_type; anonymous nodes ignored for ancestry | named-node edge | CAT | JSD2 |
| code.parser.named_depth | histogram, top-code 32 | named-node depth; root named node depth 0; depth = number of named ancestors | named node | HIST | W1 |
| code.parser.identifier_length | histogram, top-code 64 | code-point length of every node classified as identifier by the checked language mapping resource | identifier occurrence | HIST | W1 |
| code.parser.identifier_style | categorical | the 8.9 identifier-style algorithm over identifier text; "_" discarded exactly as in 8.9 | identifier occurrence | CAT | JSD2 |
| code.parser.comment_kind | categorical | each checked comment node: "line" when the source begins with a checked line-comment delimiter, else "block" when it begins a checked block delimiter; the mapping resource owns the delimiters | comment | CAT | JSD2 |
| code.parser.comment_length | histogram, top-code 256 | code-point length of the entire comment node text including delimiters | comment | HIST | W1 |

Each language has a checked, versioned mapping resource
(`stylog.tree_sitter.mapping.<language>`, mapping_version `"1.0.0"`):
identifier node types, comment node types, line/block comment delimiters.
Unmapped node types MUST NOT be guessed at runtime. Missing applicable events
produce typed no-support/not-applicable observations. These features are
comparable only when language and parser/grammar scientific compatibility
match.

### 6.14 Optional linguistic registry (stylog[nlp], spaCy)

Owner: `stylog.text.linguistic`; semantic_version `"1.0.0"`. A linguistic
token is a spaCy token with `token.is_space == False`; the core tokenizer is
not reused for these labels.

| Feature | Geometry | Definition | Support | Reducer | Metric |
|---|---|---|---|---|---|
| text.linguistic.upos | categorical | count of token.pos_ per linguistic token; empty labels do not become an empty-string category | linguistic token | CAT | JSD2 |
| text.linguistic.dependency_relation | categorical | count of token.dep_ per linguistic token with a nonempty label, including root | linguistic token | CAT | JSD2 |
| text.linguistic.dependency_distance | histogram, top-code 32 | non-root token with head in the same doc: abs(token.i - token.head.i) | non-root dependency arc | HIST | W1 |
| text.linguistic.morph_attribute | categorical | per linguistic token, one category per morph attribute/value pair "Feature=Value"; a token can contribute multiple; multi-valued values are sorted as returned through the normalized mapping before category construction | morphology attribute event | CAT | JSD2 |
| text.linguistic.morph_coverage | ratio x1 | tokens with >= 1 morph attribute / all linguistic tokens | linguistic token | RATIO | ABS |

No lemma, token, or dependency text is serialized. Missing annotation
produces typed `unavailable`/`not_applicable` observations - never heuristic
substitution. The analyzer records exact model provenance (a `ModelSignature`
on the backend, section 5.17).

## 7. Exact text algorithms

### 7.1 No Unicode normalization

Never NFC/NFD/NFKC/NFKD. `character_count` is `len(decoded_string)`.

### 7.2 White_Space resource

Unicode 17.0 White_Space, hard-coded 25 code points: U+0009-U+000D, U+0020,
U+0085, U+00A0, U+1680, U+2000-U+200A, U+2028, U+2029, U+202F, U+205F,
U+3000.

`whitespace_class` categories: space_ascii U+0020; tab U+0009; line_feed
U+000A; carriage_return U+000D; line_separator U+2028; paragraph_separator
U+2029; other_white_space (all remaining White_Space code points). CRLF
contributes one carriage_return + one line_feed here, while `line_ending`
treats CRLF as one sequence.

### 7.3 General category

`unicodedata.category(ch)` directly; `unicodedata.unidata_version` is part of
the `RuntimeSignature`.

### 7.4 Token model

Tokenizer id `stylog.text.tokenizer/1.0.0`. Kinds: WORD and NUMBER only;
punctuation is measured separately.

WORD: begins with a category L* code point. Continues through L*, M*
combining marks, internal U+0027 `'`, and internal U+2019 - an apostrophe
continues the token only when preceded inside the same WORD by L or M AND
immediately followed by a Unicode L code point. Apostrophes are retained in
token text. M* cannot begin a WORD. Hyphens and dashes always terminate a
WORD.

NUMBER: begins with a category Nd code point. Continues through Nd, ASCII
`.`, `,`, `_` - each separator is included only when immediately preceded and
immediately followed by Nd. Signs are never part of a NUMBER. Nl/No are not
NUMBER starts.

Everything else ends the current token and is not itself a token.

### 7.5 Tokenizer conformance

`Don't re-enter\u2014now. na\u00efve caf\u00e9 3.14` -> WORD(Don't) WORD(re) WORD(enter)
WORD(now) WORD(na\u00efve) WORD(caf\u00e9) NUMBER(3.14).
`'dogs' rock\u2019n\u2019roll x_y 1,000.50` -> WORD(dogs) WORD(rock\u2019n\u2019roll)
WORD(x) WORD(y) NUMBER(1,000.50). A letter followed by a combining mark is
one WORD. `-12 + 5` -> NUMBER(12), NUMBER(5).

### 7.6 Lexical type equality

Type equality is `token.casefold()` exactly. No normalization.

### 7.7 ascii_three_dots

Counted left-to-right, non-overlapping: `......` yields 2 ascii_three_dots
events and 6 U+002E punctuation code points.

### 7.8 Sentence segmentation

Segmenter id `stylog.text.sentence_segmenter/1.0.0`. Segmentation runs
separately inside each paragraph. Terminal code points: U+002E, U+0021,
U+003F, U+2026, U+3002, U+FF01, U+FF1F. Special decimal rule: U+002E is NOT
terminal when the immediate previous and immediate next code points are both
Nd. A maximal consecutive sequence of terminals is one terminal cluster.
After the cluster, following closers are included: `'`, `"`, `)`, `]`, `}`,
U+2019, U+201D, U+00BB, U+203A, U+300D, U+300F, U+3011. A boundary occurs at
terminal cluster + zero or more closers + a next code point that is
White_Space, or at paragraph end. Leading/trailing White_Space is excluded
from the sentence span. Non-whitespace residual paragraph text after the
final terminal forms a sentence. There is NO abbreviation dictionary.

Conformance: `Dr. Smith left. Value 3.14! Really?` -> `Dr.`, `Smith left.`,
`Value 3.14!`, `Really?`. `\u201cWhat?!\u201d She asked` -> `\u201cWhat?!\u201d`, `She asked`.

### 7.9 Paragraph segmentation

The line-break scanner recognizes CRLF, LF, CR, U+2028, U+2029 (CRLF is one
break). A physical line is the characters between break sequences. A blank
line has zero characters or only Unicode White_Space (excluding the break). A
paragraph is a maximal run of >= 1 nonblank physical lines; >= 1 blank lines
separate paragraphs. U+2029 always terminates the current paragraph. U+2028
terminates a physical line but does not force an extra paragraph beyond the
blank-line rule. Empty or all-blank text yields paragraph_count = 0 and
sentence_count = 0.

### 7.10 Casing

Lu -> upper, Ll -> lower, Lt -> title, other L* -> uncased. Never
`.isupper()`/`.islower()`.

### 7.11 TTR and hapax

word_count = n; type_count = distinct casefold types; TTR = types / n;
hapax_type_count = types with frequency 1; hapax_token_share =
hapax_type_count / n. When n = 0: word_count ok(0), type/hapax counts ok(0),
TTR and hapax share `insufficient_support`.

### 7.12 Window TTR

Non-overlapping windows tokens[0:100], [100:200], ...; the tail below 100 is
discarded; ttr = distinct_casefold_types / 100 per complete window, emitted
as a SummaryStatisticsValue. No sliding windows, no random offset. Below 100
words: `insufficient_support`.

### 7.13 Entropy and concentration

p_i = c_i / n; entropy_bits = -sum p_i log2(p_i); simpson = sum p_i^2. No
smoothing. n = 0 -> `insufficient_support`. Summation uses `math.fsum`.

## 8. Exact Python analysis

### 8.1 Source encoding

Python source bytes MUST use `tokenize.detect_encoding()` exactly (BOM and
PEP 263, conflicts rejected, default UTF-8). A generic text-encoding override
MUST NOT override Python source encoding.

### 8.2 Physical lines

For nonempty source: number of line-break sequences + 1. `""` -> 0; `"a"` ->
1; `"a\n"` -> 2; `"\n"` -> 2.

### 8.3 Line classification

Per physical line: remove the line ending; the line is blank if all remaining
characters are Unicode White_Space; nonblank leading indentation is the run
of consecutive ASCII space/tab from column 0. `indent_kind`:
none/spaces/tabs/mixed. `indent_char_count` counts characters, never visual
columns; tab width is never assumed. A trailing-space line ends in >= 1
U+0020/U+0009 before the line ending.

### 8.4 Tokenizer execution

`tokenize.tokenize(BytesIO(raw_bytes).readline)`. The token stream is ATOMIC:
tokens are collected to a temporary list; on TokenError, IndentationError,
SyntaxError, or UnicodeDecodeError (or equivalent) ALL partial token-derived
observations are discarded, every token-owned feature becomes `parser_error`,
generic code-surface observations are preserved, and AST is NOT attempted
after tokenization failure.

### 8.5 Token classes

`token_class` ignores ENCODING, ENDMARKER, NL, NEWLINE, INDENT, DEDENT, and
COMMENT tokens. Classes: NAME + `keyword.iskeyword` -> keyword; else NAME +
`keyword.issoftkeyword` (where the runtime provides it) -> soft_keyword;
other NAME -> identifier; NUMBER -> number; STRING -> string; token-name
prefix FSTRING_/TSTRING_ -> string_fragment; OP -> operator; everything else
-> other. `keyword_distribution` keys are exact case-sensitive keyword
strings; `operator_distribution` keys are `token.string`.

### 8.6 Number forms

Case-insensitive on the exact spelling: endswith `j` -> imaginary; starts
`0x` -> hexadecimal; `0o` -> octal; `0b` -> binary; contains `.` or `e` ->
floating; else decimal_integer. Underscores do not alter the class. Signs are
separate operators.

### 8.7 String tokens

Find the shortest position beginning a valid quote delimiter; the characters
before it are the prefix; lowercase the prefix; delimiter categories `'`,
`"`, `'''`, `"""`. Prefix categories: none (empty), r, u, b, br, rb, f, fr,
rf, other. Only whole STRING tokens participate; FSTRING_/TSTRING_ fragments
appear in `token_class` only.

### 8.8 AST

AST analysis runs only after successful tokenization and if resource guards
permit: `ast.parse(decoded_source, mode="exec", type_comments=False,
feature_version=None)` - feature_version is NEVER pinned. Resource guards are
mandatory (memory and nesting, section 16.3). On AST failure: token features
stay `ok`; AST/naming-binding/structure/docstring features become
`parser_error`; embedded ordinary comments remain; embedded docstrings become
unavailable via `parser_error`.

### 8.9 Identifier case style

Special cases: name == "_" -> discard; `^__[A-Za-z0-9_]+__$` -> dunder.

Component splitting: (A) split on one or more `_`, discarding empties; if
none remain -> other. (B) case-transition split inside each component: split
before c_i when (1) the previous character is a lowercase letter or Nd and
the current is an uppercase letter; or (2) previous and current are uppercase
and the next is lowercase. Digits stay attached until a case transition or
underscore. Examples: HTTPServer -> HTTP, Server; myURL2 -> my, URL2;
foo_bar -> foo, bar.

Style classification (on the original identifier): discard exactly "_";
dunder starts `__` ends `__` with length > 4; snake_lower contains `_` and
all cased letters lower; snake_upper contains `_` and all cased letters
upper; camel_lower has no `_`, first cased letter lower, and a later
uppercase exists; pascal has no `_`, first cased letter upper, and a later
lowercase exists; lower has no `_`, all cased letters lower; upper has no
`_`, all cased letters upper; mixed is anything else with cased letters;
uncased has no cased letters. Leading/trailing single underscores do not by
themselves prevent the underlying style: strip leading/trailing `_` for style
determination AFTER checking dunder and "_". `binding_component_length`
measures the code-point lengths of transition-split components.

### 8.10 Binding-event extraction

Every event is recorded; no deduplication. Roles: function, async_function,
class, parameter_posonly, parameter_positional, parameter_kwonly,
parameter_vararg, parameter_kwarg, assignment, walrus, loop_target,
comprehension_target, with_target, exception_target, import_binding,
pattern_binding.

- FunctionDef.name -> function; AsyncFunctionDef.name -> async_function;
  ClassDef.name -> class.
- Each ast.arg exactly once, with its structural parameter role. self/cls are
  not special-cased.
- Assign/AnnAssign/AugAssign: recursively collect Name targets, traversing
  tuple/list/starred; Attribute and Subscript are not bindings. A NamedExpr
  target -> walrus.
- For/AsyncFor targets -> loop_target (NOT also assignment).
- Comprehension targets -> comprehension_target.
- withitem.optional_vars -> with_target.
- ExceptHandler.name, when nonempty -> exception_target.
- Imports: `import a.b` binds `a`; `import a.b as x` binds `x`;
  `from a import b` binds `b`; `from a import b as x` binds `x`;
  `from a import *` emits no binding.
- Match patterns: MatchAs.name, MatchStar.name, and MatchMapping.rest are
  bindings (nested patterns traversed recursively). MatchClass.kwd_attrs are
  attribute names, not bindings.
- Exclusions: Global, Nonlocal, Delete, Attribute, Subscript, keyword
  argument labels, decorator names.
- "_" remains a binding event, classified discard.

### 8.11 AST included nodes

Every `ast.AST` instance EXCEPT ast.Module, ast.operator subclasses,
ast.unaryop, ast.boolop, ast.cmpop, and ast.expr_context subclasses.
arguments/arg/keyword/alias/comprehension/match-pattern nodes remain
included. Parent-child pairs consider direct `ast.iter_child_nodes()`
children only; an excluded direct child is ignored, with no bridging through
excluded nodes. Node depth: Module = 0; every raw AST edge adds 1; an
included child records its raw-tree depth (excluded intermediate edges still
contribute depth). Top-code 26.

### 8.12 Function measurements

Every nested FunctionDef/AsyncFunctionDef is measured.

- parameter_count = len(posonlyargs) + len(args) + len(kwonlyargs)
  + (1 if vararg) + (1 if kwarg).
- function_length_lines = end_lineno - lineno + 1; when location metadata is
  missing, that function is excluded from the length feature with diagnostic
  `PYTHON_LOCATION_UNAVAILABLE`; other function features stay valid.
- return_count: Return nodes in the function body, NOT descending into nested
  FunctionDef/AsyncFunctionDef/ClassDef/Lambda; the root function itself is
  traversed.
- Nonterminal return: max_stmt_end = the maximum end_lineno over statements
  belonging to that function, excluding nested functions/classes; a Return is
  nonterminal when return.end_lineno < max_stmt_end; a function contributes
  to the numerator when it has >= 1 such Return. This is a positional
  structural measurement, not a control-flow proof.

### 8.13 Control constructs

if, for, async_for, while, try, try_star, with, async_with, match,
list_comprehension, set_comprehension, dict_comprehension,
generator_expression, if_expression. try_star exists only where the runtime
AST exposes TryStar; an absent runtime class does not create a zero category.

### 8.14 Control nesting

Measured per function. The root function body statement depth is 0. Entering
the body/orelse/finally/handler/case-body of If, For, AsyncFor, While, With,
AsyncWith, Try, TryStar, or a Match case increments depth by 1. Nested
function/class/lambda bodies are not traversed. Comprehension expressions do
not affect this feature. The per-function maximum is recorded; no nesting
yields 0.

### 8.15 Branch constructs

Per function, excluding nested scopes: +1 for each If, IfExp, For, AsyncFor,
While, ExceptHandler, Match case, comprehension generator, and comprehension
`if` clause. Try itself is not a branch beyond its handlers. This feature is
NEVER named cyclomatic_complexity.

### 8.16 Exception constructs

try, try_star, except_handler, bare_except, finally_block, raise. A
Try/TryStar node emits a try event; each handler emits except_handler; a
handler with type None additionally emits bare_except; a nonempty finalbody
emits one finally_block; every Raise node emits raise.

### 8.17 Import alias share

Denominator: all ast.alias objects, including alias(name="*"). Numerator:
aliases with asname not None. A zero denominator yields
`insufficient_support`.

### 8.18 Match case count

Each Match node contributes len(node.cases). No Match nodes ->
`insufficient_support`.

### 8.19 Runtime determinism

Deterministic output is guaranteed only when all compatibility signatures
match: feature semantics, analyzer implementation versions, resource hashes,
CPython implementation, full Python version, cache tag, Unicode database
version, and analysis configuration. NO byte-identical promise is made across
Python 3.12/3.13/3.14, and no cross-platform byte claim is made (C libm
exp/log may differ in the last ulp). The same stance applies to verifier
fitting (section 23.11).

## 9. Embedded comment and docstring text

### 9.1 Comment cleaning

Remove exactly the first `#`; if the next character is exactly U+0020, remove
that one space; preserve everything else. No strip, no dedent. `# hello` ->
`hello`; `#hello` -> `hello`; `#  hello` -> ` hello`; `#` -> ``.

### 9.2 Exclusions from embedded prose

Excluded (but still measured as code/toolchain observations): the shebang;
the PEP 263 encoding cookie; any recognized toolchain marker (9.7); a comment
that is only whitespace after cleaning.

### 9.3 Full-line comment

Every character before the comment's `#` start column on its physical line is
ASCII space or tab.

### 9.4 Comment blocks

A `comment_block` is a maximal sequence of ordinary comments, all full-line,
on consecutive physical rows, at the same token start column. Each comment is
cleaned per 9.1 and the parts are joined with exactly LF, with no final LF.
An isolated full-line comment is a one-comment block.

### 9.5 Inline comments

An `inline_comment` is an ordinary comment with non-whitespace source before
its `#`. Inline comments are never grouped.

### 9.6 Docstrings

Docstrings follow normal AST semantics (module/class/function/async-function
first statement). Text extraction MUST use `ast.get_docstring(node,
clean=True)` with no additional stripping. Only standard docstrings qualify;
standalone string literals are not docstrings.

### 9.7 Toolchain markers

Recognized after marker removal and stripping of surrounding ASCII space/tab:

- shebang: row == 1, column == 0, raw text starts `#!`.
- encoding_cookie: first or second physical line matching PEP 263 coding
  syntax.
- type_ignore: case-sensitive `^type:\s*ignore(?:\[[^\]]+\])?\s*$`.
- ruff_noqa: case-insensitive `^ruff:\s*noqa\b`.
- noqa: case-insensitive `^noqa(?:\s*:\s*[A-Z0-9_,\-\s]+)?$`.
- fmt: case-insensitive `^fmt:\s*(off|on|skip)\b` -> fmt_off/fmt_on/fmt_skip.
- isort: case-insensitive `^isort:\s*(skip|off|on|split)\b` ->
  isort_skip/isort_off/isort_on/isort_split.
- pylint/mypy/pyright: case-insensitive prefix `pylint:`/`mypy:`/`pyright:`.
- generated_notice: only comments on the first 20 physical lines are
  eligible; the casefolded cleaned text contains "generated" and at least one
  of "do not edit" / "auto-generated" / "autogenerated".

### 9.8 Embedded identities

Embedded artifact IDs: `<parent>/comment_block/000001`,
`<parent>/inline_comment/000001`, `<parent>/docstring/000001`, with ordinals
in source order per kind. Embedded content identity is the SHA-256 of the
exact UTF-8 encoding of the cleaned embedded text (or the suppressed mode).

### 9.9 Embedded language

Embedded text language defaults to `und`; the configuration key
`analysis.code.python.embedded_text_language.language` (section 16.3) MAY set
another language such as `en`.

### 9.10 Embedded participation

Primary code fingerprints MUST NOT mix embedded text features into code
namespaces; the AnalysisBundle carries embedded fingerprints separately.
Comparing two code bundles with embedded analysis produces separate sections
`embedded.comments` and `embedded.docstrings`: each side's comment artifacts
form a temporary EvidenceSet, as do its docstring artifacts; text features
are aggregated per section 11 and the aggregates compared per section 12.
When one side has no embedded artifacts of a section, that section is
unavailable - never a zero distance. Normal EvidenceSet aggregation MUST NOT
merge embedded children into parent code features.

## 10. Status, warnings, and failure

### 10.1 Status truth table

- `ok`: completed, value defined (valid zero counts are ok).
- `insufficient_support`: the concept applies, the analyzer succeeded, and
  the denominator/event population is zero (for example TTR on zero WORD
  tokens).
- `not_applicable`: the concept does not apply to known artifact semantics
  (for example English function words for `fr`).
- `unavailable`: the concept applies but a non-parser prerequisite is missing
  (for example function words under language `und`, or a missing tree-sitter
  grammar).
- `parser_error`: a required tokenizer/parser rejected the input.
- `disabled`: user configuration explicitly disabled the owning
  feature/analyzer.

### 10.2 Valid zero versus missing support

A zero occurrence is valid when the opportunity is defined:
`character_count` on empty text is ok(0); `function_count` on a valid module
with no functions is ok(0); `parameter_count` with no functions is
`insufficient_support`; TTR on empty text is `insufficient_support`. Zero,
missing, and not-applicable MUST NOT be encoded interchangeably (never as
0/null/empty conflation).

### 10.3 Analyzer atomicity

Each analyzer declares its complete owned feature-ID set. On an unexpected
internal exception, the engine discards all of that analyzer's observations,
emits `unavailable` for every owned feature plus a diagnostic
`ANALYZER_INTERNAL_ERROR` (severity error), continues the independent
analyzers, marks the operation as an internal error, and the CLI exits 5;
machine output MAY contain the partial bundle. No traceback appears in
portable output; a stderr traceback appears only under the debug flag.

### 10.4 Stable diagnostic codes

Input: INPUT_NOT_FOUND, INPUT_DECODE_ERROR, INPUT_NUL, INPUT_SURROGATE,
INPUT_UNSUPPORTED, INPUT_TOO_LARGE, DIRECTORY_TOO_MANY_FILES,
DIRECTORY_TOO_MANY_BYTES, SYMLINK_REJECTED.

Text and resources: LANGUAGE_UNSPECIFIED, RESOURCE_MISMATCH.

Python: PYTHON_ENCODING_ERROR, PYTHON_TOKENIZE_ERROR,
PYTHON_AST_PARSE_ERROR, PYTHON_AST_RESOURCE_LIMIT,
PYTHON_LOCATION_UNAVAILABLE.

Tree-sitter: TREE_SITTER_UNAVAILABLE, TREE_SITTER_PARSE_ERROR.

Analyzer: ANALYZER_INTERNAL_ERROR.

Aggregation and comparison: FEATURE_SEMANTIC_MISMATCH,
RESOURCE_SIGNATURE_MISMATCH, RUNTIME_SIGNATURE_MISMATCH,
FEATURE_NOT_COMPARABLE, NO_COMPARABLE_FEATURES.

Baseline and profile: BASELINE_NOT_FOUND, BASELINE_INVALID,
BASELINE_INCOMPATIBLE, BASELINE_INSUFFICIENT_SUPPORT, PROFILE_ZERO_MAD.

Cache (warnings, operational only - section 14.9): CACHE_CORRUPT,
CACHE_READ_FAILED, CACHE_WRITE_FAILED.

Benchmark: BENCHMARK_INVALID, DATASET_MISSING, DATASET_CHECKSUM_MISMATCH,
SPLIT_IMPOSSIBLE, SPLIT_LEAKAGE, PAIR_INVALID, PAIRWISE_AUC_OMITTED,
VERIFICATION_AUC_OMITTED.

Exception-surface codes (CLI stderr rendering of typed errors; section
19.12): CONFIGURATION_ERROR, CAPABILITY_UNAVAILABLE, INPUT_ERROR,
PORTABLE_ARTIFACT_INVALID, MODEL_INCOMPATIBLE, VERIFIER_FIT_FAILED,
BENCHMARK_INVALID.

Verifier fit diagnostics (ephemeral; emitted on stderr by `fit`, never
portable): VERIFIER_ELIGIBILITY, VERIFIER_ZERO_VARIANCE_FEATURE,
VERIFIER_UNCONSTRAINED_LANGUAGE, VERIFIER_CALIBRATION_PAIRS_EXCLUDED,
THRESHOLD_BAND_COLLAPSED (sections 23.9-23.12).

## 11. Evidence set aggregation

### 11.1 Aggregation kinds

`exact_sum`, `ratio_pool`, `categorical_pool`, `histogram_pool`,
`sample_summary`, `not_aggregatable`.

### 11.2 Compatibility gate

Observations pool only when all samples agree on feature_id, feature semantic
version, relevant resource signatures, and the relevant runtime compatibility
signature. Otherwise the aggregate status is `unavailable` with
FEATURE_SEMANTIC_MISMATCH, RESOURCE_SIGNATURE_MISMATCH, or
RUNTIME_SIGNATURE_MISMATCH. There is no coercion.

### 11.3 Contributing samples

Only `status = ok` observations contribute. Missing statuses are counted
individually in `missing`. There is no zero imputation.

### 11.4 exact_sum

Sum of integer values as Python ints.

### 11.5 ratio_pool

pooled_num = sum(numerators); pooled_den = sum(denominators); value =
pooled_num / pooled_den x multiplier. Multipliers must match.

### 11.6 categorical_pool

Per key, sum counts; a key absent from a valid observation counts as zero;
zero-count keys are omitted from serialization.

### 11.7 histogram_pool

Per point, sum counts; `top_code` MUST match across all samples.

### 11.8 sample_summary

Reduce each ok sample's primary numeric value to one scalar; store the sorted
`sample_values` plus a `SummaryStatisticsValue`. Applies to
`text.lexical.type_count_casefold`, `text.lexical.ttr_casefold`,
`text.lexical.hapax_type_count_casefold`,
`text.lexical.hapax_token_share_casefold`,
`text.lexical.word_entropy_bits_casefold`, and
`text.lexical.word_simpson_concentration_casefold`.

### 11.9 window_ttr_100

`text.lexical.window_ttr_100` is not aggregated.

### 11.10 Sample summary mathematics

mean = math.fsum(x) / n; quantiles are type 7 (section 13.3); when n == 1:
minimum = q25 = median = q75 = maximum = mean = x_1 and `sample_sd` is
omitted; when n >= 2: sample_sd = sqrt(fsum((x - mean)^2) / (n - 1)).

### 11.11 Empty contribution

With zero ok samples: no pooled value, no sample summary,
contributing_samples = 0, and the full missing-status counts are retained.

### 11.12 Duplicate content

Identical content hashes remain two samples; there is no automatic
deduplication.

## 12. Comparison mathematics

### 12.1 No global similarity

There is no global similarity score, and per-feature distances MUST NOT be
averaged or otherwise combined into a global score inside a Comparison
(section 5.15).

### 12.2 Comparability gate

Two observations compare only when they share feature_id and
semantic_version, relevant resources are compatible, the runtime signature is
compatible for runtime-sensitive features, and both are `ok`. A language
mismatch does not forbid language-independent features (a diagnostic at
comparison scope); language-specific features fail naturally. Text and code
primary fingerprints MUST NOT cross-compare.

### 12.3 ABS

d = |a - b|; unit "proportion points on [0,1]" for ratios.

### 12.4 SPD

0 when a == b == 0, else 2|a - b| / (|a| + |b|). No required v0.1 feature
uses SPD, but the implementation MUST provide it tested.

### 12.5 JSD2

Distance, not divergence. Union keys K; P_k = count / total; M = (P + Q) / 2;
KL2(P||M) = sum over P_k > 0 of P_k log2(P_k / M_k); JSD2 = sqrt(0.5 KL2(P||M)
+ 0.5 KL2(Q||M)). 0 = identical, 1 = maximally disjoint. No smoothing; a zero
probability contributes zero. The pure-Python `math.fsum` implementation in
`analysis/stats.py` is the semantic reference. An optimized kernel (for
example NumPy) is permitted only with exact conformance; none ships in v0.1 -
the reference implementation is the only implementation. SciPy remains a
dev-only oracle (section 4.8).

### 12.6 W1 on top-coded support

Sort the union support z_1..z_m; cdf_diff after z_i = F_P(z_i) - F_Q(z_i);
W1 = sum_{i=1}^{m-1} |cdf_diff(z_i)| (z_{i+1} - z_i). The result is in the
native transformed unit of the feature; no normalization.

### 12.7 Raw counts

Raw counts are not compared unless the registry assigns a metric (most are
NONE).

### 12.8 Summary statistics

A SummaryStatisticsValue is not directly compared.

### 12.9 Missing features

Missing/unavailable features are omitted from components, represented in
diagnostics, and NEVER encoded as maximal or zero distance. Each registry-
comparable feature that is absent or non-`ok` on either side emits
`FEATURE_NOT_COMPARABLE` with `left_status` and `right_status` context.

### 12.10 Aggregate-to-aggregate comparison

Pooled values compare with the ordinary metric; SAMPLE_SUMMARY features
compare their `sample_values` via Wasserstein-1 over the actual scalar
values, with metric name `sample_wasserstein_1`; non-aggregatable features
produce no component.

### 12.11 Subject kinds

Fingerprint-to-EvidenceSet comparison is forbidden; construct a one-member
EvidenceSet explicitly. Comparisons run fingerprint-to-fingerprint,
aggregate-to-aggregate, or bundle-to-bundle (with embedded sections per
9.10); mixed subject kinds are a typed error.

### 12.12 Support

There is no minimum-support rule: a component exists whenever both values are
`ok`; support counts are included for evidence volume.

## 13. Baselines and profiling

### 13.1 Profileable values

IntegerValue, FloatValue, and RatioValue (via the normalized `value`) are
profileable. Categorical, histogram, and summary values are NOT profiled.

### 13.2 Baseline units

The baseline sample unit is one independent artifact (default
`unit = "artifact"`) or explicitly `"evidence_set"`. Every unit has equal
weight; units are never mixed.

### 13.3 Type-7 quantiles

Sorted zero-indexed x[0..n-1]; h = (n - 1) p; j = floor(h); g = h - j;
Q(p) = (1 - g) x[j] + g x[min(j + 1, n - 1)]. Q1 = Q(.25), median = Q(.5),
Q3 = Q(.75), IQR = Q3 - Q1.

### 13.4 Midrank percentile

L = count below x, E = count equal to x, N = total count;
percentile_midrank = 100 (L + 0.5 E) / N. No interpolation.

### 13.5 MAD

mad_raw = median(|x_i - median(x)|); mad_normal_scaled = mad_raw x
1.482602218505602.

### 13.6 Robust z

When mad_raw > 0: robust_z = (observed - median) / mad_normal_scaled. When
mad_raw == 0: omit `robust_z` and emit PROFILE_ZERO_MAD (whether or not the
observed value equals the median). Never emit infinity.

### 13.7 Support threshold

There is no positive minimum-support threshold. For every non-empty
compatible feature distribution, compute a ProfileObservation from all valid
values and record the exact `baseline_n`. When baseline_n == 0 the statistics
are undefined: emit BASELINE_INSUFFICIENT_SUPPORT and no ProfileObservation.

### 13.8 Baseline construction

Only `ok` baseline observations are included. The baseline retains, per
feature, the total source units and the valid values. `build_baseline`
(section 20.1) builds a local baseline from analyzed units; building from
zero units is BASELINE_INVALID.

### 13.9 Confidence intervals

No confidence intervals exist in v0.1.

### 13.10 Baseline portable format

```json
{"schema":"stylog.baseline","schema_version":"0.1.0",
 "baseline_id":"...","baseline_version":"...",
 "descriptor":{"kind":"text","language":"en","domain":"...",
               "unit":"artifact","source":"..."},
 "source_manifest_sha256":"...",
 "compatibility":{"feature_registry_version":"1.0.0"},
 "features":[{"feature_id":"...","semantic_version":"1.0.0",
              "compatibility_sha256":"...","total_units":25,
              "values":["...ascending..."]}]}
```

`values` are ascending and never exceed `total_units` in count (baseline
files conventionally use the suffix `.stylog-baseline.json`).
`compatibility_sha256` is the JCS hash of feature_id, semantic_version, the
relevant resource signatures, and the relevant runtime-compatibility fields.

### 13.11 Baseline resolution

If the reference contains a path separator or ends in `.json`, it is an
explicit path; otherwise the baseline ID is searched in the configured
`baseline.search_paths` and then in
`platformdirs.user_data_path("stylog")/"baselines"`, by exact `baseline_id`
match. Zero matches -> BASELINE_NOT_FOUND; multiple same-ID baselines with
different hashes -> BASELINE_INVALID. Resolution never touches the network.

### 13.12 Per-feature compatibility

A baseline feature participates only with exact feature_id, semantic_version,
resource compatibility, and runtime compatibility for runtime-sensitive
features. An incompatible feature is omitted with BASELINE_INCOMPATIBLE; one
incompatible feature does not invalidate the baseline.

### 13.13 Large baselines

`stylog[data]` MAY persist one feature's ascending values as a
content-addressed Parquet value resource (a single non-null float64 column
named `value`) referenced from the canonical baseline manifest: identity and
version, source manifest SHA, feature id + semantic version, compatibility
identity, row/value count, Parquet logical schema version, and the Parquet
resource SHA-256. No absolute paths. Readers MUST validate the single-column
non-null float64 shape exactly.

## 14. Canonical serialization and hashing

### 14.1 Canonical form

RFC 8785 (JCS) via `rfc8785==0.1.4`.

### 14.2 Portable JSON exclusions

Stylog portable JSON MUST NOT contain: null; NaN or infinity; lone
surrogates; negative-zero floats; integers outside the JCS safe range
(magnitude <= 2^53 - 1); absolute local paths; source text; stack traces;
timestamps; process IDs, usernames, hostnames, the current working directory,
or argv; secrets or environment values.

### 14.3 No normalization

No Unicode normalization is applied anywhere.

### 14.4 Float normalization

`value == 0.0` normalizes to `0.0` (removing -0.0) both at model validation
and at serialization. All feature algorithms produce finite floats or a typed
non-ok status.

### 14.5 Array order

Every portable array has a normative order; scientific sets are sorted before
becoming arrays. JCS sorts object keys (UTF-16 code-unit order) and preserves
array order.

### 14.6 Byte forms

`canonical_bytes(model)` = `rfc8785.dumps` of the validated, dumped portable
tree (with the section 14.2 restrictions enforced). A standalone `.json` file
is `canonical_bytes` + exactly one LF; the trailing LF is NOT part of any
hash. JSONL is one canonical object per line + LF, with no blank lines.
Parsing strips trailing LF characters and then requires one single-line JSON
object: any interior line break or trailing garbage is rejected. Writes are
atomic (section 19.11).

### 14.7 Canonical ordering

Features by feature_id ascending (Unicode scalar comparison before JCS);
analyzers by analyzer_id; resources by id; diagnostics by (severity
error > warning > info, code, artifact_id, analyzer_id, feature_id,
canonicalized context); categories by key; evidence members by member_id;
embedded analyses by (source_span.start.line, source_span.start.column,
embedded_kind, ordinal); comparison families by family name and components by
feature_id; baseline features by feature_id; benchmark artifact IDs in
lexicographic Unicode order. Missing optional diagnostic fields sort as the
empty string.

### 14.8 scientific_sha256

`scientific_sha256(model)` = SHA-256 over the canonical JCS bytes (the
trailing file LF excluded). This is the identity of every portable artifact,
including the complete-model `verifier_id` and the decision identity of a
Verification.

### 14.9 Volatile metadata

Volatile execution metadata (timings, cache hit, absolute paths, host facts)
lives only in in-memory result metadata and human diagnostics - never in
portable JSON. Cache diagnostics (CACHE_CORRUPT, CACHE_READ_FAILED,
CACHE_WRITE_FAILED) travel in the result's warning channel, not inside the
portable fingerprint.

### 14.10 Semantic equality

Semantic equality of stored artifacts is identical validated JCS bytes. There
is no epsilon equality for stored artifacts.

## 15. Provenance, compatibility, and versioning

### 15.1 Provenance versus compatibility

Package provenance is not scientific compatibility. A compatibility signature
includes ONLY fields capable of affecting a feature's meaning or value:
feature_id, semantic version, the analyzer's scientific_compatibility_id, the
relevant resource/parser-grammar/model signatures, and the relevant runtime
fields. Whole-environment dumps are never used.

### 15.2 Compatibility identifiers

Stylog-issued scientific compatibility IDs in v0.1: `stylog.text-core/1`,
`stylog.code-surface/1`, `stylog.python-native-tokenize/1`,
`stylog.python-native-ast/1`, `stylog.tree-sitter.<language>/1` for each
supported tree-sitter language, `stylog.sklearn-vectorizers/1`, and
`stylog.verifier.logreg/1`. A compatibility ID changes only when conformance
shows stored values changed.

### 15.3 Grammar manifest

One checked manifest (`stylog.tree_sitter.grammar_manifest`, resource
`src/stylog/resources/grammar_manifest.json`) records, per supported
language: language ID, grammar_id, grammar package name and module, supported
version range, installed version, upstream repository identifier, upstream
revision when known, node-types.json SHA-256 (or equivalent), the Tree-sitter
ABI version, and the Stylog parser compatibility ID. The manifest is
canonical JSON + one LF, carries `manifest_version` and `node_types_source`,
and is content-hashed; `tools/generate_grammar_manifest.py` regenerates it
after grammar upgrades and the manifest test asserts an exact match against
the installed grammar packages.

### 15.4 Runtime signature

`platform.python_implementation()`, `platform.python_version()`,
`sys.implementation.cache_tag`, and `unicodedata.unidata_version`
(section 5.5).

### 15.5 Resource identity

Every non-code resource resolves through the resource port to a
`ResourceSignature` (id, version, sha256). Resource hashes are the SHA-256 of
the exact stored resource bytes (for checked package resources: the file
bytes, including the trailing LF of canonical JSON resources). Resource
resolution records sorted relative paths and per-file SHA-256 digests; no
absolute paths. A hash mismatch against a pinned identity is
RESOURCE_MISMATCH.

### 15.6 Observation versioning

Every FeatureObservation serializes `feature_id` and `semantic_version`
directly (section 5.9), so aggregation (11.2), comparison (12.2), profiling
(13.12), and verification (23.17) gate on exact versions per observation.

### 15.7 Version bump rules

- Byte-identical refactor -> implementation patch only.
- Value-affecting dependency change -> new compatibility ID before release.
- Tokenizer/segmenter/top-code/denominator change -> incompatible feature
  semantic bump.
- New feature -> registry minor bump at semantic 1.0.0.
- Feature removal or rename -> incompatible registry/schema bump.
- Portable schema shape change -> incompatible schema bump.

## 16. Configuration

### 16.1 Format and validation

Configuration is TOML only, parsed with `tomllib` and validated by a strict
Pydantic model. Unknown keys are errors. The `version` key, when present,
MUST be `1`; omission defaults to 1.

### 16.2 Discovery and precedence

Precedence, highest first: CLI flags > explicit supported environment
overrides > `./stylog.toml` > `[tool.stylog]` in `./pyproject.toml` >
built-in defaults. `--config PATH` or `STYLOG_CONFIG` disables discovery
(a missing explicit file is a configuration error). Only the current working
directory is inspected.

### 16.3 Core keys

```toml
version = 1
[input]            text_encoding="utf-8", include_hidden=false,
                   include=["**/*.py", "**/*.js", "**/*.mjs", "**/*.cjs",
                            "**/*.ts", "**/*.tsx", "**/*.c", "**/*.rs",
                            "**/*.txt", "**/*.md", "**/*.rst"],
                   exclude=["**/__pycache__/**", "**/node_modules/**",
                            "**/venv/**", "**/dist/**", "**/build/**"],
                   max_file_bytes=8388608, max_files=10000,
                   max_total_bytes=536870912
[analysis]         language="und", export_content_hashes=true
[analysis.text]    enabled=true, function_words_en=true, window_ttr_100=true
[analysis.code]    enabled=true
[analysis.code.python]
                   enabled=true, max_ast_bytes=2097152, max_ast_nesting=200,
                   embedded_text=true, max_embedded_artifacts=5000
[analysis.code.python.embedded_text_language]
                   language="und"
[analysis.code.tree_sitter]
                   enabled=true
[execution]        mode="serial", workers=0, max_in_flight=0
[cache]            enabled=true
[baseline]         search_paths=[]
```

### 16.4 Optional capability blocks

Available only when the capability is installed: `[nlp] enabled=false,
model=""`; `[ml] enabled=false`; `[data] parquet_compression="zstd",
row_group_size=65536`. A block present without its capability installed is
rejected with CapabilityUnavailableError (section 4.12).

### 16.5 analysis_config_sha256

`analysis_config_sha256` covers ONLY value/status-affecting settings:
`input.text_encoding`; `analysis.language`; `analysis.text.*`;
`analysis.code.*` (including AST resource limits and the embedded-text
language); the active tree-sitter and linguistic scientific settings (the
`nlp` block participates only when enabled). Excluded: file discovery
patterns, max_files/max_total_bytes, cache location and enablement,
`export_content_hashes`, baseline search paths, output format,
color/verbosity, absolute paths, worker counts, and parquet settings.

### 16.6 Environment variables

`STYLOG_CONFIG` (explicit config path), `STYLOG_CACHE_DIR` (cache root
override), `STYLOG_NO_CACHE` (parsed case-insensitively:
0/false/no/off vs 1/true/yes/on; anything else is a configuration error), and
`STYLOG_FITS_DIR` (representation fit-state root override, section 20.7).

## 17. Cache

### 17.1 Root

The cache root is `platformdirs.user_cache_path("stylog")/"v1"`;
`STYLOG_CACHE_DIR` or `--cache-dir` replaces the root. The root directory
version is independent of the key-format version (17.2); a future
cache-incompatible change MUST bump the key prefix and MAY bump the root.

### 17.2 Key

The fingerprint cache key is scientific identity, never file identity:

```text
key = SHA256(
    "stylog-cache-v2" NUL
    raw content sha256 (32 bytes)
    kind NUL language NUL
    analysis_config_sha256 (32 bytes)
    schema_version NUL
    for each (analyzer_id, implementation_version), sorted:
        analyzer_id NUL implementation_version NUL
    for each (resource_id, version, sha256), sorted:
        resource_id NUL version NUL sha256 (32 bytes)
    for each (field, value) of the RuntimeSignature, sorted by field:
        field NUL value NUL
)
```

Kind and language are part of the scientific identity: language-gated
analyzers produce different observations per language, so identical content
under different kinds or languages MUST NOT share a cache entry (the v2 key
format). All four runtime-signature fields are always included. No
timestamps, paths, or mtimes participate. joblib/diskcache/SQLite MUST NOT
define fingerprint cache identity.

### 17.3 Stored object

A cache entry is the same canonical portable object (canonical bytes + one
LF). The cache always stores the full content-hash object: content-hash
suppression is an export-only concern (section 22.3) and never changes the
internal key or the stored bytes.

### 17.4 Layout

`<root>/objects/ab/cdef....json` - the first two key characters form the
shard directory.

### 17.5 Atomic writes

Write to a temp file in the same directory, write canonical bytes + LF,
flush, `os.fsync`, close, `os.replace`, then a best-effort directory fsync on
POSIX, with cleanup on failure. No locks are taken. Permissions are
best-effort 0700 directories and 0600 files on POSIX.

### 17.6 Read validation

On read: parse, validate against the model, and recompute compatibility. A
corrupt or mismatched entry produces a CACHE_CORRUPT warning, is removed or
ignored, and the analysis is recomputed - a bad cache entry never fails an
analysis. A cache hit MUST rewrite the descriptor's `artifact_id` to the
requesting artifact (artifact_id is instance metadata, section 18.1);
duplicate-content artifacts never receive a foreign instance id.

### 17.7 Failure and bypass modes

A cache read failure produces CACHE_READ_FAILED and is treated as a miss; a
write failure produces a CACHE_WRITE_FAILED warning and the valid result is
returned anyway. `--no-cache` performs no reads or writes. `--refresh` skips
reads, recomputes, and atomically overwrites. The cache schema is disposable:
a future incompatible version changes the key prefix (17.2).

## 18. Ingest, files, languages, and parallelism

### 18.1 Content identity

File and bytes input: SHA-256 over the exact raw bytes, including any BOM.
In-memory `str` input: reject lone surrogates and hash the strict UTF-8
encoding. Raw content is NEVER a field of a portable model. The runtime
artifact (`RuntimeArtifact`, a frozen dataclass, not Pydantic) carries the
descriptor fields plus `raw_bytes` and `text` and is never reachable from
model_dump trees. `artifact_id` identifies the artifact instance (a file
name, a root-relative path, `stdin`, or an API label); it is instance
metadata, not content identity (section 17.6).

### 18.2 Text decoding

UTF-8 strict by default; a UTF-8 BOM decodes via `utf-8-sig` (content
identity retains the BOM bytes); an explicit Python codec via
`codecs.lookup` is allowed for ordinary text; errors are always strict.

### 18.3 Python source decoding

Python bytes follow section 8.1 (`tokenize.detect_encoding`); a user text
encoding override does not apply.

### 18.4 Tree-sitter source decoding

Tree-sitter source is UTF-8 strict; the parser consumes the exact raw bytes.

### 18.5 Invalid input

U+0000 in decoded text -> INPUT_NUL and no analyzers run; a lone surrogate in
in-memory text -> INPUT_SURROGATE; a decode failure -> INPUT_DECODE_ERROR (or
PYTHON_ENCODING_ERROR for Python source). There is no probabilistic binary
detection.

### 18.6 File input

A symlink passed explicitly as input is rejected (SYMLINK_REJECTED); a
missing file is INPUT_NOT_FOUND; an oversized file is INPUT_TOO_LARGE.

### 18.7 Standard input

Read from `sys.stdin.buffer`; defaults are kind=text, language=und,
encoding=utf-8. Code on stdin requires explicit `--kind code --language`.
Only one stdin source is accepted per invocation.

### 18.8 Extension mapping

`.py` -> code/python; `.js`/`.mjs`/`.cjs`/`.jsx` -> code/javascript;
`.ts`/`.tsx` -> code/typescript; `.c` -> code/c; `.rs` -> code/rust;
`.txt`/`.md`/`.rst` -> text/und. Ambiguous extensions (for example `.h`) are
not guessed: an unknown extension with `kind=auto` is INPUT_UNSUPPORTED, and
explicit `--kind`/`--language` always win. For code input with an unmapped
extension, an explicit language is required.

### 18.9 Directory traversal

Recursive. Symlinks are never followed: a symlinked input is rejected (18.6);
a symlink encountered during traversal is skipped with a SYMLINK_REJECTED
diagnostic. Any hidden path component (a segment starting with `.`) is
excluded by default before glob matching.

### 18.10 Glob semantics

Globs are case-sensitive on all operating systems and use `/` separators;
`*`, `?`, and `[abc]` match within one segment; `**` is a whole segment
matching zero or more segments. Patterns are root-anchored. There is no
gitignore support. Evaluation order: hidden exclusion, then include, then
exclude; exclude wins.

### 18.11 Selection order and artifact IDs

Selected paths are sorted by Unicode scalar lexical order of the normalized
POSIX relative path. Artifact IDs and ordinals are assigned before parallel
dispatch. Absolute roots are never portable: directory members carry their
root-relative path as `artifact_id`; single-file inputs carry the file name.

### 18.12 Collection mode

Without `--collection`, inputs are independent artifacts. With
`--collection`, the inputs form one EvidenceSet with explicit linkage kind
and source (no implicit same-author linkage). When any member fails ingest,
the successful analyses MAY be returned but the aggregate is omitted.

### 18.13 Selection limits

Directory selection fails before analysis when it exceeds
`input.max_files` (DIRECTORY_TOO_MANY_FILES) or `input.max_total_bytes`
(DIRECTORY_TOO_MANY_BYTES).

### 18.14 Input safety limits

Defaults: 8 MiB per file; 10000 files; 512 MiB total; 2 MiB Python AST
source; 200 AST nesting guard; 5000 embedded artifacts per parent.

### 18.15 Streaming batch

Batch APIs are iterators/generators with memory bounded by in-flight work
(`analyze_iter`, `fingerprint_iter`; section 20.5).

### 18.16 Parallelism

Execution uses `concurrent.futures.ProcessPoolExecutor`. The coordinator
assigns stable ordinals before dispatch, bounds in-flight work, and emits
results in ordinal order; aggregation uses the normative reducers over a
stable sorted contribution order, never completion order. Worker/PID/host/
timing facts never enter artifacts. Serial and process paths run the same
scientific functions; changing the worker count MUST NOT change outputs or
hashes (section 25.26).

### 18.17 Incremental reuse

Corpus-level incremental reuse (for example the parquet fingerprint index in
`stylog[data]`) is keyed by Stylog scientific identity - content SHA-256 and
`scientific_sha256` - never by file mtime or path.

## 19. CLI contract

The CLI is Typer-based with thin callbacks; every command delegates to the
application layer (section 20.2). Terminal renderers are ASCII-only (Windows
cp1252 consoles).

### 19.1 Command inventory

Commands: `fingerprint`, `analyze`, `profile`, `compare`, `report`,
`benchmark`, `represent`, `verify`, `fit`, `info`. Compatibility behavior:
`verify-fit` is a hidden alias of `fit` and `capabilities` is a hidden alias
of `info`, with identical behavior. No `attribute`, `detect`, or `anonymize`
aliases exist. Every command that writes files accepts `--output` with the
`-o` short form, plus `--force`.

Bare `stylog` prints a concise command landing screen and exits 0. Root and
command help accept both `-h` and `--help` and exit 0; version accepts `-V` and
`--version`. An unknown command is a usage error (exit 2). Root help uses
plain ASCII, includes examples, and points to command-specific help.

### 19.2 fingerprint

`fingerprint INPUT...` over files, one directory, or stdin. Default format:
canonical JSON for a single input, JSONL for multiple.
`--format json|jsonl|terminal|parquet` (`parquet` requires the `data`
capability, writes to `--output`, and is an analytics export, not canonical
interchange). Options: `--output PATH` (`-o`), `--force`, `--kind text|code`,
`--language VALUE`, `--collection`, `--linkage`, `--linkage-source`,
`--workers N`, `--no-cache`, `--refresh`, `--cache-dir PATH`,
`--no-content-hash`, `--config PATH`, `--nlp-model MODEL` (requires the `nlp`
capability; a locally provisioned model resource only). With `--collection`
and successful members, an EvidenceAggregate is appended to the output.

### 19.3 analyze

`analyze INPUT` (one file, one directory, or stdin): fingerprints plus
embedded analyses, plus an EvidenceSet aggregate when `--collection` is
given, plus a profile when `--baseline REF` is given (never an implicit
baseline; `--baseline` requires exactly one input artifact and cannot combine
with `--collection`). Default format is terminal; machine formats emit the
bundle(s) then the aggregate and profile when present.

### 19.4 profile

`profile SOURCE --baseline BASELINE` or `profile RESULT.json --from-artifact
--baseline BASELINE`. The baseline reference is mandatory.

### 19.5 compare

`compare LEFT RIGHT` or `compare LEFT.json RIGHT.json --from-artifacts`
(subjects: two fingerprints, two bundles, or two aggregates; section 12.11).
Default terminal; `--format json` emits the canonical Comparison. There is
never a global similarity.

### 19.6 verify

`verify LEFT RIGHT --model MODEL.json (-m)` or `verify LEFT.json RIGHT.json
--from-artifacts --model MODEL.json`. The model is mandatory (a usage error,
exit 2, otherwise); it is loaded as portable `stylog.verifier-fit` JSON and
fully validated. Compatibility gates (section 23.17) are hard errors (exit
4). Default terminal: an ASCII rendering showing the verdict, the score and
the probability-if-present (explicitly marked absent on insufficient
evidence), features used/missing, the model id, and both evidence hashes.
`--format json` emits the canonical Verification. Exit 0 on success,
including abstain.

### 19.7 fit

`fit TRAINING.toml --output MODEL.json (-o) [--force]`: fits a verifier from
a `stylog.verifier-training` manifest (dataset reference + `[verifier]` block
+ `[[pair]]` populations; sections 21.13 and 23). Emits ONE canonical
VerifierFit JSON (atomic write, no sidecar). Fit diagnostics (pair/feature
eligibility counts, degenerate scales, threshold-band collapse,
unconstrained-language warning) go to stderr; stdout stays empty and the
`verifier_id` is printed to stderr. `verify-fit` is a hidden alias with
identical behavior.

### 19.8 report

`report RESULT.json`: renders an existing validated portable artifact
(fingerprint, bundle, comparison, profile, aggregate, or verification have
dedicated renderers; other portable schemas render a validated generic
summary). No re-analysis, no cache, no baseline lookup.

### 19.9 benchmark

`benchmark SPEC.toml`: runs a declarative benchmark spec (section 21); TOML
only.

### 19.10 represent and info

`represent INPUT...` (requires the `ml` capability): exactly one of
`--fit-output FIT.json` (fit; requires `--representation
char-ngram-count|word-ngram-count|char-tfidf|word-tfidf`) or
`--fit-resource FIT.json` (aliases `--model`, `-m`; transform). Emits
portable RepresentationFit / Representation objects, never sklearn objects.

`info`: a local-only capability report: version, supported core languages,
Python parser/runtime compatibility, Tree-sitter grammar identities,
installed optional capabilities, provisioned representation implementations,
verification implementation identity and fitted-model availability, and the
scientific compatibility IDs. It never
touches the network and never loads or enumerates provisioned NLP models.
`capabilities` is a hidden alias with identical behavior.

### 19.11 Output discipline

Machine modes write only machine data to stdout; diagnostics and progress go
to stderr. Output files are written atomically (temp file in the same
directory, flush, fsync, os.replace); an existing output file is refused
without `--force`. Stdout is empty on a successful `--output` write. The
`parquet` format exists only on `fingerprint` (section 19.2); every other
command rejects `--format parquet` as a usage error (exit 2).
Terminal output is sparse, aligned ASCII intended for humans. `json` emits one
portable JSON object where the command permits a single result; `jsonl` emits
exactly one compact JSON object per line. Human labels, spacing, and tables are
presentation and MUST NOT be parsed as a machine interface.

### 19.12 Exit codes

| Code | Meaning | Raised by |
|---|---|---|
| 0 | success (including scientific warnings, insufficient support, abstain) | - |
| 2 | CLI usage, configuration, capability, verifier-fit failure | typer usage errors, ConfigurationError, CapabilityUnavailableError, VerifierFitError |
| 3 | input/read/decode/unsupported (including partial-batch failures) | InputError (DecodeError, UnsupportedInputError) |
| 4 | invalid portable artifact, baseline, resource, or model incompatibility | PortableArtifactError, BaselineError, ResourceError, ModelIncompatibilityError |
| 5 | unexpected internal or analyzer failure | InternalStylogError and unguarded exceptions |
| 6 | benchmark manifest/data/checksum/split errors | BenchmarkError |
| 7 | resource-limit violation | ResourceLimitError |
| 130 | interrupted | KeyboardInterrupt |

Typed ordinary feature missingness and parser errors alone do not cause a
nonzero exit. A non-collection batch continues after per-file ingest
failures, returns the successes plus diagnostics, and exits 3 when any input
failed; a collection omits the aggregate on member failure (section 18.12).

## 20. Python API, application, and bootstrap

### 20.1 Public API

Exported lazily from the `stylog` top level (importing the package stays
light, section 4.14); thin convenience wrappers over the application layer:

- `fingerprint_file(path, *, kind="auto", language="auto", config=None)`
- `fingerprint_text(text, *, language="und", config=None)`
- `fingerprint_bytes(data, *, kind, language, encoding="utf-8", config=None)`
- `analyze_file(path, *, kind="auto", language="auto", config=None)`
- `analyze_text(text, *, language="und", config=None)`
- `compare_files(left, right, *, config=None)`
- `compare_fingerprints(left, right, *, left_ref="left", right_ref="right")`
- `profile_fingerprint(fingerprint, baseline_ref, *, config=None,
  subject_ref="subject")`
- `build_baseline(fingerprints, *, baseline_id, baseline_version="1.0.0",
  kind="text", language="und", domain="general", source="local")` (sections
  13.8 and 13.10)
- `verify_fingerprints(left, right, model, *, left_ref="left",
  right_ref="right")` and `verify_files(left, right, model, *, config=None)`
  (section 23)
- `fit_verifier(spec, pairs, *, calibration_pairs=None,
  tuning_manifest_sha256=None, config=None)` and `load_verifier(path)`
  (section 23)
- `fit_representation(spec, corpus, *, config=None)` and
  `transform_representation(fit_or_spec, subject, *, config=None)` (section
  20.7)

There are NO path-versus-literal heuristics anywhere in the API.

### 20.2 Application use cases

CLI and API both delegate to the application layer:
`fingerprint_artifact(artifact, *, config, services, ctx)`;
`analyze_artifact`; `aggregate_evidence`; `compare_subjects`;
`profile_subject`; `verify_subjects`; `fit_representation`;
`transform_representation`; `run_benchmark`. These are functions with small
immutable request/result structures, not service hierarchies.

### 20.3 RuntimeArtifact

`RuntimeArtifact` is a `@dataclass(frozen=True)` carrying the descriptor
fields, `raw_bytes`, `text`, and `content_sha256`. It is not a Pydantic model
and is never reachable from model_dump trees (section 18.1).

### 20.4 Bootstrap

`bootstrap.py` builds the default local services (filesystem cache, local
baseline resolver, package resource resolver) and the analysis context
(config, runtime signature, resolved resource handles). There is no global
singleton and no DI framework.

### 20.5 Batch API

`analyze_iter(artifacts, *, config, execution="serial", workers=0,
max_in_flight=0)` yields `(AnalysisBundle, internal_error)` and
`fingerprint_iter(...)` yields bundles, both in ordinal order with memory
bounded by in-flight work. Serial and process modes are identical and
ordered (section 18.16).

### 20.6 Capability availability

Optional capabilities are resolved lazily at first use; a missing capability
raises `CapabilityUnavailableError` (section 4.12). Feature-level gaps are
typed observations (section 10.1), never exceptions.

### 20.7 Representation API (ml)

`fit_representation(spec, corpus, *, config=None)`;
`transform_representation(fit_or_spec, subject, *, config=None)`.
Representation IDs: `stylog.representation.char_ngram_count/1`,
`stylog.representation.word_ngram_count/1`,
`stylog.representation.char_tfidf/1`, `stylog.representation.word_tfidf/1`;
`semantic_version` and `preprocessing_version` are `"1.0.0"`; backend
`scikit-learn` with scientific compatibility id
`stylog.sklearn-vectorizers/1`.

v0.1 representation semantics (all sklearn parameters explicit, never
defaults): char representations consume the exact decoded Unicode text - no
normalization, no lowercasing, no accent stripping (`analyzer="char"`,
`lowercase=False`, `strip_accents=None`, `preprocessor=None`), with
`ngram_range=(3,5)`; word representations consume the precomputed Stylog WORD
token sequence, casefolded token by token, through an identity tokenizer with
`ngram_range=(1,3)` - sklearn's default token regex is NEVER used
(`token_pattern=None`). Count representations use raw counts (`binary=False`)
with no length normalization. TF-IDF: `use_idf=True`, `smooth_idf=True`,
`sublinear_tf=False`, `norm="l2"`; no stop words, no min_df/max_df pruning,
no max_features truncation; dtype float64. The fitted vocabulary is
canonicalized to Unicode scalar lexical term order after fitting; sparse
dimensions and IDF values are reordered to the canonical vocabulary before
hashing and serialization. The fit corpus is processed in deterministic
subject-ID order. The fitted state records the representation id/version, all
parameters, the fit corpus/manifest hash, the canonical vocabulary hash, the
IDF/state hash, and the sklearn scientific compatibility identity.

Fitted state is a LOCAL content-addressed resource under
`platformdirs.user_data_path("stylog")/"fits"/<state_sha256>.json`
(`STYLOG_FITS_DIR` overrides the root), holding canonical JCS bytes + one LF
of `{"representation_id", "semantic_version", "params", "vocabulary",
"idf"?}`; the state sha256 is the SHA-256 of those canonical bytes and equals
the content-addressed file name. Portable representations contain only
coordinates/values plus resource signatures; the raw vocabulary stays in the
local fitted resource. Bulk transform reuses loaded fit state
(`transform_many`-style batching) and MUST produce byte-identical outputs to
repeated single transforms.

### 20.8 Data API (data)

`to_arrow`, `from_arrow`, `to_polars`, `to_pandas`, `write_parquet`,
`scan_parquet`, `scan_parquet_polars`, `read_parquet_objects`,
`fingerprint_index`, `open_corpus`, `query_corpus`, `catalog`,
`write_baseline_values`, `read_baseline_values`.
These are runtime conveniences returning external objects (Arrow tables,
frames, connections); every scientific row retains `schema`,
`schema_version`, `scientific_sha256`, and `canonical_json` (section 5.22).

### 20.9 Verification API

See section 23 for the decision-layer contract. The API functions are
`verify_fingerprints`, `verify_files`, `fit_verifier`, and `load_verifier`
(section 20.1); there is no decision cache (section 23.18).

### 20.10 Exceptions

```text
StylogError                              (base; exit 1)
+-- ConfigurationError                   (exit 2)
+-- CapabilityUnavailableError           (exit 2)
+-- InputError                           (exit 3)
|   +-- DecodeError
|   +-- UnsupportedInputError
|   `-- ResourceLimitError               (exit 7)
+-- PortableArtifactError                (exit 4)
+-- BaselineError                        (exit 4)
+-- ResourceError                        (exit 4)
+-- ModelIncompatibilityError            (exit 4)
+-- VerifierFitError                     (exit 2)
+-- BenchmarkError                       (exit 6)
`-- InternalStylogError                  (exit 5)
```

Feature missingness and feature status are data, not exceptions
(section 10.1). Each exception carries a stable `diagnostic_code` for the CLI
stderr rendering (sections 10.4 and 19.12).

## 21. Benchmark contract

### 21.1 Spec format and tasks

Benchmark specs are TOML only, with header `schema = "stylog.benchmark"`,
`schema_version = "0.1.0"`, `id`, `task`. Tasks are exactly: `split_audit`,
`pairwise_comparison`, `transformation_stability`, `verification`. Results
are portable `stylog.benchmark-result` objects (section 5.23); decision-level
metrics exist only as benchmark outputs and never feed back into domain
semantics.

### 21.2 Dataset manifest

`schema = "stylog.dataset"`, `schema_version = "0.1.0"`, `id`, `version`,
`license`, `redistribution` (`allowed` | `restricted` | `unknown`), `source`.
Each `[[artifact]]` declares `id`, `path` (relative to the dataset root),
`sha256`, `kind`, `language`, and optional context fields: `author_id`,
`domain`, `genre`, `platform`, `repository_id`, `file_id`, `problem_id`,
`framework_id`, `commit_time`, `formatter`, `transformation_id`,
`label_source`, `label_reliability`. Optional `[[transformation]]` entries
declare `original`, `variant`, `transformation_id` (distinct endpoints).

### 21.3 Downloads and checksums

There are no automatic downloads. Artifact files resolve relative to the
dataset manifest; when checksum verification runs, a missing file is
DATASET_MISSING and a hash mismatch is DATASET_CHECKSUM_MISMATCH.

### 21.4 Split declaration

`[split]`: `seed` (string), `train_ppm`/`dev_ppm`/`test_ppm` (integers in
[0, 1000000] summing to exactly 1_000_000 - no float fractions),
`disjoint_by` (a list of context fields), `require_nonempty` (boolean), and
optionally `disjoint_content` (boolean, default false).

### 21.5 Split algorithm

Union-find over artifacts: each `disjoint_by` field unions all artifacts
sharing the same nonempty value; when `require_nonempty` is true, an artifact
missing a `disjoint_by` value is SPLIT_IMPOSSIBLE. With
`disjoint_content=true`, artifacts with identical content hashes are also
unioned (the duplicate-content guard; default off, and it is omitted from the
split-config hash when off so existing goldens stay byte-identical). The
component key is the lexicographically smallest artifact id in the component;
digest = SHA256("stylog-split-v1" NUL seed NUL component_key); bucket =
int_be(digest) mod 1_000_000; ranges [0, train_ppm) -> train,
[train_ppm, train_ppm + dev_ppm) -> dev, else test. All artifacts of a
component land in the same split part. There is no `random` module and no
fallback: an impossible or leaky split fails with SPLIT_IMPOSSIBLE or
SPLIT_LEAKAGE. The post-check verifies that no `disjoint_by` value appears in
two parts.

### 21.6 Split outputs

The result records the sorted artifact IDs per split part, the dataset
manifest scientific hash, the split config scientific hash, and the split
algorithm version `stylog-split-v1`.

### 21.7 Pair validation

Pairs are supplied explicitly as `[[pair]]` `left`/`right`/`label` with
`label` in {`same`, `different`}; both endpoints must exist in the dataset,
left != right, the label must be exact, and no pair may violate the realized
split (both members of a pair must land in the same split part when a split
exists) - otherwise PAIR_INVALID.

### 21.8 pairwise_comparison task

Per feature with enough valid pairs, the result carries: `same_count`,
`different_count`, `same_mean_distance`, `different_mean_distance`,
`same_median_distance`, `different_median_distance`, `roc_auc` (mean/median
omitted when a class has no valid pair; AUC omitted with a
PAIRWISE_AUC_OMITTED diagnostic unless both classes are present). The task
stays descriptive: no EER, no thresholds, no probability. Decision-level
metrics live only in the verification task (21.9).

### 21.9 verification task

The benchmark spec's `[verifier]` block names an explicit VerifierFit model
file; every pair is verified under that model, and decision metrics are
computed as benchmark-only outputs (never domain types): `roc_auc` over
scored rows (omitted with VERIFICATION_AUC_OMITTED unless both classes have
scores), `f1`, `c_at_1`, and `f_05u` using the PAN-derived formulas pinned in
21.12, and Brier loss (`brier`) over calibrated probabilities when present.
These are Stylog metrics, not an assertion of drop-in compatibility with the
official PAN evaluator. Abstentions are counted by the rules in 21.12;
insufficient-evidence rows (no score or probability) are excluded from AUC
and Brier and counted separately
(`abstain_uncertain_count`, `abstain_insufficient_evidence_count`; the
decision counts sum to `pair_count`).

### 21.10 transformation_stability and split_audit tasks

transformation_stability: each `[[transformation]]` supplies original and
variant files; both are fingerprinted, corresponding features are compared,
and per-feature distances/availability are reported; the transformation is
never executed. split_audit: manifest and checksum validation, split
construction, disjointness, and contamination-risk reporting; no content
analysis beyond checksums. `[risks]` block values are descriptive manifest
strings echoed as risk declarations - never claimed as verified.

### 21.11 Pairwise AUC kernel

For one feature over labeled pairs: score = -distance, positive class =
same. The AUC is the Mann-Whitney midrank estimator: sort all scores,
assign midranks within ties, R_pos = sum of positive-class ranks, AUC =
(R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg). Both classes are required:
the kernel raises on a single-class population rather than collapsing to a
constant (the task layer converts that to the omitted-with-diagnostic outcome
of 21.8/21.9).

### 21.12 Decision metrics

The c@1, F1, and F0.5u formulas are adapted from the PAN20-23
authorship-verification evaluators, but Stylog evaluates typed verdict rows
rather than PAN's single numeric prediction field:

- c@1 (Penas & Rodrigo 2011): (1/N) (nc + nu nc / N), where nc counts
  answered-and-correct and nu counts abstentions; every abstention counts as
  unanswered.
- F1: abstentions excluded entirely; standard binary F1 with the same-author
  class as positive; zero-division yields 0.0; omitted when no row was
  answered.
- F0.5u (Bevendorff et al. 2019): 1.25 TP / (1.25 TP + 0.25 (FN + u) + FP),
  where u counts ALL abstentions regardless of true class.
- ROC AUC: the 21.11 kernel over raw scores, positive class = same.
- Brier: the plain mean squared error of probability versus the 0/1 label
  over rows with probabilities; lower is better.

This contract deliberately differs from the official PAN evaluator. PAN
represents a non-answer as prediction 0.5 (and imputes 0.5 for a missing
answer), includes that value in AUC and Brier, and reports the Brier
complement `1 - BS` so higher is better. Stylog preserves the distinction
between an uncertain abstention (score present) and insufficient evidence
(score absent), excludes score-absent rows from AUC, excludes
probability-absent rows from Brier, and reports Brier loss itself. Stylog also
omits an undefined AUC or F1 instead of substituting a constant and does not
compute PAN's rounded aggregate `overall` score. Consequently these metrics
are PAN-derived and intentionally adapted, not evaluator-identical.

### 21.13 Training manifests

`stylog.verifier-training` TOML: `schema`, `schema_version = "0.1.0"`, `id`,
`dataset` (a dataset manifest path relative to the training file), a
`[verifier]` block, and `[[pair]]` entries with `left`, `right`, `label`, and
`population` in {`train`, `tuning`, `calibration`}. The `[verifier]` block
keys: `kind`, `l2_lambda`, `max_iterations`, `tolerance`,
`min_support_fraction`, `min_class_support_fraction`, `min_pairs`,
`threshold_rule`, `threshold_alpha`, `threshold_fixed`,
`calibration_method`, `include_linguistic`, `allow_unconstrained_language`,
`languages` (explicit scope override, section 23.14), `feature_ids`, and a
`[verifier.pair_policy]` sub-table with `max_pairs_per_author`,
`max_pairs_per_problem`, `negative_positive_ratio`, `selection_version`.
Unknown keys are BENCHMARK_INVALID. The train population MUST be nonempty.
Populations are author-disjoint per the four-population discipline
(section 23.15); an evaluation population is never part of a training
manifest - it is evaluated through the verification benchmark task (21.9).
Every fit parameter is an explicit value in `[verifier]`. `stylog fit` does
not search a hyperparameter space or score tuning candidates. If an external
selection procedure used a tuning population to choose those values, tuning
`[[pair]]` entries record that population's content identity in the fitted
model; their feature distances and labels are not passed to the fitter.

## 22. Privacy and security

### 22.1 No network

No deterministic v0.1 operation performs network access: no telemetry, no
update checks, no downloads. The offline gate (section 26.9) blocks sockets
and runs the core workflows.

### 22.2 Portable-output privacy

Portable output omits: absolute paths, the current working directory,
usernames, hostnames, process IDs, source text, raw comment/docstring text,
environment values, repository URLs, secrets, and stack traces.

### 22.3 Content-hash suppression

`export_content_hashes` defaults to true. With suppression (`--no-content-hash`
or configuration), every exported content identity becomes
`{"mode": "suppressed"}`, recursively, including embedded artifacts. The
internal cache key and the stored cache object still use the raw content hash
(sections 17.2-17.3); suppression is export-only.

### 22.4 Source labels

Batch wrappers MAY carry deterministic root-relative source labels (artifact
IDs) outside the Fingerprint identity; these are suppressible by the caller.

### 22.5 Embedded spans

Embedded source spans are portable (they locate, but do not contain, source
text).

### 22.6 Confound tags

The confound tag vocabulary is `stylog.confound-tags/1.0.0`: api_sensitive,
comment_policy_sensitive, content_reduced, content_sensitive,
documentation_policy_sensitive, formatter_sensitive, framework_sensitive,
generated_code_sensitive, language_specific, length_sensitive,
parser_dependent, refactoring_sensitive, repository_sensitive,
resource_sensitive, runtime_sensitive, structural, surface, task_sensitive,
toolchain_sensitive, topic_sensitive, unicode_version_sensitive. Per-feature
assignments are pinned in `analysis/registry.py`.

### 22.7 Code signals posture

Code signals are source observations, never intrinsic programmer traits.

### 22.8 Local-only model loading

NLP models come exclusively from locally installed/provisioned model
packages; there is no download path. An unprovisioned model reference is a
typed RESOURCE_MISMATCH error (exit 4), never a fetch.

## 23. Pairwise authorship verification (Decision layer)

### 23.1 Scope

Pairwise authorship verification answers exactly one question: given two
measured subjects (Fingerprints, or AnalysisBundles reduced to their primary
fingerprints) of the same kind and an explicit fitted VerifierFit model, does
the model support `same_author`, `different_author`, or `abstain`? Verdicts
are model-relative support statements, NEVER identity claims. There is no
attribution, retrieval, style-change detection, or machine-vs-human
detection. There is no default model, no model search path, and no downloaded
weights: the model is always an explicit local artifact or object.

### 23.2 Decision score

s = sigmoid(w . z + b) in the open interval (0, 1), where z is the model's
ordered feature-distance vector standardized per feature:
z_i = (x_i - mean_i) / scale_i with the embedded training-fitted mean/scale.
The score is monotonic in the fitted linear combination of standardized
distances; it is unitless model output and MUST NOT be documented or surfaced
as a probability. Similarity is never introduced anywhere.

### 23.3 Calibrated probability

Present only when the model carries explicit Platt calibration state (a, b
fitted on a disjoint calibration split with hyperparameters already frozen)
AND a score exists: probability = sigmoid(a logit(s) + b). Only `probability`
may be interpreted as an estimated same-author probability, and only under
the model's stated kind/language assumptions. score != probability is
enforced structurally: they are separate omissible fields, and
`calibration_method` is mirrored onto the Verification so that "probability
implies explicit calibration" is locally checkable; a full audit recomputes
against the VerifierFit named by `verifier_id`. Calibration never changes the
verdict - the threshold band applies to `score`.

Prevalence caveat (normative): a calibrated probability is conditional on the
calibration population - its class prevalence, domain, and language mix.
Platt scaling fits the (logit, label) relationship of THAT calibration split;
it is not a universal real-world same-author prior and MUST NOT be presented
as one. Calibration prevalence is recorded in validation reports, and reuse
of a calibrated model on a differently-prevalenced population is a new
evaluation question, not an assumption.

### 23.4 Decision band

score >= t_same -> `same_author`; score <= t_diff -> `different_author`;
otherwise `abstain` with `abstain_reason = "uncertain"` (score present;
probability present iff calibrated). Under the deterministic collapse rule
(23.9) t_same == t_diff == t*; the same-check runs first, so the boundary is
deterministic.

### 23.5 Insufficient evidence

If any model feature's observation is present but non-ok on either side, is
dropped by the pairwise compatibility gate (11.2/12.2), or was measured under
a different semantic version: `verdict = abstain`,
`abstain_reason = "insufficient_evidence"`, `features_missing` lists the
missing features (sorted), and score/probability are ABSENT - there is no
complete evidence vector, so there is nothing to score. Uncertainty is never
encoded as 0.5; incompatibility is never encoded as abstain; capability gaps
are never zero-filled.

### 23.6 Typed non-verdict outcomes

A model feature entirely ABSENT from a fingerprint (a capability or
configuration gap - for example `text.linguistic.*` without the nlp
capability, or an English-scoped feature under a non-English fingerprint) is
a `CapabilityUnavailableError` (exit 2), never a silently reduced model. A
model-id mismatch, kind mismatch, language-constraint violation,
feature-registry-version mismatch, per-feature semantic-version or metric
mismatch, or backend scientific-compatibility-id mismatch is a
`ModelIncompatibilityError` (exit 4, MODEL_INCOMPATIBLE). A malformed model
file is a `PortableArtifactError` (exit 4). Unreadable or undecodable input
uses the existing InputError family (exit 3).

### 23.7 Evidence binding

Every Verification carries `left_fingerprint_sha256` and
`right_fingerprint_sha256` (the scientific hashes of the two primary
fingerprints used) and `verifier_id` (the scientific hash of the complete
VerifierFit). `left_ref`/`right_ref` are caller-supplied readability labels
only and are never canonicalized in v0.1. For AnalysisBundle inputs the bound
hashes are those of the PRIMARY fingerprints (bundle embedded sections are
not verifier inputs in v0.1).

### 23.8 Symmetry contract

All registry comparison metrics are symmetric, so verify(A, B) and
verify(B, A) MUST agree on every scientific field (verdict, abstain_reason,
score, probability, calibration_method, features_used, features_missing,
diagnostics, verifier_id) while left_ref/right_ref and
left/right_fingerprint_sha256 swap positions.

### 23.9 Thresholds

A higher score is more same-author evidence.
`threshold_rule = "calibration_quantile_band"` (declared
`threshold_alpha` in (0, 0.5], fitted on the CALIBRATION split with
hyperparameters frozen): t_diff = Q_{1-alpha} of the different-class
calibration scores (at or below -> confidently different); t_same = Q_alpha
of the same-class calibration scores (at or above -> confidently same);
type-7 quantiles. Collapse rule (deterministic): if t_diff > t_same, the band
collapses to t* = (t_diff + t_same) / 2, recorded as t_diff = t_same = t*
with fit diagnostic THRESHOLD_BAND_COLLAPSED.
`threshold_rule = "fixed"` uses the explicit declared `threshold_fixed` in
(0, 1) with t_same = t_diff = t. Thresholds are NEVER derived from training
or tuning scores. No default alpha is pinned; the rule and its parameters are
explicit in the fit config.

### 23.10 Fitting

A single algorithm, `VERIFIER_MODEL_ID = "stylog.verifier.logreg/1"` (one
constant, one implementation, no algorithm registry), with
`model_semantic_version = "1.0.0"`. Fully specified, pure Python: no NumPy,
no BLAS, no sklearn in the fit or verify path. Objective: minimize
sum logloss(y_i, s_i) + (lambda/2) ||w||^2 with the intercept unpenalized and
lambda > 0 - the penalized optimum is finite even on perfectly separable
data, so there is NO separation failure rule. Pairs are processed sorted by
(left content sha, right content sha); features in sorted model order; no
shuffling, no RNG. IRLS: parameters start at zero; each iteration recomputes
scores in fixed pair order with the overflow-safe sigmoid (exponent arguments
clamped to +/-700; results rounding to exactly 0.0/1.0 are moved one
representable step inward so the portable 0 < score < 1 invariant always
holds); the gradient Z^T(y - s) and Hessian Z^T W Z accumulate over pairs in
that fixed order via `math.fsum`; the small symmetric system
(Z^T W Z + lambda I~) delta = gradient is solved by specified Gaussian
elimination with partial pivoting (columns left to right; the pivot is the
first row attaining the maximum absolute column value; a pivot magnitude
below the floor 1e-12 is a typed numerical-fit error); convergence is
max|delta| <= `tolerance` (default 1e-12); exhausting `max_iterations`
(default 100) is a typed fit error (VerifierFitError, exit 2) - never a
silent half-fit. Platt calibration uses the same specified Newton rule with
two parameters and no penalty over calibration (logit, label) pairs;
perfectly separable calibration scores fail as typed non-convergence.

### 23.11 Fit determinism scope

Fixed input order plus correctly-rounded summation give byte-identical
repeated fits within the same recorded runtime environment
(`VerifierFit.runtime`: Python implementation/version/cache tag/Unicode DB).
Cross-platform byte-identity is NOT claimed (C libm exp/log may differ in the
last ulp) - the same stance as section 8.19. Worker counts never affect fits
(fitting is serial by construction). Correctness is anchored by differential
oracle tests against sklearn/SciPy in the dev lane (skip-guarded; tolerance
1e-6 on coefficients and 1e-9 on scores).

### 23.12 Feature candidacy and eligibility

Training-only, deterministic, language-aware. The candidate universe is the
registry features with metric != NONE applicable to the model kind, filtered
by: (a) `text.linguistic.*` excluded unless the fit config sets
`include_linguistic = true` (base default is core-only); (b) language-scoped
features excluded unless the model's pinned languages are nonempty and fully
inside the feature's scope - the scope map `FEATURE_LANGUAGE_SCOPE` lives in
`verification/fit.py` as the single documented source (today
`text.function_words.en.*` -> {en}); (c) an explicit `feature_ids` list
bypasses universe construction but still passes both filters and the
kind/metric checks (an unknown, metric-less, or wrong-kind id is a typed fit
error). Eligibility: a training pair supports a feature iff both sides'
observations are `ok` and the compatibility gate passes; a feature is
selected iff supporting/eligible >= `min_support_fraction` overall AND >=
`min_class_support_fraction` within each label class. The selected set is
frozen into `VerifierFit.features` and never re-derived. Training pairs
lacking complete evidence over the selected set are excluded from coefficient
fitting, counted in eligibility, and reported in fit diagnostics;
`eligible_pair_count < min_pairs` is a typed fit failure. Normalization
(per-feature mean and population std, with scale = 1.0 for zero variance plus
a VERIFIER_ZERO_VARIANCE_FEATURE fit diagnostic) is computed on eligible
training pairs only, in fixed order with `math.fsum`.

### 23.13 Pair balance policy

Training manifests are built under a deterministic balance policy embedded in
`fit_config.pair_policy`: candidate pairs are enumerated in a fully specified
sorted order (no RNG); selection within a stratum uses deterministic SHA-256
ranking - selection_key = SHA256("stylog-pair-select/" + selection_version +
NUL + canonical_pair_identity), keeping the N lowest keys; the canonical pair
identity is the canonically ordered sha256_left NUL sha256_right NUL label.
The caps `max_pairs_per_author` / `max_pairs_per_problem` and the
`negative_positive_ratio` cap are applied over the SHA-256-ranked order
within each stratum: positives are taken first in ascending selection-key
order under the caps, then negatives in ascending key order under the caps up
to a budget of int(`negative_positive_ratio` x positive_count); the ratio is
a CAP, never a target - it does not manufacture negatives. This removes
ordering bias (author ids, problem ids, filenames, chronology) while
remaining exactly reproducible.
`selection_version` versions the ranking rule. The deterministic validation
builders (vendored at `tests/verifbuild.py`) additionally assign authors to
author-disjoint populations by SHA-256 hash buckets over author ids
(algorithm id `stylog-verif-split-v1`, the same ppm-bucket construction as
21.5) and drop exact duplicate-content artifacts before population
assignment; these builder rules are validation-lane machinery, not core
semantics.

### 23.14 Language applicability

`VerifierFit.languages` is the sorted set of distinct artifact languages in
the eligible training pairs; a fit spec MAY instead pin `languages`
explicitly, and the pinned set is then used both for candidacy (23.12) and
recorded on the model. An empty (unconstrained) set requires explicit
`allow_unconstrained_language = true` and emits a fit diagnostic
(VERIFIER_UNCONSTRAINED_LANGUAGE). At verify time, a nonempty `languages`
requires both artifacts' languages to be members, else
ModelIncompatibilityError (exit 4). A model pinned outside a feature's
language scope can never contain that scoped feature (23.12).

### 23.15 Four-population discipline

The recommended evaluation workflow uses author-disjoint populations in this
order: TRAIN (eligibility, normalization, coefficients) -> TUNING (external
selection of explicit `l2_lambda`, eligibility fractions, pair-policy
settings, threshold alpha, or other fit choices) -> CALIBRATION (thresholds
and Platt, with fit choices frozen) -> EVALUATION (reported metrics, computed
exactly once). Stylog implements the TRAIN and CALIBRATION stages and the
separate EVALUATION benchmark task. It intentionally provides no
hyperparameter search space, candidate scorer, or internal TUNING stage:
every fit parameter is already explicit in `[verifier]`/`VerifierSpec` before
the fitter starts.

When tuning pairs are present in a training manifest, `stylog fit`
fingerprints them only to compute `tuning_manifest_sha256`; it does not use
their observations or labels to choose or fit any parameter. Presence of that
hash means the caller has declared the identity of an external tuning
population that informed the explicit configuration. Omission means no tuning
population identity was declared or recorded; it cannot prove that no
external model selection occurred. When no calibration split exists,
`threshold_rule = "fixed"` and no probability is ever emitted. All declared
population manifest hashes are recorded in the model
(`source_manifest_sha256`, `tuning_manifest_sha256`,
`calibration_manifest_sha256`); a population manifest hash is
`sha256_of_tree({"pairs": [[left_sha, right_sha, label], ...]})` over the
sorted triples of the two fingerprint scientific hashes and the label, so it
is order-invariant and content-sensitive.

### 23.16 Leakage rules for labeled tasks

No pair (in either orientation) may appear in more than one population; pair
builders assert canonical pair ids unique across populations.
Duplicate-content artifacts are neutralized before splitting (canonical pair
identity over sorted content hashes; builders drop exact duplicates
deterministically; benchmark splits MAY set `disjoint_content = true`,
section 21.5). Statistics discipline per 23.15: eligibility, normalization,
and coefficients come from TRAIN only; any external model selection should
use TUNING only; thresholds and calibration come from CALIBRATION only;
EVALUATION is used exactly once for final reporting. The fitter enforces its
TRAIN/CALIBRATION uses, but it neither performs nor audits the external TUNING
and EVALUATION workflow. Population construction rules live in `benchmark/`,
manifests, and validation builders - never in core domain semantics.

### 23.17 Model compatibility gates at verify time

Hard errors (ModelIncompatibilityError, exit 4): model id equality with
`stylog.verifier.logreg/1`; feature_registry_version equality; per-feature
semantic_version equality; per-feature metric equality with the registry;
kind equality; languages membership (23.14); backend
scientific_compatibility_id equality. Every model feature must exist in the
runtime registry. Per-observation compatibility (the pairwise gate of
11.2/12.2) is enforced inside comparison and surfaces as feature-missing ->
abstain (23.5).

### 23.18 Bump rules and caching

Inheriting section 15.7: a coefficient-affecting algorithm change -> a new
scientific_compatibility_id; a feature-semantics change -> a new
model_semantic_version; a schema shape change -> an incompatible schema bump.
No decision cache exists in v0.1 (scoring is pure-Python O(#features),
sub-millisecond); fingerprint caching is unchanged and verification consumes
it transparently (worker-count and cache-state invariance hold for
Verification bytes).

## 24. Reserved

This section number is reserved for a future decision-layer extension. v0.1
defines no section 24 content; the number is held so that sections 25-27 keep
their historical numbering.

## 25. Conformance fixtures (mandatory goldens)

The following fixtures are mandatory; the test suite pins them.

### 25.1 Tokenizer

`Don't re-enter\u2014now. na\u00efve caf\u00e9 3.14` -> 6 WORDs, 1 NUMBER; `token_kind`
{word: 6, number: 1}. `'dogs' rock\u2019n\u2019roll x_y 1,000.50` -> dogs,
rock\u2019n\u2019roll, x, y, NUMBER 1,000.50.

### 25.2 Combining marks

A letter plus a combining acute accent is one WORD of the full code-point
length; `-12 + 5` -> NUMBER(12), NUMBER(5).

### 25.3 Lexical statistics

`a a b c`: word_count = 4, type_count = 3, TTR = 0.75, hapax_type_count = 2,
hapax_token_share = 0.5, entropy = 1.5 bits, simpson = 0.375.

### 25.4 Sentences

`Dr. Smith left. Value 3.14! Really?` -> 4 sentences per 7.8; an
abbreviation-smart segmenter is a failure.

### 25.5 Paragraphs and whitespace

`a\r\n\r\nbc` behaves per 7.9. All 25 White_Space code points classify into
the exact 7.2 categories with the correct total.

### 25.6 Function words

The resource hash is asserted (6.5). `I would go to the house, but she
wouldn't.` under `en` yields the exact matches; under `und` -> unavailable +
LANGUAGE_UNSPECIFIED; under `fr` -> not_applicable.

### 25.7 Python tokens

`import os as operating_system` / `async def fetch(myURL2: str, *,
retries=3):` / `if retries>0: return f"{myURL2=}"` / `return None # noqa`:
token classes, keywords, operators, string/f-string handling, bindings, the
myURL2 split/style, alias share, async function kind, parameter count, return
count, nonterminal return, control constructs, and the noqa marker.
Runtime-specific goldens are allowed but must be explicit.

### 25.8 Bindings

Every binding role of 8.10, including posonly/kwonly/*args/**kwargs,
tuple/annotated/augmented assignment, walrus, loop/comprehension/with/
exception targets, imports, MatchAs/MatchStar/MatchMapping.rest, "_", a
dunder function, and attribute assignment (not a binding).

### 25.9 AST failure

`def f(` -> generic surface ok, token analyzer parser_error, AST parser_error
or not run, no partial tokens. A tokenize-ok-but-AST-failing source keeps the
token observations and marks AST features parser_error.

### 25.10 AST resource limits

Valid Python above `max_ast_bytes` -> tokens ok, AST unavailable with
PYTHON_AST_RESOURCE_LIMIT.

### 25.11 Embedded comments

shebang + coding cookie + `# First` / `# second` + `x = 1  # inline` +
`# fmt: off` + `# Third` -> comment_block "First\nsecond", inline_comment
"inline", comment_block "Third"; nothing embedded for the shebang, cookie, or
fmt marker.

### 25.12 Docstrings

Nested module/class/sync/async docstrings use `ast.get_docstring(clean=True)`
text with deterministic spans; standalone strings are not docstrings.

### 25.13 Aggregation

ratio 1/10 pooled with 9/90 -> 10/100 = 0.1, with a counterexample where the
unweighted mean differs.

### 25.14 Missing aggregation

ok + parser_error + disabled -> total_samples = 3, contributing_samples = 1,
and the exact missing-status counts.

### 25.15 JSD

[1,0] vs [1,0] -> 0; [1,0] vs [0,1] -> 1; union and tie cases per 12.5.

### 25.16 W1

Point-0 mass 1 vs point-3 mass 1 -> 3; raw 500 vs 999 under top_code 201 ->
both transform to 201 -> distance 0.

### 25.17 Quantiles

[0,10,20,30,40] type 7 -> Q.25 = 10, Q.5 = 20, Q.75 = 30, plus non-knot
cases.

### 25.18 Percentile ties

Baseline [1,2,2,4], observed 2 -> L = 1, E = 2, N = 4 -> 50.

### 25.19 MAD

An all-identical baseline -> robust_z omitted, PROFILE_ZERO_MAD, no inf/NaN.

### 25.20 JCS

RFC 8785 vectors plus non-ASCII keys, escaped controls, -0.0 normalization,
surrogate rejection, NaN rejection, null rejection, feature ordering, and
diagnostic ordering. JCS sorts object properties by UTF-16 code-unit order -
rely on the canonicalizer, never on insertion order.

### 25.21 Cache

Same content under different paths shares a key; a scientific config change
misses; an output-format change hits; content-hash suppression keeps the
internal key; an analyzer version change misses; a kind or language change
misses; a simulated runtime change misses; a corrupt entry warns and
recomputes; a cache hit rewrites only the instance `artifact_id` (17.6);
concurrent writers stay valid.

### 25.22 Directory

Hidden exclusion, symlink rejection, `**` matching zero directories,
case-sensitivity, canonical order, default excludes, duplicate content
preserved as distinct samples, and count/byte preflight failures.

### 25.23 Benchmark split

Fixed IDs/seed/disjoint graph -> hard-coded exact split IDs (21.5).

### 25.24 AUC ties

The midrank implementation of 21.11 handles tied scores exactly.

### 25.25 Numeric differential

Reference versus oracle for quantiles, midranks, MAD, JSD2, W1, and AUC;
integer accumulation never wraps via fixed-width dtypes (section 4.8).

### 25.26 Parallelism

Worker counts 1/2/N produce identical ordered results and hashes (18.16).

### 25.27 Tree-sitter

The grammar manifest hash is checked against the installed packages; the
backend signature is golden; node-type, parent-child, and depth fixtures per
language; malformed input behavior is explicit; a grammar upgrade requires a
differential review before compatibility reuse.

### 25.28 spaCy

Model tree hash fixed; signatures fixed; POS/dependency/morphology goldens;
no download path (22.8).

### 25.29 sklearn

Fit corpus hash and fitted-state hash are stable; n-gram boundaries follow
Stylog semantics (20.7); parameters are explicit; sparse output is sorted and
zero-elided; repeated transforms are identical.

### 25.30 Data roundtrips

Canonical JSON survives Arrow/Parquet/DuckDB roundtrips byte-identical with
the scientific hash unchanged; partitioning and compression never change
identity.

### 25.31 Verification

Score-algebra goldens (23.2); threshold direction and collapse goldens
(23.9); repeated fits, shuffled input order, and fresh processes are
byte-identical within one runtime (23.11); repeated verifications are
byte-identical; the sklearn/SciPy differential oracles hold within their
tolerances (23.11); offline fit + verify succeeds with sockets blocked
(26.9); worker-count and cache-state invariance hold for Verification bytes
(23.18).

## 26. CI and packaging

### 26.1 Supported runtimes

Python 3.12, 3.13, and 3.14 (CPython). `requires-python = ">=3.12,<3.15"`.

### 26.2 Packaging

One wheel and one sdist named `stylog` (section 4.9).

### 26.3 Clean-install gates

Clean-install tests cover base, `nlp`, `ml`, `data`, and `all`, including a
clean-venv offline install of the full extra set.

### 26.4 Base-install language proof

The base install MUST prove that javascript, typescript, c, and rust analysis
works without any extras (section 4.13).

### 26.5 Schema drift gate

Deterministic JSON Schemas (canonical JCS bytes + one LF) are committed under
`schemas/` and generated by `python tools/generate_schemas.py`; CI fails on
drift via `python tools/generate_schemas.py --check` (section 5.23).

### 26.6 Test categories

Unit, conformance, golden, property/invariant, integration,
application/use-case, CLI, benchmark, packaging, offline,
dependency-differential, parallel-determinism, adapter-roundtrip, and
architecture/import-boundary tests.

### 26.7 Determinism gates

Repeat serial runs give identical canonical hashes; reversed input order
gives identical per-artifact hashes; cache cold/warm/disabled runs are
identical; representation refits give identical fit identities.

### 26.8 Lint gate

`python -m ruff check src tests` is clean.

### 26.9 Offline gate

CI blocks sockets and runs all five languages plus aggregate, compare,
profile, benchmark, parallel batch, and verifier fit + verify
(`tests/test_offline.py`, `tests/test_verify_hardening.py`).

### 26.10 Architecture gate

The architecture/import-boundary tests of section 4.15 run as a release gate
(`tests/test_architecture.py`).

## 27. Migration

This is a greenfield repository: no released legacy API or artifacts exist.
The following rules are recorded for future releases: ambiguous path/string
APIs must become explicit; learned vectors never migrate into a Fingerprint;
uncalibrated similarity/probability fields are never populated from
comparisons (probability exists only under the explicit calibration contract
of sections 5.20-5.21: embedded Platt state in the VerifierFit, mirrored
`calibration_method` on the Verification); legacy null features require
re-analysis; new portable objects start at `schema_version` `"0.1.0"`.

## 28. Conformance

Conformance with this specification is checked mechanically: the schema drift
gate (`python tools/generate_schemas.py --check`, section 26.5); the
architecture gate (`tests/test_architecture.py`, section 26.10); the offline
gate (`tests/test_offline.py`, section 26.9); the parallel-determinism gate
(`tests/test_parallel.py`, section 18.16); the conformance fixtures of
section 25 as pinned by the test suite; and the lint gate (section 26.8).
Where this document marks behavior as specified-but-not-implemented (section
12.5 optimized kernels), the reference implementation named here is the only
conforming implementation until a conforming alternative lands. An
implementation that cannot satisfy a MUST or MUST NOT in this document is
non-conforming; an explanatory document that contradicts this document is
wrong and must be corrected.
