# Methodology

This document explains how Stylog works and how to interpret its outputs. For
normative requirements and exact serialization behavior, see
[Specification v0.1](spec-v0.1.md).

## Artifacts

An artifact is one observable input sample: a file, stdin, or a string via the
API. Stylog classifies every artifact by kind (`text` or `code`) and language.
Classification is explicit, never guessed: file inputs map from extension
(`.py`, `.js`/`.mjs`/`.cjs`/`.jsx`, `.ts`/`.tsx`, `.c`, `.rs` are code;
`.txt`, `.md`, `.rst` are text), and `--kind` / `--language` override the
mapping. An unknown extension requires explicit options. Decoding is strict;
Python source uses CPython `tokenize.detect_encoding` exactly.

Natural-language text carries a language tag (default `und`). All core text
measurements run on any tag. Two feature sets are language-sensitive: the
English function-word features measure only `language = "en"` (they report
`not_applicable` for other languages and `unavailable` for `und`), and the
optional `text.linguistic.*` features measure whatever the configured local
spaCy pipeline supports.

Code languages are Python (CPython `tokenize` plus `ast`) and JavaScript,
TypeScript, C, and Rust (Tree-sitter grammars pinned in
`src/stylog/resources/grammar_manifest.json`). A code artifact in any other
language still receives the generic `code.sample.*` and `code.surface.*`
measurements; parser-backed features are not produced.

## Fingerprints

A fingerprint is the portable, deterministic measurement record for one
primary artifact. It contains:

- the artifact descriptor: kind, language, encoding, byte and Unicode
  code-point counts, and the content identity (SHA-256 of the exact input bytes, or a
  suppressed marker when content hashes are disabled);
- the runtime signature (Python implementation, version, cache tag, Unicode
  database version) and the analysis configuration hash;
- one analyzer signature per analyzer (implementation version, feature
  registry version, backend and resource hashes, parser grammar hashes where
  applicable);
- the feature observations, sorted by feature id.

Fingerprints serialize as canonical JSON under RFC 8785 (JCS). The
`scientific_sha256` of an artifact is the SHA-256 of those canonical bytes
(the single trailing LF of a standalone `.json` file is never hashed).
Repeating an analysis with the same artifact descriptor (including
`artifact_id`, kind, language, and encoding), input bytes, runtime signature,
and configuration yields an identical fingerprint. Two differently named
artifacts with identical bytes share a content identity but intentionally have
different fingerprint records and scientific hashes.

## Feature families

In Stylog 0.1.0 the registry (`src/stylog/analysis/registry.py`, version
`stylog.features/1.0.0`) defines 90 features in 15 families. Each feature has
a geometry (`integer`, `float`, `ratio`, `categorical_distribution`,
`ordered_histogram`, `summary`), an aggregation reducer, a comparison metric,
and a support kind.

| Family | Count | Measures | Examples | Extra |
| --- | ---: | --- | --- | --- |
| `text.sample` | 2 | Byte and Unicode code-point counts | `text.sample.byte_count` | base |
| `text.surface` | 6 | Line endings, whitespace, Unicode categories, case, punctuation | `text.surface.line_ending`, `text.surface.punctuation_codepoint` | base |
| `text.lexical` | 11 | Tokens, word length, type-token ratio, hapax, word entropy | `text.lexical.ttr_casefold`, `text.lexical.word_length` | base |
| `text.structure` | 6 | Sentence and paragraph segmentation and lengths | `text.structure.sentence_length_tokens` | base |
| `text.function_words.en` | 2 | English function-word share and lexeme distribution | `text.function_words.en.token_share` | base |
| `text.linguistic` | 5 | UPOS tags, dependency relations and distances, morphology | `text.linguistic.upos`, `text.linguistic.dependency_distance` | `nlp` (spaCy) |
| `code.sample` | 3 | Byte, character, and physical line counts | `code.sample.physical_line_count` | base |
| `code.surface` | 8 | Indentation, line lengths, blank lines, trailing space | `code.surface.indent_kind`, `code.surface.blank_line_share` | base |
| `code.python.lexical` | 6 | Token classes, keywords, operators, number forms, string styles | `code.python.lexical.token_class` | base (Python) |
| `code.python.naming` | 8 | Identifier, binding, and attribute lengths and case styles | `code.python.naming.binding_length` | base (Python) |
| `code.python.syntax` | 3 | AST node and edge distributions, node depth | `code.python.syntax.node_distribution` | base (Python) |
| `code.python.structure` | 17 | Functions, classes, control flow, imports, decorators, nesting | `code.python.structure.function_length_lines` | base (Python) |
| `code.python.comments` | 5 | Comment and docstring counts, shares, lengths, kinds | `code.python.comments.comment_length` | base (Python) |
| `code.toolchain` | 1 | Toolchain markers in Python comments | `code.toolchain.python_comment_marker` | base (Python) |
| `code.parser` | 7 | Tree-sitter named nodes, identifiers, comments (JavaScript, TypeScript, C, Rust) | `code.parser.named_node_type`, `code.parser.identifier_length` | base |

Only the `text.linguistic` family requires an optional extra (`nlp`, a
locally provisioned spaCy model). Every other family is part of the base
install; Tree-sitter grammars are base dependencies pinned by version range.

## Feature status

Every feature observation carries a typed status; a non-ok observation never
encodes a placeholder value:

- `ok` -- measurement completed; value and support are defined (a valid zero
  count is `ok`).
- `insufficient_support` -- the concept applies but the event population was
  zero (type-token ratio on an empty word stream).
- `not_applicable` -- the concept does not apply to this artifact (English
  function words on French text).
- `unavailable` -- a non-parser prerequisite was missing (function words under
  `language = "und"`, a missing resource, an analyzer internal error).
- `parser_error` -- the required tokenizer or parser rejected the input.
- `disabled` -- configuration explicitly disabled the owning analyzer.

## Embedded artifacts

`analyze` returns an `AnalysisBundle`: the primary fingerprint plus separate
fingerprints for embedded natural-language text. For Python code, when
`analysis.code.python.embedded_text` is enabled (the default), Stylog extracts
comment blocks, inline comments, and docstrings as embedded text artifacts and
fingerprints each one independently. Embedded content identity is the SHA-256
of the cleaned text's exact UTF-8 encoding; embedded features never mix into
the parent code fingerprint's namespaces.

## Comparing fingerprints

`compare` computes per-feature distances between two fingerprints of the same
kind. There is deliberately no global similarity score: distances are never
averaged or combined, and text and code fingerprints never cross-compare. The
registry assigns each feature one metric:

- `ABS` -- absolute difference of scalar values (`|a - b|`; in proportion
  points on `[0, 1]` for ratios).
- `JSD2` -- Jensen-Shannon distance (base 2) between categorical
  distributions; 0 is identical, 1 is maximally disjoint.
- `W1` -- Wasserstein-1 distance between ordered histograms over their
  top-coded integer support, reported in the feature's native unit.
- `SPD` -- symmetric proportional distance, `2|a - b| / (|a| + |b|)`; provided
  and tested, but no v0.1 feature uses it.
- `NONE` -- no comparison (raw counts such as byte counts are evidence volume,
  not compared quantities).

A feature enters the comparison only when both sides report `ok` under the
same semantic version with compatible resources and runtime. Missing or
non-ok features are omitted from the components and surfaced as diagnostics --
never as zero or maximal distance.

Keep the output quantities distinct; none of them implies the next:

```text
feature value != feature distance != profile percentile != robust z-score
!= verifier score != calibrated probability != proof of authorship
```

## Population baselines

A baseline is a local, versioned reference distribution built from a corpus of
fingerprints. `build_baseline` collects, per feature, the sorted scalar values
of all `ok` observations with a profileable geometry (`integer`, `float`,
`ratio`); each baseline feature records the exact source-unit count and a
compatibility hash, so a baseline is bound to the registry version, analyzer
versions, resources, and runtime that produced it. A baseline requires at
least one unit; beyond that there is no minimum size -- every profile
observation reports the exact `baseline_n`. (The registry's reducer types --
`exact_sum`, `ratio_pool`, `categorical_pool`, `histogram_pool`,
`sample_summary`, `not_aggregatable` -- govern a different operation: pooling
several artifacts into one evidence aggregate.)

A baseline reference containing a path separator or ending in `.json` resolves
as an explicit path. Otherwise Stylog searches the configured
`baseline.search_paths` and then `platformdirs.user_data_path("stylog") /
"baselines"` for an exact `baseline_id` match; multiple distinct baselines
with one id are an error. Resolution never touches the network.

## Profiles

A profile interprets one fingerprint against one explicit baseline, feature by
feature. For each compatible scalar feature Stylog reports the midrank
percentile, the type-7 quartiles and IQR, the MAD, and a robust z-score:

```text
percentile_midrank = 100 * (L + 0.5 * E) / N     # L below, E equal, N baseline size
mad_raw            = median(|x_i - median|)
mad_normal_scaled  = mad_raw * 1.482602218505602
robust_z           = (observed - median) / mad_normal_scaled
```

When `mad_raw` is zero, `robust_z` is omitted entirely and a `PROFILE_ZERO_MAD`
diagnostic is emitted -- never an infinity. Incompatible or missing features
are skipped with diagnostics, not scored.

## Sparse representations

A representation is a sparse model-space vector with provenance; it is not a
fingerprint and is never a fingerprint feature. Four kinds exist, all
scikit-learn vectorizers with fully explicit parameters, behind the `ml`
extra:

| CLI token | Representation id | Space |
| --- | --- | --- |
| `char-ngram-count` | `stylog.representation.char_ngram_count/1` | character 3-5-grams, raw counts |
| `word-ngram-count` | `stylog.representation.word_ngram_count/1` | word 1-3-grams, raw counts |
| `char-tfidf` | `stylog.representation.char_tfidf/1` | character 3-5-grams, TF-IDF |
| `word-tfidf` | `stylog.representation.word_tfidf/1` | word 1-3-grams, TF-IDF |

Character representations consume the exact decoded text; word
representations consume Stylog's own casefolded word tokens, never sklearn's
default token regex. `fit_representation` learns the vocabulary (and IDF for
TF-IDF kinds) from a corpus and stores it as a local content-addressed state
file; `transform_representation` vectorizes one subject against that state. The
result records the representation id, the fit id, the scikit-learn backend
signature, and a sparse vector (sorted unique indices, zero values omitted).

## Authorship verification

Verification is a decision layer over pairs of fingerprints under an explicit
fitted model. A verifier is trained from a `stylog.verifier-training` TOML
manifest, which references a checksummed `stylog.dataset` manifest and
declares labeled same/different pairs in train, tuning, and calibration
populations. Author-disjoint populations are a construction contract: the
deterministic builders assign authors to populations by SHA-256 hash buckets
and drop pairs whose authors span populations, but the manifest format and
the fit validate structure, labels, and eligibility only -- a manifest built
by other means is trusted (see [Limitations](limitations.md#verification)).
Train drives feature eligibility, normalization, and coefficients;
calibration drives thresholds and Platt scaling with hyperparameters frozen.
Stylog's v0.1 fitter does not search hyperparameters. If a separate, pre-fit
selection procedure used tuning data to choose the explicit manifest values,
the tuning pairs are recorded by content identity only and are not passed to
the fitter.

Candidate features are the registry's comparable features for the model kind,
filtered by language scope and by the `include_linguistic` flag. A feature is
selected only when enough training pairs support it overall
(`min_support_fraction`) and within each class
(`min_class_support_fraction`); pairs without complete evidence over the
selected set are excluded from fitting, and too few eligible pairs
(`min_pairs`) fails the fit. The model (`stylog.verifier.logreg/1`) is
L2-penalized logistic regression over standardized per-feature distances,
fitted by a fully specified pure-Python IRLS solver -- no NumPy, BLAS, or
sklearn in the fit or verify path.

The decision score is the sigmoid of the linear combination, unitless and not
a probability:

```text
z_i   = (x_i - mean_i) / scale_i      # training-fitted mean/scale per feature
logit = intercept + sum(w_i * z_i)    # fixed order, math.fsum, clamped to +/-700
score = 1 / (1 + exp(-logit))         # kept inside the open interval (0, 1)
```

A two-threshold band maps the score to a verdict:

```text
score >= t_same  ->  same_author
score <= t_diff  ->  different_author
otherwise        ->  abstain (abstain_reason = "uncertain")
```

Thresholds come from `threshold_rule = "calibration_quantile_band"` (type-7
quantiles of the calibration-split class scores at a declared alpha; crossed
quantiles collapse both thresholds to their midpoint, recorded in fit
diagnostics) or `"fixed"` (an explicit declared threshold used for both
bounds, so every scored pair receives a verdict). If any model feature is
present but lacks comparable evidence on either side, the verdict is `abstain`
with `abstain_reason = "insufficient_evidence"` and no score at all; a model
feature absent from a fingerprint entirely is a typed capability error, not an
abstain. An abstain is a typed non-decision, not a weak verdict.

When the model carries Platt calibration state fitted on the disjoint
calibration split, a verification also reports:

```text
probability = sigmoid(a * logit + b)   # calibration parameters a, b
```

Only `probability` may be read as an estimated same-author probability, and
only under the calibration population's prevalence and the model's stated
kind/language scope. Calibration never changes the verdict; the threshold band
applies to `score` alone. A `same_author` verdict indicates model-relative
support for the same-author hypothesis under the fitted verifier and its
training distribution. It does not establish identity.

## Provenance and hashing

Every portable artifact records what produced it. Content identity is the
SHA-256 of the exact input bytes. Analyzer signatures record implementation
versions, backend packages and versions, resource identities with hashes (for
example the English function-word list `stylog.function_words.en` version
`1.0.0`), and Tree-sitter grammar identities including node-types and
grammar-manifest hashes. The runtime signature pins the Python implementation,
version, cache tag, and Unicode database version. A verifier's identity
(`verifier_id`) is the `scientific_sha256` of the complete `VerifierFit`,
including its training, tuning, and calibration manifest hashes; each
verification binds the scientific hashes of both input fingerprints and the
model.

## Reproducibility

Stylog is local-first and deterministic by construction. Canonical
serialization (RFC 8785, no nulls, no non-finite floats, normative array
orders) makes semantic equality byte equality. Determinism is scoped to the
recorded runtime signature: an unchanged artifact descriptor, input bytes,
configuration, and runtime signature yield identical fingerprints, but there
is no byte-identity promise across Python versions or platforms (library
`exp`/`log` may differ in the last ulp); fit and verify are byte-identical on
repeated runs within the same recorded runtime. The test suite pins these
guarantees: `tests/test_offline.py`
runs the core workflows with sockets blocked, `tests/test_parallel.py` shows
worker count never changes output hashes, and `tests/test_architecture.py`
keeps `import stylog` light (no optional or CLI modules) and the fitting stack
free of heavy dependencies.
