# CLI reference

The `stylog` command line measures, compares, profiles, and verifies text and
source-code artifacts; every command runs locally with no network access.

The base install provides the `stylog` console script on Python 3.12-3.14.
`represent` requires the `ml` extra, Parquet export requires the `data` extra,
and `--nlp-model` requires the `nlp` extra. Install details live in the
[README](../README.md).

## Command overview

| Command             | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `stylog fingerprint` | Measure inputs into portable fingerprints                  |
| `stylog analyze`    | Fingerprint one input and list its embedded artifacts       |
| `stylog compare`    | Report per-feature distances between two subjects           |
| `stylog profile`    | Interpret one subject against an explicit baseline          |
| `stylog fit`        | Fit a verifier from a training manifest                     |
| `stylog verify`     | Decide one pair under an explicit fitted verifier           |
| `stylog represent`  | Fit or apply a sparse representation (`ml` extra)           |
| `stylog report`     | Render an existing portable artifact                        |
| `stylog benchmark`  | Run a declarative benchmark spec                            |
| `stylog info`       | Report versions, grammars, and optional capabilities        |

Bare `stylog` is a successful landing screen (exit 0), not invalid usage. It
shows the command map, examples, and the important output-default distinction:

```text
$ stylog
Stylog 0.1.0 - stylometry for text and source code

Usage:
  stylog <command> [options]

Commands:
  fingerprint  Measure artifacts and emit portable fingerprints
  analyze      Inspect an artifact and its embedded text artifacts
  compare      Compare two artifacts feature-by-feature
  ...
```

## Conventions

These rules apply to every command; per-command sections state only what is
specific to that command.

### Global options

- `--debug` -- show tracebacks for internal errors; without it an internal
  error prints a concise message and a debug hint. Place it before the command:
  `stylog --debug fingerprint alice_1.txt`.
- `-V`, `--version` -- print the installed Stylog version and exit 0.
- `-h`, `--help` -- show root or command-specific help and exit 0.

### Inputs

- Commands accept files, one directory, or `-` for stdin. At most one stdin
  source is allowed per invocation.
- Stdin defaults to kind `text` with language `und`; `--kind code` on stdin
  requires an explicit `--language`.
- A directory argument is expanded into its files under the configured input
  rules; symlinks are rejected with a `SYMLINK_REJECTED` warning.
- `--kind text|code` overrides artifact kind detection and `--language LANG`
  overrides language detection for every input of the invocation.
- A file that fails to read or decode prints an error to stderr; the remaining
  inputs are still processed and the command exits 3.

### Output

- Machine formats (`json`, `jsonl`, `parquet`) write only machine bytes to
  stdout; diagnostics, progress, and warnings always go to stderr. Terminal
  renderings are plain ASCII.
- `-o PATH` (alias of `--output`) writes atomically to a file instead of
  stdout; an existing file is kept unless `--force` is given (exit 4
  otherwise).
- `--format parquet` is implemented only by `fingerprint`; it is not advertised
  as a format on other commands. `info --format json` emits pretty JSON;
  `info --format jsonl` emits one compact JSON object on one line.
- Human terminal layouts are presentation, not a parsing interface. Scripts
  should request `json` or `jsonl` explicitly.
- `-m` is the short alias of `--model` on `verify` and of `--fit-resource` on
  `represent`.

### Configuration

Configuration resolves in this order: `--config PATH`, the `STYLOG_CONFIG`
environment variable, `stylog.toml` in the current directory, a
`[tool.stylog]` table in `pyproject.toml`, then built-in defaults.

### Cache

Fingerprinting caches analysis results under
`platformdirs.user_cache_path("stylog") / "v1"` by default. `--cache-dir PATH`
or `STYLOG_CACHE_DIR` overrides the root; `--no-cache` or `STYLOG_NO_CACHE=1`
disables reads and writes; `--refresh` skips reads, recomputes, and rewrites
(and forces serial execution when `--workers` is greater than 1).

### Diagnostics

A diagnostic is a stable, machine-readable warning or error fact with a stable
code. Terminal output reports `Diagnostics  none` or a count followed by
`SEVERITY CODE key=value` detail lines; portable JSON carries them as typed
entries. Diagnostics do not by themselves cause a nonzero exit.

### Exit codes

| Code | Meaning                                                                  |
| ---- | ------------------------------------------------------------------------ |
| 0    | Success, including `abstain` verdicts and scientific warnings            |
| 2    | Usage error, configuration error, or unavailable capability              |
| 3    | Input error: missing, unreadable, undecodable, or unsupported input      |
| 4    | Invalid portable artifact, baseline error, model/resource incompatibility, or overwrite refused |
| 5    | Internal error (analyzer or unexpected exception)                        |
| 6    | Benchmark or training manifest/dataset error                             |
| 7    | Input resource limit exceeded                                            |
| 130  | Interrupted (Ctrl+C)                                                     |

## `stylog fingerprint`

Measure inputs and emit one portable fingerprint per input artifact.

### Usage

```bash
stylog fingerprint [OPTIONS] INPUTS...
```

### Arguments

- `INPUTS` -- one or more files, one directory, or `-` for stdin (required).

### Options

- `--format json|jsonl|terminal|parquet` -- output format; default `json` for a
  single input without `--collection`, otherwise `jsonl`.
- `-o, --output PATH` -- write to `PATH` instead of stdout.
- `--force` -- allow overwriting an existing `--output`.
- `--kind text|code` -- override artifact kind.
- `--language LANG` -- override artifact language.
- `--collection` -- treat all inputs as one evidence set and append an
  `stylog.evidence-aggregate` record; requires `--linkage` and
  `--linkage-source`.
- `--linkage KIND` -- evidence linkage kind recorded on the aggregate.
- `--linkage-source SOURCE` -- evidence linkage source recorded on the aggregate.
- `--workers N` -- process-pool workers; values above 1 enable the pool
  (default 1).
- `--no-cache` -- disable cache reads and writes.
- `--refresh` -- skip cache reads; recompute and rewrite.
- `--cache-dir PATH` -- override the cache root.
- `--no-content-hash` -- suppress exported content hashes.
- `--config PATH` -- explicit stylog TOML config.
- `--nlp-model NAME` -- provisioned spaCy model (requires the `nlp` extra).

### Behavior

- `--format json` with several inputs still emits one JSON object per line;
  single-object JSON requires exactly one result.
- `--format parquet` requires `--output` and the `data` extra.
- With `--collection`, the aggregate is omitted (with a stderr note) if any
  member failed ingestion.
- If any input fails ingestion, successfully processed inputs are still
  written and the command exits 3.

### Output

Terminal rendering shows the artifact id, kind, language, encoding, byte and
Unicode code-point counts, feature status counts, and diagnostics. Machine output is
the canonical `stylog.fingerprint` record (see
`schemas/stylog.fingerprint.schema.json`).

### Examples

Fingerprint one file with a terminal summary:

```bash
$ stylog fingerprint alice_1.txt --language en --format terminal
Fingerprint

Artifact     alice_1.txt
Kind         text
Language     en
Encoding     utf-8
Size         340 bytes; 340 Unicode code points
Features     27 total; 25 ok; 2 insufficient support
Diagnostics  none
```

Write a directory as JSONL:

```bash
stylog fingerprint corpus/ --language en --output fingerprints.jsonl
```

## `stylog analyze`

Analyze one input into a full `stylog.analysis` bundle: fingerprint plus
embedded artifacts, with optional aggregation or profiling.

### Usage

```bash
stylog analyze [OPTIONS] INPUT
```

### Arguments

- `INPUT` -- one file, one directory, or `-` for stdin (required).

### Options

`analyze` accepts the same options as `fingerprint`, plus:

- `--baseline REF` -- also profile the input against this baseline; requires
  exactly one input artifact and cannot be combined with `--collection`.

The default `--format` is `terminal`.

### Behavior

- Embedded artifacts (for example comments and docstrings inside source code)
  are reported in the bundle.
- With `--collection`, the expanded inputs of the single directory argument
  are aggregated under the given linkage.
- With `--baseline`, a `stylog.profile` record is appended to the output; the
  baseline resolves as described under `stylog profile`.
- Exits 3 if the input fails ingestion.

### Output

Terminal rendering lists feature families with `ok` counts, the number of
embedded artifacts, and diagnostics, followed by the aggregate or profile
block when requested. Machine formats write one JSON object per result
(bundle, then aggregate or profile when present).

### Examples

```bash
$ stylog analyze app.py
Analysis

Artifact       app.py
Kind           code
Language       python
Embedded text  2 comments/docstrings
Diagnostics    none

Feature families

  FAMILY                STATUS
  code.python.comments  5 total; 5 ok
  code.python.lexical   6 total; 5 ok; 1 insufficient support
  ...
```

## `stylog profile`

Place one subject against an explicit baseline and report population-relative
positions.

### Usage

```bash
stylog profile [OPTIONS] --baseline REF SOURCE
```

### Arguments

- `SOURCE` -- input file, `-` for stdin, or a portable `stylog.fingerprint`
  JSON file with `--from-artifact` (required).

### Options

- `--baseline REF` -- baseline id or path (required).
- `--from-artifact` -- read `SOURCE` as a portable fingerprint JSON instead of
  analyzing it; stdin is not accepted in this mode.
- `--format json|jsonl|terminal` -- output format (default `terminal`).
- `-o, --output PATH`, `--force` -- file output and overwrite control.
- `--kind text|code`, `--language LANG` -- override kind or language detection.
- `--no-cache`, `--refresh`, `--cache-dir PATH` -- cache control.
- `--no-content-hash` -- suppress exported content hashes.
- `--config PATH` -- explicit stylog TOML config.
- `--nlp-model NAME` -- provisioned spaCy model (requires the `nlp` extra).

### Behavior

A baseline reference resolves as an explicit path or as a baseline id under
the configured search paths (see
[Population baselines](methodology.md#population-baselines)). A missing path
or unmatched id is `BASELINE_NOT_FOUND` (exit 4); two distinct baselines with
the same id are `BASELINE_INVALID` (exit 4).

Only profileable scalar features appear in the result. Percentile and robust-z
semantics are covered in [Profiles](methodology.md#profiles).

### Output

Terminal rendering is one row per feature with the observed value, baseline
count, midrank percentile, and robust z; `-` marks a robust z that is
undefined (for example a zero-MAD feature where the subject equals the
median).

### Examples

```bash
$ stylog profile alice_1.txt --baseline demo-base.stylog-baseline.json --language en
Profile

Subject      alice_1.txt
Baseline     demo-base 1.0.0
Features     17
Diagnostics  none

  FEATURE                                           OBSERVED       N   MIDRANK %    ROBUST Z
  text.function_words.en.token_share                0.484375       3     83.3333     0.67449
  text.lexical.ttr_casefold                         0.765625       3     16.6667    -0.67449
...
```

Baselines are built with the Python API (`stylog.build_baseline`); see the
[README](../README.md).

## `stylog compare`

Compare two subjects feature by feature; Stylog never reports a global
similarity score.

### Usage

```bash
stylog compare [OPTIONS] LEFT RIGHT
```

### Arguments

- `LEFT`, `RIGHT` -- input files, or portable JSON artifacts with
  `--from-artifacts` (both required).

### Options

- `--from-artifacts` -- read both arguments as portable `stylog.fingerprint` or
  `stylog.analysis` JSON instead of analyzing them.
- `--format json|jsonl|terminal` -- output format (default `terminal`).
- `-o, --output PATH`, `--force` -- file output and overwrite control.
- `--kind text|code`, `--language LANG` -- override kind or language detection.
- `--no-cache`, `--refresh`, `--cache-dir PATH` -- cache control.
- `--no-content-hash` -- suppress exported content hashes.
- `--config PATH` -- explicit stylog TOML config.
- `--nlp-model NAME` -- provisioned spaCy model (requires the `nlp` extra).

### Behavior

Each comparable feature contributes one component with its distance metric,
value, unit, and left/right support, grouped by feature family. A registry-
comparable feature missing or non-`ok` on either side is omitted with a
`FEATURE_NOT_COMPARABLE` diagnostic carrying both statuses. Exactly two input
artifacts are required; ingestion failure exits 3.

### Output

```bash
$ stylog compare alice_1.txt alice_2.txt --language en
Comparison

Left                 alice_1.txt
Right                alice_2.txt
Comparable features  17
Diagnostics          none

Metric definitions
  ABS   Absolute difference; units are feature-specific
  JSD2  Base-2 Jensen-Shannon distance (sqrt divergence); range [0,1]
  W1    Wasserstein-1 distance; units are feature-specific

text.lexical

  FEATURE       METRIC  DISTANCE  SUPPORT L/R  UNIT
  ttr_casefold  ABS      0.054375  64/61 word   proportion points on [0,1]
  word_length   W1         0.4125  64/61 word   word code points
...
```

Distance metrics and units are covered in
[Comparing fingerprints](methodology.md#comparing-fingerprints).

## `stylog verify`

Verify two subjects under an explicit fitted verifier model.

### Usage

```bash
stylog verify [OPTIONS] --model MODEL.json LEFT RIGHT
```

### Arguments

- `LEFT`, `RIGHT` -- input files, or portable JSON artifacts with
  `--from-artifacts` (both required).

### Options

- `-m, --model PATH` -- portable `stylog.verifier-fit` JSON (required).
- `--from-artifacts` -- read both arguments as portable `stylog.fingerprint` or
  `stylog.analysis` JSON instead of analyzing them.
- `--format json|jsonl|terminal` -- output format (default `terminal`).
- `-o, --output PATH`, `--force` -- file output and overwrite control.
- `--kind text|code`, `--language LANG` -- override kind or language detection.
- `--no-cache`, `--refresh`, `--cache-dir PATH` -- cache control.
- `--no-content-hash` -- suppress exported content hashes.
- `--config PATH` -- explicit stylog TOML config.
- `--nlp-model NAME` -- provisioned spaCy model (requires the `nlp` extra).

### Behavior

- The verdict is `same_author`, `different_author`, or `abstain`; an abstain
  carries a typed reason (`uncertain` or `insufficient_evidence`) and is a
  successful outcome (exit 0).
- Hard gates on model id, backend compatibility id, feature registry version,
  per-feature semantic version, kind, and language scope fail with
  `MODEL_INCOMPATIBLE` (exit 4); a malformed model file also exits 4. A model
  feature that is entirely absent from a fingerprint because a capability is
  missing (for example linguistic features without the `nlp` extra) exits 2.
- A `same_author` verdict indicates model-relative support for the
  same-author hypothesis under the fitted verifier and its training
  distribution. It does not establish identity.

### Output

Terminal rendering shows the verdict, the score, the calibrated probability
(explicitly marked absent for uncalibrated models or insufficient evidence),
features used and missing, the model and verifier ids, and both subject refs
with their fingerprint hashes. The score is in `(0,1)` but is explicitly not
labeled a probability; only the separately calibrated value is a probability.

```bash
$ stylog verify alice_1.txt alice_2.txt --model model.json --language en
Verification

Verdict      same_author
Model score  0.997604 (range (0,1); not a probability)
Probability  not available (uncalibrated model)
Features     16 used; 0 unavailable
Model        stylog.verifier.logreg/1 1.0.0
Verifier ID  a2ee9d9d...
...
```

Score, probability, and abstain semantics are covered in
[Authorship verification](methodology.md#authorship-verification).

## `stylog fit`

Fit a self-contained authorship verifier (`stylog.verifier-fit`) from a
training manifest.

### Usage

```bash
stylog fit --output MODEL.json TRAINING.toml
```

### Arguments

- `TRAINING` -- a `stylog.verifier-training` TOML manifest (required).

### Options

- `-o, --output PATH` -- write the `VerifierFit` JSON here (required).
- `--force` -- allow overwriting an existing `--output`.
- `--config PATH` -- explicit stylog TOML config.

### Behavior

- The manifest declares a `dataset` (a relative path to a `stylog.dataset`
  manifest), a `[verifier]` block with fit hyperparameters, and `[[pair]]`
  entries whose `population` is `train`, `tuning`, or `calibration`.
- Train pairs drive feature eligibility, normalization, and coefficients;
  calibration pairs drive thresholds and Platt calibration state; tuning pairs
  are recorded by identity only. Any selection using tuning data is an
  external, pre-fit procedure; `stylog fit` does not search hyperparameters.
- Fit diagnostics and the `verifier_id` (the model's scientific hash) go to
  stderr; stdout stays empty.
- Repeated fits on the same manifest in the same runtime are byte-identical.
- Manifest or dataset errors exit 6; a failed fit exits 2.

### Examples

```bash
$ stylog fit training.toml -o model.json
Diagnostic  INFO VERIFIER_ELIGIBILITY candidate_feature_count=17 eligible_pair_count=8 ...
Verifier ID  a2ee9d9d...
```

## `stylog report`

Render an existing portable artifact as terminal text without re-analysis,
cache access, or baseline resolution.

### Usage

```bash
stylog report RESULT.json
```

### Arguments

- `RESULT` -- portable artifact JSON to render (required).

### Behavior

The file's `schema` member selects validation and rendering. Fingerprints,
analysis bundles, comparisons, profiles, evidence aggregates, and
verifications use the same renderers as the producing commands; other portable
schemas (baseline, evidence set, representation, representation fit, benchmark
result) render as a generic summary of their scalar members. A missing file
exits 3; invalid JSON or an unsupported schema exits 4.

### Examples

```bash
$ stylog report fingerprint.json
Schema  stylog.fingerprint

Fingerprint

Artifact  alice_1.txt
Kind      text
Language  en
...
```

## `stylog benchmark`

Run a declarative benchmark spec and emit a `stylog.benchmark-result` record.

### Usage

```bash
stylog benchmark [OPTIONS] SPEC.toml
```

### Arguments

- `SPEC` -- a `stylog.benchmark` TOML spec (required).

### Options

- `--format json|jsonl|terminal` -- output format (default `terminal`).
- `-o, --output PATH`, `--force` -- file output and overwrite control.

### Behavior

- A spec names a `task`, a `dataset` (a relative path to a `stylog.dataset`
  manifest), and task inputs such as `[split]`, `[[pair]]`, or `[verifier]`.
  Dataset artifact checksums are verified unless `checksums = false`.
- Task kinds: `split_audit` (deterministic split integrity audit),
  `pairwise_comparison` (labeled-pair distance metrics),
  `transformation_stability` (original/variant stability), and `verification`
  (decision metrics under the verifier named in `[verifier] model`).
- All paths resolve relative to the spec file; nothing is downloaded.
- Spec or dataset errors exit 6.

### Output

Terminal output is a short summary; `--format json` writes the canonical
result record. The result structure is defined by
`schemas/stylog.benchmark-result.schema.json`.

### Examples

```bash
$ stylog benchmark spec.toml
Benchmark

Schema        stylog.benchmark-result
Benchmark id  tiny-bench
Task          split_audit
Diagnostics   none
```

## `stylog represent`

Fit a sparse representation on inputs, or apply an existing representation
fit. Requires the `ml` extra (scikit-learn); without it the command exits 2
and prints the install hint.

### Usage

```bash
stylog represent [OPTIONS] INPUTS...
```

### Arguments

- `INPUTS` -- input files or `-` for stdin (required).

### Options

- `--representation TOKEN` -- representation to fit: `char-ngram-count`,
  `word-ngram-count`, `char-tfidf`, or `word-tfidf`; required with
  `--fit-output`.
- `--fit-output PATH` -- fit on the inputs and write the
  `stylog.representation-fit` resource here.
- `-m, --model, --fit-resource PATH` -- transform the inputs using an existing
  fit resource.
- `--format json|jsonl|terminal` -- output format; default `json` for
  one representation, otherwise `jsonl`.
- `-o, --output PATH`, `--force` -- file output and overwrite control.
- `--kind text|code`, `--language LANG` -- override kind or language detection.
- `--config PATH` -- explicit stylog TOML config.

### Behavior

- Exactly one of `--fit-output` or `--fit-resource` is required (exit 2
  otherwise).
- `--fit-output` writes the fit atomically and refuses to overwrite an
  existing file unless `--force` is given (exit 4 otherwise); the fit JSON is
  also written to stdout or `--output`.
- Each transformed input yields one `stylog.representation` record carrying
  its full representation id (for example
  `stylog.representation.word_tfidf/1`) and backend provenance.
- Exits 3 if any input fails ingestion.

### Examples

```bash
stylog represent alice_1.txt alice_2.txt --representation word-tfidf --fit-output fit.json
```

```bash
$ stylog represent alice_1.txt --fit-resource fit.json --format terminal
Representation

Implementation  stylog.representation.word_tfidf/1
Subject         alice_1.txt
```

## `stylog info`

Print a local-only capability report; no network access, and provisioned NLP
models are never loaded or enumerated.

### Usage

```bash
stylog info [--format json]
```

### Options

- `--format json|jsonl|terminal` -- output format; `json` writes the full
  report as pretty JSON and `jsonl` writes one compact record (default
  `terminal`).

### Output

The report covers the Stylog version, supported code languages, the runtime
signature, checked Tree-sitter grammars with their compatibility ids, optional
capability availability (`spacy`, `sklearn`, `pyarrow`, `polars`, `duckdb`,
`pandas`), representation availability and ids, and the verification
implementation id and version. Stylog does not bundle a fitted verifier; the
terminal report says to run `stylog fit`, while JSON exposes the full runtime
and compatibility data.

```bash
$ stylog info
Stylog info

Version         0.1.0
Python          CPython 3.14.3 (cpython-314)
Code languages  python, javascript, typescript, c, rust
Network         not used

Specialized analyzers
  text        native text features (language-neutral core; English function words)
  python      code-surface plus CPython tokenize and AST
  ...

Verification
  Implementation        stylog.verifier.logreg/1 1.0.0
  Bundled fitted model  none
```

## Hidden aliases

Two pre-0.2 command names remain available but do not appear in `--help`:

- `stylog verify-fit` -- alias of `stylog fit`.
- `stylog capabilities` -- alias of `stylog info`.
