<p align="center">
  <img src="docs/assets/stylog-logo.png" alt="Stylog logo" width="100">
</p>

<div align="center">

# Stylog

### Stylometry for text and source code

Stylog measures writing style in natural-language text and source code. It runs locally and provides fingerprints, feature comparisons, population baselines, sparse representations, and fitted authorship verifiers through a CLI and Python API.

<br/>

<a href="docs/spec-v0.1.md"><img src="https://img.shields.io/badge/Spec-v0.1-0f172a?style=for-the-badge" alt="Spec"></a> <a href="https://pypi.org/project/stylog/"><img src="https://img.shields.io/badge/PyPI-stylog-4f46e5?style=for-the-badge" alt="PyPI"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-3b82f6?style=for-the-badge" alt="License"></a> <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>

</div>

## Install

```bash
pip install stylog             # text + Python/JS/TS/C/Rust analysis
pip install "stylog[nlp]"      # spaCy linguistic features
pip install "stylog[ml]"       # scikit-learn representations
pip install "stylog[data]"     # Arrow/Parquet corpus I/O
pip install "stylog[all]"      # all optional features
```

Stylog requires Python 3.12+.

The base package uses Python's `tokenize` and `ast` modules for Python source and checked Tree-sitter grammars for JavaScript, TypeScript, C, and Rust.

## Commands

| Command       | Purpose                                               |
| ------------- | ----------------------------------------------------- |
| `fingerprint` | Measure files and emit fingerprints                   |
| `analyze`     | Inspect one artifact and its embedded artifacts       |
| `compare`     | Compare two artifacts feature by feature              |
| `profile`     | Place an artifact against a population baseline       |
| `fit`         | Fit an authorship verifier from labeled training data |
| `verify`      | Evaluate two artifacts with a fitted verifier         |
| `represent`   | Convert documents to sparse vectors                   |
| `report`      | Render an existing portable artifact                  |
| `benchmark`   | Run a declarative benchmark spec                      |
| `info`        | Report local capabilities, versions, and extras       |

See the [CLI reference](docs/cli.md) for arguments, options, output formats, and exit codes.
Bare `stylog` shows a concise command map and exits successfully. Use `-h` or
`--help` for full help, `stylog COMMAND -h` for a workflow, and `-V` for the
version.

## Quick start

### Fingerprint

```bash
$ stylog fingerprint alice_1.txt --language en --format terminal

Fingerprint

Artifact     alice_1.txt
Kind         text
Language     en
Encoding     utf-8
Size         221 bytes; 221 Unicode code points
Features     27 total; 26 ok; 1 insufficient support
Diagnostics  none
```

By default, a single input produces canonical JSON:

```bash
stylog fingerprint alice_1.txt | jq '.features[] | select(.status != "ok")'
```

Write a directory as JSONL:

```bash
stylog fingerprint src/ --output fingerprints.jsonl
```

### Analyze

```text
$ stylog analyze app.py

Analysis

Artifact       app.py
Kind           code
Language       python
Embedded text  3 comments/docstrings
Diagnostics    none

Feature families

  FAMILY                 STATUS
  code.python.comments   5 total; 5 ok
  code.python.lexical    6 total; 5 ok; 1 insufficient support
  code.python.naming     8 total; 6 ok; 2 insufficient support
  code.python.structure  17 total; 10 ok; 7 insufficient support
  ...
```

Embedded artifacts are the docstrings and comment blocks found inside the source file; each is analyzed separately.

### Compare

`compare` reports a distance for each comparable feature.

```text
$ stylog compare alice_1.txt alice_2.txt

Comparison

Left                 alice_1.txt
Right                alice_2.txt
Comparable features  17
Diagnostics          none

text.lexical

  FEATURE       METRIC   DISTANCE  SUPPORT L/R  UNIT
  ttr_casefold  ABS     0.00757576  44/46 word   proportion points on [0,1]
  word_length   W1        0.469697  44/46 word   word code points
  ...

$ stylog compare alice_1.txt bob_1.txt

Comparison

Left                 alice_1.txt
Right                bob_1.txt
Comparable features  17
Diagnostics          none

text.lexical

  FEATURE       METRIC  DISTANCE  SUPPORT L/R  UNIT
  ttr_casefold  ABS      0.132064  44/51 word   proportion points on [0,1]
  word_length   W1        2.76229  44/51 word   word code points
  ...
```

## Baselines

A baseline is a versioned reference distribution built from one or more documents in your corpus.

Build one with the Python API:

```python
from pathlib import Path

import stylog

from stylog.serialization.jsonio import write_json_atomic

fps = [
    stylog.fingerprint_file(path, language="en")
    for path in Path("corpus").glob("*.txt")
]

baseline = stylog.build_baseline(
    fps,
    baseline_id="my-base",
    kind="text",
    language="en",
    domain="news",
)

write_json_atomic("my-base.stylog-baseline.json", baseline)
```

Profile a document against it:

```bash
$ stylog profile alice_1.txt \
    --baseline my-base.stylog-baseline.json \
    --language en

Profile

Subject      alice_1.txt
Baseline     my-base 1.0.0
Features     17
Diagnostics  none

  FEATURE                                      OBSERVED       N   MIDRANK %    ROBUST Z
  text.function_words.en.token_share           0.545455      24         100     4.93604
  text.lexical.hapax_token_share_casefold          0.75      24     10.4167    -2.20666
  text.lexical.ttr_casefold                    0.840909      24        12.5    -2.28798
...
```

Baseline references containing a path separator or ending in `.json` resolve directly as paths. Baseline ids are resolved through `baseline.search_paths`, followed by:

```python
platformdirs.user_data_path("stylog") / "baselines"
```

## Authorship verification

Verification evaluates two documents under a fitted model and returns `same_author`, `different_author`, or `abstain`. Each verdict expresses model-relative support for an authorship hypothesis.

Training uses a `stylog.verifier-training` manifest that references a checksummed `stylog.dataset` manifest. Labeled pairs belong to author-disjoint train, tuning, and calibration populations. Train pairs fit the model and calibration pairs fit thresholds/calibration. Stylog records tuning-pair identity only; any hyperparameter selection using that population happens externally before `stylog fit`.

```toml
schema = "stylog.verifier-training"
schema_version = "0.1.0"
id = "demo-training"
dataset = "dataset.toml"

[verifier]
kind = "text"
l2_lambda = 1.0
min_support_fraction = 0.9
min_class_support_fraction = 0.8
min_pairs = 50
threshold_rule = "calibration_quantile_band"
threshold_alpha = 0.05
calibration_method = "platt"

[[pair]]
left = "t23663134a2a7a629"
right = "t38db1c322047e46a"
label = "same"
population = "train"

# ... more pairs ...
```

Fit the model:

```bash
$ stylog fit training.toml -o model.json

Diagnostic  INFO VERIFIER_ELIGIBILITY candidate_feature_count=17 eligible_pair_count=150 ...
Verifier ID  3f71213b3f5be729d1f77b0b12ca23b3e4c45b19fcd0ca067d9efe183f8a9601
```

Verify two documents:

```bash
$ stylog verify doc_a.txt doc_b.txt --model model.json --language en

Verification

Verdict           same_author
Model score       0.901019 (range (0,1); not a probability)
Probability       0.94066 same-author (platt; calibration-population conditional)
Features          16 used; 0 unavailable
Model             stylog.verifier.logreg/1 1.0.0
Verifier ID       3f71213b3f5be729d1f77b0b12ca23b3e4c45b19fcd0ca067d9efe183f8a9601
Left              doc_a.txt
Left fingerprint  8ccdfc25e1a9f9b20379aa90d5795abaea2907ce17d07f56a7b9f47929206d88
Right             doc_b.txt
Right fingerprint 9d360c97af337f0a162c2e3cae46c5f155c7e16dd9de76c5408ce66c90c96c38
Diagnostics       none
```

`score` is the model's unitless decision value. Models fitted with Platt calibration on a disjoint calibration split also report `probability`.

The verifier returns `abstain` when the available evidence is insufficient or the result falls within its uncertainty region. Typed reasons include `insufficient_evidence` and `uncertain`.

Use `--format json` for the machine-readable `Verification` object.

A fitted `VerifierFit` is stored as JSON. The fitting solver is implemented in pure Python.

From Python:

```python
import stylog

model = stylog.load_verifier("model.json")

verification = stylog.verify_files(
    "doc_a.txt",
    "doc_b.txt",
    model,
)
```

Verification records bind the hashes of both input fingerprints and the fitted model.

## Representations

Install `stylog[ml]` to build sparse representations.

Fit a word-TFIDF vocabulary:

```bash
stylog represent docs/ \
    --representation word-tfidf \
    --fit-output fit.json
```

Transform another document with the saved fit:

```bash
stylog represent new-doc.md --fit-resource fit.json
```

Representation fits and vectors use the `stylog.representation-fit` and `stylog.representation` JSON schemas and record their backend provenance.

Common shortcuts:

```text
-o    --output
-m    --model
```

For representation fits, `--model` and `-m` are aliases for `--fit-resource`.

The pre-0.2 command names `verify-fit` and `capabilities` remain available as hidden aliases for `fit` and `info`.

## Python API

```python
import stylog

fp = stylog.fingerprint_text(
    "Don't re-enter now.",
    language="en",
)

fp = stylog.fingerprint_file("app.py")

bundle = stylog.analyze_file("app.py")

comparison = stylog.compare_files("a.py", "b.py")

profile = stylog.profile_fingerprint(fp, "my-base")
```

Results are frozen Pydantic v2 models. Canonical JSON (RFC 8785) and hashing helpers live in `stylog.serialization`; the machine-readable contracts are the schemas in `schemas/`. See the [Python API reference](docs/python-api.md).

## Documentation

- [Getting started](docs/getting-started.md) — a guided first workflow
- [CLI reference](docs/cli.md) — every command, option, and exit code
- [Python API](docs/python-api.md) — the supported Python interface
- [Methodology](docs/methodology.md) — how Stylog measures, compares, and decides
- [Limitations](docs/limitations.md) — interpretation, uncertainty, failure modes
- [Specification v0.1](docs/spec-v0.1.md) — the normative contract

## Development

```bash
pip install -e ".[all,dev]"
python -m pytest
python tools/generate_schemas.py --check
```

## License

MIT. See [LICENSE](LICENSE).
