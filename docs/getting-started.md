# Getting started

This guide walks one small example set through the complete Stylog workflow.
To follow along, create a working directory with three short texts of about
100 words each (`alice_1.txt` and `alice_2.txt` in one style, `bob_1.txt` in
another), a small Python file `app.py` with a docstring and a comment or two,
and a `corpus/` directory holding several short paragraphs. Every command
runs locally; Stylog never touches the network.

## Install

Stylog requires Python 3.12-3.14. Install it from PyPI:

```bash
pip install stylog
```

The base install measures natural-language text and Python, JavaScript,
TypeScript, C, and Rust code. Optional extras add capabilities: `nlp`
(spaCy linguistic features), `ml` (scikit-learn representations), `data`
(Arrow/Parquet corpus I/O), or `all`. This guide needs none of them until
the representation step:

```bash
pip install "stylog[ml]"
```

## Check the installation

`stylog info` prints the Stylog version, the pinned grammars, and which
optional capabilities are importable. The report is local-only and never
loads NLP models.

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

Optional capabilities

  PACKAGE     EXTRA  STATUS
  spacy       nlp    installed
  sklearn     ml     installed
  ...

Verification
  Implementation        stylog.verifier.logreg/1 1.0.0
  Bundled fitted model  none
```

A capability reported as not installed only means the matching extra is
absent; the base workflows in this guide work without it.

## Fingerprint a document

An artifact is one observable input sample: a file, stdin, or a string via
the API. A fingerprint is the portable, deterministic measurement record for
one artifact. Fingerprint `alice_1.txt`, telling Stylog the text is English:

```bash
$ stylog fingerprint alice_1.txt --language en --format terminal
Fingerprint

Artifact     alice_1.txt
Kind         text
Language     en
Encoding     utf-8
Size         698 bytes; 698 Unicode code points
Features     27 total; 26 ok; 1 insufficient support
Diagnostics  none
```

The `.txt` extension and your `--language` option classified the input as
English text, and Stylog measured 27 text features. One feature had no
events to measure, so its status is `insufficient_support`; a non-`ok`
feature never carries a placeholder value. Feature statuses are covered in
[Methodology](methodology.md#feature-status).

Without `--format`, a single input prints the canonical `stylog.fingerprint`
JSON; add `-o` to write it to a file instead:

```bash
stylog fingerprint alice_1.txt --language en -o fingerprint.json
```

The record's top-level members include `artifact` (id, kind, language,
encoding, byte and Unicode code-point counts, content identity), `runtime`,
`analyzers`, `features`, and `diagnostics`. Repeating an analysis with the
same artifact descriptor, input bytes, configuration, and runtime yields a
byte-identical fingerprint. Content-identical files with different artifact
ids intentionally have different fingerprint records, although they share
the same content identity and may share a cache entry. The structure is
defined in [Methodology](methodology.md#fingerprints) and
`schemas/stylog.fingerprint.schema.json`.

## Analyze an artifact

`analyze` fingerprints one input and lists the natural-language artifacts
embedded in it -- for Python code, the docstrings and comment blocks:

```bash
$ stylog analyze app.py
Analysis

Artifact       app.py
Kind           code
Language       python
Embedded text  4 comments/docstrings
Diagnostics    none

Feature families

  FAMILY                STATUS
  code.python.comments  5 total; 5 ok
  code.python.lexical   6 total; 6 ok
  ...
```

The `.py` extension made this a Python code artifact, measured by the
`code.*` feature families instead of the `text.*` ones. Each embedded
artifact receives its own independent fingerprint; see
[Embedded artifacts](methodology.md#embedded-artifacts).

## Compare two artifacts

`compare` reports per-feature distances between two artifacts -- never a
global similarity score. Compare the two Alice texts:

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
  ttr_casefold  ABS      0.059383  107/104 word  proportion points on [0,1]
  word_length   W1        0.44744  107/104 word  word code points
...
text.structure

  FEATURE                 METRIC  DISTANCE  SUPPORT L/R    UNIT
  sentence_length_tokens  W1        3.69048  6/7 sentence   sentence tokens
...
```

Each line is one feature's distance under its registry-assigned metric:
`ABS` (absolute difference), `JSD2` (the square root of base-2 Jensen-Shannon
divergence, on `[0,1]`), or `W1` (Wasserstein-1 in the feature's native unit).
Comparing across authors
shows which features separate the styles: against `bob_1.txt`, the sentence
length distance grows from 3.69 to 9.34 tokens.

```bash
$ stylog compare alice_1.txt bob_1.txt --language en
...
  sentence_length_tokens  W1        9.34127  6/5 sentence  sentence tokens
...
```

A distance is descriptive: it says how far apart two artifacts sit on one
measurement, not who wrote either one. Metrics and units are covered in
[Comparing fingerprints](methodology.md#comparing-fingerprints).

## Profile against a baseline

A baseline is a local, versioned reference distribution built from a corpus;
a profile interprets one fingerprint against one explicit baseline. Build a
baseline from the `corpus/` texts with the Python API:

```python
from pathlib import Path

import stylog

from stylog.serialization.jsonio import write_json_atomic

fps = [
    stylog.fingerprint_file(path, language="en")
    for path in sorted(Path("corpus").glob("*.txt"))
]
baseline = stylog.build_baseline(
    fps, baseline_id="demo-base", kind="text", language="en", domain="news"
)
write_json_atomic("demo-base.stylog-baseline.json", baseline)
```

Then profile `alice_1.txt` against it:

```bash
$ stylog profile alice_1.txt --baseline demo-base.stylog-baseline.json --language en
Profile

Subject      alice_1.txt
Baseline     demo-base 1.0.0
Features     17
Diagnostics  3

  FEATURE                                      OBSERVED       N   MIDRANK %    ROBUST Z
  text.function_words.en.token_share           0.373832       8        37.5   -0.270008
...
  text.lexical.number_count                    0               8          50           -
  text.lexical.ttr_casefold                    0.785047        8           0    -2.51438
...
  text.lexical.word_count                     107              8         100     14.9737
...
Diagnostic details
  INFO PROFILE_ZERO_MAD feature=text.lexical.number_count
  ...
```

Each row places one feature in the baseline distribution: `percentile` is
the midrank percentile and `robust_z` is a MAD-based robust z-score.
`-` means the baseline's MAD is zero, so no z-score is defined; the
`PROFILE_ZERO_MAD` diagnostics say the same. Here `alice_1.txt` is longer
than any corpus text (`word_count` percentile 100) with lower vocabulary
richness (`ttr_casefold` percentile 0). With `baseline_n` of 8, these
positions are illustrative only. Exact definitions and the zero-MAD rule
are covered in [Profiles](methodology.md#profiles).

## Fit a verifier

A verifier (`VerifierFit`) is a self-contained pairwise authorship model,
fitted from labeled pairs of artifacts. The CLI fits one from a
`stylog.verifier-training` TOML manifest, which names a `stylog.dataset`
manifest (the corpus files with their SHA-256 checksums) and declares
same/different pairs in author-disjoint populations.

The toy example below uses eight synthetic files from two fictional authors
(`a01`-`a04` and `b01`-`b04` in `verifier_data/`): four same-author pairs
and four different-author pairs, all in the train population.

```toml
schema = "stylog.verifier-training"
schema_version = "0.1.0"
id = "tiny-training"
dataset = "dataset.toml"

[verifier]
kind = "text"
l2_lambda = 1.0
min_support_fraction = 0.75
min_class_support_fraction = 0.5
min_pairs = 4
threshold_rule = "fixed"
threshold_fixed = 0.5
include_linguistic = false
allow_unconstrained_language = false

[verifier.pair_policy]
selection_version = "1"

[[pair]]
left = "a01.txt"
right = "a02.txt"
label = "same"
population = "train"

# ... seven more pairs, four "same" and four "different" in total
```

The referenced dataset manifest lists every training file with its
checksum:

```toml
schema = "stylog.dataset"
schema_version = "0.1.0"
id = "tiny"
version = "1"
license = "CC0"
redistribution = "allowed"
source = "synthetic tutorial corpus"

[[artifact]]
id = "a01.txt"
path = "verifier_data/a01.txt"
sha256 = "..."
kind = "text"
language = "en"

# ... one block per file
```

Fit the model (`-o` is required):

```bash
$ stylog fit training.toml -o model.json
Diagnostic  INFO VERIFIER_ELIGIBILITY candidate_feature_count=17 eligible_pair_count=8 selected_feature_count=16 training_pair_count=8
Diagnostic  INFO VERIFIER_ZERO_VARIANCE_FEATURE feature=text.lexical.token_kind
...
Verifier ID  ec7551a2...
```

The diagnostics report how many candidate features and pairs survived
eligibility; the `verifier_id` is the model's scientific hash. This example
uses `threshold_rule = "fixed"` because the data-driven
`calibration_quantile_band` threshold -- and Platt calibration, which adds a
`probability` to verification output -- both require a disjoint calibration
population far larger than a toy corpus provides. Eight pairs demonstrate
the mechanics and nothing more: real fits need a real corpus with
author-disjoint train, tuning, and calibration populations, as covered in
[Authorship verification](methodology.md#authorship-verification).
The v0.1 fitter does not tune hyperparameters: tuning pairs only record the
identity of a population used by any external, pre-fit selection procedure.

## Verify two artifacts

Verify the two Alice texts under the fitted model:

```bash
$ stylog verify alice_1.txt alice_2.txt --model model.json --language en
Verification

Verdict            same_author
Model score        0.947887 (range (0,1); not a probability)
Probability        not available (uncalibrated model)
Features           16 used; 0 unavailable
Model              stylog.verifier.logreg/1 1.0.0
Verifier ID        ec7551a2...
Left               alice_1.txt
Left fingerprint   8cb2686d...
Right              alice_2.txt
Right fingerprint  4817ce45...
Diagnostics        none
```

The verdict is `same_author`, `different_author`, or `abstain` -- a typed
non-decision with reason `uncertain` or `insufficient_evidence`. The score
is the verifier's unitless decision value, not a probability; `probability`
appears only for Platt-calibrated models and is explicitly absent here. A
`same_author` verdict indicates model-relative support for the same-author
hypothesis under the fitted verifier and its training distribution. It does
not establish identity. Under this toy model, `alice_1.txt` against
`bob_1.txt` scores 0.20 and draws a `different_author` verdict. Score,
probability, and abstain semantics are covered in
[Authorship verification](methodology.md#authorship-verification).

## Represent documents

A representation is a sparse model-space vector with provenance, produced by
a scikit-learn vectorizer; it is not a fingerprint. This step requires the
`ml` extra. Fit a word TF-IDF representation on the corpus directory:

```bash
stylog represent corpus/ --representation word-tfidf --fit-output fit.json
```

Then transform `alice_1.txt` against the saved fit:

```bash
$ stylog represent alice_1.txt --fit-resource fit.json --format terminal
Representation

Implementation  stylog.representation.word_tfidf/1
Subject         alice_1.txt
```

`--fit-output` writes the `stylog.representation-fit` resource (the learned
vocabulary and IDF state) and echoes it to stdout; `--fit-resource` (alias
`-m`) applies a saved fit. The transformed record is a sparse vector -- here
33 nonzero entries in a 1044-dimensional space -- carrying fit and backend
provenance. Fit state is a local content-addressed resource: provision it on
each machine by re-running the fit there. The four representation kinds are
covered in [Sparse representations](methodology.md#sparse-representations).

## Next steps

- [CLI reference](cli.md) -- every command, option, diagnostic convention,
  and exit code.
- [Python API](python-api.md) -- the same workflows as function calls over
  frozen result models.
- [Methodology](methodology.md) -- how the measurements, distances,
  profiles, representations, and verification work and how to interpret
  them.
- [Limitations](limitations.md) -- uncertainty and failure modes.
- [Specification v0.1](spec-v0.1.md) -- the normative contract; the
  machine-readable structures live in [schemas/](../schemas/).
