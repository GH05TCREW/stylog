# Limitations

This page collects what Stylog's outputs do not tell you -- interpretation
limits, uncertainty, and realistic failure modes -- and it is not a list of
defects.

Mechanisms live in [Methodology](methodology.md) and normative behavior in
the [specification](spec-v0.1.md). Where this page says "Stylog enforces",
the implementation checks the rule; everything else is your responsibility.

## Measurements

- **Text formats are measured as decoded text, not rendered prose.** Markdown,
  HTML, reStructuredText, and similar inputs retain their markup, links, code
  blocks, and metadata. Distances can therefore reflect formatting conventions
  as well as prose style. Preprocess both sides consistently when the intended
  subject is prose alone.
- **Small inputs yield missing evidence.** Features report
  `insufficient_support` when their event population is empty, and a non-ok
  observation never carries a placeholder value (see
  [Feature status](methodology.md#feature-status)). Short or unusual
  artifacts accumulate non-ok observations, so downstream comparisons,
  profiles, and verifications degrade to omissions and abstains. Stylog
  enforces typed missingness; whether a sample is large enough to measure is
  your call -- check the support counts on observations.
- **Features respond to genre, topic, and register, not only to authorship.**
  A distance or a verification difference can reflect a domain change rather
  than an author change, and a verifier trained on one domain or problem mix
  can specialize to it.
- **Language coverage is uneven.** The English function-word features measure
  only `language = "en"` (they report `not_applicable` for other languages
  and `unavailable` for `und`); the remaining core text features run on any
  language tag but are language-agnostic surface and lexical measurements,
  not language-aware ones. The `text.linguistic.*` family requires the `nlp`
  extra and a locally provisioned spaCy pipeline, at substantially lower
  throughput than the core analyzers.
- **Code features measure tooling as well as authors.** The `code.surface`
  family measures exactly what formatters, linters, and project style guides
  homogenize (indentation, line endings, blank lines, trailing space); shared
  tooling can mask author differences, and a reformat can mimic an author
  change. Generated code and copied framework code are measured as-is:
  matching boilerplate is shared environment, not authorship evidence.
  Parser-backed features exist only for the supported grammars (Python,
  JavaScript, TypeScript, C, Rust); other languages still receive the generic
  `code.sample.*` and `code.surface.*` measurements, and parse failures
  produce typed `parser_error` with the generic surface measurements intact.
- **Embedded text stays separate.** Comments and docstrings are fingerprinted
  as independent embedded artifacts and never mix into the parent code
  fingerprint (see [Embedded artifacts](methodology.md#embedded-artifacts)).
  In Stylog 0.1.0 embedded sections are not verifier inputs: verifying a code
  pair compares code features only.

## Baselines and profiles

- **A baseline is your corpus, not a population norm.** Percentiles and
  robust z-scores are relative to the exact baseline you built (see
  [Profiles](methodology.md#profiles)); they say nothing about any wider
  population. Stylog requires only one baseline unit and always reports the
  exact `baseline_n`; baseline adequacy is your responsibility.
- **Zero MAD omits the robust z-score.** When the baseline values have no
  dispersion around their median, `mad_raw` is zero and `robust_z` is omitted
  with a `PROFILE_ZERO_MAD` diagnostic. Read the omission as "no dispersion
  measured", never as agreement or as an extreme deviation.

## Comparisons

- **Distances are descriptive and per-feature by design.** Stylog computes no
  global similarity score (see
  [Comparing fingerprints](methodology.md#comparing-fingerprints)); combining
  distances into one number is your modeling choice, with your own
  validation. Features missing or non-ok on either side are omitted with
  diagnostics, so a comparison with few components carries little evidence in
  either direction.

## Verification

- **A verdict is model-relative support, never identity.** A `same_author`
  verdict indicates model-relative support for the same-author hypothesis
  under the fitted verifier and its training distribution. It does not
  establish identity.
- **Every verdict depends on your training data.** Coefficients,
  normalization, thresholds, and calibration all derive from the populations
  in your training manifest, and the same pair can receive different verdicts
  under different verifiers. No default or bundled verifier ships with
  Stylog: fitting a verifier on data representative of your use, and
  evaluating it on held-out data, is your responsibility. Each verification
  binds the `verifier_id`, so a decision is always attributable to the exact
  model.
- **The author-disjoint contract is enforced by builders, not by the fit.**
  Author-disjoint train/tuning/calibration populations, duplicate-content
  neutralization, and pair uniqueness across populations are construction
  rules upheld by the deterministic manifest builders. The manifest schema
  validates structure, labels, and checksums; `fit_verifier` validates kinds,
  label classes, feature eligibility, and `min_pairs`. Neither inspects
  authorship structure -- a manifest built by other means is trusted. The
  model records its source, tuning, and calibration manifest hashes, so the
  populations behind any verifier remain auditable.
- **The score is not a probability, and the probability is not universal.**
  The decision `score` is a unitless value monotonic in the fitted model
  output. `probability` exists only when the model carries Platt calibration
  state fitted on the disjoint calibration split, and it is conditional on
  that split's class prevalence, domain, and language mix. Reusing a
  calibrated model under a different prevalence is a new evaluation question,
  not an assumption.
- **An abstain is a non-decision.** `abstain` with reason `uncertain` means
  the score fell inside the threshold band; `abstain` with reason
  `insufficient_evidence` means a model feature lacked comparable evidence
  and no score exists at all. Neither is a weak verdict in either direction.
  Under the `fixed` threshold rule both thresholds coincide, so every scored
  pair is decided; the same holds after a threshold-band collapse under
  `calibration_quantile_band` (a `THRESHOLD_BAND_COLLAPSED` fit diagnostic).
  The abstain rate is a property of the threshold rule and calibration data,
  not a universal uncertainty measure.
- **Scope violations are hard errors, not verdicts.** Kind, language,
  registry, or model mismatches raise `ModelIncompatibilityError`; a model
  feature absent from a fingerprint raises `CapabilityUnavailableError`.
  Neither is ever silently reduced to a verdict. An unconstrained-language
  verifier (explicit opt-in at fit time) accepts any language at verify time
  but can contain no language-scoped features.
- **Read `features_used` and `features_missing` on abstains.** A scored
  verification always uses the full model feature set. On
  `insufficient_evidence`, `features_missing` lists the model features
  without comparable evidence and `features_used` counts the remainder;
  recurring abstains point at evidence-quality problems, not at difficult
  authorship.
- **Verification is symmetric.** `verify(A, B)` and `verify(B, A)` agree on
  every scientific field (verdict, abstain reason, score, probability,
  features, diagnostics, verifier id); only the refs and evidence hashes swap
  positions.
- **False verdicts are possible.** No accuracy guarantee attaches to any
  single decision. The benchmark task evaluates verifiers with PAN-derived,
  intentionally adapted metrics (AUC, c@1, F1, F0.5u, and Brier loss) computed
  once on a held-out
  author-disjoint evaluation population; evaluate any verifier on your own
  held-out data before relying on it. "Stylog supports code authorship
  verification" is a capability statement, not an effectiveness claim.
- **Small training data fails loudly, but adequacy is unchecked.** Stylog
  enforces typed minima: pairs of both classes, at least `min_pairs`
  eligible pairs, at least one surviving feature, and a usable two-class
  calibration split; perfectly separable calibration scores fail Platt
  fitting as typed non-convergence. It does not and cannot enforce that your
  training data is large or representative enough.

## Representations

- **Representations are not fingerprints.** They require the `ml` extra and a
  local content-addressed fit state, and a vector inherits the vocabulary of
  its fit corpus; vectors produced under different fits never align (see
  [Sparse representations](methodology.md#sparse-representations)).

## Determinism

- **Byte identity is scoped to one runtime.** Fingerprints, fits, and
  verifications are byte-identical on repetition within the same recorded
  runtime; worker counts and cache state never change outputs (see
  [Reproducibility](methodology.md#reproducibility)). There is no
  cross-platform byte guarantee (library `exp`/`log` may differ in the last
  ulp). Determinism is repeatability, not accuracy: a deterministic pipeline
  reproduces the same answer, including the same errors.
