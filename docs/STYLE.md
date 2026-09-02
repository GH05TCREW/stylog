# Documentation style guide

This guide governs all prose in `docs/`, the README, and user-facing
documentation. It exists to keep the documentation set small, consistent, and
trustworthy.

## Audience

Write for developers, researchers, and technically proficient users. Assume
basic Python and command-line familiarity. Do not assume prior stylometry
knowledge; define Stylog-specific terms where they first appear.

## Voice

Professional, concise, neutral, direct, and technically precise. Explain when
explanation is needed; do not editorialize.

- Use second person for instructions ("Run `stylog fingerprint` ...").
- Prefer active voice ("Stylog compares features ..." over "Features are
  compared ...").
- Describe capabilities; never promote them. Words such as *powerful*,
  *revolutionary*, *state-of-the-art*, *advanced*, and *seamless* do not
  belong in the documentation.
- No introductory filler ("In this section we will look at ..."). Begin with
  the information.
- Define a term at first use, then use it. Do not restate definitions.

## Headings

- One H1 (`#`) per document, matching the document's purpose.
- Sentence case: `## Build a baseline`, not `## Build a Baseline`.
- Do not skip heading levels.
- Headings should name the thing, not announce a discussion of it:
  `## Authorship verification`, not `## Understanding authorship verification`.

## Terminology

Use these terms with exactly these meanings, taken from the
[specification](spec-v0.1.md):

- **artifact** -- one observable input sample (a file, stdin, or a string via
  the API).
- **fingerprint** -- the portable, deterministic measurement record for one
  primary artifact.
- **feature** -- one versioned measurement; **feature family** -- its grouping
  prefix (for example `text.lexical`).
- **comparison** -- descriptive per-feature distances between two artifacts;
  never a global similarity score.
- **baseline** -- a local, versioned reference distribution built from a
  corpus.
- **profile** -- the population-relative interpretation of one fingerprint
  against one explicit baseline.
- **dataset** / **training manifest** -- the declared corpus and the labeled
  pair manifest used to fit a verifier.
- **verifier** / **verifier fit** -- a fitted, self-contained pairwise
  authorship-verification model (`VerifierFit`).
- **verification** -- the model-relative decision record for one pair of
  artifacts.
- **score** -- the verifier's unitless decision value; not a probability.
- **probability** -- the calibrated same-author probability, present only when
  the verifier carries Platt calibration state.
- **verdict** -- `same_author`, `different_author`, or `abstain`.
- **abstain** -- a typed non-decision (`uncertain` or
  `insufficient_evidence`), not a weak verdict.
- **representation** -- a sparse model-space vector with provenance; not a
  fingerprint.
- **representation fit** -- the fitted vocabulary/vectorizer state used to
  produce representations.
- **diagnostic** -- a stable, machine-readable warning or error fact with a
  stable code.
- **provenance** -- recorded identity of inputs, resources, runtime, and
  models behind an artifact.

Do not use these terms interchangeably, and do not blur these distinctions:

```text
feature value != distance != authorship score != calibrated probability != certainty
same_author != proof of identity
```

## Authorship language

Never frame model output as proof of identity. A verdict expresses
model-relative support for an authorship hypothesis under the fitted verifier
and its training distribution. Prefer:

> A `same_author` verdict indicates model-relative support for the
> same-author hypothesis under the fitted verifier and its training
> distribution. It does not establish identity.

## Code formatting

Use backticks for commands, options, filenames, functions, classes, schema
identifiers (`stylog.fingerprint`), feature identifiers
(`text.lexical.ttr_casefold`), and literal statuses or verdicts (`ok`,
`abstain`).

Use fenced code blocks with a language identifier (```` ```bash ````,
```` ```python ````, ```` ```toml ````, ```` ```text ````). For copyable
commands, omit the shell prompt. Use `$` only in terminal transcripts that
also show output.

## Links

Use descriptive repository-relative links:

> See [Authorship verification](methodology.md#authorship-verification).

Never use "click here". Link the first mention of a topic that has its own
page; do not link every occurrence.

## Examples

- Prefer one small coherent example set across pages: `alice_1.txt`,
  `alice_2.txt`, `bob_1.txt`, `app.py`, `corpus/`.
- Introduce substantial examples with a sentence.
- Keep examples executable and verified against the implementation.
- Show output only when it adds understanding; use `...` for elided lines.
  Do not paste large JSON documents or volatile values (full hashes, long
  floats) when a short excerpt carries the point.

## Version language

Use explicit versions ("in Stylog 0.1.0", "schema version `0.1.0`"). Avoid
"currently", "for now", and "in the new version".

## Normative versus explanatory language

`spec-v0.1.md` is the only normative document; it uses **MUST**, **SHOULD**,
and **MAY** with their RFC 2119 meanings. All other documents are
explanatory: they describe behavior ("Stylog rejects explicit JSON null"),
they do not legislate it ("The serializer must reject ...").

## Each fact has one home

State each fact once, in its authoritative page, and cross-link elsewhere:

| Fact | Home |
| --- | --- |
| Project overview, quick start | `README.md` |
| Guided first-use workflow | `getting-started.md` |
| CLI behavior | `cli.md` |
| Public Python behavior | `python-api.md` |
| Explanations and interpretation | `methodology.md` |
| Uncertainty and failure modes | `limitations.md` |
| Normative requirements | `spec-v0.1.md` |
| Machine-readable structure | `schemas/` |
