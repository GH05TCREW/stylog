# Stylog documentation

Stylog is a Python package for deterministic, local-only stylometry of
natural-language text and source code: it measures artifacts into portable
fingerprints and builds per-feature comparisons, population profiles, sparse
representations, and pairwise authorship verifications on top of them,
through a CLI and a Python API.

## Start here

New to Stylog? Work through [Getting started](getting-started.md): it walks
one small example set from installation through fingerprinting, comparison,
profiling, verifier fitting, verification, and representation.

## Documentation

- [Getting started](getting-started.md) -- guided first-use workflow.
- [CLI reference](cli.md) -- commands, options, diagnostics, and exit codes.
- [Python API](python-api.md) -- public functions and result models.
- [Methodology](methodology.md) -- concepts, quantities, and interpretation.
- [Limitations](limitations.md) -- uncertainty and failure modes.
- [Specification v0.1](spec-v0.1.md) -- the normative contract.

The [schemas/](../schemas/) directory holds the machine-readable JSON schema
for every portable artifact.
