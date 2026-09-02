"""Stylog command-line interface (spec section 19).

Thin Typer callbacks only: parse options, delegate to the application layer
(fingerprint/analyze/aggregate/compare/profile/benchmark), render or write
portable output, and map exceptions to the contract exit codes. No scientific
logic lives here.

Stdout/stderr discipline: machine formats (json/jsonl) write only machine
bytes to stdout; all diagnostics, progress, and warnings go to stderr.
Terminal renderers emit plain ASCII so the Windows cp1252 console is safe.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import json
import os
import platform
import sys
import traceback
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel
from typer.core import TyperGroup

from stylog.application.aggregate import aggregate_evidence
from stylog.application.analyze import analyze_artifact
from stylog.application.batch import analyze_iter
from stylog.application.compare import compare_subjects
from stylog.application.profile import profile_subject
from stylog.application.verify import verify_subjects
from stylog.bootstrap import build_context, build_default_services
from stylog.capability import require_capability
from stylog.config import NlpConfig, StylogConfig, load_config
from stylog.domain import PORTABLE_MODELS_BY_SCHEMA
from stylog.domain.evidence import (
    EvidenceAggregate,
    EvidenceMember,
    EvidenceSet,
    LinkageDescriptor,
)
from stylog.domain.fingerprint import AnalysisBundle, Fingerprint
from stylog.domain.interpretation import Comparison, Profile
from stylog.domain.verification import Verification, VerifierFit
from stylog.exceptions import (
    CapabilityUnavailableError,
    InputError,
    InternalStylogError,
    PortableArtifactError,
    StylogError,
)
from stylog.infrastructure.files import select_files
from stylog.infrastructure.ingest import artifact_from_file, artifact_from_stdin
from stylog.representations.spec import CLI_TOKENS, SPECS
from stylog.runtime import AnalysisContext, RuntimeArtifact
from stylog.serialization.canonical import sha256_hex
from stylog.serialization.jsonio import file_bytes, jsonl_bytes, write_bytes_atomic

_COMMAND_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "fingerprint",
            "analyze",
            "compare",
            "profile",
            "fit",
            "verify",
            "represent",
            "report",
            "benchmark",
            "info",
        )
    )
}


class _StylogGroup(TyperGroup):
    """Keep the task-oriented command order stable in generated help."""

    def list_commands(self, ctx: typer.Context) -> list[str]:
        commands = super().list_commands(ctx)
        return sorted(commands, key=lambda name: (_COMMAND_ORDER.get(name, 1000), name))

app = typer.Typer(
    name="stylog",
    cls=_StylogGroup,
    help=(
        "Local, reproducible style measurement for text and source code. "
        "Files stay on your machine."
    ),
    epilog=(
        "Examples:\n\n\b\n"
        "  stylog analyze app.py\n"
        "  stylog compare draft-a.txt draft-b.txt --language en\n"
        "  stylog fingerprint document.txt --format terminal\n\n\b\n"
        "Run 'stylog COMMAND --help' for command-specific help."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
    no_args_is_help=False,
    add_completion=False,
    rich_markup_mode=None,
    pretty_exceptions_enable=False,
)

_STATE = {"debug": False}


# ---------------------------------------------------------------------------
# Output primitives (ASCII-only human text; bytes for machine output)
# ---------------------------------------------------------------------------


def _safe(text: object) -> str:
    """Render any value as plain ASCII for cp1252 consoles (spec 19/22)."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _stdout(text: str) -> None:
    sys.stdout.write(_safe(text) + "\n")


def _stderr(text: str) -> None:
    sys.stderr.write(_safe(text) + "\n")


def _usage_error(message: str) -> None:
    _stderr(f"Error: {message}")
    raise typer.Exit(2)


def _write_output(data: bytes, output: Path | None, force: bool) -> None:
    """Machine/terminal payload to stdout, or atomically to --output."""
    if output is None:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    else:
        write_bytes_atomic(output, data, force=force)


def _models_payload(models: list[BaseModel], single_json: bool) -> bytes:
    if single_json and len(models) == 1:
        return file_bytes(models[0])
    return jsonl_bytes(models)


def _emit(
    rendered: str,
    model: BaseModel,
    format: MachineFormat | None,
    output: Path | None,
    force: bool,
) -> None:
    """Single-model output dispatch: terminal rendering or portable bytes."""
    chosen = format or MachineFormat.TERMINAL
    if chosen is MachineFormat.TERMINAL:
        _write_output((_safe(rendered) + "\n").encode("ascii"), output, force)
    else:
        _write_output(file_bytes(model), output, force)


# ---------------------------------------------------------------------------
# Error mapping (spec 19.12)
# ---------------------------------------------------------------------------


def _command_guard(fn):
    """Map exceptions to CLI exit codes; keep callbacks free of try/except."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except typer.Exit:
            raise
        except KeyboardInterrupt:
            raise typer.Exit(130) from None
        except StylogError as exc:
            _stderr(f"Error: {exc}")
            raise typer.Exit(exc.exit_code) from None
        except Exception:
            if _STATE.get("debug"):
                traceback.print_exc()
            else:
                _stderr("Error: an internal Stylog failure occurred.")
                _stderr("Run again with '--debug' before the command for a traceback.")
            raise typer.Exit(5) from None

    return wrapper


def _welcome() -> str:
    """Task-oriented landing text for a bare ``stylog`` invocation."""
    from stylog import __version__

    return "\n".join(
        (
            f"Stylog {__version__} - stylometry for text and source code",
            "",
            "Usage:",
            "  stylog <command> [options]",
            "",
            "Commands:",
            "  fingerprint  Measure artifacts and emit portable fingerprints",
            "  analyze      Inspect an artifact and its embedded text artifacts",
            "  compare      Compare two artifacts feature-by-feature",
            "  profile      Compare an artifact with a population baseline",
            "  fit          Fit an authorship verifier from labeled data",
            "  verify       Evaluate two artifacts with a fitted verifier",
            "  represent    Fit or apply a sparse representation",
            "  report       Render a saved portable Stylog artifact",
            "  benchmark    Run a declarative benchmark specification",
            "  info         Show local capabilities and versions",
            "",
            "Examples:",
            "  stylog analyze app.py",
            "  stylog compare draft-a.txt draft-b.txt --language en",
            "  stylog fingerprint document.txt --format terminal",
            "",
            "Fingerprint writes canonical JSON by default; use --format terminal",
            "for a human summary or --format json on analytical commands for scripts.",
            "",
            "Run 'stylog --help' for global options or 'stylog COMMAND --help'.",
        )
    )


@app.callback(invoke_without_command=True)
def _app_callback(
    ctx: typer.Context,
    debug: Annotated[bool, typer.Option("--debug", help="Show tracebacks for internal errors.")] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the Stylog version and exit.",
            is_eager=True,
        ),
    ] = False,
) -> None:
    _STATE["debug"] = debug
    if version:
        from stylog import __version__

        _stdout(f"stylog {__version__}")
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        _stdout(_welcome())


# ---------------------------------------------------------------------------
# Shared option enums
# ---------------------------------------------------------------------------


class MachineFormat(str, Enum):
    JSON = "json"
    JSONL = "jsonl"
    TERMINAL = "terminal"


class FingerprintFormat(str, Enum):
    """Fingerprint formats, including its analytics-only Parquet export."""

    JSON = "json"
    JSONL = "jsonl"
    TERMINAL = "terminal"
    PARQUET = "parquet"  # analytics export only (requires stylog[data]); not canonical interchange


class Kind(str, Enum):
    TEXT = "text"
    CODE = "code"


class RepresentationId(str, Enum):
    CHAR_NGRAM_COUNT = CLI_TOKENS["char_ngram_count"]
    WORD_NGRAM_COUNT = CLI_TOKENS["word_ngram_count"]
    CHAR_TFIDF = CLI_TOKENS["char_tfidf"]
    WORD_TFIDF = CLI_TOKENS["word_tfidf"]


_REPRESENTATION_IDS = {
    RepresentationId.CHAR_NGRAM_COUNT: SPECS["char_ngram_count"].short_id,
    RepresentationId.WORD_NGRAM_COUNT: SPECS["word_ngram_count"].short_id,
    RepresentationId.CHAR_TFIDF: SPECS["char_tfidf"].short_id,
    RepresentationId.WORD_TFIDF: SPECS["word_tfidf"].short_id,
}


# ---------------------------------------------------------------------------
# Shared runtime preparation and input expansion
# ---------------------------------------------------------------------------


class _Runtime:
    def __init__(self, config: StylogConfig, services, ctx: AnalysisContext) -> None:
        self.config = config
        self.services = services
        self.ctx = ctx


def _prepare(
    config_path: Path | None,
    *,
    no_content_hash: bool = False,
    nlp_model: str | None = None,
    cache_dir: Path | None = None,
    no_cache: bool = False,
) -> _Runtime:
    """Resolve config and build default services/context (bootstrap owns env)."""
    config = load_config(config_path)
    if no_content_hash:
        config = config.model_copy(
            update={
                "analysis": config.analysis.model_copy(
                    update={"export_content_hashes": False}
                )
            }
        )
    if nlp_model:
        config = config.model_copy(update={"nlp": NlpConfig(enabled=True, model=nlp_model)})
    if no_cache:
        # Cache enablement is excluded from analysis_config_sha256, so this
        # copy only steers bootstrap (also inside batch workers).
        config = config.model_copy(
            update={"cache": config.cache.model_copy(update={"enabled": False})}
        )
    services = build_default_services(config, cache_dir=cache_dir, no_cache=no_cache)
    ctx = build_context(config, services, nlp_model_name=nlp_model)
    return _Runtime(config, services, ctx)


@contextlib.contextmanager
def _cache_dir_env(cache_dir: Path | None):
    """Batch workers rebuild services from env; mirror --cache-dir there."""
    if cache_dir is None:
        yield
        return
    previous = os.environ.get("STYLOG_CACHE_DIR")
    os.environ["STYLOG_CACHE_DIR"] = str(cache_dir)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("STYLOG_CACHE_DIR", None)
        else:
            os.environ["STYLOG_CACHE_DIR"] = previous


def _expand_inputs(
    inputs: list[str],
    config: StylogConfig,
    kind: Kind | None,
    language: str | None,
) -> tuple[list[RuntimeArtifact], int]:
    """Expand files/one directory/stdin into runtime artifacts.

    Per-file ingest failures are reported to stderr and counted; the caller
    decides the final exit code (3 when any input failed).
    """
    kind_value = kind.value if kind is not None else "auto"
    language_value = language if language is not None else "auto"
    artifacts: list[RuntimeArtifact] = []
    failures = 0
    stdin_seen = False
    for raw in inputs:
        if raw == "-":
            if stdin_seen:
                _usage_error("only one stdin source ('-') is allowed")
            stdin_seen = True
            stdin_kind = kind.value if kind is not None else "text"
            if language is not None:
                stdin_language = language
            elif stdin_kind == "text":
                stdin_language = "und"
            else:
                _usage_error("--language is required for stdin when --kind code is used")
            try:
                artifacts.append(
                    artifact_from_stdin(
                        sys.stdin.buffer.read(),
                        artifact_id="stdin",
                        kind=stdin_kind,
                        language=stdin_language,
                        encoding=config.input.text_encoding,
                        config=config,
                    )
                )
            except StylogError as exc:
                _stderr(f"Error: {exc}")
                failures += 1
            continue
        path = Path(raw)
        if path.is_dir():
            selected = select_files(path, config.input)
            for item in selected:
                if item.absolute_path.is_symlink():
                    _stderr(f"Warning  SYMLINK_REJECTED artifact={item.relative_path}")
                    continue
                try:
                    artifacts.append(
                        artifact_from_file(
                            item.absolute_path,
                            artifact_id=item.relative_path,
                            kind=kind_value,
                            language=language_value,
                            config=config,
                        )
                    )
                except StylogError as exc:
                    _stderr(f"Error: {exc}")
                    failures += 1
            continue
        try:
            artifacts.append(
                artifact_from_file(
                    path,
                    artifact_id=path.name,
                    kind=kind_value,
                    language=language_value,
                    config=config,
                )
            )
        except StylogError as exc:
            _stderr(f"Error: {exc}")
            failures += 1
    return artifacts, failures


def _analyze(
    runtime: _Runtime,
    artifacts: list[RuntimeArtifact],
    *,
    workers: int,
    refresh: bool,
) -> list[AnalysisBundle]:
    """Run the application analysis path in ordinal order."""
    if refresh:
        # fingerprint_iter exposes no refresh channel; refresh runs the same
        # analyze use case serially in-process (skips cache reads, rewrites).
        if workers > 1:
            _stderr("Notice: --refresh uses serial execution.")
        return [
            analyze_artifact(
                artifact,
                config=runtime.config,
                services=runtime.services,
                ctx=runtime.ctx,
                refresh=True,
            )[0]
            for artifact in artifacts
        ]
    execution = "process" if workers > 1 else "serial"
    bundles: list[AnalysisBundle] = []
    internal_error = False
    for bundle, had_internal_error in analyze_iter(
        artifacts,
        config=runtime.config,
        execution=execution,
        workers=workers,
    ):
        bundles.append(bundle)
        internal_error = internal_error or had_internal_error
    if internal_error:
        raise InternalStylogError("an analyzer reported an internal error")
    return bundles


def _make_evidence_set(
    artifacts: list[RuntimeArtifact], linkage: str, linkage_source: str
) -> EvidenceSet:
    members = tuple(
        EvidenceMember(member_id=f"m{ordinal:06d}", artifact_id=artifact.artifact_id)
        for ordinal, artifact in enumerate(artifacts)
    )
    material = json.dumps(
        {
            "linkage": [linkage, linkage_source],
            "members": [member.artifact_id for member in members],
        },
        sort_keys=True,
    )
    return EvidenceSet(
        evidence_set_id="es-" + sha256_hex(material.encode("utf-8"))[:16],
        members=members,
        linkage=LinkageDescriptor(kind=linkage, source=linkage_source),
    )


def _check_collection_args(
    collection: bool, linkage: str | None, linkage_source: str | None
) -> None:
    if collection and (not linkage or not linkage_source):
        _usage_error("--collection requires --linkage and --linkage-source")
    if (linkage or linkage_source) and not collection:
        _usage_error("--linkage and --linkage-source require --collection")


# ---------------------------------------------------------------------------
# Terminal renderers (human mode; ASCII only; stdout)
# ---------------------------------------------------------------------------


def _fmt_float(value: float) -> str:
    return f"{value:.6g}"


_FEATURE_STATUS_ORDER = (
    "ok",
    "insufficient_support",
    "not_applicable",
    "unavailable",
    "parser_error",
    "disabled",
)


def _kv_lines(pairs, *, indent: int = 0) -> list[str]:
    """Render a compact, aligned vertical record."""
    values = [(str(label), str(value)) for label, value in pairs]
    if not values:
        return []
    width = max(len(label) for label, _ in values)
    prefix = " " * indent
    return [f"{prefix}{label:<{width}}  {value}" for label, value in values]


def _feature_status_counts(observations) -> dict[str, int]:
    """Return complete, non-overlapping feature-status counts."""
    counts = {status: 0 for status in _FEATURE_STATUS_ORDER}
    for observation in observations:
        status = str(observation.status)
        counts[status] = counts.get(status, 0) + 1
    return counts


def _feature_status_summary(observations) -> str:
    counts = _feature_status_counts(observations)
    ordered = list(_FEATURE_STATUS_ORDER) + sorted(
        status for status in counts if status not in _FEATURE_STATUS_ORDER
    )
    parts = [f"{sum(counts.values()):,} total"]
    parts.extend(
        f"{counts[status]:,} {status.replace('_', ' ')}"
        for status in ordered
        if counts.get(status, 0) > 0
    )
    return "; ".join(parts)


def _render_diagnostic(diagnostic) -> str:
    parts = [diagnostic.severity.value.upper(), diagnostic.code]
    if diagnostic.analyzer_id:
        parts.append(f"analyzer={diagnostic.analyzer_id}")
    if diagnostic.feature_id:
        parts.append(f"feature={diagnostic.feature_id}")
    if diagnostic.artifact_id:
        parts.append(f"artifact={diagnostic.artifact_id}")
    for entry in diagnostic.context:
        parts.append(f"{entry.key}={entry.value}")
    return " ".join(parts)


def _diagnostic_count(diagnostics) -> str:
    return "none" if not diagnostics else f"{len(diagnostics):,}"


def _diagnostic_details(diagnostics) -> list[str]:
    if not diagnostics:
        return []
    lines = ["", "Diagnostic details"]
    lines.extend(f"  {_render_diagnostic(diagnostic)}" for diagnostic in diagnostics)
    return lines


def _render_diagnostics(diagnostics) -> list[str]:
    lines = _kv_lines((("Diagnostics", _diagnostic_count(diagnostics)),))
    lines.extend(_diagnostic_details(diagnostics))
    return lines


def _render_fingerprint(fingerprint: Fingerprint) -> str:
    artifact = fingerprint.artifact
    lines = ["Fingerprint", ""]
    lines.extend(
        _kv_lines(
            (
                ("Artifact", artifact.artifact_id),
                ("Kind", artifact.kind.value),
                ("Language", artifact.language),
                ("Encoding", artifact.encoding),
                (
                    "Size",
                    (
                        f"{artifact.byte_count:,} bytes; "
                        f"{artifact.character_count:,} Unicode code points"
                    ),
                ),
                ("Features", _feature_status_summary(fingerprint.features)),
                ("Diagnostics", _diagnostic_count(fingerprint.diagnostics)),
            )
        )
    )
    lines.extend(_diagnostic_details(fingerprint.diagnostics))
    return "\n".join(lines)


def _render_bundle(bundle: AnalysisBundle) -> str:
    primary = bundle.primary
    artifact = primary.artifact
    families: dict[str, list] = {}
    for obs in primary.features:
        family = obs.feature_id.rsplit(".", 1)[0]
        families.setdefault(family, []).append(obs)
    diagnostics = tuple(primary.diagnostics) + tuple(bundle.diagnostics)
    lines = ["Analysis", ""]
    lines.extend(
        _kv_lines(
            (
                ("Artifact", artifact.artifact_id),
                ("Kind", artifact.kind.value),
                ("Language", artifact.language),
                ("Embedded text", f"{len(bundle.embedded):,} comments/docstrings"),
                ("Diagnostics", _diagnostic_count(diagnostics)),
            )
        )
    )
    lines.extend(("", "Feature families", ""))
    family_width = max((len(family) for family in families), default=len("FAMILY"))
    lines.append(f"  {'FAMILY':<{family_width}}  STATUS")
    for family in sorted(families):
        lines.append(
            f"  {family:<{family_width}}  {_feature_status_summary(families[family])}"
        )
    lines.extend(_diagnostic_details(diagnostics))
    return "\n".join(lines)


def _render_aggregate(aggregate: EvidenceAggregate) -> str:
    evidence_set = aggregate.evidence_set
    lines = ["Evidence aggregate", ""]
    lines.extend(
        _kv_lines(
            (
                ("Evidence set", evidence_set.evidence_set_id),
                ("Members", f"{len(evidence_set.members):,}"),
                (
                    "Linkage",
                    f"{evidence_set.linkage.kind} / {evidence_set.linkage.source}",
                ),
                ("Features", f"{len(aggregate.aggregates):,}"),
                ("Diagnostics", _diagnostic_count(aggregate.diagnostics)),
            )
        )
    )
    lines.extend(("", "Aggregate features", ""))
    feature_width = max(
        (len(obs.feature_id) for obs in aggregate.aggregates), default=len("FEATURE")
    )
    lines.append(f"  {'FEATURE':<{feature_width}}  REDUCER          SUPPORT")
    for obs in aggregate.aggregates:
        lines.append(
            f"  {obs.feature_id:<{feature_width}}  {obs.reducer.value:<15}  "
            f"{obs.contributing_samples}/{obs.total_samples} samples"
        )
    lines.extend(_diagnostic_details(aggregate.diagnostics))
    return "\n".join(lines)


def _render_comparison(comparison: Comparison) -> str:
    metric_explanations = {
        "ABS": "Absolute difference; units are feature-specific",
        "SPD": "Symmetric proportional distance; range [0,2]",
        "JSD2": "Base-2 Jensen-Shannon distance (sqrt divergence); range [0,1]",
        "W1": "Wasserstein-1 distance; units are feature-specific",
        "sample_wasserstein_1": "Wasserstein-1 distance across sample values",
    }
    metrics = {
        component.metric
        for family in comparison.families
        for component in family.components
    }
    component_count = sum(len(family.components) for family in comparison.families)
    lines = ["Comparison", ""]
    lines.extend(
        _kv_lines(
            (
                ("Left", comparison.left_ref),
                ("Right", comparison.right_ref),
                ("Comparable features", f"{component_count:,}"),
                ("Diagnostics", _diagnostic_count(comparison.diagnostics)),
            )
        )
    )
    lines.extend(
        (
            "",
            "Interpretation",
            "  Distances are feature-specific; Stylog does not combine them into",
            "  an overall similarity score. A distance of 0 means no measured",
            "  difference for that feature.",
        )
    )
    if metrics:
        lines.extend(("", "Metric definitions"))
        metric_width = max(len(metric) for metric in metrics)
        for metric, explanation in metric_explanations.items():
            if metric in metrics:
                lines.append(f"  {metric:<{metric_width}}  {explanation}")
    for family in comparison.families:
        lines.extend(("", family.family, ""))
        rows = []
        for component in family.components:
            prefix = family.family + "."
            feature = component.feature_id.removeprefix(prefix)
            if component.left_support.kind == component.right_support.kind:
                support = (
                    f"{component.left_support.count}/{component.right_support.count} "
                    f"{component.left_support.kind}"
                )
            else:
                support = (
                    f"{component.left_support.count} {component.left_support.kind} / "
                    f"{component.right_support.count} {component.right_support.kind}"
                )
            rows.append(
                (
                    feature,
                    component.metric,
                    _fmt_float(component.value),
                    support,
                    component.unit,
                )
            )
        feature_width = max(len("FEATURE"), *(len(row[0]) for row in rows))
        metric_width = max(len("METRIC"), *(len(row[1]) for row in rows))
        value_width = max(len("DISTANCE"), *(len(row[2]) for row in rows))
        support_width = max(len("SUPPORT L/R"), *(len(row[3]) for row in rows))
        lines.append(
            f"  {'FEATURE':<{feature_width}}  {'METRIC':<{metric_width}}  "
            f"{'DISTANCE':>{value_width}}  {'SUPPORT L/R':<{support_width}}  UNIT"
        )
        for feature, metric, value, support, unit in rows:
            lines.append(
                f"  {feature:<{feature_width}}  {metric:<{metric_width}}  "
                f"{value:>{value_width}}  {support:<{support_width}}  {unit}"
            )
    lines.extend(_diagnostic_details(comparison.diagnostics))
    return "\n".join(lines)


def _render_profile(profile: Profile) -> str:
    lines = ["Profile", ""]
    lines.extend(
        _kv_lines(
            (
                ("Subject", profile.subject_ref),
                ("Baseline", f"{profile.baseline_id} {profile.baseline_version}"),
                ("Features", f"{len(profile.observations):,}"),
                ("Diagnostics", _diagnostic_count(profile.diagnostics)),
            )
        )
    )
    lines.extend(
        (
            "",
            "Interpretation",
            "  Midrank is a percentile from 0 to 100. Robust z uses the",
            "  baseline median and normal-scaled MAD; '-' means unavailable.",
            "",
        )
    )
    feature_width = max(
        (len(obs.feature_id) for obs in profile.observations), default=len("FEATURE")
    )
    lines.append(
        f"  {'FEATURE':<{feature_width}}  {'OBSERVED':>10}  {'N':>6}  "
        f"{'MIDRANK %':>10}  {'ROBUST Z':>10}"
    )
    for obs in profile.observations:
        robust_z = "-" if obs.robust_z is None else _fmt_float(obs.robust_z)
        lines.append(
            f"  {obs.feature_id:<{feature_width}}  "
            f"{_fmt_float(obs.observed_value):>10}  {obs.baseline_n:>6}  "
            f"{_fmt_float(obs.percentile_midrank):>10}  {robust_z:>10}"
        )
    lines.extend(_diagnostic_details(profile.diagnostics))
    return "\n".join(lines)


def _render_verification(verification: Verification) -> str:
    """ASCII terminal rendering of a Verification (spec 19/23)."""
    if verification.score is None:
        score = "not available (insufficient evidence)"
        probability = "not available (insufficient evidence)"
    else:
        score = f"{_fmt_float(verification.score)} (range (0,1); not a probability)"
        if verification.probability is not None:
            probability = (
                f"{_fmt_float(verification.probability)} same-author "
                f"({verification.calibration_method}; calibration-population conditional)"
            )
        else:
            probability = "not available (uncalibrated model)"
    lines = ["Verification", ""]
    pairs = [
        ("Verdict", verification.verdict),
        ("Model score", score),
        ("Probability", probability),
        (
            "Features",
            (
                f"{verification.features_used} used; "
                f"{len(verification.features_missing)} unavailable"
            ),
        ),
        ("Model", f"{verification.model_id} {verification.model_semantic_version}"),
        ("Verifier ID", verification.verifier_id),
        ("Left", verification.left_ref),
        ("Left fingerprint", verification.left_fingerprint_sha256),
        ("Right", verification.right_ref),
        ("Right fingerprint", verification.right_fingerprint_sha256),
        ("Diagnostics", _diagnostic_count(verification.diagnostics)),
    ]
    if verification.abstain_reason is not None:
        pairs.insert(1, ("Abstain reason", verification.abstain_reason))
    lines.extend(_kv_lines(pairs))
    lines.extend(
        (
            "",
            "Interpretation",
            "  Higher model scores support same_author under this fitted verifier.",
            "  Scores and verdicts are model-relative evidence, not proof of identity.",
            "  A calibrated probability is conditional on its calibration population.",
        )
    )
    if verification.features_missing:
        lines.extend(("", "Unavailable features"))
    for feature_id in verification.features_missing:
        lines.append(f"  {feature_id}")
    lines.extend(_diagnostic_details(verification.diagnostics))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portable artifact loading (report / --from-artifact(s))
# ---------------------------------------------------------------------------

_PORTABLE_MODELS: dict[str, type[BaseModel]] = PORTABLE_MODELS_BY_SCHEMA

def _validate_portable(data: bytes, model_type: type[BaseModel], source: str):
    from stylog.serialization.jsonio import model_from_bytes

    try:
        return model_from_bytes(data, model_type)
    except PortableArtifactError as exc:
        raise PortableArtifactError(f"{source}: {exc}") from exc


def _read_portable(path_value: str, allowed: dict[str, type[BaseModel]] | None = None):
    path = Path(path_value)
    if not path.is_file():
        raise InputError(f"input not found (INPUT_NOT_FOUND): {path.name}")
    data = path.read_bytes()
    try:
        tree = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableArtifactError(f"invalid portable JSON in {path.name}: {exc}") from exc
    if not isinstance(tree, dict) or not isinstance(tree.get("schema"), str):
        raise PortableArtifactError(f"portable artifact has no schema member: {path.name}")
    models = _PORTABLE_MODELS if allowed is None else allowed
    model_type = models.get(tree["schema"])
    if model_type is None:
        raise PortableArtifactError(
            f"unsupported portable schema {tree['schema']!r}: {path.name}"
        )
    return _validate_portable(data, model_type, path.name)


# ---------------------------------------------------------------------------
# Common option annotations
# ---------------------------------------------------------------------------

_ConfigOpt = Annotated[Path | None, typer.Option("--config", help="Explicit stylog TOML config.")]
_NlpModelOpt = Annotated[str | None, typer.Option("--nlp-model", help="Provisioned spaCy model (nlp extra).")]
_CacheDirOpt = Annotated[Path | None, typer.Option("--cache-dir", help="Override the cache root.")]
_NoCacheOpt = Annotated[bool, typer.Option("--no-cache", help="Disable cache reads and writes.")]
_RefreshOpt = Annotated[bool, typer.Option("--refresh", help="Skip cache reads; recompute and rewrite.")]
_NoContentHashOpt = Annotated[bool, typer.Option("--no-content-hash", help="Suppress exported content hashes.")]
_OutputOpt = Annotated[Path | None, typer.Option("--output", "-o", help="Write to PATH instead of stdout.")]
_ForceOpt = Annotated[bool, typer.Option("--force", help="Allow overwriting an existing --output.")]
_KindOpt = Annotated[Kind | None, typer.Option("--kind", help="Override artifact kind.")]
_LanguageOpt = Annotated[str | None, typer.Option("--language", help="Override artifact language.")]
_WorkersOpt = Annotated[int, typer.Option("--workers", help="Process-pool workers (>1 enables it).")]
_CollectionOpt = Annotated[bool, typer.Option("--collection", help="Treat inputs as one EvidenceSet.")]
_LinkageOpt = Annotated[str | None, typer.Option("--linkage", help="Evidence linkage kind.")]
_LinkageSourceOpt = Annotated[str | None, typer.Option("--linkage-source", help="Evidence linkage source.")]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _command_examples(*commands: str) -> str:
    return "Examples:\n\n\b\n" + "\n".join(f"  {command}" for command in commands)


@app.command(
    epilog=_command_examples(
        "stylog fingerprint document.txt --format terminal",
        "stylog fingerprint corpus -o fingerprints.jsonl",
    )
)
@_command_guard
def fingerprint(
    inputs: Annotated[
        list[str],
        typer.Argument(help="Files, one directory, or '-' for stdin.", metavar="INPUT..."),
    ],
    format: Annotated[
        FingerprintFormat | None,
        typer.Option(
            "--format",
            help=(
                "Output format. Default: JSON for one result, JSONL for several; "
                "Parquet requires stylog[data]."
            ),
        ),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    collection: _CollectionOpt = False,
    linkage: _LinkageOpt = None,
    linkage_source: _LinkageSourceOpt = None,
    workers: _WorkersOpt = 1,
    no_cache: _NoCacheOpt = False,
    refresh: _RefreshOpt = False,
    cache_dir: _CacheDirOpt = None,
    no_content_hash: _NoContentHashOpt = False,
    config: _ConfigOpt = None,
    nlp_model: _NlpModelOpt = None,
) -> None:
    """Measure artifacts and emit portable fingerprints."""
    _check_collection_args(collection, linkage, linkage_source)
    runtime = _prepare(
        config,
        no_content_hash=no_content_hash,
        nlp_model=nlp_model,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    artifacts, failures = _expand_inputs(inputs, runtime.config, kind, language)
    with _cache_dir_env(cache_dir):
        bundles = _analyze(runtime, artifacts, workers=workers, refresh=refresh)
    fingerprints = [bundle.primary for bundle in bundles]

    aggregate: EvidenceAggregate | None = None
    if collection:
        if failures == 0 and artifacts:
            evidence_set = _make_evidence_set(artifacts, linkage or "", linkage_source or "")
            aggregate = aggregate_evidence(evidence_set, fingerprints)
        else:
            _stderr("Warning: aggregate omitted because one or more inputs failed.")

    chosen = format
    if chosen is None:
        chosen = (
            FingerprintFormat.JSON
            if len(fingerprints) == 1 and aggregate is None
            else FingerprintFormat.JSONL
        )
    if chosen is FingerprintFormat.TERMINAL:
        blocks = [_render_fingerprint(fp) for fp in fingerprints]
        if aggregate is not None:
            blocks.append(_render_aggregate(aggregate))
        _write_output((_safe("\n\n".join(blocks)) + "\n").encode("ascii"), output, force)
    elif chosen is FingerprintFormat.PARQUET:
        if output is None:
            _usage_error("--format parquet requires --output PATH")
        try:
            from stylog.data import write_parquet
        except ImportError:
            raise CapabilityUnavailableError(
                "the 'data' capability requires: pip install stylog[data]"
            ) from None
        write_parquet(list(fingerprints), output, config=runtime.config, force=force)
    else:
        models: list[BaseModel] = list(fingerprints)
        if aggregate is not None:
            models.append(aggregate)
        _write_output(
            _models_payload(models, single_json=chosen is FingerprintFormat.JSON),
            output,
            force,
        )
    if failures:
        raise typer.Exit(3)


@app.command(
    epilog=_command_examples(
        "stylog analyze app.py",
        "stylog analyze document.txt --format json",
    )
)
@_command_guard
def analyze(
    input_: Annotated[
        str,
        typer.Argument(help="One file, one directory, or '-' for stdin.", metavar="INPUT"),
    ],
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: terminal."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    collection: _CollectionOpt = False,
    linkage: _LinkageOpt = None,
    linkage_source: _LinkageSourceOpt = None,
    baseline: Annotated[str | None, typer.Option("--baseline", help="Baseline ref or path for profiling.")] = None,
    workers: _WorkersOpt = 1,
    no_cache: _NoCacheOpt = False,
    refresh: _RefreshOpt = False,
    cache_dir: _CacheDirOpt = None,
    no_content_hash: _NoContentHashOpt = False,
    config: _ConfigOpt = None,
    nlp_model: _NlpModelOpt = None,
) -> None:
    """Inspect an artifact and its embedded text artifacts."""
    _check_collection_args(collection, linkage, linkage_source)
    if baseline is not None and collection:
        _usage_error("--baseline cannot be combined with --collection")
    runtime = _prepare(
        config,
        no_content_hash=no_content_hash,
        nlp_model=nlp_model,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    artifacts, failures = _expand_inputs([input_], runtime.config, kind, language)
    with _cache_dir_env(cache_dir):
        bundles = _analyze(runtime, artifacts, workers=workers, refresh=refresh)
    if baseline is not None and len(bundles) != 1:
        _usage_error("--baseline requires exactly one input artifact")

    aggregate: EvidenceAggregate | None = None
    if collection:
        if failures == 0 and artifacts:
            evidence_set = _make_evidence_set(artifacts, linkage or "", linkage_source or "")
            aggregate = aggregate_evidence(
                evidence_set, [bundle.primary for bundle in bundles]
            )
        else:
            _stderr("Warning: aggregate omitted because one or more inputs failed.")

    profile: Profile | None = None
    if baseline is not None:
        primary = bundles[0].primary
        profile = profile_subject(
            primary,
            baseline,
            services=runtime.services,
            subject_ref=primary.artifact.artifact_id,
        )

    chosen = format or MachineFormat.TERMINAL
    if chosen is MachineFormat.TERMINAL:
        blocks = [_render_bundle(bundle) for bundle in bundles]
        if aggregate is not None:
            blocks.append(_render_aggregate(aggregate))
        if profile is not None:
            blocks.append(_render_profile(profile))
        _write_output((_safe("\n\n".join(blocks)) + "\n").encode("ascii"), output, force)
    else:
        models: list[BaseModel] = list(bundles)
        if aggregate is not None:
            models.append(aggregate)
        if profile is not None:
            models.append(profile)
        _write_output(
            _models_payload(models, single_json=chosen is MachineFormat.JSON),
            output,
            force,
        )
    if failures:
        raise typer.Exit(3)


@app.command(
    epilog=_command_examples(
        "stylog profile document.txt --baseline baseline.json",
        "stylog profile fingerprint.json --from-artifact --baseline baseline.json",
    )
)
@_command_guard
def profile(
    source: Annotated[
        str,
        typer.Argument(
            help="Input file, '-' for stdin, or Fingerprint JSON with --from-artifact.",
            metavar="SOURCE",
        ),
    ],
    baseline: Annotated[str, typer.Option("--baseline", help="Baseline id or path (mandatory).")],
    from_artifact: Annotated[bool, typer.Option("--from-artifact", help="SOURCE is a portable Fingerprint JSON.")] = False,
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: terminal."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    no_cache: _NoCacheOpt = False,
    refresh: _RefreshOpt = False,
    cache_dir: _CacheDirOpt = None,
    no_content_hash: _NoContentHashOpt = False,
    config: _ConfigOpt = None,
    nlp_model: _NlpModelOpt = None,
) -> None:
    """Compare an artifact with an explicit population baseline."""
    runtime = _prepare(
        config,
        no_content_hash=no_content_hash,
        nlp_model=nlp_model,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    failures = 0
    if from_artifact:
        if source == "-":
            _usage_error("--from-artifact requires a file path, not stdin")
        subject_path = Path(source)
        if not subject_path.is_file():
            raise InputError(f"input not found (INPUT_NOT_FOUND): {subject_path.name}")
        subject = _validate_portable(subject_path.read_bytes(), Fingerprint, subject_path.name)
    else:
        artifacts, failures = _expand_inputs([source], runtime.config, kind, language)
        if failures or not artifacts:
            raise typer.Exit(3)
        with _cache_dir_env(cache_dir):
            bundle = _analyze(runtime, artifacts, workers=1, refresh=refresh)[0]
        subject = bundle.primary
    result = profile_subject(
        subject,
        baseline,
        services=runtime.services,
        subject_ref=subject.artifact.artifact_id,
    )
    _emit(_render_profile(result), result, format, output, force)


def _dual_subject(
    left: str,
    right: str,
    *,
    from_artifacts: bool,
    command: str,
    operation: Callable[..., BaseModel],
    kind: Kind | None,
    language: str | None,
    no_cache: bool,
    refresh: bool,
    cache_dir: Path | None,
    no_content_hash: bool,
    config: Path | None,
    nlp_model: str | None,
) -> BaseModel:
    """Resolve two subjects (portable artifacts or fresh analysis) and run ``operation``."""
    if from_artifacts:
        allowed = {
            name: PORTABLE_MODELS_BY_SCHEMA[name]
            for name in ("stylog.fingerprint", "stylog.analysis")
        }
        left_obj = _read_portable(left, allowed)
        right_obj = _read_portable(right, allowed)
        return operation(left_obj, right_obj, left_ref=Path(left).name, right_ref=Path(right).name)
    runtime = _prepare(
        config,
        no_content_hash=no_content_hash,
        nlp_model=nlp_model,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )
    artifacts, failures = _expand_inputs([left, right], runtime.config, kind, language)
    if failures:
        raise typer.Exit(3)
    if len(artifacts) != 2:
        _usage_error(f"{command} requires exactly two input artifacts")
    with _cache_dir_env(cache_dir):
        bundles = _analyze(runtime, artifacts, workers=1, refresh=refresh)
    return operation(
        bundles[0],
        bundles[1],
        left_ref=artifacts[0].artifact_id,
        right_ref=artifacts[1].artifact_id,
    )


@app.command(
    epilog=_command_examples(
        "stylog compare draft-a.txt draft-b.txt --language en",
        "stylog compare left.json right.json --from-artifacts",
    )
)
@_command_guard
def compare(
    left: Annotated[
        str,
        typer.Argument(
            help="Left input file (or portable JSON with --from-artifacts).",
            metavar="LEFT",
        ),
    ],
    right: Annotated[
        str,
        typer.Argument(
            help="Right input file (or portable JSON with --from-artifacts).",
            metavar="RIGHT",
        ),
    ],
    from_artifacts: Annotated[bool, typer.Option("--from-artifacts", help="LEFT/RIGHT are portable JSON artifacts.")] = False,
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: terminal."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    no_cache: _NoCacheOpt = False,
    refresh: _RefreshOpt = False,
    cache_dir: _CacheDirOpt = None,
    no_content_hash: _NoContentHashOpt = False,
    config: _ConfigOpt = None,
    nlp_model: _NlpModelOpt = None,
) -> None:
    """Compare two artifacts feature-by-feature; no overall score."""
    comparison = _dual_subject(
        left,
        right,
        from_artifacts=from_artifacts,
        command="compare",
        operation=compare_subjects,
        kind=kind,
        language=language,
        no_cache=no_cache,
        refresh=refresh,
        cache_dir=cache_dir,
        no_content_hash=no_content_hash,
        config=config,
        nlp_model=nlp_model,
    )
    _emit(_render_comparison(comparison), comparison, format, output, force)


@app.command(
    epilog=_command_examples(
        "stylog verify left.txt right.txt --model verifier.json --language en",
        "stylog verify left.json right.json --from-artifacts --model verifier.json",
    )
)
@_command_guard
def verify(
    left: Annotated[
        str,
        typer.Argument(
            help="Left input file (or portable JSON with --from-artifacts).",
            metavar="LEFT",
        ),
    ],
    right: Annotated[
        str,
        typer.Argument(
            help="Right input file (or portable JSON with --from-artifacts).",
            metavar="RIGHT",
        ),
    ],
    model: Annotated[Path, typer.Option("--model", "-m", help="Portable stylog.verifier-fit JSON (required).")],
    from_artifacts: Annotated[bool, typer.Option("--from-artifacts", help="LEFT/RIGHT are portable JSON artifacts.")] = False,
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: terminal."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    no_cache: _NoCacheOpt = False,
    refresh: _RefreshOpt = False,
    cache_dir: _CacheDirOpt = None,
    no_content_hash: _NoContentHashOpt = False,
    config: _ConfigOpt = None,
    nlp_model: _NlpModelOpt = None,
) -> None:
    """Evaluate two artifacts with a fitted authorship verifier.

    The verdict is model-relative support, never an identity claim. Abstain is
    a successful outcome (exit 0); incompatibility is a typed error (exit 4).
    """
    verifier = _read_portable(str(model), {"stylog.verifier-fit": VerifierFit})
    verification = _dual_subject(
        left,
        right,
        from_artifacts=from_artifacts,
        command="verify",
        operation=lambda left_obj, right_obj, **refs: verify_subjects(
            left_obj, right_obj, verifier, **refs
        ),
        kind=kind,
        language=language,
        no_cache=no_cache,
        refresh=refresh,
        cache_dir=cache_dir,
        no_content_hash=no_content_hash,
        config=config,
        nlp_model=nlp_model,
    )
    _emit(_render_verification(verification), verification, format, output, force)


@app.command(
    name="fit",
    epilog=_command_examples("stylog fit training.toml -o verifier.json"),
)
@app.command(name="verify-fit", hidden=True)  # backwards-compatible alias
@_command_guard
def verify_fit(
    training: Annotated[
        str,
        typer.Argument(
            help="stylog.verifier-training TOML manifest.", metavar="MANIFEST"
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Write the VerifierFit JSON here.")],
    force: _ForceOpt = False,
    config: _ConfigOpt = None,
) -> None:
    """Fit an authorship verifier from a labeled training manifest.

    Train pairs drive eligibility, normalization, and coefficients. Calibration
    pairs drive thresholds and Platt scaling. Tuning pairs record only the
    identity of an external, pre-fit selection population. Diagnostics go to
    stderr; stdout stays empty.
    """
    from stylog.benchmark.train import fit_verifier_from_manifest

    cfg = load_config(config)
    model, diagnostics = fit_verifier_from_manifest(Path(training), config=cfg)
    from stylog.serialization.canonical import scientific_sha256

    for diagnostic in diagnostics:
        _stderr(f"Diagnostic  {_render_diagnostic(diagnostic)}")
    _stderr(f"Verifier ID  {scientific_sha256(model)}")
    _write_output(file_bytes(model), output, force)


@app.command(
    epilog=_command_examples("stylog report fingerprint.json")
)
@_command_guard
def report(
    result: Annotated[
        str,
        typer.Argument(help="Portable artifact JSON to display.", metavar="RESULT"),
    ],
) -> None:
    """Render a saved portable Stylog artifact without re-analysis."""
    obj = _read_portable(result)
    if isinstance(obj, Fingerprint):
        body = _render_fingerprint(obj)
    elif isinstance(obj, AnalysisBundle):
        body = _render_bundle(obj)
    elif isinstance(obj, Comparison):
        body = _render_comparison(obj)
    elif isinstance(obj, Profile):
        body = _render_profile(obj)
    elif isinstance(obj, EvidenceAggregate):
        body = _render_aggregate(obj)
    elif isinstance(obj, Verification):
        body = _render_verification(obj)
    else:
        # Baseline, EvidenceSet, Representation, RepresentationFit,
        # BenchmarkResult: validated generic summary.
        scalar_pairs = [
            (name.replace("_", " ").capitalize(), value)
            for name, value in obj.model_dump(exclude_none=True).items()
            if isinstance(value, (str, int, float)) and name != "schema"
        ]
        body = "\n".join(
            [type(obj).__name__, "", *_kv_lines(scalar_pairs)]
        )
    _stdout(f"Schema  {obj.schema}\n\n{body}")


@app.command(
    epilog=_command_examples(
        "stylog benchmark benchmark.toml",
        "stylog benchmark benchmark.toml --format json -o result.json",
    )
)
@_command_guard
def benchmark(
    spec: Annotated[
        str, typer.Argument(help="Benchmark specification TOML file.", metavar="SPEC")
    ],
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: terminal."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
) -> None:
    """Run a declarative benchmark specification.

    Tasks: split_audit, pairwise_comparison, transformation_stability,
    verification.
    """
    from stylog.benchmark.api import run_benchmark_file

    result = run_benchmark_file(Path(spec))
    pairs = [("Schema", getattr(result, "schema", "stylog.benchmark-result"))]
    for name in ("benchmark_id", "id", "task", "dataset_id"):
        value = getattr(result, name, None)
        if isinstance(value, str):
            pairs.append((name.replace("_", " ").capitalize(), value))
    diagnostics = getattr(result, "diagnostics", ())
    pairs.append(("Diagnostics", _diagnostic_count(diagnostics)))
    lines = ["Benchmark", "", *_kv_lines(pairs)]
    lines.extend(_diagnostic_details(diagnostics))
    _emit("\n".join(lines), result, format, output, force)


@app.command(
    epilog=_command_examples(
        "stylog represent corpus --representation word-tfidf --fit-output fit.json",
        "stylog represent document.txt --fit-resource fit.json --format terminal",
    )
)
@_command_guard
def represent(
    inputs: Annotated[
        list[str],
        typer.Argument(help="Input files or '-' for stdin.", metavar="INPUT..."),
    ],
    representation: Annotated[RepresentationId | None, typer.Option("--representation")] = None,
    fit_output: Annotated[Path | None, typer.Option("--fit-output", help="Fit and write the fit resource here.")] = None,
    fit_resource: Annotated[Path | None, typer.Option("--fit-resource", "--model", "-m", help="Transform using an existing fit resource.")] = None,
    format: Annotated[
        MachineFormat | None,
        typer.Option("--format", help="Output format. Default: JSON/JSONL."),
    ] = None,
    output: _OutputOpt = None,
    force: _ForceOpt = False,
    kind: _KindOpt = None,
    language: _LanguageOpt = None,
    config: _ConfigOpt = None,
) -> None:
    """Fit or apply a sparse representation (requires stylog[ml])."""
    repfit = require_capability("stylog.representations.fit", "ml")
    if (fit_output is None) == (fit_resource is None):
        _usage_error("exactly one of --fit-output or --fit-resource is required")
    runtime = _prepare(config)
    artifacts, failures = _expand_inputs(inputs, runtime.config, kind, language)

    if fit_output is not None:
        if representation is None:
            _usage_error("--representation is required together with --fit-output")
        fitted = repfit.fit_representation_cli(
            _REPRESENTATION_IDS[representation], artifacts, fit_output, force=force
        )
        if isinstance(fitted, BaseModel):
            _write_output(file_bytes(fitted), output, force)
    else:
        representations = list(repfit.transform_representation_cli(fit_resource, artifacts))
        chosen = format
        if chosen is None:
            chosen = MachineFormat.JSON if len(representations) == 1 else MachineFormat.JSONL
        if chosen is MachineFormat.TERMINAL:
            lines = [
                "\n".join(
                    [
                        "Representation",
                        "",
                        *_kv_lines(
                            (
                                ("Implementation", rep.representation_id),
                                ("Subject", rep.subject_ref),
                            )
                        ),
                    ]
                )
                for rep in representations
            ]
            _write_output(
                (_safe("\n\n".join(lines)) + "\n").encode("ascii"), output, force
            )
        else:
            _write_output(
                _models_payload(
                    list(representations), single_json=chosen is MachineFormat.JSON
                ),
                output,
                force,
            )
    if failures:
        raise typer.Exit(3)


@app.command(
    name="info",
    epilog=_command_examples(
        "stylog info",
        "stylog info --format json",
    ),
)
@app.command(name="capabilities", hidden=True)  # backwards-compatible alias
@_command_guard
def info(
    format: Annotated[
        MachineFormat | None, typer.Option("--format", help="Output format.")
    ] = None,
) -> None:
    """Show local capabilities and versions; no network."""
    from stylog import __version__
    from stylog.analysis.base import (
        CODE_SURFACE_BACKEND,
        PYTHON_AST_BACKEND,
        PYTHON_TOKENS_BACKEND,
        TEXT_BACKEND,
    )
    from stylog.analysis.engine import TREE_SITTER_LANGUAGES
    from stylog.analysis.verify import VERIFIER_MODEL_ID, VERIFIER_MODEL_SEMANTIC_VERSION
    from stylog.domain.provenance import current_runtime_signature
    from stylog.parsers.tree_sitter import load_manifest

    manifest = load_manifest()
    runtime = current_runtime_signature()
    code_languages = ["python", *TREE_SITTER_LANGUAGES]
    analyzers = {
        "text": "native text features (language-neutral core; English function words)",
        "python": "code-surface plus CPython tokenize and AST",
        **{
            language: "code-surface plus tree-sitter grammar"
            for language in TREE_SITTER_LANGUAGES
        },
    }

    optional = {}
    for module_name, extra in (
        ("spacy", "nlp"),
        ("sklearn", "ml"),
        ("pyarrow", "data"),
        ("polars", "data"),
        ("duckdb", "data"),
        ("pandas", "data"),
    ):
        optional[module_name] = {
            "extra": extra,
            "installed": importlib.util.find_spec(module_name) is not None,
        }

    sklearn_available = optional["sklearn"]["installed"]
    representation_ids = sorted(_REPRESENTATION_IDS.values()) if sklearn_available else []

    compat_ids: dict[str, object] = {
        "stylog.native.text": TEXT_BACKEND.scientific_compatibility_id,
        "stylog.native.code": CODE_SURFACE_BACKEND.scientific_compatibility_id,
        "cpython.tokenize": PYTHON_TOKENS_BACKEND.scientific_compatibility_id,
        "cpython.ast": PYTHON_AST_BACKEND.scientific_compatibility_id,
        "tree-sitter": {
            language: manifest[language].parser_compat_id
            for language in sorted(manifest)
        },
    }

    report = {
        "stylog_version": __version__,
        "code_languages": code_languages,
        "specialized_analyzers": analyzers,
        "runtime": {
            "platform": platform.platform(),
            "python_implementation": runtime.python_implementation,
            "python_version": runtime.python_version,
            "python_cache_tag": runtime.python_cache_tag,
            "unicode_database_version": runtime.unicode_database_version,
        },
        "tree_sitter_grammars": {
            language: {
                "grammar_id": manifest[language].grammar_id,
                "package": manifest[language].package,
                "installed_version": manifest[language].installed_version,
                "upstream_revision": manifest[language].upstream_revision,
                "node_types_sha256": manifest[language].node_types_sha256,
                "abi_version": manifest[language].abi_version,
                "parser_compat_id": manifest[language].parser_compat_id,
            }
            for language in sorted(manifest)
        },
        "optional_capabilities": optional,
        "representations": {
            "available": sklearn_available,
            "implementation_ids": representation_ids,
        },
        "verification": {
            "model_id": VERIFIER_MODEL_ID,
            "model_semantic_version": VERIFIER_MODEL_SEMANTIC_VERSION,
            "implementation_id": VERIFIER_MODEL_ID,
            "implementation_semantic_version": VERIFIER_MODEL_SEMANTIC_VERSION,
            "bundled_fitted_model": False,
            "scientific_compatibility_id": VERIFIER_MODEL_ID,
        },
        "scientific_compatibility_ids": compat_ids,
        "notes": [
            (
                "installed means importable; provisioned NLP models are never "
                "loaded or enumerated by this command"
            )
        ],
    }

    chosen = format or MachineFormat.TERMINAL
    if chosen is MachineFormat.JSON:
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        sys.stdout.buffer.write(payload.encode("utf-8"))
        return
    if chosen is MachineFormat.JSONL:
        payload = json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n"
        sys.stdout.buffer.write(payload.encode("utf-8"))
        return
    lines = ["Stylog info", ""]
    lines.extend(
        _kv_lines(
            (
                ("Version", report["stylog_version"]),
                (
                    "Python",
                    (
                        f"{runtime.python_implementation} {runtime.python_version} "
                        f"({runtime.python_cache_tag})"
                    ),
                ),
                ("Platform", report["runtime"]["platform"]),
                ("Unicode data", runtime.unicode_database_version),
                ("Code languages", ", ".join(code_languages)),
                ("Network", "not used"),
            )
        )
    )
    lines.extend(("", "Specialized analyzers"))
    for name, description in report["specialized_analyzers"].items():
        lines.append(f"  {name:<10}  {description}")
    lines.extend(("", "Tree-sitter grammars"))
    for language, entry in report["tree_sitter_grammars"].items():
        lines.append(
            f"  {language:<10}  {entry['grammar_id']} {entry['installed_version']} "
            f"abi={entry['abi_version']} compat={entry['parser_compat_id']}"
        )
    lines.extend(("", "Optional capabilities", "", "  PACKAGE     EXTRA  STATUS"))
    for module_name, entry in report["optional_capabilities"].items():
        state = "installed" if entry["installed"] else "not installed"
        lines.append(f"  {module_name:<11} {entry['extra']:<6} {state}")
    lines.extend(("", "Representations"))
    if report["representations"]["available"]:
        for rep_id in report["representations"]["implementation_ids"]:
            lines.append(f"  {rep_id}")
    else:
        lines.append("  Not available. Install stylog[ml] to enable them.")
    lines.extend(("", "Verification"))
    lines.extend(
        _kv_lines(
            (
                (
                    "Implementation",
                    (
                        f"{report['verification']['implementation_id']} "
                        f"{report['verification']['implementation_semantic_version']}"
                    ),
                ),
                ("Bundled fitted model", "none"),
                ("Next step", "run 'stylog fit', then pass its output with --model"),
            ),
            indent=2,
        )
    )
    lines.extend(
        (
            "",
            "Importable packages are reported as installed; NLP models are not probed.",
            "Use 'stylog info --format json' for full runtime and compatibility IDs.",
        )
    )
    _stdout("\n".join(lines))
