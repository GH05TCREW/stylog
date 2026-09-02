"""CLI contract tests (spec section 19).

Runs the Typer app in-process via ``typer.testing.CliRunner``. Machine output
is asserted on stdout only; diagnostics belong on stderr.

Note: portable models are validated here with ``strict=False`` because the
serialization layer currently cannot round-trip its own artifacts under the
models' strict=True config (pydantic v2 rejects JSON strings for StrEnum
fields). See the KNOWN UPSTREAM BUG note in src/stylog/cli.py.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from stylog.cli import app
from stylog.domain.evidence import EvidenceAggregate
from stylog.domain.fingerprint import AnalysisBundle, Fingerprint
from stylog.domain.interpretation import Comparison, Profile
from stylog.serialization.canonical import scientific_sha256

runner = CliRunner()

TEXT_ONE = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.\n"
TEXT_TWO = "How vexingly quick daft zebras jump! Bright vixens jump; dozy fowl quack.\n"
PY_SOURCE = 'def greet(name):\n    # say hello\n    return f"hello {name}"\n\n\nx = greet("world")\n'
JS_SOURCE = 'function add(a, b) {\n  // sum them\n  return a + b;\n}\nconst total = add(1, 2);\n'


@pytest.fixture(autouse=True)
def _clean_stylog_env(monkeypatch):
    for var in ("STYLOG_CONFIG", "STYLOG_CACHE_DIR", "STYLOG_NO_CACHE"):
        monkeypatch.delenv(var, raising=False)


def _parse(model_type: type[BaseModel], payload: str | bytes) -> BaseModel:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return model_type.model_validate(json.loads(data), strict=False)


def _parse_jsonl(model_type: type[BaseModel], payload: str) -> list[BaseModel]:
    return [
        model_type.model_validate(json.loads(line), strict=False)
        for line in payload.splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# root help and interface conventions
# ---------------------------------------------------------------------------


def test_bare_root_is_a_useful_successful_landing_screen():
    result = runner.invoke(app, [])
    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert "Stylog 0.1.0 - stylometry for text and source code" in result.stdout
    assert "Usage:\n  stylog <command> [options]" in result.stdout
    assert "fingerprint  Measure artifacts" in result.stdout
    assert "Run 'stylog --help'" in result.stdout
    result.stdout.encode("ascii")


def test_root_and_command_help_are_plain_predictable_and_successful():
    for argv in (["--help"], ["-h"], ["compare", "--help"], ["compare", "-h"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, (argv, result.stderr)
        assert "Usage:" in result.stdout
        assert "\u250c" not in result.stdout and "\u256d" not in result.stdout
        result.stdout.encode("ascii")
    root_help = runner.invoke(app, ["--help"]).stdout
    assert "Examples:" in root_help
    assert "-V, --version" in root_help
    assert "-h, --help" in root_help


def test_short_version_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert result.stdout == "stylog 0.1.0\n"


@pytest.fixture()
def text_file(tmp_path):
    path = tmp_path / "sample_one.txt"
    path.write_text(TEXT_ONE, encoding="utf-8")
    return path


@pytest.fixture()
def second_text_file(tmp_path):
    path = tmp_path / "sample_two.txt"
    path.write_text(TEXT_TWO, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_text_file_json_stdout(text_file):
    result = runner.invoke(app, ["fingerprint", str(text_file), "--no-cache"])
    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    fp = _parse(Fingerprint, result.stdout)
    assert fp.schema == "stylog.fingerprint"
    assert fp.artifact.artifact_id == "sample_one.txt"
    assert fp.artifact.kind.value == "text"
    assert any(obs.status == "ok" for obs in fp.features)


def test_fingerprint_canonical_hash_stable_across_runs(text_file):
    args = ["fingerprint", str(text_file), "--no-cache"]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0 and second.exit_code == 0
    assert first.stdout == second.stdout  # canonical bytes are identical
    hash_one = scientific_sha256(_parse(Fingerprint, first.stdout))
    hash_two = scientific_sha256(_parse(Fingerprint, second.stdout))
    assert hash_one == hash_two


def test_fingerprint_python_file(tmp_path):
    path = tmp_path / "prog.py"
    path.write_text(PY_SOURCE, encoding="utf-8")
    result = runner.invoke(app, ["fingerprint", str(path), "--no-cache"])
    assert result.exit_code == 0, result.stderr
    fp = _parse(Fingerprint, result.stdout)
    assert fp.artifact.kind.value == "code"
    assert fp.artifact.language == "python"
    by_id = {obs.feature_id: obs for obs in fp.features}
    assert by_id["code.python.lexical.token_class"].status == "ok"


def test_fingerprint_javascript_tree_sitter(tmp_path):
    path = tmp_path / "prog.js"
    path.write_text(JS_SOURCE, encoding="utf-8")
    result = runner.invoke(app, ["fingerprint", str(path), "--no-cache"])
    assert result.exit_code == 0, result.stderr
    fp = _parse(Fingerprint, result.stdout)
    assert fp.artifact.language == "javascript"
    parser_features = {
        obs.feature_id: obs for obs in fp.features if obs.feature_id.startswith("code.parser.")
    }
    assert parser_features, "expected tree-sitter features"
    assert any(obs.status == "ok" for obs in parser_features.values())


def test_fingerprint_stdin():
    result = runner.invoke(
        app, ["fingerprint", "-", "--no-cache"], input=b"some text from stdin here."
    )
    assert result.exit_code == 0, result.stderr
    fp = _parse(Fingerprint, result.stdout)
    assert fp.artifact.artifact_id == "stdin"
    assert fp.artifact.kind.value == "text"
    assert fp.artifact.language == "und"


def test_fingerprint_multiple_files_default_jsonl(text_file, second_text_file):
    result = runner.invoke(
        app, ["fingerprint", str(text_file), str(second_text_file), "--no-cache"]
    )
    assert result.exit_code == 0, result.stderr
    lines = _parse_jsonl(Fingerprint, result.stdout)
    assert len(lines) == 2
    assert {fp.artifact.artifact_id for fp in lines} == {"sample_one.txt", "sample_two.txt"}


def test_fingerprint_directory_relative_ids(tmp_path):
    root = tmp_path / "corpus"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_text(TEXT_ONE, encoding="utf-8")
    (root / "nested" / "b.txt").write_text(TEXT_TWO, encoding="utf-8")
    result = runner.invoke(app, ["fingerprint", str(root), "--no-cache"])
    assert result.exit_code == 0, result.stderr
    lines = _parse_jsonl(Fingerprint, result.stdout)
    assert [fp.artifact.artifact_id for fp in lines] == ["a.txt", "nested/b.txt"]


def test_fingerprint_parquet_requires_output(text_file):
    result = runner.invoke(app, ["fingerprint", str(text_file), "--format", "parquet"])
    assert result.exit_code == 2


def test_fingerprint_parquet_export(text_file, tmp_path):
    pytest.importorskip("pyarrow")
    from stylog.data import read_parquet_objects

    out = tmp_path / "fp.parquet"
    result = runner.invoke(
        app,
        ["fingerprint", str(text_file), "--format", "parquet", "--output", str(out)],
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""
    objects = read_parquet_objects(out)
    assert len(objects) == 1
    assert objects[0].artifact.artifact_id == "sample_one.txt"


def test_parquet_rejected_outside_fingerprint(text_file, tmp_path):
    other = tmp_path / "sample_two.txt"
    other.write_text(TEXT_TWO, encoding="utf-8")
    for argv in (
        ["analyze", str(text_file), "--format", "parquet"],
        ["compare", str(text_file), str(other), "--format", "parquet"],
        ["info", "--format", "parquet"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 2, argv
        assert "Invalid value for '--format'" in result.stderr
        assert "'parquet' is not one of 'json', 'jsonl', 'terminal'" in result.stderr

    assert "parquet" in runner.invoke(app, ["fingerprint", "--help"]).stdout
    assert "parquet" not in runner.invoke(app, ["analyze", "--help"]).stdout


def test_fingerprint_with_nlp_model(text_file):
    pytest.importorskip("en_core_web_sm")
    result = runner.invoke(
        app,
        ["fingerprint", str(text_file), "--language", "en",
         "--nlp-model", "en_core_web_sm", "--no-cache"],
    )
    assert result.exit_code == 0, result.stderr
    fp = Fingerprint.model_validate(json.loads(result.stdout))
    linguistic = [f for f in fp.features if f.feature_id.startswith("text.linguistic")]
    assert len(linguistic) == 5
    assert all(f.status == "ok" for f in linguistic)


def test_fingerprint_collection_aggregate_line(text_file, second_text_file):
    result = runner.invoke(
        app,
        [
            "fingerprint",
            str(text_file),
            str(second_text_file),
            "--no-cache",
            "--collection",
            "--linkage",
            "same_author",
            "--linkage-source",
            "declared-by-test",
        ],
    )
    assert result.exit_code == 0, result.stderr
    raw_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(raw_lines) == 3
    for line in raw_lines[:2]:
        _parse(Fingerprint, line)
    aggregate = _parse(EvidenceAggregate, raw_lines[2])
    assert aggregate.schema == "stylog.evidence-aggregate"
    assert len(aggregate.evidence_set.members) == 2
    assert aggregate.evidence_set.linkage.kind == "same_author"
    assert aggregate.evidence_set.linkage.source == "declared-by-test"
    assert aggregate.aggregates  # at least one aggregated observation


def test_fingerprint_collection_requires_linkage(text_file, second_text_file):
    result = runner.invoke(
        app, ["fingerprint", str(text_file), str(second_text_file), "--collection"]
    )
    assert result.exit_code == 2
    assert "--linkage" in result.stderr


def test_fingerprint_missing_input_exits_3(tmp_path):
    missing = tmp_path / "definitely_not_here.txt"
    result = runner.invoke(app, ["fingerprint", str(missing), "--no-cache"])
    assert result.exit_code == 3
    assert result.stdout == ""
    assert "INPUT_NOT_FOUND" in result.stderr


def test_fingerprint_partial_batch_continues(text_file, tmp_path):
    missing = tmp_path / "missing.txt"
    result = runner.invoke(
        app, ["fingerprint", str(missing), str(text_file), "--no-cache"]
    )
    assert result.exit_code == 3
    assert "INPUT_NOT_FOUND" in result.stderr
    lines = _parse_jsonl(Fingerprint, result.stdout)
    assert len(lines) == 1  # the successful artifact is still returned


def test_fingerprint_no_content_hash_suppressed(text_file):
    result = runner.invoke(
        app, ["fingerprint", str(text_file), "--no-cache", "--no-content-hash"]
    )
    assert result.exit_code == 0, result.stderr
    fp = _parse(Fingerprint, result.stdout)
    assert fp.artifact.content_identity.mode == "suppressed"


def test_fingerprint_content_hash_default(text_file):
    result = runner.invoke(app, ["fingerprint", str(text_file), "--no-cache"])
    fp = _parse(Fingerprint, result.stdout)
    assert fp.artifact.content_identity.mode == "sha256"
    raw = (text_file.read_bytes())
    assert fp.artifact.content_identity.sha256 == hashlib.sha256(raw).hexdigest()


def test_fingerprint_output_atomic_and_force(text_file, tmp_path):
    target = tmp_path / "out.json"
    args = ["fingerprint", str(text_file), "--no-cache", "--output", str(target)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # stdout empty on successful --output
    _parse(Fingerprint, target.read_bytes())
    again = runner.invoke(app, args)
    assert again.exit_code == 4  # refuses overwrite without --force
    forced = runner.invoke(app, args + ["--force"])
    assert forced.exit_code == 0, forced.stderr


def test_fingerprint_rejects_second_stdin():
    result = runner.invoke(app, ["fingerprint", "-", "-"], input=b"x")
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_terminal_default(text_file):
    result = runner.invoke(app, ["analyze", str(text_file), "--no-cache"])
    assert result.exit_code == 0, result.stderr
    assert "Feature families" in result.stdout
    assert "text.lexical" in result.stdout
    assert "Embedded text" in result.stdout


def test_terminal_output_escapes_non_ascii_artifact_ids(tmp_path):
    path = tmp_path / "caf\u00e9.txt"
    path.write_text(TEXT_ONE, encoding="utf-8")
    result = runner.invoke(
        app, ["fingerprint", str(path), "--format", "terminal", "--no-cache"]
    )
    assert result.exit_code == 0, result.stderr
    assert "caf\\xe9.txt" in result.stdout
    result.stdout.encode("ascii")


def test_analyze_json_bundle(text_file):
    result = runner.invoke(app, ["analyze", str(text_file), "--no-cache", "--format", "json"])
    assert result.exit_code == 0, result.stderr
    bundle = _parse(AnalysisBundle, result.stdout)
    assert bundle.schema == "stylog.analysis"
    assert bundle.primary.artifact.artifact_id == "sample_one.txt"


def test_analyze_collection_json_bundle_then_aggregate(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text(TEXT_ONE, encoding="utf-8")
    (root / "b.txt").write_text(TEXT_TWO, encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "analyze",
            str(root),
            "--no-cache",
            "--format",
            "json",
            "--collection",
            "--linkage",
            "same_author",
            "--linkage-source",
            "declared-by-test",
        ],
    )
    assert result.exit_code == 0, result.stderr
    raw_lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(raw_lines) == 3
    for line in raw_lines[:2]:
        _parse(AnalysisBundle, line)
    _parse(EvidenceAggregate, raw_lines[2])


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def test_compare_terminal_has_components_no_similarity(text_file, second_text_file):
    result = runner.invoke(
        app, ["compare", str(text_file), str(second_text_file), "--no-cache"]
    )
    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "Comparison" in out
    assert "Left" in out and "sample_one.txt" in out
    assert "Right" in out and "sample_two.txt" in out
    assert "Metric definitions" in out
    assert "SUPPORT L/R" in out
    assert "overall similarity score" in out  # explicitly disclaims one


def test_compare_json_validates(text_file, second_text_file):
    result = runner.invoke(
        app, ["compare", str(text_file), str(second_text_file), "--no-cache", "--format", "json"]
    )
    assert result.exit_code == 0, result.stderr
    comparison = _parse(Comparison, result.stdout)
    assert comparison.schema == "stylog.comparison"
    assert comparison.left_ref == "sample_one.txt"
    assert comparison.right_ref == "sample_two.txt"
    assert comparison.families


def test_compare_from_artifacts(text_file, second_text_file, tmp_path):
    left_json = tmp_path / "left.json"
    right_json = tmp_path / "right.json"
    for source, target in ((text_file, left_json), (second_text_file, right_json)):
        written = runner.invoke(
            app, ["fingerprint", str(source), "--no-cache", "--output", str(target)]
        )
        assert written.exit_code == 0, written.stderr
    result = runner.invoke(
        app, ["compare", str(left_json), str(right_json), "--from-artifacts"]
    )
    assert result.exit_code == 0, result.stderr
    assert "Left" in result.stdout and "left.json" in result.stdout
    assert "Right" in result.stdout and "right.json" in result.stdout
    assert "DISTANCE" in result.stdout


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------


@pytest.fixture()
def baseline_path(tmp_path):
    """Build a local baseline over 25 synthetic texts via the application layer."""
    from stylog.application.fingerprint import fingerprint_artifact
    from stylog.application.profile import build_baseline
    from stylog.bootstrap import build_context, build_default_services
    from stylog.config import load_config
    from stylog.domain.baseline import BaselineDescriptor
    from stylog.infrastructure.ingest import artifact_from_text
    from stylog.serialization.canonical import file_bytes

    config = load_config()
    services = build_default_services(config, no_cache=True)
    ctx = build_context(config, services)
    fingerprints = []
    for index in range(25):
        text = (
            f"Synthetic baseline unit number {index}. "
            + " ".join(f"token{slot}" for slot in range(8 + index % 9))
            + " A closing sentence follows here."
        )
        artifact = artifact_from_text(text, artifact_id=f"unit{index:02d}", language="und")
        fingerprints.append(
            fingerprint_artifact(artifact, config=config, services=services, ctx=ctx).fingerprint
        )
    baseline = build_baseline(
        fingerprints,
        baseline_id="synthetic-test-baseline",
        baseline_version="1.0.0",
        descriptor=BaselineDescriptor(
            kind="text",
            language="und",
            domain="synthetic",
            unit="artifact",
            source="pytest",
        ),
    )
    path = tmp_path / "baseline.json"
    path.write_bytes(file_bytes(baseline))
    return path


def test_profile_terminal(text_file, baseline_path):
    result = runner.invoke(
        app,
        ["profile", str(text_file), "--baseline", str(baseline_path), "--no-cache"],
    )
    assert result.exit_code == 0, result.stderr
    assert "Profile" in result.stdout
    assert "Subject" in result.stdout and "sample_one.txt" in result.stdout
    assert "MIDRANK %" in result.stdout
    assert "ROBUST Z" in result.stdout


def test_build_baseline_public_api(tmp_path):
    """stylog.build_baseline exposes baseline construction (spec 13.8, 13.10)."""
    import stylog
    from stylog.serialization.canonical import file_bytes

    fingerprints = [
        stylog.fingerprint_text(
            f"Public API baseline unit {index}. "
            + " ".join(f"token{slot}" for slot in range(8 + index % 9))
            + " A closing sentence follows here.",
        )
        for index in range(1)
    ]
    baseline = stylog.build_baseline(
        fingerprints, baseline_id="api-test-baseline", language="und", domain="synthetic"
    )
    assert baseline.schema == "stylog.baseline"
    assert baseline.baseline_id == "api-test-baseline"
    assert baseline.baseline_version == "1.0.0"
    assert baseline.descriptor.unit == "artifact"
    assert baseline.features
    assert {feature.total_units for feature in baseline.features} == {1}

    path = tmp_path / "baseline.json"
    path.write_bytes(file_bytes(baseline))
    subject = tmp_path / "subject.txt"
    subject.write_text(TEXT_ONE, encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "profile",
            str(subject),
            "--baseline",
            str(path),
            "--no-cache",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    profile = _parse(Profile, result.stdout)
    assert profile.observations
    assert {observation.baseline_n for observation in profile.observations} == {1}


def test_profile_json_validates(text_file, baseline_path):
    result = runner.invoke(
        app,
        [
            "profile",
            str(text_file),
            "--baseline",
            str(baseline_path),
            "--no-cache",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    profile = _parse(Profile, result.stdout)
    assert profile.schema == "stylog.profile"
    assert profile.baseline_id == "synthetic-test-baseline"
    assert profile.observations


def test_profile_from_artifact(text_file, baseline_path, tmp_path):
    fp_path = tmp_path / "subject.json"
    written = runner.invoke(
        app, ["fingerprint", str(text_file), "--no-cache", "--output", str(fp_path)]
    )
    assert written.exit_code == 0, written.stderr
    result = runner.invoke(
        app,
        [
            "profile",
            str(fp_path),
            "--from-artifact",
            "--baseline",
            str(baseline_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    profile = _parse(Profile, result.stdout)
    assert profile.subject_ref == "sample_one.txt"


def test_profile_missing_baseline_exits_4(text_file):
    result = runner.invoke(
        app, ["profile", str(text_file), "--baseline", "no/such/baseline.json", "--no-cache"]
    )
    assert result.exit_code == 4
    assert "BASELINE_NOT_FOUND" in result.stderr


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_fingerprint(text_file, tmp_path):
    fp_path = tmp_path / "subject.json"
    written = runner.invoke(
        app, ["fingerprint", str(text_file), "--no-cache", "--output", str(fp_path)]
    )
    assert written.exit_code == 0, written.stderr
    result = runner.invoke(app, ["report", str(fp_path)])
    assert result.exit_code == 0, result.stderr
    assert "Schema  stylog.fingerprint" in result.stdout
    assert "Artifact" in result.stdout and "sample_one.txt" in result.stdout


def test_report_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "stylog.unknown"}\n', encoding="utf-8")
    result = runner.invoke(app, ["report", str(bad)])
    assert result.exit_code == 4


# ---------------------------------------------------------------------------
# info (capabilities is a hidden alias)
# ---------------------------------------------------------------------------


def test_info_terminal_lists_five_languages():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.stderr
    out = result.stdout
    assert "Code languages  python, javascript, typescript, c, rust" in out
    for language in ("python", "javascript", "typescript", "c", "rust"):
        assert language in out
    out.encode("ascii")  # terminal renderers must stay ASCII (cp1252 consoles)


def test_info_json():
    result = runner.invoke(app, ["info", "--format", "json"])
    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report["code_languages"]) == 5
    assert set(report["tree_sitter_grammars"]) == {"javascript", "typescript", "c", "rust"}
    compat = report["scientific_compatibility_ids"]
    assert compat["stylog.native.text"] == "stylog.text-core/1"
    assert compat["stylog.native.code"] == "stylog.code-surface/1"
    assert compat["cpython.tokenize"] == "stylog.python-native-tokenize/1"
    assert compat["cpython.ast"] == "stylog.python-native-ast/1"


def test_info_jsonl_is_one_compact_json_record():
    result = runner.invoke(app, ["info", "--format", "jsonl"])
    assert result.exit_code == 0, result.stderr
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout)["stylog_version"] == "0.1.0"


def test_capabilities_is_hidden_alias_for_info():
    """capabilities stays a hidden alias with identical output (spec 19)."""
    primary = runner.invoke(app, ["info", "--format", "json"])
    alias = runner.invoke(app, ["capabilities", "--format", "json"])
    assert primary.exit_code == 0 and alias.exit_code == 0
    assert primary.stdout == alias.stdout
    help_out = runner.invoke(app, ["--help"]).stdout
    assert "info" in help_out
    assert "\n  capabilities " not in help_out


# ---------------------------------------------------------------------------
# represent (stylog[ml]); another agent provides stylog.representations
# ---------------------------------------------------------------------------

_HAS_REPRESENTATIONS = (
    importlib.util.find_spec("stylog.representations") is not None
    and importlib.util.find_spec("sklearn") is not None
)


@pytest.mark.skipif(_HAS_REPRESENTATIONS, reason="representations module is present")
def test_represent_reports_missing_ml_capability(text_file, tmp_path):
    result = runner.invoke(
        app,
        [
            "represent",
            str(text_file),
            "--representation",
            "word-ngram-count",
            "--fit-output",
            str(tmp_path / "fit.json"),
        ],
    )
    assert result.exit_code == 2
    assert "pip install stylog[ml]" in result.stderr
    assert result.stdout == ""


@pytest.mark.skipif(not _HAS_REPRESENTATIONS, reason="representations module not present")
def test_represent_fit_then_transform(text_file, second_text_file, tmp_path):
    from stylog.representations import fit as repfit

    if not (
        hasattr(repfit, "fit_representation_cli")
        and hasattr(repfit, "transform_representation_cli")
    ):
        pytest.skip("representations CLI helpers not available")
    fit_path = tmp_path / "fit.json"
    fit_result = runner.invoke(
        app,
        [
            "represent",
            str(text_file),
            str(second_text_file),
            "--representation",
            "word-ngram-count",
            "--fit-output",
            str(fit_path),
        ],
    )
    assert fit_result.exit_code == 0, fit_result.stderr
    assert fit_path.is_file()
    transform_result = runner.invoke(
        app, ["represent", str(text_file), "--fit-resource", str(fit_path)]
    )
    assert transform_result.exit_code == 0, transform_result.stderr
    representation = json.loads(transform_result.stdout)
    assert representation["schema"] == "stylog.representation"


@pytest.mark.skipif(not _HAS_REPRESENTATIONS, reason="representations module not present")
def test_represent_flag_aliases(text_file, second_text_file, tmp_path):
    """--model/-m alias --fit-resource; -o aliases --output (spec 19)."""
    from stylog.representations import fit as repfit

    if not (
        hasattr(repfit, "fit_representation_cli")
        and hasattr(repfit, "transform_representation_cli")
    ):
        pytest.skip("representations CLI helpers not available")
    fit_path = tmp_path / "fit.json"
    fit_result = runner.invoke(
        app,
        [
            "represent",
            str(text_file),
            str(second_text_file),
            "--representation",
            "word-ngram-count",
            "--fit-output",
            str(fit_path),
        ],
    )
    assert fit_result.exit_code == 0, fit_result.stderr
    out_path = tmp_path / "rep.json"
    for flag in ("--fit-resource", "--model", "-m"):
        result = runner.invoke(
            app, ["represent", str(text_file), flag, str(fit_path), "-o", str(out_path)]
        )
        assert result.exit_code == 0, f"{flag}: {result.stderr}"
        assert json.loads(out_path.read_text(encoding="utf-8"))["schema"] == (
            "stylog.representation"
        )
        out_path.unlink()


# ---------------------------------------------------------------------------
# benchmark; another agent provides stylog.benchmark
# ---------------------------------------------------------------------------

_HAS_BENCHMARK = importlib.util.find_spec("stylog.benchmark") is not None


def _write_benchmark_fixture(root):
    """Tiny local dataset + split_audit spec per spec section 21."""
    data_dir = root / "data"
    data_dir.mkdir()
    artifact_entries = []
    for index in range(4):
        name = f"a{index}"
        body = f"benchmark sample text {index} with a handful of words.".encode()
        (data_dir / f"{name}.txt").write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        artifact_entries.append(
            f'[[artifact]]\nid = "{name}"\npath = "data/{name}.txt"\n'
            f'sha256 = "{digest}"\nkind = "text"\nlanguage = "und"\n'
            f'repository_id = "repo{index}"\n'
        )
    manifest = (
        'schema = "stylog.dataset"\nschema_version = "0.1.0"\nid = "tiny"\n'
        'version = "1"\nlicense = "MIT"\nredistribution = "allowed"\n'
        'source = "local test fixture"\n\n' + "\n".join(artifact_entries)
    )
    (root / "dataset.toml").write_text(manifest, encoding="utf-8")
    spec = (
        'schema = "stylog.benchmark"\nschema_version = "0.1.0"\nid = "tiny-bench"\n'
        'task = "split_audit"\ndataset = "dataset.toml"\n\n'
        "[split]\n"
        'seed = "cli-test"\ntrain_ppm = 500000\ndev_ppm = 250000\n'
        "test_ppm = 250000\n"
        'disjoint_by = ["repository_id"]\nrequire_nonempty = false\n'
    )
    (root / "spec.toml").write_text(spec, encoding="utf-8")
    return root / "spec.toml"


@pytest.mark.skipif(not _HAS_BENCHMARK, reason="benchmark module not present")
def test_benchmark_split_audit_json(tmp_path):
    spec = _write_benchmark_fixture(tmp_path)
    result = runner.invoke(app, ["benchmark", str(spec), "--format", "json"])
    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "stylog.benchmark-result"
    assert payload["task"] == "split_audit"


@pytest.mark.skipif(not _HAS_BENCHMARK, reason="benchmark module not present")
def test_benchmark_terminal(tmp_path):
    spec = _write_benchmark_fixture(tmp_path)
    result = runner.invoke(app, ["benchmark", str(spec)])
    assert result.exit_code == 0, result.stderr
    assert any(
        line.split() == ["Task", "split_audit"] for line in result.stdout.splitlines()
    )


def test_benchmark_missing_spec_exits_6(tmp_path):
    if not _HAS_BENCHMARK:
        pytest.skip("benchmark module not present")
    result = runner.invoke(app, ["benchmark", str(tmp_path / "missing.toml")])
    assert result.exit_code == 6
