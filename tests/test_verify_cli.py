"""CLI contract tests for verify / fit (spec 19, 23)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_verify import TTR, make_fingerprint, make_model, ok_obs, ratio
from typer.testing import CliRunner

from stylog.cli import app
from stylog.serialization.canonical import file_bytes
from stylog.serialization.jsonio import write_json_atomic

runner = CliRunner()

TEXT_A = "the quick brown fox jumps over the lazy dog. " * 12 + "\n"
TEXT_B = "How vexingly quick daft zebras jump! Bright vixens jump; dozy fowl quack. " * 12 + "\n"


@pytest.fixture(autouse=True)
def _clean_stylog_env(monkeypatch):
    for var in ("STYLOG_CONFIG", "STYLOG_CACHE_DIR", "STYLOG_NO_CACHE"):
        monkeypatch.delenv(var, raising=False)


def _write_bytes(path, data: bytes):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture_dataset(tmp_path):
    """Eight a-style + eight b-style text files plus a dataset manifest."""
    texts = {}
    for i in range(8):
        texts[f"a{i:02d}.txt"] = ("the quick brown fox jumps over the lazy dog. " * (10 + i)).strip() + "\n"
        texts[f"b{i:02d}.txt"] = ("How vexingly quick daft zebras jump! Bright vixens jump; dozy fowl quack. " * (10 + i)).strip() + "\n"
    lines = [
        'schema = "stylog.dataset"',
        'schema_version = "0.1.0"',
        'id = "tiny"',
        'version = "1"',
        'license = "CC0"',
        'redistribution = "allowed"',
        'source = "synthetic"',
    ]
    for name, text in sorted(texts.items()):
        sha = _write_bytes(tmp_path / name, text.encode("utf-8"))
        lines += [
            "[[artifact]]",
            f'id = "{name}"',
            f'path = "{name}"',
            f'sha256 = "{sha}"',
            'kind = "text"',
            'language = "en"',
        ]
    (tmp_path / "dataset.toml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return texts


def _training_toml(pairs_block: str, verifier_extra: str = "") -> str:
    return f'''schema = "stylog.verifier-training"
schema_version = "0.1.0"
id = "tiny-training"
dataset = "dataset.toml"

[verifier]
kind = "text"
l2_lambda = 1.0
min_support_fraction = 0.9
min_class_support_fraction = 0.8
min_pairs = 4
threshold_rule = "fixed"
threshold_fixed = 0.5
include_linguistic = false
allow_unconstrained_language = false
{verifier_extra}

[verifier.pair_policy]
selection_version = "1"

{pairs_block}
'''


def _default_pairs() -> str:
    rows = []
    for i in range(4):
        rows.append(
            f'[[pair]]\nleft = "a{i:02d}.txt"\nright = "a{i + 1:02d}.txt"\n'
            'label = "same"\npopulation = "train"'
        )
    for i in range(4):
        rows.append(
            f'[[pair]]\nleft = "a{i:02d}.txt"\nright = "b{i:02d}.txt"\n'
            'label = "different"\npopulation = "train"'
        )
    return "\n".join(rows)


def _fit_tiny_model(tmp_path) -> Path:
    _fixture_dataset(tmp_path)
    (tmp_path / "training.toml").write_text(
        _training_toml(_default_pairs()), encoding="utf-8", newline="\n"
    )
    model_path = tmp_path / "model.json"
    result = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "--output", str(model_path)]
    )
    assert result.exit_code == 0, result.stderr
    return model_path


# --- verify ---------------------------------------------------------------


def test_verify_terminal_and_json(tmp_path):
    model_path = _fit_tiny_model(tmp_path)
    left = tmp_path / "a00.txt"
    right = tmp_path / "a01.txt"
    result = runner.invoke(
        app,
        ["verify", str(left), str(right), "--model", str(model_path), "--language", "en"],
    )
    assert result.exit_code == 0, result.stderr
    result.stdout.encode("ascii")  # ASCII-only renderer
    assert "Verdict" in result.stdout
    assert "Model score" in result.stdout
    assert "not a probability" in result.stdout
    assert "Features" in result.stdout and "used" in result.stdout
    assert "Verifier ID" in result.stdout
    assert "Left fingerprint" in result.stdout

    result_json = runner.invoke(
        app,
        [
            "verify",
            str(left),
            str(right),
            "--model",
            str(model_path),
            "--language",
            "en",
            "--format",
            "json",
        ],
    )
    assert result_json.exit_code == 0, result_json.stderr
    obj = json.loads(result_json.stdout)
    assert obj["schema"] == "stylog.verification"
    assert obj["verdict"] in ("same_author", "different_author", "abstain")
    assert obj["left_fingerprint_sha256"]
    assert obj["right_fingerprint_sha256"]
    assert obj["verifier_id"]


def test_verify_cli_json_matches_api(tmp_path):
    model_path = _fit_tiny_model(tmp_path)
    left = tmp_path / "a00.txt"
    right = tmp_path / "a01.txt"
    result = runner.invoke(
        app,
        [
            "verify",
            str(left),
            str(right),
            "--model",
            str(model_path),
            "--language",
            "en",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.stderr
    from stylog import api

    model = api.load_verifier(model_path)
    via_api = api.verify_fingerprints(
        api.fingerprint_file(left, language="en"),
        api.fingerprint_file(right, language="en"),
        model,
        left_ref=left.name,
        right_ref=right.name,
    )
    assert result.stdout.encode() == file_bytes(via_api)


def test_verify_from_artifacts_abstain_rendering(tmp_path):
    # hand-built model with a wide abstention band; score lands inside it
    model = make_model((TTR,), t_same=0.7, t_diff=0.3, languages=("en",))
    model_path = tmp_path / "model.json"
    write_json_atomic(model_path, model)
    left_fp = make_fingerprint("left", features=(ok_obs(TTR, ratio(0.9)),))
    right_fp = make_fingerprint("right", features=(ok_obs(TTR, ratio(0.275)),))
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    write_json_atomic(left_path, left_fp)
    write_json_atomic(right_path, right_fp)
    result = runner.invoke(
        app,
        [
            "verify",
            str(left_path),
            str(right_path),
            "--from-artifacts",
            "--model",
            str(model_path),
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert any(
        line.split() == ["Verdict", "abstain"] for line in result.stdout.splitlines()
    )
    assert any(
        line.split() == ["Abstain", "reason", "uncertain"]
        for line in result.stdout.splitlines()
    )
    result.stdout.encode("ascii")

    result_json = runner.invoke(
        app,
        [
            "verify",
            str(left_path),
            str(right_path),
            "--from-artifacts",
            "--model",
            str(model_path),
            "--format",
            "json",
        ],
    )
    obj = json.loads(result_json.stdout)
    assert obj["verdict"] == "abstain"
    assert obj["abstain_reason"] == "uncertain"
    assert 0.3 < obj["score"] < 0.7


def test_verify_exit_codes(tmp_path):
    model_path = _fit_tiny_model(tmp_path)
    left = tmp_path / "a00.txt"
    right = tmp_path / "a01.txt"
    # missing --model -> usage (2)
    result = runner.invoke(app, ["verify", str(left), str(right)])
    assert result.exit_code == 2
    # missing model file -> input error (3)
    result = runner.invoke(
        app, ["verify", str(left), str(right), "--model", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 3
    # missing input file -> exit 3
    result = runner.invoke(
        app, ["verify", str(tmp_path / "nope.txt"), str(right), "--model", str(model_path)]
    )
    assert result.exit_code == 3
    # malformed model -> exit 4
    bad_model = tmp_path / "bad.json"
    bad_model.write_text('{"schema": "stylog.verifier-fit", "bogus": true}\n', encoding="utf-8")
    result = runner.invoke(
        app, ["verify", str(left), str(right), "--model", str(bad_model), "--language", "en"]
    )
    assert result.exit_code == 4
    # wrong-schema model -> exit 4
    fingerprint_path = tmp_path / "fp.json"
    write_json_atomic(fingerprint_path, make_fingerprint("x", features=(ok_obs(TTR, ratio(0.5)),)))
    result = runner.invoke(
        app, ["verify", str(left), str(right), "--model", str(fingerprint_path), "--language", "en"]
    )
    assert result.exit_code == 4


def test_verify_incompatible_model_exit_4(tmp_path):
    from stylog.domain.artifact import ArtifactKind

    model = make_model((TTR,), kind=ArtifactKind.CODE)
    model_path = tmp_path / "model.json"
    write_json_atomic(model_path, model)
    left_fp = make_fingerprint("left", features=(ok_obs(TTR, ratio(0.9)),))
    right_fp = make_fingerprint("right", features=(ok_obs(TTR, ratio(0.8)),))
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    write_json_atomic(left_path, left_fp)
    write_json_atomic(right_path, right_fp)
    result = runner.invoke(
        app,
        [
            "verify",
            str(left_path),
            str(right_path),
            "--from-artifacts",
            "--model",
            str(model_path),
        ],
    )
    assert result.exit_code == 4
    assert "kind" in result.stderr


def test_verify_no_overwrite_exit_4(tmp_path):
    model_path = _fit_tiny_model(tmp_path)
    left = tmp_path / "a00.txt"
    right = tmp_path / "a01.txt"
    output = tmp_path / "out.json"
    output.write_bytes(b"existing")
    result = runner.invoke(
        app,
        [
            "verify",
            str(left),
            str(right),
            "--model",
            str(model_path),
            "--language",
            "en",
            "--format",
            "json",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 4
    assert output.read_bytes() == b"existing"


# --- fit -----------------------------------------------------------


def test_verify_fit_deterministic_and_stderr_discipline(tmp_path):
    _fixture_dataset(tmp_path)
    (tmp_path / "training.toml").write_text(
        _training_toml(_default_pairs()), encoding="utf-8", newline="\n"
    )
    first = tmp_path / "m1.json"
    second = tmp_path / "m2.json"
    result = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "--output", str(first)]
    )
    assert result.exit_code == 0, result.stderr
    assert result.stdout == ""  # stdout empty on successful --output
    assert "Verifier ID  " in result.stderr
    assert "VERIFIER_ELIGIBILITY" in result.stderr
    result2 = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "--output", str(second)]
    )
    assert result2.exit_code == 0, result2.stderr
    assert first.read_bytes() == second.read_bytes()
    # refuse overwrite without --force
    result3 = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "--output", str(first)]
    )
    assert result3.exit_code == 4
    result4 = runner.invoke(
        app,
        ["fit", str(tmp_path / "training.toml"), "--output", str(first), "--force"],
    )
    assert result4.exit_code == 0


def test_verify_fit_alias_and_short_flags(tmp_path):
    """verify-fit is a hidden alias for fit; --output accepts -o (spec 19)."""
    _fixture_dataset(tmp_path)
    (tmp_path / "training.toml").write_text(
        _training_toml(_default_pairs()), encoding="utf-8", newline="\n"
    )
    primary = tmp_path / "primary.json"
    alias = tmp_path / "alias.json"
    result = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "-o", str(primary)]
    )
    assert result.exit_code == 0, result.stderr
    result_alias = runner.invoke(
        app, ["verify-fit", str(tmp_path / "training.toml"), "--output", str(alias)]
    )
    assert result_alias.exit_code == 0, result_alias.stderr
    assert primary.read_bytes() == alias.read_bytes()


def test_verify_fit_tuning_population_recorded(tmp_path):
    _fixture_dataset(tmp_path)
    pairs = _default_pairs() + (
        '\n[[pair]]\nleft = "a05.txt"\nright = "a06.txt"\n'
        'label = "same"\npopulation = "tuning"'
    )
    (tmp_path / "training.toml").write_text(
        _training_toml(pairs), encoding="utf-8", newline="\n"
    )
    model_path = tmp_path / "model.json"
    result = runner.invoke(
        app, ["fit", str(tmp_path / "training.toml"), "--output", str(model_path)]
    )
    assert result.exit_code == 0, result.stderr
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["tuning_manifest_sha256"]
    assert "calibration_manifest_sha256" not in model


def test_verify_fit_error_paths(tmp_path):
    _fixture_dataset(tmp_path)
    # unknown artifact endpoint
    bad_pairs = _default_pairs() + (
        '\n[[pair]]\nleft = "a00.txt"\nright = "ghost.txt"\nlabel = "same"'
    )
    (tmp_path / "bad1.toml").write_text(
        _training_toml(bad_pairs), encoding="utf-8", newline="\n"
    )
    result = runner.invoke(
        app, ["fit", str(tmp_path / "bad1.toml"), "--output", str(tmp_path / "m.json")]
    )
    assert result.exit_code == 6
    # evaluation population is rejected in training manifests
    bad_pairs2 = _default_pairs() + (
        '\n[[pair]]\nleft = "a00.txt"\nright = "a01.txt"\n'
        'label = "same"\npopulation = "evaluation"'
    )
    (tmp_path / "bad2.toml").write_text(
        _training_toml(bad_pairs2), encoding="utf-8", newline="\n"
    )
    result = runner.invoke(
        app, ["fit", str(tmp_path / "bad2.toml"), "--output", str(tmp_path / "m.json")]
    )
    assert result.exit_code == 6
    # invalid verifier block (negative lambda)
    (tmp_path / "bad3.toml").write_text(
        _training_toml(_default_pairs()).replace("l2_lambda = 1.0", "l2_lambda = -1.0"),
        encoding="utf-8",
        newline="\n",
    )
    result = runner.invoke(
        app, ["fit", str(tmp_path / "bad3.toml"), "--output", str(tmp_path / "m.json")]
    )
    assert result.exit_code == 6
    # wrong schema
    (tmp_path / "bad4.toml").write_text(
        _training_toml(_default_pairs()).replace(
            'schema = "stylog.verifier-training"', 'schema = "stylog.dataset"'
        ),
        encoding="utf-8",
        newline="\n",
    )
    result = runner.invoke(
        app, ["fit", str(tmp_path / "bad4.toml"), "--output", str(tmp_path / "m.json")]
    )
    assert result.exit_code == 6


def test_report_renders_verification(tmp_path):
    model = make_model((TTR,), t_same=0.7, t_diff=0.3, languages=("en",), coefficients=(-2.0,))
    from stylog import api

    left_fp = make_fingerprint("left", features=(ok_obs(TTR, ratio(0.9)),))
    right_fp = make_fingerprint("right", features=(ok_obs(TTR, ratio(0.9)),))
    verification = api.verify_fingerprints(left_fp, right_fp, model)
    path = tmp_path / "verification.json"
    write_json_atomic(path, verification)
    result = runner.invoke(app, ["report", str(path)])
    assert result.exit_code == 0, result.stderr
    assert "Schema  stylog.verification" in result.stdout
    assert "Verdict" in result.stdout and "same_author" in result.stdout
    result.stdout.encode("ascii")


def test_info_includes_verification():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.stderr
    assert "Verification" in result.stdout
    assert "stylog.verifier.logreg/1" in result.stdout
    report = json.loads(
        runner.invoke(app, ["info", "--format", "json"]).stdout
    )
    assert report["verification"]["model_id"] == "stylog.verifier.logreg/1"
    assert report["verification"]["scientific_compatibility_id"] == "stylog.verifier.logreg/1"
