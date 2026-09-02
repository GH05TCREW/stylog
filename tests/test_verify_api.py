"""Public API verification tests (spec 20, 23)."""

from __future__ import annotations

import pytest
from test_verify import TTR, make_fingerprint, make_model, ok_obs, ratio
from test_verify_fit import make_spec, separable_pairs

from stylog import api
from stylog.application.verify import verify_subjects
from stylog.domain.fingerprint import AnalysisBundle
from stylog.exceptions import StylogError
from stylog.serialization.canonical import canonical_bytes, scientific_sha256
from stylog.serialization.jsonio import write_json_atomic
from stylog.verification.fit import fit_verifier_model


def test_verify_fingerprints_api_parity() -> None:
    model = make_model((TTR,))
    left = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    right = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.8)),))
    via_api = api.verify_fingerprints(left, right, model)
    via_application = verify_subjects(left, right, model)
    assert canonical_bytes(via_api) == canonical_bytes(via_application)


def test_verify_files_end_to_end(tmp_path) -> None:
    left_path = tmp_path / "left.txt"
    right_path = tmp_path / "right.txt"
    left_path.write_text("the quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8")
    right_path.write_text("the quick brown fox jumps over the lazy dog. " * 20, encoding="utf-8")
    model = make_model((TTR,), languages=("und",))
    verification = api.verify_files(left_path, right_path, model)
    assert verification.score is not None
    assert verification.left_ref == str(left_path)
    assert verification.right_ref == str(right_path)
    assert verification.features_used == 1


def test_bundle_subjects_bind_primary_hashes() -> None:
    model = make_model((TTR,))
    left_fp = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    right_fp = make_fingerprint("b", features=(ok_obs(TTR, ratio(0.9)),))
    left = AnalysisBundle(primary=left_fp)
    right = AnalysisBundle(primary=right_fp)
    verification = verify_subjects(left, right, model)
    assert verification.left_fingerprint_sha256 == scientific_sha256(left_fp)
    assert verification.right_fingerprint_sha256 == scientific_sha256(right_fp)


def test_mixed_subjects_error() -> None:
    model = make_model((TTR,))
    fp = make_fingerprint("a", features=(ok_obs(TTR, ratio(0.8)),))
    bundle = AnalysisBundle(primary=fp)
    with pytest.raises(StylogError, match="verification subjects"):
        verify_subjects(fp, bundle, model)


def test_fit_verifier_api_and_load_verifier(tmp_path) -> None:
    spec = make_spec()
    pairs = separable_pairs()
    model = api.fit_verifier(spec, pairs)
    direct, _ = fit_verifier_model(spec, pairs)
    assert scientific_sha256(model) == scientific_sha256(direct)
    path = tmp_path / "model.json"
    write_json_atomic(path, model)
    loaded = api.load_verifier(path)
    assert scientific_sha256(loaded) == scientific_sha256(model)


def test_fit_verifier_records_tuning_manifest() -> None:
    model = api.fit_verifier(
        make_spec(), separable_pairs(), tuning_manifest_sha256="7" * 64
    )
    assert model.tuning_manifest_sha256 == "7" * 64


def test_api_exports() -> None:
    import stylog

    for name in ("verify_fingerprints", "verify_files", "fit_verifier", "load_verifier"):
        assert callable(getattr(stylog, name))
