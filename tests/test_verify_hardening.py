"""Verifier hardening: offline fit+verify, execution/cache invariance (spec 23, 26)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
from test_verify import TTR, make_model
from test_verify_fit import make_spec, separable_pairs

from stylog.application.batch import fingerprint_iter
from stylog.application.fingerprint import fingerprint_artifact
from stylog.bootstrap import build_context, build_default_services
from stylog.config import StylogConfig
from stylog.domain.verification import VerifierFit
from stylog.infrastructure.ingest import artifact_from_bytes
from stylog.serialization.canonical import canonical_bytes, scientific_sha256
from stylog.serialization.jsonio import write_json_atomic
from stylog.verification.fit import fit_verifier_model


@pytest.fixture(autouse=True)
def _hermetic_cache_dir(tmp_path, monkeypatch):
    """Redirect the default cache root into the test tmp dir."""
    monkeypatch.setenv("STYLOG_CACHE_DIR", str(tmp_path / "cache"))


def test_offline_fit_and_verify_with_sockets_blocked(tmp_path) -> None:
    script = textwrap.dedent(
        """
        import socket

        class _NoSocket:
            def __init__(self, *a, **k):
                raise OSError("network is blocked in this test")

        socket.socket = _NoSocket
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))

        import tempfile
        import stylog
        from stylog.verification.spec import TrainingPair, VerifierSpec

        same_texts = [
            ("the quick brown fox jumps over the lazy dog. " * 12,
             "the quick brown fox jumps over the lazy dog. " * 12 + "again. "),
            ("pack my box with five dozen liquor jugs tonight. " * 12,
             "pack my box with five dozen liquor jugs tonight. " * 12 + "yes. "),
        ]
        diff_texts = [
            ("the quick brown fox jumps over the lazy dog. " * 12,
             "How vexingly quick daft zebras jump! Bright vixens leap. " * 12),
            ("pack my box with five dozen liquor jugs tonight. " * 12,
             "Sphinx of black quartz, judge my vow! Hear me now. " * 12),
        ]
        pairs = []
        for left_text, right_text in same_texts:
            pairs.append(TrainingPair(
                left=stylog.fingerprint_text(left_text, language="en"),
                right=stylog.fingerprint_text(right_text, language="en"),
                label="same",
            ))
        for left_text, right_text in diff_texts:
            pairs.append(TrainingPair(
                left=stylog.fingerprint_text(left_text, language="en"),
                right=stylog.fingerprint_text(right_text, language="en"),
                label="different",
            ))
        spec = VerifierSpec(
            kind="text", l2_lambda=1.0, min_support_fraction=0.9,
            min_class_support_fraction=0.8, min_pairs=2,
            threshold_rule="fixed", threshold_fixed=0.5,
            feature_ids=("text.lexical.ttr_casefold",),
        )
        model = stylog.fit_verifier(spec, pairs)
        verification = stylog.verify_fingerprints(pairs[0].left, pairs[0].right, model)
        assert verification.verdict in ("same_author", "different_author", "abstain")
        assert verification.left_fingerprint_sha256
        with tempfile.TemporaryDirectory() as d:
            from stylog.serialization.jsonio import write_json_atomic
            path = d + "/model.json"
            write_json_atomic(path, model)
            loaded = stylog.load_verifier(path)
            from stylog.serialization.canonical import scientific_sha256
            assert scientific_sha256(loaded) == scientific_sha256(model)
        print("OFFLINE_VERIFY_OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    assert result.returncode == 0, result.stderr
    assert "OFFLINE_VERIFY_OK" in result.stdout


def _pair_artifacts(config: StylogConfig):
    texts = [
        "the quick brown fox jumps over the lazy dog. " * 8,
        "pack my box with five dozen liquor jugs tonight. " * 8,
        "How vexingly quick daft zebras jump! Bright vixens leap. " * 8,
        "Sphinx of black quartz, judge my vow! Hear me now. " * 8,
    ]
    return [
        artifact_from_bytes(
            text.encode(),
            artifact_id=f"doc{index}",
            kind="text",
            language="en",
            encoding="utf-8",
            config=config,
        )
        for index, text in enumerate(texts)
    ]


def test_worker_count_invariance_for_pair_verification() -> None:
    config = StylogConfig()
    model = make_model((TTR,), languages=("en",))

    def verify_all(bundles):
        from stylog.application.verify import verify_subjects

        fps = [bundle.primary for bundle in bundles]
        return [
            verify_subjects(fps[0], fps[1], model, left_ref="doc0", right_ref="doc1"),
            verify_subjects(fps[0], fps[2], model, left_ref="doc0", right_ref="doc2"),
            verify_subjects(fps[2], fps[3], model, left_ref="doc2", right_ref="doc3"),
        ]

    artifacts = _pair_artifacts(config)
    serial = verify_all(fingerprint_iter(list(artifacts), config=config, execution="serial"))
    artifacts = _pair_artifacts(config)
    parallel = verify_all(
        fingerprint_iter(list(artifacts), config=config, execution="process", workers=2)
    )
    for left, right in zip(serial, parallel, strict=True):
        assert canonical_bytes(left) == canonical_bytes(right)


def test_cache_state_invariance_for_verification() -> None:
    # warm / cold / disabled cache -> identical Verification bytes incl. hashes
    texts = ("the quick brown fox jumps over the lazy dog. " * 8, "pack my box. " * 8)
    model = make_model((TTR,), languages=("en",))

    def run(services, config):
        ctx = build_context(config, services)
        fps = []
        for index, text in enumerate(texts):
            artifact = artifact_from_bytes(
                text.encode(),
                artifact_id=f"doc{index}",
                kind="text",
                language="en",
                encoding="utf-8",
                config=config,
            )
            fps.append(
                fingerprint_artifact(
                    artifact, config=config, services=services, ctx=ctx
                ).fingerprint
            )
        from stylog.application.verify import verify_subjects

        return verify_subjects(fps[0], fps[1], model, left_ref="doc0", right_ref="doc1")

    from stylog.infrastructure.cache import MemoryCacheStore, NullCacheStore
    from stylog.ports import RuntimeServices

    config = StylogConfig()
    defaults = build_default_services(config)
    cold = run(defaults, config)
    shared_cache = MemoryCacheStore()
    services = RuntimeServices(
        cache=shared_cache,
        baselines=defaults.baselines,
        resources=defaults.resources,
    )
    first = run(services, config)
    second = run(services, config)  # warm hit
    disabled = run(
        RuntimeServices(
            cache=NullCacheStore(),
            baselines=services.baselines,
            resources=services.resources,
        ),
        config,
    )
    assert canonical_bytes(first) == canonical_bytes(second) == canonical_bytes(disabled)
    assert canonical_bytes(first) == canonical_bytes(cold)


def test_duplicate_content_pair_members_refs_and_hashes(tmp_path) -> None:
    # identical content under two artifact ids: same evidence hash, distinct refs
    config = StylogConfig()
    services = build_default_services(config)
    ctx = build_context(config, services)
    text = "the quick brown fox jumps over the lazy dog. " * 8
    fps = []
    for name in ("original.txt", "copy.txt"):
        artifact = artifact_from_bytes(
            text.encode(),
            artifact_id=name,
            kind="text",
            language="en",
            encoding="utf-8",
            config=config,
        )
        fps.append(
            fingerprint_artifact(artifact, config=config, services=services, ctx=ctx).fingerprint
        )
    assert fps[0].artifact.artifact_id == "original.txt"
    assert fps[1].artifact.artifact_id == "copy.txt"
    assert scientific_sha256(fps[0]) != scientific_sha256(fps[1])  # instance id differs
    model = make_model((TTR,), languages=("en",))
    from stylog.application.verify import verify_subjects

    verification = verify_subjects(
        fps[0], fps[1], model, left_ref="original.txt", right_ref="copy.txt"
    )
    assert verification.left_ref == "original.txt"
    assert verification.right_ref == "copy.txt"
    assert verification.left_fingerprint_sha256 == scientific_sha256(fps[0])
    assert verification.right_fingerprint_sha256 == scientific_sha256(fps[1])


def test_load_verifier_malformed_typed_error(tmp_path) -> None:
    import pytest

    from stylog.exceptions import PortableArtifactError
    from stylog.serialization.jsonio import read_json

    model, _ = fit_verifier_model(make_spec(feature_ids=(TTR,)), separable_pairs())
    path = tmp_path / "model.json"
    write_json_atomic(path, model)
    # corrupt the thresholds: t_diff > t_same violates the band invariant
    data = path.read_text(encoding="utf-8")
    tree = __import__("json").loads(data)
    tree["thresholds"] = {"t_same": 0.3, "t_diff": 0.7}
    bad = tmp_path / "bad.json"
    bad.write_text(__import__("json").dumps(tree) + "\n", encoding="utf-8")
    with pytest.raises(PortableArtifactError):
        read_json(bad, VerifierFit)
