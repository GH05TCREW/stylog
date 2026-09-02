"""Parallel determinism (spec 18.16, 25): worker count never changes output."""

from __future__ import annotations

from stylog.application.batch import fingerprint_iter
from stylog.config import StylogConfig
from stylog.infrastructure.ingest import artifact_from_bytes
from stylog.serialization.canonical import scientific_sha256

SAMPLES = [
    "The quick brown fox jumps over the lazy dog. " * 5,
    "def f(x):\n    return x + 1  # add one\n",
    "const answer = 42;\n// comment\n",
    "int main(void) { return 0; }\n",
    "fn main() { let x: i32 = 1; }\n",
    "Another document, with punctuation! And questions? Yes.",
]

LANGS = ["und", "python", "javascript", "c", "rust", "und"]
KINDS = ["text", "code", "code", "code", "code", "text"]


def _artifacts(config: StylogConfig):
    artifacts = []
    for index, (sample, language, kind) in enumerate(zip(SAMPLES, LANGS, KINDS, strict=True)):
        raw = sample.encode("utf-8")
        artifacts.append(
            artifact_from_bytes(
                raw,
                artifact_id=f"a{index}",
                kind=kind,
                language=language,
                encoding="utf-8",
                config=config,
            )
        )
    return artifacts


def _hashes(bundles) -> list[str]:
    return [scientific_sha256(bundle) for bundle in bundles]


def test_serial_and_process_outputs_identical() -> None:
    config = StylogConfig()
    artifacts = _artifacts(config)
    serial = _hashes(fingerprint_iter(list(artifacts), config=config, execution="serial"))
    process = _hashes(
        fingerprint_iter(list(artifacts), config=config, execution="process", workers=2)
    )
    assert serial == process


def test_worker_count_invariance() -> None:
    config = StylogConfig()
    artifacts = _artifacts(config)
    one = _hashes(
        fingerprint_iter(list(artifacts), config=config, execution="process", workers=1)
    )
    two = _hashes(
        fingerprint_iter(list(artifacts), config=config, execution="process", workers=2)
    )
    assert one == two


def test_serial_matches_direct_api() -> None:
    import stylog

    config = StylogConfig()
    artifacts = _artifacts(config)
    batch = list(fingerprint_iter(artifacts[:1], config=config, execution="serial"))
    # Same artifact identity through the public API path (artifact ids must
    # match for canonical equality; content identity alone is not enough).
    direct = stylog.fingerprint_bytes(
        SAMPLES[0].encode("utf-8"), kind="text", language="und", config=config
    )
    direct = direct.model_copy(
        update={
            "artifact": direct.artifact.model_copy(
                update={"artifact_id": artifacts[0].artifact_id}
            )
        }
    )
    assert scientific_sha256(batch[0].primary) == scientific_sha256(direct)
