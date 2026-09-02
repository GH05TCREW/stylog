"""Offline end-to-end test (spec 26.9): sockets blocked, core workflows pass."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_core_workflows_with_sockets_blocked(tmp_path) -> None:
    script = textwrap.dedent(
        """
        import socket

        class _NoSocket:
            def __init__(self, *a, **k):
                raise OSError("network is blocked in this test")

        socket.socket = _NoSocket
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(OSError("blocked"))

        import pathlib, tempfile
        import stylog
        from stylog.application.compare import compare_subjects
        from stylog.application.aggregate import aggregate_evidence
        from stylog.domain import EvidenceMember, EvidenceSet, LinkageDescriptor

        fp_text = stylog.fingerprint_text("Hello world. Another sentence here.", language="en")
        fp_py = stylog.fingerprint_bytes(b"def f(x):\\n    return x + 1\\n", kind="code", language="python")
        fp_js = stylog.fingerprint_bytes(b"const x = 1; // c\\n", kind="code", language="javascript")
        fp_c = stylog.fingerprint_bytes(b"int main(void) { return 0; }\\n", kind="code", language="c")
        fp_rs = stylog.fingerprint_bytes(b"fn main() { let x = 1; }\\n", kind="code", language="rust")
        for fp in (fp_text, fp_py, fp_js, fp_c, fp_rs):
            assert any(f.status == "ok" for f in fp.features)
        assert any(f.feature_id.startswith("code.parser") for f in fp_js.features)

        es = EvidenceSet(
            evidence_set_id="offline",
            members=(
                EvidenceMember(member_id="m0", artifact_id=fp_text.artifact.artifact_id),
                EvidenceMember(member_id="m1", artifact_id=fp_text.artifact.artifact_id),
            ),
            linkage=LinkageDescriptor(kind="test", source="offline"),
        )
        agg = aggregate_evidence(es, [fp_text, fp_text])
        assert agg.aggregates

        comparison = compare_subjects(fp_text, fp_text)
        assert comparison.families

        with tempfile.TemporaryDirectory() as d:
            from stylog.application.profile import build_baseline
            from stylog.domain import BaselineDescriptor
            fps = [stylog.fingerprint_text(f"sample text number {i} with words.", language="en") for i in range(25)]
            baseline = build_baseline(
                fps, baseline_id="offline-base", baseline_version="1.0.0",
                descriptor=BaselineDescriptor(kind="text", language="en", domain="test",
                                              unit="artifact", source="offline"),
            )
            from stylog.serialization.jsonio import write_json_atomic
            from stylog.application.profile import profile_subject
            from stylog.bootstrap import build_default_services
            from stylog.config import StylogConfig
            bpath = pathlib.Path(d) / "b.stylog-baseline.json"
            write_json_atomic(bpath, baseline)
            cfg = StylogConfig(baseline={"search_paths": [str(d)]})
            services = build_default_services(cfg)
            profile = profile_subject(fps[0], "offline-base", services=services)
            assert profile.observations
        print("OFFLINE OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL
    )
    assert result.returncode == 0, result.stderr
    assert "OFFLINE OK" in result.stdout
