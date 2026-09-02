"""Local baseline resolution (spec 13.11). No network resolution, ever."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_path

from stylog.config import StylogConfig
from stylog.domain.baseline import Baseline
from stylog.exceptions import BaselineError
from stylog.serialization.canonical import scientific_sha256
from stylog.serialization.jsonio import read_json


class FilesystemBaselineResolver:
    def __init__(self, config: StylogConfig) -> None:
        self.search_paths = [Path(p) for p in config.baseline.search_paths]

    def resolve(self, baseline_ref: str) -> Baseline:
        if ("/" in baseline_ref or "\\" in baseline_ref) or baseline_ref.endswith(".json"):
            path = Path(baseline_ref)
            if not path.is_file():
                raise BaselineError(f"BASELINE_NOT_FOUND: no baseline file at {baseline_ref}")
            return _load_baseline(path)

        candidates: list[Baseline] = []
        roots = list(self.search_paths) + [user_data_path("stylog") / "baselines"]
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.json")):
                try:
                    baseline = _load_baseline(path)
                except BaselineError:
                    continue
                if baseline.baseline_id == baseline_ref:
                    candidates.append(baseline)
        if not candidates:
            raise BaselineError(f"BASELINE_NOT_FOUND: no baseline with id {baseline_ref!r}")
        hashes = {scientific_sha256(candidate) for candidate in candidates}
        if len(hashes) > 1:
            raise BaselineError(
                f"BASELINE_INVALID: multiple distinct baselines with id {baseline_ref!r}"
            )
        return candidates[0]


def _load_baseline(path: Path) -> Baseline:
    try:
        return read_json(path, Baseline)
    except Exception as exc:
        raise BaselineError(f"BASELINE_INVALID: {path.name}: {exc}") from exc


class StaticBaselineResolver:
    """In-memory fake for tests."""

    def __init__(self, baselines: dict[str, Baseline]) -> None:
        self.baselines = baselines

    def resolve(self, baseline_ref: str) -> Baseline:
        try:
            return self.baselines[baseline_ref]
        except KeyError as exc:
            raise BaselineError(f"BASELINE_NOT_FOUND: {baseline_ref}") from exc
