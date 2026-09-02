"""Generic code analyzers (spec 6.6, 8.2-8.3).

Language-independent measurements over ``kind="code"`` artifacts: raw size,
physical lines, line endings, whitespace, indentation, blank runs, and
trailing space. All work happens on the already-decoded artifact text; no
parser is involved (``needs="none"``).
"""

from __future__ import annotations

from stylog.analysis import build
from stylog.analysis.base import CODE_SURFACE_BACKEND, AnalyzerOutput, BaseAnalyzer
from stylog.analysis.lines import is_blank, scan_lines
from stylog.analysis.registry import (
    ANALYZER_CODE_SAMPLE,
    ANALYZER_CODE_SURFACE,
    features_owned_by,
)
from stylog.analysis.whitespace import is_white_space, whitespace_class
from stylog.domain.feature import FeatureObservation
from stylog.runtime import AnalysisContext, RuntimeArtifact


def _indent(line_content: str) -> tuple[str, int]:
    """Leading ASCII space/tab run from column 0 (spec 8.3)."""
    has_space = False
    has_tab = False
    count = 0
    for char in line_content:
        if char == " ":
            has_space = True
            count += 1
        elif char == "\t":
            has_tab = True
            count += 1
        else:
            break
    if count == 0:
        return "none", 0
    if has_space and has_tab:
        return "mixed", count
    return ("spaces" if has_space else "tabs"), count


class CodeSampleAnalyzer(BaseAnalyzer):
    """Raw size and physical-line counts (spec 6.6, 8.2). Always ok."""

    analyzer_id = ANALYZER_CODE_SAMPLE
    backend = CODE_SURFACE_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        values = {
            "code.sample.byte_count": len(artifact.raw_bytes),
            "code.sample.character_count": len(artifact.text),
            "code.sample.physical_line_count": len(scan_lines(artifact.text)),
        }
        observations = tuple(
            build.ok(
                fdef,
                self.analyzer_id,
                self.implementation_version,
                build.int_value(values[fdef.feature_id]),
                1,
            )
            for fdef in features_owned_by(self.analyzer_id)
        )
        return AnalyzerOutput(observations=observations)


class CodeSurfaceAnalyzer(BaseAnalyzer):
    """Surface code measurements (spec 6.6, 8.2-8.3)."""

    analyzer_id = ANALYZER_CODE_SURFACE
    backend = CODE_SURFACE_BACKEND

    def analyze(
        self,
        artifact: RuntimeArtifact,
        ctx: AnalysisContext,
        facts: object | None = None,
    ) -> AnalyzerOutput:
        lines = scan_lines(artifact.text)
        nonblank = [line for line in lines if not is_blank(line.content)]

        ending_counts: dict[str, int] = {}
        for line in lines:
            if line.ending is not None:
                ending_counts[line.ending] = ending_counts.get(line.ending, 0) + 1

        ws_counts: dict[str, int] = {}
        for char in artifact.text:
            if is_white_space(char):
                category = whitespace_class(char)
                ws_counts[category] = ws_counts.get(category, 0) + 1

        indent_kinds: dict[str, int] = {}
        indent_counts: list[int] = []
        line_lengths: list[int] = []
        for line in nonblank:
            kind, count = _indent(line.content)
            indent_kinds[kind] = indent_kinds.get(kind, 0) + 1
            indent_counts.append(count)
            line_lengths.append(len(line.content))

        blank_count = sum(1 for line in lines if is_blank(line.content))
        blank_runs: list[int] = []
        run = 0
        for line in lines:
            if is_blank(line.content):
                run += 1
            elif run:
                blank_runs.append(run)
                run = 0
        if run:
            blank_runs.append(run)

        trailing_count = sum(
            1 for line in lines if line.content.endswith((" ", "\t"))
        )
        total_lines = len(lines)

        observations: list[FeatureObservation] = []
        for fdef in features_owned_by(self.analyzer_id):
            fid = fdef.feature_id
            if fid == "code.surface.line_ending":
                obs = build.categorical_observation(fdef, self.analyzer_id, self.implementation_version, ending_counts)
            elif fid == "code.surface.whitespace_class":
                obs = build.categorical_observation(fdef, self.analyzer_id, self.implementation_version, ws_counts)
            elif fid == "code.surface.indent_kind":
                obs = build.categorical_observation(fdef, self.analyzer_id, self.implementation_version, indent_kinds)
            elif fid == "code.surface.indent_char_count":
                obs = build.histogram_observation(fdef, self.analyzer_id, self.implementation_version, indent_counts)
            elif fid == "code.surface.nonblank_line_length":
                obs = build.histogram_observation(fdef, self.analyzer_id, self.implementation_version, line_lengths)
            elif fid == "code.surface.blank_line_share":
                obs = build.ratio_observation(fdef, self.analyzer_id, self.implementation_version, blank_count, total_lines)
            elif fid == "code.surface.blank_run_length":
                obs = build.histogram_observation(fdef, self.analyzer_id, self.implementation_version, blank_runs)
            elif fid == "code.surface.trailing_space_line_share":
                obs = build.ratio_observation(
                    fdef, self.analyzer_id, self.implementation_version, trailing_count, total_lines
                )
            else:  # pragma: no cover - registry drift guard
                raise AssertionError(f"unowned feature {fid}")
            observations.append(obs)
        return AnalyzerOutput(observations=tuple(observations))
