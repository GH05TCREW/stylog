"""Configuration: TOML only, tomllib + strict validation (spec section 16)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from stylog.domain._base import PortableModel, tuple_of
from stylog.exceptions import CapabilityUnavailableError, ConfigurationError
from stylog.serialization.canonical import sha256_of_tree

DEFAULT_INCLUDE = (
    "**/*.py",
    "**/*.js",
    "**/*.mjs",
    "**/*.cjs",
    "**/*.ts",
    "**/*.tsx",
    "**/*.c",
    "**/*.rs",
    "**/*.txt",
    "**/*.md",
    "**/*.rst",
)
DEFAULT_EXCLUDE = (
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/venv/**",
    "**/dist/**",
    "**/build/**",
)


class InputConfig(PortableModel):
    text_encoding: str = "utf-8"
    include_hidden: bool = False
    include: tuple_of(str) = DEFAULT_INCLUDE
    exclude: tuple_of(str) = DEFAULT_EXCLUDE
    max_file_bytes: int = 8_388_608
    max_files: int = 10_000
    max_total_bytes: int = 536_870_912


class TextAnalysisConfig(PortableModel):
    enabled: bool = True
    function_words_en: bool = True
    window_ttr_100: bool = True


class EmbeddedTextConfig(PortableModel):
    language: str = "und"


class PythonAnalysisConfig(PortableModel):
    enabled: bool = True
    max_ast_bytes: int = 2_097_152
    max_ast_nesting: int = 200
    embedded_text: bool = True
    max_embedded_artifacts: int = 5_000
    embedded_text_language: EmbeddedTextConfig = EmbeddedTextConfig()


class TreeSitterAnalysisConfig(PortableModel):
    enabled: bool = True


class CodeAnalysisConfig(PortableModel):
    enabled: bool = True
    python: PythonAnalysisConfig = PythonAnalysisConfig()
    tree_sitter: TreeSitterAnalysisConfig = TreeSitterAnalysisConfig()


class AnalysisConfig(PortableModel):
    language: str = "und"
    export_content_hashes: bool = True
    text: TextAnalysisConfig = TextAnalysisConfig()
    code: CodeAnalysisConfig = CodeAnalysisConfig()


class ExecutionConfig(PortableModel):
    mode: Literal["serial", "process"] = "serial"
    workers: int = 0
    max_in_flight: int = 0


class CacheConfig(PortableModel):
    enabled: bool = True


class BaselineConfig(PortableModel):
    search_paths: tuple_of(str) = ()


class NlpConfig(PortableModel):
    enabled: bool = False
    model: str = ""


class MlConfig(PortableModel):
    # reserved for future ML defaults
    enabled: bool = False


class DataConfig(PortableModel):
    parquet_compression: str = "zstd"
    row_group_size: int = 65_536


class StylogConfig(PortableModel):
    version: Literal[1] = 1
    input: InputConfig = InputConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    execution: ExecutionConfig = ExecutionConfig()
    cache: CacheConfig = CacheConfig()
    baseline: BaselineConfig = BaselineConfig()
    nlp: NlpConfig | None = None
    ml: MlConfig | None = None
    data: DataConfig | None = None

    @model_validator(mode="after")
    def _optional_blocks_require_capability(self) -> StylogConfig:
        import importlib.util

        for block_name, module_name in (("nlp", "spacy"), ("ml", "sklearn")):
            block = getattr(self, block_name)
            if block is not None and importlib.util.find_spec(module_name) is None:
                raise CapabilityUnavailableError(
                    f"configuration block [{block_name}] requires the '{block_name}' extra"
                )
        if self.data is not None:
            for module_name in ("pyarrow", "polars", "duckdb", "pandas"):
                if importlib.util.find_spec(module_name) is None:
                    raise CapabilityUnavailableError(
                        "configuration block [data] requires the 'data' extra"
                    )
        return self

    def scientific_subset(self) -> dict[str, Any]:
        """Settings capable of changing deterministic fingerprint values/status (16.5)."""
        analysis = self.analysis
        subset: dict[str, Any] = {
            "input": {"text_encoding": self.input.text_encoding},
            "analysis": {
                "language": analysis.language,
                "text": {
                    "enabled": analysis.text.enabled,
                    "function_words_en": analysis.text.function_words_en,
                    "window_ttr_100": analysis.text.window_ttr_100,
                },
                "code": {
                    "enabled": analysis.code.enabled,
                    "python": {
                        "enabled": analysis.code.python.enabled,
                        "max_ast_bytes": analysis.code.python.max_ast_bytes,
                        "max_ast_nesting": analysis.code.python.max_ast_nesting,
                        "embedded_text": analysis.code.python.embedded_text,
                        "max_embedded_artifacts": analysis.code.python.max_embedded_artifacts,
                        "embedded_text_language": {
                            "language": analysis.code.python.embedded_text_language.language
                        },
                    },
                    "tree_sitter": {"enabled": analysis.code.tree_sitter.enabled},
                },
            },
        }
        if self.nlp is not None and self.nlp.enabled:
            subset["nlp"] = {"enabled": True, "model": self.nlp.model}
        return subset

    def analysis_config_sha256(self) -> str:
        return sha256_of_tree(self.scientific_subset())


DEFAULT_CONFIG = StylogConfig()


def parse_config_dict(data: dict[str, Any]) -> StylogConfig:
    try:
        return StylogConfig.model_validate(data)
    except CapabilityUnavailableError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"invalid Stylog configuration: {exc}") from exc


def load_config(
    explicit_path: str | os.PathLike[str] | None = None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> StylogConfig:
    """Discovery/precedence per spec 16.2. Only the current directory is inspected."""
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd

    env_path = env.get("STYLOG_CONFIG")
    chosen: Path | None = None
    if explicit_path is not None:
        chosen = Path(explicit_path)
        if not chosen.is_file():
            raise ConfigurationError(f"config file not found: {chosen}")
        return _load_toml_config(chosen)
    if env_path:
        chosen = Path(env_path)
        if not chosen.is_file():
            raise ConfigurationError(f"config file not found: {chosen}")
        return _load_toml_config(chosen)

    stylog_toml = cwd / "stylog.toml"
    if stylog_toml.is_file():
        return _load_toml_config(stylog_toml)

    pyproject = cwd / "pyproject.toml"
    if pyproject.is_file():
        with pyproject.open("rb") as handle:
            data = tomllib.load(handle)
        section = data.get("tool", {}).get("stylog")
        if section is not None:
            return parse_config_dict(section)

    return DEFAULT_CONFIG


def _load_toml_config(path: Path) -> StylogConfig:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {path}: {exc}") from exc
    return parse_config_dict(data)


def parse_no_cache_env(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    raise ConfigurationError(f"invalid STYLOG_NO_CACHE value: {value!r}")
