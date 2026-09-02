"""Shared base machinery for portable scientific models.

Portable Stylog objects are strict Pydantic v2 models. The rules enforced here
are normative (spec section 5): no unknown fields, no explicit JSON null, no
lone surrogates, no unsafe integers, no negative zero floats, immutable nested
containers (tuples).
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from enum import StrEnum

# The portable contract normatively names a field "schema"; silence the
# shadowing notice emitted when models are defined.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "schema".*shadows an attribute.*',
    category=UserWarning,
)

from typing import Annotated, Any, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, StringConstraints, model_validator

HexDigest64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

MAX_SAFE_INT = 9_007_199_254_740_991  # 2**53 - 1


def _normalize_float(value: Any) -> Any:
    """Normalize -0.0 to 0.0 before validation."""
    if isinstance(value, float) and value == 0.0:
        return 0.0
    return value


PortableFloat = Annotated[float, BeforeValidator(_normalize_float)]


def _list_to_tuple(value: Any) -> Any:
    """Accept JSON arrays (lists) at the validation boundary; store tuples."""
    if isinstance(value, list):
        return tuple(value)
    return value


T = TypeVar("T")


def tuple_of(inner: Any) -> Any:
    """A strict tuple field type that also accepts a JSON array when parsing."""
    return Annotated[tuple[inner, ...], BeforeValidator(_list_to_tuple)]


def is_sorted_unique(values: Iterable[T]) -> bool:
    """True when ``values`` is sorted ascending with no duplicates."""
    items = list(values)
    return items == sorted(items) and len(set(items)) == len(items)


def _has_lone_surrogate(text: str) -> bool:
    return any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _check_portable_tree(value: Any, path: str = "$") -> None:
    """Reject null, lone surrogates, and unsafe integers anywhere in input."""
    if value is None:
        raise ValueError(f"portable JSON must not contain null (at {path})")
    if isinstance(value, str):
        if _has_lone_surrogate(value):
            raise ValueError(f"portable JSON must not contain lone surrogates (at {path})")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INT:
            raise ValueError(f"portable integer out of safe interchange range (at {path})")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _has_lone_surrogate(key):
                raise ValueError(f"portable JSON key has lone surrogates (at {path})")
            _check_portable_tree(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_portable_tree(item, f"{path}[{index}]")


class PortableModel(BaseModel):
    """Base class for every portable Stylog scientific model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _reject_nonportable_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _check_portable_tree(data)
        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_enum_strings(cls, data: Any) -> Any:
        """Accept plain strings for StrEnum fields when parsing portable JSON.

        Strict mode requires enum instances for enum fields; portable JSON
        carries their string values. Converting here keeps ``strict=True``
        semantics for every other type.
        """
        if not isinstance(data, dict):
            return data
        for name, model_field in cls.model_fields.items():
            annotation = model_field.annotation
            if isinstance(annotation, type) and issubclass(annotation, StrEnum):
                value = data.get(name)
                if isinstance(value, str):
                    data[name] = annotation(value)
        return data
