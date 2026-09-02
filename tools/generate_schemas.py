"""Generate committed JSON Schemas for portable models (spec 26.5).

Schemas are written as canonical JCS bytes + one LF. ``--check`` fails on any
drift so CI can gate on uncommitted schema changes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rfc8785
from pydantic import BaseModel

from stylog.domain import PORTABLE_MODELS_BY_SCHEMA

SCHEMA_MODELS: dict[str, type[BaseModel]] = PORTABLE_MODELS_BY_SCHEMA

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


def render_schema(model: type[BaseModel]) -> bytes:
    tree = model.model_json_schema(mode="serialization")
    return rfc8785.dumps(tree) + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    drift: list[str] = []
    for name, model in sorted(SCHEMA_MODELS.items()):
        data = render_schema(model)
        path = SCHEMA_DIR / f"{name}.schema.json"
        if args.check:
            if not path.is_file() or path.read_bytes() != data:
                drift.append(name)
        else:
            SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    if drift:
        print("schema drift:", ", ".join(drift), file=sys.stderr)
        return 1
    if not args.check:
        print(f"wrote {len(SCHEMA_MODELS)} schemas to {SCHEMA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
