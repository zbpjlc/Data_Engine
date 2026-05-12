from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any


try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover
    _yaml = None


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if _yaml is not None:
        data = _yaml.safe_load(text)
        return data or {}
    return _parse_simple_sources_yaml(text)


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    if _yaml is not None:
        text = _yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    else:
        text = _dump_simple_sources_yaml(data)
    path.write_text(text, encoding="utf-8")


def _strip_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _parse_simple_sources_yaml(text: str) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.strip() == "sources:":
            continue
        if re.match(r"^\s*-\s+", line):
            if current:
                sources.append(current)
            current = {}
            tail = re.sub(r"^\s*-\s+", "", line)
            if tail:
                key, _, value = tail.partition(":")
                current[key.strip()] = _coerce_scalar(_strip_value(value))
            continue
        if current is None:
            raise ValueError("Unsupported YAML format without PyYAML")
        key, _, value = line.strip().partition(":")
        current[key.strip()] = _coerce_scalar(_strip_value(value))
    if current:
        sources.append(current)
    return {"sources": sources}


def _dump_simple_sources_yaml(data: dict[str, Any]) -> str:
    sources = data.get("sources", [])
    lines = ["sources:"]
    for source in sources:
        first = True
        for key, value in source.items():
            dumped = json.dumps(value, ensure_ascii=False)
            prefix = "  - " if first else "    "
            lines.append(f"{prefix}{key}: {dumped}")
            first = False
    return "\n".join(lines) + "\n"


def _coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    return value
