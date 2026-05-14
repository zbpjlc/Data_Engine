from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict[str, Any]:
    global _CONFIG
    if _CONFIG is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _CONFIG = yaml.safe_load(f) or {}
        else:
            _CONFIG = {}
    return _CONFIG


def get_config(*keys: str, default: Any = None) -> Any:
    cfg = load_config()
    for key in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(key)
        else:
            return default
        if cfg is None:
            return default
    return cfg
