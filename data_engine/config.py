from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_CONFIG_CACHE: dict[str, Any] | None = None
_CONFIG_MTIME: float = 0
_CONFIG_LOCK = threading.Lock()


def load_config() -> dict[str, Any]:
    global _CONFIG_CACHE, _CONFIG_MTIME
    try:
        mtime = _CONFIG_PATH.stat().st_mtime if _CONFIG_PATH.exists() else 0
    except OSError:
        mtime = 0
    if _CONFIG_CACHE is not None and mtime == _CONFIG_MTIME:
        return _CONFIG_CACHE
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None and mtime == _CONFIG_MTIME:
            return _CONFIG_CACHE
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                _CONFIG_CACHE = yaml.safe_load(f) or {}
        else:
            _CONFIG_CACHE = {}
        _CONFIG_MTIME = mtime
        return _CONFIG_CACHE


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
