from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any


_CONTAINER_NAME = os.getenv("OMNIDOCBENCH_CONTAINER", "omnidocbench")
_BATCH_BRIDGE_SCRIPT = os.path.join(os.path.dirname(__file__), "teds_container_bridge_batch.py")
_DEFAULT_PYTHON = os.getenv(
    "OMNIDOCBENCH_PYTHON",
    "/opt/miniconda310/envs/omnidocbench_v16_smoke_20260408_py310/bin/python",
)


@dataclass(frozen=True)
class TEDSScore:
    score: float
    source: str
    stderr: str | None = None


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "ps"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _container_running(name: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            check=True,
            text=True,
        )
        return any(line.strip() == name for line in out.stdout.splitlines())
    except Exception:
        return False


def _start_container(name: str) -> bool:
    try:
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", name,
                "--workdir", "/workspace",
                "--entrypoint", "/bin/sh",
                "omnidocbench:latest",
                "-c", "tail -f /dev/null",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def _ensure_container() -> tuple[bool, str]:
    if not _docker_available():
        return False, "docker unavailable"
    if _container_running(_CONTAINER_NAME):
        return True, ""
    if _start_container(_CONTAINER_NAME):
        return True, "started"
    return False, f"container {_CONTAINER_NAME} unavailable"


def _run_in_container(payload: dict, *, timeout_sec: int = 120) -> Any | None:
    ok, _ = _ensure_container()
    if not ok:
        return None

    try:
        subprocess.run(
            ["docker", "cp", str(_BATCH_BRIDGE_SCRIPT), f"{_CONTAINER_NAME}:/workspace/src/metrics/teds_metric_bridge.py"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    try:
        proc = subprocess.run(
            [
                "docker", "exec", "-i", "--workdir", "/workspace",
                _CONTAINER_NAME,
                _DEFAULT_PYTHON,
                "/workspace/src/metrics/teds_metric_bridge.py",
            ],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout_sec + 30,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return json.loads(proc.stdout)
    except Exception:
        return None


def compute_teds(html_a: str, html_b: str, *, structure_only: bool = False, timeout_sec: int = 120) -> TEDSScore | None:
    payload = {
        "pairs": [{"html_a": html_a or "", "html_b": html_b or ""}],
        "structure_only": structure_only,
        "timeout_sec": timeout_sec,
    }
    data = _run_in_container(payload, timeout_sec=timeout_sec)
    if not data:
        return None
    results = data.get("results") or []
    if not results:
        return None
    item = results[0]
    return TEDSScore(score=float(item.get("score", 0.0)), source="container_teds", stderr=item.get("error"))


def compute_teds_batch(pairs: list[tuple[str, str]], *, structure_only: bool = False, timeout_sec: int = 120) -> list[TEDSScore] | None:
    payload = {
        "pairs": [{"html_a": a or "", "html_b": b or ""} for a, b in pairs],
        "structure_only": structure_only,
        "timeout_sec": timeout_sec,
    }
    data = _run_in_container(payload, timeout_sec=timeout_sec)
    if not data:
        return None
    results = data.get("results") or []
    return [TEDSScore(score=float(item.get("score", 0.0)), source="container_teds", stderr=item.get("error")) for item in results]
