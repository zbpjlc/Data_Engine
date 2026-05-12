from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[BaseModel | dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                payload = record.model_dump(mode="json")
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_manifest_candidates(manifests_dir: Path, stage: str) -> Sequence[Path]:
    return (
        manifests_dir / f"{stage}.jsonl",
        manifests_dir / f"{stage}.parquet",
    )


def find_stage_manifest(manifests_dir: Path, stage: str) -> Path | None:
    for candidate in stage_manifest_candidates(manifests_dir, stage):
        if candidate.exists():
            return candidate
    return None


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
