from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sample_id_for_page(
    source_id: str,
    category: str,
    batch_id: str,
    page_id: str,
    task_type: str = "page",
) -> str:
    payload = f"{source_id}|{category}|{batch_id}|{page_id}|{task_type}".encode("utf-8")
    # 使用前12位哈希值，足够唯一且更友好
    full_hash = sha256_bytes(payload)
    return full_hash[:12]
