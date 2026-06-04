from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
import subprocess
import pyarrow as pa
import lance
from enum import Enum
from pydantic import BaseModel
from data_engine.config import get_config

_lance_write_lock = threading.Lock()

# ─── Lance schema for ingest manifest ──────────────────────────────────────────

MANIFEST_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("source_id", pa.large_string(), nullable=False),
    pa.field("category", pa.large_string(), nullable=False),
    pa.field("batch_id", pa.large_string(), nullable=False),
    pa.field("input_type", pa.large_string(), nullable=False),
    pa.field("original_ext", pa.large_string(), nullable=False),
    pa.field("page_id", pa.large_string(), nullable=False),
    pa.field("task_type", pa.large_string(), nullable=False),
    pa.field("relative_path", pa.large_string(), nullable=False),
    pa.field("page_image", pa.large_string(), nullable=False),
    pa.field("block_list", pa.large_string(), nullable=False),        # JSON string
    pa.field("reading_order", pa.large_string(), nullable=False),      # JSON string
    pa.field("table_structure", pa.large_string(), nullable=True),     # JSON string or null
    pa.field("formula_spans", pa.large_string(), nullable=False),      # JSON string
    pa.field("text_spans", pa.large_string(), nullable=False),         # JSON string
    pa.field("bbox", pa.large_string(), nullable=True),                # JSON string or null
    pa.field("cluster_id", pa.large_string(), nullable=True),
    pa.field("difficulty", pa.large_string(), nullable=True),
    pa.field("annotation_source", pa.large_string(), nullable=True),
    pa.field("stage_status", pa.large_string(), nullable=False),
    pa.field("process_log", pa.large_string(), nullable=False),        # JSON string
    pa.field("data_version", pa.large_string(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32(), get_config("embedding", "embedding_dim", default=768)), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("threshold_version", pa.large_string(), nullable=False),
    pa.field("is_active", pa.bool_(), nullable=False),
    pa.field("page_image_sha256", pa.large_string(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),            # image bytes
    pa.field("source_metadata", pa.large_string(), nullable=True),     # JSON string or null
    pa.field("created_at", pa.large_string(), nullable=True),
])

# Fields that are stored as JSON strings in Lance but as dicts/lists in Python
_JSON_FIELDS = {
    "block_list", "reading_order", "table_structure", "formula_spans",
    "text_spans", "bbox", "process_log", "source_metadata",
}


# ─── Lance helpers ─────────────────────────────────────────────────────────────

def _record_to_arrow(record: dict) -> dict:
    """Convert a Python dict record to Arrow-compatible types."""
    row = {}
    for field in MANIFEST_SCHEMA:
        name = field.name
        val = record.get(name)
        if name == "embedding":
            row[name] = val if val else None
        elif name == "image_data":
            row[name] = val
        elif name in _JSON_FIELDS:
            row[name] = json.dumps(val, ensure_ascii=False) if val is not None else None
        elif name == "created_at":
            if val is None:
                row[name] = None
            elif isinstance(val, datetime):
                row[name] = val.isoformat()
            else:
                row[name] = str(val)
        elif name == "is_active":
            row[name] = bool(val) if val is not None else True
        elif isinstance(val, Enum):
            row[name] = val.value
        else:
            row[name] = str(val) if val is not None else None
    return row


def _arrow_to_record(row: dict) -> dict:
    """Convert an Arrow row dict back to a Python dict with native types."""
    record = {}
    for name, val in row.items():
        if name in _JSON_FIELDS:
            if val is None:
                record[name] = None
            else:
                try:
                    record[name] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    record[name] = val
        elif name == "is_active":
            record[name] = bool(val) if val is not None else True
        elif name == "created_at":
            record[name] = val  # keep as string
        elif name == "input_type":
            # 兼容旧格式 "InputType.PDF" -> "pdf"
            if val and "." in str(val):
                val = str(val).split(".")[-1].lower()
            record[name] = val
        elif name == "stage_status":
            # 兼容旧格式 "StageStatus.INGESTED" -> "ingested"
            if val and "." in str(val):
                val = str(val).split(".")[-1].lower()
            record[name] = val
        else:
            record[name] = val
    return record


# ─── Lance manifest I/O ────────────────────────────────────────────────────────

def _rows_to_table(rows: list[dict]) -> pa.Table:
    """Convert a list of row dicts to a PyArrow table with the correct schema."""
    return pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)


def _supports_atomic_rename(path: Path) -> bool:
    """Check if the filesystem supports atomic rename (POSIX)."""
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            capture_output=True, text=True
        )
        fs_type = result.stdout.strip()
        # exFAT/FAT32 不支持原子 rename
        return fs_type not in ("exfat", "vfat", "fuseblk")
    except Exception:
        return True  # 默认假设支持


def _cleanup_lance_versions(target_path: Path) -> None:
    """Keep only the most recent N versions of a Lance dataset."""
    try:
        keep = get_config("lance", "keep_versions", default=5)
        if target_path.exists():
            ds = lance.dataset(str(target_path))
            ds.cleanup_old_versions(retain_versions=keep)
    except Exception:
        pass


def _parse_size(size_str: str | None) -> int | None:
    """Parse size string like '2GB', '500MB' to bytes."""
    if not size_str:
        return None
    size_str = str(size_str).strip().upper()
    units = {"GB": 1024**3, "MB": 1024**2, "KB": 1024, "B": 1}
    for suffix, multiplier in units.items():
        if size_str.endswith(suffix):
            try:
                return int(float(size_str[:-len(suffix)]) * multiplier)
            except ValueError:
                return None
    try:
        return int(size_str)
    except ValueError:
        return None


def _safe_write_lance(table: pa.Table, target_path: Path, mode: str = "overwrite") -> None:
    """Write Lance dataset. Detects filesystem and uses appropriate strategy."""
    with _lance_write_lock:
        _safe_write_lance_inner(table, target_path, mode)


def _safe_write_lance_inner(table: pa.Table, target_path: Path, mode: str = "overwrite") -> None:
    """Internal write implementation (must be called under _lance_write_lock)."""
    ensure_parent(target_path)

    max_bytes = _parse_size(get_config("lance", "max_file_size", default=None))
    write_kwargs = {"mode": mode}
    if max_bytes:
        write_kwargs["max_bytes_per_file"] = max_bytes

    if _supports_atomic_rename(target_path.parent):
        # ext4/NFS: 直接写入
        lance.write_dataset(table, str(target_path), **write_kwargs)
        _cleanup_lance_versions(target_path)
        return

    # exFAT: 先写本地再 move
    print(f"[Lance] 检测到非POSIX文件系统，使用fallback写入", file=sys.stderr)
    tmp_dir = Path(tempfile.mkdtemp(prefix="lance_tmp_"))
    try:
        tmp_lance = tmp_dir / "data.lance"
        if mode == "append" and target_path.exists():
            # 尝试读取已有数据合并，损坏则丢弃旧数据重写
            try:
                existing_ds = lance.dataset(str(target_path))
                existing_table = existing_ds.to_table()
                combined = pa.concat_tables([existing_table, table])
                lance.write_dataset(combined, str(tmp_lance), **{**write_kwargs, "mode": "overwrite"})
            except Exception as e:
                print(f"[Lance] 旧数据损坏({e})，丢弃重写", file=sys.stderr)
                lance.write_dataset(table, str(tmp_lance), **{**write_kwargs, "mode": "overwrite"})
        else:
            lance.write_dataset(table, str(tmp_lance), **{**write_kwargs, "mode": "overwrite"})
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.move(str(tmp_lance), str(target_path))
        _cleanup_lance_versions(target_path)
    except Exception as e:
        print(f"[Lance] 写入失败: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def write_manifest(path: Path, records: list[dict]) -> None:
    """Write records to a Lance dataset (overwrite)."""
    if not records:
        return
    rows = [_record_to_arrow(r) for r in records]
    table = _rows_to_table(rows)
    _safe_write_lance(table, path, mode="overwrite")


def append_manifest(path: Path, records: list[dict]) -> None:
    """Append records to an existing Lance dataset."""
    if not records:
        return
    rows = [_record_to_arrow(r) for r in records]
    table = _rows_to_table(rows)
    _safe_write_lance(table, path, mode="append")


def read_manifest(path: Path, columns: list[str] | None = None) -> list[dict]:
    """Read records from a Lance dataset.
    
    Args:
        path: Path to the Lance dataset
        columns: List of column names to read. If None, reads all columns.
    """
    if not path.exists():
        return []
    with _lance_write_lock:
        ds = lance.dataset(str(path))
        table = ds.to_table(columns=columns)
        return [_arrow_to_record(row) for row in table.to_pylist()]


def query_relative_paths(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        ds = lance.dataset(str(path))
        col = ds.to_table(columns=["relative_path"]).column("relative_path")
        return set(col.to_pylist())
    except Exception as e:
        print(f"读取 Lance 数据集失败 {path}: {e}", file=sys.stderr)
        return set()


def query_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        ds = lance.dataset(str(path))
        col = ds.to_table(columns=["sample_id"]).column("sample_id")
        return set(col.to_pylist())
    except Exception as e:
        print(f"读取 Lance 数据集失败 {path}: {e}", file=sys.stderr)
        return set()


def manifest_count(path: Path) -> int:
    """Return the number of records in a Lance dataset."""
    if not path.exists():
        return 0
    
    # 仅仅是将 Lance 数据集加载进内存，这通常是只读且线程安全的
    try:
        ds = lance.dataset(str(path))
        return ds.count_rows()
    except Exception as e:
        # 防止底层报错导致整个服务挂掉，加个安全的日志打印
        print(f"读取 Lance 数据集失败 {path}: {e}")
        return 0


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ─── Manifest discovery ────────────────────────────────────────────────────────

def stage_manifest_candidates(manifests_dir: Path, stage: str) -> Sequence[Path]:
    return (
        manifests_dir / f"{stage}.lance",
        manifests_dir / f"{stage}.parquet",
    )


def find_stage_manifest(manifests_dir: Path, stage: str) -> Path | None:
    for candidate in stage_manifest_candidates(manifests_dir, stage):
        if candidate.exists():
            return candidate
    return None


def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ─── Element lance I/O ────────────────────────────────────────────────────────

def _element_record_to_arrow(record: dict) -> dict:
    from data_engine.ocr import ELEMENT_SCHEMA, ELEMENT_JSON_FIELDS
    row = {}
    for field in ELEMENT_SCHEMA:
        name = field.name
        val = record.get(name)
        if name in ELEMENT_JSON_FIELDS:
            if val is None:
                row[name] = None
            elif isinstance(val, (dict, list)):
                row[name] = json.dumps(val, ensure_ascii=False)
            else:
                row[name] = str(val)
        elif name == "layout_confidence":
            row[name] = float(val) if val is not None else 0.0
        elif name == "block_idx":
            row[name] = int(val) if val is not None else 0
        elif name in ("paddle_confidence", "glm_confidence", "self_confidence"):
            row[name] = float(val) if val is not None else None
        elif isinstance(val, Enum):
            row[name] = val.value
        else:
            row[name] = str(val) if val is not None else None
    return row


def _element_arrow_to_record(row: dict) -> dict:
    from data_engine.ocr import ELEMENT_JSON_FIELDS
    record = {}
    for name, val in row.items():
        if name in ELEMENT_JSON_FIELDS:
            if val is None:
                record[name] = None
            else:
                try:
                    record[name] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    record[name] = val
        else:
            record[name] = val
    return record


def write_element_manifest(path: Path, records: list[dict]) -> None:
    from data_engine.ocr import ELEMENT_SCHEMA
    if not records:
        return
    rows = [_element_record_to_arrow(r) for r in records]
    table = pa.Table.from_pylist(rows, schema=ELEMENT_SCHEMA)
    _safe_write_lance(table, path, mode="overwrite")


def append_element_manifest(path: Path, records: list[dict]) -> None:
    from data_engine.ocr import ELEMENT_SCHEMA
    if not records:
        return
    rows = [_element_record_to_arrow(r) for r in records]
    table = pa.Table.from_pylist(rows, schema=ELEMENT_SCHEMA)
    _safe_write_lance(table, path, mode="append")


def read_element_manifest(path: Path, columns: list[str] | None = None) -> list[dict]:
    if not path.exists():
        return []
    with _lance_write_lock:
        ds = lance.dataset(str(path))
        table = ds.to_table(columns=columns)
        return [_element_arrow_to_record(row) for row in table.to_pylist()]


def merge_insert_element(path: Path, records: list[dict], on_columns: list[str] | None = None) -> None:
    from data_engine.ocr import ELEMENT_SCHEMA
    if not records:
        return
    on_cols = on_columns or ["sample_id", "block_idx"]
    rows = [_element_record_to_arrow(r) for r in records]
    table = pa.Table.from_pylist(rows, schema=ELEMENT_SCHEMA)
    with _lance_write_lock:
        ds = lance.dataset(str(path))
        ds.merge_insert(on_cols).when_matched_update_all().execute(table)
