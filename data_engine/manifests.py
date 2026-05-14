from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import lance

from pydantic import BaseModel


# ─── Lance schema for ingest manifest ──────────────────────────────────────────

MANIFEST_SCHEMA = pa.schema([
    pa.field("sample_id", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("category", pa.string(), nullable=False),
    pa.field("batch_id", pa.string(), nullable=False),
    pa.field("input_type", pa.string(), nullable=False),
    pa.field("original_ext", pa.string(), nullable=False),
    pa.field("page_id", pa.string(), nullable=False),
    pa.field("task_type", pa.string(), nullable=False),
    pa.field("relative_path", pa.string(), nullable=False),
    pa.field("page_image", pa.string(), nullable=False),
    pa.field("block_list", pa.string(), nullable=False),        # JSON string
    pa.field("reading_order", pa.string(), nullable=False),      # JSON string
    pa.field("table_structure", pa.string(), nullable=True),     # JSON string or null
    pa.field("formula_spans", pa.string(), nullable=False),      # JSON string
    pa.field("text_spans", pa.string(), nullable=False),         # JSON string
    pa.field("bbox", pa.string(), nullable=True),                # JSON string or null
    pa.field("cluster_id", pa.string(), nullable=True),
    pa.field("difficulty", pa.string(), nullable=True),
    pa.field("annotation_source", pa.string(), nullable=True),
    pa.field("stage_status", pa.string(), nullable=False),
    pa.field("process_log", pa.string(), nullable=False),        # JSON string
    pa.field("data_version", pa.string(), nullable=False),
    pa.field("embedding", pa.list_(pa.float32()), nullable=True),
    pa.field("schema_version", pa.string(), nullable=False),
    pa.field("threshold_version", pa.string(), nullable=False),
    pa.field("is_active", pa.bool_(), nullable=False),
    pa.field("page_image_sha256", pa.string(), nullable=False),
    pa.field("image_data", pa.binary(), nullable=True),            # image bytes
    pa.field("source_metadata", pa.string(), nullable=True),     # JSON string or null
    pa.field("created_at", pa.string(), nullable=True),
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
            # binary data, pass through as-is (bytes or None)
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
        else:
            record[name] = val
    return record


# ─── Lance manifest I/O ────────────────────────────────────────────────────────

def _rows_to_table(rows: list[dict]) -> pa.Table:
    """Convert a list of row dicts to a PyArrow table with the correct schema."""
    return pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)


def write_manifest(path: Path, records: list[dict]) -> None:
    """Write records to a Lance dataset (overwrite)."""
    if not records:
        return
    ensure_parent(path)
    rows = [_record_to_arrow(r) for r in records]
    table = _rows_to_table(rows)
    lance.write_dataset(table, str(path), mode="overwrite")


def append_manifest(path: Path, records: list[dict]) -> None:
    """Append records to an existing Lance dataset."""
    if not records:
        return
    rows = [_record_to_arrow(r) for r in records]
    table = _rows_to_table(rows)
    lance.write_dataset(table, str(path), mode="append")


def read_manifest(path: Path) -> list[dict]:
    """Read all records from a Lance dataset."""
    if not path.exists():
        return []
    ds = lance.dataset(str(path))
    table = ds.to_table()
    records = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        records.append(_arrow_to_record(row))
    return records


def query_relative_paths(path: Path) -> set[str]:
    """Efficiently query only the relative_path column (for resume logic)."""
    if not path.exists():
        return set()
    ds = lance.dataset(str(path))
    col = ds.to_table(columns=["relative_path"]).column("relative_path")
    return set(col.to_pylist())


def manifest_count(path: Path) -> int:
    """Return the number of records in a Lance dataset."""
    if not path.exists():
        return 0
    ds = lance.dataset(str(path))
    return ds.count_rows()


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
