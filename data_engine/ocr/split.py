"""将 element.lance 中的 block 按类型分拆到 text/formula/table 三个 Lance 数据集。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

import lance
import pyarrow as pa

from data_engine.manifests import _lance_write_lock, iso_now
from data_engine.ocr import (
    CATEGORY_SCHEMAS,
    FORMULA_BLOCK_TYPES,
    SKIP_BLOCK_TYPES,
    TABLE_BLOCK_TYPES,
    TEXT_BLOCK_TYPES,
)

logger = logging.getLogger(__name__)

# 进度回调类型: (current: int, total: int, message: str) -> None
ProgressCallback = Optional[Callable[[int, int, str], None]]
# 图片懒加载回调: (sample_id: str, bbox_json) -> Optional[bytes]
ImageLoader = Optional[Callable[[str, object], Optional[bytes]]]


def _classify_block(block_type: str) -> str | None:
    if block_type in TEXT_BLOCK_TYPES:
        return "text"
    if block_type in FORMULA_BLOCK_TYPES:
        return "formula"
    if block_type in TABLE_BLOCK_TYPES:
        return "table"
    return None


def _element_row_to_category_row(row: dict, category: str, image_data: bytes | None = None) -> dict:
    base = {
        "sample_id": row["sample_id"],
        "block_idx": row.get("block_idx", 0),
        "block_type": row.get("block_type", ""),
        "bbox_json": json.dumps(row.get("bbox_json", [])) if isinstance(row.get("bbox_json"), list) else str(row.get("bbox_json", "[]")),
        "layout_confidence": row.get("layout_confidence", 0),
        "image_data": image_data,
        "consistency_pattern": row.get("consistency_pattern"),
        "schema_version": "v1",
        "created_at": iso_now(),
    }

    if category == "text":
        for model in ("paddle", "glm", "self"):
            txt = row.get(f"{model}_text")
            if txt:
                base["text_content"] = txt
                base["text_confidence"] = row.get(f"{model}_confidence")
                base["source_model"] = model
                break

    elif category == "formula":
        for model in ("paddle", "glm", "self"):
            frm = row.get(f"{model}_formula")
            if frm:
                base["formula_latex"] = frm
                base["formula_confidence"] = row.get(f"{model}_confidence")
                base["source_model"] = model
                break
        if not base.get("formula_latex"):
            for model in ("paddle", "glm", "self"):
                txt = row.get(f"{model}_text")
                if txt:
                    base["formula_latex"] = txt
                    base["source_model"] = model
                    break

    elif category == "table":
        for model in ("paddle", "glm", "self"):
            tbl = row.get(f"{model}_table")
            if tbl:
                if isinstance(tbl, str):
                    try:
                        tbl = json.loads(tbl)
                    except (json.JSONDecodeError, TypeError):
                        tbl = {"raw": tbl}
                base["table_json"] = json.dumps(tbl, ensure_ascii=False)
                base["table_html"] = tbl.get("html", "") if isinstance(tbl, dict) else ""
                base["table_confidence"] = row.get(f"{model}_confidence")
                base["source_model"] = model
                break

    return base


def split_element_to_category_lances(
    manifests_dir: Path,
    image_map: dict[str, bytes] | None = None,
    progress_callback: ProgressCallback = None,
    image_loader: ImageLoader = None,
) -> dict[str, int]:
    """读 element.lance，按 block_type 分拆到 text.lance / formula.lance / table.lance。

    Args:
        manifests_dir: 包含 element.lance 的目录
        image_map: sample_id -> image_bytes 映射（可选，旧接口，一次性加载）
        progress_callback: 进度回调 (current, total, message)
        image_loader: 懒加载回调 (sample_id, bbox_json) -> bytes | None

    Returns:
        {"text": N, "formula": M, "table": K, "skipped": S}
    """
    def _report(current: int, total: int, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(current, total, msg)
            except Exception:
                pass

    element_path = manifests_dir / "element.lance"
    if not element_path.exists():
        raise FileNotFoundError(f"element.lance 不存在: {element_path}")

    _report(0, 0, "读取 element.lance 数据...")
    with _lance_write_lock:
        ds = lance.dataset(str(element_path))
        rows = ds.to_table().to_pylist()

    total_rows = len(rows)
    _report(0, total_rows, f"读取完成，共 {total_rows} 条 block")

    category_rows: dict[str, list[dict]] = {"text": [], "formula": [], "table": []}
    skipped = 0

    # 分块报告进度，每处理 2000 行报告一次
    REPORT_INTERVAL = 2000
    for idx, row in enumerate(rows):
        bt = row.get("block_type", "")
        cat = _classify_block(bt)
        if cat is None:
            skipped += 1
            continue

        img_data = None
        if image_loader:
            # 懒加载模式：按需读取单张图片并裁剪
            img_data = image_loader(row["sample_id"], row.get("bbox_json"))
        elif image_map and row["sample_id"] in image_map:
            img_bytes = image_map[row["sample_id"]]
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(img_bytes))
                bbox = row.get("bbox_json", [])
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(c) for c in bbox]
                    cropped = img.crop((x1, y1, x2, y2))
                    buf = io.BytesIO()
                    cropped.save(buf, format="JPEG", quality=90)
                    img_data = buf.getvalue()
            except Exception:
                pass

        cat_row = _element_row_to_category_row(row, cat, img_data)
        category_rows[cat].append(cat_row)

        if (idx + 1) % REPORT_INTERVAL == 0 or (idx + 1) == total_rows:
            _report(idx + 1, total_rows, f"分类处理中 {idx + 1}/{total_rows}")

    result = {"skipped": skipped}

    _report(total_rows, total_rows, "正在写入 Lance 数据集...")
    for cat in ("text", "formula", "table"):
        rows_list = category_rows[cat]
        if not rows_list:
            result[cat] = 0
            continue

        schema = CATEGORY_SCHEMAS[cat]
        arrow_rows = []
        for r in rows_list:
            clean = {}
            for field in schema:
                clean[field.name] = r.get(field.name)
            arrow_rows.append(clean)

        table = pa.Table.from_pylist(arrow_rows, schema=schema)
        lance_path = manifests_dir / f"{cat}.lance"

        with _lance_write_lock:
            if lance_path.exists():
                lance.write_dataset(table, str(lance_path), mode="append")
            else:
                lance.write_dataset(table, str(lance_path), mode="create")

        result[cat] = len(rows_list)
        logger.info("[split] %s.lance: %d rows", cat, len(rows_list))

    return result
