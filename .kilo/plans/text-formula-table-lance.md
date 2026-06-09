# 文本/公式/表格三类 Lance 数据集实现方案

## 目标
对抽样样本做 layout 后，将 block 按类型分拆到三个独立 Lance 数据集：
- `text.lance` — 文本类 block（text, title, paragraph_title, number, header, footer, code_txt, reference）
- `formula.lance` — 公式类 block（equation_isolated, equation_inline, formula）
- `table.lance` — 表格类 block（table）
- figure/image/abandon 等直接跳过

## 实现步骤

### 1. 修改 `data_engine/ocr/__init__.py` — 添加三个 Schema

在 `ELEMENT_JSON_FIELDS` 后面添加：

```python
# ─── 文本 Lance Schema ─────────────────────────────────────────────────────

TEXT_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("text_content", pa.large_string(), nullable=True),
    pa.field("text_confidence", pa.float32(), nullable=True),
    pa.field("source_model", pa.large_string(), nullable=True),
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
])

# ─── 公式 Lance Schema ─────────────────────────────────────────────────────

FORMULA_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("formula_latex", pa.large_string(), nullable=True),
    pa.field("formula_confidence", pa.float32(), nullable=True),
    pa.field("source_model", pa.large_string(), nullable=True),
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
])

# ─── 表格 Lance Schema ─────────────────────────────────────────────────────

TABLE_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("table_html", pa.large_string(), nullable=True),
    pa.field("table_json", pa.large_string(), nullable=True),
    pa.field("table_confidence", pa.float32(), nullable=True),
    pa.field("source_model", pa.large_string(), nullable=True),
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
])

# ─── 类型分组映射 ──────────────────────────────────────────────────────────

TEXT_BLOCK_TYPES = {"text", "title", "paragraph_title", "number", "header", "footer", "code_txt", "reference"}
FORMULA_BLOCK_TYPES = {"equation_isolated", "equation_inline", "formula"}
TABLE_BLOCK_TYPES = {"table"}
SKIP_BLOCK_TYPES = {"figure", "figure_caption", "figure_footnote", "abandon", "image"}

CATEGORY_SCHEMAS = {
    "text": TEXT_LANCE_SCHEMA,
    "formula": FORMULA_LANCE_SCHEMA,
    "table": TABLE_LANCE_SCHEMA,
}
```

### 2. 新建 `data_engine/ocr/split.py` — 分类写入逻辑

```python
"""将 element.lance 中的 block 按类型分拆到 text/formula/table 三个 Lance 数据集。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

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
            tbl = row.get(f"{model}_table_json")
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
) -> dict[str, int]:
    """读 element.lance，按 block_type 分拆到 text.lance / formula.lance / table.lance。

    Args:
        manifests_dir: 包含 element.lance 的目录
        image_map: sample_id -> image_bytes 映射（可选，用于裁剪 block 图片）

    Returns:
        {"text": N, "formula": M, "table": K, "skipped": S}
    """
    element_path = manifests_dir / "element.lance"
    if not element_path.exists():
        raise FileNotFoundError(f"element.lance 不存在: {element_path}")

    with _lance_write_lock:
        ds = lance.dataset(str(element_path))
        rows = ds.to_table().to_pylist()

    category_rows: dict[str, list[dict]] = {"text": [], "formula": [], "table": []}
    skipped = 0

    for row in rows:
        bt = row.get("block_type", "")
        cat = _classify_block(bt)
        if cat is None:
            skipped += 1
            continue

        img_data = None
        if image_map and row["sample_id"] in image_map:
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

    result = {"skipped": skipped}

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
```

### 3. 修改 `data_engine/manifests.py` — 添加分类 Lance 查找函数

在 `find_stage_manifest` 函数后添加：

```python
def find_category_manifest(manifests_dir: Path, category: str) -> Path | None:
    """查找分类 Lance（text/formula/table）"""
    path = manifests_dir / f"{category}.lance"
    if path.exists():
        return path
    return None
```

### 4. 修改 `data_engine/web_app.py` — 添加拆分 API

在 `layout-batch` 结果 API 后面添加：

```python
@app.post("/api/split-categories/{source_id}/{batch_id}")
async def split_categories(source_id: str, batch_id: str, with_images: bool = False):
    """将 element.lance 中的 block 按类型分拆到 text/formula/table Lance"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        element_path = manifests_dir / "element.lance"

        if not element_path.exists():
            raise HTTPException(status_code=404, detail="element.lance 不存在，请先运行 OCR")

        image_map: dict[str, bytes] = {}
        if with_images:
            ingest_path = find_stage_manifest(manifests_dir, "ingest")
            if ingest_path:
                with _lance_write_lock:
                    ds = lance.dataset(str(ingest_path))
                    recs = ds.to_table(columns=["sample_id", "image_data"]).to_pylist()
                for r in recs:
                    if r.get("image_data"):
                        image_map[r["sample_id"]] = r["image_data"]

        from data_engine.ocr.split import split_element_to_category_lances
        result = split_element_to_category_lances(manifests_dir, image_map if with_images else None)
        invalidate_status_cache()

        return {"source_id": source_id, "batch_id": batch_id, "result": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/category-stats/{source_id}/{batch_id}")
async def get_category_stats(source_id: str, batch_id: str):
    """获取 text/formula/table 三个 Lance 的统计"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        stats = {}
        for cat in ("text", "formula", "table"):
            lance_path = manifests_dir / f"{cat}.lance"
            if lance_path.exists():
                with _lance_write_lock:
                    ds = lance.dataset(str(lance_path))
                    n = ds.count_rows()
                    cols = ds.schema.names
                stats[cat] = {"count": n, "columns": cols}
            else:
                stats[cat] = {"count": 0, "columns": []}

        return {"source_id": source_id, "batch_id": batch_id, "categories": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 5. 修改 `data_engine/templates/data_filter.html` — 在「难度感知抽样」Tab 下添加拆分按钮

在 `diff-detail` div 后面添加：

```html
<div class="mt-4 border-t pt-4">
    <div class="flex items-center gap-4 mb-3">
        <button id="btn-split-categories" class="px-4 py-2 bg-teal-600 text-white rounded-md hover:bg-teal-700 font-medium text-sm">
            按类型拆分到 Lance
        </button>
        <label class="flex items-center gap-2 text-sm text-gray-600">
            <input type="checkbox" id="split-with-images" class="rounded">
            包含裁剪图片
        </label>
        <div id="split-status" class="text-sm text-gray-500"></div>
    </div>
    <div id="split-results" class="hidden">
        <div class="grid grid-cols-3 gap-3">
            <div class="bg-blue-50 border border-blue-200 rounded-lg p-3 text-center">
                <div class="text-xs text-blue-600">Text</div>
                <div id="split-text-count" class="text-xl font-bold text-blue-800">-</div>
            </div>
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-3 text-center">
                <div class="text-xs text-purple-600">Formula</div>
                <div id="split-formula-count" class="text-xl font-bold text-purple-800">-</div>
            </div>
            <div class="bg-orange-50 border border-orange-200 rounded-lg p-3 text-center">
                <div class="text-xs text-orange-600">Table</div>
                <div id="split-table-count" class="text-xl font-bold text-orange-800">-</div>
            </div>
        </div>
    </div>
</div>
```

JS 部分（在 `renderDiffSummary` 函数后面）：

```javascript
document.getElementById('btn-split-categories').addEventListener('click', function() {
    if (!bucketSampleData) { alert('请先抽样'); return; }
    const sourceId = bucketSampleData.source_id;
    const batchId = bucketSampleData.batch_id;
    const withImages = document.getElementById('split-with-images').checked;
    const btn = this;
    const status = document.getElementById('split-status');

    btn.disabled = true;
    btn.textContent = '拆分中...';
    status.textContent = '';

    // 逐批次拆分
    const targets = batchId ? [{source_id: sourceId, batch_id: batchId}] :
        (bucketSampleData.batch_info || []).map(b => ({source_id: b.source_id, batch_id: b.batch_id}));

    let done = 0;
    const totalStats = {text: 0, formula: 0, table: 0, skipped: 0};

    const promises = targets.map(t => {
        let url = `/api/split-categories/${t.source_id}/${t.batch_id}`;
        if (withImages) url += '?with_images=true';
        return fetch(url, {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                if (data.result) {
                    for (const [k, v] of Object.entries(data.result)) {
                        totalStats[k] = (totalStats[k] || 0) + v;
                    }
                }
                done++;
                status.textContent = `${done}/${targets.length} 批次完成`;
            })
            .catch(err => {
                done++;
                console.error('[split]', t.batch_id, err);
            });
    });

    Promise.all(promises).then(() => {
        btn.disabled = false;
        btn.textContent = '按类型拆分到 Lance';
        status.textContent = '完成';
        document.getElementById('split-results').classList.remove('hidden');
        document.getElementById('split-text-count').textContent = (totalStats.text || 0).toLocaleString();
        document.getElementById('split-formula-count').textContent = (totalStats.formula || 0).toLocaleString();
        document.getElementById('split-table-count').textContent = (totalStats.table || 0).toLocaleString();
    });
});
```

### 6. 修改 LanceData 页面 — 在批次下拉中显示 text/formula/table

修改 `data_engine/web_app.py` 的 `lancedb_list_sources` 函数，除了 `ingest` 和 `element` 外，也查找 `text.lance`、`formula.lance`、`table.lance`：

在 `for dataset_name in ["ingest", "element"]:` 改为：
```python
for dataset_name in ["ingest", "element", "text", "formula", "table"]:
```

### 7. 存储路径

```
manifests/
├── ingest.lance      # 页面级
├── element.lance     # block 级（三模型 OCR）
├── text.lance        # 文本类 block
├── formula.lance     # 公式类 block
└── table.lance       # 表格类 block
```

### 8. 数据流

```
抽样样本 → pp-layout 检测 block → 三模型 OCR → element.lance
                                                    ↓
                                        split-categories API
                                                    ↓
                                    ┌───────────────┼───────────────┐
                                    ↓               ↓               ↓
                              text.lance      formula.lance    table.lance
                            (text_content)   (formula_latex)  (table_html)
```

每个 Lance 的字段：
- **text.lance**: sample_id, block_idx, block_type, bbox_json, image_data, text_content, text_confidence, source_model, consistency_pattern
- **formula.lance**: sample_id, block_idx, block_type, bbox_json, image_data, formula_latex, formula_confidence, source_model, consistency_pattern
- **table.lance**: sample_id, block_idx, block_type, bbox_json, image_data, table_html, table_json, table_confidence, source_model, consistency_pattern

每个 Lance 都有 `image_data` 字段（可选），勾选「包含裁剪图片」时会从 ingest.lance 读整页图并按 bbox 裁剪。
