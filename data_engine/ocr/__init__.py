from __future__ import annotations

import pyarrow as pa

ELEMENT_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),

    pa.field("paddle_text", pa.large_string(), nullable=True),
    pa.field("paddle_confidence", pa.float32(), nullable=True),
    pa.field("paddle_table_json", pa.large_string(), nullable=True),
    pa.field("paddle_formula", pa.large_string(), nullable=True),
    pa.field("paddle_raw_json", pa.large_string(), nullable=True),

    pa.field("glm_text", pa.large_string(), nullable=True),
    pa.field("glm_confidence", pa.float32(), nullable=True),
    pa.field("glm_table_json", pa.large_string(), nullable=True),
    pa.field("glm_formula", pa.large_string(), nullable=True),
    pa.field("glm_raw_json", pa.large_string(), nullable=True),

    pa.field("self_text", pa.large_string(), nullable=True),
    pa.field("self_confidence", pa.float32(), nullable=True),
    pa.field("self_table_json", pa.large_string(), nullable=True),
    pa.field("self_formula", pa.large_string(), nullable=True),
    pa.field("self_raw_json", pa.large_string(), nullable=True),

    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("block_diff_json", pa.large_string(), nullable=True),

    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
])

ELEMENT_JSON_FIELDS = {
    "bbox_json", "paddle_table_json", "paddle_raw_json",
    "glm_table_json", "glm_raw_json",
    "self_table_json", "self_raw_json",
    "block_diff_json",
}

from data_engine.ocr.base import LayoutBlock, OCRResult, BaseOCREngine
from data_engine.ocr.cmcv import CMCVEngine

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
