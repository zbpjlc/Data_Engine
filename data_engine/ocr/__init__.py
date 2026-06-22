from __future__ import annotations

import pyarrow as pa

from data_engine.ocr.base import LayoutBlock, OCRResult, BaseOCREngine
from data_engine.ocr.cmcv import CMCVEngine

# ─── Judge-Refine 公共列定义 ──────────────────────────────────────────────────

JUDGE_REFINE_COLUMNS = [
    pa.field("judged", pa.bool_(), nullable=True),
    pa.field("corrected", pa.bool_(), nullable=True),
    pa.field("needs_expert", pa.bool_(), nullable=True),
    pa.field("judge_refined_text", pa.large_string(), nullable=True),
    pa.field("judge_refined_table", pa.large_string(), nullable=True),
    pa.field("judge_refined_formula", pa.large_string(), nullable=True),
    pa.field("judge_confidence", pa.float32(), nullable=True),
    pa.field("judge_rounds", pa.int32(), nullable=True),
    pa.field("judge_error_locations", pa.large_string(), nullable=True),
]

# ─── 文本 Lance Schema ─────────────────────────────────────────────────────

TEXT_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("embedding", pa.list_(pa.float32(), 768), nullable=True),
    # 多模型 OCR 结果
    pa.field("paddle_text", pa.large_string(), nullable=True),
    pa.field("paddle_confidence", pa.float32(), nullable=True),
    pa.field("glm_text", pa.large_string(), nullable=True),
    pa.field("glm_confidence", pa.float32(), nullable=True),
    pa.field("self_text", pa.large_string(), nullable=True),
    pa.field("self_confidence", pa.float32(), nullable=True),
    # CMCV 结果
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("block_diff_json", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
] + JUDGE_REFINE_COLUMNS)

# ─── 公式 Lance Schema ─────────────────────────────────────────────────────

FORMULA_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("embedding", pa.list_(pa.float32(), 768), nullable=True),
    # 多模型 OCR 结果
    pa.field("paddle_formula", pa.large_string(), nullable=True),
    pa.field("paddle_confidence", pa.float32(), nullable=True),
    pa.field("glm_formula", pa.large_string(), nullable=True),
    pa.field("glm_confidence", pa.float32(), nullable=True),
    pa.field("self_formula", pa.large_string(), nullable=True),
    pa.field("self_confidence", pa.float32(), nullable=True),
    # CMCV 结果
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("block_diff_json", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
] + JUDGE_REFINE_COLUMNS)

# ─── 表格 Lance Schema ─────────────────────────────────────────────────────

TABLE_LANCE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.large_string(), nullable=False),
    pa.field("block_idx", pa.int32(), nullable=False),
    pa.field("block_type", pa.large_string(), nullable=False),
    pa.field("bbox_json", pa.large_string(), nullable=False),
    pa.field("layout_confidence", pa.float32(), nullable=False),
    pa.field("image_data", pa.large_binary(), nullable=True),
    pa.field("embedding", pa.list_(pa.float32(), 768), nullable=True),
    # 多模型 OCR 结果
    pa.field("paddle_table", pa.large_string(), nullable=True),
    pa.field("paddle_confidence", pa.float32(), nullable=True),
    pa.field("glm_table", pa.large_string(), nullable=True),
    pa.field("glm_confidence", pa.float32(), nullable=True),
    pa.field("self_table", pa.large_string(), nullable=True),
    pa.field("self_confidence", pa.float32(), nullable=True),
    # CMCV 结果
    pa.field("consistency_pattern", pa.large_string(), nullable=True),
    pa.field("block_diff_json", pa.large_string(), nullable=True),
    pa.field("schema_version", pa.large_string(), nullable=False),
    pa.field("created_at", pa.large_string(), nullable=True),
] + JUDGE_REFINE_COLUMNS)

# ─── 类型分组映射 ──────────────────────────────────────────────────────────

TEXT_BLOCK_TYPES = {"text", "title", "paragraph_title", "number", "header", "footer", "code_txt", "reference"}
FORMULA_BLOCK_TYPES = {"equation_isolated", "equation_inline", "formula",
                       "display_formula", "formula_number", "inline_formula"}
TABLE_BLOCK_TYPES = {"table"}
SKIP_BLOCK_TYPES = {"figure", "figure_caption", "figure_footnote", "abandon", "image"}

CATEGORY_SCHEMAS = {
    "text": TEXT_LANCE_SCHEMA,
    "formula": FORMULA_LANCE_SCHEMA,
    "table": TABLE_LANCE_SCHEMA,
}
