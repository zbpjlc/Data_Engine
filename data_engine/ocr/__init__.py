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
