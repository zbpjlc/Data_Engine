from __future__ import annotations

from data_engine.ocr.base import LayoutBlock, OCRResult


def results_to_element_rows(
    sample_id: str,
    blocks: list[LayoutBlock],
    engine_results: dict[str, list[OCRResult]],
    schema_version: str = "v1",
    created_at: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for idx, block in enumerate(blocks):
        row: dict = {
            "sample_id": sample_id,
            "block_idx": idx,
            "block_type": block.block_type,
            "bbox_json": block.bbox,
            "layout_confidence": block.confidence,
            "schema_version": schema_version,
            "created_at": created_at,
        }
        for prefix, results in engine_results.items():
            if idx < len(results):
                r = results[idx]
                row[f"{prefix}_text"] = r.text_content or None
                row[f"{prefix}_confidence"] = r.confidence or None
                row[f"{prefix}_table_json"] = r.table_structure
                row[f"{prefix}_formula"] = r.formula_latex or None
                row[f"{prefix}_raw_json"] = r.raw_output or None
            else:
                row[f"{prefix}_text"] = None
                row[f"{prefix}_confidence"] = None
                row[f"{prefix}_table_json"] = None
                row[f"{prefix}_formula"] = None
                row[f"{prefix}_raw_json"] = None
        rows.append(row)
    return rows


def ocr_results_to_ingest_fields(results: list[OCRResult]) -> dict:
    block_list: list[dict] = []
    text_spans: list[dict] = []
    table_structure: dict | None = None
    formula_spans: list[dict] = []

    for idx, r in enumerate(results):
        block_entry: dict = {
            "block_idx": idx,
            "block_type": r.block_type,
            "bbox": r.bbox,
            "confidence": r.confidence,
        }
        if r.text_content:
            block_entry["text"] = r.text_content
            text_spans.append({
                "block_idx": idx,
                "text": r.text_content,
                "confidence": r.confidence,
            })
        if r.table_structure:
            block_entry["table"] = r.table_structure
            table_structure = r.table_structure
        if r.formula_latex:
            block_entry["formula"] = r.formula_latex
            formula_spans.append({
                "block_idx": idx,
                "latex": r.formula_latex,
            })
        block_list.append(block_entry)

    reading_order = list(range(len(results)))

    return {
        "block_list": block_list,
        "reading_order": reading_order,
        "text_spans": text_spans,
        "table_structure": table_structure,
        "formula_spans": formula_spans,
    }
