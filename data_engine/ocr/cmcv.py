from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

from data_engine.config import get_config

logger = logging.getLogger(__name__)

_TEXT_SIM_THRESHOLD = 0.9
_TABLE_SIM_THRESHOLD = 0.85
_FORMULA_SIM_THRESHOLD = 0.85


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = _levenshtein_distance(a, b)
    return 1.0 - dist / max_len


def _flatten_table_to_html(table: dict | None) -> str:
    if not table:
        return ""
    if isinstance(table, str):
        return table
    rows = table.get("rows", table.get("data", []))
    if not rows:
        return str(table)
    parts: list[str] = []
    for row in rows:
        if isinstance(row, list):
            cells = "".join(f"<td>{c}</td>" for c in row)
        elif isinstance(row, dict):
            cells = "".join(f"<td>{v}</td>" for v in row.values())
        else:
            cells = f"<td>{row}</td>"
        parts.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(parts) + "</table>"


def _tree_edit_distance(html_a: str, html_b: str) -> int:
    return _levenshtein_distance(html_a, html_b)


def table_similarity(table_a: dict | None, table_b: dict | None) -> float:
    html_a = _flatten_table_to_html(table_a)
    html_b = _flatten_table_to_html(table_b)
    if not html_a and not html_b:
        return 1.0
    if not html_a or not html_b:
        return 0.0
    max_len = max(len(html_a), len(html_b))
    if max_len == 0:
        return 1.0
    dist = _tree_edit_distance(html_a, html_b)
    return max(0.0, 1.0 - dist / max_len)


def formula_similarity(a: str, b: str) -> float:
    return text_similarity(a, b)


def compare_block(
    paddle_text: str | None,
    glm_text: str | None,
    self_text: str | None,
    block_type: str,
    paddle_table: dict | None = None,
    glm_table: dict | None = None,
    self_table: dict | None = None,
    paddle_formula: str | None = None,
    glm_formula: str | None = None,
    self_formula: str | None = None,
    agreement_threshold: float | None = None,
) -> tuple[str, dict]:
    threshold = agreement_threshold or float(
        get_config("ocr", "cmcv", "agreement_threshold", default=0.9)
    )

    if block_type == "table":
        sim_fn = table_similarity
        a_val, b_val, c_val = paddle_table, glm_table, self_table
        thr = get_config("ocr", "cmcv", "table_threshold", default=None) or threshold
    elif block_type == "formula":
        sim_fn = formula_similarity
        a_val = paddle_formula or ""
        b_val = glm_formula or ""
        c_val = self_formula or ""
        thr = get_config("ocr", "cmcv", "formula_threshold", default=None) or threshold
    else:
        sim_fn = text_similarity
        a_val = paddle_text or ""
        b_val = glm_text or ""
        c_val = self_text or ""
        thr = get_config("ocr", "cmcv", "text_threshold", default=None) or threshold

    sim_pg = sim_fn(a_val, b_val)  # type: ignore[arg-type]
    sim_ps = sim_fn(a_val, c_val)  # type: ignore[arg-type]
    sim_gs = sim_fn(b_val, c_val)  # type: ignore[arg-type]

    diff_detail = {
        "sim_paddle_glm": round(sim_pg, 4),
        "sim_paddle_self": round(sim_ps, 4),
        "sim_glm_self": round(sim_gs, 4),
        "threshold": thr,
    }

    if sim_pg >= thr and sim_ps >= thr and sim_gs >= thr:
        pattern = "all_agree"
    elif sim_pg >= thr and (sim_ps < thr or sim_gs < thr):
        pattern = "partial_agree"
    else:
        pattern = "all_disagree"

    return pattern, diff_detail


class CMCVEngine:

    def __init__(self) -> None:
        self._threshold = float(
            get_config("ocr", "cmcv", "agreement_threshold", default=0.9)
        )

    def compare_page(self, blocks: list[dict]) -> dict:
        details: list[dict] = []
        patterns: list[str] = []

        for block in blocks:
            pattern, diff = compare_block(
                paddle_text=block.get("paddle_text"),
                glm_text=block.get("glm_text"),
                self_text=block.get("self_text"),
                block_type=block.get("block_type", "text"),
                paddle_table=block.get("paddle_table_json"),
                glm_table=block.get("glm_table_json"),
                self_table=block.get("self_table_json"),
                paddle_formula=block.get("paddle_formula"),
                glm_formula=block.get("glm_formula"),
                self_formula=block.get("self_formula"),
                agreement_threshold=self._threshold,
            )
            details.append({
                "block_idx": block.get("block_idx", 0),
                "type": block.get("block_type", "text"),
                "pattern": pattern,
                "diff": diff,
            })
            patterns.append(pattern)

        tier = self._assign_tier(patterns)
        all_agree_count = patterns.count("all_agree")
        partial_agree_count = patterns.count("partial_agree")
        all_disagree_count = patterns.count("all_disagree")

        return {
            "block_count": len(blocks),
            "all_agree_count": all_agree_count,
            "partial_agree_count": partial_agree_count,
            "all_disagree_count": all_disagree_count,
            "tier": tier,
            "details": details,
        }

    @staticmethod
    def _assign_tier(block_patterns: list[str]) -> str:
        if "all_disagree" in block_patterns:
            return "hard"
        if "partial_agree" in block_patterns:
            return "medium"
        return "easy"

    def process_element_batch(self, element_rows: list[dict]) -> tuple[list[dict], dict[str, str]]:
        by_sample: dict[str, list[dict]] = defaultdict(list)
        for row in element_rows:
            by_sample[row["sample_id"]].append(row)

        updated_rows: list[dict] = []
        page_tiers: dict[str, str] = {}

        for sample_id, blocks in by_sample.items():
            blocks_sorted = sorted(blocks, key=lambda b: b.get("block_idx", 0))
            page_result = self.compare_page(blocks_sorted)

            for block, detail in zip(blocks_sorted, page_result["details"]):
                block["consistency_pattern"] = detail["pattern"]
                block["block_diff_json"] = detail["diff"]
                updated_rows.append(block)

            page_tiers[sample_id] = page_result["tier"]

        return updated_rows, page_tiers
