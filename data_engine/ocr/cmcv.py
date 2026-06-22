from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from data_engine.config import get_config

logger = logging.getLogger(__name__)


# ─── Levenshtein（优先用 OmniDocBench 的 C 实现） ────────────────────────────

try:
    import Levenshtein as _Lev

    def text_similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        return 1.0 - _Lev.distance(a, b) / max_len
except ImportError:
    def text_similarity(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        # Python fallback
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, c1 in enumerate(a):
            curr = [i + 1]
            for j, c2 in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1 != c2)))
            prev = curr
        return 1.0 - prev[-1] / max_len


# ─── TEDS（OmniDocBench 实现） ──────────────────────────────────────────────

try:
    from src.metrics.table_metric import TEDS as _OmniTEDS
    _teds_engine = _OmniTEDS()

    def _table_to_full_html(table: dict | str | None) -> str:
        if not table:
            return ""
        if isinstance(table, str):
            if "<table" in table.lower():
                if "<html" not in table.lower():
                    return f"<html><body>{table}</body></html>"
                return table
            try:
                table = json.loads(table)
            except (json.JSONDecodeError, TypeError):
                return f"<html><body><table><tr><td>{table}</td></tr></table></body></html>"
        rows = table.get("rows", table.get("data", []))
        if not rows:
            return ""
        parts = []
        for row in rows:
            if isinstance(row, list):
                cells = "".join(f"<td>{c}</td>" for c in row)
            elif isinstance(row, dict):
                cells = "".join(f"<td>{v}</td>" for v in row.values())
            else:
                cells = f"<td>{row}</td>"
            parts.append(f"<tr>{cells}</tr>")
        return "<html><body><table>" + "".join(parts) + "</table></body></html>"

    def teds_similarity(table_a: dict | str | None, table_b: dict | str | None) -> float:
        html_a = _table_to_full_html(table_a)
        html_b = _table_to_full_html(table_b)
        if not html_a and not html_b:
            return 1.0
        if not html_a or not html_b:
            return 0.0
        return _teds_engine.evaluate(html_a, html_b)

except ImportError:
    logger.warning("OmniDocBench TEDS not available, using fallback table comparison")

    def _table_to_cells(table: dict | str | None) -> list[list[str]]:
        if not table:
            return []
        if isinstance(table, str):
            try:
                table = json.loads(table)
            except (json.JSONDecodeError, TypeError):
                return [[table]]
        if isinstance(table, dict):
            rows = table.get("rows", table.get("data", table.get("cells", [])))
        elif isinstance(table, list):
            rows = table
        else:
            return [[str(table)]]
        result = []
        for row in rows:
            if isinstance(row, list):
                result.append([str(c).strip() for c in row])
            elif isinstance(row, dict):
                result.append([str(v).strip() for v in row.values()])
            else:
                result.append([str(row).strip()])
        return result

    def teds_similarity(table_a: dict | str | None, table_b: dict | str | None) -> float:
        cells_a = _table_to_cells(table_a)
        cells_b = _table_to_cells(table_b)
        if not cells_a and not cells_b:
            return 1.0
        if not cells_a or not cells_b:
            return 0.0
        max_rows = max(len(cells_a), len(cells_b))
        if max_rows == 0:
            return 1.0
        total_cells = 0
        match_cells = 0.0
        for i in range(max_rows):
            row_a = cells_a[i] if i < len(cells_a) else []
            row_b = cells_b[i] if i < len(cells_b) else []
            max_cols = max(len(row_a), len(row_b))
            for j in range(max_cols):
                total_cells += 1
                val_a = row_a[j] if j < len(row_a) else ""
                val_b = row_b[j] if j < len(row_b) else ""
                if val_a == val_b:
                    match_cells += 1.0
                else:
                    match_cells += text_similarity(val_a, val_b)
        return match_cells / total_cells if total_cells > 0 else 0.0


# ─── CDM（Character Detection Matching） ─────────────────────────────────────

# 尝试加载 OmniDocBench CDM（需要 TeX Live + ImageMagick）
_CDM_ENGINE = None
try:
    from src.metrics.cdm_metric import CDM as _OmniCDM
    # 测试 pdflatex 是否可用
    import subprocess as _sp
    _test = _sp.run(["pdflatex", "--version"], capture_output=True, timeout=5)
    if _test.returncode == 0:
        _CDM_ENGINE = _OmniCDM
        logger.info("CDM: using OmniDocBench (pdflatex available)")
    else:
        logger.warning("CDM: pdflatex not working, using token fallback")
except Exception:
    logger.warning("CDM: OmniDocBench not available, using token fallback")

_LATEX_CMD_NORMALIZE = [
    (r"\\left\s*\(", "("),
    (r"\\right\s*\)", ")"),
    (r"\\left\s*\[", "["),
    (r"\\right\s*\]", "]"),
    (r"\\left\s*\\{", "{"),
    (r"\\right\s*\\}", "}"),
    (r"\\,", " "),
    (r"\\;", " "),
    (r"\\!", ""),
    (r"\\quad", " "),
    (r"\\qquad", " "),
    (r"\\text\s*\{([^}]*)\}", r"\1"),
    (r"\\mathrm\s*\{([^}]*)\}", r"\1"),
    (r"\\operatorname\s*\{([^}]*)\}", r"\1"),
]

_LATEX_SPACES = re.compile(r"\s+")


def _normalize_latex(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    for pattern, repl in _LATEX_CMD_NORMALIZE:
        s = re.sub(pattern, repl, s)
    s = _LATEX_SPACES.sub(" ", s).strip()
    return s


def _tokenize_latex(s: str) -> list[str]:
    return re.findall(r"\\[a-zA-Z]+|[0-9]+\.?[0-9]*|[a-zA-Z]|[^\s]", s)


def cdm_similarity(formula_a: str, formula_b: str) -> float:
    """CDM: 优先用 OmniDocBench 渲染比较，不可用时用 token 匹配"""
    if _CDM_ENGINE is not None:
        try:
            cdm = _CDM_ENGINE(output_root="/tmp/cdm_eval")
            result = cdm.evaluate(formula_a or "", formula_b or "", "inline")
            return float(result.get("F1_score", 0.0))
        except Exception as exc:
            logger.debug("CDM OmniDocBench failed, fallback: %s", exc)

    # Token 匹配 fallback
    na = _normalize_latex(formula_a)
    nb = _normalize_latex(formula_b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0

    tokens_a = _tokenize_latex(na)
    tokens_b = _tokenize_latex(nb)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    if not intersection:
        return 0.0
    precision = len(intersection) / len(set_b)
    recall = len(intersection) / len(set_a)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    pos_match = 0
    idx_b = 0
    for tok in tokens_a:
        for j in range(idx_b, len(tokens_b)):
            if tokens_b[j] == tok:
                pos_match += 1
                idx_b = j + 1
                break
    pos_ratio = pos_match / max(len(tokens_a), len(tokens_b))

    return 0.5 * f1 + 0.5 * pos_ratio


# ─── 标准化层 ────────────────────────────────────────────────────────────────

def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_table(table: dict | None) -> dict | None:
    if not table:
        return None
    return table


def normalize_formula(formula: str | None) -> str:
    return _normalize_latex(formula or "")


# ─── Block 级比较 ────────────────────────────────────────────────────────────

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
        sim_fn = teds_similarity
        a_val = normalize_table(paddle_table)
        b_val = normalize_table(glm_table)
        c_val = normalize_table(self_table)
        thr = get_config("ocr", "cmcv", "table_threshold", default=None) or threshold
        method = "teds"
    elif block_type == "formula":
        sim_fn = cdm_similarity
        a_val = normalize_formula(paddle_formula)
        b_val = normalize_formula(glm_formula)
        c_val = normalize_formula(self_formula)
        thr = get_config("ocr", "cmcv", "formula_threshold", default=None) or threshold
        method = "cdm"
    else:
        sim_fn = text_similarity
        a_val = normalize_text(paddle_text)
        b_val = normalize_text(glm_text)
        c_val = normalize_text(self_text)
        thr = get_config("ocr", "cmcv", "text_threshold", default=None) or threshold
        method = "levenshtein"

    sim_pg = sim_fn(a_val, b_val)
    sim_ps = sim_fn(a_val, c_val)
    sim_gs = sim_fn(b_val, c_val)

    diff_detail = {
        "method": method,
        "sim_paddle_glm": round(sim_pg, 4),
        "sim_paddle_self": round(sim_ps, 4),
        "sim_glm_self": round(sim_gs, 4),
        "threshold": thr,
    }

    # Plan §6.6:
    # easy   — paddle 和 glm 一致，且 self 也一致
    # medium — paddle 和 glm 一致，但 self 不一致
    # hard   — paddle 和 glm 不一致
    external_agree = sim_pg >= thr  # paddle 与 glm 一致
    self_agree_with_external = sim_ps >= thr and sim_gs >= thr  # self 与两者都一致

    if external_agree and self_agree_with_external:
        pattern = "all_agree"
    elif external_agree:
        pattern = "partial_agree"
    else:
        pattern = "all_disagree"

    return pattern, diff_detail


# ─── CMCV 引擎 ──────────────────────────────────────────────────────────────

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
                paddle_table=block.get("paddle_table"),
                glm_table=block.get("glm_table"),
                self_table=block.get("self_table"),
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

        return {
            "block_count": len(blocks),
            "all_agree_count": patterns.count("all_agree"),
            "partial_agree_count": patterns.count("partial_agree"),
            "all_disagree_count": patterns.count("all_disagree"),
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
                # 只保留 key 列 + CMCV 结果列，避免写入磁盘 schema 中不存在的列
                updated_rows.append({
                    "sample_id": block["sample_id"],
                    "block_idx": block.get("block_idx", 0),
                    "consistency_pattern": detail["pattern"],
                    "block_diff_json": json.dumps(detail["diff"], ensure_ascii=False) if detail.get("diff") else None,
                })

            page_tiers[sample_id] = page_result["tier"]

        return updated_rows, page_tiers
