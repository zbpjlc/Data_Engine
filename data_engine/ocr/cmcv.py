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
    from data_engine.ocr.omnidocbench_local.table_metric import TEDS as _OmniTEDS
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

# CDM 视觉渲染引擎：Page CMCV 默认用 token fallback（快），
# Element CMCV 按需启用视觉渲染（准）。引擎统一初始化，由调用方选择是否使用。
_CDM_ENGINE = None

try:
    from data_engine.ocr.omnidocbench_local.cdm_metric import CDM as _OmniCDM
    import subprocess as _sp
    
    # 测试 pdflatex 是否可用
    _test = _sp.run(["pdflatex", "--version"], capture_output=True, timeout=5)
    if _test.returncode == 0:
        # 测试 ImageMagick 是否可用
        _magick_test = _sp.run(["magick", "--version"], capture_output=True, timeout=5)
        if _magick_test.returncode == 0:
            _CDM_ENGINE = _OmniCDM
            logger.info("✅ CDM: 视觉渲染引擎已就绪 (pdflatex + ImageMagick)")
        else:
            logger.warning("⚠️ CDM: ImageMagick 不可用，视觉渲染降级到 token fallback")
    else:
        logger.warning("⚠️ CDM: pdflatex 不可用，视觉渲染降级到 token fallback")
        
except ImportError as e:
    logger.warning(f"⚠️ CDM: OmniDocBench 未安装 ({e})，视觉渲染降级到 token fallback")
except Exception as e:
    logger.warning(f"⚠️ CDM: 初始化失败 ({e})，视觉渲染降级到 token fallback")

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
    # 合并单字符花括号 {x} -> x，避免花括号干扰比较
    # 多次执行以处理嵌套 {a{b}} -> {ab} -> ab
    for _ in range(3):
        new_s = re.sub(r"\{([^{}]+)\}", r"\1", s)
        if new_s == s:
            break
        s = new_s
    s = _LATEX_SPACES.sub(" ", s).strip()
    return s


def _tokenize_latex(s: str) -> list[str]:
    # 归一化阶段已合并单字符花括号，这里直接分词
    return re.findall(r"\\[a-zA-Z]+|[0-9]+\.?[0-9]*|[a-zA-Z]|[^\s]", s)


def _cdm_token_similarity(formula_a: str, formula_b: str) -> float:
    """Token 级别的 LaTeX 公式相似度（快速，用于 Page CMCV 批量分类）
    
    综合三个维度：
    - F1（token 集合，无序，权重 0.25）
    - 位置匹配率（token 顺序，权重 0.25）
    - Levenshtein 字符串相似度（整体结构，权重 0.5）
    """
    na = _normalize_latex(formula_a)
    nb = _normalize_latex(formula_b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    tokens_a = _tokenize_latex(na)
    tokens_b = _tokenize_latex(nb)
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    # 维度 1: F1（token 集合相似度，无序）
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b
    if intersection:
        precision = len(intersection) / len(set_b)
        recall = len(intersection) / len(set_a)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = 0.0

    # 维度 2: 位置匹配率（token 顺序，贪心匹配）
    pos_match = 0
    idx_b = 0
    for tok in tokens_a:
        for j in range(idx_b, len(tokens_b)):
            if tokens_b[j] == tok:
                pos_match += 1
                idx_b = j + 1
                break
    pos_ratio = pos_match / max(len(tokens_a), len(tokens_b))

    # 维度 3: Levenshtein 字符串相似度（整体结构，考虑顺序）
    lev = _text_lev_ratio(na, nb)

    return 0.25 * f1 + 0.25 * pos_ratio + 0.5 * lev


def _text_lev_ratio(a: str, b: str) -> float:
    """Levenshtein 相似度比例 (0-1)，优先用 C 实现"""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    try:
        import Levenshtein
        return Levenshtein.ratio(a, b)
    except ImportError:
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        # 纯 Python fallback
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (0 if a[i-1] == b[j-1] else 1))
                prev = tmp
        dist = dp[n]
        return 1.0 - dist / max_len


def _cdm_visual_similarity(formula_a: str, formula_b: str) -> float | None:
    """OmniDocBench 视觉渲染公式相似度（精度高但慢，用于 Element CMCV）
    
    仅走容器 bridge；失败返回 None，由调用方降级到 token similarity。
    
    Returns:
        相似度得分 (0-1)，失败返回 None
    """
    if _normalize_latex(formula_a) == _normalize_latex(formula_b):
        return 1.0
    try:
        from data_engine.ocr.omnidocbench_local.cdm_bridge import compute_cdm_visual
        container_score = compute_cdm_visual(formula_a, formula_b)
        if container_score is not None:
            return container_score.score
    except Exception as exc:
        logger.debug(f"CDM 容器执行失败: {exc}")
    return None


def _cdm_visual_similarity_batch(formula_pairs: list[tuple[str, str]]) -> list[float] | None:
    """批量 CDM 视觉渲染，减少 docker exec 开销。"""
    if not formula_pairs:
        return []
    try:
        from data_engine.ocr.omnidocbench_local.cdm_bridge import compute_cdm_visual_batch
        batch_results = compute_cdm_visual_batch(formula_pairs)
        if batch_results is not None:
            return [r.score for r in batch_results]
    except Exception as exc:
        logger.debug(f"CDM 批量容器执行失败: {exc}")
    return None


def cdm_similarity(formula_a: str, formula_b: str) -> float:
    """CDM 默认：token fallback（快速，用于 Page CMCV 批量分类）"""
    return _cdm_token_similarity(formula_a, formula_b)


def cdm_similarity_visual(formula_a: str, formula_b: str) -> float | None:
    """CDM 视觉渲染（精度高，用于 Element CMCV）
    
    仅使用 OmniDocBench 视觉渲染；失败返回 None。
    """
    return _cdm_visual_similarity(formula_a, formula_b)


# ─── 标准化层 ────────────────────────────────────────────────────────────────

def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    # 全角 → 半角（ASCII 范围 0xFF01-0xFF5E → 0x0021-0x007E）
    text = "".join(chr(ord(c) - 0xFEE0) if "\uff01" <= c <= "\uff5e" else c for c in text)
    # 全角空格 → 半角空格
    text = text.replace("\u3000", " ")
    # 去掉标点符号周围的空格
    text = re.sub(r"\s+([,\.\/\;\:\!\?\-\+\=\(\)\[\]\{\}])", r"\1", text)
    text = re.sub(r"([,\.\/\;\:\!\?\-\+\=\(\)\[\]\{\}])\s+", r"\1", text)
    # 合并空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    use_visual_cdm: bool = False,
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
        sim_fn = cdm_similarity_visual if use_visual_cdm else cdm_similarity
        a_val = normalize_formula(paddle_formula)
        b_val = normalize_formula(glm_formula)
        c_val = normalize_formula(self_formula)
        thr = get_config("ocr", "cmcv", "formula_threshold", default=None) or threshold
        method = "cdm_visual" if use_visual_cdm else "cdm_token"
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

    if sim_pg is None or sim_ps is None or sim_gs is None:
        diff_detail = {
            "method": method,
            "sim_paddle_glm": None,
            "sim_paddle_self": None,
            "sim_glm_self": None,
            "threshold": thr,
            "error": "cdm_visual_unavailable",
        }
        return "all_disagree", diff_detail

    diff_detail = {
        "method": method,
        "sim_paddle_glm": round(sim_pg, 4),
        "sim_paddle_self": round(sim_ps, 4),
        "sim_glm_self": round(sim_gs, 4),
        "threshold": thr,
    }

    # Plan §6.6:
    # easy   — paddle 和 glm 一致，且 self 与其中至少一个一致
    # medium — paddle 和 glm 一致，但 self 与两者都不一致
    # hard   — paddle 和 glm 不一致
    external_agree = sim_pg >= thr  # paddle 与 glm 一致
    self_agree_with_any = sim_ps >= thr or sim_gs >= thr  # self 与至少一个一致

    if external_agree and self_agree_with_any:
        pattern = "all_agree"
    elif external_agree:
        pattern = "partial_agree"
    else:
        pattern = "all_disagree"

    return pattern, diff_detail


# ─── CMCV 引擎 ──────────────────────────────────────────────────────────────

class CMCVEngine:

    def __init__(self, use_visual_cdm: bool = False) -> None:
        """
        Args:
            use_visual_cdm: 是否使用 OmniDocBench 视觉渲染评估公式相似度
                - False（默认）: token fallback，快，用于 Page CMCV
                - True: 视觉渲染，准但慢，用于 Element CMCV
        """
        self._threshold = float(
            get_config("ocr", "cmcv", "agreement_threshold", default=0.9)
        )
        self._use_visual_cdm = use_visual_cdm

    def compare_page(self, blocks: list[dict]) -> dict:
        details: list[dict] = []
        patterns: list[str] = []
        formula_blocks: list[dict] = []
        formula_indices: list[int] = []

        for idx, block in enumerate(blocks):
            block_type = block.get("block_type", "text")
            if block_type == "formula" and self._use_visual_cdm:
                formula_blocks.append(block)
                formula_indices.append(idx)
                details.append({
                    "block_idx": block.get("block_idx", 0),
                    "type": block_type,
                    "pattern": "",
                    "diff": {},
                })
                patterns.append("")
                continue

            pattern, diff = compare_block(
                paddle_text=block.get("paddle_text"),
                glm_text=block.get("glm_text"),
                self_text=block.get("self_text"),
                block_type=block_type,
                paddle_table=block.get("paddle_table"),
                glm_table=block.get("glm_table"),
                self_table=block.get("self_table"),
                paddle_formula=block.get("paddle_formula"),
                glm_formula=block.get("glm_formula"),
                self_formula=block.get("self_formula"),
                agreement_threshold=self._threshold,
                use_visual_cdm=self._use_visual_cdm,
            )
            details.append({
                "block_idx": block.get("block_idx", 0),
                "type": block_type,
                "pattern": pattern,
                "diff": diff,
            })
            patterns.append(pattern)

        if formula_blocks and self._use_visual_cdm:
            formula_pairs: list[tuple[str, str]] = []
            for block in formula_blocks:
                a = normalize_formula(block.get("paddle_formula")) or ""
                b = normalize_formula(block.get("glm_formula")) or ""
                c = normalize_formula(block.get("self_formula")) or ""
                formula_pairs.extend([(a, b), (a, c), (b, c)])

            batch_scores = _cdm_visual_similarity_batch(formula_pairs)
            if batch_scores is not None:
                for offset, idx in enumerate(formula_indices):
                    block = formula_blocks[offset]
                    sim_pg = batch_scores[offset * 3]
                    sim_ps = batch_scores[offset * 3 + 1]
                    sim_gs = batch_scores[offset * 3 + 2]

                    thr = get_config("ocr", "cmcv", "formula_threshold", default=None) or self._threshold
                    external_agree = sim_pg >= thr
                    self_agree_with_external = sim_ps >= thr and sim_gs >= thr
                    if external_agree and self_agree_with_external:
                        pattern = "all_agree"
                    elif external_agree:
                        pattern = "partial_agree"
                    else:
                        pattern = "all_disagree"

                    patterns[idx] = pattern
                    details[idx] = {
                        "block_idx": block.get("block_idx", 0),
                        "type": block.get("block_type", "formula"),
                        "pattern": pattern,
                        "diff": {
                            "method": "cdm_visual_batch",
                            "sim_paddle_glm": round(sim_pg, 4),
                            "sim_paddle_self": round(sim_ps, 4),
                            "sim_glm_self": round(sim_gs, 4),
                            "threshold": thr,
                        },
                    }

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

    def process_element_batch(self, element_rows: list[dict], progress_callback=None) -> tuple[list[dict], dict[str, str]]:
        by_sample: dict[str, list[dict]] = defaultdict(list)
        for row in element_rows:
            by_sample[row["sample_id"]].append(row)

        total_pages = len(by_sample)
        processed_pages = 0
        updated_rows: list[dict] = []
        page_tiers: dict[str, str] = {}

        for sample_id, blocks in by_sample.items():
            blocks_sorted = sorted(blocks, key=lambda b: b.get("block_idx", 0))
            page_result = self.compare_page(blocks_sorted)

            for block, detail in zip(blocks_sorted, page_result["details"]):
                updated_rows.append({
                    "sample_id": block["sample_id"],
                    "block_idx": block.get("block_idx", 0),
                    "consistency_pattern": detail["pattern"],
                    "block_diff_json": json.dumps(detail["diff"], ensure_ascii=False) if detail.get("diff") else None,
                })

            page_tiers[sample_id] = page_result["tier"]

            processed_pages += 1
            if progress_callback:
                try:
                    progress_callback(processed_pages, total_pages, f"对比页面 {sample_id}")
                except Exception:
                    pass

        return updated_rows, page_tiers

    def process_element_batch_arrow(self, table, progress_callback=None) -> tuple[pa.Table, dict[str, str]]:
        import pyarrow as pa
        n = len(table)
        sid_col = table.column("sample_id")
        bidx_col = table.column("block_idx")
        col_names = set(table.column_names)

        by_sample: dict[str, list[int]] = defaultdict(list)
        for i in range(n):
            by_sample[sid_col[i].as_py()].append(i)

        ocr_keys = [k for k in ("paddle_text", "glm_text", "self_text",
                                 "paddle_table", "glm_table", "self_table",
                                 "paddle_formula", "glm_formula", "self_formula",
                                 "block_type") if k in col_names]
        ocr_cols = {k: table.column(k) for k in ocr_keys}

        total_pages = len(by_sample)
        processed_pages = 0
        out_sids: list[str] = []
        out_bidxs: list[int] = []
        out_patterns: list[str | None] = []
        out_diffs: list[str | None] = []
        page_tiers: dict[str, str] = {}

        for sample_id, indices in by_sample.items():
            blocks = []
            for i in indices:
                row = {"sample_id": sid_col[i].as_py(), "block_idx": bidx_col[i].as_py()}
                for k in ocr_keys:
                    row[k] = ocr_cols[k][i].as_py()
                blocks.append(row)
            blocks_sorted = sorted(blocks, key=lambda b: b.get("block_idx", 0))
            page_result = self.compare_page(blocks_sorted)

            for block, detail in zip(blocks_sorted, page_result["details"]):
                out_sids.append(block["sample_id"])
                out_bidxs.append(block.get("block_idx", 0))
                out_patterns.append(detail["pattern"])
                out_diffs.append(
                    json.dumps(detail["diff"], ensure_ascii=False) if detail.get("diff") else None
                )
            page_tiers[sample_id] = page_result["tier"]

            processed_pages += 1
            if progress_callback:
                try:
                    progress_callback(processed_pages, total_pages, f"对比页面 {sample_id}")
                except Exception:
                    pass

        result_table = pa.table({
            "sample_id": pa.array(out_sids, type=pa.large_string()),
            "block_idx": pa.array(out_bidxs, type=table.schema.field("block_idx").type),
            "consistency_pattern": pa.array(out_patterns, type=pa.large_string()),
            "block_diff_json": pa.array(out_diffs, type=pa.large_string()),
        })
        return result_table, page_tiers