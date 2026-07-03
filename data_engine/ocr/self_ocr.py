from __future__ import annotations

import base64
import io
import json
import logging
import random
from pathlib import Path

import requests

from data_engine.config import get_config
from data_engine.ocr.base import BaseOCREngine, LayoutBlock, OCRResult

logger = logging.getLogger(__name__)


def _mutate_text(text: str, char_error_rate: float = 0.05) -> str:
    """For test mode: slightly mutate text to simulate OCR errors."""
    if not text:
        return text
    chars = list(text)
    n_mutations = max(1, int(len(chars) * char_error_rate))
    for _ in range(n_mutations):
        idx = random.randint(0, len(chars) - 1)
        op = random.choice(["delete", "replace", "insert"])
        if op == "delete" and len(chars) > 1:
            chars.pop(idx)
        elif op == "replace":
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz0123456789 ")
        elif op == "insert":
            chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz0123456789 "))
    return "".join(chars)


_TASK_MAP = {
    "text": "text",
    "title": "text",
    "paragraph_title": "text",
    "number": "text",
    "header": "text",
    "footer": "text",
    "table": "table",
    "formula": "formula",
}

_ENDPOINTS = {
    "text": {"url": "/gen_ocr_cord", "prompt": "OCR:"},
    "table": {"url": "/tsr", "prompt": "Table Recognition:"},
    "formula": {"url": "/formula_infer", "prompt": "Formula Recognition:"},
}


class SelfOCREngine(BaseOCREngine):

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        test_mode: bool = False,
        score_threshold: float | None = None,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "self_ocr", "api_url", default="http://10.112.64.56:8801")
        self._formula_url = get_config("ocr", "engines", "self_ocr", "formula_url", default="http://10.112.64.56:20107")
        self._table_url = get_config("ocr", "engines", "self_ocr", "table_url", default="http://10.112.64.56:8801")
        self._api_key = api_key or get_config("ocr", "engines", "self_ocr", "api_key", default="")
        self._timeout = timeout or int(get_config("ocr", "engines", "self_ocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "self_ocr", "max_retries", default=3))
        self._score_threshold = score_threshold if score_threshold is not None else float(get_config("ocr", "engines", "self_ocr", "score_threshold", default=0.5))
        self._test_mode = test_mode
        # Test mode: existing results from other models, keyed by (sample_id, block_idx)
        self._test_ref_map: dict[tuple[str, int], dict] = {}
        self._test_current_sample_id: str = ""

    @property
    def model_name(self) -> str:
        return "self_ocr"

    @property
    def model_prefix(self) -> str:
        return "self"

    def set_test_context(self, sample_id: str, ref_map: dict[tuple[str, int], dict]) -> None:
        """Test mode: set the current sample_id and reference results from Paddle/GLM."""
        self._test_current_sample_id = sample_id
        self._test_ref_map = ref_map

    def recognize_regions(
        self,
        image_path: Path,
        regions: list[LayoutBlock],
    ) -> list[OCRResult]:
        if self._test_mode:
            return self._recognize_test_mode(regions)

        from PIL import Image

        img = Image.open(image_path)
        out: list[OCRResult] = []

        # 文本识别：对裁剪后的 block 图片 padding 到 1000x1000 后发送
        text_regions = [r for r in regions if _TASK_MAP.get(r.block_type, "text") == "text"]
        
        for region in text_regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            
            # 裁剪 block 区域
            cropped = img.crop((x1, y1, x2, y2))
            
            # Padding 到 1000x1000
            padded_img = Image.new('RGB', (1000, 1000), color='white')
            padded_img.paste(cropped, (0, 0))
            
            # 转换为 base64
            padded_buf = io.BytesIO()
            padded_img.save(padded_buf, format="PNG")
            padded_bytes = padded_buf.getvalue()
            padded_b64 = base64.b64encode(padded_bytes).decode("utf-8")
            
            # 发送 padding 后的图片
            url = self._api_url + "/gen_ocr_cord"
            try:
                raw_json = self._post_json_with_retry(url, {"image": padded_b64})
                shapes = raw_json.get("shapes", [])
                # 使用 shapes 中的最高 score 作为 confidence
                max_score = max((s.get("score", 0) for s in shapes if s.get("score")), default=0)
                text = "\n".join(s.get("label", "") for s in shapes if s.get("label") and s.get("score", 0) >= self._score_threshold)
                out.append(OCRResult(
                    block_type=region.block_type,
                    bbox=region.bbox,
                    text_content=text,
                    confidence=max_score if text else 0.0,  # 使用 OCR 自身的 confidence
                    table_structure=None,
                    formula_latex="",
                    raw_output=raw_json,
                ))
            except Exception as exc:
                logger.warning("SelfOCR failed for region %s: %s", region.block_type, exc)
                out.append(OCRResult(
                    block_type=region.block_type,
                    bbox=region.bbox,
                    text_content="",
                    confidence=region.confidence,
                    table_structure=None,
                    formula_latex="",
                    raw_output={"error": str(exc)},
                ))

        # 表格识别：裁剪表格区域后直接调用 API（不需要 padding）
        table_regions = [r for r in regions if _TASK_MAP.get(r.block_type, "text") == "table"]
        for region in table_regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            
            # 裁剪表格区域
            cropped = img.crop((x1, y1, x2, y2))
            
            # 转换为 base64
            table_buf = io.BytesIO()
            cropped.save(table_buf, format="PNG")
            table_bytes = table_buf.getvalue()
            table_b64 = base64.b64encode(table_bytes).decode("utf-8")
            
            # 先获取 OCR 结果（使用裁剪后的表格图片）
            url_ocr = self._api_url + "/gen_ocr_cord"
            ocr_raw = self._post_json_with_retry(url_ocr, {"image": table_b64})
            
            # 调用表格识别 API
            payload = {
                "file_id": f"{region.block_type}_{x1}_{y1}.png",
                "table_regions": [[0, 0, cropped.size[0], cropped.size[1]]],
                "table_types": 0,
                "image_binary": table_b64,
                "ocr_result": json.dumps(ocr_raw),
            }
            url = self._table_url + "/tsr"
            raw_json = self._post_json_with_retry(url, payload)
            table_results = raw_json.get("data", {}).get("table_results", [])
            table_data = table_results[0] if table_results else None
            # 表格识别：从 table_data 中提取 HTML 作为 text_content
            table_html = table_data.get("html", "") if table_data else ""
            # 使用 API 返回的 confidence，如果没有则使用 OCR 结果的 score
            table_confidence = raw_json.get("confidence", 0.0)
            if table_confidence == 0.0 and ocr_raw.get("shapes"):
                table_confidence = max((s.get("score", 0) for s in ocr_raw["shapes"]), default=0.0)
            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=table_html,
                confidence=table_confidence if table_confidence > 0 else region.confidence,
                table_structure=table_data,
                formula_latex="",
                raw_output=raw_json,
            ))

        # 公式识别：裁剪后直接调用公式 API（不需要 padding）
        formula_regions = [r for r in regions if _TASK_MAP.get(r.block_type, "text") == "formula"]
        for region in formula_regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            cropped = img.crop((x1, y1, x2, y2))
            formula_buf = io.BytesIO()
            cropped.save(formula_buf, format="PNG")
            files = {"image_binary": ("image.png", formula_buf.getvalue(), "image/png")}
            url = self._formula_url + "/formula_infer"
            raw_json = self._post_with_retry(url, files)
            data_list = raw_json.get("data", [])
            formula = data_list[0] if isinstance(data_list, list) and data_list else ""
            # 公式识别使用 API 返回的 confidence，如果没有则使用 layout confidence
            formula_confidence = raw_json.get("confidence", region.confidence)
            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=formula,
                confidence=formula_confidence,
                table_structure=None,
                formula_latex=formula,
                raw_output=raw_json,
            ))

        return out

    def _post_json_with_retry(self, url: str, payload: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    url,
                    data=__import__("json").dumps(payload),
                    headers={"content-type": "application/json"},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning("SelfOCR JSON attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"SelfOCR JSON failed after {self._max_retries} retries: {last_exc}")

    def _post_with_retry(self, url: str, files: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(url, files=files, timeout=self._timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning("SelfOCR attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"SelfOCR failed after {self._max_retries} retries: {last_exc}")

    def _recognize_test_mode(self, regions: list[LayoutBlock]) -> list[OCRResult]:
        """Test mode: generate mock self_ocr results based on Paddle/GLM results.

        - 60% chance: copy one model's result exactly (all_agree)
        - 30% chance: copy with small mutation (partial_agree)
        - 10% chance: generate different text (all_disagree)
        """
        out: list[OCRResult] = []
        for idx, region in enumerate(regions):
            ref_key = (self._test_current_sample_id, idx)
            ref = self._test_ref_map.get(ref_key)

            # Try to get reference text from paddle or glm
            ref_text = None
            ref_table = None
            ref_formula = None
            if ref:
                ref_text = ref.get("paddle_text") or ref.get("glm_text")
                ref_table = ref.get("paddle_table") or ref.get("glm_table")
                ref_formula = ref.get("paddle_formula") or ref.get("glm_formula")

            roll = random.random()

            if ref_text is None and ref_table is None and ref_formula is None:
                # No reference data — produce empty result
                text = ""
                table_data = None
                formula = ""
            elif roll < 0.6:
                # Exact copy (all_agree)
                text = ref_text or ""
                table_data = ref_table
                formula = ref_formula or ""
            elif roll < 0.9:
                # Small mutation (partial_agree)
                text = _mutate_text(ref_text or "", char_error_rate=0.05) if ref_text else ""
                table_data = ref_table
                formula = _mutate_text(ref_formula or "", char_error_rate=0.03) if ref_formula else ""
            else:
                # Big mutation (all_disagree)
                text = _mutate_text(ref_text or "", char_error_rate=0.3) if ref_text else ""
                table_data = None
                formula = _mutate_text(ref_formula or "", char_error_rate=0.2) if ref_formula else ""

            # Respect block type
            task = _TASK_MAP.get(region.block_type, "text")
            if task == "text":
                out.append(OCRResult(
                    block_type=region.block_type, bbox=region.bbox,
                    text_content=text, confidence=region.confidence,
                    table_structure=None, formula_latex="",
                    raw_output={"test_mode": True, "roll": roll},
                ))
            elif task == "table":
                out.append(OCRResult(
                    block_type=region.block_type, bbox=region.bbox,
                    text_content="", confidence=region.confidence,
                    table_structure=table_data, formula_latex="",
                    raw_output={"test_mode": True, "roll": roll},
                ))
            else:  # formula
                out.append(OCRResult(
                    block_type=region.block_type, bbox=region.bbox,
                    text_content="", confidence=region.confidence,
                    table_structure=None, formula_latex=formula,
                    raw_output={"test_mode": True, "roll": roll},
                ))

        return out