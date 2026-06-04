from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import requests

from data_engine.config import get_config
from data_engine.ocr.base import BaseOCREngine, LayoutBlock, OCRResult

logger = logging.getLogger(__name__)

_PROMPTS = {
    "text": "请识别这个区域内的所有文字，原样输出",
    "table": "请识别这个表格的结构和内容，输出为JSON格式的行列数据",
    "formula": "请识别这个公式的LaTeX表达式",
    "figure": "请描述这张图片的内容",
    "title": "请识别这个标题的文字，原样输出",
}


class GLMOCREngine(BaseOCREngine):

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "glm_ocr", "api_url", default="http://localhost:8081")
        self._api_key = api_key or get_config("ocr", "engines", "glm_ocr", "api_key", default=None)
        self._timeout = timeout or int(get_config("ocr", "engines", "glm_ocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "glm_ocr", "max_retries", default=3))
        self._max_tokens = int(get_config("ocr", "engines", "glm_ocr", "max_tokens", default=2048))

    @property
    def model_name(self) -> str:
        return "glm_ocr"

    @property
    def model_prefix(self) -> str:
        return "glm"

    def recognize_regions(
        self,
        image_path: Path,
        regions: list[LayoutBlock],
    ) -> list[OCRResult]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        out: list[OCRResult] = []
        for region in regions:
            prompt = _PROMPTS.get(region.block_type, _PROMPTS["text"])
            payload = {
                "image": image_b64,
                "prompt": prompt,
                "bbox": region.bbox,
                "max_tokens": self._max_tokens,
            }
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            try:
                raw_text = self._post_with_retry(payload, headers)
                raw_json = json.loads(raw_text) if raw_text else {}
            except Exception as exc:
                logger.warning("GLM OCR failed for region %s: %s", region.block_type, exc)
                raw_json = {}

            text = str(raw_json.get("text", raw_json.get("result", "")))
            table_data = raw_json.get("table") if region.block_type == "table" else None
            formula = text if region.block_type == "formula" else ""

            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=text if region.block_type != "formula" else "",
                confidence=float(raw_json.get("confidence", 0)),
                table_structure=table_data,
                formula_latex=formula,
                raw_output=raw_json,
            ))
        return out

    def _post_with_retry(self, payload: dict, headers: dict) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    f"{self._api_url}/recognize",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                last_exc = exc
                logger.warning("GLM OCR attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"GLM OCR failed after {self._max_retries} retries: {last_exc}")
