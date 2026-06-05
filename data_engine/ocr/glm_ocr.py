from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import requests

from data_engine.config import get_config
from data_engine.ocr.base import BaseOCREngine, LayoutBlock, OCRResult

logger = logging.getLogger(__name__)

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

_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
}


class GLMOCREngine(BaseOCREngine):

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "glm_ocr", "api_url", default="http://localhost:8089")
        self._api_key = api_key or get_config("ocr", "engines", "glm_ocr", "api_key", default="")
        self._timeout = timeout or int(get_config("ocr", "engines", "glm_ocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "glm_ocr", "max_retries", default=3))
        self._model = get_config("ocr", "engines", "glm_ocr", "model", default=None)

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
        import io
        from PIL import Image

        img = Image.open(image_path)
        out: list[OCRResult] = []

        for region in regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            cropped = img.crop((x1, y1, x2, y2))

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            task = _TASK_MAP.get(region.block_type, "text")
            prompt = _PROMPTS[task]

            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0,
            }
            if self._model:
                payload["model"] = self._model

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"

            try:
                raw_json = self._post_with_retry(payload, headers)
                text = self._extract_text(raw_json)
            except Exception as exc:
                logger.warning("GLM OCR failed for region %s: %s", region.block_type, exc)
                text = ""

            table_data = None
            formula = ""
            if region.block_type == "table" and text:
                table_data = {"html": text}
            elif region.block_type == "formula":
                formula = text

            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=text if region.block_type not in ("table", "formula") else "",
                confidence=region.confidence,
                table_structure=table_data,
                formula_latex=formula,
                raw_output={"vl_response": text},
            ))

        return out

    def _post_with_retry(self, payload: dict, headers: dict) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(
                    f"{self._api_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                logger.warning("GLM OCR attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"GLM OCR failed after {self._max_retries} retries: {last_exc}")

    @staticmethod
    def _extract_text(resp: dict) -> str:
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""
