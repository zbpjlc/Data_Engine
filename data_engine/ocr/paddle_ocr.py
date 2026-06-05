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
    "text": "ocr",
    "title": "ocr",
    "paragraph_title": "ocr",
    "number": "ocr",
    "header": "ocr",
    "footer": "ocr",
    "table": "table",
    "formula": "formula",
    "seal": "seal",
}

_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "chart": "Chart Recognition:",
    "formula": "Formula Recognition:",
    "seal": "Seal Recognition:",
    "spotting": "Spotting:",
}


class PaddleOCREngine(BaseOCREngine):

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "paddleocr", "api_url", default="http://localhost:8085")
        self._api_key = api_key or get_config("ocr", "engines", "paddleocr", "api_key", default="")
        self._timeout = timeout or int(get_config("ocr", "engines", "paddleocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "paddleocr", "max_retries", default=3))
        self._model = get_config("ocr", "engines", "paddleocr", "model", default=None)

    @property
    def model_name(self) -> str:
        return "paddleocr_vl"

    @property
    def model_prefix(self) -> str:
        return "paddle"

    def recognize_regions(
        self,
        image_path: Path,
        regions: list[LayoutBlock],
    ) -> list[OCRResult]:
        import io
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for image cropping. Install via: pip install Pillow")

        img = Image.open(image_path)
        out: list[OCRResult] = []

        for region in regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            cropped = img.crop((x1, y1, x2, y2))

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            task = _TASK_MAP.get(region.block_type, "ocr")
            prompt = _PROMPTS.get(task, _PROMPTS["ocr"])

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
                "extra_body": {"task": task},
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
                logger.warning("PaddleOCR VL failed for region %s: %s", region.block_type, exc)
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
                logger.warning("PaddleOCR VL attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"PaddleOCR VL failed after {self._max_retries} retries: {last_exc}")

    @staticmethod
    def _extract_text(resp: dict) -> str:
        try:
            return resp["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""
