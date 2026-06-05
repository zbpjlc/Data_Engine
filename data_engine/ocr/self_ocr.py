from __future__ import annotations

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

_ENDPOINTS = {
    "text": {"url": "/ocr", "prompt": "OCR:"},
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
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "self_ocr", "api_url", default="http://10.112.64.56:8801")
        self._formula_url = get_config("ocr", "engines", "self_ocr", "formula_url", default="http://10.112.64.56:20107")
        self._table_url = get_config("ocr", "engines", "self_ocr", "table_url", default="http://10.112.64.56:8801")
        self._api_key = api_key or get_config("ocr", "engines", "self_ocr", "api_key", default="")
        self._timeout = timeout or int(get_config("ocr", "engines", "self_ocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "self_ocr", "max_retries", default=3))

    @property
    def model_name(self) -> str:
        return "self_ocr"

    @property
    def model_prefix(self) -> str:
        return "self"

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
            task = _TASK_MAP.get(region.block_type, "text")
            endpoint = _ENDPOINTS[task]

            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            cropped = img.crop((x1, y1, x2, y2))

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            buf.seek(0)

            # 选择对应的 URL
            if task == "formula":
                base_url = self._formula_url
            elif task == "table":
                base_url = self._table_url
            else:
                base_url = self._api_url

            url = base_url + endpoint["url"]
            files = {"image_binary": ("image.png", buf, "image/png")}

            try:
                raw_json = self._post_with_retry(url, files)
            except Exception as exc:
                logger.warning("SelfOCR failed for region %s: %s", region.block_type, exc)
                raw_json = {}

            text = str(raw_json.get("text", raw_json.get("result", raw_json.get("formula", ""))))
            table_data = raw_json.get("table") if task == "table" else None
            formula = text if task == "formula" else ""

            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=text if task not in ("table", "formula") else "",
                confidence=region.confidence,
                table_structure=table_data,
                formula_latex=formula,
                raw_output=raw_json,
            ))

        return out

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
