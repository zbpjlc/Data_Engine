from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import requests

from data_engine.config import get_config
from data_engine.ocr.base import BaseOCREngine, LayoutBlock, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "paddleocr", "api_url", default="http://localhost:8080")
        self._api_key = api_key or get_config("ocr", "engines", "paddleocr", "api_key", default=None)
        self._timeout = timeout or int(get_config("ocr", "engines", "paddleocr", "timeout", default=30))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "paddleocr", "max_retries", default=3))
        self._lang = get_config("ocr", "engines", "paddleocr", "lang", default="ch")

    @property
    def model_name(self) -> str:
        return "paddleocr_v1.6"

    @property
    def model_prefix(self) -> str:
        return "paddle"

    def recognize_regions(
        self,
        image_path: Path,
        regions: list[LayoutBlock],
    ) -> list[OCRResult]:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        payload = {
            "image": image_b64,
            "lang": self._lang,
            "regions": [
                {"type": r.block_type, "bbox": r.bbox}
                for r in regions
            ],
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        raw_text = self._post_with_retry(payload, headers)
        raw_json = json.loads(raw_text) if raw_text else {}
        return self._parse_response(raw_json, regions)

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
                logger.warning("PaddleOCR attempt %d failed: %s", attempt + 1, exc)
        raise RuntimeError(f"PaddleOCR failed after {self._max_retries} retries: {last_exc}")

    def _parse_response(self, raw: dict, regions: list[LayoutBlock]) -> list[OCRResult]:
        results_raw = raw.get("results", raw.get("data", []))
        if not isinstance(results_raw, list):
            results_raw = [results_raw]
        out: list[OCRResult] = []
        for idx, region in enumerate(regions):
            item = results_raw[idx] if idx < len(results_raw) else {}
            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=str(item.get("text", "")),
                confidence=float(item.get("confidence", 0)),
                table_structure=item.get("table") if region.block_type == "table" else None,
                formula_latex=str(item.get("formula", "")) if region.block_type == "formula" else "",
                raw_output=item,
            ))
        return out
