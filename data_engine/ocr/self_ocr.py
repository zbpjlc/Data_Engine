from __future__ import annotations

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
        test_mode: bool = False,
    ) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "self_ocr", "api_url", default="http://10.112.64.56:8801")
        self._formula_url = get_config("ocr", "engines", "self_ocr", "formula_url", default="http://10.112.64.56:20107")
        self._table_url = get_config("ocr", "engines", "self_ocr", "table_url", default="http://10.112.64.56:8801")
        self._api_key = api_key or get_config("ocr", "engines", "self_ocr", "api_key", default="")
        self._timeout = timeout or int(get_config("ocr", "engines", "self_ocr", "timeout", default=60))
        self._max_retries = max_retries or int(get_config("ocr", "engines", "self_ocr", "max_retries", default=3))
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

    def _recognize_test_mode(self, regions: list[LayoutBlock]) -> list[OCRResult]:
        """Test mode: generate mock self_ocr results based on Paddle/GLM results in element.lance.

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
                ref_table = ref.get("paddle_table_json") or ref.get("glm_table_json")
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
