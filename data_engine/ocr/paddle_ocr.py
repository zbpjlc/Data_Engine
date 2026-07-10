from __future__ import annotations

import logging
from pathlib import Path

from data_engine.config import get_config
from data_engine.ocr.base import BaseOCREngine, LayoutBlock, OCRResult

logger = logging.getLogger(__name__)

_PROMPT_LABEL_MAP = {
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


class PaddleOCREngine(BaseOCREngine):

    def __init__(self, api_url: str | None = None) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "paddleocr", "api_url", default="http://localhost:8085")
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL
            server_url = self._api_url.rstrip("/")
            if not server_url.endswith("/v1"):
                server_url += "/v1"
            self._pipeline = PaddleOCRVL(
                pipeline_version="v1.5",
                vl_rec_backend="vllm-server",
                vl_rec_server_url=server_url,
            )
        return self._pipeline

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
        import numpy as np
        from PIL import Image

        pipeline = self._get_pipeline()
        img = Image.open(image_path)
        out: list[OCRResult] = []

        for region in regions:
            x1, y1, x2, y2 = [int(c) for c in region.bbox]
            cropped = img.crop((x1, y1, x2, y2))
            img_array = np.array(cropped)

            prompt_label = _PROMPT_LABEL_MAP.get(region.block_type)

            results = list(pipeline.predict(
                img_array,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                prompt_label=prompt_label,
            ))

            text = ""
            table_html = ""
            for res in results:
                for block in res.get("parsing_res_list", []):
                    if block.label == "table":
                        table_html = block.content
                    else:
                        text = block.content
                        break

            table_data = None
            formula = ""
            if region.block_type == "table" and table_html:
                table_data = {"html": table_html}
            elif region.block_type == "formula":
                formula = text

            out.append(OCRResult(
                block_type=region.block_type,
                bbox=region.bbox,
                text_content=text if region.block_type not in ("table", "formula") else "",
                confidence=region.confidence,
                table_structure=table_data,
                formula_latex=formula,
                raw_output={"vl_response": table_html or text},
            ))

        return out

