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


class _PaddleNoiseFilter(logging.Filter):
    """抑制 paddlex 反复打印的噪音日志。"""
    _NOISE = ("Creating model", "Model files already exist")
    def filter(self, record):
        msg = record.getMessage()
        return not any(n in msg for n in self._NOISE)


class PaddleOCREngine(BaseOCREngine):

    def __init__(self, api_url: str | None = None) -> None:
        self._api_url = api_url or get_config("ocr", "engines", "paddleocr", "api_url", default="http://localhost:8085")
        self._pipeline = None
        # 抑制 paddlex 的噪音日志（"Creating model" / "Model files already exist"）
        _filter = _PaddleNoiseFilter()
        logging.getLogger("paddlex").addFilter(_filter)
        logging.getLogger().addFilter(_filter)  # 根 logger（paddlex 部分代码用 logging.info）

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
                use_layout_detection=False,  # 不需要 layout 检测，block 已预裁剪
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

    def recognize_regions_batch(
        self,
        image_path: Path,
        blocks: list[dict],
    ) -> list[OCRResult]:
        """Batch OCR: 同一 page 的所有 block 按 prompt_label 分组，每组一次 predict()。
        blocks: [{"block_type": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        bbox 为绝对坐标，仅用于 crop。
        """
        import numpy as np
        from PIL import Image

        pipeline = self._get_pipeline()
        img = Image.open(image_path)

        # 按 prompt_label 分组: {label: [(idx, cropped_array), ...]}
        groups: dict[str | None, list[tuple[int, np.ndarray]]] = {}
        for i, b in enumerate(blocks):
            x1, y1, x2, y2 = [int(c) for c in b["bbox"]]
            cropped = img.crop((x1, y1, x2, y2))
            label = _PROMPT_LABEL_MAP.get(b.get("block_type", "text"))
            groups.setdefault(label, []).append((i, np.array(cropped)))

        results_map: dict[int, list] = {i: [] for i in range(len(blocks))}

        for label, items in groups.items():
            if not items:
                continue
            indices = [idx for idx, _ in items]
            batch_images = [arr for _, arr in items]

            batch_results = list(pipeline.predict(
                batch_images,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                prompt_label=label,
            ))

            for idx, res in zip(indices, batch_results):
                results_map[idx] = res.get("parsing_res_list", []) if res else []

        out: list[OCRResult] = []
        for i, b in enumerate(blocks):
            parsing_res = results_map.get(i, [])
            text = ""
            table_html = ""
            for block in parsing_res:
                if block.label == "table":
                    table_html = block.content
                else:
                    text = block.content
                    break

            bt = b.get("block_type", "text")
            table_data = None
            formula = ""
            if bt == "table" and table_html:
                table_data = {"html": table_html}
            elif bt == "formula":
                formula = text

            out.append(OCRResult(
                block_type=bt,
                bbox=b["bbox"],
                text_content=text if bt not in ("table", "formula") else "",
                confidence=b.get("confidence", 1.0),
                table_structure=table_data,
                formula_latex=formula,
                raw_output={"vl_response": table_html or text},
            ))

        return out
