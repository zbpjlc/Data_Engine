from __future__ import annotations

import logging
from pathlib import Path

from data_engine.config import get_config
from data_engine.ocr.base import LayoutBlock

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "PicoDet-L_layout_17cls"


class PPLayoutProvider:

    def __init__(self) -> None:
        self._model = None
        self._score_threshold: float = float(
            get_config("ocr", "layout", "score_threshold", default=0.5)
        )
        self._gpu_id: int = int(
            get_config("ocr", "layout", "gpu_id", default=-1)
        )

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from paddlex import create_model
        except ImportError:
            raise ImportError(
                "paddlex is required for pp-layout. Install via: pip install paddlex"
            )

        model_name = get_config("ocr", "layout", "model_name", default=_DEFAULT_MODEL_NAME)
        model_dir = get_config("ocr", "layout", "model_dir", default=None)
        device = "cpu" if self._gpu_id < 0 else f"gpu:{self._gpu_id}"

        kwargs: dict = {"device": device}
        if model_dir:
            kwargs["model_dir"] = model_dir

        self._model = create_model(model_name, **kwargs)
        logger.info("pp-layout model loaded: %s (device=%s)", model_name, device)

    def detect_layout(self, image_path: Path) -> list[LayoutBlock]:
        self._ensure_model()
        blocks: list[LayoutBlock] = []
        for result in self._model(str(image_path)):
            for det in result.get("boxes", []):
                score = float(det.get("score", 0))
                if score < self._score_threshold:
                    continue
                coord = det.get("coordinate", [0, 0, 0, 0])
                blocks.append(LayoutBlock(
                    block_type=str(det.get("label", "text")),
                    bbox=[float(c) for c in coord],
                    confidence=score,
                ))
        return blocks

    def detect_layout_from_bytes(self, image_bytes: bytes) -> list[LayoutBlock]:
        import tempfile
        import os
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        try:
            os.write(fd, image_bytes)
            os.close(fd)
            return self.detect_layout(Path(tmp_path))
        finally:
            os.unlink(tmp_path)
