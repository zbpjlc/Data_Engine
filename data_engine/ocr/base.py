from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayoutBlock:
    block_type: str
    bbox: list[float]
    confidence: float


@dataclass
class OCRResult:
    block_type: str
    bbox: list[float]
    text_content: str = ""
    confidence: float = 0.0
    table_structure: dict | None = None
    formula_latex: str = ""
    raw_output: dict = field(default_factory=dict)


class BaseOCREngine(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @property
    @abstractmethod
    def model_prefix(self) -> str: ...

    @abstractmethod
    def recognize_regions(
        self,
        image_path: Path,
        regions: list[LayoutBlock],
    ) -> list[OCRResult]: ...

    def recognize_regions_batch(
        self,
        image_path: Path,
        blocks: list[dict],
    ) -> list[OCRResult]:
        """Batch 接口。默认回退到逐 block 的 recognize_regions。
        blocks: [{"block_type": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
        """
        regions = [LayoutBlock(block_type=b.get("block_type", "text"), bbox=b["bbox"], confidence=b.get("confidence", 1.0)) for b in blocks]
        return self.recognize_regions(image_path, regions)
