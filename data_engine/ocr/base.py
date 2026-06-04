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
