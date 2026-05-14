from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class InputType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"


class StageStatus(str, Enum):
    INGESTED = "ingested"
    EMBEDDED = "embedded"
    CLUSTERED = "clustered"
    SCORED = "scored"
    INFERRED = "inferred"
    COMPARED = "compared"
    BUCKETED = "bucketed"
    ANNOTATED = "annotated"
    QAED = "qaed"
    RELEASED = "released"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SourceConfig(BaseModel):
    """全局导航图配置 - 只负责物理路径"""
    id: str
    root_path: str | list[str]
    enabled: bool = True

    def root_paths(self) -> list[str]:
        if isinstance(self.root_path, list):
            return self.root_path
        return [self.root_path]

    def resolve_batch_dir(self, batch_id: str) -> Path:
        for rp in self.root_paths():
            candidate = Path(rp) / batch_id
            if candidate.is_dir():
                return candidate
            if Path(rp).name == batch_id and Path(rp).is_dir():
                return Path(rp)
        raise FileNotFoundError(f"Batch '{batch_id}' not found under any root_path of source '{self.id}'")


class BatchMetadata(BaseModel):
    """批次身份证 - 负责逻辑属性"""
    batch_id: str | None = None
    category: str  # 必需字段，实际使用
    task_type: str | None = None
    created_at: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "BatchMetadata":
        """从YAML文件加载批次元数据"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_batch_dir(cls, batch_dir: Path) -> "BatchMetadata":
        """从批次目录加载元数据"""
        meta_file = batch_dir / ".engine_meta.yaml"
        if not meta_file.exists():
            raise FileNotFoundError(f"批次元数据文件不存在: {meta_file}")
        return cls.from_yaml(meta_file)


class SourceRegistryModel(BaseModel):
    sources: list[SourceConfig] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    original_width: int | None = None
    original_height: int | None = None
    normalized_width: int | None = None
    normalized_height: int | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    original_dpi: float | None = None
    normalized_dpi: float | None = None
    color_mode: str | None = None
    pdf_page_count: int | None = None
    pdf_text_excerpt: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class UnifiedSampleRecord(BaseModel):
    sample_id: str
    source_id: str
    category: str
    batch_id: str
    input_type: InputType
    original_ext: str
    page_id: str
    task_type: str = "page"
    relative_path: str
    page_image: str
    block_list: list[dict[str, Any]] = Field(default_factory=list)
    reading_order: list[Any] = Field(default_factory=list)
    table_structure: dict[str, Any] | None = None
    formula_spans: list[dict[str, Any]] = Field(default_factory=list)
    text_spans: list[dict[str, Any]] = Field(default_factory=list)
    bbox: list[float] | None = None
    cluster_id: str | None = None
    difficulty: str | None = None
    annotation_source: str | None = None
    stage_status: StageStatus = StageStatus.INGESTED
    process_log: list[str] = Field(default_factory=lambda: ["ingest_v1"])
    data_version: str = "v1"
    embedding: list[float] | None = None  # CLIP embedding向量
    schema_version: str = "v1"
    threshold_version: str = "v1"
    is_active: bool = True
    page_image_sha256: str
    image_data: bytes | None = None  # 图像二进制数据
    source_metadata: SourceMetadata | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("sample_id", "page_id", "page_image_sha256")
    @classmethod
    def validate_hex(cls, value: str) -> str:
        value = value.lower()
        if not value or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("must be lowercase hex")
        return value


class SourceScanSummary(BaseModel):
    source_id: str
    category: str
    root_path: str
    enabled: bool
    online: bool
    batch_count: int = 0
    manifest_count: int = 0
    last_scan_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BatchStatusSummary(BaseModel):
    source_id: str
    category: str
    batch_id: str
    stage_status: str
    sample_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    difficulty_histogram: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime | None = None
