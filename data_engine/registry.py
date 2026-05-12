from __future__ import annotations

from pathlib import Path
import re

from data_engine.models import SourceConfig, SourceRegistryModel, SourceScanSummary, BatchMetadata
from data_engine.yaml_support import dump_yaml, load_yaml


DEFAULT_REGISTRY_PATH = Path("sources.yaml")


class SourceRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_REGISTRY_PATH
        self.model = self._load()

    def _load(self) -> SourceRegistryModel:
        if not self.path.exists():
            return SourceRegistryModel()
        return SourceRegistryModel.model_validate(load_yaml(self.path))

    def save(self) -> None:
        dump_yaml(self.model.model_dump(mode="json"), self.path)

    def list_sources(self) -> list[SourceConfig]:
        return list(self.model.sources)

    def get(self, source_id: str) -> SourceConfig:
        for source in self.model.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"Unknown source_id: {source_id}")

    def register(self, root_path: Path, source_id: str | None = None, category: str | None = None) -> SourceConfig:
        root_path = root_path.resolve()
        for source in self.model.sources:
            if Path(source.root_path).resolve() == root_path:
                if source_id and source.id != source_id:
                    source.id = source_id
                # category现在从.engine_meta.yaml读取，不再存储在source中
                self.save()
                return source

        source = SourceConfig(
            id=source_id or derive_source_id(root_path),
            root_path=str(root_path),
            enabled=True,
        )
        self.model.sources.append(source)
        self.save()
        return source

    def scan(self) -> list[SourceScanSummary]:
        summaries: list[SourceScanSummary] = []
        for source in self.model.sources:
            root = Path(source.root_path)
            online = root.exists()
            batch_count = 0
            manifest_count = 0
            categories = set()
            
            if online:
                for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith("batch_") or (p / ".engine_meta.yaml").exists())):
                    batch_count += 1
                    manifests_dir = batch_dir / "manifests"
                    if manifests_dir.exists():
                        manifest_count += len([p for p in manifests_dir.iterdir() if p.is_file()])
                    
                    # 去中心化元数据：从.engine_meta.yaml读取category
                    try:
                        batch_meta = BatchMetadata.from_batch_dir(batch_dir)
                        categories.add(batch_meta.category)
                    except FileNotFoundError:
                        # 如果没有.engine_meta.yaml，使用默认category
                        pass
            
            # 使用找到的categories，如果没有则使用默认值
            category = categories.pop() if categories else "unknown"
            
            summaries.append(
                SourceScanSummary(
                    source_id=source.id,
                    category=category,
                    root_path=source.root_path,
                    enabled=source.enabled,
                    online=online,
                    batch_count=batch_count,
                    manifest_count=manifest_count,
                )
            )
        return summaries


def derive_source_id(path: Path) -> str:
    base = path.name or "source"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return slug or "source"
