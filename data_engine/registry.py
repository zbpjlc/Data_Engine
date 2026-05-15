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
        self.model = self._load()
        return list(self.model.sources)

    def get(self, source_id: str) -> SourceConfig:
        self.model = self._load()
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
        import lance
        self.model = self._load()
        summaries: list[SourceScanSummary] = []
        for source in self.model.sources:
            online = False
            batch_count = 0
            manifest_count = 0
            lance_version = 0
            categories = set()
            roots = source.root_paths()
            root_display = roots[0] if len(roots) == 1 else str(roots)

            for root_path in roots:
                root = Path(root_path)
                if not root.exists():
                    continue
                online = True
                has_meta = (root / ".engine_meta.yaml").exists()
                subdirs = list(root.iterdir()) if root.is_dir() else []
                batch_dirs = [root] if has_meta else sorted(
                    p for p in subdirs
                    if p.is_dir() and (p.name.startswith("batch_") or (p / ".engine_meta.yaml").exists())
                )
                for batch_dir in batch_dirs:
                    batch_count += 1
                    manifests_dir = batch_dir / "manifests"
                    if manifests_dir.exists():
                        manifest_count += len([p for p in manifests_dir.iterdir() if p.is_file()])
                        lance_path = manifests_dir / "ingest.lance"
                        if lance_path.exists():
                            try:
                                ds = lance.dataset(str(lance_path))
                                lance_version = max(lance_version, ds.version)
                            except Exception:
                                pass
                    try:
                        batch_meta = BatchMetadata.from_batch_dir(batch_dir)
                        categories.add(batch_meta.category)
                    except FileNotFoundError:
                        pass

            category = categories.pop() if categories else "unknown"
            summaries.append(
                SourceScanSummary(
                    source_id=source.id,
                    category=category,
                    root_path=root_display,
                    enabled=source.enabled,
                    online=online,
                    batch_count=batch_count,
                    manifest_count=manifest_count,
                    lance_version=lance_version,
                )
            )
        return summaries


def derive_source_id(path: Path) -> str:
    base = path.name or "source"
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    return slug or "source"
