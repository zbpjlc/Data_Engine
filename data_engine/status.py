from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from data_engine.manifests import find_stage_manifest, read_json, manifest_count
from data_engine.models import BatchStatusSummary, SourceScanSummary
from data_engine.registry import SourceRegistry


STAGES_IN_ORDER = ["export", "refine", "cmcv", "element_sampling", "page_sampling", "ingest"]

_STATUS_CACHE: GlobalStatus | None = None
_STATUS_CACHE_TIME: float = 0
_STATUS_CACHE_TTL: float = 30  # 缓存30秒


@dataclass
class GlobalStatus:
    sources: list[SourceScanSummary]
    batches: list[BatchStatusSummary]


def invalidate_status_cache() -> None:
    """手动清除缓存（在 ingest/embed/cluster 完成后调用）"""
    global _STATUS_CACHE, _STATUS_CACHE_TIME
    _STATUS_CACHE = None
    _STATUS_CACHE_TIME = 0


def collect_global_status(registry: SourceRegistry, force_refresh: bool = False) -> GlobalStatus:
    global _STATUS_CACHE, _STATUS_CACHE_TIME
    now = time.time()
    if not force_refresh and _STATUS_CACHE and (now - _STATUS_CACHE_TIME) < _STATUS_CACHE_TTL:
        return _STATUS_CACHE
    # ... existing code below
    source_summaries = registry.scan()
    batch_summaries: list[BatchStatusSummary] = []
    for source_summary in source_summaries:
        if not source_summary.online:
            continue
        source_config = registry.get(source_summary.source_id)
        for root_path in source_config.root_paths():
            root = Path(root_path)
            if not root.exists():
                continue
            if (root / ".engine_meta.yaml").exists():
                batch_summaries.append(_summarize_batch(source_summary.source_id, source_summary.category, root))
            else:
                for batch_dir in sorted(p for p in root.iterdir() if p.is_dir() and (p.name.startswith("batch_") or (p / ".engine_meta.yaml").exists())):
                    batch_summaries.append(_summarize_batch(source_summary.source_id, source_summary.category, batch_dir))
    result = GlobalStatus(sources=source_summaries, batches=batch_summaries)
    _STATUS_CACHE = result
    _STATUS_CACHE_TIME = time.time()
    return result


def _summarize_batch(source_id: str, category: str, batch_dir: Path) -> BatchStatusSummary:
    manifests_dir = batch_dir / "manifests"
    artifacts_dir = batch_dir / "artifacts"
    stats = read_json(artifacts_dir / "stats.json")

    stage_status = _detect_stage_status(manifests_dir)
    sample_count = int(stats.get("total_samples", 0))
    difficulty_histogram = stats.get("difficulty_histogram", {}) or {}
    pending_count = max(sample_count - _completed_count(stage_status, sample_count), 0)
    failed_count = int(stats.get("failed_samples", 0))
    updated_at = _parse_time(stats.get("updated_at"))

    ingest_manifest = find_stage_manifest(manifests_dir, "ingest")
    lance_version = 0
    if ingest_manifest and ingest_manifest.suffix == ".lance":
        try:
            import lance
            ds = lance.dataset(str(ingest_manifest))
            lance_version = ds.version
            # 用 lance 实际行数校验 sample_count，不一致时以 lance 为准
            lance_count = manifest_count(ingest_manifest)
            if lance_count > 0 and lance_count != sample_count:
                sample_count = lance_count
            pending_count = max(sample_count - _completed_count(stage_status, sample_count), 0)
        except Exception:
            pass

    return BatchStatusSummary(
        source_id=source_id,
        category=category,
        batch_id=batch_dir.name,
        stage_status=stage_status,
        sample_count=sample_count,
        failed_count=failed_count,
        pending_count=pending_count,
        lance_version=lance_version,
        difficulty_histogram=difficulty_histogram,
        updated_at=updated_at,
    )


def _detect_stage_status(manifests_dir: Path) -> str:
    ingest_manifest = find_stage_manifest(manifests_dir, "ingest")
    if ingest_manifest:
        if ingest_manifest.suffix == ".lance":
            try:
                import lance
                ds = lance.dataset(str(ingest_manifest))
                table = ds.to_table(limit=20)
                sample_records = []
                for i in range(table.num_rows):
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    sample_records.append(row)
            except Exception:
                return "ingested"
        else:
            return "ingested"

        if sample_records:
            has_cluster = any(r.get("cluster_id") for r in sample_records)
            if has_cluster:
                return "clustered"
            has_embedding = any(r.get("embedding") is not None for r in sample_records)
            if has_embedding:
                return "embedded"
        return "ingested"
    for stage in STAGES_IN_ORDER:
        if find_stage_manifest(manifests_dir, stage):
            return _stage_name_to_status(stage)
    return "not_started"


def _stage_name_to_status(stage: str) -> str:
    return {
        "ingest": "ingested",
        "page_sampling": "scored",
        "element_sampling": "scored",
        "cmcv": "bucketed",
        "refine": "annotated",
        "export": "released",
    }.get(stage, stage)


def _completed_count(stage_status: str, sample_count: int) -> int:
    return sample_count if stage_status in {"released", "annotated", "bucketed", "scored", "ingested", "embedded", "clustered"} else 0


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def format_status_report(global_status: GlobalStatus) -> str:
    lines: list[str] = []
    lines.append("Sources")
    for source in global_status.sources:
        lines.append(
            f"- {source.source_id} [{source.category}] online={source.online} "
            f"enabled={source.enabled} batches={source.batch_count} manifests={source.manifest_count}"
        )
    lines.append("")
    lines.append("Batches")
    for batch in global_status.batches:
        histogram = ", ".join(f"{k}={v}" for k, v in sorted(batch.difficulty_histogram.items()))
        lines.append(
            f"- {batch.batch_id} source={batch.source_id} category={batch.category} "
            f"stage={batch.stage_status} samples={batch.sample_count} pending={batch.pending_count} "
            f"failed={batch.failed_count}" + (f" difficulty[{histogram}]" if histogram else "")
        )
    return "\n".join(lines)
