"""向量索引构建与统计（IVF_PQ）"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lance

from data_engine.config import get_config
from data_engine.manifests import _lance_write_lock, manifest_count


def _index_stats_path(manifest_path: Path) -> Path:
    """索引记录文件路径: <batch_dir>/artifacts/index_stats.json"""
    return manifest_path.parent.parent / "artifacts" / "index_stats.json"


def _extract_index_name(index_obj: Any) -> str:
    """从 Lance list_indices() 返回对象中提取索引名。"""
    if isinstance(index_obj, dict):
        return index_obj.get("name") or index_obj.get("index_name") or ""
    return getattr(index_obj, "name", "") or getattr(index_obj, "index_name", "")


def list_index_names(manifest_path: Path) -> list[str]:
    """读取 Lance 数据集上所有已存在的索引名称。"""
    if not manifest_path.exists():
        return []
    with _lance_write_lock:
        ds = lance.dataset(str(manifest_path))
        return [_extract_index_name(idx) for idx in ds.list_indices() if _extract_index_name(idx)]


def _save_index_stats(manifest_path: Path, result: dict[str, Any]) -> None:
    """持久化索引构建记录。"""
    path = _index_stats_path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {**result, "built_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_stats(manifest_path: Path) -> dict[str, Any] | None:
    """读取已有的索引构建记录，无记录返回 None。"""
    path = _index_stats_path(manifest_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_ivf_pq_index(
    manifest_path: Path,
    num_partitions: int | None = None,
    num_sub_vectors: int | None = None,
    accelerator: str | None = None,
) -> dict[str, Any]:
    """在 Lance 数据集上构建 IVF_PQ 向量索引。

    Args:
        manifest_path: ingest.lance 路径
        num_partitions: 聚类分区数，默认从 config 读取
        num_sub_vectors: PQ 子向量数，默认从 config 读取
        accelerator: "cuda" 或 None

    Returns:
        构建结果字典
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Lance 数据集不存在: {manifest_path}")

    num_partitions = num_partitions or get_config("index", "num_partitions", default=256)
    num_sub_vectors = num_sub_vectors or get_config("index", "num_sub_vectors", default=16)

    with _lance_write_lock:
        ds = lance.dataset(str(manifest_path))
        total_rows = ds.count_rows()

        if "embedding" not in ds.schema.names:
            raise ValueError("数据集不包含 embedding 列，请先执行 embedding 提取")

        ds.create_index(
            column="embedding",
            index_type="IVF_PQ",
            name="idx_embedding_ivf",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            replace=True,
            accelerator=accelerator,
        )

        result = {
            "success": True,
            "index_name": "idx_embedding_ivf",
            "index_type": "IVF_PQ",
            "column": "embedding",
            "num_partitions": num_partitions,
            "num_sub_vectors": num_sub_vectors,
            "total_rows": total_rows,
            "lance_version": lance.__version__,
            "manifest_path": str(manifest_path),
        }
        _save_index_stats(manifest_path, result)
        return result


def get_index_stats(manifest_path: Path) -> dict[str, Any]:
    """获取向量索引的桶分布统计。

    Args:
        manifest_path: ingest.lance 路径

    Returns:
        统计字典，含 has_index 字段标记是否有索引
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Lance 数据集不存在: {manifest_path}")

    build_record = load_index_stats(manifest_path)

    with _lance_write_lock:
        ds = lance.dataset(str(manifest_path))
        indices = ds.list_indices()

        bucket_counts: list[int] = []
        index_name = None

        for idx in indices:
            idx_name = _extract_index_name(idx)
            if not idx_name:
                continue
            try:
                stats = ds.index_statistics(idx_name)
                if isinstance(stats, str):
                    stats = json.loads(stats)
                # 尝试多种格式提取分区大小
                if isinstance(stats, dict):
                    if "indices" in stats:
                        for sub_idx in stats["indices"]:
                            if "partitions" in sub_idx:
                                for part in sub_idx["partitions"]:
                                    if "size" in part:
                                        bucket_counts.append(part["size"])
                                break
                    elif "num_partitions" in stats and "num_rows" in stats:
                        # 兜底：无详细分区信息时，根据分区数和总行数估算
                        np = stats.get("num_partitions", 0)
                        nr = stats.get("num_rows", 0)
                        if np > 0 and nr > 0:
                            avg = nr // np
                            bucket_counts = [avg] * np
                if bucket_counts:
                    break
            except Exception:
                continue

    # 如果从 Lance 获取失败，尝试从保存的构建记录恢复
    if not bucket_counts and build_record:
        nr = build_record.get("total_rows", 0)
        np = build_record.get("num_partitions", 0)
        if nr > 0 and np > 0:
            avg = nr // np
            bucket_counts = [avg] * np
            index_name = build_record.get("index_name", "idx_embedding_ivf")

    # 最终兜底：直接从 Lance 数据集推断
    if not bucket_counts:
        try:
            with _lance_write_lock:
                ds2 = lance.dataset(str(manifest_path))
                if ds2.list_indices():
                    nr = ds2.count_rows()
                    # IVF_PQ 默认分区数通常为 256
                    np_default = get_config("index", "num_partitions", default=256)
                    avg = nr // np_default
                    bucket_counts = [avg] * np_default
                    index_name = "idx_embedding_ivf"
        except Exception:
            pass

    if not bucket_counts:
        return {"has_index": False, "message": "未找到向量索引，请先构建索引"}

    total = sum(bucket_counts)
    n = len(bucket_counts)
    build_record = load_index_stats(manifest_path)
    return {
        "has_index": True,
        "index_name": index_name,
        "total_partitions": n,
        "total_vectors": total,
        "max_bucket": max(bucket_counts),
        "min_bucket": min(bucket_counts),
        "avg_bucket": int(total / n),
        "empty_buckets": sum(1 for c in bucket_counts if c == 0),
        "small_buckets": sum(1 for c in bucket_counts if c < 50),
        "large_buckets": sum(1 for c in bucket_counts if c > 3000),
        "buckets": bucket_counts[:50],
        "build_record": build_record,
    }
