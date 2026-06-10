"""从 text/formula/table.lance 读取 block 裁剪图，用 SigLIP2(ViT) 提取特征向量，按类型独立做 KMeans 聚类。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data_engine.clustering import KMeansClusterer, find_optimal_clusters
from data_engine.embedding import CLIPEmbeddingExtractor, _coerce_image_bytes

logger = logging.getLogger(__name__)

# ─── 特征名称（用于 bbox 补充统计） ───────────────────────────────────────────

_BASE_FEATURE_NAMES = ["x1", "y1", "width", "height", "area", "aspect_ratio", "confidence"]


# ─── bbox 辅助特征（用于聚类统计展示） ────────────────────────────────────────

def _parse_bbox(bbox_json: Any) -> list[float]:
    if isinstance(bbox_json, str):
        try:
            return json.loads(bbox_json)
        except (json.JSONDecodeError, TypeError):
            return [0.0, 0.0, 0.0, 0.0]
    if isinstance(bbox_json, (list, tuple)):
        return [float(v) for v in bbox_json]
    return [0.0, 0.0, 0.0, 0.0]


def _bbox_stats(row: dict) -> dict[str, float]:
    """从 bbox 计算辅助统计（不参与聚类，仅用于展示）。"""
    bbox = _parse_bbox(row.get("bbox_json"))
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    w, h = max(x2 - x1, 0), max(y2 - y1, 0)
    return {
        "x1": round(x1, 1), "y1": round(y1, 1),
        "width": round(w, 1), "height": round(h, 1),
        "area": round(w * h, 1),
        "aspect_ratio": round(w / max(h, 1), 3),
        "confidence": round(float(row.get("layout_confidence", 0) or 0), 3),
    }


# ─── Lance 读取 ─────────────────────────────────────────────────────────────

def _read_lance_rows(lance_path: Path) -> list[dict]:
    """读取 lance 文件全部行。"""
    import lance
    if not lance_path.exists():
        return []
    try:
        ds = lance.dataset(str(lance_path))
        return ds.to_table().to_pylist()
    except Exception as e:
        logger.warning("[layout_features] 读取 %s 失败: %s", lance_path, e)
        return []


# ─── 聚类 ─────────────────────────────────────────────────────────────────────

def cluster_by_type(
    manifests_dir: Path,
    cat: str,
    extractor: CLIPEmbeddingExtractor,
    max_k: int = 10,
    progress_callback=None,
) -> dict:
    """读取 {cat}.lance，用 SigLIP2 提取每个 block 的 embedding，KMeans 聚类。

    Args:
        manifests_dir: manifests 目录
        cat: text / formula / table
        extractor: SigLIP2 embedding 提取器
        max_k: 最大聚类数
        progress_callback: (current, total, message) 回调

    Returns:
        {
            "n_clusters": int,
            "silhouette": float,
            "total_blocks": int,
            "clusters": {"0": {"size": int, "bbox_mean": {...}}, ...},
            "labels": {"sample_id:block_idx": int, ...}
        }
    """
    lance_path = manifests_dir / f"{cat}.lance"
    rows = _read_lance_rows(lance_path)
    if not rows:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": 0, "clusters": {}, "labels": {}}

    total = len(rows)

    # 1. 提取 embedding
    keys: list[str] = []
    embeddings: list[list[float] | None] = []
    bbox_info: list[dict] = []

    for idx, row in enumerate(rows):
        sid = row.get("sample_id", "")
        bidx = row.get("block_idx", 0)
        keys.append(f"{sid}:{bidx}")
        bbox_info.append(_bbox_stats(row))

        img_data = row.get("image_data")
        emb = None
        if img_data:
            img_bytes = _coerce_image_bytes(img_data)
            if img_bytes:
                try:
                    emb = extractor.extract_embedding_from_bytes(img_bytes)
                except Exception as e:
                    logger.warning("[layout_features] %s embedding 失败 %s:%s: %s", cat, sid, bidx, e)

        embeddings.append(emb)

        if progress_callback and (idx + 1) % 50 == 0 or idx == total - 1:
            progress_callback(idx + 1, total, f"{cat} embedding {idx + 1}/{total}")

    # 2. 过滤有效 embedding
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None and len(emb) > 0]
    if len(valid_indices) < 2:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": total, "clusters": {}, "labels": {}}

    X = np.array([embeddings[i] for i in valid_indices])

    # 3. KMeans 聚类
    optimal_k = find_optimal_clusters(X.tolist(), max_k=min(max_k, len(valid_indices)))
    clusterer = KMeansClusterer(n_clusters=optimal_k)
    labels_array = clusterer.fit_predict(X.tolist())
    stats = clusterer.get_cluster_stats()

    # 4. 构建 cluster 统计（bbox 均值）
    clusters_info: dict[str, dict] = {}
    label_to_idx: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels_array):
        if lbl < 0:
            continue
        label_to_idx.setdefault(lbl, []).append(i)

    for lbl, indices in label_to_idx.items():
        # 聚合 bbox 统计
        bbox_means: dict[str, float] = {}
        for key in _BASE_FEATURE_NAMES:
            vals = [bbox_info[valid_indices[i]][key] for i in indices if key in bbox_info[valid_indices[i]]]
            bbox_means[key] = round(sum(vals) / max(len(vals), 1), 2)

        clusters_info[str(lbl)] = {
            "size": len(indices),
            "bbox_mean": bbox_means,
        }

    # 5. 构建 labels 映射
    labels_map: dict[str, int] = {}
    for i, lbl in enumerate(labels_array):
        if lbl >= 0:
            labels_map[keys[valid_indices[i]]] = int(lbl)

    # 清理显存
    del X
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "n_clusters": stats.get("n_clusters", optimal_k),
        "silhouette": round(stats.get("silhouette_score", 0), 4),
        "total_blocks": total,
        "clusters": clusters_info,
        "labels": labels_map,
    }


# ─── 全类型聚类入口 ──────────────────────────────────────────────────────────

def cluster_all_types(
    manifests_dir: Path,
    max_k: int = 10,
    progress_callback=None,
) -> dict:
    """对 text / formula / table 分别做 SigLIP2 embedding + KMeans 聚类。

    Returns:
        {"text": {...}, "formula": {...}, "table": {...}}
    """
    extractor = CLIPEmbeddingExtractor()

    result: dict[str, dict] = {}
    for cat in ("text", "formula", "table"):

        def _cb(cur, tot, msg):
            if progress_callback:
                progress_callback(cur, tot, f"[{cat}] {msg}")

        result[cat] = cluster_by_type(manifests_dir, cat, extractor, max_k=max_k, progress_callback=_cb)
        logger.info(
            "[layout_features] %s: %d blocks -> %d clusters (silhouette=%.3f)",
            cat, result[cat]["total_blocks"], result[cat]["n_clusters"], result[cat]["silhouette"],
        )

    # 清理
    import gc, torch
    del extractor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
