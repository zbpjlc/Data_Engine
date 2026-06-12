"""从 text/formula/table.lance 读取 block 裁剪图，用 ViT 提取特征向量，按类型独立做 MiniBatchKMeans 聚类。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from data_engine.clustering import KMeansClusterer, find_optimal_clusters
from data_engine.config import get_config
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

# 聚类扫描需要的列（含 image_data，裁剪图已随 layout 拆分保存到 text/formula/table.lance）
_SCAN_COLUMNS = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence", "embedding", "image_data"]

def _read_lance_rows(lance_path: Path, columns: list[str] | None = None) -> list[dict]:
    """读取 lance 文件行。默认只读取轻量列（不含 image_data）。"""
    import lance
    if not lance_path.exists():
        return []
    try:
        ds = lance.dataset(str(lance_path))
        # 过滤掉 schema 中不存在的列
        if columns:
            available = set(ds.schema.names)
            columns = [c for c in columns if c in available]
        table = ds.to_table(columns=columns) if columns else ds.to_table()
        return table.to_pylist()
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
    """读取 {cat}.lance，用 ViT 提取/复用每个 block 的 embedding，MiniBatchKMeans 聚类。

    Embedding 缓存策略：
    - 读取 lance 中已有的 embedding 列
    - 仅为缺失 embedding 的 block 生成新向量
    - 生成后写回 lance（原地更新 embedding 列）

    聚类策略（与 Page-Level 一致）：
    - find_optimal_clusters 自动选最优 K
    - KMeansClusterer (MiniBatchKMeans) 执行聚类

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
    if not lance_path.exists():
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": 0, "clusters": {}, "labels": {}}

    # 第一次读取：只读轻量列（不含 image_data），避免内存爆炸
    _LIGHT_COLUMNS = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence", "embedding"]
    rows = _read_lance_rows(lance_path, columns=_LIGHT_COLUMNS)
    if not rows:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": 0, "clusters": {}, "labels": {}}

    total = len(rows)
    has_embedding_col = "embedding" in (rows[0] if rows else {})

    # 1. 读取已有 embedding，仅为缺失的生成新向量
    keys: list[str] = []
    embeddings: list[list[float] | None] = []
    bbox_info: list[dict] = []
    need_generate: list[int] = []

    for idx, row in enumerate(rows):
        sid = row.get("sample_id", "")
        bidx = row.get("block_idx", 0)
        keys.append(f"{sid}:{bidx}")
        bbox_info.append(_bbox_stats(row))

        existing_emb = row.get("embedding") if has_embedding_col else None
        if existing_emb and len(existing_emb) > 0:
            embeddings.append(list(existing_emb))
        else:
            embeddings.append(None)
            need_generate.append(idx)

        if progress_callback and (idx + 1) % 5000 == 0:
            progress_callback(idx + 1, total, f"{cat} 扫描 {idx + 1}/{total}")

    # 2. 为缺失 embedding 的 block 生成（分批从 lance 读取 image_data）
    generated_count = 0
    if need_generate:
        logger.info("[layout_features] %s: %d/%d blocks need embedding generation", cat, len(need_generate), total)

        # 分批处理：每次从 lance 读取一批 image_data
        chunk_size = 200
        processed_count = 0

        for chunk_start in range(0, len(need_generate), chunk_size):
            chunk_indices = need_generate[chunk_start:chunk_start + chunk_size]

            # 从 lance 只读取本批需要的 image_data
            chunk_images: dict[int, bytes] = {}  # row_idx -> image_bytes
            try:
                import lance
                ds = lance.dataset(str(lance_path))
                # 构建 sample_id IN (...) 过滤条件
                chunk_sids = [rows[idx]["sample_id"] for idx in chunk_indices]
                sids_str = ",".join(f"'{sid}'" for sid in set(chunk_sids))
                filter_str = f"sample_id IN ({sids_str})"
                
                # 批量查询 image_data
                result = ds.to_table(columns=["sample_id", "block_idx", "image_data"], filter=filter_str)
                if result.num_rows > 0:
                    res_sids = result.column("sample_id").to_pylist()
                    res_bidxs = result.column("block_idx").to_pylist()
                    res_imgs = result.column("image_data").to_pylist()
                    
                    # 构建 sid:bidx -> image_bytes 映射
                    img_map: dict[tuple, bytes] = {}
                    for sid, bidx, img in zip(res_sids, res_bidxs, res_imgs):
                        if img:
                            img_bytes = _coerce_image_bytes(img)
                            if img_bytes:
                                img_map[(sid, bidx)] = img_bytes
                    
                    # 按 row_idx 匹配
                    for row_idx in chunk_indices:
                        key = (rows[row_idx]["sample_id"], rows[row_idx]["block_idx"])
                        if key in img_map:
                            chunk_images[row_idx] = img_map[key]
                    
                    del img_map
                del result
            except Exception as e:
                logger.warning("[layout_features] %s: failed to load image_data chunk: %s", cat, e)

            # 批量 ViT 推理
            if chunk_images:
                image_list = list(chunk_images.values())
                row_idx_list = list(chunk_images.keys())
                vit_batch = 1  # 强制 batch=1 避免 cuBLAS LT 在线程中崩溃
                emb_results = extractor.extract_embeddings_from_bytes_batch(image_list, batch_size=vit_batch)

                for row_idx, emb in zip(row_idx_list, emb_results):
                    if emb is not None:
                        embeddings[row_idx] = emb
                        generated_count += 1

            processed_count += len(chunk_indices)
            if progress_callback:
                skipped = processed_count - generated_count
                skip_hint = f" (跳过 {skipped} 无图)" if skipped > 0 else ""
                progress_callback(
                    processed_count, len(need_generate),
                    f"{cat} embedding {generated_count}/{len(need_generate)}{skip_hint}"
                )

            # 清理本批内存
            del chunk_images

        # 3. 写回 lance（更新 embedding 列）
        if generated_count > 0:
            _write_embeddings_back(lance_path, rows, keys, embeddings)
            logger.info("[layout_features] %s: wrote %d embeddings back to lance", cat, generated_count)
    else:
        logger.info("[layout_features] %s: all %d embeddings cached, skip generation", cat, total)
        if progress_callback:
            progress_callback(total, total, f"{cat} embedding 全部已缓存 ({total})")

    # 4. 过滤有效 embedding
    valid_embeddings = [emb for emb in embeddings if emb and len(emb) > 0]
    if len(valid_embeddings) < 2:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": total, "clusters": {}, "labels": {}}

    # 5. MiniBatchKMeans 聚类（与 Page-Level 相同：find_optimal_clusters + KMeansClusterer）
    optimal_k = find_optimal_clusters(valid_embeddings, max_clusters=min(max_k, len(valid_embeddings)))
    if progress_callback:
        progress_callback(0, 1, f"{cat} 聚类 K={optimal_k}...")

    clusterer = KMeansClusterer(n_clusters=optimal_k)
    labels_full = clusterer.fit_predict(embeddings)  # -1 for invalid embeddings
    stats = clusterer.get_cluster_stats()

    # 6. 构建 cluster 统计（bbox 均值）
    clusters_info: dict[str, dict] = {}
    label_to_idx: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels_full):
        if lbl < 0:
            continue
        label_to_idx.setdefault(lbl, []).append(i)

    for lbl, indices in label_to_idx.items():
        bbox_means: dict[str, float] = {}
        for key in _BASE_FEATURE_NAMES:
            vals = [bbox_info[idx][key] for idx in indices if key in bbox_info[idx]]
            bbox_means[key] = round(sum(vals) / max(len(vals), 1), 2)

        clusters_info[str(lbl)] = {
            "size": len(indices),
            "bbox_mean": bbox_means,
        }

    # 7. 构建 labels 映射
    labels_map: dict[str, int] = {}
    for i, lbl in enumerate(labels_full):
        if lbl >= 0:
            labels_map[keys[i]] = int(lbl)

    # 清理显存
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


def _write_embeddings_back(lance_path: Path, rows: list[dict], keys: list[str], embeddings: list):
    """将生成的 embedding 写回 lance 文件（更新 embedding 列）。"""
    import lance
    import pyarrow as pa

    try:
        ds = lance.dataset(str(lance_path))
        # 构建 key -> embedding 映射
        emb_map: dict[str, list[float]] = {}
        for key, emb in zip(keys, embeddings):
            if emb is not None:
                emb_map[key] = emb

        # 读取全部数据，更新 embedding 列，重写
        table = ds.to_table()
        sids = table.column("sample_id").to_pylist()
        bidxs = table.column("block_idx").to_pylist()

        new_embeddings = []
        for sid, bidx in zip(sids, bidxs):
            key = f"{sid}:{bidx}"
            emb = emb_map.get(key)
            new_embeddings.append(emb)

        # 用新 embedding 列替换或添加
        new_col = pa.array(new_embeddings, type=pa.large_list(pa.float32()))
        try:
            col_idx = table.schema.get_field_index("embedding")
            table = table.set_column(col_idx, "embedding", new_col)
        except (KeyError, ValueError):
            # embedding 列不存在（旧 lance），添加新列
            table = table.append_column("embedding", new_col)

        lance.write_dataset(table, str(lance_path), mode="overwrite")
    except Exception as e:
        logger.warning("[layout_features] write embeddings back failed: %s", e)


# ─── 全类型聚类入口 ──────────────────────────────────────────────────────────

def cluster_all_types(
    manifests_dir: Path,
    max_k: int = 10,
    progress_callback=None,
    extractor: CLIPEmbeddingExtractor | None = None,
) -> dict:
    """对 text / formula / table 分别做 ViT embedding + MiniBatchKMeans 聚类。

    Args:
        extractor: 可选的预加载模型，避免在线程中首次初始化 CUDA

    Returns:
        {"text": {...}, "formula": {...}, "table": {...}}
    """
    import lance as _lance

    if extractor is None:
        extractor = CLIPEmbeddingExtractor()

    # 预先统计所有类型的总 block 数（用于固定进度 total）
    cat_totals: dict[str, int] = {}
    grand_total = 0
    for cat in ("text", "formula", "table"):
        lp = manifests_dir / f"{cat}.lance"
        if lp.exists():
            try:
                n = _lance.dataset(str(lp)).count_rows()
                cat_totals[cat] = n
                grand_total += n
            except Exception:
                cat_totals[cat] = 0
        else:
            cat_totals[cat] = 0

    if grand_total == 0:
        grand_total = 1  # avoid division by zero

    result: dict[str, dict] = {}
    completed = 0  # 已完成的 block 数（跨类别累计）

    for cat in ("text", "formula", "table"):
        cat_total = cat_totals.get(cat, 0)

        def _cb(cur, tot, msg, _completed=completed, _grand=grand_total, _cat_total=cat_total):
            """进度回调：将每个类别的局部进度映射到全局 [0, grand_total] 区间。

            局部进度范围：
            - 扫描阶段: cur ∈ [0, total], tot = total
            - embedding 阶段: cur ∈ [0, need_generate], tot = need_generate
            统一映射: global_cur = _completed + cur / tot * _cat_total
            """
            if progress_callback:
                if tot > 0:
                    mapped = int(cur / tot * _cat_total)
                else:
                    mapped = 0
                global_cur = min(_completed + mapped, _grand)
                progress_callback(global_cur, _grand, f"[{cat}] {msg}")

        print(f"[DEBUG] cluster_all_types: processing cat={cat}, total={cat_total}, extractor={extractor is not None}", flush=True)
        result[cat] = cluster_by_type(manifests_dir, cat, extractor, max_k=max_k, progress_callback=_cb)
        completed += cat_total
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
