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

# SigLIP2 base 的 embedding 维度（与 schema 中 pa.list_(pa.float32(), EMB_DIM) 保持一致）
EMB_DIM = 768

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
    max_k: int = 10,
    progress_callback=None,
    sample_ids: set[str] | None = None,
    extractor: CLIPEmbeddingExtractor | None = None,
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

    # 复用外部传入的 extractor，避免重复初始化 HTTP 连接
    _local_extractor = extractor or CLIPEmbeddingExtractor()

    # 第一次读取：只读元数据（不含 embedding 和 image_data），启动快
    _META_COLUMNS = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
    rows = _read_lance_rows(lance_path, columns=_META_COLUMNS)
    if not rows:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": 0, "clusters": {}, "labels": {}}

    # 按抽样 sample_ids 过滤（如有）
    if sample_ids is not None:
        rows = [r for r in rows if r.get("sample_id", "") in sample_ids]
        if not rows:
            return {"n_clusters": 0, "silhouette": 0, "total_blocks": 0, "clusters": {}, "labels": {}}

    total = len(rows)

    # 立即发送进度，避免前端长时间显示"启动中..."
    if progress_callback:
        progress_callback(0, total, f"{cat} 扫描 {total} blocks...")

    # 1. 用 Lance filter 快速定位缺失 embedding 的行（不加载 embedding 数据本身）
    keys: list[str] = []
    bbox_info: list[dict] = []
    need_generate: list[int] = []

    try:
        import lance as _lance_scan
        ds_scan = _lance_scan.dataset(str(lance_path))
        has_emb_col = "embedding" in ds_scan.schema.names
        if has_emb_col:
            # 找出 embedding IS NULL 的行
            null_table = ds_scan.to_table(
                columns=["sample_id", "block_idx"],
                filter="embedding IS NULL"
            )
            null_keys = set(
                f"{s}:{b}" for s, b in zip(
                    null_table.column("sample_id").to_pylist(),
                    null_table.column("block_idx").to_pylist()
                )
            )
        else:
            null_keys = set()  # 没有 embedding 列 → 全部需要生成
    except Exception:
        null_keys = set()  # filter 失败时保守处理
        has_emb_col = False

    for idx, row in enumerate(rows):
        sid = row.get("sample_id", "")
        bidx = row.get("block_idx", 0)
        key = f"{sid}:{bidx}"
        keys.append(key)
        bbox_info.append(_bbox_stats(row))

        if not has_emb_col or key in null_keys:
            need_generate.append(idx)

        if progress_callback and (idx + 1) % 5000 == 0:
            progress_callback(idx + 1, total, f"{cat} 扫描 {idx + 1}/{total}")

    if progress_callback:
        n_gen = len(need_generate)
        hint = f"，{n_gen} 条需生成 embedding" if n_gen > 0 else "，embedding 已全部就绪"
        progress_callback(total, total, f"{cat} 扫描完成 ({total} blocks){hint}")

    logger.info("[layout_features] %s: scan done, %d/%d need embedding", cat, len(need_generate), total)

    # 新生成的 embedding 暂存为 dict，避免维护大数组
    new_embeddings: dict[str, list[float]] = {}  # key -> embedding

    # 2. 为缺失 embedding 的 block 生成（分批从 lance 读取 image_data）
    generated_count = 0
    unsaved_count = 0  # 自上次保存以来新生成的数量
    save_interval = get_config("embedding", "save_interval", default=10000)

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

            # 批量 ViT 推理（通过 HTTP server，无本地 CUDA 问题）
            if chunk_images:
                image_list = list(chunk_images.values())
                row_idx_list = list(chunk_images.keys())
                emb_batch_size = get_config("embedding", "batch_size", default=32)
                emb_results = _local_extractor.extract_embeddings_from_bytes_batch(image_list, batch_size=emb_batch_size)

                for row_idx, emb in zip(row_idx_list, emb_results):
                    if emb is not None:
                        new_embeddings[keys[row_idx]] = emb
                        generated_count += 1
                        unsaved_count += 1

            processed_count += len(chunk_indices)
            if progress_callback:
                skipped = processed_count - generated_count
                skip_hint = f" (跳过 {skipped} 无图)" if skipped > 0 else ""
                progress_callback(
                    processed_count, len(need_generate),
                    f"{cat} embedding {generated_count}/{len(need_generate)}{skip_hint}"
                )

            # 增量保存：每生成 save_interval 条就 merge_insert 写回 lance
            if unsaved_count >= save_interval:
                _merge_embeddings_back(lance_path, new_embeddings)
                logger.info("[layout_features] %s: incremental save %d embeddings", cat, generated_count)
                new_embeddings.clear()
                unsaved_count = 0

            # 清理本批内存
            del chunk_images

        # 3. 写回剩余未保存的 embedding
        if new_embeddings:
            _merge_embeddings_back(lance_path, new_embeddings)
            logger.info("[layout_features] %s: final save, total %d embeddings", cat, generated_count)
            new_embeddings.clear()
    else:
        logger.info("[layout_features] %s: all %d embeddings cached, skip generation", cat, total)
        if progress_callback:
            progress_callback(total, total, f"{cat} embedding 全部已缓存 ({total})")

    # 4. 从 Lance 流式加载所有有效 embedding（生成后已全部写回）
    if progress_callback:
        progress_callback(total, total, f"{cat} 加载 embedding 向量...")
    import numpy as _np
    embeddings_arr = _np.full((total, EMB_DIM), _np.nan, dtype=_np.float32)
    valid_mask = _np.zeros(total, dtype=bool)
    try:
        import lance as _lance_read
        ds_read = _lance_read.dataset(str(lance_path))
        batch_size = get_config("embedding", "load_batch_size", default=50000)
        scanner = ds_read.scanner(
            columns=["sample_id", "block_idx", "embedding"],
            filter="embedding IS NOT NULL",
            batch_size=batch_size,
        )
        key_to_idx = {key: i for i, key in enumerate(keys)}
        for batch in scanner.to_batches():
            sids = batch.column("sample_id").to_pylist()
            bidxs = batch.column("block_idx").to_pylist()
            emb_col = batch.column("embedding")
            for row_idx, (s, b) in enumerate(zip(sids, bidxs)):
                emb = emb_col[row_idx]
                if emb.is_valid and len(emb) == EMB_DIM:
                    g_idx = key_to_idx.get(f"{s}:{b}")
                    if g_idx is not None:
                        embeddings_arr[g_idx] = _np.array(emb.as_py(), dtype=_np.float32)
                        valid_mask[g_idx] = True
    except Exception as e:
        logger.warning("[layout_features] %s: 流式加载 embedding 失败: %s", cat, e)

    valid_count = int(valid_mask.sum())
    if valid_count < 2:
        return {"n_clusters": 0, "silhouette": 0, "total_blocks": total, "clusters": {}, "labels": {}}

    valid_embeddings_np = embeddings_arr[valid_mask]

    # 5. MiniBatchKMeans 聚类（与 Page-Level 相同：find_optimal_clusters + KMeansClusterer）
    if progress_callback:
        progress_callback(total, total, f"{cat} 开始聚类 ({valid_count} 有效向量)...")
    def _cluster_cb(cur, tot, msg):
        if progress_callback:
            progress_callback(total, total, f"{cat} {msg}")

    optimal_k = find_optimal_clusters(valid_embeddings_np.tolist(), max_clusters=min(max_k, valid_count), progress_callback=_cluster_cb)
    if progress_callback:
        progress_callback(total, total, f"{cat} 聚类 K={optimal_k}...")

    clusterer = KMeansClusterer(n_clusters=optimal_k)
    labels_full = clusterer.fit_predict(embeddings_arr.tolist(), progress_callback=_cluster_cb)  # -1 for invalid embeddings
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


def _merge_embeddings_back(lance_path: Path, emb_dict: dict[str, list[float]]):
    """将新 embedding 写回 lance。

    优先用 merge_insert（增量更新）。若数据集 embedding 列仍为 legacy
    large_list 类型导致 merge_insert 失败（definition buffer 超限），
    则自动降级为一次性 overwrite —— 同时完成 schema 迁移和 embedding 写入。

    Args:
        lance_path: lance 文件路径
        emb_dict: key("sample_id:block_idx") -> embedding 映射
    """
    import lance
    import pyarrow as pa

    if not emb_dict:
        return

    # Fixed-size list type — no definition buffer, no offset array.
    emb_type = pa.list_(pa.float32(), EMB_DIM)

    try:
        ds = lance.dataset(str(lance_path))

        # 构建更新表：只包含需要更新的行
        update_sids = []
        update_bidxs = []
        update_embs = []
        for key, emb in emb_dict.items():
            sid, bidx_str = key.rsplit(":", 1)
            bidx = int(bidx_str)
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            elif not isinstance(emb, list):
                emb = list(emb)
            update_sids.append(sid)
            update_bidxs.append(bidx)
            update_embs.append([float(x) for x in emb])

        update_table = pa.table({
            "sample_id": pa.array(update_sids, type=pa.large_string()),
            "block_idx": pa.array(update_bidxs, type=pa.int32()),
            "embedding": pa.array(update_embs, type=emb_type),
        })

        # merge_insert（fast path — 要求 embedding 列已是 fixed-size list）
        try:
            ds.merge_insert(on=["sample_id", "block_idx"]) \
                .when_matched_update_all() \
                .execute(update_table)
        except Exception as merge_err:
            logger.warning(
                "[layout_features] merge_insert failed (will fallback to overwrite): %s",
                merge_err,
            )
            # ── fallback：全量读取 + 合并 + overwrite ───────────────────
            # 如果 embedding 列是 legacy large_list，此处同时完成 schema 迁移。
            logger.info("[layout_features] fallback: reading full table from %s ...", lance_path.name)
            table = ds.to_table()
            emb_lookup = dict(zip(
                [f"{s}:{b}" for s, b in zip(update_sids, update_bidxs)],
                update_embs
            ))
            sids_all = table.column("sample_id").to_pylist()
            bidxs_all = table.column("block_idx").to_pylist()
            try:
                old_embs = table.column("embedding").to_pylist()
            except (KeyError, ValueError):
                old_embs = [None] * len(sids_all)
            merged = []
            for s, b, old in zip(sids_all, bidxs_all, old_embs):
                new_emb = emb_lookup.get(f"{s}:{b}")
                merged.append(new_emb if new_emb is not None else old)
            new_col = pa.array(merged, type=emb_type)
            try:
                col_idx = table.schema.get_field_index("embedding")
                table = table.set_column(col_idx, "embedding", new_col)
            except (KeyError, ValueError):
                table = table.append_column("embedding", new_col)
            logger.info("[layout_features] fallback: writing %d rows to %s ...", table.num_rows, lance_path.name)
            lance.write_dataset(table, str(lance_path), mode="overwrite")
    except Exception as e:
        logger.warning("[layout_features] merge embeddings back failed: %s", e)


# ─── 全类型聚类入口 ──────────────────────────────────────────────────────────

def cluster_all_types(
    manifests_dir: Path,
    max_k: int = 10,
    progress_callback=None,
    sample_ids: set[str] | None = None,
) -> dict:
    """对 text / formula / table 分别做 ViT embedding + MiniBatchKMeans 聚类。

    Args:
        sample_ids: 可选，仅处理这些 sample_id 对应的 block（过滤旧数据）

    Returns:
        {"text": {...}, "formula": {...}, "table": {...}}
    """
    import lance as _lance

    # 预先统计所有类型的总 block 数（用于固定进度 total）
    cat_totals: dict[str, int] = {}
    grand_total = 0
    for cat in ("text", "formula", "table"):
        lp = manifests_dir / f"{cat}.lance"
        if lp.exists():
            try:
                ds = _lance.dataset(str(lp))
                if sample_ids is not None:
                    # 用 Lance filter 计数，不加载全量 sample_id
                    sids_str = ",".join(f"'{s}'" for s in sample_ids)
                    n = ds.count_rows(filter=f"sample_id IN ({sids_str})")
                else:
                    n = ds.count_rows()
                cat_totals[cat] = n
                grand_total += n
            except Exception:
                cat_totals[cat] = 0
        else:
            cat_totals[cat] = 0

    if grand_total == 0:
        grand_total = 1  # avoid division by zero

    # 发送初始进度（让前端立即响应，current=1 确保进度条可见）
    if progress_callback:
        progress_callback(1, grand_total, "初始化 embedding 客户端...")

    # 创建一次 extractor，三个类型共用（避免 3 次 HTTP health check）
    _shared_extractor = CLIPEmbeddingExtractor()

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
                    mapped = min(int(cur / tot * _cat_total), _cat_total)
                else:
                    mapped = 0
                global_cur = min(_completed + mapped, _grand)
                # 确保初始化阶段进度条可见（至少 1）
                if global_cur == 0 and _cat_total > 0:
                    global_cur = 1
                progress_callback(global_cur, _grand, f"[{cat}] {msg}")

        print(f"[DEBUG] cluster_all_types: processing cat={cat}, total={cat_total}", flush=True)
        result[cat] = cluster_by_type(manifests_dir, cat, max_k=max_k, progress_callback=_cb, sample_ids=sample_ids, extractor=_shared_extractor)
        completed += cat_total
        logger.info(
            "[layout_features] %s: %d blocks -> %d clusters (silhouette=%.3f)",
            cat, result[cat]["total_blocks"], result[cat]["n_clusters"], result[cat]["silhouette"],
        )

    # 清理
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
