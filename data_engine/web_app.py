from __future__ import annotations

import os
# 限制所有底层 C/C++ 库的并发线程为 1，避免多层线程嵌套导致 malloc 崩溃
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import sys
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Any
import shutil
import pyarrow as pa
# 可选导入 torch，仅用于 CUDA 内存清理
try:
    import torch
    HAS_TORCH = True
    # 禁用 TF32 避免 cuBLAS LT 问题
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # 预初始化 CUDA/cuBLAS，避免后台线程首次初始化时崩溃
    if torch.cuda.is_available():
        try:
            _old_omp = os.environ.get("OMP_NUM_THREADS", "1")
            os.environ["OMP_NUM_THREADS"] = "4"
            _dummy = torch.zeros(1, device="cuda")
            _dummy = _dummy @ _dummy.unsqueeze(0)
            del _dummy
            torch.cuda.synchronize()
            os.environ["OMP_NUM_THREADS"] = _old_omp
            print("[CUDA] cuBLAS 预初始化成功", file=sys.stderr)
        except Exception as _e:
            print(f"[CUDA] cuBLAS 预初始化失败 (非致命): {_e}", file=sys.stderr)
except ImportError:
    HAS_TORCH = False
    torch = None

# 预连接 embedding server（HTTP 模式，无本地 CUDA 依赖）
try:
    from data_engine.embedding import CLIPEmbeddingExtractor
    _embedding_client = CLIPEmbeddingExtractor()
    print(f"[Embedding] 客户端已初始化: {_embedding_client.server_url}", file=sys.stderr)
except Exception as _e:
    print(f"[Embedding] 客户端初始化失败 (非致命): {_e}", file=sys.stderr)
    _embedding_client = None

# 设置 Lance 内存限制（必须在 import lance 之前）
from data_engine.config import get_config
_memory_limit = get_config("lance", "memory_limit", default=None)
if _memory_limit:
    os.environ["LANCE_DEFAULT_MEMORY_LIMIT"] = str(_memory_limit)

import lance
import threading
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import Response
from data_engine.manifests import (
    read_manifest, write_manifest, find_stage_manifest, manifest_count,
    _lance_write_lock, _safe_write_lance,
)
from data_engine.registry import SourceRegistry
from data_engine.status import collect_global_status, format_status_report, invalidate_status_cache
from data_engine.progress_tracker import progress_tracker

app = FastAPI(title="Data Engine Web Console", version="1.0.0")

# 启动时清理 stale 任务（重启后线程已不存在）
for _tid, _task in progress_tracker.get_all_tasks().items():
    if _task.status.value in ("running", "pending"):
        progress_tracker.stop_task(_tid, "服务重启，任务已中断")
        print(f"[startup] marked stale task {_tid} as stopped", file=__import__('sys').stderr)

# Static files and templates
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Global registry
registry = SourceRegistry(Path("sources.yaml"))


def _migrate_sampling_caches() -> None:
    """One-time: merge dual cache files into single bucket_samples.json."""
    import sys as _sys
    for src_info in registry.scan():
        try:
            root = Path(src_info.root_path)
            batch_dirs = []
            if (root / "manifests").exists():
                batch_dirs.append(("", root))
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "manifests").exists():
                    batch_dirs.append((d.name, d))
            for _bid, bdir in batch_dirs:
                regular = bdir / "artifacts" / "bucket_samples.json"
                diff = bdir / "artifacts" / "bucket_samples_diff.json"
                if diff.exists() and not regular.exists():
                    diff.rename(regular)
                    print(f"[migrate] renamed {diff} -> {regular}", file=_sys.stderr)
                elif diff.exists() and regular.exists():
                    diff.unlink()
                    print(f"[migrate] deleted stale {diff}", file=_sys.stderr)
        except Exception as e:
            print(f"[migrate] skip {src_info.source_id}: {e}", file=_sys.stderr)


_migrate_sampling_caches()


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, refresh: int = 0):
    """数据地图首页 - The Map"""
    try:
        global_status = collect_global_status(registry, force_refresh=bool(refresh))
        
        batches_json = [
            {"source_id": b.source_id, "batch_id": b.batch_id, "sample_count": b.sample_count,
             "category": b.category, "stage_status": b.stage_status}
            for b in global_status.batches
        ]
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "sources": global_status.sources,
                "batches": global_status.batches,
                "batches_json": batches_json,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """获取全局状态API"""
    try:
        global_status = collect_global_status(registry)
        return {
            "sources": [item.model_dump(mode="json") for item in global_status.sources],
            "batches": [item.model_dump(mode="json") for item in global_status.batches],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sources/{source_id}/batches")
async def get_source_batches(source_id: str):
    """获取指定数据源的批次信息"""
    try:
        global_status = collect_global_status(registry)
        source_batches = [b for b in global_status.batches if b.source_id == source_id]
        return {
            "source_id": source_id,
            "batches": [item.model_dump(mode="json") for item in source_batches]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/batches/{batch_id}/samples")
async def get_batch_samples(batch_id: str):
    """获取批次样本信息"""
    try:
        # 这里需要实现从manifest读取样本的逻辑
        # 暂时返回空列表
        return {
            "batch_id": batch_id,
            "samples": [],
            "total": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data-filter", response_class=HTMLResponse)
async def data_filter(request: Request):
    """数据筛选页面"""
    try:
        global_status = collect_global_status(registry)
        batches_json = [
            {"source_id": b.source_id, "batch_id": b.batch_id, "sample_count": b.sample_count}
            for b in global_status.batches
        ]
        return templates.TemplateResponse(
            request,
            "data_filter.html",
            {
                "sources": global_status.sources,
                "batches": global_status.batches,
                "batches_json": batches_json,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/filter-data")
async def filter_data(
    sources: str = None,
    batches: str = None,
    statuses: str = None,
    page: int = 1,
    page_size: int = 10
):
    """筛选数据API"""
    try:
        # 解析筛选参数
        source_list = sources.split(',') if sources else []
        batch_list = batches.split(',') if batches else []
        status_list = statuses.split(',') if statuses else []
        
        # 读取所有manifest文件
        all_samples = []
        global_status = collect_global_status(registry)
        
        for batch in global_status.batches:
            # 应用筛选条件
            if source_list and batch.source_id not in source_list:
                continue
            if batch_list and batch.batch_id not in batch_list:
                continue
            
            # 读取批次manifest
            batch_dir = None
            try:
                source_config = registry.get(batch.source_id)
                batch_dir = source_config.resolve_batch_dir(batch.batch_id)
            except (KeyError, FileNotFoundError):
                pass
            
            if not batch_dir:
                continue

            manifests_dir = batch_dir / "manifests"
            manifest_path = find_stage_manifest(manifests_dir, "ingest")
            if manifest_path:
                records = read_manifest(manifest_path)
                for record in records:
                    # 应用数据状态筛选
                    if status_list:
                        has_embedding = record.get("embedding") is not None
                        has_cluster = record.get("cluster_id") is not None
                        
                        if "has_embedding" in status_list and not has_embedding:
                            continue
                        if "has_cluster" in status_list and not has_cluster:
                            continue
                        if "no_cluster" in status_list and has_cluster:
                            continue
                    
                    # 添加到结果
                    all_samples.append({
                        "sample_id": record.get("sample_id"),
                        "source_id": record.get("source_id"),
                        "batch_id": record.get("batch_id"),
                        "input_type": record.get("input_type"),
                        "cluster_id": record.get("cluster_id"),
                        "difficulty": record.get("difficulty"),
                    })
        
        # 分页处理
        total = len(all_samples)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_samples = all_samples[start_idx:end_idx]
        
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "samples": page_samples
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/batches/{batch_id}/hard-cases")
async def get_hard_cases(batch_id: str):
    """获取困难样本"""
    try:
        # 这里需要实现从cmcv.parquet读取hard样本的逻辑
        return {
            "batch_id": batch_id,
            "hard_cases": [],
            "total": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/{source_id}")
async def start_ingest(source_id: str, batch_id: str = None):
    """启动INGEST任务"""
    try:
        
        # 检查是否已有运行中的任务
        if batch_id:
            task_id = f"ingest_{source_id}_{batch_id}"
            existing = progress_tracker.get_task(task_id)
            if existing and existing.status.value in ["running", "pending"]:
                # 检查线程是否还活着，如果死了则标记为stopped
                alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
                if task_id not in alive_threads:
                    progress_tracker.stop_task(task_id, "线程已终止，任务停止")
                    print(f"[INGEST] 检测到线程 {task_id} 已终止，标记为stopped", file=sys.stderr)
                else:
                    return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}
        
        print(f"\n========== 启动INGEST任务请求 ==========")
        print(f"source_id: '{source_id}' (类型: {type(source_id).__name__})")
        
        def execute_ingest_task():
            try:
                print(f"[INGEST后台线程] 开始执行INGEST任务: {source_id}")
                # 导入外部的run_ingest函数
                from data_engine.ingest import run_ingest as ingest_run
                
                # 获取该数据源的所有批次
                global_status = collect_global_status(registry)
                print(f"[INGEST后台线程] 系统中total batches: {len(global_status.batches)}")
                
                source_batches = [b for b in global_status.batches if b.source_id == source_id]
                if batch_id:
                    source_batches = [b for b in source_batches if b.batch_id == batch_id]
                print(f"[INGEST后台线程] 找到 {len(source_batches)} 个source_id='{source_id}'的批次")
                
                if not source_batches:
                    print(f"[INGEST后台线程] ⚠ 未找到数据源 {source_id} 的批次")
                    # 列出所有存在的source_id
                    all_source_ids = set(b.source_id for b in global_status.batches)
                    print(f"[INGEST后台线程] 系统中存在的source_id: {all_source_ids}")
                    return
                
                for batch in source_batches:
                    print(f"[INGEST后台线程] 开始处理批次: {batch.batch_id}")
                    try:
                        # 调用run_ingest函数，传入registry参数
                        result = ingest_run(registry, source_id, batch.batch_id)
                        print(f"[INGEST后台线程] ✓ 批次 {batch.batch_id} 处理成功")
                        print(f"[INGEST后台线程]   - 处理记录数: {len(result.records)}")
                        print(f"[INGEST后台线程]   - 统计信息: {result.stats}")
                    except Exception as e:
                        print(f"[INGEST后台线程] ✗ 批次 {batch.batch_id} 处理失败: {e}")
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"[INGEST后台线程] ✗ INGEST任务执行失败: {e}")
                traceback.print_exc()
        
        # 在后台线程中运行
        print(f"[主线程] 启动后台线程进行INGEST处理...")
        task_id = f"ingest_{source_id}_{batch_id}"
        thread = threading.Thread(target=execute_ingest_task, name=task_id)
        thread.daemon = True
        thread.start()
        print(f"[主线程] 后台线程已启动，立即返回响应")
        
        return {
            "message": f"已启动 {source_id} 的INGEST任务",
            "status": "started"
        }
    except Exception as e:
        print(f"启动INGEST API错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/ingest/{source_id}")
async def clear_ingest(source_id: str, batch_id: str = None):
    """清除INGEST数据"""
    try:
        print(f"\n========== 开始清除INGEST数据 ==========")
        print(f"要清除的source_id: '{source_id}' (类型: {type(source_id).__name__})")
        
        global_status = collect_global_status(registry)
        
        # 调试信息：打印所有source_id
        print(f"[调试] global_status中的total batches: {len(global_status.batches)}")
        all_source_ids = set()
        for batch in global_status.batches:
            all_source_ids.add(batch.source_id)
            print(f"[调试] 批次 {batch.batch_id} 的source_id: '{batch.source_id}' (类型: {type(batch.source_id).__name__})")
        
        print(f"[调试] 系统中存在的所有source_id: {all_source_ids}")
        
        source_batches = [b for b in global_status.batches if b.source_id == source_id]
        if batch_id:
            source_batches = [b for b in source_batches if b.batch_id == batch_id]
        
        print(f"找到 {len(source_batches)} 个批次 (source_id='{source_id}')")
        
        cleared_count = 0
        for batch in source_batches:
            print(f"处理批次: {batch.batch_id}")
            try:
                source_config = registry.get(source_id)
                batch_dir = source_config.resolve_batch_dir(batch.batch_id)
                
                for fname in ["ingest.lance", "ingest.jsonl", "input_files.json"]:
                    fpath = batch_dir / "manifests" / fname
                    if fpath.exists():
                        if fpath.is_dir():
                            shutil.rmtree(fpath)
                        else:
                            fpath.unlink()
                        cleared_count += 1
                        print(f"已删除: {fpath}")
                
                stats_path = batch_dir / "artifacts" / "stats.json"
                if stats_path.exists():
                    stats_path.unlink()
                    print(f"已删除: {stats_path}")
                
                page_images_dir = batch_dir / "page_images"
                if page_images_dir.exists():
                    shutil.rmtree(page_images_dir)
                    page_images_dir.mkdir(parents=True, exist_ok=True)
                    print(f"已清空: {page_images_dir}")
            except Exception as e:
                print(f"清除批次 {batch.batch_id} 失败: {e}")
        
        # 无论是否找到批次，都要清除进度跟踪器中的相关任务
        removed_tasks_info = []
        try:
            from data_engine.progress_tracker import progress_tracker
            
            # 第1步：获取所有当前任务
            tasks = progress_tracker.get_all_tasks()
            print(f"[清除进度] 总任务数量: {len(tasks)}")
            
            # 第2步：找出所有相关的任务（source_id匹配的）
            related_tasks = []
            for task_id, task in tasks.items():
                if task.source_id == source_id:
                    if batch_id and task.batch_id != batch_id:
                        continue
                    related_tasks.append((task_id, task))
                    print(f"[清除进度] 找到相关任务: {task_id} | 类型: {task.task_type} | 状态: {task.status.value} | 批次: {task.batch_id}")
            
            print(f"[清除进度] 找到 {len(related_tasks)} 个相关任务")
            
            # 第3步：移除匹配条件的任务
            removed_tasks = 0
            for task_id, task in related_tasks:
                task_type_value = task.task_type.value if hasattr(task.task_type, 'value') else str(task.task_type)
                print(f"[清除进度] 检查任务: {task_id} | task_type: {task_type_value}")
                
                if task_type_value in ['ingest', 'embed', 'cluster']:
                    progress_tracker.remove_task(task_id)
                    removed_tasks += 1
                    task_info = {
                        "task_id": task_id,
                        "task_type": task_type_value,
                        "batch_id": task.batch_id,
                        "status": task.status.value
                    }
                    removed_tasks_info.append(task_info)
                    print(f"[清除进度] ✓ 已移除任务: {task_id}")
            
            print(f"[清除进度] 成功移除 {removed_tasks} 个任务")
            
            # 第4步：验证是否真的被移除
            updated_tasks = progress_tracker.get_all_tasks()
            remaining_related = {tid: t for tid, t in updated_tasks.items() if t.source_id == source_id and (not batch_id or t.batch_id == batch_id)}
            print(f"[清除进度] 验证：清除后仍有 {len(remaining_related)} 个相关任务")
            
            if remaining_related:
                for task_id, task in remaining_related.items():
                    print(f"[清除进度] ⚠ 仍存在任务: {task_id} | 类型: {task.task_type}")
            
            # 第5步：强制刷新进度状态
            progress_tracker._save_state()
            print(f"[清除进度] 已保存进度状态文件")
            
        except Exception as e:
            print(f"[清除进度] ✗ 清除进度任务失败: {e}")
            traceback.print_exc()
        
        invalidate_status_cache()
        return {
            "message": f"已清除 {cleared_count} 个批次的INGEST数据和 {len(removed_tasks_info)} 个相关任务进度",
            "status": "cleared",
            "cleared_batches": cleared_count,
            "removed_tasks_count": len(removed_tasks_info),
            "removed_tasks": removed_tasks_info
        }
    except Exception as e:
        print(f"清除INGEST API错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/{source_id}/restart")
async def restart_ingest(source_id: str):
    """重新执行INGEST任务"""
    try:
        # 先清除
        await clear_ingest(source_id)
        # 再启动
        return await start_ingest(source_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/embed/{source_id}")
async def start_embed(source_id: str, batch_id: str = None):
    """启动embedding生成任务"""
    try:
        
        # 检查是否已有运行中的任务
        if batch_id:
            task_id = f"embed_{source_id}_{batch_id}"
            existing = progress_tracker.get_task(task_id)
            if existing and existing.status.value in ["running", "pending"]:
                # 检查线程是否还活着，如果死了则标记为stopped
                alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
                if task_id not in alive_threads:
                    progress_tracker.stop_task(task_id, "线程已终止，任务停止")
                    print(f"[Embedding] 检测到线程 {task_id} 已终止，标记为stopped", file=sys.stderr)
                else:
                    return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}
        
        print(f"\n========== 启动Embedding任务请求 ==========")
        print(f"source_id: '{source_id}' (类型: {type(source_id).__name__})")
        
        def execute_embed_task():
            try:
                print(f"[Embedding后台线程] 开始执行Embedding任务: {source_id}")
                # 导入embedding函数
                from data_engine.embedding import extract_embeddings_for_records
                from data_engine.manifests import read_manifest, write_manifest, find_stage_manifest
                
                # 获取该数据源的所有批次
                global_status = collect_global_status(registry)
                print(f"[Embedding后台线程] 系统中total batches: {len(global_status.batches)}")
                
                source_batches = [b for b in global_status.batches if b.source_id == source_id]
                if batch_id:
                    source_batches = [b for b in source_batches if b.batch_id == batch_id]
                print(f"[Embedding后台线程] 找到 {len(source_batches)} 个source_id='{source_id}'的批次")
                
                if not source_batches:
                    print(f"[Embedding后台线程] ⚠ 未找到数据源 {source_id} 的批次")
                    all_source_ids = set(b.source_id for b in global_status.batches)
                    print(f"[Embedding后台线程] 系统中存在的source_id: {all_source_ids}")
                    return
                
                for batch in source_batches:
                    print(f"[Embedding后台线程] 开始处理批次: {batch.batch_id}")
                    try:
                        # 获取source信息
                        source = registry.get(source_id)
                        batch_dir = source.resolve_batch_dir(batch.batch_id)
                        manifests_dir = batch_dir / "manifests"
                        manifest_path = find_stage_manifest(manifests_dir, "ingest")
                        
                        if not manifest_path or not manifest_path.exists():
                            print(f"[Embedding后台线程] ⚠ manifest文件不存在")
                            continue
                        
                        # 使用 manifest_count 获取总数，避免读取全部数据导致 overflow
                        total_count = manifest_count(manifest_path)
                        task_id = f"embed_{source_id}_{batch.batch_id}"
                        
                        # 开始任务
                        progress_tracker.start_task(
                            task_id=task_id,
                            task_type="embed",
                            source_id=source_id,
                            batch_id=batch.batch_id,
                            total=total_count,
                            message=f"开始提取 {total_count} 个样本的embedding"
                        )
                        
                        # 分批读取和处理记录，避免 offset overflow
                        with _lance_write_lock:
                            ds = lance.dataset(str(manifest_path))
                        chunk_size = get_config("embedding", "batch_update_interval", default=10000)
                        
                        # 创建一次 extractor，复用模型
                        from data_engine.embedding import CLIPEmbeddingExtractor
                        extractor = CLIPEmbeddingExtractor()
                        
                        # 用 count_rows 计算实际已处理数，作为续跑起点
                        try:
                            with _lance_write_lock:
                                processed_count = ds.count_rows(filter="embedding IS NOT NULL")
                            start_offset = (processed_count // chunk_size) * chunk_size
                            print(f"[Embedding] 实际已有 {processed_count} 条记录含embedding，从 offset={start_offset} 开始")
                        except Exception:
                            start_offset = 0
                        
                        for offset in range(start_offset, total_count, chunk_size):
                            if progress_tracker.is_stopped(task_id):
                                print(f"[Embedding] 收到停止信号，已处理到第 {offset} 条", file=sys.stderr)
                                progress_tracker.stop_task(task_id, f"用户停止，已处理 {offset}/{total_count} 个样本")
                                break
                            
                            limit = min(chunk_size, total_count - offset)
                            with _lance_write_lock:
                                chunk_table = ds.to_table(offset=offset, limit=limit)
                            chunk_records = chunk_table.to_pylist()
                            
                            # 跳过已处理的记录（在当前 chunk 内检查）
                            records_to_process = [r for r in chunk_records if r.get("embedding") is None]
                            if not records_to_process:
                                print(f"[Embedding] 跳过第 {offset+1}-{offset+len(chunk_records)} 条（已处理）")
                                continue
                            
                            print(f"[Embedding] 处理第 {offset+1}-{offset+len(chunk_records)} 条，需处理 {len(records_to_process)} 条...")
                            updated_records = extract_embeddings_for_records(records_to_process, batch_dir, extractor=extractor, task_id=task_id, offset=offset)
                            
                            # 构建 sample_id -> embedding 映射
                            embedding_map = {r["sample_id"]: r.get("embedding") for r in updated_records}
                            
                            # 用 Lance update 原地更新 embedding
                            update_ids = list(embedding_map.keys())
                            update_embeddings = [embedding_map[sid] for sid in update_ids]
                            embedding_dim = get_config("embedding", "embedding_dim", default=768)
                            update_table = pa.table({
                                "sample_id": pa.array(update_ids, type=pa.large_string()),
                                "embedding": pa.array(update_embeddings, type=pa.list_(pa.float32(), embedding_dim)),
                            })
                            # 写入 Lance 时加锁，避免多线程并发写入导致 glibc 内存死锁
                            with _lance_write_lock:
                                current_ds = lance.dataset(str(manifest_path))
                                current_ds.merge_insert("sample_id").when_matched_update_all().execute(update_table)
                            print(f"[Embedding] 已更新 {len(update_ids)} 条记录的 embedding")

                            # merge_insert 后重新获取 dataset（版本已变）
                            with _lance_write_lock:
                                ds = lance.dataset(str(manifest_path))
                            
                            # 保存 chunk 长度用于进度更新
                            chunk_len = len(chunk_records)
                            
                            # 清理 chunk 内存，避免大规模处理时 segfault
                            del chunk_table, chunk_records, records_to_process, updated_records, embedding_map, update_table
                            gc.collect()
                            if HAS_TORCH and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # 更新进度到下一个 chunk 位置
                            next_offset = offset + chunk_len
                            progress_tracker.update_progress(
                                task_id=task_id,
                                current=next_offset,
                                message=f"已处理 {next_offset}/{total_count}"
                            )
                        
                        task_obj = progress_tracker.get_task(task_id)
                        if task_obj and task_obj.status.value == "stopped":
                            print(f"[Embedding] 任务已停止，不标记完成", file=sys.stderr)
                        else:
                            # 统计实际成功的embedding数
                            with _lance_write_lock:
                                verify_table = ds.to_table(columns=["embedding"])
                            total_rows = verify_table.num_rows
                            actual_success = total_rows - verify_table.column("embedding").null_count
                            msg = f"完成: {actual_success}/{total_rows} 条成功"
                            if actual_success == 0:
                                msg += " (全部失败，请检查GPU显存)"
                            progress_tracker.complete_task(
                                task_id=task_id,
                                message=msg
                            )
                        invalidate_status_cache()
                        
                        print(f"[Embedding后台线程] ✓ 批次 {batch.batch_id} embedding生成成功")
                        print(f"[Embedding后台线程]   - 已处理记录数: {actual_success}/{total_rows}")
                        
                    except Exception as e:
                        print(f"[Embedding后台线程] ✗ 批次 {batch.batch_id} embedding生成失败: {e}")
                        traceback.print_exc()
                        progress_tracker.fail_task(
                            task_id=f"embed_{source_id}_{batch.batch_id}",
                            error_message=str(e)
                        )
                        
            except Exception as e:
                print(f"[Embedding后台线程] ✗ Embedding任务执行失败: {e}")
                traceback.print_exc()
        
        # 在后台线程中运行
        print(f"[主线程] 启动后台线程进行Embedding处理...")
        task_id = f"embed_{source_id}_{batch_id}"
        thread = threading.Thread(target=execute_embed_task, name=task_id)
        thread.daemon = True
        thread.start()
        print(f"[主线程] 后台线程已启动，立即返回响应")
        
        return {
            "message": f"已启动 {source_id} 的Embedding任务",
            "status": "started"
        }
    except Exception as e:
        print(f"启动Embedding API错误: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop/{source_id}")
async def stop_task(source_id: str, batch_id: str = None):
    """停止运行中的任务"""
    if not batch_id:
        raise HTTPException(status_code=400, detail="batch_id is required")
    
    stopped = []
    for task_type in ["ingest", "embed", "cluster", "element_sample", "cmcv", "layout_batch", "repair"]:
        task_id = f"{task_type}_{source_id}_{batch_id}"
        task = progress_tracker.get_task(task_id)
        if task and task.status.value in ["running", "pending"]:
            progress_tracker.request_stop(task_id)
            # 检测死线程：如果线程已不存在，直接标记为 stopped
            alive_threads = {t.name for t in threading.enumerate() if t.is_alive()}
            if task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务异常停止")
            stopped.append(task_id)
    
    if stopped:
        return {"message": f"已发送停止信号: {', '.join(stopped)}", "status": "stopping"}
    return {"message": "没有运行中的任务", "status": "idle"}


@app.post("/api/cluster/{source_id}")
async def start_cluster(source_id: str, batch_id: str = None, n_clusters: int = 5, auto_optimize: bool = True):
    """启动聚类任务"""
    try:

        is_all = source_id == "__all__"

        # 检查是否已有运行中的任务
        if batch_id and not is_all:
            task_id = f"cluster_{source_id}_{batch_id}"
            existing = progress_tracker.get_task(task_id)
            if existing and existing.status.value in ["running", "pending"]:
                alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
                if task_id not in alive_threads:
                    progress_tracker.stop_task(task_id, "线程已终止，任务停止")
                    print(f"[Cluster] 检测到线程 {task_id} 已终止，标记为stopped", file=sys.stderr)
                else:
                    return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}

        def execute_cluster_task():
            try:
                from data_engine.clustering import cluster_records
                from data_engine.manifests import read_manifest, write_manifest, find_stage_manifest

                global_status = collect_global_status(registry)
                if is_all:
                    source_batches = global_status.batches
                else:
                    source_batches = [b for b in global_status.batches if b.source_id == source_id]
                if batch_id:
                    source_batches = [b for b in source_batches if b.batch_id == batch_id]

                for batch in source_batches:
                    try:
                        bid = batch.batch_id
                        s_id = batch.source_id
                        source = registry.get(s_id)
                        batch_dir = source.resolve_batch_dir(bid)
                        manifests_dir = batch_dir / "manifests"
                        manifest_path = find_stage_manifest(manifests_dir, "ingest")

                        if not manifest_path or not manifest_path.exists():
                            continue

                        records = read_manifest(manifest_path)
                        t_id = f"cluster_{s_id}_{bid}"

                        progress_tracker.start_task(
                            task_id=t_id,
                            task_type="cluster",
                            source_id=s_id,
                            batch_id=bid,
                            total=len(records),
                            message=f"开始对 {len(records)} 个样本聚类"
                        )

                        updated_records, stats = cluster_records(
                            records, n_clusters=n_clusters, auto_optimize=auto_optimize)

                        with _lance_write_lock:
                            write_manifest(manifest_path, updated_records)

                        progress_tracker.complete_task(
                            task_id=t_id,
                            message=f"聚类完成: {stats.get('n_clusters', '?')} 簇, 轮廓系数 {stats.get('silhouette_score', 0):.3f}"
                        )
                        invalidate_status_cache()

                    except Exception as e:
                        print(f"[聚类后台线程] ✗ 批次 {batch.batch_id} 聚类失败: {e}")
                        traceback.print_exc()
                        progress_tracker.fail_task(
                            task_id=f"cluster_{batch.source_id}_{batch.batch_id}",
                            error_message=str(e)
                        )

            except Exception as e:
                print(f"[聚类后台线程] ✗ 聚类任务执行失败: {e}")
                traceback.print_exc()

        task_id = f"cluster_{source_id}_{batch_id}"
        thread = threading.Thread(target=execute_cluster_task, name=task_id)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 {label} 的聚类任务", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index/{source_id}")
async def start_index_build(source_id: str, batch_id: str = None, num_partitions: int = 256, num_sub_vectors: int = 16):
    """构建向量索引（IVF_PQ）"""
    try:
        task_id = f"index_{source_id}_{batch_id}" if batch_id else f"index_{source_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
            if task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务停止")
            else:
                return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}

        def execute_index_task():
            try:
                from data_engine.index import build_ivf_pq_index

                global_status = collect_global_status(registry)
                source_batches = [b for b in global_status.batches if b.source_id == source_id]
                if batch_id:
                    source_batches = [b for b in source_batches if b.batch_id == batch_id]

                for batch in source_batches:
                    bid = batch.batch_id
                    s_id = batch.source_id
                    source = registry.get(s_id)
                    batch_dir = source.resolve_batch_dir(bid)
                    manifest_path = batch_dir / "manifests" / "ingest.lance"

                    if not manifest_path.exists():
                        print(f"[Index] 跳过 {bid}: ingest.lance 不存在", file=sys.stderr)
                        continue

                    t_id = f"index_{s_id}_{bid}"
                    total_count = manifest_count(manifest_path)

                    progress_tracker.start_task(
                        task_id=t_id,
                        task_type="index",
                        source_id=s_id,
                        batch_id=bid,
                        total=total_count,
                        message=f"开始构建向量索引 ({total_count} 条)"
                    )

                    try:
                        accelerator = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else None
                        result = build_ivf_pq_index(
                            manifest_path,
                            num_partitions=num_partitions,
                            num_sub_vectors=num_sub_vectors,
                            accelerator=accelerator,
                            task_id=t_id,
                        )

                        progress_tracker.complete_task(
                            task_id=t_id,
                            message=f"索引构建完成: {num_partitions} 分区, 版本 {result['lance_version']}"
                        )
                        invalidate_status_cache()
                        print(f"[Index] ✓ 批次 {bid} 索引构建成功", file=sys.stderr)

                    except Exception as e:
                        print(f"[Index] ✗ 批次 {bid} 索引构建失败: {e}", file=sys.stderr)
                        traceback.print_exc()
                        progress_tracker.fail_task(task_id=t_id, error_message=str(e))

            except Exception as e:
                print(f"[Index] ✗ 索引任务执行失败: {e}", file=sys.stderr)
                traceback.print_exc()

        thread = threading.Thread(target=execute_index_task, name=task_id)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 {source_id} 的向量索引构建任务", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/{source_id}/{batch_id}/stats")
async def get_index_stats(source_id: str, batch_id: str):
    """获取向量索引的桶分布统计"""
    try:
        from data_engine.index import get_index_stats as _get_stats

        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifest_path = batch_dir / "manifests" / "ingest.lance"

        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Lance 数据集不存在")

        return _get_stats(manifest_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/{source_id}/{batch_id}/partition/{partition_id}")
async def get_partition_samples(source_id: str, batch_id: str, partition_id: int, page: int = 1, page_size: int = 20):
    """获取指定分区的样本列表"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifest_path = batch_dir / "manifests" / "ingest.lance"

        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Lance 数据集不存在")

        import pyarrow as pa

        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))

            # 获取分区对应的质心向量
            try:
                stats = ds.index_statistics("idx_embedding_ivf")
                centroids = stats["indices"][0]["centroids"]
                if partition_id < 0 or partition_id >= len(centroids):
                    raise HTTPException(status_code=400, detail=f"分区ID超出范围 (0-{len(centroids)-1})")
                centroid = centroids[partition_id]
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=404, detail="未找到向量索引")

            # 用质心向量查询该分区的样本
            query_vec = pa.array(centroid, type=pa.float32())
            results = ds.to_table(
                columns=[c for c in ds.schema.names if c != "embedding"],
                nearest={"column": "embedding", "q": query_vec, "k": page_size * page},
                disable_scoring_autoprojection=True,
            )
            total = min(results.num_rows, page_size * 20)  # 限制最大返回
            offset = (page - 1) * page_size
            limit = min(page_size, total - offset)
            if limit <= 0:
                return {"partition_id": partition_id, "samples": [], "total": total, "page": page, "page_size": page_size}

            # 取当前页数据
            if offset > 0 or limit < results.num_rows:
                table = results.slice(offset, limit)
            else:
                table = results

            samples = table.to_pylist()
            for row in samples:
                for k, v in list(row.items()):
                    if k == "image_data":
                        row[k] = {"has_image": True, "size": len(v)} if v is not None else None
                    elif isinstance(v, bytes):
                        row[k] = f"<{len(v)} bytes>"
                    elif k in ("_distance",):
                        row[k] = round(float(v), 4) if v is not None else None
            return {"partition_id": partition_id, "samples": samples, "total": total, "page": page, "page_size": page_size}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/index/{source_id}/{batch_id}/sample/{sample_id}/image")
async def get_sample_image(source_id: str, batch_id: str, sample_id: str):
    """获取指定样本的图片"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifest_path = batch_dir / "manifests" / "ingest.lance"

        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Lance 数据集不存在")

        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))
            table = ds.to_table(filter=f"sample_id = '{sample_id}'", columns=["sample_id", "image_data"])
            if table.num_rows == 0:
                raise HTTPException(status_code=404, detail="样本不存在")
            row = table.to_pylist()[0]
            image_data = row.get("image_data")
            if not image_data:
                raise HTTPException(status_code=404, detail="样本无图片数据")

            import base64
            b64 = base64.b64encode(image_data).decode("utf-8")
            return {"sample_id": sample_id, "image_base64": b64, "size": len(image_data)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hard-case-review")
async def hard_case_review(request: Request):
    """Hard Case复核页面"""
    return templates.TemplateResponse(
        request,
        "hard_case_review.html",
        {}
    )


@app.get("/qa-sampling")
async def qa_sampling(request: Request):
    """QA抽检页面"""
    return templates.TemplateResponse(
        request,
        "qa_sampling.html",
        {}
    )


@app.get("/lancedb")
async def lancedb_view(request: Request):
    """LanceData 页面"""
    return templates.TemplateResponse(
        request,
        "lancedb.html",
        {}
    )


@app.get("/api/lancedb/sources")
async def lancedb_list_sources():
    """列出所有有Lance数据的数据源和批次（ingest + element）"""
    try:
        global_status = collect_global_status(registry)
        result = []
        for batch in global_status.batches:
            source_config = registry.get(batch.source_id)
            batch_dir = source_config.resolve_batch_dir(batch.batch_id)
            manifests_dir = batch_dir / "manifests"

            for dataset_name in ["ingest", "element", "text", "formula", "table"]:
                manifest_path = find_stage_manifest(manifests_dir, dataset_name)
                if not manifest_path or manifest_path.suffix != ".lance" or not manifest_path.exists():
                    continue
                try:
                    with _lance_write_lock:
                        ds = lance.dataset(str(manifest_path))
                        schema_fields = [f.name for f in ds.schema]
                        row_count = ds.count_rows()
                        versions = []
                        for v in ds.versions():
                            ts = v.get("timestamp")
                            versions.append({
                                "version": v["version"],
                                "timestamp": ts.isoformat() if ts else None,
                                "num_rows": int(v.get("metadata", {}).get("total_rows", 0)),
                            })
                        result.append({
                            "source_id": batch.source_id,
                            "batch_id": batch.batch_id,
                            "dataset": dataset_name,
                            "category": batch.category,
                            "stage_status": batch.stage_status,
                            "sample_count": row_count,
                            "current_version": getattr(ds, "version", None),
                            "columns": schema_fields,
                            "versions": versions,
                        })
                except Exception:
                    pass
        return {"sources": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lancedb/{source_id}/{batch_id}/data")
async def lancedb_query_data(
    source_id: str,
    batch_id: str,
    page: int = 1,
    page_size: int = 20,
    columns: str = None,
    search: str = None,
    search_col: str = None,
    version: int = None,
    dataset: str = "ingest",
):
    """查询Lance数据（支持 ingest / element）"""
    try:
        global_status = collect_global_status(registry)
        batch = next(
            (b for b in global_status.batches if b.source_id == source_id and b.batch_id == batch_id),
            None,
        )
        if not batch:
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

        source_config = registry.get(source_id)
        batch_dir = source_config.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, dataset)
        if not manifest_path or manifest_path.suffix != ".lance":
            raise HTTPException(status_code=404, detail=f"{dataset}.lance 不存在")

        with _lance_write_lock:
            if version:
                ds = lance.dataset(str(manifest_path)).checkout_version(version)
            else:
                ds = lance.dataset(str(manifest_path))
            total = ds.count_rows()

            select_cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
            if select_cols and "sample_id" not in select_cols:
                select_cols.append("sample_id")

            schema_fields = [f.name for f in ds.schema]

            if search and search_col and search_col in schema_fields:
                safe_search = search.replace("'", "''")
                try:
                    results = ds.to_table(
                        columns=select_cols,
                        filter=f"contains(cast({search_col} as string), '{safe_search}')",
                    )
                except Exception:
                    results = ds.to_table(columns=select_cols)
                total = results.num_rows

                start = (page - 1) * page_size
                end = min(start + page_size, total)
                page_rows = results.to_pylist()[start:end]
                for row in page_rows:
                    for k, v in list(row.items()):
                        if k == "image_data":
                            row[k] = {"has_image": True, "size": len(v)} if v is not None else None
                        elif isinstance(v, bytes):
                            row[k] = f"<{len(v)} bytes>"

                return {
                    "columns": list(results.column_names),
                    "schema": schema_fields,
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": (total + page_size - 1) // page_size,
                    "data": page_rows,
                }

            start = (page - 1) * page_size
            if start >= total:
                page_rows = []
            else:
                limit = min(page_size, total - start)
                results = ds.to_table(offset=start, limit=limit, columns=select_cols)
                page_rows = results.to_pylist()
                for row in page_rows:
                    for k, v in list(row.items()):
                        if k == "image_data":
                            row[k] = {"has_image": True, "size": len(v)} if v is not None else None
                        elif isinstance(v, bytes):
                            row[k] = f"<{len(v)} bytes>"

            return {
                "columns": select_cols or schema_fields,
                "schema": schema_fields,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size,
                "data": page_rows,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/lancedb/{source_id}/{batch_id}/image/{sample_id}")
async def lancedb_get_image(source_id: str, batch_id: str, sample_id: str, version: int = None, dataset: str = "ingest", block_idx: int = None):
    """获取样本图片。
    - ingest/element: 从 ingest.lance 读整页图
    - text/formula/table: 优先从 category lance 读裁剪后的 block 图（需 block_idx），
      若无则 fallback 到 ingest.lance
    """
    try:
        source_config = registry.get(source_id)
        batch_dir = source_config.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        image_bytes = None

        # text/formula/table: 尝试从 category lance 读裁剪后的 block 图
        if dataset in ("text", "formula", "table"):
            cat_path = find_stage_manifest(manifests_dir, dataset)
            if cat_path and cat_path.suffix == ".lance" and cat_path.exists():
                try:
                    with _lance_write_lock:
                        ds = lance.dataset(str(cat_path))
                        if version:
                            ds = ds.checkout_version(version)
                        cols = ["sample_id", "image_data"]
                        if "block_idx" in ds.schema.names:
                            cols.append("block_idx")
                        flt = f"sample_id = '{sample_id}'"
                        if block_idx is not None and "block_idx" in ds.schema.names:
                            flt += f" AND block_idx = {block_idx}"
                        tbl = ds.to_table(columns=cols, filter=flt)
                        if tbl.num_rows > 0 and tbl.column("image_data")[0].as_py() is not None:
                            image_bytes = tbl.column("image_data")[0].as_py()
                except Exception as e:
                    print(f"[lancedb] read {dataset}.lance image failed: {e}", file=sys.stderr)

        # fallback: 从 ingest.lance 读整页图（始终用最新版本，不传 category 的 version）
        if image_bytes is None:
            manifest_path = find_stage_manifest(manifests_dir, "ingest")
            if not manifest_path or manifest_path.suffix != ".lance":
                raise HTTPException(status_code=404, detail="ingest.lance 不存在")

            with _lance_write_lock:
                ds = lance.dataset(str(manifest_path))

                results = ds.to_table(
                    columns=["sample_id", "image_data"],
                    filter=f"sample_id = '{sample_id}'",
                )
                if results.num_rows == 0:
                    raise HTTPException(status_code=404, detail="Sample not found")

                image_bytes = results.column("image_data")[0].as_py()
                if image_bytes is None:
                    raise HTTPException(status_code=404, detail="No image data")

        return Response(content=image_bytes, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progress")
async def get_progress():
    """获取所有任务进度（自动检测死线程）"""
    try:
        # 自动检测死线程：如果任务状态为 running/pending 但线程已死，标记为 stopped
        alive_threads = {t.name for t in threading.enumerate() if t.is_alive()}
        for task_id, task in progress_tracker.get_all_tasks().items():
            if task.status.value in ("running", "pending") and task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务异常停止")

        tasks = progress_tracker.get_all_tasks()
        return {
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "source_id": task.source_id,
                    "batch_id": task.batch_id,
                    "status": task.status.value,
                    "current": task.current,
                    "total": task.total,
                    "progress_percentage": task.progress_percentage,
                    "message": task.message,
                    "error_message": task.error_message,
                    "elapsed_time": task.elapsed_time,
                    "elapsed_seconds": round(task.elapsed_time, 1),
                    "start_time": task.start_time
                }
                for task_id, task in tasks.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取单个任务进度（供 TaskPoller 轮询）"""
    try:
        # 自动检测死线程
        alive_threads = {t.name for t in threading.enumerate() if t.is_alive()}
        task = progress_tracker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.status.value in ("running", "pending") and task_id not in alive_threads:
            progress_tracker.stop_task(task_id, "线程已终止，任务异常停止")
            task = progress_tracker.get_task(task_id)

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "message": task.message,
            "error_message": task.error_message,
            "progress": {
                "current": task.current,
                "total": task.total,
            },
            "elapsed_seconds": round(task.elapsed_time, 1),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progress/active")
async def get_active_progress():
    """获取活跃任务进度"""
    try:
        active_tasks = progress_tracker.get_active_tasks()
        return {
            "active_tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "source_id": task.source_id,
                    "batch_id": task.batch_id,
                    "status": task.status.value,
                    "current": task.current,
                    "total": task.total,
                    "progress_percentage": task.progress_percentage,
                    "message": task.message,
                    "elapsed_time": task.elapsed_time
                }
                for task_id, task in active_tasks.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progress/batch/{source_id}/{batch_id}")
async def get_batch_progress(source_id: str, batch_id: str):
    """获取指定批次的任务进度"""
    try:
        batch_tasks = progress_tracker.get_tasks_by_batch(source_id, batch_id)
        return {
            "source_id": source_id,
            "batch_id": batch_id,
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "status": task.status.value,
                    "current": task.current,
                    "total": task.total,
                    "progress_percentage": task.progress_percentage,
                    "message": task.message,
                    "error_message": task.error_message,
                    "elapsed_time": task.elapsed_time
                }
                for task_id, task in batch_tasks.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cmcv/{source_id}")
async def start_cmcv(source_id: str, batch_id: str = None):
    """启动 CMCV 一致性比较任务"""
    try:
        task_id = f"cmcv_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
            if task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务停止")
            else:
                return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}

        def execute_cmcv_task():
            try:
                from data_engine.ocr.cmcv import CMCVEngine

                source = registry.get(source_id)
                batch_dir = source.resolve_batch_dir(batch_id)
                manifests_dir = batch_dir / "manifests"
                ingest_path = find_stage_manifest(manifests_dir, "ingest")

                def _read_blocks_from_category_lance() -> list[dict]:
                    """从 text/formula/table.lance 读取所有 block，拼成 CMCV 所需格式"""
                    all_rows = []
                    for cat in ("text", "formula", "table"):
                        lp = manifests_dir / f"{cat}.lance"
                        if not lp.exists():
                            continue
                        try:
                            with _lance_write_lock:
                                ds = lance.dataset(str(lp))
                                all_cols = ds.schema.names
                                read_cols = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
                                for prefix in ("paddle", "glm", "self"):
                                    for suffix in ("_text", "_confidence", "_table", "_formula"):
                                        col = f"{prefix}{suffix}"
                                        if col in all_cols:
                                            read_cols.append(col)
                                rows = ds.to_table(columns=[c for c in read_cols if c in all_cols]).to_pylist()
                            for row in rows:
                                for key in ("paddle_table", "glm_table", "self_table",
                                            "paddle_formula", "glm_formula", "self_formula"):
                                    if key in row and isinstance(row[key], str) and row[key]:
                                        try:
                                            row[key] = json.loads(row[key])
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                                all_rows.append(row)
                        except Exception as e:
                            print(f"[CMCV] 读 {cat}.lance 失败: {e}", file=sys.stderr)
                    return all_rows

                element_rows = _read_blocks_from_category_lance()
                if not element_rows:
                    progress_tracker.fail_task(task_id, "text/formula/table.lance 中无可用 block")
                    return

                progress_tracker.start_task(
                    task_id=task_id, task_type="cmcv",
                    source_id=source_id, batch_id=batch_id,
                    total=len(element_rows), message="开始一致性比较",
                )

                cmcv = CMCVEngine()
                updated_rows, page_tiers = cmcv.process_element_batch(element_rows)

                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    if not lp.exists():
                        continue
                    try:
                        with _lance_write_lock:
                            ds = lance.dataset(str(lp))
                            col_names = set(ds.schema.names)
                            if "consistency_pattern" not in col_names or "block_diff_json" not in col_names:
                                continue
                        update_map: dict[str, dict] = {}
                        for r in updated_rows:
                            key = (r["sample_id"], r["block_idx"])
                            if r.get("consistency_pattern"):
                                update_map.setdefault("consistency_pattern", {})[key] = r["consistency_pattern"]
                            if r.get("block_diff_json"):
                                update_map.setdefault("block_diff_json", {})[key] = r["block_diff_json"]
                        if not update_map:
                            continue
                        with _lance_write_lock:
                            ds = lance.dataset(str(lp))
                            t = ds.to_table()
                            patterns = t.column("consistency_pattern").to_pylist() if "consistency_pattern" in t.column_names else [None] * len(t)
                            diffs = t.column("block_diff_json").to_pylist() if "block_diff_json" in t.column_names else [None] * len(t)
                            sids = t.column("sample_id").to_pylist()
                            bidxs = t.column("block_idx").to_pylist()
                            pat_map = update_map.get("consistency_pattern", {})
                            diff_map = update_map.get("block_diff_json", {})
                            new_patterns = []
                            new_diffs = []
                            for i in range(len(t)):
                                key = (sids[i], bidxs[i])
                                new_patterns.append(pat_map.get(key, patterns[i]))
                                new_diffs.append(diff_map.get(key, diffs[i]))
                            import pyarrow as pa
                            update_table = pa.table({
                                "sample_id": pa.array(sids, type=pa.large_string()),
                                "block_idx": pa.array(bidxs, type=pa.int32()),
                                "consistency_pattern": pa.array(new_patterns, type=pa.large_string()),
                                "block_diff_json": pa.array(new_diffs, type=pa.large_string()),
                            })
                            ds.merge_insert(["sample_id", "block_idx"]).when_matched_update_all().execute(update_table)
                        print(f"[CMCV] 已将一致性结果写回 {cat}.lance", file=sys.stderr)
                    except Exception as e:
                        print(f"[CMCV] 写回 {cat}.lance 失败: {e}", file=sys.stderr)

                if ingest_path and ingest_path.exists() and page_tiers:
                    import pyarrow as pa
                    sample_ids = list(page_tiers.keys())
                    tiers = [page_tiers[sid] for sid in sample_ids]
                    update_table = pa.table({
                        "sample_id": pa.array(sample_ids, type=pa.large_string()),
                        "difficulty": pa.array(tiers, type=pa.large_string()),
                    })
                    with _lance_write_lock:
                        ds = lance.dataset(str(ingest_path))
                        ds.merge_insert(["sample_id"]).when_matched_update_all().execute(update_table)

                progress_tracker.complete_task(task_id=task_id, message=f"完成 {len(page_tiers)} 页面")
                invalidate_status_cache()

            except Exception as e:
                print(f"[CMCV] 任务失败: {e}", file=sys.stderr)
                traceback.print_exc()
                progress_tracker.fail_task(task_id=task_id, error_message=str(e))

        thread = threading.Thread(target=execute_cmcv_task, name=task_id)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 {source_id} 的 CMCV 任务", "status": "started", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cmcv/{source_id}/{batch_id}/results")
async def get_cmcv_results(source_id: str, batch_id: str, tier: str = None):
    """获取 CMCV 比较结果"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        rows = []
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                with _lance_write_lock:
                    ds = lance.dataset(str(lp))
                    col_names = set(ds.schema.names)
                    read_cols = ["sample_id", "block_idx", "block_type",
                                 "consistency_pattern", "block_diff_json"]
                    for prefix in ("paddle", "glm", "self"):
                        col = f"{prefix}_text"
                        if col in col_names:
                            read_cols.append(col)
                    rows.extend(ds.to_table(columns=[c for c in read_cols if c in col_names]).to_pylist())
            except Exception:
                pass

        if tier:
            rows = [r for r in rows if r.get("consistency_pattern") == tier]

        page_stats: dict[str, dict] = {}
        for row in rows:
            sid = row["sample_id"]
            if sid not in page_stats:
                page_stats[sid] = {"sample_id": sid, "blocks": [], "worst_pattern": "all_agree"}
            page_stats[sid]["blocks"].append(row)
            pat = row.get("consistency_pattern", "")
            if pat == "all_disagree":
                page_stats[sid]["worst_pattern"] = "all_disagree"
            elif pat == "partial_agree" and page_stats[sid]["worst_pattern"] != "all_disagree":
                page_stats[sid]["worst_pattern"] = "partial_agree"

        tier_histogram = {"easy": 0, "medium": 0, "hard": 0}
        for ps in page_stats.values():
            wp = ps["worst_pattern"]
            if wp == "all_agree":
                tier_histogram["easy"] += 1
            elif wp == "partial_agree":
                tier_histogram["medium"] += 1
            else:
                tier_histogram["hard"] += 1

        return {
            "total_blocks": len(rows),
            "total_pages": len(page_stats),
            "tier_histogram": tier_histogram,
            "pages": list(page_stats.values())[:100],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cmcv/{source_id}/{batch_id}/sample/{sample_id}/compare")
async def get_sample_compare(source_id: str, batch_id: str, sample_id: str):
    """获取单样本的三模型对比"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        sample_blocks = []
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                with _lance_write_lock:
                    ds = lance.dataset(str(lp))
                    col_names = set(ds.schema.names)
                    read_cols = ["sample_id", "block_idx", "block_type", "bbox_json",
                                 "paddle_text", "glm_text", "self_text",
                                 "paddle_table", "glm_table", "self_table",
                                 "paddle_formula", "glm_formula", "self_formula",
                                 "consistency_pattern", "block_diff_json"]
                    available = [c for c in read_cols if c in col_names]
                    tbl = ds.to_table(columns=available)
                    t_sample_id = tbl.column("sample_id").to_pylist()
                    for i, sid in enumerate(t_sample_id):
                        if sid == sample_id:
                            row = {col: tbl.column(col)[i].as_py() for col in available}
                            for key in ("paddle_table", "glm_table", "self_table"):
                                if key in row and isinstance(row[key], str) and row[key]:
                                    try:
                                        row[key] = json.loads(row[key])
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                            sample_blocks.append(row)
            except Exception:
                pass
        if not sample_blocks:
            raise HTTPException(status_code=404, detail=f"样本 {sample_id} 无 block 数据")

        return {
            "sample_id": sample_id,
            "blocks": sorted(sample_blocks, key=lambda r: r.get("block_idx", 0)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/category-stats/{source_id}/{batch_id}")
async def get_category_stats(source_id: str, batch_id: str):
    """获取 text/formula/table 三个 Lance 的统计"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        stats = {}
        for cat in ("text", "formula", "table"):
            lance_path = manifests_dir / f"{cat}.lance"
            if lance_path.exists():
                ds = lance.dataset(str(lance_path))
                n = ds.count_rows()
                cols = ds.schema.names
                no_img = 0
                if "image_data" in cols:
                    try:
                        no_img = ds.count_rows("image_data IS NULL")
                    except Exception:
                        pass
                stats[cat] = {"count": n, "columns": cols, "no_image": no_img}
            else:
                stats[cat] = {"count": 0, "columns": [], "no_image": 0}

        return {"source_id": source_id, "batch_id": batch_id, "categories": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/category-stats-batch")
async def get_category_stats_batch(request: Request):
    """批量获取多个批次的 category-stats，一次请求返回所有结果。"""
    try:
        targets = await request.json()  # [{source_id, batch_id}, ...]
        results = {}
        for t in targets:
            sid = t.get("source_id", "")
            bid = t.get("batch_id", "")
            if not sid or not bid:
                continue
            try:
                source = registry.get(sid)
                batch_dir = source.resolve_batch_dir(bid)
                manifests_dir = batch_dir / "manifests"
                stats = {}
                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    if lp.exists():
                        ds = lance.dataset(str(lp))
                        n = ds.count_rows()
                        cols = ds.schema.names
                        no_img = 0
                        if "image_data" in cols:
                            try:
                                no_img = ds.count_rows("image_data IS NULL")
                            except Exception:
                                pass
                        stats[cat] = {"count": n, "columns": cols, "no_image": no_img}
                    else:
                        stats[cat] = {"count": 0, "columns": [], "no_image": 0}
                results[f"{sid}/{bid}"] = {"source_id": sid, "batch_id": bid, "categories": stats}
            except Exception as e:
                print(f"[category-stats-batch] {sid}/{bid} error: {e}", file=sys.stderr)
                results[f"{sid}/{bid}"] = {"source_id": sid, "batch_id": bid, "error": str(e)}
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _merge_all_bucket_samples(registry, count: int, force: bool) -> dict:
    """合并所有源所有批次的缓存（只返回汇总统计，不返回具体样本）"""
    merged_sizes = {}
    merged_total = 0
    batch_info = []

    for src_info in registry.scan():
        try:
            root = Path(src_info.root_path)
            batch_dirs = []
            if (root / "manifests").exists():
                batch_dirs.append(("", root))
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "manifests").exists():
                    batch_dirs.append((d.name, d))

            for bid, bdir in batch_dirs:
                # 确定 batch_id：用目录名或 resolve_batch_dir 的结果
                actual_bid = bid if bid else bdir.name
                cpath = bdir / "artifacts" / "bucket_samples.json"
                if cpath.exists():
                    try:
                        cached = json.loads(cpath.read_text(encoding="utf-8"))
                        total = cached.get("total_sampled", 0)
                        if total > 0:
                            merged_total += total
                            batch_info.append({
                                "source_id": src_info.source_id,
                                "batch_id": actual_bid,
                                "total_sampled": total,
                                "count_per_bucket": cached.get("count_per_bucket", 0),
                                "bucket_count": cached.get("bucket_count", 0),
                                "partition_tiers": cached.get("partition_tiers"),
                                "strategy": cached.get("strategy"),
                            })
                            for k, v in (cached.get("bucket_sizes") or {}).items():
                                merged_sizes[f"{src_info.source_id}/{actual_bid}/{k}"] = v
                    except Exception:
                        pass
        except Exception:
            pass

    return {
        "source_id": "all", "batch_id": "",
        "count_per_bucket": count,
        "bucket_count": len(merged_sizes),
        "bucket_sizes": merged_sizes,
        "total_sampled": merged_total,
        "batch_info": batch_info,
        "buckets": {},
    }


@app.get("/api/bucket-samples")
@app.get("/api/bucket-samples/{source_id}")
@app.get("/api/bucket-samples/{source_id}/{batch_id}")
async def get_bucket_samples(source_id: str = "", batch_id: str = "", count: int = 10, force: bool = False):
    """从向量索引的每个分区中随机抽取 N 个样本。结果缓存到 artifacts/bucket_samples.json。"""
    try:
        # 空 sourceId 时直接走合并逻辑
        if not source_id:
            return _merge_all_bucket_samples(registry, count, force)

        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path:
            raise HTTPException(status_code=404, detail="ingest manifest 不存在")

        cache_path = batch_dir / "artifacts" / "bucket_samples.json"

        if not force:
            if cache_path.exists():
                try:
                    return json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    pass

            # 空 batchId 时：合并该源所有批次的缓存
            if not batch_id:
                merged_buckets = {}
                merged_sizes = {}
                merged_total = 0
                merged_count = 0
                has_any = False

                # 确定要遍历的源
                if source_id:
                    all_sources = [source]
                else:
                    all_sources = [registry.get(s.source_id) for s in registry.scan()]

                for src in all_sources:
                    src_status = registry.scan()
                    src_batches = [b for b in src_status if b.source_id == src.source_id]
                    for b in src_batches:
                        bdir = src.resolve_batch_dir(b.batch_id)
                        cpath = bdir / "artifacts" / "bucket_samples.json"
                        if cpath.exists():
                            try:
                                cached = json.loads(cpath.read_text(encoding="utf-8"))
                                for k, v in (cached.get("buckets") or {}).items():
                                    merged_buckets[f"{b.source_id}/{b.batch_id}/{k}"] = v
                                for k, v in (cached.get("bucket_sizes") or {}).items():
                                    merged_sizes[f"{b.source_id}/{b.batch_id}/{k}"] = v
                                merged_total += cached.get("total_sampled", 0)
                                merged_count = cached.get("count_per_bucket", count)
                                has_any = True
                            except Exception:
                                pass

                if has_any:
                    return {
                        "source_id": source_id or "all",
                        "batch_id": "",
                        "count_per_bucket": merged_count,
                        "bucket_count": len(merged_sizes),
                        "bucket_sizes": merged_sizes,
                        "total_sampled": merged_total,
                        "buckets": merged_buckets,
                    }

        if not force:
            return {
                "source_id": source_id, "batch_id": batch_id,
                "count_per_bucket": count, "bucket_count": 0,
                "bucket_sizes": {}, "total_sampled": 0, "buckets": {},
            }

        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))

            indices = ds.list_indices()
            index_name = None
            for idx in indices:
                name = idx.get("name", "") if isinstance(idx, dict) else getattr(idx, "name", "")
                if name:
                    index_name = name
                    break

            if not index_name:
                return {
                    "source_id": source_id, "batch_id": batch_id,
                    "count_per_bucket": count, "bucket_count": 0,
                    "bucket_sizes": {}, "total_sampled": 0, "buckets": {},
                    "error": "未找到向量索引，请先构建索引",
                }

            stats = ds.index_statistics(index_name)
            indices_data = stats.get("indices", [{}])
            partitions = indices_data[0].get("partitions", []) if indices_data else []
            centroids = indices_data[0].get("centroids", []) if indices_data else []

            buckets = {}
            bucket_sizes = {}
            for i, part_info in enumerate(partitions):
                part_size = part_info.get("size", 0)
                bucket_sizes[f"P{i}"] = part_size
                if part_size == 0 or not centroids:
                    continue
                k = min(count, part_size)
                try:
                    centroid = centroids[i]
                    query_vec = pa.array(centroid, type=pa.float32())
                    scanner = ds.scanner(
                        nearest={"column": "embedding", "q": query_vec, "k": k},
                        disable_scoring_autoprojection=True,
                    )
                    tbl = scanner.to_table()
                    cols = [c for c in ["sample_id", "difficulty", "page_image", "input_type"] if c in tbl.column_names]
                    rows = tbl.select(cols).to_pylist()
                    buckets[f"P{i}"] = rows
                except Exception as e:
                    print(f"[bucket-samples] P{i} error: {e}", file=sys.stderr)

        total_sampled = sum(len(v) for v in buckets.values())
        result = {
            "source_id": source_id,
            "batch_id": batch_id,
            "count_per_bucket": count,
            "bucket_count": len(partitions),
            "bucket_sizes": bucket_sizes,
            "total_sampled": total_sampled,
            "buckets": buckets,
            "strategy": "regular",
            "cached_at": datetime.now().isoformat(timespec='seconds'),
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # 重新抽样时清除预览模式的 layout 结果缓存（不影响 lance 拆分文件）
        layout_cache = cache_path.parent / "layout_results.json"
        if layout_cache.exists():
            layout_cache.unlink()

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bucket-samples-batch")
async def get_bucket_samples_batch(request: Request, full: bool = False):
    """批量获取多个批次的抽样缓存摘要，一次请求返回所有结果。
    full=false 时只返回摘要（不含 buckets 详情），减少传输量。
    full=true 时返回完整数据（含 buckets）。"""
    try:
        targets = await request.json()  # [{source_id, batch_id}, ...]
        results = {}
        for t in targets:
            sid = t.get("source_id", "")
            bid = t.get("batch_id", "")
            if not sid or not bid:
                continue
            try:
                source = registry.get(sid)
                batch_dir = source.resolve_batch_dir(bid)
                cache_path = batch_dir / "artifacts" / "bucket_samples.json"
                if cache_path.exists():
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    total = cached.get("total_sampled", 0)
                    if total > 0:
                        if full:
                            # 返回完整数据
                            results[f"{sid}/{bid}"] = cached
                        else:
                            # 只返回摘要
                            results[f"{sid}/{bid}"] = {
                                "source_id": cached.get("source_id", sid),
                                "batch_id": cached.get("batch_id", bid),
                                "count_per_bucket": cached.get("count_per_bucket", 0),
                                "bucket_count": cached.get("bucket_count", 0),
                                "bucket_sizes": cached.get("bucket_sizes", {}),
                                "total_sampled": total,
                                "strategy": cached.get("strategy"),
                                "partition_tiers": cached.get("partition_tiers"),
                                "ratios": cached.get("ratios"),
                                "batch_info": cached.get("batch_info"),
                            }
            except Exception as e:
                print(f"[bucket-samples-batch] {sid}/{bid} error: {e}", file=sys.stderr)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bucket-samples-difficulty/{source_id}/{batch_id}")
async def get_difficulty_aware_samples(
    source_id: str,
    batch_id: str,
    base_count: int | None = None,
    easy_ratio: float | None = None,
    medium_ratio: float | None = None,
    hard_ratio: float | None = None,
    force: bool = False,
):
    """难度感知抽样。比例是占分区向量数的百分比，如 0.5 = 0.5%。"""
    if base_count is None:
        base_count = int(get_config("ocr", "sampling", "base_count", default=10))
    if easy_ratio is None:
        easy_ratio = float(get_config("ocr", "sampling", "easy_ratio", default=0.5))
    if medium_ratio is None:
        medium_ratio = float(get_config("ocr", "sampling", "medium_ratio", default=1.0))
    if hard_ratio is None:
        hard_ratio = float(get_config("ocr", "sampling", "hard_ratio", default=2.0))
    """难度感知抽样：先按分区抽样判难度，再按难度比例从分区中抽样。

    ratio 含义：占该分区向量数的千分比（‰）。
    例：easy_ratio=0.3 → 从 easy 分区抽 0.3% 的向量
        medium_ratio=1.0 → 从 medium 分区抽 1.0%
        hard_ratio=2.0 → 从 hard 分区抽 2.0%
    """
    import random
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path:
            raise HTTPException(status_code=404, detail="ingest manifest 不存在")

        cache_path = batch_dir / "artifacts" / "bucket_samples.json"

        if not force and cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("base_count") == base_count:
                    cached_ratios = cached.get("ratios", {})
                    if (float(cached_ratios.get("easy", 0)) == easy_ratio and
                        float(cached_ratios.get("medium", 0)) == medium_ratio and
                        float(cached_ratios.get("hard", 0)) == hard_ratio):
                        return cached
            except Exception:
                pass

        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))

            indices = ds.list_indices()
            index_name = None
            for idx in indices:
                name = idx.get("name", "") if isinstance(idx, dict) else getattr(idx, "name", "")
                if name:
                    index_name = name
                    break

            if not index_name:
                return {
                    "source_id": source_id, "batch_id": batch_id,
                    "error": "未找到向量索引，请先构建索引",
                    "bucket_count": 0, "total_sampled": 0, "buckets": {},
                }

            stats = ds.index_statistics(index_name)
            indices_data = stats.get("indices", [{}])
            partitions = indices_data[0].get("partitions", []) if indices_data else []
            centroids = indices_data[0].get("centroids", []) if indices_data else []

            # ── 第一步：从每个分区抽 base_count 个，判定分区难度 ──────────────
            ratio_map = {"easy": easy_ratio, "medium": medium_ratio, "hard": hard_ratio}
            bucket_sizes = {}
            partition_tiers = {}  # P{i} -> easy/medium/hard
            all_probe_ids = []

            for i, part_info in enumerate(partitions):
                part_size = part_info.get("size", 0)
                bucket_sizes[f"P{i}"] = part_size
                if part_size == 0 or not centroids:
                    partition_tiers[f"P{i}"] = "medium"
                    continue

                k = min(base_count, part_size)
                try:
                    centroid = centroids[i]
                    query_vec = pa.array(centroid, type=pa.float32())
                    with _lance_write_lock:
                        scanner = ds.scanner(
                            columns=["sample_id"],
                            nearest={"column": "embedding", "q": query_vec, "k": k},
                            disable_scoring_autoprojection=True,
                        )
                        probe_rows = scanner.to_table().to_pylist()
                    for r in probe_rows:
                        all_probe_ids.append(r["sample_id"])
                except Exception:
                    partition_tiers[f"P{i}"] = "medium"

            # 只读 probe 样本的 difficulty（而不是全量 2.5M）
            diff_map = {}
            if all_probe_ids:
                try:
                    ids_str = ",".join(repr(s) for s in set(all_probe_ids))
                    with _lance_write_lock:
                        diff_tbl = ds.to_table(
                            columns=["sample_id", "difficulty"],
                            filter=f"sample_id IN ({ids_str})",
                        )
                        for r in diff_tbl.to_pylist():
                            diff_map[r["sample_id"]] = r.get("difficulty") or "unlabeled"
                except Exception:
                    pass

            # 判定每个分区的难度
            probe_idx = 0
            for i, part_info in enumerate(partitions):
                part_key = f"P{i}"
                if part_key in partition_tiers:
                    continue
                k = min(base_count, part_info.get("size", 0))
                probe_ids = all_probe_ids[probe_idx:probe_idx + k]
                probe_idx += k

                diff_counts = {"easy": 0, "medium": 0, "hard": 0, "unlabeled": 0}
                for sid in probe_ids:
                    d = diff_map.get(sid, "unlabeled")
                    diff_counts[d] = diff_counts.get(d, 0) + 1

                if diff_counts["hard"] > diff_counts["easy"] and diff_counts["hard"] > diff_counts.get("medium", 0):
                    partition_tiers[part_key] = "hard"
                elif diff_counts["easy"] > diff_counts["hard"] and diff_counts["easy"] > diff_counts.get("medium", 0):
                    partition_tiers[part_key] = "easy"
                else:
                    partition_tiers[part_key] = "medium"

        # ── 第二步：按难度比例抽样（ratio 是目标输出占比）──────────
        # 先统计每个难度层级的总向量数
        tier_totals = {"easy": 0, "medium": 0, "hard": 0}
        tier_partitions: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
        for i, part_info in enumerate(partitions):
            part_size = part_info.get("size", 0)
            part_key = f"P{i}"
            tier = partition_tiers.get(part_key, "medium")
            tier_totals[tier] += part_size
            tier_partitions[tier].append(i)

        grand_total = sum(tier_totals.values())
        if grand_total == 0:
            return {"source_id": source_id, "batch_id": batch_id,
                    "error": "无数据", "bucket_count": 0, "total_sampled": 0, "buckets": {}}

        # 每个层级独立抽样：ratio 是该层级的抽样百分比
        tier_target = {}
        for tier_name in ["easy", "medium", "hard"]:
            pct = ratio_map.get(tier_name, 0)
            tier_target[tier_name] = max(0, int(tier_totals[tier_name] * pct / 100))

        buckets = {}
        bucket_diff_stats = {}

        for tier_name in ["easy", "medium", "hard"]:
            target = tier_target.get(tier_name, 0)
            total_size = tier_totals.get(tier_name, 0)
            if target == 0 or total_size == 0:
                continue
            # 每个分区按占比分配采样数
            for i in tier_partitions.get(tier_name, []):
                part_size = partitions[i].get("size", 0)
                part_key = f"P{i}"
                if part_size == 0 or not centroids:
                    continue
                # 按分区大小占该层级总量的比例分配
                k = max(1, int(target * part_size / total_size))
                k = min(k, part_size)

                try:
                    centroid = centroids[i]
                    query_vec = pa.array(centroid, type=pa.float32())
                    with _lance_write_lock:
                        scanner = ds.scanner(
                            columns=["sample_id"],
                            nearest={"column": "embedding", "q": query_vec, "k": k},
                            disable_scoring_autoprojection=True,
                        )
                        rows = scanner.to_table().to_pylist()
                    buckets[part_key] = [r["sample_id"] for r in rows]
                    bucket_diff_stats[part_key] = {
                        "tier": tier_name,
                        "ratio": ratio_map.get(tier_name, 0),
                        "sampled": len(rows),
                        "total": part_size,
                    }
                except Exception as e:
                    print(f"[difficulty-samples] P{i} error: {e}", file=sys.stderr)

        total_sampled = sum(len(v) for v in buckets.values())

        # 统计分区难度分布
        tier_counts = {"easy": 0, "medium": 0, "hard": 0}
        for t in partition_tiers.values():
            tier_counts[t] = tier_counts.get(t, 0) + 1

        result = {
            "source_id": source_id,
            "batch_id": batch_id,
            "strategy": "difficulty_aware",
            "base_count": base_count,
            "ratios": {"easy": easy_ratio, "medium": medium_ratio, "hard": hard_ratio},
            "partition_tiers": tier_counts,
            "bucket_count": len(partitions),
            "bucket_sizes": bucket_sizes,
            "bucket_diff_stats": bucket_diff_stats,
            "total_sampled": total_sampled,
            "buckets": buckets,
            "cached_at": datetime.now().isoformat(timespec='seconds'),
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # 难度抽样后清除 layout 结果缓存
        layout_cache = batch_dir / "artifacts" / "layout_results.json"
        if layout_cache.exists():
            layout_cache.unlink()

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/layout-preview")
async def run_layout_preview(request: Request):
    """预览 layout 结果：优先读 Lance 缓存，无缓存时才跑模型"""
    try:
        body = await request.json()
        source_id: str = body["source_id"]
        batch_id: str = body["batch_id"]
        sample_ids: list[str] = body["sample_ids"]

        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path:
            raise HTTPException(status_code=404, detail="ingest manifest 不存在")

        import base64

        # 1. 从 Lance 读取已有的 layout 结果
        cached_blocks: dict[str, list[dict]] = {}
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if lp.exists():
                try:
                    with _lance_write_lock:
                        ds = lance.dataset(str(lp))
                        ids_str = ",".join(repr(s) for s in sample_ids)
                        rows = ds.to_table(
                            columns=["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"],
                            filter=f"sample_id IN ({ids_str})",
                        ).to_pylist()
                    for row in rows:
                        sid = row["sample_id"]
                        if sid not in cached_blocks:
                            cached_blocks[sid] = []
                        cached_blocks[sid].append({
                            "block_type": row["block_type"],
                            "bbox": json.loads(row["bbox_json"]) if row.get("bbox_json") else [],
                            "confidence": round(row.get("layout_confidence", 0), 3),
                        })
                except Exception:
                    pass

        # 2. 从 ingest.lance 读取图片（仅需要有缓存或需要跑模型的样本）
        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))
            ids_str = ",".join(repr(s) for s in sample_ids)
            recs = ds.to_table(
                columns=["sample_id", "image_data", "difficulty"],
                filter=f"sample_id IN ({ids_str})",
            ).to_pylist()
        record_map = {r["sample_id"]: r for r in recs}

        # 3. 确定哪些样本需要跑模型
        need_model_ids = [sid for sid in sample_ids if sid not in cached_blocks]
        layout = None
        model_results: dict[str, list[dict]] = {}  # sid -> blocks

        if need_model_ids:
            from data_engine.ocr.layout_provider import get_layout_provider
            layout = get_layout_provider()

            # 批量推理：写临时文件，一次性调用
            import tempfile, shutil
            tmp_dir = tempfile.mkdtemp(prefix="layout_preview_")
            tmp_paths: list[Path] = []
            valid_sids: list[str] = []
            try:
                for sid in need_model_ids:
                    rec = record_map.get(sid)
                    if rec and rec.get("image_data"):
                        tmp_path = Path(tmp_dir) / f"{sid}.png"
                        tmp_path.write_bytes(rec["image_data"])
                        tmp_paths.append(tmp_path)
                        valid_sids.append(sid)

                if valid_sids:
                    try:
                        all_blocks_list = layout.detect_layout_batch(tmp_paths)
                        for sid, detected in zip(valid_sids, all_blocks_list):
                            model_results[sid] = [
                                {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                for b in detected
                            ]
                    except Exception as e:
                        print(f"[layout-preview] batch failed, fallback: {e}", file=sys.stderr)
                        for sid in valid_sids:
                            rec = record_map.get(sid)
                            if rec and rec.get("image_data"):
                                try:
                                    detected = layout.detect_layout_from_bytes(rec["image_data"])
                                    model_results[sid] = [
                                        {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                        for b in detected
                                    ]
                                except Exception as exc:
                                    model_results[sid] = [{"error": str(exc)}]
            finally:
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        results = []
        for sid in sample_ids:
            rec = record_map.get(sid)
            if not rec:
                results.append({"sample_id": sid, "error": "not found"})
                continue
            image_bytes = rec.get("image_data")
            if not image_bytes:
                results.append({"sample_id": sid, "error": "no image_data"})
                continue

            # 优先用缓存
            if sid in cached_blocks:
                blocks = cached_blocks[sid]
                from_cache = True
            elif sid in model_results:
                blocks = model_results[sid]
                from_cache = False
                if blocks and isinstance(blocks[0], dict) and "error" in blocks[0]:
                    results.append({"sample_id": sid, "error": blocks[0]["error"]})
                    continue
            else:
                results.append({"sample_id": sid, "error": "no result"})
                continue

            results.append({
                "sample_id": sid,
                "difficulty": rec.get("difficulty") or "unlabeled",
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "blocks": blocks,
                "block_count": len(blocks),
                "from_cache": from_cache,
            })

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/layout-batch")
async def start_layout_batch(request: Request):
    """批量运行 pp-layout（后台任务，有进度）。支持全部批次。"""
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        source_id: str = body.get("source_id", "")
        batch_id: str = body.get("batch_id", "")
        sample_ids: list[str] = body.get("sample_ids", [])
        write_lance: bool = body.get("write_lance", True)

        # 全部批次：遍历所有源的所有批次，逐个启动 layout
        if source_id == "all" or (not source_id and not batch_id):
            started = []
            for src_info in registry.scan():
                try:
                    src = registry.get(src_info.source_id)
                    root = Path(src_info.root_path)
                    batch_dirs = []
                    if (root / "manifests").exists():
                        batch_dirs.append(("", root))
                    for d in sorted(root.iterdir()):
                        if d.is_dir() and (d / "manifests").exists():
                            batch_dirs.append((d.name, d))
                    for bid, bdir in batch_dirs:
                        actual_bid = bid if bid else bdir.name
                        # 检查是否有抽样缓存
                        has_cache = (bdir / "artifacts" / "bucket_samples.json").exists()
                        if has_cache:
                            t_id = f"layout_batch_{src_info.source_id}_{actual_bid}"
                            if _start_single_layout(t_id, src_info.source_id, actual_bid, write_lance=write_lance):
                                started.append(t_id)
                except Exception:
                    pass
            return {"message": f"已启动 {len(started)} 个 layout 任务", "task_ids": started, "status": "started"}

        # 处理逗号分隔的多个 source_id（前端可能拼接多个源）
        if "," in source_id:
            started = []
            for sid in [s.strip() for s in source_id.split(",") if s.strip()]:
                try:
                    src = registry.get(sid)
                    root = Path(src.root_path)
                    batch_dirs = []
                    if (root / "manifests").exists():
                        batch_dirs.append(("", root))
                    for d in sorted(root.iterdir()):
                        if d.is_dir() and (d / "manifests").exists():
                            batch_dirs.append((d.name, d))
                    for bid, bdir in batch_dirs:
                        actual_bid = bid if bid else bdir.name
                        has_cache = (bdir / "artifacts" / "bucket_samples.json").exists()
                        if has_cache:
                            t_id = f"layout_batch_{sid}_{actual_bid}"
                            if _start_single_layout(t_id, sid, actual_bid, write_lance=write_lance):
                                started.append(t_id)
                except Exception as e:
                    print(f"[layout-batch] skip source {sid}: {e}", file=sys.stderr)
            return {"message": f"已启动 {len(started)} 个 layout 任务", "task_ids": started, "status": "started"}

        # 单批次（或单源全部批次）
        if not batch_id:
            # 单源全部批次：查找该源下所有有缓存的批次
            started = []
            try:
                src = registry.get(source_id)
                root = Path(src.root_path)
                batch_dirs = []
                if (root / "manifests").exists():
                    batch_dirs.append(("", root))
                for d in sorted(root.iterdir()):
                    if d.is_dir() and (d / "manifests").exists():
                        batch_dirs.append((d.name, d))
                for bid, bdir in batch_dirs:
                    actual_bid = bid if bid else bdir.name
                    has_cache = (bdir / "artifacts" / "bucket_samples.json").exists()
                    if has_cache:
                        t_id = f"layout_batch_{source_id}_{actual_bid}"
                        if _start_single_layout(t_id, source_id, actual_bid, write_lance=write_lance):
                            started.append(t_id)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"数据源 '{source_id}' 不存在: {e}")
            return {"message": f"已启动 {len(started)} 个 layout 任务", "task_ids": started, "status": "started"}

        # 单批次
        task_id = f"layout_batch_{source_id}_{batch_id}"
        if _start_single_layout(task_id, source_id, batch_id, sample_ids, write_lance=write_lance):
            return {"message": "已启动 layout 批量检测", "task_id": task_id, "total": len(sample_ids) if sample_ids else 0}
        return {"message": "任务已在运行中", "task_id": task_id, "status": "already_running"}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[layout-batch] ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


def _write_layout_to_category_lances(
    results: list[dict],
    manifests_dir: Path,
    flush_size: int = 10_000,
    progress_callback=None,
    write_mode: str = "overwrite",
    image_map: dict[str, bytes] | None = None,
) -> dict[str, int]:
    """将 layout 结果按 block_type 分块写入 text.lance / formula.lance / table.lance

    Args:
        results: layout 检测结果列表
        manifests_dir: manifests 目录
        flush_size: 每处理多少个样本 flush 一次（默认 10000）
        progress_callback: 可选回调 callback(current, total, message)
        write_mode: 初始写入模式 ("overwrite" 或 "append")
        image_map: 可选，sample_id -> 原始图片字节；提供时将裁剪每个 block 区域并写入 image_data
    """
    import pyarrow as pa
    from datetime import datetime, timezone
    from data_engine.ocr import (
        FORMULA_BLOCK_TYPES, TABLE_BLOCK_TYPES, SKIP_BLOCK_TYPES,
        TEXT_LANCE_SCHEMA, FORMULA_LANCE_SCHEMA, TABLE_LANCE_SCHEMA,
    )

    schema_map = {"text": TEXT_LANCE_SCHEMA, "formula": FORMULA_LANCE_SCHEMA, "table": TABLE_LANCE_SCHEMA}
    stats: dict[str, int] = {"text": 0, "formula": 0, "table": 0}
    written_once: dict[str, bool] = {"text": False, "formula": False, "table": False}
    now = datetime.now(timezone.utc).isoformat()
    total = len(results)

    # 预加载 PIL，仅当需要裁剪图片时
    _PIL_Image = None
    if image_map:
        try:
            from PIL import Image as _PIL_Image
            import io as _io_mod
        except ImportError:
            _PIL_Image = None
            image_map = None  # PIL 不可用，退化为不保存图片

    def _crop_block_image(sid: str, bbox: list) -> bytes | None:
        """从原始图片裁剪 block 区域，返回 JPEG 字节"""
        if not image_map or _PIL_Image is None:
            return None
        raw = image_map.get(sid)
        if not raw:
            return None
        try:
            img = _PIL_Image.open(_io_mod.BytesIO(raw))
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cropped = img.crop((x1, y1, x2, y2))
                buf = _io_mod.BytesIO()
                cropped.save(buf, format="JPEG", quality=90)
                return buf.getvalue()
            return raw
        except Exception:
            return None

    def _classify_block(sid: str, idx: int, b: dict) -> tuple[str, dict] | None:
        bt = b.get("block_type", "text")
        if bt in SKIP_BLOCK_TYPES:
            return None
        # 裁剪 block 图片（如 image_map 可用）
        cropped_img = _crop_block_image(sid, b.get("bbox", []))
        row = {
            "sample_id": sid,
            "block_idx": idx,
            "block_type": bt,
            "bbox_json": json.dumps(b.get("bbox", [])),
            "layout_confidence": float(b.get("confidence", 0)),
            "embedding": None,
            "schema_version": "v1",
            "created_at": now,
        }
        if bt in FORMULA_BLOCK_TYPES:
            row.update({"image_data": cropped_img,
                        "consistency_pattern": None})
            return "formula", row
        elif bt in TABLE_BLOCK_TYPES:
            row.update({"image_data": cropped_img,
                        "consistency_pattern": None})
            return "table", row
        else:
            row.update({"image_data": cropped_img,
                        "consistency_pattern": None})
            return "text", row

    def _flush_chunk(buffers: dict[str, list[dict]], mode_map: dict[str, str]):
        for cat, rows in buffers.items():
            if not rows:
                continue
            lance_path = manifests_dir / f"{cat}.lance"
            tbl = pa.Table.from_pylist(rows, schema=schema_map[cat])
            _safe_write_lance(tbl, lance_path, mode=mode_map[cat])
            stats[cat] += len(rows)
            written_once[cat] = True
            mode_map[cat] = "append"  # 后续都改为 append
            print(f"[layout-split] flushed {len(rows)} rows to {cat}.lance (total: {stats[cat]})", file=sys.stderr)

    # 分块处理
    buffers: dict[str, list[dict]] = {"text": [], "formula": [], "table": []}
    mode_map = {"text": write_mode, "formula": write_mode, "table": write_mode}
    processed = 0

    for r in results:
        sid = r.get("sample_id", "")
        for idx, b in enumerate(r.get("blocks", [])):
            result = _classify_block(sid, idx, b)
            if result:
                cat, row = result
                buffers[cat].append(row)

        processed += 1
        if processed % flush_size == 0:
            _flush_chunk(buffers, mode_map)
            buffers = {"text": [], "formula": [], "table": []}
            if progress_callback:
                progress_callback(processed, total, f"拆分写入 {processed}/{total}")

    # 写入剩余数据
    _flush_chunk(buffers, mode_map)
    if progress_callback:
        progress_callback(total, total, f"拆分完成 {total}/{total}")

    return stats


def _start_single_layout(task_id: str, source_id: str, batch_id: str, sample_ids: list[str] | None = None, write_lance: bool = True) -> bool:
    """启动单个 layout 批量任务。返回 True 表示已启动，False 表示已在运行。

    Args:
        write_lance: True 时将结果写入 text/formula/table.lance（流水线模式）；
                     False 时仅保存到 layout_results.json（预览/分布查看模式）。
    """
    existing = progress_tracker.get_task(task_id)
    if existing and existing.status.value in ["running", "pending"]:
        return False

    def execute():
        try:
            source = registry.get(source_id)
            batch_dir = source.resolve_batch_dir(batch_id)
            manifests_dir = batch_dir / "manifests"
            manifest_path = find_stage_manifest(manifests_dir, "ingest")
            if not manifest_path:
                progress_tracker.fail_task(task_id, "ingest manifest 不存在")
                return

            # 从缓存读取 sample_ids
            ids = list(sample_ids) if sample_ids else []
            if not ids:
                cpath = batch_dir / "artifacts" / "bucket_samples.json"
                if cpath.exists():
                    try:
                        cached = json.loads(cpath.read_text(encoding="utf-8"))
                        for samples in cached.get("buckets", {}).values():
                            for s in samples:
                                sid = s if isinstance(s, str) else s.get("sample_id", "")
                                if sid:
                                    ids.append(sid)
                    except Exception:
                        pass

            if not ids:
                progress_tracker.fail_task(task_id, "无抽样样本")
                return

            mode_label = "写入" if write_lance else "预览"
            progress_tracker.start_task(
                task_id=task_id, task_type="layout_batch",
                source_id=source_id, batch_id=batch_id,
                total=len(ids), message=f"layout({mode_label}) {len(ids)} 样本",
            )

            from data_engine.ocr.layout_provider import get_layout_provider
            layout = get_layout_provider()
            errors: list[dict] = []
            layout_stats: dict[str, int] = {"text": 0, "formula": 0, "table": 0}
            all_results: list[dict] = []  # write_lance=False 时积累全部结果（含续跑已有）

            # 断点续跑：根据模式从不同来源读取已处理的 sample_id
            done_ids: set[str] = set()
            lance_write_mode = "overwrite"
            if write_lance:
                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    if lp.exists():
                        try:
                            with _lance_write_lock:
                                ds = lance.dataset(str(lp))
                                ids_col = ds.to_table(columns=["sample_id"]).column("sample_id").to_pylist()
                                done_ids.update(ids_col)
                        except Exception as e:
                            print(f"[layout] resume: failed to read {cat}.lance: {e}", file=sys.stderr)
                if done_ids:
                    lance_write_mode = "append"
                    print(f"[layout] resume: {len(done_ids)} done sample_ids from Lance", file=sys.stderr)
            else:
                # 预览模式：从 layout_results.json 读取已有结果
                lr_path = batch_dir / "artifacts" / "layout_results.json"
                if lr_path.exists():
                    try:
                        cached = json.loads(lr_path.read_text(encoding="utf-8"))
                        existing = cached.get("results", [])
                        for r in existing:
                            sid = r.get("sample_id", "")
                            if sid:
                                done_ids.add(sid)
                        all_results = list(existing)  # 预加载已有结果，后续追加新结果
                        print(f"[layout] resume: {len(done_ids)} done sample_ids from layout_results.json", file=sys.stderr)
                    except Exception as e:
                        print(f"[layout] resume: failed to read layout_results.json: {e}", file=sys.stderr)

            # 过滤掉已处理的
            pending_ids = [sid for sid in ids if sid not in done_ids]
            done = len(done_ids)
            print(f"[layout] mode={mode_label} total={len(ids)} done={done} pending={len(pending_ids)}", file=sys.stderr)

            if not pending_ids:
                progress_tracker.update_progress(task_id, current=done, message=f"全部 {done} 样本已完成，跳过 layout")
            else:
                # 先触发模型加载
                progress_tracker.update_progress(task_id, current=done, message="加载模型中...")
                try:
                    _ = layout._ensure_model()
                    progress_tracker.update_progress(task_id, current=done, message=f"模型就绪，开始处理 {done}/{len(ids)}")
                except Exception as e:
                    progress_tracker.fail_task(task_id, f"模型加载失败: {e}")
                    return

                # 流式处理：分批加载图片 + 跑 layout + 报进度
                process_batch_size = 500

                for batch_start in range(0, len(pending_ids), process_batch_size):
                    if progress_tracker.is_stopped(task_id):
                        break

                    batch_ids = pending_ids[batch_start:batch_start + process_batch_size]

                    # 加载本批图片（分块加载，逐块更新进度）
                    image_map: dict[str, bytes] = {}
                    loaded_in_batch = 0
                    try:
                        with _lance_write_lock:
                            ds = lance.dataset(str(manifest_path))
                            query_batch = get_config("layout", "query_batch", default=200)
                            for qi in range(0, len(batch_ids), query_batch):
                                if progress_tracker.is_stopped(task_id):
                                    break
                                chunk = batch_ids[qi:qi + query_batch]
                                ids_str = ",".join(repr(s) for s in chunk)
                                try:
                                    recs = ds.to_table(
                                        columns=["sample_id", "image_data"],
                                        filter=f"sample_id IN ({ids_str})",
                                    ).to_pylist()
                                    for r in recs:
                                        if r.get("image_data"):
                                            image_map[r["sample_id"]] = r["image_data"]
                                except Exception:
                                    pass
                                loaded_in_batch += len(chunk)
                                progress_tracker.update_progress(
                                    task_id, current=done + loaded_in_batch,
                                    message=f"加载图片 {done+1}-{done+loaded_in_batch}/{len(ids)}"
                                )
                    except Exception as e:
                        print(f"[layout] 加载图片失败 batch {batch_start}: {e}", file=sys.stderr)

                    # 跑 layout（批量推理，比逐张快 10-16 倍）
                    batch_results: list[dict] = []
                    if progress_tracker.is_stopped(task_id):
                        break

                    # 收集有效图片的 sample_id
                    valid_sids = [sid for sid in batch_ids if image_map.get(sid)]
                    missing_sids = [sid for sid in batch_ids if not image_map.get(sid)]
                    done += len(missing_sids)

                    if valid_sids:
                        # 写临时文件（一次性）
                        import tempfile
                        tmp_dir = tempfile.mkdtemp(prefix="layout_batch_")
                        tmp_paths: list[Path] = []
                        try:
                            for sid in valid_sids:
                                tmp_path = Path(tmp_dir) / f"{sid}.png"
                                tmp_path.write_bytes(image_map[sid])
                                tmp_paths.append(tmp_path)

                            # 批量推理
                            progress_tracker.update_progress(
                                task_id, current=done,
                                message=f"layout 推理 {done+1}-{done+len(valid_sids)}/{len(ids)}"
                            )
                            try:
                                all_blocks_list = layout.detect_layout_batch(tmp_paths)
                                for sid, blocks in zip(valid_sids, all_blocks_list):
                                    result_entry = {
                                        "sample_id": sid,
                                        "blocks": [
                                            {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                            for b in blocks
                                        ],
                                        "block_count": len(blocks),
                                    }
                                    batch_results.append(result_entry)
                                    if not write_lance:
                                        all_results.append(result_entry)
                                    done += 1
                            except Exception as e:
                                # 批量失败时回退到逐张
                                print(f"[layout] batch failed, fallback to single: {e}", file=sys.stderr)
                                for sid in valid_sids:
                                    try:
                                        blocks = layout.detect_layout_from_bytes(image_map[sid])
                                        result_entry = {
                                            "sample_id": sid,
                                            "blocks": [
                                                {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                                for b in blocks
                                            ],
                                            "block_count": len(blocks),
                                        }
                                        batch_results.append(result_entry)
                                        if not write_lance:
                                            all_results.append(result_entry)
                                    except Exception as e2:
                                        errors.append({"sample_id": sid, "error": str(e2)})
                                    done += 1
                        finally:
                            # 清理临时目录
                            import shutil
                            try:
                                shutil.rmtree(tmp_dir, ignore_errors=True)
                            except Exception:
                                pass

                    # 每批结束：更新进度
                    error_summary = f"（{len(errors)} 错误）" if errors else ""
                    progress_tracker.update_progress(
                        task_id, current=done,
                        message=f"layout {done}/{len(ids)}{error_summary}"
                    )
                    # write_lance 模式：增量写 Lance
                    if write_lance and batch_results:
                        try:
                            batch_write_stats = _write_layout_to_category_lances(
                                batch_results, manifests_dir,
                                flush_size=get_config("layout", "flush_size", default=10_000),
                                write_mode=lance_write_mode,
                                image_map=image_map,
                            )
                            for k in layout_stats:
                                layout_stats[k] += batch_write_stats.get(k, 0)
                            lance_write_mode = "append"
                        except Exception as e:
                            print(f"[layout] incremental lance write failed: {e}", file=sys.stderr)

            # 无论完成还是停止
            stopped = progress_tracker.is_stopped(task_id)

            # 完成消息
            if write_lance:
                split_msg = ""
                for cat in ("text", "formula", "table"):
                    n = layout_stats.get(cat, 0)
                    if n:
                        split_msg += f"{cat}: {n}, "
                split_msg = split_msg.rstrip(", ")
            else:
                split_msg = f"{len(all_results)} 样本"

            if stopped:
                final_msg = f"已停止（已处理 {done}/{len(ids)}"
                if split_msg:
                    final_msg += f"，{split_msg}"
                final_msg += "）"
                progress_tracker.stop_task(task_id, final_msg)
            else:
                final_msg = f"完成 {done} 样本"
                if split_msg:
                    final_msg += f"，{split_msg}"
                progress_tracker.complete_task(task_id, message=final_msg)

            # write_lance=False 模式：保存结果到 layout_results.json
            if not write_lance and all_results:
                try:
                    out_path = batch_dir / "artifacts" / "layout_results.json"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps({"results": all_results}, ensure_ascii=False), encoding="utf-8")
                    print(f"[layout] saved {len(all_results)} results to {out_path}", file=sys.stderr)
                except Exception as e:
                    print(f"[layout] failed to save layout_results.json: {e}", file=sys.stderr)

            invalidate_status_cache()
        except Exception as e:
            progress_tracker.fail_task(task_id, str(e))

    thread = threading.Thread(target=execute, name=task_id)
    thread.daemon = True
    thread.start()
    return True


@app.get("/api/layout-batch/{source_id}/{batch_id}/results")
async def get_layout_batch_results(source_id: str, batch_id: str):
    """获取 layout 结果。优先读 layout_results.json（预览模式），其次从 Lance 文件读取。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)

        # 优先读 layout_results.json（预览模式产物）
        json_path = batch_dir / "artifacts" / "layout_results.json"
        if json_path.exists():
            try:
                cached = json.loads(json_path.read_text(encoding="utf-8"))
                results = cached.get("results", [])
                return {"results": results, "total": len(results)}
            except Exception:
                pass

        # 回退：从 Lance 文件读取
        manifests_dir = batch_dir / "manifests"
        results_map: dict[str, dict] = {}
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if lp.exists():
                with _lance_write_lock:
                    ds = lance.dataset(str(lp))
                    rows = ds.to_table(
                        columns=["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
                    ).to_pylist()
                for row in rows:
                    sid = row["sample_id"]
                    if sid not in results_map:
                        results_map[sid] = {"sample_id": sid, "blocks": [], "block_count": 0}
                    results_map[sid]["blocks"].append({
                        "block_type": row["block_type"],
                        "bbox": json.loads(row["bbox_json"]) if row.get("bbox_json") else [],
                        "confidence": row.get("layout_confidence", 0),
                        "block_idx": row["block_idx"],
                    })
        for sid in results_map:
            results_map[sid]["blocks"].sort(key=lambda b: b["block_idx"])
            results_map[sid]["block_count"] = len(results_map[sid]["blocks"])
        results = list(results_map.values())
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 补齐无图 API ───────────────────────────────────────────────────────────────────────

@app.post("/api/repair-images/{source_id}/{batch_id}")
async def repair_images(source_id: str, batch_id: str):
    """补齐 text/formula/table.lance 中缺失 image_data 的 block。

    扫描无图 block → 找出涉及页面 → 重新跑 layout + 拆分 → 替换旧行写入。
    """
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        # 1. 扫描无图 block 涉及的 sample_id
        no_image_sids: set[str] = set()
        no_image_counts: dict[str, int] = {}
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if lp.exists():
                with _lance_write_lock:
                    ds = lance.dataset(str(lp))
                    cols = ds.schema.names
                    if "image_data" in cols:
                        try:
                            null_rows = ds.to_table(
                                columns=["sample_id"],
                                filter="image_data IS NULL"
                            ).to_pylist()
                            sids = {r["sample_id"] for r in null_rows}
                            no_image_sids.update(sids)
                            no_image_counts[cat] = len(null_rows)
                        except Exception:
                            no_image_counts[cat] = 0

        if not no_image_sids:
            return {"message": "所有 block 均已有图片，无需补齐", "no_image": 0}

        # 检查 ingest.lance 是否存在
        ingest_path = find_stage_manifest(manifests_dir, "ingest")
        if not ingest_path or not ingest_path.exists():
            raise HTTPException(status_code=400, detail="ingest.lance 不存在，无法获取页面原图")

        task_id = f"repair_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            return {"message": "补齐任务已在运行中", "task_id": task_id}

        total_pages = len(no_image_sids)

        def execute():
            try:
                progress_tracker.start_task(
                    task_id=task_id, task_type="repair",
                    source_id=source_id, batch_id=batch_id,
                    total=total_pages,
                    message=f"补齐 {total_pages} 个页面的无图 block",
                )

                # 2. 从 ingest.lance 加载页面图片
                image_map: dict[str, bytes] = {}
                try:
                    with _lance_write_lock:
                        ds = lance.dataset(str(ingest_path))
                        all_sids = list(no_image_sids)
                        chunk_size = 200
                        for ci in range(0, len(all_sids), chunk_size):
                            chunk = all_sids[ci:ci + chunk_size]
                            ids_str = ",".join(repr(s) for s in chunk)
                            try:
                                recs = ds.to_table(
                                    columns=["sample_id", "image_data"],
                                    filter=f"sample_id IN ({ids_str})",
                                ).to_pylist()
                                for r in recs:
                                    if r.get("image_data"):
                                        image_map[r["sample_id"]] = r["image_data"]
                            except Exception:
                                pass
                except Exception as e:
                    progress_tracker.fail_task(task_id, f"加载页面图片失败: {e}")
                    return

                if not image_map:
                    progress_tracker.complete_task(task_id, "无法加载任何页面图片")
                    return

                # 3. 重新跑 layout 检测
                from data_engine.ocr.layout_provider import get_layout_provider
                layout = get_layout_provider()

                progress_tracker.update_progress(task_id, current=0, message="加载模型中...")
                try:
                    _ = layout._ensure_model()
                except Exception as e:
                    progress_tracker.fail_task(task_id, f"模型加载失败: {e}")
                    return

                # 写入临时文件，批量推理
                import tempfile, shutil
                tmp_dir = tempfile.mkdtemp(prefix="repair_")
                errors: list[dict] = []
                repaired_results: list[dict] = []
                done = 0
                process_batch_size = 200
                pending_sids = [sid for sid in no_image_sids if sid in image_map]

                for batch_start in range(0, len(pending_sids), process_batch_size):
                    if progress_tracker.is_stopped(task_id):
                        break

                    batch_ids = pending_sids[batch_start:batch_start + process_batch_size]
                    tmp_paths: list[Path] = []
                    try:
                        for sid in batch_ids:
                            tmp_path = Path(tmp_dir) / f"{sid}.png"
                            tmp_path.write_bytes(image_map[sid])
                            tmp_paths.append(tmp_path)

                        try:
                            all_blocks_list = layout.detect_layout_batch(tmp_paths)
                            for sid, blocks in zip(batch_ids, all_blocks_list):
                                repaired_results.append({
                                    "sample_id": sid,
                                    "blocks": [
                                        {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                        for b in blocks
                                    ],
                                    "block_count": len(blocks),
                                })
                        except Exception as e:
                            print(f"[repair] batch failed, fallback to single: {e}", file=sys.stderr)
                            for sid in batch_ids:
                                try:
                                    blocks = layout.detect_layout_from_bytes(image_map[sid])
                                    repaired_results.append({
                                        "sample_id": sid,
                                        "blocks": [
                                            {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                            for b in blocks
                                        ],
                                        "block_count": len(blocks),
                                    })
                                except Exception as e2:
                                    errors.append({"sample_id": sid, "error": str(e2)})
                    finally:
                        for p in tmp_paths:
                            try:
                                p.unlink()
                            except Exception:
                                pass

                    done += len(batch_ids)
                    error_summary = f"（{len(errors)} 错误）" if errors else ""
                    progress_tracker.update_progress(
                        task_id, current=done,
                        message=f"layout {done}/{total_pages}{error_summary}",
                    )

                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

                if progress_tracker.is_stopped(task_id):
                    progress_tracker.stop_task(task_id, f"已停止（已处理 {done}/{total_pages}）")
                    return

                # 4. 删除旧行 + 写入新行
                progress_tracker.update_progress(task_id, current=done, message="写入 Lance...")
                repair_sids_set = set(no_image_sids)

                # 先生成新行（写入临时 lance 文件）
                new_stats: dict[str, int] = {"text": 0, "formula": 0, "table": 0}
                if repaired_results:
                    # 用临时目录避免覆盖现有数据
                    tmp_manifests = Path(tempfile.mkdtemp(prefix="repair_manifests_"))
                    try:
                        new_stats = _write_layout_to_category_lances(
                            repaired_results, tmp_manifests,
                            flush_size=10_000,
                            write_mode="create",
                            image_map=image_map,
                        )

                        # 对每个 category，合并：删除旧行 + 追加新行
                        for cat in ("text", "formula", "table"):
                            lp = manifests_dir / f"{cat}.lance"
                            new_lp = tmp_manifests / f"{cat}.lance"
                            if not lp.exists():
                                # 如果原来不存在，直接复制新文件
                                if new_lp.exists():
                                    import shutil as _sh
                                    _sh.copytree(str(new_lp), str(lp))
                                continue

                            with _lance_write_lock:
                                ds = lance.dataset(str(lp))
                                full_table = ds.to_table()

                            # 过滤掉需要替换的 sample_id 行
                            sids_col = full_table.column("sample_id").to_pylist()
                            keep_indices = [i for i, sid in enumerate(sids_col) if sid not in repair_sids_set]
                            if keep_indices:
                                kept_table = full_table.take(keep_indices)
                            else:
                                kept_table = full_table.slice(0, 0)

                            # overwrite 保留的行
                            with _lance_write_lock:
                                lance.write_dataset(kept_table, str(lp), mode="overwrite")

                            # append 新行
                            if new_lp.exists() and new_stats.get(cat, 0) > 0:
                                with _lance_write_lock:
                                    new_ds = lance.dataset(str(new_lp))
                                    new_table = new_ds.to_table()
                                    ds2 = lance.dataset(str(lp))
                                    ds2.merge_insert(["sample_id", "block_idx"]).when_not_matched_insert_all().execute(new_table)

                            print(f"[repair] {cat}.lance: 保留 {kept_table.num_rows} 行, 新写入 {new_stats.get(cat, 0)} 行", file=sys.stderr)
                    finally:
                        try:
                            shutil.rmtree(str(tmp_manifests), ignore_errors=True)
                        except Exception:
                            pass

                # 5. 完成
                err_msg = f"（{len(errors)} 错误）" if errors else ""
                final_msg = f"补齐完成: {done} 页面{err_msg}"
                progress_tracker.complete_task(task_id, message=final_msg)
                invalidate_status_cache()

            except Exception as e:
                print(f"[repair] 任务失败: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                progress_tracker.fail_task(task_id=task_id, error_message=str(e))

        thread = threading.Thread(target=execute, name=task_id)
        thread.daemon = True
        thread.start()

        return {
            "message": f"已启动补齐任务: {total_pages} 个页面",
            "task_id": task_id,
            "no_image": no_image_counts,
            "pages": total_pages,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Element-Type Layout 聚类 API ─────────────────────────────────────────────────────


@app.post("/api/element-clusters/{source_id}/{batch_id}")
async def start_element_clusters(source_id: str, batch_id: str, request: Request):
    """触发 Element-Type Layout 聚类（后台任务）。
    
    对 text/formula/table.lance 中的 block 分别用 SigLIP2 提取特征向量，独立做 KMeans 聚类。
    """
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        max_k: int = body.get("max_k", 10)
        
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        
        # 检查是否有 text/formula/table.lance
        has_any = any((manifests_dir / f"{cat}.lance").exists() for cat in ("text", "formula", "table"))
        if not has_any:
            raise HTTPException(status_code=400, detail="未找到 text/formula/table.lance，请先运行 layout 拆分")
        
        task_id = f"element_clusters_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            return {"message": "任务已在运行中", "task_id": task_id, "status": "already_running"}
        
        def execute_element_clustering():
            try:
                print(f"[DEBUG-Thread] step 1: imports...", flush=True)
                import json
                import lance
                
                print(f"[DEBUG-Thread] step 2: import layout_features...", flush=True)
                from data_engine.ocr.layout_features import cluster_all_types
                
                print(f"[DEBUG-Thread] step 3: counting blocks...", flush=True)
                total_blocks = 0
                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    if lp.exists():
                        try:
                            total_blocks += lance.dataset(str(lp)).count_rows()
                        except Exception:
                            pass
                
                progress_tracker.start_task(
                    task_id, task_type="element_clusters",
                    source_id=source_id, batch_id=batch_id,
                    total=max(total_blocks, 1),
                    message=f"Element-Type 聚类 {total_blocks} blocks..."
                )
                
                def cb(cur, tot, msg):
                    progress_tracker.update_progress(task_id, current=cur, total=tot, message=msg)
                
                print(f"[DEBUG-Thread] step 4: calling cluster_all_types...", flush=True)
                result = cluster_all_types(manifests_dir, max_k=max_k, progress_callback=cb)
                print(f"[DEBUG-Thread] step 5: cluster_all_types done!", flush=True)
                
                # 保存结果
                out_path = batch_dir / "artifacts" / "element_clusters.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                
                summary_parts = []
                for cat in ("text", "formula", "table"):
                    info = result.get(cat, {})
                    if info.get("total_blocks", 0) > 0:
                        summary_parts.append(f"{cat}: {info['n_clusters']}簇/{info['total_blocks']}块")
                summary = ", ".join(summary_parts) if summary_parts else "无结果"
                
                progress_tracker.complete_task(task_id, f"完成: {summary}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                try:
                    progress_tracker.fail_task(task_id, str(e))
                except Exception:
                    pass
        
        thread = threading.Thread(target=execute_element_clustering, name=task_id)
        thread.daemon = True
        thread.start()
        
        print(f"[INFO] Element clustering started in thread: {task_id}", flush=True)
        
        return {"message": "已启动 Element-Type 聚类", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] start_element_clusters failed: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/api/element-clusters/{source_id}/{batch_id}")
async def get_element_clusters(source_id: str, batch_id: str):
    """获取 Element-Type Layout 聚类结果。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        
        # 检查任务状态
        task_id = f"element_clusters_{source_id}_{batch_id}"
        task_info = progress_tracker.get_task(task_id)
        if task_info:
            status = task_info.status.value
            if status in ["running", "pending"]:
                return {
                    "status": status,
                    "message": task_info.message,
                    "progress": {
                        "current": task_info.current,
                        "total": task_info.total,
                        "elapsed_seconds": round(task_info.elapsed_time, 1),
                    },
                }
            if status == "failed":
                return {
                    "status": "failed",
                    "message": task_info.message or task_info.error_message or "聚类失败",
                }
            if status == "stopped":
                return {
                    "status": "stopped",
                    "message": task_info.message or "已停止",
                }
        
        # 读取结果
        out_path = batch_dir / "artifacts" / "element_clusters.json"
        if not out_path.exists():
            return {"status": "not_started", "result": None}
        
        result = json.loads(out_path.read_text(encoding="utf-8"))
        resp = {"status": "completed", "result": result}
        if task_info and task_info.elapsed_time > 0:
            resp["elapsed_seconds"] = round(task_info.elapsed_time, 1)
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 每桶抽样 (Per-Cluster Sampling) ──────────────────────────────────────────

@app.post("/api/element-sample/{source_id}/{batch_id}")
async def element_sample(source_id: str, batch_id: str, request: Request):
    """基于聚类结果，从每个 cluster 中随机抽取指定数量的样本。

    读取 element_clusters.json 的 labels，再从 text/formula/table.lance 取对应行。
    """
    try:
        body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
        per_cluster: int = body.get("per_cluster", 5)

        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        # 检查聚类结果
        cluster_path = batch_dir / "artifacts" / "element_clusters.json"
        if not cluster_path.exists():
            raise HTTPException(status_code=400, detail="未找到聚类结果，请先运行 Element-Type 聚类")

        cluster_data = json.loads(cluster_path.read_text(encoding="utf-8"))

        import lance
        import random

        summary = {}
        all_samples = []

        for cat in ("text", "formula", "table"):
            cat_info = cluster_data.get(cat, {})
            labels_map = cat_info.get("labels", {})  # {"sample_id:block_idx": cluster_id}
            if not labels_map:
                summary[cat] = {"clusters": 0, "sampled": 0}
                continue

            # 按 cluster 分组
            cluster_members: dict[int, list[str]] = {}
            for key, lbl in labels_map.items():
                cluster_members.setdefault(lbl, []).append(key)

            # 读取 lance 数据
            lance_path = manifests_dir / f"{cat}.lance"
            if not lance_path.exists():
                summary[cat] = {"clusters": len(cluster_members), "sampled": 0}
                continue

            try:
                ds = lance.dataset(str(lance_path))
                # 只读 key 列，避免加载 image_data 二进制
                key_table = ds.to_table(columns=["sample_id", "block_idx"])
                key_rows = key_table.to_pylist()
            except Exception:
                summary[cat] = {"clusters": len(cluster_members), "sampled": 0}
                continue

            # 构建 key 集合（只用于存在性检查）
            key_set: set[str] = set()
            for row in key_rows:
                key = f"{row.get('sample_id', '')}:{row.get('block_idx', 0)}"
                key_set.add(key)

            # 每桶抽样
            cat_sampled = 0
            cluster_details = []
            for cid, members in sorted(cluster_members.items()):
                available = [m for m in members if m in key_set]
                n = min(per_cluster, len(available))
                sampled_keys = random.sample(available, n) if n > 0 else []
                for k in sampled_keys:
                    sid, bidx = k.rsplit(":", 1)
                    all_samples.append({
                        "category": cat,
                        "cluster_id": cid,
                        "sample_id": sid,
                        "block_idx": int(bidx),
                    })
                cat_sampled += len(sampled_keys)
                cluster_details.append({"cluster_id": cid, "total": len(available), "sampled": len(sampled_keys)})

            summary[cat] = {
                "clusters": len(cluster_members),
                "sampled": cat_sampled,
                "details": cluster_details,
            }

        # 保存抽样结果
        out_path = batch_dir / "artifacts" / "element_samples.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "per_cluster": per_cluster,
            "summary": summary,
            "samples": all_samples,
        }, ensure_ascii=False, default=str), encoding="utf-8")

        total_sampled = sum(s.get("sampled", 0) for s in summary.values())
        return {
            "message": f"抽样完成: {total_sampled} 条",
            "per_cluster": per_cluster,
            "summary": summary,
            "total_sampled": total_sampled,
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/element-sample/{source_id}/{batch_id}")
async def get_element_sample_status(source_id: str, batch_id: str):
    """获取已有抽样结果状态。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        out_path = batch_dir / "artifacts" / "element_samples.json"
        if not out_path.exists():
            return {"status": "not_started", "total_sampled": 0}
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return {
            "status": "completed",
            "per_cluster": data.get("per_cluster", 0),
            "summary": data.get("summary", {}),
            "total_sampled": data.get("total_sampled", sum(s.get("sampled", 0) for s in data.get("summary", {}).values())),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/element-ocr/{source_id}/{batch_id}")
async def element_ocr(source_id: str, batch_id: str, request: Request,
                      model: str = "paddleocr", test_mode: bool = False,
                      force: bool = False):
    """对 Element 抽样 block 运行 OCR，结果写回 text/formula/table.lance。
    force=True 时清空该模型已有结果后全部重跑。
    """
    try:
        prefix_map = {"paddleocr": "paddle", "glm_ocr": "glm", "self_ocr": "self"}
        prefix = prefix_map.get(model)
        if not prefix:
            raise HTTPException(status_code=400, detail=f"未知模型: {model}")

        task_id = f"el_ocr_{model}_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
            if task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务停止")
            else:
                return {"message": "任务已在运行中", "task_id": task_id, "status": "already_running"}

        use_test_mode = test_mode and model == "self_ocr"

        def execute():
            try:
                import tempfile, shutil, lance

                # 立即启动任务，让前端马上看到进度
                progress_tracker.start_task(
                    task_id=task_id, task_type="el_ocr",
                    source_id=source_id, batch_id=batch_id,
                    total=1, message=f"{model}: 启动中..."
                )

                source = registry.get(source_id)
                batch_dir = source.resolve_batch_dir(batch_id)
                manifests_dir = batch_dir / "manifests"

                # 阶段 1：仅读元数据统计总数（不加载 image_data）
                cat_paths: list[tuple[str, str]] = []  # (category, lance_path)
                total_blocks = 0
                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    if not lp.exists():
                        continue
                    try:
                        ds = lance.dataset(str(lp))
                        if "image_data" not in ds.schema.names:
                            continue
                        total_blocks += ds.count_rows()
                        cat_paths.append((cat, str(lp)))
                    except Exception as e:
                        print(f"[el-ocr] 读 {cat}.lance 失败: {e}", file=sys.stderr)

                if not cat_paths or total_blocks == 0:
                    progress_tracker.fail_task(task_id, "三个 category lance 中无可用 block")
                    return

                progress_tracker.update_progress(task_id, current=0, message=f"{model}: 扫描已完成，共 {total_blocks} blocks...")

                # 初始化 OCR 引擎
                if model == "paddleocr":
                    from data_engine.ocr.paddle_ocr import PaddleOCREngine
                    engine = PaddleOCREngine()
                elif model == "glm_ocr":
                    from data_engine.ocr.glm_ocr import GLMOCREngine
                    engine = GLMOCREngine()
                else:
                    from data_engine.ocr.self_ocr import SelfOCREngine
                    engine = SelfOCREngine(test_mode=use_test_mode)

                text_col = f"{prefix}_text"
                conf_col = f"{prefix}_confidence"

                # force 模式：清空该模型在所有 category lance 中的结果列
                if force:
                    progress_tracker.update_progress(task_id, current=0, message=f"{model}: 强制重跑，清空已有结果...")
                    # 按类别定义需要清空的列及重置值（SQL 表达式）
                    _clear_map = {
                        "text":    {text_col: "''", conf_col: "0.0"},
                        "formula": {text_col: "''", conf_col: "0.0", f"{prefix}_formula": "''"},
                        "table":   {text_col: "''", conf_col: "0.0", f"{prefix}_table": "''"},
                    }
                    for cat, lp_str in cat_paths:
                        try:
                            ds = lance.dataset(lp_str)
                            updates = {c: v for c, v in _clear_map[cat].items() if c in ds.schema.names}
                            if updates:
                                with _lance_write_lock:
                                    ds.update(updates)
                                print(f"[el-ocr] force: 已清空 {cat}.lance 中 {list(updates.keys())}", file=sys.stderr)
                        except Exception as e:
                            print(f"[el-ocr] force: 清空 {cat} 失败: {e}", file=sys.stderr)

                # 断点续跑：收集已有结果的 (sample_id, block_idx)
                # 优化：使用 filter 下推，不读大文本列，IO 减少 90%+
                done_keys: set[tuple[str, int]] = set()
                for cat, lp_str in cat_paths:
                    try:
                        ds = lance.dataset(lp_str)
                        if text_col in ds.schema.names:
                            tbl = ds.to_table(
                                columns=["sample_id", "block_idx"],
                                filter=f"{text_col} IS NOT NULL AND {text_col} != ''"
                            )
                            done_keys.update(
                                (row["sample_id"], row["block_idx"])
                                for row in tbl.to_pylist()
                            )
                    except Exception as e:
                        print(f"[el-ocr] resume 读 {cat} 失败: {e}", file=sys.stderr)

                if done_keys:
                    print(f"[el-ocr] resume: {model} 已有 {len(done_keys)} 个 block 完成，跳过", file=sys.stderr)

                # 重新计算待处理总数
                remaining_blocks = total_blocks - len(done_keys)
                if remaining_blocks <= 0:
                    progress_tracker.complete_task(task_id, f"{model} 所有 block 已完成（{len(done_keys)} 个），无需重跑")
                    return

                # 更新为实际总数和待处理信息
                progress_tracker.update_progress(
                    task_id, current=len(done_keys), total=total_blocks,
                    message=f"{model}: {total_blocks} 个 block（跳过 {len(done_keys)} 已完成，待处理 {remaining_blocks}）",
                )

                from data_engine.ocr.base import LayoutBlock
                tmp_dir = tempfile.mkdtemp(prefix="el_ocr_")

                done = 0
                errors = 0
                skipped = 0
                save_interval = int(get_config("ocr", "save_interval", default=20))
                pending_rows: dict[str, list[dict]] = {"text": [], "formula": [], "table": []}

                # test_mode: 预加载所有 category lance 的 paddle/glm 结果作为 ref_map
                _test_ref_map: dict[tuple[str, int], dict] = {}
                if use_test_mode:
                    for _cat, _lp in cat_paths:
                        try:
                            _ds = lance.dataset(_lp)
                            _cols_needed = ["sample_id", "block_idx"]
                            for _col in ("paddle_text", "glm_text", "paddle_table", "glm_table", "paddle_formula", "glm_formula"):
                                if _col in _ds.schema.names:
                                    _cols_needed.append(_col)
                            if len(_cols_needed) > 2:
                                for _batch in _ds.to_table(columns=_cols_needed).to_batches():
                                    for _row in _batch.to_pylist():
                                        _key = (_row["sample_id"], _row["block_idx"])
                                        _ref = {k: v for k, v in _row.items() if k not in ("sample_id", "block_idx")}
                                        _test_ref_map[_key] = _ref
                        except Exception as _e:
                            print(f"[el-ocr] test_mode 加载 {_cat} ref_map 失败: {_e}", file=sys.stderr)
                    if _test_ref_map:
                        print(f"[el-ocr] test_mode: 加载 {len(_test_ref_map)} 条 ref_map", file=sys.stderr)

                # 阶段 2：按类别流式处理，逐批读取并处理
                for cat, lp_str in cat_paths:
                    if progress_tracker.is_stopped(task_id):
                        break
                    try:
                        ds = lance.dataset(lp_str)
                    except Exception as e:
                        print(f"[el-ocr] 打开 {cat}.lance 失败: {e}", file=sys.stderr)
                        continue

                    progress_tracker.update_progress(
                        task_id, current=done,
                        message=f"{model}: 扫描 {cat} 数据..."
                    )

                    # 单次流式读取所有列（含 image_data），小批量处理
                    # 不依赖 _rowid，避免 merge_insert 后 rowid 失效
                    STREAM_BATCH = 100  # 每次流式读取的行数
                    scan_done = 0
                    for batch in ds.to_batches(batch_size=STREAM_BATCH):
                        if progress_tracker.is_stopped(task_id):
                            break
                        rows = batch.to_pylist()
                        for row in rows:
                            if progress_tracker.is_stopped(task_id):
                                break

                            sid = row["sample_id"]
                            bidx = row["block_idx"]
                            key = (sid, bidx)
                            if key in done_keys:
                                done += 1
                                scan_done += 1
                                continue

                            img_bytes = row.get("image_data")
                            if not img_bytes:
                                skipped += 1
                                done += 1
                                scan_done += 1
                                if scan_done % 50 == 0:
                                    progress_tracker.update_progress(
                                        task_id, current=done,
                                        message=f"{model}: 跳过 {skipped} 无图"
                                    )
                                continue

                            tmp_path = Path(tmp_dir) / f"{cat}_{sid}_{bidx}.png"
                            if isinstance(img_bytes, bytes):
                                tmp_path.write_bytes(img_bytes)
                            elif isinstance(img_bytes, str):
                                tmp_path.write_bytes(img_bytes.encode("latin-1"))
                            else:
                                tmp_path.write_bytes(bytes(img_bytes))

                            # image_data 已是裁剪好的 block 图，直接用原始尺寸避免黑边填充
                            _w, _h = 0, 0
                            try:
                                import struct
                                _raw = img_bytes if isinstance(img_bytes, bytes) else bytes(img_bytes)
                                if _raw[:8] == b'\x89PNG\r\n\x1a\n' and len(_raw) > 24:
                                    _w, _h = struct.unpack('>II', _raw[16:24])
                                elif _raw[:2] == b'\xff\xd8':  # JPEG
                                    _i = 2
                                    while _i < len(_raw) - 1:
                                        if _raw[_i] != 0xFF: break
                                        _m = _raw[_i+1]
                                        if _m in (0xC0, 0xC1, 0xC2):
                                            _h, _w = struct.unpack('>HH', _raw[_i+5:_i+9])
                                            break
                                        _ln = struct.unpack('>H', _raw[_i+2:_i+4])[0]
                                        _i += 2 + _ln
                            except Exception:
                                pass
                            if _w == 0 or _h == 0:
                                _w, _h = 9999, 9999  # 足够大，让 crop 返回完整图片
                            region = LayoutBlock(
                                block_type=row.get("block_type", cat),
                                bbox=[0, 0, _w, _h],
                                confidence=row.get("layout_confidence", 1.0),
                            )
                            if use_test_mode:
                                # idx 在 _recognize_test_mode 中是 regions 列表下标（始终=0），
                                # 需要把实际 block_idx 的数据映射到 key=(sample_id, 0)
                                _single_ref = {(sid, 0): _test_ref_map.get((sid, bidx), {})}
                                engine.set_test_context(sid, _single_ref)
                            try:
                                results = engine.recognize_regions(tmp_path, [region])
                                r = results[0] if results else None
                                ocr_row = {
                                    "sample_id": sid,
                                    "block_idx": bidx,
                                    text_col: (r.text_content or "") if r else "",
                                    conf_col: r.confidence if r else 0.0,
                                }
                                if cat == "table" and r and r.table_structure:
                                    ocr_row[f"{prefix}_table"] = json.dumps(r.table_structure, ensure_ascii=False)
                                if cat == "formula" and r and r.formula_latex:
                                    ocr_row[f"{prefix}_formula"] = r.formula_latex
                                pending_rows[cat].append(ocr_row)
                            except Exception as e:
                                errors += 1
                                if errors <= 5:
                                    print(f"[el-ocr] {model} error {sid}/{bidx}: {e}", file=sys.stderr)
                            finally:
                                tmp_path.unlink(missing_ok=True)

                            done += 1
                            scan_done += 1
                            err_msg = f"（{errors} 错误）" if errors else ""
                            progress_tracker.update_progress(task_id, current=done, message=f"{model}: {err_msg}" if err_msg else f"{model}")

                            if done % save_interval == 0:
                                _flush_el_ocr_rows(manifests_dir, pending_rows, text_col, conf_col, prefix)
                                pending_rows = {"text": [], "formula": [], "table": []}

                # 最终 flush
                _flush_el_ocr_rows(manifests_dir, pending_rows, text_col, conf_col, prefix)
                shutil.rmtree(tmp_dir, ignore_errors=True)

                if progress_tracker.is_stopped(task_id):
                    progress_tracker.stop_task(task_id, f"已停止（已处理 {done}/{total_blocks}）")
                else:
                    progress_tracker.complete_task(task_id, f"{model} 完成: {done} block（跳过 {len(done_keys)} 已完成{f'，{skipped} 无图' if skipped else ''}{f'，{errors} 错误' if errors else ''}）")
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                # 异常时也尝试 flush 已积累的结果，避免数据丢失
                try:
                    _flush_el_ocr_rows(manifests_dir, pending_rows, text_col, conf_col, prefix)
                except Exception:
                    pass
                progress_tracker.fail_task(task_id, str(e))

        thread = threading.Thread(target=execute, name=task_id, daemon=True)
        thread.start()
        return {"message": f"Element OCR {model} 已启动", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_CAT_TEXT_COL = {"text": "_text", "formula": "_formula", "table": "_table"}


def _flush_el_ocr_rows(manifests_dir: Path, pending_rows: dict, text_col: str, conf_col: str, prefix: str):
    """将 Element OCR 结果 merge_insert 到对应的 category lance。
    带重试：Lance 并发事务冲突时自动重试（最多 5 次，指数退避）。
    自动检测磁盘 schema，只写入兼容的列。
    """
    import time as _time
    import lance as _lance
    MAX_RETRIES = 5
    for cat, rows in pending_rows.items():
        if not rows:
            continue
        lp = manifests_dir / f"{cat}.lance"
        if not lp.exists():
            continue
        try:
            import pyarrow as pa
            cat_text_col = f"{prefix}{_CAT_TEXT_COL.get(cat, '_text')}"

            # 读取磁盘 schema，确保只写入兼容的列
            with _lance_write_lock:
                ds = _lance.dataset(str(lp))
            disk_names = {f.name for f in ds.schema}

            arrays = {}
            for r in rows:
                # text_col (e.g. paddle_text) 存储了 OCR 文本结果
                # cat_text_col 是目标 Lance 列名（对 text 类同，对 formula/table 不同）
                mapped = {
                    "sample_id": r["sample_id"],
                    "block_idx": r["block_idx"],
                }
                # 主文本内容：始终从 text_col 读取（OCR 引擎统一存储位置）
                mapped[cat_text_col] = r.get(text_col, "")
                mapped[conf_col] = r.get(conf_col, 0.0)
                # 额外列：表格结构、公式 LaTeX（如存在且磁盘 schema 支持）
                for extra_key in (f"{prefix}_table", f"{prefix}_formula"):
                    if extra_key in r and extra_key in disk_names:
                        mapped[extra_key] = r[extra_key]

                for k, v in mapped.items():
                    if k not in arrays:
                        arrays[k] = []
                    arrays[k].append(v)

            # 只保留磁盘 schema 中存在的列
            arrays = {k: v for k, v in arrays.items() if k in disk_names}
            if "sample_id" not in arrays or "block_idx" not in arrays:
                continue  # 缺少 key 列，跳过

            pa_arrays = {}
            for col, vals in arrays.items():
                field = ds.schema.field(col)
                if field.type == pa.large_string():
                    pa_arrays[col] = pa.array([v if v is not None else "" for v in vals], type=pa.large_string())
                elif field.type == pa.float32():
                    pa_arrays[col] = pa.array([float(v or 0) for v in vals], type=pa.float32())
                elif field.type == pa.int32():
                    pa_arrays[col] = pa.array([int(v or 0) for v in vals], type=pa.int32())
                else:
                    pa_arrays[col] = pa.array(vals)
            table = pa.table(pa_arrays)

            # 重试写入：Lance 事务冲突时指数退避
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    with _lance_write_lock:
                        ds = _lance.dataset(str(lp))
                        builder = ds.merge_insert(["sample_id", "block_idx"]).when_matched_update_all()
                        # 只有当写入列包含磁盘 schema 所有列时才启用 insert
                        if set(arrays.keys()) >= disk_names:
                            builder = builder.when_not_matched_insert_all()
                        builder.execute(table)
                    break
                except Exception as we:
                    if attempt < MAX_RETRIES and ("Incompatible transaction" in str(we) or "conflict" in str(we).lower()):
                        wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s, 4s
                        print(f"[el-ocr] flush {cat} 事务冲突 (attempt {attempt}/{MAX_RETRIES})，{wait}s 后重试...", file=sys.stderr)
                        _time.sleep(wait)
                    else:
                        raise
        except Exception as e:
            print(f"[el-ocr] flush {cat} 失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    import os
    import signal
    import uvicorn

    PID_FILE = Path("/tmp/data_engine_web_app.pid")

    def _check_and_kill_old():
        if not PID_FILE.exists():
            return
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if old_pid == os.getpid():
                return
            os.kill(old_pid, 0)  # 检查进程是否存在
            print(f"[web_app] 发现旧进程 pid={old_pid}，正在终止...", file=sys.stderr)
            os.kill(old_pid, signal.SIGTERM)
            import time
            time.sleep(1)
            try:
                os.kill(old_pid, signal.SIGKILL)
            except OSError:
                pass
        except (OSError, ValueError, ProcessLookupError):
            pass

    _check_and_kill_old()
    PID_FILE.write_text(str(os.getpid()))

    import atexit
    atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    uvicorn.run(app, host="0.0.0.0", port=8001)
