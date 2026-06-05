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
from pathlib import Path
from typing import Any
import shutil
import pyarrow as pa
# 可选导入 torch，仅用于 CUDA 内存清理
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

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
    _lance_write_lock,
    read_element_manifest, write_element_manifest, merge_insert_element,
    append_element_manifest,
)
from data_engine.registry import SourceRegistry
from data_engine.status import collect_global_status, format_status_report, invalidate_status_cache
from data_engine.progress_tracker import progress_tracker

app = FastAPI(title="Data Engine Web Console", version="1.0.0")

# Static files and templates
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Global registry
registry = SourceRegistry(Path("sources.yaml"))


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
    for task_type in ["ingest", "embed", "cluster", "element_sample", "cmcv"]:
        task_id = f"{task_type}_{source_id}_{batch_id}"
        task = progress_tracker.get_task(task_id)
        if task and task.status.value in ["running", "pending"]:
            progress_tracker.request_stop(task_id)
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
                nearest={"column": "embedding", "q": query_vec, "k": page_size * page}
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

            for dataset_name in ["ingest", "element"]:
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
async def lancedb_get_image(source_id: str, batch_id: str, sample_id: str, version: int = None, dataset: str = "ingest"):
    """获取样本图片（始终从 ingest.lance 读取 image_data）"""
    try:
        source_config = registry.get(source_id)
        batch_dir = source_config.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        # 图片始终从 ingest.lance 读取
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path or manifest_path.suffix != ".lance":
            raise HTTPException(status_code=404, detail="ingest.lance 不存在")

        with _lance_write_lock:
            if version:
                ds = lance.dataset(str(manifest_path)).checkout_version(version)
            else:
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
    """获取所有任务进度"""
    try:
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
                    "start_time": task.start_time
                }
                for task_id, task in tasks.items()
            }
        }
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


@app.post("/api/ocr/{source_id}")
async def start_ocr(source_id: str, request: Request, model: str = "paddleocr", batch_id: str = "", resume: bool = True):
    """对指定批次的抽样样本运行 OCR，使用已有 layout 结果，支持断点续跑"""
    try:
        prefix_map = {"paddleocr": "paddle", "glm_ocr": "glm", "self_ocr": "self"}
        prefix = prefix_map.get(model)
        if not prefix:
            raise HTTPException(status_code=400, detail=f"未知模型: {model}")

        task_id = f"ocr_{model}_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            alive_threads = [t.name for t in threading.enumerate() if t.is_alive()]
            if task_id not in alive_threads:
                progress_tracker.stop_task(task_id, "线程已终止，任务停止")
            else:
                return {"message": f"任务已在运行中", "task_id": task_id, "status": "already_running"}

        def execute():
            try:
                source = registry.get(source_id)
                batch_dir = source.resolve_batch_dir(batch_id)
                artifacts_dir = batch_dir / "artifacts"
                element_path = batch_dir / "manifests" / "element.lance"

                # 1. 读该批次的抽样结果
                sample_cache = artifacts_dir / "bucket_samples.json"
                if not sample_cache.exists():
                    progress_tracker.fail_task(task_id, "无抽样数据，请先抽样")
                    return

                cached = json.loads(sample_cache.read_text(encoding="utf-8"))
                sample_map: dict[str, dict] = {}
                for samples in cached.get("buckets", {}).values():
                    for s in samples:
                        sample_map[s["sample_id"]] = s

                # 2. 读 layout 结果
                layout_cache = artifacts_dir / "layout_results.json"
                layout_map: dict[str, list[dict]] = {}
                _SKIP_BLOCK_TYPES = {"figure"}
                if layout_cache.exists():
                    for r in json.loads(layout_cache.read_text(encoding="utf-8")):
                        if r.get("blocks"):
                            filtered_blocks = [b for b in r["blocks"] if b.get("block_type") not in _SKIP_BLOCK_TYPES]
                            if filtered_blocks:
                                layout_map[r["sample_id"]] = filtered_blocks

                if not layout_map:
                    progress_tracker.fail_task(task_id, "无 layout 结果，请先运行 pp-layout")
                    return

                # 3. 初始化 OCR 引擎
                if model == "paddleocr":
                    from data_engine.ocr.paddle_ocr import PaddleOCREngine
                    engine = PaddleOCREngine()
                elif model == "glm_ocr":
                    from data_engine.ocr.glm_ocr import GLMOCREngine
                    engine = GLMOCREngine()
                else:
                    from data_engine.ocr.self_ocr import SelfOCREngine
                    engine = SelfOCREngine()

                # 4. 断点续跑：跳过 element.lance 中已有的 sample
                done_ids: set[str] = set()
                if resume and element_path.exists():
                    try:
                        with _lance_write_lock:
                            ds = lance.dataset(str(element_path))
                            existing_rows = ds.to_table(columns=["sample_id"]).to_pylist()
                            done_ids = {r["sample_id"] for r in existing_rows}
                        print(f"[{model}] resume: element.lance 有 {len(done_ids)} 个唯一 sample_id", file=sys.stderr)
                    except Exception as e:
                        print(f"[{model}] resume 读取 element.lance 失败: {e}", file=sys.stderr)
                else:
                    print(f"[{model}] resume={resume} element.exists={element_path.exists()}", file=sys.stderr)

                # 5. 读图片数据
                ids_to_load = [sid for sid in layout_map if sid in sample_map]
                image_map: dict[str, bytes] = {}
                try:
                    ipath = find_stage_manifest(batch_dir / "manifests", "ingest")
                    if ipath:
                        with _lance_write_lock:
                            ds = lance.dataset(str(ipath))
                            ids_str = ",".join(repr(s) for s in ids_to_load)
                            try:
                                recs = ds.to_table(columns=["sample_id", "image_data"], filter=f"sample_id IN ({ids_str})").to_pylist()
                            except Exception:
                                recs = ds.to_table(columns=["sample_id", "image_data"]).to_pylist()
                                recs = [r for r in recs if r["sample_id"] in set(ids_to_load)]
                        for r in recs:
                            if r.get("image_data"):
                                image_map[r["sample_id"]] = r["image_data"]
                except Exception as e:
                    print(f"[OCR] 读图失败: {e}", file=sys.stderr)

                # 6. 处理
                pending = [sid for sid in layout_map if sid not in done_ids and sid in image_map]
                total = len(layout_map)
                skipped = total - len(pending)
                print(f"[{model}] layout_map={len(layout_map)} image_map={len(image_map)} done_ids={len(done_ids)} pending={len(pending)} skipped={skipped}", file=sys.stderr)

                progress_tracker.start_task(
                    task_id=task_id, task_type="element_sample",
                    source_id=source_id, batch_id=batch_id,
                    total=total,
                    message=f"{model}: 跳过 {skipped}，剩余 {len(pending)}",
                )
                if skipped > 0:
                    progress_tracker.update_progress(task_id=task_id, current=skipped)

                from data_engine.ocr.base import LayoutBlock
                from data_engine.ocr.normalizer import results_to_element_rows

                all_rows: list[dict] = []
                fail_count = 0
                save_interval = int(get_config("ocr", "save_interval", default=10))

                def flush_rows(rows_to_save: list[dict]) -> None:
                    """加锁：读已有 → 合并 → 写入"""
                    if not rows_to_save:
                        return
                    with _lance_write_lock:
                        if element_path.exists():
                            try:
                                ds_el = lance.dataset(str(element_path))
                                existing_cols = [c for c in [
                                    "sample_id", "block_idx",
                                    "paddle_text", "glm_text", "self_text",
                                    "paddle_confidence", "glm_confidence", "self_confidence",
                                    "paddle_table_json", "glm_table_json", "self_table_json",
                                    "paddle_formula", "glm_formula", "self_formula",
                                    "paddle_raw_json", "glm_raw_json", "self_raw_json",
                                    "consistency_pattern", "block_diff_json",
                                ] if c in ds_el.schema.names]
                                existing_rows = ds_el.to_table(columns=existing_cols).to_pylist()
                                existing_map = {(r["sample_id"], r["block_idx"]): r for r in existing_rows}
                                for row in rows_to_save:
                                    key = (row["sample_id"], row["block_idx"])
                                    if key in existing_map:
                                        old = existing_map[key]
                                        for col in old:
                                            if col not in ("sample_id", "block_idx") and old.get(col) is not None and row.get(col) is None:
                                                row[col] = old[col]
                            except Exception:
                                pass

                        if element_path.exists():
                            from data_engine.ocr import ELEMENT_SCHEMA
                            arrow_rows = [__import__('data_engine.manifests', fromlist=['_element_record_to_arrow'])._element_record_to_arrow(r) for r in rows_to_save]
                            import pyarrow as pa
                            table = pa.Table.from_pylist(arrow_rows, schema=ELEMENT_SCHEMA)
                            ds_el = lance.dataset(str(element_path))
                            ds_el.merge_insert(["sample_id", "block_idx"]).when_matched_update_all().when_not_matched_insert_all().execute(table)
                        else:
                            write_element_manifest(element_path, rows_to_save)

                for idx, sid in enumerate(pending):
                    if progress_tracker.is_stopped(task_id):
                        progress_tracker.stop_task(task_id, f"停止 {skipped + idx}/{total}")
                        break

                    if sid not in layout_map or sid not in image_map:
                        continue

                    blocks = [
                        LayoutBlock(block_type=b["block_type"], bbox=b["bbox"], confidence=b.get("confidence", 0))
                        for b in layout_map[sid]
                    ]
                    image_bytes = image_map[sid]

                    try:
                        import tempfile, os
                        fd, tmp = tempfile.mkstemp(suffix=".png")
                        os.write(fd, image_bytes)
                        os.close(fd)
                        results = engine.recognize_regions(Path(tmp), blocks)
                        os.unlink(tmp)
                    except Exception as exc:
                        fail_count += 1
                        if fail_count <= 3:
                            print(f"[{model}] failed {sid}: {exc}", file=sys.stderr)
                        continue

                    rows = results_to_element_rows(
                        sample_id=sid, blocks=blocks,
                        engine_results={prefix: results},
                    )
                    all_rows.extend(rows)

                    progress_tracker.update_progress(
                        task_id=task_id, current=skipped + idx + 1,
                        message=f"{model}: {skipped + idx + 1}/{total}",
                    )

                    if len(all_rows) >= save_interval:
                        flush_rows(all_rows)
                        all_rows = []

                # 保存剩余
                flush_rows(all_rows)
                all_rows = []

                print(f"[{model}] 完成: pending={len(pending)} skipped={skipped} failed={fail_count}", file=sys.stderr)
                progress_tracker.complete_task(task_id=task_id, message=f"{model} 完成: skip={skipped} fail={fail_count}")
                invalidate_status_cache()
            except Exception as e:
                print(f"[OCR] {model} 失败: {e}", file=sys.stderr)
                traceback.print_exc()
                progress_tracker.fail_task(task_id=task_id, error_message=str(e))

        thread = threading.Thread(target=execute, name=task_id)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 {model}", "task_id": task_id, "status": "started"}
    except HTTPException:
        raise
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
                element_path = manifests_dir / "element.lance"
                ingest_path = find_stage_manifest(manifests_dir, "ingest")

                if not element_path.exists():
                    print("[CMCV] element.lance 不存在", file=sys.stderr)
                    return

                element_rows = read_element_manifest(element_path)
                progress_tracker.start_task(
                    task_id=task_id, task_type="cmcv",
                    source_id=source_id, batch_id=batch_id,
                    total=len(element_rows), message="开始一致性比较",
                )

                cmcv = CMCVEngine()
                updated_rows, page_tiers = cmcv.process_element_batch(element_rows)
                merge_insert_element(element_path, updated_rows)

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

        return {"message": f"已启动 {source_id} 的 CMCV 任务", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cmcv/{source_id}/{batch_id}/results")
async def get_cmcv_results(source_id: str, batch_id: str, tier: str = None):
    """获取 CMCV 比较结果"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        element_path = manifests_dir / "element.lance"

        if not element_path.exists():
            raise HTTPException(status_code=404, detail="element.lance 不存在")

        rows = read_element_manifest(element_path, columns=[
            "sample_id", "block_idx", "block_type",
            "paddle_text", "glm_text", "self_text",
            "consistency_pattern", "block_diff_json",
        ])

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

        return {
            "total_blocks": len(rows),
            "total_pages": len(page_stats),
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
        element_path = manifests_dir / "element.lance"

        if not element_path.exists():
            raise HTTPException(status_code=404, detail="element.lance 不存在")

        rows = read_element_manifest(element_path)
        sample_blocks = [r for r in rows if r.get("sample_id") == sample_id]
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


@app.get("/api/bucket-samples/{source_id}")
@app.get("/api/bucket-samples/{source_id}/{batch_id}")
async def get_bucket_samples(source_id: str, batch_id: str = "", count: int = 10, force: bool = False):
    """从向量索引的每个分区中随机抽取 N 个样本。结果缓存到 artifacts/bucket_samples.json。"""
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
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

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
                    )
                    tbl = scanner.to_table()
                    # 只取需要的列（nearest 查询会带 _distance 和 embedding）
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
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        # 重新抽样时清除旧 layout 结果
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
    """对单个样本运行 pp-layout（同步，返回图片 + bbox）"""
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

        with _lance_write_lock:
            ds = lance.dataset(str(manifest_path))
            recs = ds.to_table(
                columns=["sample_id", "image_data", "difficulty"],
                filter=f"sample_id IN ({','.join(repr(s) for s in sample_ids)})",
            ).to_pylist()
        record_map = {r["sample_id"]: r for r in recs}

        from data_engine.ocr.layout_provider import PPLayoutProvider
        layout = PPLayoutProvider()

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
            try:
                blocks = layout.detect_layout_from_bytes(image_bytes)
                import base64
                results.append({
                    "sample_id": sid,
                    "difficulty": rec.get("difficulty") or "unlabeled",
                    "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                    "blocks": [
                        {
                            "block_type": b.block_type,
                            "bbox": [round(c, 1) for c in b.bbox],
                            "confidence": round(b.confidence, 3),
                        }
                        for b in blocks
                    ],
                    "block_count": len(blocks),
                })
            except Exception as exc:
                results.append({"sample_id": sid, "error": str(exc)})

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/layout-batch")
async def start_layout_batch(request: Request):
    """批量运行 pp-layout（后台任务，有进度）"""
    try:
        body = await request.json()
        source_id: str = body["source_id"]
        batch_id: str = body["batch_id"]
        sample_ids: list[str] = body["sample_ids"]

        task_id = f"layout_batch_{source_id}_{batch_id}"
        existing = progress_tracker.get_task(task_id)
        if existing and existing.status.value in ["running", "pending"]:
            return {"message": "任务已在运行中", "task_id": task_id, "status": "already_running"}

        def execute():
            try:
                source = registry.get(source_id)
                batch_dir = source.resolve_batch_dir(batch_id)
                manifests_dir = batch_dir / "manifests"
                manifest_path = find_stage_manifest(manifests_dir, "ingest")

                with _lance_write_lock:
                    ds = lance.dataset(str(manifest_path))
                    recs = ds.to_table(
                        columns=["sample_id", "image_data", "difficulty"],
                        filter=f"sample_id IN ({','.join(repr(s) for s in sample_ids)})",
                    ).to_pylist()
                record_map = {r["sample_id"]: r for r in recs}

                from data_engine.ocr.layout_provider import PPLayoutProvider
                layout = PPLayoutProvider()

                progress_tracker.start_task(
                    task_id=task_id, task_type="layout_batch",
                    source_id=source_id, batch_id=batch_id,
                    total=len(sample_ids), message="开始 layout 检测",
                )

                all_results = []
                for i, sid in enumerate(sample_ids):
                    if progress_tracker.is_stopped(task_id):
                        progress_tracker.stop_task(task_id, f"用户停止 {i}/{len(sample_ids)}")
                        break

                    rec = record_map.get(sid)
                    if not rec or not rec.get("image_data"):
                        all_results.append({"sample_id": sid, "error": "no image"})
                        progress_tracker.update_progress(task_id, current=i + 1)
                        continue
                    try:
                        blocks = layout.detect_layout_from_bytes(rec["image_data"])
                        all_results.append({
                            "sample_id": sid,
                            "difficulty": rec.get("difficulty") or "unlabeled",
                            "blocks": [
                                {"block_type": b.block_type, "bbox": [round(c, 1) for c in b.bbox], "confidence": round(b.confidence, 3)}
                                for b in blocks
                            ],
                            "block_count": len(blocks),
                        })
                    except Exception as exc:
                        all_results.append({"sample_id": sid, "error": str(exc)})

                    progress_tracker.update_progress(task_id, current=i + 1, message=f"layout {i+1}/{len(sample_ids)}")

                # 保存结果到缓存
                result_path = batch_dir / "artifacts" / "layout_results.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(json.dumps(all_results, ensure_ascii=False), encoding="utf-8")

                progress_tracker.complete_task(task_id=task_id, message=f"完成 {len(all_results)} 个样本")
            except Exception as e:
                progress_tracker.fail_task(task_id=task_id, error_message=str(e))

        thread = threading.Thread(target=execute, name=task_id)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 layout 批量检测", "task_id": task_id, "total": len(sample_ids)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/layout-batch/{source_id}/{batch_id}/results")
async def get_layout_batch_results(source_id: str, batch_id: str):
    """获取 layout 批量检测结果"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        result_path = batch_dir / "artifacts" / "layout_results.json"
        if not result_path.exists():
            return {"results": [], "total": 0}
        results = json.loads(result_path.read_text(encoding="utf-8"))
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
