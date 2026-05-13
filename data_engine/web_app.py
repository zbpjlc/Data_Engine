from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from data_engine.manifests import read_jsonl
from data_engine.registry import SourceRegistry
from data_engine.status import collect_global_status, format_status_report
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
async def dashboard(request: Request):
    """数据地图首页 - The Map"""
    try:
        global_status = collect_global_status(registry)
        
        batches_json = [
            {"source_id": b.source_id, "batch_id": b.batch_id, "sample_count": b.sample_count,
             "category": b.category, "stage_status": b.stage_status}
            for b in global_status.batches
        ]
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
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
            "data_filter.html",
            {
                "request": request,
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
                
            manifest_path = batch_dir / "manifests" / "ingest.jsonl"
            if manifest_path.exists():
                records = read_jsonl(manifest_path)
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
        import threading
        
        # 检查是否已有运行中的任务
        if batch_id:
            task_id = f"ingest_{source_id}_{batch_id}"
            existing = progress_tracker.get_task(task_id)
            if existing and existing.status.value in ["running", "pending"]:
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
                        import traceback
                        traceback.print_exc()
                        
            except Exception as e:
                print(f"[INGEST后台线程] ✗ INGEST任务执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 在后台线程中运行
        print(f"[主线程] 启动后台线程进行INGEST处理...")
        thread = threading.Thread(target=execute_ingest_task)
        thread.daemon = True
        thread.start()
        print(f"[主线程] 后台线程已启动，立即返回响应")
        
        return {
            "message": f"已启动 {source_id} 的INGEST任务",
            "status": "started"
        }
    except Exception as e:
        print(f"启动INGEST API错误: {e}")
        import traceback
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
                
                for fname in ["ingest.jsonl", "input_files.json"]:
                    fpath = batch_dir / "manifests" / fname
                    if fpath.exists():
                        fpath.unlink()
                        cleared_count += 1
                        print(f"已删除: {fpath}")
                
                stats_path = batch_dir / "artifacts" / "stats.json"
                if stats_path.exists():
                    stats_path.unlink()
                    print(f"已删除: {stats_path}")
                
                page_images_dir = batch_dir / "page_images"
                if page_images_dir.exists():
                    import shutil
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
            import traceback
            traceback.print_exc()
        
        return {
            "message": f"已清除 {cleared_count} 个批次的INGEST数据和 {len(removed_tasks_info)} 个相关任务进度",
            "status": "cleared",
            "cleared_batches": cleared_count,
            "removed_tasks_count": len(removed_tasks_info),
            "removed_tasks": removed_tasks_info
        }
    except Exception as e:
        print(f"清除INGEST API错误: {e}")
        import traceback
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
        import threading
        
        # 检查是否已有运行中的任务
        if batch_id:
            task_id = f"embed_{source_id}_{batch_id}"
            existing = progress_tracker.get_task(task_id)
            if existing and existing.status.value in ["running", "pending"]:
                return {"message": f"任务 {task_id} 已在运行中", "status": "already_running"}
        
        print(f"\n========== 启动Embedding任务请求 ==========")
        print(f"source_id: '{source_id}' (类型: {type(source_id).__name__})")
        
        def execute_embed_task():
            try:
                print(f"[Embedding后台线程] 开始执行Embedding任务: {source_id}")
                # 导入embedding函数
                from data_engine.embedding import extract_embeddings_for_records
                from data_engine.manifests import read_jsonl, write_jsonl
                
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
                        manifest_path = manifests_dir / "ingest.jsonl"
                        
                        if not manifest_path.exists():
                            print(f"[Embedding后台线程] ⚠ manifest文件不存在: {manifest_path}")
                            continue
                        
                        # 读取现有记录
                        records = read_jsonl(manifest_path)
                        task_id = f"embed_{source_id}_{batch.batch_id}"
                        
                        # 开始任务
                        progress_tracker.start_task(
                            task_id=task_id,
                            task_type="embed",
                            source_id=source_id,
                            batch_id=batch.batch_id,
                            total=len(records),
                            message=f"开始提取 {len(records)} 个样本的embedding"
                        )
                        
                        # 提取embedding
                        print(f"[Embedding后台线程] 开始提取 {len(records)} 个样本的embedding...")
                        updated_records = extract_embeddings_for_records(records, batch_dir, task_id=task_id)
                        
                        write_jsonl(manifest_path, updated_records)
                        
                        task_obj = progress_tracker.get_task(task_id)
                        if task_obj and task_obj.status.value == "stopped":
                            print(f"[Embedding] 任务已停止，不标记完成", file=__import__("sys").stderr)
                        else:
                            progress_tracker.complete_task(
                                task_id=task_id,
                                message=f"成功提取 {len(updated_records)} 个样本的embedding"
                            )
                        
                        print(f"[Embedding后台线程] ✓ 批次 {batch.batch_id} embedding生成成功")
                        print(f"[Embedding后台线程]   - 已处理记录数: {len(updated_records)}")
                        
                    except Exception as e:
                        print(f"[Embedding后台线程] ✗ 批次 {batch.batch_id} embedding生成失败: {e}")
                        import traceback
                        traceback.print_exc()
                        progress_tracker.fail_task(
                            task_id=f"embed_{source_id}_{batch.batch_id}",
                            error_message=str(e)
                        )
                        
            except Exception as e:
                print(f"[Embedding后台线程] ✗ Embedding任务执行失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 在后台线程中运行
        print(f"[主线程] 启动后台线程进行Embedding处理...")
        thread = threading.Thread(target=execute_embed_task)
        thread.daemon = True
        thread.start()
        print(f"[主线程] 后台线程已启动，立即返回响应")
        
        return {
            "message": f"已启动 {source_id} 的Embedding任务",
            "status": "started"
        }
    except Exception as e:
        print(f"启动Embedding API错误: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop/{source_id}")
async def stop_task(source_id: str, batch_id: str = None):
    """停止运行中的任务"""
    if not batch_id:
        raise HTTPException(status_code=400, detail="batch_id is required")
    
    stopped = []
    for task_type in ["ingest", "embed", "cluster"]:
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
        import threading

        def execute_cluster_task():
            try:
                from data_engine.clustering import cluster_records
                from data_engine.manifests import read_jsonl, write_jsonl

                global_status = collect_global_status(registry)
                source_batches = [b for b in global_status.batches if b.source_id == source_id]
                if batch_id:
                    source_batches = [b for b in source_batches if b.batch_id == batch_id]

                for batch in source_batches:
                    try:
                        source = registry.get(source_id)
                        batch_dir = source.resolve_batch_dir(batch.batch_id)
                        manifest_path = batch_dir / "manifests" / "ingest.jsonl"

                        if not manifest_path.exists():
                            continue

                        records = read_jsonl(manifest_path)
                        task_id = f"cluster_{source_id}_{batch.batch_id}"

                        progress_tracker.start_task(
                            task_id=task_id,
                            task_type="cluster",
                            source_id=source_id,
                            batch_id=batch.batch_id,
                            total=len(records),
                            message=f"开始对 {len(records)} 个样本聚类"
                        )

                        updated_records, stats = cluster_records(
                            records, n_clusters=n_clusters, auto_optimize=auto_optimize)

                        write_jsonl(manifest_path, updated_records)

                        progress_tracker.complete_task(
                            task_id=task_id,
                            message=f"聚类完成: {stats.get('n_clusters', '?')} 簇, 轮廓系数 {stats.get('silhouette_score', 0):.3f}"
                        )

                    except Exception as e:
                        print(f"[聚类后台线程] ✗ 批次 {batch.batch_id} 聚类失败: {e}")
                        import traceback
                        traceback.print_exc()
                        progress_tracker.fail_task(
                            task_id=f"cluster_{source_id}_{batch.batch_id}",
                            error_message=str(e)
                        )

            except Exception as e:
                print(f"[聚类后台线程] ✗ 聚类任务执行失败: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=execute_cluster_task)
        thread.daemon = True
        thread.start()

        return {"message": f"已启动 {source_id} 的聚类任务", "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hard-case-review")
async def hard_case_review(request: Request):
    """Hard Case复核页面"""
    return templates.TemplateResponse(
        "hard_case_review.html",
        {"request": request}
    )


@app.get("/qa-sampling")
async def qa_sampling(request: Request):
    """QA抽检页面"""
    return templates.TemplateResponse(
        "qa_sampling.html",
        {"request": request}
    )


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
