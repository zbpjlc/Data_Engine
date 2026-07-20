from __future__ import annotations

import os
# 限制所有底层 C/C++ 库的并发线程为 1，避免多层线程嵌套导致 malloc 崩溃
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import asyncio
import sys
import json
import gc
import threading
import resource
from collections import OrderedDict

# 启动时提高文件描述符限制
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
except Exception:
    pass
from datetime import datetime
from pathlib import Path


class LanceDatasetCache:
    """LRU cache for lance.dataset() to avoid FD exhaustion."""
    def __init__(self, max_size=8):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, path: str):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        ds = lance.dataset(key)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if len(self._cache) >= self._max_size:
                old_key, old_ds = self._cache.popitem(last=False)
                try: del old_ds
                except: pass
                gc.collect()
            self._cache[key] = ds
            return ds

    def invalidate(self, path: str = None):
        with self._lock:
            if path: self._cache.pop(str(path), None)
            else: self._cache.clear()
            gc.collect()


_lance_cache = LanceDatasetCache(max_size=8)


def _open_lance(path):
    """使用缓存打开 Lance dataset，避免 FD 耗尽。仅用于读操作。"""
    return _lance_cache.get(str(path))


class BucketSummaryCache:
    """LRU cache for parsed bucket_samples.json content (avoids 4s json.loads on every request)."""
    def __init__(self, max_size=16):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, path: str):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def set(self, path: str, data: dict):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = data

    def invalidate(self, path: str = None):
        with self._lock:
            if path:
                self._cache.pop(str(path), None)
            else:
                self._cache.clear()


_bucket_summary_cache = BucketSummaryCache(max_size=16)


class JsonFileCache:
    """LRU cache for parsed JSON files (avoids repeated json.loads of large files)."""
    def __init__(self, max_size=16):
        self._cache = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, path: str):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, path: str, data):
        key = str(path)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = data

    def invalidate(self, path: str = None):
        with self._lock:
            if path:
                self._cache.pop(str(path), None)
            else:
                self._cache.clear()


_json_cache = JsonFileCache(max_size=16)


def _read_json_cached(path):
    """带缓存的 JSON 文件读取，避免重复解析大文件。"""
    cached = _json_cache.get(str(path))
    if cached is not None:
        return cached
    import json as _json
    data = _json.loads(path.read_text(encoding="utf-8"))
    _json_cache.put(str(path), data)
    return data


# ─── 统一后台任务模型 (P0) ─────────────────────────────────────────────────────

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger("data_engine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class TaskContext:
    """后台任务上下文，包含停止信号和进度更新工具。"""
    task_id: str
    task_type: str
    source_id: str
    batch_id: str
    stop_event: threading.Event = field(default_factory=threading.Event)

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    def check_stop(self) -> bool:
        """检查停止信号，如果已停止则抛出异常终止任务。"""
        if self.stop_event.is_set():
            raise StopTaskError(self.task_id)
        return False


class StopTaskError(Exception):
    """任务被主动停止时抛出。"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        super().__init__(f"任务 {task_id} 已停止")


@dataclass
class TaskHandle:
    """运行中任务的句柄，用于统一管理。"""
    thread: threading.Thread
    context: TaskContext
    start_time: float = 0.0

    @property
    def task_id(self) -> str:
        return self.context.task_id

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self, message: str = "用户手动停止"):
        """发送停止信号并更新进度状态。"""
        self.context.stop_event.set()
        progress_tracker.request_stop(self.task_id)
        progress_tracker.stop_task(self.task_id, message)

    def join(self, timeout: float = 30.0) -> bool:
        """等待线程结束，返回是否在超时内完成。"""
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()


class TaskManager:
    """统一的后台任务管理器。

    职责：
    - start(): 启动后台任务
    - stop(): 停止指定任务
    - stop_all(): 停止所有任务
    - join(): 等待所有任务结束
    - shutdown(): 优雅关闭（stop_all + join）
    - get(): 获取任务句柄
    - list(): 列出所有运行中的任务
    """

    def __init__(self):
        self._tasks: dict[str, TaskHandle] = {}
        self._lock = threading.Lock()

    def start(
        self,
        *,
        task_id: str,
        task_type: str,
        source_id: str,
        batch_id: str,
        target: Callable[[TaskContext], None],
        **kwargs,
    ) -> dict:
        """启动后台任务。"""
        with self._lock:
            existing = self._tasks.get(task_id)
            if existing and existing.is_alive:
                return {"message": f"任务 {task_id} 已在运行中", "status": "already_running", "task_id": task_id}

            tracked = progress_tracker.get_task(task_id)
            if tracked and tracked.status.value in ("running", "pending"):
                if existing and existing.is_alive:
                    return {"message": f"任务 {task_id} 已在运行中", "status": "already_running", "task_id": task_id}
                else:
                    progress_tracker.stop_task(task_id, "线程已终止，任务停止")

        ctx = TaskContext(
            task_id=task_id,
            task_type=task_type,
            source_id=source_id,
            batch_id=batch_id,
        )

        def runner():
            import time as _time
            try:
                handle = self.get(task_id)
                if handle:
                    handle.start_time = _time.time()

                target(ctx, **kwargs)

                if ctx.is_stopped():
                    return

                progress_tracker.complete_task(task_id)
            except StopTaskError:
                pass
            except Exception as e:
                logger.exception(f"[{task_id}] 任务执行失败")
                try:
                    progress_tracker.fail_task(task_id, error_message=str(e))
                except Exception:
                    pass
            finally:
                with self._lock:
                    self._tasks.pop(task_id, None)

        thread = threading.Thread(target=runner, name=task_id, daemon=False)

        handle = TaskHandle(thread=thread, context=ctx)
        with self._lock:
            self._tasks[task_id] = handle

        thread.start()
        logger.info(f"[{task_id}] 后台任务已启动 (type={task_type}, source={source_id}, batch={batch_id})")

        return {"message": f"已启动 {source_id} 的 {task_type} 任务", "status": "started", "task_id": task_id}

    def stop(self, task_id: str, message: str = "用户手动停止") -> bool:
        """停止指定任务。返回是否成功发送停止信号。"""
        handle = self.get(task_id)
        if handle and handle.is_alive:
            handle.stop(message)
            return True
        return False

    def stop_all(self, message: str = "服务关闭，任务已停止"):
        """停止所有运行中的任务。"""
        with self._lock:
            tasks = list(self._tasks.values())

        for handle in tasks:
            try:
                handle.stop(message)
            except Exception as e:
                logger.warning(f"[stop_all] 停止任务 {handle.task_id} 失败: {e}")

    def join(self, timeout: float = 30.0) -> dict[str, bool]:
        """等待所有任务结束。返回每个任务是否在超时内完成。"""
        with self._lock:
            tasks = list(self._tasks.values())

        results = {}
        for handle in tasks:
            try:
                results[handle.task_id] = handle.join(timeout=timeout)
            except Exception as e:
                logger.warning(f"[join] 等待任务 {handle.task_id} 失败: {e}")
                results[handle.task_id] = False
        return results

    def shutdown(self, timeout: float = 30.0):
        """优雅关闭：stop_all + join。在 FastAPI shutdown 事件中调用。"""
        with self._lock:
            tasks = list(self._tasks.values())

        if not tasks:
            return

        logger.info(f"[shutdown] 正在停止 {len(tasks)} 个运行中的任务...")

        self.stop_all()
        results = self.join(timeout=timeout)

        for task_id, ok in results.items():
            if not ok:
                logger.warning(f"[shutdown] 任务 {task_id} 超时未结束")

        logger.info("[shutdown] 所有任务已停止")

    def get(self, task_id: str) -> Optional[TaskHandle]:
        """获取任务句柄。"""
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> dict[str, TaskHandle]:
        """列出所有运行中的任务。"""
        with self._lock:
            return {tid: h for tid, h in self._tasks.items() if h.is_alive}

    def is_alive(self, task_id: str) -> bool:
        """检查任务是否存活。"""
        handle = self.get(task_id)
        return handle.is_alive if handle else False


# 全局 TaskManager 单例
task_manager = TaskManager()


# ─── 分页安全限制 ──────────────────────────────────────────────────────────────

MAX_PAGE_SIZE = 1000  # 分页接口最大 page_size


def clamp_page_size(page_size: int, max_size: int = MAX_PAGE_SIZE) -> int:
    """限制 page_size 在合理范围内，防止恶意请求。"""
    return min(max(page_size, 1), max_size)


from typing import Any
import shutil
import pyarrow as pa
import pyarrow.compute as pc
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
    _lance_write_lock, _safe_write_lance, safe_merge, ensure_lance_indexes,
    ocr_complete_filter,
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


@app.on_event("shutdown")
def _on_shutdown():
    """服务关闭时优雅终止所有后台任务，确保 Lance 写入不中断。"""
    task_manager.shutdown(timeout=30.0)

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
        page_size = clamp_page_size(page_size)
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
        task_id = f"ingest_{source_id}_{batch_id}" if batch_id else f"ingest_{source_id}"

        def _execute_ingest(ctx: TaskContext):
            from data_engine.ingest import run_ingest as ingest_run

            global_status = collect_global_status(registry)
            source_batches = [b for b in global_status.batches if b.source_id == ctx.source_id]
            if ctx.batch_id:
                source_batches = [b for b in source_batches if b.batch_id == ctx.batch_id]

            if not source_batches:
                progress_tracker.start_task(
                    task_id=ctx.task_id, task_type="ingest",
                    source_id=ctx.source_id, batch_id=ctx.batch_id or "",
                    total=0, message="无可处理的批次"
                )
                progress_tracker.fail_task(task_id=ctx.task_id, error_message="无可处理的批次")
                return

            for batch in source_batches:
                ctx.check_stop()
                progress_tracker.start_task(
                    task_id=ctx.task_id, task_type="ingest",
                    source_id=ctx.source_id, batch_id=batch.batch_id,
                    total=0, message=f"处理批次 {batch.batch_id}"
                )
                result = ingest_run(registry, ctx.source_id, batch.batch_id)
                progress_tracker.update_progress(
                    ctx.task_id, current=1, total=1,
                    message=f"批次 {batch.batch_id} 完成: {len(result.records)} 条"
                )

        return task_manager.start(
            task_id=task_id, task_type="ingest",
            source_id=source_id, batch_id=batch_id or "",
            target=_execute_ingest,
        )
    except Exception as e:
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
    task_id = f"embed_{source_id}_{batch_id}" if batch_id else f"embed_{source_id}"

    def _execute_embed(ctx: TaskContext):
        from data_engine.embedding import extract_embeddings_for_records
        from data_engine.manifests import read_manifest, write_manifest, find_stage_manifest

        global_status = collect_global_status(registry)
        source_batches = [b for b in global_status.batches if b.source_id == ctx.source_id]
        if ctx.batch_id:
            source_batches = [b for b in source_batches if b.batch_id == ctx.batch_id]

        if not source_batches:
            progress_tracker.start_task(
                task_id=ctx.task_id, task_type="embed",
                source_id=ctx.source_id, batch_id=ctx.batch_id or "",
                total=0, message="无可处理的批次"
            )
            progress_tracker.fail_task(task_id=ctx.task_id, error_message="无可处理的批次")
            return

        for batch in source_batches:
            ctx.check_stop()
            source = registry.get(ctx.source_id)
            batch_dir = source.resolve_batch_dir(batch.batch_id)
            manifests_dir = batch_dir / "manifests"
            manifest_path = find_stage_manifest(manifests_dir, "ingest")

            if not manifest_path or not manifest_path.exists():
                continue

            total_count = manifest_count(manifest_path)
            batch_task_id = f"embed_{ctx.source_id}_{batch.batch_id}"

            progress_tracker.start_task(
                task_id=batch_task_id, task_type="embed",
                source_id=ctx.source_id, batch_id=batch.batch_id,
                total=total_count,
                message=f"开始提取 {total_count} 个样本的embedding"
            )

            with _lance_write_lock:
                ds = lance.dataset(str(manifest_path))
            chunk_size = get_config("embedding", "batch_update_interval", default=10000)

            from data_engine.embedding import CLIPEmbeddingExtractor
            extractor = CLIPEmbeddingExtractor()

            try:
                with _lance_write_lock:
                    processed_count = ds.count_rows(filter="embedding IS NOT NULL")
                start_offset = (processed_count // chunk_size) * chunk_size
            except Exception:
                start_offset = 0

            for offset in range(start_offset, total_count, chunk_size):
                ctx.check_stop()

                limit = min(chunk_size, total_count - offset)
                with _lance_write_lock:
                    chunk_table = ds.to_table(offset=offset, limit=limit)
                chunk_records = chunk_table.to_pylist()

                records_to_process = [r for r in chunk_records if r.get("embedding") is None]
                if not records_to_process:
                    continue

                updated_records = extract_embeddings_for_records(records_to_process, batch_dir, extractor=extractor, task_id=batch_task_id, offset=offset)

                embedding_map = {r["sample_id"]: r.get("embedding") for r in updated_records}
                update_ids = list(embedding_map.keys())
                update_embeddings = [embedding_map[sid] for sid in update_ids]
                embedding_dim = get_config("embedding", "embedding_dim", default=768)
                update_table = pa.table({
                    "sample_id": pa.array(update_ids, type=pa.large_string()),
                    "embedding": pa.array(update_embeddings, type=pa.list_(pa.float32(), embedding_dim)),
                })
                with _lance_write_lock:
                    current_ds = lance.dataset(str(manifest_path))
                    safe_merge(current_ds, update_table, ["sample_id"], context="embed")
                _lance_cache.invalidate(str(manifest_path))

                with _lance_write_lock:
                    ds = lance.dataset(str(manifest_path))

                chunk_len = len(chunk_records)
                del chunk_table, chunk_records, records_to_process, updated_records, embedding_map, update_table
                gc.collect()
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                next_offset = offset + chunk_len
                progress_tracker.update_progress(
                    task_id=batch_task_id, current=next_offset,
                    message=f"已处理 {next_offset}/{total_count}"
                )

            with _lance_write_lock:
                verify_table = ds.to_table(columns=["embedding"])
            total_rows = verify_table.num_rows
            actual_success = total_rows - verify_table.column("embedding").null_count
            msg = f"完成: {actual_success}/{total_rows} 条成功"
            if actual_success == 0:
                msg += " (全部失败，请检查GPU显存)"
            progress_tracker.complete_task(task_id=batch_task_id, message=msg)
            invalidate_status_cache()

    return task_manager.start(
        task_id=task_id, task_type="embed",
        source_id=source_id, batch_id=batch_id or "",
        target=_execute_embed,
    )


@app.post("/api/stop/{source_id}")
async def stop_task(source_id: str, batch_id: str = None, task_id: str = None):
    """停止运行中的任务"""
    from data_engine.progress_tracker import TaskStatus
    stopped = []

    # 优先通过 TaskManager 停止
    if task_id:
        if task_manager.stop(task_id):
            stopped.append(task_id)
    else:
        for tid, handle in task_manager.list().items():
            if source_id and source_id not in tid:
                continue
            if batch_id and batch_id not in tid:
                continue
            if task_manager.stop(tid):
                stopped.append(tid)

    # 兼容旧任务（不在 TaskManager 中的）
    all_tasks = progress_tracker.get_all_tasks()
    for tid, task in all_tasks.items():
        if tid in stopped:
            continue
        if task_id and tid != task_id:
            continue
        if not task_id:
            if source_id and source_id not in tid:
                continue
            if batch_id and batch_id not in tid:
                continue
        if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
            continue
        progress_tracker.request_stop(tid)
        stopped.append(tid)

    if stopped:
        return {"message": f"已发送停止信号: {', '.join(stopped)}", "status": "stopping"}
    return {"message": "没有运行中的任务", "status": "idle"}


@app.post("/api/cluster/{source_id}")
async def start_cluster(source_id: str, batch_id: str = None, n_clusters: int = 5, auto_optimize: bool = True):
    """启动聚类任务"""
    is_all = source_id == "__all__"
    task_id = f"cluster_{source_id}_{batch_id}" if batch_id else f"cluster_{source_id}"

    def _execute_cluster(ctx: TaskContext):
        from data_engine.clustering import cluster_records
        from data_engine.manifests import read_manifest, write_manifest, find_stage_manifest

        global_status = collect_global_status(registry)
        if is_all:
            source_batches = global_status.batches
        else:
            source_batches = [b for b in global_status.batches if b.source_id == ctx.source_id]
        if ctx.batch_id:
            source_batches = [b for b in source_batches if b.batch_id == ctx.batch_id]

        for batch in source_batches:
            ctx.check_stop()
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
                task_id=t_id, task_type="cluster",
                source_id=s_id, batch_id=bid,
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

    return task_manager.start(
        task_id=task_id, task_type="cluster",
        source_id=source_id, batch_id=batch_id or "",
        target=_execute_cluster,
    )


@app.post("/api/index/{source_id}")
async def start_index_build(source_id: str, batch_id: str = None, num_partitions: int = 256, num_sub_vectors: int = 16):
    """构建向量索引（IVF_PQ）"""
    task_id = f"index_{source_id}_{batch_id}" if batch_id else f"index_{source_id}"

    def _execute_index(ctx: TaskContext):
        from data_engine.index import build_ivf_pq_index

        global_status = collect_global_status(registry)
        source_batches = [b for b in global_status.batches if b.source_id == ctx.source_id]
        if ctx.batch_id:
            source_batches = [b for b in source_batches if b.batch_id == ctx.batch_id]

        if not source_batches:
            progress_tracker.start_task(
                task_id=ctx.task_id, task_type="index",
                source_id=ctx.source_id, batch_id=ctx.batch_id or "",
                total=0, message="无可处理的批次"
            )
            progress_tracker.fail_task(task_id=ctx.task_id, error_message="无可处理的批次")
            return

        for batch in source_batches:
            ctx.check_stop()
            bid = batch.batch_id
            s_id = batch.source_id
            source = registry.get(s_id)
            batch_dir = source.resolve_batch_dir(bid)
            manifest_path = batch_dir / "manifests" / "ingest.lance"

            if not manifest_path.exists():
                continue

            t_id = f"index_{s_id}_{bid}" if len(source_batches) > 1 else ctx.task_id
            total_count = manifest_count(manifest_path)

            progress_tracker.start_task(
                task_id=t_id, task_type="index",
                source_id=s_id, batch_id=bid,
                total=total_count,
                message=f"开始构建向量索引 ({total_count} 条)"
            )

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

    return task_manager.start(
        task_id=task_id, task_type="index",
        source_id=source_id, batch_id=batch_id or "",
        target=_execute_index,
    )


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
        page_size = clamp_page_size(page_size)
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifest_path = batch_dir / "manifests" / "ingest.lance"

        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Lance 数据集不存在")

        ds = _open_lance(manifest_path)

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

        ds = _open_lance(manifest_path)
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


@app.get("/api/batches/light")
async def list_batches_light():
    """轻量级批次列表（不扫描 Lance，仅返回 source_id/batch_id）。"""
    try:
        global_status = collect_global_status(registry)
        return {
            "sources": [
                {"source_id": b.source_id, "batch_id": b.batch_id}
                for b in global_status.batches
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                    ds = _open_lance(manifest_path)
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
        page_size = clamp_page_size(page_size)
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

        if version:
            ds = _open_lance(manifest_path).checkout_version(version)
        else:
            ds = _open_lance(manifest_path)
        total = ds.count_rows()

        select_cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        if select_cols and "sample_id" not in select_cols:
            select_cols.append("sample_id")

        schema_fields = [f.name for f in ds.schema]

        if search and search_col and search_col in schema_fields:
            safe_search = search.replace("'", "''")
            search_filter = f"contains(cast({search_col} as string), '{safe_search}')"
            start = (page - 1) * page_size
            try:
                total = ds.count_rows(filter=search_filter)
                results = ds.to_table(
                    columns=select_cols,
                    filter=search_filter,
                    offset=start,
                    limit=page_size,
                )
            except Exception:
                total = ds.count_rows()
                results = ds.to_table(columns=select_cols, offset=start, limit=page_size)

            page_rows = results.to_pylist()
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
                    ds = _open_lance(cat_path)
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

            ds = _open_lance(manifest_path)

            results = ds.to_table(
                columns=["sample_id", "image_data"],
                filter=f"sample_id = '{sample_id}'",
            )
            if results.num_rows == 0:
                raise HTTPException(status_code=404, detail="Sample not found")

            image_bytes = results.column("image_data")[0].as_py()
            if image_bytes is None:
                raise HTTPException(status_code=404, detail="No image data")

        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/progress")
async def get_progress():
    """获取所有任务进度（自动检测死线程）"""
    try:
        # 自动检测死线程：如果任务状态为 running/pending 但线程已死，标记为 stopped
        for task_id, task in progress_tracker.get_all_tasks().items():
            if task.status.value in ("running", "pending") and not task_manager.is_alive(task_id):
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
        task = progress_tracker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.status.value in ("running", "pending") and not task_manager.is_alive(task_id):
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
async def start_cmcv(source_id: str, batch_id: str = None, full: bool = False, force: bool = False, block_types: str = None):
    """启动 CMCV 一致性比较任务

    Args:
        block_types: 逗号分隔的 block 类型过滤，如 "text" 或 "text,table"。默认全部（text,formula,table）。
    """
    task_id = f"cmcv_{source_id}_{batch_id}"

    # 解析 block_types 过滤
    allowed_cats = ("text", "formula", "table")
    if block_types:
        selected = [c.strip() for c in block_types.split(",") if c.strip() in allowed_cats]
        cats_to_process = tuple(selected) if selected else allowed_cats
    else:
        cats_to_process = allowed_cats

    def _execute_cmcv(ctx: TaskContext):
        import sys
        from data_engine.ocr.cmcv import CMCVEngine

        global_status = collect_global_status(registry)
        source = registry.get(ctx.source_id)
        batch_dir = source.resolve_batch_dir(ctx.batch_id)
        manifests_dir = batch_dir / "manifests"
        ingest_path = find_stage_manifest(manifests_dir, "ingest")

        def _all_models_have_result(row: dict, cat: str) -> bool:
            """检查三个 OCR 模型是否都有非空结果"""
            if cat == "text":
                return bool(row.get("paddle_text") and row.get("glm_text") and row.get("self_text"))
            elif cat == "table":
                return bool(row.get("paddle_table") and row.get("glm_table") and row.get("self_table"))
            elif cat == "formula":
                return bool(row.get("paddle_formula") and row.get("glm_formula") and row.get("self_formula"))
            return False

        def _read_blocks_from_category_lance(full_mode: bool = False, force: bool = False) -> list[dict]:
            if full_mode:
                return _read_all_blocks_from_category_lance(manifests_dir, force=force)

            sample_keys = set()
            for b in global_status.batches:
                try:
                    sp = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "artifacts" / "element_samples.json"
                    if sp.exists():
                        sd = _read_json_cached(sp)
                        for s in sd.get("samples", []):
                            sample_keys.add((s["category"], s["sample_id"], s["block_idx"]))
                except Exception:
                    pass

            if not sample_keys:
                return _read_all_blocks_from_category_lance(manifests_dir, force=force)

            cmcv_filter = None if force else "consistency_pattern IS NULL"
            all_rows = []
            json_keys = ("paddle_table", "glm_table", "self_table",
                         "paddle_formula", "glm_formula", "self_formula")
            for b in global_status.batches:
                b_manifests = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "manifests"
                for cat in cats_to_process:
                    lp = b_manifests / f"{cat}.lance"
                    if not lp.exists():
                        continue
                    try:
                        with _lance_write_lock:
                            ds = _open_lance(lp)
                            all_cols = set(ds.schema.names)
                            ocr_filter = ocr_complete_filter(all_cols)
                            batch_filter = cmcv_filter
                            if ocr_filter:
                                batch_filter = f"({cmcv_filter}) AND ({ocr_filter})" if cmcv_filter else ocr_filter
                            ds = _open_lance(lp)
                            all_cols = ds.schema.names
                            if not force and "consistency_pattern" not in all_cols:
                                continue
                            read_cols = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
                            for prefix in ("paddle", "glm", "self"):
                                for suffix in ("_text", "_confidence", "_table", "_formula"):
                                    col = f"{prefix}{suffix}"
                                    if col in all_cols:
                                        read_cols.append(col)
                            cols = [c for c in read_cols if c in all_cols]
                            batches = list(ds.to_batches(columns=cols, filter=batch_filter))
                        for batch in batches:
                            sid_col = batch.column("sample_id")
                            bidx_col = batch.column("block_idx")
                            for i in range(len(batch)):
                                key = (cat, sid_col[i].as_py(), bidx_col[i].as_py())
                                if key not in sample_keys:
                                    continue
                                row = {c: batch.column(c)[i].as_py() for c in cols}
                                for k in json_keys:
                                    val = row.get(k)
                                    if isinstance(val, str) and val:
                                        try:
                                            row[k] = json.loads(val)
                                        except (json.JSONDecodeError, TypeError):
                                            pass
                                if not _all_models_have_result(row, cat):
                                    continue
                                all_rows.append(row)
                    except Exception as e:
                        print(f"[CMCV] 读 {b.source_id}/{b.batch_id}/{cat}.lance 失败: {e}", file=sys.stderr)
            return all_rows

        def _read_all_blocks_from_category_lance(manifests_dir: Path, force: bool = False) -> list[dict]:
            base_filter = None if force else "consistency_pattern IS NULL"
            json_keys = ("paddle_table", "glm_table", "self_table",
                         "paddle_formula", "glm_formula", "self_formula")
            all_rows = []
            for cat in cats_to_process:
                lp = manifests_dir / f"{cat}.lance"
                if not lp.exists():
                    continue
                try:
                    with _lance_write_lock:
                        ds = _open_lance(lp)
                        all_cols = set(ds.schema.names)
                        if not force and "consistency_pattern" not in all_cols:
                            continue
                        ocr_filter = ocr_complete_filter(all_cols)
                        batch_filter = base_filter
                        if ocr_filter:
                            batch_filter = f"({base_filter}) AND ({ocr_filter})" if base_filter else ocr_filter
                        read_cols = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
                        for prefix in ("paddle", "glm", "self"):
                            for suffix in ("_text", "_confidence", "_table", "_formula"):
                                col = f"{prefix}{suffix}"
                                if col in all_cols:
                                    read_cols.append(col)
                        cols = [c for c in read_cols if c in all_cols]
                        batches = list(ds.to_batches(columns=cols, filter=batch_filter))
                    for batch in batches:
                        for i in range(len(batch)):
                            row = {c: batch.column(c)[i].as_py() for c in cols}
                            for k in json_keys:
                                val = row.get(k)
                                if isinstance(val, str) and val:
                                    try:
                                        row[k] = json.loads(val)
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                            if not _all_models_have_result(row, cat):
                                continue
                            all_rows.append(row)
                except Exception as e:
                    print(f"[CMCV] 读 {cat}.lance 失败: {e}", file=sys.stderr)
            return all_rows

        element_rows = _read_blocks_from_category_lance(full_mode=full, force=force)
        if not element_rows:
            progress_tracker.fail_task(ctx.task_id, "text/formula/table.lance 中无可用 block")
            return

        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="cmcv",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=len(element_rows), message="开始一致性比较",
        )

        def _flush_cmcv_rows(rows: list[dict], g_status, write_lock, open_fn, cache_fn):
            """将 CMCV 结果批量写入 lance 文件"""
            for b in g_status.batches:
                b_manifests = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "manifests"
                for cat in ("text", "formula", "table"):
                    lp = b_manifests / f"{cat}.lance"
                    if not lp.exists():
                        continue
                    try:
                        with write_lock:
                            ds = open_fn(lp)
                            col_names = set(ds.schema.names)
                            if "consistency_pattern" not in col_names or "block_diff_json" not in col_names:
                                continue
                        update_map: dict[str, dict] = {}
                        for r in rows:
                            key = (r["sample_id"], r["block_idx"])
                            if r.get("consistency_pattern"):
                                update_map.setdefault("consistency_pattern", {})[key] = r["consistency_pattern"]
                            if r.get("block_diff_json"):
                                update_map.setdefault("block_diff_json", {})[key] = r["block_diff_json"]
                        if not update_map:
                            continue
                        with write_lock:
                            ds = open_fn(lp)
                            read_cols = ["sample_id", "block_idx", "consistency_pattern", "block_diff_json"]
                            scanner = ds.scanner(columns=[c for c in read_cols if c in ds.schema.names])
                            update_batches = []
                            pat_map = update_map.get("consistency_pattern", {})
                            diff_map = update_map.get("block_diff_json", {})
                            for batch in scanner.to_batches():
                                sids = batch.column("sample_id").to_pylist()
                                bidxs = batch.column("block_idx").to_pylist()
                                patterns = batch.column("consistency_pattern").to_pylist() if "consistency_pattern" in batch.column_names else [None] * len(batch)
                                diffs = batch.column("block_diff_json").to_pylist() if "block_diff_json" in batch.column_names else [None] * len(batch)
                                new_patterns = []
                                new_diffs = []
                                for i in range(len(batch)):
                                    key = (sids[i], bidxs[i])
                                    new_patterns.append(pat_map.get(key, patterns[i]))
                                    new_diffs.append(diff_map.get(key, diffs[i]))
                                update_batches.append(pa.table({
                                    "sample_id": pa.array(sids, type=pa.large_string()),
                                    "block_idx": pa.array(bidxs, type=pa.int32()),
                                    "consistency_pattern": pa.array(new_patterns, type=pa.large_string()),
                                    "block_diff_json": pa.array(new_diffs, type=pa.large_string()),
                                }))
                            cache_fn.invalidate(str(lp))
                        if update_batches:
                            update_table = pa.concat_tables(update_batches)
                            safe_merge(ds, update_table, ["sample_id", "block_idx"], context=f"cmcv/{cat}")
                    except Exception as e:
                        print(f"[CMCV] flush {cat}.lance 失败: {e}", file=sys.stderr)

        cmcv = CMCVEngine()  # token fallback，快；视觉渲染由 cli/full 模式按需启用

        FLUSH_INTERVAL = int(get_config("ocr", "cmcv", "flush_interval", default=200))
        pending_rows: list[dict] = []

        def _cmcv_step_cb(cur, tot, msg):
            progress_tracker.update_progress(task_id=ctx.task_id, current=cur, message=f"[{cur}/{tot}] {msg}", total=tot)

        def _cmcv_flush_cb(rows_batch, page_tiers_batch):
            """每 200 个 block 写入一次 lance 文件"""
            pending_rows.extend(rows_batch)
            if len(pending_rows) >= FLUSH_INTERVAL:
                _flush_cmcv_rows(pending_rows, global_status, _lance_write_lock, _open_lance, _lance_cache)
                print(f"[CMCV] flush {len(pending_rows)} rows to lance", file=sys.stderr)
                pending_rows.clear()

        updated_rows, page_tiers = cmcv.process_element_batch(
            element_rows,
            progress_callback=_cmcv_step_cb,
            flush_callback=_cmcv_flush_cb,
        )
        print(f"[CMCV] updated_rows={len(updated_rows)}, page_tiers={len(page_tiers)}", file=sys.stderr)

        # 写入剩余未 flush 的行
        if pending_rows:
            _flush_cmcv_rows(pending_rows, global_status, _lance_write_lock, _open_lance, _lance_cache)
            print(f"[CMCV] final flush {len(pending_rows)} rows", file=sys.stderr)
            pending_rows.clear()

        for b in global_status.batches:
            b_manifests = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "manifests"
            for cat in ("text", "formula", "table"):
                lp = b_manifests / f"{cat}.lance"
                if not lp.exists():
                    continue
                try:
                    with _lance_write_lock:
                        ds = _open_lance(lp)
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
                        print(f"[CMCV DEBUG] update_map 为空，跳过写入", file=sys.stderr)
                        continue
                    print(f"[CMCV DEBUG] update_map 有 {len(update_map)} keys", file=sys.stderr)
                    with _lance_write_lock:
                        ds = _open_lance(lp)
                        # 流式读取，逐批更新，避免全表加载
                        read_cols = ["sample_id", "block_idx", "consistency_pattern", "block_diff_json"]
                        scanner = ds.scanner(columns=[c for c in read_cols if c in ds.schema.names])
                        update_batches = []
                        pat_map = update_map.get("consistency_pattern", {})
                        diff_map = update_map.get("block_diff_json", {})
                        for batch in scanner.to_batches():
                            sids = batch.column("sample_id").to_pylist()
                            bidxs = batch.column("block_idx").to_pylist()
                            patterns = batch.column("consistency_pattern").to_pylist() if "consistency_pattern" in batch.column_names else [None] * len(batch)
                            diffs = batch.column("block_diff_json").to_pylist() if "block_diff_json" in batch.column_names else [None] * len(batch)
                            new_patterns = []
                            new_diffs = []
                            for i in range(len(batch)):
                                key = (sids[i], bidxs[i])
                                new_patterns.append(pat_map.get(key, patterns[i]))
                                new_diffs.append(diff_map.get(key, diffs[i]))
                            update_batches.append(pa.table({
                                "sample_id": pa.array(sids, type=pa.large_string()),
                                "block_idx": pa.array(bidxs, type=pa.int32()),
                                "consistency_pattern": pa.array(new_patterns, type=pa.large_string()),
                                "block_diff_json": pa.array(new_diffs, type=pa.large_string()),
                            }))
                        _lance_cache.invalidate(str(lp))
                    if update_batches:
                        update_table = pa.concat_tables(update_batches)
                        safe_merge(ds, update_table, ["sample_id", "block_idx"], context=f"cmcv/{cat}")
                except Exception as e:
                    print(f"[CMCV] 写回 {b.source_id}/{b.batch_id}/{cat}.lance 失败: {e}", file=sys.stderr)

        if ingest_path and ingest_path.exists() and page_tiers:
            sample_ids = list(page_tiers.keys())
            tiers = [page_tiers[sid] for sid in sample_ids]
            update_table = pa.table({
                "sample_id": pa.array(sample_ids, type=pa.large_string()),
                "difficulty": pa.array(tiers, type=pa.large_string()),
            })
            with _lance_write_lock:
                ds = _open_lance(ingest_path)
                safe_merge(ds, update_table, ["sample_id"], context="cmcv/ingest_tiers")
            _lance_cache.invalidate(str(ingest_path))

        for b in global_status.batches:
            b_manifests = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "manifests"
            for cat in ("text", "formula", "table"):
                lp = b_manifests / f"{cat}.lance"
                if lp.exists():
                    ensure_lance_indexes(lp, ["consistency_pattern", "block_idx"])

        progress_tracker.complete_task(task_id=ctx.task_id, message=f"完成 {len(page_tiers)} 页面")
        invalidate_status_cache()

    return task_manager.start(
        task_id=task_id, task_type="cmcv",
        source_id=source_id, batch_id=batch_id or "",
        target=_execute_cmcv,
    )


_cmcv_results_cache = {}  # key: (source_id, batch_id) → (result, lance_version)


@app.get("/api/cmcv/{source_id}/{batch_id}/results")
async def get_cmcv_results(source_id: str, batch_id: str, tier: str = None):
    """获取单个 batch 的 CMCV 比较结果"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        # 检查缓存：基于 Lance 版本
        cache_key = (source_id, batch_id)
        cached_result = _cmcv_results_cache.get(cache_key)
        if cached_result is not None:
            cached_data, cached_version = cached_result
            # 验证版本是否一致
            text_lp = manifests_dir / "text.lance"
            if text_lp.exists():
                try:
                    ds = _open_lance(text_lp)
                    current_version = getattr(ds, "version", None)
                    if current_version == cached_version:
                        # 应用 tier 过滤
                        if tier:
                            filtered_rows = [r for r in cached_data["rows"] if r.get("consistency_pattern") == tier]
                            # 重新计算 histogram
                            tier_histogram = {"easy": 0, "medium": 0, "hard": 0}
                            tier_type_histogram = {
                                "easy": {"text": 0, "formula": 0, "table": 0},
                                "medium": {"text": 0, "formula": 0, "table": 0},
                                "hard": {"text": 0, "formula": 0, "table": 0},
                            }
                            for row in filtered_rows:
                                pat = row.get("consistency_pattern", "")
                                bt = row.get("block_type", "text")
                                if pat == "all_agree":
                                    tier_histogram["easy"] += 1
                                    tier_type_histogram["easy"][bt] = tier_type_histogram["easy"].get(bt, 0) + 1
                                elif pat == "partial_agree":
                                    tier_histogram["medium"] += 1
                                    tier_type_histogram["medium"][bt] = tier_type_histogram["medium"].get(bt, 0) + 1
                                elif pat == "all_disagree":
                                    tier_histogram["hard"] += 1
                                    tier_type_histogram["hard"][bt] = tier_type_histogram["hard"].get(bt, 0) + 1
                            return {
                                "total_blocks": len(filtered_rows),
                                "total_pages": len(set(r["sample_id"] for r in filtered_rows)),
                                "tier_histogram": tier_histogram,
                                "tier_type_histogram": tier_type_histogram,
                                "pages": list({r["sample_id"]: r for r in filtered_rows}.values())[:100],
                            }
                        return {
                            "total_blocks": cached_data["total_blocks"],
                            "total_pages": cached_data["total_pages"],
                            "tier_histogram": cached_data["tier_histogram"],
                            "tier_type_histogram": cached_data.get("tier_type_histogram"),
                            "pages": cached_data["pages"],
                        }
                except Exception:
                    pass

        sample_keys = set()
        try:
            sp = batch_dir / "artifacts" / "element_samples.json"
            if sp.exists():
                sd = _read_json_cached(sp)
                for s in sd.get("samples", []):
                    sample_keys.add((s["sample_id"], s["block_idx"]))
        except Exception:
            pass

        rows = []
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                col_names = set(ds.schema.names)
                read_cols = ["sample_id", "block_idx", "block_type", "consistency_pattern", "block_diff_json"]
                # 流式读取，避免全表加载
                available = [c for c in read_cols if c in col_names]
                scanner = ds.scanner(columns=available)
                for batch in scanner.to_batches():
                    rows.extend(batch.to_pylist())
            except Exception:
                pass

        if tier:
            rows = [r for r in rows if r.get("consistency_pattern") == tier]

        tier_histogram = {"easy": 0, "medium": 0, "hard": 0}
        tier_type_histogram = {
            "easy": {"text": 0, "formula": 0, "table": 0},
            "medium": {"text": 0, "formula": 0, "table": 0},
            "hard": {"text": 0, "formula": 0, "table": 0},
        }
        for row in rows:
            pat = row.get("consistency_pattern", "")
            bt = row.get("block_type", "text")
            if pat == "all_agree":
                tier_histogram["easy"] += 1
                tier_type_histogram["easy"][bt] = tier_type_histogram["easy"].get(bt, 0) + 1
            elif pat == "partial_agree":
                tier_histogram["medium"] += 1
                tier_type_histogram["medium"][bt] = tier_type_histogram["medium"].get(bt, 0) + 1
            elif pat == "all_disagree":
                tier_histogram["hard"] += 1
                tier_type_histogram["hard"][bt] = tier_type_histogram["hard"].get(bt, 0) + 1

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

        result = {
            "total_blocks": len(rows),
            "total_pages": len(page_stats),
            "tier_histogram": tier_histogram,
            "tier_type_histogram": tier_type_histogram,
            "pages": list(page_stats.values())[:100],
            "rows": rows,  # 用于缓存
        }

        # 存储缓存（基于 Lance 版本）
        try:
            text_lp = manifests_dir / "text.lance"
            if text_lp.exists():
                ds = _open_lance(text_lp)
                lance_version = getattr(ds, "version", None)
                _cmcv_results_cache[(source_id, batch_id)] = (result, lance_version)
        except Exception:
            pass

        return {
            "total_blocks": result["total_blocks"],
            "total_pages": result["total_pages"],
            "tier_histogram": result["tier_histogram"],
            "tier_type_histogram": result.get("tier_type_histogram"),
            "pages": result["pages"],
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
                ds = _open_lance(lp)
                col_names = set(ds.schema.names)
                read_cols = ["sample_id", "block_idx", "block_type", "bbox_json",
                             "paddle_text", "glm_text", "self_text",
                             "paddle_table", "glm_table", "self_table",
                             "paddle_formula", "glm_formula", "self_formula",
                             "consistency_pattern", "block_diff_json"]
                available = [c for c in read_cols if c in col_names]
                # 使用 Lance filter 直接查询，避免加载全表
                tbl = ds.to_table(columns=available, filter=f"sample_id = '{sample_id}'")
                for i in range(tbl.num_rows):
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
                ds = _open_lance(lance_path)
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
                        ds = _open_lance(lp)
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
                        cached = _read_json_cached(cpath)
                        if cached.get("strategy") in ("element_cluster", "difficulty_aware"):
                            continue
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
                    cached = _read_json_cached(cache_path)
                    # Element Clustering 和 Difficulty Aware 需要重新生成，不返回缓存
                    # Page-Level 聚类（page_cmcv_*）应该返回缓存
                    if cached.get("strategy") in ("element_cluster", "difficulty_aware"):
                        pass  # 继续生成新的
                    else:
                        return cached  # 直接返回缓存（包括 page_cmcv_* 和无 strategy）
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
                                cached = _read_json_cached(cpath)
                                if cached.get("strategy") in ("element_cluster", "difficulty_aware"):
                                    continue
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

        ds = _open_lance(manifest_path)

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
        import gc as _gc
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
                del tbl, scanner
                _gc.collect()
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
        _bucket_summary_cache.invalidate(cache_path)

        # 重新抽样时清除预览模式的 layout 结果缓存（不影响 lance 拆分文件）
        layout_cache = cache_path.parent / "layout_results.json"
        if layout_cache.exists():
            layout_cache.unlink()

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _strip_heavy_summary_fields(page_cmcv, page_cmcv_sample):
    """去掉 block_details(29MB) 和 page_cmcv_sample.buckets(38MB)——摘要模式下前端不需要。"""
    if isinstance(page_cmcv, dict):
        page_cmcv.pop("block_details", None)
    if isinstance(page_cmcv_sample, dict):
        page_cmcv_sample.pop("buckets", None)
    return page_cmcv, page_cmcv_sample


@app.get("/api/bucket-summary")
@app.get("/api/bucket-summary/{source_id}")
@app.get("/api/bucket-summary/{source_id}/{batch_id}")
async def get_bucket_summary(source_id: str = "", batch_id: str = ""):
    """返回 bucket_samples.json 的摘要字段，不含 buckets 样本数据（占 95% 体积）。"""
    try:
        if not source_id:
            # 没有选源时返回空摘要
            return {
                "source_id": "", "batch_id": "",
                "count_per_bucket": 0, "bucket_count": 0,
                "bucket_sizes": {}, "total_sampled": 0,
                "strategy": None, "partition_tiers": None,
                "ratios": None, "batch_info": [],
                "page_cmcv": None, "page_cmcv_sample": None,
            }

        source = registry.get(source_id)

        if not batch_id:
            # 空 batchId 时：直接读源根目录下的 bucket_samples.json
            batch_dir = source.resolve_batch_dir("")
            cache_path = batch_dir / "artifacts" / "bucket_samples.json"
            if not cache_path.exists():
                return {
                    "source_id": source_id, "batch_id": "",
                    "count_per_bucket": 0, "bucket_count": 0,
                    "bucket_sizes": {}, "total_sampled": 0,
                    "strategy": None, "partition_tiers": None,
                    "ratios": None, "batch_info": [],
                    "page_cmcv": None, "page_cmcv_sample": None,
                }
            cached = _bucket_summary_cache.get(cache_path)
            if cached is None:
                cached = _read_json_cached(cache_path)
                _bucket_summary_cache.set(cache_path, cached)
            page_cmcv, page_cmcv_sample = _strip_heavy_summary_fields(
                cached.get("page_cmcv"), cached.get("page_cmcv_sample"))
            return {
                "source_id": source_id,
                "batch_id": cached.get("batch_id", ""),
                "count_per_bucket": cached.get("count_per_bucket", 0),
                "bucket_count": cached.get("bucket_count", 0),
                "bucket_sizes": cached.get("bucket_sizes", {}),
                "total_sampled": cached.get("total_sampled", 0),
                "strategy": cached.get("strategy"),
                "partition_tiers": cached.get("partition_tiers"),
                "ratios": cached.get("ratios"),
                "batch_info": [{
                    "source_id": source_id,
                    "batch_id": cached.get("batch_id", ""),
                    "total_sampled": cached.get("total_sampled", 0),
                    "partition_tiers": cached.get("partition_tiers"),
                    "strategy": cached.get("strategy"),
                }],
                "page_cmcv": page_cmcv,
                "page_cmcv_sample": page_cmcv_sample,
                "cached_at": cached.get("cached_at"),
            }

        batch_dir = source.resolve_batch_dir(batch_id)
        cache_path = batch_dir / "artifacts" / "bucket_samples.json"

        if not cache_path.exists():
            return {
                "source_id": source_id, "batch_id": batch_id,
                "count_per_bucket": 0, "bucket_count": 0,
                "bucket_sizes": {}, "total_sampled": 0,
                "strategy": None, "partition_tiers": None,
                "ratios": None, "batch_info": [],
                "page_cmcv": None, "page_cmcv_sample": None,
            }

        cached = _bucket_summary_cache.get(cache_path)
        if cached is None:
            cached = _read_json_cached(cache_path)
            _bucket_summary_cache.set(cache_path, cached)

        page_cmcv, page_cmcv_sample = _strip_heavy_summary_fields(
            cached.get("page_cmcv"), cached.get("page_cmcv_sample"))

        return {
            "source_id": source_id,
            "batch_id": batch_id,
            "count_per_bucket": cached.get("count_per_bucket", 0),
            "bucket_count": cached.get("bucket_count", 0),
            "bucket_sizes": cached.get("bucket_sizes", {}),
            "total_sampled": cached.get("total_sampled", 0),
            "strategy": cached.get("strategy"),
            "partition_tiers": cached.get("partition_tiers"),
            "ratios": cached.get("ratios"),
            "batch_info": [{
                "source_id": source_id,
                "batch_id": batch_id,
                "total_sampled": cached.get("total_sampled", 0),
                "partition_tiers": cached.get("partition_tiers"),
                "strategy": cached.get("strategy"),
            }],
            "page_cmcv": page_cmcv,
            "page_cmcv_sample": page_cmcv_sample,
            "cached_at": cached.get("cached_at"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bucket-samples-batch")
async def get_bucket_samples_batch(request: Request, full: bool = False):
    """批量获取多个批次的抽样缓存摘要，一次请求返回所有结果。
    full=false 时只返回摘要（不含 buckets 详情），减少传输量。
    full=true 时返回完整数据（含 buckets）。"""
    try:
        targets = await request.json()  # [{source_id, batch_id}, ...]
        if not targets:
            return {}
        results = {}
        for t in targets:
            sid = t.get("source_id", "")
            bid = t.get("batch_id", "")
            if not sid or not bid:
                continue
            try:
                source = registry.get(sid)
                if not source:
                    print(f"[bucket-samples-batch] Source not found: {sid}", file=sys.stderr)
                    continue
                batch_dir = source.resolve_batch_dir(bid)
                cache_path = batch_dir / "artifacts" / "bucket_samples.json"
                if cache_path.exists():
                    cached = _bucket_summary_cache.get(cache_path)
                    if cached is None:
                        cached = _read_json_cached(cache_path)
                        _bucket_summary_cache.set(cache_path, cached)
                    # 如果缓存中 page_cmcv_sample 缺少 buckets，说明缓存过期
                    sample = cached.get("page_cmcv_sample")
                    if sample and not sample.get("buckets"):
                        _json_cache.invalidate(str(cache_path))
                        _bucket_summary_cache.invalidate(str(cache_path))
                        cached = _read_json_cached(cache_path)
                        _bucket_summary_cache.set(cache_path, cached)
                    # 跳过 Element Clustering 和 Difficulty Aware 抽样（这些是 Block 级别的）
                    # Page-Level 聚类（page_cmcv_*）应该正常返回
                    if cached.get("strategy") in ("element_cluster", "difficulty_aware"):
                        continue
                    total = cached.get("total_sampled", 0)
                    if total > 0:
                        if full:
                            # 返回完整数据
                            results[f"{sid}/{bid}"] = cached
                        else:
                            # 只返回摘要
                            page_cmcv = cached.get("page_cmcv")
                            if page_cmcv:
                                page_cmcv = dict(page_cmcv)
                                page_cmcv.pop("block_details", None)
                            page_cmcv_sample = cached.get("page_cmcv_sample")
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
                                "page_cmcv": page_cmcv,
                                "page_cmcv_sample": page_cmcv_sample,
                            }
            except Exception as e:
                print(f"[bucket-samples-batch] {sid}/{bid} error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
        return results
    except json.JSONDecodeError as e:
        print(f"[bucket-samples-batch] JSON decode error: {e}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[bucket-samples-batch] Unexpected error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bucket-samples-difficulty/{source_id}/{batch_id}")
async def get_difficulty_aware_samples(
    source_id: str,
    batch_id: str,
    easy_ratio: float | None = None,
    medium_ratio: float | None = None,
    hard_ratio: float | None = None,
    force: bool = False,
):
    """基于 Element Clustering + CMCV 的 Block 级别难度抽样。

    流程：
    1. 读取 element_clusters.json 获取每个 block 的 cluster_id
    2. 从 text/formula/table.lance 读取 consistency_pattern（CMCV 结果）
    3. 每个 block 根据 consistency_pattern 获得难度标签
    4. 按用户指定比例从各层级中抽样，保持 cluster 多样性
    """
    if easy_ratio is None:
        easy_ratio = float(get_config("ocr", "sampling", "easy_ratio", default=50))
    if medium_ratio is None:
        medium_ratio = float(get_config("ocr", "sampling", "medium_ratio", default=35))
    if hard_ratio is None:
        hard_ratio = float(get_config("ocr", "sampling", "hard_ratio", default=15))

    ratio_map = {"easy": easy_ratio, "medium": medium_ratio, "hard": hard_ratio}

    try:
        global_status = collect_global_status(registry)
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        # 检查缓存
        cache_path = batch_dir / "artifacts" / "element_bucket_samples.json"
        if not force and cache_path.exists():
            try:
                cached = _read_json_cached(cache_path)
                if cached.get("strategy") == "element_cluster":
                    cached_ratios = cached.get("ratios", {})
                    if (float(cached_ratios.get("easy", 0)) == easy_ratio and
                        float(cached_ratios.get("medium", 0)) == medium_ratio and
                        float(cached_ratios.get("hard", 0)) == hard_ratio):
                        return cached
            except Exception:
                pass

        # 1. 读取 element_clusters.json
        clusters_path = batch_dir / "artifacts" / "element_clusters.json"
        if not clusters_path.exists():
            raise HTTPException(status_code=400, detail="未找到 Element 聚类结果，请先运行 Element Clustering")
        clusters_data = _read_json_cached(clusters_path)

        # 收集所有 batch 的抽样 keys（用于确定哪些 block 有 CMCV 结果）
        all_sample_keys = set()
        for b in global_status.batches:
            try:
                sp = registry.get(b.source_id).resolve_batch_dir(b.batch_id) / "artifacts" / "element_samples.json"
                if sp.exists():
                    sd = _read_json_cached(sp)
                    for s in sd.get("samples", []):
                        all_sample_keys.add((s["sample_id"], s["block_idx"]))
            except Exception:
                pass

        # 2. 从抽样 block 的 CMCV 结果推算每个 cluster 的难度分布
        PATTERN_TIER = {"all_agree": "easy", "partial_agree": "medium", "all_disagree": "hard"}

        # cluster_diff_map[(cat, cid)] = {"easy": N, "medium": N, "hard": N}
        cluster_diff_map: dict[tuple, dict] = {}
        # 同时构建 block_tier_map，避免第二次 Lance 扫描
        block_tier_map_global: dict[str, str] = {}

        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                if "consistency_pattern" not in ds.schema.names:
                    continue
                # 流式读取，避免全表加载
                scanner = ds.scanner(columns=["sample_id", "block_idx", "consistency_pattern"])
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        bk = f"{row['sample_id']}:{row['block_idx']}"
                        pattern = row.get("consistency_pattern")
                        if pattern and pattern in PATTERN_TIER:
                            block_tier_map_global[bk] = PATTERN_TIER[pattern]

                        key = (row["sample_id"], row["block_idx"])
                        if key not in all_sample_keys:
                            continue
                        if not pattern:
                            continue
                        tier = PATTERN_TIER.get(pattern)
                        if not tier:
                            continue
                        cat_info = clusters_data.get(cat, {})
                        labels_map = cat_info.get("labels", {})
                        ckey = f"{row['sample_id']}:{row['block_idx']}"
                        cid = labels_map.get(ckey)
                    if cid is None:
                        continue
                    cluster_key = (cat, cid)
                    cluster_diff_map.setdefault(cluster_key, {"easy": 0, "medium": 0, "hard": 0})
                    cluster_diff_map[cluster_key][tier] += 1
            except Exception as e:
                print(f"[difficulty-samples] 读取 {cat}.lance 失败: {e}", file=sys.stderr)
                continue

        if not cluster_diff_map:
            return {"source_id": source_id, "batch_id": batch_id,
                    "error": "无 CMCV 结果，请先运行 Element CMCV",
                    "bucket_count": 0, "total_sampled": 0, "buckets": {}}

        # 3. 按比例归属：每个 cluster 的 block 按 CMCV 分布比例分摊到三个 tier
        tier_totals = {"easy": 0, "medium": 0, "hard": 0}
        tier_cluster_sizes: dict[str, list] = {"easy": [], "medium": [], "hard": []}

        for cat in ("text", "formula", "table"):
            cat_info = clusters_data.get(cat, {})
            labels_map = cat_info.get("labels", {})
            if not labels_map:
                continue
            cluster_sizes: dict[int, int] = {}
            for key, cid in labels_map.items():
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
            for cid, cluster_size in cluster_sizes.items():
                diff_counts = cluster_diff_map.get((cat, cid), {"easy": 0, "medium": 0, "hard": 0})
                total_labeled = sum(diff_counts.values())
                for tier in ("easy", "medium", "hard"):
                    if total_labeled > 0:
                        proportion = diff_counts[tier] / total_labeled
                    else:
                        proportion = 1.0 / 3
                    proportional_size = int(cluster_size * proportion)
                    tier_totals[tier] += proportional_size
                    tier_cluster_sizes[tier].append((cat, cid, proportional_size))

        grand_total = sum(tier_totals.values())
        if grand_total == 0:
            return {"source_id": source_id, "batch_id": batch_id,
                    "error": "无聚类数据", "bucket_count": 0, "total_sampled": 0, "buckets": {}}

        # 5. 从所有 cluster 按比例抽样，然后按 block 的 CMCV 结果分配 tier
        import random

        buckets = {}
        cluster_diff_stats = {}

        for cat in ("text", "formula", "table"):
            cat_info = clusters_data.get(cat, {})
            labels_map = cat_info.get("labels", {})
            if not labels_map:
                continue
            cluster_sizes: dict[int, int] = {}
            for key, cid in labels_map.items():
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

            for cid, cluster_size in cluster_sizes.items():
                diff_counts = cluster_diff_map.get((cat, cid), {"easy": 0, "medium": 0, "hard": 0})
                total_labeled = sum(diff_counts.values())
                if total_labeled == 0:
                    continue

                cluster_key = f"{cat}_C{cid}"
                cluster_keys = [k for k, cid_val in labels_map.items() if cid_val == cid]
                if not cluster_keys:
                    continue

                # 按比例确定每个 tier 从该 cluster 抽多少
                tier_samples = {}
                for tier in ("easy", "medium", "hard"):
                    proportion = diff_counts[tier] / total_labeled
                    pct = ratio_map.get(tier, 0)
                    target = max(1, int(cluster_size * proportion * pct / 100)) if pct > 0 else 0
                    if target > 0:
                        tier_samples[tier] = min(target, cluster_size)

                # 从 cluster 抽一个总池，然后按比例分配
                total_target = sum(tier_samples.values())
                if total_target == 0:
                    continue
                sampled_keys = random.sample(cluster_keys, min(total_target, len(cluster_keys)))

                # 按 tier 分配 sampled blocks（使用第一轮扫描构建的 block_tier_map_global）
                for tier_name in ("easy", "medium", "hard"):
                    tier_target = tier_samples.get(tier_name, 0)
                    if tier_target == 0:
                        continue
                    tier_bucket_key = f"{cluster_key}_{tier_name}"
                    # 优先选属于该 tier 的 block
                    tier_blocks = [k for k in sampled_keys if block_tier_map_global.get(k) == tier_name]
                    other_blocks = [k for k in sampled_keys if block_tier_map.get(k) != tier_name]
                    chosen = tier_blocks[:tier_target]
                    if len(chosen) < tier_target:
                        chosen.extend(other_blocks[:tier_target - len(chosen)])
                    sampled_ids = [k.split(":")[0] for k in chosen]
                    buckets[tier_bucket_key] = sampled_ids
                    pct = ratio_map.get(tier_name, 0)
                    cluster_diff_stats[tier_bucket_key] = {
                        "tier": tier_name,
                        "ratio": pct,
                        "sampled": len(sampled_ids),
                        "total": tier_target,
                        "easy_count": diff_counts.get("easy", 0),
                        "medium_count": diff_counts.get("medium", 0),
                        "hard_count": diff_counts.get("hard", 0),
                    }

        total_sampled = sum(len(v) for v in buckets.values())

        tier_counts = {}
        for tier in ("easy", "medium", "hard"):
            tier_counts[tier] = {"blocks": tier_totals[tier], "ratio": ratio_map.get(tier, 0),
                                 "target": max(0, int(tier_totals[tier] * ratio_map.get(tier, 0) / 100))}

        result = {
            "source_id": source_id,
            "batch_id": batch_id,
            "strategy": "element_cluster",
            "ratios": {"easy": easy_ratio, "medium": medium_ratio, "hard": hard_ratio},
            "tier_summary": tier_counts,
            "bucket_count": len(buckets),
            "bucket_diff_stats": cluster_diff_stats,
            "total_sampled": total_sampled,
            "buckets": buckets,
            "cached_at": datetime.now().isoformat(timespec='seconds'),
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Judge-and-Refine (Hard Case 自动纠错) ──────────────────────────────────

@app.get("/api/hard-cases/{source_id}/{batch_id}")
async def get_hard_cases_list(source_id: str, batch_id: str, tier: str = "all_disagree", block_type: str = "all", limit: int = 1000):
    """轻量级 block 列表（按 tier 筛选，向量化防爆 OOM）。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        tier_filter = f"consistency_pattern = '{tier}'" if tier != "all" else None
        hard_blocks = []
        real_total = 0
        safe_limit = min(limit, 1000)

        for cat in ("text", "formula", "table"):
            if block_type != "all" and cat != block_type:
                continue
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                cols = ds.schema.names
                if "consistency_pattern" not in cols:
                    continue
                if tier_filter:
                    real_total += ds.count_rows(filter=tier_filter)
                else:
                    real_total += ds.count_rows()
                remaining = safe_limit - len(hard_blocks)
                if remaining <= 0:
                    continue
                light_cols = [c for c in ("sample_id", "block_idx", "block_type",
                                          "consistency_pattern", "block_diff_json",
                                          "judged", "corrected",
                                          "needs_expert", "judge_confidence") if c in cols]
                scanner = ds.scanner(
                    columns=light_cols,
                    filter=tier_filter,
                    limit=remaining,
                )
                pa_table = scanner.to_table()
                if len(pa_table) > 0:
                    rows = pa_table.to_pylist()
                    if block_type != "all" and "block_type" in light_cols:
                        rows = [r for r in rows if r.get("block_type") == block_type]
                    hard_blocks.extend(rows)
            except Exception as e:
                print(f"[hard-cases] 读 {cat}.lance 失败: {e}", file=sys.stderr)

        # 单次循环完成统计，避免 3 次全量遍历
        judged_count = 0
        corrected_count = 0
        needs_expert_count = 0
        for b in hard_blocks:
            if b.get("judged"):
                judged_count += 1
            if b.get("corrected"):
                corrected_count += 1
            if b.get("needs_expert"):
                needs_expert_count += 1

        return {
            "total": real_total,
            "loaded": len(hard_blocks),
            "judged": judged_count,
            "corrected": corrected_count,
            "needs_expert": needs_expert_count,
            "blocks": hard_blocks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hard-cases/{source_id}/{batch_id}/{cat}/{sample_id}/{block_idx}")
async def get_hard_cases_detail(source_id: str, batch_id: str, cat: str, sample_id: str, block_idx: int):
    """单条 Hard block 详情（含 OCR 文本 + diff JSON）。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        lp = batch_dir / "manifests" / f"{cat}.lance"
        if not lp.exists():
            raise HTTPException(status_code=404, detail=f"{cat}.lance 不存在")

        ds = _open_lance(lp)
        cols = ds.schema.names
        detail_cols = [c for c in ("sample_id", "block_idx", "block_type", "bbox_json",
                                    "consistency_pattern", "block_diff_json",
                                    "paddle_text", "glm_text", "self_text",
                                    "paddle_table", "glm_table", "self_table",
                                    "paddle_formula", "glm_formula", "self_formula",
                                    "judged", "corrected", "needs_expert",
                                    "judge_confidence", "judge_rounds") if c in cols]
        batches = list(ds.to_batches(
            columns=detail_cols,
            filter=f"sample_id = '{sample_id}' AND block_idx = {block_idx}",
        ))
        for batch in batches:
            if len(batch) > 0:
                row = {c: batch.column(c)[0].as_py() for c in detail_cols}
                if "block_diff_json" in row and isinstance(row["block_diff_json"], str):
                    try:
                        row["block_diff_json"] = json.loads(row["block_diff_json"])
                    except Exception:
                        pass
                if "bbox_json" in row and isinstance(row["bbox_json"], str):
                    try:
                        row["bbox_json"] = json.loads(row["bbox_json"])
                    except Exception:
                        pass
                return row
        raise HTTPException(status_code=404, detail="block 未找到")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/judge-refine/{source_id}/{batch_id}")
async def start_judge_refine(source_id: str, batch_id: str, request: Request):
    """启动 Judge-and-Refine 纠错流程。"""
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    max_rounds: int = body.get("max_rounds", 3)
    force: bool = body.get("force", False)

    task_id = f"judge_refine_{source_id}_{batch_id}"

    def _execute_judge_refine(ctx: TaskContext):
        from data_engine.ocr.judge_refine import JudgeRefineEngine

        source = registry.get(ctx.source_id)
        batch_dir = source.resolve_batch_dir(ctx.batch_id)
        manifests_dir = batch_dir / "manifests"

        hard_blocks = []
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                cols = ds.schema.names
                read_cols = ["sample_id", "block_idx", "block_type", "bbox_json"]
                for c in ("paddle_text", "glm_text", "self_text",
                          "paddle_table", "glm_table", "self_table",
                          "paddle_formula", "glm_formula", "self_formula",
                          "image_data", "judged", "consistency_pattern"):
                    if c in cols:
                        read_cols.append(c)
                # 流式读取，避免全表加载
                available_cols = [c for c in read_cols if c in cols]
                scanner = ds.scanner(columns=available_cols)
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        if row.get("consistency_pattern") != "all_disagree":
                            continue
                        if not force and row.get("judged"):
                            continue
                        row["_cat"] = cat
                        hard_blocks.append(row)
            except Exception:
                pass

        if not hard_blocks:
            progress_tracker.complete_task(task_id=ctx.task_id, message="无需纠错的 Hard block")
            return

        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="judge_refine",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=len(hard_blocks), message=f"开始纠错 {len(hard_blocks)} 个 Hard block",
        )

        engine = JudgeRefineEngine(max_rounds=max_rounds)

        for i, block in enumerate(hard_blocks):
            ctx.check_stop()

            cat = block.pop("_cat")
            sid = block["sample_id"]
            bidx = block["block_idx"]

            img_bytes = block.get("image_data")
            if isinstance(img_bytes, str):
                img_bytes = img_bytes.encode("latin-1")

            try:
                rounds = engine.judge_and_refine(
                    block_type=block.get("block_type", "text"),
                    paddle_text=block.get("paddle_text"),
                    glm_text=block.get("glm_text"),
                    self_text=block.get("self_text"),
                    paddle_table=block.get("paddle_table"),
                    glm_table=block.get("glm_table"),
                    self_table=block.get("self_table"),
                    paddle_formula=block.get("paddle_formula"),
                    glm_formula=block.get("glm_formula"),
                    self_formula=block.get("self_formula"),
                    original_image=img_bytes,
                )
            except Exception:
                rounds = []

            judged = True
            corrected = False
            needs_expert = True
            refined_text = None
            refined_table = None
            refined_formula = None
            confidence = 0.0
            rounds_count = len(rounds)
            error_locations = []

            if rounds:
                last = rounds[-1].result
                confidence = last.confidence
                needs_expert = last.needs_expert
                refined_text = last.corrected_text
                refined_table = last.corrected_table
                refined_formula = last.corrected_formula
                error_locations = last.error_locations
                corrected = (refined_text is not None or refined_table is not None or refined_formula is not None) and not needs_expert

            try:
                lp = manifests_dir / f"{cat}.lance"
                with _lance_write_lock:
                    ds = _open_lance(lp)
                    update_data = {
                        "judged": [True],
                        "corrected": [corrected],
                        "needs_expert": [needs_expert],
                        "judge_confidence": [confidence],
                        "judge_rounds": [rounds_count],
                        "judge_error_locations": [json.dumps(error_locations, ensure_ascii=False)],
                    }
                    if refined_text is not None:
                        update_data["judge_refined_text"] = [refined_text]
                    if refined_table is not None:
                        update_data["judge_refined_table"] = [refined_table if isinstance(refined_table, str) else json.dumps(refined_table, ensure_ascii=False)]
                    if refined_formula is not None:
                        update_data["judge_refined_formula"] = [refined_formula]
                    ds.update(update_data, where=f"sample_id = '{sid}' AND block_idx = {bidx}")
            except Exception:
                pass

            progress_tracker.update_progress(
                task_id=ctx.task_id, current=i + 1,
                message=f"Judge-Refine: {i + 1}/{len(hard_blocks)} ({'corrected' if corrected else 'needs_expert'})",
            )

        progress_tracker.complete_task(task_id=ctx.task_id, message=f"完成 {len(hard_blocks)} 个 Hard block 纠错")

    return task_manager.start(
        task_id=task_id, task_type="judge_refine",
        source_id=source_id, batch_id=batch_id,
        target=_execute_judge_refine,
    )


@app.get("/api/judge-refine/{source_id}/{batch_id}/results")
async def get_judge_refine_results(source_id: str, batch_id: str):
    """获取 Judge-and-Refine 纠错结果统计。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        stats = {"total": 0, "judged": 0, "corrected": 0, "needs_expert": 0}
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                if "judged" not in ds.schema.names:
                    continue
                # 流式读取
                scanner = ds.scanner(columns=["consistency_pattern", "judged", "corrected", "needs_expert"])
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        if row.get("consistency_pattern") != "all_disagree":
                            continue
                        stats["total"] += 1
                        if row.get("judged"):
                            stats["judged"] += 1
                        if row.get("corrected"):
                            stats["corrected"] += 1
                        if row.get("needs_expert"):
                            stats["needs_expert"] += 1
            except Exception:
                pass

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/layout-preview")
async def run_layout_preview(request: Request):
    """预览 layout 结果：优先读 Lance 缓存，无缓存时才跑模型"""
    try:
        body = await request.json()
        source_id: str = body.get("source_id", "")
        batch_id: str = body.get("batch_id", "")
        sample_ids: list[str] = body.get("sample_ids", [])
        print(f"[layout-preview] source_id={source_id!r} batch_id={batch_id!r} sample_ids={sample_ids[:3]}", file=sys.stderr)

        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path:
            raise HTTPException(status_code=404, detail="ingest manifest 不存在")

        import base64

        # 1. 从 Lance 读取已有的 layout 结果（只读，无需写锁）
        cached_blocks: dict[str, list[dict]] = {}
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if lp.exists():
                try:
                    ds = _open_lance(lp)
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

        # 2. 从 ingest.lance 读取图片（只读，无需写锁）
        ds = _open_lance(manifest_path)
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
                "image_url": f"/api/lancedb/{source_id}/{batch_id}/image/{sid}",
                "blocks": blocks,
                "block_count": len(blocks),
                "from_cache": from_cache,
            })

        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[layout-preview] ERROR: {e}", file=sys.stderr)
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
        cropped_img = _crop_block_image(sid, b.get("bbox", []))
        row = {
            "sample_id": sid,
            "block_idx": idx,
            "block_type": bt,
            "bbox_json": json.dumps(b.get("bbox", [])),
            "layout_confidence": float(b.get("confidence", 0)),
            "embedding": None,
            "consistency_pattern": None,
            "block_diff_json": None,
            "schema_version": "v1",
            "created_at": now,
            "judged": None, "corrected": None, "needs_expert": None,
            "judge_refined_text": None, "judge_refined_table": None,
            "judge_refined_formula": None, "judge_confidence": None,
            "judge_rounds": None, "judge_error_locations": None,
        }
        if bt in FORMULA_BLOCK_TYPES:
            row["image_data"] = cropped_img
            return "formula", row
        elif bt in TABLE_BLOCK_TYPES:
            row["image_data"] = cropped_img
            return "table", row
        else:
            row["image_data"] = cropped_img
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

    def _execute_layout(ctx: TaskContext):
        source = registry.get(ctx.source_id)
        batch_dir = source.resolve_batch_dir(ctx.batch_id)
        manifests_dir = batch_dir / "manifests"
        manifest_path = find_stage_manifest(manifests_dir, "ingest")
        if not manifest_path:
            progress_tracker.fail_task(ctx.task_id, "ingest manifest 不存在")
            return

        ids = list(sample_ids) if sample_ids else []
        if not ids:
            cpath = batch_dir / "artifacts" / "bucket_samples.json"
            if cpath.exists():
                try:
                    cached = _read_json_cached(cpath)
                    for samples in cached.get("buckets", {}).values():
                        for s in samples:
                            sid = s if isinstance(s, str) else s.get("sample_id", "")
                            if sid:
                                ids.append(sid)
                except Exception:
                    pass

        if not ids:
            progress_tracker.fail_task(ctx.task_id, "无抽样样本")
            return

        mode_label = "写入" if write_lance else "预览"
        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="layout_batch",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=len(ids), message=f"layout({mode_label}) {len(ids)} 样本",
        )

        from data_engine.ocr.layout_provider import get_layout_provider
        layout = get_layout_provider()
        errors: list[dict] = []
        layout_stats: dict[str, int] = {"text": 0, "formula": 0, "table": 0}
        all_results: list[dict] = []

        done_ids: set[str] = set()
        lance_write_mode = "overwrite"
        if write_lance:
            for cat in ("text", "formula", "table"):
                lp = manifests_dir / f"{cat}.lance"
                if lp.exists():
                    try:
                        ds = _open_lance(lp)
                        ids_col = ds.to_table(columns=["sample_id"]).column("sample_id").to_pylist()
                        done_ids.update(ids_col)
                    except Exception:
                        pass
            if done_ids:
                lance_write_mode = "append"
        else:
            lr_path = batch_dir / "artifacts" / "layout_results.json"
            if lr_path.exists():
                try:
                    cached = _read_json_cached(lr_path)
                    existing_results = cached.get("results", [])
                    for r in existing_results:
                        sid = r.get("sample_id", "")
                        if sid:
                            done_ids.add(sid)
                    all_results = list(existing_results)
                except Exception:
                    pass

        pending_ids = [sid for sid in ids if sid not in done_ids]
        done = len(ids) - len(pending_ids)

        if not pending_ids:
            progress_tracker.update_progress(ctx.task_id, current=done, message=f"全部 {done} 样本已完成，跳过 layout")
        else:
            progress_tracker.update_progress(ctx.task_id, current=done, message="加载模型中...")
            try:
                _ = layout._ensure_model()
                progress_tracker.update_progress(ctx.task_id, current=done, message=f"模型就绪，开始处理 {done}/{len(ids)}")
            except Exception as e:
                progress_tracker.fail_task(ctx.task_id, f"模型加载失败: {e}")
                return

            process_batch_size = 500

            for batch_start in range(0, len(pending_ids), process_batch_size):
                ctx.check_stop()

                batch_ids = pending_ids[batch_start:batch_start + process_batch_size]

                image_map: dict[str, bytes] = {}
                loaded_in_batch = 0
                try:
                    ds = _open_lance(manifest_path)
                    query_batch = get_config("layout", "query_batch", default=200)
                    for qi in range(0, len(batch_ids), query_batch):
                        ctx.check_stop()
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
                            ctx.task_id, current=done + loaded_in_batch,
                            message=f"加载图片 {done+1}-{done+loaded_in_batch}/{len(ids)}"
                        )
                except Exception as e:
                    print(f"[layout] 加载图片失败 batch {batch_start}: {e}", file=sys.stderr)

                batch_results: list[dict] = []
                ctx.check_stop()

                valid_sids = [sid for sid in batch_ids if image_map.get(sid)]
                missing_sids = [sid for sid in batch_ids if not image_map.get(sid)]
                done += len(missing_sids)

                if valid_sids:
                    import tempfile
                    tmp_dir = tempfile.mkdtemp(prefix="layout_batch_")
                    try:
                        for si, sid in enumerate(valid_sids):
                            ctx.check_stop()
                            tmp_path = Path(tmp_dir) / f"{sid}.png"
                            tmp_path.write_bytes(image_map[sid])
                            try:
                                blocks = layout.detect_layout(tmp_path)
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
                            except Exception as e:
                                errors.append({"sample_id": sid, "error": str(e)})
                            done += 1
                            if (si + 1) % 10 == 0 or si == len(valid_sids) - 1:
                                progress_tracker.update_progress(
                                    ctx.task_id, current=done,
                                    message=f"layout 推理 {done}/{len(ids)}"
                                )
                    finally:
                        import shutil
                        try:
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                        except Exception:
                            pass

                error_summary = f"（{len(errors)} 错误）" if errors else ""
                progress_tracker.update_progress(
                    ctx.task_id, current=done,
                    message=f"layout {done}/{len(ids)}{error_summary}"
                )
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
                        if "different schema" in str(e) or "did not match" in str(e):
                            try:
                                batch_write_stats = _write_layout_to_category_lances(
                                    batch_results, manifests_dir,
                                    flush_size=get_config("layout", "flush_size", default=10_000),
                                    write_mode="overwrite",
                                    image_map=image_map,
                                )
                                for k in layout_stats:
                                    layout_stats[k] += batch_write_stats.get(k, 0)
                                lance_write_mode = "append"
                            except Exception:
                                pass

        if write_lance:
            for cat in ("text", "formula", "table"):
                lp = manifests_dir / f"{cat}.lance"
                if lp.exists():
                    try:
                        ds = _open_lance(lp)
                        ds.optimize.compact_files()
                        ds.cleanup_old_versions(keep_versions=1)
                    except Exception:
                        pass

        if write_lance:
            split_msg = ""
            for cat in ("text", "formula", "table"):
                n = layout_stats.get(cat, 0)
                if n:
                    split_msg += f"{cat}: {n}, "
            split_msg = split_msg.rstrip(", ")
        else:
            split_msg = f"{len(all_results)} 样本"

        final_msg = f"完成 {done} 样本"
        if split_msg:
            final_msg += f"，{split_msg}"
        progress_tracker.complete_task(ctx.task_id, message=final_msg)

        if not write_lance and all_results:
            try:
                out_path = batch_dir / "artifacts" / "layout_results.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps({"results": all_results}, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

        invalidate_status_cache()

    task_manager.start(
        task_id=task_id, task_type="layout_batch",
        source_id=source_id, batch_id=batch_id,
        target=_execute_layout,
    )
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
                cached = _read_json_cached(json_path)
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
                ds = _open_lance(lp)
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
    source = registry.get(source_id)
    batch_dir = source.resolve_batch_dir(batch_id)
    manifests_dir = batch_dir / "manifests"

    no_image_sids: set[str] = set()
    no_image_counts: dict[str, int] = {}
    for cat in ("text", "formula", "table"):
        lp = manifests_dir / f"{cat}.lance"
        if lp.exists():
            ds = _open_lance(lp)
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

    ingest_path = find_stage_manifest(manifests_dir, "ingest")
    if not ingest_path or not ingest_path.exists():
        raise HTTPException(status_code=400, detail="ingest.lance 不存在，无法获取页面原图")

    task_id = f"repair_{source_id}_{batch_id}"
    total_pages = len(no_image_sids)

    def _execute_repair(ctx: TaskContext):
        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="repair",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=total_pages,
            message=f"补齐 {total_pages} 个页面的无图 block",
        )

        image_map: dict[str, bytes] = {}
        try:
            with _lance_write_lock:
                ds = _open_lance(ingest_path)
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
            progress_tracker.fail_task(ctx.task_id, f"加载页面图片失败: {e}")
            return

        if not image_map:
            progress_tracker.complete_task(ctx.task_id, "无法加载任何页面图片")
            return

        from data_engine.ocr.layout_provider import get_layout_provider
        layout = get_layout_provider()

        progress_tracker.update_progress(ctx.task_id, current=0, message="加载模型中...")
        try:
            _ = layout._ensure_model()
        except Exception as e:
            progress_tracker.fail_task(ctx.task_id, f"模型加载失败: {e}")
            return

        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp(prefix="repair_")
        errors: list[dict] = []
        repaired_results: list[dict] = []
        done = 0
        process_batch_size = 200
        pending_sids = [sid for sid in no_image_sids if sid in image_map]

        for batch_start in range(0, len(pending_sids), process_batch_size):
            ctx.check_stop()

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
                except Exception:
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
                ctx.task_id, current=done,
                message=f"layout {done}/{total_pages}{error_summary}",
            )

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        ctx.check_stop()

        progress_tracker.update_progress(ctx.task_id, current=done, message="写入 Lance...")
        repair_sids_set = set(no_image_sids)

        new_stats: dict[str, int] = {"text": 0, "formula": 0, "table": 0}
        if repaired_results:
            tmp_manifests = Path(tempfile.mkdtemp(prefix="repair_manifests_"))
            try:
                new_stats = _write_layout_to_category_lances(
                    repaired_results, tmp_manifests,
                    flush_size=10_000,
                    write_mode="create",
                    image_map=image_map,
                )

                for cat in ("text", "formula", "table"):
                    lp = manifests_dir / f"{cat}.lance"
                    new_lp = tmp_manifests / f"{cat}.lance"
                    if not lp.exists():
                        if new_lp.exists():
                            import shutil as _sh
                            _sh.copytree(str(new_lp), str(lp))
                        continue

                    # 流式过滤：逐批读取，只保留不在 repair_sids_set 中的行
                    with _lance_write_lock:
                        ds = _open_lance(lp)
                        scanner = ds.scanner()
                        kept_batches = []
                        for batch in scanner.to_batches():
                            sids = batch.column("sample_id").to_pylist()
                            keep_mask = [sid not in repair_sids_set for sid in sids]
                            if any(keep_mask):
                                kept_batches.append(batch.filter(pa.array(keep_mask)))
                        _lance_cache.invalidate(str(lp))

                    if kept_batches:
                        kept_table = pa.concat_tables(kept_batches)
                        with _lance_write_lock:
                            lance.write_dataset(kept_table, str(lp), mode="overwrite")
                    else:
                        with _lance_write_lock:
                            empty_table = pa.table({f.name: pa.array([], type=f.type) for f in ds.schema})
                            lance.write_dataset(empty_table, str(lp), mode="overwrite")

                    if new_lp.exists() and new_stats.get(cat, 0) > 0:
                        with _lance_write_lock:
                            new_ds = lance.dataset(str(new_lp))
                            # 流式读取新数据，避免全表加载
                            new_batches = list(new_ds.scanner().to_batches())
                            if new_batches:
                                new_table = pa.concat_tables(new_batches)
                                ds2 = lance.dataset(str(lp))
                                safe_merge(ds2, new_table, ["sample_id", "block_idx"],
                                           when_not_matched=True, context=f"repair/{cat}")
                                _lance_cache.invalidate(str(lp))
            finally:
                try:
                    shutil.rmtree(str(tmp_manifests), ignore_errors=True)
                except Exception:
                    pass

        err_msg = f"（{len(errors)} 错误）" if errors else ""
        final_msg = f"补齐完成: {done} 页面{err_msg}"
        progress_tracker.complete_task(ctx.task_id, message=final_msg)
        invalidate_status_cache()

    task_manager.start(
        task_id=task_id, task_type="repair",
        source_id=source_id, batch_id=batch_id,
        target=_execute_repair,
    )

    return {
        "message": f"已启动补齐任务: {total_pages} 个页面",
        "task_id": task_id,
        "no_image": no_image_counts,
        "pages": total_pages,
    }


# ─── Element-Type Layout 聚类 API ─────────────────────────────────────────────────────


@app.post("/api/element-clusters/{source_id}/{batch_id}")
async def start_element_clusters(source_id: str, batch_id: str, request: Request):
    """触发 Element-Type Layout 聚类（后台任务）。

    对 text/formula/table.lance 中的 block 分别用 SigLIP2 提取特征向量，独立做 KMeans 聚类。
    """
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    max_k: int = body.get("max_k", 10)

    source = registry.get(source_id)
    batch_dir = source.resolve_batch_dir(batch_id)
    manifests_dir = batch_dir / "manifests"

    has_any = any((manifests_dir / f"{cat}.lance").exists() for cat in ("text", "formula", "table"))
    if not has_any:
        raise HTTPException(status_code=400, detail="未找到 text/formula/table.lance，请先运行 layout 拆分")

    task_id = f"element_clusters_{source_id}_{batch_id}"

    def _execute_element_clusters(ctx: TaskContext):
        import json
        import lance
        from data_engine.ocr.layout_features import cluster_all_types

        total_blocks = 0
        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if lp.exists():
                try:
                    total_blocks += lance.dataset(str(lp)).count_rows()
                except Exception:
                    pass

        progress_tracker.start_task(
            ctx.task_id, task_type="element_clusters",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=max(total_blocks, 1),
            message=f"Element-Type 聚类 {total_blocks} blocks..."
        )

        def cb(cur, tot, msg):
            progress_tracker.update_progress(ctx.task_id, current=cur, total=tot, message=msg)

        result = cluster_all_types(manifests_dir, max_k=max_k, progress_callback=cb)

        out_path = batch_dir / "artifacts" / "element_clusters.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

        summary_parts = []
        for cat in ("text", "formula", "table"):
            info = result.get(cat, {})
            if info.get("total_blocks", 0) > 0:
                summary_parts.append(f"{cat}: {info['n_clusters']}簇/{info['total_blocks']}块")
        summary = ", ".join(summary_parts) if summary_parts else "无结果"

        progress_tracker.complete_task(ctx.task_id, f"完成: {summary}")

    return task_manager.start(
        task_id=task_id, task_type="element_clusters",
        source_id=source_id, batch_id=batch_id,
        target=_execute_element_clusters,
    )


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
        
        result = _read_json_cached(out_path)
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

        cluster_data = _read_json_cached(cluster_path)

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
                ds = _open_lance(lance_path)
                # 流式读取 key 列，避免全表加载
                key_set: set[str] = set()
                scanner = ds.scanner(columns=["sample_id", "block_idx"])
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        key = f"{row.get('sample_id', '')}:{row.get('block_idx', 0)}"
                        key_set.add(key)
            except Exception:
                summary[cat] = {"clusters": len(cluster_members), "sampled": 0}
                continue

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
        data = _read_json_cached(out_path)
        return {
            "status": "completed",
            "per_cluster": data.get("per_cluster", 0),
            "summary": data.get("summary", {}),
            "total_sampled": data.get("total_sampled", sum(s.get("sampled", 0) for s in data.get("summary", {}).values())),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/page-cmcv/{source_id}/{batch_id}")
async def page_cmcv_classify(
    source_id: str,
    batch_id: str,
):
    """从 bucket_samples.json 读取每个 sample 的 blocks，按 block 比较多模型 OCR 一致性，
    判定 easy/medium/hard。分类模式，不做采样。
    """
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        ingest_path = batch_dir / "manifests" / "ingest.lance"
        cache_path = batch_dir / "artifacts" / "bucket_samples.json"

        if not cache_path.exists():
            raise HTTPException(status_code=400, detail="bucket_samples.json 不存在，请先运行 Page OCR")

        cached = _read_json_cached(cache_path)

        buckets = cached.get("buckets", {})

        # 获取每个 IVF 分区的实际大小
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        ingest_path = batch_dir / "manifests" / "ingest.lance"
        partition_sizes = {}  # partition_name -> actual size
        try:
            ds = _open_lance(ingest_path)
            stats = ds.index_statistics("idx_embedding_ivf")
            parts = stats["indices"][0].get("partitions", [])
            for i, p in enumerate(parts):
                partition_sizes[f"P{i}"] = p.get("size", 0)
        except Exception as e:
            print(f"[page-cmcv] 获取分区大小失败: {e}", file=sys.stderr)

        # CMCV：对每个 sample 的每个 block 计算模型一致性
        # Page CMCV 用 token fallback（快），不启用视觉渲染
        from data_engine.ocr.cmcv import (
            text_similarity, cdm_similarity, _normalize_latex,
            normalize_table, teds_similarity,
        )

        PATTERN_TIER = {"all_agree": "easy", "partial_agree": "medium", "all_disagree": "hard"}
        page_tiers = {}  # sid -> "easy"/"medium"/"hard"
        block_details = {}  # sid -> [{"block_idx", "pattern", "similarities"}, ...]
        page_high = float(get_config("ocr", "cmcv", "page_high_threshold", default=0.8))
        page_low = float(get_config("ocr", "cmcv", "page_low_threshold", default=0.4))

        # 按 block 类型收集可用的模型列
        model_cols: dict[str, list[str]] = {"text": [], "formula": [], "table": []}
        for bt in ("text", "formula", "table"):
            for prefix in ("paddle", "glm", "self"):
                col = f"{prefix}_{bt}"
                found = False
                for tier_samples in buckets.values():
                    for s in tier_samples:
                        if isinstance(s, dict):
                            for b in s.get("blocks", []):
                                if b.get(col):
                                    model_cols[bt].append(col)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break

        def _pick_sim_fn(block_type: str):
            """根据 block 类型选择相似度函数（Page CMCV 用 token fallback）"""
            if block_type == "formula":
                return cdm_similarity, "cdm_token"
            elif block_type == "table":
                return lambda a, b: teds_similarity(
                    {"html": a} if a else None,
                    {"html": b} if b else None,
                ), "teds"
            else:
                return text_similarity, "levenshtein"

        for tier_name, tier_samples in buckets.items():
            for s in tier_samples:
                if not isinstance(s, dict):
                    continue
                sid = s.get("sample_id", "")
                blocks = s.get("blocks", [])
                if not sid or not blocks:
                    continue

                block_patterns = []
                block_scores = []
                for b in blocks:
                    bt = b.get("block_type", "text")
                    cols = model_cols.get(bt, model_cols["text"])
                    if not cols:
                        block_scores.append({"block_idx": b.get("block_idx", 0), "type": bt, "method": None, "pattern": None, "avg_similarity": None})
                        continue
                    texts = [b.get(col, "") for col in cols]
                    sim_fn, method = _pick_sim_fn(bt)
                    # formula 需要 normalize，table 需要 normalize_table
                    if bt == "formula":
                        texts = [_normalize_latex(t) for t in texts]
                    # 计算两两相似度
                    sims = []
                    for i in range(len(texts)):
                        for j in range(i + 1, len(texts)):
                            s = sim_fn(texts[i], texts[j])
                            sims.append(s if s is not None else 0.0)
                    avg_sim = sum(sims) / len(sims) if sims else 1.0

                    if avg_sim >= page_high:
                        pattern = "all_agree"
                    elif avg_sim >= page_low:
                        pattern = "partial_agree"
                    else:
                        pattern = "all_disagree"

                    block_patterns.append(pattern)
                    block_scores.append({"block_idx": b.get("block_idx", 0), "type": bt, "method": method, "pattern": pattern, "avg_similarity": round(avg_sim, 3)})

                # 页面难度 = 所有 block 中最差的
                if "all_disagree" in block_patterns:
                    page_tier = "hard"
                elif "partial_agree" in block_patterns:
                    page_tier = "medium"
                else:
                    page_tier = "easy"

                page_tiers[sid] = page_tier
                # 给每个 block 标注 page 级别的 tier
                for bs in block_scores:
                    bs["_tier"] = page_tier
                block_details[sid] = block_scores

        # 按分区分组，每个分区内按难度分层
        partition_tier_pages = {}  # P0 -> {easy: [...], medium: [...], hard: [...]}
        for tier_name, tier_samples in buckets.items():
            partition_tier_pages[tier_name] = {"easy": [], "medium": [], "hard": []}
            for s in tier_samples:
                if not isinstance(s, dict):
                    continue
                sid = s.get("sample_id", "")
                if sid in page_tiers:
                    partition_tier_pages[tier_name][page_tiers[sid]].append({
                        "tier": tier_name, "sample_id": sid, "blocks": block_details.get(sid, [])
                    })

        # 全局 tier 统计
        tier_counts = {"easy": 0, "medium": 0, "hard": 0}
        for ptp in partition_tier_pages.values():
            for ctier in ("easy", "medium", "hard"):
                tier_counts[ctier] += len(ptp[ctier])
        # 统计每个 tier 的实际 block 数
        tier_block_counts = {"easy": 0, "medium": 0, "hard": 0}
        for ptp in partition_tier_pages.values():
            for ctier in ("easy", "medium", "hard"):
                tier_block_counts[ctier] += sum(len(p.get("blocks", [])) for p in ptp[ctier])
        total_blocks = sum(tier_block_counts.values())
        total_pages = sum(tier_counts.values())

        result = {
            "source_id": source_id, "batch_id": batch_id,
            "strategy": "page_cmcv_classify",
            "tier_summary": tier_counts,
            "tier_block_summary": tier_block_counts,
            "total_sampled": sum(len(v) for ptp in partition_tier_pages.values() for v in ptp.values()),
            "total_blocks": total_blocks,
            "total_pages": total_pages,
            "block_details": block_details,
        }

        cached["page_cmcv"] = result
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
        _bucket_summary_cache.invalidate(cache_path)
        _json_cache.invalidate(str(cache_path))

        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/page-cmcv-sample/{source_id}/{batch_id}")
async def page_cmcv_sample(
    source_id: str,
    batch_id: str,
    easy_ratio: float = 0.5,
    medium_ratio: float = 1.0,
    hard_ratio: float = 2.0,
    force: bool = False,
):
    """基于 Page CMCV 分类结果，按分区+比例从 ingest.lance 抽取原始页面图片。"""
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        ingest_path = batch_dir / "manifests" / "ingest.lance"
        cache_path = batch_dir / "artifacts" / "bucket_samples.json"

        if not cache_path.exists():
            raise HTTPException(status_code=400, detail="bucket_samples.json 不存在，请先运行 Page OCR")

        cached = _read_json_cached(cache_path)

        # 缓存短路：非 force 且已有 page_cmcv_sample 结果直接返回
        if not force:
            existing = cached.get("page_cmcv_sample")
            if existing and existing.get("total_sampled", 0) > 0:
                return existing

        buckets = cached.get("buckets", {})

        # 检测可用的 OCR 模型列
        model_cols = []
        for prefix in ("paddle", "glm", "self"):
            col = f"{prefix}_text"
            for tier_samples in buckets.values():
                for s in tier_samples:
                    if isinstance(s, dict):
                        for b in s.get("blocks", []):
                            if b.get(col):
                                model_cols.append(col)
                                break
                    if model_cols:
                        break
                if model_cols:
                    break

        if len(model_cols) < 2:
            raise HTTPException(status_code=400, detail=f"需要至少 2 个模型的 OCR 结果做比较，当前只有 {model_cols}")

        from data_engine.ocr.cmcv import text_similarity

        page_high = float(get_config("ocr", "cmcv", "page_high_threshold", default=0.8))
        page_low = float(get_config("ocr", "cmcv", "page_low_threshold", default=0.4))

        page_tiers = {}
        for tier_name, tier_samples in buckets.items():
            for s in tier_samples:
                if not isinstance(s, dict):
                    continue
                sid = s.get("sample_id", "")
                blocks = s.get("blocks", [])
                if not sid or not blocks:
                    continue
                block_patterns = []
                for b in blocks:
                    texts = [b.get(col, "") for col in model_cols]
                    sims = []
                    for i in range(len(texts)):
                        for j in range(i + 1, len(texts)):
                            sims.append(text_similarity(texts[i], texts[j]))
                    avg_sim = sum(sims) / len(sims) if sims else 1.0
                    if avg_sim >= page_high:
                        block_patterns.append("all_agree")
                    elif avg_sim >= page_low:
                        block_patterns.append("partial_agree")
                    else:
                        block_patterns.append("all_disagree")
                if "all_disagree" in block_patterns:
                    page_tiers[sid] = "hard"
                elif "partial_agree" in block_patterns:
                    page_tiers[sid] = "medium"
                else:
                    page_tiers[sid] = "easy"

        partition_tier_pages = {}
        for tier_name, tier_samples in buckets.items():
            partition_tier_pages[tier_name] = {"easy": [], "medium": [], "hard": []}
            for s in tier_samples:
                if not isinstance(s, dict):
                    continue
                sid = s.get("sample_id", "")
                if sid in page_tiers:
                    partition_tier_pages[tier_name][page_tiers[sid]].append({
                        "tier": tier_name, "sample_id": sid,
                    })

        ratios = {"easy": easy_ratio, "medium": medium_ratio, "hard": hard_ratio}
        sampled = {"easy": [], "medium": [], "hard": []}
        partition_stats = {}

        import random, gc

        # 只打开一次 dataset，循环内复用，避免 FD 耗尽
        _ds = _lance_cache.get(str(ingest_path))
        stats = _ds.index_statistics("idx_embedding_ivf")
        parts = stats["indices"][0].get("partitions", [])
        centroids = stats["indices"][0].get("centroids", [])

        for i, part_info in enumerate(parts):
            part_size = part_info.get("size", 0)
            if part_size == 0:
                continue
            tier_name = f"P{i}"
            ptp = partition_tier_pages.get(tier_name, {"easy": [], "medium": [], "hard": []})
            sampled_in_partition = len(ptp["easy"]) + len(ptp["medium"]) + len(ptp["hard"])
            if sampled_in_partition == 0:
                # 无 CMCV 数据的分区：用全局平均比例估算
                easy_ratio_est = 1/3
                med_ratio_est = 1/3
                hard_ratio_est = 1/3
            else:
                easy_ratio_est = len(ptp["easy"]) / sampled_in_partition
                med_ratio_est = len(ptp["medium"]) / sampled_in_partition
                hard_ratio_est = len(ptp["hard"]) / sampled_in_partition
            ps = {"sampled_in_partition": sampled_in_partition, "actual_size": part_size,
                  "easy_ratio": round(easy_ratio_est, 3), "medium_ratio": round(med_ratio_est, 3),
                  "hard_ratio": round(hard_ratio_est, 3), "sampled_easy": 0, "sampled_medium": 0, "sampled_hard": 0}
            centroid = centroids[i]
            query_vec = pa.array(centroid, type=pa.float32())
            total_target = 0
            for ctier, pct in ratios.items():
                est_ratio = {"easy": easy_ratio_est, "medium": med_ratio_est, "hard": hard_ratio_est}[ctier]
                if pct >= 100:
                    total_target += int(part_size * est_ratio)
                else:
                    total_target += max(1, int(part_size * est_ratio * pct / 100))
            total_target = max(total_target, part_size)
            total_target = min(total_target, part_size)
            try:
                scanner = _ds.scanner(
                    nearest={"column": "embedding", "q": query_vec, "k": total_target},
                    disable_scoring_autoprojection=True, columns=["sample_id", "difficulty", "input_type"],
                )
                tbl = scanner.to_table()
                candidates = tbl.to_pylist()
                del tbl, scanner
                if i % 8 == 0:
                    gc.collect()
            except Exception as e:
                print(f"[page-cmcv-sample] {tier_name} 采样失败: {e}", file=sys.stderr)
                continue
            for ctier, pct in ratios.items():
                est_ratio = {"easy": easy_ratio_est, "medium": med_ratio_est, "hard": hard_ratio_est}[ctier]
                if pct >= 100:
                    target = int(part_size * est_ratio)
                else:
                    target = max(1, int(part_size * est_ratio * pct / 100))
                available = candidates
                if len(available) <= target:
                    chosen = available
                else:
                    chosen = random.sample(available, target)
                for c in chosen:
                    c["partition"] = tier_name
                sampled[ctier].extend(chosen)
                ps[f"sampled_{ctier}"] = len(chosen)
            partition_stats[tier_name] = ps

        result = {
            "source_id": source_id, "batch_id": batch_id,
            "strategy": "page_cmcv_sample",
            "ratios": ratios,
            "partition_stats": partition_stats,
            "total_sampled": sum(len(v) for v in sampled.values()),
            "buckets": {f"page_{t}": [s["sample_id"] for s in samples] for t, samples in sampled.items()},
        }
        cached["page_cmcv_sample"] = result
        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
        _bucket_summary_cache.invalidate(cache_path)
        _json_cache.invalidate(str(cache_path))
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/{source_id}")
async def page_ocr(source_id: str, model: str = "paddleocr", batch_id: str = "",
                   resume: bool = True):
    """对抽样页面运行 Page OCR，结果写回 bucket_samples.json。"""
    prefix_map = {"paddleocr": "paddle", "glm_ocr": "glm", "self_ocr": "self"}
    prefix = prefix_map.get(model)
    if not prefix:
        raise HTTPException(status_code=400, detail=f"未知模型：{model}")

    task_id = f"ocr_{model}_{source_id}_{batch_id}"

    def _execute_page_ocr(ctx: TaskContext):
        import tempfile, lance

        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="page_ocr",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=1, message=f"{model}: 启动中..."
        )

        source = registry.get(ctx.source_id)
        batch_dir = source.resolve_batch_dir(ctx.batch_id)
        manifests_dir = batch_dir / "artifacts"
        cache_path = manifests_dir / "bucket_samples.json"

        if not cache_path.exists():
            progress_tracker.fail_task(ctx.task_id, "bucket_samples.json 不存在，请先抽样")
            return

        cached = _read_json_cached(cache_path)
        buckets = cached.get("buckets", {})
        if not buckets:
            progress_tracker.fail_task(ctx.task_id, "抽样数据为空")
            return

        all_samples = []
        for tier, samples in buckets.items():
            for s in samples:
                sid = s if isinstance(s, str) else s.get("sample_id", "")
                if sid:
                    all_samples.append((tier, sid))

        if not all_samples:
            progress_tracker.fail_task(ctx.task_id, "无抽样样本")
            return

        progress_tracker.update_progress(ctx.task_id, current=0, total=len(all_samples), message=f"{model}: 共 {len(all_samples)} 页面，准备中...")

        ingest_dir = batch_dir / "manifests"
        ingest_path = find_stage_manifest(ingest_dir, "ingest")
        if not ingest_path:
            progress_tracker.fail_task(ctx.task_id, "ingest manifest 不存在")
            return

        ds = _open_lance(ingest_path)

        lr_path = batch_dir / "artifacts" / "layout_results.json"
        layout_map: dict[str, list] = {}
        if lr_path.exists():
            try:
                lr_data = _read_json_cached(lr_path)
                for r in lr_data.get("results", []):
                    sid = r.get("sample_id", "")
                    blocks = r.get("blocks", [])
                    if sid and blocks:
                        layout_map[sid] = [b for b in blocks if "bbox" in b]
            except Exception:
                pass

        if model == "paddleocr":
            from data_engine.ocr.paddle_ocr import PaddleOCREngine
            engine = PaddleOCREngine()
        elif model == "glm_ocr":
            from data_engine.ocr.glm_ocr import GLMOCREngine
            engine = GLMOCREngine()
        else:
            from data_engine.ocr.self_ocr import SelfOCREngine
            engine = SelfOCREngine()

        from data_engine.ocr.base import LayoutBlock

        text_col = f"{prefix}_text"
        conf_col = f"{prefix}_confidence"

        done_count = 0
        if resume:
            for tier, samples in buckets.items():
                for s in samples:
                    if isinstance(s, dict):
                        blocks = s.get("blocks", [])
                        all_done = True
                        for b in blocks:
                            if text_col not in b:
                                all_done = False
                                break
                            if b.get(conf_col, 1.0) == 0.0:
                                all_done = False
                                break
                            raw_output = b.get("raw_output", {})
                            if isinstance(raw_output, dict) and "error" in raw_output:
                                all_done = False
                                break
                        if all_done and blocks:
                            done_count += 1

        tasks = []
        skipped = 0
        for tier, samples in buckets.items():
            for idx, s in enumerate(samples):
                sid = s if isinstance(s, str) else s.get("sample_id", "")
                if not sid:
                    continue
                if isinstance(s, dict) and resume:
                    blocks = s.get("blocks", [])
                    if blocks:
                        all_done = True
                        for b in blocks:
                            if text_col not in b:
                                all_done = False
                                break
                            block_type = b.get("block_type", "text")
                            if model == "self_ocr" and block_type not in ("formula", "table"):
                                if not b.get(text_col, ""):
                                    all_done = False
                                    break
                            if b.get(conf_col, 1.0) == 0.0:
                                all_done = False
                                break
                            raw_output = b.get("raw_output", {})
                            if isinstance(raw_output, dict) and "error" in raw_output:
                                all_done = False
                                break
                        if all_done:
                            skipped += 1
                            continue
                blocks = layout_map.get(sid, [])
                if not blocks:
                    skipped += 1
                    continue
                tasks.append((tier, idx, sid, s, blocks))

        remaining = len(tasks)
        if remaining <= 0:
            progress_tracker.complete_task(ctx.task_id, f"{model} 所有 {len(all_samples)} 个页面已完成（跳过 {skipped}）")
            return

        progress_tracker.update_progress(
            ctx.task_id, current=skipped, total=len(all_samples),
            message=f"{model}: 共 {len(all_samples)} 页面（跳过 {skipped}，待处理 {remaining}）",
        )

        tmp_dir = tempfile.mkdtemp(prefix="page_ocr_")
        processed = 0
        errors = 0
        save_interval = int(get_config("ocr", "save_interval", default=10))
        max_workers = int(get_config("ocr", "page_concurrency", default=4))

        # 批量预加载图片（避免逐页查询 Lance）
        image_cache: dict[str, bytes] = {}
        BATCH_SIZE = 100
        pending_sids = [t[2] for t in tasks]  # sid 列表
        progress_tracker.update_progress(
            ctx.task_id, current=skipped, total=len(all_samples),
            message=f"{model}: 预加载图片中...",
        )
        for i in range(0, len(pending_sids), BATCH_SIZE):
            ctx.check_stop()
            chunk = pending_sids[i:i + BATCH_SIZE]
            sids_str = ",".join(f"'{sid}'" for sid in chunk)
            try:
                scanner = ds.scanner(
                    columns=["sample_id", "image_data"],
                    filter=f"sample_id IN ({sids_str})",
                )
                for batch in scanner.to_batches():
                    for row in batch.to_pylist():
                        sid = row["sample_id"]
                        img = row.get("image_data")
                        if img:
                            image_cache[sid] = img
            except Exception:
                pass

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _ocr_one(tier, idx, sid, s, blocks):
            if ctx.is_stopped():
                return None

            tmp_path = Path(tmp_dir) / f"{sid}.png"
            try:
                img_bytes = image_cache.get(sid)
                if not img_bytes:
                    return (tier, idx, sid, s, [], "Image not found")

                tmp_path.write_bytes(img_bytes)

                # 一次 batch predict 处理整页所有 block
                batch_blocks = [
                    {"block_type": b.get("block_type", "text"), "bbox": [int(c) for c in b.get("bbox", [0, 0, 9999, 9999])], "confidence": b.get("confidence", 1.0)}
                    for b in blocks
                ]
                ocr_results = engine.recognize_regions_batch(tmp_path, batch_blocks)

                block_results = []
                for bi, b in enumerate(blocks):
                    bbox = b.get("bbox", [0, 0, 9999, 9999])
                    br = {"block_idx": bi, "block_type": b.get("block_type", "text"), "bbox": [round(c, 1) for c in bbox]}
                    r = ocr_results[bi] if bi < len(ocr_results) else None
                    # 表格：paddle_text 存 HTML，formula：存 LaTeX
                    if r and r.table_structure and r.table_structure.get("html"):
                        br[text_col] = r.table_structure["html"]
                    elif r and r.formula_latex:
                        br[text_col] = r.formula_latex
                    else:
                        br[text_col] = (r.text_content or "") if r else ""
                    br[conf_col] = r.confidence if r else 0.0
                    block_results.append(br)

                return (tier, idx, sid, s, block_results, None)
            except Exception as e:
                import traceback
                logger.error(f"[{task_id}] _ocr_one failed sid={sid}: {e}\n{traceback.format_exc()}")
                return (tier, idx, sid, s, [], e)
            finally:
                tmp_path.unlink(missing_ok=True)

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_ocr_one, *t): t for t in tasks}
                for future in as_completed(futures):
                    ctx.check_stop()
                    result = future.result()
                    if not result:
                        continue
                    tier, idx, sid, s, block_results, err = result
                    if err:
                        errors += 1
                        if errors <= 5:
                            logger.warning(f"[{task_id}] result error sid={sid}: {err}")
                    else:
                        if isinstance(s, str):
                            buckets[tier][idx] = {"sample_id": s, "blocks": block_results}
                        else:
                            existing_blocks = s.get("blocks", [])
                            if existing_blocks:
                                for br in block_results:
                                    bi = br.get("block_idx", 0)
                                    if bi < len(existing_blocks):
                                        existing_blocks[bi].update(br)
                                    else:
                                        existing_blocks.append(br)
                            else:
                                s["blocks"] = block_results

                    done_count += 1
                    processed += 1
                    err_msg = f"（{errors} 错误）" if errors else ""
                    progress_tracker.update_progress(ctx.task_id, current=skipped + processed, total=len(all_samples), message=f"{model}: {processed}/{remaining} {err_msg}")
                    if processed % save_interval == 0:
                        cached["buckets"] = buckets
                        cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
                        _bucket_summary_cache.invalidate(cache_path)
                        _json_cache.invalidate(str(cache_path))

        finally:
            cached["buckets"] = buckets
            cached[f"{prefix}_ocr_at"] = datetime.now().isoformat(timespec='seconds')
            cache_path.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
            _json_cache.invalidate(str(cache_path))
            _bucket_summary_cache.invalidate(cache_path)
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        progress_tracker.complete_task(ctx.task_id, f"{model} 完成: {processed} 页面（跳过 {skipped}{f'，{errors} 错误' if errors else ''}）")

    return task_manager.start(
        task_id=task_id, task_type="page_ocr",
        source_id=source_id, batch_id=batch_id,
        target=_execute_page_ocr,
    )


@app.get("/api/page-image/{source_id}/{batch_id}/{sample_id}")
async def get_page_image(source_id: str, batch_id: str, sample_id: str):
    """返回指定 sample 的页面图片（base64）。"""
    import base64
    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        ingest_path = find_stage_manifest(batch_dir / "manifests", "ingest")
        if not ingest_path:
            raise HTTPException(status_code=404, detail="ingest not found")
        ds = _open_lance(ingest_path)
        tbl = ds.to_table(columns=["sample_id", "image_data"], filter=f"sample_id = '{sample_id}'")
        if len(tbl) == 0:
            raise HTTPException(status_code=404, detail=f"sample {sample_id} not found")
        img = tbl.column("image_data")[0].as_py()
        if not img:
            raise HTTPException(status_code=404, detail="no image data")
        return {"sample_id": sample_id, "image": base64.b64encode(img).decode(), "mime": "image/png"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/element-ocr-status/{source_id}/{batch_id}")
async def get_element_ocr_status(source_id: str, batch_id: str):
    """查询各 OCR 模型在 Element block 上的完成状态。"""
    models_prefix = {"paddleocr": "paddle", "glm_ocr": "glm", "self_ocr": "self"}
    result = {m: {"total": 0, "done": 0} for m in models_prefix}

    try:
        source = registry.get(source_id)
        batch_dir = source.resolve_batch_dir(batch_id)
        manifests_dir = batch_dir / "manifests"

        for cat in ("text", "formula", "table"):
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                if "image_data" not in ds.schema.names:
                    continue
                total = ds.count_rows()
                for model, prefix in models_prefix.items():
                    text_col = f"{prefix}_text"
                    conf_col = f"{prefix}_confidence"
                    if text_col not in ds.schema.names:
                        continue
                    tbl = ds.to_table(
                        columns=["sample_id"],
                        filter=f"{text_col} IS NOT NULL AND {text_col} != '' AND {conf_col} > 0"
                    )
                    result[model]["total"] += total
                    result[model]["done"] += len(tbl)
            except Exception:
                pass
    except Exception:
        pass

    return result


@app.post("/api/element-ocr/{source_id}/{batch_id}")
async def element_ocr(source_id: str, batch_id: str, request: Request,
                      model: str = "paddleocr", force: bool = False, categories: str = "text,formula,table"):
    """对 Element 抽样 block 运行 OCR，结果写回 text/formula/table.lance。
    force=True 时清空该模型已有结果后全部重跑。
    categories: 逗号分隔的类别列表，如 "text" 或 "text,formula,table"。
    """
    prefix_map = {"paddleocr": "paddle", "glm_ocr": "glm", "self_ocr": "self"}
    prefix = prefix_map.get(model)
    if not prefix:
        raise HTTPException(status_code=400, detail=f"未知模型: {model}")

    task_id = f"el_ocr_{model}_{source_id}_{batch_id}"

    def _execute_element_ocr(ctx: TaskContext):
        import tempfile, shutil, lance

        progress_tracker.start_task(
            task_id=ctx.task_id, task_type="el_ocr",
            source_id=ctx.source_id, batch_id=ctx.batch_id,
            total=1, message=f"{model}: 启动中..."
        )

        source = registry.get(ctx.source_id)
        batch_dir = source.resolve_batch_dir(ctx.batch_id)
        manifests_dir = batch_dir / "manifests"

        selected_cats = [c.strip() for c in categories.split(",") if c.strip()]
        cat_paths: list[tuple[str, str]] = []
        total_blocks = 0
        for cat in selected_cats:
            lp = manifests_dir / f"{cat}.lance"
            if not lp.exists():
                continue
            try:
                ds = _open_lance(lp)
                if "image_data" not in ds.schema.names:
                    continue
                total_blocks += ds.count_rows()
                cat_paths.append((cat, str(lp)))
            except Exception:
                pass

        if not cat_paths or total_blocks == 0:
            progress_tracker.fail_task(ctx.task_id, "三个 category lance 中无可用 block")
            return

        progress_tracker.update_progress(ctx.task_id, current=0, message=f"{model}: 扫描已完成，共 {total_blocks} blocks...")

        if model == "paddleocr":
            from data_engine.ocr.paddle_ocr import PaddleOCREngine
            engine = PaddleOCREngine()
        elif model == "glm_ocr":
            from data_engine.ocr.glm_ocr import GLMOCREngine
            engine = GLMOCREngine()
        else:
            from data_engine.ocr.self_ocr import SelfOCREngine
            engine = SelfOCREngine()

        text_col = f"{prefix}_text"
        conf_col = f"{prefix}_confidence"

        if force:
            progress_tracker.update_progress(ctx.task_id, current=0, message=f"{model}: 强制重跑，清空已有结果...")
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
                except Exception:
                    pass

        done_keys: set[tuple[str, int]] = set()
        for cat, lp_str in cat_paths:
            try:
                ds = lance.dataset(lp_str)
                if text_col in ds.schema.names:
                    tbl = ds.to_table(
                        columns=["sample_id", "block_idx"],
                        filter=f"{text_col} IS NOT NULL AND {text_col} != '' AND {conf_col} > 0"
                    )
                    sid_col = tbl.column("sample_id")
                    bidx_col = tbl.column("block_idx")
                    done_keys.update(
                        (sid_col[i].as_py(), bidx_col[i].as_py())
                        for i in range(len(tbl))
                    )
            except Exception:
                pass

        remaining_blocks = total_blocks - len(done_keys)
        if remaining_blocks <= 0:
            progress_tracker.complete_task(ctx.task_id, f"{model} 所有 block 已完成（{len(done_keys)} 个），无需重跑")
            return

        progress_tracker.update_progress(
            ctx.task_id, current=len(done_keys), total=total_blocks,
            message=f"{model}: {total_blocks} 个 block（跳过 {len(done_keys)} 已完成，待处理 {remaining_blocks}）",
        )

        from data_engine.ocr.base import LayoutBlock
        tmp_dir = tempfile.mkdtemp(prefix="el_ocr_")

        done = 0
        errors = 0
        skipped = 0
        save_interval = int(get_config("ocr", "save_interval", default=20))
        pending_rows: dict[str, list[dict]] = {"text": [], "formula": [], "table": []}

        for cat, lp_str in cat_paths:
            ctx.check_stop()
            try:
                ds = lance.dataset(lp_str)
            except Exception:
                continue

            progress_tracker.update_progress(
                ctx.task_id, current=done,
                message=f"{model}: 扫描 {cat} 数据..."
            )

            STREAM_BATCH = 100
            scan_done = 0
            for batch in ds.to_batches(batch_size=STREAM_BATCH):
                ctx.check_stop()
                rows = batch.to_pylist()
                for row in rows:
                    ctx.check_stop()

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
                        continue

                    tmp_path = Path(tmp_dir) / f"{cat}_{sid}_{bidx}.png"
                    if isinstance(img_bytes, bytes):
                        tmp_path.write_bytes(img_bytes)
                    elif isinstance(img_bytes, str):
                        tmp_path.write_bytes(img_bytes.encode("latin-1"))
                    else:
                        tmp_path.write_bytes(bytes(img_bytes))

                    _w, _h = 0, 0
                    try:
                        import struct
                        _raw = img_bytes if isinstance(img_bytes, bytes) else bytes(img_bytes)
                        if _raw[:8] == b'\x89PNG\r\n\x1a\n' and len(_raw) > 24:
                            _w, _h = struct.unpack('>II', _raw[16:24])
                        elif _raw[:2] == b'\xff\xd8':
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
                        _w, _h = 9999, 9999
                    region = LayoutBlock(
                        block_type=row.get("block_type", cat),
                        bbox=[0, 0, _w, _h],
                        confidence=row.get("layout_confidence", 1.0),
                    )
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
                    finally:
                        tmp_path.unlink(missing_ok=True)

                    done += 1
                    scan_done += 1
                    pct = round(done / total_blocks * 100) if total_blocks else 0
                    err_msg = f"（{errors} 错误）" if errors else ""
                    progress_tracker.update_progress(
                        ctx.task_id, current=done,
                        message=f"{model}: {cat} {done}/{total_blocks} ({pct}%){err_msg}"
                    )

                    if done % save_interval == 0:
                        _flush_el_ocr_rows(manifests_dir, pending_rows, text_col, conf_col, prefix)
                        pending_rows = {"text": [], "formula": [], "table": []}

        _flush_el_ocr_rows(manifests_dir, pending_rows, text_col, conf_col, prefix)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        progress_tracker.complete_task(ctx.task_id, f"{model} 完成: {done} block（跳过 {len(done_keys)} 已完成{f'，{skipped} 无图' if skipped else ''}{f'，{errors} 错误' if errors else ''}）")

        try:
            from data_engine.ocr.cmcv import CMCVEngine
            cmcv_engine = CMCVEngine(use_visual_cdm=True)
            json_keys = ("paddle_table", "glm_table", "self_table",
                         "paddle_formula", "glm_formula", "self_formula")
            for cat in ("text", "formula", "table"):
                lp = manifests_dir / f"{cat}.lance"
                if not lp.exists():
                    continue
                try:
                    with _lance_write_lock:
                        ds = _open_lance(lp)
                        all_cols_set = set(ds.schema.names)
                        if "consistency_pattern" not in all_cols_set:
                            continue
                        ocr_filter = ocr_complete_filter(all_cols_set)
                        read_cols = ["sample_id", "block_idx", "block_type", "bbox_json", "layout_confidence"]
                        for _pfx in ("paddle", "glm", "self"):
                            for suffix in ("_text", "_confidence", "_table", "_formula"):
                                col = f"{_pfx}{suffix}"
                                if col in all_cols_set:
                                    read_cols.append(col)
                        cols = [c for c in read_cols if c in all_cols_set]
                        auto_filter = None  # force 模式下不过滤
                        if ocr_filter:
                            auto_filter = f"({ocr_filter})"
                        batches = list(ds.to_batches(columns=cols, filter=auto_filter))
                    rows = []
                    for batch in batches:
                        for i in range(len(batch)):
                            row = {c: batch.column(c)[i].as_py() for c in cols}
                            for key in json_keys:
                                val = row.get(key)
                                if isinstance(val, str) and val:
                                    try:
                                        row[key] = json.loads(val)
                                    except (json.JSONDecodeError, TypeError):
                                        pass
                            rows.append(row)
                    updated_rows, _ = cmcv_engine.process_element_batch(rows)
                    update_map = {}
                    for r in updated_rows:
                        if r.get("consistency_pattern"):
                            update_map.setdefault("consistency_pattern", {})[(r["sample_id"], r["block_idx"])] = r["consistency_pattern"]
                        if r.get("block_diff_json"):
                            update_map.setdefault("block_diff_json", {})[(r["sample_id"], r["block_idx"])] = r["block_diff_json"]
                    if update_map:
                        with _lance_write_lock:
                            ds = _open_lance(lp)
                            # 流式读取，逐批更新
                            read_cols = ["sample_id", "block_idx", "consistency_pattern", "block_diff_json"]
                            available = [c for c in read_cols if c in ds.schema.names]
                            scanner = ds.scanner(columns=available)
                            update_batches = []
                            pat_map = update_map.get("consistency_pattern", {})
                            diff_map = update_map.get("block_diff_json", {})
                            for batch in scanner.to_batches():
                                sids = batch.column("sample_id").to_pylist()
                                bidxs = batch.column("block_idx").to_pylist()
                                patterns = batch.column("consistency_pattern").to_pylist() if "consistency_pattern" in batch.column_names else [None] * len(batch)
                                diffs = batch.column("block_diff_json").to_pylist() if "block_diff_json" in batch.column_names else [None] * len(batch)
                                new_p = [pat_map.get((sids[i], bidxs[i]), patterns[i]) for i in range(len(batch))]
                                new_d = [diff_map.get((sids[i], bidxs[i]), diffs[i]) for i in range(len(batch))]
                                update_batches.append(pa.table({
                                    "sample_id": pa.array(sids, type=pa.large_string()),
                                    "block_idx": pa.array(bidxs, type=pa.int32()),
                                    "consistency_pattern": pa.array(new_p, type=pa.large_string()),
                                    "block_diff_json": pa.array(new_d, type=pa.large_string()),
                                }))
                            _lance_cache.invalidate(str(lp))
                            if update_batches:
                                update_table = pa.concat_tables(update_batches)
                                safe_merge(ds, update_table, ["sample_id", "block_idx"], context=f"el-ocr-cmcv/{cat}")
                except Exception:
                    pass
        except Exception:
            pass

    return task_manager.start(
        task_id=task_id, task_type="el_ocr",
        source_id=source_id, batch_id=batch_id,
        target=_execute_element_ocr,
    )


_CAT_TEXT_COL = {"text": "_text", "formula": "_formula", "table": "_table"}


def _flush_el_ocr_rows(manifests_dir: Path, pending_rows: dict, text_col: str, conf_col: str, prefix: str):
    """将 Element OCR 结果 merge_insert 到对应的 category lance。
    自动检测磁盘 schema，只写入兼容的列。
    重试和去重由 safe_merge 统一处理。
    """
    for cat, rows in pending_rows.items():
        if not rows:
            continue
        lp = manifests_dir / f"{cat}.lance"
        if not lp.exists():
            continue
        try:
            cat_text_col = f"{prefix}{_CAT_TEXT_COL.get(cat, '_text')}"

            with _lance_write_lock:
                ds = lance.dataset(str(lp))
            disk_names = {f.name for f in ds.schema}

            arrays = {}
            for r in rows:
                mapped = {
                    "sample_id": r["sample_id"],
                    "block_idx": r["block_idx"],
                }
                mapped[cat_text_col] = r.get(text_col, "")
                mapped[conf_col] = r.get(conf_col, 0.0)
                for extra_key in (f"{prefix}_table", f"{prefix}_formula"):
                    if extra_key in r and extra_key in disk_names:
                        mapped[extra_key] = r[extra_key]

                for k, v in mapped.items():
                    if k not in arrays:
                        arrays[k] = []
                    arrays[k].append(v)

            arrays = {k: v for k, v in arrays.items() if k in disk_names}
            if "sample_id" not in arrays or "block_idx" not in arrays:
                continue

            pa_arrays = {}
            for col, vals in arrays.items():
                field = ds.schema.field(col)
                if field.type == pa.large_string():
                    pa_arrays[col] = pc.fill_null(pa.array(vals), "").cast(pa.large_string())
                elif field.type == pa.float32():
                    pa_arrays[col] = pc.fill_null(pa.array(vals, type=pa.float64()), 0.0).cast(pa.float32())
                elif field.type == pa.int32():
                    pa_arrays[col] = pc.fill_null(pa.array(vals, type=pa.int64()), 0).cast(pa.int32())
                else:
                    pa_arrays[col] = pa.array(vals)
            table = pa.table(pa_arrays)

            with _lance_write_lock:
                ds = lance.dataset(str(lp))
                safe_merge(ds, table, ["sample_id", "block_idx"],
                           when_not_matched=set(arrays.keys()) >= disk_names,
                           context=f"el-ocr/{cat}")
            _lance_cache.invalidate(str(lp))
        except Exception as e:
            print(f"[el-ocr] flush {cat} 失败: {e}", file=sys.stderr)


if __name__ == "__main__":
    import os
    import signal
    import uvicorn

    # Docker socket: 宿主机 daemon 只支持 API v1.43，镜像 CLI 是 29.1.3 (v1.52)
    os.environ.setdefault("DOCKER_API_VERSION", "1.43")
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