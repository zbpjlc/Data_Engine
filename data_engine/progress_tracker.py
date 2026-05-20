from __future__ import annotations
import os
import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class TaskProgress:
    task_id: str
    task_type: str  # ingest, page_sample, element_sample, etc.
    source_id: str
    batch_id: str
    status: TaskStatus
    current: int = 0
    total: int = 0
    message: str = ""
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def progress_percentage(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        if not self.start_time:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class ProgressTracker:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.tasks: Dict[str, TaskProgress] = {}
                    cls._instance._stop_flags: set = set()
                    cls._instance._file_path = Path("progress_state.json")
                    cls._instance._file_lock = threading.Lock()
        return cls._instance
    
    def start_task(self, task_id: str, task_type: str, source_id: str, batch_id: str, total: int = 0, message: str = "") -> None:
        """开始一个新任务"""
        with self._lock:
            self._stop_flags.discard(task_id)
            self.tasks[task_id] = TaskProgress(
                task_id=task_id,
                task_type=task_type,
                source_id=source_id,
                batch_id=batch_id,
                status=TaskStatus.RUNNING,
                total=total,
                message=message,
                start_time=time.time()
            )
            self._save_state()
    
    def update_progress(self, task_id: str, current: int, message: str = "", total: int = None) -> None:
        """更新任务进度"""
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.current = current
                if total is not None:
                    task.total = total
                if message:
                    task.message = message
                self._save_state()
    
    def complete_task(self, task_id: str, message: str = "") -> None:
        """完成任务"""
        with self._lock:
            self._stop_flags.discard(task_id)
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.current = task.total
                task.end_time = time.time()
                if message:
                    task.message = message
                self._save_state()
    
    def remove_task(self, task_id: str) -> None:
        """移除任务"""
        with self._lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                self._save_state()
    
    def fail_task(self, task_id: str, error_message: str = "") -> None:
        """任务失败"""
        with self._lock:
            self._stop_flags.discard(task_id)
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.end_time = time.time()
                task.error_message = error_message
                self._save_state()
    
    def request_stop(self, task_id: str) -> None:
        """请求停止任务"""
        with self._lock:
            self._stop_flags.add(task_id)
    
    def is_stopped(self, task_id: str) -> bool:
        """检查任务是否被请求停止"""
        with self._lock:
            return task_id in self._stop_flags
    
    def stop_task(self, task_id: str, message: str = "用户手动停止") -> None:
        """停止任务"""
        with self._lock:
            self._stop_flags.discard(task_id)
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.STOPPED
                task.end_time = time.time()
                task.message = message
                self._save_state()
    
    def get_task(self, task_id: str) -> Optional[TaskProgress]:
        """获取指定任务"""
        with self._lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> Dict[str, TaskProgress]:
        """获取所有任务"""
        with self._lock:
            return self.tasks.copy()
    
    def get_active_tasks(self) -> Dict[str, TaskProgress]:
        """获取活跃任务"""
        with self._lock:
            return {tid: task for tid, task in self.tasks.items() 
                   if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]}
    
    def get_tasks_by_batch(self, source_id: str, batch_id: str) -> Dict[str, TaskProgress]:
        """获取指定批次的任务"""
        with self._lock:
            return {tid: task for tid, task in self.tasks.items() 
                   if task.source_id == source_id and task.batch_id == batch_id}
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> None:
        """清理旧任务"""
        with self._lock:
            current_time = time.time()
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.end_time and (current_time - task.end_time) > max_age_hours * 3600:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
            
            if to_remove:
                self._save_state()
    
    def _save_state(self) -> None:
        """保存状态到文件（原子写入，避免损坏）"""
        try:
            def _fmt_time(ts):
                if ts is None:
                    return None
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

            state = {
                "tasks": {
                    task_id: {
                        **asdict(task),
                        "status": task.status.value,
                        "start_time": _fmt_time(task.start_time),
                        "end_time": _fmt_time(task.end_time),
                        "elapsed_seconds": round(task.elapsed_time, 1),
                    }
                    for task_id, task in self.tasks.items()
                }
            }
            tmp_path = self._file_path.with_suffix('.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._file_path)
        except Exception:
            pass
            with open(self._file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass  # 忽略保存错误
    
    def _load_state(self) -> None:
        """从文件加载状态"""
        try:
            if self._file_path.exists():
                with open(self._file_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                self.tasks.clear()
                
                for task_id, task_data in state.get("tasks", {}).items():
                    if "status" in task_data:
                        status_value = task_data["status"]
                        if isinstance(status_value, str):
                            try:
                                task_data["status"] = TaskStatus(status_value)
                            except ValueError:
                                continue
                    
                    for ts_field in ["start_time", "end_time"]:
                        v = task_data.get(ts_field)
                        if isinstance(v, str):
                            try:
                                task_data[ts_field] = datetime.strptime(v, "%Y-%m-%d %H:%M:%S").timestamp()
                            except ValueError:
                                task_data[ts_field] = None
                    
                    task_data.pop("elapsed_seconds", None)
                    
                    try:
                        self.tasks[task_id] = TaskProgress(**task_data)
                    except Exception:
                        continue
        except Exception:
            pass


# 全局实例
progress_tracker = ProgressTracker()
# 加载已有状态
progress_tracker._load_state()
# 服务器重启后，将所有running状态改为stopped
for task_id, task in progress_tracker.tasks.items():
    if task.status == TaskStatus.RUNNING:
        task.status = TaskStatus.STOPPED
        task.message = "服务器重启，任务已停止"
progress_tracker._save_state()
