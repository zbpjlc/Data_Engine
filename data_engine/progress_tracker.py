from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"


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
                    cls._instance._file_path = Path("progress_state.json")
                    cls._instance._file_lock = threading.Lock()
        return cls._instance
    
    def start_task(self, task_id: str, task_type: str, source_id: str, batch_id: str, total: int = 0, message: str = "") -> None:
        """开始一个新任务"""
        with self._lock:
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
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.end_time = time.time()
                task.error_message = error_message
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
        """保存状态到文件"""
        try:
            state = {
                "tasks": {
                    task_id: {
                        **asdict(task),
                        "status": task.status.value  # 保存枚举值而不是枚举对象
                    }
                    for task_id, task in self.tasks.items()
                }
            }
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
                
                # 清空当前任务
                self.tasks.clear()
                
                for task_id, task_data in state.get("tasks", {}).items():
                    # 转换状态枚举
                    if "status" in task_data:
                        status_value = task_data["status"]
                        if isinstance(status_value, str):
                            # 如果是字符串，转换为枚举
                            try:
                                task_data["status"] = TaskStatus(status_value)
                            except ValueError:
                                # 如果枚举值无效，跳过这个任务
                                print(f"Invalid status value '{status_value}' for task {task_id}, skipping")
                                continue
                        else:
                            # 如果已经是枚举，直接使用
                            task_data["status"] = status_value
                    
                    try:
                        self.tasks[task_id] = TaskProgress(**task_data)
                        print(f"Loaded task: {task_id}, status: {self.tasks[task_id].status}")
                    except Exception as e:
                        print(f"Error creating task {task_id}: {e}")
                        continue
        except Exception as e:
            print(f"Error loading state: {e}")
            pass  # 忽略加载错误


# 全局实例
progress_tracker = ProgressTracker()
# 加载已有状态
progress_tracker._load_state()
