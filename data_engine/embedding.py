import os
# 限制所有底层 C/C++ 库的并发线程为 1，避免多层线程嵌套导致 malloc 崩溃
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import gc
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
from modelscope import AutoModel, AutoProcessor

from data_engine.config import get_config


def _get_progress_interval() -> int:
    return get_config("embedding", "progress_update_interval", default=100)


def _coerce_image_bytes(image_data: Any) -> bytes | None:
    """将 Lance 读出的 image_data 统一转为 bytes，无法转换则返回 None。"""
    if isinstance(image_data, (bytes, bytearray, memoryview)):
        return bytes(image_data)
    if isinstance(image_data, str):
        try:
            return image_data.encode("latin-1")
        except Exception:
            return None
    return None


class CLIPEmbeddingExtractor:
    """SigLIP2图像embedding提取器"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_config("embedding", "model_name", default="google/siglip2-base-patch16-224")
        gpu_id = get_config("embedding", "gpu_id", default=0)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """加载SigLIP2模型"""
        local_path = get_config("embedding", "local_path", default=None)
        if local_path:
            local_path = str(Path(local_path).expanduser())
        model_source = local_path if local_path else self.model_name
        print(f"加载模型: {model_source}", file=sys.stderr)
        self.model = AutoModel.from_pretrained(
            model_source, device_map=self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_source)
        self.device = self.model.device
        print(f"模型已加载到: {self.device}", file=sys.stderr)

    def extract_embedding(self, image_path: Path) -> list[float]:
        """从文件路径提取图像embedding"""
        image = Image.open(image_path).convert("RGB")
        try:
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.get_image_features(**inputs)
            embedding = outputs
            if hasattr(outputs, "pooler_output"):
                embedding = outputs.pooler_output
            result = embedding.cpu().numpy().flatten().tolist()
            del inputs, outputs, embedding
            return result
        finally:
            image.close()

    def extract_embedding_from_bytes(self, image_bytes: bytes) -> list[float]:
        """从二进制数据提取图像embedding"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        try:
            inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model.get_image_features(**inputs)
            embedding = outputs
            if hasattr(outputs, "pooler_output"):
                embedding = outputs.pooler_output
            result = embedding.cpu().numpy().flatten().tolist()
            del inputs, outputs, embedding
            return result
        finally:
            image.close()

    def extract_embeddings_batch(self, image_paths: list[Path]) -> dict[Path, list[float]]:
        """批量提取图像embedding"""
        embeddings = {}
        for image_path in image_paths:
            try:
                embeddings[image_path] = self.extract_embedding(image_path)
            except Exception as e:
                print(f"提取embedding失败 {image_path}: {e}", file=sys.stderr)
        return embeddings

    def extract_embeddings_from_bytes_batch(self, image_bytes_list: list[bytes], batch_size: int = 32) -> list[list[float] | None]:
        """批量从二进制数据提取图像embedding（GPU 并行推理，比单张快 10-30x）

        Args:
            image_bytes_list: 图片 bytes 列表
            batch_size: GPU 推理批大小（默认 32）

        Returns:
            embedding 列表，失败的为 None
        """
        if not image_bytes_list:
            return []

        results: list[list[float] | None] = [None] * len(image_bytes_list)

        for start in range(0, len(image_bytes_list), batch_size):
            end = min(start + batch_size, len(image_bytes_list))
            images = []
            valid_indices = []

            for i in range(start, end):
                try:
                    img = Image.open(io.BytesIO(image_bytes_list[i])).convert("RGB")
                    images.append(img)
                    valid_indices.append(i)
                except Exception:
                    pass

            if not images:
                continue

            try:
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                with torch.inference_mode():
                    outputs = self.model.get_image_features(**inputs)
                # 处理 BaseModelOutputWithPooling 对象
                if hasattr(outputs, "pooler_output"):
                    outputs = outputs.pooler_output
                embeddings = outputs.cpu().numpy()

                for j, idx in enumerate(valid_indices):
                    results[idx] = embeddings[j].flatten().tolist()

                del inputs, outputs, embeddings
            except Exception as e:
                print(f"[Embedding] batch inference failed: {e}", file=sys.stderr)
            finally:
                for img in images:
                    img.close()

        gc.collect()
        return results


# ─── HTTP 客户端：调用 embedding 服务 ─────────────────────────────────────────

def extract_embeddings_via_http(
    image_bytes_list: list[bytes],
    batch_size: int = 32,
    timeout: float = 300,
) -> list[list[float] | None]:
    """通过 HTTP 调用独立 embedding 服务提取 embedding。

    这是推荐的方式：embedding 服务是独立进程，有自己的 CUDA 上下文，
    不受 web_app 主进程的 CUDA 状态影响，支持 batch_size > 1。

    Args:
        image_bytes_list: 图片 bytes 列表
        batch_size: 批量推理批大小
        timeout: HTTP 请求超时时间（秒）

    Returns:
        embedding 列表，失败的为 None
    """
    import requests
    import base64

    if not image_bytes_list:
        return []

    host = get_config("embedding", "server", "host", default="127.0.0.1")
    port = get_config("embedding", "server", "port", default=8090)
    url = f"http://{host}:{port}/embed"

    # 编码图片为 base64
    images_b64 = [base64.b64encode(img).decode("utf-8") for img in image_bytes_list]

    payload = {"images": images_b64, "batch_size": batch_size}

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]
    except requests.exceptions.ConnectionError:
        print(f"[Embedding] 无法连接 embedding 服务 {url}", file=sys.stderr)
        return [None] * len(image_bytes_list)
    except requests.exceptions.Timeout:
        print(f"[Embedding] embedding 服务请求超时 ({timeout}s)", file=sys.stderr)
        return [None] * len(image_bytes_list)
    except Exception as e:
        print(f"[Embedding] HTTP 调用失败: {e}", file=sys.stderr)
        return [None] * len(image_bytes_list)


def _check_embedding_server() -> bool:
    """检查 embedding 服务是否可用。"""
    import requests
    host = get_config("embedding", "server", "host", default="127.0.0.1")
    port = get_config("embedding", "server", "port", default=8090)
    try:
        resp = requests.get(f"http://{host}:{port}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "ready"
    except Exception:
        return False


# ─── 常驻子进程 Embedding Worker（避免 cuBLAS LT 在线程中崩溃） ──────────────

_WORKER_SCRIPT = '''\
import sys, os, json, pickle, signal
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_id}"

import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, "{project_root}")
from data_engine.embedding import CLIPEmbeddingExtractor

extractor = CLIPEmbeddingExtractor()
print(json.dumps({{"status": "ready"}}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        if req.get("action") == "shutdown":
            break
        img_file = req.get("img_file")
        batch_size = req.get("batch_size", 32)
        result_file = req.get("result_file")
        if not img_file or not result_file:
            print(json.dumps({{"status": "error", "message": "missing img_file or result_file"}}), flush=True)
            continue
        with open(img_file, "rb") as f:
            image_bytes_list = pickle.load(f)
        results = extractor.extract_embeddings_from_bytes_batch(image_bytes_list, batch_size=batch_size)
        with open(result_file, "wb") as f:
            pickle.dump(results, f)
        print(json.dumps({{"status": "ok", "count": len(results)}}), flush=True)
    except Exception as e:
        print(json.dumps({{"status": "error", "message": str(e)}}), flush=True)

del extractor
'''


class EmbeddingWorker:
    """常驻子进程 embedding worker，通过 stdin/stdout JSON 协议通信。

    用法:
        worker = EmbeddingWorker(gpu_id=1)
        worker.start()
        results = worker.extract(image_bytes_list, batch_size=32)
        worker.shutdown()
    """

    def __init__(self, gpu_id: int = 1, batch_size: int = 32):
        self.gpu_id = gpu_id
        self.default_batch_size = batch_size
        self._process = None
        self._script_path = None
        self._lock = None

    def start(self):
        """启动 worker 子进程。"""
        import subprocess
        import tempfile
        import threading

        if self._lock is None:
            self._lock = threading.Lock()

        project_root = str(Path(__file__).parent.parent)
        script_content = _WORKER_SCRIPT.format(
            gpu_id=self.gpu_id,
            project_root=project_root,
        )

        fd, self._script_path = tempfile.mkstemp(suffix="_embed_worker.py")
        with os.fdopen(fd, "w") as f:
            f.write(script_content)

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "TORCH_NUM_THREADS": "1"}

        self._process = subprocess.Popen(
            [sys.executable, self._script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )

        # 等待 "ready" 信号
        import select
        ready = False
        for _ in range(60):
            if select.select([self._process.stdout], [], [], 1.0)[0]:
                line = self._process.stdout.readline()
                if line:
                    try:
                        msg = json.loads(line.strip())
                        if msg.get("status") == "ready":
                            ready = True
                            break
                    except json.JSONDecodeError:
                        pass

        if ready:
            print(f"[EmbeddingWorker] 启动成功 (pid={self._process.pid}, gpu={self.gpu_id})", file=sys.stderr)
        else:
            stderr_out = self._process.stderr.read(500).decode(errors="replace") if self._process.stderr else ""
            print(f"[EmbeddingWorker] 启动失败: {stderr_out[-300:]}", file=sys.stderr)
            self._process.terminate()

    def extract(self, image_bytes_list: list[bytes], batch_size: int | None = None, timeout: float = 300) -> list[list[float] | None]:
        """发送 embedding 请求并等待结果。"""
        import tempfile
        import pickle

        if not image_bytes_list:
            return []
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("EmbeddingWorker 未运行")

        with self._lock:
            # 将图片数据写入临时文件（避免通过 stdin 传输大数据）
            img_fd, img_file = tempfile.mkstemp(suffix=".pkl")
            os.close(img_fd)
            result_fd, result_file = tempfile.mkstemp(suffix=".pkl")
            os.close(result_fd)

            try:
                with open(img_file, "wb") as f:
                    pickle.dump(image_bytes_list, f)

                req = json.dumps({
                    "action": "embed",
                    "img_file": img_file,
                    "batch_size": batch_size or self.default_batch_size,
                    "result_file": result_file,
                })
                self._process.stdin.write((req + "\n").encode())
                self._process.stdin.flush()

                # 等待结果文件
                import time
                deadline = time.time() + timeout
                while time.time() < deadline:
                    if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                        with open(result_file, "rb") as f:
                            results = pickle.load(f)
                        return results
                    time.sleep(0.1)

                print(f"[EmbeddingWorker] 超时 ({timeout}s)", file=sys.stderr)
                return [None] * len(image_bytes_list)
            finally:
                for p in (img_file, result_file):
                    try:
                        os.unlink(p)
                    except Exception:
                        pass

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def shutdown(self):
        """关闭 worker 子进程。"""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write(b'{"action":"shutdown"}\n')
                self._process.stdin.flush()
                self._process.wait(timeout=10)
            except Exception:
                pass
            if self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=5)
        if self._script_path:
            try:
                os.unlink(self._script_path)
            except Exception:
                pass
        print("[EmbeddingWorker] 已关闭", file=sys.stderr)


def extract_embeddings_in_subprocess(
    image_bytes_list: list[bytes],
    batch_size: int = 32,
    model_name: str | None = None,
    local_path: str | None = None,
    gpu_id: int = 0,
    timeout: float = 600,
) -> list[list[float] | None]:
    """兼容旧接口。直接在调用线程中提取 embedding。

    注：cuBLAS LT 在线程中 batch>1 会崩溃，因此 batch_size 固定为 1。
    """
    if not image_bytes_list:
        return []

    try:
        from data_engine.embedding import CLIPEmbeddingExtractor
        extractor = CLIPEmbeddingExtractor()
        results = extractor.extract_embeddings_from_bytes_batch(image_bytes_list, batch_size=1)
        del extractor
        return results
    except Exception as e:
        print(f"[Embedding] failed: {e}", file=sys.stderr)
        return [None] * len(image_bytes_list)


def _ensure_pure_list(embedding: Any) -> list[float] | None:
    """确保 embedding 是纯 Python list，切断 PyTorch/numpy 底层引用。"""
    if embedding is None:
        return None
    if isinstance(embedding, list):
        return embedding
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def extract_embeddings_for_records(
    records: list[dict[str, Any]],
    batch_dir: Path,
    extractor: CLIPEmbeddingExtractor | None = None,
    task_id: str | None = None,
    offset: int = 0
) -> list[dict[str, Any]]:
    """为样本记录提取embedding（已优化内存管理）"""
    if extractor is None:
        extractor = CLIPEmbeddingExtractor()

    progress_tracker = None
    if task_id:
        try:
            from data_engine.progress_tracker import progress_tracker
        except ImportError:
            pass

    interval = _get_progress_interval()
    gc_interval = get_config("embedding", "gc_interval", default=1000)
    updated_records = []

    for idx, record in enumerate(records):
        if progress_tracker and task_id and progress_tracker.is_stopped(task_id):
            print(f"[Embedding] 收到停止信号，中断处理", file=sys.stderr)
            progress_tracker.stop_task(task_id, f"用户停止，已处理 {idx}/{len(records)} 个样本")
            break

        try:
            image_data = record.get("image_data")
            embedding = None

            if image_data:
                image_bytes = _coerce_image_bytes(image_data)
                if image_bytes is None:
                    print(f"[Embedding] record {record.get('sample_id')} image_data 类型异常: {type(image_data).__name__}, 跳过", file=sys.stderr)
                else:
                    embedding = extractor.extract_embedding_from_bytes(image_bytes)
            else:
                page_image = record.get("page_image")
                if not page_image or not isinstance(page_image, str):
                    print(f"[Embedding] record {record.get('sample_id')} page_image 异常: {page_image!r}, 跳过", file=sys.stderr)
                else:
                    image_path = batch_dir / page_image
                    if not image_path.exists():
                        print(f"图像文件不存在: {image_path}", file=sys.stderr)
                    else:
                        embedding = extractor.extract_embedding(image_path)

            # 确保 embedding 是纯 Python list，切断底层 C/显存 引用
            record["embedding"] = _ensure_pure_list(embedding)
            updated_records.append(record)

            if progress_tracker and task_id:
                if (idx + 1) % interval == 0 or idx == len(records) - 1:
                    progress_tracker.update_progress(
                        task_id=task_id,
                        current=offset + idx + 1,
                        message=f"已处理 {offset + idx + 1} 个样本"
                    )

        except Exception as e:
            print(f"处理记录embedding失败 {record.get('sample_id')}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            record["embedding"] = None
            updated_records.append(record)

            if progress_tracker and task_id:
                if (idx + 1) % interval == 0 or idx == len(records) - 1:
                    progress_tracker.update_progress(
                        task_id=task_id,
                        current=offset + idx + 1,
                        message=f"已处理 {offset + idx + 1} 个样本（含失败）"
                    )

        # 降频 GC：每 gc_interval 条做一次轻量回收，不再调用 empty_cache
        if (idx + 1) % gc_interval == 0:
            gc.collect()

    # 批次结束做一次终极清理
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return updated_records
