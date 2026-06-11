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
