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


def extract_embeddings_for_records(
    records: list[dict[str, Any]],
    batch_dir: Path,
    extractor: CLIPEmbeddingExtractor | None = None,
    task_id: str | None = None,
    offset: int = 0
) -> list[dict[str, Any]]:
    """为样本记录提取embedding"""
    if extractor is None:
        extractor = CLIPEmbeddingExtractor()

    progress_tracker = None
    if task_id:
        try:
            from data_engine.progress_tracker import progress_tracker
        except ImportError:
            pass

    interval = _get_progress_interval()
    updated_records = []
    for idx, record in enumerate(records):
        if progress_tracker and task_id and progress_tracker.is_stopped(task_id):
            print(f"[Embedding] 收到停止信号，中断处理", file=sys.stderr)
            progress_tracker.stop_task(task_id, f"用户停止，已处理 {idx}/{len(records)} 个样本")
            break
        
        try:
            image_data = record.get("image_data")

            if image_data:
                image_bytes = _coerce_image_bytes(image_data)
                if image_bytes is None:
                    print(f"[Embedding] record {record.get('sample_id')} image_data 类型异常: {type(image_data).__name__}, 跳过", file=sys.stderr)
                    record["embedding"] = None
                    updated_records.append(record)
                else:
                    embedding = extractor.extract_embedding_from_bytes(image_bytes)
                    record["embedding"] = embedding
                    updated_records.append(record)
            else:
                page_image = record.get("page_image")
                if not page_image or not isinstance(page_image, str):
                    print(f"[Embedding] record {record.get('sample_id')} page_image 异常: {page_image!r}, 跳过", file=sys.stderr)
                    record["embedding"] = None
                    updated_records.append(record)
                else:
                    image_path = batch_dir / page_image
                    if not image_path.exists():
                        print(f"图像文件不存在: {image_path}", file=sys.stderr)
                        record["embedding"] = None
                        updated_records.append(record)
                    else:
                        embedding = extractor.extract_embedding(image_path)
                        record["embedding"] = embedding
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

        finally:
            # 更频繁的内存清理，避免大规模处理时 OOM
            if idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # 处理完成后强制清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return updated_records
