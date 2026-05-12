from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
from modelscope import AutoModel, AutoProcessor


class CLIPEmbeddingExtractor:
    """SigLIP2图像embedding提取器"""

    def __init__(self, model_name: str = "google/siglip2-base-patch16-224"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self) -> None:
        """加载SigLIP2模型"""
        import sys
        print(f"加载模型: {self.model_name}", file=sys.stderr)
        self.model = AutoModel.from_pretrained(
            self.model_name, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.device = self.model.device
        print(f"模型已加载到: {self.device}", file=sys.stderr)

    def extract_embedding(self, image_path: Path) -> list[float]:
        """提取图像embedding"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            embedding = outputs.pooler_output
        elif hasattr(outputs, "cpu"):
            embedding = outputs
        else:
            embedding = outputs[0]
        embedding = embedding.cpu().numpy().flatten().tolist()
        return embedding

    def extract_embeddings_batch(self, image_paths: list[Path]) -> dict[Path, list[float]]:
        """批量提取图像embedding"""
        embeddings = {}
        for image_path in image_paths:
            try:
                embeddings[image_path] = self.extract_embedding(image_path)
            except Exception as e:
                print(f"提取embedding失败 {image_path}: {e}", file=__import__("sys").stderr)
        return embeddings


def extract_embeddings_for_records(
    records: list[dict[str, Any]],
    batch_dir: Path,
    extractor: CLIPEmbeddingExtractor | None = None,
    task_id: str | None = None
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

    updated_records = []
    for idx, record in enumerate(records):
        try:
            image_path = batch_dir / record["page_image"]

            if progress_tracker and task_id:
                progress_tracker.update_progress(
                    task_id=task_id,
                    current=idx + 1,
                    message=f"开始处理第 {idx + 1}/{len(records)} 个样本"
                )

            if not image_path.exists():
                print(f"图像文件不存在: {image_path}", file=__import__("sys").stderr)
                record["embedding"] = None
                updated_records.append(record)
            else:
                embedding = extractor.extract_embedding(image_path)
                record["embedding"] = embedding
                updated_records.append(record)

            if progress_tracker and task_id:
                progress_tracker.update_progress(
                    task_id=task_id,
                    current=idx + 1,
                    message=f"已处理 {idx + 1}/{len(records)} 个样本"
                )

        except Exception as e:
            print(f"处理记录embedding失败 {record.get('sample_id')}: {e}", file=__import__("sys").stderr)
            record["embedding"] = None
            updated_records.append(record)

            if progress_tracker and task_id:
                progress_tracker.update_progress(
                    task_id=task_id,
                    current=idx + 1,
                    message=f"已处理 {idx + 1}/{len(records)} 个样本（含失败）"
                )

    return updated_records
